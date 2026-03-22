#!/bin/bash
# =============================================================================
# Protenix Fine-tune: RNA + DNA LLM Embeddings (AIDO)
#
# Uses both RNA (2048-dim) and DNA (1024-dim) embeddings with separate
# projection layers. All key embedding parameters are configurable via
# command-line arguments.
#
# Usage:
#   bash finetune/finetune_rna_dna.sh                          # defaults: input injection
#   bash finetune/finetune_rna_dna.sh --injection_mode diffusion
#   bash finetune/finetune_rna_dna.sh --injection_mode both --gate_mode scalar
#   bash finetune/finetune_rna_dna.sh --two_stage true
#   bash finetune/finetune_rna_dna.sh --max_steps 50000 --lr 0.0005
# =============================================================================
set -euo pipefail

# ===================== Configurable Parameters =====================
INJECTION_MODE="input"          # "diffusion" | "input" | "both"
GATE_MODE="none"                # "none" | "scalar" | "token" | "dual"
GATE_INIT_LOGIT="-3.0"
TWO_STAGE="false"               # "true" | "false"
MAX_STEPS=100000
LR=0.0001
WARMUP_STEPS=2000
TRAIN_CROP_SIZE=384
EVAL_INTERVAL=400
CHECKPOINT_INTERVAL=1000
USE_WANDB="true"
RUN_NAME="0311_16_rna_dna_input"                     # auto-generated if empty
# ===================================================================

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --injection_mode)   INJECTION_MODE="$2";    shift 2 ;;
        --gate_mode)        GATE_MODE="$2";         shift 2 ;;
        --gate_init_logit)  GATE_INIT_LOGIT="$2";   shift 2 ;;
        --two_stage)        TWO_STAGE="$2";         shift 2 ;;
        --max_steps)        MAX_STEPS="$2";         shift 2 ;;
        --lr)               LR="$2";               shift 2 ;;
        --warmup_steps)     WARMUP_STEPS="$2";      shift 2 ;;
        --train_crop_size)  TRAIN_CROP_SIZE="$2";   shift 2 ;;
        --eval_interval)    EVAL_INTERVAL="$2";     shift 2 ;;
        --checkpoint_interval) CHECKPOINT_INTERVAL="$2"; shift 2 ;;
        --use_wandb)        USE_WANDB="$2";         shift 2 ;;
        --run_name)         RUN_NAME="$2";          shift 2 ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo "Valid args: --injection_mode, --gate_mode, --gate_init_logit,"
            echo "            --two_stage, --max_steps, --lr, --warmup_steps, --train_crop_size,"
            echo "            --eval_interval, --checkpoint_interval, --use_wandb, --run_name"
            exit 1
            ;;
    esac
done

# Validate injection_mode
if [[ "${INJECTION_MODE}" != "input" && "${INJECTION_MODE}" != "diffusion" && "${INJECTION_MODE}" != "both" ]]; then
    echo "ERROR: --injection_mode must be 'input', 'diffusion', or 'both' (got: ${INJECTION_MODE})"
    exit 1
fi

# Validate gate_mode
if [[ "${GATE_MODE}" != "none" && "${GATE_MODE}" != "scalar" && "${GATE_MODE}" != "token" && "${GATE_MODE}" != "dual" ]]; then
    echo "ERROR: --gate_mode must be 'none', 'scalar', 'token', or 'dual' (got: ${GATE_MODE})"
    exit 1
fi

# ===================== Paths =====================
PROJECT_ROOT="/inspire/ssd/project/sais-bio/public/ash_proj"
PROTENIX_DIR="${PROJECT_ROOT}/code/protenix_new/Protenix"
DATA_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/part2"
PREPARED_DATA_DIR="${DATA_DIR}/protenix_prepared"

CHECKPOINT_PATH="${PROTENIX_DIR}/checkpoints/protenix_base_20250630_v1.0.0.pt"
RNA_CIF_DIR="${DATA_DIR}/PDB_RNA"
BIOASSEMBLY_DIR="${PREPARED_DATA_DIR}/rna_bioassembly"

# AIDO Embedding paths
AIDO_EMBEDDING_DIR="${DATA_DIR}/aido_embeddings"
RNA_EMBEDDING_DIR="${AIDO_EMBEDDING_DIR}/rna"
RNA_SEQUENCE_FPATH="${RNA_EMBEDDING_DIR}/rna_sequences.csv"
DNA_EMBEDDING_DIR="${AIDO_EMBEDDING_DIR}/dna"
DNA_SEQUENCE_FPATH="${DNA_EMBEDDING_DIR}/dna_sequences.csv"

RNA_EMBEDDING_DIM=2048  # AIDO.RNA-1.6B
DNA_EMBEDDING_DIM=1024  # AIDO.DNA-300M (native dim)
MODEL_NAME="aido"

# Auto-generate run name if not set
if [ -z "${RUN_NAME}" ]; then
    RUN_NAME="ft_rna_dna_${INJECTION_MODE}"
    if [ "${GATE_MODE}" != "none" ]; then
        RUN_NAME="${RUN_NAME}_gate_${GATE_MODE}"
    fi
    if [ "${TWO_STAGE}" = "true" ]; then
        RUN_NAME="${RUN_NAME}_2stage"
    fi
fi

OUTPUT_DIR="${PROTENIX_DIR}/output/${RUN_NAME}"

# ===================== Validate Paths =====================
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi
if [ ! -f "${RNA_SEQUENCE_FPATH}" ]; then
    echo "ERROR: RNA sequence CSV not found: ${RNA_SEQUENCE_FPATH}"
    echo "Please run generate_aido_embeddings.py first."
    exit 1
fi

USE_DNA_EMBEDDING_DIR=""
USE_DNA_SEQUENCE_FPATH=""
if [ -f "${DNA_SEQUENCE_FPATH}" ]; then
    echo "DNA embeddings found: ${DNA_SEQUENCE_FPATH}"
    USE_DNA_EMBEDDING_DIR="${DNA_EMBEDDING_DIR}"
    USE_DNA_SEQUENCE_FPATH="${DNA_SEQUENCE_FPATH}"
else
    echo "WARNING: DNA sequence CSV not found: ${DNA_SEQUENCE_FPATH}"
    echo "DNA tokens will receive zero vectors."
fi

# ===================== Environment =====================
export PROTENIX_ROOT_DIR="${PROJECT_ROOT}/data"
CONDA_PREFIX="${PROJECT_ROOT}/conda/envs/protenix"
export CUDA_HOME="${CONDA_PREFIX}"
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"

cd "${PROTENIX_DIR}"
export PYTHONPATH="${PROTENIX_DIR}:${PYTHONPATH:-}"
export TRIANGLE_ATTENTION="cuequivariance"
export TRIANGLE_MULTIPLICATIVE="cuequivariance"

TRAIN_SET="weightedPDB_before2109_wopb_nometalc_0925"
VAL_SET="recentPDB_1536_sample384_0925"

mkdir -p "${OUTPUT_DIR}"

# ===================== Banner =====================
echo "========================================================"
echo "  Protenix Fine-tune: RNA + DNA (AIDO)"
echo "  RNA: ${RNA_EMBEDDING_DIM}-dim  |  DNA: ${DNA_EMBEDDING_DIM}-dim"
echo "  injection_mode:  ${INJECTION_MODE}"
echo "  gate_mode:       ${GATE_MODE}"
echo "  two_stage:       ${TWO_STAGE}"
echo "  max_steps:       ${MAX_STEPS}"
echo "  lr:              ${LR}"
echo "  run_name:        ${RUN_NAME}"
echo "  output_dir:      ${OUTPUT_DIR}"
echo "========================================================"

python3 ./runner/train.py \
    --model_name "protenix_base_20250630_v1.0.0" \
    --run_name "${RUN_NAME}" \
    --seed 42 \
    --base_dir "${OUTPUT_DIR}" \
    --dtype "bf16" \
    --project "protenix_rna_aido" \
    --use_wandb "${USE_WANDB}" \
    --diffusion_batch_size 48 \
    --eval_interval "${EVAL_INTERVAL}" \
    --log_interval 50 \
    --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
    --train_crop_size "${TRAIN_CROP_SIZE}" \
    --max_steps "${MAX_STEPS}" \
    --lr_scheduler "af3" \
    --grad_clip_norm 10 \
    --model.N_cycle 4 \
    --sample_diffusion.N_step 20 \
    --triangle_attention "cuequivariance" \
    --triangle_multiplicative "cuequivariance" \
    --load_checkpoint_path "${CHECKPOINT_PATH}" \
    --load_ema_checkpoint_path "${CHECKPOINT_PATH}" \
    --load_strict false \
    --data.num_dl_workers 0 \
    \
    --lr "${LR}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --ema_decay 0.999 \
    \
    --adam.use_adamw true \
    --adam.beta1 0.9 \
    --adam.beta2 0.999 \
    --adam.weight_decay 0.01 \
    \
    --loss.weight.alpha_bond 0.5 \
    --model.confidence_head.stop_gradient true \
    --rna_loss.enable false \
    --two_stage.enable "${TWO_STAGE}" \
    \
    --rnalm.enable true \
    --rnalm.use_rna_embed true \
    --rnalm.use_dna_embed true \
    --rnalm.model_name "${MODEL_NAME}" \
    --rnalm.embedding_dim "${RNA_EMBEDDING_DIM}" \
    --rnalm.dna_embedding_dim "${DNA_EMBEDDING_DIM}" \
    --rnalm.injection_mode "${INJECTION_MODE}" \
    --rnalm.gate_mode "${GATE_MODE}" \
    --rnalm.gate_init_logit "${GATE_INIT_LOGIT}" \
    --rnalm.separate_dna_projection true \
    --rnalm.embedding_dir "${RNA_EMBEDDING_DIR}" \
    --rnalm.sequence_fpath "${RNA_SEQUENCE_FPATH}" \
    --rnalm.dna_embedding_dir "${USE_DNA_EMBEDDING_DIR}" \
    --rnalm.dna_sequence_fpath "${USE_DNA_SEQUENCE_FPATH}" \
    \
    --data.train_sets "${TRAIN_SET}" \
    --data.${TRAIN_SET}.base_info.mmcif_dir "${RNA_CIF_DIR}" \
    --data.${TRAIN_SET}.base_info.bioassembly_dict_dir "${BIOASSEMBLY_DIR}" \
    --data.${TRAIN_SET}.base_info.indices_fpath "${PREPARED_DATA_DIR}/indices/rna_bioassembly_indices.csv" \
    --data.${TRAIN_SET}.base_info.pdb_list "${PREPARED_DATA_DIR}/rna_train_pdb_list_filtered.txt" \
    --data.test_sets "${VAL_SET}" \
    --data.${VAL_SET}.base_info.mmcif_dir "${RNA_CIF_DIR}" \
    --data.${VAL_SET}.base_info.bioassembly_dict_dir "${BIOASSEMBLY_DIR}" \
    --data.${VAL_SET}.base_info.indices_fpath "${PREPARED_DATA_DIR}/indices/rna_bioassembly_indices.csv" \
    --data.${VAL_SET}.base_info.pdb_list "${PREPARED_DATA_DIR}/rna_val_pdb_list_filtered.txt" \
    --data.${VAL_SET}.base_info.find_eval_chain_interface false \
    --data.${VAL_SET}.base_info.group_by_pdb_id false \
    --data.${VAL_SET}.base_info.max_n_token 1536 \
    --data.msa.enable_rna_msa true \
    --data.msa.rna_msadir_raw_paths "${PREPARED_DATA_DIR}/rna_msa/msas" \
    --data.msa.rna_seq_or_filename_to_msadir_jsons "${PREPARED_DATA_DIR}/rna_msa/rna_sequence_to_pdb_chains.json" \
    --data.msa.rna_indexing_methods "sequence" \
    --data.msa.enable_prot_msa false \
    --data.template.enable_prot_template false

echo ""
echo "Fine-tune complete: RNA+DNA | injection=${INJECTION_MODE} | gate=${GATE_MODE}"
echo "Output: ${OUTPUT_DIR}"
