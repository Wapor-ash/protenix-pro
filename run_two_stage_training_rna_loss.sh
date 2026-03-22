#!/bin/bash
# =============================================================================
# Two-Stage Training with RNA-Optimized Loss Weights
#
# Same as run_two_stage_training.sh but with RNA loss overrides enabled:
#   - alpha_distogram: 0.03 -> 0.10 (enhance base-pairing distance learning)
#   - alpha_bond: 0.0 -> 0.5 (enable backbone bond constraints)
#   - weight_rna: 5.0 -> 8.0 (increase RNA atom weight)
#
# Stage 1 (Adapter Warmup):
#   - Freeze Protenix backbone (lr=0), only train rnalm_projection adapter
#   - High LR (5e-3) with cosine schedule, no EMA
#
# Stage 2 (Joint Training):
#   - Unfreeze all parameters, train backbone + adapter together
#   - Standard LR (1e-3) with warmup (100 steps), EMA decay=0.999
#
# The transition happens automatically within a single run.
# =============================================================================
set -e

# Activate conda environment

PROJECT_ROOT="/inspire/ssd/project/sais-bio/public/ash_proj"
PROTENIX_DIR="${PROJECT_ROOT}/code/protenix_new/Protenix"
DATA_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/part2"
PREPARED_DATA_DIR="${DATA_DIR}/protenix_prepared"
SCRIPT_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/protenix_rna"

CHECKPOINT_PATH="${PROTENIX_DIR}/checkpoints/protenix_base_20250630_v1.0.0.pt"
RNA_CIF_DIR="${DATA_DIR}/PDB_RNA"
OUTPUT_DIR="${SCRIPT_DIR}/output/two_stage_training_rna_loss"
BIOASSEMBLY_DIR="${PREPARED_DATA_DIR}/rna_bioassembly"

# === Real RiNALMo Embedding Paths ===
RNALM_EMBEDDING_DIR="${SCRIPT_DIR}/rnalm_embeddings_real"
RNALM_SEQUENCE_FPATH="${RNALM_EMBEDDING_DIR}/rna_sequences.csv"

export PROTENIX_ROOT_DIR="${PROJECT_ROOT}/data"

# CUDA headers for JIT compilation of fast layernorm kernel
CONDA_PREFIX="${PROJECT_ROOT}/conda/envs/protenix"
export CUDA_HOME="${CONDA_PREFIX}"
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH}"

echo "============================================================================="
echo "  Two-Stage Training + RNA-Optimized Loss Weights"
echo "============================================================================="

# Verify files
echo "Checking prerequisites..."
[ ! -f "${CHECKPOINT_PATH}" ] && echo "ERROR: No checkpoint" && exit 1
echo "  Checkpoint: OK"
echo "  Bioassembly: $(ls -1 ${BIOASSEMBLY_DIR}/*.pkl.gz 2>/dev/null | wc -l) pkl.gz files"
[ ! -d "${RNALM_EMBEDDING_DIR}" ] && echo "ERROR: No rnalm_embeddings_real dir. Run generate_real_rnalm_embeddings.py first." && exit 1
[ ! -f "${RNALM_SEQUENCE_FPATH}" ] && echo "ERROR: No rna_sequences.csv in rnalm_embeddings_real" && exit 1
echo "  RiNALMo REAL embeddings dir: OK ($(ls -1d ${RNALM_EMBEDDING_DIR}/*/ 2>/dev/null | wc -l) entities)"
echo "  RiNALMo sequence CSV: OK"

cd "${PROTENIX_DIR}"
export PYTHONPATH="${PROTENIX_DIR}:${PYTHONPATH}"
export TRIANGLE_ATTENTION="cuequivariance"
export TRIANGLE_MULTIPLICATIVE="cuequivariance"

TRAIN_SET="weightedPDB_before2109_wopb_nometalc_0925"
VAL_SET="recentPDB_1536_sample384_0925"

TRAIN_INDICES="${PREPARED_DATA_DIR}/indices/rna_bioassembly_indices.csv"
VAL_INDICES="${PREPARED_DATA_DIR}/indices/rna_bioassembly_indices.csv"
TRAIN_PDB_LIST="${PREPARED_DATA_DIR}/rna_train_pdb_list_filtered.txt"
VAL_PDB_LIST="${PREPARED_DATA_DIR}/rna_val_pdb_list_filtered.txt"
RNA_MSA_DIR="${PREPARED_DATA_DIR}/rna_msa/msas"
RNA_MSA_JSON="${PREPARED_DATA_DIR}/rna_msa/rna_sequence_to_pdb_chains.json"

mkdir -p "${OUTPUT_DIR}"

# =============================================
# Two-Stage Training Parameters
# =============================================
# Stage 1: Adapter Warmup
STAGE1_MAX_STEPS=400         # Train adapter for 400 steps
STAGE1_LR=5e-3              # Higher LR for adapter warmup
STAGE1_WARMUP=1             # Minimal warmup (adapter starts from zero-init)

# Stage 2: Joint Training
STAGE2_LR=1e-3              # Standard LR for full model
STAGE2_WARMUP=100           # Warmup for backbone unfreezing
STAGE2_EMA_DECAY=0.999      # EMA only in Stage 2

# Total training
TOTAL_MAX_STEPS=100000       # Total steps (Stage1 + Stage2)

# =============================================
# RNA Loss Override Parameters
# =============================================
RNA_ALPHA_DISTOGRAM=0.10     # Default: 0.03 -> 0.10
RNA_ALPHA_BOND=0.5           # Default: 0.0  -> 0.5
RNA_WEIGHT_RNA=8.0           # Default: 5.0  -> 8.0

echo ""
echo "Training Configuration:"
echo "  Stage 1 (Adapter Warmup): ${STAGE1_MAX_STEPS} steps, lr=${STAGE1_LR}, warmup=${STAGE1_WARMUP}, no EMA"
echo "  Stage 2 (Joint Training): $((TOTAL_MAX_STEPS - STAGE1_MAX_STEPS)) steps, lr=${STAGE2_LR}, warmup=${STAGE2_WARMUP}, EMA=${STAGE2_EMA_DECAY}"
echo "  Total: ${TOTAL_MAX_STEPS} steps"
echo ""
echo "RNA Loss Overrides:"
echo "  alpha_distogram: 0.03 -> ${RNA_ALPHA_DISTOGRAM}"
echo "  alpha_bond:      0.0  -> ${RNA_ALPHA_BOND} (actual weight: 4.0 * ${RNA_ALPHA_BOND})"
echo "  weight_rna:      5.0  -> ${RNA_WEIGHT_RNA}"
echo ""

python3 ./runner/train.py \
    --model_name "protenix_base_20250630_v1.0.0" \
    --run_name "two_stage_rna_loss" \
    --seed 42 \
    --base_dir "${OUTPUT_DIR}" \
    --dtype "bf16" \
    --project "protenix_rna_two_stage_rna_loss" \
    --use_wandb True \
    --diffusion_batch_size 48 \
    --eval_interval 400 \
    --log_interval 50 \
    --checkpoint_interval 400 \
    --ema_decay -1 \
    --train_crop_size 384 \
    --max_steps ${TOTAL_MAX_STEPS} \
    --warmup_steps 1 \
    --lr 0.001 \
    --lr_scheduler "af3" \
    --grad_clip_norm 10 \
    --model.N_cycle 4 \
    --sample_diffusion.N_step 20 \
    --triangle_attention "cuequivariance" \
    --triangle_multiplicative "cuequivariance" \
    --load_checkpoint_path "${CHECKPOINT_PATH}" \
    --load_strict false \
    --data.num_dl_workers 0 \
    \
    --data.train_sets "${TRAIN_SET}" \
    --data.${TRAIN_SET}.base_info.mmcif_dir "${RNA_CIF_DIR}" \
    --data.${TRAIN_SET}.base_info.bioassembly_dict_dir "${BIOASSEMBLY_DIR}" \
    --data.${TRAIN_SET}.base_info.indices_fpath "${TRAIN_INDICES}" \
    --data.${TRAIN_SET}.base_info.pdb_list "${TRAIN_PDB_LIST}" \
    \
    --data.test_sets "${VAL_SET}" \
    --data.${VAL_SET}.base_info.mmcif_dir "${RNA_CIF_DIR}" \
    --data.${VAL_SET}.base_info.bioassembly_dict_dir "${BIOASSEMBLY_DIR}" \
    --data.${VAL_SET}.base_info.indices_fpath "${VAL_INDICES}" \
    --data.${VAL_SET}.base_info.pdb_list "${VAL_PDB_LIST}" \
    --data.${VAL_SET}.base_info.find_eval_chain_interface false \
    --data.${VAL_SET}.base_info.group_by_pdb_id false \
    --data.${VAL_SET}.base_info.max_n_token 1536 \
    \
    --data.msa.enable_rna_msa true \
    --data.msa.rna_msadir_raw_paths "${RNA_MSA_DIR}" \
    --data.msa.rna_seq_or_filename_to_msadir_jsons "${RNA_MSA_JSON}" \
    --data.msa.rna_indexing_methods "sequence" \
    --data.msa.enable_prot_msa false \
    --data.template.enable_prot_template false \
    \
    --rnalm.enable true \
    --rnalm.embedding_dim 1280 \
    --rnalm.embedding_dir "${RNALM_EMBEDDING_DIR}" \
    --rnalm.sequence_fpath "${RNALM_SEQUENCE_FPATH}" \
    \
    --two_stage.enable true \
    --two_stage.stage1_max_steps ${STAGE1_MAX_STEPS} \
    --two_stage.stage1_lr ${STAGE1_LR} \
    --two_stage.stage1_warmup_steps ${STAGE1_WARMUP} \
    --two_stage.stage2_lr ${STAGE2_LR} \
    --two_stage.stage2_warmup_steps ${STAGE2_WARMUP} \
    --two_stage.stage2_ema_decay ${STAGE2_EMA_DECAY} \
    --two_stage.adapter_keywords "rnalm_projection" \
    \
    --rna_loss.enable true \
    --rna_loss.alpha_distogram ${RNA_ALPHA_DISTOGRAM} \
    --rna_loss.alpha_bond ${RNA_ALPHA_BOND} \
    --rna_loss.weight_rna ${RNA_WEIGHT_RNA}

echo ""
echo "============================================================================="
echo "Two-stage training with RNA loss completed!"
echo "============================================================================="
