#!/bin/bash
# =============================================================================
# Protenix + AIDO RNA/DNA LLM + ESM Protein : Full LM Embedding Pipeline
#
# Design:
#   - RNA LLM embedding (2048-dim) -> rna_projection -> add to RNA features
#   - DNA LLM embedding (1024-dim) -> dna_projection -> add to DNA features
#   - ESM2-3B protein embedding (2560-dim) -> esm_projection -> add to protein features
#   - Injection at InputFeatureEmbedder for all three
#   - separate_dna_projection=True: independent RNA/DNA projection layers
#   - Fallback: RNA<->DNA cross-manifest lookup for hybrid entities
# =============================================================================
set -euo pipefail

PROJECT_ROOT="/inspire/ssd/project/sais-bio/public/ash_proj"
PROTENIX_DIR="${PROJECT_ROOT}/code/protenix_new/Protenix"
DATA_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/part2"
PREPARED_DATA_DIR="${DATA_DIR}/protenix_prepared"

CHECKPOINT_PATH="${PROTENIX_DIR}/checkpoints/protenix_base_20250630_v1.0.0.pt"
RNA_CIF_DIR="${DATA_DIR}/PDB_RNA"
BIOASSEMBLY_DIR="${PREPARED_DATA_DIR}/rna_bioassembly"

# ============= AIDO RNA/DNA Embedding Configuration =============
AIDO_EMBEDDING_DIR="${DATA_DIR}/aido_embeddings"
RNA_EMBEDDING_DIR="${AIDO_EMBEDDING_DIR}/rna"
RNA_SEQUENCE_FPATH="${RNA_EMBEDDING_DIR}/rna_sequences.csv"
DNA_EMBEDDING_DIR="${AIDO_EMBEDDING_DIR}/dna"
DNA_SEQUENCE_FPATH="${DNA_EMBEDDING_DIR}/dna_sequences.csv"

RNA_EMBEDDING_DIM=2048  # AIDO.RNA-1.6B
DNA_EMBEDDING_DIM=1024  # AIDO.DNA-300M (native, no zero-padding)
MODEL_NAME="aido"

# ============= ESM Protein Embedding Configuration =============
ESM_EMBEDDING_DIR="${DATA_DIR}/esm_embeddings/esm2-3b"
ESM_SEQUENCE_FPATH="${ESM_EMBEDDING_DIR}/prot_sequences.csv"
ESM_EMBEDDING_DIM=2560  # ESM2-3B

OUTPUT_DIR="${PROTENIX_DIR}/output/aido_esm_input"
# ===========================================================

# Validate AIDO paths
if [ ! -f "${RNA_SEQUENCE_FPATH}" ]; then
    echo "ERROR: RNA sequence CSV not found: ${RNA_SEQUENCE_FPATH}"
    echo "Please run extract_sequences.py + generate_aido_embeddings.py first."
    exit 1
fi

USE_DNA_EMBEDDING_DIR=""
USE_DNA_SEQUENCE_FPATH=""
if [ -f "${DNA_SEQUENCE_FPATH}" ]; then
    echo "DNA embeddings found: ${DNA_SEQUENCE_FPATH}"
    USE_DNA_EMBEDDING_DIR="${DNA_EMBEDDING_DIR}"
    USE_DNA_SEQUENCE_FPATH="${DNA_SEQUENCE_FPATH}"
else
    echo "WARNING: DNA sequence CSV not found. DNA tokens will get zeros."
fi

# Validate ESM paths
USE_ESM_ENABLE="false"
USE_ESM_EMBEDDING_DIR=""
USE_ESM_SEQUENCE_FPATH=""
if [ -f "${ESM_SEQUENCE_FPATH}" ] && [ -d "${ESM_EMBEDDING_DIR}" ]; then
    echo "ESM protein embeddings found: ${ESM_SEQUENCE_FPATH}"
    USE_ESM_ENABLE="true"
    USE_ESM_EMBEDDING_DIR="${ESM_EMBEDDING_DIR}"
    USE_ESM_SEQUENCE_FPATH="${ESM_SEQUENCE_FPATH}"
else
    echo "WARNING: ESM sequence CSV not found at ${ESM_SEQUENCE_FPATH}. Protein tokens will get zeros."
fi

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

echo "========================================================"
echo "  Protenix + AIDO RNA/DNA + ESM Protein — Full LM Pipeline"
echo "  RNA:     ${RNA_EMBEDDING_DIM}-dim -> rna_projection"
echo "  DNA:     ${DNA_EMBEDDING_DIM}-dim -> dna_projection"
echo "  Protein: ${ESM_EMBEDDING_DIM}-dim -> esm_projection (enable=${USE_ESM_ENABLE})"
echo "  injection_mode: input (like ESM)"
echo "  separate_dna_projection: true"
echo "========================================================"

python3 ./runner/train.py \
    --model_name "protenix_base_20250630_v1.0.0" \
    --run_name "aido_esm_input" \
    --seed 42 \
    --base_dir "${OUTPUT_DIR}" \
    --dtype "bf16" \
    --project "protenix_rna_sota" \
    --use_wandb true \
    --diffusion_batch_size 48 \
    --eval_interval 400 \
    --log_interval 50 \
    --checkpoint_interval 1000 \
    --train_crop_size 384 \
    --max_steps 100000 \
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
    --lr 0.0001 \
    --warmup_steps 2000 \
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
    --two_stage.enable false \
    \
    --esm.enable ${USE_ESM_ENABLE} \
    --esm.model_name "esm2-3b" \
    --esm.embedding_dim "${ESM_EMBEDDING_DIM}" \
    --esm.embedding_dir "${USE_ESM_EMBEDDING_DIR}" \
    --esm.sequence_fpath "${USE_ESM_SEQUENCE_FPATH}" \
    \
    --rnalm.enable true \
    --rnalm.model_name "${MODEL_NAME}" \
    --rnalm.embedding_dim "${RNA_EMBEDDING_DIM}" \
    --rnalm.dna_embedding_dim "${DNA_EMBEDDING_DIM}" \
    --rnalm.injection_mode "input" \
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

echo "AIDO + ESM full LM pipeline fine-tuning complete!"
