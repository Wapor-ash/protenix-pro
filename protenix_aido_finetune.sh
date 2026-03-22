#!/bin/bash
# =============================================================================
# Protenix + AIDO RNA/DNA LLM : 1-stage joint finetune
#
# This script replaces RiNALMo with AIDO embeddings for RNA and DNA:
# - AIDO.RNA-1.6B: 2048-dim per-nucleotide RNA embeddings
# - AIDO.DNA-300M: 2048-dim per-nucleotide DNA embeddings (optional)
#
# Key differences from protenix_rna_llm_input_inject.sh:
# - embedding_dim: 2048 (AIDO) instead of 1280 (RiNALMo)
# - model_name: "aido" instead of "rinalmo"
# - dna_embedding_dir/dna_sequence_fpath: DNA embedding paths
# - Training logic and hyperparameters are UNCHANGED
# =============================================================================
set -euo pipefail

PROJECT_ROOT="/inspire/ssd/project/sais-bio/public/ash_proj"
PROTENIX_DIR="${PROJECT_ROOT}/code/protenix_new/Protenix"
DATA_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/part2"
PREPARED_DATA_DIR="${DATA_DIR}/protenix_prepared"

CHECKPOINT_PATH="${PROTENIX_DIR}/checkpoints/protenix_base_20250630_v1.0.0.pt"
RNA_CIF_DIR="${DATA_DIR}/PDB_RNA"
BIOASSEMBLY_DIR="${PREPARED_DATA_DIR}/rna_bioassembly"

# ============= AIDO Embedding Configuration =============
# RNA embeddings from AIDO.RNA-1.6B (2048-dim)
AIDO_EMBEDDING_DIR="${DATA_DIR}/aido_embeddings"
RNA_EMBEDDING_DIR="${AIDO_EMBEDDING_DIR}/rna"
RNA_SEQUENCE_FPATH="${RNA_EMBEDDING_DIR}/rna_sequences.csv"

# DNA embeddings from AIDO.DNA-300M (2048-dim)
DNA_EMBEDDING_DIR="${AIDO_EMBEDDING_DIR}/dna"
DNA_SEQUENCE_FPATH="${DNA_EMBEDDING_DIR}/dna_sequences.csv"

# Embedding dimension (AIDO models output 2048-dim embeddings)
EMBEDDING_DIM=2048

# Model name identifier
MODEL_NAME="aido"

# Output directory
OUTPUT_DIR="${PROTENIX_DIR}/output/aido_finetune"
# ===========================================================

# Check if RNA embeddings exist
if [ ! -f "${RNA_SEQUENCE_FPATH}" ]; then
    echo "ERROR: RNA sequence CSV not found: ${RNA_SEQUENCE_FPATH}"
    echo "Please run generate_aido_embeddings.py first."
    exit 1
fi

# Enable/disable DNA embeddings based on file existence
USE_DNA_EMBEDDING_DIR=""
USE_DNA_SEQUENCE_FPATH=""
if [ -f "${DNA_SEQUENCE_FPATH}" ]; then
    echo "DNA embeddings found: ${DNA_SEQUENCE_FPATH}"
    USE_DNA_EMBEDDING_DIR="${DNA_EMBEDDING_DIR}"
    USE_DNA_SEQUENCE_FPATH="${DNA_SEQUENCE_FPATH}"
else
    echo "WARNING: DNA sequence CSV not found: ${DNA_SEQUENCE_FPATH}"
    echo "Continuing without DNA embeddings (DNA tokens will get zeros)."
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
echo "  Protenix + AIDO RNA/DNA LLM Fine-tuning"
echo "  RNA embedding dim: ${EMBEDDING_DIM}"
echo "  RNA embedding dir: ${RNA_EMBEDDING_DIR}"
echo "  DNA embedding dir: ${USE_DNA_EMBEDDING_DIR:-none}"
echo "  Model name: ${MODEL_NAME}"
echo "========================================================"

python3 ./runner/train.py \
    --model_name "protenix_base_20250630_v1.0.0" \
    --run_name "aido_rna_dna_finetune" \
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
    --warmup_steps 1000 \
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
    --rnalm.enable true \
    --rnalm.model_name "${MODEL_NAME}" \
    --rnalm.embedding_dim "${EMBEDDING_DIM}" \
    --rnalm.injection_mode "diffusion" \
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

echo "AIDO RNA/DNA fine-tuning complete!"
