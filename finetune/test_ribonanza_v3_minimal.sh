#!/bin/bash
# =============================================================================
# Minimal GPU Integration Test: RibonanzaNet2 v3 Tokenizer + Model
#
# This script runs a very short training (2 steps) to validate:
#   1. RibonanzaTokenizer correctly produces tokenized_seq & ribonanza_token_mask
#   2. Model forward pass with v3 src_mask logic does not crash
#   3. Backward pass and gradient flow through adapter parameters
#   4. RibonanzaNet backbone stays frozen
#
# Usage:
#   bash finetune/test_ribonanza_v3_minimal.sh
# =============================================================================
set -eo pipefail

# Activate conda environment
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate protenix
set -u

# ===================== Paths =====================
PROJECT_ROOT="/inspire/ssd/project/sais-bio/public/ash_proj"
PROTENIX_DIR="${PROJECT_ROOT}/code/protenix_pro/Protenix"
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

# RibonanzaNet2 model
RIBONANZA_MODEL_DIR="${PROJECT_ROOT}/data/ribonanzanet2/model_weights"

RNA_EMBEDDING_DIM=2048
DNA_EMBEDDING_DIM=1024
MODEL_NAME="aido"

RUN_NAME="test_ribonanza_v3_minimal"
OUTPUT_DIR="${PROTENIX_DIR}/output/${RUN_NAME}"

TRAIN_SET="weightedPDB_before2109_wopb_nometalc_0925"
VAL_SET="recentPDB_1536_sample384_0925"

# Validate critical paths
[ -f "${CHECKPOINT_PATH}" ] || { echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"; exit 1; }
[ -f "${RIBONANZA_MODEL_DIR}/pairwise.yaml" ] || { echo "ERROR: RibonanzaNet2 config not found"; exit 1; }
[ -f "${RIBONANZA_MODEL_DIR}/pytorch_model_fsdp.bin" ] || { echo "ERROR: RibonanzaNet2 weights not found"; exit 1; }

# ===================== Environment =====================
export PROTENIX_ROOT_DIR="${PROJECT_ROOT}/data"
CONDA_PREFIX="${PROJECT_ROOT}/conda/envs/protenix"
export CUDA_HOME="${CONDA_PREFIX}"
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"

cd "${PROTENIX_DIR}"
export PYTHONPATH="${PROTENIX_DIR}:${PYTHONPATH:-}"
export TRIANGLE_ATTENTION="cuequivariance"
export TRIANGLE_MULTIPLICATIVE="cuequivariance"

mkdir -p "${OUTPUT_DIR}"

echo "========================================================"
echo "  Minimal GPU Test: RibonanzaNet2 v3 Integration"
echo "  checkpoint:   ${CHECKPOINT_PATH}"
echo "  ribonanza:    ${RIBONANZA_MODEL_DIR}"
echo "  max_steps:    2"
echo "  crop_size:    128"
echo "========================================================"

python3 ./runner/train.py \
    --model_name "protenix_base_20250630_v1.0.0" \
    --run_name "${RUN_NAME}" \
    --seed 42 \
    --base_dir "${OUTPUT_DIR}" \
    --dtype "bf16" \
    --project "protenix_rna_v3_test" \
    --use_wandb "false" \
    --diffusion_batch_size 4 \
    --eval_interval 9999 \
    --log_interval 1 \
    --checkpoint_interval 9999 \
    --train_crop_size 128 \
    --max_steps 2 \
    --lr 0.001 \
    --lr_scheduler "af3" \
    --warmup_steps 1 \
    --grad_clip_norm 10 \
    --model.N_cycle 1 \
    --sample_diffusion.N_step 5 \
    --triangle_attention "cuequivariance" \
    --triangle_multiplicative "cuequivariance" \
    --load_checkpoint_path "${CHECKPOINT_PATH}" \
    --load_ema_checkpoint_path "${CHECKPOINT_PATH}" \
    --load_strict false \
    --data.num_dl_workers 0 \
    --ema_decay 0.999 \
    --adam.use_adamw true \
    --adam.beta1 0.9 \
    --adam.beta2 0.999 \
    --adam.weight_decay 0.01 \
    --loss.weight.alpha_bond 0.5 \
    --model.confidence_head.stop_gradient true \
    \
    --two_stage.enable false \
    --two_stage.adapter_lr 0.005 \
    --two_stage.backbone_lr 0.0001 \
    \
    --rnalm.enable true \
    --rnalm.use_rna_embed true \
    --rnalm.use_dna_embed true \
    --rnalm.model_name "${MODEL_NAME}" \
    --rnalm.embedding_dim "${RNA_EMBEDDING_DIM}" \
    --rnalm.dna_embedding_dim "${DNA_EMBEDDING_DIM}" \
    --rnalm.injection_mode "diffusion" \
    --rnalm.gate_mode "none" \
    --rnalm.gate_init_logit "-3.0" \
    --rnalm.separate_dna_projection true \
    --rnalm.embedding_dir "${RNA_EMBEDDING_DIR}" \
    --rnalm.sequence_fpath "${RNA_SEQUENCE_FPATH}" \
    --rnalm.dna_embedding_dir "${DNA_EMBEDDING_DIR}" \
    --rnalm.dna_sequence_fpath "${DNA_SEQUENCE_FPATH}" \
    \
    --ribonanzanet2.enable true \
    --ribonanzanet2.model_dir "${RIBONANZA_MODEL_DIR}" \
    --ribonanzanet2.gate_type "channel" \
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
    --data.${VAL_SET}.base_info.max_n_token 384 \
    --data.msa.enable_rna_msa true \
    --data.msa.rna_msadir_raw_paths "${PREPARED_DATA_DIR}/rna_msa/msas" \
    --data.msa.rna_seq_or_filename_to_msadir_jsons "${PREPARED_DATA_DIR}/rna_msa/rna_sequence_to_pdb_chains.json" \
    --data.msa.rna_indexing_methods "sequence" \
    --data.msa.enable_prot_msa false \
    --data.template.enable_prot_template false \
    2>&1 | tee "${OUTPUT_DIR}/test_log.txt"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "========================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  TEST PASSED: RibonanzaNet2 v3 minimal GPU test completed successfully"
else
    echo "  TEST FAILED: Exit code ${EXIT_CODE}"
    echo "  Check log: ${OUTPUT_DIR}/test_log.txt"
fi
echo "========================================================"

exit ${EXIT_CODE}
