#!/bin/bash
# =============================================================================
# Quick backward-compatibility smoke test: existing separate_input config
# should work unchanged (use_rna_embed=True, use_dna_embed=True are defaults)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="/inspire/ssd/project/sais-bio/public/ash_proj"
PROTENIX_DIR="${PROJECT_ROOT}/code/protenix_new/Protenix"
DATA_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/part2"
PREPARED_DATA_DIR="${DATA_DIR}/protenix_prepared"

CHECKPOINT_PATH="${PROTENIX_DIR}/checkpoints/protenix_base_20250630_v1.0.0.pt"
RNA_CIF_DIR="${DATA_DIR}/PDB_RNA"
BIOASSEMBLY_DIR="${PREPARED_DATA_DIR}/rna_bioassembly"

AIDO_EMBEDDING_DIR="${DATA_DIR}/aido_embeddings"
RNA_EMBEDDING_DIR="${AIDO_EMBEDDING_DIR}/rna"
RNA_SEQUENCE_FPATH="${RNA_EMBEDDING_DIR}/rna_sequences.csv"
DNA_EMBEDDING_DIR="${AIDO_EMBEDDING_DIR}/dna"
DNA_SEQUENCE_FPATH="${DNA_EMBEDDING_DIR}/dna_sequences.csv"

OUTPUT_DIR="${PROTENIX_DIR}/output/test_toggle_compat"

export PROTENIX_ROOT_DIR="${PROJECT_ROOT}/data"
CONDA_PREFIX="${PROJECT_ROOT}/conda/envs/protenix"
export CUDA_HOME="${CONDA_PREFIX}"
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"

cd "${PROTENIX_DIR}"
export PYTHONPATH="${PROTENIX_DIR}:${PYTHONPATH:-}"

TRAIN_SET="weightedPDB_before2109_wopb_nometalc_0925"
VAL_SET="recentPDB_1536_sample384_0925"

mkdir -p "${OUTPUT_DIR}"

echo "========================================================"
echo "  TEST 1: Backward-compatibility (both RNA+DNA enabled)"
echo "========================================================"

python3 ./runner/train.py \
    --model_name "protenix_base_20250630_v1.0.0" \
    --run_name "test_toggle_compat" \
    --seed 42 \
    --base_dir "${OUTPUT_DIR}" \
    --dtype "bf16" \
    --project "protenix_test" \
    --use_wandb false \
    --diffusion_batch_size 48 \
    --eval_interval 9999 \
    --log_interval 1 \
    --checkpoint_interval -1 \
    --train_crop_size 256 \
    --max_steps 3 \
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
    --lr 0.0001 \
    --warmup_steps 1 \
    --ema_decay 0.999 \
    --adam.use_adamw true \
    --adam.beta1 0.9 \
    --adam.beta2 0.999 \
    --adam.weight_decay 0.01 \
    --loss.weight.alpha_bond 0.5 \
    --model.confidence_head.stop_gradient true \
    --rna_loss.enable false \
    --two_stage.enable false \
    \
    --rnalm.enable true \
    --rnalm.model_name "aido" \
    --rnalm.embedding_dim 2048 \
    --rnalm.dna_embedding_dim 1024 \
    --rnalm.injection_mode "input" \
    --rnalm.separate_dna_projection true \
    --rnalm.embedding_dir "${RNA_EMBEDDING_DIR}" \
    --rnalm.sequence_fpath "${RNA_SEQUENCE_FPATH}" \
    --rnalm.dna_embedding_dir "${DNA_EMBEDDING_DIR}" \
    --rnalm.dna_sequence_fpath "${DNA_SEQUENCE_FPATH}" \
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
echo "TEST 1 PASSED: Backward-compatibility with RNA+DNA enabled"
echo ""
