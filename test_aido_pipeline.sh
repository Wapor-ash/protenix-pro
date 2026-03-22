#!/bin/bash
# Quick test: 2 training steps with AIDO embeddings to verify pipeline
set -euo pipefail

PROJECT_ROOT="/inspire/ssd/project/sais-bio/public/ash_proj"
PROTENIX_DIR="${PROJECT_ROOT}/code/protenix_new/Protenix"
DATA_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/part2"
PREPARED_DATA_DIR="${DATA_DIR}/protenix_prepared"

CHECKPOINT_PATH="${PROTENIX_DIR}/checkpoints/protenix_base_20250630_v1.0.0.pt"
RNA_CIF_DIR="${DATA_DIR}/PDB_RNA"
BIOASSEMBLY_DIR="${PREPARED_DATA_DIR}/rna_bioassembly"

AIDO_DIR="${DATA_DIR}/aido_embeddings"
RNA_EMB_DIR="${AIDO_DIR}/rna"
RNA_SEQ_CSV="${RNA_EMB_DIR}/rna_sequences.csv"
DNA_EMB_DIR="${AIDO_DIR}/dna"
DNA_SEQ_CSV="${DNA_EMB_DIR}/dna_sequences.csv"
OUTPUT_DIR="${PROTENIX_DIR}/output/aido_test"

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

# DNA embedding args (if available)
DNA_EMB_ARG=""
DNA_SEQ_ARG=""
if [ -f "${DNA_SEQ_CSV}" ]; then
    DNA_EMB_ARG="${DNA_EMB_DIR}"
    DNA_SEQ_ARG="${DNA_SEQ_CSV}"
fi

mkdir -p "${OUTPUT_DIR}"

echo "Testing AIDO pipeline (2 steps)..."

python3 ./runner/train.py \
    --model_name "protenix_base_20250630_v1.0.0" \
    --run_name "aido_pipeline_test" \
    --seed 42 \
    --base_dir "${OUTPUT_DIR}" \
    --dtype "bf16" \
    --project "protenix_rna_test" \
    --use_wandb false \
    --diffusion_batch_size 48 \
    --eval_interval 2 \
    --log_interval 1 \
    --checkpoint_interval 999999 \
    --train_crop_size 384 \
    --max_steps 2 \
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
    --lr 0.001 \
    --warmup_steps 1 \
    --ema_decay 0.999 \
    \
    --adam.use_adamw true \
    --adam.beta1 0.9 \
    --adam.beta2 0.95 \
    --adam.weight_decay 0.01 \
    \
    --loss.weight.alpha_bond 0.5 \
    --model.confidence_head.stop_gradient true \
    --rna_loss.enable false \
    --two_stage.enable false \
    \
    --rnalm.enable true \
    --rnalm.model_name "aido" \
    --rnalm.embedding_dim 2048 \
    --rnalm.injection_mode "diffusion" \
    --rnalm.embedding_dir "${RNA_EMB_DIR}" \
    --rnalm.sequence_fpath "${RNA_SEQ_CSV}" \
    --rnalm.dna_embedding_dir "${DNA_EMB_ARG}" \
    --rnalm.dna_sequence_fpath "${DNA_SEQ_ARG}" \
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

echo "Pipeline test complete!"
