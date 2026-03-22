#!/bin/bash
set -euo pipefail

PROJECT_ROOT="/inspire/ssd/project/sais-bio/public/ash_proj"
PROTENIX_DIR="${PROJECT_ROOT}/code/protenix_new/Protenix"
DATA_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/part2"
PREPARED_DATA_DIR="${DATA_DIR}/protenix_prepared"
CONDA_PREFIX="${PROJECT_ROOT}/conda/envs/protenix"

export PROTENIX_ROOT_DIR="${PROJECT_ROOT}/data"
export CUDA_HOME="${CONDA_PREFIX}"
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export PYTHONPATH="${PROTENIX_DIR}:${PYTHONPATH:-}"
export TRIANGLE_ATTENTION="cuequivariance"
export TRIANGLE_MULTIPLICATIVE="cuequivariance"

cd "${PROTENIX_DIR}"

TRAIN_SET="weightedPDB_before2109_wopb_nometalc_0925"
VAL_SET="recentPDB_1536_sample384_0925"

${CONDA_PREFIX}/bin/python3 ./runner/train.py \
    --model_name "protenix_base_20250630_v1.0.0" \
    --run_name "smoke_test" \
    --seed 42 \
    --base_dir "./output/smoke_test" \
    --dtype "bf16" \
    --project "smoke_test" \
    --use_wandb false \
    --diffusion_batch_size 4 \
    --eval_interval 999999 \
    --log_interval 1 \
    --train_crop_size 128 \
    --max_steps 2 \
    --lr_scheduler "af3" \
    --grad_clip_norm 10 \
    --model.N_cycle 1 \
    --sample_diffusion.N_step 5 \
    --triangle_attention "cuequivariance" \
    --triangle_multiplicative "cuequivariance" \
    --load_checkpoint_path "./checkpoints/protenix_base_20250630_v1.0.0.pt" \
    --load_ema_checkpoint_path "./checkpoints/protenix_base_20250630_v1.0.0.pt" \
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
    --rnalm.enable true \
    --rnalm.model_name "aido" \
    --rnalm.embedding_dim 2048 \
    --rnalm.dna_embedding_dim 1024 \
    --rnalm.injection_mode "input" \
    --rnalm.separate_dna_projection true \
    --rnalm.embedding_dir "${DATA_DIR}/aido_embeddings/rna" \
    --rnalm.sequence_fpath "${DATA_DIR}/aido_embeddings/rna/rna_sequences.csv" \
    --rnalm.dna_embedding_dir "${DATA_DIR}/aido_embeddings/dna" \
    --rnalm.dna_sequence_fpath "${DATA_DIR}/aido_embeddings/dna/dna_sequences.csv" \
    --data.train_sets "${TRAIN_SET}" \
    --data.${TRAIN_SET}.base_info.mmcif_dir "${DATA_DIR}/PDB_RNA" \
    --data.${TRAIN_SET}.base_info.bioassembly_dict_dir "${PREPARED_DATA_DIR}/rna_bioassembly" \
    --data.${TRAIN_SET}.base_info.indices_fpath "${PREPARED_DATA_DIR}/indices/rna_bioassembly_indices.csv" \
    --data.${TRAIN_SET}.base_info.pdb_list "${PREPARED_DATA_DIR}/rna_train_pdb_list_filtered.txt" \
    --data.test_sets "${VAL_SET}" \
    --data.${VAL_SET}.base_info.mmcif_dir "${DATA_DIR}/PDB_RNA" \
    --data.${VAL_SET}.base_info.bioassembly_dict_dir "${PREPARED_DATA_DIR}/rna_bioassembly" \
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

echo "SMOKE TEST PASSED!"
