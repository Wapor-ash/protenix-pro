#!/bin/bash
# =============================================================================
# Protenix + RNA LLM : 1-stage joint finetune — INPUT injection mode
# 思路：
# 1. 以稳定的 1-stage baseline 为底
# 2. RNA LLM embedding 在 InputFeatureEmbedder 处注入 s_inputs（模仿 ESM 注入方式）
# 3. Zero-init projection: 训练开始时模型行为与不使用 LLM 完全一致
# 4. 1-stage joint training，不使用 two_stage
# 5. injection_mode="input" — 独立于 diffusion 注入模式
# =============================================================================
set -euo pipefail

PROJECT_ROOT="/inspire/ssd/project/sais-bio/public/ash_proj"
PROTENIX_DIR="${PROJECT_ROOT}/code/protenix_new/Protenix"
DATA_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/part2"
PREPARED_DATA_DIR="${DATA_DIR}/protenix_prepared"
SCRIPT_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/protenix_rna"

CHECKPOINT_PATH="${PROTENIX_DIR}/checkpoints/protenix_base_20250630_v1.0.0.pt"
RNA_CIF_DIR="${DATA_DIR}/PDB_RNA"
BIOASSEMBLY_DIR="${PREPARED_DATA_DIR}/rna_bioassembly"
OUTPUT_DIR="${SCRIPT_DIR}/output/sota_one_stage_llm_input_inject"

RNALM_EMBEDDING_DIR="${SCRIPT_DIR}/rnalm_embeddings_real"
RNALM_SEQUENCE_FPATH="${RNALM_EMBEDDING_DIR}/rna_sequences.csv"

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

echo "🚀 Launching 1-stage joint finetuning with RNA LLM (INPUT injection, like ESM)..."

python3 ./runner/train.py \
    --model_name "protenix_base_20250630_v1.0.0" \
    --run_name "sota_one_stage_llm_input_inject" \
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
    --rnalm.enable true \
    --rnalm.embedding_dim 1280 \
    --rnalm.injection_mode "input" \
    --rnalm.embedding_dir "${RNALM_EMBEDDING_DIR}" \
    --rnalm.sequence_fpath "${RNALM_SEQUENCE_FPATH}" \
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

echo "1-stage joint finetuning with RNA LLM (INPUT injection) complete!"
