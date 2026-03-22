#!/bin/bash
set -euo pipefail

# =============================================================
# Protenix RNA finetune WITH RNA LLM (two-stage)
# Balanced recipe:
# - follows the paper's best fusion choice: single-conditioning + add + RNA MSA
# - keeps official Protenix fine-tuning loss weights
# - uses short two-stage schedule instead of 100k-step long run
# =============================================================

PROJECT_ROOT="/inspire/ssd/project/sais-bio/public/ash_proj"
PROTENIX_DIR="${PROJECT_ROOT}/code/protenix_new/Protenix"
DATA_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/part2"
PREPARED_DATA_DIR="${DATA_DIR}/protenix_prepared"
SCRIPT_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/protenix_rna"

CHECKPOINT_PATH="${PROTENIX_DIR}/checkpoints/protenix_base_20250630_v1.0.0.pt"
RNA_CIF_DIR="${DATA_DIR}/PDB_RNA"
BIOASSEMBLY_DIR="${PREPARED_DATA_DIR}/rna_bioassembly"
OUTPUT_DIR="${SCRIPT_DIR}/output/finetune_with_llm_2stage_balanced"

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
TRAIN_INDICES="${PREPARED_DATA_DIR}/indices/rna_bioassembly_indices.csv"
VAL_INDICES="${PREPARED_DATA_DIR}/indices/rna_bioassembly_indices.csv"
TRAIN_PDB_LIST="${PREPARED_DATA_DIR}/rna_train_pdb_list_filtered.txt"
VAL_PDB_LIST="${PREPARED_DATA_DIR}/rna_val_pdb_list_filtered.txt"
RNA_MSA_DIR="${PREPARED_DATA_DIR}/rna_msa/msas"
RNA_MSA_JSON="${PREPARED_DATA_DIR}/rna_msa/rna_sequence_to_pdb_chains.json"

STAGE1_MAX_STEPS=400
STAGE1_LR=1e-2
STAGE1_WARMUP=1
STAGE2_LR=1e-3
STAGE2_WARMUP=200
STAGE2_EMA_DECAY=0.999
TOTAL_MAX_STEPS=50000

mkdir -p "${OUTPUT_DIR}"

python3 ./runner/train.py \
  --model_name "protenix_base_20250630_v1.0.0" \
  --run_name "rna_finetune_with_llm_2stage_balanced" \
  --seed 42 \
  --base_dir "${OUTPUT_DIR}" \
  --dtype "bf16" \
  --project "protenix_rna_with_llm_2stage_balanced" \
  --use_wandb true \
  --diffusion_batch_size 48 \
  --eval_interval 200 \
  --log_interval 50 \
  --checkpoint_interval 200 \
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
  --loss.weight.alpha_pae 0.0 \
  --loss.weight.alpha_bond 0.5 \
  --loss.weight.smooth_lddt 0.0 \
  --loss.weight.alpha_confidence 1e-4 \
  --loss.weight.alpha_diffusion 4.0 \
  --loss.weight.alpha_distogram 0.03 \
  --model.confidence_head.stop_gradient true \
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
  --two_stage.adapter_keywords "rnalm_projection"
