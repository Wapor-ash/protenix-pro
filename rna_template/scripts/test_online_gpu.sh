#!/bin/bash
# =============================================================================
# Quick GPU validation for RNA Template v3 (Online Mode)
#
# Tests the online pipeline on GPU:
#   1. Online featurizer unit tests (PDB API, CIF build, filtering)
#   2. Short GPU training run with online mode enabled
#
# Usage:
#   bash rna_template/scripts/test_online_gpu.sh
#   bash rna_template/scripts/test_online_gpu.sh --max_steps 10
# =============================================================================
set -eo pipefail

# Activate conda
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate protenix 2>/dev/null

PROJECT_ROOT="/inspire/ssd/project/sais-bio/public/ash_proj"
PROTENIX_DIR="${PROJECT_ROOT}/code/protenix_new/Protenix"
DATA_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/part2"
PREPARED_DATA_DIR="${DATA_DIR}/protenix_prepared"
RNA_DATABASE_DIR="${PROTENIX_DIR}/rna_database"

MAX_STEPS="${1:-10}"

cd "${PROTENIX_DIR}"
export PYTHONPATH="${PROTENIX_DIR}:${PYTHONPATH:-}"
export PROTENIX_ROOT_DIR="${PROJECT_ROOT}/data"

CONDA_PREFIX="${CONDA_PREFIX:-${PROJECT_ROOT}/conda/envs/protenix}"
export CUDA_HOME="${CONDA_PREFIX}"
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
export TRIANGLE_ATTENTION="cuequivariance"
export TRIANGLE_MULTIPLICATIVE="cuequivariance"

echo "========================================================"
echo "  RNA Template v3 — Online Mode GPU Validation"
echo "========================================================"

# --- Phase 1: Unit tests ---
echo ""
echo "=== Phase 1: Online featurizer unit tests ==="
python3 rna_template/scripts/test_online_featurizer.py
UNIT_EXIT=$?

if [ ${UNIT_EXIT} -ne 0 ]; then
    echo "FAIL: Unit tests failed. Aborting GPU test."
    exit 1
fi
echo "Unit tests: ALL PASSED"

# --- Phase 2: GPU training with online mode ---
echo ""
echo "=== Phase 2: GPU training with online mode (${MAX_STEPS} steps) ==="

CHECKPOINT_PATH="${PROTENIX_DIR}/checkpoints/protenix_base_20250630_v1.0.0.pt"
RNA_CIF_DIR="${DATA_DIR}/PDB_RNA"
BIOASSEMBLY_DIR="${PREPARED_DATA_DIR}/rna_bioassembly"
SEARCH_RESULTS_PATH="${RNA_DATABASE_DIR}/search_results.json"
RNA3DB_METADATA="${PROJECT_ROOT}/data/RNA3D/rna3db-jsons/filter.json"
INDICES_FPATH="${PREPARED_DATA_DIR}/indices/rna_bioassembly_indices.csv"
VAL_PDB_LIST="${PREPARED_DATA_DIR}/rna_val_pdb_list_filtered.txt"
OUTPUT_DIR="${PROTENIX_DIR}/output/test_online_mode"

# Validate required files
for f in "${CHECKPOINT_PATH}" "${SEARCH_RESULTS_PATH}" "${RNA3DB_METADATA}" "${VAL_PDB_LIST}" "${INDICES_FPATH}"; do
    [ -f "$f" ] || { echo "Missing: $f"; exit 1; }
done
[ -d "${RNA_CIF_DIR}" ] || { echo "Missing CIF dir: ${RNA_CIF_DIR}"; exit 1; }
[ -d "${BIOASSEMBLY_DIR}" ] || { echo "Missing bioassembly dir: ${BIOASSEMBLY_DIR}"; exit 1; }

mkdir -p "${OUTPUT_DIR}"

TRAIN_SET="weightedPDB_before2109_wopb_nometalc_0925"
VAL_SET="recentPDB_1536_sample384_0925"

python3 ./runner/train.py \
    --model_name "protenix_base_20250630_v1.0.0" \
    --run_name "test_online_mode" \
    --seed 42 \
    --base_dir "${OUTPUT_DIR}" \
    --dtype "bf16" \
    --project "test_online" \
    --use_wandb false \
    --diffusion_batch_size 8 \
    --eval_interval 999999 \
    --log_interval 5 \
    --checkpoint_interval 999999 \
    --train_crop_size 128 \
    --max_steps "${MAX_STEPS}" \
    --lr 0.0001 \
    --lr_scheduler "af3" \
    --warmup_steps 3 \
    --grad_clip_norm 10 \
    --model.N_cycle 1 \
    --sample_diffusion.N_step 5 \
    --load_checkpoint_path "${CHECKPOINT_PATH}" \
    --load_ema_checkpoint_path "${CHECKPOINT_PATH}" \
    --load_strict false \
    --data.num_dl_workers 0 \
    --ema_decay 0.999 \
    --adam.use_adamw true \
    --adam.beta1 0.9 \
    --adam.beta2 0.999 \
    --loss.weight.alpha_bond 0.5 \
    --model.confidence_head.stop_gradient true \
    \
    --two_stage.enable false \
    \
    --rna_template.enable true \
    --rna_template.template_database_dir "${RNA_DATABASE_DIR}" \
    --rna_template.max_rna_templates 4 \
    --rna_template.rna3db_metadata_path "${RNA3DB_METADATA}" \
    --rna_template.projector_init "protein" \
    --rna_template.alpha_init 0.01 \
    --rna_template.search_results_path "${SEARCH_RESULTS_PATH}" \
    --rna_template.cif_database_dir "${RNA_CIF_DIR}" \
    --model.template_embedder.n_blocks 2 \
    \
    --rnalm.enable false \
    \
    --data.train_sets "${TRAIN_SET}" \
    --data.${TRAIN_SET}.base_info.mmcif_dir "${RNA_CIF_DIR}" \
    --data.${TRAIN_SET}.base_info.bioassembly_dict_dir "${BIOASSEMBLY_DIR}" \
    --data.${TRAIN_SET}.base_info.indices_fpath "${INDICES_FPATH}" \
    --data.${TRAIN_SET}.base_info.pdb_list "${VAL_PDB_LIST}" \
    --data.test_sets "${VAL_SET}" \
    --data.${VAL_SET}.base_info.mmcif_dir "${RNA_CIF_DIR}" \
    --data.${VAL_SET}.base_info.bioassembly_dict_dir "${BIOASSEMBLY_DIR}" \
    --data.${VAL_SET}.base_info.indices_fpath "${INDICES_FPATH}" \
    --data.${VAL_SET}.base_info.pdb_list "${VAL_PDB_LIST}" \
    --data.${VAL_SET}.base_info.find_eval_chain_interface false \
    --data.${VAL_SET}.base_info.group_by_pdb_id false \
    --data.${VAL_SET}.base_info.max_n_token 512 \
    --data.msa.enable_rna_msa true \
    --data.msa.rna_msadir_raw_paths "${PREPARED_DATA_DIR}/rna_msa/msas" \
    --data.msa.rna_seq_or_filename_to_msadir_jsons "${PREPARED_DATA_DIR}/rna_msa/rna_sequence_to_pdb_chains.json" \
    --data.msa.rna_indexing_methods "sequence" \
    --data.msa.enable_prot_msa false \
    --data.template.enable_prot_template false \
    2>&1 | tee "${OUTPUT_DIR}/training.log"
TRAIN_EXIT=$?

echo ""
if [ ${TRAIN_EXIT} -eq 0 ]; then
    echo "GPU training with ONLINE mode: PASSED (${MAX_STEPS} steps)"
else
    echo "GPU training with ONLINE mode: FAILED (exit code ${TRAIN_EXIT})"
fi

echo ""
echo "========================================================"
echo "  Result: Unit tests PASSED, GPU training exit=${TRAIN_EXIT}"
echo "========================================================"

exit ${TRAIN_EXIT}
