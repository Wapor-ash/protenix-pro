#!/bin/bash
# =============================================================================
# RNA3D Template E2E Test
#
# Validates the full rna3db-mmcifs → Arena → MMseqs2 → Template → Finetune pipeline.
#
# Steps:
#   1. Extract catalog from rna3db-mmcifs (small subset)
#   2. MMseqs2 search for cross-template matches
#   3. Build cross-templates from search results
#   4. Build template index
#   5. Validate cross-template NPZ files
#   6. Run short GPU training test
#
# Usage:
#   bash test_rna3d_e2e.sh
#   bash test_rna3d_e2e.sh --num_test 20 --max_steps 50
#   bash test_rna3d_e2e.sh --skip_training   # Only test template pipeline
# =============================================================================
set -eo pipefail

# ===================== Paths =====================
PROJECT_ROOT="/inspire/ssd/project/sais-bio/public/ash_proj"
PROTENIX_DIR="${PROJECT_ROOT}/code/protenix_new/Protenix"
DATA_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/part2"
RNA3D_DIR="${PROJECT_ROOT}/data/RNA3D/rna3db-mmcifs"
PREPARED_DATA_DIR="${DATA_DIR}/protenix_prepared"
ARENA_BINARY="${PROJECT_ROOT}/Arena/Arena"
SCRIPTS_DIR="${PROTENIX_DIR}/rna_template/scripts"

# Test output directory (separate from production)
TEST_DIR="${PROTENIX_DIR}/rna_database_e2e_test"
TEST_TEMPLATE_DIR="${TEST_DIR}/cross_templates"
TEST_CATALOG="${TEST_DIR}/rna_catalog.json"
TEST_INDEX="${TEST_DIR}/rna_template_index.json"
TEST_SEARCH_RESULTS="${TEST_DIR}/search_results.json"
TEST_TRAINING_SEQ_JSON="${TEST_DIR}/test_training_sequences.json"
TEST_OUTPUT_DIR="${PROTENIX_DIR}/output/rna3d_e2e_test"

# ===================== Parameters =====================
NUM_TEST=30            # Number of test structures
MAX_STEPS=20           # Training steps for validation
TRAIN_CROP_SIZE=128    # Small crop for speed
MAX_RNA_TEMPLATES=2    # Fewer templates for speed
SKIP_TRAINING=false
USE_ARENA=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num_test)         NUM_TEST="$2";         shift 2 ;;
        --max_steps)        MAX_STEPS="$2";        shift 2 ;;
        --train_crop_size)  TRAIN_CROP_SIZE="$2";  shift 2 ;;
        --skip_training)    SKIP_TRAINING=true;    shift ;;
        --no_arena)         USE_ARENA=false;       shift ;;
        *)
            echo "ERROR: Unknown argument: $1"; exit 1 ;;
    esac
done

# Activate conda environment (try multiple locations)
eval "$(conda shell.bash hook 2>/dev/null)"
if conda activate protenix 2>/dev/null; then
    echo "  Activated conda env: protenix"
elif conda activate "${PROJECT_ROOT}/conda/envs/protenix" 2>/dev/null; then
    echo "  Activated conda env: ${PROJECT_ROOT}/conda/envs/protenix"
elif conda activate "${PROJECT_ROOT}/conda/envs/r1126_rna" 2>/dev/null; then
    echo "  Activated conda env: r1126_rna (fallback)"
else
    echo "WARNING: Could not activate conda environment. Using current Python."
fi

echo "========================================================"
echo "  RNA3D Template E2E Test"
echo "  Test structures: ${NUM_TEST}"
echo "  Use Arena:       ${USE_ARENA}"
echo "  Training steps:  ${MAX_STEPS}"
echo "  Crop size:       ${TRAIN_CROP_SIZE}"
echo "========================================================"

# Clean and create test directory
rm -rf "${TEST_DIR}"
mkdir -p "${TEST_DIR}" "${TEST_TEMPLATE_DIR}"

# ===================== Step 1: Extract Catalog =====================
echo ""
echo "=== Step 1: Extracting RNA Catalog from rna3db-mmcifs ==="

python3 "${SCRIPTS_DIR}/01_extract_rna_catalog.py" \
    --pdb_rna_dir "${RNA3D_DIR}" \
    --output "${TEST_CATALOG}" \
    --max_structures "${NUM_TEST}" \
    --min_length 10 \
    --max_length 500 \
    --num_workers 4

if [ ! -f "${TEST_CATALOG}" ]; then
    echo "FAIL: Catalog not created!"
    exit 1
fi

N_CATALOG=$(python3 -c "import json; print(len(json.load(open('${TEST_CATALOG}'))))")
echo "  Catalog: ${N_CATALOG} structures"

# ===================== Step 1.5: Generate Test training_sequences JSON =====================
# Bug fix: production uses training_sequences JSON ({sequence: [pdb_id, ...]}) as query source,
# but this test previously used catalog directly. Generate a JSON in production format
# from the catalog to validate the same code path.
echo ""
echo "=== Step 1.5: Generating Test training_sequences JSON (production format) ==="

python3 -c "
import json, sys

catalog = json.load(open('${TEST_CATALOG}'))
# Convert catalog to production format: {sequence: [pdb_id, ...]}
seq_to_pdbs = {}
for entry_id, chains in catalog.items():
    base_pdb = entry_id.split('_')[0].upper()
    for chain in chains:
        seq = chain.get('sequence', '')
        if len(seq) >= 10:
            seq_to_pdbs.setdefault(seq, [])
            if base_pdb not in seq_to_pdbs[seq]:
                seq_to_pdbs[seq].append(base_pdb)

with open('${TEST_TRAINING_SEQ_JSON}', 'w') as f:
    json.dump(seq_to_pdbs, f, indent=2)

print(f'  Generated training_sequences JSON: {len(seq_to_pdbs)} unique sequences')
print(f'  Total PDB mappings: {sum(len(v) for v in seq_to_pdbs.values())}')
"

if [ ! -f "${TEST_TRAINING_SEQ_JSON}" ]; then
    echo "FAIL: Test training_sequences JSON not created!"
    exit 1
fi

# ===================== Step 2: MMseqs2 Search =====================
echo ""
echo "=== Step 2: MMseqs2 Template Search (using training_sequences JSON) ==="

ARENA_ARGS=""
if [ "${USE_ARENA}" = true ]; then
    ARENA_ARGS="--use_arena --arena_binary ${ARENA_BINARY}"
    echo "  Arena atom-filling: ENABLED"
else
    echo "  Arena atom-filling: DISABLED"
fi

# Use training_sequences JSON as query source (same as production path)
python3 "${SCRIPTS_DIR}/03_mmseqs2_search.py" \
    --catalog "${TEST_CATALOG}" \
    --template_dir "${TEST_TEMPLATE_DIR}" \
    --training_sequences "${TEST_TRAINING_SEQ_JSON}" \
    --output_index "${TEST_INDEX}.tmp" \
    --output_search "${TEST_SEARCH_RESULTS}" \
    --strategy mmseqs2 \
    --min_identity 0.3 \
    --max_templates "${MAX_RNA_TEMPLATES}" \
    --num_threads 4

if [ ! -f "${TEST_SEARCH_RESULTS}" ]; then
    echo "FAIL: Search results not created!"
    exit 1
fi

N_SEARCH=$(python3 -c "import json; print(len(json.load(open('${TEST_SEARCH_RESULTS}'))))")
echo "  Queries with search hits: ${N_SEARCH}"

if [ "${N_SEARCH}" -eq 0 ]; then
    echo "FAIL: No cross-template hits found. Cross-only test stops here by design."
    exit 1
fi

# ===================== Step 3: Build Cross-Templates =====================
echo ""
echo "=== Step 3: Building Cross-Templates from Search Results ==="

python3 "${SCRIPTS_DIR}/02_build_rna_templates.py" \
    --catalog "${TEST_CATALOG}" \
    --pdb_rna_dir "${RNA3D_DIR}" \
    --output_dir "${TEST_TEMPLATE_DIR}" \
    --mode cross \
    --search_results "${TEST_SEARCH_RESULTS}" \
    --max_templates "${MAX_RNA_TEMPLATES}" \
    ${ARENA_ARGS}

N_CROSS=$(find "${TEST_TEMPLATE_DIR}" -name "*.npz" | wc -l)
echo "  Cross-templates built: ${N_CROSS}"

if [ "${N_CROSS}" -eq 0 ]; then
    echo "FAIL: No cross-templates were built. Cross-only test does not fall back to self-templates."
    exit 1
fi

# ===================== Step 4: Rebuild Index After Cross-Templates =====================
echo ""
echo "=== Step 4: Building Template Index ==="

python3 "${SCRIPTS_DIR}/03_mmseqs2_search.py" \
    --catalog "${TEST_CATALOG}" \
    --template_dir "${TEST_TEMPLATE_DIR}" \
    --training_sequences "${TEST_TRAINING_SEQ_JSON}" \
    --output_index "${TEST_INDEX}" \
    --output_search "${TEST_SEARCH_RESULTS}" \
    --strategy mmseqs2 \
    --skip_search \
    --num_threads 4

rm -f "${TEST_INDEX}.tmp"

if [ ! -f "${TEST_INDEX}" ]; then
    echo "FAIL: Index not created!"
    exit 1
fi

N_INDEX=$(python3 -c "import json; print(len(json.load(open('${TEST_INDEX}'))))")
echo "  Sequences in index: ${N_INDEX}"

# ===================== Step 5: NPZ Validation =====================
echo ""
echo "=== Step 5: NPZ Validation (Cross templates only) ==="
python3 -c "
import numpy as np
import glob
import os

template_dir = '${TEST_TEMPLATE_DIR}'
npz_files = sorted(glob.glob(os.path.join(template_dir, '*.npz')))

required_keys = [
    'template_aatype', 'template_distogram',
    'template_pseudo_beta_mask', 'template_unit_vector',
    'template_backbone_frame_mask',
]

total = len(npz_files)
passed = 0
failed = 0

for npz_path in npz_files:
    name = os.path.basename(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    missing = [k for k in required_keys if k not in keys]
    if missing:
        print(f'  FAIL {name}: missing keys {missing}')
        failed += 1
    else:
        aatype = data['template_aatype']
        T, N = aatype.shape
        dist = data['template_distogram']
        mask = data['template_pseudo_beta_mask']
        uv = data['template_unit_vector']
        frame = data['template_backbone_frame_mask']

        # Shape checks
        ok = True
        if dist.shape != (T, N, N, 39):
            print(f'  FAIL {name}: distogram shape {dist.shape}')
            ok = False
        if mask.shape != (T, N, N):
            print(f'  FAIL {name}: mask shape {mask.shape}')
            ok = False
        if uv.shape != (T, N, N, 3):
            print(f'  FAIL {name}: uv shape {uv.shape}')
            ok = False
        if frame.shape != (T, N, N):
            print(f'  FAIL {name}: frame shape {frame.shape}')
            ok = False

        if ok:
            coverage = mask[0].sum() / max(N*N, 1) * 100
            passed += 1
        else:
            failed += 1

print(f'')
print(f'NPZ Validation: {passed}/{total} passed, {failed}/{total} failed')
if failed == 0:
    print('ALL NPZ CHECKS PASSED!')
else:
    print('SOME NPZ CHECKS FAILED!')
    exit(1)
"

if [ "${SKIP_TRAINING}" = true ]; then
    echo ""
    echo "========================================================"
    echo "  PIPELINE TEST PASSED (training skipped)"
    echo "  Catalog:        ${N_CATALOG} structures"
    echo "  Cross-templates: ${N_CROSS} NPZ files"
    echo "  Index:           ${N_INDEX} sequences"
    echo "========================================================"
    exit 0
fi

# ===================== Step 6: GPU Training Test =====================
echo ""
echo "=== Step 6: GPU Training Test ==="

CHECKPOINT_PATH="${PROTENIX_DIR}/checkpoints/protenix_base_20250630_v1.0.0.pt"
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "WARNING: Checkpoint not found: ${CHECKPOINT_PATH}"
    CHECKPOINT_PATH=$(find "${PROTENIX_DIR}/checkpoints" -name "*.pt" -type f 2>/dev/null | head -1)
    if [ -z "${CHECKPOINT_PATH}" ]; then
        echo "No checkpoint found. Pipeline test passed, training skipped."
        exit 0
    fi
fi

cd "${PROTENIX_DIR}"
export PYTHONPATH="${PROTENIX_DIR}:${PYTHONPATH:-}"
export PROTENIX_ROOT_DIR="${PROJECT_ROOT}/data"

# Use CONDA_PREFIX from the activated env, fallback to known locations
if [ -z "${CONDA_PREFIX:-}" ]; then
    if [ -d "/opt/conda/envs/protenix" ]; then
        CONDA_PREFIX="/opt/conda/envs/protenix"
    else
        CONDA_PREFIX="${PROJECT_ROOT}/conda/envs/protenix"
    fi
fi
export CUDA_HOME="${CONDA_PREFIX}"
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"

# Training data paths
BIOASSEMBLY_DIR="${PREPARED_DATA_DIR}/rna_bioassembly"
INDICES_FPATH="${PREPARED_DATA_DIR}/indices/rna_bioassembly_indices.csv"

# Use a subset of val PDBs for both train and test
VAL_PDB_LIST="${PREPARED_DATA_DIR}/rna_val_pdb_list.txt"

TRAIN_SET="weightedPDB_before2109_wopb_nometalc_0925"
VAL_SET="recentPDB_1536_sample384_0925"
RNA_CIF_DIR="${DATA_DIR}/PDB_RNA"

mkdir -p "${TEST_OUTPUT_DIR}"

echo "Starting training test (${MAX_STEPS} steps)..."
echo "  RNA template DB: ${TEST_DIR}"
echo "  Index:          ${TEST_INDEX}"

python3 ./runner/train.py \
    --model_name "protenix_base_20250630_v1.0.0" \
    --run_name "rna3d_e2e_test" \
    --seed 42 \
    --base_dir "${TEST_OUTPUT_DIR}" \
    --dtype "bf16" \
    --project "rna3d_e2e_test" \
    --use_wandb false \
    --diffusion_batch_size 8 \
    --eval_interval 999999 \
    --log_interval 5 \
    --checkpoint_interval 999999 \
    --train_crop_size "${TRAIN_CROP_SIZE}" \
    --max_steps "${MAX_STEPS}" \
    --lr 0.0001 \
    --lr_scheduler "af3" \
    --warmup_steps 5 \
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
    --rna_template.template_database_dir "${TEST_DIR}" \
    --rna_template.template_index_path "${TEST_INDEX}" \
    --rna_template.max_rna_templates "${MAX_RNA_TEMPLATES}" \
    --rna_template.projector_init "protein" \
    --rna_template.alpha_init 0.01 \
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
    --data.msa.enable_rna_msa false \
    --data.msa.enable_prot_msa false \
    --data.msa.rna_seq_or_filename_to_msadir_jsons "${PREPARED_DATA_DIR}/rna_msa/rna_sequence_to_pdb_chains.json" \
    --data.msa.rna_msadir_raw_paths "${PREPARED_DATA_DIR}/rna_msa/msas" \
    --data.template.enable_prot_template false \
    2>&1 | tee "${TEST_OUTPUT_DIR}/training_test.log"

TRAIN_EXIT=$?
echo ""
if [ ${TRAIN_EXIT} -eq 0 ]; then
    echo "========================================================"
    echo "  RNA3D E2E TEST PASSED!"
    echo "  Pipeline: catalog → MMseqs2 → cross-templates → index → train"
    echo "  Training: ${MAX_STEPS} steps completed"
    echo "  Catalog:        ${N_CATALOG} structures"
    echo "  Cross-templates: ${N_CROSS} NPZ files"
    echo "  Index:           ${N_INDEX} sequences"
    echo "========================================================"
else
    echo "========================================================"
    echo "  RNA3D E2E TEST FAILED (exit code: ${TRAIN_EXIT})"
    echo "  Check log: ${TEST_OUTPUT_DIR}/training_test.log"
    echo "========================================================"
    exit ${TRAIN_EXIT}
fi
