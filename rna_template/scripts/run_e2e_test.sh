#!/bin/bash
# =============================================================================
# RNA Template E2E Test Pipeline
#
# Full end-to-end test: catalog → pairwise search → build templates → index
# → GPU training validation. Designed to verify the entire RNA template
# pipeline works correctly with a small subset of data.
#
# Steps:
#   1. Extract RNA catalog from PDB_RNA CIF files (or reuse existing)
#   2. Pairwise alignment search for templates
#   3. Build template .npz files from search results
#   4. Build template index (seq → .npz mapping)
#   5. Run GPU training validation (1-2 steps)
#
# ---- CONFIGURABLE INTERFACE ----
# To swap the search backend later, replace Step 2 with your own search
# script that produces the same search_results.json format:
#   {query_id: {"query_sequence": str, "templates": [{pdb_id, chain_id, identity}]}}
# ---- END CONFIGURABLE ----
#
# Usage:
#   bash run_e2e_test.sh                          # Default: 5 PDBs, 2 train steps
#   bash run_e2e_test.sh --n_test_pdbs 10         # More PDBs
#   bash run_e2e_test.sh --max_train_steps 5      # More train steps
#   bash run_e2e_test.sh --skip_pipeline          # Only run GPU test
#   bash run_e2e_test.sh --skip_gpu_test          # Only run pipeline
#   bash run_e2e_test.sh --reuse_catalog          # Skip catalog extraction
#   bash run_e2e_test.sh --search_strategy pairwise  # Search algorithm
# =============================================================================
set -eo pipefail

# ===================== Activate Environment =====================
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate protenix
set -u

# ===================== Paths =====================
PROJECT_ROOT="/inspire/ssd/project/sais-bio/public/ash_proj"
PROTENIX_DIR="${PROJECT_ROOT}/code/protenix_new/Protenix"
DATA_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/part2"
PREPARED_DATA_DIR="${DATA_DIR}/protenix_prepared"
PDB_RNA_DIR="${DATA_DIR}/PDB_RNA"
SCRIPTS_DIR="${PROTENIX_DIR}/rna_template/scripts"

# Test output directory (separate from production rna_database)
TEST_OUTPUT_DIR="${PROTENIX_DIR}/rna_database_test"
TEST_TEMPLATE_DIR="${TEST_OUTPUT_DIR}/templates"
TEST_CATALOG="${TEST_OUTPUT_DIR}/rna_catalog.json"
TEST_SEARCH_RESULTS="${TEST_OUTPUT_DIR}/search_results.json"
TEST_INDEX="${TEST_OUTPUT_DIR}/rna_template_index.json"
TEST_PDB_LIST="${TEST_OUTPUT_DIR}/test_pdb_list.txt"

# Training resources
CHECKPOINT_PATH="${PROTENIX_DIR}/checkpoints/protenix_base_20250630_v1.0.0.pt"
RNA_CIF_DIR="${PDB_RNA_DIR}"
BIOASSEMBLY_DIR="${PREPARED_DATA_DIR}/rna_bioassembly"
INDICES_FILE="${PREPARED_DATA_DIR}/indices/rna_bioassembly_indices.csv"
TRAIN_PDB_LIST="${PREPARED_DATA_DIR}/rna_train_pdb_list_filtered.txt"
VAL_PDB_LIST="${PREPARED_DATA_DIR}/rna_val_pdb_list_filtered.txt"
RNA_MSA_DIR="${PREPARED_DATA_DIR}/rna_msa/msas"
RNA_SEQ_TO_PDB_JSON="${PREPARED_DATA_DIR}/rna_msa/rna_sequence_to_pdb_chains.json"

# ===================== Test Parameters =====================
N_TEST_PDBS=5                 # Number of small PDBs to test with
MAX_TRAIN_STEPS=2             # Training steps for GPU validation
TRAIN_CROP_SIZE=64            # Small crop for fast testing
MAX_TEMPLATES=4               # Max RNA templates per chain
MIN_IDENTITY=0.3              # Min sequence identity for search
SEARCH_STRATEGY="pairwise"    # ---- CONFIGURABLE: pairwise | mmseqs2 | custom ----
SKIP_PIPELINE=false
SKIP_GPU_TEST=false
REUSE_CATALOG=false
REUSE_EXISTING_DB=false       # Use production rna_database instead of test
NUM_WORKERS=4

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_test_pdbs)        N_TEST_PDBS="$2";         shift 2 ;;
        --max_train_steps)    MAX_TRAIN_STEPS="$2";     shift 2 ;;
        --train_crop_size)    TRAIN_CROP_SIZE="$2";     shift 2 ;;
        --max_templates)      MAX_TEMPLATES="$2";       shift 2 ;;
        --min_identity)       MIN_IDENTITY="$2";        shift 2 ;;
        --search_strategy)    SEARCH_STRATEGY="$2";     shift 2 ;;
        --skip_pipeline)      SKIP_PIPELINE=true;       shift ;;
        --skip_gpu_test)      SKIP_GPU_TEST=true;       shift ;;
        --reuse_catalog)      REUSE_CATALOG=true;       shift ;;
        --reuse_existing_db)  REUSE_EXISTING_DB=true;   shift ;;
        --num_workers)        NUM_WORKERS="$2";         shift 2 ;;
        --pdb_rna_dir)        PDB_RNA_DIR="$2";         shift 2 ;;
        --output_dir)         TEST_OUTPUT_DIR="$2";     shift 2 ;;
        *)
            echo "ERROR: Unknown argument: $1"
            exit 1 ;;
    esac
done

# If reusing existing DB, point to production rna_database
if [ "${REUSE_EXISTING_DB}" = true ]; then
    TEST_OUTPUT_DIR="${PROTENIX_DIR}/rna_database"
    TEST_TEMPLATE_DIR="${TEST_OUTPUT_DIR}/templates"
    TEST_CATALOG="${TEST_OUTPUT_DIR}/rna_catalog.json"
    TEST_SEARCH_RESULTS="${TEST_OUTPUT_DIR}/search_results.json"
    TEST_INDEX="${TEST_OUTPUT_DIR}/rna_template_index.json"
fi

# Update derived paths
TEST_TEMPLATE_DIR="${TEST_OUTPUT_DIR}/templates"
TEST_CATALOG="${TEST_OUTPUT_DIR}/rna_catalog.json"
TEST_SEARCH_RESULTS="${TEST_OUTPUT_DIR}/search_results.json"
TEST_INDEX="${TEST_OUTPUT_DIR}/rna_template_index.json"
TEST_PDB_LIST="${TEST_OUTPUT_DIR}/test_pdb_list.txt"

GPU_TEST_OUTPUT="${PROTENIX_DIR}/output/e2e_rna_template_test"

echo "========================================================"
echo "  RNA Template E2E Test Pipeline"
echo "  PDB_RNA dir:        ${PDB_RNA_DIR}"
echo "  Test output:        ${TEST_OUTPUT_DIR}"
echo "  N test PDBs:        ${N_TEST_PDBS}"
echo "  Search strategy:    ${SEARCH_STRATEGY}"
echo "  Max train steps:    ${MAX_TRAIN_STEPS}"
echo "  Train crop size:    ${TRAIN_CROP_SIZE}"
echo "  Skip pipeline:      ${SKIP_PIPELINE}"
echo "  Skip GPU test:      ${SKIP_GPU_TEST}"
echo "  Reuse catalog:      ${REUSE_CATALOG}"
echo "  Reuse existing DB:  ${REUSE_EXISTING_DB}"
echo "========================================================"

mkdir -p "${TEST_OUTPUT_DIR}" "${TEST_TEMPLATE_DIR}"

# ===================== Pipeline =====================
if [ "${SKIP_PIPELINE}" = false ] && [ "${REUSE_EXISTING_DB}" = false ]; then

    # --- Step 0: Select small test PDBs ---
    echo ""
    echo "=== Step 0: Selecting ${N_TEST_PDBS} small test PDBs ==="
    python3 "${SCRIPTS_DIR}/select_test_pdbs.py" \
        --train_pdb_list "${TRAIN_PDB_LIST}" \
        --indices_file "${INDICES_FILE}" \
        --cif_dir "${RNA_CIF_DIR}" \
        --bioassembly_dir "${BIOASSEMBLY_DIR}" \
        --output "${TEST_PDB_LIST}" \
        --n_pdbs "${N_TEST_PDBS}" \
        --max_tokens 100

    echo "Selected PDBs:"
    cat "${TEST_PDB_LIST}"

    # --- Step 1: Extract RNA catalog ---
    if [ "${REUSE_CATALOG}" = false ]; then
        echo ""
        echo "=== Step 1: Extracting RNA Catalog ==="
        # Use the full PDB_RNA database as the template source
        python3 "${SCRIPTS_DIR}/01_extract_rna_catalog.py" \
            --pdb_rna_dir "${PDB_RNA_DIR}" \
            --output "${TEST_CATALOG}" \
            --max_structures 200 \
            --min_length 10 \
            --max_length 500 \
            --num_workers "${NUM_WORKERS}"
    else
        if [ ! -f "${TEST_CATALOG}" ]; then
            # Copy from production
            PROD_CATALOG="${PROTENIX_DIR}/rna_database/rna_catalog.json"
            if [ -f "${PROD_CATALOG}" ]; then
                echo "Copying production catalog..."
                cp "${PROD_CATALOG}" "${TEST_CATALOG}"
            else
                echo "ERROR: No catalog found to reuse"
                exit 1
            fi
        fi
        echo "Reusing existing catalog: ${TEST_CATALOG}"
    fi

    # --- Step 2: Pairwise alignment search ---
    echo ""
    echo "=== Step 2: Template Search (strategy=${SEARCH_STRATEGY}) ==="
    python3 "${SCRIPTS_DIR}/03_search_and_index.py" \
        --catalog "${TEST_CATALOG}" \
        --template_dir "${TEST_TEMPLATE_DIR}" \
        --output_index "${TEST_INDEX}.tmp" \
        --output_search "${TEST_SEARCH_RESULTS}" \
        --strategy "${SEARCH_STRATEGY}" \
        --training_pdb_list "${TEST_PDB_LIST}" \
        --min_identity "${MIN_IDENTITY}" \
        --max_templates "${MAX_TEMPLATES}"

    # --- Step 3: Build template .npz files ---
    echo ""
    echo "=== Step 3: Building Template .npz Files ==="
    python3 "${SCRIPTS_DIR}/02_build_rna_templates.py" \
        --catalog "${TEST_CATALOG}" \
        --pdb_rna_dir "${PDB_RNA_DIR}" \
        --output_dir "${TEST_TEMPLATE_DIR}" \
        --mode cross \
        --search_results "${TEST_SEARCH_RESULTS}" \
        --max_templates "${MAX_TEMPLATES}"

    # --- Step 4: Rebuild index after templates exist ---
    echo ""
    echo "=== Step 4: Building Template Index ==="
    python3 "${SCRIPTS_DIR}/03_search_and_index.py" \
        --catalog "${TEST_CATALOG}" \
        --template_dir "${TEST_TEMPLATE_DIR}" \
        --output_index "${TEST_INDEX}" \
        --strategy "${SEARCH_STRATEGY}" \
        --training_pdb_list "${TEST_PDB_LIST}" \
        --min_identity "${MIN_IDENTITY}" \
        --max_templates "${MAX_TEMPLATES}"

    rm -f "${TEST_INDEX}.tmp"

    # --- Pipeline Summary ---
    echo ""
    echo "=== Pipeline Results ==="
    N_CATALOG=$(python3 -c "import json; d=json.load(open('${TEST_CATALOG}')); print(len(d))" 2>/dev/null || echo "?")
    N_TEMPLATES=$(find "${TEST_TEMPLATE_DIR}" -name "*.npz" 2>/dev/null | wc -l || echo "0")
    N_INDEX=$(python3 -c "import json; d=json.load(open('${TEST_INDEX}')); print(len(d))" 2>/dev/null || echo "?")
    echo "  Catalog entries:     ${N_CATALOG}"
    echo "  Template .npz files: ${N_TEMPLATES}"
    echo "  Index sequences:     ${N_INDEX}"

    if [ "${N_INDEX}" = "0" ] || [ "${N_INDEX}" = "?" ]; then
        echo ""
        echo "WARNING: Index is empty. No templates were found for the test PDBs."
        echo "This may be expected if the test PDBs have no similar sequences in the database."
        echo "The GPU test will still run but RNA template features will be all zeros."
    fi
fi

# ===================== GPU Training Test =====================
if [ "${SKIP_GPU_TEST}" = false ]; then
    echo ""
    echo "========================================================"
    echo "  GPU Training Validation Test"
    echo "========================================================"

    # Determine which database dir & index to use
    if [ "${REUSE_EXISTING_DB}" = true ]; then
        RNA_DB_DIR="${PROTENIX_DIR}/rna_database"
        RNA_IDX="${PROTENIX_DIR}/rna_database/rna_template_index.json"
    else
        RNA_DB_DIR="${TEST_OUTPUT_DIR}"
        RNA_IDX="${TEST_INDEX}"
    fi

    # Validate prerequisites
    [ -f "${CHECKPOINT_PATH}" ] || { echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"; exit 1; }
    [ -d "${RNA_DB_DIR}" ] || { echo "ERROR: RNA database dir not found: ${RNA_DB_DIR}"; exit 1; }
    [ -f "${RNA_IDX}" ] || { echo "ERROR: RNA template index not found: ${RNA_IDX}"; exit 1; }

    cd "${PROTENIX_DIR}"
    export PYTHONPATH="${PROTENIX_DIR}:${PYTHONPATH:-}"
    export PROTENIX_ROOT_DIR="${PROJECT_ROOT}/data"

    CONDA_PREFIX="${PROJECT_ROOT}/conda/envs/protenix"
    export CUDA_HOME="${CONDA_PREFIX}"
    export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
    export TRIANGLE_ATTENTION="cuequivariance"
    export TRIANGLE_MULTIPLICATIVE="cuequivariance"

    TRAIN_SET="weightedPDB_before2109_wopb_nometalc_0925"
    VAL_SET="recentPDB_1536_sample384_0925"

    # Use test PDB list if we built one, otherwise use first few from training
    if [ -f "${TEST_PDB_LIST}" ]; then
        GPU_PDB_LIST="${TEST_PDB_LIST}"
    else
        GPU_PDB_LIST="${TRAIN_PDB_LIST}"
    fi

    mkdir -p "${GPU_TEST_OUTPUT}"

    echo "Running GPU training test..."
    echo "  RNA database:    ${RNA_DB_DIR}"
    echo "  RNA index:       ${RNA_IDX}"
    echo "  PDB list:        ${GPU_PDB_LIST}"
    echo "  Max steps:       ${MAX_TRAIN_STEPS}"
    echo "  Crop size:       ${TRAIN_CROP_SIZE}"

    python3 ./runner/train.py \
        --model_name "protenix_base_20250630_v1.0.0" \
        --run_name "e2e_rna_template_test" \
        --seed 42 \
        --base_dir "${GPU_TEST_OUTPUT}" \
        --dtype "bf16" \
        --project "e2e_test" \
        --use_wandb false \
        --diffusion_batch_size 48 \
        --eval_interval 999999 \
        --log_interval 1 \
        --checkpoint_interval 999999 \
        --train_crop_size "${TRAIN_CROP_SIZE}" \
        --max_steps "${MAX_TRAIN_STEPS}" \
        --lr 0.0001 \
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
        --rnalm.enable false \
        \
        --rna_template.enable true \
        --rna_template.template_database_dir "${RNA_DB_DIR}" \
        --rna_template.template_index_path "${RNA_IDX}" \
        --rna_template.max_rna_templates "${MAX_TEMPLATES}" \
        --rna_template.projector_init "protein" \
        --rna_template.alpha_init 0.01 \
        --model.template_embedder.n_blocks 2 \
        \
        --data.train_sets "${TRAIN_SET}" \
        --data.${TRAIN_SET}.base_info.mmcif_dir "${RNA_CIF_DIR}" \
        --data.${TRAIN_SET}.base_info.bioassembly_dict_dir "${BIOASSEMBLY_DIR}" \
        --data.${TRAIN_SET}.base_info.indices_fpath "${INDICES_FILE}" \
        --data.${TRAIN_SET}.base_info.pdb_list "${GPU_PDB_LIST}" \
        --data.test_sets "${VAL_SET}" \
        --data.${VAL_SET}.base_info.mmcif_dir "${RNA_CIF_DIR}" \
        --data.${VAL_SET}.base_info.bioassembly_dict_dir "${BIOASSEMBLY_DIR}" \
        --data.${VAL_SET}.base_info.indices_fpath "${INDICES_FILE}" \
        --data.${VAL_SET}.base_info.pdb_list "${GPU_PDB_LIST}" \
        --data.${VAL_SET}.base_info.find_eval_chain_interface false \
        --data.${VAL_SET}.base_info.group_by_pdb_id false \
        --data.${VAL_SET}.base_info.max_n_token 256 \
        --data.msa.enable_rna_msa true \
        --data.msa.rna_msadir_raw_paths "${RNA_MSA_DIR}" \
        --data.msa.rna_seq_or_filename_to_msadir_jsons "${RNA_SEQ_TO_PDB_JSON}" \
        --data.msa.rna_indexing_methods "sequence" \
        --data.msa.enable_prot_msa false \
        --data.template.enable_prot_template false

    GPU_EXIT=$?

    echo ""
    if [ ${GPU_EXIT} -eq 0 ]; then
        echo "========================================================"
        echo "  [SUCCESS] GPU Training Validation PASSED!"
        echo "  RNA template pipeline is fully functional."
        echo "========================================================"
    else
        echo "========================================================"
        echo "  [FAILED] GPU Training Validation FAILED (exit code: ${GPU_EXIT})"
        echo "  Check logs above for details."
        echo "========================================================"
        exit ${GPU_EXIT}
    fi
fi

echo ""
echo "E2E test complete."
echo "  Pipeline output:  ${TEST_OUTPUT_DIR}"
echo "  GPU test output:  ${GPU_TEST_OUTPUT}"
