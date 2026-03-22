#!/bin/bash
# =============================================================================
# RNA Template Pipeline: End-to-End
#
# Chains together all steps:
#   1. Extract RNA catalog from CIF files (supports flat and nested layouts)
#   2. Search for templates using MMseqs2
#   3. Build template .npz files (with optional Arena atom refinement)
#   4. Build template index
#
# The pipeline produces:
#   - rna_catalog.json: Catalog of all RNA chains with sequences
#   - rna_database/templates/*.npz: Template feature files
#   - rna_template_index.json: Index mapping sequences to templates
#   - search_results.json: (mmseqs2 mode) search results with identities
#
# Usage:
#   bash run_pipeline.sh                                    # Cross-template mode
#   bash run_pipeline.sh --use_arena                        # Cross-template mode with Arena
#   bash run_pipeline.sh --max_structures 100               # Test with 100
#   bash run_pipeline.sh --pdb_rna_dir /path/to/rna3db-mmcifs  # Custom data
# =============================================================================
set -eo pipefail

# ===================== Default Paths =====================
PROJECT_ROOT="/inspire/ssd/project/sais-bio/public/ash_proj"
PROTENIX_DIR="${PROJECT_ROOT}/code/protenix_new/Protenix"
DATA_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/part2"
RNA3D_DIR="${PROJECT_ROOT}/data/RNA3D/rna3db-mmcifs"
PREPARED_DATA_DIR="${DATA_DIR}/protenix_prepared"
ARENA_BINARY="${PROJECT_ROOT}/Arena/Arena"

SCRIPTS_DIR="${PROTENIX_DIR}/rna_template/scripts"
RNA_DATABASE_DIR="${PROTENIX_DIR}/rna_database"
TEMPLATE_DIR="${RNA_DATABASE_DIR}/templates"
CATALOG_PATH="${RNA_DATABASE_DIR}/rna_catalog.json"
INDEX_PATH="${RNA_DATABASE_DIR}/rna_template_index.json"
SEARCH_RESULTS_PATH="${RNA_DATABASE_DIR}/search_results.json"

# ===================== Parameters =====================
PDB_RNA_DIR="${RNA3D_DIR}"   # Default: use rna3db-mmcifs
MAX_STRUCTURES=0              # 0 = all
MIN_LENGTH=10
MAX_LENGTH=2000
NUM_WORKERS=8
STRATEGY="mmseqs2"            # cross-template mode only
MAX_TEMPLATES=4
MIN_IDENTITY=0.3
PDB_LIST=""
SKIP_CATALOG=false
SKIP_BUILD=false
SKIP_SEARCH=false
USE_ARENA=false
ARENA_OPTION=5
SENSITIVITY=7.5
EVALUE="1e-3"
RELEASE_DATE_CUTOFF=""
RNA3DB_METADATA=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max_structures)     MAX_STRUCTURES="$2";    shift 2 ;;
        --min_length)         MIN_LENGTH="$2";        shift 2 ;;
        --max_length)         MAX_LENGTH="$2";        shift 2 ;;
        --num_workers)        NUM_WORKERS="$2";       shift 2 ;;
        --strategy)           STRATEGY="$2";          shift 2 ;;
        --max_templates)      MAX_TEMPLATES="$2";     shift 2 ;;
        --min_identity)       MIN_IDENTITY="$2";      shift 2 ;;
        --pdb_list)           PDB_LIST="$2";          shift 2 ;;
        --pdb_rna_dir)        PDB_RNA_DIR="$2";       shift 2 ;;
        --output_dir)         RNA_DATABASE_DIR="$2";  shift 2 ;;
        --arena_binary)       ARENA_BINARY="$2";      shift 2 ;;
        --arena_option)       ARENA_OPTION="$2";      shift 2 ;;
        --sensitivity)        SENSITIVITY="$2";       shift 2 ;;
        --evalue)             EVALUE="$2";            shift 2 ;;
        --release_date_cutoff) RELEASE_DATE_CUTOFF="$2"; shift 2 ;;
        --rna3db_metadata)    RNA3DB_METADATA="$2";   shift 2 ;;
        --skip_catalog)       SKIP_CATALOG=true;      shift ;;
        --skip_build)         SKIP_BUILD=true;        shift ;;
        --skip_search)        SKIP_SEARCH=true;       shift ;;
        --use_arena)          USE_ARENA=true;         shift ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo "Usage: bash run_pipeline.sh [--strategy mmseqs2] [--use_arena] [--max_structures N]"
            exit 1 ;;
    esac
done

# Update derived paths after parsing
TEMPLATE_DIR="${RNA_DATABASE_DIR}/templates"
CATALOG_PATH="${RNA_DATABASE_DIR}/rna_catalog.json"
INDEX_PATH="${RNA_DATABASE_DIR}/rna_template_index.json"
SEARCH_RESULTS_PATH="${RNA_DATABASE_DIR}/search_results.json"

echo "========================================================"
echo "  RNA Template Pipeline"
echo "  PDB_RNA dir:     ${PDB_RNA_DIR}"
echo "  Output dir:      ${RNA_DATABASE_DIR}"
echo "  Strategy:        ${STRATEGY}"
echo "  Use Arena:       ${USE_ARENA}"
echo "  Max structures:  ${MAX_STRUCTURES} (0=all)"
echo "  Max templates:   ${MAX_TEMPLATES}"
echo "  Min identity:    ${MIN_IDENTITY}"
echo "========================================================"

mkdir -p "${RNA_DATABASE_DIR}" "${TEMPLATE_DIR}"

# Build PDB list args
PDB_LIST_ARGS=""
if [ -n "${PDB_LIST}" ]; then
    PDB_LIST_ARGS="--pdb_list ${PDB_LIST}"
fi

# Release date cutoff args
DATE_CUTOFF_ARGS=""
if [ -n "${RELEASE_DATE_CUTOFF}" ]; then
    if [ -z "${RNA3DB_METADATA}" ]; then
        RNA3DB_METADATA="${PROJECT_ROOT}/data/RNA3D/rna3db-jsons/filter.json"
    fi
    DATE_CUTOFF_ARGS="--release_date_cutoff ${RELEASE_DATE_CUTOFF} --rna3db_metadata ${RNA3DB_METADATA}"
fi

# Arena args
ARENA_ARGS=""
if [ "${USE_ARENA}" = true ]; then
    ARENA_ARGS="--use_arena --arena_binary ${ARENA_BINARY} --arena_option ${ARENA_OPTION}"
fi

# ===================== Step 1: Extract Catalog =====================
if [ "${SKIP_CATALOG}" = false ]; then
    echo ""
    echo "=== Step 1: Extracting RNA Catalog ==="
    python3 "${SCRIPTS_DIR}/01_extract_rna_catalog.py" \
        --pdb_rna_dir "${PDB_RNA_DIR}" \
        --output "${CATALOG_PATH}" \
        --max_structures "${MAX_STRUCTURES}" \
        --min_length "${MIN_LENGTH}" \
        --max_length "${MAX_LENGTH}" \
        --num_workers "${NUM_WORKERS}" \
        ${PDB_LIST_ARGS}
else
    echo "Skipping catalog extraction (using existing ${CATALOG_PATH})"
fi

if [ ! -f "${CATALOG_PATH}" ]; then
    echo "ERROR: Catalog not found: ${CATALOG_PATH}"
    exit 1
fi

# ===================== Step 2-4: Cross-template pipeline =====================
if [ "${STRATEGY}" != "mmseqs2" ]; then
    echo "ERROR: Unsupported strategy '${STRATEGY}'. Only 'mmseqs2' cross-template mode is allowed."
    exit 1
fi

# FIXED pipeline order (code review issue #5):
#   Step 2: MMseqs2 search (produces search_results.json)
#   Step 3: Build cross-templates from search results (produces .npz)
#   Step 4: Build index AFTER templates exist on disk

TRAINING_SEQ_JSON="${PREPARED_DATA_DIR}/rna_msa/rna_sequence_to_pdb_chains.json"

# Step 2: MMseqs2 search
if [ "${SKIP_SEARCH}" = false ]; then
    echo ""
    echo "=== Step 2: MMseqs2 Template Search ==="
    python3 "${SCRIPTS_DIR}/03_mmseqs2_search.py" \
        --catalog "${CATALOG_PATH}" \
        --template_dir "${TEMPLATE_DIR}" \
        --output_index "${INDEX_PATH}.tmp" \
        --output_search "${SEARCH_RESULTS_PATH}" \
        --strategy mmseqs2 \
        --training_sequences "${TRAINING_SEQ_JSON}" \
        --min_identity "${MIN_IDENTITY}" \
        --max_templates "${MAX_TEMPLATES}" \
        --sensitivity "${SENSITIVITY}" \
        --evalue "${EVALUE}" \
        --num_threads "${NUM_WORKERS}" \
        ${PDB_LIST_ARGS} \
        ${DATE_CUTOFF_ARGS}
fi

# Step 3: Build cross-templates from search results
if [ "${SKIP_BUILD}" = false ]; then
    echo ""
    echo "=== Step 3: Building Cross-Template .npz Files ==="
    python3 "${SCRIPTS_DIR}/02_build_rna_templates.py" \
        --catalog "${CATALOG_PATH}" \
        --pdb_rna_dir "${PDB_RNA_DIR}" \
        --output_dir "${TEMPLATE_DIR}" \
        --mode cross \
        --search_results "${SEARCH_RESULTS_PATH}" \
        --max_templates "${MAX_TEMPLATES}" \
        ${ARENA_ARGS}
fi

# Step 4: Rebuild index AFTER templates are on disk
echo ""
echo "=== Step 4: Building Template Index ==="
python3 "${SCRIPTS_DIR}/03_mmseqs2_search.py" \
    --catalog "${CATALOG_PATH}" \
    --template_dir "${TEMPLATE_DIR}" \
    --output_index "${INDEX_PATH}" \
    --output_search "${SEARCH_RESULTS_PATH}" \
    --strategy mmseqs2 \
    --training_sequences "${TRAINING_SEQ_JSON}" \
    --min_identity "${MIN_IDENTITY}" \
    --max_templates "${MAX_TEMPLATES}" \
    --skip_search \
    ${PDB_LIST_ARGS} \
    ${DATE_CUTOFF_ARGS}

# Clean up temp index
rm -f "${INDEX_PATH}.tmp"

# ===================== Summary =====================
echo ""
echo "========================================================"
echo "  Pipeline Complete!"
echo "  Catalog:    ${CATALOG_PATH}"
echo "  Templates:  ${TEMPLATE_DIR}/"
echo "  Index:      ${INDEX_PATH}"
echo ""

# Count results
N_CATALOG=$(python3 -c "import json; d=json.load(open('${CATALOG_PATH}')); print(len(d))" 2>/dev/null || echo "?")
N_TEMPLATES=$(find "${TEMPLATE_DIR}" -name "*.npz" 2>/dev/null | wc -l || echo "0")
N_INDEX=$(python3 -c "import json; d=json.load(open('${INDEX_PATH}')); print(len(d))" 2>/dev/null || echo "?")

echo "  Structures in catalog: ${N_CATALOG}"
echo "  Template .npz files:   ${N_TEMPLATES}"
echo "  Sequences in index:    ${N_INDEX}"
echo "========================================================"
