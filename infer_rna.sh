#!/bin/bash
# =============================================================================
# Protenix Inference with RNA/DNA LLM Embeddings
#
# Runs inference using a finetuned checkpoint that includes RNA/DNA LLM
# embedding injection. Supports all injection modes and toggle combinations.
#
# Prerequisites:
#   1. A finetuned checkpoint (from finetune_rna_only_diffuse.sh or finetune_rna_dna.sh)
#   2. Pre-computed RNA/DNA embeddings for the input sequences
#   3. A sequences CSV mapping input sequences to embedding files
#
# Usage:
#   bash infer_rna.sh --input_json /path/to/input.json --checkpoint /path/to/checkpoint.pt
#   bash infer_rna.sh --input_json /path/to/input.json --checkpoint /path/to/checkpoint.pt \
#       --injection_mode input --use_dna true
#   bash infer_rna.sh --input_json /path/to/input.json --checkpoint /path/to/checkpoint.pt \
#       --n_sample 5 --n_step 200
# =============================================================================
set -euo pipefail

# ===================== Configurable Parameters =====================
INPUT_JSON=""
CHECKPOINT_PATH=""
INJECTION_MODE="input"          # "diffusion" | "input" | "both"  — must match training
USE_RNALM="true"
USE_RNA="true"
USE_DNA="true"
GATE_MODE="none"                # "none" | "scalar" | "token" | "dual"  — must match training
GATE_INIT_LOGIT="-3.0"
N_SAMPLE=5
N_STEP=200
N_CYCLE=10
DUMP_DIR="./output/infer_rna"
DTYPE="bf16"
USE_MSA="false"
USE_TEMPLATE="false"

# ===================== RNA Template Parameters =====================
USE_RNA_TEMPLATE="false"
RNA_PROJECTOR_INIT="protein"
RNA_TEMPLATE_ALPHA="0.01"
MAX_RNA_TEMPLATES=4
TEMPLATE_N_BLOCKS=2
RNA_SEARCH_RESULTS=""              # search_results.json path (online mode)
RNA_CIF_DIR=""                     # CIF database directory (online mode)
RNA3DB_METADATA=""                 # RNA3DB filter.json path
# ===================================================================

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_json)       INPUT_JSON="$2";        shift 2 ;;
        --checkpoint)       CHECKPOINT_PATH="$2";   shift 2 ;;
        --injection_mode)   INJECTION_MODE="$2";    shift 2 ;;
        --use_rnalm)        USE_RNALM="$2";         shift 2 ;;
        --use_rna)          USE_RNA="$2";           shift 2 ;;
        --use_dna)          USE_DNA="$2";           shift 2 ;;
        --gate_mode)        GATE_MODE="$2";         shift 2 ;;
        --gate_init_logit)  GATE_INIT_LOGIT="$2";   shift 2 ;;
        --n_sample)         N_SAMPLE="$2";          shift 2 ;;
        --n_step)           N_STEP="$2";            shift 2 ;;
        --n_cycle)          N_CYCLE="$2";           shift 2 ;;
        --dump_dir)         DUMP_DIR="$2";          shift 2 ;;
        --dtype)            DTYPE="$2";             shift 2 ;;
        --use_msa)          USE_MSA="$2";           shift 2 ;;
        --use_template)     USE_TEMPLATE="$2";      shift 2 ;;
        --use_rna_template) USE_RNA_TEMPLATE="$2";  shift 2 ;;
        --rna_search_results) RNA_SEARCH_RESULTS="$2"; shift 2 ;;
        --rna_cif_dir)      RNA_CIF_DIR="$2";       shift 2 ;;
        --rna3db_metadata)  RNA3DB_METADATA="$2";   shift 2 ;;
        --rna_projector_init) RNA_PROJECTOR_INIT="$2"; shift 2 ;;
        --rna_template_alpha) RNA_TEMPLATE_ALPHA="$2"; shift 2 ;;
        --max_rna_templates) MAX_RNA_TEMPLATES="$2"; shift 2 ;;
        --template_n_blocks) TEMPLATE_N_BLOCKS="$2"; shift 2 ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo "Required: --input_json, --checkpoint"
            echo "Optional: --injection_mode, --use_rnalm, --use_rna, --use_dna, --gate_mode,"
            echo "          --n_sample, --n_step, --n_cycle, --dump_dir, --dtype,"
            echo "          --use_msa, --use_template,"
            echo "          --use_rna_template, --rna_search_results, --rna_cif_dir,"
            echo "          --rna3db_metadata, --rna_projector_init, --rna_template_alpha"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "${INPUT_JSON}" ]; then
    echo "ERROR: --input_json is required"
    exit 1
fi
if [ -z "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: --checkpoint is required (path to finetuned .pt file)"
    exit 1
fi
if [ ! -f "${INPUT_JSON}" ]; then
    echo "ERROR: Input JSON not found: ${INPUT_JSON}"
    exit 1
fi
if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT_PATH}"
    exit 1
fi

# ===================== Paths =====================
PROJECT_ROOT="/inspire/ssd/project/sais-bio/public/ash_proj"
PROTENIX_DIR="${PROJECT_ROOT}/code/protenix_new/Protenix"
DATA_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/part2"

# AIDO Embedding paths
AIDO_EMBEDDING_DIR="${DATA_DIR}/aido_embeddings"
RNA_EMBEDDING_DIR="${AIDO_EMBEDDING_DIR}/rna"
RNA_SEQUENCE_FPATH="${RNA_EMBEDDING_DIR}/rna_sequences.csv"
DNA_EMBEDDING_DIR="${AIDO_EMBEDDING_DIR}/dna"
DNA_SEQUENCE_FPATH="${DNA_EMBEDDING_DIR}/dna_sequences.csv"

RNA_EMBEDDING_DIM=2048  # AIDO.RNA-1.6B
DNA_EMBEDDING_DIM=1024  # AIDO.DNA-300M

# Prepare checkpoint: inference expects {load_checkpoint_dir}/{model_name}.pt
# We symlink the finetuned checkpoint to a temp dir with a known name
INFER_CKPT_DIR=$(mktemp -d)
INFER_MODEL_NAME="finetuned_rnalm"
ln -sf "$(realpath "${CHECKPOINT_PATH}")" "${INFER_CKPT_DIR}/${INFER_MODEL_NAME}.pt"
trap "rm -rf ${INFER_CKPT_DIR}" EXIT

# Validate embedding paths
RNA_EMB_DIR_ARG=""
RNA_SEQ_FPATH_ARG=""
if [ "${USE_RNALM}" = "true" ] && [ "${USE_RNA}" = "true" ]; then
    if [ ! -f "${RNA_SEQUENCE_FPATH}" ]; then
        echo "ERROR: RNA sequence CSV not found: ${RNA_SEQUENCE_FPATH}"
        echo "Please run generate_aido_embeddings.py first."
        exit 1
    fi
    RNA_EMB_DIR_ARG="${RNA_EMBEDDING_DIR}"
    RNA_SEQ_FPATH_ARG="${RNA_SEQUENCE_FPATH}"
fi

DNA_EMB_DIR_ARG=""
DNA_SEQ_FPATH_ARG=""
if [ "${USE_RNALM}" = "true" ] && [ "${USE_DNA}" = "true" ]; then
    if [ -f "${DNA_SEQUENCE_FPATH}" ]; then
        DNA_EMB_DIR_ARG="${DNA_EMBEDDING_DIR}"
        DNA_SEQ_FPATH_ARG="${DNA_SEQUENCE_FPATH}"
    else
        echo "WARNING: DNA sequence CSV not found: ${DNA_SEQUENCE_FPATH}"
        echo "DNA embeddings disabled."
        USE_DNA="false"
    fi
fi

# ===================== RNA Template args =====================
RNA_TEMPLATE_ARGS=""
if [ "${USE_RNA_TEMPLATE}" = "true" ]; then
    # Default paths if not specified
    RNA_DATABASE_DIR="${PROTENIX_DIR}/rna_database"
    [ -z "${RNA_SEARCH_RESULTS}" ] && RNA_SEARCH_RESULTS="${RNA_DATABASE_DIR}/search_results.json"
    [ -z "${RNA_CIF_DIR}" ] && RNA_CIF_DIR="${DATA_DIR}/PDB_RNA"
    [ -z "${RNA3DB_METADATA}" ] && RNA3DB_METADATA="${PROJECT_ROOT}/data/RNA3D/rna3db-jsons/filter.json"

    [ -f "${RNA_SEARCH_RESULTS}" ] || { echo "ERROR: RNA search results not found: ${RNA_SEARCH_RESULTS}"; exit 1; }
    [ -d "${RNA_CIF_DIR}" ] || { echo "ERROR: RNA CIF dir not found: ${RNA_CIF_DIR}"; exit 1; }

    RNA_TEMPLATE_ARGS="--rna_template.enable true \
        --rna_template.template_database_dir ${RNA_DATABASE_DIR} \
        --rna_template.search_results_path ${RNA_SEARCH_RESULTS} \
        --rna_template.cif_database_dir ${RNA_CIF_DIR} \
        --rna_template.max_rna_templates ${MAX_RNA_TEMPLATES} \
        --rna_template.rna3db_metadata_path ${RNA3DB_METADATA} \
        --rna_template.projector_init ${RNA_PROJECTOR_INIT} \
        --rna_template.alpha_init ${RNA_TEMPLATE_ALPHA} \
        --model.template_embedder.n_blocks ${TEMPLATE_N_BLOCKS}"
else
    RNA_TEMPLATE_ARGS="--rna_template.enable false"
fi

# ===================== Environment =====================
export PROTENIX_ROOT_DIR="${PROJECT_ROOT}/data"
CONDA_PREFIX="${PROJECT_ROOT}/conda/envs/protenix"
export CUDA_HOME="${CONDA_PREFIX}"
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"

cd "${PROTENIX_DIR}"
export PYTHONPATH="${PROTENIX_DIR}:${PYTHONPATH:-}"
export TRIANGLE_ATTENTION="cuequivariance"
export TRIANGLE_MULTIPLICATIVE="cuequivariance"

mkdir -p "${DUMP_DIR}"

# ===================== Banner =====================
echo "========================================================"
echo "  Protenix Inference: RNA/DNA LLM Embeddings"
echo "  checkpoint:        ${CHECKPOINT_PATH}"
echo "  input_json:        ${INPUT_JSON}"
echo "  use_rnalm:         ${USE_RNALM}"
echo "  use_rna:           ${USE_RNA}"
echo "  use_dna:           ${USE_DNA}"
echo "  injection_mode:    ${INJECTION_MODE}"
echo "  gate_mode:         ${GATE_MODE}"
echo "  use_rna_template:  ${USE_RNA_TEMPLATE}"
echo "  N_sample:          ${N_SAMPLE}"
echo "  N_step:            ${N_STEP}"
echo "  N_cycle:           ${N_CYCLE}"
echo "  dump_dir:          ${DUMP_DIR}"
echo "========================================================"

python3 ./runner/inference.py \
    --model_name "${INFER_MODEL_NAME}" \
    --load_checkpoint_dir "${INFER_CKPT_DIR}" \
    --load_strict false \
    --input_json_path "${INPUT_JSON}" \
    --dump_dir "${DUMP_DIR}" \
    --dtype "${DTYPE}" \
    --use_msa "${USE_MSA}" \
    --use_template "${USE_TEMPLATE}" \
    --model.N_cycle "${N_CYCLE}" \
    --sample_diffusion.N_sample "${N_SAMPLE}" \
    --sample_diffusion.N_step "${N_STEP}" \
    --triangle_attention "cuequivariance" \
    --triangle_multiplicative "cuequivariance" \
    --enable_tf32 true \
    --enable_efficient_fusion true \
    --enable_diffusion_shared_vars_cache true \
    \
    --rnalm.enable "${USE_RNALM}" \
    --rnalm.use_rna_embed "${USE_RNA}" \
    --rnalm.use_dna_embed "${USE_DNA}" \
    --rnalm.model_name "aido" \
    --rnalm.embedding_dim "${RNA_EMBEDDING_DIM}" \
    --rnalm.dna_embedding_dim "${DNA_EMBEDDING_DIM}" \
    --rnalm.injection_mode "${INJECTION_MODE}" \
    --rnalm.gate_mode "${GATE_MODE}" \
    --rnalm.gate_init_logit "${GATE_INIT_LOGIT}" \
    --rnalm.separate_dna_projection true \
    --rnalm.embedding_dir "${RNA_EMB_DIR_ARG}" \
    --rnalm.sequence_fpath "${RNA_SEQ_FPATH_ARG}" \
    --rnalm.dna_embedding_dir "${DNA_EMB_DIR_ARG}" \
    --rnalm.dna_sequence_fpath "${DNA_SEQ_FPATH_ARG}" \
    \
    ${RNA_TEMPLATE_ARGS}

echo ""
echo "Inference complete. Results saved to: ${DUMP_DIR}"
