#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONFIG_PATH="${SCRIPT_DIR}/train_config.sh"

CONFIG_PATH="${1:-${DEFAULT_CONFIG_PATH}}"
if [[ "${CONFIG_PATH}" == "--config" ]]; then
    CONFIG_PATH="${2:-${DEFAULT_CONFIG_PATH}}"
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_PATH}" >&2
    exit 1
fi

# shellcheck source=/dev/null
source "${CONFIG_PATH}"
set -u

apply_defaults() {
    : "${OUTPUT_ROOT:=${PROTENIX_DIR}/output}"
    : "${LAYERNORM_TYPE_VALUE:=}"
    : "${USE_RNA_SS:=false}"
    : "${RNA_SS_SEQUENCE_FPATH:=}"
    : "${RNA_SS_FEATURE_DIR:=}"
    : "${RNA_SS_FORMAT:=sparse_npz}"
    : "${RNA_SS_N_CLASSES:=6}"
    : "${RNA_SS_COVERAGE_WINDOW:=8}"
    : "${RNA_SS_STRICT:=false}"
    : "${RNA_SS_MIN_PROB:=0.0}"
    : "${RNA_SS_ARCHITECTURE:=mlp}"
    : "${RNA_SS_HIDDEN_DIM:=128}"
    : "${RNA_SS_N_LAYERS:=3}"
    : "${RNA_SS_ALPHA_INIT:=0.01}"
    : "${RNA_SS_INIT_METHOD:=kaiming}"
    : "${INFER_CHECKPOINT_PATH:=${CHECKPOINT_PATH}}"
    : "${INFER_MODEL_NAME:=${MODEL_NAME_ARG}}"
    : "${INFER_RUN_NAME:=}"
    : "${INFER_INPUT_JSON:=}"
    : "${INFER_DUMP_DIR:=}"
    : "${INFER_LOAD_STRICT:=false}"
    : "${INFER_NUM_WORKERS:=0}"
    : "${INFER_SEEDS:=101}"
    : "${INFER_USE_SEEDS_IN_JSON:=false}"
    : "${INFER_DTYPE:=${DTYPE}}"
    : "${INFER_N_SAMPLE:=1}"
    : "${INFER_N_STEP:=${SAMPLE_DIFFUSION_N_STEP}}"
    : "${INFER_N_CYCLE:=${MODEL_N_CYCLE}}"
    : "${INFER_USE_MSA:=false}"
    : "${INFER_USE_TEMPLATE:=false}"
    : "${INFER_USE_RNA_MSA:=false}"
    : "${INFER_NEED_ATOM_CONFIDENCE:=false}"
    : "${INFER_SORTED_BY_RANKING_SCORE:=true}"
    : "${INFER_ENABLE_TF32:=true}"
    : "${INFER_ENABLE_EFFICIENT_FUSION:=true}"
    : "${INFER_ENABLE_DIFFUSION_SHARED_VARS_CACHE:=true}"
    : "${INFER_EXTRA_ARGS:=}"
}

apply_defaults

as_bool() {
    local value="${1:-}"
    case "${value}" in
        true|false) printf "%s" "${value}" ;;
        *)
            echo "ERROR: invalid boolean value '${value}', expected true/false" >&2
            exit 1
            ;;
    esac
}

warn() {
    echo "WARN: $*" >&2
}

require_file() {
    local path="$1"
    local label="$2"
    [ -f "${path}" ] || { echo "ERROR: ${label} not found: ${path}" >&2; exit 1; }
}

require_dir() {
    local path="$1"
    local label="$2"
    [ -d "${path}" ] || { echo "ERROR: ${label} not found: ${path}" >&2; exit 1; }
}

append_args_from_string() {
    local target_name="$1"
    local raw="${2:-}"
    [ -n "${raw}" ] || return 0
    local -n target_ref="${target_name}"
    # shellcheck disable=SC2206
    local extra_args=( ${raw} )
    target_ref+=("${extra_args[@]}")
}

validate_choice() {
    local value="$1"
    local label="$2"
    shift 2
    local allowed
    for allowed in "$@"; do
        if [ "${value}" = "${allowed}" ]; then
            return 0
        fi
    done
    echo "ERROR: ${label}='${value}' is invalid. Allowed: $*" >&2
    exit 1
}

resolve_conda_env_spec() {
    local env_spec="${1:-}"
    if [ -z "${env_spec}" ]; then
        return 0
    fi
    if [ -d "${env_spec}" ]; then
        printf "%s" "${env_spec}"
    else
        printf "%s" "${PROJECT_ROOT}/conda/envs/${env_spec}"
    fi
}

activate_conda_env() {
    local env_spec="${1:-}"
    [ -n "${env_spec}" ] || return 0

    local conda_sh=""
    if command -v conda >/dev/null 2>&1; then
        local conda_base=""
        conda_base="$(conda info --base 2>/dev/null || true)"
        if [ -n "${conda_base}" ] && [ -f "${conda_base}/etc/profile.d/conda.sh" ]; then
            conda_sh="${conda_base}/etc/profile.d/conda.sh"
        fi
    fi

    if [ -z "${conda_sh}" ]; then
        local candidate
        for candidate in \
            "/opt/conda/etc/profile.d/conda.sh" \
            "${HOME}/miniconda3/etc/profile.d/conda.sh" \
            "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
            if [ -f "${candidate}" ]; then
                conda_sh="${candidate}"
                break
            fi
        done
    fi

    [ -n "${conda_sh}" ] || {
        echo "ERROR: Could not locate conda.sh for CONDA_ENV='${env_spec}'" >&2
        exit 1
    }

    set +u
    # shellcheck source=/dev/null
    source "${conda_sh}"
    command -v conda >/dev/null 2>&1 || {
        set -u
        echo "ERROR: 'conda' command unavailable after sourcing ${conda_sh}" >&2
        exit 1
    }
    conda activate "${env_spec}"
    set -u
}

validate_config() {
    validate_choice "${INJECTION_MODE}" "INJECTION_MODE" "input" "diffusion" "both"
    validate_choice "${GATE_MODE}" "GATE_MODE" "none" "scalar" "token" "dual"
    validate_choice "${RNA_PROJECTOR_INIT}" "RNA_PROJECTOR_INIT" "protein" "zero"
    validate_choice "${RNA_SS_FORMAT}" "RNA_SS_FORMAT" "sparse_npz" "dense_npz"
    validate_choice "${RNA_SS_ARCHITECTURE}" "RNA_SS_ARCHITECTURE" "mlp" "transformer"
    validate_choice "${RNA_SS_INIT_METHOD}" "RNA_SS_INIT_METHOD" "kaiming"

    as_bool "${PRINT_ONLY}" >/dev/null
    as_bool "${USE_RNALM}" >/dev/null
    as_bool "${USE_RIBONANZA}" >/dev/null
    as_bool "${USE_RNA}" >/dev/null
    as_bool "${USE_DNA}" >/dev/null
    as_bool "${USE_RNA_TEMPLATE}" >/dev/null
    as_bool "${USE_RNA_SS}" >/dev/null
    as_bool "${RNA_SS_STRICT}" >/dev/null
    as_bool "${RNALM_SEPARATE_DNA_PROJECTION}" >/dev/null
    as_bool "${INFER_LOAD_STRICT}" >/dev/null
    as_bool "${INFER_USE_SEEDS_IN_JSON}" >/dev/null
    as_bool "${INFER_USE_MSA}" >/dev/null
    as_bool "${INFER_USE_TEMPLATE}" >/dev/null
    as_bool "${INFER_USE_RNA_MSA}" >/dev/null
    as_bool "${INFER_NEED_ATOM_CONFIDENCE}" >/dev/null
    as_bool "${INFER_SORTED_BY_RANKING_SCORE}" >/dev/null
    as_bool "${INFER_ENABLE_TF32}" >/dev/null
    as_bool "${INFER_ENABLE_EFFICIENT_FUSION}" >/dev/null
    as_bool "${INFER_ENABLE_DIFFUSION_SHARED_VARS_CACHE}" >/dev/null

    require_dir "${PROTENIX_DIR}" "PROTENIX_DIR"
    require_file "${INFER_CHECKPOINT_PATH}" "INFER_CHECKPOINT_PATH"
    require_file "${INFER_INPUT_JSON}" "INFER_INPUT_JSON"

    if [ "$(as_bool "${USE_RNALM}")" = "true" ]; then
        if [ "$(as_bool "${USE_RNA}")" = "true" ]; then
            require_file "${RNA_SEQUENCE_FPATH}" "RNA_SEQUENCE_FPATH"
            require_dir "${RNA_EMBEDDING_DIR}" "RNA_EMBEDDING_DIR"
        fi
        if [ "$(as_bool "${USE_DNA}")" = "true" ]; then
            require_file "${DNA_SEQUENCE_FPATH}" "DNA_SEQUENCE_FPATH"
            require_dir "${DNA_EMBEDDING_DIR}" "DNA_EMBEDDING_DIR"
        fi
    fi

    if [ "$(as_bool "${USE_RNA_TEMPLATE}")" = "true" ]; then
        require_dir "${RNA_DATABASE_DIR}" "RNA_DATABASE_DIR"
        require_file "${RNA_SEARCH_RESULTS}" "RNA_SEARCH_RESULTS"
        require_dir "${PDB_RNA_DIR}" "PDB_RNA_DIR"
        require_file "${RNA3DB_METADATA_PATH}" "RNA3DB_METADATA_PATH"
    fi

    if [ "$(as_bool "${USE_RIBONANZA}")" = "true" ]; then
        require_dir "${RIBONANZA_MODEL_DIR}" "RIBONANZA_MODEL_DIR"
        require_file "${RIBONANZA_MODEL_DIR}/pairwise.yaml" "Ribonanza pairwise.yaml"
        require_file "${RIBONANZA_MODEL_DIR}/pytorch_model_fsdp.bin" "Ribonanza weights"
        validate_choice "${RIBONANZA_GATE_TYPE}" "RIBONANZA_GATE_TYPE" "channel" "scalar"
    fi

    if [ "$(as_bool "${USE_RNA_SS}")" = "true" ]; then
        if [ "${RNA_SS_N_CLASSES}" != "6" ]; then
            echo "ERROR: RNA_SS_N_CLASSES must be 6, got ${RNA_SS_N_CLASSES}" >&2
            exit 1
        fi
        if [ -n "${RNA_SS_SEQUENCE_FPATH}" ]; then
            require_file "${RNA_SS_SEQUENCE_FPATH}" "RNA_SS_SEQUENCE_FPATH"
        elif [ "$(as_bool "${RNA_SS_STRICT}")" = "true" ]; then
            echo "ERROR: RNA_SS_SEQUENCE_FPATH is required when USE_RNA_SS=true and RNA_SS_STRICT=true" >&2
            exit 1
        else
            warn "USE_RNA_SS=true but RNA_SS_SEQUENCE_FPATH is empty; priors will gracefully fall back to zeros."
        fi
        if [ -n "${RNA_SS_FEATURE_DIR}" ]; then
            require_dir "${RNA_SS_FEATURE_DIR}" "RNA_SS_FEATURE_DIR"
        fi
    fi
}

build_rnalm_args() {
    RNALM_ARGS=()
    if [ "$(as_bool "${USE_RNALM}")" != "true" ]; then
        RNALM_ARGS=(--rnalm.enable false)
        return 0
    fi

    local rna_args=()
    local dna_args=()
    if [ "$(as_bool "${USE_RNA}")" = "true" ]; then
        rna_args+=(--rnalm.embedding_dir "${RNA_EMBEDDING_DIR}")
        rna_args+=(--rnalm.sequence_fpath "${RNA_SEQUENCE_FPATH}")
    fi
    if [ "$(as_bool "${USE_DNA}")" = "true" ]; then
        dna_args+=(--rnalm.dna_embedding_dir "${DNA_EMBEDDING_DIR}")
        dna_args+=(--rnalm.dna_sequence_fpath "${DNA_SEQUENCE_FPATH}")
    fi

    RNALM_ARGS=(
        --rnalm.enable true
        --rnalm.use_rna_embed "$(as_bool "${USE_RNA}")"
        --rnalm.use_dna_embed "$(as_bool "${USE_DNA}")"
        --rnalm.model_name "${RNALM_MODEL_NAME}"
        --rnalm.embedding_dim "${RNA_EMBEDDING_DIM}"
        --rnalm.dna_embedding_dim "${DNA_EMBEDDING_DIM}"
        --rnalm.injection_mode "${INJECTION_MODE}"
        --rnalm.gate_mode "${GATE_MODE}"
        --rnalm.gate_init_logit "${GATE_INIT_LOGIT}"
        --rnalm.separate_dna_projection "$(as_bool "${RNALM_SEPARATE_DNA_PROJECTION}")"
        "${rna_args[@]}"
        "${dna_args[@]}"
    )
}

build_rna_template_args() {
    RNA_TEMPLATE_ARGS=()
    if [ "$(as_bool "${USE_RNA_TEMPLATE}")" != "true" ]; then
        RNA_TEMPLATE_ARGS=(--rna_template.enable false)
        return 0
    fi

    RNA_TEMPLATE_ARGS=(
        --rna_template.enable true
        --rna_template.template_database_dir "${RNA_DATABASE_DIR}"
        --rna_template.search_results_path "${RNA_SEARCH_RESULTS}"
        --rna_template.cif_database_dir "${PDB_RNA_DIR}"
        --rna_template.max_rna_templates "${MAX_RNA_TEMPLATES}"
        --rna_template.rna3db_metadata_path "${RNA3DB_METADATA_PATH}"
        --rna_template.projector_init "${RNA_PROJECTOR_INIT}"
        --rna_template.alpha_init "${RNA_TEMPLATE_ALPHA}"
        --model.template_embedder.n_blocks "${TEMPLATE_N_BLOCKS}"
    )
}

build_ribonanza_args() {
    RIBONANZA_ARGS=()
    if [ "$(as_bool "${USE_RIBONANZA}")" != "true" ]; then
        RIBONANZA_ARGS=(--ribonanzanet2.enable false)
        return 0
    fi

    RIBONANZA_ARGS=(
        --ribonanzanet2.enable true
        --ribonanzanet2.model_dir "${RIBONANZA_MODEL_DIR}"
        --ribonanzanet2.gate_type "${RIBONANZA_GATE_TYPE}"
        --ribonanzanet2.n_pairformer_blocks "${RIBONANZA_N_PAIRFORMER_BLOCKS}"
    )
}

build_rna_ss_args() {
    RNA_SS_ARGS=(
        --rna_ss.enable false
        --model.constraint_embedder.substructure_embedder.enable false
    )

    if [ "$(as_bool "${USE_RNA_SS}")" != "true" ]; then
        return 0
    fi

    RNA_SS_ARGS=(
        --rna_ss.enable true
        --rna_ss.format "${RNA_SS_FORMAT}"
        --rna_ss.n_classes "${RNA_SS_N_CLASSES}"
        --rna_ss.coverage_window "${RNA_SS_COVERAGE_WINDOW}"
        --rna_ss.strict "$(as_bool "${RNA_SS_STRICT}")"
        --rna_ss.min_prob "${RNA_SS_MIN_PROB}"
        --model.constraint_embedder.initialize_method "${RNA_SS_INIT_METHOD}"
        --model.constraint_embedder.substructure_embedder.enable true
        --model.constraint_embedder.substructure_embedder.n_classes "${RNA_SS_N_CLASSES}"
        --model.constraint_embedder.substructure_embedder.architecture "${RNA_SS_ARCHITECTURE}"
        --model.constraint_embedder.substructure_embedder.hidden_dim "${RNA_SS_HIDDEN_DIM}"
        --model.constraint_embedder.substructure_embedder.n_layers "${RNA_SS_N_LAYERS}"
        --model.constraint_embedder.substructure_embedder.alpha_init "${RNA_SS_ALPHA_INIT}"
    )

    if [ -n "${RNA_SS_SEQUENCE_FPATH}" ]; then
        RNA_SS_ARGS+=(--rna_ss.sequence_fpath "${RNA_SS_SEQUENCE_FPATH}")
    fi
    if [ -n "${RNA_SS_FEATURE_DIR}" ]; then
        RNA_SS_ARGS+=(--rna_ss.feature_dir "${RNA_SS_FEATURE_DIR}")
    fi
}

prepare_checkpoint_link() {
    INFER_CKPT_DIR="$(mktemp -d)"
    trap 'rm -rf "${INFER_CKPT_DIR}"' EXIT
    ln -sf "$(realpath "${INFER_CHECKPOINT_PATH}")" "${INFER_CKPT_DIR}/${INFER_MODEL_NAME}.pt"
}

build_common_args() {
    local infer_name="${INFER_RUN_NAME}"
    if [ -z "${infer_name}" ]; then
        infer_name="$(basename "${INFER_INPUT_JSON}")"
        infer_name="${infer_name%.json}"
    fi

    if [ -n "${INFER_DUMP_DIR}" ]; then
        RESOLVED_INFER_DUMP_DIR="${INFER_DUMP_DIR}"
    else
        RESOLVED_INFER_DUMP_DIR="${OUTPUT_ROOT}/inference/${infer_name}"
    fi

    COMMON_ARGS=(
        --model_name "${INFER_MODEL_NAME}"
        --load_checkpoint_dir "${INFER_CKPT_DIR}"
        --load_strict "$(as_bool "${INFER_LOAD_STRICT}")"
        --input_json_path "${INFER_INPUT_JSON}"
        --dump_dir "${RESOLVED_INFER_DUMP_DIR}"
        --num_workers "${INFER_NUM_WORKERS}"
        --seeds "${INFER_SEEDS}"
        --use_seeds_in_json "$(as_bool "${INFER_USE_SEEDS_IN_JSON}")"
        --dtype "${INFER_DTYPE}"
        --use_msa "$(as_bool "${INFER_USE_MSA}")"
        --use_template "$(as_bool "${INFER_USE_TEMPLATE}")"
        --use_rna_msa "$(as_bool "${INFER_USE_RNA_MSA}")"
        --need_atom_confidence "$(as_bool "${INFER_NEED_ATOM_CONFIDENCE}")"
        --sorted_by_ranking_score "$(as_bool "${INFER_SORTED_BY_RANKING_SCORE}")"
        --enable_tf32 "$(as_bool "${INFER_ENABLE_TF32}")"
        --enable_efficient_fusion "$(as_bool "${INFER_ENABLE_EFFICIENT_FUSION}")"
        --enable_diffusion_shared_vars_cache "$(as_bool "${INFER_ENABLE_DIFFUSION_SHARED_VARS_CACHE}")"
        --model.N_cycle "${INFER_N_CYCLE}"
        --sample_diffusion.N_sample "${INFER_N_SAMPLE}"
        --sample_diffusion.N_step "${INFER_N_STEP}"
        --triangle_attention "${TRIANGLE_ATTENTION_IMPL}"
        --triangle_multiplicative "${TRIANGLE_MULTIPLICATIVE_IMPL}"
    )

    append_args_from_string "COMMON_ARGS" "${INFER_EXTRA_ARGS}"
}

print_summary() {
    echo "========================================================"
    echo "  Protenix Unified Inference Pipeline"
    echo "  config:             ${CONFIG_PATH}"
    echo "  checkpoint:         ${INFER_CHECKPOINT_PATH}"
    echo "  input_json:         ${INFER_INPUT_JSON}"
    echo "  dump_dir:           ${RESOLVED_INFER_DUMP_DIR}"
    echo "  model_name:         ${INFER_MODEL_NAME}"
    echo "  use_rnalm:          ${USE_RNALM}"
    echo "  use_ribonanza:      ${USE_RIBONANZA}"
    echo "  use_rna_ss:         ${USE_RNA_SS}"
    echo "  use_rna_template:   ${USE_RNA_TEMPLATE}"
    echo "  use_msa:            ${INFER_USE_MSA}"
    echo "  use_template:       ${INFER_USE_TEMPLATE}"
    echo "  seeds:              ${INFER_SEEDS}"
    if [ "$(as_bool "${USE_RNA_SS}")" = "true" ]; then
        echo "  rna_ss_index:       ${RNA_SS_SEQUENCE_FPATH:-<empty>}"
        echo "  rna_ss_feature_dir: ${RNA_SS_FEATURE_DIR:-<empty>}"
        echo "  rna_ss_strict:      ${RNA_SS_STRICT}"
    fi
    echo "========================================================"
}

main() {
    validate_config

    export PROTENIX_ROOT_DIR="${PROJECT_ROOT}/data"
    if [ -n "${CONDA_ENV}" ]; then
        activate_conda_env "${CONDA_ENV}"
        CONDA_PREFIX_ACTUAL="${CONDA_PREFIX:-$(resolve_conda_env_spec "${CONDA_ENV}")}"
        export CUDA_HOME="${CONDA_PREFIX_ACTUAL}"
        export CPLUS_INCLUDE_PATH="${CONDA_PREFIX_ACTUAL}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"
    fi
    if [ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]; then
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
    fi

    cd "${PROTENIX_DIR}"
    export PYTHONPATH="${PROTENIX_DIR}:${PYTHONPATH:-}"
    export TRIANGLE_ATTENTION="${TRIANGLE_ATTENTION_IMPL}"
    export TRIANGLE_MULTIPLICATIVE="${TRIANGLE_MULTIPLICATIVE_IMPL}"
    if [ -n "${LAYERNORM_TYPE_VALUE}" ]; then
        export LAYERNORM_TYPE="${LAYERNORM_TYPE_VALUE}"
    fi

    build_rnalm_args
    build_rna_template_args
    build_ribonanza_args
    build_rna_ss_args
    prepare_checkpoint_link
    build_common_args

    mkdir -p "${RESOLVED_INFER_DUMP_DIR}"
    print_summary

    CMD=(
        "${PYTHON_BIN}" ./runner/inference.py
        "${COMMON_ARGS[@]}"
        "${RNALM_ARGS[@]}"
        "${RIBONANZA_ARGS[@]}"
        "${RNA_SS_ARGS[@]}"
        "${RNA_TEMPLATE_ARGS[@]}"
    )

    if [ "$(as_bool "${PRINT_ONLY}")" = "true" ]; then
        printf 'Command:\n'
        printf '  %q' "${CMD[@]}"
        printf '\n'
        exit 0
    fi

    "${CMD[@]}"
}

main "$@"
