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
    : "${LOAD_EMA_CHECKPOINT_PATH:=}"
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
    : "${TRAIN_EXTRA_ARGS:=}"
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
    # Deliberately split on shell words for config-authored extra args.
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

validate_config() {
    validate_choice "${TRAIN_MODE}" "TRAIN_MODE" "1stage" "2stage"
    validate_choice "${INJECTION_MODE}" "INJECTION_MODE" "input" "diffusion" "both"
    validate_choice "${GATE_MODE}" "GATE_MODE" "none" "scalar" "token" "dual"
    validate_choice "${RNA_PROJECTOR_INIT}" "RNA_PROJECTOR_INIT" "protein" "zero"
    validate_choice "${RNA_SS_FORMAT}" "RNA_SS_FORMAT" "sparse_npz" "dense_npz"
    validate_choice "${RNA_SS_ARCHITECTURE}" "RNA_SS_ARCHITECTURE" "mlp" "transformer"
    validate_choice "${RNA_SS_INIT_METHOD}" "RNA_SS_INIT_METHOD" "kaiming"

    as_bool "${USE_WANDB}" >/dev/null
    as_bool "${PRINT_ONLY}" >/dev/null
    as_bool "${USE_RNALM}" >/dev/null
    as_bool "${USE_RIBONANZA}" >/dev/null
    as_bool "${USE_RNA}" >/dev/null
    as_bool "${USE_DNA}" >/dev/null
    as_bool "${USE_RNA_TEMPLATE}" >/dev/null
    as_bool "${USE_RNA_MSA}" >/dev/null
    as_bool "${USE_PROT_MSA}" >/dev/null
    as_bool "${USE_PROT_TEMPLATE}" >/dev/null
    as_bool "${VAL_FIND_EVAL_CHAIN_INTERFACE}" >/dev/null
    as_bool "${VAL_GROUP_BY_PDB_ID}" >/dev/null
    as_bool "${ADAM_USE_ADAMW}" >/dev/null
    as_bool "${CONFIDENCE_STOP_GRADIENT}" >/dev/null
    as_bool "${RNALM_SEPARATE_DNA_PROJECTION}" >/dev/null
    as_bool "${RNA_LOSS_ENABLE}" >/dev/null
    as_bool "${LOAD_STRICT}" >/dev/null
    as_bool "${USE_RNA_SS}" >/dev/null
    as_bool "${RNA_SS_STRICT}" >/dev/null

    require_dir "${PROTENIX_DIR}" "PROTENIX_DIR"
    require_file "${CHECKPOINT_PATH}" "CHECKPOINT_PATH"
    if [ -n "${LOAD_EMA_CHECKPOINT_PATH}" ]; then
        require_file "${LOAD_EMA_CHECKPOINT_PATH}" "LOAD_EMA_CHECKPOINT_PATH"
    fi
    require_dir "${RNA_CIF_DIR}" "RNA_CIF_DIR"
    require_dir "${BIOASSEMBLY_DIR}" "BIOASSEMBLY_DIR"
    require_file "${TRAIN_INDICES_FPATH}" "TRAIN_INDICES_FPATH"
    require_file "${TRAIN_PDB_LIST}" "TRAIN_PDB_LIST"
    require_file "${VAL_INDICES_FPATH}" "VAL_INDICES_FPATH"
    require_file "${VAL_PDB_LIST}" "VAL_PDB_LIST"

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
        if [ -n "${MANUAL_TEMPLATE_HINTS}" ]; then
            require_file "${MANUAL_TEMPLATE_HINTS}" "MANUAL_TEMPLATE_HINTS"
        fi
    fi

    if [ "$(as_bool "${USE_RNA_MSA}")" = "true" ]; then
        require_dir "${RNA_MSA_RAW_DIR}" "RNA_MSA_RAW_DIR"
        require_file "${RNA_MSA_INDEX_JSON}" "RNA_MSA_INDEX_JSON"
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

build_rnalm_args() {
    RNALM_ARGS=()
    if [ "$(as_bool "${USE_RNALM}")" != "true" ]; then
        RNALM_ARGS+=(--rnalm.enable false)
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

    if [ -n "${MANUAL_TEMPLATE_HINTS}" ]; then
        RNA_TEMPLATE_ARGS+=(--rna_template.manual_template_hints_path "${MANUAL_TEMPLATE_HINTS}")
    fi
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
        --model.constraint_embedder.substructure_embedder.enable true
        --model.constraint_embedder.substructure_embedder.n_classes "${RNA_SS_N_CLASSES}"
        --model.constraint_embedder.substructure_embedder.architecture "${RNA_SS_ARCHITECTURE}"
        --model.constraint_embedder.substructure_embedder.hidden_dim "${RNA_SS_HIDDEN_DIM}"
        --model.constraint_embedder.substructure_embedder.n_layers "${RNA_SS_N_LAYERS}"
        --model.constraint_embedder.substructure_embedder.alpha_init "${RNA_SS_ALPHA_INIT}"
        --model.constraint_embedder.substructure_embedder.initialize_method "${RNA_SS_INIT_METHOD}"
    )

    if [ -n "${RNA_SS_SEQUENCE_FPATH}" ]; then
        RNA_SS_ARGS+=(--rna_ss.sequence_fpath "${RNA_SS_SEQUENCE_FPATH}")
    fi
    if [ -n "${RNA_SS_FEATURE_DIR}" ]; then
        RNA_SS_ARGS+=(--rna_ss.feature_dir "${RNA_SS_FEATURE_DIR}")
    fi
}

build_common_args() {
    local resolved_run_name="${RUN_NAME}"
    if [ -z "${resolved_run_name}" ]; then
        if [ "${TRAIN_MODE}" = "1stage" ]; then
            resolved_run_name="1stage_rna_finetune"
            [ "$(as_bool "${USE_RIBONANZA}")" = "true" ] && resolved_run_name="${resolved_run_name}_rnet2"
            [ "$(as_bool "${USE_RNALM}")" = "true" ] && resolved_run_name="${resolved_run_name}_llm_${INJECTION_MODE}"
            [ "$(as_bool "${USE_RNA_SS}")" = "true" ] && resolved_run_name="${resolved_run_name}_rna_ss"
            [ "$(as_bool "${USE_RNA_MSA}")" = "true" ] && resolved_run_name="${resolved_run_name}_rna_msa"
            [ "$(as_bool "${USE_RNA_TEMPLATE}")" = "true" ] && resolved_run_name="${resolved_run_name}_rna_template"
            resolved_run_name="${resolved_run_name}_alr${ADAPTER_LR}_blr${BACKBONE_LR}"
        else
            resolved_run_name="2stage_rna_finetune"
            [ "$(as_bool "${USE_RIBONANZA}")" = "true" ] && resolved_run_name="${resolved_run_name}_rnet2"
            [ "$(as_bool "${USE_RNALM}")" = "true" ] && resolved_run_name="${resolved_run_name}_llm_${INJECTION_MODE}"
            [ "$(as_bool "${USE_RNA_SS}")" = "true" ] && resolved_run_name="${resolved_run_name}_rna_ss"
            [ "$(as_bool "${USE_RNA_MSA}")" = "true" ] && resolved_run_name="${resolved_run_name}_rna_msa"
            [ "$(as_bool "${USE_RNA_TEMPLATE}")" = "true" ] && resolved_run_name="${resolved_run_name}_rna_template"
            resolved_run_name="${resolved_run_name}_s1a${STAGE1_ADAPTER_LR}_s1b${STAGE1_BACKBONE_LR}"
        fi
        [ "${GATE_MODE}" != "none" ] && resolved_run_name="${resolved_run_name}_gate_${GATE_MODE}"
    fi
    RUN_NAME_RESOLVED="${resolved_run_name}"

    if [ -n "${PROJECT_NAME}" ]; then
        PROJECT_NAME_RESOLVED="${PROJECT_NAME}"
    elif [ "${TRAIN_MODE}" = "1stage" ]; then
        PROJECT_NAME_RESOLVED="protenix_rna_finetune_1stage"
    else
        PROJECT_NAME_RESOLVED="protenix_rna_finetune_2stage"
    fi

    OUTPUT_DIR="${OUTPUT_ROOT}"

    COMMON_ARGS=(
        --model_name "${MODEL_NAME_ARG}"
        --run_name "${RUN_NAME_RESOLVED}"
        --seed "${SEED}"
        --base_dir "${OUTPUT_DIR}"
        --dtype "${DTYPE}"
        --project "${PROJECT_NAME_RESOLVED}"
        --use_wandb "$(as_bool "${USE_WANDB}")"
        --diffusion_batch_size "${DIFFUSION_BATCH_SIZE}"
        --eval_interval "${EVAL_INTERVAL}"
        --log_interval "${LOG_INTERVAL}"
        --checkpoint_interval "${CHECKPOINT_INTERVAL}"
        --train_crop_size "${TRAIN_CROP_SIZE}"
        --max_steps "${MAX_STEPS}"
        --lr "${LR}"
        --lr_scheduler "${LR_SCHEDULER}"
        --warmup_steps "${WARMUP_STEPS}"
        --grad_clip_norm "${GRAD_CLIP_NORM}"
        --model.N_cycle "${MODEL_N_CYCLE}"
        --sample_diffusion.N_step "${SAMPLE_DIFFUSION_N_STEP}"
        --triangle_attention "${TRIANGLE_ATTENTION_IMPL}"
        --triangle_multiplicative "${TRIANGLE_MULTIPLICATIVE_IMPL}"
        --load_checkpoint_path "${CHECKPOINT_PATH}"
        --load_strict "$(as_bool "${LOAD_STRICT}")"
        --data.num_dl_workers "${DATA_NUM_DL_WORKERS}"
        --adam.use_adamw "$(as_bool "${ADAM_USE_ADAMW}")"
        --adam.beta1 "${ADAM_BETA1}"
        --adam.beta2 "${ADAM_BETA2}"
        --adam.weight_decay "${ADAM_WEIGHT_DECAY}"
        --loss.weight.alpha_bond "${LOSS_ALPHA_BOND}"
        --rna_loss.enable "$(as_bool "${RNA_LOSS_ENABLE}")"
        --model.confidence_head.stop_gradient "$(as_bool "${CONFIDENCE_STOP_GRADIENT}")"
    )

    if [ "${TRAIN_MODE}" = "1stage" ]; then
        COMMON_ARGS+=(--ema_decay "${EMA_DECAY}")
    fi

    if [ -n "${LOAD_EMA_CHECKPOINT_PATH}" ]; then
        COMMON_ARGS+=(--load_ema_checkpoint_path "${LOAD_EMA_CHECKPOINT_PATH}")
    fi

    append_args_from_string "COMMON_ARGS" "${TRAIN_EXTRA_ARGS}"
}

build_stage_args() {
    STAGE_ARGS=()
    if [ "${TRAIN_MODE}" = "1stage" ]; then
        STAGE_ARGS=(
            --two_stage.enable false
            --two_stage.adapter_lr "${ADAPTER_LR}"
            --two_stage.backbone_lr "${BACKBONE_LR}"
        )
    else
        STAGE_ARGS=(
            --two_stage.enable true
            --two_stage.stage1_max_steps "${STAGE1_MAX_STEPS}"
            --two_stage.stage1_adapter_lr "${STAGE1_ADAPTER_LR}"
            --two_stage.stage1_backbone_lr "${STAGE1_BACKBONE_LR}"
            --two_stage.stage1_warmup_steps "${STAGE1_WARMUP_STEPS}"
            --two_stage.stage2_adapter_lr "${STAGE2_ADAPTER_LR}"
            --two_stage.stage2_backbone_lr "${STAGE2_BACKBONE_LR}"
            --two_stage.stage2_warmup_steps "${STAGE2_WARMUP_STEPS}"
            --two_stage.stage2_ema_decay "${STAGE2_EMA_DECAY}"
        )
    fi
}

build_data_args() {
    DATA_ARGS=(
        --data.train_sets "${TRAIN_SET}"
        --data.${TRAIN_SET}.base_info.mmcif_dir "${RNA_CIF_DIR}"
        --data.${TRAIN_SET}.base_info.bioassembly_dict_dir "${BIOASSEMBLY_DIR}"
        --data.${TRAIN_SET}.base_info.indices_fpath "${TRAIN_INDICES_FPATH}"
        --data.${TRAIN_SET}.base_info.pdb_list "${TRAIN_PDB_LIST}"
        --data.test_sets "${VAL_SET}"
        --data.${VAL_SET}.base_info.mmcif_dir "${RNA_CIF_DIR}"
        --data.${VAL_SET}.base_info.bioassembly_dict_dir "${BIOASSEMBLY_DIR}"
        --data.${VAL_SET}.base_info.indices_fpath "${VAL_INDICES_FPATH}"
        --data.${VAL_SET}.base_info.pdb_list "${VAL_PDB_LIST}"
        --data.${VAL_SET}.base_info.find_eval_chain_interface "$(as_bool "${VAL_FIND_EVAL_CHAIN_INTERFACE}")"
        --data.${VAL_SET}.base_info.group_by_pdb_id "$(as_bool "${VAL_GROUP_BY_PDB_ID}")"
        --data.${VAL_SET}.base_info.max_n_token "${VAL_MAX_N_TOKEN}"
        --data.msa.enable_rna_msa "$(as_bool "${USE_RNA_MSA}")"
        --data.msa.rna_msadir_raw_paths "${RNA_MSA_RAW_DIR}"
        --data.msa.rna_seq_or_filename_to_msadir_jsons "${RNA_MSA_INDEX_JSON}"
        --data.msa.rna_indexing_methods "${RNA_MSA_INDEXING_METHOD}"
        --data.msa.enable_prot_msa "$(as_bool "${USE_PROT_MSA}")"
        --data.template.enable_prot_template "$(as_bool "${USE_PROT_TEMPLATE}")"
    )
}

print_summary() {
    echo "========================================================"
    echo "  Protenix Unified Train Pipeline"
    echo "  config:             ${CONFIG_PATH}"
    echo "  train_mode:         ${TRAIN_MODE}"
    echo "  use_ribonanza:      ${USE_RIBONANZA}"
    echo "  use_rnalm:          ${USE_RNALM}"
    echo "  use_rna_ss:         ${USE_RNA_SS}"
    echo "  use_rna_template:   ${USE_RNA_TEMPLATE}"
    echo "  use_rna_msa:        ${USE_RNA_MSA}"
    echo "  use_rna_loss:       ${RNA_LOSS_ENABLE}"
    echo "  injection_mode:     ${INJECTION_MODE}"
    echo "  gate_mode:          ${GATE_MODE}"
    if [ "$(as_bool "${USE_RIBONANZA}")" = "true" ]; then
        echo "  rnet_model_dir:     ${RIBONANZA_MODEL_DIR}"
        echo "  rnet_gate_type:     ${RIBONANZA_GATE_TYPE}"
        echo "  rnet_pf_blocks:     ${RIBONANZA_N_PAIRFORMER_BLOCKS}"
    fi
    if [ "$(as_bool "${USE_RNA_SS}")" = "true" ]; then
        echo "  rna_ss_index:       ${RNA_SS_SEQUENCE_FPATH:-<empty>}"
        echo "  rna_ss_feature_dir: ${RNA_SS_FEATURE_DIR:-<empty>}"
        echo "  rna_ss_strict:      ${RNA_SS_STRICT}"
        echo "  rna_ss_arch:        ${RNA_SS_ARCHITECTURE}"
    fi
    echo "  rna_projector_init: ${RNA_PROJECTOR_INIT}"
    echo "  rna_template_alpha: ${RNA_TEMPLATE_ALPHA}"
    echo "  template_n_blocks:  ${TEMPLATE_N_BLOCKS}"
    echo "  max_steps:          ${MAX_STEPS}"
    echo "  run_name:           ${RUN_NAME_RESOLVED}"
    echo "  output_root:        ${OUTPUT_DIR}"
    if [ -n "${MANUAL_TEMPLATE_HINTS}" ]; then
        echo "  manual_hints:       ${MANUAL_TEMPLATE_HINTS}"
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
    build_common_args
    build_stage_args
    build_data_args

    mkdir -p "${OUTPUT_DIR}"
    print_summary

    CMD=(
        "${PYTHON_BIN}" ./runner/train.py
        "${COMMON_ARGS[@]}"
        "${STAGE_ARGS[@]}"
        "${RNALM_ARGS[@]}"
        "${RIBONANZA_ARGS[@]}"
        "${RNA_SS_ARGS[@]}"
        "${RNA_TEMPLATE_ARGS[@]}"
        "${DATA_ARGS[@]}"
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
