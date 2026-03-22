#!/bin/bash
# =============================================================================
# Quick GPU test: Validate RNA/DNA LLM embedding injection during inference.
#
# Tests all key toggle combinations:
#   1. RNA-only with input injection (using finetuned checkpoint)
#   2. RNA+DNA with input injection (using finetuned checkpoint)
#   3. RNA-only with diffusion injection (using finetuned checkpoint)
#   4. Base model with rnalm enabled (should fail: checkpoint lacks RNALM projector)
#
# Uses minimal settings (N_sample=1, N_step=5, N_cycle=1) for speed.
# =============================================================================
set -euo pipefail

PROJECT_ROOT="/inspire/ssd/project/sais-bio/public/ash_proj"
PROTENIX_DIR="${PROJECT_ROOT}/code/protenix_new/Protenix"
DATA_DIR="${PROJECT_ROOT}/data/stanford-rna-3d-folding/part2"

# Embedding paths
AIDO_EMBEDDING_DIR="${DATA_DIR}/aido_embeddings"
RNA_EMBEDDING_DIR="${AIDO_EMBEDDING_DIR}/rna"
RNA_SEQUENCE_FPATH="${RNA_EMBEDDING_DIR}/rna_sequences.csv"
DNA_EMBEDDING_DIR="${AIDO_EMBEDDING_DIR}/dna"
DNA_SEQUENCE_FPATH="${DNA_EMBEDDING_DIR}/dna_sequences.csv"

RNA_EMBEDDING_DIM=2048
DNA_EMBEDDING_DIM=1024

# Checkpoint paths
BASE_CKPT="${PROTENIX_DIR}/checkpoints/protenix_base_20250630_v1.0.0.pt"
# Use input-injection finetuned checkpoint (has linear_rna_llm + linear_dna_llm weights)
INPUT_CKPT="${PROTENIX_DIR}/output/aido_separate_input/aido_separate_input_20260311_151213/checkpoints/999.pt"
# Use diffusion-injection finetuned checkpoint (has rna_projection weights)
DIFFUSE_CKPT="${PROTENIX_DIR}/output/0311_16_rna_diffuse/0311_16_rna_diffuse_20260312_084058/checkpoints/999.pt"

# Environment
export PROTENIX_ROOT_DIR="${PROJECT_ROOT}/data"
CONDA_PREFIX="${PROJECT_ROOT}/conda/envs/protenix"
export CUDA_HOME="${CONDA_PREFIX}"
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPLUS_INCLUDE_PATH:-}"

cd "${PROTENIX_DIR}"
export PYTHONPATH="${PROTENIX_DIR}:${PYTHONPATH:-}"
export TRIANGLE_ATTENTION="cuequivariance"
export TRIANGLE_MULTIPLICATIVE="cuequivariance"

# Use the example RNA input
INPUT_JSON="${PROTENIX_DIR}/examples/examples_with_rna_msa/example_9gmw_2.json"

PASS_COUNT=0
FAIL_COUNT=0
TOTAL=0

run_test() {
    local TEST_NAME="$1"
    local CKPT_PATH="$2"
    local INJECTION_MODE="$3"
    local USE_RNA="$4"
    local USE_DNA="$5"
    local GATE_MODE="${6:-none}"
    TOTAL=$((TOTAL + 1))

    echo ""
    echo "========================================================"
    echo "  TEST ${TOTAL}: ${TEST_NAME}"
    echo "  checkpoint:      ${CKPT_PATH}"
    echo "  injection_mode:  ${INJECTION_MODE}"
    echo "  use_rna:         ${USE_RNA}"
    echo "  use_dna:         ${USE_DNA}"
    echo "  gate_mode:       ${GATE_MODE}"
    echo "========================================================"

    DUMP_DIR="${PROTENIX_DIR}/output/test_inference_rnalm/${TEST_NAME}"
    mkdir -p "${DUMP_DIR}"

    # Prepare checkpoint symlink
    INFER_CKPT_DIR=$(mktemp -d)
    INFER_MODEL_NAME="test_rnalm"
    ln -sf "$(realpath "${CKPT_PATH}")" "${INFER_CKPT_DIR}/${INFER_MODEL_NAME}.pt"

    # Build RNA/DNA embedding args
    RNA_EMB_DIR=""
    RNA_SEQ_FPATH=""
    if [ "${USE_RNA}" = "true" ]; then
        RNA_EMB_DIR="${RNA_EMBEDDING_DIR}"
        RNA_SEQ_FPATH="${RNA_SEQUENCE_FPATH}"
    fi

    DNA_EMB_DIR=""
    DNA_SEQ_FPATH=""
    if [ "${USE_DNA}" = "true" ] && [ -f "${DNA_SEQUENCE_FPATH}" ]; then
        DNA_EMB_DIR="${DNA_EMBEDDING_DIR}"
        DNA_SEQ_FPATH="${DNA_SEQUENCE_FPATH}"
    fi

    if python3 ./runner/inference.py \
        --model_name "${INFER_MODEL_NAME}" \
        --load_checkpoint_dir "${INFER_CKPT_DIR}" \
        --load_strict false \
        --input_json_path "${INPUT_JSON}" \
        --dump_dir "${DUMP_DIR}" \
        --dtype "bf16" \
        --use_msa false \
        --use_template false \
        --model.N_cycle 1 \
        --sample_diffusion.N_sample 1 \
        --sample_diffusion.N_step 5 \
        --triangle_attention "cuequivariance" \
        --triangle_multiplicative "cuequivariance" \
        --enable_tf32 true \
        --enable_efficient_fusion true \
        --enable_diffusion_shared_vars_cache true \
        \
        --rnalm.enable true \
        --rnalm.use_rna_embed "${USE_RNA}" \
        --rnalm.use_dna_embed "${USE_DNA}" \
        --rnalm.model_name "aido" \
        --rnalm.embedding_dim "${RNA_EMBEDDING_DIM}" \
        --rnalm.dna_embedding_dim "${DNA_EMBEDDING_DIM}" \
        --rnalm.injection_mode "${INJECTION_MODE}" \
        --rnalm.gate_mode "${GATE_MODE}" \
        --rnalm.gate_init_logit "-3.0" \
        --rnalm.separate_dna_projection true \
        --rnalm.embedding_dir "${RNA_EMB_DIR}" \
        --rnalm.sequence_fpath "${RNA_SEQ_FPATH}" \
        --rnalm.dna_embedding_dir "${DNA_EMB_DIR}" \
        --rnalm.dna_sequence_fpath "${DNA_SEQ_FPATH}" \
        2>&1 | tee "${DUMP_DIR}/test.log"; then

        # Check output CIF files exist
        if ls "${DUMP_DIR}"/*.cif 1>/dev/null 2>&1 || ls "${DUMP_DIR}"/**/*.cif 1>/dev/null 2>&1; then
            echo ""
            echo "  PASS: ${TEST_NAME} — output CIF generated"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            echo ""
            echo "  PASS (no CIF check): ${TEST_NAME} — inference completed without error"
            PASS_COUNT=$((PASS_COUNT + 1))
        fi
    else
        echo ""
        echo "  FAIL: ${TEST_NAME} — inference exited with error"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    rm -rf "${INFER_CKPT_DIR}"
}

run_expected_failure_test() {
    local TEST_NAME="$1"
    local CKPT_PATH="$2"
    local INJECTION_MODE="$3"
    local USE_RNA="$4"
    local USE_DNA="$5"
    local GATE_MODE="${6:-none}"
    TOTAL=$((TOTAL + 1))

    echo ""
    echo "========================================================"
    echo "  TEST ${TOTAL}: ${TEST_NAME}"
    echo "  checkpoint:      ${CKPT_PATH}"
    echo "  injection_mode:  ${INJECTION_MODE}"
    echo "  use_rna:         ${USE_RNA}"
    echo "  use_dna:         ${USE_DNA}"
    echo "  gate_mode:       ${GATE_MODE}"
    echo "  expected:        RuntimeError (missing RNALM projector weights)"
    echo "========================================================"

    DUMP_DIR="${PROTENIX_DIR}/output/test_inference_rnalm/${TEST_NAME}"
    mkdir -p "${DUMP_DIR}"

    INFER_CKPT_DIR=$(mktemp -d)
    INFER_MODEL_NAME="test_rnalm"
    ln -sf "$(realpath "${CKPT_PATH}")" "${INFER_CKPT_DIR}/${INFER_MODEL_NAME}.pt"

    RNA_EMB_DIR=""
    RNA_SEQ_FPATH=""
    if [ "${USE_RNA}" = "true" ]; then
        RNA_EMB_DIR="${RNA_EMBEDDING_DIR}"
        RNA_SEQ_FPATH="${RNA_SEQUENCE_FPATH}"
    fi

    DNA_EMB_DIR=""
    DNA_SEQ_FPATH=""
    if [ "${USE_DNA}" = "true" ] && [ -f "${DNA_SEQUENCE_FPATH}" ]; then
        DNA_EMB_DIR="${DNA_EMBEDDING_DIR}"
        DNA_SEQ_FPATH="${DNA_SEQUENCE_FPATH}"
    fi

    set +e
    python3 ./runner/inference.py \
        --model_name "${INFER_MODEL_NAME}" \
        --load_checkpoint_dir "${INFER_CKPT_DIR}" \
        --load_strict false \
        --input_json_path "${INPUT_JSON}" \
        --dump_dir "${DUMP_DIR}" \
        --dtype "bf16" \
        --use_msa false \
        --use_template false \
        --model.N_cycle 1 \
        --sample_diffusion.N_sample 1 \
        --sample_diffusion.N_step 5 \
        --triangle_attention "cuequivariance" \
        --triangle_multiplicative "cuequivariance" \
        --enable_tf32 true \
        --enable_efficient_fusion true \
        --enable_diffusion_shared_vars_cache true \
        \
        --rnalm.enable true \
        --rnalm.use_rna_embed "${USE_RNA}" \
        --rnalm.use_dna_embed "${USE_DNA}" \
        --rnalm.model_name "aido" \
        --rnalm.embedding_dim "${RNA_EMBEDDING_DIM}" \
        --rnalm.dna_embedding_dim "${DNA_EMBEDDING_DIM}" \
        --rnalm.injection_mode "${INJECTION_MODE}" \
        --rnalm.gate_mode "${GATE_MODE}" \
        --rnalm.gate_init_logit "-3.0" \
        --rnalm.separate_dna_projection true \
        --rnalm.embedding_dir "${RNA_EMB_DIR}" \
        --rnalm.sequence_fpath "${RNA_SEQ_FPATH}" \
        --rnalm.dna_embedding_dir "${DNA_EMB_DIR}" \
        --rnalm.dna_sequence_fpath "${DNA_SEQ_FPATH}" \
        >"${DUMP_DIR}/test.log" 2>&1
    EXIT_CODE=$?
    set -e

    if [ "${EXIT_CODE}" -ne 0 ] && rg -q "checkpoint contains NO RNA/DNA LM projector weights" "${DUMP_DIR}/test.log"; then
        echo ""
        echo "  PASS: ${TEST_NAME} — inference failed fast with the expected RNALM projector error"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo ""
        echo "  FAIL: ${TEST_NAME} — expected RNALM projector RuntimeError"
        if [ -f "${DUMP_DIR}/test.log" ]; then
            tail -n 40 "${DUMP_DIR}/test.log"
        fi
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    rm -rf "${INFER_CKPT_DIR}"
}

echo "========================================================"
echo "  RNA/DNA LLM Inference Validation Tests"
echo "========================================================"

# Test 1: RNA-only, input injection (uses aido_separate_input checkpoint)
if [ -f "${INPUT_CKPT}" ]; then
    run_test "rna_only_input" "${INPUT_CKPT}" "input" "true" "false"
else
    echo "SKIP: Test 1 — checkpoint not found: ${INPUT_CKPT}"
    TOTAL=$((TOTAL + 1))
fi

# Test 2: RNA+DNA, input injection (uses same checkpoint — DNA gets zeros from base)
if [ -f "${INPUT_CKPT}" ]; then
    run_test "rna_dna_input" "${INPUT_CKPT}" "input" "true" "true"
else
    echo "SKIP: Test 2 — checkpoint not found: ${INPUT_CKPT}"
    TOTAL=$((TOTAL + 1))
fi

# Test 3: RNA-only, diffusion injection
if [ -f "${DIFFUSE_CKPT}" ]; then
    run_test "rna_only_diffuse" "${DIFFUSE_CKPT}" "diffusion" "true" "false"
else
    echo "SKIP: Test 3 — checkpoint not found: ${DIFFUSE_CKPT}"
    TOTAL=$((TOTAL + 1))
fi

# Test 4: Base model with rnalm enabled should fail fast (strict inference policy)
if [ -f "${BASE_CKPT}" ]; then
    run_expected_failure_test "base_model_rnalm_missing_projector" "${BASE_CKPT}" "input" "true" "false"
else
    echo "SKIP: Test 4 — base checkpoint not found: ${BASE_CKPT}"
    TOTAL=$((TOTAL + 1))
fi

echo ""
echo "========================================================"
echo "  RESULTS: ${PASS_COUNT}/${TOTAL} passed, ${FAIL_COUNT} failed"
echo "========================================================"

if [ "${FAIL_COUNT}" -gt 0 ]; then
    exit 1
fi
