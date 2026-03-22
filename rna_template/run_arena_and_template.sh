#!/usr/bin/env bash
# ============================================================================
# run_arena_and_template.sh
#
# End-to-end pipeline:
#   1. Arena: fill missing atoms in an RNA PDB file
#   2. build_rna_template_protenix.py: compute Protenix-compatible template features
#
# Usage:
#   bash run_arena_and_template.sh <input.pdb> <chain_id> <output_dir> [arena_option]
#
# Example:
#   bash run_arena_and_template.sh /path/to/rna.pdb A /path/to/output 5
#
# Prerequisites:
#   - conda activate protenix
#   - Arena compiled at ARENA_DIR
#   - BioPython, numpy installed in the conda env
# ============================================================================
set -euo pipefail

# ---------- paths (edit if moved) ----------
ARENA_DIR="/inspire/ssd/project/sais-bio/public/ash_proj/Arena"
COMPUTE_DIR="/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute"

# ---------- args ----------
INPUT_PDB="${1:?Usage: $0 <input.pdb> <chain_id> <output_dir> [arena_option]}"
CHAIN_ID="${2:?Please specify chain ID (e.g., A)}"
OUTPUT_DIR="${3:?Please specify output directory}"
ARENA_OPT="${4:-5}"   # default: option 5 (fill all missing atoms)

mkdir -p "${OUTPUT_DIR}"

BASENAME="$(basename "${INPUT_PDB}" .pdb)"
ARENA_OUTPUT="${OUTPUT_DIR}/${BASENAME}_arena.pdb"
NPZ_OUTPUT="${OUTPUT_DIR}/${BASENAME}_template.npz"

# ===== Step 1: Arena =====
echo "========================================"
echo "[Step 1] Arena: filling missing atoms"
echo "  input:  ${INPUT_PDB}"
echo "  output: ${ARENA_OUTPUT}"
echo "  option: ${ARENA_OPT}"
echo "========================================"
"${ARENA_DIR}/Arena" "${INPUT_PDB}" "${ARENA_OUTPUT}" "${ARENA_OPT}"

if [ ! -s "${ARENA_OUTPUT}" ]; then
    echo "ERROR: Arena output is empty or missing." >&2
    exit 1
fi

ATOM_COUNT=$(grep -c "^ATOM" "${ARENA_OUTPUT}" || true)
echo "  Arena output: ${ATOM_COUNT} atoms"

# ===== Extract sequence from Arena output =====
QUERY_SEQ=$(grep "^ATOM" "${ARENA_OUTPUT}" \
    | awk -v chain="${CHAIN_ID}" '$5 == chain {printf "%s %s\n", $6, $4}' \
    | sort -un \
    | awk '{printf $2}')

if [ -z "${QUERY_SEQ}" ]; then
    # fallback: try without chain filter
    QUERY_SEQ=$(grep "^ATOM" "${ARENA_OUTPUT}" \
        | awk '{printf "%s %s\n", $6, $4}' \
        | sort -un \
        | awk '{printf $2}')
fi

SEQ_LEN=${#QUERY_SEQ}
echo "  Extracted sequence (${SEQ_LEN} nt): ${QUERY_SEQ}"

# ===== Step 2: build_rna_template_protenix.py =====
echo ""
echo "========================================"
echo "[Step 2] Computing RNA template features"
echo "  template: ${ARENA_OUTPUT}:${CHAIN_ID}"
echo "  output:   ${NPZ_OUTPUT}"
echo "========================================"
cd "${COMPUTE_DIR}"
python build_rna_template_protenix.py \
    --query_seq "${QUERY_SEQ}" \
    --template "${ARENA_OUTPUT}:${CHAIN_ID}" \
    --output "${NPZ_OUTPUT}" \
    --anchor_mode base_center_fallback

if [ ! -s "${NPZ_OUTPUT}" ]; then
    echo "ERROR: Template NPZ output is empty or missing." >&2
    exit 1
fi

# ===== Step 3: Quick validation =====
echo ""
echo "========================================"
echo "[Step 3] Validating output"
echo "========================================"
python3 -c "
import numpy as np
data = np.load('${NPZ_OUTPUT}', allow_pickle=True)
core = ['template_aatype','template_distogram','template_pseudo_beta_mask',
        'template_unit_vector','template_backbone_frame_mask']
ok = True
for k in core:
    if k not in data:
        print(f'  MISSING: {k}')
        ok = False
    else:
        arr = data[k]
        print(f'  {k}: shape={arr.shape}, dtype={arr.dtype}')
mask = data['template_anchor_mask_1d']
valid = int(mask[0].sum())
total = mask.shape[1]
print(f'  anchor coverage: {valid}/{total} ({100*valid/total:.1f}%)')
uv = data['template_unit_vector']
norms = np.linalg.norm(uv, axis=-1)
valid_norms = norms[norms > 0]
if len(valid_norms) > 0 and np.allclose(valid_norms, 1.0, atol=1e-4):
    print('  unit vectors: PASS (all norm=1)')
else:
    print('  unit vectors: WARN')
    ok = False
if ok:
    print('  => ALL CHECKS PASSED')
else:
    print('  => SOME CHECKS FAILED')
"

echo ""
echo "========================================"
echo "DONE"
echo "  Arena output:    ${ARENA_OUTPUT}"
echo "  Template NPZ:    ${NPZ_OUTPUT}"
echo "========================================"
