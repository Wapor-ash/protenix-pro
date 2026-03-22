#!/bin/bash
# Cross-template-only small E2E wrapper.
#
# The previous self-template smoke test was removed to avoid accidental
# self-template usage and data leakage. Use the cross-only RNA3D E2E test
# with a small subset instead.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/test_rna3d_e2e.sh" --num_test 10 "$@"
