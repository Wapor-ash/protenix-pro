"""
Thin import bridge: re-exports rna_template_common functions for use in the
online featurizer without modifying the compute/ directory or sys.path hacks.
"""

import os
import sys

# Add compute dir to sys.path so we can import rna_template_common
_COMPUTE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "rna_template", "compute"
)
if _COMPUTE_DIR not in sys.path:
    sys.path.insert(0, _COMPUTE_DIR)

from rna_template_common import (  # noqa: E402, F401
    load_structure_residues,
    build_minimal_template_arrays,
    normalize_query_sequence,
    stack_template_dicts,
    residues_to_sequence,
    align_query_to_template,
    compute_anchor,
    compute_frame,
    compute_distogram,
    compute_unit_vectors,
    ResidueRecord,
)
