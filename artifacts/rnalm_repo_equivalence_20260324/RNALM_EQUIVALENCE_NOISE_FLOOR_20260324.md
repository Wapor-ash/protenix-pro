# RNALM Equivalence Noise Floor

- Date: 2026-03-24
- Purpose: establish same-repo / same-case GPU repeat noise floor, so cross-repo tiny numeric drift is not misread as behavior drift.

## Repeat 1

- Case: `protenix_new`, `rna_157d_A`, `rnalm_off`
- raw input max abs diff: `0.0`
- prepared input max abs diff: `0.0`
- pairformer max abs diff: `0.00030517578125`
- prediction max abs diff: `0.0009765625`

## Repeat 2

- Case: `protenix_pro`, `hybrid_165d_A`, `rnalm_on_diffusion`
- raw input max abs diff: `0.0`
- prepared input max abs diff: `0.0`
- pairformer max abs diff: `0.0003662109375`
- prediction max abs diff: `0.006799221038818359`

## Interpretation

- Same-repo repeated GPU runs are not bitwise identical after the model trunk begins.
- The observed cross-repo deltas in the main report fall within this repeat-run noise floor.
- Therefore the correct conclusion is "numerically consistent within GPU repeat noise", not "bitwise identical".
