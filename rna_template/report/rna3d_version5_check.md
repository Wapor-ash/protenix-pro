# RNA Template v5 Review Check

Date: 2026-03-16

Scope:
- Prompt / target: `/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build-manual.txt`
- Design: `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/understanding/rna_template_manual_override_and_fallback_design_20260316.md`
- Risk review: `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/understanding/rna_template_resolver_integration_risks_20260316.md`
- Claimed completion report: `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_version5.md`
- Implementation reviewed: `protenix/data/rna_template/rna_template_featurizer.py`
- Tests reviewed/run: `finetune/test_manual_template_override.py`, `finetune/test_manual_template_e2e.py`

## Verdict

`v5` is **partially implemented**, but I do **not** think it is fully closed as written in `rna3d_version5.md`.

What is clearly present:
- manual structure / manual NPZ entry points
- entity-level `templateHints`
- `manual_only / prefer_manual / hybrid / default_only`
- slot-level merge with default search fallback
- default no-hint path left largely intact

What is still materially incomplete or risky:
- manual override is effectively **inference-only**, not finetune/training-integrated
- manual templates bypass provenance / temporal / self-hit governance
- the test evidence in `rna3d_version5.md` is not reproducible in the current environment and overstates actual coverage
- public JSON/schema docs are not updated for `templateHints`

## Findings

### 1. High: manual override is not wired for training/finetune, despite the prompt emphasizing finetune validation

The original task explicitly emphasized real finetune validation and manual-template behavior during finetune-like usage:

- `/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build-manual.txt:37`
- `/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build-manual.txt:40`

The current implementation only extracts `templateHints` in `inference_mode=True`:

- `protenix/data/rna_template/rna_template_featurizer.py:1276`
- `protenix/data/rna_template/rna_template_featurizer.py:1293`

In the training branch, RNA sequences are collected, but manual hints are not extracted at all:

- `protenix/data/rna_template/rna_template_featurizer.py:1306`
- `protenix/data/rna_template/rna_template_featurizer.py:1324`

The v5 report also states this explicitly:

- `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_version5.md:180`

Impact:
- If the intended contract was "manual template can be used in finetune samples / training data", this task is not actually complete.
- What exists today is an inference-time manual override layer, not a training-integrated resolver.


### 4. Medium: `templateHints` is implemented as a private extension, but public JSON/docs were not updated

The implementation reads `templateHints` from RNA inference JSON:

- `protenix/data/rna_template/rna_template_featurizer.py:1243`
- `protenix/data/rna_template/rna_template_featurizer.py:1293`

But the public input format doc still documents `rnaSequence` without any template field:

- `docs/infer_json_format.md:111`
- `docs/infer_json_format.md:138`

I also did not find `templateHints` mentioned in `docs/`, `examples/`, `runner/`, or `protenix/web_service`.

Impact:
- This is schema drift.
- The feature may work for a user who already knows the hidden field, but it is not yet a stable/public interface.

### 5. Medium: RNA-first reclassified DNA-with-U entities can enter RNA template search, but cannot carry manual hints through the same path

In inference mode, manual hints are extracted only for `rnaSequence` / `rnaChain`:

- `protenix/data/rna_template/rna_template_featurizer.py:1279`
- `protenix/data/rna_template/rna_template_featurizer.py:1288`

`dnaSequence` / `dnaChain` with uracil are later reclassified as RNA for search:

- `protenix/data/rna_template/rna_template_featurizer.py:1294`
- `protenix/data/rna_template/rna_template_featurizer.py:1305`

But in that DNA reclassification branch, no `templateHints` are extracted.

Impact:
- If a user has a hybrid/mislabelled DNA-style entity that is intentionally reclassified into RNA handling, default RNA template search can apply, but manual override cannot be attached via the same JSON contract.
- This is a boundary inconsistency in the current RNA-first logic.

## What Looks Correct

These parts look consistent with the stated narrow design:

- manual builder is separate from the search-hit builder:
  - `protenix/data/rna_template/rna_template_featurizer.py:473`
  - `protenix/data/rna_template/rna_template_featurizer.py:548`

- slot-level merge is narrow and easy to reason about:
  - `protenix/data/rna_template/rna_template_featurizer.py:995`
  - `protenix/data/rna_template/rna_template_featurizer.py:1032`

- entities without hints still go through the existing online/offline branches:
  - `protenix/data/rna_template/rna_template_featurizer.py:1109`
  - `protenix/data/rna_template/rna_template_featurizer.py:1155`

So the core implementation direction is reasonable. The main issues are contract scope and validation completeness, not that the feature is entirely absent.

## Test / Reproduction Notes

Commands run:

```bash
python /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/finetune/test_manual_template_override.py
python /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/finetune/test_manual_template_e2e.py
python - <<'PY'
import sys
print(sys.executable)
try:
    import Bio
    print("Bio OK")
except Exception as e:
    print("Bio FAIL", repr(e))
PY
```

Observed:

```text
/opt/conda/bin/python
Bio FAIL ModuleNotFoundError("No module named 'Bio'")
```

This blocks local reproduction of the claimed pass counts. Since `biopython==1.85` is declared in `requirements.txt`, this is best described as a validation-environment gap rather than a direct code defect.

## Bottom Line

My review conclusion is:

1. The narrow inference-side manual override mechanism is implemented.
2. The task is **not fully closed** against the original prompt if finetune/training manual support was intended.
4. The biggest process risk is that the report currently over-claims verification relative to what is reproducible from this workspace.

## Recommended Next Steps

4. Re-run the validation in a fully provisioned environment, especially for the GPU claim.
