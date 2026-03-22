# RNA/DNA LLM Embedding Integration — Code Review Fix Report

**Date:** 2026-03-12
**Scope:** Three issues identified in the RNA/DNA LLM embedding injection pipeline
**Status:** All three issues fixed and verified

---

## Overview

This report documents the three issues found during code review of the RNA/DNA LLM embedding integration into Protenix, and the fixes applied.

### Files Modified

| File | Changes |
|------|---------|
| `configs/configs_base.py` | Removed `fusion_method` config option |
| `protenix/model/modules/diffusion.py` | Removed concat branch in `DiffusionConditioning`; added use_rna/use_dna sync guard |
| `protenix/model/modules/embedders.py` | Added use_rna/use_dna sync guard in `InputFeatureEmbedder` |
| `protenix/model/protenix.py` | Added use_rna/use_dna sync guard; cleaned fusion_method log |
| `protenix/data/pipeline/dataset.py` | Changed `get_rnalm_featurizer()` from warn-and-return-None to fail-fast |
| `protenix/data/inference/infer_dataloader.py` | Changed inference entry from "fall back to random" to fail-fast |
| `finetune_rna_only_diffuse.sh` | Removed all fusion_method references |
| `finetune_rna_dna.sh` | Removed all fusion_method references |
| `finetune_rna_only.sh` | Removed all fusion_method references |
| `finetune/finetune_1stage.sh` | Removed all fusion_method references |
| `finetune/finetune_2stage.sh` | Removed all fusion_method references |
| `finetune/test_gpu_small.sh` | Removed `--rnalm.fusion_method` arg |
| `rna_llm_gate_tune.sh` | Removed `--rnalm.fusion_method` arg |
| `rna_2stage_slow.sh` | Removed `--rnalm.fusion_method` arg |
| `rna_2stage_fast.sh` | Removed `--rnalm.fusion_method` arg |
| `run_two_stage_training_rna_loss.sh` | Removed `--rnalm.fusion_method` arg |
| `protenix_rna_llm_onestage_joint.sh` | Removed `--rnalm.fusion_method` arg |
| `protenix_rna_llm_both_inject.sh` | Removed `--rnalm.fusion_method` arg |

---

## Issue 1: Remove `concat` Fusion Method (Keep Only `add`)

### Problem

The `fusion_method` config supported both `"add"` and `"concat"` modes for injecting RNA LLM embeddings into `DiffusionConditioning`:

- **`add`**: `s_i = concat([s_trunk + s_rnalm, s_inputs])` — reuses existing `layernorm_s` and `linear_no_bias_s` layers, adds zero new parameters.
- **`concat`**: `s_i = concat([s_trunk, s_inputs, s_rnalm])` — requires new `layernorm_s_concat` and `linear_no_bias_s_concat` layers with wider input dim.

The `concat` mode was never properly tested with all config combinations (especially `separate_dna_projection=True`), had no clear benefit over `add` (which is the approach used by ESM), and added code complexity.

### Fix

**Removed all `concat` code paths:**

1. **`DiffusionConditioning.__init__`** — Removed creation of `layernorm_s_concat` and `linear_no_bias_s_concat` layers. Removed `self.rnalm_fusion_method` attribute.

2. **`DiffusionConditioning.forward`** — Simplified the single conditioning branch. Instead of if/else on fusion_method, the code now always uses the `add` path when `s_rnalm` is provided:
   ```python
   if s_rnalm is not None and self.rnalm_enable:
       single_s = torch.cat([s_trunk + s_rnalm, s_inputs], dim=-1)
   else:
       single_s = torch.cat([s_trunk, s_inputs], dim=-1)
   single_s = self.linear_no_bias_s(self.layernorm_s(single_s))
   ```

3. **`configs/configs_base.py`** — Removed `"fusion_method": "add"` from `rnalm` config. Updated docstring.

4. **All shell scripts** — Removed `FUSION_METHOD` variable definitions, `--fusion_method` arg parsing, validation blocks, echo lines, and `--rnalm.fusion_method` python args from 12 scripts.

### Impact

- Reduces model code complexity
- Eliminates an untested code path
- Existing checkpoints that used `add` mode are fully compatible (the `layernorm_s_concat` / `linear_no_bias_s_concat` keys were never in any saved checkpoint)
- The `add` approach is architecturally consistent with how ESM embeddings are injected

---

## Issue 2: Fail-Fast for Missing Embeddings

### Problem

When `rnalm.enable=True` but embedding paths were missing or invalid:

1. **Training (`get_rnalm_featurizer()` in `dataset.py`)**: Logged a warning and returned `None`, causing training to silently proceed without any RNA/DNA embeddings. The model would still expect embeddings (especially in the combined pathway), leading to either:
   - RuntimeError from missing `rnalm_token_embedding` key (noisy failure)
   - Or silently training with no LLM signal (worse — silent quality degradation)

2. **Inference (`InferenceDataset.__init__()` in `infer_dataloader.py`)**: Logged "Model will fall back to random embeddings" — actively misleading. If the model was finetuned with embeddings, inference without them would produce degraded results silently.

### Fix

**Both entry points now raise `ValueError` immediately when paths are missing:**

#### Training (`get_rnalm_featurizer()`)
```python
# Before (warn and return None):
if use_rna and (not embedding_dir or not sequence_fpath):
    logger.warning("use_rna_embed=True but no RNA embedding paths.")

# After (fail-fast):
if use_rna and (not embedding_dir or not sequence_fpath):
    raise ValueError(
        "rnalm.enable=True and use_rna_embed=True but RNA embedding paths are missing. "
        f"embedding_dir='{embedding_dir}', sequence_fpath='{sequence_fpath}'. "
        "Either provide valid paths or set rnalm.use_rna_embed=false."
    )
```

Same pattern for DNA paths.

#### Inference (`InferenceDataset.__init__()`)
```python
# Before (warn and set featurizer=None, model falls back to "random"):
logger.warning("RiNALMo embedding paths not found. Model will fall back to random embeddings.")

# After (fail-fast):
if use_rna and (not rnalm_embedding_dir or not rnalm_sequence_fpath):
    raise ValueError(
        "rnalm.enable=True and use_rna_embed=True but RNA embedding paths "
        f"are missing for inference. ..."
    )
```

### Impact

- **Training**: Immediately surfaces misconfiguration instead of silent quality degradation
- **Inference**: Prevents running a finetuned model without the embeddings it was trained with
- Error messages include the actual path values and suggest the fix (`set use_rna_embed=false`)
- Follows the same fail-fast pattern already used by `RiNALMoFeaturizer.__init__()` internally

### Note on `RiNALMoFeaturizer` Constructor

The constructor itself already had fail-fast (lines 80-123 of `rnalm_featurizer.py`). The issue was that the *callers* (`get_rnalm_featurizer()` and `InferenceDataset`) would catch the case *before* reaching the constructor and silently skip it. Now the callers also enforce the same contract.

---

## Issue 3: `use_rna_embed=false && use_dna_embed=false` Not Equivalent to `rnalm.enable=false`

### Problem

When both `use_rna_embed` and `use_dna_embed` are set to `false`:

- **Data layer**: `get_rnalm_featurizer()` would return `None` or disable the featurizer — **no embedding tensors produced**.
- **Model layer**: `rnalm_enable` was still `True`, causing:
  - In `separate_dna_projection=True` path: Model creates `rna_projection` and/or `dna_projection` layers, then `_get_s_rnalm()` tries to access `rna_llm_embedding`/`dna_llm_embedding` from `input_feature_dict` → **RuntimeError**
  - In `separate_dna_projection=False` path: Model creates `rnalm_projection` layer, then `_get_s_rnalm()` tries to access `rnalm_token_embedding` → **RuntimeError**
  - In `DiffusionConditioning`: `self.rnalm_enable=True` but `s_rnalm=None` → falls through to original path (happens to work but is semantically wrong)

The report claim "use_rna_embed/use_dna_embed=False is equivalent to rnalm.enable=False" was **not strictly true** in the model layer.

### Fix

**Added sync guards at all three model initialization points:**

1. **`Protenix.__init__()` (protenix.py)**:
   ```python
   if self.rnalm_enable and not self.rnalm_use_rna and not self.rnalm_use_dna:
       logger.warning(
           "rnalm.enable=True but both use_rna_embed and use_dna_embed are False. "
           "Disabling rnalm at model layer to stay consistent with data layer."
       )
       self.rnalm_enable = False
   ```
   This prevents creation of projection layers and gates, and ensures `_get_s_rnalm()` returns `None`.

2. **`InputFeatureEmbedder.__init__()` (embedders.py)**:
   ```python
   if not _use_rna and not _use_dna:
       rnalm_enable = False
   ```
   Prevents creation of `linear_rnalm` / `linear_rna_llm` / `linear_dna_llm` layers.

3. **`DiffusionConditioning.__init__()` (diffusion.py)**:
   ```python
   if self.rnalm_enable:
       if not _use_rna and not _use_dna:
           self.rnalm_enable = False
   ```
   Ensures `forward()` always takes the original path when no embeddings are available.

### Impact

- **`use_rna_embed=false && use_dna_embed=false` is now truly equivalent to `rnalm.enable=false`** across all layers
- No orphan projection layers are created
- No RuntimeErrors from missing embedding keys
- The model behaves identically to the base model (no rnalm) when both embed flags are off
- Works correctly for all config combinations:

| `use_rna` | `use_dna` | `separate_dna` | Behavior |
|-----------|-----------|----------------|----------|
| true | true | true | RNA + DNA separate projections |
| true | true | false | Combined rnalm_projection |
| true | false | true | RNA projection only, no dna_projection created |
| true | false | false | Combined rnalm_projection (DNA entities get zeros) |
| false | true | true | DNA projection only, no rna_projection created |
| false | true | false | Combined rnalm_projection (RNA entities get zeros) |
| false | false | any | **rnalm disabled entirely** (no projections, no embedding requirements) |

---

## Verification Checklist

### Training Path
- [x] `get_rnalm_featurizer()` raises `ValueError` when `use_rna=True` but paths missing
- [x] `get_rnalm_featurizer()` raises `ValueError` when `use_dna=True` but paths missing
- [x] `get_rnalm_featurizer()` returns `None` when `enable=False`
- [x] `get_rnalm_featurizer()` returns `None` when both `use_rna=False` and `use_dna=False`
- [x] `RiNALMoFeaturizer` constructor fail-fast works (pre-existing)
- [x] Model layer syncs `rnalm_enable=False` when both use flags off

### Inference Path
- [x] `InferenceDataset` raises `ValueError` when `use_rna=True` but paths missing
- [x] `InferenceDataset` raises `ValueError` when `use_dna=True` but paths missing
- [x] `InferenceDataset` disables when both `use_rna=False` and `use_dna=False`
- [x] `InferenceDataset` creates featurizer correctly when paths valid
- [x] Inference `process_one()` correctly produces `rna_llm_embedding` / `dna_llm_embedding` / `rnalm_token_embedding` in `features_dict`
- [x] Model forward pass receives embeddings via `input_feature_dict` and projects them

### Config Consistency
- [x] `fusion_method` removed from `configs_base.py`
- [x] No remaining `fusion_method` / `FUSION_METHOD` in any `.py` or `.sh` file (except benign comments)
- [x] All shell scripts updated and consistent
- [x] `DiffusionConditioning` no longer has `layernorm_s_concat` or `linear_no_bias_s_concat`

### Backward Compatibility
- [x] Existing checkpoints trained with `fusion_method="add"` load correctly (no state_dict key changes)
- [x] Existing checkpoints without rnalm keys load correctly (`load_strict=false`)
- [x] Config files that still specify `fusion_method` will get a harmless ignored key (no crash)

---

## Summary of Architectural Invariants (Post-Fix)

1. **Single fusion strategy**: RNA LLM embeddings are always injected via `add` (added to `s_trunk` before concatenation with `s_inputs`). This mirrors ESM injection and introduces zero new weight matrices in the DiffusionConditioning.

2. **Fail-fast everywhere**: If `rnalm.enable=True` and any required embedding path is missing, the system raises `ValueError` immediately — no silent fallback to zeros or random embeddings.

3. **Data-model layer consistency**: When both `use_rna_embed` and `use_dna_embed` are `False`, the model layer disables rnalm entirely. No projection layers are created, `_get_s_rnalm()` returns `None`, and the model behaves identically to base Protenix.
