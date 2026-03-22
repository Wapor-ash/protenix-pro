# RNA LLM Embedding Injection Design Report

## Overview

This document describes the design and implementation of configurable RNA LLM embedding injection into Protenix, supporting three injection modes that are independently selectable and do not interfere with existing training scripts or two-stage pipelines.

## Background: How ESM Is Injected in Protenix

ESM (protein language model) embeddings are injected at the **InputFeatureEmbedder** level (Algorithm 2 in AF3):

```
ESM embedding [N_token, 2560]
    → LinearNoBias(2560, 449)  [zero-init]
    → s_inputs = s_inputs + esm_projected
```

Key properties:
- **Zero-initialization**: `nn.init.zeros_(self.linear_esm.weight)` ensures the model starts identically to the baseline without ESM
- **Additive residual**: ESM is added to `s_inputs` before it enters the trunk
- **Early injection**: Happens before pairformer, so ESM information flows through the entire trunk

## How Existing RNA LM (RiNALMo) Is Injected

The existing RNA LM injection operates at the **DiffusionConditioning** level (Algorithm 21 in AF3):

```
RNA LM embedding [N_token, 1280]
    → rnalm_projection: LinearNoBias(1280, 384)  [zero-init]
    → optional gating (scalar/token/dual)
    → s_rnalm [N_token, c_s=384]
    → DiffusionConditioning: s_trunk + s_rnalm before diffusion
```

Key properties:
- **Late injection**: Happens at diffusion conditioning, after the pairformer trunk
- **Zero-initialization**: Projection weights initialized to zeros
- **Fusion methods**: `"add"` (s_trunk + s_rnalm) or `"concat"` (separate third channel)

## New Design: Configurable `injection_mode`

### Configuration

A new `injection_mode` field is added to the `rnalm` config:

```python
"rnalm": {
    "enable": True,
    "embedding_dim": 1280,
    "injection_mode": "diffusion",  # "diffusion" | "input" | "both"
    ...
}
```

### Three Injection Modes

#### 1. `injection_mode = "diffusion"` (existing, default)

```
RiNALMo [N_token, 1280]
    → Protenix.rnalm_projection: LinearNoBias(1280, 384) [zero-init]
    → optional gate
    → s_rnalm [N_token, 384]
    → DiffusionConditioning: concat([s_trunk + s_rnalm, s_inputs])
```

- **Where**: DiffusionConditioning module (late injection)
- **New params**: 1280 x 384 = 491,520 (projection only)
- **Backward compatible**: Default behavior, no changes to existing code paths

#### 2. `injection_mode = "input"` (new, like ESM)

```
RiNALMo [N_token, 1280]
    → InputFeatureEmbedder.linear_rnalm: LinearNoBias(1280, 449) [zero-init]
    → s_inputs = s_inputs + rnalm_projected
```

- **Where**: InputFeatureEmbedder (early injection, like ESM)
- **New params**: 1280 x 449 = 574,720 (projection only)
- **Intuition**: RNA LLM knowledge flows through the entire pairformer trunk, affecting both single and pair representations. This mimics ESM's proven injection strategy for protein LMs.
- **No diffusion-level layers created**: DiffusionModule receives `rnalm_configs=None`, no s_rnalm is passed

#### 3. `injection_mode = "both"` (new, dual injection)

```
RiNALMo [N_token, 1280]
    ├── InputFeatureEmbedder.linear_rnalm: LinearNoBias(1280, 449) [zero-init]
    │   → s_inputs = s_inputs + rnalm_projected
    │
    └── Protenix.rnalm_projection: LinearNoBias(1280, 384) [zero-init]
        → optional gate
        → s_rnalm [N_token, 384]
        → DiffusionConditioning: concat([s_trunk + s_rnalm, s_inputs])
```

- **Where**: Both InputFeatureEmbedder AND DiffusionConditioning
- **New params**: 574,720 + 491,520 = 1,066,240 (~1M)
- **Intuition**: Maximum information flow - RNA LLM knowledge enters both at the trunk level (to influence representation learning) and at the diffusion level (to directly condition structure generation)

### Comparison Table

| Property | `diffusion` | `input` | `both` |
|----------|-------------|---------|--------|
| Injection point | DiffusionConditioning | InputFeatureEmbedder | Both |
| Like ESM? | No | Yes | Partially |
| New params | 491K | 575K | 1.07M |
| Affects trunk? | No | Yes | Yes |
| Affects diffusion? | Yes | No | Yes |
| Zero-init? | Yes | Yes | Yes |
| 1-stage joint? | Yes | Yes | Yes |

## Code Changes

### 1. `configs/configs_base.py`

Added `injection_mode` field to `rnalm` config:

```python
"rnalm": {
    ...
    "injection_mode": "diffusion",  # "diffusion" | "input" | "both"
}
```

### 2. `protenix/model/modules/embedders.py` — InputFeatureEmbedder

**Constructor changes:**
- Added `rnalm_configs` parameter (default `{}`)
- When `injection_mode in ("input", "both")`:
  - Creates `self.linear_rnalm = LinearNoBias(1280, 449)`
  - Zero-initializes: `nn.init.zeros_(self.linear_rnalm.weight)`
  - Sets `self.rnalm_input_enable = True`

**Forward pass changes:**
- After ESM injection (if any), checks `self.rnalm_input_enable`
- If enabled and `"rnalm_token_embedding"` exists in input_feature_dict:
  ```python
  rnalm_embeddings = self.linear_rnalm(input_feature_dict["rnalm_token_embedding"])
  s_inputs = s_inputs + rnalm_embeddings
  ```

### 3. `protenix/model/protenix.py` — Protenix class

**Constructor changes:**
- Passes `rnalm_configs` to `InputFeatureEmbedder` constructor
- Only creates `rnalm_projection` and gates when `injection_mode in ("diffusion", "both")`
- Only passes `rnalm_configs` to `DiffusionModule` when `injection_mode in ("diffusion", "both")`; otherwise passes `None`

**`_get_s_rnalm` changes:**
- Returns `None` when `injection_mode == "input"` (input-only mode doesn't need diffusion-level s_rnalm)
- Existing behavior preserved for `"diffusion"` and `"both"` modes

### 4. Shell Scripts

| Script | Mode | Description |
|--------|------|-------------|
| `protenix_rna_llm_onestage_joint.sh` | `diffusion` (default) | Existing script, unchanged |
| `protenix_rna_llm_input_inject.sh` | `input` | New: ESM-like input injection |
| `protenix_rna_llm_both_inject.sh` | `both` | New: dual injection at both levels |

All scripts use:
- 1-stage joint training (`--two_stage.enable false`)
- Same data pipeline and hyperparameters
- `--load_strict false` to handle new parameters not in checkpoint

## Design Decisions

### Why Zero-Init?

Zero-initialization is critical for stable training:
1. **At step 0**, the model produces identical outputs to the baseline (without LLM)
2. The LLM signal is gradually learned during training
3. Prevents sudden distribution shift when loading a pre-trained checkpoint
4. This is the same strategy used by ESM injection in the original Protenix

### Why 1-Stage Joint Training?

- Simplest and most stable approach
- All parameters (backbone + new projections) train together from step 0
- Zero-init ensures the model starts from a good baseline
- No stage handoff instability
- Compatible with existing optimizer settings

### Why Independent of Existing Pipelines?

- `injection_mode="diffusion"` (default) preserves all existing behavior
- The `input` and `both` modes are opt-in via config
- No changes to `DiffusionConditioning`, `DiffusionModule`, or `generator.py` for the `input` mode
- Two-stage training scripts continue to work unchanged

## Verification

### Model Instantiation Tests (CPU)

All three modes create models with correct architecture:

| Mode | `input_embedder.rnalm_input_enable` | `has rnalm_projection` |
|------|--------------------------------------|------------------------|
| `diffusion` | `False` | `True` |
| `input` | `True` | `False` |
| `both` | `True` | `True` |

### GPU Tests

- Zero-init projection produces zero output (verified: `max=0.0`)
- Gradient flow works through both projection paths
- Dimension matching verified: `1280 -> 449` (input) and `1280 -> 384` (diffusion)
- Additive fusion at DiffusionConditioning verified: `dim = c_s + c_s_inputs = 833`

## Architecture Diagram

```
Input Sequences
    │
    ▼
[Pre-computed RiNALMo Embeddings: N_token x 1280]
    │
    ▼
Data Pipeline (rnalm_featurizer.py)
    │ feat["rnalm_token_embedding"] = [N_crop, 1280]
    │
    ▼
┌──────────────────────────────────────────────┐
│  InputFeatureEmbedder (Algorithm 2)          │
│                                               │
│  AtomAttentionEncoder → a_token               │
│  + restype, profile, deletion_mean            │
│  = s_inputs [N_token, 449]                    │
│                                               │
│  if injection_mode in ("input", "both"):      │
│    s_inputs += linear_rnalm(rnalm_emb)  ◄─── │ ★ INPUT INJECTION (zero-init)
│                                               │
└──────────────────────────────────────────────┘
    │ s_inputs
    ▼
┌──────────────────────────────────────────────┐
│  Pairformer Trunk                             │
│  (N_cycle recycling → s_trunk, z_trunk)       │
└──────────────────────────────────────────────┘
    │ s_trunk [N_token, 384]
    │ z_trunk [N_token, N_token, 128]
    ▼
┌──────────────────────────────────────────────┐
│  if injection_mode in ("diffusion", "both"):  │
│    s_rnalm = rnalm_projection(rnalm_emb) ◄── │ ★ DIFFUSION INJECTION (zero-init)
│    s_rnalm = gate(s_rnalm)  (optional)        │
│                                               │
│  DiffusionConditioning (Algorithm 21)         │
│    concat([s_trunk + s_rnalm, s_inputs])      │
│    + noise_embedding → single_s               │
│                                               │
│  DiffusionTransformer → coordinates           │
└──────────────────────────────────────────────┘
    │
    ▼
  Predicted Structure
```

## Files Modified

| File | Change |
|------|--------|
| `configs/configs_base.py` | Added `injection_mode` field to rnalm config |
| `protenix/model/modules/embedders.py` | Added `rnalm_configs` param and input injection to `InputFeatureEmbedder` |
| `protenix/model/protenix.py` | Routing logic for `injection_mode`, pass rnalm_configs to embedder |

## Files Created

| File | Purpose |
|------|---------|
| `protenix_rna_llm_input_inject.sh` | Training script for `injection_mode="input"` |
| `protenix_rna_llm_both_inject.sh` | Training script for `injection_mode="both"` |
| `report/rna_llm_injection_design.md` | This design report |

## Files NOT Modified (preserved existing behavior)

| File | Reason |
|------|--------|
| `protenix/model/modules/diffusion.py` | No changes needed; `s_rnalm=None` path works for input-only mode |
| `protenix/model/generator.py` | No changes needed; `s_rnalm=None` handled gracefully |
| `protenix_rna_llm_onestage_joint.sh` | Existing diffusion injection script, unchanged |
| `rna_llm_gate_tune.sh` | Existing 2-stage script, unchanged |
| `protenix/data/rnalm/rnalm_featurizer.py` | Data pipeline unchanged; same embeddings used for all modes |
