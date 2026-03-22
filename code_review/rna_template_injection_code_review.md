# RNA Structural Template Injection — Code Review Report

**Date:** 2026-03-13
**Scope:** Integration of RNA structural templates into the Protenix model
**Status:** Implementation complete, GPU validation passed (9/9 tests)

---

## 1. Executive Summary

This implementation adds RNA structural template support to Protenix, allowing the model to leverage pre-computed RNA template features (distogram, unit vectors, backbone frame masks) alongside existing protein templates. The design uses a **separate RNA projector** (`W_rna`) in the TemplateEmbedder that shares the PairformerStack and output layers with protein templates. Contribution is controlled by a learnable scalar alpha (init=0.01).

**Formula:** `v = W_z·LN(z) + W_prot·a_pp + α·W_rna·a_rr`

The implementation is fully backward-compatible — when `rna_template.enable=False` (default), behavior is identical to the original model.

---

## 2. Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │        TemplateEmbedder              │
                    │                                     │
                    │  ┌─────────┐    ┌─────────────┐     │
                    │  │ Protein │    │    RNA       │     │
                    │  │ Template│    │  Template    │     │
                    │  │ Features│    │  Features    │     │
                    │  └────┬────┘    └──────┬──────┘     │
                    │       │                │             │
                    │  ┌────▼────┐    ┌──────▼──────┐     │
                    │  │ W_prot  │    │   W_rna     │     │  ← separate projectors
                    │  │(linear) │    │  (linear)   │     │
                    │  └────┬────┘    └──────┬──────┘     │
                    │       │                │·α          │  ← learnable gate
                    │       │                │             │
                    │  ┌────▼────────────────▼────┐       │
                    │  │  Shared: W_z·LN(z) +     │       │  ← shared z projection
                    │  │  PairformerStack (2 blks) │       │
                    │  │  layernorm_v              │       │
                    │  └──────────┬────────────────┘       │
                    │             │                         │
                    │        mean pool                     │
                    │     (prot + RNA combined)             │
                    │             │                         │
                    │      linear_u(relu(u))                │
                    └─────────────┼─────────────────────────┘
                                  │
                              += z_init
```

### Key Design Decisions

1. **Separate projector, shared backbone:** RNA templates use `linear_no_bias_a_rna` for the initial feature-to-hidden mapping but share `linear_no_bias_z`, `pairformer_stack`, and `layernorm_v` with protein templates. This reduces parameter count while allowing RNA-specific feature interpretation.

2. **Combined mean pooling:** Protein and RNA template outputs are combined in the denominator: `u = Σv / (num_prot + num_rna)`. This naturally scales contributions when both types are present.

3. **RNA block mask:** A `rna_template_block_mask` tensor of shape `[N_token, N_token]` restricts RNA template features to RNA-RNA token pairs only. Cross-species (RNA-protein) blocks are zero in this v1 implementation.

4. **Alpha gating:** A learnable scalar `rna_template_alpha` (init=0.01) controls the RNA template contribution strength. This starts small to prevent destabilizing the pre-trained model during early fine-tuning.

---

## 3. Files Modified

### 3.1 `configs/configs_base.py`

**Changes:**
- Added `rna_template` config section (lines 136-153)
- Added `linear_no_bias_a_rna,rna_template_alpha,rna_template_gate` to `adapter_keywords` (line 66)

**Config keys:**
| Key | Default | Description |
|-----|---------|-------------|
| `enable` | `False` | Enable RNA template injection |
| `template_database_dir` | `""` | Directory with pre-computed .npz files |
| `template_index_path` | `""` | JSON mapping sequences to template paths |
| `max_rna_templates` | `4` | Max templates per RNA chain |
| `injection_mode` | `"z_init"` | How templates are injected |
| `alpha_init` | `0.01` | Initial value for learnable alpha gate |

### 3.2 `protenix/model/modules/pairformer.py` — TemplateEmbedder

**Changes:**
- **Constructor:** Added `rna_template_configs` parameter. When enabled, creates:
  - `linear_no_bias_a_rna` (same dims as protein projector: input=116, output=c=64)
  - `rna_template_alpha` (learnable scalar, init from config)
- **`forward()`:** Checks for both protein (`template_aatype`) and RNA (`rna_template_aatype`) templates. Processes each through respective projectors, combines via mean pooling.
- **`_single_rna_template_forward()`:** New method. Same structure as `single_template_forward()` but uses `rna_template_*` keys and `linear_no_bias_a_rna`. Applies `rna_block_mask` to restrict features to RNA-RNA pairs.

**New parameters added to model (when enabled):**
| Parameter | Shape | Description |
|-----------|-------|-------------|
| `template_embedder.linear_no_bias_a_rna.weight` | `[64, 116]` | RNA feature projector |
| `template_embedder.rna_template_alpha` | `[1]` | Learnable alpha gate |

**Total new parameters:** ~7,488 (negligible compared to full model)

### 3.3 `protenix/model/protenix.py` — Main Model

**Changes:**
- **Line 130-136:** Pass `rna_template_configs` to TemplateEmbedder constructor. When `rna_template.enable=False`, passes `None` (backward compat).
- **No other changes needed:** The existing `template_` key cleanup (line 699) already handles `rna_template_*` keys. The TemplateEmbedder forward method reads RNA template features directly from `input_feature_dict`.

### 3.4 `protenix/data/pipeline/dataset.py` — Data Pipeline

**Changes:**
- **`__init__`:** Added `self.rna_template_featurizer = kwargs.get("rna_template_featurizer", None)`
- **`process_one()`:** Added RNA template feature loading block after RNA LLM section. Converts numpy arrays to torch tensors and adds to feature dict.
- **`get_rna_template_featurizer()`:** New factory function that creates `RNATemplateFeaturizer` when config enables it.
- **`get_datasets()`:** Added `"rna_template_featurizer"` to dataset kwargs via `_get_dataset_param()`.

### 3.5 `protenix/data/rna_template/__init__.py` (NEW)

Empty init file for the new `rna_template` package.

### 3.6 `protenix/data/rna_template/rna_template_featurizer.py` (NEW, ~425 lines)

**Core class: `RNATemplateFeaturizer`**

Mirrors protein template pipeline (TemplateFeaturizer → Templates → TemplateEmbedder) but loads pre-computed tensors from .npz files.

**Key methods:**
- `__init__()`: Loads JSON index mapping RNA sequences to template .npz paths
- `_find_templates_for_sequence()`: Exact match + T→U fallback
- `get_rna_template_features()`: Main feature assembly:
  1. For each RNA entity, find template .npz files via index
  2. Load pre-computed features (`template_aatype`, `template_distogram`, etc.)
  3. Place chain features at correct token positions using entity/residue IDs
  4. Build `rna_template_block_mask` (1 where both tokens are RNA)
- `__call__()`: Entry point. Identifies RNA entities using RNA-First strategy (same as RiNALMoFeaturizer), then delegates to `get_rna_template_features()`

**RNA entity identification (training mode):**
```python
is_rna = centre_atom_array.chain_mol_type == "rna"
rna_entity_ids = set(centre_atom_array.label_entity_id[is_rna])
# Also reclassify DNA-labeled entities with uracil
```

**Output features:**
| Key | Shape | Description |
|-----|-------|-------------|
| `rna_template_aatype` | `[T, N]` | RNA residue type IDs (21-25) |
| `rna_template_distogram` | `[T, N, N, 39]` | Pairwise distance bins |
| `rna_template_pseudo_beta_mask` | `[T, N, N]` | Valid pair mask |
| `rna_template_unit_vector` | `[T, N, N, 3]` | Inter-residue unit vectors |
| `rna_template_backbone_frame_mask` | `[T, N, N]` | Valid backbone frame mask |
| `rna_template_block_mask` | `[N, N]` | RNA-RNA token pair mask |

### 3.7 `protenix/data/rna_template/build_rna_template_index.py` (NEW, ~285 lines)

Placeholder index builder using pairwise sequence alignment.

**Key functions:**
- `pairwise_identity()`: BioPython global alignment with simple fallback
- `collect_database_templates()`: Scans .npz files, extracts sequences from metadata
- `load_training_sequences()`: Reads from Protenix training data indices
- `build_index_pairwise()`: O(n*m) pairwise comparison with length ratio filter

**Note:** This is a placeholder. Production use should replace with nhmmer/cmscan/BLAST for scalable template search.

---

## 4. Training Scripts

### 4.1 `finetune/finetune_rna_template_1stage.sh`

1-stage training with per-group LR for RNA template + optional LLM embeddings.

Key flags:
- `--use_rna_template true/false` — toggle RNA template injection
- `--use_rnalm true/false` — toggle RNA/DNA LLM embeddings
- `--rna_template_alpha 0.01` — initial alpha gate value
- `--template_n_blocks 2` — PairformerStack blocks in TemplateEmbedder
- `--adapter_lr / --backbone_lr` — per-group learning rates

### 4.2 `finetune/finetune_rna_template_2stage.sh`

2-stage training:
- **Stage 1:** Adapter warmup (backbone frozen, only RNA template projector + LLM projections train)
- **Stage 2:** Joint training with EMA

Both scripts support `--use_rnalm false` for template-only mode (no LLM embeddings).

---

## 5. Validation Results

All 9 tests pass on H800 GPU:

```
  [PASS] Config loads with rna_template section
  [PASS] Adapter keywords include RNA template params
  [PASS] RNATemplateFeaturizer returns correct feature shapes
  [PASS] TemplateEmbedder initializes with RNA template configs
  [PASS] TemplateEmbedder forward pass with RNA templates (GPU)
  [PASS] RNA template parameters receive gradients
  [PASS] Backward compatibility: model works without RNA templates
  [PASS] Combined protein + RNA templates forward pass
  [PASS] get_rna_template_featurizer factory function works
```

**Tests cover:**
- Config loading and defaults
- Data pipeline feature shapes
- Model initialization with/without RNA templates
- Forward pass on GPU (RNA-only, protein-only, combined)
- Gradient flow through RNA-specific parameters
- Backward compatibility (disabled = identical to original)
- Factory function for dataset integration

---

## 6. Backward Compatibility

| Scenario | Behavior |
|----------|----------|
| `rna_template.enable=False` (default) | No RNA template modules created. TemplateEmbedder unchanged. |
| `rna_template.enable=True`, no .npz files | Empty features (all zeros/gaps). Alpha-scaled zero contribution. |
| `rna_template.enable=True`, `n_blocks=0` | TemplateEmbedder returns 0 (same as before). |
| Existing protein-only training | No change in behavior. |
| Loading pre-trained checkpoint with `load_strict=false` | New RNA template params initialized from scratch, existing params loaded. |

---

## 7. Parameter Grouping for Training

The `adapter_keywords` in `two_stage` config include:
```
linear_no_bias_a_rna    # RNA template projector
rna_template_alpha      # RNA template alpha gate
rna_template_gate       # Future gate parameters
```

These are automatically grouped as "adapter" parameters for separate LR in both 1-stage (per-group LR) and 2-stage training.

---

## 8. Known Limitations & Future Work

1. **Template search is placeholder:** `build_rna_template_index.py` uses O(n*m) pairwise alignment. Replace with nhmmer/cmscan for production.

2. **No cross-species template features:** `rna_template_block_mask` zeros out RNA-protein pairs. Future v2 could explore cross-species template information.

3. **injection_mode="diffusion" not yet implemented for RNA templates:** Currently only `z_init` injection is supported (RNA template features flow through TemplateEmbedder → z_init). Diffusion-level injection (like RNA LLM) would require additional projection layers in protenix.py.

4. **No online template computation:** Templates must be pre-computed. Online template search + feature computation at training time is not supported.

5. **Fixed alpha init:** The `rna_template_alpha` is a simple learnable scalar. Consider per-head or per-block alpha for more fine-grained control.

---

## 9. Files Summary

| File | Status | Lines Changed/Added |
|------|--------|-------------------|
| `configs/configs_base.py` | Modified | ~20 lines |
| `protenix/model/modules/pairformer.py` | Modified | ~150 lines |
| `protenix/model/protenix.py` | Modified | ~7 lines |
| `protenix/data/pipeline/dataset.py` | Modified | ~30 lines |
| `protenix/data/rna_template/__init__.py` | New | 0 lines |
| `protenix/data/rna_template/rna_template_featurizer.py` | New | ~425 lines |
| `protenix/data/rna_template/build_rna_template_index.py` | New | ~285 lines |
| `finetune/finetune_rna_template_1stage.sh` | New | ~215 lines |
| `finetune/finetune_rna_template_2stage.sh` | New | ~230 lines |
| `finetune/test_rna_template_integration.py` | New | ~280 lines |
