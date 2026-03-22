# RNA Template Pipeline v5 — Manual Template Override & Fallback

**Date**: 2026-03-16
**Author**: Claude Opus 4.6
**Status**: Implemented & GPU Validated
**Basis**: `rna_template_manual_override_and_fallback_design_20260316.md` + `rna_template_resolver_integration_risks_20260316.md`

---

## 0. Background

v4 established the online RNA template pipeline (MMseqs2 → search_results.json → CIF build). This version adds the ability to **manually specify** CIF/PDB structures or pre-computed NPZ files as RNA templates, with configurable fallback to the existing online search pipeline.

**Design principle**: The manual override is implemented as a thin layer *on top of* the existing pipeline, not as a replacement. When no `templateHints` are provided, the pipeline behaves identically to v4.

---

## 1. What Was Added

### 1.1 New JSON Interface: `templateHints`

An optional `templateHints` field can be added to any `rnaSequence` entity in the inference JSON:

```json
{
  "rnaSequence": {
    "sequence": "GGGAAAUCC",
    "count": 1,
    "templateHints": {
      "mode": "prefer_manual",
      "manual_templates": [
        {
          "type": "structure",
          "path": "/abs/path/to/template.cif",
          "chain_id": "A"
        }
      ]
    }
  }
}
```

**Supported `type` values:**

| Type | Description | Requirements |
|------|-------------|-------------|
| `structure` | CIF/PDB structure file | `.cif`, `.mmcif`, or `.pdb` format. Optional `chain_id` (required for multi-chain files). |
| `npz` | Pre-computed NPZ template | Must match Protenix RNA template feature format (`template_aatype`, `template_distogram`, etc.). |

**Supported `mode` values:**

| Mode | Behavior |
|------|----------|
| `manual_only` | Only use manual templates. No search fallback. If all manual templates fail → empty. |
| `prefer_manual` | Use manual templates if any succeed. If ALL fail → fallback to default search pipeline. |
| `hybrid` | Manual templates fill front slots. Remaining slots filled by search results. |
| `default_only` | Ignore manual hints entirely, use search only. (For A/B testing.) |

Default mode when `templateHints` is present but `mode` is not specified: **`prefer_manual`**.

### 1.2 Slot Organization

Templates are organized at the slot level (not residue level):

```
max_rna_templates = 4 (example)

Slot 0:  manual template 1  (if available)
Slot 1:  manual template 2  (if available)
Slot 2:  search template 1  (if hybrid/fallback)
Slot 3:  search template 2  (if hybrid/fallback)
```

Manual templates always occupy front slots. Search templates fill remaining slots. No mixing within a single slot.

### 1.3 Per-Entity Granularity

Each RNA entity can independently have or not have `templateHints`. In a multi-entity input:

- Entity 1 with `templateHints` → uses manual resolution
- Entity 2 without `templateHints` → uses existing search pipeline (unchanged)

This is entity-level, not copy-level (per risk doc recommendation).

---

## 2. Code Changes

### 2.1 New Standalone Function: `_build_single_template_from_structure()`

**File**: `protenix/data/rna_template/rna_template_featurizer.py`

Builds template features directly from a user-specified CIF/PDB file, bypassing the search-hit-based CIF resolution used by the online pipeline.

```python
def _build_single_template_from_structure(
    query_seq: str,
    structure_path: str,
    chain_id: str = "",
    anchor_mode: str = "base_center_fallback",
) -> Optional[Dict[str, np.ndarray]]:
```

**Key difference from `_build_single_template_online()`**:
- Takes a direct file path instead of a search hit dict
- Does not require `cif_database_dir` or PDB ID resolution
- Supports `.cif`, `.mmcif`, and `.pdb` formats via BioPython
- Auto-detects chain for single-chain structures; requires explicit `chain_id` for multi-chain

### 2.2 New Methods on `RNATemplateFeaturizer`

#### `_build_from_manual_spec(query_seq, spec)`
Dispatches based on `spec["type"]`:
- `"structure"` → calls `_build_single_template_from_structure()`
- `"npz"` → calls `_load_and_crop_rna_template()` and extracts first template slot

Returns a single template dict `[N, ...]` or `None`.

#### `_collect_search_templates(sequence, query_pdb_id, cutoff_date, max_count)`
Extracts individual template dicts from the existing search pipeline (online or offline). Used by `_resolve_with_manual()` for fallback/hybrid slots.

Returns `(list_of_template_dicts, filter_stats)`.

#### `_resolve_with_manual(entity_id, sequence, hints, query_pdb_id, cutoff_date)`
Main manual template resolution logic:

1. Build manual templates from `hints["manual_templates"]`
2. Determine if search fallback is needed based on `hints["mode"]`
3. If needed, collect search templates via `_collect_search_templates()`
4. Merge: manual first, then search, up to `max_templates`
5. Stack into `[T, N, ...]` format

Returns stacked features or `None`.

### 2.3 Modified: `get_rna_template_features()`

**Signature change**:
```python
def get_rna_template_features(
    self,
    rna_sequences, token_entity_ids, token_res_ids, num_tokens,
    query_pdb_id=None, query_release_date=None,
    manual_template_hints=None,   # ← NEW (optional, default None)
) -> Dict[str, np.ndarray]:
```

**Entity loop change** — added manual hints dispatch before existing online/offline branches:

```python
for entity_id, sequence in rna_sequences.items():
    entity_hints = (manual_template_hints or {}).get(str(entity_id))
    if entity_hints and entity_hints.get("manual_templates"):
        chain_features = self._resolve_with_manual(...)  # NEW
    elif self.online_mode:
        # ONLINE PATH (unchanged)
        ...
    else:
        # OFFLINE PATH (unchanged)
        ...
```

When `manual_template_hints` is `None` or empty for an entity, the existing online/offline paths are used **exactly as before**. Zero behavior change for the default case.

### 2.4 Modified: `__call__()`

In inference mode, extracts `templateHints` from `bioassembly_dict["sequences"]`:

```python
manual_template_hints: Dict[str, dict] = {}
if inference_mode:
    for i, entity_info_wrapper in enumerate(bioassembly_dict["sequences"]):
        entity_type = list(entity_info_wrapper.keys())[0]
        if entity_type in ("rnaSequence", "rnaChain"):
            entity_info = entity_info_wrapper[entity_type]
            hints = entity_info.get("templateHints")
            if hints:
                manual_template_hints[entity_id] = hints
```

In training mode, `manual_template_hints` remains empty — training data does not include `templateHints`.

---

## 3. What Was NOT Changed

These paths remain completely unmodified:

| Component | Status |
|-----------|--------|
| Online search (MMseqs2 → CIF build) | Unchanged |
| Offline NPZ loading | Unchanged |
| Temporal filtering / self-hit exclusion | Unchanged |
| RNA entity identification (RNA-First) | Unchanged |
| Token mapping / feature placement | Unchanged |
| RNA/DNA LLM embedding pipeline | Unchanged |
| MSA pipeline | Unchanged |
| TemplateEmbedder model code | Unchanged |
| Projector initialization (protein/zero) | Unchanged |
| 1-stage / 2-stage training scripts | Unchanged |
| Inference script (`infer_rna.sh`) | Unchanged |
| Configs (`configs_base.py`) | Unchanged |

---

## 4. Comparison: Before vs After

| Scenario | v4 (Before) | v5 (After) |
|----------|-------------|------------|
| No `templateHints` in JSON | Online search pipeline | Online search pipeline (identical) |
| `templateHints` with `mode=manual_only` | N/A | Manual templates only; no search |
| `templateHints` with `mode=prefer_manual` | N/A | Manual if any succeed; else search fallback |
| `templateHints` with `mode=hybrid` | N/A | Manual front slots + search back slots |
| `templateHints` with `mode=default_only` | N/A | Manual ignored, search only |
| Manual CIF/PDB structure | N/A | Built via `_build_single_template_from_structure()` |
| Manual NPZ file | N/A | Loaded via `_load_and_crop_rna_template()` |
| Training mode | Online search | Online search (no change, hints not extracted) |
| RNA/DNA LLM embeddings | Work normally | Work normally (no interference) |
| MSA processing | Work normally | Work normally (no interference) |

---

## 5. Usage Examples

### 5.1 User has a specific CIF structure to use as template

```json
{
  "sequences": [
    {
      "rnaSequence": {
        "sequence": "GGGAAAUCC",
        "count": 1,
        "templateHints": {
          "mode": "prefer_manual",
          "manual_templates": [
            {
              "type": "structure",
              "path": "/data/my_rna_model/predicted.cif",
              "chain_id": "A"
            }
          ]
        }
      }
    }
  ]
}
```

Behavior: Builds template from the CIF file. If it fails (wrong chain, bad file), falls back to online search.

### 5.2 User has a pre-computed NPZ, no fallback

```json
{
  "rnaSequence": {
    "sequence": "AUGCUAGCUA",
    "count": 1,
    "templateHints": {
      "mode": "manual_only",
      "manual_templates": [
        {
          "type": "npz",
          "path": "/data/templates/my_custom_template.npz"
        }
      ]
    }
  }
}
```

### 5.3 Hybrid: manual + search

```json
{
  "rnaSequence": {
    "sequence": "GGGAAAUCC",
    "count": 1,
    "templateHints": {
      "mode": "hybrid",
      "manual_templates": [
        {
          "type": "structure",
          "path": "/data/rnajp_output/model_1.cif",
          "chain_id": "A"
        }
      ]
    }
  }
}
```

Behavior: Manual CIF fills slot 0. Slots 1-3 filled by MMseqs2 search results.

### 5.4 No template hints (backward compatible)

```json
{
  "rnaSequence": {
    "sequence": "GGGAAAUCC",
    "count": 1
  }
}
```

Behavior: Identical to v4 — uses online search pipeline.

---

## 6. Testing

### 6.1 Unit Tests (11/11 passed)

```
  [✅ PASS] imports
  [✅ PASS] backward_compat_sig
  [✅ PASS] no_hints_passthrough
  [✅ PASS] hints_extraction
  [✅ PASS] build_from_structure (real CIF: 100d.cif chain A)
  [✅ PASS] build_from_manual_npz
  [✅ PASS] build_from_manual_missing
  [✅ PASS] resolve_manual_only
  [✅ PASS] resolve_manual_only_fail
  [✅ PASS] resolve_default_only
  [✅ PASS] prefer_manual_fallback
```

### 6.2 End-to-End Integration Tests (7/7 passed)

```
  [✅ PASS] manual_npz_get_features — diag_val=0.990
  [✅ PASS] manual_npz_block_mask — block_mask sum=64
  [✅ PASS] no_hints_backward_compat — empty templates as expected
  [✅ PASS] manual_structure_real_cif — mask[0] sum=100.0
  [✅ PASS] manual_structure_block_mask
  [✅ PASS] prefer_manual_fallback — correct fallback: manual+search both empty
  [✅ PASS] multi_entity_partial — entity1_sum=7.9, entity2_sum=0.0
```

### 6.3 GPU Training Validation (1-stage, H800)

```
GPU:             NVIDIA H800
Mode:            ONLINE (search_results.json + PDB_RNA CIF)
RNA LLM:         AIDO RNA (2048d) + DNA (1024d) enabled
RNA Template:    Online mode, projector_init=protein, alpha=0.01
Steps:           5
Result:          Checkpoints saved (4.pt, 4_ema_0.999.pt)
Errors:          None

Config verification:
  rna_template.enable: true
  rna_template.search_results_path: .../search_results.json    ✅
  rna_template.cif_database_dir: .../PDB_RNA                   ✅
  rnalm.enable: true                                           ✅
  rnalm.injection_mode: diffusion                              ✅
```

---

## 7. Risk Mitigations (from integration risk review)

| Risk | Mitigation |
|------|-----------|
| 1.1: Entity vs copy level | v5 only supports entity-level manual override. No copy-level support. |
| 1.2: Multi-manual template stacking | v5 supports multiple manual templates per entity (one per slot). |
| 1.3: Manual structure ≠ search hit | `_build_single_template_from_structure()` is a separate builder contract; does not fake search hits. |
| 3.1: JSON schema drift | `templateHints` is purely optional; existing JSON without it works identically. |
| 3.2: Resolver too strong | Implementation is narrow: entity-level, slot-level merge only, no adapter/external_job support in v5. |

---

## 8. Modified File List

| File | Operation | Lines Changed |
|------|-----------|---------------|
| `protenix/data/rna_template/rna_template_featurizer.py` | Modified | +~250 lines (new functions/methods + dispatch logic) |
| `finetune/test_manual_template_override.py` | New | Unit test suite for manual override |
| `finetune/test_manual_template_e2e.py` | New | E2E integration test suite |

---

## 9. Architecture Diagram

```
Input JSON (inference)
  └── rnaSequence
        ├── sequence: "GGGAAAUCC"
        ├── templateHints (optional)
        │     ├── mode: "prefer_manual" | "manual_only" | "hybrid" | "default_only"
        │     └── manual_templates:
        │           ├── {type: "structure", path: "/...", chain_id: "A"}
        │           └── {type: "npz", path: "/..."}
        └── (other fields: count, unpairedMsaPath, etc.)

RNATemplateFeaturizer.__call__()
  ├── Extract templateHints (inference mode only)
  ├── Identify RNA entities (RNA-First)
  └── get_rna_template_features()
        └── Per entity:
              ├── IF templateHints present:
              │     └── _resolve_with_manual()
              │           ├── Build manual templates
              │           │     ├── _build_from_manual_spec() → "structure" or "npz"
              │           │     │     ├── _build_single_template_from_structure() [NEW]
              │           │     │     └── _load_and_crop_rna_template() [existing]
              │           │     └── Fill front slots
              │           ├── Fallback search (if mode allows)
              │           │     └── _collect_search_templates() [NEW]
              │           │           ├── Online: _find_hits → _filter → _build_single_template_online
              │           │           └── Offline: _find_templates → _filter → _load_and_crop
              │           └── Merge: manual slots + search slots → stack [T, N, ...]
              │
              ├── ELIF online_mode:
              │     └── (existing v4 online path, unchanged)
              │
              └── ELSE (offline):
                    └── (existing legacy path, unchanged)
```

---

## 10. Future Extensions (Not in v5)

| Feature | Status | Notes |
|---------|--------|-------|
| `external_job` type (RNAJP adapter) | Planned (Phase 2) | Requires adapter module |
| Copy-level override | Deferred | Entity-level only in v5 |
| Config: `allow_manual_override` | Not needed | Presence of `templateHints` in JSON is sufficient |
| Residue-level hybrid merge | Deferred | Slot-level merge is simpler and sufficient |
| `nhmmer`/`cmsearch` as search provider | Planned (Phase 3) | Requires search pipeline extension |
