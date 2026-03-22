# RNA Template Pipeline v6 — Training Manual Template Support

Date: 2026-03-16
Author: Claude Opus 4.6

## 0. Background

v5 implemented manual template override for **inference only**: users could attach
`templateHints` to `rnaSequence` entities in the input JSON, choosing between
`manual_only`, `prefer_manual`, `hybrid`, and `default_only` modes. However, the
v5 review ([`rna3d_version5_check.md`](rna3d_version5_check.md)) identified three
findings:

| # | Severity | Issue |
|---|----------|-------|
| 1 | **High** | Manual override not wired for training/finetune — `templateHints` extracted only in `inference_mode=True` |
| 4 | Medium | Public JSON docs not updated for `templateHints` (schema drift) |
| 5 | Medium | DNA-with-U reclassified entities cannot carry manual hints in inference mode |

v6 addresses **Finding 1** (the high-priority training gap) and **Finding 5**
(the DNA-with-U boundary inconsistency).

---

## 1. What Was Added (v6)

### 1.1 Training-time manual template hints

A new config parameter `rna_template.manual_template_hints_path` accepts a JSON
file that maps training PDB IDs to per-entity template hints. This enables the
same `manual_only / prefer_manual / hybrid / default_only` modes that were
previously inference-only to work during training and finetuning.

**JSON format:**

```json
{
  "pdb_id": {
    "entity_id": {
      "mode": "hybrid",
      "manual_templates": [
        {
          "type": "structure",
          "path": "/path/to/template.cif",
          "chain_id": "A",
          "label": "my_template"
        }
      ]
    }
  }
}
```

Special entity key `"*"` acts as a wildcard — matches any RNA entity in that PDB.
Specific entity IDs take priority over the wildcard.

**Dispatch flow during training:**

```
bioassembly_dict["pdb_id"]
      │
      ▼
 _training_manual_hints[pdb_id]?
      │
      ├─ Yes → per-entity lookup (entity_id or "*" wildcard)
      │         │
      │         ├─ hints found → _resolve_with_manual()
      │         │                 (with temporal filtering on search fallback)
      │         │
      │         └─ no hints → existing online/offline pipeline
      │
      └─ No  → existing online/offline pipeline (unchanged)
```

### 1.2 DNA-with-U reclassification now carries manual hints (inference)

In v5, `dnaSequence`/`dnaChain` entities containing uracil were reclassified
as RNA for template search, but `templateHints` were only extracted from
`rnaSequence`/`rnaChain` entities. v6 also extracts `templateHints` from
reclassified DNA-with-U entities in inference mode, closing the boundary gap.

### 1.3 Backward compatibility guarantee

When `manual_template_hints_path` is empty (default), the entire manual hints
subsystem is dormant:
- `self._training_manual_hints` is an empty dict
- The conditional `if self._training_manual_hints and query_pdb_id:` short-circuits
- Zero overhead, zero behavior change

---

## 2. Code Changes — Detailed

### 2.1 `configs/configs_base.py`

Added `manual_template_hints_path` to the `rna_template` config block:

```python
# Before (v5):
"rna_template": {
    ...
    "search_results_path": "",
    "cif_database_dir": "",
},

# After (v6):
"rna_template": {
    ...
    "search_results_path": "",
    "cif_database_dir": "",
    # === Manual template hints (v6) ===
    "manual_template_hints_path": "",
},
```

### 2.2 `protenix/data/rna_template/rna_template_featurizer.py`

**Change 1: Constructor — new parameter + JSON loading**

```python
# __init__ signature: added manual_template_hints_path=""
def __init__(
    self,
    template_database_dir: str = "",
    template_index_path: str = "",
    max_templates: int = 4,
    rna3db_metadata_path: str = "",
    search_results_path: str = "",
    cif_database_dir: str = "",
    manual_template_hints_path: str = "",  # ← NEW (v6)
):
    ...
    # ── Load training-time manual template hints (v6) ──
    self._training_manual_hints: Dict[str, Dict[str, dict]] = {}
    if manual_template_hints_path:
        if os.path.exists(manual_template_hints_path):
            with open(manual_template_hints_path, "r") as f:
                self._training_manual_hints = json.load(f)
            logger.info(
                f"Training manual template hints loaded: "
                f"{len(self._training_manual_hints)} PDB entries "
                f"from {manual_template_hints_path}"
            )
        else:
            logger.warning(
                f"manual_template_hints_path='{manual_template_hints_path}' "
                f"does not exist — training manual hints disabled."
            )
```

**Change 2: Training branch in `__call__` — per-PDB hint lookup**

After RNA entity identification and RNA-First reclassification, the training branch
now looks up the current sample's `query_pdb_id` in the loaded hints dict:

```python
# ── Training manual template hints (v6) ──
if self._training_manual_hints and query_pdb_id:
    pdb_hints = self._training_manual_hints.get(
        query_pdb_id,
        self._training_manual_hints.get(query_pdb_id.upper(), {}),
    )
    if pdb_hints:
        for entity_id in rna_sequences:
            entity_hints = pdb_hints.get(
                str(entity_id),
                pdb_hints.get("*", None),  # "*" = wildcard
            )
            if entity_hints:
                manual_template_hints[str(entity_id)] = entity_hints
                logger.info(
                    f"Training manual hints for pdb={query_pdb_id} "
                    f"entity={entity_id}: mode=..."
                )
```

The `manual_template_hints` dict is then passed to `get_rna_template_features()`,
which routes entities with hints through `_resolve_with_manual()` — exactly the
same resolver used in inference mode. Data-leakage prevention (temporal filtering,
self-hit exclusion) applies to the **search fallback** portion within
`_resolve_with_manual()`.

**Change 3: Inference DNA-with-U branch — extract templateHints**

```python
# In the inference dnaSequence/dnaChain reclassification block:
elif entity_type in ("dnaSequence", "dnaChain"):
    entity_info = entity_info_wrapper[entity_type]
    seq = entity_info["sequence"]
    if "U" in seq or "u" in seq:
        rna_sequences[entity_id] = seq
        # NEW (v6): Also extract templateHints for reclassified entities
        hints = entity_info.get("templateHints")
        if hints:
            manual_template_hints[entity_id] = hints
```

### 2.3 `protenix/data/pipeline/dataset.py`

Updated `get_rna_template_featurizer()` factory to read and pass the new config:

```python
manual_template_hints_path = rna_template_info.get("manual_template_hints_path", "")
...
return RNATemplateFeaturizer(
    ...,
    manual_template_hints_path=manual_template_hints_path,
)
```

### 2.4 `finetune/finetune_rna_template_1stage.sh` and `2stage.sh`

Added `--manual_template_hints` CLI argument (optional, default empty):

```bash
MANUAL_TEMPLATE_HINTS=""  # (v6) JSON file mapping PDB IDs to manual template hints

# In arg parser:
--manual_template_hints) MANUAL_TEMPLATE_HINTS="$2"; shift 2 ;;

# In RNA_TEMPLATE_ARGS builder:
if [ -n "${MANUAL_TEMPLATE_HINTS}" ]; then
    [ -f "${MANUAL_TEMPLATE_HINTS}" ] || { echo "ERROR: ..."; exit 1; }
    RNA_TEMPLATE_ARGS="${RNA_TEMPLATE_ARGS} --rna_template.manual_template_hints_path ${MANUAL_TEMPLATE_HINTS}"
fi
```

Usage:
```bash
# Without manual hints (default — identical to v5 behavior):
bash finetune/finetune_rna_template_1stage.sh

# With manual hints:
bash finetune/finetune_rna_template_1stage.sh --manual_template_hints my_hints.json
```

---

## 3. What Was NOT Changed

| Component | Status |
|-----------|--------|
| Online search pipeline (MMseqs2 → CIF → features) | Unchanged |
| Offline NPZ loading (legacy) | Unchanged |
| Temporal filtering / self-hit exclusion | Unchanged (applied to search fallback in manual resolver) |
| RNA-First entity identification | Unchanged |
| Token mapping / feature assembly | Unchanged |
| RNA/DNA LLM embedding (RNALM / AIDO) | Unchanged |
| MSA processing | Unchanged |
| TemplateEmbedder model code | Unchanged |
| Projector init (protein copy + alpha gate) | Unchanged |
| 1-stage / 2-stage training logic | Unchanged |
| Inference `templateHints` from JSON | Unchanged (still works as in v5) |

---

## 4. Fallback Logic Summary

| Scenario | Behavior |
|----------|----------|
| No `manual_template_hints_path` (default) | Online/offline search — identical to v4/v5 |
| `manual_template_hints_path` set, PDB not in file | Online/offline search — no manual hints |
| PDB in file, entity ID matches or `"*"` wildcard | `_resolve_with_manual()` with configured mode |
| `mode=hybrid` | Manual templates fill front slots, search fills remaining |
| `mode=prefer_manual` | Manual if any succeed, else full search fallback |
| `mode=manual_only` | Only manual templates, no search |
| `mode=default_only` | Ignore manual specs, use search only |
| Manual CIF file missing or corrupt | Warning logged, falls through to search (in hybrid/prefer_manual) |
| Inference: no `templateHints` in JSON | Online/offline search — identical to v4 |
| Inference: `dnaSequence` with uracil + `templateHints` | Reclassified as RNA + manual hints applied (v6 fix) |

---

## 5. Testing Results

### 5.1 Unit Tests — Manual Hints Logic (6/6 passed)

```
PASS: test_load_manual_hints
PASS: test_empty_path_no_hints
PASS: test_missing_file_warning
PASS: test_wildcard_entity_matching
PASS: test_specific_entity_overrides_wildcard
PASS: test_pdb_not_in_hints_returns_none
```

### 5.2 GPU Training — 1-Stage Without Manual Hints (fallback to online)

- H800 GPU, 5 training steps
- `manual_template_hints_path: ''` in config output
- Checkpoint produced: `checkpoints/4.pt`, `4_ema_0.999.pt`
- RNA template online mode active, RNALM + MSA enabled
- **PASS**: No regression from v5

### 5.3 GPU Training — 2-Stage Without Manual Hints

- H800 GPU, stage1=3 steps (adapter warmup, backbone frozen), stage2=3 steps (joint)
- Checkpoints: `3.pt` (end stage1), `5.pt` + `5_ema_0.999.pt` (end stage2)
- **PASS**: 2-stage pipeline works with new config parameter (empty)

### 5.4 GPU Training — 1-Stage With Manual Hints

- H800 GPU, 5 training steps
- `manual_template_hints_path: finetune/test_manual_hints.json`
- Test JSON maps 2 PDB IDs (`157d`, `1a1t`) to manual CIF templates
- Checkpoint produced: `checkpoints/4.pt`, `4_ema_0.999.pt`
- **PASS**: Training with manual hints completes without errors

### 5.5 Pipeline Integrity Verification

Checked config.yaml from all test runs — confirmed:
- `rnalm.enable: true`, `use_rna_embed: true`, `use_dna_embed: true`
- `data.msa.enable_rna_msa: true`
- `rna_template.enable: true`, online mode (search_results_path + cif_database_dir)
- RNA/DNA LLM, MSA, and RNA template pipelines all active simultaneously

---

## 6. Modified Files

| File | Change |
|------|--------|
| `configs/configs_base.py` | Added `manual_template_hints_path` config key |
| `protenix/data/rna_template/rna_template_featurizer.py` | Constructor: load hints JSON; `__call__` training branch: PDB→entity lookup; inference DNA-with-U: extract hints |
| `protenix/data/pipeline/dataset.py` | Pass `manual_template_hints_path` to featurizer factory |
| `finetune/finetune_rna_template_1stage.sh` | Added `--manual_template_hints` CLI arg |
| `finetune/finetune_rna_template_2stage.sh` | Added `--manual_template_hints` CLI arg |
| `finetune/test_v6_training_manual_hints.py` | **New**: Unit tests for training manual hints |
| `finetune/test_manual_hints.json` | **New**: Sample manual hints JSON for testing |

---

## 7. Design Rationale

### Why an external JSON file (not inline in training data)?

Training samples come from PDB files via `bioassembly_dict`, which has a fixed
schema (`{"sequences": {"entity_id": "seq", ...}, "pdb_id": "xxxx", ...}`).
Embedding manual hints into this dict would require modifying the data preparation
pipeline. An external JSON lookup is:
- Non-invasive: zero changes to data prep, parser, or bioassembly format
- Composable: can be generated by any script/tool and passed as a config arg
- Selective: only PDBs in the file get manual treatment; everything else is unchanged

### Why wildcard `"*"` entity matching?

Many RNA PDBs have a single RNA entity. Requiring exact entity IDs would mean the
user needs to know internal entity numbering. `"*"` provides a convenient shorthand
for "apply this template to all RNA entities in this PDB."

### Why data-leakage prevention still applies to search fallback?

In `hybrid` and `prefer_manual` modes, the resolver calls `_collect_search_templates()`
for remaining slots, which passes through `_filter_hits_online()` with temporal
filtering and self-hit exclusion. Manual templates are user-specified and intentional,
so they bypass temporal checks — but any search-based fills are still governed.

---

## 8. Architecture Diagram (v6)

```
                         ┌──────────────────────────────────────────────┐
                         │  RNATemplateFeaturizer.__call__()            │
                         │                                              │
                         │  ┌──────────┐     ┌──────────────────────┐  │
                         │  │ Inference │     │ Training             │  │
                         │  │ Mode     │     │ Mode                 │  │
                         │  └────┬─────┘     └────┬─────────────────┘  │
                         │       │                 │                    │
                         │  JSON entity-level      │  chain_mol_type   │
                         │  templateHints           │  + RNA-First      │
                         │  (rna + dna-with-U)     │                    │
                         │       │                 │  ┌───────────────┐ │
                         │       │                 ├──┤ _training_    │ │
                         │       │                 │  │ manual_hints  │ │
                         │       │                 │  │ [pdb_id]      │ │
                         │       │                 │  │ [entity_id|*] │ │
                         │       │                 │  └───────┬───────┘ │
                         │       ▼                 ▼          │ (v6)    │
                         │  ┌─────────────────────────────────▼──────┐  │
                         │  │ manual_template_hints populated?       │  │
                         │  └────┬──────────────────────────┬────────┘  │
                         │  Yes  │                          │  No       │
                         │  ┌────▼──────────────┐  ┌───────▼────────┐  │
                         │  │_resolve_with_manual│  │existing search │  │
                         │  │ manual→search fill │  │(online/offline)│  │
                         │  │ temporal filter on │  │temporal filter │  │
                         │  │ search portion     │  │self-hit check  │  │
                         │  └────────────────────┘  └────────────────┘  │
                         │                                              │
                         │  get_rna_template_features()                 │
                         │  → token-level placement → return            │
                         └──────────────────────────────────────────────┘
```

---

## 9. v5 → v6 Comparison

| Aspect | v5 | v6 |
|--------|----|----|
| Manual templates in inference | Yes | Yes (unchanged) |
| Manual templates in training | **No** | **Yes** — via `manual_template_hints_path` |
| DNA-with-U manual hints (inference) | **No** | **Yes** — now extracted |
| Config parameter count | 11 | 12 (+`manual_template_hints_path`) |
| Finetune script args | No manual arg | `--manual_template_hints` (optional) |
| Backward compatible | — | Yes (empty path = no behavior change) |
| Data-leakage prevention for search | Yes | Yes (unchanged) |
| Projector init / alpha gate / 1-stage / 2-stage | Yes | Yes (unchanged) |

---

## 10. Usage Examples

### Training with manual templates for specific PDBs

```bash
# Create manual hints JSON
cat > my_hints.json << 'EOF'
{
  "7xyz": {
    "*": {
      "mode": "hybrid",
      "manual_templates": [
        {"type": "structure", "path": "/data/my_known_good_template.cif", "chain_id": "A"}
      ]
    }
  }
}
EOF

# Run finetune with manual hints
bash finetune/finetune_rna_template_1stage.sh --manual_template_hints my_hints.json
```

### Training without manual templates (default — no change from v5)

```bash
bash finetune/finetune_rna_template_1stage.sh
# manual_template_hints_path defaults to "", all entities use online search
```

### Inference with templateHints in JSON (unchanged from v5)

```json
{
  "sequences": [
    {
      "rnaSequence": {
        "sequence": "AGCUAGCU...",
        "templateHints": {
          "mode": "hybrid",
          "manual_templates": [
            {"type": "structure", "path": "/path/to/template.cif", "chain_id": "A"}
          ]
        }
      }
    }
  ]
}
```
