# RNA Template Temporal Filtering & Self-Hit Exclusion — Implementation Report

**Date**: 2026-03-15
**Status**: Implemented and GPU-verified

---

## 1. Problem Statement

The RNA template pipeline had a critical **data leakage** problem:

| Issue | Protein Pipeline | RNA Pipeline (Before Fix) |
|-------|-----------------|--------------------------|
| **Per-query temporal filtering** | `query_release_date - 60 days` cutoff per sample | Only global `--release_date_cutoff` at search time; **none at runtime** |
| **Self-hit exclusion** | Rejects templates from same PDB online | Only during offline MMseqs2 search; **none at runtime** |
| **Query identity tracking** | Per-query matching via HHR/A3M | Sequence-keyed index loses query provenance |

### Root Cause

The `RNATemplateFeaturizer.__call__()` method loaded templates based solely on sequence matching, without checking:
1. Whether the template PDB was the **same structure** as the training query (self-hit)
2. Whether the template was **released after** the query's release date (temporal leakage)

This meant during training, a query PDB `X` could use its own 3D structure (or a structure released years later) as a "template", giving the model access to the answer it was supposed to predict.

---

## 2. Solution Architecture

We mirror the protein pipeline's online filtering approach (from `template_featurizer.py` and `template_utils.py`), applying it per-query at runtime in the RNA featurizer:

```
Training sample (bioassembly_dict)
  ├── pdb_id: "7abc"
  └── release_date: "2021-05-15"
           │
           ▼
  RNATemplateFeaturizer._filter_candidates()
           │
           ├── Self-hit exclusion:
           │     template PDB == query PDB → REJECT
           │
           └── Temporal filtering:
                 cutoff = query_release_date - DAYS_BEFORE_QUERY_DATE (60)
                 template_release_date > cutoff → REJECT
                 template_release_date ≤ cutoff → KEEP
```

### Key Design Decisions

1. **Reuse protein constant**: `DAYS_BEFORE_QUERY_DATE = 60` from `template_utils.py` — single source of truth for the safety margin.

2. **PDB ID extracted from NPZ filename**: Template files are named `{pdb_id}_{chain}_{chain}_template.npz`, so `_extract_pdb_from_npz_path()` takes the first underscore-delimited segment.

3. **Release dates from RNA3DB metadata**: Loaded once from `filter.json` (15,441 entries → 5,389 unique PDBs). The mapping is `base_pdb → earliest_release_date` across all chains.

4. **Training-only filtering**: During inference (`inference_mode=True`), no filtering is applied — mirrors protein pipeline behavior.

5. **Graceful degradation**: If `rna3db_metadata_path` is empty, temporal filtering is silently disabled but self-hit exclusion still works.

6. **No protein/LLM interference**: Only touches files with `rna_template` prefix. The protein template pipeline, LLM embedding pipeline, and all other data paths are completely untouched.

---

## 3. Files Modified

| File | Change |
|------|--------|
| `protenix/data/rna_template/rna_template_featurizer.py` | **Core fix**: Added `_load_rna_release_dates()`, `_extract_pdb_from_npz_path()`, `_filter_candidates()`. Modified `__init__`, `get_rna_template_features`, `__call__` to accept and use metadata. |
| `protenix/data/pipeline/dataset.py` | `get_rna_template_featurizer()` now reads and passes `rna3db_metadata_path` config. |
| `protenix/data/inference/infer_dataloader.py` | Passes `rna3db_metadata_path` to `RNATemplateFeaturizer` constructor. |
| `configs/configs_base.py` | Added `"rna3db_metadata_path": ""` to `rna_template` config block. |
| `finetune/finetune_rna_template_1stage.sh` | Added `RNA3DB_METADATA_PATH` variable and `--rna_template.rna3db_metadata_path`. |
| `finetune/finetune_rna_template_2stage.sh` | Same metadata path addition. |
| `finetune/finetune_rna_template_validate.sh` | Same metadata path addition (was already partially done). |

---

## 4. New Code: Key Functions

### `_load_rna_release_dates(metadata_path)` → `Dict[str, datetime]`
- Loads RNA3DB `filter.json`
- Builds `base_pdb → earliest_release_date` mapping
- Cached per-process (avoids re-loading on each sample)
- Example: `{"7zpi": datetime(2022, 11, 16), ...}` — 5,389 PDB entries

### `_extract_pdb_from_npz_path(npz_path)` → `str`
- `"templates/1asy_R_R_template.npz"` → `"1asy"`
- Fast, no I/O — pure string parsing

### `_filter_candidates(candidate_paths, query_pdb_id, cutoff_date)` → `(filtered, stats)`
- Iterates candidate NPZ paths
- Rejects self-hits (template PDB == query PDB)
- Rejects future templates (template release > cutoff)
- Returns filtered list and stats dict `{self_hit, future, no_date}`

---

## 5. Verification

### Unit Tests (all passed)
```
[PASS] _extract_pdb_from_npz_path — correct PDB extraction
[PASS] _load_rna_release_dates — 5,389 PDBs loaded
[PASS] DAYS_BEFORE_QUERY_DATE = 60 (imported from template_utils)
[PASS] Self-hit exclusion: 3 → 2 candidates (1 self-hit removed)
[PASS] Temporal filtering: 3 → N candidates (depending on cutoff)
[PASS] No filtering during inference (both args None) — all candidates kept
```

### GPU Training Test (H800, 5 steps)
```
Training completed successfully.
Filtering actions observed:
  - pdb=9iwf: self_hit=2, future=0   (self-template blocked)
  - pdb=9jgm: self_hit=4, future=0   (self-template blocked)
  - pdb=9kgg: self_hit=1, future=0   (self-template blocked)
  - pdb=9ljn: self_hit=1, future=10  (self-hit + temporal filtering active!)
No CUDA errors, no crashes.
```

### Temporal Filtering Coverage (full index)

| Cutoff Date | Templates Removed | % Removed | Templates Kept |
|------------|-------------------|-----------|---------------|
| 2015-01-01 | 10,883 | 82.9% | 2,245 |
| 2018-01-01 | 9,215 | 70.2% | 3,913 |
| 2021-09-30 | 6,324 | 48.2% | 6,804 |
| 2024-01-01 | 3,801 | 29.0% | 9,327 |

Total template candidates in index: 13,128
RNA3DB PDBs with release dates: 5,389
PDBs with no date in metadata (kept by default): 0

---

## 6. How to Adjust RNA Template Search Sensitivity

### Control 1: `DAYS_BEFORE_QUERY_DATE` (temporal margin)
- **Current**: 60 days (shared with protein pipeline)
- **Location**: `protenix/data/template/template_utils.py:72`
- **Effect**: Increasing the margin is more conservative (fewer templates), decreasing allows more recent templates
- **Recommendation**: Keep at 60 days to match protein pipeline unless you have a specific reason to diverge

### Control 2: `rna3db_metadata_path` (enable/disable temporal filtering)
- **Set to `""`**: Temporal filtering disabled, only self-hit exclusion active
- **Set to filter.json path**: Full temporal filtering + self-hit exclusion
- **Config**: `--rna_template.rna3db_metadata_path /path/to/filter.json`

### Control 3: `max_rna_templates` (per-chain cap)
- **Current**: 4 (matches protein pipeline)
- **Config**: `--rna_template.max_rna_templates N`
- **Effect**: After filtering, at most N candidates are loaded per RNA chain

### Control 4: MMseqs2 search sensitivity (offline, in `03_mmseqs2_search.py`)
- **`--min_identity`**: Minimum sequence identity (default 0.3 = 30%)
  - Lower → more templates but lower quality
  - Higher → fewer but more reliable templates
- **`--sensitivity`**: MMseqs2 sensitivity parameter (default 7.5)
  - Range 1-8, higher is more sensitive but slower
- **`--evalue`**: E-value threshold (default 1e-3)
  - Lower → more stringent, fewer hits
- **`--max_templates`**: Max templates per query during search (default 4)

### Control 5: `release_date_cutoff` (global offline cutoff in `03_mmseqs2_search.py`)
- **Usage**: `--release_date_cutoff 2021-09-30` during catalog creation
- **Effect**: Removes all catalog entries released after this date BEFORE search
- **Relation to runtime filtering**: This is a coarse pre-filter; the runtime per-query filter provides finer-grained control

### Sensitivity Tuning Recipe

For **maximum template coverage** (production training):
```bash
--rna_template.rna3db_metadata_path /path/to/filter.json  # Temporal active
# In 03_mmseqs2_search.py:
--min_identity 0.2 --sensitivity 7.5 --evalue 0.01 --max_templates 8
```

For **strict data leakage prevention** (benchmarking / Kaggle):
```bash
--rna_template.rna3db_metadata_path /path/to/filter.json  # Temporal active
# DAYS_BEFORE_QUERY_DATE = 60 (default)
# In 03_mmseqs2_search.py:
--min_identity 0.3 --sensitivity 7.5 --evalue 1e-3 --max_templates 4
```

For **inference** (no filtering needed):
```bash
# rna3db_metadata_path can be set or empty — inference_mode=True skips all filtering
```

---

## 7. Architectural Comparison: Before vs After

```
BEFORE (leaky):
  sequence_index["AGCU..."] → [npz1, npz2, npz3]
  Load first valid NPZ → template features
  ❌ No date check, no self-hit check

AFTER (hardened):
  sequence_index["AGCU..."] → [npz1, npz2, npz3]
       │
       ▼
  _filter_candidates(paths,
    query_pdb="7abc",
    cutoff=query_date - 60 days)
       │
       ├── npz1: template_pdb="7abc" → REJECT (self-hit)
       ├── npz2: template_date > cutoff → REJECT (future)
       └── npz3: template_date ≤ cutoff, different PDB → KEEP ✓
       │
       ▼
  Load first valid NPZ from filtered list → template features
```

---

## 8. What Is NOT Changed

- Protein template pipeline (`template_featurizer.py`, `template_utils.py`) — untouched
- RNA/DNA LLM embedding pipeline (`rnalm_featurizer.py`) — untouched
- MSA pipeline — untouched
- Model architecture — untouched
- Offline search pipeline (`03_mmseqs2_search.py`) — untouched (its date filtering is complementary)
- NPZ file format — untouched
- Template index format — untouched
