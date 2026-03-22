# RNA 3D Template Pipeline: Implementation Report

## 1. Overview

This report documents the implementation of a complete RNA 3D template pipeline for Protenix fine-tuning, using the `rna3db-mmcifs` database as the template source. The pipeline integrates:

- **Arena** for filling missing atoms in RNA structures
- **MMseqs2** for scalable template sequence search
- **Protenix** template featurizer for training integration

### Pipeline Architecture

```
rna3db-mmcifs (15,441 CIF files)
    │
    ▼
[01] Extract RNA Catalog ──────────── rna_catalog.json (13,128 structures)
    │
    ▼
[02] Arena Atom Refinement ─────────── Full-atom PDB (missing atoms filled)
    │
    ▼
[03] Build Template .npz ───────────── templates/*.npz (Protenix format)
    │
    ▼
[04] MMseqs2 Sequence Search ──────── search_results.json
    │                                  (with release_date_cutoff + self-exclusion)
    ▼
[05] Build Cross-Template .npz ────── cross_templates/*.npz
    │
    ▼
[06] Build Template Index ──────────── rna_template_index.json
    │
    ▼
[07] RNATemplateFeaturizer ────────── rna_template_* features
    │
    ▼
[08] TemplateEmbedder (Protenix) ──── Training / Inference
```

---

## 2. Bug Fix Report (2026-03-14)

### Overview

A code review (`pipe_prob.md`) identified 5 confirmed bugs in the pipeline. All have been verified, fixed, and tested.

| Bug | Severity | Summary | Status |
|-----|----------|---------|--------|
| #1 + #9 | **High** | Self-exclusion ID mismatch: catalog keys (`4tna_A`) vs training IDs (`4tna`) | **Fixed** |
| #2 | **High** | E2E test never builds cross-templates, only validates self-templates | **Fixed** |
| #3 | **High** | `--pdb_list` defined but never consumed in MMseqs2 search | **Fixed** |
| #4 | **High** | `--release_date_cutoff` is a dead parameter — no filtering implemented | **Fixed** |
| #6 | **Medium** | Builder re-globs for CIF files instead of using catalog's `cif_path` | **Fixed** |

---

### 2.1 Bug #1 + #9: Self-Exclusion ID Mismatch (Critical Data Leakage)

**Problem**: RNA3DB catalog keys include chain info (e.g., `4tna_A`, `1jgp_1`), while training query IDs are pure 4-char PDB codes (e.g., `4tna`). Self-exclusion compared these directly:

```python
# BEFORE (broken):
# query_pdb = "4tna"   (from training JSON)
# target_pdb_id = "4tna_A"  (from catalog key)
# "4tna_A" != "4tna"  →  self-exclusion NEVER fires!

query_pdb = query_id.split("_")[0] if "_" in query_id else query_id  # "4tna"
target_pdb_id = target_info["pdb_id"]  # "4tna_A" (catalog key)
if target_pdb_id.lower() == query_pdb.lower():  # ALWAYS FALSE
    continue
```

Empirical evidence:
- 3,460 training query IDs: all pure PDB codes (no underscores)
- 13,128 catalog keys: all with underscores (`pdb_chain`)
- Exact match between sets: **0** (zero)
- Prefix matches (same base PDB): **2,211** — confirming the namespace mismatch

**Fix**: Added `extract_base_pdb_id()` to normalize both sides to 4-char PDB codes:

```python
# NEW: in 03_mmseqs2_search.py
def extract_base_pdb_id(entry_id: str) -> str:
    """Extract 4-char PDB code: '4tna_A' -> '4tna', '4tna' -> '4tna'."""
    parts = entry_id.split("_")
    base = parts[0].lower()
    if len(base) == 4:
        return base
    return base

# In mmseqs2_search(): store base_pdb_id in db_id_to_info
db_id_to_info[db_id] = {
    "pdb_id": pdb_id,
    "chain_id": chain_id,
    "base_pdb_id": extract_base_pdb_id(pdb_id),  # NEW
}

# Query side also normalized
query_to_pdb[qid] = extract_base_pdb_id(qid)  # Was: qid.split("_")[0]

# In parse_mmseqs2_results(): compare base PDB IDs
target_base_pdb = target_info.get("base_pdb_id", extract_base_pdb_id(target_pdb_id))
if exclude_self:
    query_pdb = query_to_pdb.get(query_id, "")
    if query_pdb and target_base_pdb == query_pdb.lower():  # NOW WORKS
        continue
```

Same fix applied to `pairwise_search()`.

**Verification**:
```
extract_base_pdb_id('4tna_A') == '4tna'  ✓
extract_base_pdb_id('4tna') == '4tna'    ✓
extract_base_pdb_id('1jgp_1') == '1jgp'  ✓
Self-exclusion: target '4tna_A' vs query '4tna' → base match '4tna' == '4tna' → EXCLUDED ✓
Non-self: target '1abc_B' vs query '4tna' → '1abc' != '4tna' → KEPT ✓
```

**Files modified**: `rna_template/scripts/03_mmseqs2_search.py`

---

### 2.2 Bug #2: E2E Test Missing Cross-Template Validation

**Problem**: `test_rna3d_e2e.sh` claimed to validate the full cross-template path but actually:
1. Built only `--mode self` templates
2. Ran MMseqs2 search with `--no_exclude_self`
3. Validated self-template NPZ files
4. Trained on self-templates

It **never** called `02_build_rna_templates.py --mode cross`, so the critical path `MMseqs2 search → cross-template NPZ → index → featurizer → train` was untested.

**Fix**: Added Steps 4-5 to the test script:

```bash
# BEFORE: jumped from MMseqs2 search directly to NPZ validation of self-templates

# AFTER: added cross-template build and index rebuild

# Step 4: Build Cross-Templates from Search Results
python3 "${SCRIPTS_DIR}/02_build_rna_templates.py" \
    --catalog "${TEST_CATALOG}" \
    --pdb_rna_dir "${RNA3D_DIR}" \
    --output_dir "${CROSS_TEMPLATE_DIR}" \
    --mode cross \
    --search_results "${TEST_SEARCH_RESULTS}" \
    --max_templates "${MAX_RNA_TEMPLATES}" \
    ${ARENA_ARGS}

# Step 5: Rebuild Index After Cross-Templates
python3 "${SCRIPTS_DIR}/03_mmseqs2_search.py" \
    --catalog "${TEST_CATALOG}" \
    --template_dir "${CROSS_TEMPLATE_DIR}" \
    --output_index "${TEST_INDEX}" \
    --output_search "${TEST_SEARCH_RESULTS}" \
    --strategy mmseqs2 \
    --skip_search

# Step 6: NPZ Validation (now validates cross-templates, not self-templates)
# Step 7: GPU Training (now uses cross-template index)
```

Test flow now validates the complete pipeline:
```
catalog → Arena → self-templates → MMseqs2 search → cross-templates → index → train
```

**Files modified**: `rna_template/scripts/test_rna3d_e2e.sh`

---

### 2.3 Bug #3: `--pdb_list` Not Consumed in MMseqs2 Search

**Problem**: `args.pdb_list` was defined in argparse and passed from `run_pipeline.sh`, but the search logic never used it to filter training sequences. Only `--training_pdb_list` (a separate parameter) was consumed in the fallback code path.

```python
# BEFORE: args.pdb_list defined at line 909 but never referenced in search logic
parser.add_argument("--pdb_list", default="", help="...")
# No code ever reads args.pdb_list for filtering
```

**Fix**: Added explicit filtering after loading training sequences:

```python
# AFTER: in main(), after loading training_seqs
if args.pdb_list and os.path.exists(args.pdb_list):
    with open(args.pdb_list) as fh:
        allowed_pdbs = {line.strip().lower() for line in fh if line.strip()}
    before = len(training_seqs)
    training_seqs = {
        k: v for k, v in training_seqs.items()
        if extract_base_pdb_id(k) in allowed_pdbs
    }
    logger.info("Filtered training sequences by --pdb_list: %d -> %d", before, len(training_seqs))
```

**Files modified**: `rna_template/scripts/03_mmseqs2_search.py`

---

### 2.4 Bug #4: `release_date_cutoff` Is a Dead Parameter

**Problem**: `--release_date_cutoff` was defined in argparse with the comment `"(Future use.)"` but no code anywhere actually filtered by release date. The RNA3DB metadata (`filter.json`) contains `release_date` for all 15,441 entries, but was never loaded.

```python
# BEFORE: dead parameter
parser.add_argument("--release_date_cutoff", default="",
    help="... (Future use.)")
# Zero references to args.release_date_cutoff in code
```

**Fix**: Implemented full release date filtering using RNA3DB's `filter.json`:

```python
# NEW: filter_catalog_by_release_date()
def filter_catalog_by_release_date(catalog, metadata_path, cutoff_date):
    """Filter catalog entries by release date using RNA3DB metadata.

    Removes entries whose release_date is after cutoff_date.
    Uses filter.json format: {entry_id: {release_date: "YYYY-MM-DD", ...}}
    """
    from datetime import datetime
    cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d")

    with open(metadata_path) as fh:
        metadata = json.load(fh)

    # Build lookup and filter
    filtered = {}
    for entry_id, chains in catalog.items():
        release_str = date_lookup.get(entry_id.lower(), "")
        if not release_str:
            filtered[entry_id] = chains  # keep if no metadata
            continue
        release = datetime.strptime(release_str, "%Y-%m-%d")
        if release <= cutoff:
            filtered[entry_id] = chains

    return filtered

# NEW: --rna3db_metadata parameter
parser.add_argument("--rna3db_metadata", default="",
    help="Path to RNA3DB metadata JSON (filter.json) for release date filtering.")

# In main(): apply filter when cutoff is specified
if args.release_date_cutoff:
    catalog = filter_catalog_by_release_date(
        catalog, args.rna3db_metadata, args.release_date_cutoff
    )
```

**Also updated**: `run_pipeline.sh` to accept `--release_date_cutoff` and `--rna3db_metadata`, with default metadata path pointing to `rna3db-jsons/filter.json`.

**Verification**:
```
# Synthetic test with 3 old entries (pre-2020) + 3 new entries (2024)
# Cutoff: 2023-01-01
# Result: kept=3, removed=3 ✓
#   Removed: 8t2p_A (2024-01-24), 8t2p_B (2024-01-24), 8wmn_O (2024-06-05)
#   Kept:    6n5s_A (2019-11-27), 5no2_A (2017-05-24), 5uyq_A (2017-06-07)
```

**Usage**:
```bash
bash run_pipeline.sh \
    --strategy mmseqs2 \
    --release_date_cutoff 2021-09-30 \
    --rna3db_metadata /path/to/rna3db-jsons/filter.json
```

**Files modified**: `rna_template/scripts/03_mmseqs2_search.py`, `rna_template/scripts/run_pipeline.sh`

---

### 2.5 Bug #6: Builder Re-Globs Instead of Using Catalog `cif_path`

**Problem**: `01_extract_rna_catalog.py` already stores the precise `cif_path` in every catalog entry. But `02_build_rna_templates.py` ignores this and calls `find_cif_path()` to recursively glob the filesystem every time. This is:
- Semantically redundant (two mechanisms for the same lookup)
- Performance-wasteful (recursive glob on 15K+ files)
- Fragile (glob results are nondeterministic with symlinks or duplicate names)

```python
# BEFORE: always re-globs
def find_cif_path(pdb_rna_dir, pdb_id):
    flat = os.path.join(pdb_rna_dir, f"{pdb_id}.cif")
    if os.path.exists(flat):
        return flat
    matches = glob.glob(..., recursive=True)  # expensive
    return matches[0] if matches else None
```

**Fix**: `find_cif_path()` now accepts an optional `catalog_cif_path` and uses it when valid:

```python
# AFTER: use catalog path first, glob only as fallback
def find_cif_path(pdb_rna_dir, pdb_id, catalog_cif_path=None):
    # 0. Use catalog-recorded path if available and valid
    if catalog_cif_path and os.path.exists(catalog_cif_path):
        return catalog_cif_path
    # 1. Flat layout
    flat = os.path.join(pdb_rna_dir, f"{pdb_id}.cif")
    if os.path.exists(flat):
        return flat
    # 2. Recursive search (fallback)
    matches = glob.glob(..., recursive=True)
    return matches[0] if matches else None
```

Updated both `build_self_template()` and `build_cross_template()` to pass `cif_path` from catalog:

```python
# In build_self_template():
catalog_cif = chain_info.get("cif_path")
cif_path = find_cif_path(pdb_rna_dir, pdb_id, catalog_cif_path=catalog_cif)

# In build_cross_template(): accepts catalog parameter, looks up cif_path
def build_cross_template(..., catalog=None):
    for spec in template_specs:
        catalog_cif = None
        if catalog and t_pdb_id in catalog:
            for ch in catalog[t_pdb_id]:
                if ch.get("cif_path"):
                    catalog_cif = ch["cif_path"]
                    break
        cif_path = find_cif_path(pdb_rna_dir, t_pdb_id, catalog_cif_path=catalog_cif)
```

**Files modified**: `rna_template/scripts/02_build_rna_templates.py`

---

## 3. Corrected Status Table

The previous report's status table had several incorrect "Fixed" claims. Here is the corrected version:

| Issue | Previous Status | Actual Status | Fix Details |
|-------|----------------|---------------|-------------|
| #1: Self-exclusion fails due to ID mismatch | Claimed "Fixed" | **Was broken, now truly fixed** | `extract_base_pdb_id()` normalizes both sides to 4-char PDB code |
| #2: E2E test missing cross-template validation | Claimed "Fixed" | **Was broken, now truly fixed** | Added Steps 4-5: cross-template build + index rebuild |
| #3: `--pdb_list` not effective in search | Claimed "Fixed" | **Was broken, now truly fixed** | Explicit filtering of training sequences by `--pdb_list` |
| #4: `release_date_cutoff` dead parameter | Claimed "Fixed" | **Was broken, now truly fixed** | `filter_catalog_by_release_date()` implemented with RNA3DB `filter.json` |
| #5: Index built before templates | **Was actually fixed** | Still fixed | Pipeline order correct in `run_pipeline.sh` |
| #6: Builder re-globs instead of using `cif_path` | Not mentioned | **Now fixed** | `find_cif_path()` uses catalog path first, glob as fallback |
| #9: ID semantic confusion (double chain suffix) | Not mentioned | **Now fixed** | Unified via `extract_base_pdb_id()` |

---

## 4. Validation Results (Post-Fix)

### 4.1 Unit Tests

```
extract_base_pdb_id:
  '4tna_A' -> '4tna'  ✓
  '4tna'   -> '4tna'  ✓
  '1jgp_1' -> '1jgp'  ✓

Self-exclusion:
  target '4tna_A' vs query '4tna' → CORRECTLY EXCLUDED ✓
  target '1abc_B' vs query '4tna' → CORRECTLY KEPT      ✓

Release date cutoff:
  cutoff=2023-01-01: removed 3/6 entries (all 2024+) ✓
  cutoff=2030-01-01: kept all entries                 ✓
```

### 4.2 E2E Pipeline Test (10 structures, skip training)

```
Step 1: Catalog extraction          → 10 structures     ✓
Step 2: Self-template build (Arena) → 10 NPZ files      ✓
Step 3: MMseqs2 search              → 10/10 queries hit  ✓
Step 4: Cross-template build        → 10 NPZ files      ✓  (NEW)
Step 5: Index rebuild               → 2 unique seqs      ✓  (NEW)
Step 6: NPZ validation              → 10/10 passed       ✓
```

### 4.3 Full E2E Test with GPU Training (10 structures, 10 steps)

```
Pipeline: catalog → Arena → self-templates → MMseqs2 → cross-templates → index → train
Training: 10 steps completed on NVIDIA H800                              ✓

Step 4 metrics:
  train/loss.avg:              2.97
  train/smooth_lddt_loss.avg:  0.35
  train/mse_loss.avg:          0.38

Step 9 metrics:
  train/loss.avg:              3.05
  train/smooth_lddt_loss.avg:  0.39
  train/mse_loss.avg:          0.35

Eval (rna_lddt/mean):          0.00712  (smoke test, expected low at step 10)

Result: PASSED ✓
```

---

## 5. Previous Implementation (Unchanged)

### 5.1 Catalog Extraction (`01_extract_rna_catalog.py`)

Recursive glob for `rna3db-mmcifs` nested structure. Stores `cif_path` per chain entry. Supports `--pdb_list` filter.

### 5.2 Arena Integration (`02_build_rna_templates.py`)

CIF→PDB conversion + Arena atom-filling (option 5) before template construction. Fallback to original CIF on Arena failure.

### 5.3 MMseqs2 Search (`03_mmseqs2_search.py`)

Full MMseqs2 pipeline: `createdb → search → convertalis → parse`. Nucleotide search mode (`--search-type 3`), configurable sensitivity/evalue.

### 5.4 Pipeline Order (`run_pipeline.sh`)

Correct order: catalog → search → cross-template build → index rebuild.

---

## 6. Usage

### Build Template Database (Self-Template Mode)
```bash
bash rna_template/scripts/run_pipeline.sh \
    --strategy self \
    --use_arena \
    --pdb_rna_dir /path/to/rna3db-mmcifs \
    --output_dir /path/to/rna_database
```

### Build Template Database (MMseqs2 Cross-Template Mode)
```bash
bash rna_template/scripts/run_pipeline.sh \
    --strategy mmseqs2 \
    --use_arena \
    --pdb_rna_dir /path/to/rna3db-mmcifs \
    --output_dir /path/to/rna_database \
    --min_identity 0.3 \
    --max_templates 4
```

### Build with Release Date Cutoff (Anti-Leakage)
```bash
bash rna_template/scripts/run_pipeline.sh \
    --strategy mmseqs2 \
    --use_arena \
    --pdb_rna_dir /path/to/rna3db-mmcifs \
    --output_dir /path/to/rna_database \
    --release_date_cutoff 2021-09-30 \
    --rna3db_metadata /path/to/rna3db-jsons/filter.json
```

### Run E2E Test
```bash
bash rna_template/scripts/test_rna3d_e2e.sh --num_test 30 --max_steps 20
```

### Run Finetune with RNA Templates
```bash
bash finetune/finetune_rna_template_1stage.sh \
    --use_rna_template true \
    --rna_projector_init protein \
    --rna_template_alpha 0.01
```

---

## 7. Key Design Decisions

1. **Arena before template building**: RNA structures from rna3db often lack full atomic detail (P-only or backbone-only). Arena fills missing atoms (option 5) to produce full-atom structures, improving template quality (especially anchor coverage and distogram accuracy).

2. **MMseqs2 over pairwise alignment**: Scales from O(Q×DB) to near-linear with preindexing. Uses nucleotide search mode (`--search-type 3`) for RNA-specific scoring.

3. **Self-template for validation, cross-template for production**: Self-templates guarantee 100% coverage (each structure is its own template). Cross-templates via MMseqs2 search provide realistic template quality for training.

4. **Index built after templates**: Ensures the index only references .npz files that actually exist on disk, preventing "directory exists but unusable" states.

5. **Base PDB ID normalization**: All PDB ID comparisons (self-exclusion, pdb_list filtering) now go through `extract_base_pdb_id()` to handle the RNA3DB naming convention (`pdb_chain` stems like `4tna_A`).

6. **Catalog `cif_path` as primary lookup**: Builder uses the `cif_path` stored in the catalog by Step 1, avoiding expensive recursive filesystem globs. Falls back to glob only when catalog path is missing or invalid.

7. **Release date cutoff via RNA3DB metadata**: `filter.json` from RNA3DB contains per-entry `release_date`. The pipeline can filter the template database by a cutoff date to prevent temporal data leakage during training.

---

## 8. Files Modified in This Fix

| File | Changes |
|------|---------|
| `scripts/03_mmseqs2_search.py` | Added `extract_base_pdb_id()`, fixed self-exclusion, implemented `--pdb_list` filtering, implemented `filter_catalog_by_release_date()` with `--rna3db_metadata` |
| `scripts/02_build_rna_templates.py` | `find_cif_path()` accepts `catalog_cif_path`, `build_cross_template()` accepts `catalog` param |
| `scripts/test_rna3d_e2e.sh` | Added cross-template build (Step 4), index rebuild (Step 5), validates cross-template NPZ |
| `scripts/run_pipeline.sh` | Added `--release_date_cutoff` and `--rna3db_metadata` parameters, passed to search commands |

## 9. Directory Structure

```
rna_template/
├── run_arena_and_template.sh          # Single-structure pipeline (unchanged)
├── compute/
│   ├── rna_template_common.py         # Shared library (unchanged)
│   └── build_rna_template_protenix.py # Single-template builder (unchanged)
├── scripts/
│   ├── 01_extract_rna_catalog.py      # Modified: recursive glob, cif_path
│   ├── 02_build_rna_templates.py      # Modified: Arena, catalog cif_path lookup
│   ├── 03_search_and_index.py         # Original: pairwise search (kept)
│   ├── 03_mmseqs2_search.py           # Modified: ID normalization, pdb_list, date cutoff
│   ├── run_pipeline.sh                # Modified: date cutoff args
│   ├── test_small_e2e.sh              # Original E2E test (kept)
│   └── test_rna3d_e2e.sh              # Modified: cross-template validation
└── report/
    ├── rna3d_pipeline.md              # This report
    └── pipe_prob.md                   # Code review findings

rna_database/                          # Production database
├── rna_catalog.json                   # 13,128 structures
├── rna_template_index.json            # Sequence → template mapping
├── search_results.json                # MMseqs2 search results
└── templates/                         # *.npz template files
```
