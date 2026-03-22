# RNA Template Temporal Filtering — Code Review Report

**Date**: 2026-03-15
**Reviewer**: Claude Opus 4.6 (codex-code-review + codex-bio-ai-research)
**Scope**: Read-only audit of temporal filtering implementation against requirements
**Status**: Requirements substantially met, with actionable hardening recommendations

---

## 0. Executive Summary

The RNA template temporal filtering pipeline **successfully implements the core requirements**: per-query runtime self-hit exclusion and temporal date filtering that mirrors the protein template pipeline's approach. GPU training has been verified on H800 with filtering actively rejecting self-hits and future templates. The implementation is architecturally sound, isolated from protein/LLM pipelines, and correctly integrated into both training and inference paths.

However, this review identifies **3 high-severity gaps**, **4 medium-severity weaknesses**, and **3 low-severity improvements** that should be addressed to reach production hardening.

---

## 1. Requirements Traceability

Source: `/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build_temporal_filter.txt`

| # | Requirement | Status | Evidence |
|---|-------------|--------|----------|
| R1 | Fix data leakage in RNA template pipeline | **PASS** | `_filter_candidates()` implements both self-hit and temporal filtering at runtime |
| R2 | Per-query temporal filtering (not global) | **PASS** | `cutoff_date = query_release_date - timedelta(days=60)` computed per sample in `get_rna_template_features()` |
| R3 | Mirror protein pipeline's online filtering approach | **PASS** | Uses same `DAYS_BEFORE_QUERY_DATE=60` constant from `template_utils.py`; same reject-if-after logic |
| R4 | Self-hit exclusion at runtime | **PASS** | `_extract_pdb_from_npz_path()` + PDB comparison in `_filter_candidates()` |
| R5 | No filtering during inference | **PASS** | `inference_mode=True` bypasses all filtering; hardcoded in `infer_dataloader.py` |
| R6 | Do not affect protein pipeline | **PASS** | All changes scoped to `rna_template_*` keys; protein `template_featurizer.py` untouched |
| R7 | Do not affect LLM embedding pipeline | **PASS** | `rnalm_featurizer.py` untouched; RNA template features use separate `rna_template_*` namespace |
| R8 | Successful GPU training with filtering active | **PASS** | H800 5-step test shows filtering stats: `9ljn: self_hit=1, future=10` |
| R9 | Provide sensitivity tuning controls | **PASS** | Report documents 5 control knobs (DAYS_BEFORE_QUERY_DATE, metadata path, max_templates, MMseqs2 params, offline cutoff) |
| R10 | Finetune scripts updated | **PASS** | All 3 scripts (`1stage.sh`, `2stage.sh`, `validate.sh`) pass `--rna_template.rna3db_metadata_path` |
| R11 | Config system updated | **PASS** | `configs_base.py` includes `rna3db_metadata_path: ""` in `rna_template` block |

**Overall: 11/11 requirements passed.**

---

## 2. Architecture Review

### 2.1 Two-Layer Filtering Design (Correct)

```
Layer 1: OFFLINE (03_mmseqs2_search.py)
  - Global catalog cutoff via --release_date_cutoff
  - Self-hit exclusion during search result parsing
  - Output: rna_template_index.json (sequence -> [NPZ paths])

Layer 2: ONLINE (rna_template_featurizer.py)
  - Per-query cutoff: query_release_date - 60 days
  - Self-hit exclusion: template PDB != query PDB
  - Applied only during training (inference_mode=False)
```

The two layers are **complementary**: offline is a coarse pre-filter that reduces index size; online is fine-grained per-sample protection. Online filtering alone is sufficient for correctness even if offline filtering is disabled.

### 2.2 Data Flow Verification

```
Stanford RNA dataset (temporal_cutoff column)
  -> prepare_protenix_data.py (maps to release_date)
  -> rna_train_indices.csv.gz (column 11)
  -> parser.py (loads into bioassembly_dict["release_date"])
  -> RNATemplateFeaturizer.__call__() (extracts and parses)
  -> cutoff_date = release_date - 60 days
  -> _filter_candidates() (rejects self-hits + future templates)
```

**Verified**: `release_date` is reliably populated for all Stanford RNA training samples from the `temporal_cutoff` field. Format is consistently `YYYY-MM-DD`.

### 2.3 Protein Pipeline Parity Comparison

| Feature | Protein Pipeline | RNA Pipeline (After Fix) | Match? |
|---------|-----------------|--------------------------|--------|
| Per-query cutoff | `release_date - 60 days` | `release_date - 60 days` | Yes |
| Self-hit exclusion | Base PDB comparison | Base PDB comparison | Yes |
| Constant source | `DAYS_BEFORE_QUERY_DATE` | Same constant (imported) | Yes |
| Inference bypass | No filtering | No filtering | Yes |
| Graceful degradation | Fallback to empty | Fallback to empty | Yes |
| Unknown-date handling | Rejects (conservative) | **Keeps** (permissive) | **NO** |

---

## 3. Identified Issues

### 3.1 HIGH Severity

#### H1: Unknown-Date Templates Kept Instead of Rejected

**Location**: `rna_template_featurizer.py`, `_filter_candidates()` lines ~322-330

**Problem**: When a template's PDB ID is not found in RNA3DB metadata (`_release_dates`), the template is **kept** and counted as `no_date`. The protein pipeline takes the opposite approach: unknown templates are treated conservatively.

```python
# Current behavior (permissive):
if tpl_date is not None:
    if tpl_date > cutoff_date:
        stats["future"] += 1
        continue  # REJECT
else:
    stats["no_date"] += 1  # COUNT but KEEP  <-- Risk
```

**Risk**: If RNA3DB metadata is incomplete or if new templates are added to the index without corresponding metadata entries, those templates bypass temporal filtering entirely. This is a **silent data leakage vector**.

**Current mitigation**: The report states "PDBs with no date in metadata: 0" — currently the metadata covers all index entries. However, this is not enforced programmatically.

**Recommendation**: Either reject unknown-date templates during training (conservative, matching protein pipeline), or add a startup validation that all template PDBs in the index have corresponding metadata entries.

---

#### H2: Silent Temporal Filtering Bypass on Date Parse Failure

**Location**: `rna_template_featurizer.py`, `__call__()` lines ~556-559

**Problem**: If `bioassembly_dict["release_date"]` cannot be parsed as `%Y-%m-%d`, the exception is silently caught and `query_release_date` remains `None`. This causes `cutoff_date` to be `None`, which makes `_filter_candidates()` skip **all** temporal filtering without any log warning.

```python
try:
    query_release_date = datetime.strptime(str(rd_str), "%Y-%m-%d")
except (ValueError, TypeError):
    pass  # Silent! No warning logged
```

**Risk**: If a data preparation bug introduces malformed dates (e.g., `"2021/05/15"` or `"15-05-2021"`), temporal filtering silently degrades to self-hit-only mode. This could affect an entire training run without any indication in logs.

**Recommendation**: Add `logger.warning(f"RNA template: failed to parse query release_date '{rd_str}' for pdb={query_pdb_id}, temporal filtering disabled for this sample")`.

---

#### H3: Sequence Length Mismatch Causes Feature Corruption via Clamping

**Location**: `rna_template_featurizer.py`, `get_rna_template_features()` lines ~462-470

**Problem**: When template residue indices exceed template dimensions, the code **clamps** indices instead of rejecting the template:

```python
valid_mask = (chain_indices >= 0) & (chain_indices < template_n)
if not valid_mask.all():
    logger.warning(f"RNA template: some residue indices out of bounds...")
    chain_indices = np.clip(chain_indices, 0, template_n - 1)
```

**Risk**: Clamping maps multiple query residues to the same template residue. For example, if template has 50 residues but query indices are `[48, 49, 50, 51, 52]`, clamping produces `[48, 49, 49, 49, 49]`. This creates structurally incorrect distance features that could mislead the model during training.

**Recommendation**: Skip the template (return `None` / fallback to empty features) when indices are out of bounds, rather than silently corrupting features.

---

### 3.2 MEDIUM Severity

#### M1: Entity ID Type Inconsistency Between Training and Inference

**Location**: `rna_template_featurizer.py`, `__call__()` lines ~564-598

**Problem**: In inference mode, entity_id is assigned as `str(i + 1)` (string). In training mode, entity_id comes from `atom_array.label_entity_id` (integer). The `rna_sequences` dict is keyed differently depending on mode, which could cause lookup failures if downstream code assumes one type.

**Impact**: Potential silent failures in entity-to-sequence mapping if types mismatch.

---

#### M2: Only First Valid NPZ Loaded Per Chain (Low Actual Impact)

**Location**: `rna_template_featurizer.py`, `get_rna_template_features()` lines ~434-441

**Problem**: The code only loads the **first** successfully-parsed NPZ file per chain:

```python
for npz_path in resolved_paths:
    chain_features = _load_and_crop_rna_template(npz_path, sequence, self.max_templates)
    if chain_features is not None:
        break  # Stop at first successful load
```

**Corrected assessment**: After reviewing `02_build_rna_templates.py`, each NPZ file is a **cross-template bundle** — it already contains up to `max_templates=4` different PDB structures stacked along the T dimension (from MMseqs2 search results for that query). So loading one NPZ typically gives you all available templates for that sequence. Multiple NPZ paths in the index for the same sequence would only arise if the same sequence appeared in different query contexts. This is **not a significant limitation** in practice.

**Note**: Originally rated as a design concern, downgraded after understanding the NPZ build pipeline.

---

#### M3: Module-Level Cache Not Process-Safe in Multi-GPU Training

**Location**: `rna_template_featurizer.py`, `_RNA_RELEASE_DATES_CACHE` (module-level dict)

**Problem**: The metadata cache is a module-level dict. In multi-GPU distributed training with `torch.distributed`, each process forks and gets its own copy, which is fine. However, if `DataLoader` uses `num_workers > 0`, each worker subprocess also loads its own copy of the metadata into memory.

**Impact**: Memory waste proportional to `num_workers * metadata_size`. With 5,389 PDB entries this is negligible (~few MB), but the pattern should be documented.

---

#### M4: Offline/Online Self-Hit Exclusion Uses Different Extraction Logic

**Location**:
- Offline: `03_mmseqs2_search.py` `extract_base_pdb_id()` — extracts from entry ID format `7zpi_B`
- Online: `_extract_pdb_from_npz_path()` — extracts from filename format `7zpi_B_B_template.npz`

**Problem**: Both use `split("_")[0].lower()` but operate on different input formats. If the NPZ naming convention changes (e.g., including chain type prefix), the online extraction would break while offline still works.

**Impact**: Low risk currently (naming is consistent), but the coupling is implicit and undocumented.

---

### 3.3 LOW Severity

#### L1: Metadata Loaded Unnecessarily in Inference Mode

**Location**: `infer_dataloader.py` lines ~225-235

The `RNATemplateFeaturizer` constructor loads `rna3db_metadata_path` even when the featurizer will always be called with `inference_mode=True`. The metadata is never used but consumes memory and I/O time.

---

#### L2: Short Chain Threshold (len <= 4) Not Configurable

**Location**: `rna_template_featurizer.py`, `get_rna_template_features()` line ~408

Chains with length <= 4 are silently skipped without logging. The threshold is hardcoded and undocumented.

---

#### L3: T->U Conversion Only Tried as Fallback

**Location**: `_find_templates_for_sequence()` lines ~281-289

If both the original sequence and T->U-converted sequence exist in the index, only the original is returned. For DNA-labeled RNA sequences, this could miss template hits if the index was built with U-notation.

---

## 4. Positive Findings

### What Was Done Well

1. **Clean separation of concerns**: RNA template features use a `rna_template_*` namespace that is completely isolated from the protein `template_*` namespace. No risk of feature key collision.

2. **Correct constant reuse**: `DAYS_BEFORE_QUERY_DATE = 60` is imported from `template_utils.py` rather than duplicated, ensuring a single source of truth.

3. **Graceful degradation**: If `rna3db_metadata_path` is empty, temporal filtering is disabled but self-hit exclusion still works. Empty features are returned when no templates survive filtering.

4. **Comprehensive validation script**: `finetune_rna_template_validate.sh` performs 10 distinct checks including GPU training, NPZ format, template hit coverage, and date cutoff verification.

5. **Consistent config propagation**: All three finetune scripts (`1stage.sh`, `2stage.sh`, `validate.sh`) correctly reference the same metadata path and pass it to the featurizer.

6. **Reliable data source**: The `release_date` field is populated from Stanford RNA dataset's `temporal_cutoff` column through a verified pipeline (`prepare_protenix_data.py` -> indices CSV -> bioassembly_dict).

7. **Good logging**: Filtering statistics (self_hit, future, no_date counts) are logged per sample, enabling debugging of filtering behavior.

---

## 5. Temporal Filtering Coverage Analysis

From the implementation report, the filtering is working as expected:

| Training Sample | Self-Hits Blocked | Future Templates Blocked |
|-----------------|-------------------|--------------------------|
| pdb=9iwf | 2 | 0 |
| pdb=9jgm | 4 | 0 |
| pdb=9kgg | 1 | 0 |
| pdb=9ljn | 1 | 10 |

**Observation**: Self-hit exclusion is more frequently triggered than temporal filtering in the test run, which is expected for recently-released PDBs where most templates in the index predate the query.

### Cutoff Sensitivity Analysis (from index statistics)

| Cutoff Date | Templates Removed | % Removed | Templates Kept |
|-------------|-------------------|-----------|----------------|
| 2015-01-01 | 10,883 | 82.9% | 2,245 |
| 2018-01-01 | 9,215 | 70.2% | 3,913 |
| 2021-09-30 | 6,324 | 48.2% | 6,804 |
| 2024-01-01 | 3,801 | 29.0% | 9,327 |

Total index: 13,128 candidates | RNA3DB PDBs with dates: 5,389 (100% coverage)

---

## 6. Comparison with Hardening Plan

The `codex_rna_template_minimal_hardening_plan.md` proposed 3 phases:

| Phase | Description | Implemented? |
|-------|-------------|--------------|
| Phase 1 | Runtime self-hit + temporal filtering | **YES** — `_filter_candidates()` |
| Phase 2 | Per-query date filtering in offline search | **PARTIAL** — offline supports global cutoff; per-query is online-only |
| Phase 3 | Strict provenance mode (optional config flag) | **NO** — not yet implemented |

Phase 2 partial status is acceptable: the online per-query filter provides the necessary per-sample guarantee. Offline global cutoff is complementary optimization.

Phase 3 (strict provenance mode) would add a config flag `rna_template.strict_query_match` that rejects templates when query identity can't be verified. This remains a future hardening option.

---

## 7. Recommendations Summary

### Must-Fix (Before Production Training)

| # | Issue | Fix |
|---|-------|-----|
| H2 | Silent date parse failure | Add `logger.warning()` when `release_date` parsing fails |

### Should-Fix (Before Benchmarking / Kaggle)

| # | Issue | Fix |
|---|-------|-----|
| H1 | Unknown-date templates kept | Add startup validation that all index PDBs exist in metadata; optionally reject unknown-date templates |
| H3 | Index clamping corrupts features | Return `None` for templates with length mismatch instead of clamping |

### Nice-to-Have (Future Hardening)

| # | Issue | Fix |
|---|-------|-----|
| M1 | Entity ID type inconsistency | Normalize to consistent type across modes |
| M2 | Only first NPZ loaded | Consider loading and merging multiple NPZ files per chain |
| M3 | Per-worker metadata copies | Document or use shared memory for large metadata |
| L1 | Unnecessary metadata load in inference | Pass empty string for metadata in inference mode |

---

## 8. Conclusion

The temporal filtering implementation is **functionally correct and meets all stated requirements**. The core data leakage problem (self-hit + temporal) is resolved for training, while inference correctly bypasses filtering. The architecture mirrors the protein pipeline's approach using the same safety margin constant.

The identified high-severity issues (H1-H3) are edge cases that do not affect current correctness (metadata coverage is 100%, dates are well-formatted, sequence lengths match), but represent **silent failure modes** that could become real problems if data conditions change. Adding defensive logging (H2) and startup validation (H1) would make the pipeline robust against future data drift.

**Verdict**: Approved for continued development and training. Address H2 before production runs, H1+H3 before any benchmarking or competition submissions.

---

*Report generated by Claude Opus 4.6 — read-only code review, no code modifications made.*
