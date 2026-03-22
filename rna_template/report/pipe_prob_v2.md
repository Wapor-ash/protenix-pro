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

#### H1: Unknown-Date Templates Kept Instead of Rejected

**Location**: `rna_template_featurizer.py`, `_filter_candidates()` lines ~322-330

**Problem**: When a template's PDB ID is not found in RNA3DB metadata (`_release_dates`), the template is **kept** and counted as `no_date`. 

**Risk**: If RNA3DB metadata is incomplete or if new templates are added to the index without corresponding metadata entries, those templates bypass temporal filtering entirely. This is a **silent data leakage vector**.


TODO: for the sturctures unfound, please help me to use PDB API to check for release date, because RNA3DB may not be comphrehenseive and update the maintained release date file to keep track
---

#### H2: Only First Valid Template Loaded Per Chain

**Location**: `rna_template_featurizer.py`, `get_rna_template_features()` lines ~434-441

**Problem**: Although `max_templates=4` is configured and the NPZ format supports multiple templates per file, the code only loads the **first** successfully-parsed NPZ file per chain:

```python
for npz_path in resolved_paths:
    chain_features = _load_and_crop_rna_template(npz_path, sequence, self.max_templates)
    if chain_features is not None:
        break  # Stop at first successful load
```

**Impact**: If the first NPZ contains only 1-2 templates but other NPZ files contain additional valid templates, those are never loaded. The protein pipeline uses a different strategy: it collects hits from multiple sources and selects the top-N.

**Note**: This is a design limitation, not a bug. The current approach is functional but may underutilize available template data.


！！TODO:  fix this design limitation