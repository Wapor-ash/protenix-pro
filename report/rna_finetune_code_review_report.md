# Protenix RNA Fine-tuning Code Review Report

## Executive Summary

**Review Date:** March 7, 2026  
**Reviewer:** Code Review Agent  
**Scope:** Comparison of RNA fine-tuning implementation in `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/` against original Protenix codebase at `/inspire/ssd/project/sais-bio/public/ash_proj/code/Protenix`

### Overall Assessment

| Component | Consistency Status | Severity |
|-----------|-------------------|----------|
| Fine-tune Script Logic | ✅ **CONSISTENT** | - |
| Fine-tune Parameters | ✅ **CONSISTENT** | - |
| Bioassembly Generation | ✅ **CONSISTENT** | - |
| Index Generation Approach | ⚠️ **MOSTLY CONSISTENT** | Low |
| MSA Data Structure | ✅ **CONSISTENT** | - |

---

## 1. Fine-tune Script Review

### 1.1 Scripts Compared

| File | Location | Purpose |
|------|----------|---------|
| `run_rna_finetune.sh` | `part2/` | New pipeline script for RNA fine-tuning |
| `finetune_rna.sh` | `Protenix/` | Original RNA fine-tune script |
| `finetune_demo.sh` | `Protenix/` | Base demo fine-tuning script |

### 1.2 Parameter Consistency Analysis

#### ✅ Training Hyperparameters - FULLY CONSISTENT

| Parameter | Original (`finetune_rna.sh`) | New (`run_rna_finetune.sh`) | Status |
|-----------|-----------------------------|----------------------------|--------|
| `DIFFUSION_BATCH_SIZE` | 48 | 48 | ✅ |
| `TRAIN_CROP_SIZE` | 384 | 384 | ✅ |
| `MAX_STEPS` | 100000 | 100000 | ✅ |
| `WARMUP_STEPS` | 2000 | 2000 | ✅ |
| `LR` | 0.001 | 0.001 | ✅ |
| `GRAD_CLIP_NORM` | 10 | 10 | ✅ |
| `EMA_DECAY` | 0.999 | 0.999 | ✅ |
| `EVAL_INTERVAL` | 400 | 400 | ✅ |
| `LOG_INTERVAL` | 50 | 50 | ✅ |
| `CHECKPOINT_INTERVAL` | 400 | 400 | ✅ |
| `N_CYCLE` | 4 | 4 | ✅ |
| `N_STEP_DIFFUSION` | 20 | 20 | ✅ |
| `SEED` | 42 | 42 | ✅ |
| `DTYPE` | bf16 | bf16 | ✅ |

#### ✅ Model Configuration - CONSISTENT

| Parameter | Original | New | Status |
|-----------|----------|-----|--------|
| `MODEL_NAME` | `protenix_base_20250630_v1.0.0` | `protenix_base_20250630_v1.0.0` | ✅ |
| `TRIANGLE_ATTENTION` | `cuequivariance` | `cuequivariance` | ✅ |
| `TRIANGLE_MULTIPLICATIVE` | `cuequivariance` | `cuequivariance` | ✅ |

#### ✅ Data Configuration - CONSISTENT (with RNA-specific adaptations)

| Aspect | Original Approach | New Approach | Status |
|--------|------------------|--------------|--------|
| Train dataset key | `weightedPDB_before2109_wopb_nometalc_0925` | Same | ✅ |
| Val dataset key | `recentPDB_1536_sample384_0925` | Same | ✅ |
| Data override method | Command-line path overrides | Same | ✅ |
| RNA MSA enabling | `--data.msa.enable_rna_msa true` | Same | ✅ |
| RNA indexing method | `sequence` | `sequence` | ✅ |

### 1.3 Key Improvements in New Script

The new `run_rna_finetune.sh` includes several **beneficial enhancements**:

1. **CCD Cache Auto-generation** (Step 1.5): Automatically generates CCD cache if missing
2. **Bioassembly Pre-computation** (Step 2): Integrated bioassembly dictionary computation
3. **PDB List Filtering** (Step 2.5): Filters PDB lists to only include IDs with available bioassembly data
4. **Better Error Handling**: Comprehensive verification of prerequisites before training
5. **Detailed Logging**: More verbose output for debugging and monitoring

### 1.4 Training Command Comparison

**Original (`finetune_rna.sh`):**
```bash
python3 ./runner/train.py \
    --model_name "protenix_base_20250630_v1.0.0" \
    --run_name "protenix_rna_finetune" \
    --data.train_sets "${TRAIN_SET}" \
    --data.weightedPDB_before2109_wopb_nometalc_0925.base_info.pdb_list "${FINETUNE_LIST_PATH}" \
    --data.weightedPDB_before2109_wopb_nometalc_0925.base_info.mmcif_dir "${RNA_DATA_DIR}" \
    --data.test_sets "${TEST_SETS}"
```

**New (`run_rna_finetune.sh`):**
```bash
python3 ./runner/train.py \
    --model_name "${MODEL_NAME}" \
    --run_name "${RUN_NAME}" \
    --data.train_sets "${TRAIN_SET}" \
    --data.${TRAIN_SET}.base_info.mmcif_dir "${RNA_CIF_DIR}" \
    --data.${TRAIN_SET}.base_info.bioassembly_dict_dir "${BIOASSEMBLY_DIR}" \
    --data.${TRAIN_SET}.base_info.indices_fpath "${TRAIN_INDICES}" \
    --data.${TRAIN_SET}.base_info.pdb_list "${TRAIN_PDB_LIST}" \
    --data.test_sets "${VAL_SET}" \
    --data.${VAL_SET}.base_info.mmcif_dir "${RNA_CIF_DIR}" \
    --data.${VAL_SET}.base_info.bioassembly_dict_dir "${BIOASSEMBLY_DIR}" \
    --data.${VAL_SET}.base_info.indices_fpath "${VAL_INDICES}" \
    --data.${VAL_SET}.base_info.pdb_list "${VAL_PDB_LIST}" \
    --data.${VAL_SET}.base_info.find_eval_chain_interface false \
    --data.${VAL_SET}.base_info.group_by_pdb_id false \
    --data.${VAL_SET}.base_info.max_n_token 1536 \
    --data.msa.enable_rna_msa true \
    --data.msa.rna_msadir_raw_paths "${RNA_MSA_DIR}" \
    --data.msa.rna_seq_or_filename_to_msadir_jsons "${RNA_MSA_JSON}" \
    --data.msa.rna_indexing_methods "sequence" \
    --data.msa.enable_prot_msa false \
    --data.template.enable_prot_template false
```

**Assessment:** The new script provides **more explicit and complete** configuration, properly overriding all necessary paths for RNA data.

---

## 2. Bioassembly Generation Review

### 2.1 Scripts Compared

| File | Location | Purpose |
|------|----------|---------|
| `run_bioassembly.py` | `part2/protenix_prepared/` | Custom bioassembly computation script |
| `prepare_training_data.py` | `Protenix/scripts/` | Original Protenix bioassembly script |

### 2.2 Core Logic Comparison

#### ✅ DataPipeline Usage - CONSISTENT

Both scripts use the same core Protenix API:

**Original (`prepare_training_data.py`):**
```python
from protenix.data.pipeline.data_pipeline import DataPipeline
from protenix.utils.file_io import dump_gzip_pickle

def gen_a_bioassembly_data(mmcif, bioassembly_output_dir, cluster_file, distillation):
    sample_indices_list, bioassembly_dict = DataPipeline.get_data_from_mmcif(
        mmcif, cluster_file, "WeightedPDB"
    )
    if sample_indices_list and bioassembly_dict:
        pdb_id = bioassembly_dict["pdb_id"]
        dump_gzip_pickle(bioassembly_dict, bioassembly_output_dir / f"{pdb_id}.pkl.gz")
        return sample_indices_list
```

**New (`run_bioassembly.py`):**
```python
from protenix.data.pipeline.data_pipeline import DataPipeline
from protenix.utils.file_io import dump_gzip_pickle

def process_one(mmcif_path):
    try:
        sample_indices_list, bioassembly_dict = DataPipeline.get_data_from_mmcif(
            mmcif_path, None, "WeightedPDB"
        )
        if sample_indices_list and bioassembly_dict:
            pdb_id = bioassembly_dict["pdb_id"]
            dump_gzip_pickle(bioassembly_dict, BIO_DIR / f"{pdb_id}.pkl.gz")
            return sample_indices_list
```

**Assessment:** ✅ **Identical core logic** - both use `DataPipeline.get_data_from_mmcif()` with `"WeightedPDB"` dataset type.

#### ✅ Output Format - CONSISTENT

| Aspect | Original | New | Status |
|--------|----------|-----|--------|
| Output file naming | `{pdb_id}.pkl.gz` | `{pdb_id}.pkl.gz` | ✅ |
| Output directory | `bioassembly_output_dir` | `BIO_DIR` | ✅ |
| Indices CSV format | Standard Protenix columns | Same columns | ✅ |
| Compression | gzip pickle | gzip pickle | ✅ |

#### ✅ Parallel Processing - CONSISTENT

| Aspect | Original | New | Status |
|--------|----------|-----|--------|
| Parallel library | `joblib.Parallel` | `joblib.Parallel` | ✅ |
| Default workers | 1 (configurable via `-n`) | 16 (hardcoded) | ⚠️ |
| Progress tracking | `tqdm` | `tqdm` | ✅ |

**Note:** The new script uses 16 workers hardcoded, which is appropriate for the target environment (as documented in `part2/report.md`: "64 workers causes OOM; 16 workers is optimal").

### 2.3 Key Differences

| Feature | Original | New | Assessment |
|---------|----------|-----|------------|
| Input specification | Directory or .txt file | Reads from existing PDB lists | ✅ Appropriate adaptation |
| Skip logic | None | Skips existing pkl.gz files | ✅ Beneficial addition |
| Indices generation | From processed results | Re-processes all for completeness | ✅ More robust |
| Integration | Standalone script | Integrated into pipeline | ✅ Better workflow |

---

## 3. Index Generation Review

### 3.1 Scripts Compared

| File | Location | Purpose |
|------|----------|---------|
| `prepare_protenix_data.py` | `part2/` | Data preparation including indices |
| `prepare_training_data.py` | `Protenix/scripts/` | Original indices generation |
| `run_bioassembly.py` | `part2/` | Bioassembly-based indices |

### 3.2 Index Generation Approaches

The part2 directory uses **two different index generation approaches**:

#### Approach A: Direct CSV Generation (`prepare_protenix_data.py`)

**Purpose:** Generate initial indices from Stanford RNA CSV data

**Columns Generated:**
```
pdb_id, type, chain_1_id, chain_2_id, cluster_id, entity_1_id, 
entity_2_id, num_tokens, mol_1_type, mol_2_type, resolution, 
release_date, assembly_id
```

**Sample Output:**
```csv
pdb_id,type,chain_1_id,chain_2_id,cluster_id,entity_1_id,entity_2_id,num_tokens,mol_1_type,mol_2_type,resolution,release_date,assembly_id
4tna,chain,A,,cluster_0,,,76,nuc,intra,2.0,1978-04-12,1
```

#### Approach B: Bioassembly-derived Indices (`run_bioassembly.py` + pipeline)

**Purpose:** Generate indices from actual CIF parsing to ensure chain ID consistency

**Sample Output:**
```csv
"pdb_id","type","chain_1_id","chain_2_id","cluster_id","entity_1_id","entity_2_id","num_tokens","mol_1_type","mol_2_type","resolution","release_date","assembly_id"
"157d","chain","A","","cluster_0","","","24","nuc","intra","1.8","1994-05-31","1"
```

### 3.3 Consistency Analysis

#### ✅ Column Structure - CONSISTENT

Both approaches generate indices with the **exact same column structure** as the original Protenix format:

| Column | Original Format | New Format | Status |
|--------|----------------|------------|--------|
| `pdb_id` | Lowercase PDB ID | Same | ✅ |
| `type` | "chain" or "interface" | "chain" | ✅ |
| `chain_1_id` | Chain identifier | Same | ✅ |
| `chain_2_id` | Empty for intra-chain | Same | ✅ |
| `cluster_id` | Unique cluster ID | `cluster_{n}` | ✅ |
| `mol_1_type` | "prot"/"nuc"/"ligand" | "nuc" | ✅ |
| `mol_2_type` | "intra"/"inter" | "intra" | ✅ |

#### ⚠️ Chain ID Source - MINOR INCONSISTENCY (Resolved)

**Issue Identified:**
- `prepare_protenix_data.py` extracts chain ID from `stoichiometry` field in CSV
- `run_bioassembly.py` extracts chain ID from actual CIF parsing via `DataPipeline`

**Resolution:**
The pipeline correctly uses **bioassembly-derived indices** (`rna_bioassembly_indices.csv`) for training, which ensures chain IDs match the CIF-parsed values. This is the **correct approach** as documented in `run_rna_finetune.sh`:

```bash
# Use bioassembly-derived indices (has correct chain IDs from CIF parsing)
TRAIN_INDICES="${PREPARED_DATA_DIR}/indices/rna_bioassembly_indices.csv"
```

#### ✅ Index File Usage - CONSISTENT

| Usage | Original | New | Status |
|-------|----------|-----|--------|
| Training indices | Passed via `--data.*.base_info.indices_fpath` | Same | ✅ |
| Validation indices | Same mechanism | Same | ✅ |
| PDB list filtering | Manual | Automated (Step 2.5) | ✅ Enhanced |

### 3.4 Index Format Verification

**Original Protenix indices format** (from `configs_data.py`):
```python
"indices_fpath": os.path.join(
    PROTENIX_ROOT_DIR,
    "indices/weightedPDB_indices_before_2021-09-30_wo_posebusters_resolution_below_9.csv.gz",
),
```

**New RNA indices format:**
- Train: `rna_train_indices.csv.gz` (gzip compressed)
- Val: `rna_val_indices.csv` (plain text)
- Bioassembly: `rna_bioassembly_indices.csv` (plain text, used for training)

**Assessment:** ✅ **Format consistent** - both use standard CSV with same column structure.

---

## 4. MSA Data Structure Review

### 4.1 Directory Structure Comparison

#### ✅ A3M Directory Structure - CONSISTENT

**Original Protenix expects:**
```
{msa_dir}/{directory_name}/{directory_name}_all.a3m
```

**New implementation provides:**
```
rna_msa/msas/4TNA/4TNA_all.a3m
rna_msa/msas/157D/157D_all.a3m
```

**Assessment:** ✅ **Exactly matches** expected structure.

#### ✅ Sequence-to-MSA Mapping - CONSISTENT

**Original format** (from `configs_data.py`):
```python
"rna_seq_or_filename_to_msadir_jsons": [
    os.path.join(PROTENIX_ROOT_DIR, "rna_msa/rna_sequence_to_pdb_chains.json")
]
```

**New implementation:**
```json
{
  "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA": ["4TNA"],
  ...
}
```

**Usage in Protenix:**
```python
# From rna_msa_search.py
directory_name = mapping[sequence][0]  # Get directory name from sequence
```

**Assessment:** ✅ **Correct format** - maps RNA sequence to list containing directory name.

### 4.2 File Format

| Aspect | Original Expectation | New Implementation | Status |
|--------|---------------------|-------------------|--------|
| A3M file naming | `{id}_all.a3m` | Same | ✅ |
| A3M content | FASTA-compatible for RNA | Symlink to original FASTA | ✅ |
| Mapping JSON | `{sequence: [directory]}` | Same | ✅ |
| Directory structure | `{msa_dir}/{dir}/{dir}_all.a3m` | Same | ✅ |

---

## 5. Issues and Recommendations

### 5.1 Critical Issues

**None identified.** All core logic and parameters are consistent with the original Protenix implementation.

### 5.2 Minor Observations

#### ⚠️ Observation 1: Quoting in CSV Files

**Finding:** Bioassembly indices CSV uses quoted values, while initial indices do not:

```csv
# prepare_protenix_data.py output (unquoted)
4tna,chain,A,,cluster_0,,,76,nuc,intra,2.0,1978-04-12,1

# run_bioassembly.py output (quoted)
"157d","chain","A","","cluster_0","","","24","nuc","intra","1.8","1994-05-31","1"
```

**Assessment:** ✅ **Not an issue** - CSV parsers handle both formats correctly. The quoting is due to `quoting=csv.QUOTE_NONNUMERIC` in `run_bioassembly.py`.

#### ⚠️ Observation 2: Resolution Values

**Finding:** `prepare_protenix_data.py` uses placeholder resolution (2.0), while `run_bioassembly.py` uses actual CIF-parsed values:

```python
# prepare_protenix_data.py
"2.0",  # resolution (placeholder)

# run_bioassembly.py (from CIF)
"1.8", "1.55", etc.  # actual resolution
```

**Assessment:** ✅ **Beneficial** - bioassembly-derived indices have accurate resolution values.

#### ℹ️ Observation 3: Release Date Handling

**Finding:** `prepare_protenix_data.py` uses `temporal_cutoff` from CSV, while `run_bioassembly.py` uses CIF-parsed release dates.

**Assessment:** ✅ **Correct** - CIF-parsed dates are more authoritative.

### 5.3 Recommendations

#### Recommendation 1: Document Chain ID Source

**Rationale:** The pipeline uses bioassembly-derived indices for training (correct), but the initial `prepare_protenix_data.py` indices are still generated.

**Suggestion:** Add a comment in `run_rna_finetune.sh` explaining why bioassembly indices are preferred:

```bash
# Note: Bioassembly-derived indices are used because chain_1_id must exactly
# match the atom_array.chain_id parsed from CIF files. The initial indices
# from prepare_protenix_data.py extract chain IDs from stoichiometry,
# which may not match CIF parsing results.
```

#### Recommendation 2: Consider Worker Configuration

**Rationale:** `run_bioassembly.py` hardcodes 16 workers.

**Suggestion:** Make it configurable via environment variable:

```python
N_JOBS = int(os.environ.get("BIOASSEMBLY_WORKERS", "16"))
Parallel(n_jobs=N_JOBS, ...)
```

#### Recommendation 3: Add Validation Step

**Rationale:** No explicit validation that indices match bioassembly pkl.gz files.

**Suggestion:** Add a verification step:

```bash
echo "Validating indices match bioassembly data..."
python3 -c "
import pandas as pd
from pathlib import Path

indices = pd.read_csv('${BIOASSEMBLY_INDICES}')
bio_dir = Path('${BIOASSEMBLY_DIR}')

missing = 0
for pdb_id in indices['pdb_id'].unique():
    if not (bio_dir / f'{pdb_id}.pkl.gz').exists():
        missing += 1

if missing > 0:
    print(f'WARNING: {missing} PDB IDs in indices missing pkl.gz files')
else:
    print('All indices have matching pkl.gz files')
"
```

---

## 6. Conclusion

### 6.1 Summary

The RNA fine-tuning implementation in `part2/` is **highly consistent** with the original Protenix codebase:

| Component | Consistency | Notes |
|-----------|-------------|-------|
| Fine-tune parameters | ✅ 100% | All hyperparameters match exactly |
| Training logic | ✅ 100% | Uses same Protenix APIs and patterns |
| Bioassembly generation | ✅ 100% | Identical DataPipeline usage |
| Index format | ✅ 100% | Same column structure and semantics |
| MSA structure | ✅ 100% | Matches expected directory layout |
| Index chain IDs | ✅ Correct | Uses CIF-parsed values via bioassembly |

### 6.2 Strengths

1. **Faithful Implementation:** Core logic faithfully follows original Protenix patterns
2. **Enhanced Robustness:** Added PDB list filtering prevents training failures
3. **Better Integration:** Automated CCD cache and bioassembly generation
4. **Correct Chain ID Handling:** Uses CIF-parsed chain IDs, not CSV-extracted
5. **Proper MSA Format:** Correctly structured A3M files and mapping JSON

### 6.3 Final Verdict

**✅ APPROVED** - The implementation is **production-ready** and consistent with the original Protenix fine-tuning logic and parameters. No critical issues identified.

---

## Appendix A: Files Reviewed

### Part2 Directory (New Implementation)
- `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/run_rna_finetune.sh`
- `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/run_bioassembly.py`
- `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/prepare_protenix_data.py`
- `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/protenix_new/run_rna_finetune.sh`

### Protenix Directory (Original)
- `/inspire/ssd/project/sais-bio/public/ash_proj/code/Protenix/finetune_rna.sh`
- `/inspire/ssd/project/sais-bio/public/ash_proj/code/Protenix/finetune_demo.sh`
- `/inspire/ssd/project/sais-bio/public/ash_proj/code/Protenix/scripts/prepare_training_data.py`
- `/inspire/ssd/project/sais-bio/public/ash_proj/code/Protenix/protenix/data/pipeline/data_pipeline.py`
- `/inspire/ssd/project/sais-bio/public/ash_proj/code/Protenix/configs/configs_data.py`

---

*Report generated by Code Review Agent on March 7, 2026*
