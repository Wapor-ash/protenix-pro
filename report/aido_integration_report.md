# AIDO RNA/DNA Embedding Integration Report

## Overview

This report documents the integration of **AIDO foundation models** (AIDO.RNA-1.6B and AIDO.DNA-300M) into the Protenix fine-tuning pipeline, replacing the previous RiNALMo RNA language model. The integration enables both RNA and DNA per-residue embeddings to be injected into Protenix during structure prediction fine-tuning.

**Key changes:**
- AIDO.RNA-1.6B produces **2048-dim** per-nucleotide embeddings (vs RiNALMo's 1280-dim)
- AIDO.DNA-300M produces **1024-dim** per-nucleotide embeddings (zero-padded to 2048 to share projection layers)
- Both RNA and DNA embeddings are supported simultaneously through a unified featurizer
- All existing training logic, hyperparameters, and model architecture remain unchanged

---

## 1. Scripts Created

### 1.1 Sequence Extraction Script

**File:** `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/extract_sequences.py`

**Purpose:** Step 1 — Extract RNA and DNA sequences from bioassembly pkl.gz files.

**Environment:** `protenix` conda env (requires rdkit, protenix dependencies for unpickling)

**Usage:**
```bash
conda activate /inspire/ssd/project/sais-bio/public/ash_proj/conda/envs/protenix
python3 extract_sequences.py \
    --bioassembly_dir ./protenix_prepared/rna_bioassembly \
    --output_json ./aido_embeddings/sequences.json
```

**Key design decisions:**
- Split into a separate script from embedding generation because bioassembly pkl.gz files contain pickled protenix/rdkit objects that require the `protenix` conda environment, while AIDO models require the `aido_env` environment
- Classifies entities via `entity_poly_type`:
  - `"polyribonucleotide"` → RNA
  - `"polydeoxyribonucleotide"` → DNA
  - Hybrids (e.g., `"polydeoxyribonucleotide/polyribonucleotide hybrid"`) → treated as RNA
- Outputs JSON with separate `"rna"` and `"dna"` arrays, each containing `{pdb_id, entity_id, part_id, sequence, poly_type}`
- Skips sequences longer than 10,000 nt by default

**Results:**
| Metric | Value |
|--------|-------|
| Total bioassembly files | 6,479 |
| Corrupted/skipped files | 2 (8ucw, 9ash) |
| RNA entities extracted | 12,725 |
| Unique RNA sequences | 4,842 |
| DNA entities extracted | 1,267 |
| Unique DNA sequences | 755 |

---

### 1.2 AIDO Embedding Generation Script

**File:** `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/generate_aido_embeddings.py`

**Purpose:** Step 2 — Generate per-nucleotide AIDO embeddings from pre-extracted sequences.

**Environment:** `aido_env` conda env (has ModelGenerator/AIDO models)

**Usage:**
```bash
conda activate /inspire/ssd/project/sais-bio/public/ash_proj/conda/envs/aido_env

# RNA embeddings (AIDO.RNA-1.6B, 2048-dim)
python3 generate_aido_embeddings.py \
    --sequences_json ./aido_embeddings/sequences.json \
    --output_dir ./aido_embeddings \
    --mol_type rna \
    --device cuda:0

# DNA embeddings (AIDO.DNA-300M, 1024-dim)
python3 generate_aido_embeddings.py \
    --sequences_json ./aido_embeddings/sequences.json \
    --output_dir ./aido_embeddings \
    --mol_type dna \
    --device cuda:0
```

**Key features:**
- **Sliding window for long sequences:** Window size 1000, overlap 200, average pooling for overlapping regions
- **Sequence caching:** Identical sequences (across different PDB entities) reuse cached embeddings — avoids redundant GPU computation
- **Special token filtering:** Uses `attention_mask` and `special_tokens_mask` from ModelGenerator output to extract only real nucleotide embeddings (strips CLS, SEP, PAD tokens)
- **Resume support:** Skips already-generated `.pt` files, so the script can be interrupted and resumed

**Output format:**
```
aido_embeddings/
├── rna/
│   ├── {pdb_id}_{entity_id}/
│   │   └── {pdb_id}_{entity_id}.pt    # [seq_len, 2048] tensor
│   └── rna_sequences.csv               # seq → file path mapping
├── dna/
│   ├── {pdb_id}_{entity_id}/
│   │   └── {pdb_id}_{entity_id}.pt    # [seq_len, 1024] tensor
│   └── dna_sequences.csv               # seq → file path mapping
└── sequences.json                       # intermediate sequence data
```

**Results:**
| Metric | RNA | DNA |
|--------|-----|-----|
| Backbone | aido_rna_1b600m | aido_dna_300m |
| Embedding dim | 2048 | 1024 |
| Total entities | 12,725 | 1,267 |
| Unique computed | 4,842 | 755 |
| Errors | 0 | 0 |

---

### 1.3 AIDO Fine-tuning Training Script

**File:** `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix_aido_finetune.sh`

**Purpose:** Main training script for Protenix fine-tuning with AIDO RNA+DNA embeddings.

**Key configuration:**
```bash
--rnalm.enable true
--rnalm.model_name "aido"
--rnalm.embedding_dim 2048
--rnalm.injection_mode "diffusion"
--rnalm.embedding_dir "${RNA_EMBEDDING_DIR}"
--rnalm.sequence_fpath "${RNA_SEQUENCE_FPATH}"
--rnalm.dna_embedding_dir "${DNA_EMBEDDING_DIR}"      # auto-detected
--rnalm.dna_sequence_fpath "${DNA_SEQUENCE_FPATH}"     # auto-detected
```

**Design:** All training hyperparameters are identical to `protenix_rna_llm_input_inject.sh`. The script only changes:
- Embedding model from RiNALMo to AIDO
- Embedding dimension from 1280 to 2048
- Addition of DNA embedding paths
- Auto-detection of DNA embedding availability (graceful fallback to RNA-only)

---

### 1.4 Pipeline Test Script

**File:** `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/test_aido_pipeline.sh`

**Purpose:** Quick 2-step training + evaluation to verify the full AIDO pipeline works end-to-end.

**Configuration:** Same as `protenix_aido_finetune.sh` but with:
- `--max_steps 2` (instead of full training)
- `--use_wandb false`
- `--data.num_dl_workers 0` (for debugging)
- Output to `output/aido_test/`

---

## 2. Files Modified

### 2.1 RiNALMo Featurizer

**File:** `protenix/data/rnalm/rnalm_featurizer.py`

**Changes:** Extended `RiNALMoFeaturizer` class to support DNA embeddings alongside RNA.

| Change | Description |
|--------|-------------|
| Constructor | Added `dna_embedding_dir`, `dna_sequence_fpath` parameters |
| `load_dna_embedding()` | New method: loads DNA `.pt` files, zero-pads from 1024→2048 dim |
| `_fill_entities()` | New helper: unified entity-filling logic for both RNA and DNA |
| `__call__()` | Extended to identify DNA entities (via `chain_mol_type == "dna"`) and fill DNA embeddings |

**DNA dimension handling:**
```python
def load_dna_embedding(self, sequence: str) -> torch.Tensor:
    x = torch.load(os.path.join(self.dna_embedding_dir, self.dna_seq_to_filename[sequence]))
    # Zero-pad if DNA embedding dim < RNA embedding dim (1024 → 2048)
    if x.size(1) < self.embedding_dim:
        pad = torch.zeros(x.size(0), self.embedding_dim - x.size(1))
        x = torch.cat([x, pad], dim=1)
    return x
```

**Rationale for zero-padding:** Both RNA and DNA embeddings pass through the same projection layer (`rnalm_projection: 2048 → 384`). Zero-padding the DNA embeddings to 2048 ensures the first 1024 dimensions carry the DNA signal while the remaining 1024 are zeros — the projection layer learns to use only the relevant dimensions.

---

### 2.2 Configuration

**File:** `configs/configs_base.py`

**Changes:** Added fields to the `rnalm` config section:

| Field | Default | Description |
|-------|---------|-------------|
| `model_name` | `"rinalmo"` | Model identifier: `"rinalmo"` or `"aido"` |
| `dna_embedding_dir` | `""` | Directory containing pre-computed DNA `.pt` files |
| `dna_sequence_fpath` | `""` | CSV mapping DNA sequences to file paths |

---

### 2.3 Dataset Pipeline

**File:** `protenix/data/pipeline/dataset.py`

**Changes:** Updated `get_rnalm_featurizer()` to pass DNA embedding paths from config to `RiNALMoFeaturizer`:

```python
dna_embedding_dir = rnalm_info.get("dna_embedding_dir", "")
dna_sequence_fpath = rnalm_info.get("dna_sequence_fpath", "")
# ...
self.rnalm_featurizer = RiNALMoFeaturizer(
    embedding_dir=rnalm_embedding_dir,
    sequence_fpath=rnalm_sequence_fpath,
    embedding_dim=rnalm_info.get("embedding_dim", 1280),
    error_dir=error_dir,
    dna_embedding_dir=dna_embedding_dir,
    dna_sequence_fpath=dna_sequence_fpath,
)
```

---

### 2.4 Inference Dataloader

**File:** `protenix/data/inference/infer_dataloader.py`

**Changes:** Updated `RiNALMoFeaturizer` instantiation in `InferenceDataset.__init__()` to pass DNA embedding paths from config, mirroring the training pipeline changes.

---

## 3. Files NOT Modified (Pre-existing Support)

The following files already had the infrastructure to support configurable embedding dimensions and injection modes. No changes were needed:

| File | Component | Why No Changes Needed |
|------|-----------|----------------------|
| `protenix/model/protenix.py` | `_get_s_rnalm()`, `rnalm_projection` | Already reads `embedding_dim` from config; projection layer `LinearNoBias(embedding_dim → 384)` works for any input dimension |
| `protenix/model/modules/embedders.py` | `InputFeatureEmbedder`, `linear_rnalm` | Already parameterized by `embedding_dim` from config |
| `protenix/model/modules/diffusion.py` | `DiffusionConditioning` | RNA LM fusion via add/concat already implemented |

**Model architecture flow:**
```
AIDO RNA/DNA embeddings [N_token, 2048]
    → rnalm_projection (LinearNoBias, 2048 → 384, zero-initialized)
    → s_rnalm [N_token, 384]
    → Fused into DiffusionConditioning: s_trunk + s_rnalm
```

---

## 4. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Embedding Generation (Offline)                │
│                                                                 │
│  Step 1: extract_sequences.py (protenix env)                    │
│    bioassembly pkl.gz → sequences.json (RNA + DNA sequences)    │
│                                                                 │
│  Step 2: generate_aido_embeddings.py (aido_env)                 │
│    sequences.json → .pt files (per-nucleotide embeddings)       │
│    - RNA: AIDO.RNA-1.6B → [seq_len, 2048]                      │
│    - DNA: AIDO.DNA-300M → [seq_len, 1024]                      │
│    - Long sequences: sliding window (1000) + overlap (200)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline (GPU)                       │
│                                                                 │
│  RiNALMoFeaturizer.__call__()                                   │
│    1. Identify RNA/DNA entities from cropped token array         │
│    2. Load pre-computed .pt embeddings for each entity           │
│    3. DNA: zero-pad 1024 → 2048 to match RNA dim                │
│    4. Place into [N_token, 2048] tensor (zeros for non-RNA/DNA)  │
│                              ↓                                   │
│  Protenix Model                                                  │
│    5. rnalm_projection: [N_token, 2048] → [N_token, 384]        │
│       (zero-initialized LinearNoBias)                            │
│    6. Optional gating (scalar/token/dual)                        │
│    7. Inject at DiffusionConditioning: s_trunk += s_rnalm        │
│       OR at InputFeatureEmbedder: s_inputs += projected_rnalm    │
│       OR both                                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Pipeline Verification

A 2-step training + evaluation test was run using `test_aido_pipeline.sh` to verify the full pipeline.

### Training Results (2 steps)
| Metric | Step 0 | Step 1 |
|--------|--------|--------|
| loss | 1.460 | 2.776 |
| smooth_lddt_loss | 0.241 | 0.314 |
| mse_loss | 0.108 | 0.359 |
| rna_mse_loss | 0.632 | 2.169 |
| bond_loss | 0.0 | 0.0 |

### Verification Checklist
- [x] RNA embeddings loaded correctly (2048-dim, 12,725 entities)
- [x] DNA embeddings loaded correctly (1024-dim → zero-padded to 2048, 1,267 entities)
- [x] Both RNA and DNA featurizers initialized without errors
- [x] Forward pass completes with mixed RNA/DNA/protein structures
- [x] Loss computed and backpropagated successfully
- [x] Checkpoints saved (step 0 and step 1)
- [x] Evaluation loop started on 123 validation samples

---

## 6. Data Locations

| Data | Path |
|------|------|
| Bioassembly files | `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/protenix_prepared/rna_bioassembly/` |
| Extracted sequences | `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/aido_embeddings/sequences.json` |
| RNA embeddings | `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/aido_embeddings/rna/` |
| RNA sequences CSV | `.../aido_embeddings/rna/rna_sequences.csv` |
| DNA embeddings | `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/aido_embeddings/dna/` |
| DNA sequences CSV | `.../aido_embeddings/dna/dna_sequences.csv` |
| Protenix code | `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/` |
| Training script | `.../Protenix/protenix_aido_finetune.sh` |
| Test script | `.../Protenix/test_aido_pipeline.sh` |
| Test output | `.../Protenix/output/aido_test/` |

---

## 7. Environment Dependencies

| Task | Conda Environment | Key Dependencies |
|------|-------------------|------------------|
| Sequence extraction | `protenix` | rdkit, protenix, biotite |
| Embedding generation | `aido_env` | modelgenerator, torch, AIDO models |
| Training/fine-tuning | `protenix` | protenix, torch, cuequivariance |

---

## 8. Summary of Changes

### Created (4 files)
1. `data/.../extract_sequences.py` — Extract RNA/DNA sequences from bioassembly data
2. `data/.../generate_aido_embeddings.py` — Generate AIDO per-nucleotide embeddings
3. `Protenix/protenix_aido_finetune.sh` — Full fine-tuning training script
4. `Protenix/test_aido_pipeline.sh` — Quick pipeline verification script

### Modified (4 files)
1. `protenix/data/rnalm/rnalm_featurizer.py` — Added DNA embedding support + zero-padding
2. `configs/configs_base.py` — Added `model_name`, `dna_embedding_dir`, `dna_sequence_fpath` fields
3. `protenix/data/pipeline/dataset.py` — Pass DNA paths to RiNALMoFeaturizer
4. `protenix/data/inference/infer_dataloader.py` — Pass DNA paths for inference mode

### Unchanged (model architecture)
- `protenix/model/protenix.py` — Already parameterized by `embedding_dim`
- `protenix/model/modules/embedders.py` — Already parameterized
- `protenix/model/modules/diffusion.py` — Already parameterized
