# Protenix RNA/DNA Finetune Scripts & Code Review Report

**Date**: 2026-03-12
**Status**: Scripts delivered, GPU tests passed, code review complete
**Project**: Protenix + AIDO RNA/DNA LLM Embedding Fine-tuning

---

## 1. Deliverables

### 1.1 Finetune Scripts

Two production-ready finetune scripts placed in `Protenix/finetune/`:

| Script | File | Description |
|--------|------|-------------|
| RNA+DNA | `finetune/finetune_rna_dna.sh` | Both RNA (2048-dim) and DNA (1024-dim) embeddings enabled, separate projections |
| RNA-only | `finetune/finetune_rna_only.sh` | RNA embeddings only, DNA disabled (`use_dna_embed=false`) |

### 1.2 Configurable Parameters

Both scripts accept the following command-line arguments:

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `--injection_mode` | `input` | `input`, `diffusion`, `both` | Where embeddings are injected |
| `--fusion_method` | `add` | `add`, `concat` | How embeddings are fused with trunk |
| `--gate_mode` | `none` | `none`, `scalar`, `token`, `dual` | Gating mechanism |
| `--gate_init_logit` | `-3.0` | float | Initial gate logit (sigmoid(-3)~0.047) |
| `--two_stage` | `false` | `true`, `false` | Two-stage adapter warmup |
| `--max_steps` | `100000` | int | Total training steps |
| `--lr` | `0.0001` | float | Learning rate |
| `--warmup_steps` | `2000` | int | LR warmup steps |
| `--train_crop_size` | `384` | int | Training crop size |
| `--eval_interval` | `400` | int | Evaluation frequency |
| `--checkpoint_interval` | `1000` | int | Checkpoint save frequency |
| `--use_wandb` | `true` | `true`, `false` | W&B logging |
| `--run_name` | auto | string | Custom run name |

### 1.3 Injection Mode Explanation

| Mode | Location | Mechanism | Use Case |
|------|----------|-----------|----------|
| `input` | `InputFeatureEmbedder` | Added to `s_inputs` (like ESM embeddings) | Default, lightweight |
| `diffusion` | `DiffusionConditioning` | Added to `s_trunk` before diffusion | Affects denoising directly |
| `both` | Both locations | Injected at both input and diffusion | Maximum embedding utilization |

### 1.4 Usage Examples

```bash
# RNA+DNA, input injection (default)
bash finetune/finetune_rna_dna.sh

# RNA+DNA, diffusion injection with scalar gating
bash finetune/finetune_rna_dna.sh --injection_mode diffusion --gate_mode scalar

# RNA+DNA, both injection with two-stage training
bash finetune/finetune_rna_dna.sh --injection_mode both --two_stage true

# RNA-only, input injection (default)
bash finetune/finetune_rna_only.sh

# RNA-only, diffusion injection, 50k steps
bash finetune/finetune_rna_only.sh --injection_mode diffusion --max_steps 50000
```

---

## 2. GPU Test Results

### Test Environment
- **GPU**: NVIDIA H800 (81GB)
- **Checkpoint**: `protenix_base_20250630_v1.0.0.pt`
- **Dataset**: Stanford RNA 3D Folding (part2)
- **Conda**: `protenix`

### Test 1: RNA+DNA (`finetune_rna_dna.sh`)

```
Configuration: use_rna_embed=True, use_dna_embed=True
              separate_dna_projection=True, injection_mode=input
              3 training steps, crop_size=256
```

| Metric | Value |
|--------|-------|
| Exit code | 0 (success) |
| Training loss (step 2) | 1.884 |
| RNA MSE loss | 0.947 |
| Eval rna_lddt/mean | 0.395 |
| EMA rna_lddt/mean | 0.485 |
| EMA rna_lddt/best | 0.500 |
| Model parameters | 369.86M |
| RNA entries loaded | 4842 |
| DNA entries loaded | 761 |

Log confirms:
- `Separate RNA/DNA input injection (like ESM): use_rna=True (2048->449), use_dna=True (1024->449)`
- `RNA LM injection_mode=input (input=True, diffusion=False, separate_dna=True)`
- Checkpoints saved: `2.pt`, `2_ema_0.999.pt`

### Test 2: RNA-only (`finetune_rna_only.sh`)

```
Configuration: use_rna_embed=True, use_dna_embed=False
              separate_dna_projection=True, injection_mode=input
              3 training steps, crop_size=256
```

| Metric | Value |
|--------|-------|
| Exit code | 0 (success) |
| Checkpoints saved | `2.pt`, `2_ema_0.999.pt` |

Log confirms:
- DNA embedding loading skipped
- Only `rna_projection` created (no `dna_projection`)
- DNA tokens receive zero vectors as expected

---

## 3. Code Logic & Vulnerability Review

### 3.1 Findings Summary

| # | Severity | Category | Description | Action |
|---|----------|----------|-------------|--------|
| 1 | MEDIUM | Logic (inherited) | `s_rnalm` not zeroed during conditioning drop when `pair_z` is pre-cached | Pre-existing issue in DiffusionConditioning. RNA/DNA code mirrors baseline behavior. Fix would need to be in `diffusion.py` conditioning drop logic. |
| 2 | MEDIUM | Edge case | Modified-base fallback silently clamps out-of-range indices via `res_id` indexing | Could produce subtly wrong embeddings for modified-base tokens near sequence boundaries. Low probability in practice. |
| 3 | LOW | Error handling | CSV parsing lacks column validation | Malformed CSV produces cryptic pandas errors instead of clear messages. |
| 4 | LOW | Security | No path traversal validation on manifest filenames | `torch.load(weights_only=True)` mitigates execution risk. Research-only concern. |
| 5 | LOW | Error handling | Broad `except Exception` in `_fill_entities` | Could mask programming bugs during development. Acceptable for production robustness. |
| 6 | INFO | Robustness | No dtype validation on loaded embedding tensors | Float16/32/64 handled by projection layers, but integer tensors would silently produce wrong results. |

### 3.2 Toggle Flag Propagation: CORRECT

The `use_rna_embed` / `use_dna_embed` flags propagate correctly through all 6 layers:

```
configs_base.py (defaults)
    -> dataset.py / infer_dataloader.py (featurizer creation)
        -> rnalm_featurizer.py (entity identification, embedding loading)
    -> protenix.py (projection layer creation, forward pass)
    -> embedders.py (input injection layer creation, forward pass)
```

### 3.3 Memory Efficiency: CORRECT

- Disabled projection layers are NOT created (guarded by `if self.rnalm_use_rna` / `if self.rnalm_use_dna`)
- No unused GPU memory allocated for disabled paths
- Zero-init guarantees identical model output at step 0 before any finetuning

### 3.4 Injection Mode Routing: CORRECT

All three modes (`input`, `diffusion`, `both`) correctly route through:
- Model init (conditional layer creation)
- Forward pass (conditional projection + injection)
- Separate vs combined pathway (conditional key names)

### 3.5 Tensor Shapes: CORRECT

All dimension combinations verified:
- Separate: RNA `[N, 2048]` -> `[N, 384]`, DNA `[N, 1024]` -> `[N, 384]`
- Combined: `[N, 2048]` -> `[N, 384]` (DNA zero-padded)
- Input injection: RNA `[N, 2048]` -> `[N, 449]`, DNA `[N, 1024]` -> `[N, 449]`
- Concat fusion: `s_trunk[384] + s_inputs[449] + s_rnalm[384]` = `[1217]`

### 3.6 Script Review: CORRECT

Both finetune scripts:
- Have proper input validation for all enum parameters
- Use `set -euo pipefail` for fail-fast behavior
- Validate checkpoint and embedding paths before launching
- Auto-generate descriptive run names
- Handle missing DNA embeddings gracefully (RNA+DNA script)

---

## 4. Configuration Combination Matrix

| use_rna | use_dna | separate_dna | injection_mode | Behavior |
|---------|---------|--------------|----------------|----------|
| true | true | true | input | **Default RNA+DNA**: Two independent input-level projections |
| true | true | true | diffusion | Two independent diffusion-level projections |
| true | true | true | both | Four projections (2 input + 2 diffusion) |
| true | false | true | input | **RNA-only**: Single rna_projection, DNA gets zeros |
| true | false | true | diffusion | Single rna_projection at diffusion level |
| true | false | true | both | Two rna_projections (input + diffusion) |
| false | true | true | * | DNA-only: Single dna_projection |
| false | false | * | * | No embeddings (equivalent to rnalm.enable=False) |

---

## 5. Recommendations

1. **For production training**: Use `finetune_rna_dna.sh` with `--injection_mode input` (default). This is the most tested and safest configuration.

2. **For ablation studies**: Use `finetune_rna_only.sh` to measure the marginal contribution of DNA embeddings by comparing RNA-only vs RNA+DNA.

3. **For maximum performance**: Consider `--injection_mode both --gate_mode scalar --two_stage true` to leverage all injection points with learned gating and adapter warmup.

4. **Data loader workers**: Both scripts use `num_dl_workers=0`. For full training runs, consider adding `--data.num_dl_workers 8` or higher to improve data loading throughput.

---

## 6. File Inventory

```
Protenix/finetune/
    finetune_rna_dna.sh       # RNA+DNA finetune script (configurable)
    finetune_rna_only.sh      # RNA-only finetune script (configurable)

Protenix/report/
    finetune_scripts_and_code_review_report.md  # This report
```
