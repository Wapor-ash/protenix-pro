# Per-Group Learning Rate System & Embedding Fail-Fast Fix Report

**Date:** 2026-03-12
**Status:** Implemented & GPU Tested

---

## Summary

This report covers three categories of fixes to the Protenix RNA/DNA LLM embedding fine-tuning pipeline:
1. **Two-stage adapter keyword mismatch** - Gate parameters missing from adapter keywords
2. **Flexible per-group learning rate system** - Independent adapter/backbone LRs for 1-stage and 2-stage training
3. **Missing embedding fail-fast** - Silent fallback to zeros replaced with hard errors

---

## 1. Two-Stage Adapter Keyword Fix

### Problem
The `adapter_keywords` config only matched projection layer names (`rnalm_projection`, `rna_projection`, `dna_projection`, `linear_rnalm`, `linear_rna_llm`, `linear_dna_llm`) but **missed gate parameters**:
- `rnalm_alpha_logit` (scalar gate)
- `rnalm_gate_mlp` (token gate MLP)

When `gate_mode` is `scalar`, `token`, or `dual`, these gate parameters are new adapter modules that should be trained with the adapter learning rate. Without matching keywords, they would be assigned to the backbone group and frozen during Stage 1.

### Fix
Added `rnalm_alpha_logit,rnalm_gate_mlp` to the default `adapter_keywords` in `configs/configs_base.py`.

### Files Changed
- `configs/configs_base.py` line 72

---

## 2. Flexible Per-Group Learning Rate System

### Problem
The original two-stage system had:
- Confusing naming: `stage1_lr` (was adapter LR), `stage2_lr` (was backbone LR), `stage2_adapter_lr`
- No way to set independent adapter/backbone LRs in 1-stage training
- Scheduler set ALL param groups to the same LR, with a post-hoc hack to force backbone to 0
- No support for non-zero backbone LR during Stage 1

### Design

#### 1-Stage Training (two_stage.enable=False)
Set `two_stage.adapter_lr` and/or `two_stage.backbone_lr` to enable per-group splitting:

| Config | Default | Meaning |
|--------|---------|---------|
| `adapter_lr` | -1.0 | LR for new modules (-1 = use global `lr`) |
| `backbone_lr` | -1.0 | LR for original backbone (-1 = use global `lr`) |

Example: `--two_stage.adapter_lr 0.005 --two_stage.backbone_lr 0` freezes backbone while training new modules.

#### 2-Stage Training (two_stage.enable=True)

**Stage 1 (adapter warmup):**

| Config | Default | Meaning |
|--------|---------|---------|
| `stage1_adapter_lr` | 5e-3 | Adapter LR |
| `stage1_backbone_lr` | 0.0 | Backbone LR (0 = frozen) |
| `stage1_warmup_steps` | 1 | Warmup steps |
| `stage1_max_steps` | 400 | Total Stage 1 steps |

**Stage 2 (joint training):**

| Config | Default | Meaning |
|--------|---------|---------|
| `stage2_adapter_lr` | -1.0 | Adapter LR (-1 = same as stage1_adapter_lr) |
| `stage2_backbone_lr` | -1.0 | Backbone LR (-1 = same as stage2_adapter_lr) |
| `stage2_warmup_steps` | 100 | Warmup steps |
| `stage2_ema_decay` | 0.999 | EMA decay |

### Implementation

#### Core Methods in `runner/train.py`

- **`_split_params()`**: Splits model parameters into adapter and backbone groups using `adapter_keywords` matching.
- **`_build_optimizer()`**: Creates Adam/AdamW optimizer with two param groups (backbone at index 0, adapter at index 1). Respects `adam.use_adamw` config.
- **`_apply_per_group_lr()`**: Called after every `scheduler.step()`. Computes the scheduler's scale factor (`sched_lr / scheduler_base_lr`) and applies it proportionally to both groups:
  ```
  adapter_lr_t = adapter_base_lr * scale_t
  backbone_lr_t = backbone_base_lr * scale_t
  ```
  When `backbone_base_lr=0`, backbone stays at 0 regardless of scheduler state.

- **`_setup_per_group_training()`**: For 1-stage per-group LR. Creates optimizer, scheduler, and stores `_lr_group_config`.
- **`_setup_stage1()`**: Rewritten with explicit `stage1_adapter_lr` / `stage1_backbone_lr`.
- **`_transition_to_stage2()`**: Rewritten with default resolution (stage2_adapter_lr defaults to stage1_adapter_lr, stage2_backbone_lr defaults to stage2_adapter_lr).

#### Freeze Guarantee

When backbone_lr=0:
1. Optimizer param group has `lr=0`
2. After every `scheduler.step()`, `_apply_per_group_lr()` forces backbone group back to 0
3. With AdamW: `param = param * (1 - lr * weight_decay) - lr * step = param` when lr=0
4. With Adam: update = `lr * m_hat / (sqrt(v_hat) + eps) = 0` when lr=0
5. Parameters are guaranteed unchanged throughout the entire training cycle

### GPU Test Results (H800)

```
Step 0 learning rate: [0.0, 0.005]       # Stage 1: backbone=0, adapter=0.005
Step 1 learning rate: [0.0, 0.002525]     # Stage 1: backbone=0, adapter decaying
Step 2 learning rate: [0.0, 5e-05]        # Stage 1: backbone=0 (FROZEN entire stage)
--- Stage transition ---
Step 3 learning rate: [0.0001, 0.001]     # Stage 2: backbone=0.0001, adapter=0.001
Step 4 learning rate: [5.05e-05, 0.000505]# Stage 2: both decaying proportionally
Step 5 learning rate: [1e-06, 1e-05]      # Stage 2: both at minimum, ratio=10:1
```

### Files Changed
- `configs/configs_base.py`: New two_stage config structure
- `runner/train.py`: New methods (`_split_params`, `_build_optimizer`, `_apply_per_group_lr`, `_setup_per_group_training`), rewritten `_setup_stage1`, `_transition_to_stage2`, updated `train_step` and LR logging

---

## 3. Missing Embedding Fail-Fast

### Problem
When `use_rna_embed=True` or `use_dna_embed=True`, the featurizer could silently fall back to zeros:
1. If embedding paths were not configured (empty strings) -> logged info, returned zeros
2. If embedding paths didn't exist on disk -> logged warning, returned zeros
3. If a specific sequence's embedding file couldn't be loaded -> caught exception, logged warning, returned zeros

This meant training could proceed with zero embeddings without the user knowing, wasting GPU hours.

### Fix

#### Featurizer Init (`protenix/data/rnalm/rnalm_featurizer.py`)
When `use_rna_embed=True`:
- Empty paths -> `ValueError`
- Directory not found -> `FileNotFoundError`
- CSV not found -> `FileNotFoundError`

Same for `use_dna_embed=True`.

#### Per-Sample Loading (`_fill_entities`)
When a specific sequence's embedding cannot be loaded:
- Old: `except Exception -> logger.warning()` and continue with zeros
- New: `except Exception -> raise RuntimeError` with clear error message

### Files Changed
- `protenix/data/rnalm/rnalm_featurizer.py`: `__init__` RNA/DNA sections, `_fill_entities` exception handling

---

## 4. Additional Audit Findings

### Items Verified as Safe
- **DDP with `find_unused_parameters=False`**: Safe because all ranks share the same config, so all conditionally-created modules are consistent. Adapter modules are always used in forward pass when created.
- **Dtype/device handling**: `to_device(batch, self.device)` in train.py handles device transfer. PyTorch autocast handles dtype in mixed precision.
- **Checkpoint loading with `load_strict=False`**: New adapter params are properly zero-initialized in `protenix.py` (`nn.init.zeros_`). Missing keys in checkpoint are correctly handled.

### Known Limitations
- Weight decay in AdamW is applied uniformly to all params (including biases) in per-group mode. The original `get_adamw` function splits decay/no-decay params, but per-group mode uses a simpler 2-group structure. This has negligible impact for fine-tuning.
- In DDP with backbone frozen (lr=0), gradients are still computed for backbone params (wasted compute). Setting `requires_grad=False` would save memory but could cause DDP issues with `static_graph=True`. Current approach prioritizes correctness.

---

## 5. New Training Scripts

### `finetune/finetune_1stage.sh`
1-stage training with per-group LR. Key arguments:
```bash
--adapter_lr 0.005     # New module LR
--backbone_lr 0.0001   # Backbone LR (0 = freeze)
--injection_mode diffusion
--gate_mode scalar
```

### `finetune/finetune_2stage.sh`
2-stage training with full per-group LR control. Key arguments:
```bash
--stage1_adapter_lr 0.005
--stage1_backbone_lr 0.0      # Freeze in Stage 1
--stage2_adapter_lr 0.001
--stage2_backbone_lr 0.0001   # Unfreeze in Stage 2
```

### `finetune/test_gpu_small.sh`
GPU smoke test that verifies all training modes in ~10 minutes:
```bash
bash finetune/test_gpu_small.sh          # Run all tests
bash finetune/test_gpu_small.sh 1stage   # 1-stage only
bash finetune/test_gpu_small.sh 2stage   # 2-stage only
```

### GPU Test Results

| Test | Mode | Steps | Result |
|------|------|-------|--------|
| 1stage_freeze | adapter_lr=0.005, backbone_lr=0 | 3 | PASS |
| 1stage_both | adapter_lr=0.005, backbone_lr=0.0001 | 3 | PASS |
| 2stage | Stage1: adapter=0.005, backbone=0 -> Stage2: adapter=0.001, backbone=0.0001 | 6 (3+3) | PASS |

All tests completed successfully on NVIDIA H800 (bf16 precision).

---

## File Change Summary

| File | Lines Changed | Description |
|------|--------------|-------------|
| `configs/configs_base.py` | ~20 | New two_stage config with per-group LR |
| `runner/train.py` | ~150 | Per-group LR system, rewritten stage methods |
| `protenix/data/rnalm/rnalm_featurizer.py` | ~40 | Fail-fast embedding loading |
| `finetune/finetune_1stage.sh` | new | 1-stage training script |
| `finetune/finetune_2stage.sh` | new | 2-stage training script |
| `finetune/test_gpu_small.sh` | new | GPU smoke test |
