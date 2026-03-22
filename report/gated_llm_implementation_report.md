# Gated RNA LLM Embedding Implementation Report

## Overview

Implemented dual-gate mechanism for RNA LLM (RiNALMo) embedding injection into Protenix, following the recommendations from `protenix_rna_llm_gating_report.md`. The design treats RNA LLM embeddings as a **controllable perturbation** rather than a direct addition, preventing early-training distribution shifts that degrade the pretrained backbone.

## Gate Architecture (Method C: Dual Gate)

```
lm_delta = proj(lm_embed)                    # [N_token, 384]  (existing rnalm_projection)
g1 = sigmoid(alpha_logit)                     # scalar, global amplitude gate
g2 = sigmoid(gate_mlp(s_trunk.detach()))      # [N_token, 1], per-token confidence gate
s_rnalm = g1 * g2 * lm_delta                 # gated residual injection
```

### Gate Components

| Component | Parameters | Init | Purpose |
|-----------|-----------|------|---------|
| `rnalm_alpha_logit` | 1 scalar | -3.0 (sigmoid = 0.047) | Global injection amplitude |
| `rnalm_gate_mlp` | Linear(384,96) + ReLU + Linear(96,1) | output bias = -3.0 | Per-token confidence |
| `rnalm_projection` | LinearNoBias(1280, 384) | zeros | LLM embedding projection (existing) |

**Total new parameters**: ~37K (gate MLP) + 1 (alpha) = ~37K on top of existing 492K projection.

### Why Dual Gate

- **g1 (global)**: Controls overall injection strength. Starts near zero, lets the model gradually increase LLM influence across all tokens.
- **g2 (per-token)**: Learned from `s_trunk.detach()` (backbone output, stop-gradient). Decides which tokens should trust the LLM signal. The detach prevents gate gradients from flowing back into the backbone during Stage 1.

## Gate Modes

Four modes are available via `--rnalm.gate_mode`:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `none` | No gating, direct add (original behavior) | Backward compatibility |
| `scalar` | `g1 * lm_delta` | Simplest gate, verify if gating helps |
| `token` | `g2 * lm_delta` | Per-token control without global scaling |
| `dual` | `g1 * g2 * lm_delta` | **Recommended** - full control |

## 2-Stage Training Strategy

### Stage 1: Gate + Projection Warmup (steps 0-400)

- **Frozen**: Entire backbone (368.48M params, lr=0)
- **Trainable**: `rnalm_projection` + `rnalm_alpha_logit` + `rnalm_gate_mlp` (0.53M params)
- **LR**: 3e-3, cosine schedule, 50-step warmup
- **No EMA**
- **Purpose**: Calibrate LLM signal scale, teach gate to be conservative

### Stage 2: Joint Fine-tuning (steps 400-100000)

- **Trainable**: All parameters
- **Backbone LR**: 1e-4
- **Adapter LR**: 3e-4 (3x backbone, new `stage2_adapter_lr` option)
- **EMA**: 0.999 decay
- **Warmup**: 500 steps
- **Purpose**: Backbone adapts to the now-calibrated gated LLM signal

## Files Modified

### 1. `configs/configs_base.py`
- Added `rnalm.gate_mode` (default: `"none"`) and `rnalm.gate_init_logit` (default: `-3.0`)
- Added `two_stage.stage2_adapter_lr` (default: `-1.0`, meaning same as `stage2_lr`)

### 2. `protenix/model/protenix.py`
- Added gate layer initialization in `__init__` when `gate_mode != "none"`
- Modified `_get_s_rnalm()` to accept optional `s_trunk` argument and apply gating
- Updated both call sites (training + inference) to pass `s_trunk=s`

### 3. `runner/train.py`
- Modified `_transition_to_stage2()` to support separate adapter LR via `stage2_adapter_lr`

### 4. New: `rna_llm_gate_tune.sh`
- Training script with `--rnalm.gate_mode "dual"` and recommended hyperparameters

## Backward Compatibility

The original `rna_2stage_slow.sh` is **completely unaffected**:

1. It doesn't set `--rnalm.gate_mode` → defaults to `"none"` → no gate layers created
2. It uses `--two_stage.adapter_keywords "rnalm_projection"` → only matches projection
3. It doesn't set `--two_stage.stage2_adapter_lr` → defaults to `-1.0` → uses `stage2_lr` for both groups (same as before)

No existing code paths are changed when `gate_mode="none"`.

## GPU Test Results

Tested on NVIDIA H800 (80GB). Training launched successfully:

```
RNA LM (RiNALMo) embedding enabled: 1280 -> 384, fusion_method=add, gate_mode=dual
[Stage 1] Adapter warmup: backbone=368.48M (lr=0), adapter=0.53M (keyword='rnalm')
[Stage 1] Optimizer: Adam, backbone_lr=0.0, adapter_lr=0.003
[Stage 1] Scheduler: cosine, warmup=50, max_steps=400, lr=0.003
```

Training progressed through 40+ steps at ~8-10s/step without errors. Stage 1 is expected to complete in ~55 minutes (400 steps).

## Hyperparameter Recommendations

Based on the gating report, the following can be tuned:

| Parameter | Current | Range to Explore |
|-----------|---------|-----------------|
| `gate_init_logit` | -3.0 | -4.0 to -2.0 |
| `stage1_max_steps` | 400 | 200 to 600 |
| `stage1_lr` | 3e-3 | 2e-3 to 5e-3 |
| `stage2_lr` (backbone) | 1e-4 | 5e-5 to 2e-4 |
| `stage2_adapter_lr` | 3e-4 | 2e-4 to 5e-4 |

## How to Run

```bash
# Gated version (new)
bash rna_llm_gate_tune.sh

# Original version (unchanged)
bash rna_2stage_slow.sh
```

## Monitoring Gate Values

To inspect gate behavior during training, check wandb or add logging for:
- `sigmoid(model.rnalm_alpha_logit)` — global gate value (should start ~0.047, grow gradually)
- `model.rnalm_gate_mlp` output statistics — per-token gate distribution
