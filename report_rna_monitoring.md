# RNA Monitoring Metrics: Implementation Report

**Date**: 2026-03-08
**Purpose**: Add RNA-specific metrics to wandb for monitoring RNA fine-tuning effectiveness
**Constraint**: Backpropagation loss is completely unchanged — only display/logging is modified

---

## 1. Summary

Added RNA-specific monitoring metrics to both training and validation logging:

| Metric | Where | WandB Key | Description |
|--------|-------|-----------|-------------|
| RNA MSE Loss | Train + Val | `train/mse_loss/rna_mse_loss.avg` | Average per-atom squared error for RNA atoms only |
| RNA MSE Loss | Val | `{test_name}/mse_loss/rna_mse_loss.avg` | Same, during validation |
| RNA LDDT Mean | Val | `{test_name}/{ema_prefix}rna_lddt/mean.avg` | Mean LDDT computed over RNA atom pairs only |
| RNA LDDT Best | Val | `{test_name}/{ema_prefix}rna_lddt/best.avg` | Best sample LDDT over RNA atom pairs only |

**Backprop loss**: UNCHANGED. The `cum_loss` used for `.backward()` is identical.

---

## 2. Files Modified

### 2.1 `protenix/model/loss.py` — MSELoss.forward() (line ~1209-1226)

**Change**: After computing the standard MSE loss (unchanged), added RNA-only MSE computation inside `torch.no_grad()` and returned it as metrics.

**Before:**
```python
loss = loss_reduction(weighted_align_mse_loss, method=self.reduction)
return loss
```

**After:**
```python
loss = loss_reduction(weighted_align_mse_loss, method=self.reduction)

# RNA-only MSE for monitoring (does NOT affect backprop)
with torch.no_grad():
    rna_coord_mask = is_rna * coordinate_mask
    n_rna_atoms = rna_coord_mask.sum(dim=-1)
    if n_rna_atoms.sum() > 0:
        rna_se_per_sample = (per_atom_se * rna_coord_mask).sum(dim=-1) / (
            n_rna_atoms.unsqueeze(-1) + self.eps
            if n_rna_atoms.dim() > 0
            else n_rna_atoms + self.eps
        )
        rna_mse = loss_reduction(
            rna_se_per_sample.mean(dim=-1), method=self.reduction
        )
    else:
        rna_mse = torch.tensor(0.0, device=loss.device, dtype=loss.dtype)

return (loss, {"rna_mse_loss": rna_mse})
```

**How it works:**
- `per_atom_se` (already computed, shape `[..., N_sample, N_atom]`) is the per-atom squared error after weighted rigid alignment
- `rna_coord_mask = is_rna * coordinate_mask` selects only RNA atoms with valid coordinates
- `rna_se_per_sample` averages squared error over RNA atoms only, per sample
- `rna_mse` reduces across samples and batch dimensions
- Wrapped in `torch.no_grad()` — no gradient computation, no impact on backward pass
- `aggregate_losses()` already handles tuple returns (line 1600): `(loss, metrics)` → `loss` used for backprop, `metrics` added to logging dict

**Key property**: The first element of the returned tuple is the **exact same loss tensor** as before. The gradient graph is identical.

### 2.2 `runner/train.py` — _evaluate() (line ~706-721)

**Change**: After computing standard LDDT metrics, added RNA-only LDDT computation.

**Added code (inside `with enable_amp:` and `@torch.no_grad()` context):**
```python
# RNA-only LDDT for monitoring
is_rna = batch["input_feature_dict"].get("is_rna", None)
if is_rna is not None and is_rna.sum() > 0:
    rna_lddt_mask = (
        batch["label_dict"]["lddt_mask"]
        * is_rna.float().unsqueeze(-1)
    )
    if rna_lddt_mask.sum() > 0:
        rna_lddt = self.lddt_metrics.lddt_base.forward(
            pred_coordinate=batch["pred_dict"]["coordinate"],
            true_coordinate=batch["label_dict"]["coordinate"],
            lddt_mask=rna_lddt_mask,
            chunk_size=self.lddt_metrics.chunk_size,
        )  # [N_sample]
        simple_metrics["rna_lddt/mean"] = rna_lddt.mean()
        simple_metrics["rna_lddt/best"] = rna_lddt.max()
```

**How it works:**
- `lddt_mask` has shape `[N_atom, N_atom]` — indicates which atom pairs to consider for LDDT
- `is_rna.float().unsqueeze(-1)` has shape `[N_atom, 1]` — broadcasts to mask rows
- `rna_lddt_mask = lddt_mask * is_rna[..., None]` keeps only pairs where the "center" atom `l` is RNA
- This measures how well RNA atoms' local neighborhoods are preserved in the prediction
- Uses the same `lddt_base.forward()` already used for complex LDDT — just with a restricted mask
- Only computed when RNA atoms exist in the sample (safe for mixed datasets)
- Already inside `@torch.no_grad()` context — no gradient impact

---

## 3. Data Flow

### Training
```
train_step()
  → get_loss() → ProtenixLoss.forward() → calculate_losses() → aggregate_losses()
      → MSELoss.forward() returns (loss, {"rna_mse_loss": rna_mse})
      → aggregate_losses unpacks: loss for backprop, metrics for logging
      → loss_dict["mse_loss/rna_mse_loss"] = rna_mse  ← NEW
  → loss.backward()  ← UNCHANGED (same loss tensor)
  → loss_dict items with "loss" in key → train_metric_wrapper
  → wandb.log(metrics)  ← includes "train/mse_loss/rna_mse_loss.avg"
```

### Validation
```
_evaluate()
  → get_loss() → loss_dict includes "mse_loss/rna_mse_loss"  ← NEW
  → get_metrics() → lddt_dict (complex LDDT, unchanged)
  → RNA LDDT computation  ← NEW
      → simple_metrics["rna_lddt/mean"] = rna_lddt.mean()
      → simple_metrics["rna_lddt/best"] = rna_lddt.max()
  → simple_metric_wrapper aggregates over all samples
  → wandb.log(metrics)  ← includes RNA MSE + RNA LDDT
```

---

## 4. WandB Metric Names

### Training (every `log_interval` steps)

| WandB Key | Description |
|-----------|-------------|
| `train/mse_loss/rna_mse_loss.avg` | RNA-only MSE (avg over log interval) |
| `train/mse_loss.avg` | Total MSE (unchanged) |
| `train/loss.avg` | Total loss (unchanged) |

### Validation (every `eval_interval` steps)

| WandB Key | Description |
|-----------|-------------|
| `{test}/mse_loss/rna_mse_loss.avg` | RNA-only MSE (avg over val set) |
| `{test}/rna_lddt/mean.avg` | RNA LDDT mean over samples (avg over val set) |
| `{test}/rna_lddt/best.avg` | RNA LDDT best sample (avg over val set) |
| `{test}/ema0.999_rna_lddt/mean.avg` | Same with EMA model |
| `{test}/ema0.999_rna_lddt/best.avg` | Same with EMA model |
| `{test}/ema0.999_mse_loss/rna_mse_loss.avg` | RNA MSE with EMA model |

---

## 5. What to Look for in WandB

### Training Curves
- **`train/mse_loss/rna_mse_loss.avg`**: Should decrease over training if RNA fine-tuning is effective
- Compare with **`train/mse_loss.avg`** (total MSE) to see if RNA-specific improvement diverges from overall

### Validation Curves
- **`rna_lddt/mean.avg`**: Higher is better. Measures structural accuracy for RNA atoms specifically
- **`rna_lddt/best.avg`**: Best sample LDDT for RNA — shows the model's potential
- Compare **`rna_lddt/mean.avg`** vs **`lddt/complex/mean`** to see if RNA is improving faster/slower than the complex overall

### Key Comparisons
1. RNA MSE vs Total MSE trend divergence → RNA-specific learning signal
2. RNA LDDT vs Complex LDDT → whether RNA accuracy tracks overall accuracy
3. EMA vs non-EMA RNA metrics → EMA stabilization effect on RNA predictions

---

## 6. Backward Compatibility

- When `is_rna` is all-zero (protein-only samples): RNA metrics are simply not added to the aggregator for that sample
- RNA MSE returns 0.0 when no RNA atoms exist, and `aggregate_losses` still works correctly
- The loss tensor used for backprop is the exact same object — changing return from `loss` to `(loss, metrics)` is handled by the existing `isinstance(loss_outputs, tuple)` check in `aggregate_losses`
- No new config flags needed — RNA metrics are always computed when RNA atoms are present

---

## 7. Summary of All Changes

| File | Lines Changed | Description |
|------|---------------|-------------|
| `protenix/model/loss.py` | +17 lines (line 1209-1226) | RNA MSE in MSELoss.forward(), returned as metrics tuple |
| `runner/train.py` | +14 lines (line 706-721) | RNA LDDT in _evaluate(), added to simple_metrics |

**Total**: ~31 lines added. Zero lines of existing logic modified. Backprop is identical.

---

**Report generated**: 2026-03-08
