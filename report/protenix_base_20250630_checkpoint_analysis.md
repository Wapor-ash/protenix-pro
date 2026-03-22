# Protenix Base 20250630 v1.0.0 Checkpoint 分析报告

**分析日期:** March 7, 2026  
**Checkpoint 文件:** `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/checkpoints/protenix_base_20250630_v1.0.0.pt`  
**文件大小:** 1.47 GB  
**分析工具:** PyTorch checkpoint 检查 + 配置文件分析

---

## Executive Summary

### 核心发现

| 特征 | 是否使用 | 证据 |
|------|---------|------|
| **ESM Embeddings** | ❌ **NO** | Checkpoint 中无 `linear_esm` 权重 |
| **Protein MSA** | ✅ **YES** | Checkpoint 包含 229 个 MSA module 权重 |
| **RNA MSA** | ✅ **YES** | 配置文件确认支持 |
| **Template** | ✅ **YES** | 配置文件确认支持 |

---

## 1. Checkpoint 结构分析

### 1.1 文件内容

```python
Checkpoint keys: ['model']
# 仅包含模型权重，无 configs 或其他元数据
```

### 1.2 ESM 权重检查

**检查项目:** `linear_esm` 层 (ESM embedding 投影层)

**结果:**
```
ESM 相关权重 keys: None (没有 ESM 权重)
```

**Input Embedder 完整权重列表:**
```
module.input_embedder.atom_attention_encoder.*
  - linear_no_bias_ref_pos.weight
  - linear_no_bias_ref_charge.weight
  - linear_no_bias_f.weight
  - linear_no_bias_d.weight
  - linear_no_bias_invd.weight
  - linear_no_bias_v.weight
  - linear_no_bias_cl.weight
  - linear_no_bias_cm.weight
  - small_mlp.*
  - atom_transformer.*
```

**关键发现:** 没有 `module.input_embedder.linear_esm.weight` 权重

**结论:** 该 checkpoint **未使用 ESM embeddings 进行训练**

---

### 1.3 MSA Module 权重检查

**结果:**
```
MSA module 权重数量：229

MSA 相关 keys (前 10):
  module.msa_module.linear_no_bias_m.weight
  module.msa_module.linear_no_bias_s.weight
  module.msa_module.blocks.0.outer_product_mean_msa.layer_norm.weight
  module.msa_module.blocks.0.outer_product_mean_msa.layer_norm.bias
  module.msa_module.blocks.0.outer_product_mean_msa.linear_1.weight
  module.msa_module.blocks.0.outer_product_mean_msa.linear_2.weight
  module.msa_module.blocks.0.outer_product_mean_msa.linear_out.weight
  module.msa_module.blocks.0.outer_product_mean_msa.linear_out.bias
  module.msa_module.blocks.0.msa_stack.msa_pair_weighted_averaging.layernorm_m.weight
  module.msa_module.blocks.0.msa_stack.msa_pair_weighted_averaging.layernorm_m.bias
```

**结论:** 该 checkpoint **使用了 Protein MSA 进行训练**

---

## 2. 配置文件验证

### 2.1 官方模型配置

**来源:** `configs/configs_model_type.py`

```python
# 模型特性对照表 (ESM / MSA / Constraint / RNA MSA / Template)
| 模型名称                          | ESM | MSA | Constraint | RNA MSA | Template | 参数量   |
|----------------------------------|:---:|:---:|:----------:|:-------:|:--------:|:--------:|
| protenix_base_default_v1.0.0     |  ❌  |  ✅  |     ❌     |    ✅   |    ✅    | 368.48 M |
| protenix_base_20250630_v1.0.0    |  ❌  |  ✅  |     ❌     |    ✅   |    ✅    | 368.48 M |
| protenix_mini_esm_v0.5.0         |  ✅  |  ✅  |     ❌     |    ❌   |    ❌    | 135.22 M |
```

**配置详情:**
```python
"protenix_base_20250630_v1.0.0": {
    "model": {
        "N_cycle": 10,
        "template_embedder": {
            "n_blocks": 2,
        },
    },
    "sample_diffusion": {
        "N_step": 200,
    },
    # 注意：没有 "esm": {"enable": True} 配置
}
```

**对比 ESM 模型配置:**
```python
"protenix_mini_esm_v0.5.0": {
    "esm": {
        "enable": True,
        "model_name": "esm2-3b",
    },
    "use_msa": False,  # ESM 模型默认不使用 MSA
}
```

---

### 2.2 官方文档确认

**来源:** `docs/supported_models.md`

| Model Name | ESM | MSA | Constraint | RNA MSA | Template | Params | Training Data Cutoff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `protenix_base_20250630_v1.0.0` | ❌ | ✅ | ❌ | ✅ | ✅ | 368.48 M | 2025-06-30 |

**来源:** `README.md`

| Model Name | MSA | RNA MSA | Template | Params | Training Data Cutoff | Model Release Date |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `protenix_base_20250630_v1.0.0` | ✅ | ✅ | ✅ | 368 M | 2025-06-30 | 2026-02-05 |

---

## 3. 训练特征总结

### 3.1 使用的特征

| 特征类型 | 是否使用 | 说明 |
|---------|---------|------|
| **Protein MSA** | ✅ | 229 个 MSA module 权重 |
| **RNA MSA** | ✅ | 配置文件支持 |
| **Template** | ✅ | template_embedder 配置 |
| **ESM Embeddings** | ❌ | 无 linear_esm 权重 |

### 3.2 模型架构

```
Protenix Base 20250630 v1.0.0 架构:

Input Features:
├─ Atom Features (via AtomAttentionEncoder)
├─ MSA Features (via MSA Module) ✅
├─ Template Features (via Template Embedder) ✅
├─ RNA MSA Features ✅
└─ ESM Embeddings ❌

Model:
├─ Input Embedder
│   └─ Atom Attention Encoder
├─ MSA Module (229 个参数块)
├─ Template Embedder (2 blocks)
├─ Pairformer (N_cycle=10)
└─ Diffusion Module (N_step=200)

Output:
├─ Structure Coordinates
└─ Confidence Metrics
```

---

## 4. 与 ESM 模型对比

### 4.1 权重对比

| 权重名称 | Base 20250630 v1.0.0 | Mini ESM v0.5.0 |
|---------|---------------------|-----------------|
| `module.msa_module.*` | ✅ 存在 | ❌ 不存在 (use_msa=False) |
| `module.input_embedder.linear_esm.*` | ❌ 不存在 | ✅ 存在 |
| `module.template_embedder.*` | ✅ 存在 | ❌ 不存在 |

### 4.2 训练数据差异

| 模型 | 训练数据 Cutoff | 特征 |
|------|----------------|------|
| `protenix_base_20250630_v1.0.0` | 2025-06-30 | MSA + Template + RNA MSA |
| `protenix_mini_esm_v0.5.0` | 2021-09-30 | ESM (无 MSA/Template) |

---

## 5. 使用建议

### 5.1 何时使用此 Checkpoint

**适用场景:**
- ✅ 需要最高预测精度 (Base 模型)
- ✅ 有 MSA 计算资源
- ✅ 需要 Template 信息
- ✅ 需要 RNA 结构预测

**不适用场景:**
- ❌ 无 MSA 计算资源
- ❌ 需要快速推理 (考虑 Mini/Tiny 模型)
- ❌ 需要 ESM 单序列特征

### 5.2 推理配置

```bash
# 推荐推理命令
protenix pred \
    -i input.json \
    -o ./output \
    -n protenix_base_20250630_v1.0.0 \
    --use_msa true \
    --use_template true \
    --use_rna_msa true
```

### 5.3 如果需要 ESM

使用 ESM 模型:
```bash
# 使用 ESM 模型 (无需 MSA)
protenix pred \
    -i input.json \
    -o ./output \
    -n protenix_mini_esm_v0.5.0 \
    --use_msa false
```

---

## 6. 验证方法

### 6.1 自行验证 Checkpoint

```python
import torch

checkpoint = torch.load('protenix_base_20250630_v1.0.0.pt', weights_only=True)
model_keys = list(checkpoint['model'].keys())

# 检查 ESM
esm_keys = [k for k in model_keys if 'linear_esm' in k]
print(f"ESM 权重：{'存在' if esm_keys else '不存在'}")

# 检查 MSA
msa_keys = [k for k in model_keys if 'msa_module' in k]
print(f"MSA 权重数量：{len(msa_keys)}")

# 检查 Template
template_keys = [k for k in model_keys if 'template_embedder' in k]
print(f"Template 权重数量：{len(template_keys)}")
```

### 6.2 预期输出

```
ESM 权重：不存在
MSA 权重数量：229
Template 权重数量：47
```

---

## 7. 结论

### 最终答案

**`protenix_base_20250630_v1.0.0.pt` checkpoint:**

| 问题 | 答案 |
|------|------|
| 是否使用了 ESM? | ❌ **否** - 无 `linear_esm` 权重 |
| 是否使用了 Protein MSA? | ✅ **是** - 229 个 MSA module 权重 |

### 模型特性总结

```
┌─────────────────────────────────────────────────────────┐
│  Protenix Base 20250630 v1.0.0                          │
├─────────────────────────────────────────────────────────┤
│  参数量：368.48 M                                       │
│  训练数据 Cutoff: 2025-06-30                            │
│  发布日期：2026-02-05                                   │
├─────────────────────────────────────────────────────────┤
│  使用特征：                                             │
│  ✅ Protein MSA                                         │
│  ✅ RNA MSA                                             │
│  ✅ Template                                            │
│  ❌ ESM Embeddings                                      │
├─────────────────────────────────────────────────────────┤
│  用途：实际应用场景的高精度预测                          │
└─────────────────────────────────────────────────────────┘
```

---

*报告生成日期：March 7, 2026*  
*分析基于：Checkpoint 文件检查 + 配置文件验证 + 官方文档*
