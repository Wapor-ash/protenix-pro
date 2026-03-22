这份报告深度剖析了 **Protenix** (ByteDance) 与 **AlphaFold3** (DeepMind) 在模型架构层面对模板与 Pairwise 约束的注入机制，并提供了针对 RNA 二级结构（BPP 等）注入的**可复现补丁方案**。

---

## 1. 核心注入逻辑对照剖析

### A. Protenix：约束与模板的双重注入路径

在 Protenix 中，Pair 表征 $z$ 的生命周期始于初始化，并在每个 Recycle 循环中被模板更新。

#### (1) 初始注入点 (z_init)

* **代码指针**：`protenix/model/protenix.py` -> `get_pairformer_output`
* **数学形式**：

$$z_{init} = \text{Linear}(s_i) + \text{Linear}(s_j) + \text{RelPE}(i,j) + \text{Bond}(i,j) + \mathbf{z_{constraint}}$$


* **关键发现**：`z_constraint` 是由 `ConstraintEmbedder` 产生的。源码中已经预留了 `substructure` 接口，这正是注入 **RNA 二级结构 (BPP)** 的绝佳入口。

#### (2) 模板注入点 (Recycling)

* **代码指针**：`protenix/model/modules/pairformer.py` -> `TemplateEmbedder`
* **注入方式**：$z = z + \text{TemplateEmbedder}(z, \text{features})$
* **现状**：目前数据侧 `template_featurizer.py` 显式 `assert ctype == PROTEIN_CHAIN`。若要支持 RNA 模板，需修改数据预处理逻辑以计算核酸的 Pseudo-beta 和 Rigid frames。

---

### B. AlphaFold3：基于 Evoformer 的 additive 注入

AlphaFold3 的 `pair_activations` 采用类似的累加逻辑，但其“约束”接口相对封闭。

#### (1) 约束注入 (Bonds)

* **代码指针**：`evoformer.py` -> `_embed_bonds`
* **逻辑**：将 JSON 中的 `bondedAtomPairs` 转化为 $N \times N$ 的 Contact Matrix，通过 Linear 投影后 `pair = pair + bond_emb`。

#### (2) 模板注入 (TemplateEmbedding)

* **代码指针**：`template_modules.py`
* **逻辑**：构造 Distogram、Unit Vector 等特征，经过一个简易的 `PairFormerIteration` 后加回主干。
* **限制**：官方 `compute_features` 逻辑目前会跳过非蛋白链的模板提取。

---

## 2. 可复现补丁：为 Protenix 注入 RNA 二级结构 (BPP)

如果你希望将 RNA 二级结构（如 $N \times N$ 的配对概率矩阵 BPP）注入模型，推荐修改 Protenix 的 `substructure` 路径。

### 补丁 1：数据解析层 (JSON -> Tensor)

修改 `protenix/data/constraint/constraint_featurizer.py`，补齐 TODO 部分。

```python
# 补丁：将 BPP 矩阵 load 并存入 feature_dict
if substructure := constraint_param.get("structure", {}):
    if substructure.get("type") == "bpp":
        import numpy as np
        bpp = np.load(substructure["path"]) # [N, N]
        # 对称化与截断处理
        bpp = 0.5 * (bpp + bpp.T)
        bpp = np.clip(bpp, 0.0, 1.0).astype(np.float32)
        feature_dict["substructure"] = torch.from_numpy(bpp) # 形状 [N, N]

```

### 补丁 2：模型嵌入层 (标量 -> 向量)

修改 `protenix/model/modules/embedders.py` 中的 `ConstraintEmbedder`，使其兼容连续值输入。

```python
# 补丁：兼容 [N, N] 连续概率输入并投影
sub = constraint_feature_dict["substructure"]
if sub.dim() == 2:
    # 扩展为 [N, N, n_classes]，例如 n_classes=2 (paired/unpaired)
    sub = sub.unsqueeze(-1) 
    sub = torch.cat([sub, 1.0 - sub], dim=-1) 

z_substructure = self.substructure_z_embedder(sub)
z_constraint = z_substructure # 随后加到 z_init

```

---

## 3. 扩展步骤清单

### 若要支持 RNA/DNA 模板 (Protenix)

1. **数据侧**：修改 `InferenceTemplateFeaturizer`，移除对非蛋白链的 `assert`。
2. **几何侧**：在生成模板特征时，定义核酸的“Pseudo-beta”（通常选 C4' 或 C2）以计算 `template_distogram`。
3. **索引侧**：修改 `TemplateSourceManager`，确保其能检索并处理核酸 mmCIF 文件。

### 若要支持通用 Pairwise Bias (AlphaFold3)

1. **定义特征**：在 `features.py` 中新增 `PairwiseBias` 字段。
2. **注入 Evoformer**：在 `evoformer.py` 中模仿 `_embed_bonds` 创建一个新方法 `_embed_custom_constraints`。
3. **Zero-Init**：使用零初始化（Zero-init）线性层注入，确保在不加载新权重时模型输出保持不变。

---

## 4. 训练与测试建议

* **冻结训练 (Adapter Tuning)**：加载预训练权重时设置 `strict=False`。冻结 Trunk 部分（`pairformer_stack`），仅开放 `constraint_embedder` 和新增的线性层进行训练。
* **数值稳定性检查**：
* 断言 `z_constraint.shape == z_init.shape`。
* 在混合精度 (AMP) 下，确保 `z_constraint` 在相加前通过 `.to(z_init.dtype)` 完成对齐，防止隐式转型导致的精度损失。


* **基线验证**：输入全零的 BPP 矩阵，模型输出应与原始预训练模型完全一致。

**下一步建议：**
如果你已经准备好了 BPP 数据，需要我为你写一个 **训练脚本的配置文件 (Config YAML) 示例** 来启用这些新增的 `substructure` 通道吗？