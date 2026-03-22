这是一份根据你提供的逻辑整理后的 **Protenix Template 数据流向全解析**。

---

## 1. 核心逻辑综述

在 Protenix 中，Template（模板）信息的处理并不是在 Trunk 内部临时把一个小矩阵“贴”进去，而是遵循**“上游全量对齐，下游全局叠加”**的原则。

* **空间一致性**：进入 Trunk 的 Template 特征在第一维和第二维上已经是 $N_{token} \times N_{token}$，与全局 Pair Representation $z$ 完全对齐。
* **内容差异性**：虽然空间是全局的，但由于目前数据管线仅支持蛋白，非蛋白区域（RNA、Ligand）在输入时被填充为 `empty_template_features`，实质上是不携带有效几何信息的“空位”。

---

## 2. 完整数据流向图

### A. 上游：Template Featurizer (离线/预处理)

在 `template_featurizer.py` 中，系统按链（Chain）处理特征：

1. **蛋白链**：进行模板检索，提取 `distogram`、`unit_vector`、`aatype` 等 3D 几何特征。
2. **非蛋白链 (RNA/Ligand)**：触发 `empty_template_features` 函数，生成占位符。
3. **对齐与拼接**：
* 将各链特征按 `token` 顺序排列。
* 通过 `standard_token_idxs` 映射到全局索引。
* **结果**：输出形状为 $[N_{templ}, N_{token}, N_{token}, C_{feat}]$ 的张量。



### B. 中游：Trunk 初始化与 Recycle

在 `protenix/model/protenix.py` 的执行循环中：

1. **初始化 $z$**：由输入序列特征通过 `RelativePositionEncoding` 等初始化得到 $z_{init} \in [N_{token}, N_{token}, C_z]$。
2. **Recycle 叠加**：每一轮 Recycle 开始时，$z$ 会加上上一轮的反馈。
3. **Template 注入**：调用 `TemplateEmbedder`。

### C. 下游：TemplateEmbedder (模型内部)

这是特征真正转化为 Embedding 的地方：

```python
# 伪代码：TemplateEmbedder 内部逻辑
def forward(self, batch, z):
    # batch['template_distogram'] 已经是 [N_templ, N_token, N_token, 39]
    # 1. 投影与对称化
    template_mask = batch['template_mask']
    # 2. 关键：只在同链内部生效
    multichain_mask = (asym_id[:, None] == asym_id[None, :])
    
    # 3. 与全局 z 融合
    # 对每个 template 进行 Pairformer 类似的特征提取
    # ...
    
    # 4. 产生 Delta z
    z_template_update = self.output_projection(template_features) # [N_token, N_token, C_z]
    return z + z_template_update

```

---

## 3. 关键机制对比：你的直觉 vs 代码实现

| 维度 | 你的直觉理解 (修正前) | 代码实际实现 (Protenix) |
| --- | --- | --- |
| **形状对齐** | 局部 $N_{prot} \times N_{prot}$ 贴到全局。 | 上游直接构造全局 $N_{token} \times N_{token}$，非蛋白处补空。 |
| **空位填充** | 剩下的地方填 0。 | 使用 `empty_template_features`，几何通道 Mask 掉，类别通道用 Default/Unknown。 |
| **跨链作用** | 可能会影响到其他链。 | 内部存在 `multichain_mask`，模板信息通常被限制在单链内部。 |
| **注入位置** | 在 Pairformer 中间某个位置。 | 在每个 Recycle 开始处，作为 $z$ 的初始修正项之一。 |

---

## 4. 对引入 RNA 二级结构的启发

既然 Template 路径在代码上是**“预留了 $N_{token}$ 接口，但逻辑上只喂蛋白数据”**，这给你加 RNA 二级结构提供了两个策略：

### 方案一：硬塞进 Template 路径 (不推荐)

* **做法**：伪造 RNA 的 `template_distogram`（将二级结构配对概率映射为距离约束）。
* **缺点**：Template 路径期望的是 3D 坐标衍生的几何特征（如 unit vectors, frames），RNA 二级结构缺少这些信息，强行塞入会导致大量特征维度的浪费或不匹配。

### 方案二：走 `z_init` 或 `z_constraint` 路径 (推荐)

* **做法**：直接在 `protenix.py` 中，在 $z$ 初始化之后、进入 Pairformer 之前，加一个自定义的 `RNAStructureEmbedder`。
* **优点**：
* **维度匹配**：二级结构本质是 Pairwise 概率，直接对应 $z$ 的通道。
* **灵活性**：不受 Template 模块中蛋白特有逻辑（如 `aatype`）的限制。
* **更符合 Transformer 逻辑**：将二级结构作为一种 **Bias** 或 **Prior** 注入 $z$。



---

**下一步建议：**
如果你想深入看这部分代码，我可以为你提取 `template_featurizer.py` 中 **`empty_template_features`** 的具体实现，看看它到底给非蛋白区域填了什么具体的数值？