# RNA Template 注入 Protenix 的 v1 设计方案

当前 Protenix 的开源模板管线有 3 个与你这个设计直接相关的事实。第一，模板检索与装配现在只真正支持蛋白链：如果 `drop_template` 为真或链类型不是 `PROTEIN_CHAIN`，就直接返回空模板；装配阶段也会把“非蛋白链或长度过短的链”走 `empty_template_features`。第二，这个 fallback 不是跳过模板分支，而是返回**同 shape 的空模板**。第三，TemplateEmbedder 当前假设的 raw template 输入接口是固定的：`39` 维 `template_distogram`、`1` 维 `template_backbone_frame_mask`、`3` 维 `template_unit_vector`、`1` 维 `template_pseudo_beta_mask`，以及 `32 + 32` 维的 `template_restype_i/j`，然后统一进同一个 `linear_no_bias_a`，后面走共享的 `PairformerStack` 和 `linear_no_bias_u`。当前蛋白模板几何也是按蛋白专用语义来的：先算 pseudo-beta，再由坐标生成 39-bin distogram，并用 `C/CA/N` 局部框架去算 unit vector。([GitHub][1])

---

## 1．设计目标

### 1.1 目标

* 第一版仍然并进**同一个大 template tensor**。
* 蛋白模板路径尽量不动。
* RNA 只在**输入端单独开 projector**。
* 后半段 `PairformerStack + output projection` 全部共享。
* RNA 找不到模板时，行为尽量和蛋白的 empty-template fallback 一致。

### 1.2 不做的事

* 第一版**不单开完整 RNA template trunk**。
* 第一版**不填 protein-RNA cross-template block**。
* 第一版**不把 base-pair type／stacking／pseudoknot 直接塞进 39 个 distogram 通道**。

---

## 2．总体方案

### 2.1 高层思路

保留当前全局模板接口不变。也就是仍然构造一个复合物级别的大 template pair tensor：

[
\text{template_feat}[t, i, j, :]
]

其中：

* `t` 是 template slot。
* `i, j` 是复合物全局 token index。
* 蛋白-蛋白区域继续放原有 protein template feature。
* RNA-RNA 区域放新的 RNA template feature。
* protein-RNA、RNA-ligand、protein-ligand、ligand-ligand 在 v1 里先全部保持 0。

### 2.2 核心改动

不再让 RNA raw template feature 直接走蛋白那套 `linear_no_bias_a`，而是：

* protein raw template → `W_prot`
* RNA raw template → `W_rna`
* 两者投到**同一个 hidden template space**
* 然后再进入**同一个共享 PairformerStack**
* 最后再经**同一个共享 output projection** 回到 `z_pair`

也就是：

[
v_{ij}^{(t)} = W_z , LN(z_{ij})

* M_{pp,ij}, W_{prot}, a^{prot}_{tij}
* M_{rr,ij}, W_{rna}, a^{rna}_{tij}
  ]

然后：

[
h_{ij}^{(t)} = \text{SharedTemplatePairformer}\big(v_{ij}^{(t)}\big)
]

最后：

[
u_{ij} = W_{out}, \mathrm{ReLU}\Big(\mathrm{Avg}*t\big(LN(h*{ij}^{(t)})\big)\Big)
]

再按现有 template branch 的方式更新 `z_pair`。

### 2.3 为什么这样做

这个设计把“RNA 与 protein raw 几何语义不同”的问题，限制在**输入投影这一层**解决；但把“模板信息如何帮助 pair representation”的更高层机制，继续交给已经存在的 TemplateEmbedder 后半段去做。

---

## 3．RNA Template Raw Feature 的定义

## 3.1 设计原则

v1 的原则是：

**接口尽量像 protein template。
但内部几何语义换成 RNA-native。**

所以在 key 和 shape 上，建议先保持现有接口名不变：

* `template_distogram`
* `template_backbone_frame_mask`
* `template_unit_vector`
* `template_pseudo_beta_mask`
* `template_restype_i`
* `template_restype_j`

其中：

* `template_pseudo_beta_mask` 这个名字虽然沿用了 protein 命名，但在 RNA 里它的语义改成 **RNA anchor mask**。
* `template_backbone_frame_mask` 在 RNA 里改成 **RNA local frame valid mask**。


### `template_pseudo_beta_mask`：`[T, N, N, 1]`

语义改成 **RNA anchor pair mask**。

先算 1 维：
[
m^{anchor}_i = \mathbf{1}[\text{C4'}\ \text{存在}]
]

再变成 2 维：
[
M^{anchor}_{ij} = m^{anchor}_i \cdot m^{anchor}_j
]

最后扩成 1 通道。

### `template_backbone_frame_mask`：`[T, N, N, 1]`

语义改成 **RNA frame pair mask**。

先算 1 维：
[
m^{frame}_i = \mathbf{1}[\text{C4'}, C1', C3' \text{都存在}]
]

再变成 2 维：
[
M^{frame}_{ij} = m^{frame}_i \cdot m^{frame}_j
]

### `template_unit_vector`：`[T, N, N, 3]`

用 RNA local frame 下的相对方向。

若 `anchor_i` 或 frame 缺失，则该位置填 0，由 mask 告诉模型无效。

### `template_restype_i`、`template_restype_j`

保持与当前 Protenix token vocabulary 一致。

推荐做法：

* 如果你当前模型里 RNA token 本来就已经有统一 residue vocabulary 编码，则直接复用。
* 如果现在模板侧的 32-way one-hot 只够蛋白，那就**统一扩 vocab，一次性改完整个 template restype 编码**，而不是只在 RNA template 分支偷偷改维度。

---

## 4．全局大 Template Tensor 的拼接方式

## 4.1 全局 block 规则

对复合物总 token 数 `N_total`，每个 template slot `t` 都初始化为全 0：

* `global_template_distogram[t] = 0`
* `global_template_unit_vector[t] = 0`
* `global_template_backbone_frame_mask[t] = 0`
* `global_template_pseudo_beta_mask[t] = 0`

然后分 block 写入：

* protein-protein block：写现有 protein template feature
* RNA-RNA block：写新 RNA template feature
* protein-RNA block：保持 0
* RNA-ligand／protein-ligand／ligand-ligand：保持 0

## 4.2 block mask

建议显式构造 2 个 pair block mask：

[
M_{pp}[i,j] = \mathbf{1}[i \in protein \land j \in protein]
]

[
M_{rr}[i,j] = \mathbf{1}[i \in RNA \land j \in RNA]
]

这两个 mask 后面直接给输入 projector 用。

---

## 5．TemplateEmbedder 的改法

## 5.1 需要保留不动的部分

后半段全部共享：

* `pairformer_stack`
* `layernorm_v`
* `relu`
* `linear_no_bias_u`

当前 Protenix 的 TemplateEmbedder 的确是“一个 `linear_no_bias_a` 输入投影，再接共享 `PairformerStack` 和 `linear_no_bias_u` 输出投影”的结构。([GitHub][2])

## 5.2 需要新增的部分

在输入端新增一个 RNA projector：

```python
self.linear_no_bias_a_rna = LinearNoBias(
    in_features = 39 + 1 + 3 + 1 + 32 + 32,
    out_features = self.c,
)
```

protein 侧保持：

```python
self.linear_no_bias_a
```

RNA 侧新加：

```python
self.linear_no_bias_a_rna
```

## 5.3 推荐初始化

推荐顺序：

1. `linear_no_bias_a_rna` 先**拷贝** `linear_no_bias_a` 的初始权重。
2. 然后允许 finetune。
3. `copy protein projector + 小 gate`

先把：

\[
W_{rna} \leftarrow W_{prot}
\]

然后再加一个很小的系数：

\[
v = W_z LN(z) + W_{prot}a_{pp} + \alpha \, W_{rna}a_{rr}
\]

其中：

- `α = 1e-2`，或者
- `α` 是可学习标量，初值设成很小但**不要等于 `0`**

这个方案比“projector 全 `0`”更快，因为：

- `W_rna` 一开始就带有 protein template 的几何先验
- 但小 `α` 会限制初始扰动，不会猛冲进共享 template trunk

如果你把 `α` 设成**严格 `0`**，那也不是完全不行，但这时候 `W_rna` 本身一开始拿不到梯度，只能先靠 `α` 自己动起来，再带动 `W_rna` 学。

所以我会更偏向：

- `W_rna = copy(W_prot)`
- `α_init = 1e-2`

而不是 `α_init = 0`


原因：

* 这样最接近“同接口、相近几何语义”的 warm start。
* 比随机初始化更稳。
* 又不会强迫 RNA 永远共用 protein 的 raw-input 分布假设。

## 6.1 retrieval fallback

模仿蛋白的契约。

### 蛋白当前的 fallback 语义

当前 Protenix 中，如果模板关闭、不是蛋白链、或 hit 数为 0，就返回空 hit；装配时则调用 `TemplateFeatures.empty_template_features(num_tokens)`，其内容是 gap 的 `template_aatype`、全 0 的 `template_atom_mask`、全 0 的 `template_atom_positions`，然后继续走同一个模板接口。([GitHub][1])

### RNA v1 也这样做

定义：

```python
TemplateFeatures.empty_template_features_rna(num_tokens)
```

语义与蛋白保持一致：

* template restype 设 gap／null
* atom positions 全 0
* atom masks 全 0
* release timestamp 设 0
* pad 到固定 `max_templates`

也就是：

[
\text{no RNA template} \Rightarrow \text{empty RNA template} \Rightarrow \text{same API, same shape}
]

---

## 6.2 per-residue fallback

即使一个 hit 存在，也不是每个 residue 都能算出全部几何。

### anchor 缺失

若某 residue 没有 `C4'`：

* `anchor_mask[i] = 0`
* `template_distogram` 相关 pair 位置可保持 0
* `template_pseudo_beta_mask` 对应 pair 位置为 0

### frame 缺失

若某 residue 缺 `C4'`／`C1'`／`C3'` 任一个：

* `frame_mask[i] = 0`
* `template_unit_vector[i, :, :] = 0`
* `template_unit_vector[:, i, :] = 0`
* `template_backbone_frame_mask` 对应 pair 位置为 0

### 部分对齐

如果 hit 只覆盖 query 的一部分 residue：

* 只在对齐到的 token 上写入 feature
* 未对齐 token 位置保持 0，并由 mask 指示无效

---

## 6.3 chain-level fallback

建议 v1 直接复用 protein 的最小长度逻辑，减少分支差异。

也就是：

* 若 RNA 链长度过短，例如 `num_tokens <= 4`，直接返回 empty RNA template。
* 后面如果你发现短 RNA motif 模板有价值，再把这个阈值改成可配置。

---

## 6.4 block-level fallback

即使 RNA hit 存在，v1 里以下 block 仍然始终为空：

* protein-RNA
* RNA-ligand
* protein-ligand
* ligand-ligand

原因很简单：

* 你现在做的是**单链 RNA template 注入**。
* 不是“完整异源复合体模板重建”。
* 把交叉块硬填上，反而会引入更大的伪信号。

---

## 7．建议新增的数据结构

建议在 `input_feature_dict` 里新增：

```python
template_block_mask_pp      # [N, N]
template_block_mask_rr      # [N, N]
template_chain_type         # [N]，可选
template_has_rna_template   # 标量或 [T]，可选
```

其中最重要的是前两个：

* `template_block_mask_pp`
* `template_block_mask_rr`

这样 TemplateEmbedder 不需要猜哪些 pair 属于 protein、哪些属于 RNA。

---


---

## 12．一句话落地版结论

**第一版就做成：
“同一个大 template tensor，protein-RNA 交叉块先空；RNA-RNA 区域填 RNA template feature；RNA 在输入端走单独 projector；后半段 Template Pairformer 完全共享；找不到 RNA template 时返回 fully masked 的 empty RNA template。”**

如果你要，我下一条直接给你补一版**更像代码设计文档的 md**，按下面 4 个类来写：

* `RNATemplateFeaturizer`
* `RNATemplateUtils`
* `TemplateEmbedderRNAAware`
* `empty_template_features_rna()`

[1]: https://github.com/bytedance/Protenix/blob/main/protenix/data/template/template_featurizer.py "Protenix/protenix/data/template/template_featurizer.py at main · bytedance/Protenix · GitHub"
[2]: https://github.com/bytedance/Protenix/blob/main/protenix/model/modules/pairformer.py "Protenix/protenix/model/modules/pairformer.py at main · bytedance/Protenix · GitHub"
[3]: https://rna.bgsu.edu/rna3dhub/pdb?utm_source=chatgpt.com "RNA Structure Atlas"
