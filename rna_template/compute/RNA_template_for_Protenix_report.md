# RNA template → Protenix 输入特征方案与脚本说明

## 1．我给你的文件

本目录包含 4 个文件：

1．`rna_template_common.py`
- 公共函数库。
- 负责解析 `mmCIF/PDB`、抽取 RNA 残基、序列对齐、anchor／frame 计算、distogram、DSSR JSON 解析。

2．`build_rna_template_protenix.py`
- **最小兼容版**。
- 目标是直接产出与现有 `TemplateEmbedder` 尽量兼容的 5 组 RNA template 特征。
- 输出重点：
  - `template_aatype`
  - `template_distogram`
  - `template_pseudo_beta_mask`
  - `template_unit_vector`
  - `template_backbone_frame_mask`

3．`build_rna_template_extended.py`
- **RNA 扩展版**。
- 在最小兼容版基础上，额外输出 RNA 特有 pair/single feature。
- DSSR 可用时，会抽取：
  - base pair 类型
  - Leontis-Westhof 类别
  - stacking
  - modification flag
  - chi / delta / eta / theta 的紧凑角度编码
- 这个文件适合你后面自己写一个 `RNATemplateEmbedder` 并行加到 `z` 上。

4．`run_dssr_and_build.py`
- 自动化脚本。
- 先跑 `x3dna-dssr` 生成 JSON，再自动调用最小版和／或扩展版 builder。

---

## 2．整体设计思路

### 2.1 最稳妥的落地路线

先走 **最小兼容版**。

原因是：
- 你现在最想验证的是「RNA template 注入以后，Protenix 是否真的能吃到有效几何先验」。
- 这一步最少改 trunk。
- 只要你最终能把 key 名和 shape 对齐，你就有机会直接复用现有 template 路径。

所以第一版的核心目标不是把所有 RNA 生物学都塞满，而是：

> 先把 RNA template 变成与 protein template **同类型的几何张量**。

也就是：
- residue type
- distance bins
- anchor mask
- local frame orientation
- frame mask

---

### 2.2 为什么 RNA 不能直接照抄 protein template

protein template 里常见的是：
- pseudo-beta
- 主链局部框架
- distogram
- unit vector

RNA 不能直接沿用 protein 的原因：
- RNA 没有蛋白那种 `N-CA-C` 主链三原子框架。
- RNA 的关键结构约束除了距离，还强依赖：
  - base pairing
  - base stacking
  - 糖-磷骨架构象
  - 修饰碱基

所以我做的处理是：

#### Anchor（替代 pseudo-beta）
优先级：
1．base center
2．C4'
3．C1'

#### Frame（替代 protein backbone frame）
使用：
- 原点：`C4'`
- 第一轴：`P - C4'`
- 第二轴：`C1' - C4'`
- 再做 Gram-Schmidt 正交化，得到 `e1, e2, e3`

这个定义满足两个条件：
- 和 protein template 的“局部 frame + relative direction”数学形式保持相似。
- 又是 RNA 化学上合理的局部坐标系。

---

## 3．最小兼容版脚本的输出

### 3.1 输出 key

`build_rna_template_protenix.py` 输出：

- `template_aatype`：`[T, N]`
- `template_distogram`：`[T, N, N, 39]`
- `template_pseudo_beta_mask`：`[T, N, N]`
- `template_unit_vector`：`[T, N, N, 3]`
- `template_backbone_frame_mask`：`[T, N, N]`

同时还会额外保存一些调试友好的中间量：

- `template_anchor_pos`
- `template_anchor_mask_1d`
- `template_frame_origin`
- `template_frame_axes`
- `template_frame_mask_1d`
- `template_mask`
- `template_names`
- `template_mapping_json`

这些额外字段的用途是：
- 便于你 debug 对齐对不对。
- 如果你的 Protenix 分支实际上期待 1D mask，而不是 pairwise mask，你可以直接从这些中间量改。

---

### 3.2 关键计算方式

#### 3.2.1 `template_aatype`
使用 Protenix 官方 RNA residue id：
- `A -> 21`
- `G -> 22`
- `C -> 23`
- `U -> 24`
- `N -> 25`
- `- -> 31`

未对齐到模板的位置填 `31`（gap）。

---

#### 3.2.2 `template_distogram`
先求 anchor 间距离：

```text
D_ij = || anchor_i - anchor_j ||
```

再按照：
- `min_bin = 3.25`
- `max_bin = 50.75`
- `num_bins = 39`

做 one-hot binning。

也就是输出：

```text
template_distogram[t, i, j, :] ∈ R^39
```

---

#### 3.2.3 `template_pseudo_beta_mask`
这里实际上是 RNA anchor 的 pairwise mask：

```text
mask_anchor(i, j) = anchor_valid(i) * anchor_valid(j)
```

---

#### 3.2.4 `template_unit_vector`
把 `j` 相对 `i` 的位置向量投影到 `i` 的局部 frame 中。

```text
r_ij = anchor_j - origin_i
u_ij = normalize( frame_i * r_ij )
```

输出：

```text
template_unit_vector[t, i, j, :] ∈ R^3
```

---

#### 3.2.5 `template_backbone_frame_mask`
只有当：
- `i` 的 frame 可用
- `j` 的 anchor 可用

才给 orientation。

```text
frame_mask(i, j) = frame_valid(i) * anchor_valid(j)
```

---

## 4．扩展版脚本的输出

`build_rna_template_extended.py` 除了保留最小兼容版全部 key 外，还会额外输出：

### 4.1 RNA-specific pair 特征

`rna_template_pair`：`[T, N, N, C_pair]`

默认 `C_pair` 包含：

#### base-pair 总类
- `pair_any`
- `pair_canonical`
- `pair_wobble`
- `pair_noncanonical`

#### Leontis-Westhof 18 类
- `lw_cWW`
- `lw_tWW`
- `lw_cWH`
- `lw_tWH`
- `lw_cWS`
- `lw_tWS`
- `lw_cHW`
- `lw_tHW`
- `lw_cHH`
- `lw_tHH`
- `lw_cHS`
- `lw_tHS`
- `lw_cSW`
- `lw_tSW`
- `lw_cSH`
- `lw_tSH`
- `lw_cSS`
- `lw_tSS`

#### stacking
- `stack_any`
- `stack_consecutive`
- `stack_nonconsecutive`

所以扩展版是：

```text
39 个 distogram 通道
+ 25 个 RNA-specific pair 通道
```

如果你后面把两者 concat 到一个新的 RNA template encoder 里，就已经明显超过“只有 39 个距离桶”的表达能力了。

---

### 4.2 RNA-specific single 特征

`rna_template_single`：`[T, N, C_single]`

默认包含：
- `is_modified`
- `is_purine`
- `is_pyrimidine`
- `has_anchor`
- `has_frame`
- `chi_sin`
- `chi_cos`
- `delta_sin`
- `delta_cos`
- `eta_sin`
- `eta_cos`
- `theta_sin`
- `theta_cos`

这里的角度全部用 `sin/cos` 表示，而不是直接输出角度本身，原因是：
- 避免角度在 `-π ~ π` 的跳变不连续。
- 对网络更友好。

---

### 4.3 可选 atom-level 输出

如果你加 `--emit_dense_atoms`，还会输出：
- `template_atom_positions`：`[T, N, A, 3]`
- `template_atom_mask`：`[T, N, A]`
- `template_atom_names`

这个是为你后面自己写 atom-level RNA template encoder 预留的。

注意：
- 这个 atom list 是“通用 RNA union atom list”。
- 它更适合 **自定义分支**。
- 它不是保证能直接对上某个你本地 Protenix 分支里写死的 `24 atom slot` 版本。

---

## 5．DSSR 在这里怎么用

### 5.1 为什么要 DSSR

如果你只想做最小兼容版，其实 **不一定要 DSSR**。

因为最小版只靠结构坐标本身就能算：
- anchor
- frame
- distogram
- unit vector

但是如果你要做真正的 RNA-specific 扩展，DSSR 很重要，因为它能直接给你：
- pairs
- stacks
- nts
- modified nucleotide 标记
- 一些参考 frame／参数

所以我的建议是：
- **先最小版做 baseline**。
- **再用 DSSR 扩展版做增量实验**。

---

### 5.2 自动化脚本怎么跑

`run_dssr_and_build.py` 会做三件事：

1．对每个 template 运行：
```bash
x3dna-dssr -i=template.cif --json -o=template.dssr.json
```
或：
```bash
x3dna-dssr -i=template.cif --json=ebi -o=template.dssr.json
```

2．调用最小版 builder。

3．调用扩展版 builder。

---

## 6．最推荐的实验顺序

### 阶段 A．先验证最小兼容版

先用：
- `build_rna_template_protenix.py`

做一个最小实验：
- 同一个 RNA target
- 有无 template feature 对比
- 看精度是否上升

如果这个阶段完全没收益，再往后加 RNA 特有通道的意义就不大。

---

### 阶段 B．再加扩展版

如果阶段 A 有收益，再把：
- `rna_template_pair`
- `rna_template_single`

喂给一个并行的 `RNATemplateEmbedder`。

建议融合方式：

```text
z = z
  + TemplateEmbedder(minimal_features)
  + RNATemplateEmbedder(rna_template_pair, rna_template_single)
```

也可以先做门控版：

```text
z = z + g1 * z_template_min + g2 * z_template_rna
```

其中 `g1, g2` 可以是：
- 标量可学习参数
- 或根据 template confidence 预测出来的 gate

---

## 7．和你当前 Protenix 分支的关系

这里要特别说明一个工程细节。

你前面已经分析过：
- 你本地那条 code path 里，template 路径 historically 对非 protein chain 可能还是空模板逻辑。
- 也就是 RNA chain 可能不会自动走现成 protein template featurizer。

所以这套脚本的定位不是“保证你的当前分支零改就能跑通”，而是：

> 把 RNA template 先标准化成一组稳定张量，让你后面无论是直接注入 `input_feature_dict`，还是 patch featurizer，还是并行加一个 `RNATemplateEmbedder`，都能直接复用。

也就是说，这套脚本解决的是：
- **RNA template 怎么表示**
- **怎么计算**
- **怎么落盘**

你后面只需要决定：
- 走旧 template 分支注入
- 还是单独写一个 RNA 模块并行注入

---

## 8．最常见的坑

### 8.1 residue id 对不上
DSSR 的 `nt_id` 和你自己 parser 的 residue 编号，不一定永远完全一致。

我这里做的是比较宽松的映射：
- `(chain, resseq, insertion_code)`

如果你某些模板有复杂 insertion code 或特殊命名，你可能需要针对自己数据再补 1 层映射规则。

---

### 8.2 modified nucleotide 太多
有些修改碱基会让：
- base 推断
- atom 名称
- chi 角主原子

出现缺失或不标准。

当前脚本的策略是：
- 能算就算。
- 不能算就 mask 成 0。
- 不会硬报错。

---

### 8.3 第一位或末位 frame 不完整
有些 RNA 首位残基缺 `P`，或者结构本来缺原子。

当前脚本做法：
- `P` 缺失时，尝试 `O5'` 或 `C3'` 替代。
- 再不行就该位置 frame mask = 0。

---

### 8.4 你的具体 Protenix 分支可能期待别的 mask 形状
我这里默认输出的是：
- pairwise mask：`[T, N, N]`

同时又保留：
- `template_anchor_mask_1d`
- `template_frame_mask_1d`

如果你本地分支实际要的是 1D mask，你直接从这两个中间量改即可，不需要重跑结构解析。

---

## 9．建议你先怎么跑

### 9.1 只做最小兼容版

```bash
python build_rna_template_protenix.py \
  --query_seq query.fa \
  --template template1.cif:A \
  --template template2.pdb:A \
  --output rna_template_minimal.npz
```

---

### 9.2 跑 DSSR 扩展版

```bash
python build_rna_template_extended.py \
  --query_seq query.fa \
  --template template1.cif:A \
  --dssr_json template1.dssr.json \
  --template template2.pdb:A \
  --dssr_json template2.dssr.json \
  --output rna_template_extended.npz \
  --emit_dense_atoms
```

---

### 9.3 一键自动化

```bash
python run_dssr_and_build.py \
  --query_seq query.fa \
  --template template1.cif:A \
  --template template2.pdb:A \
  --outdir ./rna_template_build \
  --mode both \
  --emit_dense_atoms
```

---

## 10．最后怎么接到 Protenix

### 方案 1．直接塞回现有 template 路径

如果你的分支已经支持你自己手工注入 template tensor：
- 直接把 `rna_template_minimal.npz` 里的最小 5 个 key 接进去。

这是最先应该验证的方案。

---

### 方案 2．并行新建 `RNATemplateEmbedder`

如果你要用扩展版：
- 保留最小 5 key 给现有 TemplateEmbedder。
- 再把 `rna_template_pair`、`rna_template_single` 喂给你自己的 RNA 支路。

这个方案最适合你后面做：
- RNA pairing／stacking 的明确建模
- template geometry 和 secondary-structure prior 的融合
- mixture / gate / confidence weighting

---

## 11．一句话总结

这套脚本分成两层：

- **第一层**：先把 RNA template 变成与现有 Protenix template 语义尽量一致的几何张量。
- **第二层**：再把 RNA 特有的 pairing／stacking／modification／backbone torsion 作为扩展通道并行加进去。

这样做的好处是：
- 先快验证。
- 再慢扩展。
- 不会一开始就把 trunk 改得太重。
