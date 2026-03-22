1. `Protenix` 相对原版做了什么自定义。
2. `Protenix` 相对原版做了什么自定义。
3. RNA 二级结构是怎样被接入网络的。
4. RNA template 是否真的被接入网络，以及它和原版 protein template 的关系。

**核心结论：**

- `Protenix` 这条线的核心自定义是“自定义 RNA 数据集 + RNA 单体 embedding 注入”，不是 RNA template 注入。
- `Protenix_v1` 的核心自定义是“新增 RNA secondary structure pair feature，并直接加到 trunk 的初始 pair 表示 `z_init` 上”。
- `Protenix_v1` 仍然保留了官方的 **protein template** 网络通路；这条通路在结构上是完整的。
- 代码里能看到一些“RNA template 搜索/编码”的痕迹，但**真正的 template featurizer 和 template network 仍然是 protein-only**。也就是说，当前代码**没有把 RNA template 真正接入网络主干**。
- 因此，如果严格回答“怎么把 RNA 二级结构和 RNA template 加入到网络中”，答案是：
  - RNA 二级结构：**已经接入**，而且接在 pair 初始化阶段。
  - RNA template：**尚未真正接入**；目前只有前处理和接口层面的局部准备。

---

## 2. 我如何判定“原版”和“自定义版”

### 2.1 `Protenix`

`Protenix` 的远端是 ByteDance 官方仓库：

- `origin = https://github.com/bytedance/Protenix.git`

当前分支 `HEAD` 为提交：

- `19101fe add data_configs_custom_rna`

相对官方标签 `v0.7.3`，这个仓库已经引入了自定义 RNA 数据配置与 RNA embedding 相关改动。

### 2.2 `Protenix_v1`

`Protenix_v1` 当前分支 `HEAD` 在 `ss` 分支上，最近关键提交：

- `10c3adc (tag: v1.0.1) fix inference with ion`
- `a0ff3fe release RNA MSA data`
- `fadd7f4 add RNA secondary feature to protenix v1`
- `3f0af25 config update`

所以：

- `v1.0.1` 可以视为较接近官方的 v1 基线。
- `fadd7f4` 是 RNA 二级结构正式进入 `Protenix_v1` 的关键提交。
- `3f0af25` 主要是配置/路径调整，不是网络结构本身的本质变化。

---

## 3. `Protenix` 这条线到底做了什么

### 3.1 不是模板改造，而是“RNA 数据 + RNA embedding”

`Protenix` 的自定义入口主要在：

- `configs/configs_base.py`
- `configs/configs_data_rna.py`
- `protenix/data/data_pipeline.py`
- `protenix/model/modules/embedders.py`

其中最关键的是两步：

1. 新增一套专门的 RNA 训练/测试数据配置。
2. 把预先准备好的 RNA embedding 注入到单体输入 `s_inputs`。

### 3.2 自定义 RNA 数据配置

`configs/configs_base.py` 新增了 `custom_rna_data` 与 `rna_embedding` 开关，说明这条线的目标是支持 RNA 专项训练，而不是只跑官方 protein/complex 数据：

- `Protenix/configs/configs_base.py:45-78`

`configs/configs_data_rna.py` 定义了独立的 RNA 数据集：

- `train_rna_before202509`
- `test_rna_before202509`
- `test_rna_casp16`

并将数据根目录切到 `rna_data/all_cif`、`rna_data/all_pkl` 与对应索引：

- `Protenix/configs/configs_data_rna.py:182-248`

同时它把 MSA 也改成了面向 RNA 数据目录的版本：

- `Protenix/configs/configs_data_rna.py:249-278`

但是模板配置被直接关掉：

- `Protenix/configs/configs_data_rna.py:279-281`

这一步已经很关键：**这条线默认就不是“用 template 学 RNA”**。

### 3.3 RNA embedding 如何进入网络

在 `DataPipeline` 里，作者先扫描 token 序列中的 RNA 片段，再把 `bioassembly_dict["rna_embeddings"]` 里的预计算向量对齐回 token 级表示：

- 寻找 RNA 片段：`Protenix/protenix/data/data_pipeline.py:117-135`
- 生成 token 级 RNA embedding：`Protenix/protenix/data/data_pipeline.py:138-170`
- 载入 bioassembly 后写入 `rna_embeddings_combine`：`Protenix/protenix/data/data_pipeline.py:173-193`

之后，这个 `rna_embeddings_combine` 被塞进 MSA feature 字典里，一起流向下游：

- 无裁剪时注入：`Protenix/protenix/data/data_pipeline.py:327-345`
- 裁剪后注入：`Protenix/protenix/data/data_pipeline.py:376-400`

最终在 `InputFeatureEmbedder` 中，RNA embedding 通过一个线性层投影到与 `s_inputs` 同维度后直接相加：

- 初始化 `linear_rna`：`Protenix/protenix/model/modules/embedders.py:59-75`
- 前向相加：`Protenix/protenix/model/modules/embedders.py:126-131`

### 3.4 公式化描述

原版 AF3 / Protenix 的 token 输入近似可写为：

\[
s_i^{(0)} = \operatorname{concat}(a_i,\ \text{restype}_i,\ \text{profile}_i,\ \text{deletion\_mean}_i)
\]

其中：

- \(a_i\) 是 atom attention encoder 产生的 token 表示。

`Protenix` 自定义后变成：

\[
s_i^{(0,\text{RNA})} = s_i^{(0)} + W_{\text{rna}} e_i^{\text{rna}}
\]

其中：

- \(e_i^{\text{rna}} \in \mathbb{R}^{1280}\) 是预计算 RNA embedding。
- \(W_{\text{rna}}\) 是 `linear_rna`。

这说明 `Protenix` 的 RNA 改造发生在 **single representation** 侧，不是在 template branch。

### 3.5 `Protenix` 的 template 网络其实没接上

`Protenix` 的 `TemplateEmbedder` 代码虽然还在，但前向直接返回 `0`：

- `Protenix/protenix/model/modules/pairformer.py:958-960`

这意味着即使模板特征存在，**模板分支也不会贡献任何 pair update**。因此：

- `Protenix` 不是“把 RNA template 加进网络”的实现。
- 它更像“用 RNA 数据和 RNA embedding 做定制训练”的实验线。

---

## 4. `Protenix_v1` 这条线做了什么

### 4.1 这条线保留了官方 protein template 主干

与 `Protenix` 不同，`Protenix_v1` 的 `TemplateEmbedder` 是完整工作的。

在 trunk 主流程中：

- 初始化 `TemplateEmbedder`：`Protenix_v1/protenix/model/protenix.py:128`
- recycle 循环中把 template update 加到 `z` 上：见同文件后续循环逻辑

`TemplateEmbedder` 本身会对每个模板做特征拼接、Pairformer 更新、模板平均，再投影回 `c_z`：

- `Protenix_v1/protenix/model/modules/pairformer.py:937-1031`
- 单模板细节：`Protenix_v1/protenix/model/modules/pairformer.py:1048-1075`

所以 `Protenix_v1` 的结构基础是：

- **官方 protein template 路径仍然存在**
- 在此基础上，新增了 RNA secondary structure 分支

### 4.2 RNA secondary structure 从哪里来

`fadd7f4 add RNA secondary feature to protenix v1` 这个提交新增了：

- `protenix/data/ss/ss_featurizer.py`
- `protenix/data/ss/ss_utils.py`
- 对 `data_pipeline.py` / `dataset.py` / `embedders.py` / `protenix.py` 的改动

这说明它不是“训练脚本 hack”，而是贯穿数据到模型的正式功能。

### 4.3 二级结构的原始表示：dot-bracket -> 邻接矩阵

`parse_dot_bracket()` 会把 RNA 二级结构字符串解析成对称的 \(L \times L\) pairing adjacency matrix：

- `Protenix_v1/protenix/data/ss/ss_utils.py:5-52`

具体规则：

- `(` `)`、`[` `]`、`{` `}`、`<` `>` 都会形成配对。
- 伪结还支持 `A-Z` 与 `a-z` 的配对。
- 若第 \(i\) 位与第 \(j\) 位配对，则

\[
M_{ij} = M_{ji} = 1
\]

否则

\[
M_{ij} = 0
\]

这一步的本质是把一维 dot-bracket 结构变成 pair-level binary contact prior。

### 4.4 二级结构如何对齐到 token / crop

`SSFeaturizer` 做了三件事：

1. 按 RNA 序列去映射表里找到对应 `ss.txt`
2. 每条 RNA 链各自生成配对矩阵
3. 用 block diagonal 把多条链拼起来，再用 `map_to_standard()` 对齐到当前 token/crop 索引

对应代码：

- 取 `ss.txt`：`Protenix_v1/protenix/data/ss/ss_featurizer.py:37-63`
- RNA 链转 pairing matrix，非 RNA 链补零：`Protenix_v1/protenix/data/ss/ss_featurizer.py:95-109`
- 多链 block diagonal：`Protenix_v1/protenix/data/ss/ss_featurizer.py:111-116`
- 用 `std_idxs` 投影回裁剪后的 token 网格：`Protenix_v1/protenix/data/ss/ss_featurizer.py:118-146`
- 最终输出 key：`rna_sec_struct`，见 `Protenix_v1/protenix/data/ss/ss_featurizer.py:235-241`

如果写成公式，就是：

设第 \(k\) 条 RNA 链长度为 \(L_k\)，其二级结构矩阵为 \(M^{(k)} \in \{0,1\}^{L_k \times L_k}\)，则多链拼接矩阵为

\[
M^{\text{blk}} = \operatorname{blockdiag}(M^{(1)}, M^{(2)}, \dots, M^{(K)}, 0, 0, \dots)
\]

再通过标准索引映射 \(P\) 取出当前 crop/token 对应子矩阵：

\[
M^{\text{crop}} = P M^{\text{blk}} P^\top
\]

这里 \(P\) 可以理解为由 `std_idxs` 定义的选择矩阵。

### 4.5 二级结构如何被数据管线送进模型

`DataPipeline` 新增了 `get_ss_raw_features()`，并在裁剪前后都生成二级结构特征：

- 入口函数：`Protenix_v1/protenix/data/pipeline/data_pipeline.py:204-236`
- 无裁剪路径：`Protenix_v1/protenix/data/pipeline/data_pipeline.py:311-338`
- 裁剪路径：`Protenix_v1/protenix/data/pipeline/data_pipeline.py:362-399`

`Dataset` 随后把 `ss_features` 一路带到 `features_dict` 中：

- crop 返回 `cropped_ss_features`：`Protenix_v1/protenix/data/pipeline/dataset.py:507-528`
- 空则补 dummy，非空则 `dict_to_tensor` 并写入 `features_dict`：`Protenix_v1/protenix/data/pipeline/dataset.py:853-873`
- `get_ss_featurizer()` 构造器：`Protenix_v1/protenix/data/pipeline/dataset.py:919-940`

### 4.6 二级结构如何进网络

`configs_base.py` 给模型增加了一个 `ss_embedder` 配置项：

- `Protenix_v1/configs/configs_base.py:202-212`

`SSEmbedder` 非常直接：它不是 MLP，也不是 attention，而是一个离散 embedding lookup：

- `Protenix_v1/protenix/model/modules/embedders.py:335-361`

若 `rna_sec_struct[i,j] \in \{0,1\}`，则

\[
z_{ij}^{\text{ss}} = E_{\text{ss}}[\,M_{ij}^{\text{crop}}\,]
\]

其中：

- \(E_{\text{ss}} \in \mathbb{R}^{2 \times c_z}\) 是 `nn.Embedding(num_bins=2, c_z)`。

然后在主模型里，它被**直接加到 pair 初始化 `z_init`** 上：

- 初始化 `self.ss_embedder`：`Protenix_v1/protenix/model/protenix.py:133-136`
- 加到 `z_init`：`Protenix_v1/protenix/model/protenix.py:215-234`

也就是说，原始的 pair 初始化从

\[
z_{ij}^{(0)} = W_i s_i + W_j s_j + \operatorname{RelPos}_{ij} + W_b b_{ij} + z_{ij}^{\text{constraint}}
\]

变成了

\[
z_{ij}^{(0,\text{ss})} = W_i s_i + W_j s_j + \operatorname{RelPos}_{ij} + W_b b_{ij} + z_{ij}^{\text{constraint}} + z_{ij}^{\text{ss}}
\]

这和官方 protein template 的接入方式非常不同：

- **RNA secondary structure**：在 trunk 初始 pair 表示阶段加入。
- **protein template**：在 recycle 循环内部，以 template branch update 的形式加入。

这个设计说明作者把 RNA 二级结构当成一种“强先验 pair bias”，而不是当成独立模板分支。

---

## 5. 原版 protein template 是怎么进网络的

这一部分是回答“和原版 protein 对比”的关键。

### 5.1 Template feature 的生成条件仍是 protein-only

`TemplateFeaturizer` 在入口处直接限制：

- `Protenix_v1/protenix/data/template/template_featurizer.py:334-335`

也就是：

\[
\text{if } chain\_entity\_type \neq \text{PROTEIN\_CHAIN},\quad \text{return empty}
\]

这已经说明当前模板特征生成只面向 protein。

### 5.2 单模板特征张量是什么

`TemplateEmbedder` 使用的输入包括：

- `template_distogram`，39 维
- `template_backbone_frame_mask`，1 维
- `template_unit_vector`，3 维
- `template_pseudo_beta_mask`，1 维
- `template_restype_i`，32 维 one-hot
- `template_restype_j`，32 维 one-hot

见：

- `Protenix_v1/protenix/model/modules/pairformer.py:941-949`

对单个模板 \(t\) 与残基对 \((i,j)\)，它构造：

\[
a_{ij}^{(t)} =
\operatorname{concat}\Big(
d_{ij}^{(t)},
m_{ij,\text{pb}}^{(t)},
\operatorname{onehot}(r_i^{(t)}),
\operatorname{onehot}(r_j^{(t)}),
u_{ij}^{(t)},
m_{ij,\text{bb}}^{(t)}
\Big)
\]

随后与当前 pair 表示共同投影：

\[
v_{ij}^{(t)} = W_z \operatorname{LN}(z_{ij}) + W_a a_{ij}^{(t)}
\]

再走一个只处理 pair 的 `PairformerStack`：

\[
\tilde{v}^{(t)} = \operatorname{PairformerTemplate}(v^{(t)})
\]

所有模板平均后回投影：

\[
u_{ij} = \frac{1}{T}\sum_{t=1}^T \tilde{v}_{ij}^{(t)}, \qquad
\Delta z_{ij}^{\text{templ}} = W_u \operatorname{ReLU}(u_{ij})
\]

对应代码：

- 模板平均与输出投影：`Protenix_v1/protenix/model/modules/pairformer.py:1015-1031`
- 单模板特征拼接：`Protenix_v1/protenix/model/modules/pairformer.py:1048-1075`

### 5.3 它与 RNA 二级结构注入的本质区别

**protein template 路径**

- 输入是 richer geometric template tensors
- 进入方式是 recycle 中的 template branch
- 数学上是一个 learned residual update
- 粒度更像“结构模板条件化”

**RNA secondary structure 路径**

- 输入只有 binary pairing matrix
- 进入方式是 `z_init` 直接加法
- 数学上更像 pair bias / pair prior
- 粒度更像“结构约束先验”

所以不能把现在的 RNA 二级结构实现理解成“RNA template 的简化版”。两者在网络中的角色并不一样。

---

## 6. RNA template：代码里到底有没有真正接进去

### 6.1 有一些“看起来像在做 RNA template”的痕迹

我确实看到三类痕迹：

1. `template_parser.py` 的 `encode_template_restype()` 已经支持 `RNA_CHAIN` 与 `DNA_CHAIN`
   - `Protenix_v1/protenix/data/template/template_parser.py:206-221`
2. `runner/template_search.py` 的脚本尾部会遍历 `Protenix/rna_data/mmcif_msa` 去跑 `run_template_search()`
   - `Protenix_v1/runner/template_search.py:239-247`
3. `README` / `docs` / `inference_demo.sh` 强调了 v1 的 template 能力

这说明作者**考虑过**把更广义的 template 机制拓展到核酸。

### 6.2 但真正的 template featurizer 仍然卡在 protein-only

真正阻断 RNA template 进入网络的有两道硬门槛：

**门槛 1：模板检索入口就要求 `chain_entity_type == PROTEIN_CHAIN`**

- `Protenix_v1/protenix/data/template/template_featurizer.py:334-335`

**门槛 2：真正生成 `template_aatype` 时硬编码了 `PROTEIN_CHAIN`**

- `Protenix_v1/protenix/data/template/template_utils.py:599-606`

也就是：

\[
\texttt{aatype = encode\_template\_restype(PROTEIN\_CHAIN, out\_seq\_str)}
\]

这不是一个“配置没打开”的问题，而是**特征生成逻辑本身仍按 protein 模板写死**。

### 6.3 inference 路径也没有把 RNA secondary structure 接上

虽然 `InferenceSSFeaturizer` 已经被写出来了：

- `Protenix_v1/protenix/data/ss/ss_featurizer.py:243-317`

但当前 inference dataloader 实际只调用：

- `InferenceMSAFeaturizer.make_msa_feature()`
- `InferenceTemplateFeaturizer.make_template_feature()`

见：

- `Protenix_v1/protenix/data/inference/infer_dataloader.py:162-177`

没有调用 `InferenceSSFeaturizer.make_ss_feature()`。

这说明：

- **训练/数据集侧** 的 RNA secondary structure 已经接入。
- **推理 JSON 流水线侧** 目前还没有完全打通。

### 6.4 因此对“RNA template”的严格结论

如果以“模板特征被生成为 `template_*` 张量，并通过 `TemplateEmbedder` 影响 `z`”作为判据，那么：

**当前代码没有真正实现 RNA template 注入。**

更准确地说，当前状态是：

- 已经有 RNA template 搜索和 RNA residue type 编码的一部分准备工作。
- 但模板特征抽取与模板网络更新仍然停留在 protein-only。

---

## 7. `Protenix` 与 `Protenix_v1` 的对比总结

| 维度 | `Protenix` | `Protenix_v1` |
|---|---|---|
| 相对基线 | `v0.7.3` 官方线 | `v1.0.1` 官方线 |
| 自定义重点 | RNA 数据集、RNA embedding、训练配置 | RNA secondary structure、RNA MSA 数据、保留 protein template |
| RNA 特征注入位置 | `s_inputs` 单体表示 | `z_init` 成对表示 |
| 模板分支是否可用 | 基本不可用，`TemplateEmbedder` 直接返回 0 | 可用，protein template 完整 |
| 是否真的支持 RNA template | 否 | 仍然否 |
| 是否支持 RNA secondary structure | 否 | 是 |

可以把两条线理解为两种完全不同的设计思路：

- `Protenix`：把 RNA 先当作“额外单体语义 embedding”问题处理。
- `Protenix_v1`：把 RNA secondary structure 当作“额外 pair prior”问题处理，同时保留官方 protein template 机制。

---

## 8. 对“怎么把 RNA 二级结构和 RNA template 加入网络”的最终技术判断

### 8.1 RNA 二级结构

当前代码的真实做法是：

1. 从 `ss.txt` / dot-bracket 解析出 pairing adjacency matrix。
2. 对多链做 block-diagonal 合并。
3. 按 crop/token 标准索引切回当前样本。
4. 用 `nn.Embedding(2, c_z)` 映射成 pair embedding。
5. 直接加到 trunk 初始 pair 表示 `z_init`。

这条路径已经完成，且代码闭环是完整的。

### 8.2 RNA template

当前代码**没有**完成下面这条闭环：

1. RNA chain 命中 template hit
2. RNA template 生成 `template_*` 张量
3. `TemplateEmbedder` 对 RNA template 生效
4. 模板 update 写回 trunk pair 表示

它目前停在：

- 搜索脚本可能为 RNA 数据生成了某些模板搜索产物
- 解析器具备 RNA residue type 编码能力
- 但模板特征提取和 template branch 仍然明确限制为 protein-only

所以“RNA template 已经加入网络”这个说法，从当前代码证据看 **不成立**。

---

## 9. 如果要真正把 RNA template 接进网络，最小必要改动是什么

从当前代码结构推断，最小闭环至少需要四步：

1. 去掉 `TemplateFeaturizer` 中 `chain_entity_type != PROTEIN_CHAIN` 的早退限制。
2. 在 `template_utils.py::_extract_template_features()` 中，不再硬编码 `PROTEIN_CHAIN`，而是把真实链类型传给 `encode_template_restype()`。
3. 为 RNA 定义与 protein 模板等价的几何特征语义。
   - 例如 `template_backbone_frame_mask`
   - `template_unit_vector`
   - `template_distogram`
   - 这些在 RNA 上是否仍沿用 pseudo-beta / backbone frame，需要重新定义
4. 检查 `TemplateEmbedder` 的输入 one-hot 维度与核酸 residue vocabulary 是否兼容。

换句话说，真正困难的不是“搜到 RNA template”，而是：

- **如何把 RNA template 规范化成当前 protein template branch 可消费的几何张量。**

---

## 10. 最终结论

一句话总结：

**`Protenix` 主要是在做 RNA embedding 定制；`Protenix_v1` 真正做成的是 RNA 二级结构 pair-bias 注入；而 RNA template 到今天这份代码为止，还没有像原版 protein template 那样被完整接入网络。**

如果你后续要继续推进实现，最合理的路线不是继续把 RNA template 当成“搜索文件”问题，而是先决定：

- RNA template 在几何上要复用 protein template branch，还是单独设计一条 nucleic-acid template branch。

从当前代码形态看，作者已经证明了“RNA secondary structure -> pair prior”这条路很自然；但“RNA template -> protein template branch”还只走到一半。
