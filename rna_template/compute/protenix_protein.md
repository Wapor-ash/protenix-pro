我先直接核对两件事：官方仓库现在怎么说 template 支持，以及你上传那份 protein template 分析里，实际 code path 是怎么把结构变成模型输入的。

我已经确认到关键点了：主干逻辑里，检索入口不是“先搜一堆 cif/pdb 文件”，而是先读模板搜索结果 .a3m 或 .hhr，再根据命中的 template 去加载对应的 mmCIF 结构，最后再转成模型张量。

Thought for 6m 26s

对，但要改成更准确的说法：

不是“先搜到 cif/pdb 然后直接变成模型 tensor”，
而是：

模板检索结果 .a3m / .hhr → 解析 template hits → 读取对应 template 的 mmCIF 坐标 → 对齐到 query → 先生成 atom-level template 张量 → 再派生出模型真正吃的 pairwise template 张量。

你可以把 Protenix 的蛋白 template 流程理解成 5 步：

先做模板检索，不是先扫结构文件。
fetch_template_paths() 先根据 pdb_id 或 query_sequence 去找模板搜索结果文件，支持的入口是 hmmsearch.a3m 和 concat.hhr。只有 chain_entity_type == PROTEIN_CHAIN 时才会继续走这条路径。

把 .a3m/.hhr 解析成 template hits。
get_template() 里会先读这些文件内容，然后用 HHRParser 或 HmmsearchA3MParser 把命中解析出来；如果没有 hit，就直接返回空模板。

再用命中的模板去取结构坐标，官方主路径是 mmCIF。
TemplateFeaturizer 初始化时显式传入的是 prot_template_mmcif_dir，并交给 TemplateHitFeaturizer 去把 hit 变成结构特征，所以官方蛋白 template 的结构源是 mmCIF 目录，不是以 PDB 作为主实现接口。

先得到 atom-level 模板张量。
组装后的 Templates dataclass 里，核心原始字段是：

template_aatype: [T, N]

template_atom_positions: [T, N, 24, 3]

template_atom_mask: [T, N, 24]
这里已经是“对齐到 query token 长度 N”的模板张量了。

再从 atom-level 派生模型真正使用的 pairwise template 特征。
as_protenix_dict() 会从 atom positions/mask 继续算出：

template_distogram

template_pseudo_beta_mask

template_unit_vector

template_backbone_frame_mask
然后送进 TemplateEmbedder。你上传的分析里也总结了，TemplateEmbedder 最终吃的是 39 + 1 + 32 + 32 + 3 + 1 = 108 维 pairwise 模板特征。

所以一句话回答你：

是“先检索 template hit，再从 mmCIF 取结构并转成模板张量”，不是“直接搜 cif/pdb 后立刻喂模型”。
而且在当前公开主干代码里，这条 template 数据管线仍然明确只对 protein chain 开启；非蛋白链会走 empty_template_features 占位