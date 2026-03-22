# Protenix Protein Template Mechanism Report

日期: 2026-03-16

范围:
- 理解 Protenix 里 protein template 的真实获取链路
- 回答 protein template 是否来自 `nhmmer`、是否直接复用 MSA hits
- 解释 MSA 灵敏度与结构 template 可用性之间的关系
- 基于当前代码做一次机制审核, 为后续 RNA temporal template filter 对齐提供参照

---

## 1. 结论摘要

结论非常明确:

1. Protenix 的 **protein template 不走 `nhmmer`**。
2. Protenix 的 **protein MSA** 先通过 **MMseqs2** 获取。
3. 在拿到 `pairing.a3m` / `non_pairing.a3m` 之后, Protenix 会把这些 MSA 合并成 profile, 然后使用 **`hmmbuild + hmmsearch`** 去搜索 **`pdb_seqres` 结构序列库**。
4. 因此, **protein template 不是直接把 MSA hits 当 template 用**。MSA hits 的作用是提升 profile 搜索灵敏度; 真正的 template hit 仍然必须来自结构库。
5. 搜到候选后, Protenix 还会继续做:
   - release date cutoff
   - 最小对齐比例过滤
   - 重复模板过滤
   - 长度过滤
   - 读取对应 mmCIF
   - query/template 重对齐
   - 提取原子坐标并构造 template 特征

所以你的担心不能表述成“MSA 命中高了但没结构怎么办, 因为 template 直接来自 MSA”。  
更准确的表述应当是:

> Protenix 用高灵敏度 MSA 来构建更强的 profile, 再在结构序列库里找模板。  
> 如果 profile 很强但结构库覆盖不足, 结果会是 template 候选少或没有, 而不是错误地把无结构 MSA hit 当模板。

---

## 2. 端到端链路

### 2.1 Protein MSA 阶段

Protein 序列先进入 MSA 搜索流程:

- 入口: `runner/msa_search.py`
- 核心函数: `msa_search()`
- 代码:
  - `runner/msa_search.py:125-152`

这里的实现不是 HMMER, 而是调用 `RequestParser.msa_search()`:

- `runner/msa_search.py:141-146`

实际搜索落到 web service parser:

- `protenix/web_service/colab_request_parser.py:243-331`

在 `mode == "protenix"` 时, 使用的是 `run_mmseqs2_service(...)`:

- `protenix/web_service/colab_request_parser.py:263-275`

这说明 protein MSA 的上游来源是 **MMseqs2**, 不是 `nhmmer`。

### 2.2 Template Search 阶段

拿到 MSA 后, template 搜索由 `runner/template_search.py` 完成:

- 文档说明: `docs/msa_template_pipeline.md:109-128`
- 代码入口: `runner/template_search.py:45-156`

该步骤会:

1. 从 `pairing.a3m` / `non_pairing.a3m` 读入 MSA
2. 将多个 A3M 合并
3. 用 `hmmbuild` 基于 A3M 构造 HMM profile
4. 用 `hmmsearch` 搜索 `pdb_seqres_2022_09_28.fasta`
5. 输出 `hmmsearch.a3m`

关键代码:

- 读取并合并 MSA: `runner/template_search.py:127-140`
- 调用 HMMER 搜索: `runner/template_search.py:141-146`
- 写出模板搜索结果: `runner/template_search.py:148-155`

底层函数 `run_hmmsearch_with_a3m()` 进一步确认了实际机制:

- `protenix/data/tools/search.py:528-556`

这个函数的逻辑是:

1. 把输入 A3M 转成 Stockholm
2. 构造 `Hmmsearch` 对象
3. 用 `query_with_sto(...)` 搜索数据库

因此这里是标准的:

`A3M -> Stockholm -> HMM profile -> pdb_seqres search`

而不是:

`MSA hits -> 直接充当 template`

### 2.3 推理前自动补全模板路径

当 inference JSON 中没有 `templatesPath` 时, Protenix 会自动根据已有 protein MSA 路径触发 template search:

- `runner/template_search.py:159-230`

关键点:

- 先找 `pairedMsaPath` / `unpairedMsaPath`
- 从该目录取 `pairing.a3m` 和 `non_pairing.a3m`
- 若 `hmmsearch.a3m` 不存在, 执行 `run_template_search(...)`
- 最后把 `templatesPath` 指向 `hmmsearch.a3m`

这进一步说明:

> protein template 搜索依赖 protein MSA 结果作为 profile 输入, 但 template 本身并不是 MSA 命中列表。

### 2.4 Template Hit 解析

`hmmsearch.a3m` 之后会被解析成 `TemplateHit`:

- `protenix/data/template/template_parser.py:591-649`

`HmmsearchA3MParser.parse()` 的要点:

- 从 A3M 读 hit
- 只接受描述行里含 `mol:protein` 的项
- 从 hit header 中解析 `pdb_id` / `chain` / `start-end`
- 形成 `TemplateHit` 对象

这说明 template 候选来自 **PDB structure sequence hits**, 而不是来自普通数据库里的 MSA rows。

### 2.5 Template 候选过滤

模板候选不会直接进入模型, 还要经过 prefilter:

- `protenix/data/template/template_utils.py:338-386`

过滤项包括:

1. **日期过滤**
   - 模板 release date 不能晚于 cutoff
2. **最小对齐比例**
   - `align_ratio <= 0.1` 会被过滤
3. **重复/近重复过滤**
   - 如果 hit 基本是 query 的大子串, 会被过滤
4. **最短长度过滤**
   - 长度 `< 10` 的模板会被过滤

### 2.6 mmCIF 读取、重对齐和特征构造

候选 hit 真正变成 template 特征前, 还要经过结构级处理:

- `protenix/data/template/template_utils.py:389-958`

关键流程:

1. 读取本地 mmCIF 或远程抓取 PDBe CIF
   - `template_utils.py:413-433`
2. 从 mmCIF 中提取全原子坐标
   - `template_utils.py:435-470`
3. 检查连续残基 CA 距离
4. 用 `kalign` 将 query 与 template 结构链重新对齐
5. 构造 `template_all_atom_positions` / `template_all_atom_masks`
6. 再导出 pseudo-beta、distogram、unit vector 等模型特征

最终模板特征生成在:

- `protenix/data/template/template_featurizer.py:580-621`

也就是模型消费的是 **真实结构导出的几何特征**, 而不是 MSA 层面的序列特征。

---

## 3. 回答你的几个核心问题

### 3.1 它用了 `nhmmer` 去搜索蛋白质库吗?

没有。

`nhmmer` 在这个仓库里是 RNA MSA 路径使用的工具, 不是 protein template 路径。

Protein template 路径用的是:

- protein MSA: MMseqs2
- protein template search: `hmmbuild + hmmsearch`

证据:

- protein MSA 文档与代码:
  - `runner/msa_search.py:125-152`
  - `protenix/web_service/colab_request_parser.py:263-275`
- protein template 文档与代码:
  - `docs/msa_template_pipeline.md:109-128`
  - `runner/template_search.py:114-146`

### 3.2 如果用了 MSA, 是不是直接用 MSA 的 hits 当 template?

不是。

真实逻辑是:

1. MSA hits 先帮助构建 query 的 profile
2. 再用这个 profile 去打 **结构序列库** `pdb_seqres`
3. 得到的结构库命中才进入 template parser 和 template featurizer

因此:

- MSA hits 是“提升灵敏度的中间信息”
- template hits 是“必须有结构来源的最终候选”

### 3.3 既然 MSA 是序列的, 灵敏度可以更高; 但 template 必须是结构的, 那结构覆盖不够怎么办?

这是对的, 但要分清两个层次:

#### 层次 A: 进化搜索灵敏度

MSA 搜索面对的是大规模序列库, 目标是尽可能召回同源信息。  
这里高灵敏度是有价值的, 即使很多命中没有结构。

#### 层次 B: 结构模板可用性

Template 搜索只面对 `pdb_seqres` 这样的结构序列库。  
即使上一步 MSA 很强, 如果结构库里没有足够近的结构同源, template 仍然会少, 甚至没有。

这不是 bug, 而是结构数据库覆盖边界。

所以 Protenix 的设计本质上是:

> 先用大序列库建立尽可能强的 profile  
> 再把这个 profile 投射到较小但“有结构保证”的数据库里找模板

这正是为了解耦:

- “同源可检测性”
- “结构是否已知”

---

## 4. 从代码看, Protenix 对 template 搜索是偏宽松还是偏保守?

从当前实现看, 它对 template 候选的召回并不保守, 反而偏宽松:

### 4.1 hmmsearch 阈值很松

`runner/template_search.py:114-125`

当前配置:

- `filter_f1 = 0.1`
- `filter_f2 = 0.1`
- `filter_f3 = 0.1`
- `e_value = 100`
- `inc_e = 100`
- `dom_e = 100`
- `incdom_e = 100`

这说明 template search 阶段先尽量多召回, 不在 HMMER 参数层过早卡死。

### 4.2 最小对齐比例很低

`template_utils.py:345`

默认 `min_align_ratio = 0.1`, 即只要对齐覆盖超过 query 的 10%, 就能先进入后续流程。

### 4.3 inference 最后只保留少量模板

推理时 `TemplateHitFeaturizer` 配置:

- `max_hits = 4`
- `_max_template_candidates_num = 20`

见:

- `protenix/data/inference/infer_dataloader.py:105-116`

也就是说策略是:

1. 前面尽量多召回有效候选
2. 后面最多只送 4 个模板进模型

这是典型的“前宽后窄”设计。

---

## 5. 对你提出的担心, 更精确的判断

你的直觉里最重要的一点是:

> “template 必须是结构, 如果搜索太灵敏, 可能 hit 到很多没有结构的东西”

对 Protenix 当前 protein 实现来说, 这个问题已经被数据库切分解决了:

- MSA 阶段命中没有结构没关系
- template 阶段数据库已经限制为结构序列库

所以真正需要关心的不是“灵敏度太高导致无结构 hit 混进模板”, 而是下面两个问题:

### 5.1 结构库覆盖不足

如果 query 很新、很偏门、很远缘, 即使 MSA 很深, `pdb_seqres` 里也可能没有合适模板。  
结果会是:

- `raw_hit_count` 小
- prefilter 后几乎没候选
- 最终进入模型的 template 很少或为 0

### 5.2 template search 完全依赖已有 MSA

当前自动 template 搜索逻辑是:

- 先要有 `pairedMsaPath` 或 `unpairedMsaPath`
- 再从对应目录生成 `hmmsearch.a3m`

见:

- `runner/template_search.py:190-229`

这说明它没有独立的 sequence-only template fallback。  
如果你后续做 RNA template online search, 这一点值得参考:

> 你要明确你的 RNA template search 是不是也依赖上游 RNA MSA/profile, 还是允许纯 query sequence 直接打结构库。

---

## 6. 当前代码审核结论

基于本次核查, 对 protein template 机制的审核结论如下。

### 6.1 已确认的事实

1. Protein template 不使用 `nhmmer`
2. Protein template 不直接复用普通 MSA hits
3. Template 来自 `pdb_seqres` 命中
4. 最终 template 特征来自 mmCIF 结构解析与重对齐
5. 模型消耗的是结构几何特征, 不是纯序列 hit

### 6.2 当前实现中值得注意的点

#### A. inference cutoff 是硬编码日期

在 inference 中, `TemplateHitFeaturizer` 被初始化为:

- `max_template_date = "2021-09-30"`

见:

- `protenix/data/inference/infer_dataloader.py:105-116`

这说明 protein inference 的模板时间策略是固定的。  
如果你给 RNA template 增加 temporal filter, 最好明确:

- 是不是也固定到某个训练 cutoff
- 是否和 protein 保持同一语义
- 是否存在 protein/RNA cutoff 不一致的问题

#### B. 训练与推理 template policy 不完全相同

训练态 `TemplateFeaturizer` 的默认最大日期策略:

- train 普通数据集: `3000-01-01`
- distillation/openprotein: `2018-04-30`

见:

- `protenix/data/template/template_featurizer.py:270-304`

而 inference 里是 `2021-09-30`。  
这说明 temporal policy 在 train / infer 间本来就有分层。  
你为 RNA template 设计 temporal filter 时, 也应该区分:

- 训练集去泄漏
- 推理时模拟官方 benchmark cutoff

#### C. protein template 仍只支持 protein chain

inference template 入口里明确断言:

- `Only protein templates are supported.`

见:

- `protenix/data/template/template_featurizer.py:693-706`

因此你现在给 RNA template 加的逻辑本质上是并行新支路, 而不是扩展已有 protein template parser。

---

## 7. 对 RNA template temporal filter 的直接启发

你现在做的 RNA template temporal filter, 如果要和 protein 机制保持“哲学一致”, 推荐遵循下面原则:

### 7.1 把“搜索召回”和“时间过滤”分层

Protein 路径里:

- 先从结构库召回候选
- 再按日期 cutoff 过滤

RNA 也最好保持这个顺序:

1. 先找 RNA 结构候选
2. 再按 temporal cutoff 过滤
3. 再做重排/特征构造

不要把“没有结构候选”和“有结构但被时间过滤掉”混成一个错误来源。

### 7.2 让 cutoff 语义显式

Protein 使用的是“模板 release date 不得晚于 cutoff”。  
RNA 也应明确:

- 过滤依据是 PDB release date?
- 还是 RNA 结构数据库自己的 deposition date?
- 如果 query 本身来自某个结构条目, 是否还要减去保护窗口, 类似 `DAYS_BEFORE_QUERY_DATE = 60`

Protein 训练态存在:

- `query_release_date - 60 days`

见:

- `protenix/data/template/template_featurizer.py:353-380`
- `protenix/data/template/template_utils.py:24`

如果 RNA 不做类似保护, 就要明确这是设计差异还是潜在泄漏点。

### 7.3 保留“raw candidates”和“selected templates”两个统计

Protein 训练态 profile 中会统计:

- `raw_templates`
- `selected_templates`

见:

- `protenix/data/template/template_featurizer.py:452-533`

RNA template pipeline 也建议保留这两个统计量。  
这样你可以区分:

- 搜索阶段没找到
- temporal filter 删没了
- 结构解析失败删没了

---

## 8. 最终回答

把你的原问题压缩成一句话:

> Protenix 的 protein template 不是 `nhmmer`, 也不是直接拿 MSA hits 当 template。  
> 它是先用 MMseqs2 拿 protein MSA, 再基于 MSA 建 HMM profile, 用 `hmmsearch` 去搜 `pdb_seqres` 结构序列库, 然后再用 mmCIF 结构生成真正的 template 特征。

因此:

- MSA 可以高灵敏度, 因为它只是增强 profile
- template 仍然只来自有结构的库
- 如果灵敏度高但结构覆盖低, 结果是 template 少, 不是把无结构序列误当模板

---

## 9. 下一步建议

下一步最值得做的是对你当前 RNA template temporal filter 代码进行一次“与 protein policy 对齐”的专项审核, 重点检查:

1. temporal cutoff 的定义是否清楚
2. train / infer 是否使用相同或合理不同的 cutoff
3. 过滤发生在搜索前还是搜索后
4. 是否可能出现时间泄漏
5. 是否能区分“无候选”和“候选被过滤光”这两类失败

如果继续做下一轮报告, 建议优先核查这些文件:

- `protenix/data/inference/infer_dataloader.py`
- `protenix/data/rna_template/rna_template_featurizer.py`
- `rna_template/scripts/03_search_and_index.py`
- 你新增的 temporal filter 相关逻辑文件

