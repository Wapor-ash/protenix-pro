# RNA Template Resolver 融合风险审查

日期: 2026-03-16

范围:
- 审查 `RNA Template Resolver` 方案与当前 Protenix RNA template 框架融合时的漏洞、边界问题和潜在失败模式
- 基于当前代码实现判断哪些设计假设会被打破
- 给出收缩范围后的最稳实施建议

结论:

> `resolver` 方案方向是对的，但如果直接把“manual template / RNAJP / hybrid fallback”接入当前框架，最大的风险不是 search pipeline，而是**现有 RNA template 实现隐含地假设了模板是 entity 级、完整链级、PDB-like hit 级别来源**。  
> 如果这些假设不先收紧，resolver 很容易在语义上比底层实现更强，最后造成“设计支持、实现不支持”的假象。

---

## 1. 最高风险问题

### 1.1 当前 RNA template 实现实际上是 `entity` 级，不是 `copy` 级

如果 resolver 后面支持:

- 同一个 `rnaSequence.count > 1`
- 不同 copy 指向不同 manual template

那当前框架会错误地把模板施加到这个 entity 的所有 copy。

原因:

1. 一个 polymer entity 的多个 copy 会在构建 atom array 时共享同一个 `label_entity_id`
2. 只是通过 `copy_id` 区分不同 copy
3. 但 RNA template 写回时只按 `entity_id` 选 token, 没有使用 `copy_id`

相关代码:

- copy 展开:
  - [json_to_feature.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/inference/json_to_feature.py#L104)
  - [json_to_feature.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/inference/json_to_feature.py#L121)
- template 写回:
  - [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L879)
  - [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L919)

失败模式:

- 用户只想给 copy 2 一个 RNAJP 模板
- resolver 解析成功
- 下游却把模板写到了同 entity 的所有 copy 上

建议:

> 第一版 resolver 明确只支持 **entity 级 manual template**，不支持 copy 级。

---

### 1.2 当前 offline/manual `npz` 路径并不真正支持“多 manual template 合并”

resolver 设计里如果允许:

- 多个 `manual_npz`
- 多个 `manual_structure`
- `hybrid` 占多个 manual slots

当前下游并不能完整承接。

原因:

offline 分支虽然会遍历多个候选 `npz_path`，但实际逻辑是:

1. 依次尝试加载
2. 第一个成功加载的模板立刻 `break`
3. 之后不再继续组合

相关代码:

- [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L857)
  到
- [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L871)

失败模式:

- resolver 产出 3 个 manual NPZ 候选
- 文档说 manual templates 会占多个 slot
- 实际只会用第一个成功文件

建议:

> 在实现 resolver 前，先明确当前版本 manual 路径只支持:
>
> - `manual_npz`: 单一最终模板文件
> - `manual_structure`: 单一最终结构输入
>
> 不要在第一版承诺“多 manual template 堆叠”。

---

### 1.3 “手工结构模板”不能直接伪装成现有 online hit

当前 online builder `_build_single_template_online()` 假设输入是一个标准 search hit:

- 有 `pdb_id`
- 有 `chain_id`
- 能用 base PDB 去 `cif_database_dir` 中查找结构文件

相关代码:

- [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L371)
  到
- [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L446)

这里最大的隐含前提是:

> 模板来源是“可被标准化为 PDB-like 命中”的。

但 manual structure / RNAJP 结果通常是:

- 任意路径
- 任意文件名
- 不一定在 `cif_database_dir`
- 不一定有可靠的 `pdb_id`
- 甚至可能不是标准 PDB archive 条目

失败模式:

- resolver 把 manual CIF 包装成 fake hit
- builder 试图把它压成 base PDB 去查库
- 最终找不到文件或错误使用别的同名 PDB 结构

建议:

> `manual_structure` 必须走独立 builder contract，不能复用 search-hit contract。

也就是说:

- search hit builder
- manual structure builder

要分开。

---

## 3. 中等风险问题

### 3.1 JSON schema drift 风险

当前公开输入格式文档没有 RNA template 的手工覆盖字段:

- [infer_json_format.md](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/docs/infer_json_format.md)

而 inference 实际上会把原始 `bioassembly_dict["sequences"]` 传给后续 RNA template 路径:

- [infer_dataloader.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/inference/infer_dataloader.py#L333)
  到
- [infer_dataloader.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/inference/infer_dataloader.py#L341)

这意味着:

- 新字段短期可能“能跑”
- 但 CLI、web service、文档、样例 JSON、校验逻辑不一定同步

失败模式:

- 训练 JSON 使用 `templateHints`
- 文档没写
- 推理样例/自动生成工具不支持
- 后面协作者不知道字段是正式接口还是私有扩展

建议:

> 一旦决定加入 `templateHints`，就同步更新:
>
> - JSON 格式文档
> - 示例输入
> - 校验逻辑
> - CLI 帮助信息

---

### 3.2 `resolver` 设计可能比当前实现“强太多”

设计文档里 resolver 容易写成一个很强的对象:

- 多 provider
- 多 manual templates
- hybrid
- RNAJP adapter
- future `nhmmer/cmsearch`

但当前底层实现实际支持度有限:

- entity 级，不是 copy 级
- 完整链级，不是局部级
- offline 单模板优先，不是多模板堆叠
- PDB-like hit 语义较重

失败模式:

- 文档上 resolver 看起来像一个总调度器
- 第一版实现却只是“读一个手工模板，否则走默认搜索”
- 后续用户误以为更多语义已经被底层保证

建议:

> 把 resolver 的第一版 contract 写得非常窄:
>
> - entity 级
> - 单 manual template
> - 单 default fallback
> - slot 级混合最多一层

---

## 4. 当前最稳的收缩方案

如果要避免上面的风险，最稳的第一版应该是:

### 4.1 明确能力边界

第一版只支持:

1. `entity` 级 manual override
2. `manual_npz`
3. `manual_structure`
4. `manual_only`
5. `prefer_manual`
6. `hybrid` 但仅限:
   - manual template 放前 slots
   - fallback template 放后 slots

第一版明确不支持:

1. copy 级 override
2. 区间级 fallback
3. 多 manual template 真正堆叠
4. 无 provenance 的 external job template

### 4.2 明确 manual 来源要求

建议 resolver 对 manual/external 输入强制要求:

- `template_id`
- `source_type`
- `path`
- `provenance`


建议把 slot 顺序写死:

- `manual slots` 在前
- `fallback slots` 在后

并保证训练/推理完全一致。

---

## 5. 最终判断

`resolver` 方案本身没有根本性错误。  
真正的漏洞在于:

> 它很容易让设计层支持的语义，超过当前 RNA template 底层真正支持的语义。

当前最危险的三个点是:

1. **entity/copy 语义不一致**
2. **manual structure 不能伪装成标准 hit**
3. **manual/external 来源没有清晰 temporal/provenance 规则**

如果这些点不先收紧，resolver 很容易在接口上“看起来非常灵活”，但实际落地时产生:

- silent mis-application
- silent leakage
- silent template dropping

---
