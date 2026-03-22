# RNA Template 手动指定 + 默认 Pipeline 回退 设计方案

日期: 2026-03-16

目标:
- 在你当前 Protenix RNA template 框架上, 增加“用户可手动指定 RNA 模板”的能力
- 支持离线步骤和外部工具产物, 包括但不限于 RNAJP、外部 CIF/PDB、预计算 NPZ
- 当某些 RNA 链没有提供手动模板时, 自动回退到默认 pipeline
- 当前默认 pipeline 以 `MMseqs2 -> search_results.json -> online CIF build` 为主

---

## 1. 结论摘要

我建议你**不要直接把“手动模板”和“默认 pipeline”写死在 `RNATemplateFeaturizer` 里互相覆盖**，而是新增一层清晰的中间抽象:

> **RNA Template Resolver**

这层的职责是:

1. 接收输入 JSON 中对 RNA 模板的显式指定
2. 接收全局默认搜索配置
3. 统一输出“每条 RNA 链、每个模板来源”的候选模板描述
4. 再交给现有 `RNATemplateFeaturizer` 去做:
   - temporal/self-hit filter
   - CIF 读取
   - 单模板构建
   - 多模板堆叠
   - 缺失链级 fallback

一句话概括:

> **把“找哪些模板”与“怎么把模板变成特征”分开。**

这是最稳、最容易扩展、也最适合你后面接 RNAJP / CM / nhmmer / 手动 CIF 的方式。

---

## 2. 为什么当前架构适合这样改

你当前的 RNA template 系统已经具备两个关键基础:

### 2.1 已有全局 RNA template 开关和 online/offline 双模式

当前配置里 `rna_template` 已支持:

- 离线 `template_database_dir + template_index_path`
- 在线 `search_results_path + cif_database_dir`

证据:

- `configs/configs_base.py:137-162`
- `protenix/data/inference/infer_dataloader.py:194-243`

### 2.2 当前 `RNATemplateFeaturizer` 已经是按 hit 在线构建

它当前在线模式流程是:

1. sequence 查 `search_results.json`
2. per-hit 过滤
3. CIF 构建模板
4. collect top successful templates

证据:

- `protenix/data/rna_template/rna_template_featurizer.py:5-22`
- `protenix/data/rna_template/rna_template_featurizer.py:765-840`

所以你现在并不缺“模板构建器”，你缺的是:

> **一个更灵活的模板来源协调器**

---

## 3. 你现在要支持的来源, 应该统一纳入哪几类

我建议把 RNA 模板来源统一抽象成 4 类 `source_type`。

### 3.1 `manual_npz`

用户直接给已经构建好的模板特征文件:

- `.npz`
- 已符合 Protenix RNA template 特征格式

适合:

- 你已经跑完 `build_rna_template_protenix.py`
- 或外部离线流程已经生成最终 template tensor

### 3.2 `manual_structure`

用户直接给结构文件:

- `.cif`
- `.pdb`
- RNAJP / Arena / Rosetta / 其他外部工具生成的全原子结构

系统随后调用你已有的 builder:

- 例如 `build_rna_template_protenix.py`
- 或 `rna_template_common.py` 中的构建逻辑

适合:

- 用户有“我就是想用这个结构当模板”的明确意图
- 不想先手工转成 `.npz`

### 3.3 `external_job`

用户给的是一个外部离线步骤的目录或 manifest, 比如:

- RNAJP 输出目录
- 某个 docking/refinement 工作目录
- 你自己定义的 job metadata JSON

系统先执行一个 adapter, 将这些结果转换成:

- `manual_structure`
  或
- `manual_npz`

### 3.4 `default_search`

没有给手动模板时, 使用默认 pipeline:

- 当前默认: `MMseqs2`
- 将来可替换为:
  - `MMseqs2 -> nhmmer rerank`
  - `nhmmer`
  - `cmsearch`

---

## 4. 最核心设计: Resolver 层

### 4.1 新增模块职责

建议新增一层逻辑, 名字可以是:

- `RNATemplateResolver`
  或
- `RNATemplateProviderManager`

推荐放置位置:

- `protenix/data/rna_template/rna_template_resolver.py`

### 4.2 Resolver 的输入

Resolver 的输入应包括:

1. 当前样本的原始 input JSON
2. 当前全局配置 `configs.rna_template`
3. query 级元信息:
   - `query_pdb_id`
   - `query_release_date`
4. 当前 RNA chains 的 entity / copy / sequence 信息

### 4.3 Resolver 的输出

不要直接输出 tensor, 而是输出**统一候选描述**:

```python
{
  chain_key: [
    {
      "source_type": "manual_npz" | "manual_structure" | "default_search",
      "priority": 0,
      "sequence": "...",
      "template_id": "...",
      "npz_path": "...",
      "structure_path": "...",
      "builder": "protenix" | "rnajp_adapter",
      "metadata": {...}
    },
    ...
  ]
}
```

这样后续的构建器只需要消费统一候选对象，不需要关心它来自:

- 手工 CIF
- RNAJP
- offline NPZ
- search_results

---

## 5. 推荐的 JSON 扩展接口

当前 `infer_json_format.md` 里 `rnaSequence` 还没有 RNA template 专属字段。  
我建议你为 `rnaSequence` 增加一个**可选字段**:

- `templateHints`

为什么不用直接叫 `templatesPath`:

1. protein 的 `templatesPath` 是单文件语义
2. 你这里需要支持:
   - 多来源
   - 多模板
   - fallback
3. `templateHints` 比较中性，不会和 protein 的 `.a3m/.hhr` 语义混淆

### 5.1 最小推荐格式

```json
{
  "rnaSequence": {
    "sequence": "GGGAAAUCC",
    "count": 1,
    "unpairedMsaPath": "/abs/path/rna_msa.a3m",
    "templateHints": {
      "mode": "hybrid",
      "manual_templates": [
        {
          "type": "structure",
          "path": "/abs/path/manual_templates/1abc_A.cif",
          "priority": 0
        }
      ]
    }
  }
}
```

含义:

- `mode = hybrid`
  - 先使用手动模板
  - 如果模板不足, 再回退默认 pipeline

### 5.2 更完整格式

```json
{
  "rnaSequence": {
    "sequence": "GGGAAAUCCCUUUGGG",
    "count": 1,
    "templateHints": {
      "mode": "hybrid",
      "fallback": "default_search",
      "manual_templates": [
        {
          "type": "npz",
          "path": "/abs/path/manual_templates/custom_rna_template.npz",
          "priority": 0,
          "label": "handpicked_npz"
        },
        {
          "type": "structure",
          "path": "/abs/path/manual_templates/rnajp_refined_model.cif",
          "builder": "protenix",
          "priority": 1,
          "label": "rnajp_model"
        },
        {
          "type": "external_job",
          "path": "/abs/path/rnajp_jobs/job_001",
          "adapter": "rnajp",
          "priority": 2,
          "label": "rnajp_job"
        }
      ]
    }
  }
}
```

---

## 6. 回退策略怎么定义才合理

你提的关键需求是:

> 如果某条 RNA 链没给定 RNA 模板, 则默认开启默认 pipeline

### 6.1 链级 fallback

某条 RNA 链:

- 完全没提供手动模板
- 或手动模板全部构建失败

则整条链回退默认 pipeline。

这是必须支持的。

## 7. 我建议的 merge 语义

我建议采用**template slot 级混合**，而不是残基级拼接。

### 7.1 规则 1: 手动模板优先于默认模板

当前版本里，手动模板优先占用预留的 manual slots。

### 7.2 规则 2: fallback 只做链级或 slot 级补充

当前版本先不做残基级补位。  
默认 pipeline 只在“整条链没有可用手动模板”或“手动模板槽位不足”时补充。

### 7.3 规则 3: 不在同一个 template slot 里混来源

更稳的做法是:

1. 手动模板先占前几个 template slots
2. 默认 pipeline 产生的模板占后几个 slots
3. 模型通过 mask 自己融合

这样实现简单，也最符合当前 `[T, N, ...]` 的表示方式。

### 7.4 推荐 slot 组织

例如 `max_rna_templates = 4` 时:

如果 manual 不足:

- 剩余槽位由 fallback 填充

这比在单个模板槽位里做细粒度混写稳定得多，也更符合你当前阶段的复杂度目标。

---

## 8. 推荐的模式定义

建议 `templateHints.mode` 支持 4 种模式。

### 8.1 `manual_only`

只使用用户指定模板。

若模板构建失败:

- 返回空模板
- 不启用默认搜索

适合:

- 做严格对照实验
- 验证某个 handpicked template 的效果

### 8.2 `prefer_manual`

优先使用用户指定模板。  
若手动模板完全不可用, 再回退默认 pipeline。

适合:

- 你有明确想用的模板
- 但不想因为它失败而整个链没模板

### 8.3 `hybrid`

同时允许:

- 手动模板占前 slots
- 默认 pipeline 补后 slots

这是我最推荐的默认模式。

适合:

- 你想整合 RNAJP / 指定 CIF / 默认搜索
- 兼顾人工先验和系统自动召回

### 8.4 `default_only`

忽略手动模板配置, 完全按默认 pipeline 走。

适合:

- 调试
- ablation

---

## 10. 离线步骤应该如何落位

你说“还想做离线的步骤，比如 RNAJP”，这说明你需要明确区分两类离线。

### 10.1 离线搜索

例如:

- 预跑 `MMseqs2`
- 预跑 `nhmmer`
- 预跑 `cmsearch`

输出:

- `search_results.json`

这类仍属于 **default_search provider** 的准备阶段。

### 10.2 离线结构生成/整理

例如:

- RNAJP
- Arena
- refinement
- 外部折叠器

输出:

- `manual_structure`
  或
- `manual_npz`

这类属于 **manual provider**。

这两类不要混成一个字段。

---

## 11. 我推荐的整体架构

### 11.1 新增三层

建议新增三个模块概念。

#### A. `RNATemplateResolver`

负责:

- 读取 JSON 中的 `templateHints`
- 读取全局默认配置
- 汇总 manual/default 候选
- 形成 chain-level candidate list

#### B. `RNATemplateBuilder`

负责:

- 把 `manual_structure` 变成单模板特征
- 把 `manual_npz` 读取成特征
- 调用已有 builder / common functions

#### C. `RNATemplateMerger`

负责:

- 按 slot 组织 manual + fallback templates
- 产出最终 `[T, N, ...]` 特征

### 11.2 现有 `RNATemplateFeaturizer` 的演进方式

不建议推翻重写。  
建议改成:

```text
RNATemplateFeaturizer
  -> resolve candidates
  -> build template records
  -> merge slots
  -> place into global token tensor
```

也就是把它从“单一 search_results 消费者”升级成“多 provider 模板协调器”。

---

## 12. 推荐的配置扩展

除了 JSON 里的 `templateHints`, 全局 config 也建议补一层 provider 配置。

### 12.1 推荐新增配置

```python
"rna_template": {
    "enable": True,
    "default_provider": "mmseqs2_online",
    "manual_template_builder": "protenix",
    "allow_manual_override": True,
    "manual_slot_budget": 2,
    "fallback_slot_budget": 2,
    "search_results_path": "...",
    "cif_database_dir": "...",
    "template_database_dir": "...",
    "template_index_path": "...",
}
```

### 12.2 为什么要有 `manual_slot_budget`

因为这能直接避免手动模板把所有 slots 吃满，导致默认 pipeline 完全没机会参与。

推荐默认:

- `max_rna_templates = 4`
- `manual_slot_budget = 2`
- `fallback_slot_budget = 2`

---

## 13. 推荐的优先级规则

建议把优先级写死成以下顺序:

1. `manual_npz`
2. `manual_structure`
3. `external_job` 经过 adapter 转换后的 manual 结果
4. `default_search`

如果多个人工模板同时存在:

- 按 `priority` 升序
- 再按构建成功与否

如果多个 default templates 存在:

- 保持默认搜索排序

---

## 14. 推荐的最小可行实现

如果你下一步只做一个最小实现，我建议范围如下。

### Phase 1

支持:

1. `rnaSequence.templateHints.manual_templates[].type = "npz"`
2. `rnaSequence.templateHints.manual_templates[].type = "structure"`
3. `mode = manual_only | prefer_manual | hybrid`
4. `fallback = default_search`

先**不做**:

- family-aware merge
- 多外部工具 adapter

### Phase 2

再支持:

1. `external_job`
2. `adapter = "rnajp"`
3. 更完整的日志和来源追踪

### Phase 3

再支持:

1. `MMseqs2 -> nhmmer rerank`
2. `cmsearch` 作为 default provider 的可插拔实现

---

## 15. 最推荐的用户使用形态

### 15.1 用户完全不管模板

```json
{
  "rnaSequence": {
    "sequence": "....",
    "count": 1
  }
}
```

行为:

- 自动走默认 RNA template pipeline

### 15.2 用户明确指定一个手工模板, 失败时自动回退

```json
{
  "rnaSequence": {
    "sequence": "....",
    "count": 1,
    "templateHints": {
      "mode": "prefer_manual",
      "manual_templates": [
        {
          "type": "structure",
          "path": "/abs/path/template.cif"
        }
      ]
    }
  }
}
```

### 15.3 用户指定 RNAJP 结果 + 默认 pipeline 补充

```json
{
  "rnaSequence": {
    "sequence": "....",
    "count": 1,
    "templateHints": {
      "mode": "hybrid",
      "manual_templates": [
        {
          "type": "external_job",
          "path": "/abs/path/rnajp_job_001",
          "adapter": "rnajp"
        }
      ]
    }
  }
}
```

行为:

- RNAJP 结果先占手动模板槽
- 不足的模板位由默认 pipeline 填

---

## 16. 我对这个方案的最终建议

如果只说一句:

> **在你当前框架上，最合理的做法不是“新增一个 if 用户给了模板就完全跳过默认搜索”的硬开关，而是引入一个支持 manual/default/hybrid 三种模式的 RNA Template Resolver。**

这样你能同时得到:

1. 明确可控的手工模板注入
2. 对 RNAJP 等离线工具的自然接入点
3. 与现有 `MMseqs2` 默认 pipeline 的兼容
4. 手工模板失败时的链级自动 fallback
5. 后续接 `nhmmer` / `cmsearch` 的可扩展性

---

## 17. 我最推荐的实现顺序

1. 先加 `templateHints` JSON 接口
2. 先支持 `manual_npz` 和 `manual_structure`
3. 先实现 `prefer_manual` 和 `hybrid`
4. 先按 template slot 合并, 不做细粒度残基拼接
5. 再加 `external_job: rnajp`
6. 最后再把默认 provider 从 `MMseqs2-only` 升级为 `MMseqs2 -> nhmmer rerank`

这条路线最稳，也最符合你当前代码基线。
