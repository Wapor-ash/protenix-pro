# RNA Template Search Pipeline 设计建议

日期: 2026-03-16

目标:
- 为当前 Protenix RNA template 支路设计一套更合理的 search pipeline
- 兼顾生物学合理性、工程复杂度、运行效率和后续 temporal filtering
- 明确 `MMseqs2`、`nhmmer`、`cmbuild/cmsearch` 在 RNA template 检索中的定位

适用范围:
- 你当前的 RNA template 训练/推理支路
- 当前仓库已有的 online RNA template featurizer
- 你手头已经可用的 `MMseqs2`、`nhmmer`、`cmbuild`、`cmsearch`

---

## 1. 先说结论

如果只给一个推荐:

> **最推荐的现实方案是三段式混合检索**
>
> `MMseqs2 coarse retrieval -> nhmmer rerank/filter -> online CIF build + temporal/self-hit filter`

原因:

1. `MMseqs2` 负责高吞吐粗召回, 速度最好
2. `nhmmer` 负责对 top-K 候选做 RNA-specific profile 复筛, 比纯 sequence identity 更稳
3. 你当前的 online featurizer 已经支持把“真正的时间过滤、自身泄漏过滤、CIF 构建”放到训练时逐 hit 执行, 这和 protein pipeline 的思路一致

如果资源更多、目标更高:

> **长期最优方案是 family-aware / structure-aware 路线**
>
> `family assignment -> cmbuild/cmsearch (or prebuilt CM database) -> online CIF build`

但这个方案的前提更重:

- 你要有可靠 family / clan / seed alignment
- 或至少能把 template 库组织成可维护的 CM 集合
- 否则工程成本会明显高于收益

---

## 2. 你当前代码已经是什么状态

你当前 RNA template 支路已经不是一个纯粹的“离线 NPZ 查表”系统了，而是已经部分走向 protein 那种在线模板构造逻辑。

### 2.1 当前在线模式已经很像 protein pipeline

`RNATemplateFeaturizer` 的在线模式说明非常明确:

- 先 `query sequence -> hit list`
- 再做 per-hit 过滤
- 再逐 hit 读 CIF 构造 3D template
- 最后取成功构建的前 `max_templates`

证据:

- `protenix/data/rna_template/rna_template_featurizer.py:5-22`
- `protenix/data/rna_template/rna_template_featurizer.py:765-840`

### 2.2 当前 search 入口实际上是 MMseqs2 baseline

当前 `03_mmseqs2_search.py` 已经把 `MMseqs2` 定位为 production mode:

- `rna_template/scripts/03_mmseqs2_search.py:5-18`
- `rna_template/scripts/03_mmseqs2_search.py:876-882`

它现在做的是:

1. 把 query 和 template catalog 都建成 nucleotide MMseqs2 db
2. 跑 `mmseqs search`
3. 转换结果
4. 解析出每个 query 的 top template hits

证据:

- `rna_template/scripts/03_mmseqs2_search.py:352-537`

### 2.3 当前 online featurizer 负责真正的“模板合法性”

当前合法性检查是在在线构造阶段做的，而不是完全在 search 阶段做死:

- self-hit exclusion
- temporal cutoff
- PDB API fallback
- CIF 构造失败跳过

证据:

- `protenix/data/rna_template/rna_template_featurizer.py:642-684`
- `protenix/data/rna_template/rna_template_featurizer.py:728-761`
- `protenix/data/rna_template/rna_template_featurizer.py:791-840`

这其实是正确方向，因为 RNA template 的“最终可用性”不能只靠序列搜索分数判断。

---

## 3. RNA 为什么不该长期停留在纯 direct sequence-based

这是设计决策最核心的一点。

### 3.1 RNA 单序列信息量低于 protein

RNA 常见主字母只有 `A/G/C/U`，sequence alphabet 比 protein 小很多。  
因此只靠单序列相似性，容易遇到:

- 远缘同源召回不足
- 低复杂度序列假阳性
- 短 RNA 或重复 motif 误命中

### 3.2 RNA 结构保守性常高于一级序列保守性

很多 RNA family 的关键保守信号不只在一级序列, 而在:

- 配对模式
- 协变位点
- motif 局部结构
- loop / stem / junction 组织

这也是为什么对 RNA 来说:

- `pairwise identity`
- 甚至普通 sequence search

都经常不够。

### 3.3 所以 RNA 检索最自然的排序是

从生物学信号强度上看, 通常是:

1. **纯 sequence direct search**
2. **profile-based nucleotide search**
3. **covariance / structure-aware family model**

因此, 在你的工具集合里:

- `MMseqs2` 适合做粗召回
- `nhmmer` 适合做 RNA sequence-profile 复筛
- `cmbuild/cmsearch` 更适合做高质量 family-aware 检索

---

## 4. 方案总览

这里给四个可落地方案, 从简单到复杂。

| 方案 | 搜索主干 | 生物学合理性 | 工程复杂度 | 运行速度 | 推荐等级 |
| --- | --- | --- | --- | --- | --- |
| A | `MMseqs2` 直接搜 template catalog | 中 | 低 | 很高 | 可做 baseline |
| B | `MMseqs2 -> nhmmer rerank` | 高 | 中 | 高 | 最推荐 |
| C | `nhmmer` 全库主搜 | 中高 | 中 | 中低 | 小库可用 |
| D | `CM / cmsearch` family-aware 主搜 | 最高 | 高 | 低到中 | 长期最优 |

---

## 5. 方案 A: MMseqs2 直接搜 template catalog

### 5.1 流程

`query RNA sequence -> MMseqs2 search against template catalog -> top-N hits -> online CIF build -> temporal/self-hit filter`

### 5.2 优点

1. 你现在已经有实现
2. 批量处理速度最好
3. 工程改动最小
4. 和现有 `search_results.json -> online featurizer` 完全兼容

### 5.3 缺点

1. 排序主要依赖 sequence identity / bitscore
2. 对远缘 RNA family 的召回有限
3. 对结构保守但序列漂移的家族不够友好
4. 对短序列、重复 motif 容易需要更多后处理

### 5.4 适用场景

- 你想先快速稳定跑通大规模 pipeline
- 当前模板库不是特别大
- 你更关心吞吐而不是极致远缘召回

### 5.5 对现有代码的映射

这个就是当前主路径:

- `rna_template/scripts/03_mmseqs2_search.py:352-537`
- `protenix/data/rna_template/rna_template_featurizer.py:765-840`

### 5.6 我的判断

这个方案适合作为 **baseline**，但不建议作为最终上限方案。

---

## 6. 方案 B: MMseqs2 coarse retrieval + nhmmer rerank

这是我最推荐的折中方案。

### 6.1 核心思想

不要让 `nhmmer` 去扫全库；先让 `MMseqs2` 快速召回 top-K 候选，再用 `nhmmer` 对这些候选做更稳的 RNA profile 复筛。

流程:

1. `MMseqs2` 在全 template catalog 上召回每个 query 的 top `K1`
2. 为 query 构建 RNA profile
3. 用 `nhmmer` 在候选子库上复筛 / 重排
4. 输出 reranked `search_results.json`
5. 进入现有 online featurizer 做 temporal/self-hit/CIF build

### 6.2 为什么这个方案最平衡

#### 速度层面

- 全库扫描交给 `MMseqs2`
- `nhmmer` 只看候选子集

这样不会像全库 `nhmmer` 那样慢。

#### 生物学层面

`nhmmer` 的 profile 复筛比直接 identity 排序更接近“RNA 同源模板检索”的目标。

#### 工程层面

你不需要推翻当前架构。  
只需要在 `03_mmseqs2_search.py` 后面加一个 rerank stage。

### 6.3 `nhmmer` 在这里怎么用才合理

这里有两种子模式。

#### B1. 单 query 序列直接跑 `nhmmer`

这其实收益有限。  
如果 query 只有单条序列，没有扩展 profile，那么 `nhmmer` 和普通 sequence search 的差距未必很大。

不推荐作为主版本。

#### B2. 先为 query 构建 RNA MSA / profile，再跑 `nhmmer`

更合理的版本是:

1. 先为 query 准备 RNA MSA
2. 用 `hmmbuild` 生成 query HMM
3. 再对 `MMseqs2` 召回的候选序列子库跑 `nhmmer` / `hmmsearch` 风格复筛

如果你的 RNA MSA 质量还可以，这个会明显比直接按 identity 排 top hits 更稳。

### 6.4 推荐参数策略

一个比较稳的工程参数组合:

- `MMseqs2`:
  - `sensitivity = 7.5`
  - `evalue = 1e-3` 或更松
  - `K1 = 50 ~ 200`
- `nhmmer` rerank:
  - 只在 top-K 候选子库上跑
  - 输出前 `K2 = 10 ~ 20`
- online CIF build:
  - 最终保留成功构建的 `max_templates = 4`

### 6.5 优点

1. 速度和效果平衡最好
2. 不需要维护完整 CM 数据库
3. 与现有 `search_results.json` 格式兼容
4. 适合逐步替换当前 `MMseqs2-only` baseline

### 6.6 缺点

1. 需要额外维护一个 rerank stage
2. 如果 query 没有足够好的 RNA MSA, rerank 的增益会受限
3. 仍然不是 full structure-aware family model

### 6.7 我对这个方案的推荐结论

> **这是最适合你当前仓库演进的主方案。**

原因是它和你现有设计最兼容，同时能真正把 RNA 的 profile-based 优势引入进来。

---

## 7. 方案 C: nhmmer 全库主搜

### 7.1 流程

`query (or query profile) -> nhmmer against all RNA template sequences -> top-N -> online CIF build`

### 7.2 什么时候值得用

适合:

- template 数据库还不大
- 你想尽量统一到 HMMER 家族工具链
- 你已经有较好 RNA MSA / HMM 生成流程

### 7.3 优点

1. 比纯 pairwise / identity 更 RNA-aware
2. 检索逻辑更接近 protein pipeline 的 profile search 哲学
3. 工具链简单, 都在 HMMER/Infernal 家族附近

### 7.4 缺点

1. 全库速度明显慢于 `MMseqs2`
2. 数据库扩大后吞吐压力会比较大
3. 如果 query 只有单序列, 收益也不一定足够大

### 7.5 我的判断

如果你的 template 库规模中等, 这个方案可以作为研究版对照实验。  
但从生产角度, 我仍然更倾向方案 B 而不是 C。

---

## 8. 方案 D: cmbuild/cmsearch family-aware 主搜

这是长期上限最高、但前提也最重的方案。

### 8.1 核心思想

RNA 最强的检索往往不是 sequence-only, 而是 family-aware covariance model。

理想流程:

1. 给 template 库建立 family / clan / motif 组织
2. 为每个 family 准备高质量 seed alignment
3. `cmbuild` 构建 CM
4. `cmsearch` 将 query 搜到 family / template clusters
5. 在 family 内再落到具体 PDB chain
6. online CIF build

### 8.2 生物学优势

这是最能利用 RNA 特性的路线, 因为它能显式利用:

- 协变
- 配对保守性
- family motif
- 结构上下文

### 8.3 工程难点

1. 你必须先定义 template family
2. 需要 seed alignment 或可靠的 family clustering
3. 数据维护成本明显高
4. 更新 template 库时需要同步更新 CM 资产

### 8.4 什么时候值得做

适合:

- 你后续想把 RNA template 检索做到真正研究级别
- 目标 RNA 家族相对可定义
- 你有较稳定的 catalog 和 family 维护能力

### 8.5 我的判断

> 这个方案不是现在最合适的第一步，但很适合作为长期目标架构。

---

## 9. 推荐的最终架构

这里给出我认为最合理的分层式架构。

### 9.1 主推荐: Hybrid Tiered Retrieval

#### Stage 0. 数据准备

- template catalog: 结构库中的 RNA chain 序列和元数据
- per-template CIF
- release date metadata
- 可选: template family label / cluster ID

#### Stage 1. 全库粗召回

工具:

- `MMseqs2`

输出:

- 每个 query 的 top `K1` 候选

理由:

- 快
- 易扩展
- 适合做全库第一层

#### Stage 2. RNA-aware 复筛

优先推荐:

- `nhmmer`

长期升级:

- `cmsearch`

输出:

- top `K2` reranked hits

理由:

- 让 RNA template 排序不再只由 identity 主导

#### Stage 3. 结构合法性验证

工具:

- 当前 online featurizer

流程:

- self-hit exclusion
- temporal filtering
- PDB API fallback
- CIF load
- minimal template array 构造
- 成功构造的前 `max_templates`

这个阶段你已经基本实现了:

- `protenix/data/rna_template/rna_template_featurizer.py:642-684`
- `protenix/data/rna_template/rna_template_featurizer.py:728-840`

### 9.2 这套方案的关键优点

1. 把“快”和“准”拆层处理
2. 不会让 `cmsearch` 或 `nhmmer` 直接承担全库吞吐压力
3. 与现有 Protenix online build 逻辑兼容
4. temporal filtering 继续放在训练时 per-hit 执行, 能最大程度减少泄漏

---

## 10. 针对你当前仓库的具体建议

### 10.1 近期版本

保留当前:

- `03_mmseqs2_search.py`
- `search_results.json`
- `RNATemplateFeaturizer` online mode

只新增一个中间 rerank stage:

`03_mmseqs2_search.py -> 03b_nhmmer_rerank.py -> search_results_reranked.json`

然后在配置里让:

- `rna_template.search_results_path` 指向 reranked 版本

这会是改动最小、收益最大的升级。

### 10.2 中期版本

把 `search_results.json` 的 hit schema 扩展为更丰富格式，例如:

```json
{
  "query_id": {
    "query_sequence": "...",
    "templates": [
      {
        "pdb_id": "1abc",
        "chain_id": "A",
        "coarse_method": "mmseqs2",
        "coarse_identity": 0.41,
        "coarse_evalue": 1e-5,
        "rerank_method": "nhmmer",
        "rerank_score": 87.3,
        "rerank_evalue": 3e-8,
        "family_id": "RFxxxxx"
      }
    ]
  }
}
```

这样后面:

- debugging
- temporal leakage audit
- error analysis

都会容易很多。

### 10.3 长期版本

如果你后面模板库稳定、RNA family 管理成熟，再逐步升级到:

- `MMseqs2 coarse -> family assignment -> cmsearch rerank`

而不是第一天就把整条链重写成 full CM pipeline。

---

## 11. 不推荐的方案

### 11.1 不推荐长期只用 pairwise identity

原因:

- 对 RNA 远缘同源不友好
- 容易被序列长度和低复杂度影响
- 不能体现 RNA 家族的结构保守性

当前仓库里 `pairwise` 更像 legacy fallback:

- `rna_template/scripts/03_mmseqs2_search.py:879-881`

### 11.2 不推荐一开始就 full `cmsearch` 扫全库

原因:

- 工程负担过高
- 数据准备重
- 维护成本大
- 你现在已有的 online build/temporal 机制还没完全沉淀稳定

---

## 12. 最终建议

### 12.1 如果你问“现在最该做哪个版本”

答案是:

> **方案 B: `MMseqs2 coarse retrieval + nhmmer rerank + current online build`**

这是当前复杂度、速度、效果三者之间最好的平衡点。

### 12.2 如果你问“长期最强版本”

答案是:

> **family-aware `cmbuild/cmsearch` 路线**

但前提是:

- 你愿意维护 family/seed/CM 资产
- 你要把 RNA template 检索做成长期基础设施

### 12.3 如果你问“现在能否直接只用 MMseqs2”

可以，而且你当前已经在这么做。  
但它更像是:

> **高效 baseline**

而不是:

> **RNA template search 的最终形态**

---

## 13. 一个简洁决策图

### 情况 A: 我要最快上线

选:

- `MMseqs2 only`

### 情况 B: 我要速度和效果都要

选:

- `MMseqs2 -> nhmmer rerank`

### 情况 C: 我要做长期研究级 RNA template 基础设施

选:

- `MMseqs2 -> family-aware CM rerank`
  或
- `CM-first pipeline`

---

## 14. 下一步实施建议

最建议的下一步不是大改主流程，而是新增一个最小 rerank 模块:

1. 复用 `03_mmseqs2_search.py` 生成 top-K 候选
2. 对每个 query 的候选子集构建 FASTA 子库
3. 用 `nhmmer` 对子库 rerank
4. 输出新的 `search_results.json`
5. 直接接入当前 `RNATemplateFeaturizer` online mode

如果继续推进, 最值得实现的就是这个文件:

- `rna_template/scripts/03b_nhmmer_rerank.py`

它的职责应该很单纯:

- 输入: `search_results.json`, query sequences, candidate catalog
- 输出: `search_results_reranked.json`

而不是把 temporal filter / CIF build 也揉进去。

