# RNA Template Online Pipeline 改造方案

**Date**: 2026-03-15
**Status**: 方案设计（未修改代码）

# RNA Template Temporal Filtering — Code Review Report

**Date**: 2026-03-15
**Reviewer**: Claude Opus 4.6 (codex-code-review + codex-bio-ai-research)
**Scope**: Read-only audit of temporal filtering implementation against requirements
**Status**: Requirements substantially met, with actionable hardening recommendations

---

## 0. Executive Summary

The RNA template temporal filtering pipeline **successfully implements the core requirements**: per-query runtime self-hit exclusion and temporal date filtering that mirrors the protein template pipeline's approach. GPU training has been verified on H800 with filtering actively rejecting self-hits and future templates. The implementation is architecturally sound, isolated from protein/LLM pipelines, and correctly integrated into both training and inference paths.

---
Prob1

#### H1: Unknown-Date Templates Kept Instead of Rejected

**Location**: `rna_template_featurizer.py`, `_filter_candidates()` lines ~322-330

**Problem**: When a template's PDB ID is not found in RNA3DB metadata (`_release_dates`), the template is **kept** and counted as `no_date`. 

**Risk**: If RNA3DB metadata is incomplete or if new templates are added to the index without corresponding metadata entries, those templates bypass temporal filtering entirely. This is a **silent data leakage vector**.


TODO: for the sturctures unfound, please help me to use PDB API to check for release date, because RNA3DB may not be comphrehenseive and update the maintained release date file to keep track
---

Prob2
---

## 2. Protein Template Pipeline（参考标准）
TODO:!!!!
```
OFFLINE（一次性）:
  hmmsearch → .a3m / .hhr 文本文件
  ★ 只是搜索结果（hit 列表），不构建任何 3D feature

ONLINE（每个 training sample 实时）:
  TemplateFeaturizer.get_template():
    1. source_mgr.fetch_template_paths() → 读取 .a3m/.hhr
    2. HHRParser.parse() → N 个 TemplateHit 对象
    3. cutoff = min(max_template_date, query_release_date - 60 days)
    4. TemplateHitFilter.prefilter() → 逐个 hit：
       ├── release_date > cutoff? → REJECT
       ├── alignment quality 太差? → REJECT
       ├── 是 query 的 duplicate? → REJECT
       └── PDB date unknown? → REJECT
    5. 排序 (by sum_probs) → 去重 → shuffle top-k
    6. TemplateHitProcessor.process() → 逐个 hit：
       ├── 读取 mmCIF → 解析 3D 坐标
       ├── kalign alignment (query vs template)
       └── 构建 feature dict [N_query 维度]
    7. 收集到 max_hits(4) 个成功的就停
```

**核心特征**：
- Offline 只存文本搜索结果，**零 3D 预计算**
- Filter 在 feature 构建**之前**，被 reject 的不浪费构建时间
- 被 reject 后自动尝试下一个 hit，**能回填**
- `build_minimal_template_arrays()` 需要 `query_seq` 做 alignment → 输出 `[N_query]` → **必须 online 做**

---

## 2. 当前 RNA Pipeline 的问题

```
OFFLINE:
  MMseqs2 → search_results.json → 02_build_rna_templates.py → NPZ (3D 已 stack)
  ★ NPZ 是 per-query 预构建的，包含 alignment 后的 [T, N_query, ...] feature

ONLINE:
  index lookup → filter NPZ 文件名 → load 第一个通过的 NPZ → features
```

| 问题 | 说明 |
|------|------|
| **Filter 粒度 = NPZ 文件** | NPZ 内 stack 了 4 个不同 PDB 的 template，filter 只看文件名，无法逐个操作 |
| **被 filter 后不能回填** | NPZ 被 reject 后没有备选结构可加载 |
| **Offline cutoff 是全局的** | `--release_date_cutoff` 对所有 sample 一刀切，不是 per-query |
| **alignment 已固化** | NPZ 内的 feature 是按构建时的 query 做的 alignment，如果 query 变了（cropping 等）无法重新 align |

---

## 3. 目标架构（完全对齐 Protein）

```
OFFLINE（一次性，轻量）:
  03_mmseqs2_search.py → search_results.json (hit list + metadata)
  ★ 不构建 NPZ，不构建 3D feature
  ★ max_templates 设大（如 20），给 online filter 足够的候选

ONLINE（每个 training sample 实时）:
  RNATemplateFeaturizer.__call__():
    1. 从 search index 查找 sequence → hit list
    2. 计算 cutoff = query_release_date - 60 days
    3. Per-hit filter:
       ├── self-hit (hit.pdb_id == query_pdb_id)? → REJECT
       ├── temporal (release_date > cutoff)? → REJECT
       └── date unknown? → REJECT (保守)
    4. 排序 (by identity/bitscore) → 取 top 候选
    5. Per-hit 构建 3D feature（复用现有函数）:
       ├── load_structure_residues(cif_path, chain_id)  ← 已有
       ├── build_minimal_template_arrays(query_seq, residues, ...)  ← 已有
       └── 失败? → 跳过，尝试下一个 hit
    6. 收集到 max_templates(4) 个成功的就停
    7. stack_template_dicts() → [T, N_query, ...] features  ← 已有
```

---

## 4. 与 Protein Pipeline 的对应关系

| Protein Pipeline 组件 | RNA Pipeline 对应 | 现有代码位置 |
|----------------------|-------------------|-------------|
| `TemplateSourceManager.fetch_template_paths()` → 读 .a3m/.hhr | 读 `search_index.json` → hit list | `03_mmseqs2_search.py` 输出 |
| `HHRParser.parse()` → TemplateHit 对象 | JSON parse → hit dict `{pdb_id, chain_id, identity, ...}` | `search_results.json` 已有 |
| `TemplateHitFilter.prefilter()` → per-hit filter | `_filter_hit()` → per-hit filter | **新写，逻辑简单** |
| `TemplateHitProcessor.process()` → 读 mmCIF + align + 构建 feature | `load_structure_residues()` + `build_minimal_template_arrays()` | `rna_template_common.py` **已有** |
| `stack_template_dicts()` → 堆叠 | 完全相同 | `rna_template_common.py` **已有** |
| `TemplateHitFeaturizer.get_templates()` → 主循环 | `_build_templates_online()` → 主循环 | **新写，参考 protein** |
| `template_cache_dir` → 缓存 mmCIF 解析结果 | `template_cache_dir` → 缓存 per-PDB+chain NPZ | **新加** |
| `max_template_date` → 全局 cutoff | `max_template_date` 不再需要（per-query cutoff 足够） | — |

---

## 5. 可复用的现有代码

### 5.1 完全不改，直接复用

| 函数/文件 | 位置 | 用途 |
|-----------|------|------|
| `load_structure_residues()` | `rna_template_common.py:238` | 从 CIF 加载残基 ResidueRecord 列表 |
| `build_minimal_template_arrays()` | `rna_template_common.py:424` | query-template alignment + 构建 feature dict |
| `stack_template_dicts()` | `rna_template_common.py:481` | 堆叠多个 template 为 [T, N, ...] |
| `normalize_query_sequence()` | `rna_template_common.py:176` | 序列规范化 (T→U 等) |
| `_load_rna_release_dates()` | `rna_template_featurizer.py:51` | 加载 RNA3DB metadata |
| `_empty_rna_template_features()` | `rna_template_featurizer.py:133` | 空 feature fallback |
| `find_cif_path()` | `02_build_rna_templates.py:70` | CIF 文件查找（flat/nested） |
| `get_rna_template_features()` 后半段 | `rna_template_featurizer.py:449-517` | token-level 组装（entity mask, block mask 等） |
| `03_mmseqs2_search.py` | `scripts/` | MMseqs2 搜索 → search_results.json |

