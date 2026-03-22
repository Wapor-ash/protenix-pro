# RNA Template Pipeline v3 — Online Mode & PDB API Fallback

**Date**: 2026-03-15
**Author**: Claude Opus 4.6
**Status**: Implemented & GPU Validated

---

## 1. 问题概述

根据 `online_rna_template_pipeline_proposal.md` 中识别的两个核心问题：

### Problem 1: Unknown-Date Templates 静默绕过时间过滤

**位置**: `rna_template_featurizer.py`, `_filter_candidates()`

**问题描述**: 当 RNA3DB metadata (`filter.json`) 中找不到某个 template 的 PDB ID 时，该 template 被**保留**而非**拒绝**。这意味着未在 RNA3DB 中记录的结构会绕过时间过滤，造成**静默数据泄漏**。

**根因**: RNA3DB 并不包含所有 PDB 结构的 release date，metadata 的覆盖率有限。

### Problem 2: Offline → Online 模式转换

**问题描述**: 当前 RNA template pipeline 使用**离线模式**（NPZ 预构建），与 protein template pipeline 的**在线模式**存在根本架构差异：

| 维度 | Protein Pipeline (参考标准) | RNA Pipeline v2 (旧) |
|------|---------------------------|---------------------|
| 搜索结果 | 文本（.a3m/.hhr），零 3D 预计算 | NPZ（已 stack 4 个 template 的 3D feature） |
| Filter 粒度 | Per-hit，可逐个操作 | Per-NPZ 文件，无法逐个操作 |
| 被 reject 后 | 自动尝试下一个 hit（回填） | 无备选，直接放弃 |
| Cutoff | Per-query（query_release_date - 60 days） | 全局一刀切（--release_date_cutoff） |
| Alignment | Online 根据 query 实时构建 | Offline 已固化，不可重新 align |

---

## 2. 解决方案设计

### 2.1 架构对齐 Protein Pipeline

```
v3 Online 模式:
  OFFLINE（一次性，轻量）:
    03_mmseqs2_search.py → search_results.json (hit list + metadata)
    ★ 不构建 NPZ，不构建 3D feature
    ★ max_templates 设大（如 20），给 online filter 足够的候选

  ONLINE（每个 training sample 实时）:
    RNATemplateFeaturizer.__call__():
      1. 从 search_results.json 查找 sequence → hit list
      2. 计算 cutoff = query_release_date - 60 days
      3. Per-hit filter:
         ├── self-hit (hit.pdb_id == query_pdb_id)? → REJECT
         ├── temporal (release_date > cutoff)? → REJECT
         ├── date unknown (RNA3DB miss)? → PDB API 查询
         │   └── API 也查不到? → REJECT (保守策略)
         └── PASS
      4. Per-hit 构建 3D feature:
         ├── _find_cif_path_cached(pdb_base)
         ├── load_structure_residues(cif_path, chain_id)
         ├── build_minimal_template_arrays(query_seq, residues)
         └── 失败? → 跳过，尝试下一个 hit
      5. 收集到 max_templates(4) 个成功 → stack
```

### 2.2 PDB API Fallback

```
Release date 查找优先级:
  1. RNA3DB metadata (filter.json) → 5389 PDBs
  2. PDB API 内存缓存 → O(1) 查找
  3. RCSB PDB REST API → https://data.rcsb.org/rest/v1/core/entry/{pdb_id}
  4. 持久化缓存 → pdb_release_dates_cache.json (避免重复 API 调用)
  5. 全部查不到 → None → REJECT (保守策略)
```

---

## 3. 代码修改详情

### 3.1 核心文件: `rna_template_featurizer.py`

**文件路径**: `protenix/data/rna_template/rna_template_featurizer.py`

完全重写为 v3，支持 Online/Offline 双模式。

#### 3.1.1 PDB API Release Date 查询（Problem 1 修复）

```python
# ── PDB API release-date lookup with persistent cache ────────────────────

_PDB_API_DATE_CACHE: Dict[str, Optional[datetime]] = {}
_PDB_API_CACHE_LOCK = threading.Lock()
_PDB_API_CACHE_PATH: Optional[str] = None


def _fetch_pdb_release_date(pdb_id_4char: str) -> Optional[datetime]:
    """Query RCSB PDB REST API for the release date of a 4-char PDB ID.

    Uses https://data.rcsb.org/rest/v1/core/entry/{pdb_id} and extracts
    rcsb_accession_info.initial_release_date.
    """
    pdb_id_4char = pdb_id_4char.lower()

    with _PDB_API_CACHE_LOCK:
        if pdb_id_4char in _PDB_API_DATE_CACHE:
            return _PDB_API_DATE_CACHE[pdb_id_4char]

    try:
        import urllib.request
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id_4char}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        accession = data.get("rcsb_accession_info", {})
        release_date_str = accession.get("initial_release_date")
        if not release_date_str:
            release_date_str = accession.get("deposit_date")

        if release_date_str:
            dt = datetime.strptime(release_date_str[:10], "%Y-%m-%d")
            with _PDB_API_CACHE_LOCK:
                _PDB_API_DATE_CACHE[pdb_id_4char] = dt
                _save_pdb_api_cache()
            return dt

    except Exception as e:
        logger.debug(f"PDB API error for {pdb_id_4char}: {e}")

    # Cache failure too
    with _PDB_API_CACHE_LOCK:
        _PDB_API_DATE_CACHE[pdb_id_4char] = None
        _save_pdb_api_cache()
    return None
```

**关键特性**:
- 线程安全（`threading.Lock`）
- 持久化缓存到 JSON 文件（避免重复 API 调用）
- 失败也缓存（避免反复查询无效 PDB ID）
- 优先使用 `initial_release_date`，fallback 到 `deposit_date`

#### 3.1.2 Featurizer 中的 Release Date 查找

```python
def _get_release_date(self, pdb_id_4char: str) -> Optional[datetime]:
    """Get release date from RNA3DB metadata, with PDB API fallback."""
    # 1. Check RNA3DB metadata
    tpl_date = self._release_dates.get(pdb_id_4char)
    if tpl_date is not None:
        return tpl_date

    # 2. Check PDB API cache
    with _PDB_API_CACHE_LOCK:
        if pdb_id_4char in _PDB_API_DATE_CACHE:
            return _PDB_API_DATE_CACHE[pdb_id_4char]

    # 3. Query PDB API
    api_date = _fetch_pdb_release_date(pdb_id_4char)
    if api_date is not None:
        self._release_dates[pdb_id_4char] = api_date  # 更新本地缓存
    return api_date
```

#### 3.1.3 Online Template Building（Problem 2 修复）

**修复的 Bug**: `_build_single_template_online` 中 CIF 路径查找使用了 `hit["pdb_id"]`（格式为 `"1et4_D"`），但 CIF 文件名是 4 字符 PDB ID（`1et4.cif`）。

**修复前**:
```python
t_pdb_id = hit["pdb_id"]          # "1et4_D"
t_chain_id = hit["chain_id"]      # "D"

# ❌ 查找 "1et4_D.cif" — 文件不存在
cif_path = _find_cif_path_cached(cif_database_dir, t_pdb_id)
```

**修复后**:
```python
t_pdb_id_raw = hit["pdb_id"]      # "1et4_D"
t_chain_id = hit["chain_id"]      # "D"
# ✅ 提取 4 字符 base PDB ID
t_pdb_base = _extract_base_pdb_id(t_pdb_id_raw)  # "1et4"

# ✅ 查找 "1et4.cif" — 找到!
cif_path = _find_cif_path_cached(cif_database_dir, t_pdb_base)
```

**完整的 online build 流程**:
```python
def _build_single_template_online(query_seq, hit, cif_database_dir, anchor_mode="base_center_fallback"):
    # 1. Extract base PDB ID from "1et4_D" → "1et4"
    t_pdb_base = _extract_base_pdb_id(hit["pdb_id"])
    t_chain_id = hit["chain_id"]

    # 2. Find CIF file (cached lookup, supports flat + nested layouts)
    cif_path = _find_cif_path_cached(cif_database_dir, t_pdb_base)

    # 3. Load residues from CIF
    residues = load_structure_residues(cif_path, chain_id=t_chain_id)

    # 4. Build template features (alignment + 3D coordinates)
    td = build_minimal_template_arrays(
        query_seq=query_seq,
        residues=residues,
        template_name=f"{t_pdb_base}.cif:{t_chain_id}",
        anchor_mode=anchor_mode,
    )

    # 5. Return only TemplateEmbedder-required features
    return {
        "template_aatype": td["template_aatype"],           # [N]
        "template_distogram": td["template_distogram"],     # [N, N, 39]
        "template_pseudo_beta_mask": td["template_pseudo_beta_mask"],  # [N, N]
        "template_unit_vector": td["template_unit_vector"], # [N, N, 3]
        "template_backbone_frame_mask": td["template_backbone_frame_mask"],  # [N, N]
    }
```

#### 3.1.4 Online 模式下的 Per-Hit Filtering

```python
def _filter_hits_online(self, hits, query_pdb_id, cutoff_date):
    """Per-hit filter with PDB API fallback (mirrors protein pipeline)."""
    filtered = []
    stats = {"self_hit": 0, "future": 0, "no_date": 0}

    for hit in hits:
        tpl_pdb = _extract_base_pdb_id(hit["pdb_id"])

        # 1) Self-hit exclusion
        if query_pdb_id and tpl_pdb == query_pdb_id:
            stats["self_hit"] += 1
            continue

        # 2) Temporal filtering with PDB API fallback
        if cutoff_date is not None:
            tpl_date = self._get_release_date(tpl_pdb)  # RNA3DB → PDB API → None
            if tpl_date is not None:
                if tpl_date > cutoff_date:
                    stats["future"] += 1
                    continue
            else:
                # Unknown date after all lookups → REJECT (保守)
                stats["no_date"] += 1
                continue

        filtered.append(hit)

    return filtered, stats
```

### 3.2 Config 变更: `configs/configs_base.py`

新增 online mode 配置字段:

```python
"rna_template": {
    # ... existing offline fields ...

    # === Online mode (v3) ===
    # When search_results_path + cif_database_dir are both set, online mode is activated
    "search_results_path": "",   # Path to search_results.json from MMseqs2
    "cif_database_dir": "",      # Root directory of CIF structure files
},
```

### 3.3 Dataset 集成: `protenix/data/pipeline/dataset.py`

`get_rna_template_featurizer()` 函数更新以支持双模式:

```python
def get_rna_template_featurizer(configs):
    search_results_path = rna_template_info.get("search_results_path", "")
    cif_database_dir = rna_template_info.get("cif_database_dir", "")
    online_mode = bool(search_results_path and cif_database_dir)

    if online_mode:
        # Validate paths
        if not os.path.exists(search_results_path):
            raise FileNotFoundError(...)
        if not os.path.isdir(cif_database_dir):
            raise FileNotFoundError(...)
        logger.info(f"RNA template featurizer: ONLINE mode — ...")
    else:
        # Offline mode validation
        ...

    return RNATemplateFeaturizer(
        search_results_path=search_results_path,
        cif_database_dir=cif_database_dir,
        # ... other params ...
    )
```

### 3.4 Inference 集成: `protenix/data/inference/infer_dataloader.py`

同样更新以传递 `search_results_path` 和 `cif_database_dir` 到 featurizer 构造函数。

### 3.5 Import Bridge: `protenix/data/rna_template/rna_template_common_online.py`

新建文件，将 `rna_template/compute/rna_template_common.py` 的函数重新导出到 Protenix package 内:

```python
"""Thin import bridge: re-exports rna_template_common functions."""
import os, sys

_COMPUTE_DIR = os.path.join(
    os.path.dirname(...), "rna_template", "compute"
)
if _COMPUTE_DIR not in sys.path:
    sys.path.insert(0, _COMPUTE_DIR)

from rna_template_common import (
    load_structure_residues,
    build_minimal_template_arrays,
    normalize_query_sequence,
    stack_template_dicts,
    # ... etc
)
```

### 3.6 GPU 测试脚本: `rna_template/scripts/test_online_gpu.sh`

新建 GPU 验证脚本，执行:
1. Online featurizer 单元测试（31 tests）
2. 短训练 run（10 steps on GPU，使用 online mode）

---

## 4. 修改前后对比

### 4.1 Release Date 查找

| 方面 | v2 (修改前) | v3 (修改后) |
|------|------------|------------|
| RNA3DB 中找到 | ✅ 使用 release date | ✅ 使用 release date（相同） |
| RNA3DB 中未找到 | ⚠️ **保留**（静默泄漏） | ✅ PDB API 查询 → 缓存 → 使用 |
| PDB API 也查不到 | N/A | ✅ **REJECT**（保守策略） |
| API 调用开销 | N/A | 首次 ~100ms，后续 O(1) from cache |
| 缓存持久化 | N/A | ✅ JSON 文件，跨 session 复用 |

### 4.2 Template Building 模式

| 方面 | v2 Offline 模式 | v3 Online 模式 |
|------|----------------|----------------|
| 3D feature 构建时机 | 离线预计算 (NPZ) | Training time 实时构建 |
| Filter 粒度 | Per-NPZ (4 templates 捆绑) | Per-hit (逐个 template) |
| 被 reject 后 | 无法回填 | 自动尝试下一个 hit |
| Cutoff 类型 | 全局一刀切 | Per-query (query_release_date - 60d) |
| 与 protein pipeline 对齐 | ❌ | ✅ 完全对齐 |
| 向后兼容 | ✅ (原始模式) | ✅ (offline 模式保留) |

### 4.3 CIF 路径解析 Bug 修复

| 场景 | 修复前 | 修复后 |
|------|--------|--------|
| hit["pdb_id"] = "1et4_D" | 查找 `1et4_D.cif` → 不存在 → 失败 | 提取 `1et4` → 查找 `1et4.cif` → 成功 |
| hit["pdb_id"] = "7zpi" | 查找 `7zpi.cif` → 可能成功 | 查找 `7zpi.cif` → 成功 |

---

## 5. 测试验证

### 5.1 单元测试 (31/31 通过)

```
============================================================
RNA Template Featurizer v3 — Online Mode Test Suite
============================================================

=== Test 1: PDB API Release Date Lookup ===
  ✓ Known PDB (1g1x) returns valid date (date=2000-10-30)
  ✓ Cached lookup is fast (<0.01s)
  ✓ Another PDB (4tna) returns valid date (date=1978-04-12)
  ✓ Invalid PDB (zzzz) returns None
  ✓ Cache populated (cache_size=3)

=== Test 2: Online Featurizer Initialization ===
  ✓ Online mode enabled
  ✓ Search hits loaded (num_sequences=143)
  ✓ Release dates loaded (num_dates=5389)

=== Test 3: Online Template Building from CIF ===
  ✓ Single template build succeeds (elapsed=0.14s)
  ✓ template_aatype shape — (35,)
  ✓ template_distogram shape — (35, 35, 39)
  ✓ template_pseudo_beta_mask shape — (35, 35)
  ✓ template_unit_vector shape — (35, 35, 3)
  ✓ Non-zero anchor mask (mask_sum=1225.0)

=== Test 4: Chain Feature Stacking (Online) ===
  ✓ Hits found for sequence (num_hits=4)
  ✓ Chain features built (elapsed=0.51s)
  ✓ Stacked aatype shape [T, N] — (4, 35)
  ✓ Stacked distogram shape [T, N, N, 39]

=== Test 5: Temporal Filtering with PDB API Fallback ===
  ✓ Old cutoff rejects template (future filter)
  ✓ Far future cutoff keeps template

=== Test 6: Self-Hit Exclusion ===
  ✓ Self-hit excluded (1g1x → rejected, 4tna → kept)

=== Test 7: Unknown Date → REJECT (Conservative) ===
  ✓ Unknown date template rejected (no_date=1)

=== Test 8: Offline Mode Backward Compatibility ===
  ✓ Offline mode (online_mode=False)
  ✓ NPZ index loaded (num_sequences=4204)

=== Test 9: PDB API Cache Persistence ===
  ✓ Cache file path set
  ✓ Cache file exists on disk
  ✓ Cache file has entries

=== Test 10: Full Feature Assembly (Online Mode) ===
  ✓ Features returned (elapsed=0.50s)
  ✓ rna_template_aatype shape — (4, N)
  ✓ rna_template_block_mask shape — (N, N)
  ✓ Block mask populated (block_mask_sum=1225.0)

============================================================
Results: 31 passed, 0 failed, 31 total
============================================================
```

### 5.2 GPU 训练验证 (H800, 10 steps)

```
GPU: NVIDIA H800, 81GB
Mode: ONLINE
Steps: 10
Crop size: 128

训练日志关键输出:
  RNA template featurizer: ONLINE mode enabled (143 sequences)
  RNA release dates loaded: 5389 PDBs
  PDB API date cache loaded: 1 entries

  Step 4 train metrics:
    train/loss.avg: 3.16
    train/mse_loss/rna_mse_loss.avg: 16.11

  Step 9 train metrics:
    train/loss.avg: 2.50  (↓ loss 下降)
    train/mse_loss/rna_mse_loss.avg: 6.33  (↓ RNA loss 下降)

  Step 9 eval:
    rna_lddt/best.avg: 0.0079
    rna_lddt/mean.avg: 0.0070

  ✅ Finished training after 10 steps
```

---

## 6. 文件修改清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `protenix/data/rna_template/rna_template_featurizer.py` | 重写 | v3 双模式 featurizer + PDB API |
| `protenix/data/rna_template/rna_template_common_online.py` | 新建 | Import bridge |
| `configs/configs_base.py` | 修改 | 新增 `search_results_path`, `cif_database_dir` |
| `protenix/data/pipeline/dataset.py` | 修改 | `get_rna_template_featurizer()` 支持双模式 |
| `protenix/data/inference/infer_dataloader.py` | 修改 | Inference 路径支持 online 模式 |
| `rna_template/scripts/test_online_featurizer.py` | 新建 | 31 个单元测试 |
| `rna_template/scripts/test_online_gpu.sh` | 新建 | GPU 验证脚本 |

---

## 7. 使用指南

### 7.1 启用 Online 模式（训练）

在 finetune 脚本中添加:
```bash
--rna_template.enable true \
--rna_template.search_results_path "${RNA_DATABASE_DIR}/search_results.json" \
--rna_template.cif_database_dir "${PDB_RNA_DIR}" \
--rna_template.rna3db_metadata_path "${RNA3DB_METADATA}" \
--rna_template.max_rna_templates 4 \
--rna_template.projector_init "protein" \
--rna_template.alpha_init 0.01
```

注意：设置 `search_results_path` + `cif_database_dir` 后，`template_index_path` 不再需要（自动切换到 online 模式）。

### 7.2 保持 Offline 模式（向后兼容）

不设置 `search_results_path` 和 `cif_database_dir`，使用原有 NPZ 方式:
```bash
--rna_template.enable true \
--rna_template.template_database_dir "${RNA_DATABASE_DIR}" \
--rna_template.template_index_path "${RNA_TEMPLATE_INDEX}" \
```

---

## 8. 总结

| 问题 | 状态 | 解决方式 |
|------|------|----------|
| Prob1: Unknown-date templates 绕过时间过滤 | ✅ 已解决 | PDB API fallback + 保守 REJECT |
| Prob2: Offline → Online 模式 | ✅ 已解决 | Per-hit online build from CIF |
| CIF 路径解析 Bug (`1et4_D.cif` vs `1et4.cif`) | ✅ 已修复 | `_extract_base_pdb_id()` |
| GPU 训练验证 | ✅ 通过 | H800, 10 steps, loss 下降正常 |
| 向后兼容 (offline mode) | ✅ 保持 | 不设置 online 参数时自动 fallback |
| 单元测试 | ✅ 31/31 通过 | `test_online_featurizer.py` |
