# RNA/DNA Embedding KeyError 根因分析报告

**日期**: 2026-03-11
**问题**: 训练时出现 `KeyError`，找不到某些序列的 embedding 文件路径
**涉及 PDB**: `5di2` (Entity 2), `1u1y` (Entity 1)

---

## 1. 问题现象

训练使用 `protenix_aido_separate_input.sh` 脚本时，出现以下错误：

```
[5di2] DNA embedding error for entity 2: KeyError: 'GGGCGUCUGGGCAGUACCCA'
[1u1y] DNA embedding error for entity 1: KeyError: 'CCGGGGGAUCACCACGG'
```

两个序列都包含 `U`（尿嘧啶），是 RNA 特征碱基。但 featurizer 却在 **DNA** embedding 查找表中搜索它们。

---

## 2. 根因分析

### 2.1 核心问题：两套分类体系不一致

| 阶段 | 分类依据 | 分类字段 | 代码位置 |
|------|---------|----------|---------|
| **Embedding 生成** (`extract_sequences.py`) | `entity_poly_type` 元数据 | PDB mmCIF 的 entity 级别标注 | `extract_sequences.py:20-34` |
| **训练 Featurizer** (`rnalm_featurizer.py`) | `chain_mol_type` 原子级别 | CCD 化学组分字典逐残基分类后链级聚合 | `parser.py:2544-2605` |

**这两套分类体系对同一个 entity 可能给出不同结果。**

### 2.2 分类流程详解

#### Embedding 生成阶段 (`extract_sequences.py`)

```python
def classify_entity(entity_poly_type):
    if "polydeoxyribonucleotide" == poly_type:       → DNA
    elif "polyribonucleotide" == poly_type:           → RNA
    elif "hybrid" in poly_type:                       → RNA  ← 关键：hybrid 归类为 RNA
    elif "ribonucleotide" in poly_type:               → RNA
```

- 使用 PDB 元数据中的 `entity_poly_type` 字段
- **Hybrid 实体统一归为 RNA**，embedding 只存在 RNA manifest 中

#### 训练阶段 (`parser.py` → `rnalm_featurizer.py`)

**Step 1**: 逐残基分类 (`add_token_mol_type`, `parser.py:2544-2570`)
```python
# 每个残基通过 CCD (Chemical Component Dictionary) 查询其化学类型
mol_types[start:stop] = ccd.get_mol_type(res_name)
```

**Step 2**: CCD 分类逻辑 (`ccd.py:150-178`)
```python
def get_mol_type(ccd_code):
    link_type = ccd_cif[ccd_code]["chem_comp"]["type"].as_item().upper()
    if "PEPTIDE" in link_type: return "protein"
    if "DNA" in link_type:     return "dna"    # ← 优先检查 DNA
    if "RNA" in link_type:     return "rna"    # ← 其次检查 RNA
    return "ligand"
```

**注意**: "DNA" 优先于 "RNA" 检查。如果 CCD 类型包含两者（如 hybrid 类型的修饰核苷酸），会被归为 DNA。

**Step 3**: 链级聚合 (`add_atom_mol_type_mask`, `parser.py:2572-2605`)
```python
chain_starts = struc.get_chain_starts(atom_array, add_exclusive_stop=True)
for start, end in zip(chain_starts[:-1], chain_starts[1:]):
    mol_types = atom_array.mol_type[start:end]
    mol_type_count = Counter(mol_types)
    sorted_by_key = sorted(mol_type_count.items(), key=lambda x: x[0])   # 字母序
    sorted_by_value = sorted(sorted_by_key, key=lambda x: x[1])          # 计数升序
    most_freq_mol_type = sorted_by_value[0][0]                           # ← 取第一个 = 最少的!
```

**这里存在一个 Bug**: 变量名 `most_freq_mol_type`（最频繁类型），但实际取的是 `sorted_by_value[0]`（**计数最少**的类型）。`sorted()` 默认升序排列，`[0]` 是最小值。

**Step 4**: Featurizer 使用 `chain_mol_type` 决定查找哪个 embedding 表 (`rnalm_featurizer.py:250-255`)
```python
is_rna = centre_atom_array.chain_mol_type == "rna"   → 查 RNA 表
is_dna = centre_atom_array.chain_mol_type == "dna"   → 查 DNA 表
```

### 2.3 具体案例分析

#### PDB 5di2 Entity 2

| 属性 | 值 |
|------|-----|
| `entity_poly_type` | `polydeoxyribonucleotide/polyribonucleotide hybrid` |
| 序列 | `GGGCGUCUGGGCAGUACCCA` (含 U → RNA 特征) |
| Embedding 生成分类 | **RNA** (hybrid → RNA) |
| Embedding 存储位置 | `rna/5di2_2/5di2_2.pt` ✓ |
| 训练时 `chain_mol_type` | **DNA** (CCD 逐残基分类 + 链级聚合) |
| Featurizer 查找表 | **DNA** manifest → **KeyError** ✗ |

**原因**: 这是一个 DNA/RNA hybrid 实体。CCD 会将其中的 deoxy 核苷酸标记为 "dna"，ribo 核苷酸标记为 "rna"。链级聚合的 Bug（取最少类型）或字母序 tiebreaker（"dna" < "rna"）导致整条链被标为 "dna"。但 embedding 生成时将 hybrid 统一归类为 RNA。

#### PDB 1u1y Entity 1

| 属性 | 值 |
|------|-----|
| `entity_poly_type` | `polyribonucleotide` (纯 RNA) |
| 序列 | `CCGGGGGAUCACCACGG` (含 U → RNA 特征) |
| Embedding 生成分类 | **RNA** |
| Embedding 存储位置 | `rna/1u1y_1/1u1y_1.pt` ✓ |
| 训练时 `chain_mol_type` | **DNA** |
| Featurizer 查找表 | **DNA** manifest → **KeyError** ✗ |

**原因**: 虽然 `entity_poly_type` 标注为纯 RNA，但 CCD 残基级分类可能将某些修饰核苷酸归为 "dna"（CCD type 包含 "DNA" → 优先匹配 DNA）。再加上链级聚合 Bug，即使只有少量残基被错分为 DNA，也可能导致整条链被标记为 "dna"。

---

## 3. 影响范围

### 3.1 已知受影响的实体

根据 `missing_embeddings_report.txt` 的统计：

| 分类 | 数量 | 说明 |
|------|------|------|
| Hybrid 实体总数 | 157 | `polydeoxyribonucleotide/polyribonucleotide hybrid` |
| Hybrid 在 RNA manifest 中 | 157 / 157 (100%) | 都在 RNA 表里 |
| Hybrid 在 DNA manifest 中 | 1 / 157 (0.6%) | 几乎都不在 DNA 表里 |
| Hybrid 缺失于 DNA manifest | **156** | 训练时如被分为 DNA 则 KeyError |

### 3.2 潜在受影响的实体

除了 157 个 hybrid 实体外，任何满足以下条件的 **纯 RNA** 实体也可能受影响：
- 包含修饰核苷酸，其 CCD type 含 "DNA" 关键词
- 包含修饰核苷酸，其 CCD type 不含 "RNA" 关键词（被归为 "ligand"）
- 链级聚合 Bug 导致少数异常残基类型覆盖了多数正常残基类型

PDB 1u1y 就是这种情况的一个例子。

---

## 4. Bug 汇总

### Bug 1: 分类体系不一致（根本原因）

| | Embedding 生成 | 训练 Featurizer |
|--|---------------|----------------|
| 分类依据 | `entity_poly_type` (元数据) | `chain_mol_type` (CCD 化学分类) |
| Hybrid 处理 | → RNA | → DNA (大概率) |
| 分类粒度 | Entity 级别 | 残基级别 → 链级聚合 |

### Bug 2: `add_atom_mol_type_mask()` 取最少而非最多（`parser.py:2595`）

```python
# 当前代码（错误）：取计数最少的类型
most_freq_mol_type = sorted_by_value[0][0]    # [0] = 最小计数

# 正确应为：取计数最多的类型
most_freq_mol_type = sorted_by_value[-1][0]   # [-1] = 最大计数
```

**影响**: 当链中有少量修饰核苷酸（被 CCD 分为 "ligand" 或 "dna"）时，少数类型反而成为整条链的类型。

### Bug 3: CCD `get_mol_type()` 的 DNA 优先检查 (`ccd.py:174-176`)

```python
if "DNA" in link_type:  return "dna"   # 先检查 DNA
if "RNA" in link_type:  return "rna"   # 后检查 RNA
```

**影响**: 如果某个修饰核苷酸的 CCD type 同时包含 "DNA" 和 "RNA"（如 hybrid 类型），会被优先归为 DNA。

---

## 5. 建议修复方向（待确认后执行）

### 方案 A: Featurizer 端兼容（推荐 — 最小改动）

在 `rnalm_featurizer.py` 的 `_fill_entities()` 或序列查找逻辑中，当 DNA 表找不到时自动 fallback 到 RNA 表查找（反之亦然）：

```python
# 伪代码
def load_embedding(self, sequence, primary_lookup, fallback_lookup):
    if sequence in primary_lookup:
        return load_from(primary_lookup[sequence])
    elif sequence in fallback_lookup:
        logger.info(f"Fallback: found '{sequence}' in other manifest")
        return load_from(fallback_lookup[sequence])
    else:
        raise KeyError(f"Sequence not found in either manifest")
```

**优点**: 不需要修改 Protenix 核心代码（parser.py），不会影响其他功能
**缺点**: 只是绕过问题，没有修正根源

### 方案 B: 修正 `add_atom_mol_type_mask()` 的聚合逻辑

```python
# 修正为取最频繁的类型
most_freq_mol_type = sorted_by_value[-1][0]  # [-1] = 最大计数
```

**优点**: 修正了 Protenix 的一个实际 Bug
**缺点**: 改动 Protenix 核心数据流，可能影响其他下游任务，需要充分测试

### 方案 C: Embedding 生成时同时生成 RNA 和 DNA 版本

对 hybrid 实体，同时在 RNA 和 DNA manifest 中生成 embedding。

**优点**: 无论 featurizer 分类为 RNA 还是 DNA，都能找到
**缺点**: 需要重新运行 embedding 生成流程

### 推荐: 方案 A + 方案 B

- 方案 A 作为 featurizer 层的安全网，立即解决 KeyError
- 方案 B 作为长期修正，提交给 Protenix 社区或在本地 fork 中修复

---

## 6. 数据验证

| 检查项 | 结果 |
|-------|------|
| `GGGCGUCUGGGCAGUACCCA` 在 RNA CSV 中？ | ✅ 存在 (5di2_2, 5dh6_2, 5dh7_2 等) |
| `GGGCGUCUGGGCAGUACCCA` 在 DNA CSV 中？ | ❌ 不存在 |
| `CCGGGGGAUCACCACGG` 在 RNA CSV 中？ | ✅ 存在 (1u1y_1) |
| `CCGGGGGAUCACCACGG` 在 DNA CSV 中？ | ❌ 不存在 |
| 5di2.pkl.gz bioassembly 文件？ | ✅ 存在 (41 KB) |
| 1u1y.pkl.gz bioassembly 文件？ | ✅ 存在 (339 KB) |
| RNA embedding .pt 文件？ | ✅ 两个序列都有对应的 .pt 文件 |

**结论**: Embedding 本身是完整的。问题完全在于训练时 featurizer 的分类与 embedding 生成时的分类不一致，导致去了错误的查找表。

---

## 7. 相关文件

| 文件 | 作用 |
|------|------|
| `protenix/data/rnalm/rnalm_featurizer.py:250-255` | 使用 `chain_mol_type` 决定 RNA/DNA 分类 |
| `protenix/data/core/parser.py:2572-2605` | `add_atom_mol_type_mask()` 链级聚合（含 Bug） |
| `protenix/data/core/parser.py:2544-2570` | `add_token_mol_type()` 逐残基 CCD 分类 |
| `protenix/data/core/ccd.py:150-178` | `get_mol_type()` CCD 类型判断（DNA 优先） |
| `extract_sequences.py:20-34` | Embedding 生成的分类逻辑（使用 entity_poly_type） |
| `missing_embeddings_report.txt` | 已标记 156 个 hybrid 实体缺失 DNA manifest |
