# RNA Template → Protenix 输入特征：完整技术文档

> **文件清单**：`rna_template_common.py` · `build_rna_template_protenix.py` · `build_rna_template_extended.py` · `run_dssr_and_build.py`

---

## 目录

1. [总体架构](#1-总体架构)
2. [rna_template_common.py — 公共函数库](#2-rna_template_commonpy--公共函数库)
3. [build_rna_template_protenix.py — 最小兼容版](#3-build_rna_template_protenixpy--最小兼容版)
4. [build_rna_template_extended.py — RNA 扩展版](#4-build_rna_template_extendedpy--rna-扩展版)
5. [run_dssr_and_build.py — 一键自动化](#5-run_dssr_and_buildpy--一键自动化)
6. [核心算法详解](#6-核心算法详解)
7. [输出张量速查表](#7-输出张量速查表)
8. [数据流全景图](#8-数据流全景图)

---

## 1. 总体架构

### 1.1 设计目标

这套脚本要解决的核心问题是：**Protenix 原生的 template 通路是为蛋白质设计的，RNA 没有 `N-Cα-C` 主链，不能直接复用。** 因此需要一套「RNA → 类蛋白 template 张量」的转换管线，使得 RNA 结构先验能够以 Protenix `TemplateEmbedder` 可理解的数学形式注入模型。

### 1.2 两层设计

```
Layer 1 — 最小兼容版 (build_rna_template_protenix.py)
  ├─ 产出 5 个与 Protenix TemplateEmbedder 语义一致的几何张量
  └─ 不需要 DSSR，只靠结构坐标

Layer 2 — RNA 扩展版 (build_rna_template_extended.py)
  ├─ 保留 Layer 1 全部输出
  ├─ 额外产出 RNA-specific pair channel (25 维)
  ├─ 额外产出 RNA-specific single channel (13 维)
  └─ 需要 DSSR JSON 作为输入
```

### 1.3 依赖关系

```
rna_template_common.py          ← 所有公共逻辑
    ↑               ↑
build_rna_template   build_rna_template
_protenix.py         _extended.py
    ↑               ↑
    └───────┬───────┘
   run_dssr_and_build.py        ← 自动化入口
```

外部依赖：`numpy`、`BioPython`（PDB/mmCIF 解析 + 序列对齐）、可选 `x3dna-dssr`（扩展版）。

---

## 2. rna_template_common.py — 公共函数库

这是整套管线的核心引擎。下面按功能模块逐一说明。

### 2.1 常量与映射表

#### 2.1.1 `PROTENIX_TEMPLATE_RNA_SEQ_TO_ID`

```python
{"A": 21, "G": 22, "C": 23, "U": 24, "N": 25, "-": 31}
```

这是 **Protenix 官方的 RNA residue id 编码**，直接取自 ByteDance/Protenix 主分支。蛋白质占用 0–20，RNA 从 21 开始，gap（未对齐位置）编为 31。这样 `template_aatype` 就能直接喂给 Protenix 的 embedding lookup。

#### 2.1.2 `COMMON_MODIFIED_BASE_MAP`

```python
{"PSU": "U", "H2U": "U", "5MC": "C", "1MA": "A", "2MG": "G", ...}
```

将常见修饰碱基（假尿嘧啶、甲基化碱基等）映射回标准 A/G/C/U。这是因为 PDB/mmCIF 中 RNA 模板经常包含修饰残基（如 tRNA），需要退化成标准碱基才能做序列对齐和 aatype 编码。

#### 2.1.3 `BACKBONE_SUGAR_ATOMS`

```python
{"P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", ...}
```

定义了 RNA 糖-磷骨架的原子集合。用途：**计算 base center 时排除骨架原子**——base center 只应包含碱基环上的原子坐标的质心。

#### 2.1.4 `DEFAULT_DENSE_ATOM_NAMES`（27 个原子）

```python
["P", "OP1", "OP2", "O5'", "C5'", ..., "C1'", "N9", "C8", ..., "N4", "O4"]
```

这是一个 **RNA 通用联合原子列表**：覆盖了嘌呤（A/G）和嘧啶（C/U）的全部重原子。用于扩展版中的 `--emit_dense_atoms` 选项，为后续 atom-level RNA template encoder 预留接口。

#### 2.1.5 `LW_CLASSES`（18 类 Leontis-Westhof 碱基配对）

```python
["cWW", "tWW", "cWH", "tWH", "cWS", "tWS",
 "cHW", "tHW", "cHH", "tHH", "cHS", "tHS",
 "cSW", "tSW", "cSH", "tSH", "cSS", "tSS"]
```

LW 分类法用三个碱基边（Watson-Crick / Hoogsteen / Sugar）的两两组合 × cis/trans 方向来标注碱基对类型。总共 3×3×2 = 18 种。这套分类比简单的 canonical/non-canonical 区分精细得多，是 RNA 结构分析的标准工具。

---

### 2.2 数据结构：`ResidueRecord`

```python
@dataclass
class ResidueRecord:
    chain_id: str       # 链 ID
    resseq: int         # 残基序号
    icode: str          # 插入码 (insertion code)
    resname: str        # 原始残基名 (如 "PSU", "A", "5MC")
    base_letter: str    # 退化后的标准碱基 (A/G/C/U/N)
    atoms: Dict[str, np.ndarray]  # 原子名 → 3D 坐标
```

每个 RNA 残基被抽象为一个 `ResidueRecord`。关键设计点：

- `resname` 保留原始名（便于追踪修饰碱基），`base_letter` 存退化后的标准碱基。
- `atoms` 是一个 `{原子名: [x,y,z]}` 字典，已经过清洗（`*` → `'`，去除氢原子）。
- `aliases()` 方法返回 `(chain, resseq, icode)` 元组的集合，用于和 DSSR 的 `nt_id` 做宽松匹配。

---

### 2.3 结构文件解析：`load_structure_residues()`

**输入**：mmCIF 或 PDB 文件路径 + 可选链 ID

**算法流程**：

```
1. 根据文件后缀选择 BioPython 的 MMCIFParser 或 PDBParser
2. 取 model 0 的指定 chain（若未指定且只有 1 条链，自动选取）
3. 遍历 chain 中每个 residue：
   a. 遍历其所有 atom，处理 disordered atom（选 occupancy 最高的）
   b. 清洗原子名：去空格、 * → '、大写化
   c. 过滤氢原子
   d. 用 _looks_like_nucleotide() 判定是否为核苷酸
      → 检查 resname 是否在标准集 {A, G, C, U, DA, DG, DC, DT}
      → 或者含有 C1' 或 (P + C4') 原子
   e. 用 _infer_base_from_atoms() 推断标准碱基字母
      → 先查 COMMON_MODIFIED_BASE_MAP
      → 再按特征原子推断 (N6→A, O6/N2→G, N4→C, O4→U)
      → 最后 fallback 到 "N"
4. 返回 List[ResidueRecord]
```

**设计亮点**：对修饰碱基的处理非常宽松——能推断就推断，推断不了就标 "N"，不会直接报错。这对真实 RNA 模板（尤其 tRNA/rRNA）非常重要。

---

### 2.4 序列对齐：`align_query_to_template()`

**输入**：query 序列（RNA）、template 序列（从结构文件提取）

**算法**：BioPython `PairwiseAligner`，全局对齐

```
打分矩阵：
  match    = +2.0
  mismatch = -1.0
  gap open = -5.0
  gap ext  = -0.5
```

**输出**：`Dict[int, int]`，key = query 位置索引，value = template 位置索引

这个 mapping 是整条管线最关键的数据结构——后续所有特征计算都依赖它来确定「query 的第 i 个残基对应 template 的哪个残基」。

对齐参数的选择逻辑：
- gap open 惩罚较重（-5.0），鼓励尽量连续匹配。
- gap extend 较轻（-0.5），允许长插入/缺失以处理结构域差异。
- mismatch 适中（-1.0），允许修饰碱基和序列差异。

---

### 2.5 Anchor 计算：`compute_anchor()`

**作用**：替代蛋白质中的 pseudo-beta (Cβ) 位置——为每个 RNA 残基定义一个「代表性空间位置」。

**算法（以默认 `base_center_fallback` 模式为例）**：

```
优先级 1: base center
  → 取残基中所有【非骨架原子】坐标的质心
  → 这些原子包括碱基环上的 N1, C2, N3, C4, C5, C6, N6, O6, N7, C8, N9 等
  → base_center = mean(base_atom_coords)

优先级 2: C4'
  → 糖环上的 C4' 原子坐标

优先级 3: C1'
  → 糖苷键上的 C1' 原子坐标

最终 fallback: 零向量 + mask = 0
```

**为什么用 base center 而不是 C4'？**

RNA 的关键结构约束来自碱基配对和碱基堆叠——这些都是碱基平面之间的相互作用。用 base center 作为 anchor 能更准确地反映残基间的「碱基-碱基距离」，比 C4'（骨架位置）更能捕捉 RNA 特有的几何先验。

---

### 2.6 局部坐标框架：`compute_frame()`

**作用**：替代蛋白质中 N-Cα-C 定义的主链局部框架——为每个 RNA 残基建立一个局部正交坐标系。

**算法（Gram-Schmidt 正交化）**：

```
输入原子：
  origin = C4'        ← 框架原点
  v1_raw = P - C4'    ← 第一轴方向 (骨架延伸方向)
  v2_raw = C1'- C4'   ← 第二轴方向 (指向糖苷键/碱基)

Gram-Schmidt 过程：
  e1 = normalize(v1_raw)                        ← 第一正交轴
  e2 = normalize(v2_raw - dot(v2_raw, e1)·e1)   ← 第二正交轴 (从 v2 移除 e1 分量)
  e3 = normalize(cross(e1, e2))                  ← 第三正交轴

输出：
  origin: [3]      ← C4' 坐标
  axes:   [3, 3]   ← 正交基 [e1, e2, e3]
  mask:   float    ← 1.0 表示框架有效，0.0 表示缺原子
```

**Fallback 策略**：
- 如果 P 缺失（常见于 RNA 链首位），尝试用 O5' 或 C3' 替代。
- 如果任何正交化步骤中向量范数接近 0，则整个 frame mask = 0。

**化学合理性**：
- `P → C4'` 方向代表骨架延伸方向。
- `C4' → C1'` 方向指向碱基。
- 二者正交化后形成的平面近似于糖环平面的法线方向。
- 这与蛋白质 template 中「局部 frame + relative direction」的数学形式保持一致。

---

### 2.7 Distogram 计算：`compute_distogram()`

**作用**：将残基对之间的 anchor 距离转换为 one-hot 距离桶（distance bin）。

**算法**：

```
步骤 1: 构建距离桶边界
  lower_breaks = linspace(3.25, 50.75, 39)
  → 产生 39 个均匀间隔的阈值
  → 桶宽 ≈ (50.75 - 3.25) / 38 ≈ 1.25 Å

步骤 2: 计算 pairwise 距离矩阵
  D²[i,j] = ||anchor_i - anchor_j||²

步骤 3: One-hot 编码
  对每对 (i,j)：
    distogram[i,j,k] = 1.0  当且仅当  lower[k]² < D²[i,j] ≤ upper[k]²
    其中 upper[k] = lower[k+1]（最后一桶上界为 +∞）

步骤 4: 应用 pairwise mask
  distogram[i,j,:] *= anchor_valid[i] * anchor_valid[j]
```

**设计选择**：
- 默认 39 桶、3.25–50.75 Å 范围与 Protenix/AF3 蛋白 template distogram 参数一致。
- 用平方距离比较避免开方运算。
- 距离超过 50.75 Å 的残基对会落入最后一个桶。

---

### 2.8 Unit Vector 计算：`compute_unit_vectors()`

**作用**：把「j 相对于 i 的方向」投影到「i 的局部坐标系」中，产出归一化方向向量。

**算法**：

```
对每对 (i, j)：
  r_ij = anchor_j - frame_origin_i        ← 全局位移向量
  proj_ij = frame_axes_i · r_ij            ← 投影到 i 的局部坐标系
  u_ij = normalize(proj_ij)                ← 归一化

mask:
  uv_mask[i,j] = frame_valid[i] * anchor_valid[j] * (||proj_ij|| > ε)
```

**直觉**：这个特征告诉模型「从 i 的局部视角看，j 在哪个方向」。蛋白质 template 中用 Cα 的 N-Cα-C frame 做同样的事情。RNA 版本用 C4'-P-C1' frame 替代，数学形式完全一致。

---

### 2.9 二面角计算函数

#### `dihedral_angle(p0, p1, p2, p3)`

标准四原子二面角计算：

```
b0 = p0 - p1
b1 = p2 - p1    (中心键)
b2 = p3 - p2

将 b1 归一化后：
  v = b0 - (b0·b1)·b1    ← b0 在垂直于 b1 平面上的投影
  w = b2 - (b2·b1)·b1    ← b2 在垂直于 b1 平面上的投影

atan2(cross(b1,v)·w, v·w) → 二面角 ∈ [-π, π]
```

#### `compute_chi_angle(record)`

**χ 角 (糖苷键扭转角)**：定义碱基相对于糖环的旋转。

```
嘌呤 (A, G): O4' - C1' - N9 - C4
嘧啶 (C, U): O4' - C1' - N1 - C2
```

χ 角是 RNA 结构中最重要的二面角之一——它决定了碱基是 anti 还是 syn 构象。

#### `compute_delta_angle(record)`

**δ 角 (糖环构象角)**：

```
C5' - C4' - C3' - O3'
```

δ 角直接反映糖环的折叠状态（C2'-endo vs C3'-endo），是区分 A-form 和 B-form 核酸的关键指标。

#### `compute_eta_theta(prev, current, next)`

**η 和 θ 角 (虚拟骨架扭转角)**：由 Wadley & Pyle 2007 定义，用简化骨架描述 RNA 的整体构象。

```
η = dihedral(prev_C4', P, C4', next_P)
    → 描述链上相邻核苷酸之间的「翻转」
    → 需要前后相邻残基的原子

θ = dihedral(P, C4', next_P, next_C4')
    → 描述当前核苷酸到下一个的「扭转」
    → 需要下一个残基的原子
```

#### `angle_to_sin_cos(angle)`

```
angle → (sin(angle), cos(angle))
None  → (0.0, 0.0)
```

将角度编码为 sin/cos 对而不是直接输出弧度值。原因：避免 -π ~ π 边界处的不连续跳变，对神经网络更友好。

---

### 2.10 DSSR JSON 解析

#### `parse_dssr_nt_id(value)`

将 DSSR 输出的核苷酸标识符（如 `"A.G15"` 或 `"B.U7A"`）解析为统一的 `(chain_id, resseq, icode)` 三元组。

```
"A.G15"  → ("A", 15, "")
"B.U7A"  → ("B", 7, "A")
```

这个解析器处理了 DSSR 输出中各种可能的命名格式差异。

#### `build_residue_lookup(residues, mapping)`

构建 `{(chain, resseq, icode) → query_index}` 的反向查找表。这样就能将 DSSR 报告中的核苷酸 ID 直接映射到 query 序列的位置索引。

#### `collect_dssr_pair_edges(dssr, lookup, query_seq)`

从 DSSR JSON 中提取碱基配对信息。

```
算法：
1. 从 dssr["pairs"], dssr["basePairs"], dssr["bps"] 中收集碱基对
2. 对每个碱基对：
   a. 解析两端核苷酸 ID → 查找 query 索引 (i, j)
   b. 提取 LW 分类标签
   c. 判断配对类别：
      - cWW + {A,U} 或 {G,C} → "canonical"
      - cWW + {G,U}          → "wobble"
      - 其余                  → "noncanonical"
3. 返回 [{i, j, lw, category, raw}, ...]
```

#### `collect_dssr_stack_edges(dssr, lookup)`

从 DSSR JSON 中提取碱基堆叠信息。

```
算法：
1. 遍历 dssr 中所有 key 名含 "stack" 的列表
2. 对每个 stacking entry：
   a. 从 nts_long/nts/nt1+nt2 中提取涉及的核苷酸
   b. 解析成 query 索引
   c. 对连续的索引对 (a, b) 生成边
3. 返回 [{i, j, consecutive_in_stack, raw}, ...]
```

---

### 2.11 模板堆叠与保存

#### `stack_template_dicts(dicts, max_templates)`

将多个模板的特征字典堆叠为统一的批量张量。

```
输入：[template_dict_1, template_dict_2, ...]  (每个含 N 维特征)
输出：合并后的字典，每个 key 的 shape 从 [N, ...] 变为 [T, N, ...]

T = max_templates (默认 4)
实际模板数 < T 时，多余位置填零 + template_mask = 0
aatype 的填充值用 gap id 31
```

#### `save_npz_with_metadata(path, arrays, metadata)`

将张量保存为压缩 `.npz` 文件，同时在旁边写一个 `.npz.meta.json` 文件记录所有元信息（query 序列、anchor 模式、binning 参数、模板来源等）。

---

## 3. build_rna_template_protenix.py — 最小兼容版

### 3.1 定位

这个脚本的目标是：**只产出与现有 Protenix `TemplateEmbedder` 直接兼容的 5 个核心张量**。不需要 DSSR，不需要额外的 RNA 知识——纯粹基于 3D 结构坐标。

### 3.2 运行方式

```bash
python build_rna_template_protenix.py \
  --query_seq query.fa \
  --template template1.cif:A \
  --template template2.pdb:B \
  --output rna_template_minimal.npz \
  --anchor_mode base_center_fallback \
  --max_templates 4
```

### 3.3 完整处理流程

```
对每个 --template：
  ①  parse_template_spec()
      → 解析 "path:CHAIN" 格式
  
  ②  load_structure_residues()
      → 读取 mmCIF/PDB → List[ResidueRecord]
  
  ③  build_minimal_template_arrays()
      内部流程：
      a. normalize_query_sequence()   → 清洗 query 序列
      b. residues_to_sequence()       → 从结构提取 template 序列
      c. align_query_to_template()    → 全局序列比对 → mapping
      d. 对每个已对齐位置 (q_idx, t_idx)：
         - 编码 aatype (A→21, G→22, ...)
         - compute_anchor()  → anchor 坐标 + mask
         - compute_frame()   → 局部框架 + mask
      e. compute_distogram()      → [N, N, 39]
      f. compute_unit_vectors()   → [N, N, 3] + mask
      g. 组装成字典返回

合并所有模板：
  ④  stack_template_dicts()
      → [T, N, ...] 批量张量

  ⑤  save_npz_with_metadata()
      → 保存 .npz + .meta.json
```

### 3.4 输出 key 与 shape

| Key | Shape | 描述 |
|-----|-------|------|
| `template_aatype` | `[T, N]` | RNA 残基类型 ID |
| `template_distogram` | `[T, N, N, 39]` | anchor 间距离的 one-hot 编码 |
| `template_pseudo_beta_mask` | `[T, N, N]` | pairwise anchor 有效性 mask |
| `template_unit_vector` | `[T, N, N, 3]` | 局部框架中的归一化方向向量 |
| `template_backbone_frame_mask` | `[T, N, N]` | pairwise frame 有效性 mask |

辅助 key（debug 用）：

| Key | Shape | 描述 |
|-----|-------|------|
| `template_anchor_pos` | `[T, N, 3]` | anchor 绝对坐标 |
| `template_anchor_mask_1d` | `[T, N]` | 1D anchor mask |
| `template_frame_origin` | `[T, N, 3]` | 框架原点 (C4') |
| `template_frame_axes` | `[T, N, 3, 3]` | 正交基 |
| `template_frame_mask_1d` | `[T, N]` | 1D frame mask |
| `template_mask` | `[T]` | 模板有效性 |
| `template_names` | `[T]` | 模板名称 |
| `template_mapping_json` | `[T]` | 对齐 mapping 的 JSON |

### 3.5 与蛋白 template 的对应关系

| 蛋白质 Template 概念 | RNA 等价实现 |
|---------------------|-------------|
| Cβ (pseudo-beta) | base center (碱基重原子质心) |
| N-Cα-C backbone frame | C4'-based frame (P→C4' 为 e1, C1'→C4' 正交化为 e2) |
| Residue type (20 aa) | RNA base type (A=21, G=22, C=23, U=24, N=25) |
| Distogram binning | 相同参数: 39 bins, 3.25–50.75 Å |
| Unit vector | 相同数学: 投影到局部 frame 后归一化 |

---

## 4. build_rna_template_extended.py — RNA 扩展版

### 4.1 定位

在最小兼容版的基础上，利用 DSSR 的结构注释输出额外的 **RNA 特有特征通道**。这些通道可以喂给一个并行的 `RNATemplateEmbedder`。

### 4.2 运行方式

```bash
python build_rna_template_extended.py \
  --query_seq query.fa \
  --template template1.cif:A \
  --dssr_json template1.dssr.json \
  --template template2.pdb:B \
  --dssr_json template2.dssr.json \
  --output rna_template_extended.npz \
  --emit_dense_atoms
```

### 4.3 完整处理流程

```
对每个 (--template, --dssr_json) 配对：
  ① 最小兼容版的全部流程（复用 build_minimal_template_arrays）

  ② 额外的 RNA 特征提取：
     a. 用 align_query_to_template() 得到 mapping
     b. 用 build_residue_lookup() 建立反向查找表
     c. load_dssr_json() 读取 DSSR 注释

     d. collect_dssr_pair_edges()  → 碱基配对边列表
     e. collect_dssr_stack_edges() → 碱基堆叠边列表
     f. _fill_pair_features()     → [N, N, 25] pair 特征张量
     g. _fill_single_features()   → [N, 13] single 特征张量

  ③ 可选: _fill_dense_atoms()    → [N, 27, 3] 原子坐标 + [N, 27] mask

合并：
  ④ stack_template_dicts() + 手动堆叠 pair/single/dense 张量
  ⑤ save_npz_with_metadata()
```

### 4.4 Pair 特征通道详解 (`rna_template_pair`)

Shape: `[T, N, N, 25]`

| 通道索引 | 通道名 | 语义 | 编码方式 |
|---------|--------|------|---------|
| 0 | `pair_any` | 存在任意碱基配对 | binary |
| 1 | `pair_canonical` | Watson-Crick 标准配对 (A-U, G-C) | binary |
| 2 | `pair_wobble` | G-U 摆动配对 | binary |
| 3 | `pair_noncanonical` | 非标准配对 | binary |
| 4–21 | `lw_cWW` ... `lw_tSS` | 18 种 Leontis-Westhof 类型 | binary |
| 22 | `stack_any` | 存在碱基堆叠 | binary |
| 23 | `stack_consecutive` | 序列连续堆叠 (\|i-j\|=1) | binary |
| 24 | `stack_nonconsecutive` | 序列不连续堆叠 (\|i-j\|>1) | binary |

**`_fill_pair_features()` 算法**：

```
初始化 pair = zeros(N, N, 25)

处理碱基配对：
  对每条 DSSR pair edge (i, j, category, lw)：
    对称填充 (i,j) 和 (j,i)：
      pair[i,j,"pair_any"] = 1.0
      pair[i,j,f"pair_{category}"] = 1.0    # canonical/wobble/noncanonical
      if lw in LW_CLASSES:
          pair[i,j,f"lw_{lw}"] = 1.0

处理碱基堆叠：
  对每条 DSSR stack edge (i, j)：
    gap = |i - j|
    对称填充 (i,j) 和 (j,i)：
      pair[i,j,"stack_any"] = 1.0
      if gap == 1: pair[i,j,"stack_consecutive"] = 1.0
      else:        pair[i,j,"stack_nonconsecutive"] = 1.0
```

### 4.5 Single 特征通道详解 (`rna_template_single`)

Shape: `[T, N, 13]`

| 通道索引 | 通道名 | 语义 | 编码方式 |
|---------|--------|------|---------|
| 0 | `is_modified` | 是否为修饰碱基 | binary |
| 1 | `is_purine` | 是否为嘌呤 (A/G) | binary |
| 2 | `is_pyrimidine` | 是否为嘧啶 (C/U) | binary |
| 3 | `has_anchor` | anchor 坐标是否有效 | binary |
| 4 | `has_frame` | 局部框架是否有效 | binary |
| 5 | `chi_sin` | sin(χ) | continuous |
| 6 | `chi_cos` | cos(χ) | continuous |
| 7 | `delta_sin` | sin(δ) | continuous |
| 8 | `delta_cos` | cos(δ) | continuous |
| 9 | `eta_sin` | sin(η) | continuous |
| 10 | `eta_cos` | cos(η) | continuous |
| 11 | `theta_sin` | sin(θ) | continuous |
| 12 | `theta_cos` | cos(θ) | continuous |

**`_fill_single_features()` 算法**：

```
初始化 single = zeros(N, 13)

对每个已对齐的 query 位置 q_idx → template 残基 rec：

  # 分类特征
  is_modified  = 0.0 if rec.resname in {A,G,C,U} else 1.0
  is_purine    = 1.0 if rec.base_letter in {A,G} else 0.0
  is_pyrimidine= 1.0 if rec.base_letter in {C,U} else 0.0
  has_anchor   = template_anchor_mask_1d[q_idx]
  has_frame    = template_frame_mask_1d[q_idx]

  # 角度特征 (sin/cos 编码)
  chi   = compute_chi_angle(rec)      → (sin, cos)
  delta = compute_delta_angle(rec)    → (sin, cos)
  eta, theta = compute_eta_theta(prev, rec, next) → 各 (sin, cos)
  
  # 缺失原子导致角度无法计算时 → (0.0, 0.0)
```

### 4.6 Dense Atom 输出（可选）

当 `--emit_dense_atoms` 启用时：

| Key | Shape | 描述 |
|-----|-------|------|
| `template_atom_positions` | `[T, N, 27, 3]` | 每个残基 27 个原子的 3D 坐标 |
| `template_atom_mask` | `[T, N, 27]` | 原子是否存在的 mask |
| `template_atom_names` | `[27]` | 原子名顺序 |

这是为后续自定义 atom-level RNA template encoder 预留的接口。

---

## 5. run_dssr_and_build.py — 一键自动化

### 5.1 定位

将「运行 DSSR → 调用 minimal builder → 调用 extended builder」三步串联为一条命令。

### 5.2 运行方式

```bash
python run_dssr_and_build.py \
  --query_seq query.fa \
  --template template1.cif:A \
  --template template2.pdb:B \
  --outdir ./rna_template_build \
  --mode both \
  --emit_dense_atoms
```

### 5.3 执行流程

```
步骤 1: 对每个 template 运行 DSSR
  x3dna-dssr -i=template.cif --json -o=outdir/template_A.dssr.json

步骤 2: 调用最小兼容版 builder (如果 mode ∈ {minimal, both})
  python build_rna_template_protenix.py \
    --query_seq ... --template ... --output outdir/rna_template_minimal.npz

步骤 3: 调用扩展版 builder (如果 mode ∈ {extended, both})
  python build_rna_template_extended.py \
    --query_seq ... --template ... --dssr_json ... --output outdir/rna_template_extended.npz
```

### 5.4 DSSR 模式选项

- `--dssr_json_mode default`：使用 `--json` 标准输出格式
- `--dssr_json_mode ebi`：使用 `--json=ebi` PDBe 兼容格式

---

## 6. 核心算法详解

### 6.1 RNA Anchor vs 蛋白 Pseudo-Beta：数学对比

**蛋白质**：
```
pseudo_beta = Cβ 坐标 (glycine 用 virtual Cβ)
distance = ||Cβ_i - Cβ_j||
```

**RNA (本方案)**：
```
anchor = mean(碱基重原子坐标)  [fallback: C4' → C1']
distance = ||anchor_i - anchor_j||
```

关键区别：蛋白质的 Cβ 是单一原子，而 RNA 的 base center 是多原子质心。这使得 RNA anchor 对单个原子缺失更鲁棒，同时更能反映碱基-碱基相互作用的实际距离。

### 6.2 RNA Frame vs 蛋白 Backbone Frame：数学对比

**蛋白质**：
```
origin = Cα
e1 = normalize(N - Cα)
e2 = normalize((C - Cα) - dot(C-Cα, e1)·e1)
e3 = cross(e1, e2)
```

**RNA (本方案)**：
```
origin = C4'
e1 = normalize(P - C4')
e2 = normalize((C1'- C4') - dot(C1'-C4', e1)·e1)
e3 = cross(e1, e2)
```

数学形式完全相同（三原子 → Gram-Schmidt → SO(3) 框架），只是参考原子不同。这保证了 `template_unit_vector` 的计算可以直接复用 Protenix 原有的 TemplateEmbedder 逻辑。

### 6.3 Leontis-Westhof 分类法

LW 分类用三个碱基边和顺/反方向描述非标准碱基对：

```
三个碱基相互作用边：
  W = Watson-Crick 边  (标准氢键面)
  H = Hoogsteen 边     (嘌呤 C6-N7-C8 区域)
  S = Sugar 边         (2'-OH 附近)

顺反：
  c = cis   (糖苷键同侧)
  t = trans (糖苷键异侧)

组合：
  第一个字母(c/t) + 第一个核苷酸的边(W/H/S) + 第二个核苷酸的边(W/H/S)
  
  例: cWH = cis, nt1 用 Watson-Crick 边, nt2 用 Hoogsteen 边
```

为什么需要 18 类而不是简单的 canonical/non-canonical？因为不同 LW 类型对应完全不同的几何构型和结构功能。例如 tHS 在 RNA 三级结构中常见于 A-minor motif，而 cWW 是经典 Watson-Crick 配对。

### 6.4 虚拟骨架角 (η, θ) 的结构意义

η 和 θ 是由 Wadley & Pyle (2007) 提出的 RNA 简化骨架描述符：

```
η (eta):   C4'(i-1) — P(i) — C4'(i) — P(i+1)
θ (theta): P(i)     — C4'(i) — P(i+1) — C4'(i+1)
```

它们的 (η, θ) 散点图构成 RNA 版的 Ramachandran plot，可以区分不同的局部构象簇。常见的 RNA 构象（A-form 螺旋、GNRA tetraloop、kink-turn 等）在此空间中占据不同区域。

---

## 7. 输出张量速查表

### 7.1 最小兼容版输出

```
rna_template_minimal.npz
├── template_aatype               [4, N]         int32
├── template_distogram            [4, N, N, 39]  float32
├── template_pseudo_beta_mask     [4, N, N]      float32
├── template_unit_vector          [4, N, N, 3]   float32
├── template_backbone_frame_mask  [4, N, N]      float32
├── template_anchor_pos           [4, N, 3]      float32   (debug)
├── template_anchor_mask_1d       [4, N]         float32   (debug)
├── template_frame_origin         [4, N, 3]      float32   (debug)
├── template_frame_axes           [4, N, 3, 3]   float32   (debug)
├── template_frame_mask_1d        [4, N]         float32   (debug)
├── template_mask                 [4]            float32
├── template_names                [4]            <U256
├── template_mapping_json         [4]            <U1048576
├── query_sequence                [N]            <U1
└── template_anchor_mode          scalar         <U*
```

### 7.2 扩展版额外输出

```
rna_template_extended.npz  (包含上述全部 + 以下)
├── rna_template_pair             [4, N, N, 25]  float32
├── rna_template_single           [4, N, 13]     float32
├── rna_template_pair_names       [25]           <U64
├── rna_template_single_names     [13]           <U64
│
│   以下仅当 --emit_dense_atoms 时：
├── template_atom_positions       [4, N, 27, 3]  float32
├── template_atom_mask            [4, N, 27]     float32
└── template_atom_names           [27]           <U8
```

---

## 8. 数据流全景图

```
┌─────────────────┐     ┌──────────────┐
│  mmCIF / PDB    │     │ Query 序列    │
│  template files │     │ (FASTA/text) │
└────────┬────────┘     └──────┬───────┘
         │                      │
         ▼                      │
  load_structure_residues()     │
         │                      │
         ▼                      ▼
  List[ResidueRecord]    normalize_query_sequence()
         │                      │
         ├──────────────────────┤
         │                      │
         ▼                      ▼
  residues_to_sequence()  query_seq (clean)
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
          align_query_to_template()
                    │
                    ▼
            mapping: {q_idx → t_idx}
                    │
         ┌──────────┴──────────┐
         │                      │
         ▼                      ▼
   ┌─ 最小兼容版 ─┐      ┌─ 扩展版 (额外) ─┐
   │              │      │                 │
   │ compute_     │      │ DSSR JSON       │
   │  anchor()    │      │    ↓            │
   │ compute_     │      │ collect_dssr_   │
   │  frame()     │      │  pair_edges()   │
   │ compute_     │      │ collect_dssr_   │
   │  distogram() │      │  stack_edges()  │
   │ compute_unit │      │ compute_chi/    │
   │  _vectors()  │      │  delta/eta/θ    │
   │              │      │                 │
   └──────┬───────┘      └───────┬─────────┘
          │                       │
          ▼                       ▼
   5 key template dict    pair [N,N,25]
                          single [N,13]
                          (opt) atoms [N,27,3]
          │                       │
          └───────────┬───────────┘
                      │
                      ▼
            stack_template_dicts()
                      │
                      ▼
            save_npz_with_metadata()
                      │
                      ▼
              .npz + .meta.json
```

---

## 附录 A：常见问题处理策略

| 场景 | 脚本行为 |
|------|---------|
| 修饰碱基 (如 PSU, 5MC) | 查 `COMMON_MODIFIED_BASE_MAP` 退化为标准碱基；查不到则按特征原子推断 |
| 原子缺失 | anchor/frame 的 fallback 链逐级降级；最终 mask=0 |
| 首位残基缺 P | frame 计算尝试 O5' 或 C3' 替代 |
| disordered atoms | 选 occupancy 最高的 alt conf |
| DSSR nt_id 格式不一致 | 宽松正则匹配 + alias 机制 |
| 序列不完全匹配 | 全局对齐容忍 mismatch 和 gap |

## 附录 B：推荐实验路线

```
阶段 A: 最小兼容版 baseline
  ├─ 同一 RNA target，有/无 template 对比
  └─ 验证 Protenix 能否吃到 RNA 几何先验

阶段 B: 扩展版增量实验
  ├─ 加入 rna_template_pair + rna_template_single
  ├─ 并行 RNATemplateEmbedder
  └─ 融合方式: z += g1·z_minimal + g2·z_rna_ext

阶段 C: Atom-level encoder (可选)
  ├─ 使用 --emit_dense_atoms 的 27 原子坐标
  └─ 自定义 atom-level template attention
```
