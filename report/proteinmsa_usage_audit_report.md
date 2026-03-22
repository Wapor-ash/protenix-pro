# Protenix RNA Fine-tuning Pipeline - ProteinMSA 使用审查报告

**审查日期**: 2026-03-07  
**审查范围**: `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix`  
**审查目标**: 验证 pipeline 是否正确使用 proteinmsa，以及当 load 的模型默认使用 protein msa 时的行为分析

---

## 1. 执行摘要

### 核心发现

| 问题 | 审查结果 |
|------|---------|
| **是否使用了 proteinmsa？** | ✅ **是**，代码中完整支持 protein MSA 和 RNA MSA |
| **RNA fine-tuning 配置** | ✅ `enable_prot_msa=false`, `enable_rna_msa=true` |
| **空 MSA 处理** | ✅ 代码有完善的空 MSA 处理逻辑 |
| **模型兼容性** | ✅ 预训练模型可以安全用于 RNA-only fine-tuning |

---

## 2. 代码审查详情

### 2.1 MSA 配置入口 (`run_rna_finetune.sh`)

```bash
# RNA MSA 配置
--data.msa.enable_rna_msa true \
--data.msa.rna_msadir_raw_paths "${RNA_MSA_DIR}" \
--data.msa.rna_seq_or_filename_to_msadir_jsons "${RNA_MSA_JSON}" \
--data.msa.rna_indexing_methods "sequence" \

# Protein MSA 配置 (明确禁用)
--data.msa.enable_prot_msa false \

# Protein Template 配置 (明确禁用)
--data.template.enable_prot_template false
```

**审查结论**: 配置正确，RNA fine-tuning 时明确禁用了 protein MSA 和 template。

---

### 2.2 MSA 数据处理流程

#### 2.2.1 `MSAFeaturizer` 初始化 (`protenix/data/msa/msa_featurizer.py`)

```python
class MSAFeaturizer:
    def __init__(
        self,
        dataset_name: str = "",
        prot_seq_or_filename_to_msadir_jsons: Sequence[str] = [""],
        prot_msadir_raw_paths: Sequence[str] = [""],
        rna_seq_or_filename_to_msadir_jsons: Sequence[str] = [""],
        rna_msadir_raw_paths: Sequence[str] = [""],
        prot_pairing_dbs: Sequence[str] = [""],
        prot_non_pairing_dbs: Sequence[str] = [""],
        prot_indexing_methods: Sequence[str] = ["sequence"],
        rna_indexing_methods: Sequence[str] = ["sequence"],
        enable_prot_msa: bool = True,      # ← 关键参数
        enable_rna_msa: bool = True,       # ← 关键参数
    ) -> None:
        # Initialize source managers for protein and RNA
        self.prot_mgr = MSASourceManager(
            prot_msadir_raw_paths,
            prot_indexing_methods,
            {i: load_json_cached(p) for i, p in enumerate(prot_seq_or_filename_to_msadir_jsons)},
            enable_prot_msa,  # ← 传入 enable_prot_msa
        )
        self.rna_mgr = MSASourceManager(
            rna_msadir_raw_paths,
            rna_indexing_methods,
            {i: load_json_cached(p) for i, p in enumerate(rna_seq_or_filename_to_msadir_jsons)},
            enable_rna_msa,  # ← 传入 enable_rna_msa
        )
```

**审查结论**: 
- `enable_prot_msa=false` 时，`self.prot_mgr.enabled=false`
- `MSASourceManager.fetch_msas()` 会在 `enabled=false` 时直接返回 `([], [])`

---

#### 2.2.2 `MSASourceManager.fetch_msas()` 行为 (`protenix/data/msa/msa_featurizer.py`)

```python
def fetch_msas(
    self,
    sequence: str,
    pdb_id: str,
    chain_type: str,
    p_dbs: Optional[Sequence[str]] = None,
    np_dbs: Optional[Sequence[str]] = None,
) -> Tuple[List[RawMsa], List[RawMsa]]:
    if not self.enabled:  # ← 当 enable_prot_msa=false 时
        return [], []     # ← 直接返回空列表，不会尝试加载 protein MSA
    
    unpaired, paired = [], []
    
    # RNA-specific loading logic
    if chain_type == RNA_CHAIN:
        for path, method, m_key in zip(self.raw_paths, self.methods, self.mappings):
            if method != "sequence":
                continue
            mapping = self.mappings[m_key]
            if sequence not in mapping:
                continue
            eid = str(mapping[sequence][0])
            fpath = opjoin(path, eid, f"{eid}_all.a3m")
            if opexists(fpath):
                with open(fpath, "r") as f:
                    content = f.read()
                if content:
                    unpaired.append(RawMsa.from_a3m(...))
        return unpaired, []
    
    # Protein-specific loading logic (仅在 enable_prot_msa=true 时执行)
    if chain_type == PROTEIN_CHAIN:
        ...
```

**审查结论**: 
- ✅ **Protein 链**: 当 `enable_prot_msa=false` 时，直接返回 `([], [])`，**不会给空 MSA**
- ✅ **RNA 链**: 当 `enable_rna_msa=true` 时，按 sequence 索引加载 RNA MSA

---

#### 2.2.3 `FeatureAssemblyLine.assemble()` 空 MSA 处理 (`protenix/data/msa/msa_featurizer.py`)

```python
def assemble(
    self, bioassembly: Mapping[int, Mapping[str, Any]], std_idxs: np.ndarray
) -> "MSAFeat":
    # 1. Base featurization
    for aid, info in bioassembly.items():
        ctype, seq = info["chain_entity_type"], info["sequence"]
        skip = ctype not in STANDARD_POLYMER_CHAIN_TYPES or len(seq) <= 4
        
        if ctype in STANDARD_POLYMER_CHAIN_TYPES:
            up_msa = RawMsa.from_a3m(
                seq,
                ctype,
                (
                    info["unpaired_msa"]  # ← 当 MSA 为空时，这里是 ""
                    if not skip and ctype in [PROTEIN_CHAIN, RNA_CHAIN]
                    else ""
                ),
                dedup=True,
            )
            p_msa = RawMsa.from_a3m(
                seq,
                ctype,
                (
                    info["paired_msa"]  # ← 当 paired MSA 为空时，这里是 ""
                    if not skip and need_pairing and ctype == PROTEIN_CHAIN
                    else ""
                ),
                dedup=False,
            )
        else:
            # Ligand placeholders
            u_msa = p_msa = RawMsa(seq, PROTEIN_CHAIN, [], [], deduplicate=False)
```

**审查结论**: 
- ✅ 空 MSA 字符串 `""` 会被 `RawMsa.from_a3m()` 处理
- ✅ `RawMsa` 类有保护逻辑确保至少有 query sequence

---

#### 2.2.4 `RawMsa` 空 MSA 保护逻辑 (`protenix/data/msa/msa_utils.py`)

```python
class RawMsa:
    def __init__(
        self,
        query: str,
        chain_type: str,
        sequences: Sequence[str],
        descriptions: Sequence[str],
        deduplicate: bool = True,
    ) -> None:
        self.query = query
        self.chain_type = chain_type
        if deduplicate:
            self.seqs, self.descs = self._deduplicate_sequences(sequences, descriptions)
        else:
            self.seqs, self.descs = list(sequences), list(descriptions)
        
        # Make sure the MSA always has at least the query.  ← 关键保护逻辑
        self.seqs = self.seqs or [query]
        self.descs = self.descs or ["Original query"]
        
        if not self._verify_query():
            raise ValueError(...)
```

**审查结论**: 
- ✅ **即使 MSA 为空，也会保留 query sequence 作为单序列 MSA**
- ✅ 不会给"真正的空 MSA"，而是给"只有 query 的 MSA"

---

### 2.3 数据 Pipeline 集成 (`protenix/data/pipeline/data_pipeline.py`)

```python
@staticmethod
def get_msa_raw_features(
    bioassembly_dict: dict[str, Any],
    selected_indices: np.ndarray,
    msa_featurizer: Optional[MSAFeaturizer],
) -> dict[str, np.ndarray]:
    if msa_featurizer is None:  # ← 当 MSA 完全禁用时
        return {}               # ← 返回空字典
    
    entity_to_asym_id_int = dict(
        DataPipeline.get_label_entity_id_to_asym_id_int(
            bioassembly_dict["atom_array"]
        )
    )
    
    msa_feats = msa_featurizer(
        bioassembly_dict=bioassembly_dict,
        selected_indices=selected_indices,
        entity_to_asym_id_int=entity_to_asym_id_int,
    )
    
    return msa_feats
```

**审查结论**: 
- ✅ MSA featurizer 为 None 时返回空字典
- ✅ MSA featurizer 存在但 `enable_prot_msa=false` 时，protein 链返回空 MSA 列表，最终生成只有 query 的 MSA

---

### 2.4 模型输入处理 (`protenix/model/protenix.py`)

```python
class Protenix(nn.Module):
    def __init__(self, configs: Any) -> None:
        ...
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data.get("msa", {}),
        )
        ...
```

**审查结论**: 
- ✅ 模型通过 `MSAModule` 处理 MSA 特征
- ✅ MSA 特征为空时，模型会使用 query-only 的 MSA 进行推理

---

## 3. 关键问题解答

### Q1: Pipeline 是否使用了 proteinmsa？

**答案**: 
- **代码层面**: ✅ 是，完整支持 protein MSA 和 RNA MSA
- **RNA fine-tuning 配置**: ❌ 否，明确设置 `enable_prot_msa=false`

### Q2: 如果 load 的模型默认使用 protein MSA，会给空的 MSA 吗？

**答案**: **不会给"真正的空 MSA"**，而是：

1. **Protein 链** (当 `enable_prot_msa=false`):
   - `MSASourceManager.fetch_msas()` 返回 `([], [])`
   - `RawMsa` 构造函数确保 `self.seqs = self.seqs or [query]`
   - **最终结果**: 只有 query sequence 的单序列 MSA

2. **RNA 链** (当 `enable_rna_msa=true`):
   - 从 `rna_msadir_raw_paths` 按 sequence 索引加载 RNA MSA
   - 如果找不到匹配的 MSA，同样回退到 query-only MSA

### Q3: 预训练模型 (默认使用 protein MSA) 能用于 RNA fine-tuning 吗？

**答案**: ✅ **可以安全使用**

**原因**:
1. 模型架构相同，MSA 输入维度一致
2. `enable_prot_msa=false` 时，protein 链使用 query-only MSA
3. `enable_rna_msa=true` 时，RNA 链使用真实 RNA MSA
4. 模型在 fine-tuning 过程中会适应新的 MSA 分布

---

## 4. 数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                    run_rna_finetune.sh                          │
│  --data.msa.enable_prot_msa false                               │
│  --data.msa.enable_rna_msa true                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MSAFeaturizer                                │
│  self.prot_mgr = MSASourceManager(..., enabled=false)           │
│  self.rna_mgr  = MSASourceManager(..., enabled=true)            │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│   Protein Chains        │     │     RNA Chains          │
│  fetch_msas() → []      │     │  fetch_msas() → [MSA]   │
│  RawMsa(seqs=[query])   │     │  RawMsa(seqs=[...])     │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              FeatureAssemblyLine.assemble()                     │
│  - Merge all chains                                             │
│  - Handle empty MSA rows                                        │
│  - Forward compatibility patch for non-protein entities         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Protenix Model                               │
│  - InputFeatureEmbedder                                         │
│  - MSAModule                                                    │
│  - PairformerStack                                              │
│  - DiffusionModule                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. 风险评估

| 风险项 | 风险等级 | 缓解措施 |
|--------|---------|---------|
| Protein MSA 缺失导致模型性能下降 | 🟡 中 | RNA fine-tuning 目标就是适应 RNA-only 场景 |
| RNA MSA 加载失败 | 🟢 低 | 代码有 query-only 回退逻辑 |
| 预训练模型与 fine-tuning 配置不兼容 | 🟢 低 | MSA 输入维度一致，模型可适应 |
| 空 MSA 导致模型崩溃 | 🟢 低 | `RawMsa` 确保至少有 query sequence |

---

## 6. 结论与建议

### 6.1 结论

1. ✅ **Pipeline 代码正确使用 proteinmsa 模块**
2. ✅ **RNA fine-tuning 配置正确** (`enable_prot_msa=false`, `enable_rna_msa=true`)
3. ✅ **空 MSA 处理完善**，不会给"真正的空 MSA"
4. ✅ **预训练模型可以安全用于 RNA fine-tuning**

### 6.2 代码执行行为总结

| 场景 | Protein 链 MSA | RNA 链 MSA |
|------|---------------|-----------|
| `enable_prot_msa=false` | Query-only (单序列) | N/A |
| `enable_rna_msa=true` + MSA 存在 | N/A | 真实多序列 MSA |
| `enable_rna_msa=true` + MSA 缺失 | N/A | Query-only (单序列) |

### 6.3 建议

1. **无需修改代码**: 当前配置和代码逻辑已经正确处理了 RNA fine-tuning 场景
2. **监控训练指标**: 关注 `N_msa_prot_unpair` 和 `N_msa_rna_unpair` 指标，确认 MSA 加载正常
3. **验证 RNA MSA 数据**: 确保 `${RNA_MSA_DIR}` 和 `${RNA_MSA_JSON}` 路径正确且包含有效数据

---

## 7. 参考文件

| 文件路径 | 描述 |
|---------|------|
| `protenix/data/msa/msa_featurizer.py` | MSA 数据加载和特征化 |
| `protenix/data/msa/msa_utils.py` | MSA 工具类和数据结构 |
| `protenix/data/pipeline/data_pipeline.py` | 数据 Pipeline |
| `protenix/data/pipeline/dataset.py` | Dataset 类和 MSAFeaturizer 工厂函数 |
| `configs/configs_data.py` | 默认数据配置 |
| `run_rna_finetune.sh` | RNA fine-tuning 启动脚本 |

---

**报告生成时间**: 2026-03-07  
**审查者**: Code Review Skill  
**状态**: ✅ 审查完成，无代码修改
