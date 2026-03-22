# Protenix RNA/DNA Pipeline 完整整合报告

**作者**: Ash
**日期**: 2026-03-16
**项目**: Protenix RNA/DNA LLM Fusion + RNA Template Integration

---

## 目录

1. [项目概述与整合总结](#1-项目概述与整合总结)
2. [数据链路 — 完整数据准备流程](#2-数据链路--完整数据准备流程)
   - 2.1 [新数据如何产生 Finetune 数据](#21-新数据如何产生-finetune-数据)
   - 2.2 [指定 LLM 模型产生 Embeddings](#22-指定-llm-模型产生-embeddings)
   - 2.3 [LLM Embeddings 整合进 Finetune 链路](#23-llm-embeddings-整合进-finetune-链路)
   - 2.4 [RNA Template 数据准备](#24-rna-template-数据准备)
   - 2.5 [Offline RNA MSA 准备](#25-offline-rna-msa-准备)
   - 2.6 [Pretrain 操作完整检查清单](#26-pretrain-操作完整检查清单)
3. [训练阶段 — LLM Part 工作流](#3-训练阶段--llm-part-工作流)
   - 3.1 [RNALM 开启时的工作流](#31-rnalm-开启时的工作流)
   - 3.2 [RNA LLM / DNA LLM 分别开启的工作流](#32-rna-llm--dna-llm-分别开启的工作流)
   - 3.3 [Separate Projection 与 Init 选项](#33-separate-projection-与-init-选项)
4. [训练阶段 — MSA Part 工作流](#4-训练阶段--msa-part-工作流)
5. [训练阶段 — Template Part 工作流](#5-训练阶段--template-part-工作流)
   - 5.1 [RNA Template 开启的完整链路](#51-rna-template-开启的完整链路)
   - 5.2 [人工指定 Template 链路](#52-人工指定-template-链路)
   - 5.3 [默认 Template 链路（Online 自动搜索）](#53-默认-template-链路online-自动搜索)
   - 5.4 [Template 执行流程详解](#54-template-执行流程详解)
6. [Projector 设置与效果](#6-projector-设置与效果)
7. [训练 Stage 链路](#7-训练-stage-链路)
   - 7.1 [1-Stage 训练链路](#71-1-stage-训练链路)
   - 7.2 [2-Stage 训练链路](#72-2-stage-训练链路)
8. [完整参数手册](#8-完整参数手册)
9. [测试用例与验证](#9-测试用例与验证)
10. [附录 — 代码索引](#10-附录--代码索引)

---

## 1. 项目概述与整合总结

Protenix 是基于 AlphaFold3 架构的蛋白质/核酸结构预测模型。Ash 对 Protenix 进行了两个核心 pipeline 整合：

### 整合一：RNA/DNA LLM 融合进 Pipeline

将预训练语言模型（RiNALMo / AIDO）的 RNA 和 DNA embedding 融合进 Protenix 的训练和推理流程。

```
┌─────────────────────────────────────────────────────────┐
│                  RNA/DNA LLM 融合架构                     │
│                                                          │
│  预计算阶段:                                              │
│  RNA序列 ──→ RiNALMo/AIDO.RNA ──→ RNA Embedding [N,1280] │
│  DNA序列 ──→ AIDO.DNA          ──→ DNA Embedding [N,1024] │
│                                                          │
│  注入点:                                                  │
│  ┌──────────────┐     ┌──────────────────┐               │
│  │ Input Embedder│     │ Diffusion Module │               │
│  │  (input注入)  │     │  (diffusion注入)  │               │
│  └──────────────┘     └──────────────────┘               │
│         ↓                      ↓                         │
│    s_inputs += proj(emb)   s_rnalm = proj(emb)           │
│         ↓                      ↓                         │
│    InputFeatureEmbedder    DiffusionConditioning          │
│    rna_projection          rna_projection                 │
│    dna_projection            dna_projection                 │
│                                                          │
│  门控机制: none / scalar / token / dual                    │
└─────────────────────────────────────────────────────────┘
```

### 整合二：RNA Template 融合进 Pipeline

将 RNA 结构模版（类似蛋白质 template）融合进 Protenix 的 PairFormer 模块。

补充说明: 当前实现和原版 Protenix 一样，**统一的是样本级对象**（`bioassembly_dict`、`token_array`、`entity_id` 等），但 **RNA LLM / RNA MSA / RNA template 在 sequence artifact lookup 这一层并没有统一成一个公共接口**。三条链共享样本语义，但各自查询各自的 embedding、MSA、template 资源。

```
┌─────────────────────────────────────────────────────────┐
│                 RNA Template 融合架构                      │
│                                                          │
│  模版准备:                                                │
│  RNA序列 ──→ 预先生成 search_results.json ──→ 命中CIF    │
│         └──────────────────────────────────────→ 3D特征提取│
│                                                          │
│  特征格式:                                                │
│  ┌─────────────────────────────────────────────┐         │
│  │ distogram [T,N,N,39]  (距离直方图)            │         │
│  │ backbone_frame_mask [T,N,N]                  │         │
│  │ unit_vector [T,N,N,3]                        │         │
│  │ pseudo_beta_mask [T,N,N]                     │         │
│  │ aatype [T,N]                                 │         │
│  │ rna_template_block_mask [N,N]  (RNA-RNA pair mask) │     │
│  └─────────────────────────────────────────────┘         │
│                                                          │
│  注入点: TemplateEmbedder (PairFormer)                    │
│  ┌──────────────────────────────────────────┐            │
│  │ protein: linear_no_bias_a  ──→ template trunk │         │
│  │ RNA:     linear_no_bias_a_rna ──→ same trunk  │         │
│  │ fusion:  template contribution加回 pair 表示   │         │
│  └──────────────────────────────────────────┘            │
│                                                          │
│  初始化: protein (copy权重+alpha门) / zero                │
│  防泄漏: 仅默认 search-based template 在 training 时执行   │
│          时间过滤 + self-hit 排除                          │
└─────────────────────────────────────────────────────────┘
```

### 关键文件位置

| 组件 | 路径 |
|------|------|
| 模型代码 | `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_pro/Protenix/protenix/` |
| 配置文件 | `configs/configs_base.py`, `configs/configs_data.py` |
| 训练入口 | `runner/train.py` |
| 推理入口 | `runner/inference.py` |
| RNA LLM Featurizer | `protenix/data/rnalm/rnalm_featurizer.py` |
| RNA Template Featurizer | `protenix/data/rna_template/rna_template_featurizer.py` |
| MSA Featurizer | `protenix/data/msa/msa_featurizer.py` |
| InputEmbedder (LLM注入) | `protenix/model/modules/embedders.py` |
| DiffusionConditioning (LLM注入) | `protenix/model/modules/diffusion.py` |
| TemplateEmbedder (模版注入) | `protenix/model/modules/pairformer.py` |
| RNA Loss | `protenix/model/rna_loss.py` |
| Protenix 主模型 | `protenix/model/protenix.py` |
| 数据 | `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/` |
| Embeddings | `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/protenix_rna/` |
| RNA Template DB | `/inspire/ssd/project/sais-bio/public/ash_proj/data/RNA3D/` |
| Template Index | `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_pro/Protenix/rna_database/` |

---

## 2. 数据链路 — 完整数据准备流程

### 2.1 新数据如何产生 Finetune 数据

当新的 train/test 数据到来时，需要执行以下步骤将原始 mmCIF/PDB 文件转化为 Protenix 可用的 finetune 数据：

```
新数据 (mmCIF/PDB)
     │
     ▼
┌─────────────────────────────────────┐
│ Step 1: prepare_training_data.py    │
│ 解析mmCIF → Bioassembly → 序列化    │
│                                     │
│ 输入: mmCIF目录                      │
│ 输出: indices.csv + bioassembly/     │
│       (pkl.gz files)                │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Step 2: 生成 ESM Embeddings (蛋白)   │
│ python protenix/data/esm/           │
│   compute_esm.py                    │
│                                     │
│ 输入: 蛋白序列                       │
│ 输出: esm_embeddings/ (.pt files)    │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Step 3: 生成 RNA/DNA LLM Embeddings  │
│ (详见 2.2)                           │
│                                     │
│ 输入: RNA/DNA序列                    │
│ 输出: rnalm_embeddings/ (.pt files)  │
│       dna_embeddings/ (.pt files)    │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Step 4: MSA搜索 (蛋白 + RNA)         │
│ (详见 2.5)                           │
│                                     │
│ 输入: 序列                           │
│ 输出: msa/ (.a3m files)             │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Step 5: RNA Template准备             │
│ (详见 2.4)                           │
│                                     │
│ 输入: RNA序列 + RNA3D数据库           │
│ 输出: search_results.json            │
│       + CIF files ready             │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Step 6: 数据组装                     │
│ 配置 configs 指向以上输出路径         │
│ 开始训练                             │
└─────────────────────────────────────┘
```

**具体命令：**

```bash
# Step 1: 准备训练数据
cd /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_pro/Protenix

python scripts/prepare_training_data.py \
    -i /path/to/new/mmcif/directory \
    -o /path/to/output/indices.csv \
    -b /path/to/output/bioassembly/ \
    -c /path/to/cluster.txt \     # 可选: 序列聚类文件
    -n 16                          # 并行进程数
```

**`prepare_training_data.py` 内部流程：**
1. `MMCIFParser` 解析每个 mmCIF 文件
2. 提取 bioassembly 结构（多聚体组装）
3. `make_sample_indices()` 生成 chain 和 interface 索引
4. `TokenArray` + `AtomArray` 序列化
5. 输出 `pkl.gz` + `indices.csv`（每行一个训练样本）

**当前工作区的一个关键前置依赖：**

无论你是跑通用的 `prepare_training_data.py`，还是跑 Stanford RNA 这套
`run_bioassembly.py` / `rerun_after_parser_logic_change.sh`，都要先确保：

```bash
export PROTENIX_ROOT_DIR=/inspire/ssd/project/sais-bio/public/ash_proj/data
ls ${PROTENIX_ROOT_DIR}/common/components.cif
ls ${PROTENIX_ROOT_DIR}/common/components.cif.rdkit_mol.pkl
```

原因是当前 `configs/configs_data.py` 默认把 `PROTENIX_ROOT_DIR` 回落到 `$HOME`。
如果不显式设置，在 root 环境里就会去找 `/root/common/components.cif`，然后出现：

```text
Gen data failed ... due to [Errno 2] No such file or directory: '/root/common/components.cif'
```

进一步会导致 `bioassembly` 部分看起来“跑完了”，但最后写出空的 `indices.csv`。

**如果你不是在接入一批全新的 mmCIF，而是在当前 Stanford RNA 数据上重刷 parser logic：**

应优先使用已经整理好的重跑脚本，而不是再手动拼命令：

```bash
conda activate protenix
export PROJECT_ROOT=/inspire/ssd/project/sais-bio/public/ash_proj
export PROTENIX_ROOT_DIR=${PROJECT_ROOT}/data

cd ${PROJECT_ROOT}/data/stanford-rna-3d-folding/part2
bash rerun_after_parser_logic_change.sh full
```

这里之所以推荐 `full`，是因为 parser logic 变更后，`bioassembly`、`sequences.json`、
AIDO RNA/DNA manifests 最好一起保持一致；只修 `indices` 往往不够。

### 2.2 指定 LLM 模型产生 Embeddings

Protenix 支持两种 LLM 模型产生 embeddings：

| 模型 | RNA Dim | DNA Dim | 说明 |
|------|---------|---------|------|
| **RiNALMo** | 1280 | — | RNA-only 模型 |
| **AIDO** | 2048 (RNA) | 1024 (DNA) | 支持 RNA + DNA |

**产生 RNA Embeddings 的流程：**

```bash
# 使用已有的embedding生成脚本
cd /inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/protenix_rna

# 生成真实的 RiNALMo embeddings
python generate_real_rnalm_embeddings.py \
    --input_csv /path/to/train_sequences.csv \
    --output_dir /path/to/rnalm_embeddings/ \
    --model_name rinalmo  # 或 aido

# 生成 DNA embeddings (AIDO.DNA)
python generate_real_rnalm_embeddings.py \
    --input_csv /path/to/dna_sequences.csv \
    --output_dir /path/to/dna_embeddings/ \
    --model_name aido_dna
```

**Embedding 目录结构：**
```
rnalm_embeddings/
├── 0/
│   └── seq_0001.pt      # torch tensor [seq_len, embedding_dim]
├── 1/
│   └── seq_0002.pt
├── ...
└── manifest.csv
    # 关键不是文件名本身，而是 CSV 必须能被 RiNALMoFeaturizer 解析。
    # 当前代码默认读取列: seq, part_id, seq_label
    # 并拼接为 `${part_id}/${seq_label}.pt`
```

更准确地说，当前 `RiNALMoFeaturizer` 不是读取通用的 `sequence,embedding_path` 两列表，而是通过 `seq / part_id / seq_label` 这些列构造相对路径。  
如果你的生成脚本输出格式不同，最终需要适配到这个 contract，或者同步修改 featurizer。

**关键配置参数：**
```bash
# 在训练命令中指定
--rnalm.enable true
--rnalm.model_name "rinalmo"           # 或 "aido"
--rnalm.embedding_dim 1280             # RiNALMo=1280, AIDO.RNA=2048
--rnalm.embedding_dir "/path/to/rnalm_embeddings/"
--rnalm.sequence_fpath "/path/to/manifest.csv"

# 如果使用 DNA embeddings
--rnalm.use_dna_embed true
--rnalm.dna_embedding_dim 1024
--rnalm.dna_embedding_dir "/path/to/dna_embeddings/"
--rnalm.dna_sequence_fpath "/path/to/dna_manifest.csv"
```

**当新数据来了，跑什么流程可以产生新数据的 LLM：**

```
新RNA/DNA序列
     │
     ▼
┌──────────────────────────────────┐
│ 1. 提取序列                      │
│ 从 bioassembly pkl.gz 中提取     │
│ 所有 RNA/DNA chain 序列          │
│ 保存为 train_sequences.csv       │
└──────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────┐
│ 2. 运行 LLM 推理                 │
│ 加载 RiNALMo/AIDO 模型           │
│ 逐序列 forward → 保存 .pt        │
│ 生成 manifest.csv                │
└──────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────┐
│ 3. 配置路径                      │
│ --rnalm.embedding_dir = 输出目录  │
│ --rnalm.sequence_fpath = manifest│
└──────────────────────────────────┘
```

### 2.3 LLM Embeddings 整合进 Finetune 链路

LLM embeddings 通过 `RiNALMoFeaturizer` 在训练时按需加载：

```
训练数据加载流程:
┌───────────────┐
│ DataLoader    │
│ (per batch)   │
└───────┬───────┘
        │
        ▼
┌───────────────────────────────────────────┐
│ DataPipeline.crop()                       │
│ 1. 选择crop区域                           │
│ 2. 裁剪 token_array, atom_array           │
│ 3. 调用各 featurizer                       │
│                                           │
│ RiNALMoFeaturizer(                        │
│     token_array,                          │
│     atom_array,                           │
│     bioassembly_dict,                     │
│     return_separate=True/False            │
│ )                                         │
│                                           │
│ 返回:                                     │
│ separate=True:                            │
│   {"rna_llm_embedding": [N,1280],         │
│    "dna_llm_embedding": [N,1024]}         │
│ separate=False:                           │
│   {"rnalm_token_embedding": [N,1280]}     │
└───────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────┐
│ input_feature_dict 包含:                   │
│ - restype (32-dim one-hot)                │
│ - profile (32-dim)                        │
│ - deletion_mean (1-dim)                   │
│ - esm_embedding (如有)                     │
│ - rna_llm_embedding / dna_llm_embedding   │
│ - msa features (如有)                      │
│ - template features (如有)                 │
│ - rna_template features (如有)             │
└───────────────────────────────────────────┘
```

**Featurizer 内部的 RNA-First 策略：**

```python
# rnalm_featurizer.py 中的实体识别逻辑
# 优先级: RNA > DNA
#
# 1. 含U碱基 → 一定是RNA → 使用RNA模型
# 2. 纯ACGT → DNA → 使用DNA模型
# 3. RNA标签但纯ACGT → 降级为DNA（Reverse-RNA-First）
# 4. 修饰碱基(X) → strip后重试manifest

# 多级fallback:
# RNA路径: 直接匹配 → T→U转换匹配 → DNA manifest兜底
# DNA路径: 直接匹配 → T→U转换匹配 → 拒绝含U序列
```

### 2.4 RNA Template 数据准备

如果要对新数据启用 RNA Template，有两种模式。当前训练主路径应优先使用 **Online 模式**。

#### 模式一：Online 模式（推荐，训练/推理时在线构建模板特征）

```
Stanford RNA 数据经 Protenix finetune 预处理后的产物
     │
     ├─ part2/train_labels.csv
     │    (残基级标签表，可从中重建每个训练 PDB 的 RNA 序列)
     ├─ part2/protenix_prepared/rna_train_pdb_list_filtered.txt
     │    (finetune 预处理后筛出的最终训练 PDB 列表；本次为 5574 个)
     ├─ part2/PDB_RNA/                    (训练时在线读 CIF)
     └─ RNA3D/rna3db-mmcifs/             (构 catalog 用)
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│ Step 1: 构建全量 RNA catalog                                │
│                                                            │
│ python rna_template/scripts/01_extract_rna_catalog.py      │
│   --pdb_rna_dir /path/to/RNA3D/rna3db-mmcifs               │
│   --output rna_database/rna_catalog.json                   │
│   --max_structures 0 --min_length 10 --max_length 2000     │
└────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│ Step 2: 生成训练 query 映射 JSON                            │
│                                                            │
│ python rna_template/scripts/                               │
│   generate_training_sequences_from_labels.py               │
│   --labels_csv /path/to/part2/train_labels.csv             │
│   --pdb_list /path/to/rna_train_pdb_list_filtered.txt      │
│   --output_json /path/to/rna_sequence_to_pdb_chains.json   │
│   --min_length 1                                            │
│                                                            │
│ 输出格式必须是: {sequence: [PDB_ID, ...]}                   │
└────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│ Step 3: 预先跑 MMseqs2 搜索                                 │
│                                                            │
│ python rna_template/scripts/03_mmseqs2_search.py          │
│   --catalog rna_database/rna_catalog.json                  │
│   --template_dir rna_database/templates                    │
│   --training_sequences /path/to/rna_sequence_to_pdb_...json│
│   --output_search rna_database/search_results.json         │
│   --output_index rna_database/rna_template_index.json      │
│   --strategy mmseqs2                                       │
│   --min_identity 0.3 --max_templates 4                     │
│   --sensitivity 7.5 --evalue 1e-3 --num_threads 8          │
│                                                            │
│ 输出: {query_id: {query_sequence, templates[]}}            │
└────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│ Step 4: 按最终 cif_database_dir 清洗 search_results.json    │
│                                                            │
│ 如果训练时要用 /path/to/part2/PDB_RNA 作为 cif_database_dir │
│ 就要确保 search_results.json 里的 hit PDB 在该目录可解析。  │
│ 否则 online 构模板时会命中 search_results 但找不到 CIF。     │
└────────────────────────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────────────────────────┐
│ Step 5: 配置训练参数                                        │
│                                                            │
│ --rna_template.enable true                                 │
│ --rna_template.search_results_path rna_database/search_... │
│ --rna_template.cif_database_dir /path/to/part2/PDB_RNA     │
│ --rna_template.rna3db_metadata_path /path/to/RNA3D/filter.json │
│ --rna_template.max_rna_templates 4                         │
└────────────────────────────────────────────────────────────┘
```

这里的 “online” 指的是 **训练/推理时根据预先生成的 `search_results.json` 在线构建 template features**，  
不是在 `runner/train.py` 内部实时再跑一遍 MMseqs2 搜索。搜索是准备阶段完成的，在线的是 CIF 读取、过滤和特征构建。

如果你关心“这里说的 template features 到底怎么从 RNA 结构坐标算出来”，见后文 [5.1 RNA Template 开启的完整链路](#51-rna-template-开启的完整链路) 下面新增的“RNA template 特征如何计算”小节；当前 online / manual 两条链最终都会走同一套 `build_minimal_template_arrays()` 几何构建逻辑。

这一步有几个容易混淆的点，必须明确：

1. `01_extract_rna_catalog.py` 当前真实参数名是 `--pdb_rna_dir`，不是旧文档里写的 `--cif_dir`。
2. `03_mmseqs2_search.py` 当前真实输入不是 `--query_fasta`，而是 `--training_sequences /path/to/rna_sequence_to_pdb_chains.json`。
3. `search_results.json` 的键是 `query_id`，不是 sequence；但 online featurizer 加载后会自动聚合成 `sequence -> hits`。
4. 因为很多训练 PDB 共享同一条 RNA sequence，`search_results.json` 的 `query_id` 数通常会明显大于 featurizer 加载后的唯一 sequence 数，这是正常现象。
5. Online 模式真正需要的文件是 `search_results.json`、`cif_database_dir`、`rna3db_metadata_path`；`rna_template_index.json` 即使为空也不影响 online 模式。

#### Stanford RNA 3D folding / Protenix finetune 的已验证命令

下面这组命令是当前工作区里已经实际跑通的版本。

```bash
# 1) 全量 RNA3D catalog
python /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_pro/Protenix/rna_template/scripts/01_extract_rna_catalog.py \
  --pdb_rna_dir /inspire/ssd/project/sais-bio/public/ash_proj/data/RNA3D/rna3db-mmcifs \
  --output /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_pro/Protenix/rna_database/rna_catalog.json \
  --max_structures 0 \
  --min_length 10 \
  --max_length 2000 \
  --num_workers 8

# 2) 5574 个训练 PDB 的 query 映射
python /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_pro/Protenix/rna_template/scripts/generate_training_sequences_from_labels.py \
  --labels_csv /inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/train_labels.csv \
  --pdb_list /inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/protenix_prepared/rna_train_pdb_list_filtered.txt \
  --output_json /inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/protenix_prepared/rna_msa/rna_sequence_to_pdb_chains_train5574.json \
  --min_length 1

# 3) MMseqs2 搜索
python /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_pro/Protenix/rna_template/scripts/03_mmseqs2_search.py \
  --catalog /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_pro/Protenix/rna_database/rna_catalog.json \
  --template_dir /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_pro/Protenix/rna_database/templates \
  --training_sequences /inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/protenix_prepared/rna_msa/rna_sequence_to_pdb_chains_train5574.json \
  --output_index /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_pro/Protenix/rna_database/rna_template_index.json \
  --output_search /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_pro/Protenix/rna_database/search_results.json \
  --strategy mmseqs2 \
  --min_identity 0.3 \
  --max_templates 4 \
  --sensitivity 7.5 \
  --evalue 1e-3 \
  --num_threads 8 \
  --mmseqs_work_dir /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_pro/Protenix/rna_database/mmseqs2_work_5574
```

#### 当前这次全量重建的结果（2026-03-16）

- `rna_catalog.json`: `13128` 个结构，`5329` 个唯一 base PDB
- 训练 query JSON: `5574` 个 PDB，对应 `2629` 个唯一 sequence
- 其中 `<5nt` 的 query 只有 `3` 个：`3S4G`、`4E6B`、`6RLP`
- `03_mmseqs2_search.py` 实际搜索了 `5571` 个有效 query
- 清洗后的 `search_results.json`: `3838` 个命中 query，`14550` 条原始 hit
- `RNATemplateFeaturizer` online 加载后会聚合为 `1465` 个唯一 sequence、`5781` 条去重 hit

换句话说：

- `search_results.json` 是按 `query_id` 存
- online featurizer 是按 `query_sequence` 读
- 所以 “3838 个 query_id 命中” 和 “1465 个唯一 sequence 命中” 同时成立，并不冲突

#### 什么时候 **不需要** 重跑 RNA Template

这个判断很容易被误解，单独说明：

如果你这次做的事情只是：

1. 修 `components.cif` / `PROTENIX_ROOT_DIR`
2. 重刷 `rna_bioassembly/*.pkl.gz`
3. 重写 `rna_bioassembly_indices.csv`
4. 重抽 `sequences.json`
5. 增量刷新 AIDO embeddings

那么 **RNA template 通常不需要自动跟着重跑**，因为它不直接依赖这些产物。
当前 online RNA template 实际依赖的是：

- `search_results.json`
- `cif_database_dir`
- `rna3db_metadata_path`
- 以及当时用于构 query 的训练 PDB 列表和训练序列来源

只有下面两种情况，才建议重新跑 RNA template：

1. `rna_train_pdb_list_filtered.txt` 的成员发生变化，不再是原来那套 train split
2. 你希望 validation 也获得 template 覆盖；当前这份 `search_results.json` 是按 train query 构建的

换句话说：

- 修 parser / bioassembly / AIDO：**先不用重跑 RNA template**
- 训练 split 变了，或要补 val template：**再重跑 RNA template**

#### 模式二：Offline 模式（预计算NPZ，兼容保留）

```
Step 1: 01_extract_rna_catalog.py
Step 2: 02_build_rna_templates.py    → templates/*.npz
Step 3: 03_search_and_index.py       → rna_template_index.json

配置:
--rna_template.template_database_dir /path/to/templates/
--rna_template.template_index_path rna_template_index.json
```

### 2.5 Offline RNA MSA 准备（已有 MSA 文件版）

这一节只讨论一个场景：

- 你已经有一批离线 RNA MSA 文件
- 不需要我再给你 RNA MSA 搜索命令
- 你只想把这些现成文件整理成 **Protenix finetune 可直接读取** 的输入格式

以当前工作区的例子：

- 原始 MSA 文件目录：[`part2/MSA`](/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/MSA)
- 文件名形式：`<PDB>.MSA.fasta`
- 例如：`157D.MSA.fasta`、`1A3M.MSA.fasta`、`4E6B.MSA.fasta`

#### Protenix 当前 RNA MSA loader 的真实要求

代码路径：[`msa_featurizer.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_pro/Protenix/protenix/data/msa/msa_featurizer.py)

当前实现里，RNA MSA 不是按 PDB 直接查文件，而是按 **sequence** 查：

1. 先用 `msa.rna_seq_or_filename_to_msadir_jsons` 里的 JSON 做 `sequence -> [eid, ...]` 查找。
2. 取这个列表的第一个元素作为 `eid`。
3. 再去 `msa.rna_msadir_raw_paths/<eid>/<eid>_all.a3m` 读取 MSA。
4. RNA 当前只读取 **unpaired MSA**，不需要 `pairing.a3m`。

也就是说，你要给训练准备的不是“随便一堆 `.fasta` 文件”，而是下面这两个东西：

```text
rna_msa/
├── rna_sequence_to_pdb_chains.json
└── msas/
    ├── 157D/
    │   └── 157D_all.a3m
    ├── 1A3M/
    │   └── 1A3M_all.a3m
    └── ...
```

其中：

- `rna_sequence_to_pdb_chains.json` 的格式是 `{sequence: [eid1, eid2, ...]}`
- 训练时真正使用的是每个 sequence 对应列表里的第一个 `eid`
- 所以对 RNA MSA 来说，这个 JSON 更准确地应该理解成 `sequence -> canonical_msa_id`

#### 用 `part2/MSA` 目录接入 finetune 的整理方式

`part2/MSA` 当前实际情况：

- 总文件数：`5744`
- 与 `5574` 个训练 PDB 的文件名覆盖：`5574/5574`
- 也就是：**每个训练 PDB 都有一个同名 `.MSA.fasta` 文件**

所以如果你只是问“这些现成文件如何接到 Protenix”，整理步骤是：

##### Step 1: 准备 sequence -> eid 映射 JSON

如果你的训练集就是这次的 `5574` 个 PDB，那么映射来源应当是训练实际使用的 RNA sequence，而不是 MSA 文件名本身。

可直接复用或生成这类 JSON：

- [`rna_sequence_to_pdb_chains_train5574.json`](/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/protenix_prepared/rna_msa/rna_sequence_to_pdb_chains_train5574.json)

格式示意：

```json
{
  "GCGGAUUUAGCUCAGUUGGGAGAGCGCCAGACUGAAGAUCUGGAGGUCCUGUGUUCGAUCCACAGAAUUCGCACCA": ["4TNA"],
  "GGCCGAUGGUAGUGUGGGGUCUCCCCAUGCGAGAGUAGGCC": ["1A51"],
  "...": ["157D"]
}
```

注意：

1. 对 RNA MSA loader 来说，列表里的第一个元素最重要。
2. 如果多个 PDB 共享同一条 sequence，只需要保留一个能找到 MSA 的 `eid` 在列表第一个位置。
3. 你也可以保留更长列表，但训练时通常只会用到第一个 `eid`。

##### Step 2: 把 `part2/MSA/*.MSA.fasta` 整理成 Protenix 期望的目录结构

Protenix 期望：

- 根目录：`rna_msa/msas`
- 每个 `eid` 一个子目录
- 子目录内文件名固定为 `<eid>_all.a3m`

所以用 `part2/MSA` 做输入时，本质上就是“重命名/软链接”：

```bash
TARGET_ROOT=/your/path/to/rna_msa/msas
SRC_ROOT=/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/MSA

mkdir -p "${TARGET_ROOT}"
for f in "${SRC_ROOT}"/*.MSA.fasta; do
  eid=$(basename "${f}" .MSA.fasta)
  mkdir -p "${TARGET_ROOT}/${eid}"
  ln -sf "${f}" "${TARGET_ROOT}/${eid}/${eid}_all.a3m"
done
```

这里虽然源文件是 `.fasta`，但当前 `RawMsa.from_a3m()` 实际调用的是通用 FASTA 解析逻辑，所以内容可以保持不变；关键是路径和文件名要符合 loader 预期。

##### Step 3: 在训练配置里指向这两个离线产物

训练侧需要：

```python
msa.enable_rna_msa = True
msa.rna_seq_or_filename_to_msadir_jsons = ["/your/path/to/rna_sequence_to_pdb_chains.json"]
msa.rna_msadir_raw_paths = ["/your/path/to/rna_msa/msas"]
msa.rna_indexing_methods = ["sequence"]
```

这里最重要的是：

- `rna_seq_or_filename_to_msadir_jsons` 提供 `sequence -> eid`
- `rna_msadir_raw_paths` 提供 `eid -> eid_all.a3m`
- `rna_indexing_methods` 必须是 `"sequence"`

##### Step 4: 在真正训练前做一致性检查

这一步非常重要，因为当前 RNA MSA loader 对 query 的要求不是“序列完全相同”，而是至少 **第一条序列长度要和训练 query 长度一致**；如果长度不一致，内部会退化成 query-only。

就这次的 `part2/MSA` 例子来说：

- 如果用当前 `5574` 训练集的代表序列定义
- 那么 `5574` 个文件里，只有 `2515` 个的首条 query 长度和训练 query 一致
- 另外 `3059` 个长度不一致

这意味着：

- `part2/MSA` 目录可以作为离线 RNA MSA 输入来源
- 但 **不能假设所有文件都能无损直接用**
- 那 `3059` 个长度不一致的样本，在当前实现下很可能退化成 query-only MSA

因此，真正稳妥的使用原则是：

1. `mapping JSON` 的 key 必须是训练时真实会出现的 RNA sequence。
2. `eid_all.a3m` 的第一条 query 长度最好与该 sequence 一致。
3. 如果长度不一致，最好先做外部预处理或重新生成这部分 MSA；否则训练虽然能跑，但那部分样本的 RNA MSA 信息会显著变弱。

#### 这一节的结论

如果你已经有现成的 RNA MSA 文件，Protenix finetune 侧真正需要的不是“再跑一次搜索”，而是：

1. 一个 `sequence -> eid` 的 JSON 映射
2. 一个符合 `msas/<eid>/<eid>_all.a3m` 约定的目录树
3. 确认 `eid_all.a3m` 第一条 query 与训练 sequence 至少长度一致

只要这三件事成立，就不需要改代码，finetune 就能直接吃你的离线 RNA MSA。

### 2.6 Pretrain 操作完整检查清单

以下所有步骤完成后，才算做完所有 pretrain 操作：

```
✅ 检查清单:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ 1. mmCIF 文件已下载/准备好
□ 2. `PROTENIX_ROOT_DIR` 已显式设置
     → 推荐: `/inspire/ssd/project/sais-bio/public/ash_proj/data`
     → `${PROTENIX_ROOT_DIR}/common/components.cif` 存在
     → `${PROTENIX_ROOT_DIR}/common/components.cif.rdkit_mol.pkl` 存在
□ 3. prepare_training_data.py / run_bioassembly.py 完成
     → indices.csv 存在
     → bioassembly/ 目录下有 pkl.gz 文件
□ 4. ESM embeddings 已生成 (蛋白链)
     → esm_embeddings/ 目录下有 .pt 文件
     → manifest.csv 存在
□ 5. RNA LLM embeddings 已生成 (RNA链)
     → rnalm_embeddings/ 目录下有 .pt 文件
     → manifest.csv 存在
□ 6. DNA LLM embeddings 已生成 (DNA链, 如果需要)
     → dna_embeddings/ 目录下有 .pt 文件
     → manifest.csv 存在
□ 7. MSA 输入已准备
     → 蛋白 MSA: msa/ 下有 .a3m 文件
     → RNA MSA: rna_msa/msas/<eid>/<eid>_all.a3m 可解析
     → RNA sequence → eid 映射JSON已生成
□ 8. RNA Template 准备 (如果需要)
     → rna_catalog.json 已生成
     → 训练 query JSON 已生成 (`rna_sequence_to_pdb_chains.json`)
     → search_results.json 已生成 (online模式)
     → search_results 与 cif_database_dir 已对齐
     → CIF 数据库目录可用
     → filter.json (RNA3DB metadata) 可用
     → 如果这次只是重刷 parser/bioassembly，先确认 train split 是否变化；未变化时可不重跑 RNA template
□ 9. 训练配置已检查
     → 所有路径指向正确
     → 模型参数设置正确
     → checkpoint 路径设置正确
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 3. 训练阶段 — LLM Part 工作流

### 3.1 RNALM 开启时的工作流

当 `--rnalm.enable true` 时，进入以下工作流：

```
┌─────────────────────────────────────────────────────────────┐
│                    rnalm.enable = true                       │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 数据层: RiNALMoFeaturizer                            │    │
│  │                                                     │    │
│  │ 1. 识别每个token的entity类型 (RNA/DNA/Protein)       │    │
│  │ 2. 对RNA token: 加载RNA embedding                   │    │
│  │ 3. 对DNA token: 加载DNA embedding                   │    │
│  │ 4. 返回 embedding tensors                           │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                   │
│                         ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 模型层: 注入点由 injection_mode 决定                  │    │
│  │                                                     │    │
│  │ injection_mode = "input":                           │    │
│  │   → InputFeatureEmbedder.forward()                  │    │
│  │   → s_inputs += linear_rnalm(emb) 或               │    │
│  │     s_inputs += linear_rna_llm(rna) +              │    │
│  │                  linear_dna_llm(dna)                │    │
│  │                                                     │    │
│  │ injection_mode = "diffusion":                       │    │
│  │   → Protenix._get_s_rnalm()                        │    │
│  │   → s_rnalm = projection(emb)                      │    │
│  │   → 可选gate: g * s_rnalm                          │    │
│  │   → DiffusionConditioning: s_trunk + s_rnalm       │    │
│  │                                                     │    │
│  │ injection_mode = "both":                            │    │
│  │   → 同时执行 input + diffusion 注入                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  关键参数:                                                   │
│  - model_name: "rinalmo" | "aido"                          │
│  - injection_mode: "input" | "diffusion" | "both"          │
│  - gate_mode: "none" | "scalar" | "token" | "dual"         │
│  - separate_dna_projection: true | false                   │
└─────────────────────────────────────────────────────────────┘
```

**代码路径 (`protenix/model/protenix.py`):**

```python
# 模型初始化时 (protenix.py)
if self.rnalm_enable:
    if injection_mode in ("diffusion", "both"):
        if separate_dna:
            self.rna_projection = LinearNoBias(1280, c_s)  # RNA投影
            self.dna_projection = LinearNoBias(1024, c_s)  # DNA投影
        else:
            self.rnalm_projection = LinearNoBias(1280, c_s)  # 统一投影

        # 门控
        if gate_mode == "scalar":
            self.rnalm_alpha_logit = nn.Parameter(tensor(gate_init_logit))
        elif gate_mode == "token":
            self.rnalm_gate_mlp = nn.Sequential(Linear, ReLU, Linear)

# Forward 时:
def _get_s_rnalm(self, input_feature_dict, N_token, s_trunk=None):
    if self.rnalm_separate_dna:
        lm_delta = zeros([N_token, c_s])
        lm_delta += self.rna_projection(rna_llm_embedding)  # RNA
        lm_delta += self.dna_projection(dna_llm_embedding)  # DNA
    else:
        lm_delta = self.rnalm_projection(rnalm_token_embedding)

    # 应用门控
    if gate_mode == "scalar":
        lm_delta = sigmoid(alpha_logit) * lm_delta
    elif gate_mode == "token":
        lm_delta = sigmoid(gate_mlp(s_trunk)) * lm_delta
    elif gate_mode == "dual":
        lm_delta = sigmoid(alpha) * sigmoid(gate_mlp(s_trunk)) * lm_delta

    return lm_delta  # → 传入 DiffusionConditioning
```

### 3.2 RNA LLM / DNA LLM 分别开启的工作流

```
┌──────────────────────────────────────────────────────────┐
│ 场景矩阵:                                                │
│                                                          │
│ use_rna_embed=T, use_dna_embed=T (默认):                  │
│   → RNA token用RNA模型, DNA token用DNA模型                │
│   → 两个embedding都加载                                  │
│                                                          │
│ use_rna_embed=T, use_dna_embed=F:                        │
│   → 只加载RNA embedding                                  │
│   → DNA token的embedding为全零                           │
│   → 如果separate_dna=true, linear_dna_llm不创建          │
│                                                          │
│ use_rna_embed=F, use_dna_embed=T:                        │
│   → 只加载DNA embedding                                  │
│   → RNA token的embedding为全零                           │
│   → 如果separate_dna=true, linear_rna_llm不创建          │
│                                                          │
│ use_rna_embed=F, use_dna_embed=F:                        │
│   → rnalm整体禁用 (即使enable=true也不注入)              │
│   → 不创建任何projection层                               │
└──────────────────────────────────────────────────────────┘
```

**RiNALMo vs AIDO 模型选择：**

```
┌───────────────┬────────────────────┬────────────────────┐
│ 配置          │ model_name=rinalmo │ model_name=aido    │
├───────────────┼────────────────────┼────────────────────┤
│ RNA dim       │ 1280               │ 2048               │
│ DNA dim       │ — (不支持)          │ 1024               │
│ use_rna_embed │ true               │ true               │
│ use_dna_embed │ false              │ true               │
│ separate_dna  │ false              │ true (推荐)         │
│ 投影层        │ linear_rnalm       │ linear_rna_llm +   │
│               │ (1280→449/384)     │ linear_dna_llm     │
│               │                    │ (2048→449, 1024→449)│
└───────────────┴────────────────────┴────────────────────┘
```

### 3.3 Separate Projection 与 Init 选项

```
┌──────────────────────────────────────────────────────────┐
│ separate_dna_projection = false (默认):                   │
│                                                          │
│ ┌──────────────┐     ┌──────────┐     ┌──────────┐      │
│ │ RNA embedding │──┐  │ linear   │     │ s_inputs │      │
│ │ [N, 1280]     │  ├──│ _rnalm   │──+──│ 或       │      │
│ │ DNA零填充到1280│──┘  │ (1280→D) │     │ s_rnalm  │      │
│ └──────────────┘     └──────────┘     └──────────┘      │
│                                                          │
│ 特点:                                                    │
│ - 单一投影矩阵处理 RNA 和 DNA                             │
│ - DNA embedding 零填充到 RNA 维度                         │
│ - 参数量少，但 RNA/DNA 共享投影空间                        │
│                                                          │
│ ════════════════════════════════════════════════════════  │
│                                                          │
│ separate_dna_projection = true:                          │
│                                                          │
│ ┌──────────────┐     ┌────────────┐                      │
│ │ RNA embedding │────│ linear_rna │──┐                    │
│ │ [N, 1280]     │     │ _llm       │  │  ┌──────────┐    │
│ └──────────────┘     │ (1280→D)   │  ├──│ s_inputs │    │
│                      └────────────┘  │  │ 或       │    │
│ ┌──────────────┐     ┌────────────┐  │  │ s_rnalm  │    │
│ │ DNA embedding │────│ linear_dna │──┘  └──────────┘    │
│ │ [N, 1024]     │     │ _llm       │                     │
│ └──────────────┘     │ (1024→D)   │                     │
│                      └────────────┘                     │
│                                                          │
│ 特点:                                                    │
│ - RNA 和 DNA 各有独立投影矩阵                             │
│ - 不需要零填充                                            │
│ - 可以独立学习 RNA/DNA 的投影空间                          │
│ - 参数量更多，但更灵活                                    │
└──────────────────────────────────────────────────────────┘
```

**投影层初始化：**
- 所有 LLM 投影层（`linear_rnalm`, `linear_rna_llm`, `linear_dna_llm`）都是 **零初始化**
- 这保证了在训练初期，LLM embedding 的贡献为零，不会破坏预训练模型
- 随训练逐渐学习有效的投影

---

## 4. 训练阶段 — MSA Part 工作流

### 4.1 RNA MSA 开启时的工作流

```
┌──────────────────────────────────────────────────────────┐
│ configs_data.py: msa.enable_rna_msa = True               │
│                                                          │
│ ┌────────────────────────────────────────────────────┐    │
│ │ MSASourceManager                                    │    │
│ │                                                    │    │
│ │ 1. 加载映射: rna_sequence_to_pdb_chains.json       │    │
│ │    RNA序列 → [eid1, eid2, ...]                     │    │
│ │                                                    │    │
│ │ 2. 查找MSA文件:                                    │    │
│ │    rna_msadir_raw_paths/                           │    │
│ │    ├── eid1/eid1_all.a3m                           │    │
│ │    ├── eid2/eid2_all.a3m                           │    │
│ │    └── ...                                         │    │
│ │                                                    │    │
│ │ 3. indexing_method = "sequence":                   │    │
│ │    根据RNA序列查找对应MSA                           │    │
│ │    (不是按PDB ID直接查；实际取映射列表第一个eid)   │    │
│ └────────────────────────────────────────────────────┘    │
│                    │                                      │
│                    ▼                                      │
│ ┌────────────────────────────────────────────────────┐    │
│ │ MSAFeaturizer                                       │    │
│ │                                                    │    │
│ │ 1. 解析 .a3m 文件                                  │    │
│ │ 2. 对齐到 query 序列                               │    │
│ │ 3. 裁剪到 sample_cutoff (max 16384 sequences)     │    │
│ │ 4. 生成 MSA features:                              │    │
│ │    - msa_profile [N_token, 32]                     │    │
│ │    - deletion_mean [N_token, 1]                    │    │
│ │    - has_deletion [N_token, 1]                     │    │
│ └────────────────────────────────────────────────────┘    │
│                    │                                      │
│                    ▼                                      │
│ ┌────────────────────────────────────────────────────┐    │
│ │ MSAModule (PairFormer)                              │    │
│ │                                                    │    │
│ │ MSA features 作为 profile 和 deletion_mean          │    │
│ │ 直接进入 InputFeatureEmbedder 的 per-token concat  │    │
│ │                                                    │    │
│ │ s_inputs = [atom_emb, restype, profile,            │    │
│ │            deletion_mean]  # 384+32+32+1 = 449     │    │
│ └────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

**启用 MSA 进入 Model 的配置：**

```python
# configs_data.py 中:
"msa": {
    "enable_rna_msa": True,
    "rna_seq_or_filename_to_msadir_jsons": [
        "rna_msa/rna_sequence_to_pdb_chains.json"
    ],
    "rna_msadir_raw_paths": [
        "rna_msa/msas"
    ],
    "rna_indexing_methods": ["sequence"],
    "min_size": {"train": 1, "test": 1},
    "max_size": {"train": 16384, "test": 16384},
    "sample_cutoff": {"train": 16384, "test": 16384},
}
```

---

## 5. 训练阶段 — Template Part 工作流

### 5.1 RNA Template 开启的完整链路

```
┌──────────────────────────────────────────────────────────────────┐
│                rna_template.enable = true                         │
│                                                                  │
│ ┌──────────────────────────────────────────────────────────┐     │
│ │ 数据层: RNATemplateFeaturizer                             │     │
│ │                                                          │     │
│ │ 判断模式:                                                │     │
│ │ if search_results_path 和 cif_database_dir 都配置:       │     │
│ │     → Online模式 (基于预计算 search_results 在线构建)     │     │
│ │ elif template_index_path 和 template_database_dir 都配置:│     │
│ │     → Offline模式 (预计算NPZ)                             │     │
│ │ else:                                                    │     │
│ │     → 配置错误；factory 会报错，而不是静默回空            │     │
│ │                                                          │     │
│ │ ┌── Online模式流程 ──────────────────────────────┐       │     │
│ │ │ 1. 查找query序列 → search_results.json中命中    │       │     │
│ │ │    (加载时先把 query_id 聚合成 sequence→hits)   │       │     │
│ │ │ 2. 对每个hit:                                   │       │     │
│ │ │    a. self-hit检查 (排除同PDB)                   │       │     │
│ │ │    b. 时间过滤 (query_date - 60天)              │       │     │
│ │ │    c. 加载CIF文件                               │       │     │
│ │ │    d. 提取残基坐标                              │       │     │
│ │ │    e. 构建minimal template arrays               │       │     │
│ │ │    f. 计算distogram, mask, unit_vector等        │       │     │
│ │ │ 3. 收集最多 max_rna_templates 个模版            │       │     │
│ │ │ 4. 堆叠为 [T, N, N, *] 格式                    │       │     │
│ │ └─────────────────────────────────────────────────┘       │     │
│ └──────────────────────────────────────────────────────────┘     │
│                          │                                       │
│                          ▼                                       │
│ ┌──────────────────────────────────────────────────────────┐     │
│ │ 模型层: TemplateEmbedder (pairformer.py)                  │     │
│ │                                                          │     │
│ │ 输入特征:                                                │     │
│ │ [distogram:39, frame_mask:1, unit_vec:3,                 │     │
│ │  pseudo_beta:1, restype_i:32, restype_j:32] = 108 dim   │     │
│ │                                                          │     │
│ │ RNA专用投影:                                             │     │
│ │ linear_no_bias_a_rna (108 → c)                          │     │
│ │                                                          │     │
│ │ 共享的Pairformer处理:                                    │     │
│ │ pairformer_stack (2 blocks) + layernorm_v                │     │
│ │                                                          │     │
│ │ 融合语义:                                                │     │
│ │ RNA template 经过独立 projector，走共享 template trunk   │     │
│ │ 然后作为 template contribution 加回 pair representation │     │
│ │ protein-init 时再乘可学习 α                              │     │
│ │                                                          │     │
│ │ rna_template_block_mask 限制RNA模版只影响RNA-RNA pairs   │     │
│ └──────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

补充说明：上面的流程图描述的是 **什么时候构建 RNA template**，但没有展开 **这些特征本身是怎么从 CIF/mmCIF 坐标算出来的**。当前代码里，online search hit 和 manual structure hint 最终都会调用 `build_minimal_template_arrays()`；`protenix/data/rna_template/rna_template_common_online.py` 只是把实现桥接到 `rna_template/compute/rna_template_common.py`。

#### RNA template 特征如何计算（CIF/mmCIF -> Protenix 张量）

当前最核心的计算函数链是：

```text
load_structure_residues()
  -> normalize_query_sequence()
  -> align_query_to_template()
  -> build_minimal_template_arrays()
     -> compute_anchor()
     -> compute_frame()
     -> compute_distogram()
     -> compute_unit_vectors()
```

**Step 1: 读取并清洗 template 链**

- `load_structure_residues()` 用 BioPython 解析 hit 对应的 mmCIF/PDB，只保留“看起来像 nucleotide”的 residue。
- 对 disordered atom，会优先取 occupancy / bfactor 更合理的那一支；氢原子会被丢掉。
- 常见修饰碱基会先退化到标准 RNA 碱基再继续算特征，例如 `PSU -> U`、`5MC -> C`、`1MA -> A`、`2MG -> G`；实在无法判断时记成 `N`。
- 最终每个残基会变成一个 `ResidueRecord`，里面保留 `chain_id / resseq / icode / base_letter / atoms`。

**Step 2: 把 query 序列和 template 序列对齐**

- `normalize_query_sequence()` 会先把 query 标准化成大写 RNA 字母表：`T -> U`，非 `A/G/C/U/N/-` 的字符统一写成 `N`。
- `residues_to_sequence()` 从 template 链提取碱基序列后，`align_query_to_template()` 用全局对齐生成 `query_idx -> template_idx` 的映射。
- 当前打分与实现一致：
  - `match = +2.0`
  - `mismatch = -1.0`
  - `gap_open = -5.0`
  - `gap_extend = -0.5`
- 没有对齐上的 query 位置不会强行补坐标，而是保留为 gap：`template_aatype` 写入 Protenix gap id `31`，对应 1D/2D mask 都会是 `0`。

**Step 3: 先算每个 query 位置的一维模板信息**

- `template_aatype [N]`
  - 对齐成功的位置，用 Protenix RNA residue id 编码：
    - `A=21, G=22, C=23, U=24, N=25, gap=31`
- `anchor_pos [N,3]` + `anchor_mask [N]`
  - 默认 `anchor_mode` 是 `base_center_fallback`
  - 优先取碱基非骨架原子的质心作为 anchor，也就是 base center
  - 如果碱基原子不全，回退到 `C4'`
  - 再不行回退到 `C1'`
  - 仍然缺失时用零向量，且 `anchor_mask=0`
- `frame_origin [N,3]` + `frame_axes [N,3,3]` + `frame_mask [N]`
  - RNA 局部 frame 的原点是 `C4'`
  - 第一轴优先取 `P - C4'`，如果 `P` 缺失则回退 `O5'`，再回退 `C3'`
  - 第二个参考方向是 `C1' - C4'`
  - 再通过 Gram-Schmidt 正交化得到三个局部坐标轴
  - 如果任一步向量退化，整个 `frame_mask=0`

可以把这个 frame 理解成 RNA 版的“蛋白 N-CA-C backbone frame”：

```text
origin = C4'
e1 = normalize(P_or_O5'_or_C3' - C4')
e2 = normalize((C1' - C4') 去掉在 e1 上的分量)
e3 = normalize(cross(e1, e2))
```

**Step 4: 再把 1D 几何转成 pair 特征**

- `template_distogram [N,N,39]`
  - 先计算所有残基对 anchor 的欧氏距离
  - 再按和 Protenix 蛋白 template 一致的 39 个桶做 one-hot：
    - `lower_breaks = linspace(3.25, 50.75, 39)`
    - 最后一桶上界视为 `+inf`
  - 实现里直接比较平方距离，避免重复开方
  - 只有 `anchor_mask[i] * anchor_mask[j] = 1` 的 pair 才保留
- `template_pseudo_beta_mask [N,N]`
  - 其实就是 pairwise anchor 有效性：
  - `anchor_mask[:,None] * anchor_mask[None,:]`
- `template_unit_vector [N,N,3]`
  - 对每个中心残基 `i`，把 `anchor_j - frame_origin_i` 投影到 `i` 的局部 frame
  - 然后单位化，得到“从 i 的局部视角看 j 在哪个方向”
- `template_backbone_frame_mask [N,N]`
  - 只有当 `i` 有合法 frame、`j` 有合法 anchor，且投影向量范数大于阈值时，这个 pair 才有效

它们在实现里的关系可以简写成：

```text
distogram[i,j,:] <- bucketize(||anchor_i - anchor_j||)
pseudo_beta_mask[i,j] <- anchor_mask[i] * anchor_mask[j]
unit_vector[i,j,:] <- normalize(frame_axes[i] @ (anchor_j - frame_origin[i]))
backbone_frame_mask[i,j] <- frame_mask[i] * anchor_mask[j] * valid_norm(i,j)
```

**Step 5: 多模板堆叠，并映射回整个 sample token 空间**

- 单个 hit 先得到链内局部张量，shape 以该 RNA chain 的 query 长度 `N_chain` 为准。
- `RNATemplateFeaturizer` 最多保留 `max_rna_templates` 个成功模板，堆叠成 `[T, N_chain, ...]`。
- 随后再根据样本里的 `token_entity_id` / `token_res_id`，把 chain 局部特征 scatter 回全样本 token 级别：
  - `rna_template_aatype [T, N_token]`
  - `rna_template_distogram [T, N_token, N_token, 39]`
  - `rna_template_pseudo_beta_mask [T, N_token, N_token]`
  - `rna_template_unit_vector [T, N_token, N_token, 3]`
  - `rna_template_backbone_frame_mask [T, N_token, N_token]`
- 最后再构造：
  - `rna_template_block_mask [N_token, N_token] = rna_token_mask[:,None] * rna_token_mask[None,:]`
  - 它只允许 RNA-RNA pair 接收 RNA template 的贡献；蛋白 token、DNA token、跨模态 pair 都会被挡掉。

**Step 6: 喂给 TemplateEmbedder 的真实拼接格式**

进入 `pairformer.py` 后，每个 RNA template 会按和蛋白 template 相同的 108 维 pair 特征格式拼起来：

```text
39  distogram
+ 1 backbone_frame_mask
+ 3 unit_vector
+ 1 pseudo_beta_mask
+32 restype_i (one-hot)
+32 restype_j (one-hot)
=108
```

区别只在于：

- RNA 走的是独立的 `linear_no_bias_a_rna`
- 但后面的 `pairformer_stack` 和 `layernorm_v` 与蛋白 template 共享
- 如果 `projector_init=protein`，还会再乘一个可学习的 `rna_template_alpha`

所以，**RNA template feature 的“计算”主要发生在数据侧**（读结构、对齐、算几何、组 mask），**模型侧做的是投影和融合**。

### 5.2 人工指定 Template 链路

```
┌──────────────────────────────────────────────────────────────┐
│           manual_template_hints_path 被设置                    │
│                                                              │
│ 当前 training 接口:                                          │
│ {                                                            │
│   "1abc": {                                                  │
│     "*": {                                                   │
│       "mode": "hybrid",                                      │
│       "manual_templates": [                                  │
│         {"type": "structure", "path": "/path/to/tpl.cif",    │
│          "chain_id": "A"}                                    │
│       ]                                                      │
│     }                                                        │
│   }                                                          │
│ }                                                            │
│                                                              │
│ 当前 inference 接口不是这个文件，而是 inline JSON            │
│ `rnaSequence.templateHints`。                                 │
│                                                              │
│ 流程:                                                        │
│ ┌────────────────────────────────────────────────────┐       │
│ │ 1. 解析 per-PDB / per-entity hints                 │       │
│ │ 2. 按 mode 决定 manual_only / prefer_manual /      │       │
│ │    hybrid / default_only                           │       │
│ │ 3. 手工模板通过 structure/npz builder 构建         │       │
│ │ 4. 如果 mode 允许，再和默认 search templates 合并  │       │
│ │ 5. 注意: manual template 本身绕过默认 search 的    │       │
│ │    temporal / self-hit 过滤；search fallback 仍过滤│       │
│ └────────────────────────────────────────────────────┘       │
│                                                              │
│ 配置:                                                        │
│ --rna_template.enable true                                   │
│ --rna_template.manual_template_hints_path hints.json          │
│ (可以同时设置 search_results_path 来补充自动搜索模版)        │
└──────────────────────────────────────────────────────────────┘
```

### 5.3 默认 Template 链路（Online 自动搜索）

```
┌──────────────────────────────────────────────────────────────┐
│  默认链路 = Online自动搜索 (无manual_template_hints)           │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │ 1. RNATemplateFeaturizer.__call__()                  │     │
│  │    └─ 检测到 search_results_path + cif_database_dir  │     │
│  │       已配置 → Online模式                             │     │
│  │                                                     │     │
│  │ 2. 默认 search 流                                   │     │
│  │    ├─ _find_hits_for_sequence()                     │     │
│  │    ├─ _filter_hits_online()                         │     │
│  │    │   ├─ self-hit? (同base PDB → skip)            │     │
│  │    │   ├─ 时间过滤? (太新 → skip)                  │     │
│  │    │   └─ RNA3DB metadata / PDB API 查日期         │     │
│  │    ├─ _build_single_template_online()              │     │
│  │    │   ├─ 定位CIF文件                               │     │
│  │    │   ├─ 解析残基/坐标                             │     │
│  │    │   └─ 构建单 template feature                   │     │
│  │    └─ 收集成功的模版 (最多 max_rna_templates)       │     │
│  │                                                     │     │
│  │ 3. 如果没有任何模版命中:                             │     │
│  │    → 返回空特征 (全零tensor)                        │     │
│  │    → TemplateEmbedder 中 mask 全为0, 无效果          │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
│  默认 search-based template 的防泄漏三层保护:                │
│  Layer 1: Self-hit exclusion                                 │
│    extract_base_pdb_id(query) != extract_base_pdb_id(hit)    │
│  Layer 2: Temporal filtering                                 │
│    hit_release_date < query_release_date - 60 days           │
│  Layer 3: Conservative unknown handling                      │
│    release date unknown → REJECT                             │
│                                                              │
│  注: inference 模式默认不做 query-side temporal/self-hit     │
│  审核；manual template 也不自动套用这三层规则。               │
└──────────────────────────────────────────────────────────────┘
```

### 5.4 Template 执行流程详解

```
TemplateEmbedder.forward() 完整执行流程:

┌────────────────────────────────────────────────────────────┐
│ 输入: input_feature_dict, z (pair representation)           │
│                                                            │
│ Step 1: 蛋白模版处理                                       │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ for t in range(N_prot_templates):                     │   │
│ │   features = concat(distogram, mask, vec, beta,       │   │
│ │                     restype_i, restype_j)  [N,N,108]  │   │
│ │   a = linear_no_bias_a(features)  [N,N,c_z]          │   │
│ │   v = pairformer_stack(a)  [N,N,c_z]                  │   │
│ │   v = layernorm_v(v)                                  │   │
│ │   u += v  (累加)                                      │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                            │
│ Step 2: RNA模版处理 (如果 rna_template_enable)              │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ for t in range(N_rna_templates):                      │   │
│ │   # 与蛋白类似但使用独立投影                           │   │
│ │   features = same concat format [N,N,108]             │   │
│ │   # 应用 rna_template_block_mask (只保留RNA-RNA pairs)│   │
│ │   effective_mask = multichain_mask * rna_template_    │   │
│ │                    block_mask                         │   │
│ │   features = features * effective_mask                │   │
│ │   a = linear_no_bias_a_rna(features)  [N,N,c]        │   │
│ │   v = pairformer_stack(a)  (共享!)                    │   │
│ │   if projector_init == "zero":                        │   │
│ │       u += v                                          │   │
│ │   elif projector_init == "protein":                   │   │
│ │       u += rna_template_alpha * v  (α-gated)          │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                            │
│ Step 3: 平均和投影                                         │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ u = u / (N_prot_templates + N_rna_templates)          │   │
│ │ u = linear_no_bias_u(relu(u)) → [N,N,c_z]            │   │
│ │ 返回给上游 trunk，再加回 pair representation           │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                            │
│ 输出: z (更新后的 pair representation)                      │
└────────────────────────────────────────────────────────────┘
```

---

## 6. Projector 设置与效果

### RNA LLM Projector

```
┌────────────────────────────────────────────────────────────┐
│ injection_mode 与 projector 的关系:                          │
│                                                            │
│ ┌────────────────┬───────────────┬──────────────────────┐  │
│ │ injection_mode │ 创建的投影层   │ 投影目标维度          │  │
│ ├────────────────┼───────────────┼──────────────────────┤  │
│ │ "input"        │ linear_rnalm  │ c_s_inputs (449)     │  │
│ │                │ 或 linear_rna │                      │  │
│ │                │ + linear_dna  │                      │  │
│ ├────────────────┼───────────────┼──────────────────────┤  │
│ │ "diffusion"    │ rna_projection│ c_s (384)            │  │
│ │                │ 或 rnalm_     │                      │  │
│ │                │ projection    │                      │  │
│ ├────────────────┼───────────────┼──────────────────────┤  │
│ │ "both"         │ 两组都创建     │ 449 + 384            │  │
│ └────────────────┴───────────────┴──────────────────────┘  │
│                                                            │
│ 所有投影层: 零初始化 (初始贡献为0)                          │
│ 训练中逐渐学习有效投影                                     │
└────────────────────────────────────────────────────────────┘
```

### RNA Template Projector

```
┌────────────────────────────────────────────────────────────┐
│ projector_init 设置的效果:                                   │
│                                                            │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ projector_init = "zero":                             │    │
│ │                                                     │    │
│ │ linear_no_bias_a_rna = LinearNoBias(108, c_z)       │    │
│ │ 初始化: 全零权重                                     │    │
│ │ 无 alpha gate                                        │    │
│ │ 效果: 训练初期 RNA template 无贡献                   │    │
│ │       类似 LLM projector 的零初始化策略               │    │
│ │ 融合: u += rna_v  (直接加)                           │    │
│ │                                                     │    │
│ │ 适用: 从零开始训练RNA template projector             │    │
│ └─────────────────────────────────────────────────────┘    │
│                                                            │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ projector_init = "protein" (默认):                   │    │
│ │                                                     │    │
│ │ linear_no_bias_a_rna = copy(linear_no_bias_a)       │    │
│ │ 初始化: 从蛋白projector复制权重                      │    │
│ │ rna_template_alpha = Parameter(alpha_init=0.01)     │    │
│ │ 效果: 利用蛋白template的先验知识                     │    │
│ │       α=0.01 → 初始贡献很小 (1%)                     │    │
│ │       训练中 α 自动调节                              │    │
│ │ 融合: u += α · rna_v  (alpha-gated)                 │    │
│ │                                                     │    │
│ │ 适用: 利用蛋白模版的迁移学习 (推荐)                  │    │
│ └─────────────────────────────────────────────────────┘    │
│                                                            │
│ alpha_init 的影响:                                         │
│ ┌─────────────────────────────────────────────────────┐    │
│ │ alpha_init = 0.01 (默认)                             │    │
│ │   → 初始RNA template贡献为蛋白的1%                   │    │
│ │   → 安全启动, 不会破坏已训练的模型                   │    │
│ │                                                     │    │
│ │ alpha_init = 0.1                                    │    │
│ │   → 初始贡献10%, 更激进                              │    │
│ │   → 适合 RNA template 质量很高时                     │    │
│ │                                                     │    │
│ │ alpha_init = 1.0                                    │    │
│ │   → 初始等权重, 与蛋白template相同重要性             │    │
│ │   → 可能导致训练不稳定                               │    │
│ └─────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────┘
```

### Gate Mode 效果矩阵

```
┌────────────────────────────────────────────────────────────┐
│ gate_mode 设置与执行:                                       │
│                                                            │
│ ┌──────────┬────────────────────┬────────────────────────┐ │
│ │ gate_mode│ 公式                │ 效果                    │ │
│ ├──────────┼────────────────────┼────────────────────────┤ │
│ │ "none"   │ s = proj(emb)      │ 无门控, 直接注入        │ │
│ │          │                    │ 最简单, 参数最少        │ │
│ ├──────────┼────────────────────┼────────────────────────┤ │
│ │ "scalar" │ s = σ(α) · proj(e) │ 全局标量门              │ │
│ │          │ α初始=-3.0         │ σ(-3)≈0.047            │ │
│ │          │ (可学习)           │ 初始很小, 渐进学习      │ │
│ ├──────────┼────────────────────┼────────────────────────┤ │
│ │ "token"  │ s = σ(MLP(s_t)) ·  │ 逐token门控             │ │
│ │          │     proj(emb)      │ 可学习哪些位置需要LLM   │ │
│ │          │                    │ 参数量中等              │ │
│ ├──────────┼────────────────────┼────────────────────────┤ │
│ │ "dual"   │ s = σ(α)·σ(MLP) ·  │ 双重门控               │ │
│ │          │     proj(emb)      │ 全局+逐token            │ │
│ │          │                    │ 最灵活, 参数最多        │ │
│ └──────────┴────────────────────┴────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

---

## 7. 训练 Stage 链路

### 7.1 1-Stage 训练链路

```
┌────────────────────────────────────────────────────────────┐
│ two_stage.enable = false (默认) → 1-Stage 训练              │
│                                                            │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 初始化 (init_model):                                  │   │
│ │ 1. 创建 Protenix 模型                                │   │
│ │ 2. 加载 checkpoint (如有)                             │   │
│ │ 3. DDP 封装 (如多GPU)                                 │   │
│ │ 4. 创建优化器:                                       │   │
│ │    a. 检查 two_stage.adapter_lr 和 backbone_lr       │   │
│ │    b. 如果 adapter_lr > 0:                           │   │
│ │       → 分两组参数 (adapter, backbone)               │   │
│ │       → adapter 用 adapter_lr, backbone 用 backbone_lr│   │
│ │    c. 否则: 统一学习率                                │   │
│ │ 5. 启用 EMA                                          │   │
│ │ 6. 创建 lr_scheduler                                 │   │
│ └──────────────────────────────────────────────────────┘   │
│                         │                                   │
│                         ▼                                   │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ 训练循环:                                             │   │
│ │ while step < max_steps:                              │   │
│ │   batch = next(train_dl)                             │   │
│ │   loss = model(batch)                                │   │
│ │   loss.backward()                                    │   │
│ │   optimizer.step()                                   │   │
│ │   ema.update()                                       │   │
│ │   lr_scheduler.step()                                │   │
│ │   if step % eval_interval == 0:                      │   │
│ │       evaluate()                                     │   │
│ │   if step % checkpoint_interval == 0:                │   │
│ │       save_checkpoint()                              │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                            │
│ 特点:                                                      │
│ - 所有参数同时训练                                          │
│ - 可以通过 adapter_lr/backbone_lr 设置不同学习率            │
│ - 始终有 EMA                                               │
│ - 简单直接                                                  │
└────────────────────────────────────────────────────────────┘
```

### 7.2 2-Stage 训练链路

```
┌────────────────────────────────────────────────────────────┐
│ two_stage.enable = true → 2-Stage 训练                      │
│                                                            │
│ ═══════════════ Stage 1: Adapter Warmup ═══════════════    │
│                                                            │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ _setup_stage1():                                      │   │
│ │                                                      │   │
│ │ 1. 分类参数:                                         │   │
│ │    adapter_keywords = [                              │   │
│ │      "rnalm_projection", "rna_projection",           │   │
│ │      "dna_projection", "linear_rnalm",               │   │
│ │      "linear_rna_llm", "linear_dna_llm",             │   │
│ │      "rnalm_alpha_logit", "rnalm_gate_mlp",          │   │
│ │      "linear_no_bias_a_rna", "rna_template_alpha",   │   │
│ │      "rna_template_gate"                             │   │
│ │    ]                                                 │   │
│ │    adapter_params = 匹配 keyword 的参数              │   │
│ │    backbone_params = 其余所有参数                     │   │
│ │                                                      │   │
│ │ 2. 创建优化器:                                       │   │
│ │    param_groups = [                                   │   │
│ │      {"params": backbone, "lr": 0.0},   # 冻结!      │   │
│ │      {"params": adapter,  "lr": 5e-3},  # 活跃       │   │
│ │    ]                                                 │   │
│ │                                                      │   │
│ │ 3. 调度器:                                           │   │
│ │    CosineAnnealing(warmup=1, max_steps=400)          │   │
│ │                                                      │   │
│ │ 4. 无 EMA                                            │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                            │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Stage 1 训练循环 (steps 0 → 400):                     │   │
│ │                                                      │   │
│ │ ⚡ Backbone 冻结 (lr=0)                               │   │
│ │ ⚡ 只有 adapter 参数更新                               │   │
│ │ ⚡ 学习新注入的 LLM/Template 投影层                    │   │
│ │ ⚡ 不会破坏已训练的 backbone                           │   │
│ └──────────────────────────────────────────────────────┘   │
│                         │                                   │
│                    step >= 400                               │
│                         │                                   │
│                         ▼                                   │
│ ═══════════════ Stage 2: Joint Training ═══════════════    │
│                                                            │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ _transition_to_stage2():                              │   │
│ │                                                      │   │
│ │ 1. 解冻所有参数                                      │   │
│ │    for p in model.parameters():                      │   │
│ │        p.requires_grad = True                        │   │
│ │                                                      │   │
│ │ 2. 解析学习率:                                       │   │
│ │    stage2_adapter_lr:                                 │   │
│ │      -1 → 继承 stage1_adapter_lr (5e-3)             │   │
│ │    stage2_backbone_lr:                               │   │
│ │      -1 → 等于 stage2_adapter_lr                    │   │
│ │      指定值 → 使用指定值                             │   │
│ │                                                      │   │
│ │ 3. 新优化器:                                         │   │
│ │    param_groups = [                                   │   │
│ │      {"params": backbone, "lr": backbone_lr},        │   │
│ │      {"params": adapter,  "lr": adapter_lr},         │   │
│ │    ]                                                 │   │
│ │                                                      │   │
│ │ 4. 新调度器:                                         │   │
│ │    CosineAnnealing(warmup=100, max_steps=remaining)  │   │
│ │                                                      │   │
│ │ 5. 启用 EMA (decay=0.999)                            │   │
│ └──────────────────────────────────────────────────────┘   │
│                                                            │
│ ┌──────────────────────────────────────────────────────┐   │
│ │ Stage 2 训练循环 (steps 400 → max_steps):             │   │
│ │                                                      │   │
│ │ ⚡ 所有参数都训练                                     │   │
│ │ ⚡ adapter 和 backbone 可以有不同学习率               │   │
│ │ ⚡ EMA 平滑模型参数                                   │   │
│ │ ⚡ backbone 逐步适应新注入的信号                       │   │
│ └──────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

**2-Stage 训练的意义：**
```
为什么用 2-Stage?

Stage 1 (Adapter Warmup):
  - 新添加的投影层(rnalm_projection, rna_template等)都是随机/零初始化
  - 如果直接和backbone一起训练, 随机梯度会破坏已训练好的backbone
  - Stage 1 冻结backbone, 只训练adapter → adapter学到有意义的投影

Stage 2 (Joint Training):
  - Adapter已经学到了合理的LLM/Template信号投影
  - 现在解冻backbone, 让backbone适应新信号
  - EMA保证训练稳定性

这类似于LoRA/Adapter的思路, 但更灵活:
  - 可以控制每个stage的步数
  - 可以控制每个stage的学习率
  - backbone不需要量化, 可以全精度微调
```

---

## 8. 完整参数手册

### 8.1 RNA/DNA LLM 参数

| 参数 | 类型 | 默认值 | 说明 | 影响的链路 |
|------|------|--------|------|-----------|
| `rnalm.enable` | bool | `false` | 总开关 | 开启后激活所有RNALM相关模块 |
| `rnalm.model_name` | str | `"rinalmo"` | LLM模型名 | 决定embedding维度和模型选择 |
| `rnalm.embedding_dim` | int | `1280` | RNA embedding维度 | RiNALMo=1280, AIDO=2048 |
| `rnalm.dna_embedding_dim` | int | `1024` | DNA embedding维度 | AIDO.DNA=1024 |
| `rnalm.use_rna_embed` | bool | `true` | 加载RNA embedding | false→RNA token embedding为零 |
| `rnalm.use_dna_embed` | bool | `true` | 加载DNA embedding | false→DNA token embedding为零 |
| `rnalm.injection_mode` | str | `"diffusion"` | 注入位置 | "input"→InputEmbedder, "diffusion"→DiffusionCond, "both"→两处 |
| `rnalm.separate_dna_projection` | bool | `false` | RNA/DNA独立投影 | true→两个独立投影层; false→共享投影 |
| `rnalm.gate_mode` | str | `"none"` | 门控模式 | 控制LLM信号强度的方式 |
| `rnalm.gate_init_logit` | float | `-3.0` | 门控初始值 | sigmoid(-3)≈0.047, 初始贡献约5% |
| `rnalm.embedding_dir` | str | `""` | RNA embedding目录 | .pt文件存放路径 |
| `rnalm.sequence_fpath` | str | `""` | RNA序列映射CSV | 当前 featurizer 默认读取 `seq/part_id/seq_label` 并拼接成 `.pt` 路径 |
| `rnalm.dna_embedding_dir` | str | `""` | DNA embedding目录 | DNA .pt文件存放路径 |
| `rnalm.dna_sequence_fpath` | str | `""` | DNA序列映射CSV | 当前 featurizer 默认读取 `seq/part_id/seq_label` 并拼接成 `.pt` 路径 |

### 8.2 RNA Template 参数

| 参数 | 类型 | 默认值 | 说明 | 影响的链路 |
|------|------|--------|------|-----------|
| `rna_template.enable` | bool | `false` | 总开关 | 开启RNA模版注入 |
| `rna_template.search_results_path` | str | `""` | 搜索结果JSON | 当前格式是 `{query_id: {query_sequence, templates}}`；与 `cif_database_dir` 同时设置才进入 Online 模式 |
| `rna_template.cif_database_dir` | str | `""` | CIF数据库目录 | Online 模式需要；必须能解析 `search_results.json` 里的 hit PDB |
| `rna_template.template_database_dir` | str | `""` | 预计算NPZ目录 | Offline模式 (已过时) |
| `rna_template.template_index_path` | str | `""` | 模版索引JSON | Offline模式 (已过时)；online 模式不依赖它 |
| `rna_template.max_rna_templates` | int | `4` | 最大模版数 | 每条query最多使用的模版数 |
| `rna_template.rna3db_metadata_path` | str | `""` | RNA3DB元数据 | 用于时间过滤 |
| `rna_template.injection_mode` | str | `"z_init"` | 保留配置 | 当前实现主路径仍是 TemplateEmbedder/pair 表示注入，不建议把它当成熟开关 |
| `rna_template.projector_init` | str | `"protein"` | 投影器初始化 | "protein"=copy+alpha; "zero"=零初始化 |
| `rna_template.alpha_init` | float | `0.01` | alpha初始值 | 初始RNA template贡献比例 |
| `rna_template.manual_template_hints_path` | str | `""` | training 人工模版JSON | 当前格式是 `pdb_id -> entity_id/* -> templateHints`，不是简单 chain→path 映射 |

### 8.3 Training Stage 参数

| 参数 | 类型 | 默认值 | 说明 | 影响的链路 |
|------|------|--------|------|-----------|
| `two_stage.enable` | bool | `false` | 启用2-stage训练 | 1-stage vs 2-stage |
| `two_stage.adapter_keywords` | str | (见下) | adapter参数关键词 | 决定哪些参数被视为adapter |
| `two_stage.adapter_lr` | float | `-1.0` | 1-stage adapter学习率 | -1→不使用per-group LR |
| `two_stage.backbone_lr` | float | `-1.0` | 1-stage backbone学习率 | -1→不使用per-group LR |
| `two_stage.stage1_max_steps` | int | `400` | Stage 1步数 | 冻结backbone的步数 |
| `two_stage.stage1_adapter_lr` | float | `5e-3` | Stage 1 adapter LR | adapter warmup的学习率 |
| `two_stage.stage1_backbone_lr` | float | `0.0` | Stage 1 backbone LR | 0=完全冻结 |
| `two_stage.stage1_warmup_steps` | int | `1` | Stage 1 warmup | 极短warmup |
| `two_stage.stage2_adapter_lr` | float | `-1.0` | Stage 2 adapter LR | -1→继承stage1_adapter_lr |
| `two_stage.stage2_backbone_lr` | float | `-1.0` | Stage 2 backbone LR | -1→等于stage2_adapter_lr |
| `two_stage.stage2_warmup_steps` | int | `100` | Stage 2 warmup | 解冻后的warmup |
| `two_stage.stage2_ema_decay` | float | `0.999` | Stage 2 EMA衰减 | EMA平滑系数 |

**adapter_keywords 默认值：**
```
"rnalm_projection,rna_projection,dna_projection,
 linear_rnalm,linear_rna_llm,linear_dna_llm,
 rnalm_alpha_logit,rnalm_gate_mlp,
 linear_no_bias_a_rna,rna_template_alpha,rna_template_gate"
```

### 8.4 RNA Loss 参数

| 参数 | 类型 | 默认值 | 说明 | 影响 |
|------|------|--------|------|------|
| `rna_loss.enable` | bool | `false` | 启用RNA loss覆盖 | 使用RNA优化的loss权重 |
| `rna_loss.alpha_distogram` | float | `0.10` | Distogram loss权重 | 0.03→0.10 (增强碱基对距离学习) |
| `rna_loss.alpha_bond` | float | `0.5` | Bond loss权重 | 0.0→0.5 (启用骨架键约束) |
| `rna_loss.weight_rna` | float | `8.0` | RNA原子权重 | 5.0→8.0 (增加RNA原子重要性) |

### 8.5 MSA 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `msa.enable_rna_msa` | bool | `true` | 启用RNA MSA |
| `msa.rna_seq_or_filename_to_msadir_jsons` | list | `[]` | 序列→MSA目录映射JSON |
| `msa.rna_msadir_raw_paths` | list | `[]` | MSA文件存放路径 |
| `msa.rna_indexing_methods` | list | `["sequence"]` | 索引方式 |
| `msa.min_size.train` | int | `1` | 训练集最小MSA大小 |
| `msa.max_size.train` | int | `16384` | 训练集最大MSA大小 |
| `msa.sample_cutoff.train` | int | `16384` | 训练集采样截断 |

---

## 9. 测试用例与验证

### 9.1 测试设计

为验证所有链路连通，设计以下测试用例（不修改项目代码，只有测试脚本）：

#### Case 1: 纯 RNA LLM (RiNALMo, diffusion注入)
```bash
# 测试: rnalm.enable=true, injection_mode=diffusion, separate_dna=false
python runner/train.py \
    --rnalm.enable true \
    --rnalm.model_name rinalmo \
    --rnalm.injection_mode diffusion \
    --rnalm.separate_dna_projection false \
    --max_steps 5
```

#### Case 2: RNA+DNA LLM (AIDO, input注入, separate projection)
```bash
# 测试: rnalm.enable=true, injection_mode=input, separate_dna=true
python runner/train.py \
    --rnalm.enable true \
    --rnalm.model_name aido \
    --rnalm.injection_mode input \
    --rnalm.separate_dna_projection true \
    --rnalm.use_rna_embed true \
    --rnalm.use_dna_embed true \
    --max_steps 5
```

#### Case 3: RNA LLM + both注入 + scalar gate
```bash
python runner/train.py \
    --rnalm.enable true \
    --rnalm.injection_mode both \
    --rnalm.gate_mode scalar \
    --max_steps 5
```

#### Case 4: RNA Template (Online模式, protein init)
```bash
python runner/train.py \
    --rna_template.enable true \
    --rna_template.search_results_path rna_database/search_results.json \
    --rna_template.cif_database_dir /path/to/PDB_RNA \
    --rna_template.projector_init protein \
    --rna_template.alpha_init 0.01 \
    --max_steps 5
```

#### Case 5: RNA Template (zero init)
```bash
python runner/train.py \
    --rna_template.enable true \
    --rna_template.projector_init zero \
    --max_steps 5
```

#### Case 6: RNA LLM + RNA Template (full pipeline)
```bash
python runner/train.py \
    --rnalm.enable true \
    --rnalm.injection_mode diffusion \
    --rna_template.enable true \
    --rna_template.projector_init protein \
    --max_steps 5
```

#### Case 7: 2-Stage训练
```bash
python runner/train.py \
    --rnalm.enable true \
    --rna_template.enable true \
    --two_stage.enable true \
    --two_stage.stage1_max_steps 3 \
    --max_steps 8 \
    # stage1: steps 0-2, stage2: steps 3-7
```

#### Case 8: RNA Loss Override
```bash
python runner/train.py \
    --rna_loss.enable true \
    --rna_loss.alpha_distogram 0.10 \
    --rna_loss.alpha_bond 0.5 \
    --rna_loss.weight_rna 8.0 \
    --max_steps 5
```

#### Case 9: 推理测试 (RNA LLM)
```bash
bash infer_rna.sh \
    --injection_mode diffusion \
    --gate_mode none \
    --use_rnalm true \
    --use_rna true \
    --use_dna true
```

#### Case 10: 推理测试 (RNA Template)
```bash
bash infer_rna.sh \
    --use_rna_template true \
    --rna_projector_init protein \
    --rna_template_alpha 0.01
```

### 9.2 验证检查点

每个测试用例需要验证：

```
✅ 验证检查点:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ 模型初始化成功 (无报错)
□ 正确的投影层被创建
□ 数据加载成功 (embedding/template加载无报错)
□ Forward pass 成功 (loss计算无NaN/Inf)
□ Backward pass 成功 (梯度存在且合理)
□ 参数更新正确 (adapter/backbone按预期更新)
□ Loss下降趋势正常
□ Checkpoint保存成功
□ (推理) 输出结构文件生成
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 10. 附录 — 代码索引

### 10.1 核心模型文件

| 文件 | 关键类/函数 | 说明 |
|------|------------|------|
| `protenix/model/protenix.py` | `Protenix`, `_get_s_rnalm()` | 主模型, RNA LLM diffusion注入 |
| `protenix/model/modules/embedders.py` | `InputFeatureEmbedder` | Input级别的LLM注入 |
| `protenix/model/modules/diffusion.py` | `DiffusionConditioning`, `DiffusionModule` | Diffusion级别的LLM注入 |
| `protenix/model/modules/pairformer.py` | `TemplateEmbedder`, `PairformerStack` | Template处理+PairFormer |
| `protenix/model/rna_loss.py` | `apply_rna_loss_overrides()` | RNA loss权重覆盖 |
| `protenix/model/loss.py` | `ProtenixLoss` | 主loss函数 |

### 10.2 数据处理文件

| 文件 | 关键类/函数 | 说明 |
|------|------------|------|
| `protenix/data/rnalm/rnalm_featurizer.py` | `RiNALMoFeaturizer` | RNA/DNA embedding加载 |
| `protenix/data/rna_template/rna_template_featurizer.py` | `RNATemplateFeaturizer` | RNA模版特征提取 |
| `protenix/data/msa/msa_featurizer.py` | `MSASourceManager`, `MSAFeaturizer` | MSA特征处理 |
| `protenix/data/pipeline/data_pipeline.py` | `DataPipeline` | 数据流水线 |
| `protenix/data/pipeline/dataloader.py` | 训练DataLoader | PyTorch数据加载 |
| `protenix/data/inference/infer_dataloader.py` | `InferenceDataset` | 推理数据加载 |

### 10.3 运行脚本

| 脚本 | 说明 |
|------|------|
| `runner/train.py` | 训练入口 (`AF3Trainer`) |
| `runner/inference.py` | 推理入口 (`InferenceRunner`) |
| `runner/rna_msa_search.py` | RNA MSA搜索 |
| `runner/template_search.py` | 模版搜索 |
| `scripts/prepare_training_data.py` | 准备训练数据 |
| `rna_template/scripts/01_extract_rna_catalog.py` | 提取RNA catalog |
| `rna_template/scripts/generate_training_sequences_from_labels.py` | 从 `train_labels.csv` 生成 `{sequence: [PDB_ID...]}` |
| `rna_template/scripts/03_mmseqs2_search.py` | MMseqs2搜索 |

### 10.4 配置文件

| 文件 | 说明 |
|------|------|
| `configs/configs_base.py` | 基础配置 (rnalm, rna_template, two_stage, rna_loss) |
| `configs/configs_data.py` | 数据配置 (MSA, ESM, 路径) |
| `configs/configs_model_type.py` | 模型架构配置 |
| `configs/configs_inference.py` | 推理配置 |

### 10.5 训练Shell脚本

| 脚本 | 模式 | 说明 |
|------|------|------|
| `finetune_rna_template_1stage.sh` | 1-Stage + Template | Online RNA Template + per-group LR |
| `finetune_rna_template_2stage.sh` | 2-Stage + Template | Stage1冻结backbone → Stage2联合训练 |
| `finetune_rna_template_validate.sh` | 验证 | 完整pipeline验证 |
| `finetune_rna_only.sh` | RNA LLM only | RiNALMo diffusion注入 |
| `finetune_rna_dna.sh` | RNA+DNA LLM | AIDO separate projection |
| `run_two_stage_training_rna_loss.sh` | 2-Stage + RNA Loss | 带RNA loss覆盖的2-stage |
| `infer_rna.sh` | 推理 | RNA LLM + Template推理 |

---

## 完整数据流图

```
                           ┌──────────────────────┐
                           │   Raw mmCIF/PDB Files │
                           └──────────┬───────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
          ┌─────────────────┐                 ┌─────────────────┐
          │ prepare_training │                 │ 序列提取          │
          │ _data.py         │                 │ (RNA/DNA/Protein) │
          └────────┬────────┘                 └────────┬────────┘
                   │                                   │
                   ▼                          ┌────────┴────────┐
          ┌─────────────────┐                 │                 │
          │ bioassembly/     │                 ▼                 ▼
          │ indices.csv      │       ┌─────────────┐   ┌─────────────┐
          └────────┬────────┘       │ RiNALMo/AIDO│   │ MMseqs2     │
                   │                │ Embedding   │   │ MSA Search  │
                   │                │ Generation  │   │ RNA Template│
                   │                └──────┬──────┘   └──────┬──────┘
                   │                       │                 │
                   │              ┌────────┴────────┐       │
                   │              │                 │       │
                   │              ▼                 ▼       ▼
                   │    ┌──────────────┐  ┌──────────┐  ┌──────────┐
                   │    │ RNA .pt files│  │ DNA .pt  │  │search_   │
                   │    │ manifest.csv │  │ files    │  │results.  │
                   │    └──────┬───────┘  └────┬─────┘  │json      │
                   │           │               │        └────┬─────┘
                   │           └───────┬───────┘             │
                   │                   │                     │
                   ▼                   ▼                     ▼
          ┌────────────────────────────────────────────────────────┐
          │                    训练 DataLoader                      │
          │                                                        │
          │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
          │  │RiNALMo       │ │MSAFeaturizer │ │RNATemplate   │   │
          │  │Featurizer    │ │              │ │Featurizer    │   │
          │  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘   │
          │         │                │                │            │
          │         ▼                ▼                ▼            │
          │    rna/dna_llm_emb  profile/del_mean  rna_template_*  │
          └────────────────────────┬───────────────────────────────┘
                                   │
                                   ▼
          ┌────────────────────────────────────────────────────────┐
          │                   Protenix Model                       │
          │                                                        │
          │  ┌──────────────────┐    ┌──────────────────┐          │
          │  │InputFeatureEmbed │    │ TemplateEmbedder  │          │
          │  │(input注入:LLM)   │    │ (template注入)    │          │
          │  └────────┬─────────┘    └────────┬─────────┘          │
          │           │                       │                    │
          │           ▼                       ▼                    │
          │      s_inputs [N,449]      z_template [N,N,c_z]       │
          │           │                       │                    │
          │           ▼                       ▼                    │
          │  ┌────────────────────────────────────────────┐        │
          │  │              PairFormer Stack               │        │
          │  │       s_trunk [N,384], z_trunk [N,N,128]    │        │
          │  └─────────────────────┬──────────────────────┘        │
          │                        │                               │
          │                        ▼                               │
          │  ┌────────────────────────────────────────────┐        │
          │  │         DiffusionModule                     │        │
          │  │  s_rnalm = proj(LLM_emb) (diffusion注入)   │        │
          │  │  conditioning: s_trunk + s_rnalm            │        │
          │  │  denoising → predicted coordinates          │        │
          │  └─────────────────────┬──────────────────────┘        │
          │                        │                               │
          │                        ▼                               │
          │               Predicted Structure                      │
          └────────────────────────────────────────────────────────┘
```

---

## 11. 实际测试结果

### 11.1 单元测试结果（20/20 全部通过）

测试脚本: `test_all_pipeline_cases.py`

```
======================================================================
SUMMARY: 20 PASSED, 0 FAILED out of 20 tests
======================================================================
✅ Config loading: All 4 config sections present, defaults=False
✅ RNALM disabled: No RNALM layers created
✅ Diffusion-only mode: Diffusion mode: no input layers (correct)
✅ Input mode (combined): Combined projection: linear_rnalm (449, 1280), zero-init
✅ Input mode (separate RNA/DNA): Separate: RNA (449, 2048), DNA (449, 1024)
✅ Both mode: Both mode: input layers created
✅ RNA-only (no DNA): RNA-only: has_rna=True, has_dna=False
✅ DiffusionConditioning disabled: rnalm_enable=False
✅ DiffusionConditioning enabled: rnalm_enable=True
✅ Template disabled: No RNA projector created
✅ Protein init (copy + alpha): Protein init: alpha=0.0100, weights copied from protein
✅ Zero init: Zero init: has_alpha=False, weights_are_zero=True
✅ Alpha init values: Alpha init: 0.001→0.0010, 0.01→0.0100, 0.1→0.1000, 1.0→1.0000
✅ RNA loss defaults: distogram=0.10, bond=0.5, weight_rna=8.0
✅ RiNALMoFeaturizer AIDO init: AIDO featurizer: RNA entries=4842, DNA entries=761
✅ RNATemplateFeaturizer online: Online mode enabled (1465 sequences, 5781 dedup hits)
✅ Gate mode code in Protenix: scalar, token, dual, alpha_logit, gate_mlp all present
✅ Two-stage methods exist: setup_stage1, transition_stage2, adapter_keywords, EMA
✅ Adapter keywords coverage: 8/8 adapter keywords present
✅ Data paths: ALL 9 paths verified (bioassembly, indices, checkpoint, RNA/DNA embeddings,
   search_results, PDB_RNA, rna3db_metadata, rna_msa)
```

### 11.2 端到端训练初始化与烟测

使用 `finetune_rna_template_1stage.sh` 进行完整初始化测试：

```
测试配置:
- AIDO 模型 (RNA: 2048-dim, DNA: 1024-dim)
- Separate DNA projection
- RNA Template Online 模式
- Protein projector init + alpha=0.01
- Per-group LR (adapter=0.005, backbone=0.0001)

初始化结果:
✅ 模型参数: 369.86M (包含新增的RNA/DNA投影层)
✅ RNALM Featurizer: RNA 4842 entries, DNA 761 entries loaded
✅ RNA Template: raw search_results=3838 query_ids / 14550 raw hits
✅ RNA Template Featurizer: ONLINE mode loaded 1465 unique sequences / 5781 dedup hits
✅ RNA3DB Metadata: 5389 PDBs with release dates
✅ MSA Featurizer: initialized (sequence indexing mode)
✅ Checkpoint loaded (non-strict, step=0)
✅ RNA projector init: copied_from_protein (confirmed)
✅ EMA: enabled
✅ Training data: 61,771 samples, 5,574 PDB IDs
✅ Validation data: 123 samples, 26 PDB IDs

已知问题:
⚠️ torch.multinomial 采样权重为负 — 这是预存在的数据权重计算问题
   (calc_weights_for_df 中 cluster-based 权重在某些情况下产生负值)
   此问题与 RNA/DNA pipeline 整合无关

说明:
- 这里的结论更准确地应理解为“初始化 / 配置链路 / 模块装配和小规模烟测通过”。
- 这不等同于“完整长程训练已全部验证无问题”。
- 如果要把它升级成严格可复现结论，需要同时记录执行环境、checkpoint 版本、数据版本和具体测试命令。
```

### 11.3 模块级测试验证矩阵

| 测试维度 | 配置组合 | 状态 | 验证内容 |
|---------|---------|------|---------|
| InputEmbedder × rnalm=off | — | ✅ | 无多余层创建 |
| InputEmbedder × diffusion | injection=diffusion | ✅ | 不创建input层 |
| InputEmbedder × input+combined | injection=input, separate=false | ✅ | linear_rnalm (449,1280) 零初始化 |
| InputEmbedder × input+separate | injection=input, separate=true | ✅ | linear_rna_llm + linear_dna_llm 分别创建 |
| InputEmbedder × both | injection=both | ✅ | input层创建 |
| InputEmbedder × rna_only | use_dna=false | ✅ | 只有RNA投影 |
| DiffusionCond × off | rnalm=null | ✅ | rnalm_enable=False |
| DiffusionCond × on | rnalm=enabled | ✅ | rnalm_enable=True |
| TemplateEmb × off | enable=false | ✅ | 无RNA projector |
| TemplateEmb × protein init | projector=protein, α=0.01 | ✅ | 权重复制, alpha=0.01 |
| TemplateEmb × zero init | projector=zero | ✅ | 零初始化, 无alpha |
| TemplateEmb × alpha values | 0.001/0.01/0.1/1.0 | ✅ | 所有值正确设置 |
| RNA Loss | defaults | ✅ | distogram=0.10, bond=0.5, weight=8.0 |
| RiNALMoFeaturizer | AIDO model | ✅ | RNA=4842, DNA=761 entries |
| RNATemplateFeaturizer | Online mode | ✅ | raw file=3838 query_ids/14550 hits; loader聚合后=1465 sequences/5781 dedup hits |
| Gate modes | source analysis | ✅ | scalar/token/dual/none 全支持 |
| Two-stage | code paths | ✅ | stage1/stage2 转换 + EMA |
| Adapter keywords | coverage | ✅ | 8/8 关键词覆盖 |
| Data paths | 9 paths | ✅ | 全部存在 |

### 11.4 各组件初始化确认日志

```
[INFO] RNA embedding enabled: dir=.../aido_embeddings/rna, entries=4842
[INFO] DNA embedding enabled: dir=.../aido_embeddings/dna, entries=761
[INFO] RNA template featurizer: ONLINE mode enabled (1465 sequences, cif_dir=.../PDB_RNA)
[INFO] Search results loaded: 1465 unique sequences, 5781 total hits
[INFO] RNA release dates loaded: 5389 PDBs from filter.json
[INFO] MSAFeaturizer initialized.
[INFO] Model Parameters: 369.86M
[INFO] RNA projector init after checkpoint load: copied_from_protein
[INFO] Using EMA: True
[INFO] Separate RNA/DNA input injection: use_rna=True (2048->449), use_dna=True (1024->449), zero-init
```

---

*报告结束。此报告记录了 Protenix RNA/DNA LLM 融合和 RNA Template 整合的主要 pipeline、工作流、参数配置和测试方案。按当前代码审阅，整体机制描述已经与实现基本对齐；其中最需要额外注意的是 manual template 与默认 search template 在 temporal / self-hit 审核上的语义并不相同，且当前系统统一的是样本级对象而不是底层 sequence artifact lookup 接口。*
