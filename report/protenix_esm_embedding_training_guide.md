# Protenix ESM Embedding 配置与训练完整指南

**Review Date:** March 7, 2026  
**Project:** `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix`  
**Author:** Code Review Agent

---

## Executive Summary

本报告详细说明了在 Protenix 项目中如何配置和使用 **ESM (Evolutionary Scale Modeling) protein embeddings** 进行训练。报告涵盖：

1. **训练配置设置** - 如何启用 ESM 并配置相关参数
2. **Pipeline 执行流程** - ESM 特征如何被处理和缓存
3. **模型下载机制** - ESM 模型权重如何下载和存储
4. **Embedding 注入机制** - ESM 特征如何整合到模型输入中
5. **训练流程** - 完整的训练循环和 ESM 集成

---

## Table of Contents

1. [ESM 配置选项](#1-esm-配置选项)
2. [训练设置指南](#2-训练设置指南)
3. [Pipeline 执行流程](#3-pipeline-执行流程)
4. [ESM 模型下载与处理](#4-esm-模型下载与处理)
5. [Embedding 注入机制](#5-embedding-注入机制)
6. [训练流程详解](#6-训练流程详解)
7. [关键代码路径](#7-关键代码路径)
8. [常见问题与解决方案](#8-常见问题与解决方案)

---

## 1. ESM 配置选项

### 1.1 支持的 ESM 模型

Protenix 支持以下 ESM 模型变体：

| 模型名称 | 类型 | 参数量 | Embedding 维度 | 层数 | 用途 |
|---------|------|--------|---------------|------|------|
| `esm2-3b` | ESM-2 | 3B | 2560 | 36 | 标准蛋白 embedding |
| `esm2-3b-ism` | ESM-2 ISM | 3B | 2560 | 36 | Inverse Folding 任务 |

**配置文件位置:** `configs/configs_base.py`, `configs/configs_model_type.py`

### 1.2 默认 ESM 配置

```python
# configs/configs_base.py
"esm": {
    "enable": False,           # 是否启用 ESM
    "model_name": "esm2-3b",   # ESM 模型名称
    "embedding_dim": 2560,     # ESM embedding 维度
}
```

### 1.3 预定义模型配置

项目提供了多个预配置的模型，其中包含 ESM 支持的模型：

```python
# configs/configs_model_type.py

# protenix_mini_esm_v0.5.0 - 轻量级 ESM-only 模型
"protenix_mini_esm_v0.5.0": {
    "esm": {
        "enable": True,
        "model_name": "esm2-3b",
    },
    "use_msa": False,  # 不使用 MSA，仅依赖 ESM
}

# protenix_mini_ism_v0.5.0 - ISM 变体
"protenix_mini_ism_v0.5.0": {
    "esm": {
        "enable": True,
        "model_name": "esm2-3b-ism",
    },
    "use_msa": False,
}

# protenix_base_constraint_v0.5.0 - 基础模型 + 约束 + ESM
"protenix_base_constraint_v0.5.0": {
    "esm": {
        "enable": True,
        "model_name": "esm2-3b",
    },
    "load_strict": False,  # 从 base 模型 finetune 时需设为 False
}
```

---

## 2. 训练设置指南

### 2.1 启用 ESM 的训练配置

#### 方法 A: 使用预定义模型（推荐）

```bash
# 使用轻量级 ESM 模型
python3 ./runner/train.py \
    --model_name "protenix_mini_esm_v0.5.0" \
    --run_name "my_esm_finetune" \
    --seed 42 \
    --base_dir ./output \
    --dtype bf16 \
    --project protenix \
    --use_wandb false \
    --diffusion_batch_size 48 \
    --train_crop_size 384 \
    --max_steps 100000 \
    --warmup_steps 2000 \
    --lr 0.001 \
    --model.N_cycle 4 \
    --sample_diffusion.N_step 20 \
    --load_checkpoint_path ./checkpoints/protenix_mini_esm_v0.5.0.pt \
    --load_ema_checkpoint_path ./checkpoints/protenix_mini_esm_v0.5.0.pt \
    --data.train_sets weightedPDB_before2109_wopb_nometalc_0925 \
    --data.test_sets recentPDB_1536_sample384_0925
```

#### 方法 B: 手动启用 ESM

```bash
python3 ./runner/train.py \
    --model_name "protenix_base_default_v1.0.0" \
    --esm.enable true \
    --esm.model_name "esm2-3b" \
    --esm.embedding_dim 2560 \
    ...其他参数
```

### 2.2 完整训练脚本示例

```bash
#!/bin/bash
# finetune_with_esm.sh

set -e

# 环境设置
export PROTENIX_ROOT_DIR="/inspire/ssd/project/sais-bio/public/ash_proj/data"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 内核配置
export TRIANGLE_ATTENTION="cuequivariance"
export TRIANGLE_MULTIPLICATIVE="cuequivariance"

# 训练配置
MODEL_NAME="protenix_mini_esm_v0.5.0"
RUN_NAME="esm_finetune"
CHECKPOINT_PATH="./checkpoints/protenix_mini_esm_v0.5.0.pt"
OUTPUT_DIR="./output/esm_finetune"

# 启动训练
python3 ./runner/train.py \
    --model_name "${MODEL_NAME}" \
    --run_name "${RUN_NAME}" \
    --seed 42 \
    --base_dir "${OUTPUT_DIR}" \
    --dtype bf16 \
    --project protenix \
    --use_wandb false \
    --diffusion_batch_size 48 \
    --eval_interval 400 \
    --log_interval 50 \
    --checkpoint_interval 400 \
    --ema_decay 0.999 \
    --train_crop_size 384 \
    --max_steps 100000 \
    --warmup_steps 2000 \
    --lr 0.001 \
    --grad_clip_norm 10 \
    --model.N_cycle 4 \
    --sample_diffusion.N_step 20 \
    --load_checkpoint_path "${CHECKPOINT_PATH}" \
    --load_ema_checkpoint_path "${CHECKPOINT_PATH}" \
    --data.train_sets weightedPDB_before2109_wopb_nometalc_0925 \
    --data.weightedPDB_before2109_wopb_nometalc_0925.base_info.mmcif_dir "${PROTENIX_ROOT_DIR}/mmcif" \
    --data.weightedPDB_before2109_wopb_nometalc_0925.base_info.bioassembly_dict_dir "${PROTENIX_ROOT_DIR}/mmcif_bioassembly" \
    --data.test_sets recentPDB_1536_sample384_0925
```

### 2.3 关键配置参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--esm.enable` | `false` | 是否启用 ESM embeddings |
| `--esm.model_name` | `"esm2-3b"` | ESM 模型名称 |
| `--esm.embedding_dim` | `2560` | ESM embedding 维度 |
| `--use_msa` | `true` | 是否使用 MSA (ESM 模型通常设为 false) |
| `--load_strict` | `true` | 严格加载 checkpoint (模型架构变化时需设为 false) |

---

## 3. Pipeline 执行流程

### 3.1 训练数据加载流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    Training Pipeline Flow                        │
└─────────────────────────────────────────────────────────────────┘

1. 读取 mmCIF 文件
   ↓
2. DataPipeline.get_data_from_mmcif()
   - 解析生物组装结构
   - 生成 atom_array 和 token_array
   ↓
3. 加载预计算的 ESM embeddings
   - 从缓存文件加载 (.pt 文件)
   - 映射到对应的 protein entities
   ↓
4. Featurizer 处理
   - MSAFeaturizer (如果启用)
   - TemplateFeaturizer (如果启用)
   - ESMFeaturizer (如果启用)
   ↓
5. InputFeatureEmbedder
   - 整合所有输入特征
   - 通过 linear_esm 投影 ESM embeddings
   ↓
6. Protenix Model
   - Pairformer
   - Diffusion Module
   - Confidence Head
```

### 3.2 ESM 特征处理流程

**代码路径:** `protenix/data/pipeline/dataset.py` → `protenix/data/esm/esm_featurizer.py`

```python
# 训练时的 ESM 特征处理 (伪代码)
class BaseSingleDataset(Dataset):
    def __getitem__(self, idx):
        # 1. 从 mmCIF 获取结构数据
        bioassembly_dict = DataPipeline.get_data_bioassembly(pkl_path)
        atom_array = bioassembly_dict["atom_array"]
        token_array = bioassembly_dict["token_array"]
        
        # 2. 如果是训练模式，从缓存加载 ESM embeddings
        if self.esm_enable:
            esm_featurizer = ESMFeaturizer(
                embedding_dir=self.esm_embedding_dir,
                sequence_fpath=self.esm_sequence_fpath,
                embedding_dim=self.esm_embedding_dim
            )
            
            # 3. 获取 ESM embedding
            # 注意：训练时使用预计算的 embeddings，不实时计算
            x_esm = esm_featurizer(
                token_array=token_array,
                atom_array=atom_array,
                bioassembly_dict=bioassembly_dict,
                inference_mode=False
            )
            
            # 4. 添加到特征字典
            features_dict["esm_token_embedding"] = x_esm
        
        return features_dict
```

### 3.3 ESMFeaturizer 详细流程

**文件:** `protenix/data/esm/esm_featurizer.py`

```python
class ESMFeaturizer:
    def __init__(self, embedding_dir, sequence_fpath, embedding_dim=2560):
        self.embedding_dir = embedding_dir  # ESM embeddings 缓存目录
        self.sequence_fpath = sequence_fpath  # 序列到文件名的映射
        self.seq_to_filename = self.get_seq_to_filename(sequence_fpath)
        self.embedding_dim = embedding_dim
    
    def __call__(self, token_array, atom_array, bioassembly_dict, inference_mode=False):
        N_token = len(token_array)
        # 初始化为零
        x = torch.zeros([N_token, self.embedding_dim])
        
        # 获取每个 token 的中心原子
        centre_atoms_indices = token_array.get_annotation("centre_atom_index")
        centre_atom_array = atom_array[centre_atoms_indices]
        
        # 获取 protein entity IDs
        is_protein = centre_atom_array.chain_mol_type == "protein"
        protein_entity_ids = set(centre_atom_array.label_entity_id[is_protein])
        
        # 遍历每个 protein entity
        for entity_id in protein_entity_ids:
            # 1. 获取序列
            sequence = bioassembly_dict["sequences"][str(entity_id)]
            
            # 2. 加载预计算的 ESM embedding
            x_esm = self.load_esm_embedding(sequence)
            
            # 3. 获取 residue indices
            entity_mask = centre_atom_array.label_entity_id == entity_id
            res_index = centre_atom_array.res_id[entity_mask] - 1  # 从 1 开始
            
            # 4. 映射到对应的 token 位置
            x[entity_mask] = x_esm[res_index]
        
        return x
```

---

## 4. ESM 模型下载与处理

### 4.1 ESM 模型下载流程

**触发位置:** `runner/inference.py` 和 `protenix/data/inference/infer_dataloader.py`

**下载 URL 配置:** `protenix/web_service/dependency_url.py`

```python
URL = {
    "protenix_mini_esm_v0.5.0": "https://protenix.tos-cn-beijing.volces.com/checkpoint/protenix_mini_esm_v0.5.0.pt",
    "esm2_t36_3B_UR50D": "https://protenix.tos-cn-beijing.volces.com/checkpoint/esm2_t36_3B_UR50D.pt",
    "esm2_t36_3B_UR50D-contact-regression": "https://protenix.tos-cn-beijing.volces.com/checkpoint/esm2_t36_3B_UR50D-contact-regression.pt",
    "esm2_t36_3B_UR50D_ism": "https://protenix.tos-cn-beijing.volces.com/checkpoint/esm2_t36_3B_UR50D_ism.pt",
    "esm2_t36_3B_UR50D_ism-contact-regression": "https://protenix.tos-cn-beijing.volces.com/checkpoint/esm2_t36_3B_UR50D_ism-contact-regression.pt",
}
```

### 4.2 推理时 ESM 模型下载代码

**文件:** `runner/inference.py` (Lines 349-382)

```python
def prepare_model_checkpoint(configs, checkpoint_dir):
    # ... 主模型下载 ...
    
    # ESM 模型下载 (如果模型名称包含 "esm")
    if "esm" in configs.model_name:
        # 下载 ESM-2 3B 基础权重
        esm_3b_ckpt_path = f"{checkpoint_dir}/esm2_t36_3B_UR50D.pt"
        if not opexists(esm_3b_ckpt_path):
            tos_url = URL["esm2_t36_3B_UR50D"]
            logger.info(f"Downloading ESM checkpoint from {tos_url}...")
            download_from_url(tos_url, esm_3b_ckpt_path)
        
        # 下载 contact regression 头
        esm_3b_ckpt_path2 = f"{checkpoint_dir}/esm2_t36_3B_UR50D-contact-regression.pt"
        if not opexists(esm_3b_ckpt_path2):
            tos_url = URL["esm2_t36_3B_UR50D-contact-regression"]
            download_from_url(tos_url, esm_3b_ckpt_path2)
    
    # ISM 模型下载 (如果模型名称包含 "ism")
    if "ism" in configs.model_name:
        esm_3b_ism_ckpt_path = f"{checkpoint_dir}/esm2_t36_3B_UR50D_ism.pt"
        if not opexists(esm_3b_ism_ckpt_path):
            tos_url = URL["esm2_t36_3B_UR50D_ism"]
            download_from_url(tos_url, esm_3b_ism_ckpt_path)
        
        esm_3b_ism_ckpt_path2 = f"{checkpoint_dir}/esm2_t36_3B_UR50D_ism-contact-regression.pt"
        if not opexists(esm_3b_ism_ckpt_path2):
            tos_url = URL["esm2_t36_3B_UR50D_ism-contact-regression"]
            download_from_url(tos_url, esm_3b_ism_ckpt_path2)
```

### 4.3 推理时 ESM Embedding 预计算

**文件:** `protenix/data/inference/infer_dataloader.py` (Lines 119-145)

```python
class InferenceDataset(Dataset):
    def __init__(self, configs):
        self.configs = configs
        self.esm_enable = configs.get("esm", {}).get("enable", False)
        
        if self.esm_enable:
            # 设置 ESM 缓存路径
            configs.esm.embedding_dir = f"./esm_embeddings/{configs.esm.model_name}"
            configs.esm.sequence_fpath = f"./esm_embeddings/{json_task_name}_prot_sequences.csv"
            
            # 创建目录
            os.makedirs(configs.esm.embedding_dir, exist_ok=True)
            os.makedirs(os.path.dirname(configs.esm.sequence_fpath), exist_ok=True)
            
            # 预计算 ESM embeddings
            ESMFeaturizer.precompute_esm_embedding(
                inputs=self.inputs,  # 输入样本列表
                model_name=configs.esm.model_name,
                embedding_dir=configs.esm.embedding_dir,
                sequence_fpath=configs.esm.sequence_fpath,
                checkpoint_dir=configs.load_checkpoint_dir,
            )
            
            # 初始化 ESM Featurizer
            self.esm_featurizer = ESMFeaturizer(
                embedding_dir=configs.esm.embedding_dir,
                sequence_fpath=configs.esm.sequence_fpath,
                embedding_dim=configs.esm.embedding_dim,
                error_dir="./esm_embeddings/",
            )
```

### 4.4 ESM Embedding 预计算详细流程

**文件:** `protenix/data/esm/esm_featurizer.py` (Lines 137-173)

```python
@staticmethod
def precompute_esm_embedding(inputs, model_name, embedding_dir, sequence_fpath, checkpoint_dir):
    """
    为所有输入样本预计算 ESM embeddings
    
    Args:
        inputs: 输入样本列表 (来自 JSON)
        model_name: ESM 模型名称
        embedding_dir: embeddings 保存目录
        sequence_fpath: 序列 CSV 保存路径
        checkpoint_dir: ESM 模型权重目录
    """
    print("Precompute ESM embeddings")
    
    # 1. 提取所有 protein sequences
    all_seq_dict = []
    for sample_dict in inputs:
        sample_name = sample_dict["name"]
        for i, entity_info_wrapper in enumerate(sample_dict["sequences"]):
            pdb_entity_id = sample_name + "_" + str(i + 1)
            entity_type = list(entity_info_wrapper.keys())[0]
            entity_info = entity_info_wrapper[entity_type]
            
            # 只处理 protein chains
            if entity_type == "proteinChain":
                all_seq_dict.append({
                    "seq": entity_info["sequence"],
                    "pdb_entity_id": pdb_entity_id,
                    "seq_label": pdb_entity_id,
                    "part_id": pdb_entity_id,
                })
    
    # 2. 保存序列到 CSV
    df_seq = pd.DataFrame(all_seq_dict)
    df_seq.to_csv(sequence_fpath)
    
    # 3. 加载 ESM 模型
    model, alphabet = load_esm_model(model_name, local_esm_dir=checkpoint_dir)
    
    # 4. 分批计算 embeddings
    part_counts = dict(df_seq["part_id"].value_counts())
    for part_id, count in part_counts.items():
        df_part = df_seq[df_seq["part_id"] == part_id]
        labels = df_part["seq_label"].tolist()
        sequences = df_part["seq"].tolist()
        
        try:
            save_dir = os.path.join(embedding_dir, part_id)
            os.makedirs(save_dir, exist_ok=True)
            
            # 计算 ESM embeddings
            lm_embeddings = compute_ESM_embeddings(
                model_name=model_name,
                model=model,
                alphabet=alphabet,
                labels=labels,
                sequences=sequences,
                save_dir=save_dir,
                truncation_seq_length=4094,
                toks_per_batch=16384,
            )
            print(f"[{part_id}] Processed {len(lm_embeddings)} sequences")
        except Exception as e:
            print(f"[{part_id}] Error: {e}")
```

### 4.5 ESM 模型加载

**文件:** `protenix/data/esm/compute_esm.py` (Lines 35-52)

```python
ESM_CONFIG = {
    "esm2-3b": {
        "type": "esm2",
        "model_path": "esm2_t36_3B_UR50D.pt",
        "emb_dim": 2560,
        "n_layers": 36,
    },
    "esm2-3b-ism": {
        "type": "esm2",
        "model_path": "esm2_t36_3B_UR50D_ism.pt",
        "emb_dim": 2560,
        "n_layers": 36,
    },
}

def load_esm_model(model_name, local_esm_dir="release_data/checkpoint"):
    """
    加载 ESM 模型
    
    Args:
        model_name: 模型名称 ("esm2-3b" 或 "esm2-3b-ism")
        local_esm_dir: 本地模型权重目录
    
    Returns:
        model, alphabet: 加载的 ESM 模型和字母表
    """
    local_model_path = os.path.join(local_esm_dir, ESM_CONFIG[model_name]["model_path"])
    
    # 检查本地模型是否存在
    if os.path.exists(local_model_path):
        print("Loading ESM from ", local_model_path)
    
    # ISM 模型必须从本地加载 (不能从 fair-esm 下载)
    if "ism" in model_name and not os.path.exists(local_model_path):
        raise RuntimeError(
            f"esm2-3b-ism model: {local_model_path} does not exist\n"
            "Download from: https://af3-dev.tos-cn-beijing.volces.com/release_model/esm2_t36_3B_UR50D_ism.pt"
        )
    
    # 加载 ESM2 模型
    if model_name.startswith("esm2"):
        model, alphabet = _load_esm2_model(local_model_path)
    
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model, alphabet
```

### 4.6 ESM Embedding 计算

**文件:** `protenix/data/esm/compute_esm.py` (Lines 82-120)

```python
def compute_esm2_embeddings(model, alphabet, labels, sequences, save_dir,
                            toks_per_batch=4096, truncation_seq_length=1022):
    """
    计算 ESM-2 embeddings
    
    Args:
        model: ESM 模型
        alphabet: ESM 字母表
        labels: 序列标签列表
        sequences: 氨基酸序列列表
        save_dir: embeddings 保存目录
        toks_per_batch: 每批次的 token 数
        truncation_seq_length: 截断长度
    
    Returns:
        embeddings: 计算得到的 embeddings 字典
    """
    # 创建批量数据集
    dataset = FastaBatchedDataset(labels, sequences)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length),
        batch_sampler=batches,
    )
    
    repr_layer = model.num_layers  # 使用最后一层
    embeddings = {}
    
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(tqdm(data_loader)):
            # 检查是否已存在
            if _check_files_exist(save_dir, labels):
                continue
            
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)
            
            # 前向传播
            out = model(toks, repr_layers=[repr_layer], return_contacts=False)
            representation = out["representations"][repr_layer].to(device="cpu")
            
            # 保存每个序列的 embedding
            for i, label in enumerate(labels):
                truncate_len = min(truncation_seq_length, len(strs[i]))
                # 去除 BOS token，保留氨基酸 embedding
                embeddings[label] = representation[i, 1:truncate_len + 1].clone()
                
                # 保存到文件
                save_path = os.path.join(save_dir, label + ".pt")
                torch.save(embeddings[label], save_path)
    
    return embeddings
```

---

## 5. Embedding 注入机制

### 5.1 InputFeatureEmbedder 架构

**文件:** `protenix/model/modules/embedders.py`

```python
class InputFeatureEmbedder(nn.Module):
    """
    实现 AF3 算法 2
    
    整合以下输入特征:
    - restype (32): 残基类型 one-hot
    - profile (32): MSA profile
    - deletion_mean (1): 平均 deletion 率
    - ESM embedding (2560): 如果启用
    """
    
    def __init__(self, c_atom=128, c_atompair=16, c_token=384, esm_configs={}):
        super().__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        
        # Atom Attention Encoder (处理 per-atom 特征)
        self.atom_attention_encoder = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=False,
        )
        
        # ESM 配置
        self.esm_configs = {
            "enable": esm_configs.get("enable", False),
            "embedding_dim": esm_configs.get("embedding_dim", 2560),
        }
        
        # ESM 投影层 (如果启用)
        if self.esm_configs["enable"]:
            self.linear_esm = LinearNoBias(
                self.esm_configs["embedding_dim"],  # 输入：2560
                self.c_token + 32 + 32 + 1,         # 输出：449
            )
            # 初始化为零 (finetune 时从零开始学习)
            nn.init.zeros_(self.linear_esm.weight)
        
        # 其他输入特征维度
        self.input_feature = {
            "restype": 32,      # 残基类型
            "profile": 32,      # MSA profile
            "deletion_mean": 1, # 平均 deletion
        }
    
    def forward(self, input_feature_dict, inplace_safe=False, chunk_size=None):
        """
        Args:
            input_feature_dict: 包含所有输入特征的字典
                - atom_to_token_idx: [..., N_atom]
                - ref_pos: [..., N_atom, 3]
                - ref_charge: [..., N_atom]
                - ref_mask: [..., N_atom]
                - ref_atom_name_chars: [..., N_atom, 4]
                - ref_element: [..., N_atom]
                - d_lm: [..., N_atom] (language model distance)
                - v_lm: [..., N_atom] (language model validity)
                - pad_info: [..., N_atom]
                - restype: [..., N_token, 32]
                - profile: [..., N_token, 32]
                - deletion_mean: [..., N_token, 1]
                - esm_token_embedding: [..., N_token, 2560] (如果启用)
        
        Returns:
            s_inputs: [..., N_token, 449]
                449 = c_token(384) + restype(32) + profile(32) + deletion_mean(1)
        """
        # 1. Atom Attention Encoder 处理 per-atom 特征
        a, _, _, _ = self.atom_attention_encoder(
            input_feature_dict["atom_to_token_idx"],
            input_feature_dict["ref_pos"],
            input_feature_dict["ref_charge"],
            input_feature_dict["ref_mask"],
            input_feature_dict["ref_atom_name_chars"],
            input_feature_dict["ref_element"],
            input_feature_dict["d_lm"],
            input_feature_dict["v_lm"],
            input_feature_dict["pad_info"],
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )  # [..., N_token, c_token=384]
        
        # 2. 拼接 per-token 特征
        batch_shape = input_feature_dict["restype"].shape[:-1]
        s_inputs = torch.cat(
            [a]  # [384]
            + [
                input_feature_dict[name].reshape(*batch_shape, d)
                for name, d in self.input_feature.items()
            ]  # [32] + [32] + [1]
            ,
            dim=-1,
        )  # [..., N_token, 449]
        
        # 3. 添加 ESM embedding (如果启用)
        if self.esm_configs["enable"]:
            if self.training:
                logger.warning(
                    "Make sure esm_token_embedding has been cached for training. "
                    "If not, refer to protenix/data/esm/compute_esm.py first."
                )
            
            # 投影 ESM embedding 到模型维度
            esm_embeddings = self.linear_esm(
                input_feature_dict["esm_token_embedding"]
            )  # [..., N_token, 449]
            
            # 残差连接
            s_inputs = s_inputs + esm_embeddings  # [..., N_token, 449]
        
        return s_inputs
```

### 5.2 ESM 注入流程图

```
┌─────────────────────────────────────────────────────────────────┐
│              InputFeatureEmbedder Forward Pass                  │
└─────────────────────────────────────────────────────────────────┘

输入特征字典:
├─ atom_to_token_idx  ─┐
├─ ref_pos            │
├─ ref_charge         │
├─ ref_mask           │
├─ ref_atom_name_chars├─→ AtomAttentionEncoder ─→ a [..., N_token, 384]
├─ ref_element        │
├─ d_lm               │
├─ v_lm               │
└─ pad_info           ┘

├─ restype        ────────────────┐
├─ profile        ────────────────┤
└─ deletion_mean  ────────────────┤
                                  │
                                  ↓
                            Concat + a ─→ s_inputs_base [..., 449]
                                  
┌─ esm_token_embedding [..., N_token, 2560]
│
↓
LinearNoBias(2560 → 449)
│
↓
esm_embeddings [..., N_token, 449]
│
↓ (残差连接)
s_inputs = s_inputs_base + esm_embeddings [..., 449]
```

### 5.3 ESM 维度变换

```python
# ESM embedding 维度变换
ESM 原始输出:     [N_token, 2560]
                    ↓
LinearNoBias:     [N_token, 2560] → [N_token, 449]
                    ↓
残差连接:         [N_token, 449] + [N_token, 449] = [N_token, 449]

# 449 = c_token(384) + restype(32) + profile(32) + deletion_mean(1)
```

### 5.4 为什么使用 LinearNoBias 并初始化为零？

```python
self.linear_esm = LinearNoBias(2560, 449)
nn.init.zeros_(self.linear_esm.weight)
```

**原因:**

1. **零初始化**: 训练开始时，ESM 贡献为零，模型行为与不使用 ESM 时相同
2. **渐进学习**: 随着训练进行，模型逐渐学习如何利用 ESM 特征
3. **稳定性**: 避免训练初期 ESM 特征主导模型行为
4. **Finetune 友好**: 从预训练模型 finetune 时，可以平滑引入 ESM 特征

---

## 6. 训练流程详解

### 6.1 完整训练循环

**文件:** `runner/train.py`

```python
class AF3Trainer:
    def __init__(self, configs):
        self.configs = configs
        self.init_env()       # 初始化分布式环境
        self.init_basics()    # 初始化目录和计数器
        self.init_log()       # 初始化日志
        self.init_model()     # 初始化模型 (包含 ESM 配置)
        self.init_loss()      # 初始化损失函数
        self.init_data()      # 初始化数据加载器
        self.try_load_checkpoint()
    
    def init_model(self):
        """初始化 Protenix 模型"""
        self.model = Protenix(self.configs)
        
        # 如果启用 ESM，模型会自动使用 InputFeatureEmbedder 中的 linear_esm
        if self.configs.esm.enable:
            self.print("ESM enabled with model:", self.configs.esm.model_name)
        
        # 分布式训练包装
        if DIST_WRAPPER.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[DIST_WRAPPER.local_rank],
                find_unused_parameters=self.configs.model.find_unused_parameters,
            )
        
        # 加载 checkpoint
        if self.configs.load_checkpoint_path:
            self.load_checkpoint(self.configs.load_checkpoint_path)
    
    def init_data(self):
        """初始化数据加载器"""
        self.train_loader, self.test_loader = get_dataloaders(self.configs)
        
        # 数据加载器会自动处理 ESM 特征
        # 参见 protenix/data/pipeline/dataloader.py
    
    def train_step(self, batch):
        """单个训练步骤"""
        # batch 包含 input_feature_dict，其中可能有 esm_token_embedding
        input_feature_dict = batch["input_feature_dict"]
        
        # 前向传播
        # 在 InputFeatureEmbedder 中，ESM embedding 会被投影并添加到 s_inputs
        outputs = self.model(input_feature_dict)
        
        # 计算损失
        loss = self.loss_fn(outputs, batch)
        
        # 反向传播
        loss.backward()
        
        return loss
```

### 6.2 训练时的 ESM 特征流

```
训练迭代流程:

1. DataLoader 采样 batch
   ↓
2. Dataset.__getitem__(idx)
   - 从 mmCIF 加载结构
   - 从缓存加载 ESM embeddings (.pt 文件)
   - 生成特征字典
   ↓
3. Collate function
   - 批量处理特征
   ↓
4. Model.forward(input_feature_dict)
   ↓
5. InputFeatureEmbedder
   - 如果 esm_enable=True:
     s_inputs = base_features + linear_esm(esm_token_embedding)
   ↓
6. Pairformer / Diffusion Module
   ↓
7. 计算损失
   ↓
8. 反向传播
   - linear_esm 的权重被更新
```

### 6.3 训练时的关键注意事项

**来自代码的警告:**

```python
# protenix/model/modules/embedders.py:115
if self.esm_configs["enable"]:
    if self.training:
        logger.warning(
            "Make sure esm_token_embedding has been cached for training. "
            "If not, refer to protenix/data/esm/compute_esm.py first."
        )
```

**重要提示:**

1. **预计算 ESM embeddings**: 训练前必须为所有训练样本预计算 ESM embeddings
2. **缓存路径**: embeddings 必须保存在正确的目录结构中
3. **序列映射**: `seq_to_filename.json` 必须正确映射序列到缓存文件

### 6.4 ESM 训练数据准备脚本

```bash
#!/bin/bash
# prepare_esm_embeddings.sh

# 准备训练数据的 ESM embeddings

CHECKPOINT_DIR="./checkpoints"
EMBEDDING_DIR="./esm_embeddings/esm2-3b"
SEQUENCE_FPATH="./esm_embeddings/train_sequences.csv"

# 1. 提取训练序列
python3 -c "
import json
import pandas as pd

# 从训练数据 JSON 提取 protein sequences
with open('train_data.json', 'r') as f:
    train_data = json.load(f)

all_seqs = []
for sample in train_data:
    for i, entity_wrapper in enumerate(sample['sequences']):
        entity_type = list(entity_wrapper.keys())[0]
        if entity_type == 'proteinChain':
            seq = entity_wrapper[entity_type]['sequence']
            all_seqs.append({
                'seq': seq,
                'seq_label': f\"{sample['name']}_{i+1}\",
                'part_id': f\"{sample['name']}_{i+1}\"
            })

df = pd.DataFrame(all_seqs)
df.to_csv('$SEQUENCE_FPATH', index=False)
"

# 2. 预计算 ESM embeddings
python3 ./protenix/data/esm/compute_esm.py \
    --model_name esm2-3b \
    --embedding_dir $EMBEDDING_DIR \
    --sequence_fpath $SEQUENCE_FPATH \
    --checkpoint_dir $CHECKPOINT_DIR
```

---

## 7. 关键代码路径

### 7.1 配置文件

| 文件 | 作用 |
|------|------|
| `configs/configs_base.py` | 基础配置，包含 ESM 默认设置 |
| `configs/configs_model_type.py` | 预定义模型配置 (包含 ESM 模型) |
| `configs/configs_data.py` | 数据配置 (训练/测试集路径) |

### 7.2 ESM 相关模块

| 文件 | 作用 |
|------|------|
| `protenix/data/esm/compute_esm.py` | ESM 模型加载和 embedding 计算 |
| `protenix/data/esm/esm_featurizer.py` | ESMFeaturizer 类，训练/推理时使用 |
| `protenix/data/esm/__init__.py` | ESM 模块导出 |

### 7.3 模型模块

| 文件 | 作用 |
|------|------|
| `protenix/model/modules/embedders.py` | InputFeatureEmbedder，ESM 注入点 |
| `protenix/model/protenix.py` | 主模型，调用 embedders |

### 7.4 数据管道

| 文件 | 作用 |
|------|------|
| `protenix/data/pipeline/dataset.py` | 训练数据集，加载 ESM embeddings |
| `protenix/data/pipeline/dataloader.py` | 数据加载器 |
| `protenix/data/inference/infer_dataloader.py` | 推理数据集，预计算 ESM |

### 7.5 训练/推理脚本

| 文件 | 作用 |
|------|------|
| `runner/train.py` | 训练主循环 |
| `runner/inference.py` | 推理脚本，下载 ESM 模型 |

---

## 8. 常见问题与解决方案

### 8.1 ESM 模型加载失败

**问题:** `RuntimeError: Error(s) in loading state_dict`

**原因:** PyTorch 2.6+ 与 fair-esm 兼容性问题

**解决方案:** 参见 `tests/test_esm_loading.py`

```python
# 检查 PyTorch 版本
import torch
print(f"PyTorch version: {torch.__version__}")

# PyTorch 2.6+ 可能需要降级或使用补丁
```

### 8.2 ESM Embedding 维度不匹配

**问题:** `AssertionError: ESM embedding size X not equal to sequence length Y`

**原因:** 缓存的 embeddings 与当前序列不匹配

**解决方案:** 删除缓存并重新计算

```bash
rm -rf ./esm_embeddings/esm2-3b/*
# 重新运行预计算
```

### 8.3 训练时 ESM 特征未加载

**问题:** 训练日志显示 ESM 特征全为零

**原因:** `esm_token_embedding` 未在特征字典中

**解决方案:** 检查数据加载器配置

```python
# 确认 esm.enable = true
# 确认 embedding_dir 和 sequence_fpath 正确设置
```

### 8.4 ISM 模型下载失败

**问题:** `esm2-3b-ism model does not exist`

**原因:** ISM 模型不能从 fair-esm 自动下载

**解决方案:** 手动下载

```bash
wget https://af3-dev.tos-cn-beijing.volces.com/release_model/esm2_t36_3B_UR50D_ism.pt \
    -O ./checkpoints/esm2_t36_3B_UR50D_ism.pt
```

### 8.5 内存不足 (OOM)

**问题:** 计算 ESM embeddings 时 OOM

**解决方案:** 减小 batch size

```python
# 修改 compute_esm.py
toks_per_batch=8192  # 从 16384 减小
```

---

## 9. 总结

### 9.1 ESM 配置步骤总结

1. **选择模型**: 使用预定义模型 (如 `protenix_mini_esm_v0.5.0`) 或手动配置
2. **下载权重**: 推理时自动下载，训练时需确保 checkpoint 存在
3. **预计算 embeddings**: 为所有训练样本计算并缓存 ESM embeddings
4. **配置训练**: 设置 `--esm.enable true` 和使用正确的 model_name
5. **启动训练**: 运行训练脚本，ESM 特征会自动注入

### 9.2 ESM 优势

- **无需 MSA**: 对于 `protenix_mini_esm_v0.5.0`，可以不依赖 MSA
- **快速推理**: 预计算 embeddings，推理时只需加载
- **高质量特征**: ESM-2 3B 提供强大的 protein representation

### 9.3 性能考虑

| 场景 | 建议 |
|------|------|
| 训练 | 预计算所有 embeddings，使用缓存 |
| 推理 | 批量预计算，共享相同序列的 embeddings |
| 内存 | 调整 `toks_per_batch` 避免 OOM |
| 速度 | 使用 GPU 计算 ESM，16 workers 并行 |

---

*报告生成日期: March 7, 2026*  
*项目版本: Protenix v1.0.0*  
*ESM 版本: ESM-2 3B (esm2_t36_3B_UR50D)*
