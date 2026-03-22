# 从零生成 `rna_database` 并启动 Online RNA Template Finetune

**Date**: 2026-03-15

## 目标

假设以下原始资源 **保持不变**：

- CIF template 库保持不变
- training data 保持不变

但你当前**没有**下面这些运行产物：

- `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database`
- `rna_catalog.json`
- `search_results.json`
- `rna_template_index.json`
- 各类离线 `.npz` template

本说明给出一条**从零开始**、最终能够启动 **online RNA template finetune** 的最短正确流程。

---

## 核心结论

你当前这版 online RNA template pipeline 的本质是：

- **离线准备 hit 候选池**
- **训练时在线从 CIF 现算 template feature**

所以：

- 你**不需要**预先构建 `.npz` template
- 你**仍然需要**先准备 `search_results.json`

换句话说，最小闭环是：

1. 准备 `rna_catalog.json`
2. 准备 `search_results.json`
3. 跑 `finetune_rna_template_1stage.sh` 或 `finetune_rna_template_2stage.sh`
4. 训练时在线从 CIF 计算 `rna_template_*` feature

---

## 当前主训练脚本真实依赖

按当前脚本实现，在线 finetune 主脚本要求以下路径存在：

- checkpoint：
  - `checkpoints/protenix_base_20250630_v1.0.0.pt`
- training CIF：
  - `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/PDB_RNA`
- prepared training data：
  - `.../part2/protenix_prepared/rna_bioassembly`
  - `.../part2/protenix_prepared/indices/rna_bioassembly_indices.csv`
  - `.../part2/protenix_prepared/rna_msa/rna_sequence_to_pdb_chains.json`
  - `rna_train_pdb_list_filtered.txt`
  - `rna_val_pdb_list_filtered.txt`
- RNA template metadata：
  - `/inspire/ssd/project/sais-bio/public/ash_proj/data/RNA3D/rna3db-jsons/filter.json`
- RNA template search results：
  - `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/search_results.json`

对应脚本：

- `finetune/finetune_rna_template_1stage.sh`
- `finetune/finetune_rna_template_2stage.sh`

其中 `search_results.json` 是 online mode 的关键输入；`.npz` 不是。

---

## 推荐顺序

### Step 0. 进入仓库并激活环境

```bash
cd /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate protenix
```

---

### Step 1. 新建 `rna_database` 目录

```bash
mkdir -p /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database
```

这一步只是为了让后续 catalog / search result 有地方写。

---

### Step 2. 从固定的 CIF template 库抽取 RNA catalog

如果你的 template CIF 库不变，按当前默认设置，通常使用：

- `/inspire/ssd/project/sais-bio/public/ash_proj/data/RNA3D/rna3db-mmcifs`

运行：

```bash
python rna_template/scripts/01_extract_rna_catalog.py \
  --pdb_rna_dir /inspire/ssd/project/sais-bio/public/ash_proj/data/RNA3D/rna3db-mmcifs \
  --output /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/rna_catalog.json \
  --max_structures 0 \
  --min_length 10 \
  --max_length 2000 \
  --num_workers 8
```

输出：

- `rna_database/rna_catalog.json`

作用：

- 为模板库中的每个 RNA chain 建立 `pdb_id -> chain_id -> sequence` 目录

---

### Step 3. 从固定 training data 构建 `search_results.json`

这一步是 **online finetune 的必需预处理**。

使用固定 training sequence 映射：

- `/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/protenix_prepared/rna_msa/rna_sequence_to_pdb_chains.json`

运行：

```bash
python rna_template/scripts/03_mmseqs2_search.py \
  --catalog /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/rna_catalog.json \
  --template_dir /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/templates \
  --training_sequences /inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/protenix_prepared/rna_msa/rna_sequence_to_pdb_chains.json \
  --output_index /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/rna_template_index.json \
  --output_search /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/search_results.json \
  --strategy mmseqs2 \
  --min_identity 0.3 \
  --max_templates 4 \
  --sensitivity 7.5 \
  --evalue 1e-3 \
  --num_threads 8
```

输出：

- `rna_database/search_results.json`
- `rna_database/rna_template_index.json`

其中：

- `search_results.json` 是 **online finetune 必需**
- `rna_template_index.json` 只是脚本顺带输出，online 主路径不依赖它

---

## 更省事的等价做法

如果你不想手动分 Step 2 和 Step 3，可以直接用主 pipeline 脚本，但**跳过 `.npz` 构建**：

```bash
bash rna_template/scripts/run_pipeline.sh --skip_build
```

这会做：

1. 提取 catalog
2. 运行 MMseqs2 搜索生成 `search_results.json`
3. 生成 index
4. **不会**构建离线 `.npz`

对当前 online finetune，这通常是最方便的入口。

---

## Step 4. 启动 online RNA template finetune

准备好 `rna_database/search_results.json` 之后，就可以直接跑主训练脚本。

### 1-stage

```bash
bash finetune/finetune_rna_template_1stage.sh
```

### 2-stage

```bash
bash finetune/finetune_rna_template_2stage.sh
```

这两个脚本当前会自动传入：

- `--rna_template.search_results_path`
- `--rna_template.cif_database_dir`
- `--rna_template.rna3db_metadata_path`

也就是会进入 **online mode**。

---

## 训练时真实发生了什么

训练时不会再去读预构建 `.npz`，而是对每个 sample 做：

1. 读取 sample 的 RNA sequence
2. 用 sequence 去 `search_results.json` 查 hit 列表
3. 根据 query 自身的 `pdb_id/release_date` 做：
   - self-hit exclusion
   - per-query temporal filtering
4. 对保留下来的 hit：
   - 找对应 CIF
   - 读结构残基
   - 在线计算 template feature
5. 生成 `rna_template_aatype`
6. 生成 `rna_template_distogram`
7. 生成 `rna_template_unit_vector`
8. 生成 `rna_template_backbone_frame_mask`
9. 写入模型输入，送进 `TemplateEmbedder`

所以你现在的 online 版本不是“训练时重新搜模板”，而是：

- 搜索结果预先算好
- 特征训练时现算

---

## `rna_template/compute` 在这个流程里的位置

online 训练时，真正参与 feature 计算的是：

- `load_structure_residues()`
- `build_minimal_template_arrays()`
- `compute_anchor()`
- `compute_frame()`
- `compute_distogram()`
- `compute_unit_vectors()`

它们来自：

- `rna_template/compute/rna_template_common.py`

调用链路是：

`finetune_rna_template_1stage.sh`
-> `runner/train.py`
-> `dataset.get_rna_template_featurizer()`
-> `BaseSingleDataset.process_one()`
-> `RNATemplateFeaturizer.__call__()`
-> `get_rna_template_features()`
-> `_build_single_template_online()`
-> `protenix/data/rna_template/rna_template_common_online.py`
-> `rna_template/compute/rna_template_common.py`

注意：

- 这条 online runtime 链路 **不调用 Arena**
- Arena 只在旧的离线 `.npz` 构建路径里是可选项

---

## 从零开始的最短可执行闭环

如果你只关心“最少步骤跑起来”，建议直接按下面顺序：

```bash
cd /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate protenix

mkdir -p rna_database

python rna_template/scripts/01_extract_rna_catalog.py \
  --pdb_rna_dir /inspire/ssd/project/sais-bio/public/ash_proj/data/RNA3D/rna3db-mmcifs \
  --output /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/rna_catalog.json \
  --max_structures 0 \
  --min_length 10 \
  --max_length 2000 \
  --num_workers 8

python rna_template/scripts/03_mmseqs2_search.py \
  --catalog /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/rna_catalog.json \
  --template_dir /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/templates \
  --training_sequences /inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/protenix_prepared/rna_msa/rna_sequence_to_pdb_chains.json \
  --output_index /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/rna_template_index.json \
  --output_search /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/search_results.json \
  --strategy mmseqs2 \
  --min_identity 0.3 \
  --max_templates 4 \
  --sensitivity 7.5 \
  --evalue 1e-3 \
  --num_threads 8

bash finetune/finetune_rna_template_1stage.sh
```

---

## 哪些东西不需要再做

如果你的目标是当前 online RNA template finetune，那么：

- 不需要先跑 `02_build_rna_templates.py`
- 不需要先生成 `.npz template`
- 不需要依赖 `rna_template_index.json`
- 不需要 Arena 参与在线特征计算

---

## 最终判断

从 **完全没有 `rna_database`** 的状态开始，只要：

1. CIF template 库不变
2. training prepared data 不变
3. 先建 `rna_catalog.json`
4. 再建 `search_results.json`

就已经足够把当前 **online RNA template finetune** 跑起来。

真正必需的是：

- `rna_database/search_results.json`
- `part2/PDB_RNA`
- `RNA3D/rna3db-jsons/filter.json`
- prepared training data

不是离线 `.npz`。

