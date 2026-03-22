# Protenix 项目代码审计报告

**审计日期:** 2026-03-07  
**审计类型:** 代码差异比较分析  
**原始项目:** `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix`  
**修改项目:** `/inspire/ssd/project/sais-bio/public/ash_proj/code/Protenix`

---

## 执行摘要

本次审计比较了原始 Protenix 项目与修改后的项目，识别出 **8 个被修改的核心文件** 和 **多个新增文件/目录**。修改主要集中在 MSA 特征处理、模型模块、训练工具和评估指标等方面。

---

## 一、修改的文件列表

### 1.1 核心代码修改文件 (8 个)

| 序号 | 文件路径 | 修改类型 | 影响范围 |
|------|----------|----------|----------|
| 1 | `protenix/data/msa/msa_featurizer.py` | 逻辑优化 | MSA 数据处理 |
| 2 | `protenix/metrics/clash.py` | 断言添加 | 冲突检测 |
| 3 | `protenix/model/modules/pairformer.py` | 重构 | 核心模型模块 |
| 4 | `protenix/model/modules/primitives.py` | 精度修复 | 基础运算单元 |
| 5 | `protenix/model/sample_confidence.py` | 逻辑简化 | 置信度采样 |
| 6 | `protenix/utils/torch_utils.py` | Bug 修复 | 工具函数 |
| 7 | `protenix/utils/training.py` | 功能增强 | 训练优化器 |
| 8 | `runner/train.py` | 容错处理 | 训练入口 |

---

## 二、详细修改分析

### 2.1 `protenix/data/msa/msa_featurizer.py`

**修改位置:** `MSAMultiFeaturizer.__init__` 方法 (约 401-430 行)

**修改内容:**
```diff
- self.prot_mgr = MSASourceManager(
-     prot_msadir_raw_paths,
-     prot_indexing_methods,
-     {
-         i: load_json_cached(p)
-         for i, p in enumerate(prot_seq_or_filename_to_msadir_jsons)
-     },
-     enable_prot_msa,
- )
+ # Only load JSON mappings if the corresponding MSA type is enabled
+ prot_mappings = (
+     {
+         i: load_json_cached(p)
+         for i, p in enumerate(prot_seq_or_filename_to_msadir_jsons)
+     }
+     if enable_prot_msa
+     else {}
+ )
+ rna_mappings = (
+     {
+         i: load_json_cached(p)
+         for i, p in enumerate(rna_seq_or_filename_to_msadir_jsons)
+     }
+     if enable_rna_msa
+     else {}
+ )
+ self.prot_mgr = MSASourceManager(
+     prot_msadir_raw_paths,
+     prot_indexing_methods,
+     prot_mappings,
+     enable_prot_msa,
+ )
```

**修改说明:**
- **优化点:** 添加了条件加载逻辑，仅在对应的 MSA 类型启用时才加载 JSON 映射文件
- **影响:** 减少不必要的内存占用和加载时间，支持更灵活的 MSA 配置

---

### 2.2 `protenix/metrics/clash.py`

**修改位置:** 冲突检测函数 (约 85-100 行)

**修改内容:**
```diff
- unique_asym_ids = torch.unique(asym_id)
- if len(unique_asym_ids) != asym_id.max() + 1:
-     remap = {old.item(): new for new, old in enumerate(unique_asym_ids)}
-     asym_id = torch.tensor(
-         [remap[x.item()] for x in asym_id], dtype=torch.long, device=asym_id.device
-     )
  asym_id_to_asym_mask = {
      aid.item(): asym_id == aid for aid in torch.unique(asym_id)
  }
  N_chains = len(asym_id_to_asym_mask)
+ # Make sure it is from 0 to N_chain-1
+ assert N_chains == asym_id.max() + 1
```

**修改说明:**
- **变更:** 移除了自动重映射逻辑，改为使用断言验证
- **影响:** 调用方需确保传入的 `asym_id` 是连续的 (0 到 N-1)，否则将触发断言错误

---

### 2.3 `protenix/model/modules/pairformer.py`

**修改位置:** 多处 (约 23-490 行)

**主要修改:**

#### 2.3.1 移除 fused_ops 依赖
```diff
- from protenix.model.modules.fused_ops import dropout_add_rowwise
  from protenix.model.triangular.layers import DropoutRowwise, LayerNorm, OuterProductMean
```

#### 2.3.2 移除 p_drop 属性
```diff
  self.dropout_row = DropoutRowwise(dropout)
- self.p_drop = dropout
  self.pair_transition = Transition(c_in=c_z, n=num_intermediate_factor)
```

#### 2.3.3 简化 dropout 调用
```diff
- z = dropout_add_rowwise(z, tmu_update, self.p_drop, self.training)
+ z = z + self.dropout_row(tmu_update)
```

#### 2.3.4 MSA 模块修改
```diff
  msa_pair_weighted = self.chunk_forward(
      self.msa_pair_weighted_averaging, m_new, z, chunk_size
  )
- m = dropout_add_rowwise(m, msa_pair_weighted[: m.shape[-3], :, :], self.p_drop, self.training)
+ m = m + self.dropout_row(msa_pair_weighted[: m.shape[-3], :, :])
```

**修改说明:**
- **重构:** 移除了对 `fused_ops.py` 的依赖，改用标准的 dropout + 加法操作
- **影响:** 降低了对自定义 CUDA 操作的依赖，提高代码可移植性

---

### 2.4 `protenix/model/modules/primitives.py`

**修改位置:** 注意力机制函数 (约 256-290 行)

**修改内容:**
```diff
  input_dtype = q.dtype
  q = q.to(dtype=torch.float32)
  k = k.to(dtype=torch.float32)
+ v = v.to(dtype=torch.float32)
  if attn_bias is not None:
      attn_bias = attn_bias.to(dtype=torch.float32)
  
- return attn_output
+ return attn_output.to(dtype=input_dtype)
  
  # ...
- attn_output = attn_weights.to(dtype=input_dtype) @ v
+ attn_output = (attn_weights @ v).to(dtype=input_dtype)
```

**修改说明:**
- **修复:** 添加了 `v` 的 float32 转换，确保所有输入张量精度一致
- **修复:** 添加返回值的精度转换，保持输入输出精度一致
- **影响:** 修复了混合精度训练时的数值稳定性问题

---

### 2.5 `protenix/model/sample_confidence.py`

**修改位置:** 多处 (约 498-680 行)

**修改内容:**
```diff
  # 位置 1 (约 498 行)
- unique_asym_ids = torch.unique(asym_id)
- if len(unique_asym_ids) != asym_id.max() + 1:
-     remap = {old.item(): new for new, old in enumerate(unique_asym_ids)}
-     asym_id = torch.tensor(
-         [remap[x.item()] for x in asym_id], dtype=torch.long, device=asym_id.device
-     )
  asym_id_to_asym_mask = {aid.item(): asym_id == aid for aid in torch.unique(asym_id)}

  # 位置 2 (约 614 行)
- if N_chain != asym_id.max() + 1:
-     # asym_id has gaps (chains were filtered out); remap to contiguous 0..N_chain-1
-     remap = {old.item(): new for new, old in enumerate(unique_asym_ids)}
-     asym_id = torch.tensor(
-         [remap[x.item()] for x in asym_id], dtype=torch.long, device=asym_id.device
-     )
+ assert N_chain == asym_id.max() + 1  # make sure it is from 0 to N_chain-1
```

**修改说明:**
- **简化:** 移除了多处 `asym_id` 重映射逻辑
- **影响:** 与 `clash.py` 类似，调用方需确保传入连续的 chain ID

---

### 2.6 `protenix/utils/torch_utils.py`

**修改位置:** `get_2d_mask` 函数 (约 95-98 行)

**修改内容:**
```diff
  if opposite:
      return 1.0 - torch.eye(L, device=device)
  else:
-     return torch.eye(L, device=device)
+     torch.eye(L, device=device)
```

**修改说明:**
- **Bug:** 移除了 `return` 语句，导致函数在非 opposite 模式下返回 `None`
- **注意:** ⚠️ 这是一个潜在的 Bug，可能导致调用方出错

---

### 2.7 `protenix/utils/training.py`

**修改位置:** 优化器创建函数 (约 74-90 行)

**修改内容:**
```diff
- if param_names is None or len(param_names) == 0 or param_names[0] == "":
+ if len(param_names) == 0 or param_names[0] == "":
      param_names = None
  if configs.adam.use_adamw:
      optimizer = get_adamw(
          model=model,
          weight_decay=configs.adam.weight_decay,
          learning_rate=configs.adam.lr,
+         other_learning_rate=configs.other_lr,
          betas=(configs.adam.beta1, configs.adam.beta2),
          device_type="cuda" if torch.cuda.is_available() else "cpu",
      )
```

**修改说明:**
- **变更:** 修改了 `param_names` 的空值判断逻辑
- **增强:** 添加了 `other_learning_rate` 参数传递，支持差异化学习率

---

### 2.8 `runner/train.py`

**修改位置:** 导入和初始化部分 (约 23-135 行)

**修改内容:**
```diff
- import wandb
+ # Optional wandb import - only required if use_wandb is True
+ try:
+     import wandb
+ except ImportError:
+     wandb = None

  # ...
  
  if self.configs.use_wandb and DIST_WRAPPER.rank == 0:
-     wandb.init(
-         project=self.configs.project,
-         name=self.run_name,
-         config=vars(self.configs),
-         id=self.configs.wandb_id or None,
-     )
+     if wandb is None:
+         logging.warning("WANDB is not installed. Install with 'pip install wandb' to enable logging.")
+     else:
+         wandb.init(
+             project=self.configs.project,
+             name=self.run_name,
+             config=vars(self.configs),
+             id=self.configs.wandb_id or None,
+         )
```

**修改说明:**
- **增强:** 添加了 wandb 的可选导入，未安装时不会导致程序崩溃
- **影响:** 提高了代码的健壮性和可移植性

---

## 三、修改项目独有文件/目录

### 3.1 新增文件 (非 __pycache__ 和构建产物)

| 文件/目录 | 类型 | 用途推测 |
|-----------|------|----------|
| `FINAL_REPORT.md` | 文档 | 项目最终报告 |
| `RNA_FINETUNE_GUIDE.md` | 文档 | RNA 微调指南 |
| `analyze_dataset.py` | 脚本 | 数据集分析工具 |
| `compare_part1_part2.py` | 脚本 | 比较工具 |
| `finetune_rna.sh` | 脚本 | RNA 微调启动脚本 |
| `finetune_rna_test.sh` | 脚本 | RNA 微调测试脚本 |
| `run_example.sh` | 脚本 | 示例运行脚本 |
| `start_training.sh` | 脚本 | 训练启动脚本 |
| `stanford_rna_finetune_list.txt` | 数据 | Stanford RNA 微调列表 |
| `verify_environment.sh` | 脚本 | 环境验证脚本 |

### 3.2 新增目录

| 目录 | 用途推测 |
|------|----------|
| `checkpoints/` | 模型检查点存储 |
| `claude/` | Claude 相关代码/配置 |
| `claude_new/` | Claude 新版本代码/配置 |
| `common/` | 公共模块 |
| `output/` | 输出目录 |
| `qwen/` | Qwen 相关代码/配置 |
| `qwen_rna_finetune/` | Qwen RNA 微调 |
| `report/` | 报告目录 |

### 3.3 修改项目独有但原始项目也存在的文件

| 文件 | 状态 |
|------|------|
| `protenix/model/modules/fused_ops.py` | **仅原始项目存在**，修改项目已移除依赖 |

---

## 四、修改统计

| 类别 | 数量 |
|------|------|
| 修改的核心文件 | 8 |
| 新增文件 (不含缓存/构建产物) | 10+ |
| 新增目录 | 8 |
| 移除依赖的模块 | 1 (`fused_ops.py`) |

---

## 五、审计发现与建议

### 5.1 积极改进

1. ✅ **性能优化:** `msa_featurizer.py` 的条件加载减少了不必要的资源消耗
2. ✅ **可移植性提升:** 移除 `fused_ops.py` 依赖，降低对自定义 CUDA 操作的依赖
3. ✅ **数值稳定性:** `primitives.py` 的精度修复改善了混合精度训练
4. ✅ **健壮性增强:** `train.py` 的可选 wandb 导入提高了容错能力
5. ✅ **功能扩展:** 新增 RNA 微调相关脚本和文档

### 5.2 潜在问题

1. ⚠️ **Bug 引入:** `torch_utils.py` 中 `get_2d_mask` 函数移除了 `return` 语句
2. ⚠️ **API 变更:** 多处移除 `asym_id` 重映射逻辑，要求调用方保证输入连续性
3. ⚠️ **文档缺失:** 新增的 RNA 微调功能缺少详细的 API 文档

### 5.3 建议

1. **修复 Bug:** 检查 `torch_utils.py` 的 `get_2d_mask` 函数是否为有意修改
2. **更新文档:** 为 RNA 微调功能补充使用文档和示例
3. **回归测试:** 对修改的核心模块进行完整的回归测试
4. **API 文档:** 记录 `asym_id` 连续性要求的变更

---

## 六、附录

### 6.1 命令参考

本次审计使用的比较命令:
```bash
diff -rq /path/to/original /path/to/modified
diff -u /path/to/original/file /path/to/modified/file
```

### 6.2 文件完整性检查

| 检查项 | 状态 |
|--------|------|
| 原始项目可访问 | ✅ |
| 修改项目可访问 | ✅ |
| 核心文件完整性 | ✅ |
| 无未授权修改 | ✅ (仅审计，未修改) |

---

**报告生成完成**  
**审计结论:** 修改项目相比原始项目进行了 8 处核心代码修改，主要聚焦于性能优化、可移植性提升和 RNA 微调功能扩展。发现 1 处潜在 Bug 需进一步确认。
