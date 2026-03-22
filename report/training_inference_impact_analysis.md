# Protenix 修改项目 - 训练与推理影响分析报告

**审计日期:** 2026-03-07  
**分析类型:** 训练与推理影响深度分析  
**原始项目:** `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix`  
**修改项目:** `/inspire/ssd/project/sais-bio/public/ash_proj/code/Protenix`

---

## 执行摘要

### 影响评估总览

| 修改类别 | 对训练的影响 | 对推理的影响 | 风险等级 |
|----------|-------------|-------------|----------|
| `msa_featurizer.py` | ⚠️ 轻微影响 | ⚠️ 轻微影响 | 🟡 中 |
| `clash.py` | ⚠️ 中等影响 | ⚠️ 中等影响 | 🟡 中 |
| `pairformer.py` | ✅ 无负面影响 | ✅ 无负面影响 | 🟢 低 |
| `primitives.py` | ✅ 正面影响 | ✅ 正面影响 | 🟢 低 |
| `sample_confidence.py` | ⚠️ 中等影响 | ⚠️ 中等影响 | 🟡 中 |
| `torch_utils.py` | ❌ **严重 Bug** | ❌ **严重 Bug** | 🔴 高 |
| `training.py` | ⚠️ 配置依赖 | N/A | 🟡 中 |
| `train.py` | ✅ 正面影响 | N/A | 🟢 低 |

### 关键发现

1. **🔴 严重问题:** `torch_utils.py` 中的 `eye_mask` 函数缺少 `return` 语句，将导致使用此函数的所有训练和推理流程失败
2. **⚠️ 配置问题:** `training.py` 中添加了 `other_learning_rate` 参数，但配置文件中未定义此参数
3. **⚠️ API 变更:** 多处移除 `asym_id` 重映射逻辑，要求上游数据预处理保证 chain ID 连续

---

## 详细分析

### 1. `protenix/data/msa/msa_featurizer.py` 

**修改内容:** 条件加载 MSA 映射文件

```python
# 修改后
prot_mappings = (
    {i: load_json_cached(p) for i, p in enumerate(prot_seq_or_filename_to_msadir_jsons)}
    if enable_prot_msa
    else {}
)
```

#### 对训练的影响

| 影响项 | 评估 | 说明 |
|--------|------|------|
| 内存占用 | ✅ 降低 | 仅加载启用的 MSA 类型对应的映射文件 |
| 加载速度 | ✅ 提升 | 减少不必要的 JSON 文件读取 |
| 数据正确性 | ✅ 无影响 | 逻辑等价，仅优化加载时机 |
| 训练稳定性 | ✅ 无影响 | 不影响前向传播逻辑 |

**结论:** 🟢 **正面影响** - 优化了资源使用，不影响训练正确性

#### 对推理的影响

| 影响项 | 评估 | 说明 |
|--------|------|------|
| 推理速度 | ✅ 轻微提升 | 减少初始化时间 |
| 内存占用 | ✅ 降低 | 同训练 |
| 输出一致性 | ✅ 无影响 | 逻辑等价 |

**结论:** 🟢 **正面影响**

---

### 2. `protenix/metrics/clash.py`

**修改内容:** 移除 `asym_id` 自动重映射，改为断言验证

```python
# 修改前
unique_asym_ids = torch.unique(asym_id)
if len(unique_asym_ids) != asym_id.max() + 1:
    remap = {old.item(): new for new, old in enumerate(unique_asym_ids)}
    asym_id = torch.tensor(
        [remap[x.item()] for x in asym_id], dtype=torch.long, device=asym_id.device
    )

# 修改后
# Make sure it is from 0 to N_chain-1
assert N_chains == asym_id.max() + 1
```

#### 对训练的影响

| 影响项 | 评估 | 说明 |
|--------|------|------|
| 数据兼容性 | ⚠️ **降低** | 要求输入数据的 chain ID 必须是连续的 (0,1,2,...,N-1) |
| 错误检测 | ✅ 提升 | 更早发现数据预处理问题 |
| 训练流程 | ⚠️ **可能中断** | 如果数据预处理不保证连续 ID，将触发断言失败 |
| 评估指标 | ✅ 无影响 | clash 计算逻辑本身未变 |

**潜在问题场景:**
```python
# 如果数据预处理后 chain ID 为 [0, 2, 5] (有间隔)
# 修改前：自动重映射为 [0, 1, 2]
# 修改后：触发 AssertionError，训练中断
```

**结论:** 🟡 **需要验证** - 需确保数据预处理 pipeline 输出连续的 chain ID

#### 对推理的影响

| 影响项 | 评估 | 说明 |
|--------|------|------|
| 输入要求 | ⚠️ **更严格** | 同训练，要求连续 chain ID |
| 错误处理 | ⚠️ **更脆弱** | 非连续 ID 将导致推理失败而非自动修正 |
| 输出质量 | ✅ 无影响 | clash 计算逻辑未变 |

**结论:** 🟡 **需要验证** - 推理前需验证输入数据的 chain ID 格式

---

### 3. `protenix/model/modules/pairformer.py`

**修改内容:** 移除 `fused_ops` 依赖，改用标准 dropout + 加法

```python
# 修改前
from protenix.model.modules.fused_ops import dropout_add_rowwise
z = dropout_add_rowwise(z, tmu_update, self.p_drop, self.training)

# 修改后
z = z + self.dropout_row(tmu_update)
```

#### 对训练的影响

| 影响项 | 评估 | 说明 |
|--------|------|------|
| 数值等价性 | ✅ 等价 | `DropoutRowwise` + 加法与 fused 操作数学等价 |
| 训练速度 | ⚠️ 可能略降 | fused CUDA 操作通常比分离操作更快 |
| 内存占用 | ✅ 相似 | 无显著差异 |
| 梯度计算 | ✅ 正确 | 标准 PyTorch 操作梯度更稳定 |
| 可移植性 | ✅ **提升** | 不再依赖自定义 CUDA kernel |
| 编译兼容 | ✅ **提升** | 更易于 torch.compile 优化 |

**性能对比分析:**

```
fused_ops 版本:
  - 优点：单次 kernel 完成 dropout + add，减少内存访问
  - 缺点：需要自定义 CUDA 编译，兼容性差

标准版本:
  - 优点：标准 PyTorch 操作，兼容性好，易于调试
  - 缺点：两次内存访问 (dropout 输出 + add)
```

**结论:** 🟢 **可接受** - 牺牲少量性能换取更好的可维护性和兼容性

#### 对推理的影响

| 影响项 | 评估 | 说明 |
|--------|------|------|
| 输出一致性 | ✅ 等价 | 数学上等价的操作 |
| 推理速度 | ⚠️ 可能略降 | 同训练 |
| 部署便利 | ✅ **提升** | 无需编译自定义 CUDA 扩展 |

**结论:** 🟢 **正面影响** - 更易于部署

---

### 4. `protenix/model/modules/primitives.py`

**修改内容:** 修复混合精度训练中的精度转换

```python
# 修改前
input_dtype = q.dtype
q = q.to(dtype=torch.float32)
k = k.to(dtype=torch.float32)
# v 未转换
# ...
return attn_output  # 未转换回 input_dtype
# ...
attn_output = attn_weights.to(dtype=input_dtype) @ v  # 先转换再计算

# 修改后
input_dtype = q.dtype
q = q.to(dtype=torch.float32)
k = k.to(dtype=torch.float32)
v = v.to(dtype=torch.float32)  # ✅ 添加
# ...
return attn_output.to(dtype=input_dtype)  # ✅ 添加
# ...
attn_output = (attn_weights @ v).to(dtype=input_dtype)  # ✅ 先计算再转换
```

#### 对训练的影响

| 影响项 | 评估 | 说明 |
|--------|------|------|
| 数值稳定性 | ✅ **显著提升** | 所有输入统一为 float32，避免精度不匹配 |
| 梯度正确性 | ✅ **提升** | 矩阵乘法在 float32 下进行，减少溢出风险 |
| 训练收敛 | ✅ **可能改善** | 更稳定的数值计算有助于收敛 |
| 内存占用 | ⚠️ 轻微增加 | v 需要额外的 float32 缓存 |
| 计算速度 | ⚠️ 轻微影响 | 额外的类型转换开销 |

**修复的问题:**
```python
# 修改前的问题场景 (BF16 训练):
q: BF16 -> F32 ✓
k: BF16 -> F32 ✓
v: BF16 (未转换) ✗  -> 与 F32 的 attn_weights 相乘时精度不匹配

# 修改后:
q, k, v: 全部 -> F32 ✓
attn_output: F32 -> BF16 ✓  (保持与输入一致的精度)
```

**结论:** 🟢 **重要修复** - 解决了混合精度训练的数值稳定性问题

#### 对推理的影响

| 影响项 | 评估 | 说明 |
|--------|------|------|
| 输出精度 | ✅ **提升** | 输出精度与输入保持一致 |
| 数值稳定性 | ✅ **提升** | 同训练 |
| 推理速度 | ⚠️ 轻微影响 | 可忽略的类型转换开销 |

**结论:** 🟢 **重要修复**

---

### 5. `protenix/model/sample_confidence.py`

**修改内容:** 移除 `asym_id` 自动重映射 (3 处)

```python
# 修改前 (3 处相同模式)
unique_asym_ids = torch.unique(asym_id)
if len(unique_asym_ids) != asym_id.max() + 1:
    remap = {old.item(): new for new, old in enumerate(unique_asym_ids)}
    asym_id = torch.tensor(
        [remap[x.item()] for x in asym_id], dtype=torch.long, device=asym_id.device
    )

# 修改后
# 第 1 处：直接移除重映射逻辑
# 第 2 处：改为断言
assert N_chain == asym_id.max() + 1
# 第 3 处：直接移除重映射逻辑，但保留断言
```

#### 对训练的影响

| 影响项 | 评估 | 说明 |
|--------|------|------|
| 数据兼容性 | ⚠️ **降低** | 同 clash.py，要求连续 chain ID |
| 评估指标计算 | ⚠️ **可能失败** | 非连续 ID 将触发断言 |
| 错误定位 | ✅ 提升 | 更早暴露数据问题 |

**影响的函数:**
1. `compute_tm_score` (约 498 行) - 移除重映射
2. `compute_pde_metrics` (约 614 行) - 改为断言
3. `compute_plddt_metrics` (约 679 行) - 移除重映射

**结论:** 🟡 **需要验证** - 与 clash.py 相同的问题

#### 对推理的影响

| 影响项 | 评估 | 说明 |
|--------|------|------|
| 置信度评估 | ⚠️ **可能失败** | 非连续 chain ID 导致推理后评估中断 |
| 输出文件 | ⚠️ **可能不完整** | 评估失败可能导致部分指标缺失 |

**结论:** 🟡 **需要验证**

---

### 6. `protenix/utils/torch_utils.py` ⚠️

**修改内容:** `eye_mask` 函数缺少 `return` 语句

```python
# 修改前
def eye_mask(L, device=None, opposite=False):
    if opposite:
        return 1.0 - torch.eye(L, device=device)
    else:
        return torch.eye(L, device=device)  # ✅

# 修改后
def eye_mask(L, device=None, opposite=False):
    if opposite:
        return 1.0 - torch.eye(L, device=device)
    else:
        torch.eye(L, device=device)  # ❌ 缺少 return
```

#### 对训练的影响

| 影响项 | 评估 | 说明 |
|--------|------|------|
| 功能正确性 | ❌ **严重破坏** | `opposite=False` 时返回 `None` |
| 训练流程 | ❌ **必然失败** | 任何使用此函数的地方都会出错 |
| 错误传播 | ❌ 难以定位 | 可能在下游操作中报出 confusing 的错误 |

**影响范围分析:**

```bash
# 搜索结果显示：eye_mask 在代码库中未被直接调用
# 但这是一个公共工具函数，未来可能被使用
# 且这是一个明显的 Bug，应修复
```

**潜在影响代码:**
```python
# 任何这样的调用都会失败:
mask = eye_mask(10, device='cuda', opposite=False)
# mask = None ❌

# 后续使用 mask 的操作会失败:
result = some_tensor * mask  # TypeError: unsupported operand type(s) for *: 'Tensor' and 'NoneType'
```

**结论:** 🔴 **严重 Bug** - 必须修复

#### 对推理的影响

| 影响项 | 评估 | 说明 |
|--------|------|------|
| 功能正确性 | ❌ **严重破坏** | 同训练 |
| 推理流程 | ❌ **可能失败** | 如果推理代码使用此函数 |

**结论:** 🔴 **严重 Bug** - 必须修复

---

### 7. `protenix/utils/training.py`

**修改内容:** 添加 `other_learning_rate` 参数

```python
# 修改前
if param_names is None or len(param_names) == 0 or param_names[0] == "":
    param_names = None
if configs.adam.use_adamw:
    optimizer = get_adamw(
        model=model,
        weight_decay=configs.adam.weight_decay,
        learning_rate=configs.adam.lr,
        betas=(configs.adam.beta1, configs.adam.beta2),
        device_type="cuda" if torch.cuda.is_available() else "cpu",
    )

# 修改后
if len(param_names) == 0 or param_names[0] == "":  # 移除了 param_names is None 检查
    param_names = None
if configs.adam.use_adamw:
    optimizer = get_adamw(
        model=model,
        weight_decay=configs.adam.weight_decay,
        learning_rate=configs.adam.lr,
        other_learning_rate=configs.other_lr,  # ✅ 新增
        betas=(configs.adam.beta1, configs.adam.beta2),
        device_type="cuda" if torch.cuda.is_available() else "cpu",
    )
```

#### 对训练的影响

| 影响项 | 评估 | 说明 |
|--------|------|------|
| 配置依赖 | ⚠️ **新增依赖** | 需要 `configs.other_lr` 参数 |
| 参数检查 | ⚠️ **潜在问题** | 移除了 `param_names is None` 检查 |
| 优化器功能 | ⚠️ **未实现** | `get_adamw` 函数签名未更新 |

**问题分析:**

1. **配置参数缺失:**
   ```bash
   # 搜索结果显示 configs 目录中没有 other_lr 定义
   # 这会导致训练启动时 AttributeError
   ```

2. **函数签名不匹配:**
   ```python
   # get_adamw 函数定义 (未修改):
   def get_adamw(
       model: torch.nn.Module,
       weight_decay: float,
       learning_rate: float,
       betas: tuple[float, float],
       device_type: str,
   ) -> torch.optim.AdamW:
   
   # 调用时传入了未定义的参数:
   optimizer = get_adamw(
       ...,
       other_learning_rate=configs.other_lr,  # ❌ TypeError
   )
   ```

3. **空值检查变更:**
   ```python
   # 修改前: if param_names is None or len(param_names) == 0 or param_names[0] == "":
   # 修改后: if len(param_names) == 0 or param_names[0] == "":
   
   # 如果 param_names = None:
   # 修改前: 安全，进入 if 分支
   # 修改后: TypeError: object of type 'NoneType' has no len()
   ```

**结论:** 🟡 **配置不完整** - 需要补充配置和更新函数签名

#### 对推理的影响

| 影响项 | 评估 | 说明 |
|--------|------|------|
| 推理流程 | ✅ 无影响 | 此文件仅用于训练 |

**结论:** ✅ **不影响推理**

---

### 8. `runner/train.py`

**修改内容:** wandb 可选导入

```python
# 修改前
import wandb
# ...
wandb.init(...)

# 修改后
try:
    import wandb
except ImportError:
    wandb = None
# ...
if wandb is None:
    logging.warning("WANDB is not installed...")
else:
    wandb.init(...)
```

#### 对训练的影响

| 影响项 | 评估 | 说明 |
|--------|------|------|
| 容错能力 | ✅ **提升** | wandb 未安装时不崩溃 |
| 日志功能 | ✅ 降级处理 | 未安装时仅输出警告 |
| 训练核心 | ✅ 无影响 | 仅影响日志记录 |
| 部署便利 | ✅ **提升** | 减少依赖要求 |

**结论:** 🟢 **正面影响**

#### 对推理的影响

| 影响项 | 评估 | 说明 |
|--------|------|------|
| 推理流程 | ✅ 无影响 | 此文件仅用于训练 |

**结论:** ✅ **不影响推理**

---

## 综合影响评估

### 训练流程影响

```
训练启动流程:
├── 环境初始化 (train.py)          ✅ 改进 - wandb 可选
├── 配置加载 (training.py)         ⚠️ 风险 - other_lr 未定义
├── 优化器创建 (training.py)       ⚠️ 风险 - 函数签名不匹配
├── 数据加载 (msa_featurizer.py)   ✅ 改进 - 条件加载
├── 前向传播 (pairformer.py)       ✅ 可接受 - 标准操作
├── 前向传播 (primitives.py)       ✅ 改进 - 精度修复
├── 损失计算 (clash.py)            ⚠️ 风险 - 断言可能失败
├── 评估指标 (sample_confidence.py) ⚠️ 风险 - 断言可能失败
└── 工具函数 (torch_utils.py)      ❌ Bug - eye_mask 返回 None
```

### 推理流程影响

```
推理流程:
├── 模型加载                       ✅ 无影响
├── 数据预处理 (msa_featurizer.py) ✅ 改进
├── 前向传播 (pairformer.py)       ✅ 可接受
├── 前向传播 (primitives.py)       ✅ 改进
├── 置信度评估 (sample_confidence.py) ⚠️ 风险 - 断言可能失败
├── Clash 检测 (clash.py)          ⚠️ 风险 - 断言可能失败
└── 工具函数 (torch_utils.py)      ❌ Bug - 如被调用将失败
```

---

## 风险等级汇总

| 风险等级 | 文件 | 问题描述 | 建议操作 |
|----------|------|----------|----------|
| 🔴 高 | `torch_utils.py` | `eye_mask` 缺少 return | **立即修复** |
| 🔴 高 | `training.py` | `other_lr` 配置缺失 + 函数签名不匹配 | **立即修复** |
| 🟡 中 | `clash.py` | 断言要求连续 chain ID | 验证数据预处理 |
| 🟡 中 | `sample_confidence.py` | 同 clash.py | 验证数据预处理 |
| 🟢 低 | `pairformer.py` | 移除 fused ops | 可接受，性能影响小 |
| 🟢 低 | `msa_featurizer.py` | 条件加载优化 | 正面改进 |
| 🟢 低 | `primitives.py` | 精度修复 | 正面改进 |
| 🟢 低 | `train.py` | wandb 可选导入 | 正面改进 |

---

## 修复建议

### 1. 紧急修复 (必须)

#### 1.1 `torch_utils.py`
```python
def eye_mask(L, device=None, opposite=False):
    if opposite:
        return 1.0 - torch.eye(L, device=device)
    else:
        return torch.eye(L, device=device)  # 添加 return
```

#### 1.2 `training.py`
```python
# 方案 A: 如果不需要 other_lr 功能
optimizer = get_adamw(
    model=model,
    weight_decay=configs.adam.weight_decay,
    learning_rate=configs.adam.lr,
    # 移除 other_learning_rate 参数
    betas=(configs.adam.beta1, configs.adam.beta2),
    device_type="cuda" if torch.cuda.is_available() else "cpu",
)

# 方案 B: 如果需要 other_lr 功能
# Step 1: 在 configs_base.py 中添加配置
# Step 2: 更新 get_adamw 函数签名
def get_adamw(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    device_type: str,
    other_learning_rate: Optional[float] = None,  # 新增
) -> torch.optim.AdamW:
```

#### 1.3 `training.py` 空值检查
```python
# 恢复 None 检查
if param_names is None or len(param_names) == 0 or param_names[0] == "":
    param_names = None
```

### 2. 数据验证建议

在数据预处理 pipeline 中添加 chain ID 连续性验证:

```python
def validate_chain_ids(asym_id):
    unique_ids = torch.unique(asym_id)
    expected_ids = torch.arange(len(unique_ids), device=asym_id.device)
    if not torch.equal(unique_ids, expected_ids):
        raise ValueError(
            f"Chain IDs must be contiguous from 0 to N-1. "
            f"Got: {unique_ids.tolist()}"
        )
```

### 3. 性能测试建议

由于移除了 `fused_ops`，建议进行性能对比测试:

```bash
# 训练速度对比
# - 原始版本 (with fused_ops)
# - 修改版本 (standard ops)

# 预期: 修改版本可能慢 5-15%，但兼容性和可维护性更好
```

---

## 最终结论

### 是否影响训练？

**是的，有影响:**

1. **阻断性问题 (2 个):**
   - `torch_utils.py` 的 `eye_mask` Bug 将导致任何使用此函数的代码失败
   - `training.py` 的配置和函数签名不匹配将导致优化器创建失败

2. **潜在问题 (2 个):**
   - `clash.py` 和 `sample_confidence.py` 的断言可能因数据格式问题中断训练
   - 需要验证数据预处理 pipeline 是否保证连续 chain ID

3. **正面改进 (4 个):**
   - `primitives.py` 修复了混合精度训练问题
   - `train.py` 提升了容错能力
   - `msa_featurizer.py` 优化了资源使用
   - `pairformer.py` 提升了可移植性

### 是否影响推理？

**是的，有影响:**

1. **阻断性问题 (1 个):**
   - `torch_utils.py` 的 `eye_mask` Bug (如果被推理代码调用)

2. **潜在问题 (2 个):**
   - `clash.py` 和 `sample_confidence.py` 的断言可能中断推理后评估
   - 需要确保输入数据的 chain ID 格式正确

3. **正面改进 (3 个):**
   - `primitives.py` 提升了数值稳定性
   - `pairformer.py` 更易于部署 (无需 CUDA 扩展)
   - `msa_featurizer.py` 减少了初始化时间

### 总体评估

| 评估维度 | 评分 | 说明 |
|----------|------|------|
| 代码质量 | 🟡 中 | 有重要 Bug 需要修复 |
| 功能完整性 | 🟡 中 | 配置不完整 |
| 数值正确性 | 🟢 好 | primitives.py 修复了精度问题 |
| 可维护性 | 🟢 好 | 移除 fused ops 提升可移植性 |
| 数据兼容性 | 🟡 中 | 需要验证 chain ID 格式 |

### 建议操作顺序

1. **立即修复** `torch_utils.py` 的 return Bug (5 分钟)
2. **立即修复** `training.py` 的配置和签名问题 (10 分钟)
3. **验证** 数据预处理 pipeline 的 chain ID 格式 (30 分钟)
4. **测试** 训练流程是否正常运行 (1-2 小时)
5. **测试** 推理流程是否正常运行 (1 小时)
6. **性能对比** 评估 fused ops 移除的影响 (可选)

---

**报告生成完成**  
**下一步:** 修复紧急 Bug 后进行完整的训练和推理测试
