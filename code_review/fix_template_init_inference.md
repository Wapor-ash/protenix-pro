# RNA Template Bug Fix Report: Projector Init & Inference Validation

**日期:** 2026-03-14
**范围:** 修复 RNA template projector 初始化/恢复 bug + inference projector 权重校验
**状态:** 全部修复完成，14/14 测试通过

---

## 1. 问题总览

| Bug ID | 严重级别 | 问题 | 修复状态 |
|--------|---------|------|---------|
| Bug 1 | **Medium** | `copy_prot_projector_after_load=True` 在恢复训练时会覆盖已训练的 RNA projector | ✅ 已修复 |
| Bug 1 扩展 | **Feature** | 需要提供 projector 初始化选项：zero-init 或 protein-copy | ✅ 已实现 |
| Bug 1 扩展 | **Feature** | 如果 checkpoint 已包含 RNA projector 权重，直接使用 | ✅ 已实现 |
| Inference | **High** | Inference 时 projector 权重缺失仅 warning，未 raise error | ✅ 已修复 |
| Bug 3 | **Low** | 脚本 WARNING 与代码 fail-fast 语义不一致 | ✅ 已修复 |

---

## 2. Bug 1 (Medium): projector 初始化/恢复逻辑重构

### 2.1 问题根因

旧实现中 `copy_prot_projector_after_load=True` 默认值导致：
- **冷启动场景正确**：从 protein-only checkpoint 开始 RNA finetune 时，会将 protein projector 权重拷贝到 RNA projector。
- **恢复场景错误**：从已包含 RNA projector 权重的 checkpoint 恢复训练时，`reinit_rna_projector_from_protein()` 会无条件覆盖已训练好的 RNA projector，导致 silent clobber。

旧代码：
```python
# protenix/model/protenix.py — 旧实现
def reinit_rna_projector_from_protein(self):
    te = self.template_embedder
    if not getattr(te, "rna_template_enable", False):
        return
    # 总是覆盖！不管 checkpoint 里是否有 RNA projector 权重
    with torch.no_grad():
        te.linear_no_bias_a_rna.weight.copy_(te.linear_no_bias_a.weight)

# runner/train.py — 旧实现
rna_cfg = self.configs.get("rna_template", {})
if rna_cfg.get("enable", False) and rna_cfg.get("copy_prot_projector_after_load", True):
    self.raw_model.reinit_rna_projector_from_protein()  # 无条件覆盖
```

### 2.2 设计方案

#### 新 config：`projector_init` 替代 `copy_prot_projector_after_load`

```python
"rna_template": {
    "projector_init": "protein",  # "protein" 或 "zero"
    "alpha_init": 0.01,           # 仅在 projector_init="protein" 时使用
}
```

| `projector_init` | 初始化方式 | alpha gate | 适用场景 |
|------------------|-----------|------------|---------|
| `"protein"` | 从 protein projector 拷贝权重 | 有（learnable scalar, init=0.01） | 迁移学习：RNA projector 从 protein 路径的已学习表示出发 |
| `"zero"` | 零初始化 | 无（直接加到 pairwise，和 protein 一样） | 从零开始训练 RNA path，类似 RNALM projector 的 zero-init 模式 |

#### 核心逻辑：Smart Checkpoint Detection

不管选 `"protein"` 还是 `"zero"`，如果 checkpoint 已包含 RNA projector 权重（key: `template_embedder.linear_no_bias_a_rna.weight`），`load_state_dict` 已经加载了正确的权重 → **不做任何 reinit**。

只有当 checkpoint 中 **没有** RNA projector 权重时（protein-only checkpoint），才按 `projector_init` 策略进行初始化。

### 2.3 修复代码

#### 文件 1: `configs/configs_base.py`

**删除** `copy_prot_projector_after_load`、`gate_mode`、`gate_init_logit`。
**新增** `projector_init`。

```python
# 修改后
"rna_template": {
    "enable": False,
    "template_database_dir": "",
    "template_index_path": "",
    "max_rna_templates": 4,
    "injection_mode": "z_init",
    # projector_init: 如何初始化 RNA template projector
    #   "protein" — 从 protein projector 拷贝权重 + 使用 learnable alpha gate
    #   "zero"    — 零初始化，直接加到 pairwise（无 alpha gate）
    # 不管选哪个，如果 checkpoint 已包含 RNA projector 权重，直接使用。
    "projector_init": "protein",
    "alpha_init": 0.01,  # alpha gate 初始值（仅 projector_init="protein" 时使用）
},
```

#### 文件 2: `protenix/model/modules/pairformer.py` — TemplateEmbedder

**构造函数** — 根据 `projector_init` 选择初始化策略：

```python
# 修改后
rna_cfg = rna_template_configs or {}
self.rna_template_enable = rna_cfg.get("enable", False)
self.rna_projector_init = rna_cfg.get("projector_init", "protein")
if self.rna_template_enable:
    rna_input_dim = sum(self.input_feature1.values()) + sum(self.input_feature2.values())
    self.linear_no_bias_a_rna = LinearNoBias(
        in_features=rna_input_dim, out_features=self.c,
    )

    if self.rna_projector_init == "zero":
        # Zero-init: RNA projector 从零开始，无 alpha gate
        nn.init.zeros_(self.linear_no_bias_a_rna.weight)
    else:
        # Protein-init: 从 protein projector 拷贝 + learnable alpha gate
        with torch.no_grad():
            self.linear_no_bias_a_rna.weight.copy_(self.linear_no_bias_a.weight)
        alpha_init = rna_cfg.get("alpha_init", 0.01)
        self.rna_template_alpha = nn.Parameter(torch.tensor(float(alpha_init)))
```

**forward 方法** — 根据模式选择是否使用 alpha gate：

```python
# 修改后
# Scale RNA template contribution:
# - zero mode: 直接加（projector 是 zero-init 的，不需要 gate）
# - protein mode: 乘以 learnable alpha
if self.rna_projector_init == "zero" or not hasattr(self, "rna_template_alpha"):
    u = u + rna_v
else:
    u = u + self.rna_template_alpha * rna_v
```

#### 文件 3: `protenix/model/protenix.py` — Smart Checkpoint Detection

```python
# 修改后
def reinit_rna_projector_from_protein(self, checkpoint_keys=None) -> str:
    """条件性重新初始化 RNA projector。

    Smart detection 逻辑：
    1. checkpoint 已包含 RNA projector 权重 → 不做任何事（load_state_dict 已加载）
    2. checkpoint 不包含 RNA projector 权重 → 按 projector_init 策略初始化

    Args:
        checkpoint_keys: checkpoint 中的 key 集合，用于检测是否包含 RNA projector 权重

    Returns:
        str: "loaded_from_checkpoint" | "copied_from_protein" | "zero_initialized" | "skipped"
    """
    te = self.template_embedder
    if not getattr(te, "rna_template_enable", False):
        return "skipped"

    rna_projector_key = "template_embedder.linear_no_bias_a_rna.weight"
    rna_projector_key_ddp = "module." + rna_projector_key

    if checkpoint_keys is not None:
        has_rna_weights = (
            rna_projector_key in checkpoint_keys
            or rna_projector_key_ddp in checkpoint_keys
        )
        if has_rna_weights:
            return "loaded_from_checkpoint"

    init_mode = getattr(te, "rna_projector_init", "protein")
    with torch.no_grad():
        if init_mode == "zero":
            nn.init.zeros_(te.linear_no_bias_a_rna.weight)
            return "zero_initialized"
        else:
            te.linear_no_bias_a_rna.weight.copy_(te.linear_no_bias_a.weight)
            return "copied_from_protein"
```

#### 文件 4: `runner/train.py` — 传递 checkpoint keys

```python
# 修改后
if self.configs.load_checkpoint_path:
    ckpt_keys = _load_checkpoint(...)  # _load_checkpoint 现在返回 checkpoint key set

    rna_cfg = self.configs.get("rna_template", {})
    if rna_cfg.get("enable", False):
        result = self.raw_model.reinit_rna_projector_from_protein(
            checkpoint_keys=ckpt_keys
        )
        self.print(f"RNA projector init after checkpoint load: {result}")
```

`_load_checkpoint` 内部新增：
```python
checkpoint_keys = set(checkpoint["model"].keys())
# ... load_state_dict ...
return checkpoint_keys
```

### 2.4 行为矩阵

| 场景 | `projector_init` | Checkpoint 内容 | reinit 行为 | 结果 |
|------|-------------------|-----------------|-------------|------|
| 首次从 protein-only ckpt finetune | `"protein"` | 无 RNA 权重 | `copied_from_protein` | RNA projector = loaded protein projector |
| 首次从 protein-only ckpt finetune | `"zero"` | 无 RNA 权重 | `zero_initialized` | RNA projector = 全零 |
| 恢复 RNA finetune ckpt | `"protein"` | 有 RNA 权重 | `loaded_from_checkpoint` | RNA projector = checkpoint 中的训练好的权重 |
| 恢复 RNA finetune ckpt | `"zero"` | 有 RNA 权重 | `loaded_from_checkpoint` | RNA projector = checkpoint 中的训练好的权重 |
| RNA template 未启用 | any | any | `skipped` | 不做任何事 |

---

## 3. Inference Projector Validation

### 3.1 问题根因

旧实现中，inference 路径在加载 checkpoint 后：
- RNALM projector 缺失时仅 `logger.warning()`，不 raise
- RNA template projector 完全不检查

这意味着 inference 可能使用未初始化/随机初始化的 projector 权重进行推理，产生错误结果但不报错。

### 3.2 设计思路

**Inference 的逻辑很简单：开了就必须要有。**

如果用户在 inference 配置中启用了 rnalm/rna_template，那 checkpoint 中必须包含对应的 projector 权重。不存在 "fallback 到零初始化" 的场景——inference 时零初始化的 projector 没有任何意义。

### 3.3 修复代码

**文件:** `runner/inference.py`

```python
# 修改后
ckpt_keys = set(checkpoint["model"].keys())

# RNALM projector validation
rnalm_configs = self.configs.get("rnalm", {})
if rnalm_configs.get("enable", False):
    rnalm_keys_in_ckpt = [
        k for k in ckpt_keys
        if any(s in k for s in [
            "rna_projection", "dna_projection", "rnalm_projection",
            "linear_rna_llm", "linear_dna_llm", "linear_rnalm",
            "rnalm_alpha_logit", "rnalm_gate_mlp",
        ])
    ]
    if not rnalm_keys_in_ckpt:
        raise RuntimeError(
            "rnalm.enable=True but checkpoint contains NO RNA/DNA LM projector weights. "
            "Inference requires a finetuned checkpoint that was trained with rnalm.enable=True. "
            "Either load a proper checkpoint or set rnalm.enable=false."
        )

# RNA template projector validation
rna_template_configs = self.configs.get("rna_template", {})
if rna_template_configs.get("enable", False):
    rna_tpl_keys_in_ckpt = [
        k for k in ckpt_keys
        if "linear_no_bias_a_rna" in k or "rna_template_alpha" in k
    ]
    if not rna_tpl_keys_in_ckpt:
        raise RuntimeError(
            "rna_template.enable=True but checkpoint contains NO RNA template projector weights "
            "(linear_no_bias_a_rna). Inference requires a finetuned checkpoint that was trained "
            "with rna_template.enable=True. Either load a proper checkpoint or set "
            "rna_template.enable=false."
        )
```

### 3.4 行为变化

| 场景 | 旧行为 | 新行为 |
|------|--------|--------|
| `rnalm.enable=True`, checkpoint 无 RNALM 权重 | `logger.warning()` → 静默使用零初始化 projector | `raise RuntimeError` → 立即终止 |
| `rna_template.enable=True`, checkpoint 无 RNA template 权重 | 不检查 → 使用随机初始化的 projector | `raise RuntimeError` → 立即终止 |
| 权重存在 | 正常加载 | 正常加载 + 打印确认信息 |

---

## 4. Bug 3 (Low): 脚本 fail-fast 语义修复

### 4.1 问题

两个训练脚本中 RNA database 目录缺失时只打印 WARNING：
```bash
# 旧代码
[ -d "${RNA_DATABASE_DIR}" ] || { echo "WARNING: RNA database dir not found: ${RNA_DATABASE_DIR}. Templates will be empty."; }
```

但代码层已经是 fail-fast（`raise FileNotFoundError`），所以脚本的 WARNING 信息误导用户。

### 4.2 修复

```bash
# 修改后 — 两个脚本都改为 ERROR + exit 1
[ -d "${RNA_DATABASE_DIR}" ] || { echo "ERROR: RNA database dir not found: ${RNA_DATABASE_DIR}"; exit 1; }
[ -f "${RNA_TEMPLATE_INDEX}" ] || { echo "ERROR: RNA template index not found: ${RNA_TEMPLATE_INDEX}"; exit 1; }
```

同时新增 `--rna_projector_init` CLI 选项和验证：
```bash
RNA_PROJECTOR_INIT="protein"    # 默认值
# CLI 解析
--rna_projector_init)   RNA_PROJECTOR_INIT="$2";    shift 2 ;;
# 验证
[[ "${RNA_PROJECTOR_INIT}" =~ ^(protein|zero)$ ]] || { echo "ERROR: --rna_projector_init must be protein/zero"; exit 1; }
# 传递到 python
--rna_template.projector_init ${RNA_PROJECTOR_INIT}
```

---

## 5. 修改文件汇总

| 文件 | 修改内容 | 行号 |
|------|---------|------|
| `configs/configs_base.py` | 删除 `copy_prot_projector_after_load`/`gate_mode`/`gate_init_logit`；新增 `projector_init` | 144-155 |
| `protenix/model/modules/pairformer.py` | TemplateEmbedder 支持 zero-init 模式；forward 按模式选择 alpha gate | 990-1015, 1096-1101 |
| `protenix/model/protenix.py` | `reinit_rna_projector_from_protein` 改为 smart checkpoint detection | 281-320 |
| `runner/train.py` | `_load_checkpoint` 返回 checkpoint keys；传递给 reinit 方法 | 564-661 |
| `runner/inference.py` | RNALM: warning→RuntimeError；新增 RNA template projector 校验 | 179-212 |
| `finetune/finetune_rna_template_1stage.sh` | WARNING→ERROR+exit；新增 `--rna_projector_init` | 33,80,147-157,192 |
| `finetune/finetune_rna_template_2stage.sh` | WARNING→ERROR+exit；新增 `--rna_projector_init` | 35,93,151-161,196 |
| `finetune/test_rna_template_integration.py` | 更新 Test 1/4/10；新增 Test 12/13/14 | 全文 |

---

## 6. 测试结果

```
============================================================
  RNA Template Integration Validation Tests
============================================================

  [PASS] Config loads with rna_template section and projector_init
  [PASS] Adapter keywords include RNA template params
  [PASS] RNATemplateFeaturizer returns correct feature shapes
  [PASS] TemplateEmbedder initializes with RNA template configs (protein and zero mode)
  [PASS] TemplateEmbedder forward pass with RNA templates (GPU)
  [PASS] RNA template parameters receive gradients
  [PASS] Backward compatibility: model works without RNA templates
  [PASS] Combined protein + RNA templates forward pass
  [PASS] get_rna_template_featurizer factory function works
  [PASS] RNA projector init: protein mode copies weights, zero mode zeros weights
  [PASS] RNA aatype is masked by effective_mask in forward pass
  [PASS] Zero-init mode: forward pass works and produces valid output
  [PASS] Smart checkpoint detection: skip reinit when RNA weights in checkpoint
  [PASS] Inference: missing projector weights in checkpoint raises RuntimeError

  Results: 14 passed, 0 failed, 14 total
============================================================
```

### 新增测试说明

| Test | 验证内容 |
|------|---------|
| Test 1 (更新) | 验证 `projector_init` config 存在且默认为 `"protein"` |
| Test 4 (更新) | 验证 protein 模式创建 alpha gate，zero 模式不创建 alpha gate 且权重为零 |
| Test 10 (更新) | 验证 protein 模式拷贝权重正确，zero 模式零初始化正确 |
| Test 12 (新增) | Zero-init 模式完整 forward+backward pass，验证梯度流 |
| Test 13 (新增) | Smart checkpoint detection：3 种场景（有RNA权重/无RNA权重+protein/无RNA权重+zero） |
| Test 14 (新增) | Inference projector validation：缺失权重 raise RuntimeError |

---

## 7. 使用指南

### 7.1 首次从 protein-only checkpoint finetune（protein 模式）

```bash
bash finetune/finetune_rna_template_1stage.sh \
    --rna_projector_init protein \
    --rna_template_alpha 0.01
```

RNA projector 从 protein projector 拷贝权重，通过 learnable alpha gate 控制贡献大小。

### 7.2 首次从 protein-only checkpoint finetune（zero 模式）

```bash
bash finetune/finetune_rna_template_1stage.sh \
    --rna_projector_init zero
```

RNA projector 零初始化，直接加到 pairwise 中（和 protein template 一样），不需要 alpha gate。

### 7.3 恢复 RNA finetune checkpoint

```bash
# 不需要指定 --rna_projector_init
# Smart detection 会自动检测 checkpoint 中有 RNA projector 权重，直接使用
python train.py \
    --load_checkpoint_path /path/to/rna_finetune.pt \
    --load_strict false \
    --rna_template.enable true
```

### 7.4 Inference

```bash
# 必须使用包含 RNA template projector 权重的 checkpoint
# 否则会 raise RuntimeError
python inference.py \
    --rna_template.enable true \
    --rna_template.template_database_dir /path/to/rna_templates/ \
    --rna_template.template_index_path /path/to/index.json
```

---

## 8. 设计决策记录

### 为什么用 smart detection 而不是手动开关？

旧方案用 `copy_prot_projector_after_load=True/False` 让用户自己管理，但：
1. 用户需要记住在 resume 时设为 `False`，容易忘
2. 忘了设就会 silent clobber，loss 看着正常但 RNA projector 被重置
3. 两个脚本都没暴露这个开关

新方案通过检查 checkpoint key 自动判断：
- 有 RNA 权重 → 说明是 RNA finetune checkpoint → 不覆盖
- 没有 RNA 权重 → 说明是 protein-only checkpoint → 按策略初始化

零误操作风险。

### 为什么 zero 模式不需要 alpha gate？

Zero-init 的 projector 输出在训练开始时严格为零，不会影响已有的 protein template 输出。随着训练进行，projector 权重自然从零增长。这和 RNALM projector 的 zero-init 策略一致（`nn.init.zeros_(self.linear_rnalm.weight)`），不需要额外的 gating 机制。

Protein 模式需要 alpha gate 是因为拷贝后的 RNA projector 会立即产生与 protein projector 相同数量级的输出，如果直接加上去会破坏预训练模型的平衡。

### 为什么 inference 从 warning 改为 error？

Inference 时使用未训练的 projector（零初始化或随机初始化）没有任何意义——产出的结果不可信，但又不会明显报错。用户可能误以为 inference 正常工作，浪费计算资源在错误结果上。

训练时可以从零开始学，但 inference 不可以。所以 inference 必须 fail-fast。
