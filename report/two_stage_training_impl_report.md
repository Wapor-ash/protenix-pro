# Protenix 两阶段训练实现深度解读

## 0. 结论先行

先直接回答你的两个核心问题。

### 结论 1: `stage1` 是不是“只 train projector”？

**不一定，取决于 `two_stage.adapter_keywords` 配的是什么。**

- 如果脚本里是 `--two_stage.adapter_keywords "rnalm_projection"`，那么 `stage1` 的确**只训练 projector**。
  - 这对应 [`rna_2stage_fast.sh`](../rna_2stage_fast.sh) 第 118-125 行。
- 如果脚本里是 `--two_stage.adapter_keywords "rnalm"`，那么 `stage1` 训练的是**所有名字里带 `rnalm` 的参数**，不只是 projector，还包括 gate 参数。
  - 这对应 [`rna_llm_gate_tune.sh`](../rna_llm_gate_tune.sh) 第 86-94 行。

所以，“stage1 只 train projector”这句话对 `rna_2stage_fast.sh` 是对的，但对 `rna_llm_gate_tune.sh` 是错的。

### 结论 2: `stage2` 是不是“只 train diffusion network”？pairformer/trunk 会不会一起 train？

**会一起 train。并且不是只 train diffusion，而是整个 model 做 joint training。**

`stage2` 在实现上会把**所有参数**重新放进 optimizer：

- adapter group: 名字匹配 `adapter_keywords` 的参数
- backbone group: 其余所有参数

这意味着在 `stage2`：

- `pairformer_stack` / trunk 会训练
- `diffusion_module` 会训练
- `distogram_head` 会训练
- `confidence_head` 会训练
- 其它 backbone 模块也会训练

因此 `stage2` 不是“只训 diffusion network”，而是**全模型联合训练**。

## 1. 两阶段训练的真实实现方式

核心代码在 [`runner/train.py`](../runner/train.py) 。

### 1.1 Stage 1 不是靠 `requires_grad=False` 冻结，而是靠 `lr=0`

`_setup_stage1()` 的实现见 [`runner/train.py`](../runner/train.py) 第 248-314 行。

关键点：

1. 它先按 `adapter_keyword` 把参数分成 `adapter_params` 和 `backbone_params`
2. **并没有把 backbone 的 `requires_grad` 改成 `False`**
3. 它创建了两个 optimizer param group:
   - backbone group: `lr = 0.0`
   - adapter group: `lr = stage1_lr`

对应代码证据：

- 参数按名字分组: 第 258-269 行
- backbone/adpater 参数量打印: 第 271-274 行
- optimizer 两组 LR: 第 280-291 行

这意味着 `stage1` 的“冻结”是**优化器层面的冻结**，不是 autograd 层面的冻结。

### 1.2 Stage 1 为了防 scheduler 把 backbone LR 改回来，还手动把它重置为 0

在 [`runner/train.py`](../runner/train.py) 第 799-801 行：

```python
if self.two_stage_enable and self.current_stage == 1:
    self.optimizer.param_groups[self._stage1_backbone_group_idx]["lr"] = 0.0
```

这是因为 scheduler 每步都会更新 param group LR，如果不手动改回 0，backbone group 会被 scheduler 带着走。

所以 Stage 1 的真实语义是：

- backbone 依然参与前向、反向
- 可能仍然产生梯度
- 但是 optimizer update 时 LR=0，不发生参数更新

### 1.3 Stage 2 会重建 optimizer，把全模型重新纳入训练

`_transition_to_stage2()` 见 [`runner/train.py`](../runner/train.py) 第 316-386 行。

关键点：

1. 第 324-326 行把所有参数 `requires_grad=True`
2. 第 335-341 行再次按 `adapter_keyword` 分组
3. 第 347-356 行重新创建 optimizer
   - backbone group: `lr = stage2_lr`
   - adapter group: `lr = stage2_adapter_lr` 或 `stage2_lr`
4. 第 359-376 行重新创建 stage2 的 scheduler
5. 第 379-386 行开启 EMA

注意：这里的“Unfreeze all parameters”在当前实现里更像是**语义声明**，因为 stage1 本来就没有把 backbone 的 `requires_grad` 关掉。

## 2. stage1/stage2 到底训练哪些参数

## 2.1 `adapter_keywords` 决定 adapter 边界

默认配置见 [`configs/configs_base.py`](../configs/configs_base.py) 第 62-72 行：

- 默认 `adapter_keywords = "rnalm_projection"`

这说明两阶段训练的参数划分**完全靠参数名 substring 匹配**，不是靠 module 类型，也不是靠显式白名单。

### 2.2 `rna_2stage_fast.sh`: stage1 只训 projector

脚本见 [`rna_2stage_fast.sh`](../rna_2stage_fast.sh) 第 118-125 行：

```bash
--two_stage.enable true \
--two_stage.stage1_max_steps ${STAGE1_MAX_STEPS} \
--two_stage.stage1_lr ${STAGE1_LR} \
--two_stage.stage1_warmup_steps ${STAGE1_WARMUP} \
--two_stage.stage2_lr ${STAGE2_LR} \
--two_stage.stage2_warmup_steps ${STAGE2_WARMUP} \
--two_stage.stage2_ema_decay ${STAGE2_EMA_DECAY} \
--two_stage.adapter_keywords "rnalm_projection"
```

这时命中的只有：

- `rnalm_projection.weight`

因为 projector 定义在 [`protenix/model/protenix.py`](../protenix/model/protenix.py) 第 146-153 行：

```python
self.rnalm_projection = LinearNoBias(
    in_features=rnalm_embedding_dim,
    out_features=configs.c_s,
)
```

参数量:

- `384 x 1280 = 491,520` 参数，约 `0.4915M`

因此对这个脚本：

- Stage 1 trainable: `rnalm_projection.weight`
- Stage 1 frozen by lr=0: 全部 backbone
- Stage 2 trainable: 全模型

### 2.3 `rna_llm_gate_tune.sh`: stage1 不是只训 projector，而是 projector + gates

脚本见 [`rna_llm_gate_tune.sh`](../rna_llm_gate_tune.sh) 第 80-94 行：

```bash
--rnalm.fusion_method "add" \
--rnalm.gate_mode "dual" \
...
--two_stage.stage2_adapter_lr 0.0003 \
--two_stage.adapter_keywords "rnalm"
```

而 gate 参数在 [`protenix/model/protenix.py`](../protenix/model/protenix.py) 第 155-171 行创建：

- `rnalm_alpha_logit`
- `rnalm_gate_mlp`
- `rnalm_projection`

并且 `_get_s_rnalm()` 在第 373-383 行使用它们。

所以这个脚本的 `stage1` 实际训练参数是：

- `rnalm_projection.weight`
- `rnalm_alpha_logit`
- `rnalm_gate_mlp.0.weight`
- `rnalm_gate_mlp.0.bias`
- `rnalm_gate_mlp.2.weight`
- `rnalm_gate_mlp.2.bias`

参数量:

- `rnalm_projection`: `384 x 1280 = 491,520`
- `rnalm_alpha_logit`: `1`
- `rnalm_gate_mlp`: `384x96 + 96 + 96x1 + 1 = 37,057`
- 合计: `528,578`，约 `0.5286M`

这和已有实现报告里打印的 `adapter=0.53M` 一致，见 [`report/gated_llm_implementation_report.md`](./gated_llm_implementation_report.md) 第 94 行。

## 3. stage2 为什么 pairformer/trunk 也会训练

这个问题不能只看 optimizer，还要看计算图。

## 3.1 trunk 的输出 `s/z` 来自 `pairformer_stack`

见 [`protenix/model/protenix.py`](../protenix/model/protenix.py) 第 119-145 行和第 767-772 行：

- `self.pairformer_stack = PairformerStack(...)`
- 训练时先调用 `get_pairformer_output()` 得到 `s_inputs, s, z`

`get_pairformer_output()` 内部在第 334-343 行调用：

```python
s, z = self.pairformer_stack(...)
```

也就是说：

- `s` = trunk single embedding
- `z` = trunk pair embedding

## 3.2 diffusion training 直接消费 `s/z`，而且这里没有 detach

训练主路径在 [`protenix/model/protenix.py`](../protenix/model/protenix.py) 第 890-916 行：

```python
_, x_denoised, x_noise_level = sample_diffusion_training(
    ...
    denoise_net=self.diffusion_module,
    s_inputs=s_inputs,
    s_trunk=s,
    z_trunk=None if cache["pair_z"] is not None else z,
    pair_z=cache["pair_z"],
    ...
    s_rnalm=s_rnalm,
)
```

这里的 `s_inputs/s/z/s_rnalm` 都直接喂给 diffusion denoiser，**没有 detach**。

所以只要这些参数所在 param group 的 LR 非零，diffusion loss 的梯度就会沿着：

`diffusion loss -> diffusion_module -> s/z -> pairformer_stack -> trunk/backbone`

一路回传。

因此在 `stage2`：

- `diffusion_module` 会训
- `pairformer/trunk` 也会训

这个结论是明确的，不是推测。

## 3.3 `DiffusionModule` 本身就是一个大网络，不只是一个薄头

见 [`protenix/model/modules/diffusion.py`](../protenix/model/modules/diffusion.py) 第 338-377 行。

`self.diffusion_module` 包含：

- `diffusion_conditioning`
- `atom_attention_encoder`
- `diffusion_transformer`
- `atom_attention_decoder`

并且在 `f_forward()` 里，trunk embedding 会参与 conditioning 和 atom/token attention，见第 436-519 行。

所以 stage2 的“diffusion network”本身就很大，但它仍然不是 stage2 唯一在训练的部分，因为 trunk 的 `s/z` 也在参与反传。

## 4. 哪些 loss 会把梯度传回 trunk

这个点很重要，因为它决定 stage2 虽然“全模型可训练”，但不同模块吃到的梯度来源并不一样。

## 4.1 diffusion loss: 会回传到 trunk

原因就是上一节说的：`sample_diffusion_training()` 直接吃 `s/z`，没有 detach。

所以 diffusion loss 会更新：

- diffusion_module
- pairformer/trunk
- 上游 embedding/backbone
- rnalm adapter

## 4.2 distogram loss: 也会回传到 trunk

见 [`protenix/model/protenix.py`](../protenix/model/protenix.py) 第 918-925 行：

```python
"distogram": self.distogram_head(z)
```

这里 distogram head 直接用 trunk 的 `z`。所以 distogram loss 也会把梯度打回 trunk。

## 4.3 confidence loss: 默认不会回传到 trunk

这点非常容易忽略。

在 [`protenix/model/modules/confidence.py`](../protenix/model/modules/confidence.py) 第 65 行，`stop_gradient=True` 是默认值；配置里也没改，见 [`configs/configs_base.py`](../configs/configs_base.py) 第 314-320 行附近的 `confidence_head` 配置。

在 confidence head forward 里，第 178-181 行明确写了：

```python
if self.stop_gradient:
    s_inputs = s_inputs.detach()
    s_trunk = s_trunk.detach()
    z_trunk = z_trunk.detach()
```

所以 confidence head 的 loss 默认只训练：

- confidence head 自己

而**不会**把梯度再传回 trunk / input embedder。

### 4.4 mini-rollout 也不会训练 trunk/diffusion

在训练循环里，mini-rollout 是放在 `with torch.no_grad():` 里的，见 [`protenix/model/protenix.py`](../protenix/model/protenix.py) 第 808-858 行。

这里面还显式对 `s_inputs/s/z/cache/s_rnalm` 做了 `.detach()`。

所以 mini-rollout 的作用是：

- 生成 `coordinate_mini`
- 做 label permutation
- 给 confidence head 提供输入

它不是训练梯度主路径的一部分。

## 5. 你应该如何理解“params 怎么 train”

## 5.1 Stage 1

### 当 `adapter_keywords = "rnalm_projection"`

- trainable update: 只有 `rnalm_projection.weight`
- backbone: 参与 forward/backward，但 optimizer LR=0，不更新
- EMA: 不开
- scheduler: cosine

### 当 `adapter_keywords = "rnalm"`

- trainable update: `projection + gate`
- backbone: 同样参与 forward/backward，但 LR=0
- gate 中 `s_trunk.detach()` 会阻止 gate 分支的梯度回到 trunk，见 [`protenix/model/protenix.py`](../protenix/model/protenix.py) 第 377-383 行

## 5.2 Stage 2

- optimizer 重新创建
- 所有参数重新进入 optimizer
- backbone group 用 `stage2_lr`
- adapter group 用 `stage2_adapter_lr` 或 `stage2_lr`
- EMA 开启
- 真正的训练语义是 **joint training**，不是只训 diffusion

## 6. 一个非常关键的实现细节: `stage2_adapter_lr` 很可能没有按预期持续生效

这是我认为这个实现里最值得注意的点。

### 6.1 现象

`_transition_to_stage2()` 确实创建了两个不同 LR 的 param group，见 [`runner/train.py`](../runner/train.py) 第 343-357 行：

- backbone group: `stage2_lr`
- adapter group: `stage2_adapter_lr`

但 scheduler 实现见 [`protenix/utils/lr_scheduler.py`](../protenix/utils/lr_scheduler.py) 第 22-63 行和第 147-176 行。

`CosineAnnealingWithWarmup.get_lr()` 返回的是：

```python
return [
    self._get_step_lr(self.last_epoch) for group in self.optimizer.param_groups
]
```

也就是说：

- 它对**每个 group 返回同一个 LR 值**
- 它没有保留 group 间的比例关系
- 也没有用各自的 `base_lrs` 做缩放

### 6.2 这意味着什么

在 stage2：

1. optimizer 初始化时，adapter group 可能是 `3e-4`，backbone group 是 `1e-4`
2. 但第一次 `self.lr_scheduler.step()` 之后，两个 group 都会被 scheduler 改成**同一个值**

而 stage2 没有像 stage1 那样再手动覆写某个 group LR。

所以从代码逻辑上看：

- `stage2_adapter_lr != stage2_lr` 的差异**只在 optimizer 刚创建后短暂存在**
- 后续训练里，大概率会被 scheduler 抹平成同一个 LR

### 6.3 结论

如果你的预期是：

- Stage 2 全程 adapter 用 3x LR
- backbone 用 1x LR

那么**当前实现并没有严格做到**。

更准确地说：

- 代码“想这么做”
- 但 scheduler 的写法会把这种双 LR 设计破坏掉

这个细节不会影响“stage2 是否训练 trunk”这个结论，但会影响“adapter 和 backbone 在 stage2 到底是不是按不同 LR 训练”这个结论。

## 7. 还有两个容易误判的点

## 7.1 Stage 1 的“freeze”不是省算力的 freeze

因为 backbone 没有 `requires_grad=False`，所以 Stage 1：

- 不是 LoRA 那种真正只算 adapter 的轻量训练
- 不是严格意义上的 autograd freeze
- 只是 update 不发生

所以它更像：

- **参数冻结**，不是**图冻结**

## 7.2 如果以后用 `fusion_method="concat"`，当前 `adapter_keywords` 可能漏参数

在 [`protenix/model/modules/diffusion.py`](../protenix/model/modules/diffusion.py) 第 85-92 行，`fusion_method="concat"` 时会新建：

- `layernorm_s_concat`
- `linear_no_bias_s_concat`

但这些参数名**不含 `rnalm`**。

因此如果你仍然用：

- `adapter_keywords="rnalm"`

那么这两个 concat 专用新层会被归到 backbone，而不是 adapter。

这对当前脚本没问题，因为当前脚本都用 `fusion_method="add"`；但如果将来切到 `concat`，就需要重新审视 adapter 参数匹配规则。

## 8. 最终回答

### Q1: stage1 只 train projector 吗？

- 对 `rna_2stage_fast.sh`: **是**
- 对 `rna_llm_gate_tune.sh`: **不是**，它 train `projector + gates`
- 对通用实现: **取决于 `two_stage.adapter_keywords`**

### Q2: stage2 只 train diffusion network，还是 trunk/pairformer 也 train？

- **trunk / pairformer 也会 train**
- 而且不只是 trunk + diffusion，实际上是**全模型 joint training**
- 其中 trunk 的主要梯度来源是：
  - diffusion loss
  - distogram loss
- confidence loss 默认不会回传到 trunk，因为 `ConfidenceHead.stop_gradient=True`

## 9. 一句话版

这个两阶段实现的本质不是“stage1 只开某些 `requires_grad`，stage2 只开 diffusion/trunk”，而是：

- **Stage 1**: 用 `adapter_keywords` 把 adapter 放进非零 LR group，其余参数放进 `lr=0` group
- **Stage 2**: 重建 optimizer，把**全模型**都放回可更新状态，做真正的 **joint training**

唯一需要额外小心的是：**`stage2_adapter_lr` 的双 LR 设计，按当前 scheduler 写法，大概率不会持续生效。**
