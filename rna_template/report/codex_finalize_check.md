# Codex Finalize Check

日期：2026-03-15

## 结论先行

这次修改里，`cross_only_revalidation_20260314.md` 指出的核心测试入口问题已经修复：`test_rna3d_e2e.sh` 现在确实走了 `training_sequences JSON -> mmseqs2 -> cross-template -> index` 这条 production-style query source。

但我不能直接给出“全部问题都已完全解决”的结论。原因不是 RNA template 主功能没工作，而是我在复核时确认了几处仍然成立的高风险点，尤其是：

1. `RNA template enable` 的 inference 路径并不会像训练那样从 protein projector 自动回填，`runner/inference.py` 现在会直接拒绝没有 RNA template projector 权重的 checkpoint。
2. 从 base checkpoint 开始训练且启用 RNA template 时，EMA shadow 的初始化顺序有问题，早期 EMA 评估/EMA checkpoint 可能拿到的是 stale RNA template projector。
3. 你当前的 production validate 脚本不是 clean-room 运行，`rna_database/templates` 旧产物会被直接复用，所以它不能单独证明“这次 run 从零构建也完全没问题”。

## Findings

### 1. 高严重度：`finalize_check.md` 关于 inference 自动回填 RNA projector 的结论与实际代码不一致

`finalize_check.md` 把 `fix_template_init_inference` 写成了“checkpoint 没有 RNA projector 时会从 protein projector copy，然后 inference 可以继续”。

实际代码不是这样：

- 训练路径确实会在 checkpoint load 之后调用 `reinit_rna_projector_from_protein()`：
  - `runner/train.py:651-660`
  - `protenix/model/protenix.py:281-329`
- 但 inference 路径没有调用这段回填逻辑，反而会在 checkpoint 不含 RNA template projector 权重时直接抛错：
  - `runner/inference.py:206-223`

也就是说：

- `RNA template enable + protein-only/base checkpoint`：
  - 训练：可以继续，因为会 reinit
  - inference：不能继续，会报错

所以“`fix_template_init_inference` 已验证可工作”这个结论当前不成立。

### 2. 高严重度：EMA shadow 在 RNA template reinit 之前注册，导致早期 EMA 结果可能是错的

训练初始化顺序现在是：

1. 先加载 `load_ema_checkpoint_path`
2. 立刻 `ema_wrapper.register()`
3. 再加载 `load_checkpoint_path`
4. 最后才对 RNA template projector 做 `reinit_rna_projector_from_protein()`

对应代码：

- `runner/train.py:632-660`
- `runner/ema.py:48-54`
- `runner/ema.py:70-77`

这在 RNA template 场景里有明显风险：

- 构造模型时，`linear_no_bias_a_rna` 会先从“尚未 load checkpoint 的 protein projector”复制一次：
  - `protenix/model/modules/pairformer.py:1006-1018`
- base checkpoint 本身并没有 RNA template projector 权重
- 所以 EMA shadow 先记录下来的，可能是 stale/random 的 RNA projector
- 后面虽然 raw model 被 reinit 成“copied_from_protein”，EMA shadow 不会自动同步

影响：

- 训练早期的 EMA 评估可能不可信
- 早期保存的 `*_ema_*.pt` 也可能带着错误的 RNA template projector 初值

这会直接削弱 `finalize_check.md` 里基于短程训练的 EMA 侧结论。

### 3. 中严重度：production pipeline 验证不是 clean-room，旧模板会污染结果

`run_pipeline.sh` 只会创建输出目录，不会清空旧的 `templates/`：

- `rna_template/scripts/run_pipeline.sh:107`

我实际运行：

```bash
bash finetune/finetune_rna_template_validate.sh --skip_training --num_test_structures 20
```

在 Step 2 结束、Step 3 构建之前，日志已经出现：

- `Cross-index: 38 templates found, 105 missing`

这说明当前 `rna_database/templates` 里已经有旧 `.npz` 被直接复用了，验证不是从零开始。

因此：

- 这个 validate run 能证明“主路径可以走到 search，并且会读取/使用现有模板目录”
- 不能独立证明“本轮代码在空目录上也能从零稳定构建出相同结果”

### 4. 中严重度：inference 对 RNA template database 路径缺少和训练同等级别的 fail-fast 校验

训练数据管线里，`rna_template.enable=true` 时会检查：

- `template_database_dir` / `template_index_path` 是否为空
- `template_database_dir` 是否存在

代码：

- `protenix/data/pipeline/dataset.py:1099-1128`

但 inference dataloader 只检查“字符串是否为空”，没有检查目录存在性：

- `protenix/data/inference/infer_dataloader.py:195-220`

而 `RNATemplateFeaturizer` 自身也不会在构造时校验 database dir 是否存在，真正读不到 `.npz` 时只是 warning 并跳过：

- `protenix/data/rna_template/rna_template_featurizer.py:101-109`
- `protenix/data/rna_template/rna_template_featurizer.py:279-297`

结果是：

- 训练侧：坏路径会早失败
- inference 侧：坏路径可能变成“静默退化为无 template/少 template”

这会让线上或离线推理更难排错。

### 5. 中严重度：RNA LM inference 测试脚本和实际 inference 策略已经不一致

`test_inference_rnalm.sh` 仍然写着：

- `Base model with rnalm enabled (zero-init — should warn)`
  - `test_inference_rnalm.sh:9`
  - `test_inference_rnalm.sh:177-179`

但实际 `runner/inference.py` 对 `rnalm.enable=true` 且 checkpoint 没有 RNALM projector 权重时，是直接 `RuntimeError`：

- `runner/inference.py:185-200`

这不是 RNA template 主逻辑 bug，但它会误导后续验证，尤其会让“旧 LLM inference 还应当只是 warning”这种判断失真。

## 已验证的事项

### A. 模块级整合测试通过

运行：

```bash
python finetune/test_rna_template_integration.py
```

结果：

- `14 passed, 0 failed`

它确认了这些点：

- `rna_template` 默认配置存在且默认关闭
- `TemplateEmbedder` 在 `protein` / `zero` 两种 RNA projector init 模式下都能初始化
- RNA template forward/backward 正常
- `rna_template.enable=false` 时 TemplateEmbedder backward compatibility 正常
- `reinit_rna_projector_from_protein()` 的本地逻辑按当前实现工作

注意：这里验证的是模块级行为，不是完整 inference runner。

### B. `cross_only_revalidation` 指出的 query source 问题已被真实修复

代码层面，`test_rna3d_e2e.sh` 已经改成：

- 先生成 test `training_sequences JSON`
- Step 2 / Step 4 显式传 `--training_sequences`

位置：

- `rna_template/scripts/test_rna3d_e2e.sh:104-154`
- `rna_template/scripts/test_rna3d_e2e.sh:200-208`

同时 production pipeline 入口仍然使用：

- `rna_template/scripts/run_pipeline.sh:162-181`
- `rna_template/scripts/run_pipeline.sh:198-212`

也就是现在测试入口和 production query source 已经对齐。

### C. Cross-only E2E 小规模实跑通过

运行：

```bash
bash rna_template/scripts/test_rna3d_e2e.sh --num_test 10 --skip_training
```

实际结果：

- catalog：10 structures
- generated training_sequences JSON：2 unique sequences / 3 PDB mappings
- MMseqs2 search：`2/3 queries have templates`
- cross-template build：`2 templates built`
- rebuild index：`1 sequence`, `2 template paths`
- NPZ validation：`2/2 passed`
- 脚本最终输出：`PIPELINE TEST PASSED (training skipped)`

这说明：

- `training_sequences JSON` loader 真被走到了
- search/build/index 链路在 cross-only 小样本上是通的
- 没有出现之前报告里那种“测试入口没覆盖 production query source”的问题

### D. `RNA template disable` 时，训练侧确实退回到原来的 LLM finetune 主架构

静态代码上：

- `finetune/finetune_rna_template_1stage.sh` 在 `--use_rna_template false` 时只会额外传：
  - `--rna_template.enable false`
  - 位置：`finetune/finetune_rna_template_1stage.sh:147-160`
- RNALM 参数块仍然沿用原先 LLM finetune 路径：
  - `finetune/finetune_rna_template_1stage.sh:132-145`
- 旧脚本的 RNALM 主体参数是同一套：
  - `finetune/finetune_1stage.sh:180-190`
- 模型构造时如果 `rna_template.enable=false`，就会把 `rna_template_configs` 置为 `None`，不会创建 RNA template projector/gate：
  - `protenix/model/protenix.py:130-137`
  - `protenix/model/modules/pairformer.py:990-1019`

动态验证上，我运行了：

```bash
bash test_toggle_backward_compat.sh
```

拿到的关键证据：

- RNALM input injection 正常创建：
  - `Separate RNA/DNA input injection (like ESM): use_rna=True (2048->449), use_dna=True (1024->449), zero-init`
- 数据和 embedding 正常加载
- base checkpoint 正常加载
- 成功训练到 step 2
- 成功保存：
  - `output/test_toggle_compat/test_toggle_compat_20260315_063746/checkpoints/2.pt`
  - `output/test_toggle_compat/test_toggle_compat_20260315_063746/checkpoints/2_ema_0.999.pt`
- 并成功进入 evaluation

我在 evaluation 中途手动停掉了它，原因只是避免继续占 GPU，不是因为报错。

这足以支持：

- `RNA template disable` 时，LLM finetune 训练主架构没有被 RNA template 代码破坏
- 至少训练初始化、数据管线、checkpoint load、前 3 个 step 和 eval 入口都能工作

## 对你最关心问题的直接回答

### 1. 你的 pipeline 现在是否 work？

部分成立，分层回答：

- `cross-only` 小规模 E2E：能 work，已实跑通过
- 模块级 RNA template integration：能 work，14/14 通过
- production `run_pipeline.sh`：主路径能走到 search，且使用了真实 `training_sequences JSON`，但我没有把它当作“独立通过”计入，因为当前验证被已有 `rna_database/templates` 污染了
- `RNA template enable` inference：不能说完全 work，因为 base/protein-only checkpoint 仍然会被 `runner/inference.py` 直接拒绝

### 2. 之前报告里的问题是否解决？

`cross_only_revalidation_20260314.md` 指出的那个核心测试入口问题，已经解决。

但 `finalize_check.md` 里关于 inference 自动回填和“整体都 passed”的表述，我认为过于乐观，至少 inference 和 EMA 两块还不能算完全收口。

### 3. 如果 `RNA template disable`，是否会退回你之前的 with-LLM finetune 架构？

训练侧答案是：会。

理由：

- `rna_template.enable=false` 时不会实例化 RNA template projector/gate
- RNALM 模块代码路径独立存在，没有被 template 开关改写
- backward-compat 训练冒烟已经实际跑通到 checkpoint 保存和 eval 入口

### 4. LLM 架构有没有变，是否还能 work？

如果说的是 `RNA template disable` 条件下的训练主架构，结论是：

- RNALM 架构本身没有被 RNA template 改写
- 还能 work

但要区分训练和 inference：

- 训练：当前可工作
- inference：RNALM 和 RNA template 都采用了“checkpoint 必须带对应 projector 权重”的严格策略；这和一些旧测试脚本里的“base model 只 warning”预期已经不一致

## 机制理解总结

### RNA template 机制

- 数据层：`RNATemplateFeaturizer` 从预计算 `.npz` 读取 `rna_template_*` 特征
- 模型层：`TemplateEmbedder` 额外创建 `linear_no_bias_a_rna`
- 融合位置：走 template/z-init 路径，不是 RNALM 的 input/diffusion 路径
- 默认 `protein` init：用 protein template projector 权重初始化 RNA projector，并用 `rna_template_alpha` 控制初始注入强度

### RNALM 机制

- 数据层：`RiNALMoFeaturizer`
- 输入注入：`InputFeatureEmbedder`
- diffusion 注入：`Protenix._get_s_rnalm()` / `DiffusionModule`
- 与 RNA template 是并列关系，不是同一模块

所以当 `RNA template disable` 时，移除的是 template 分支，不是 RNALM 分支。

## 建议的后续验证

1. 用一个全新的空输出目录重跑 `run_pipeline.sh`，不要复用 `rna_database/`
2. 修正训练初始化顺序：先做 RNA projector reinit，再 `ema_wrapper.register()`
3. 明确决定 inference 策略：
   - 要么支持 base checkpoint + runtime reinit
   - 要么维持当前 hard-fail，并同步修正文档/测试脚本/报告
4. 给 `finetune_rna_template_1stage.sh --use_rna_template false` 加一个 1-step 的独立 smoke test，避免以后只能靠 `test_toggle_backward_compat.sh`
5. 给 inference dataloader 补上 `template_database_dir` 存在性检查，和训练侧保持一致

## 本次实际执行的验证命令

```bash
python finetune/test_rna_template_integration.py
bash rna_template/scripts/test_rna3d_e2e.sh --num_test 10 --skip_training
bash test_toggle_backward_compat.sh
bash finetune/finetune_rna_template_validate.sh --skip_training --num_test_structures 20
```

其中：

- 前两个命令完整跑完
- 后两个命令在拿到足够证据后被我手动停止，避免继续占 GPU / 继续污染共享输出目录
