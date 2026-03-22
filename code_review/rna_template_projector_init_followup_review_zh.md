# RNA Template Projector Init / Inference Follow-up Review

日期：2026-03-14

范围：

- `code_review/fix_template_init_inference.md`
- `code_review/bug_fix_template.md`
- `configs/configs_base.py`
- `protenix/model/modules/pairformer.py`
- `protenix/model/protenix.py`
- `runner/train.py`
- `runner/inference.py`
- `finetune/finetune_rna_template_1stage.sh`
- `finetune/finetune_rna_template_2stage.sh`
- `finetune/test_rna_template_integration.py`

## Findings

### 1. Medium: `projector_init` 现在不只是“初始化策略”，它还决定是否创建 `rna_template_alpha`，因此和 checkpoint 模式不一致时会静默改变模型语义

位置：

- `configs/configs_base.py:149-154`
- `protenix/model/modules/pairformer.py:993-1018`
- `runner/train.py:604-609`
- `runner/inference.py:172-175`

问题：

- 当前实现里：
  - `projector_init="protein"` 会创建 `rna_template_alpha`
  - `projector_init="zero"` 不会创建 `rna_template_alpha`
- 这意味着 `projector_init` 不只是“初始化方式”，而是会改变模型参数集合。
- 但训练和推理的 checkpoint 加载仍然是普通 `load_state_dict(strict=self.configs.load_strict)`。

这会带来两个静默错配场景：

1. 用 `projector_init="zero"` 去加载一个 `protein` 模式训练出来的 checkpoint  
   结果：`linear_no_bias_a_rna.weight` 能加载，但 checkpoint 里的 `rna_template_alpha` 会被直接丢掉，因为模型当前没有这个参数。

2. 用 `projector_init="protein"` 去加载一个 `zero` 模式训练出来的 checkpoint  
   结果：`linear_no_bias_a_rna.weight` 能加载，但模型会额外新建一个 `rna_template_alpha`，它不是 checkpoint 里的训练结果，而是新的初始化值。

影响：

- 你在 `fix_template_init_inference.md` 里想实现的是“如果 checkpoint 已包含 RNA projector 权重，就直接用 checkpoint 的结果”。
- 但按当前实现，这个承诺只对 `linear_no_bias_a_rna.weight` 成立，不对完整的 RNA template 注入语义成立。
- 一旦 config 的 `projector_init` 和 checkpoint 的训练模式不一致，行为会静默漂移，而且现有 inference 校验也不会拦住这个问题。

为什么我把它定成 Medium：

- 这不会影响“同模式使用”的主路径。
- 但它会影响 resume / eval / deploy 时的可预测性，而且是 silent mismatch，不容易第一时间发现。

## Open Questions / Assumptions

我这次 review 基于一个假设：

- 你期望 `projector_init="protein"` 和 `projector_init="zero"` 代表两种可切换的训练/推理模式，并且“只要 checkpoint 里有 RNA template 权重，就应尽量保持 checkpoint 原语义”。

如果你的真实意图是：

- `projector_init` 只用于“冷启动建模”，并且你保证 checkpoint 的加载配置永远与训练配置严格一致，

那么上面的风险会小很多。但目前代码和脚本里没有任何显式保护去保证这件事。

## Summary

这次修复的整体判断是正面的：

- 是，大部分我之前指出、且你这次选择修的 bug，已经在代码层面修掉了。
- 你修复了 `resume` 覆盖 RNA projector 的旧逻辑。
- 你给训练脚本补了 `zero/protein` 选项。
- 你把 inference 路径上“开了 projector 但 checkpoint 里没有对应权重”改成了显式报错。
- 你还把脚本里的 fail-fast 语义和代码统一了。

我这次没有再看到上一轮那种明显的 Critical / High 级残留问题。

剩下我认为最值得你继续盯的，是上面这个 Medium 级一致性问题：

- `projector_init` 现在决定了 `rna_template_alpha` 是否存在；
- 因而它不只是 init policy，而是模型结构选择；
- config 和 checkpoint 一旦模式不一致，就会出现 silent semantic drift。

## 验证说明

我没有修改任何源码。

我做了两类检查：

- 静态代码核对：逐项比对 fix report 和真实实现。
- 语法级检查：相关 Python 文件 `py_compile` 通过。

我没有在当前机器上独立跑通 `14/14` 测试，因为本机仍然缺少你项目运行所需的完整环境与依赖；所以我这次的结论是基于代码审查，不是基于完整执行复现。
