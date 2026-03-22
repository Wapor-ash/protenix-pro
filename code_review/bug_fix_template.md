# RNA Template Bugfix Follow-up Review

日期：2026-03-14

范围：

- `code_review/rna_template_bugfix_report.md`
- `protenix/data/rna_template/rna_template_featurizer.py`
- `protenix/data/pipeline/dataset.py`
- `protenix/data/inference/infer_dataloader.py`
- `protenix/model/modules/pairformer.py`
- `protenix/model/protenix.py`
- `runner/train.py`
- `finetune/finetune_rna_template_1stage.sh`
- `finetune/finetune_rna_template_2stage.sh`
- `finetune/test_rna_template_integration.py`

## Findings

### 1. Medium: `copy_prot_projector_after_load=True` 默认会在训练恢复时覆盖已经训练好的 RNA projector

位置：

- `configs/configs_base.py:144-153`
- `runner/train.py:645-655`
- `protenix/model/protenix.py:282-309`
- `finetune/finetune_rna_template_1stage.sh:148-153`
- `finetune/finetune_rna_template_2stage.sh:152-157`

问题：

- 这次修复为了解决“从 protein-only checkpoint 开始 finetune 时，RNA projector 没有在 checkpoint load 后重新同步”的问题，新增了 `copy_prot_projector_after_load=True` 默认值，并在 `train.py` 里无条件执行 `self.raw_model.reinit_rna_projector_from_protein()`。
- 但 `reinit_rna_projector_from_protein()` 当前实现是“只要开关为真就总是 copy”，没有判断当前 checkpoint 是否已经包含 RNA projector，也没有判断当前是否是 resume RNA finetune。
- 两个训练脚本都没有显式传 `--rna_template.copy_prot_projector_after_load false` 的入口，因此默认行为就是覆盖。

影响：

- 从 base protein checkpoint 开始第一次 RNA finetune，这个修复是有益的。
- 但如果后续你要“恢复一个已经训练过 RNA template 的 checkpoint”，当前默认行为会把 checkpoint 里的 RNA projector 权重再次覆盖成 protein projector 权重，等于 silent clobber。
- 这会让恢复训练和继续训练的行为不符合直觉，也会让你以为“resume 成功了”，实际上 RNA projector 被重置了。

结论：

- 这说明 Bug 1.5 不是完全修好，而是“训练冷启动场景修好了，但 resume 场景留下了回归风险”。

注意：claude code！！你要做的
帮我修复上面的！ ，然后帮我初始化帮我在scirpt 里面提供选项 

zero init （那就删掉那个小的gate alpha， 直接加到pairwise里面 和protein一样，因为zeroinit 了）

还是protein （那就保持一样的训练逻辑）

此外，不管选哪个， 如果checkpoint里面包含了rna template projectro 的权重，就直接覆盖

————————

inference 的时候帮我实现

就是如果 rnalm rna llm dnallm ， 或者rna template 任何一个开着 就就检查那个projector 的weights 如果没有就报错，不需要你乱iniit， 就是inference 的情况开着就必须要有的逻辑。

### 3. Low: 脚本和代码的 fail-fast 语义已经不一致，容易误导后续使用者

位置：

- `protenix/data/pipeline/dataset.py:1107-1118`
- `protenix/data/rna_template/rna_template_featurizer.py:50-63`
- `finetune/finetune_rna_template_1stage.sh:146-153`
- `finetune/finetune_rna_template_2stage.sh:150-157`

问题：

- 代码层现在已经是 fail-fast：
  - 配置路径为空会 `raise ValueError`
  - 训练目录不存在会 `raise FileNotFoundError`
  - index 不存在或为空也会在 featurizer 初始化时报错
- 但两个训练脚本还保留着旧语义：数据库目录缺失时只打印 `Templates will be empty.`

影响：

- 当前行为实际上不会“静默变空”，而是更可能在 dataset/featurizer 初始化阶段直接失败。
- 脚本注释和真实行为不一致，会增加排障成本。

结论：

- 这不是功能性 bug，但属于文档/脚本语义漂移。

## 已修复确认

按我上次提出、且你这次明确说要修的范围来看，以下问题我确认已经在代码层落地：

- Bug 1.1：已修复。
  - `_empty_rna_template_features()` 现在返回 `rna_template_*` 命名空间，并补了 `rna_template_block_mask`。
  - `RNATemplateFeaturizer.__call__()` 在无 RNA 样本时不再覆盖 protein template。
  - 相关测试也已经改成检查 `rna_template_*`。

- Bug 1.3：大体修复。
  - 训练侧 `get_rna_template_featurizer()` 已从 warning 改成 raise。
  - `_load_rna_template_index(..., fail_fast=True)` 也已加入缺失/空 index 的错误处理。
  - 这意味着“配置开了但默默不用 template”的风险大幅下降。

- Bug 1.6：已修复。
  - `_single_rna_template_forward()` 里 `aatype_row/aatype_col` 现在已经乘上 `effective_mask` 和 `pair_mask`。
  - 你上次报告里提到的 block mask 语义漏洞，这次基本补上了。

- Bug 1.7：大体修复。
  - `infer_dataloader.py` 已经接入 `RNATemplateFeaturizer`，`process_one()` 也会写入 `rna_template_*` 特征。
  - 所以“完全没接 inference pipeline”这个问题已经不是当前状态了。
