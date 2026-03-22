# RNA Template Injection 代码审查报告

日期：2026-03-14

审查范围：

- `prompts/rna_template/tempale_create.txt`
- `code_review/rna_template_injection_code_review.md`
- `rna_template/pipeline_validation_report.md`
- `protenix/data/rna_template/*`
- `protenix/data/pipeline/dataset.py`
- `protenix/data/inference/infer_dataloader.py`
- `protenix/model/modules/pairformer.py`
- `protenix/model/protenix.py`
- `configs/configs_base.py`
- `finetune/finetune_rna_template_1stage.sh`
- `finetune/finetune_rna_template_2stage.sh`
- `finetune/test_rna_template_integration.py`

结论先行：

- 你的 RNA template 注入目标只完成了“第一版的一部分”。
- 已完成的部分主要是：训练数据侧加载 `rna_template_*` 特征、`TemplateEmbedder` 内新增 RNA projector、1-stage/2-stage 训练脚本新增 adapter/backbone 分组学习率。
- 未完成或完成不符合原始设计的部分主要是：`diffusion` 注入没有实现，inference pipeline 没打通，protein projector 权重拷贝初始化没有实现，RNA template 搜索链路缺少 protein template 现有的防泄漏护栏。
- 现实现里还存在至少 3 个真正的高风险问题，其中最严重的是“空 RNA 样本会返回错误 key 名，可能覆盖 protein template 特征”，这和你要求的“兼容之前 pipeline”是直接冲突的。

## 1. 主要问题

### 1.1 Critical: 非 RNA 样本返回了错误命名空间的空特征，可能直接覆盖 protein template

位置：

- `protenix/data/rna_template/rna_template_featurizer.py:52-64`
- `protenix/data/rna_template/rna_template_featurizer.py:413-415`
- `protenix/data/pipeline/dataset.py:568-576`

问题：

- `_empty_rna_template_features()` 返回的是 `template_*` 键，而不是 `rna_template_*` 键。
- `RNATemplateFeaturizer.__call__()` 在 `not rna_sequences` 时直接返回这个错误命名空间的字典。
- `dataset.process_one()` 会把返回结果逐项写回 `feat`。

影响：

- 如果样本本身没有 RNA，或者未来你把这套逻辑用于混合样本，`template_aatype/template_distogram/...` 这些 protein template 键会被一组全零的“空 RNA 特征”覆盖。
- 这会破坏你要求的 backward compatibility。
- 当前 RNA-only 脚本里又显式传了 `--data.template.enable_prot_template false`，所以这个 bug 在现有脚本下不一定暴露；但一旦你想回到“兼容之前 pipeline”的目标，这个 bug 会直接炸掉 protein template 路径。

为什么这是高优先级：

- 这不是“效果不好”，而是命名空间错误导致的功能性覆盖。
- 现有测试还把这个错误行为写进了断言里，见 `finetune/test_rna_template_integration.py:90-97`，说明验证本身也没有守住这个边界。

### 1.2 High: RNA template 搜索/索引链路缺少 protein template 现有的防泄漏机制，存在 template leakage 风险

位置：

- RNA: `protenix/data/rna_template/build_rna_template_index.py:173-220`
- Protein: `protenix/data/template/template_utils.py:331-386`
- Protein: `protenix/data/template/template_featurizer.py:441-450`

问题：

- RNA 索引构建只做了简单 pairwise identity 排序，没有：
  - query release date cutoff
  - obsolete PDB 处理
  - duplicate / large subsequence 过滤
  - template dropout
  - 训练时的时间切分保护
- Protein template 这套护栏是现成存在的，`TemplateHitFilter` 里明确做了日期过滤和 duplicate 过滤，训练时还有 template dropout。

影响：

- 训练集中的 RNA 序列很容易直接命中自身或近重复模板。
- 这会造成结构泄漏，模型可能学到“检索回答案”而不是“利用可泛化的 template 先验”。
- 如果后续评估集来自相近时间窗口或数据库未清洗，这个风险更大。

为什么这是“可被攻击”的漏洞：

- 攻击者或数据准备者只要把 query 本身或高度相似模板放进 `rna_database`，当前实现就会优先捡起来。
- 这属于典型的数据层面 information leakage，不是普通精度波动。

### 1.3 High: RNA template 可以被“静默关掉”或“静默变空”，脚本和数据层都没有 fail-fast

位置：

- `protenix/data/rna_template/rna_template_featurizer.py:46-49`
- `protenix/data/pipeline/dataset.py:1099-1124`
- `finetune/finetune_rna_template_1stage.sh:145-153`

问题：

- index 文件不存在时，`_load_rna_template_index()` 直接返回 `{}`，没有报错。
- `get_rna_template_featurizer()` 只有在路径字符串为空时才返回 `None`；如果路径字符串存在但文件不存在，仍会创建 featurizer。
- 训练脚本只检查数据库目录是否存在，不检查 index 文件是否存在，也不检查 index 是否非空。

影响：

- 很容易出现“命令行里明明开了 `--rna_template.enable true`，实际训练全程没有任何有效 template”的情况。
- 这类问题最坏的地方在于不报错，日志也未必明显。
- 你看到 loss 在跑，不代表 RNA template 真参与了训练。

### 1.4 High: 你原始 prompt 要求的 diffusion-level RNA template 注入并没有实现

位置：

- 设计意图：`configs/configs_base.py:142-152`
- 实际注入路径：`protenix/model/modules/pairformer.py:1036-1096`
- 脚本传参：`finetune/finetune_rna_template_1stage.sh:145-153`

问题：

- 配置里写了 `rna_template.injection_mode` 和 `gate_mode`，注释也写了 `"diffusion"` 选项。
- 但真正实现里，RNA template 只进入 `TemplateEmbedder`，也就是只走 `z_init` 路径。
- 训练脚本没有暴露 `rna_template.injection_mode` 参数，也没有把 `gate_mode/gate_init_logit` 传给 `rna_template`。

影响：

- 你 prompt 里明确要求“像 RNA LLM embed 那样 condition on diffusion，也可以加到 z-init，在 script 里实现”。
- 这个目标目前没有完成。
- 因此“RNA template injection 已完成”这个表述不准确，更准确的说法应该是“完成了 z_init-only 的 v1 版注入”。

### 1.5 Medium: 设计里要求的 protein projector 权重拷贝初始化没有真正落地

位置：

- 设计文档：`understanding/rna_template_design.md:223-235`
- 代码注释：`protenix/model/modules/pairformer.py:1004-1005`
- 模型初始化：`protenix/model/protenix.py:130-137`
- checkpoint 加载：`runner/train.py:558-604`

问题：

- 设计文档要求 `linear_no_bias_a_rna` 先拷贝 `linear_no_bias_a` 的权重，再用小 `alpha` 放开 finetune。
- 代码里只留下了注释“这件事会在 Protenix.__init__ after loading checkpoint 里做”。
- 但 `Protenix.__init__` 并没有这段逻辑，`runner/train.py` 的 checkpoint 加载也只是普通 `load_state_dict()`。

影响：

- 当前 `linear_no_bias_a_rna` 其实是随机初始化，不是设计中说的“copy protein projector + 小 gate”。
- 这会降低早期训练稳定性，也说明实现和你的设计语气并不一致。

### 1.6 Medium: `rna_template_block_mask` 没有真正把 RNA 路径限制到 RNA-RNA pair

位置：

- `protenix/model/modules/pairformer.py:1190-1215`

问题：

- 代码只对 `dgram/pseudo_beta_mask/unit_vector/backbone_mask` 乘了 `effective_mask`。
- `aatype` 的 two-way one-hot 展开没有乘 mask，仍然会进入 `linear_no_bias_a_rna`。

影响：

- 即使 `rna_template_block_mask` 为 0，非 RNA token 也会通过 gap one-hot 或 residue-type one-hot 给 RNA projector 一个常量输入。
- 所以“RNA template 只作用于 RNA-RNA block”这个说法并不严格成立。
- 这个问题比 1.1 小，因为它更偏“信号泄漏/污染”，不是直接功能覆盖；但它会让你对 block mask 的解释过于乐观。

### 1.7 Medium: inference pipeline 没有接入 RNA template，pipeline 不是端到端打通

位置：

- `protenix/data/inference/infer_dataloader.py:142-175`
- `protenix/data/inference/infer_dataloader.py:248-272`

问题：

- inference dataloader 只接了 RNA LLM，没有 `rna_template_featurizer`，也没有任何 `rna_template_*` 生成逻辑。
- 训练数据 pipeline 接入了 RNA template，但 inference 没有对应入口。

影响：

- 如果你的目标是“像 protein template 一样完整打通 pipeline”，那当前实现只能算训练侧打通，不是完整打通。
- 这也意味着后续推理/评估脚本无法直接复用同一套 RNA template 逻辑。

## 2. 目标完成度判断

按你原始 prompt 拆开看：

### 2.1 已完成

- RNA template tensor 的离线计算链路已经有了，而且 `pipeline_validation_report.md` 证明最小兼容版 NPZ 可以产出 5 个 core tensor。
- 训练数据 pipeline 已经能把 `rna_template_*` 特征塞进 `input_feature_dict`。
- 模型层已经有 RNA 专用输入投影 `linear_no_bias_a_rna`，并共享 `PairformerStack + linear_no_bias_u`。
- 1-stage / 2-stage 脚本都支持 adapter/backbone 分组学习率。
- `use_rna_template` / `use_rnalm` 开关都已经有。

### 2.2 部分完成

- “像 protein template 一样注入 z pair representation”：完成了，但只是 z-init / TemplateEmbedder 路径。
- “和 RNA LLM 一样可选地 condition on diffusion”：配置里写了，实际没做。
- “保持兼容以前 pipeline”：只在你当前 RNA-only 训练脚本条件下大致成立；一旦回到 protein template 共存或非 RNA 样本，这个结论不成立。

### 2.3 未完成

- RNA template 的 diffusion injection。
- inference 侧 RNA template pipeline。
- 按设计复制 protein projector 初始化 RNA projector。
- protein template 那套 release-date / duplicate / dropout 护栏。

综合判断：

- 如果把目标定义为“做出一个能在 RNA-only 训练里试起来的 v1 原型”，答案是：基本做到了。
- 如果把目标定义为你 prompt 里的“完整项目完成、兼容旧 pipeline、脚本正确、支持 z-init 和 diffusion 两种注入”，答案是：没有完成。

## 3. 和 protein template 的关键差异

### 3.1 RNA template 比 protein template 少掉的能力

- 没有在线检索和 hit 解析，完全依赖预计算 NPZ。
- 没有 release date cutoff。
- 没有 duplicate / large subsequence 过滤。
- 没有 obsolete PDB 处理。
- 没有 template dropout。
- 没有 inference 侧接入。

### 3.2 RNA template 比 protein template 多出来的东西

- 单独的 `linear_no_bias_a_rna` 输入投影。
- 可学习的 `rna_template_alpha`。
- `rna_template_block_mask`。

### 3.3 架构上是否符合你的设计

- “输入端单独 projector，后半段共享 Pairformer trunk”：是。
- “小 alpha 控制 RNA 扰动”：是。
- “copy protein projector 初始化”：不是。
- “block-level fallback，交叉块先空”：只做到了 pair 特征层面，没完全做到输入 projector 层面。

## 4. 训练脚本审查

审查对象：

- `finetune/finetune_rna_template_1stage.sh`
- `finetune/finetune_rna_template_2stage.sh`

### 4.1 脚本做对了什么

- 有 `use_rna_template` 开关。
- 有 `use_rnalm` 开关。
- 有 adapter/backbone LR 拆分。
- 2-stage 脚本能把 Stage 1 和 Stage 2 的 LR 独立配置。
- 当 `use_rna_template=false` 时，可以退回 RNA/DNA LLM-only。

### 4.2 脚本没有达到你原始要求的地方

- 没有暴露 `rna_template.injection_mode`，所以不能在脚本层切到 diffusion injection。
- 没有暴露 `rna_template.gate_mode` / `gate_init_logit`。
- 没有检查 `RNA_TEMPLATE_INDEX` 是否存在。
- 只检查数据库目录，不检查 index 是否为空。
- 当前工作区里 `conda activate protenix` 不能成功，说明脚本在这台机器上不可直接复现执行。

### 4.3 语气是否符合你的原始设计

不完全符合。

你的原始 prompt 明确写的是：

- z-init 和 diffusion 两路都要支持
- 要像 RNA LLM 一样 condition on diffusion
- 要尽量和旧 pipeline 兼容

但脚本和实现实际落下的是：

- 只支持 z-init
- compatibility 只在 RNA-only 的窄场景下大致成立
- 验证环节也没有覆盖最危险的兼容性 bug

## 5. 验证情况

我没有修改任何代码。

我额外尝试运行现有验证脚本：

- `finetune/test_rna_template_integration.py`

结果：

- 在当前工作区直接跑失败，缺少 `ml_collections`、`rdkit`、`biotite`。
- 按脚本声明尝试 `conda activate protenix` 也失败，当前机器上没有名为 `protenix` 的 conda 环境。

这意味着：

- 你现有 `rna_template_injection_code_review.md` 里“GPU validation passed (9/9 tests)”这一结论，我在当前环境下无法复现。
- 更重要的是，现有测试文件本身没有覆盖 1.1 里的命名空间覆盖 bug，且把错误 key 形式写成了预期行为，所以即使将来环境恢复，这份测试也仍然不够。

## 6. 建议修复优先级

P0：

- 修正 `_empty_rna_template_features()` 的 key 命名空间，确保始终返回 `rna_template_*`。
- 增加 mixed sample / no-RNA sample / protein-template-enabled 的回归测试。

P1：

- 给 RNA template 检索链路补上 self-hit / duplicate / release-date 护栏。
- 把 `template_database_dir`、`template_index_path`、index 非空检查改成 fail-fast。
- 明确日志里打印“本 batch / 本样本是否真正命中 RNA template”。

P2：

- 实现 `rna_template.injection_mode=diffusion`。
- 实现 protein projector -> RNA projector 的 copy init。
- 补 inference dataloader 的 RNA template 接入。

P3：

- 如果坚持 block-mask 语义，应该把 residue-type 通道也一并 mask，或者在文档里承认当前只是“pair geometry 被 mask，restype 通道未 mask”。

## 7. 最终结论

这次实现不是“没做成”，但也远没到“目标已经完整完成”的程度。

更准确的判断是：

- 你已经有了一个可继续迭代的 RNA template v1 原型。
- 它完成了离线 tensor 生成、训练数据接入、TemplateEmbedder 内的 z-init 注入和基础训练脚本。
- 但它还没有达到你 prompt 里要求的完整目标，而且当前实现存在会影响兼容性和数据正确性的高风险问题。

如果只回答一个问题：

- “RNA template inject 的目标是否已被完成？”

答案是：

- 没有完整完成。
- 目前完成的是“RNA-only 训练场景下的 z-init 原型版”，不是“完整、稳健、兼容旧 pipeline、支持 diffusion 的正式版”。
