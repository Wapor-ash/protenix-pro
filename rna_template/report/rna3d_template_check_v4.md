# RNA Template Pipeline v4 审查报告

**Date**: 2026-03-15

**Scope**: 只读审查，不修改任何业务代码。审查范围覆盖：

- `finetune` 主训练脚本是否已切到 online RNA template
- `inference` 与 `training` pipeline 是否按脚本选项一致
- temporal filtering 是否按设计工作
- 关闭 RNA template 时回退是否正常
- RNA template projector 初始化与 checkpoint 修复是否正确
- 与 RNALM 既有设计的一致性
- 当前 pipeline 的漏洞、弱点与测试缺口

---

## 结论

和上一个版本相比，**核心目标大部分已经完成**：

- 主 `finetune` 脚本已经切到 `search_results_path + cif_database_dir` 的 online mode
- inference 下 `rnaSequence/rnaChain` 不再错误漏掉不含 `U` 的合法 RNA；我用最小样例实际复现确认 `rnaSequence("ACG")` 现在会进入 RNA template 查找
- RNA template projector 的 `protein` / `zero` 两种初始化行为与脚本语义一致
- 关闭 RNA template 的训练回退路径从代码结构看是正常的

但从“是否已经完全闭环、是否所有 pipeline 都合理”这个角度，结论不是“100% 完成”，而是：

**训练主路径基本完成；推理脚本入口和综合验证入口仍有关键缺口；在线 temporal 设计本身也仍保留一个召回上的结构性弱点。**

---

## 最高优先级问题

### 1. `infer_rna.sh` 仍把 `rnalm.enable` 写死为 `true`，导致 template-only checkpoint 的推理入口不成立

这是当前最关键的剩余问题。

虽然你已经在训练侧支持了：

- `--use_rnalm false`
- `--use_rna_template true`

而且仓库里已经有一份实际跑出来的 template-only 配置：

- `output/1stage_rna_template_alr0.005_blr0.0001/.../config.yaml`
- 其中 `rnalm.enable: false`
- 同时 `rna_template.enable: true`
- 且 `search_results_path` / `cif_database_dir` 已正确写入

但 `infer_rna.sh` 仍然无条件传：

```bash
--rnalm.enable true \
```

位置：

- `infer_rna.sh:225-238`

而 `runner/inference.py` 又显式要求：如果 `rnalm.enable=true`，checkpoint 必须真的包含 RNALM 权重；否则直接报错：

- `runner/inference.py:198-200`

这意味着：

- 如果你训练的是 **RNA template only** checkpoint
- 即使 RNA template 权重存在、配置也正确
- 只要走 `infer_rna.sh`，推理入口仍可能因为 RNALM 校验被直接挡住

这和你的验收要求冲突：

- “确保 inference pipeline 正确”
- “所有其他 pipeline 按照 script 里的选项能正确”

当前答案是：**不完全正确**。脚本层面，template-only inference 还没有闭环。

---

## 中优先级问题

### 2. `finetune_rna_template_validate.sh` 仍然混合了旧 offline 假设，导致“综合验证脚本”与在线设计不完全一致

你现在的训练主入口已经转向 online mode，但综合验证脚本仍保留了不少 offline-era 验收条件：

- 仍定义 `INDEX_PATH`
- 仍检查 `rna_template_index.json`
- 仍检查 cross-template `.npz`
- 仍把“构建离线模板文件”纳入主验证逻辑

相关位置：

- `finetune_rna_template_validate.sh:35-40`
- `finetune_rna_template_validate.sh:173-178`
- `finetune_rna_template_validate.sh:191-216`
- `finetune_rna_template_validate.sh:315`

而当前最新的综合验证日志也说明这条验证链并不稳定：

- `output/rna_template_validate/validate.log`

该日志里，Phase 1 在 `02_build_rna_templates.py` 阶段失败，报错写入：

```text
FileNotFoundError: ... /rna_database/templates/4v4a_template.npz.meta.json
```

这带来的问题不是“训练一定坏了”，而是：

- 你的 **online 训练主路径** 已经不依赖这些 `.npz`
- 但你的 **综合验证脚本** 还在把离线模板构建当成主成功标准
- 结果是“实际主训练可用”和“验证脚本通过”之间出现分裂

所以目前 `rna3d_version4.md` 里“全面落地并完整验证”的说法，**不能完全由当前综合验证工件支持**。

更准确的说法应该是：

- online training 主路径基本落地
- 但 full validation 脚本仍带着一部分旧 pipeline 假设

---

### 3. `run_pipeline.sh` 仍保留全局 `release_date_cutoff` 预裁剪，会削弱 online per-query temporal filtering 的召回

当前 online 设计的正确核心是：

- search 产生较大候选池
- 真正的 anti-leakage 过滤在运行时按 query 的 `release_date - 60d` 做

但数据准备脚本仍支持并实际使用全局 cutoff：

- `run_pipeline.sh:115-122`
- `run_pipeline.sh:168-181`
- `03_mmseqs2_search.py:969-980`

这不会导致数据泄漏，方向上是保守的；但它会造成另一个问题：

- 如果 `search_results.json` 在离线阶段已经被全局 cutoff 提前裁掉
- 那么较新的 query 在 online mode 下也看不到本来“按 per-query 时间是允许”的模板

结论是：

- **正确性上基本安全**
- **召回上仍然偏保守**

这属于当前 pipeline 的设计弱点，而不是立即阻塞训练的 bug。

---

### 4. 自动化测试仍主要覆盖旧 offline 路径，没有把新的主入口脚本真正锁住

`test_online_featurizer.py` 确实覆盖了 online featurizer 本体，这部分是好的。

但 `finetune/test_rna_template_integration.py` 仍主要围绕：

- `template_index_path`
- offline fail-fast
- 模块级 forward/init

相关位置例如：

- `finetune/test_rna_template_integration.py:61`
- `finetune/test_rna_template_integration.py:118`
- `finetune/test_rna_template_integration.py:383-404`
- `finetune/test_rna_template_integration.py:662-689`

它没有覆盖两个当前最关键的新入口行为：

- `infer_rna.sh` 在 template-only 模式下是否能正确关闭 RNALM
- `finetune_rna_template_1stage.sh / 2stage.sh` 生成的最终 config 是否稳定落成 online mode

这意味着现在的回归风险主要不在 featurizer 内部，而在“脚本入口和配置落盘”这一层。

---

## 已完成且实现合理的部分

### 1. 主训练脚本已经切到 online mode

我核对了主脚本：

- `finetune/finetune_rna_template_1stage.sh:155-163`
- `finetune/finetune_rna_template_2stage.sh:159-167`
- `finetune/finetune_rna_template_validate.sh:436-443`

现在它们都传：

- `--rna_template.search_results_path`
- `--rna_template.cif_database_dir`

不再依赖 `template_index_path` 作为主训练入口。

此外，我还直接检查了一份实际训练输出：

- `output/1stage_rna_template_alr0.005_blr0.0001/.../config.yaml`

其中明确显示：

- `rna_template.enable: true`
- `search_results_path` 已设置
- `cif_database_dir` 已设置
- `template_index_path: ''`
- `rnalm.enable: false`

这说明：

- 不是只有脚本文本改了
- 实际落盘配置也确实进入了 online + template-only 训练模式

---

### 2. inference 下 `rnaSequence("ACG")` 进入 RNA template 查找的问题已经修好

当前实现位于：

- `protenix/data/rna_template/rna_template_featurizer.py:979-1005`

逻辑已经变成：

- `rnaSequence/rnaChain` 直接接受
- `dnaSequence/dnaChain` 仅在含 `U` 时重分类为 RNA

我在 `protenix` 环境里做了最小复现：

- 构造 `bioassembly_dict = {'sequences': [{'rnaSequence': {'sequence': 'ACG'}}]}`
- monkeypatch `get_rna_template_features`
- 调用 `RNATemplateFeaturizer(..., inference_mode=True)`

结果实际捕获到：

```python
rna_sequences == {'1': 'ACG'}
```

所以这个点现在是**已修复且已验证**。

---

### 3. RNA template projector 的 `protein` / `zero` 初始化语义是正确的

实现位置：

- `protenix/model/modules/pairformer.py:990-1018`
- `protenix/model/protenix.py:281-329`
- `runner/train.py:563-580`
- `runner/inference.py:206-222`

我做了最小实例化验证，结果如下：

- `projector_init="protein"`：
  - `linear_no_bias_a_rna.weight` 与 protein projector 权重一致
  - `rna_template_alpha` 存在，初值约为 `0.01`
- `projector_init="zero"`：
  - `linear_no_bias_a_rna.weight` 为全零
  - 不创建 `rna_template_alpha`

这和脚本语义是一致的，也和 checkpoint repair 逻辑一致。

---

### 4. 关闭 RNA template 的训练回退路径是正常的

主训练脚本在 `USE_RNA_TEMPLATE=false` 时会只传：

```bash
--rna_template.enable false
```

相关位置：

- `finetune/finetune_rna_template_1stage.sh:164-165`
- `finetune/finetune_rna_template_2stage.sh:168-169`
- `infer_rna.sh:174-175`

训练数据侧工厂函数也会在 disabled 时直接返回 `None`：

- `protenix/data/pipeline/dataset.py:1103-1105`

所以“不开 RNA template 回退正常”在训练/数据侧基本成立。

---

## 和 RNALM 设计的一致性判断

目前的一致性状态是：

- **已对齐的部分**：
  - inference 下 `rnaSequence/rnaChain` 不再因“没有 U”被漏掉
  - `dnaSequence/dnaChain` 含 `U` 时会被纳入 RNA template 查找

- **保留差异的部分**：
  - RNALM 有 `Reverse-RNA-First`，会把纯 `ACGT` 的 RNA-labeled entity 重分到 DNA 分支
  - RNA template 这里没有对称的 DNA template 分支，因此当前实现选择“RNA-labeled entity 直接接受”

我认为这个差异**在工程上是合理的**，因为：

- 这里不存在 DNA template projector / DNA structural template 分支
- 如果照搬 RNALM 的 reverse rule，只会把一部分输入直接排除，而不是转到另一个可用结构分支

所以这不再是我本轮的主要 bug finding；但建议在设计文档里明确写成“有意偏离”，否则后续维护者容易误以为这是遗漏。

---

## 我实际完成的验证

### 静态核对

- 核对了主训练脚本、验证脚本、推理脚本
- 核对了 dataset / infer_dataloader / featurizer / projector init / checkpoint repair 关键路径
- 核对了 `rna3d_version4.md` 的实现声明与当前代码是否一致

### 轻量运行验证

- `bash -n` 检查以下脚本语法通过：
  - `finetune_rna_template_1stage.sh`
  - `finetune_rna_template_2stage.sh`
  - `finetune_rna_template_validate.sh`
  - `infer_rna.sh`
- 在 `protenix` 环境中实例化 `TemplateEmbedder`，确认 `protein` / `zero` init 行为正确
- 在 `protenix` 环境中最小复现确认 `rnaSequence("ACG")` 现在会进入 RNA template 查找
- 读取已有 1-stage 实际输出 `config.yaml`，确认 online mode 已经真正落盘

### 已有工件核对

- `output/1stage_rna_template_alr0.005_blr0.0001/.../checkpoints/4.pt` 说明至少有一次短程训练已成功启动并保存 checkpoint
- `output/rna_template_validate/validate.log` 显示综合验证脚本的 pipeline phase 仍存在失败点
- `output/rna_template_validate/training_validate.log` 是更早一次 offline/index 路径的训练记录，不能作为当前 v4 online 主路径的直接证据

---

## 最终判断

如果验收标准是：

- “主 finetune 是否已经按 online temporal RNA template 设计切过去”

答案是：**是，基本完成。**

如果验收标准是：

- “training / inference / validate 三条主入口是否都已经完全闭环，且所有脚本选项都一致可用”

答案是：**还没有。**

当前最重要的剩余问题是：

1. `infer_rna.sh` 还不能正确承载 template-only inference，因为 `rnalm.enable` 被写死为 `true`
2. `finetune_rna_template_validate.sh` 仍混合旧 offline 验收逻辑，综合验证结果与 online 主路径不完全一致

当前最主要的弱点是：

1. `run_pipeline.sh` 的全局 cutoff 会保守地损失 online mode 召回
2. 自动化测试没有把新的在线入口脚本真正锁住，后续回归风险仍在脚本层

---

## 审查结论摘要

**已完成且合理**

- 主训练脚本切到 online mode
- inference 下 RNA entity 识别 bug 已修复
- projector init / checkpoint repair 语义正确
- 训练侧关闭 RNA template 回退正常

**仍未完全完成**

- template-only inference 脚本入口未闭环
- full validation 脚本仍未完全对齐 online 设计

**当前主要弱点**

- 全局 cutoff 降低模板召回
- 测试覆盖没有锁住新的主入口脚本

