# RNA Template Pipeline 审查报告

**Date**: 2026-03-15

**Scope**: 只读审查，不修改任何代码。审查范围覆盖 RNA template online/offline pipeline、temporal filtering、training/inference 接入、RNA template projector 初始化、与 RNALM 既有设计一致性、脚本实际可用性，以及回退路径。

## 结论

当前仓库里，`RNA template online featurizer` 本身已经落地，核心代码不是空实现；`PDB API fallback`、`per-query temporal filtering`、`self-hit exclusion`、`RNA projector init/checkpoint repair`、`disabled fallback`、`inference fail-fast` 这些关键点大体成立。

但从“需求是否已经完成”这个角度看，结论不是“完全完成”，而是“核心模块完成，但主 pipeline 和脚本层没有完全切到设计目标”。最关键的问题是：默认 finetune/validate 脚本仍然走离线 `template_index + NPZ` 路径，没有真正把训练主入口切到 `search_results + cif_database_dir` 的 online per-hit filtering 模式。因此，报告 `rna3d_version3.md` 里“整体已完成并已切到 online pipeline”的表述，和仓库当前默认训练入口并不一致。

## 高优先级问题

### 1. 主训练脚本仍然使用离线 NPZ/index，核心需求“online query-time filtering”没有在默认训练入口落地

这是本次最重要的问题。

需求里强调的目标，是把 RNA template 从离线预构建 NPZ 的思路，切到 protein template 同款的在线模式：运行时按 query 的 `pdb_id/release_date` 做 `self-hit` 和 `temporal` 过滤，然后逐 hit 从 CIF 构建模板特征并回填。

但主训练脚本和验证脚本仍然配置的是离线路径：

- [finetune_rna_template_1stage.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/finetune/finetune_rna_template_1stage.sh#L153)
- [finetune_rna_template_2stage.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/finetune/finetune_rna_template_2stage.sh#L157)
- [finetune_rna_template_validate.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/finetune/finetune_rna_template_validate.sh#L436)

这些脚本都传的是：

- `--rna_template.template_database_dir`
- `--rna_template.template_index_path`

而不是 online 模式需要的：

- `--rna_template.search_results_path`
- `--rna_template.cif_database_dir`

与之相对，真正走 online 模式的是测试脚本：

- [test_online_gpu.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_online_gpu.sh#L110)

这意味着：

- `RNATemplateFeaturizer` 的 online 实现是存在的。
- 但用户默认执行的 finetune pipeline 还没有切过去。
- 因此“确保 finetune 按需求查找 RNA template”这个目标，在默认训练脚本层面没有完成。

这不是文档问题，而是实际行为问题。只要继续用当前 `finetune/` 主脚本，训练就不会得到你要求的 online per-query temporal filtering 行为。

————————

TODO：帮我修改并核实能够按照online训练！！
执行并测试

### 2. inference 模式下，`rnaSequence/rnaChain` 里不含 `U` 的合法 RNA 会被 `RNATemplateFeaturizer` 直接漏掉

`RNATemplateFeaturizer.__call__()` 在 inference 路径里对 RNA entity 的识别逻辑是：

- [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L982)
- [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L989)

它对 `rnaSequence/rnaChain` 不是按类型直接接收，而是额外要求：

- 序列里出现 `U`
- 或出现非 `ACGTN` 字符

结果是像 `ACG`, `GGCAA`, `AGCAA` 这类完全合法、但刚好不含 `U` 的 RNA 序列，会被当成“不是 RNA”，从而完全不进入 RNA template 查找。

我在 `protenix` 环境里做了最小复现，结论是：

- `bioassembly_dict["sequences"] = [{"rnaSequence": {"sequence": "ACG"}}]`
- 调用 `RNATemplateFeaturizer.__call__(..., inference_mode=True)`
- 实际不会调用 `get_rna_template_features()`
- 输出仍是全空模板，`rna_template_block_mask.sum() == 0`

这和 RNALM 的既有设计也不一致。RNALM 在 inference 下使用的是“RNA-First / Reverse-RNA-First”分类逻辑：

- [rnalm_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rnalm/rnalm_featurizer.py#L577)
- [rnalm_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rnalm/rnalm_featurizer.py#L582)

RNALM 至少会按实体类型进入分类流程；RNA template 这里则直接跳过。对你的要求“确保和 RNA LLM 的之前设计保持一致”，这一点目前不满足。

——————

TODO：帮我按照RNALM的设计一致！！实现并核实
### 3. 用户侧 inference 脚本没有把 RNA template 暴露出来，推理 pipeline 仍不完整

底层 inference 支持已经接上：

- [infer_dataloader.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/inference/infer_dataloader.py#L194)
- [runner/inference.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/runner/inference.py#L206)

而且还做了 checkpoint fail-fast，避免推理时静默使用随机初始化 projector。

但仓库里用户实际会调用的 RNA inference 脚本 [infer_rna.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/infer_rna.sh#L1) 只暴露了 RNALM 参数，没有任何 `rna_template.*` 选项，也没有 `search_results_path/cif_database_dir` 或 `template_index_path/template_database_dir`。

这意味着：

- 底层推理代码支持 RNA template。
- 但现有用户入口脚本没有把它串起来。
- 所以“确保 inference pipeline 正确”只能说底层代码基本接通，用户脚本级 pipeline 还不完整。

## 中优先级问题与弱点

### 4. `run_pipeline.sh` 仍然保留全局 `release_date_cutoff` 思路，会削弱 online per-query filtering 的召回

在 online 设计里，理想做法是：

- 离线只做 search，尽量保留较大的候选池。
- 运行时按 query 自己的 `release_date - 60 days` 去筛。

但当前数据准备脚本 [run_pipeline.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L115) 仍支持并鼓励在离线阶段传 `--release_date_cutoff`，并且 [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L969) 会先把 catalog 全局裁掉。

这不会造成泄漏，反而是保守的；但如果这批 `search_results.json` 被拿去做 online mode，较新的 query 将看不到本来应该允许使用的较新模板，召回会被全局 cutoff 提前压低。

所以现在的状态更像是：

- 代码层面支持 online per-query temporal filtering。
- 但数据准备和默认脚本层面，仍然带着“全局 cutoff + 离线模板库”的旧思路。

## 已完成且实现合理的部分

### 1. online featurizer 核心实现已经存在，而且本地在线单测通过

核心代码位于：

- [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L475)

关键机制包括：

- sequence -> hit list lookup
- `query_release_date - 60 days` temporal cutoff
- self-hit exclusion
- RNA3DB metadata -> PDB API fallback -> conservative reject
- per-hit CIF build
- build 失败时继续尝试下一个 hit
- stack 到 `max_templates`

我实际跑了：

- [test_online_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_online_featurizer.py#L1)

结果：

- `31 passed, 0 failed`

这说明 online featurizer 的局部实现是实的，而且当前仓库中的 `search_results.json + PDB_RNA` 组合至少能跑通一批真实样本。

### 2. RNA template projector 初始化、checkpoint 修复、推理 fail-fast 设计是合理的

相关实现：

- [pairformer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/model/modules/pairformer.py#L990)
- [protenix.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/model/protenix.py#L281)
- [train.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/runner/train.py#L563)
- [inference.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/runner/inference.py#L206)

我认为这里的设计是成立的：

- `projector_init="protein"` 时，构造时复制 protein projector，并创建 `rna_template_alpha`
- `projector_init="zero"` 时，RNA projector 零初始化，不创建 alpha
- checkpoint 已经带 RNA projector 时，不会被后续 repair 覆盖
- inference 时如果 checkpoint 没有 RNA template projector，会直接报错，不会静默推理

这部分和你的需求“RNA template projector 的 init 按照 script 选项正确 init”“不开 RNA template 回退正常”基本一致。

### 3. training 路径里的 query metadata 传递是对的

`release_date` 会进入 bioassembly/sample metadata：

- [parser.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/core/parser.py#L693)
- [dataset.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/pipeline/dataset.py#L347)

`RNATemplateFeaturizer` 在 training 时会读取：

- `bioassembly_dict["pdb_id"]`
- `bioassembly_dict["release_date"]`

位置：

- [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L967)

所以从数据流机制上，training 做 per-query temporal filtering 的必要元数据是通的。

## 机制梳理

### Training 机制

当前训练侧真实链路是：

- `parser` 产出 `bioassembly_dict`，其中包含 `pdb_id/release_date/sequences`
- `dataset.get_rna_template_featurizer()` 决定创建 online 还是 offline featurizer
- `BaseSingleDataset.process_one()` 在 crop 后调用 `rna_template_featurizer`
- `RNATemplateFeaturizer` 产出 `rna_template_*`
- `TemplateEmbedder` 用 `linear_no_bias_a_rna` 和共享 `PairformerStack` 做 RNA template 编码
- `Protenix.reinit_rna_projector_from_protein()` 在 load checkpoint 后修复 RNA projector 初始状态

这条链路在代码层面是通的。

### Inference 机制

当前推理侧真实链路是：

- `SampleDictToFeatures` 把 JSON sample 转为 `atom_array/token_array`
- `InferenceDataset` 可选创建 `RNATemplateFeaturizer`
- `RNATemplateFeaturizer(..., inference_mode=True)` 生成 `rna_template_*`
- `runner/inference.py` 在 load checkpoint 后校验 RNA template 权重是否存在

所以 inference 底层机制存在，但脚本入口没有完整暴露，且 inference RNA entity 分类存在前述 bug。

## 我实际做的验证

### 已完成验证

- 阅读需求与现有报告：
  - [online_rna_template_pipeline_proposal.md](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/online_rna_template_pipeline_proposal.md)
  - [fix_pipline_copy.txt](/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/fix_pipline_copy.txt)
  - [rna3d_version3.md](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_version3.md)
- 静态审查 training/inference/model/data/script 关键路径。
- 运行 online featurizer 自测，结果 `31/31` 通过：
  - [test_online_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/test_online_featurizer.py#L1)
- 复现 inference 下 `rnaSequence="ACG"` 被 `RNATemplateFeaturizer` 漏掉的问题。
- 对比 RNALM inference 下同类输入的分类逻辑，确认两者不一致。

### 受环境限制未完成的验证

- [test_rna_template_integration.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/finetune/test_rna_template_integration.py#L1) 在当前沙箱里触发 OpenMP shared-memory 权限问题，未能作为最终证据使用。
- `py_compile` 写 `__pycache__` 时遇到当前目录权限限制，未完成这一项语法落盘检查。
- 没有重跑完整 GPU training，因为本次任务是只读审查，而且当前沙箱对输出目录写入也有限制。

## 最终判断

如果你的验收标准是“核心 online featurizer 有没有实现”，答案是：`有，而且局部验证通过`。

如果你的验收标准是“整个 training/inference pipeline 是否已经按需求完整切换到 online temporal RNA template pipeline”，答案是：`还没有`。

当前最需要修正的不是 `RNATemplateFeaturizer` 主体，而是：

- 让主 finetune/validate/inference 脚本真正使用 online mode
- 把 inference 的 RNA entity 识别逻辑对齐到 RNALM RNA-First 设计

