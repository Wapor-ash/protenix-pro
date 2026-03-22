# Protenix RNA/DNA LLM Embed Review

## Scope

Reviewed against:

- `report/per_group_lr_and_embedding_fix_report.md`
- `report/finetune_scripts_and_code_review_report.md`
- `prompts/report/rna_dna_embed_toggle_implementation_report.md`

Reviewed code paths:

- config: `configs/configs_base.py`
- data: `protenix/data/pipeline/dataset.py`, `protenix/data/inference/infer_dataloader.py`, `protenix/data/rnalm/rnalm_featurizer.py`
- model: `protenix/model/protenix.py`, `protenix/model/modules/embedders.py`, `protenix/model/modules/diffusion.py`
- training: `runner/train.py`
- scripts: `finetune/finetune_1stage.sh`, `finetune/finetune_2stage.sh`, plus older two-stage scripts still present in repo

## Executive Verdict

核心能力大体已经落地了：

- RNA/DNA 独立开关已实现
- `separate_dna_projection=True` 下的分离投影已实现
- `injection_mode=input|diffusion|both` 已实现
- 1-stage / 2-stage 的联合微调框架已实现
- 单条样本 embedding 缺失时，`RiNALMoFeaturizer` 已从静默补零改为报错

但如果按你问的标准去判断: “是不是已经把可选注入、可选是否加入、可选是否联合微调、找不到 embed 就报错退出这些逻辑都完整做完了”，结论是:

**还没有完全闭环。**

当前实现里还存在几处会让某些配置组合失真、冻结错误模块、或者把“早失败”退化成“首个 forward 才炸”的漏洞。下面按严重度列出。

## Findings

### 1. High: repo 里仍有旧两阶段脚本，和新的 LR 接口已经脱节，会 silently 跑错训练策略

**Evidence**

- `runner/train.py:442-448` 现在只读取 `stage2_adapter_lr` / `stage2_backbone_lr`
- 但以下脚本仍在传老参数 `stage2_lr`，这个值不会被消费:
  - `run_two_stage_training_rna_loss.sh:169-174`
  - `rna_2stage_fast.sh:120-125`
  - `rna_2stage_slow.sh:85-90`
- 同一批旧脚本还把 `--two_stage.adapter_keywords` 强行写死为 `"rnalm_projection"`:
  - `run_two_stage_training_rna_loss.sh:174`
  - `rna_2stage_fast.sh:125`
  - `rna_2stage_slow.sh:90`

**Impact**

- 这些旧脚本里配置的 `stage2_lr` 实际上被忽略，Stage 2 会退回到 `runner/train.py:442-448` 的默认解析逻辑。
- 如果开启 gate、input injection、separate RNA/DNA projection，这些脚本的 Stage 1 仍然只会训练 `rnalm_projection`，不会训练:
  - `rna_projection`
  - `dna_projection`
  - `linear_rna_llm`
  - `linear_dna_llm`
  - `rnalm_alpha_logit`
  - `rnalm_gate_mlp`

**Why this matters for your goal**

这直接破坏了“可以选择在哪里注入、要不要联合微调”的可靠性。不是主代码不支持，而是 repo 内仍然存在会把新能力绕回旧行为的启动入口。

### 2. High: `fusion_method=concat` 新增的参数没有被纳入 adapter 组，冻结 backbone 时这条新分支训练不到

**Evidence**

- `protenix/model/modules/diffusion.py:85-92` 在 `fusion_method=="concat"` 时新建了:
  - `layernorm_s_concat`
  - `linear_no_bias_s_concat`
- 但 adapter 关键字默认值只包含:
  - `configs/configs_base.py:66`
  - `runner/train.py:256-259`
- 这里并没有覆盖 `layernorm_s_concat` / `linear_no_bias_s_concat`

**Impact**

- 1-stage per-group LR 下，如果 `backbone_lr=0`，这两个 concat 专用模块会被分到 backbone 组并被冻结。
- 2-stage 的 Stage 1 也一样会把它们冻住。
- 结果是用户虽然可以选 `fusion_method=concat`，但在“只训新模块”的训练设定下，这个新融合分支并没有被真正训练。

**Why this matters for your goal**

这属于“可配置但不完全可训练”。如果你希望这个 project 支持用户自由切换 fusion 方案并稳定 finetune，目前这块还不算完成。

### 3. Medium: “找不到 embed 就报错退出” 还没有在所有入口做到真正 fail-fast

**Evidence**

- `RiNALMoFeaturizer` 构造器本身已经是 fail-fast:
  - `protenix/data/rnalm/rnalm_featurizer.py:80-123`
- 但训练数据入口 `get_rnalm_featurizer()` 在“双路径都没配”的情况下不会抛错，而是直接返回 `None`:
  - `protenix/data/pipeline/dataset.py:1036-1041`
- 推理入口更明显，日志仍写着“回退到 random embeddings”:
  - `protenix/data/inference/infer_dataloader.py:179-183`
- 实际上模型并没有 random fallback；后面只会在 forward 时因为 feature key 缺失而报错:
  - combined path: `protenix/model/protenix.py:452-459`
  - input path: `protenix/model/modules/embedders.py:190-195`

**Impact**

- 某些配置下不会在 dataloader/featurizer 初始化阶段立即失败，而是延迟到第一个 batch forward 才失败。
- inference 的 warning 文案是误导性的，实际不会“回退到 random embeddings”。

**Why this matters for your goal**

如果标准是“embed 找不到就立刻明确退出”，现在只能说做到了大部分，不是全入口一致。

### 4. Medium: `stage2_backbone_lr=0` 实际不可表达，Stage 2 不能显式保持 backbone 冻结

**Evidence**

- `runner/train.py:446-448`

当前逻辑是:

- `stage2_backbone_lr <= 0` 时，直接回退为 `stage2_adapter_lr`

这意味着:

- `-1` 作为“未设置”会回退，合理
- 但 `0` 这个“显式冻结”也会被当成“未设置”，被强行改成非零

**Impact**

- 你不能配置一个真正的 “Stage 2 仍只训 adapter、backbone 保持 0 LR” 模式。
- 所谓“是否联合微调”在 2-stage 语义下并不完全自由；它更接近“Stage 2 默认必然 joint”。

**Why this matters for your goal**

如果你的设计目标是“联合微调可选”，那这里还少了一档合法配置。

### 5. Medium: `use_rna_embed=false && use_dna_embed=false` 并不总是等价于 `rnalm.enable=false`

**Evidence**

- 数据层会直接禁用 featurizer:
  - `protenix/data/pipeline/dataset.py:1017-1021`
- 但模型层并不会同步关闭 `rnalm_enable`:
  - `protenix/model/protenix.py:146-162`
  - `protenix/model/modules/embedders.py:77-105`

在 `separate_dna_projection=False` 的 combined 路径下，模型仍会创建共享投影层，并在 forward 时要求:

- `rnalm_token_embedding`
  - `protenix/model/protenix.py:452-459`
  - `protenix/model/modules/embedders.py:190-195`

**Impact**

- 报告里“`use_rna_embed/use_dna_embed` 都关掉等价于 `rnalm.enable=False`”这个说法并不严格成立。
- 它只在某些路径上成立，尤其是 `separate_dna_projection=True` 时比较接近 no-op。
- 在 combined 路径下，这个配置会变成“数据不产 embedding，但模型还在要 embedding”。

**Why this matters for your goal**

这会让“可选择不加入 embed”这个能力在部分配置组合上不一致。

### 6. Medium: conditioning drop 对 `s_rnalm` 的屏蔽不完整，缓存路径下仍可能带入 RNA/DNA conditioning

**Evidence**

- `protenix/model/modules/diffusion.py:167-179` 只有在 `pair_z is None` 时才会在 `use_conditioning=False` 下把 `s_rnalm` 归零
- 训练时常见路径会直接传入缓存好的 `pair_z`:
  - `protenix/model/protenix.py:996-1004`

**Impact**

- 当 `drop_conditioning` 命中时，`s_rnalm` 不一定真的被一起 drop 掉。
- 这会让 conditioning dropout 的语义变得不纯，RNA/DNA 注入分支和 trunk conditioning 的正则化不一致。

**Why this matters for your goal**

这不是“开关没实现”，但它会影响注入分支的训练行为，属于需要知道的残余风险。

## What Is Actually Implemented Correctly

下面这些点，从当前代码看是成立的:

### 1. RNA/DNA 分离投影

已实现。

- diffusion 注入层:
  - `protenix/model/protenix.py:165-184`
- input 注入层:
  - `protenix/model/modules/embedders.py:87-101`

当 `separate_dna_projection=True` 且 toggle 打开时，RNA 和 DNA 走的是两套独立线性层。

### 2. 可以单独开关 RNA / DNA embed

已实现，但“都关掉”这一极端组合仍有上面第 5 条漏洞。

- config:
  - `configs/configs_base.py:110-137`
- featurizer:
  - `protenix/data/rnalm/rnalm_featurizer.py:68-69`
  - `protenix/data/rnalm/rnalm_featurizer.py:653-659`
- model:
  - `protenix/model/protenix.py:149-151`
  - `protenix/model/modules/embedders.py:82-85`

### 3. 可以选择注入位置

已实现。

- input 注入:
  - `protenix/model/modules/embedders.py:80-110`
  - `protenix/model/modules/embedders.py:166-199`
- diffusion 注入:
  - `protenix/model/protenix.py:153-162`
  - `protenix/model/protenix.py:400-474`

`input` / `diffusion` / `both` 三个模式都有对应路由。

### 4. 可以选择是否 joint finetune

部分实现。

- 1-stage per-group:
  - `runner/train.py:321-362`
- 2-stage:
  - `runner/train.py:364-501`

如果你的意思是:

- “只训 adapter” -> 可以
- “adapter + backbone 一起训” -> 可以
- “两阶段先只训 adapter，再 joint” -> 可以

但如果你的意思包括:

- “两阶段里 Stage 2 也允许继续只训 adapter / backbone 维持 0”

那当前还不完整，见 Finding 4。

### 5. 单条样本/单条序列 embedding 找不到就报错

这部分基本已实现。

- 构造阶段缺路径/缺目录/缺 CSV:
  - `protenix/data/rnalm/rnalm_featurizer.py:80-123`
- 运行阶段 sequence 对不上 manifest 或 `.pt` 无法加载:
  - `protenix/data/rnalm/rnalm_featurizer.py:505-538`

这一点比旧逻辑明显更安全。

## Bottom Line

如果问题是“这个 proj 是不是已经实现了你想要的 DNA/RNA LLM embed 方案”，答案是:

**大方向是，而且主体代码已经具备你要的能力。**

如果问题是“是不是已经把这些能力做到所有入口一致、所有脚本都不踩坑、所有配置组合都闭环”，答案是:

**还没有。**

最需要优先修的不是主干投影/注入代码，而是以下三类边角:

1. 清理或修复 repo 内仍然存在的旧 two-stage 脚本
2. 把 `concat` 融合新增参数纳入 adapter 分组
3. 把 fail-fast 统一到 dataset/inference 入口，而不是部分场景延迟到首个 forward

## Testing Gaps

从当前仓库内容看，以下组合没有看到足够扎实的覆盖:

- `fusion_method=concat` + `backbone_lr=0`
- `fusion_method=concat` + `two_stage.enable=true`
- inference 场景下 embedding 路径完全缺失
- `use_rna_embed=false && use_dna_embed=false`
- 旧 two-stage 脚本是否仍被文档或团队流程引用

以上几项如果不补，后续很容易出现“报告说支持，实际某个入口一跑就偏”的情况。
