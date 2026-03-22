# Protenix RNA/DNA LLM Integration Code Review

日期：2026-03-11  
审查范围：`/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix`  
约束：未修改业务代码；仅做静态审查与语法检查  
额外验证：已用 `/inspire/ssd/project/sais-bio/public/ash_proj/conda/envs/protenix/bin/python -m py_compile` 检查本次相关 Python 文件，语法通过

---

## 1. Executive Summary

结论先说：

1. **你想要的核心训练逻辑，基本已经实现。**
   在 `rnalm.enable=true`、`rnalm.injection_mode="diffusion"`、`rnalm.separate_dna_projection=true` 时，当前代码确实走的是：
   - RNA embedding `[2048] -> rna_projection -> c_s`
   - DNA embedding `[1024] -> dna_projection -> c_s`
   - 二者相加得到 `s_rnalm`
   - 在 `DiffusionConditioning` 中以 `add` 方式加到 `s_trunk`
   - 没有再走 DNA zero-padding 到 RNA 维度的旧路径

2. **如果你的目标是“像 ESM 一样加载，但注入到 diffusion”，当前训练链路是对的；如果你的目标是“像 ESM 一样加到 S_input”，那只有 `injection_mode=input` 或 `both` 才是。**
   当前 diffusion 设计不是加到 `s_inputs`，而是加到 `s_trunk`。这和你最后给出的目标设计是一致的。

3. **当前实现存在 3 个需要优先处理的问题。**
   - `rnalm_featurizer` 的 inference 分支识别了错误的 JSON entity type，导致推理态 RNA/DNA embedding 很可能根本不会加载。
   - 多处日志声称“会 fallback 到 random embeddings”，但真实实现并没有 random fallback。
   - 训练态 RNA/DNA 实体识别依赖 `chain_mol_type`，这是链级标签，不是 token 级标签；对 hybrid/mixed chain 的 separate projection 不够精确。

4. **“embedding 找不到时，会不会和 ESM 一样处理？”答案是：部分一样，部分不一样。**
   - 单条序列 embedding 文件缺失 / manifest 缺失时：**像 ESM**，featurizer 会记录 warning，保留该实体对应位置为 0，继续跑。
   - 整个 feature key 没进模型输入字典时：**不像日志里说的那样 fallback**，而是直接在模型侧报错。
   - inference JSON 场景下：当前更严重，RNA/DNA entity type 名称写错，可能直接变成“静默全 0”。

---

## 2. Highest-Priority Findings

### Finding 1: Inference RNA/DNA entity type 写错，导致推理态 embedding 加载失效

严重性：`High`

位置：
- `protenix/data/rnalm/rnalm_featurizer.py:243`
- `protenix/data/rnalm/rnalm_featurizer.py:245`
- `protenix/data/inference/json_to_feature.py:63`
- `protenix/data/inference/json_to_feature.py:64`
- `docs/infer_json_format.md` 中也使用 `rnaSequence` / `dnaSequence`

现状：

`RiNALMoFeaturizer._identify_entities(..., inference_mode=True)` 用的是：

```python
if entity_type == "rnaChain":
elif entity_type == "dnaChain":
```

但 Protenix inference JSON 入口实际使用的是：

```python
"rnaSequence"
"dnaSequence"
```

这在 `protenix/data/inference/json_to_feature.py:61-67` 可以直接确认。

影响：

- inference 模式下，RNA/DNA entity 根本不会被识别进 `rna_entity_id_to_seq` / `dna_entity_id_to_seq`
- `rna_entity_ids` / `dna_entity_ids` 会变空
- featurizer 返回的 `rna_llm_embedding` / `dna_llm_embedding` 或 `rnalm_token_embedding` 会是**全零**
- 且不会触发 `_fill_entities()`，因此不会打印“某条 embedding 缺失”的 warning
- 表面上看推理能跑，实际上 RNA/DNA LLM 注入可能完全没生效

结论：

**这是当前最重要的问题。训练态逻辑大体正确，但推理态核酸 embedding 加载逻辑是断的。**

---

### Finding 2: “fallback to random embeddings” 日志与真实行为不一致

严重性：`High`

位置：
- `protenix/data/pipeline/dataset.py:963-965`
- `protenix/data/pipeline/dataset.py:969-972`
- `protenix/data/inference/infer_dataloader.py:166-169`
- `protenix/model/modules/embedders.py:164-191`
- `protenix/model/protenix.py:426-459`

现状：

当 `rnalm.enable=true`，但 `embedding_dir/sequence_fpath` 没配置、路径不存在、或 featurizer 没被创建时：

- `get_rnalm_featurizer()` 返回 `None`
- `InferenceDataset` 里 `self.rnalm_featurizer` 也保持 `None`
- 但模型端 `self.rnalm_enable` 仍然是 `True`
- 后续：
  - `injection_mode=input/both` 时，`InputFeatureEmbedder.forward()` 直接要求 `rna_llm_embedding` / `dna_llm_embedding` 或 `rnalm_token_embedding` 存在，否则 `RuntimeError`
  - `injection_mode=diffusion/both` 时，`Protenix._get_s_rnalm()` 直接要求相应 key 存在，否则 `RuntimeError`

所以真实行为不是：

> fallback to random embeddings

而是：

> feature 不进 batch，模型侧直接报错

影响：

- 日志会误导你以为模型会自动退化成“无 RNA/DNA LLM 版本”
- 实际训练/推理会在模型 forward 阶段中断

结论：

**当前没有实现“random fallback”。有的是“日志说会 fallback，但代码实际 fail-fast”。**

---

### Finding 3: 训练态 separate projection 的 RNA/DNA 分类是链级近似，不是 token 级真分类

严重性：`Medium-High`

位置：
- `protenix/data/rnalm/rnalm_featurizer.py:250-255`
- `protenix/data/core/parser.py:2587-2604`

现状：

训练态 `RiNALMoFeaturizer._identify_entities(..., inference_mode=False)` 依赖：

```python
is_rna = centre_atom_array.chain_mol_type == "rna"
is_dna = centre_atom_array.chain_mol_type == "dna"
```

而 `chain_mol_type` 是在 `AddAtomArrayAnnot.add_atom_mol_type_mask()` 里，为**整条 chain** 赋一个统一标签。

这意味着：

- 对纯 RNA chain、纯 DNA chain：通常没问题
- 对 hybrid chain、modified chain、或链内混合情况：**不精确**
- separate projection 的“RNA residue 走 RNA projector、DNA residue 走 DNA projector”在训练态并不是严格按 residue/token 判定，而是按 chain 近似

更进一步：

`add_atom_mol_type_mask()` 当前实现是基于 chain 内 `mol_type` 计数排序后取一个类型；这本身就是 coarse approximation。你的 `rnalm_featurizer` 在训练态直接复用了这个近似标签，因此 hybrid 情况下容易偏离设计意图。

结论：

**当前 separate projection 更准确地说是“按 chain type 分流”，不是“按 token 真正分流”。**

---

### Finding 4: `concat` 融合模式下，不再保持 baseline-preserving 的 zero-init 行为

严重性：`Medium`

位置：
- `protenix/model/modules/diffusion.py:85-92`

现状：

当前默认 `fusion_method="add"`，这与你的目标设计一致，问题不大。  
但如果有人把配置改成 `fusion_method="concat"`，代码会新建：

- `layernorm_s_concat`
- `linear_no_bias_s_concat`

这两个层没有 zero-init，且会替代原始 `linear_no_bias_s` 路径。

影响：

- 即使 `rna_projection/dna_projection` 本身 zero-init，`concat` 路径也已经改变了 baseline 网络
- 这不再具备 ESM 风格的“初始时等价于不注入”的性质

结论：

**对你当前 `add` 方案不构成阻断，但它是这个实现里的一个配置性风险。**

---

## 3. Core Logic Review

### 3.1 Data Loading: training path

训练数据链路如下：

1. `get_rnalm_featurizer()` 根据 `configs.rnalm` 构造 `RiNALMoFeaturizer`  
   位置：`protenix/data/pipeline/dataset.py:944`

2. `get_datasets()` 把：
   - `rnalm_featurizer`
   - `rnalm_separate_dna`
   注入 `BaseSingleDataset`

3. `BaseSingleDataset.process_one()` 在 crop 后调用 featurizer  
   位置：`protenix/data/pipeline/dataset.py:522`

4. 若 `separate_dna_projection=true`，写入：
   - `feat["rna_llm_embedding"]`
   - `feat["dna_llm_embedding"]`

5. 否则写入：
   - `feat["rnalm_token_embedding"]`

判断：

- **这部分链路是闭合的**
- training batch 的 feature key 与模型侧读取的 key 是一致的

---

### 3.2 Data Loading: inference path

推理数据链路如下：

1. `InferenceDataset.__init__()` 构造 `RiNALMoFeaturizer`  
   位置：`protenix/data/inference/infer_dataloader.py:141`

2. `process_one()` 中，如果启用则调用 featurizer  
   位置：`protenix/data/inference/infer_dataloader.py:227`

3. separate 模式写入：
   - `features_dict["rna_llm_embedding"]`
   - `features_dict["dna_llm_embedding"]`

4. combined 模式写入：
   - `features_dict["rnalm_token_embedding"]`

判断：

- feature key 对模型是匹配的
- **但 inference entity type 识别写错，导致这条链路在 RNA/DNA JSON 输入时功能上不可靠**

---

### 3.3 Featurizer behavior

`RiNALMoFeaturizer` 的结构与 ESM 类似：

- 先初始化全零 tensor
- 按实体加载 embedding
- 根据 crop 后的 `res_id` 对齐回当前 token
- 某个实体加载失败时，catch exception，记录 warning，保留 0

对应位置：
- ESM：`protenix/data/esm/esm_featurizer.py:79-131`
- RNA/DNA：`protenix/data/rnalm/rnalm_featurizer.py:180-231`

你的 separate 设计在 featurizer 层面的落地是：

- RNA tensor：`[N_token, embedding_dim]`
- DNA tensor：`[N_token, dna_embedding_dim]`
- separate 时 DNA 不再 zero-pad 到 RNA 维度  
  位置：`protenix/data/rnalm/rnalm_featurizer.py:301-330`

判断：

- **separate/no-padding 设计已落地**
- **与 ESM 的“先对齐到 token，再由模型 projection”风格一致**

---

## 4. Model Injection Review

### 4.1 Input injection: 是否类似 ESM

是的，在 `injection_mode in ("input", "both")` 时，当前实现与 ESM 风格一致。

位置：
- 蛋白 ESM：`protenix/model/modules/embedders.py:59-68`, `151-158`
- RNA/DNA input：`protenix/model/modules/embedders.py:75-104`, `160-195`

行为：

- ESM：
  - `esm_token_embedding -> linear_esm -> add to s_inputs`
- RNA/DNA separate：
  - `rna_llm_embedding -> linear_rna_llm -> add to s_inputs`
  - `dna_llm_embedding -> linear_dna_llm -> add to s_inputs`

判断：

- **如果你把目标定义为“像 ESM 一样加到 S_input”，当前 input 模式是成立的**

---

### 4.2 Diffusion injection: 是否实现了你最后写的目标设计

是的，训练态主逻辑是成立的。

位置：
- `protenix/model/protenix.py:151-218`
- `protenix/model/protenix.py:424-460`
- `protenix/model/modules/diffusion.py:186-208`

当前实际行为：

1. `Protenix.__init__()` 在 diffusion/both 模式下创建：
   - `rna_projection: 2048 -> c_s`
   - `dna_projection: 1024 -> c_s`

2. `Protenix._get_s_rnalm()` 计算：

```python
lm_delta = self.rna_projection(rna_emb) + self.dna_projection(dna_emb)
```

3. `sample_diffusion()` / `sample_diffusion_training()` 把 `s_rnalm` 传进 `DiffusionModule`

4. `DiffusionConditioning.forward()` 在 `fusion_method="add"` 时执行：

```python
single_s = concat([s_trunk + s_rnalm, s_inputs], dim=-1)
```

判断：

- **这和你最后要求的**

```text
Separate projections
DIFFUSION injection
RNA -> RNA features
DNA -> DNA features
inject at DiffusionConditioning
no zero-padding
```

**是一致的**

补充说明：

- 它不是“加到 `s_inputs`”
- 它是“加到 diffusion conditioning 用到的 `s_trunk`”
- 所以它符合你最后的 diffusion 版设计，不符合你最前面那句 `S_input` 表述

---

## 5. Zero-Init Audit

你特别关心 projector 是否 zero-init。结论如下。

### 5.1 蛋白 ESM

有 zero-init。

位置：
- `protenix/model/modules/embedders.py:63-68`

```python
self.linear_esm = LinearNoBias(...)
nn.init.zeros_(self.linear_esm.weight)
```

### 5.2 RNA/DNA input injection

有 zero-init。

位置：
- `protenix/model/modules/embedders.py:87-90`
- `protenix/model/modules/embedders.py:99-100`

包括：
- `linear_rna_llm`
- `linear_dna_llm`
- `linear_rnalm`

### 5.3 RNA/DNA diffusion injection

有 zero-init。

位置：
- `protenix/model/protenix.py:166-177`
- `protenix/model/protenix.py:185-189`

包括：
- `rna_projection`
- `dna_projection`
- `rnalm_projection`

### 5.4 需要额外说明的点

`fusion_method="concat"` 时新建的 `linear_no_bias_s_concat` 没有 zero-init。  
这不影响你当前 `add` 方案，但会影响 concat 方案的 baseline-preserving 性。

---

## 6. Missing Embedding Behavior vs ESM

这个问题要拆成三类。

### 6.1 某个实体的 embedding 文件缺失 / manifest 缺失

行为：

- ESM：catch exception，warning，相关 token 保持 0，继续执行  
  位置：`protenix/data/esm/esm_featurizer.py:103-129`

- RNA/DNA：catch exception，warning，相关 token 保持 0，继续执行  
  位置：`protenix/data/rnalm/rnalm_featurizer.py:207-230`

结论：

**这一层面上，RNA/DNA 处理方式和 ESM 基本一致。**

---

### 6.2 整个 feature key 没有进模型输入字典

行为：

- ESM：如果 `esm.enable=true` 但 `esm_token_embedding` 不在 `input_feature_dict`，模型直接在 `InputFeatureEmbedder` 里取 key，最终会报错  
  位置：`protenix/model/modules/embedders.py:151-158`

- RNA/DNA：显式 `RuntimeError`
  - input 模式：`protenix/model/modules/embedders.py:164-191`
  - diffusion 模式：`protenix/model/protenix.py:426-459`

结论：

**这一层面上不是“继续用随机 embedding”，而是 fail-fast。**

---

### 6.3 配置了 rnalm，但 embedding_dir / csv 根本没准备好

当前行为：

- dataset / inference dataloader 打日志说“会 fallback to random embeddings”
- 实际不会生成任何 random embedding tensor
- 模型 forward 还是会因为缺少 key 而报错

结论：

**这和日志描述不一致；也不应该被理解成和 ESM 完全一样。**

---

### 6.4 inference JSON 场景

当前行为更特殊：

- 因为 `rnaChain` / `dnaChain` 写错
- 很可能根本识别不到实体
- featurizer 会安静返回全零 tensor
- 不会报缺失文件 warning

结论：

**这比 ESM 更危险，因为它可能“静默失效”。**

---

## 7. Design Match Against Your Stated Goal

你的最终目标是：

```text
Protenix + AIDO RNA/DNA LLM
Separate projections
DIFFUSION injection
RNA 2048 -> rna_projection
DNA 1024 -> dna_projection
inject at DiffusionConditioning
separate_dna_projection=True
no zero-padding
```

审查结论：

### 7.1 已实现的部分

- `separate_dna_projection=True` 时，featurizer 返回独立 RNA/DNA tensor
- DNA separate 路径不再 zero-pad 到 RNA 维度
- diffusion 模式下，模型创建独立 `rna_projection` / `dna_projection`
- `s_rnalm` 在 diffusion conditioning 里通过 `add` 方式并入 `s_trunk`
- input 模式也有 ESM-like 的独立 zero-init projector

### 7.2 没有完全实现好的部分

- inference JSON 核酸 entity type 名称不匹配，导致推理态加载不可靠
- “fallback to random embeddings” 没有真正实现
- training 态对 hybrid/mixed chain 不是 residue-level 的 RNA/DNA 分流

### 7.3 最终判断

**如果只看训练态、且只看你要求的 diffusion + separate AIDO 方案，主干逻辑已经实现。**  
**如果看整个项目完整可用性，这个实现还不能说完全稳，因为 inference 和 fallback 语义存在明显问题。**

---

## 8. Recommended Next Actions

不改代码前，我建议你先按下面顺序判断：

1. **先确认你当前训练是不是只跑 training，不依赖 inference JSON。**
   如果是，那么最关键的问题暂时是 hybrid/mixed chain 分类精度，而不是 inference type mismatch。

2. **如果你后面要做 inference 或导出 demo，必须先修 `rnaChain/dnaChain`。**
   这是阻断级问题。

3. **把“random fallback”语义统一掉。**
   要么真正实现 random / zero fallback；
   要么把日志改成“featurizer disabled, model will error if rnalm.enable remains true”。

4. **如果数据里存在 hybrid DNA/RNA chain，需要重新设计 token-level 分类。**
   否则 separate projection 在训练态并不是真正按 residue 分流。

---

## 9. Final Verdict

最终结论：

- **训练态 diffusion separate projection：基本实现到位**
- **与 ESM 的相似性：**
  - 数据对齐方式、zero-init projector 思路、单实体缺失时置零继续跑，这些是相似的
  - diffusion 注入点本身不是 ESM 的 `s_inputs` 逻辑，而是你新定义的 `s_trunk` 逻辑
- **当前最关键的问题：**
  - inference entity type 写错
  - fallback 日志和真实行为不一致
  - training 侧 hybrid/mixed chain 分类不够精细

一句话判断：

**“你想要的 AIDO RNA/DNA separate diffusion training logic 已经基本落地，但整个项目层面的 feature loading / inference robustness 还没完全收口。”**
