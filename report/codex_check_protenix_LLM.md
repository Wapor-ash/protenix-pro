# Protenix RNA/DNA LLM Integration Check

**Date**: 2026-03-11  
**Repository**: `ash_proj/code/protenix_new/Protenix`  
**Audit scope**: feature loading, input assembly, diffusion injection, ESM-style input logic, zero-init projector behavior

---

## 1. Target Design Being Checked

Expected target:

- RNA LLM embedding (`2048`) -> `rna_projection` -> add to RNA features
- DNA LLM embedding (`1024`) -> `dna_projection` -> add to DNA features
- injection point at `DiffusionConditioning`, by adding to `s_trunk`
- `separate_dna_projection=True`
- no DNA zero-padding in the separate-projection path

Reference script:

- `protenix_aido_separate_diffusion.sh`

This script sets:

- `--rnalm.enable true`
- `--rnalm.embedding_dim 2048`
- `--rnalm.dna_embedding_dim 1024`
- `--rnalm.injection_mode diffusion`
- `--rnalm.separate_dna_projection true`

So the script configuration is consistent with the intended design.

---

## 2. Executive Conclusion

**Conclusion: mostly yes, but not fully safe.**

Your current modification **does implement the core logic** of:

1. loading separate RNA and DNA token-level embeddings,
2. using **separate projection layers**,
3. injecting them at **DiffusionConditioning** through `s_trunk + s_rnalm`,
4. and using **zero-init** on the new RNA/DNA projector layers.

It also implements an **ESM-like input injection path** for RNA/DNA when `injection_mode` is `input` or `both`.

However, there is one **important correctness issue**:

- when `separate_dna_projection=True` but DNA embedding files are not available, the current featurizer returns `dna_llm_embedding` with the **wrong last dimension** in separate mode, which conflicts with `dna_projection(1024 -> 384)` / `linear_dna_llm(1024 -> 449)`.

That means the current code is **correct for the intended separate diffusion design only if DNA embedding loading is properly configured and present**.

---

## 3. End-to-End Code Path

### 3.1 Feature loading: training path

Training dataset path:

- `protenix/data/pipeline/dataset.py:523-544`

Behavior:

- after crop, if `rnalm_featurizer` exists:
  - `separate_dna_projection=True`:
    - writes `feat["rna_llm_embedding"]`
    - writes `feat["dna_llm_embedding"]`
  - otherwise:
    - writes `feat["rnalm_token_embedding"]`

This is the right place to attach token-aligned RNA/DNA embeddings, because it happens **after cropping**, so the LLM features align with the final token set used by the model.

### 3.2 Feature loading: inference path

Inference path:

- `protenix/data/inference/infer_dataloader.py:227-248`

Behavior is parallel to training:

- separate mode writes:
  - `features_dict["rna_llm_embedding"]`
  - `features_dict["dna_llm_embedding"]`
- legacy mode writes:
  - `features_dict["rnalm_token_embedding"]`

So training and inference are internally consistent.

### 3.3 RNA/DNA featurizer behavior

Featurizer implementation:

- `protenix/data/rnalm/rnalm_featurizer.py:261-354`

Separate mode:

- `rna_llm_embedding`: shape `[N_token, embedding_dim]`
- `dna_llm_embedding`: shape `[N_token, dna_embedding_dim]` when DNA loading is enabled

Token semantics:

- RNA tokens get RNA embeddings
- DNA tokens get DNA embeddings
- protein/ligand/non-target tokens stay zero

This is structurally analogous to ESM:

- `protenix/data/esm/esm_featurizer.py:77-134`

ESM also builds a full `[N_token, dim]` tensor, fills only protein tokens, and leaves other token positions zero.

So from a **feature loading pattern** perspective, your RNA/DNA implementation is indeed modeled after the ESM logic.

---

## 4. Model Injection Logic

### 4.1 InputFeatureEmbedder: ESM-like input path exists

File:

- `protenix/model/modules/embedders.py:66-104`
- `protenix/model/modules/embedders.py:160-177`

Behavior:

- if `rnalm.enable` and `injection_mode in ("input", "both")`
  - separate mode creates:
    - `linear_rna_llm: embedding_dim -> 449`
    - `linear_dna_llm: dna_embedding_dim -> 449`
  - forward adds projected tensors directly into `s_inputs`

This is genuinely **ESM-like input logic**:

- ESM path:
  - `linear_esm(2560 -> 449)`
  - add to `s_inputs`
- RNA/DNA input path:
  - `linear_rna_llm(2048 -> 449)`
  - `linear_dna_llm(1024 -> 449)`
  - add to `s_inputs`

So if you ask "有没有实现类似 ESM 的 input 逻辑", the answer is:

- **Yes**, for `input` / `both` mode.
- For your target script `separate_diffusion`, this path is not active, because injection mode is `diffusion`.

### 4.2 Main target path: separate diffusion projection

File:

- `protenix/model/protenix.py:145-177`
- `protenix/model/protenix.py:398-468`

Behavior:

- when `rnalm.enable` and `injection_mode in ("diffusion", "both")`
  - separate mode creates:
    - `rna_projection: embedding_dim -> c_s`
    - `dna_projection: dna_embedding_dim -> c_s`

Then `_get_s_rnalm()` does:

- read `rna_llm_embedding`
- read `dna_llm_embedding`
- compute:
  - `lm_delta = self.rna_projection(rna_emb) + self.dna_projection(dna_emb)`

This is exactly the intended **separate projection** design.

Because the featurizer returns full-length tensors with zero everywhere except relevant tokens:

- RNA projection contributes only on RNA token positions
- DNA projection contributes only on DNA token positions
- protein/ligand positions remain zero

So semantically this is:

- RNA embedding -> RNA positions
- DNA embedding -> DNA positions

without using a shared zero-padded projector in the separate path.

### 4.3 Injection point: DiffusionConditioning

Files:

- `protenix/model/protenix.py:669-740`
- `protenix/model/protenix.py:857-999`
- `protenix/model/modules/diffusion.py:186-209`

Flow:

1. `Protenix.get_pairformer_output()` computes `s_inputs`, `s`, `z`
2. `Protenix._get_s_rnalm()` computes projected RNA/DNA delta in `c_s`
3. `s_rnalm` is passed into diffusion sampling/training
4. `DiffusionConditioning.forward()` does:
   - if `fusion_method == "add"`:
     - `single_s = concat([s_trunk + s_rnalm, s_inputs])`

This matches your requested design:

- **Injection at DiffusionConditioning**
- **Add to `s_trunk`**

For the target script, `fusion_method` is not explicitly passed, but config default is:

- `configs/configs_base.py:103`
- `"fusion_method": "add"`

So the intended `DIFFUSION injection` behavior is active.

---

## 5. Zero-Init Projector Check

### 5.1 Protein / ESM projector

File:

- `protenix/model/modules/embedders.py:56-61`

Behavior:

- `linear_esm` is created
- `nn.init.zeros_(self.linear_esm.weight)`

So the protein ESM projector is zero-init.

### 5.2 RNA/DNA input-mode projectors

File:

- `protenix/model/modules/embedders.py:79-83`
- `protenix/model/modules/embedders.py:87-93`

Behavior:

- `linear_rna_llm` zero-init
- `linear_dna_llm` zero-init
- legacy `linear_rnalm` also zero-init

So the ESM-like RNA/DNA input projectors are zero-init.

### 5.3 RNA/DNA diffusion-mode projectors

File:

- `protenix/model/protenix.py:166-177`
- `protenix/model/protenix.py:185-189`

Behavior:

- `rna_projection.weight` zero-init
- `dna_projection.weight` zero-init
- legacy `rnalm_projection.weight` zero-init

So the separate diffusion projectors are also zero-init.

### 5.4 Final judgment on zero-init

If your question is:

- “protein / RNA / DNA projector 是否都做了 zero-init？”

Answer:

- **Protein ESM projector: yes**
- **RNA input projector: yes**
- **DNA input projector: yes**
- **RNA diffusion projector: yes**
- **DNA diffusion projector: yes**

This part is implemented correctly.

---

## 6. Important Findings

## Finding 1: Separate mode breaks the “DNA missing -> zeros” expectation

**Severity: high**

Relevant code:

- `protenix/data/rnalm/rnalm_featurizer.py:301-306`
- `protenix/data/rnalm/rnalm_featurizer.py:327-330`
- `protenix/model/protenix.py:172-177`
- `protenix/model/protenix.py:433-442`
- `protenix/model/modules/embedders.py:79-83`
- `protenix/model/modules/embedders.py:168-170`

Problem:

In separate mode, when DNA embedding files are not enabled:

```python
dna_dim = self.dna_embedding_dim if self.dna_enable else self.embedding_dim
dna_x = torch.zeros([N_token, dna_dim])
```

So if:

- RNA dim = `2048`
- DNA dim = `1024`
- but `self.dna_enable == False`

then `dna_x` becomes `[N_token, 2048]`, not `[N_token, 1024]`.

But downstream separate projectors are built as:

- `dna_projection: 1024 -> 384`
- `linear_dna_llm: 1024 -> 449`

This creates a dimension mismatch.

So the script message in `protenix_aido_separate_diffusion.sh`:

- “DNA tokens will get zeros.”

is **not actually safe in the current separate-projection implementation**.

What is true right now:

- legacy combined mode supports “missing DNA -> zero-padded combined tensor”
- separate mode does **not** safely support “missing DNA -> zero tensor” unless the zero tensor is also created in the native DNA dimension

This is the most important mismatch I found.

## Finding 2: Two-stage adapter keyword is stale for separate projectors

**Severity: medium**

Relevant code:

- `configs/configs_base.py:72`
- `runner/train.py:244-246`
- `runner/train.py:263-269`
- `protenix/model/protenix.py:166-177`

Current default:

- `two_stage.adapter_keywords = "rnalm_projection"`

But in separate diffusion mode the new parameter names are:

- `rna_projection`
- `dna_projection`

So if someone enables two-stage training while using separate projection, stage 1 adapter warmup will **not** match the new separate projectors by default.

This does **not** affect your current `protenix_aido_separate_diffusion.sh`, because it sets:

- `--two_stage.enable false`

But it is a real implementation gap for future separate-projection training.

## Finding 3: Missing embedding paths can silently fall back to random embeddings

**Severity: medium**

Relevant code:

- `protenix/data/pipeline/dataset.py:961-973`
- `protenix/data/inference/infer_dataloader.py:166-169`
- `protenix/model/protenix.py:425-454`

Behavior:

- if feature loading is unavailable, model-side `_get_s_rnalm()` can synthesize random embeddings

This is not a shape bug by itself, but it is operationally risky:

- a path/config mistake may not fail fast
- training may proceed with random embeddings instead of real RNA/DNA embeddings

For an auditing or finetuning workflow, this is easy to miss.

---

## 7. Direct Answer to Your Questions

### 7.1 “我现在的修改是否实现了这个逻辑？”

**Answer: yes, in the main path, with one important caveat.**

Implemented correctly:

- separate RNA/DNA feature loading
- separate RNA/DNA projector layers
- diffusion injection at `DiffusionConditioning`
- `s_trunk + s_rnalm` add-style fusion
- zero-init for the new projectors

Not fully correct:

- separate mode currently does **not** robustly support the “DNA embedding missing -> use zeros” fallback promised by the script/log message

### 7.2 “类似 ESM 的 input 逻辑？”

**Answer: yes.**

You added an ESM-style input pathway in `InputFeatureEmbedder`:

- full token tensor
- non-target token positions are zeros
- project to `449`
- add to `s_inputs`
- zero-init projector

This is structurally consistent with official ESM handling.

### 7.3 “zero-init projector（分别蛋白，rna，dna 部分）”

**Answer: yes.**

Confirmed zero-init:

- protein ESM projector
- RNA input projector
- DNA input projector
- RNA diffusion projector
- DNA diffusion projector

---

## 8. Final Verdict

For the requested architecture:

- **Protenix + AIDO RNA/DNA LLM**
- **Separate projections**
- **Diffusion injection**

the current codebase has the **right overall architecture and data flow**.

My final judgment is:

- **Architecture correctness: yes**
- **ESM-style input-path implementation: yes**
- **Zero-init projector implementation: yes**
- **Production safety / fallback correctness: not fully**

The main blocker is the separate-mode DNA fallback shape bug described in Finding 1.

---

## 9. Audit Method

This report is based on static code inspection of the current modified tree, mainly:

- `configs/configs_base.py`
- `protenix/data/rnalm/rnalm_featurizer.py`
- `protenix/data/pipeline/dataset.py`
- `protenix/data/inference/infer_dataloader.py`
- `protenix/model/modules/embedders.py`
- `protenix/model/modules/diffusion.py`
- `protenix/model/protenix.py`
- `runner/train.py`
- `protenix_aido_separate_diffusion.sh`

No source code was modified during this audit.
