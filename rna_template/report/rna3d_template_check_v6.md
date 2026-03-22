# RNA Template v6 Code Review

Date: 2026-03-16

Scope:
- Prompt: `/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna3d/build-manualv2.txt`
- Previous review: `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_version5_check.md`
- Claimed completion report: `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/report/rna3d_version6.md`
- Main implementation reviewed:
  - `configs/configs_base.py`
  - `protenix/data/rna_template/rna_template_featurizer.py`
  - `protenix/data/pipeline/dataset.py`
  - `finetune/finetune_rna_template_1stage.sh`
  - `finetune/finetune_rna_template_2stage.sh`
  - `finetune/test_v6_training_manual_hints.py`
  - `finetune/test_manual_template_override.py`
  - `finetune/test_manual_template_e2e.py`

## Verdict

`v6` closes the main `v5` gap in one important sense: **training now has a real manual-hints entry point** via `rna_template.manual_template_hints_path`, and that entry point is wired into the dataset featurizer path used by finetune.

But I do **not** think `v6` is fully closed as claimed in `rna3d_version6.md`.

My assessment:

1. The config and script plumbing for training-time manual hints is present and mostly reasonable.
2. The default fallback behavior still looks preserved:
   - no manual hints path -> default online/offline RNA template behavior
   - `--use_rna_template false` -> RNA template path disabled
   - RNA LLM / RNA MSA flags in the finetune scripts are unchanged
3. The biggest remaining issue is now more serious than in `v5`: **manual templates are reachable during training, but still bypass temporal/self-hit/provenance governance**.
4. The local Python test evidence is now reproducible in the proper `protenix` environment, but the **GPU training claims are still under-evidenced** relative to the report.

## Findings

### 1. High: v6 makes manual templates available during training, but manual templates still bypass temporal/self-hit/provenance controls

Training-time manual hints are now actually wired in:

- `protenix/data/rna_template/rna_template_featurizer.py:1357`
- `protenix/data/rna_template/rna_template_featurizer.py:1377`

and are passed into normal RNA template feature construction:

- `protenix/data/rna_template/rna_template_featurizer.py:1385`
- `protenix/data/rna_template/rna_template_featurizer.py:1392`

However, the manual branch itself still directly accepts a local structure or NPZ and builds features without any provenance/date/self-hit checks:

- `protenix/data/rna_template/rna_template_featurizer.py:864`
- `protenix/data/rna_template/rna_template_featurizer.py:904`
- `protenix/data/rna_template/rna_template_featurizer.py:996`
- `protenix/data/rna_template/rna_template_featurizer.py:1008`

The temporal/self-hit filters are only applied to the search fallback path:

- `protenix/data/rna_template/rna_template_featurizer.py:906`
- `protenix/data/rna_template/rna_template_featurizer.py:968`

Why this matters:
- In `v5`, this was mostly an inference-side governance gap.
- In `v6`, it becomes a **training-time leakage risk** because the manual path is now reachable from finetune.
- A user can inject a future structure, a self structure, or an externally refined structure with no explicit `allow_training_use` / `derived_from_pdb_id` / `release_date` contract.

This is the most important remaining weakness in the current design.

### 2. Medium-High: the new v6 tests do not actually verify the training pipeline behavior they claim to verify

`rna3d_version6.md` claims:

- unit tests passed
- 1-stage and 2-stage GPU training passed
- training with manual hints passed

Relevant report sections:

- `rna_template/report/rna3d_version6.md:277`
- `rna_template/report/rna3d_version6.md:304`

But the new dedicated v6 unit test file only checks JSON loading and dictionary lookup logic; it does **not** execute:

- `RNATemplateFeaturizer.__call__(..., inference_mode=False)`
- `get_rna_template_features(...)`
- dataset integration
- finetune script launch
- manual template build during training

Evidence:

- featurizer construction only:
  - `finetune/test_v6_training_manual_hints.py:44`
  - `finetune/test_v6_training_manual_hints.py:50`
- tests only inspect `_training_manual_hints` and fallback dict lookup:
  - `finetune/test_v6_training_manual_hints.py:53`
  - `finetune/test_v6_training_manual_hints.py:180`

So the file named "Verify manual template hints work during training mode" does not actually cover the training feature path end-to-end.

Why this matters:
- The new config plumbing may be correct, but the claimed behavioral closure is still under-tested.
- The report is stronger than the real evidence.

### 3. Medium: local tests are reproducible in `conda activate protenix`, but the GPU-training claims are still not independently confirmed here

I ran:

```bash
eval "$(conda shell.bash hook)"
conda activate protenix
python finetune/test_v6_training_manual_hints.py
python finetune/test_manual_template_override.py
python finetune/test_manual_template_e2e.py
```

Observed result:
- `test_v6_training_manual_hints.py`: passed
- `test_manual_template_override.py`: `11 passed, 0 failed`
- `test_manual_template_e2e.py`: `7 passed, 0 failed`

I also confirmed `Bio` is available in that environment:

```text
/opt/conda/envs/protenix/bin/python
Bio OK /opt/conda/envs/protenix/lib/python3.11/site-packages/Bio/__init__.py
```

So the **Python-side test claims are reproducible** when using the intended environment.

What I still did **not** independently confirm is the report's GPU result section:
- 1-stage no-manual full completion
- 2-stage full completion
- 1-stage manual-hints full completion with saved checkpoints

I was able to start a 1-stage smoke run successfully and verify:
- shell script launches correctly
- `runner/train.py` starts
- CUDA device is visible
- config expands with RNALM + RNA template + RNA MSA enabled

But within this review window I did not wait through a complete step/checkpoint cycle, so I am not upgrading the GPU claims to fully independently verified.

This is especially important because `rna3d_version6.md` presents the GPU checks as completed facts:

- `rna_template/report/rna3d_version6.md:286`
- `rna_template/report/rna3d_version6.md:304`

### 4. Medium: public schema/interface drift still exists

Inference still uses inline JSON `templateHints`, while training now uses an external `manual_template_hints_path` JSON.

Implementation-wise this is understandable, but the public interface is still not documented in the main JSON format doc:

- `docs/infer_json_format.md:111`
- `docs/infer_json_format.md:138`

I did not find `templateHints` documented in `docs/` or examples.

Why this matters:
- The system now has two manual-template interfaces:
  - inference: inline `templateHints`
  - training: external `manual_template_hints_path`
- both are effectively private conventions unless the docs are updated
- this raises maintenance and onboarding cost

## Confirmed Good / Reasonable Parts

These parts look correct and aligned with the prompt:

### 1. Training manual hints are actually wired into the training data path

The dataset factory now reads the new config and passes it into the featurizer:

- `protenix/data/pipeline/dataset.py:1113`
- `protenix/data/pipeline/dataset.py:1160`

The training dataset uses `rna_template_featurizer(..., inference_mode=False)` in the same path as before:

- `protenix/data/pipeline/dataset.py:567`
- `protenix/data/pipeline/dataset.py:576`

This means `v6` did close the core "training path was missing" issue from `v5`.

### 2. No-manual fallback path still looks intact

The training hints are only used when configured:

- `protenix/data/rna_template/rna_template_featurizer.py:1360`

and `rna3d_version6.md` is correct that the empty default path should short-circuit:

- `configs/configs_base.py:164`
- `configs/configs_base.py:169`

So:
- no `manual_template_hints_path` -> existing online/offline pipeline remains active
- `--use_rna_template false` in the finetune scripts still disables the feature entirely

### 3. RNA LLM, projector-init, and RNA MSA script wiring remains consistent

The finetune scripts still pass RNALM settings unchanged:

- `finetune/finetune_rna_template_1stage.sh:136`
- `finetune/finetune_rna_template_1stage.sh:146`
- `finetune/finetune_rna_template_2stage.sh:140`
- `finetune/finetune_rna_template_2stage.sh:150`

The scripts still enable RNA MSA explicitly:

- `finetune/finetune_rna_template_1stage.sh:271`
- `finetune/finetune_rna_template_1stage.sh:275`
- `finetune/finetune_rna_template_2stage.sh:280`
- `finetune/finetune_rna_template_2stage.sh:284`

The RNA template projector init path is still honored by model code:

- `protenix/model/modules/pairformer.py:991`
- `protenix/model/modules/pairformer.py:1018`

and checkpoint-load repair for the RNA projector is still present:

- `runner/train.py:563`
- `runner/train.py:575`
- `protenix/model/protenix.py:303`
- `protenix/model/protenix.py:329`

This supports the claim that `v6` did not obviously break the existing RNALM / projector-init / MSA wiring.

### 4. Inference-side DNA-with-U manual-hints gap appears fixed

This was a real `v5` boundary gap, and the new branch now extracts hints for reclassified DNA-with-U entities:

- `protenix/data/rna_template/rna_template_featurizer.py:1313`
- `protenix/data/rna_template/rna_template_featurizer.py:1331`

That part looks reasonable.

## Validation Performed In This Review

### Code-path inspection

Reviewed:
- config plumbing
- dataset factory plumbing
- training/inference featurizer branches
- finetune script argument propagation
- RNA projector-init path
- RNA MSA flags in finetune scripts

### Script sanity checks

Ran:

```bash
bash -n finetune/finetune_rna_template_1stage.sh
bash -n finetune/finetune_rna_template_2stage.sh
```

Both passed shell syntax check.

### File existence spot checks

Confirmed these referenced files exist:
- `data/stanford-rna-3d-folding/part2/PDB_RNA/100d.cif`
- `data/stanford-rna-3d-folding/part2/PDB_RNA/104d.cif`
- `rna_database/search_results.json`
- `finetune/test_manual_hints.json`

### Local test execution

Ran:

```bash
eval "$(conda shell.bash hook)"
conda activate protenix
python finetune/test_v6_training_manual_hints.py
python finetune/test_manual_template_override.py
python finetune/test_manual_template_e2e.py
```

Result:
- `test_v6_training_manual_hints.py`: passed
- `test_manual_template_override.py`: `11 passed, 0 failed`
- `test_manual_template_e2e.py`: `7 passed, 0 failed`

### Training smoke-run check

Ran a minimal 1-stage launch in the `protenix` environment:

```bash
bash finetune/finetune_rna_template_1stage.sh \
  --max_steps 1 \
  --eval_interval 1000 \
  --checkpoint_interval 1 \
  --use_wandb false \
  --run_name review_v6_nomanu_smoke
```

Confirmed:
- script starts successfully
- `runner/train.py` launches
- `LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]`
- H800 is visible on the machine
- run directory and `config.yaml` are created

I stopped short of claiming full independent checkpoint validation because I did not let this smoke run complete through the first saved step during this review window.

### Script-level small-instance closure checks

I also ran three bounded `1stage` script smokes in `conda activate protenix` using `timeout`:

```bash
timeout --signal=INT 45s bash finetune/finetune_rna_template_1stage.sh \
  --max_steps 1 --eval_interval 1000 --checkpoint_interval 1 \
  --use_wandb false --run_name review_v6_default_smoke

timeout --signal=INT 45s bash finetune/finetune_rna_template_1stage.sh \
  --max_steps 1 --eval_interval 1000 --checkpoint_interval 1 \
  --use_wandb false \
  --manual_template_hints finetune/test_manual_hints.json \
  --run_name review_v6_manual_smoke

timeout --signal=INT 45s bash finetune/finetune_rna_template_1stage.sh \
  --max_steps 1 --eval_interval 1000 --checkpoint_interval 1 \
  --use_wandb false --use_rna_template false \
  --run_name review_v6_notpl_smoke
```

All three cases showed the same pattern:
- script banner printed correctly
- distributed/CUDA initialization succeeded
- run directory and `config.yaml` were created
- process was interrupted by the review timeout during generic model initialization, not by RNA-template-specific argument handling

Config-level closure confirmed from the generated `config.yaml` files:

1. Default online template path:
   - `rna_template.enable: true`
   - `rna_template.manual_template_hints_path: ''`
   - online `search_results_path` and `cif_database_dir` populated
   - RNALM and RNA MSA still enabled

2. Manual-hints path:
   - `rna_template.enable: true`
   - `rna_template.manual_template_hints_path: finetune/test_manual_hints.json`
   - online `search_results_path` and `cif_database_dir` still populated
   - RNALM and RNA MSA still enabled

3. RNA template disabled:
   - `rna_template.enable: false`
   - `rna_template.search_results_path: ''`
   - `rna_template.cif_database_dir: ''`
   - `rna_template.manual_template_hints_path: ''`
   - RNALM and RNA MSA still enabled

What this does prove:
- the `finetune/finetune_rna_template_1stage.sh` script accepts and propagates the new manual-hints option correctly
- the no-manual default path is configured as fallback-to-online-search
- turning RNA template off does not collapse RNALM or RNA MSA script wiring

What this does not yet prove:
- first-step completion with checkpoint saved
- script-level evidence that a sampled training example actually consumed a manual template during that bounded run
- script-level evidence that a sampled training example actually took the search fallback during that bounded run

## Residual Pipeline Weaknesses

Even if the core `v6` training-hook design is accepted, these weaknesses remain:

1. Manual templates have no auditable provenance contract.
2. Training now accepts manual templates without temporal/self-hit guardrails.
3. Script-level validation is now stronger for `1stage` startup/config closure, but full GPU training evidence is still lighter than the report wording suggests.
4. The interface is split across inference JSON and training-side external JSON, but not publicly documented.

## Bottom Line

My review conclusion is:

1. `v6` **does** address the main missing training-entry issue from `v5`.
2. The implementation direction is reasonable and appears non-invasive to the default RNA template, RNA LLM, RNA projector-init, and RNA MSA script wiring.
3. The task is still **not fully closed** because the most important biological/benchmark risk is now active:
   - manual templates are usable during training
   - but still bypass provenance and temporal/self-hit governance
4. The local Python test suite is reproducible in `conda activate protenix`, but the report's GPU-training claims should still be treated as only partially independently confirmed.
