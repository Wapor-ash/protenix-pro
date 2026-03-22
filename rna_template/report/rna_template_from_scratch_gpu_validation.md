# RNA Template Online Finetune From-Scratch GPU Validation

**Date**: 2026-03-15

## Scope

This report validates the current **online RNA template finetune** path starting from a blank
`/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database`.

Constraints followed:

- No business code was changed.
- CIF template library stayed unchanged.
- Training data stayed unchanged.
- Validation was run on GPU.

Command used:

```bash
cd /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate protenix
bash finetune/finetune_rna_template_validate.sh --max_steps 2 --num_test_structures 20
```

## Findings

### No blocking finding in this run

The from-scratch online pipeline completed successfully:

- `rna_database/rna_catalog.json` was rebuilt
- `rna_database/search_results.json` was rebuilt
- GPU finetune completed for 2 steps
- training/eval logs contained no CUDA errors
- final validation summary was `17/17 passed`

Primary evidence:

- [validate.log](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/validate.log)
- [training.log](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/training.log)
- [config.yaml](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/rna_template_validate_20260315_203212/config.yaml)

### Medium: this run proves pipeline correctness, not full-production recall

This validation intentionally used `--num_test_structures 20`, so the rebuilt template catalog was a small subset of the full RNA3D library. As a result, the search stage only produced hits for `143/3460` training queries with `2.0` average templates per hit-bearing query, which is enough to prove the online path works but not enough to characterize production retrieval coverage.

Evidence:

- [validate.log#L98](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/validate.log#L98)

### Medium: online mode still emits an empty offline index, which is harmless but easy to misread

The validation rebuilt `rna_template_index.json`, but because no offline `.npz` templates were built, the index reports `0 templates found, 143 missing`. This did **not** block training in online mode, but it remains a confusing signal if someone reads the log with an offline mental model.

Evidence:

- [validate.log#L75](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/validate.log#L75)
- [validate.log#L96](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/validate.log#L96)

### Low: smoke validation is expensive because even a 2-step run performs full eval passes

This 2-step validation still took a long wall-clock time because the training run performed full evaluation over the `110`-sample validation set twice. That is not a correctness bug, but it makes “quick from-scratch smoke test” operationally heavier than the command line suggests.

Evidence:

- [training.log#L66](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/training.log#L66)
- [training.log#L298](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/training.log#L298)

## Verified Facts

### 1. `rna_database` was rebuilt from blank

Generated files:

- [rna_catalog.json](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/rna_catalog.json)
- [search_results.json](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/search_results.json)
- [rna_template_index.json](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/rna_template_index.json)

The run completed without generating offline `.npz` templates, which is expected for online mode.

Evidence:

- [validate.log#L81](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/validate.log#L81)

### 2. GPU was actually used

The training log shows `LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]`, and the environment reported one visible `NVIDIA H800`.

Evidence:

- [training.log#L3](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/training.log#L3)

### 3. Training used online RNA template mode, not offline index mode

The runtime log explicitly shows:

- `RNA template featurizer: ONLINE mode`
- `ONLINE mode enabled`

The saved config also records:

- `rna_template.enable: true`
- `search_results_path: .../rna_database/search_results.json`
- `cif_database_dir: .../part2/PDB_RNA`
- `template_index_path: ''`
- `rnalm.enable: false`

Evidence:

- [training.log#L12](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/training.log#L12)
- [training.log#L14](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/training.log#L14)
- [config.yaml#L478](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/rna_template_validate_20260315_203212/config.yaml#L478)
- [config.yaml#L489](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/rna_template_validate_20260315_203212/config.yaml#L489)

### 4. RNA template projector initialization behaved as designed

The validation confirmed:

- `linear_no_bias_a_rna` exists
- `rna_template_alpha` was initialized to `0.01`
- RNA projector weights matched the protein projector in `protein` init mode

The runtime log also shows projector repair happened after checkpoint and EMA checkpoint load.

Evidence:

- [validate.log#L121](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/validate.log#L121)
- [training.log#L52](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/training.log#L52)
- [training.log#L56](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/training.log#L56)

### 5. The finetune run truly stepped forward

This was not just initialization. The run:

- emitted `Step 1 train metrics`
- saved checkpoint `1.pt`
- saved EMA checkpoint `1_ema_0.999.pt`
- finished training after `2 steps`

Evidence:

- [training.log#L66](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/training.log#L66)
- [training.log#L298](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/output/rna_template_validate/training.log#L298)

## Conclusion

The current repository state can **rebuild `rna_database` from scratch and successfully run online RNA template finetuning on GPU without any code changes**.

What is proven by this run:

- blank `rna_database` bootstrap works
- online `search_results.json + CIF` path works
- RNA template projector init works
- GPU training works
- `rnalm=false` template-only training path works

What is not proven by this run:

- full-production retrieval recall, because the validation used only `20` template structures
- runtime cost suitability for repeated smoke tests, because evaluation dominates wall-clock time
