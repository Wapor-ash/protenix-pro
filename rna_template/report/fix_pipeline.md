# RNA Template Pipeline Code Review

Date: 2026-03-14

Scope:
- `/inspire/ssd/project/sais-bio/public/ash_proj/prompts/rna_template/run_pipeline.txt`
- `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/code_review/rna_template_search_pipeline_report_zh.md`
- `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/*`
- `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/*`
- `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py`
- `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/pipeline/dataset.py`
- `/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/model/modules/pairformer.py`

Constraint:
- Review only.
- No code changes were made.

## 1. Executive Summary

This review traces the actual RNA template pipeline used by the current Protenix integration and checks whether the implementation matches the intended end-to-end design described in the prompt and the existing technical report.

Main conclusion:
- The current pipeline is wired up and can generate `.npz` RNA template files that are consumable by Protenix.
- However, the implementation still has several structural mismatches relative to the intended "full pipeline" semantics.
- The most important issue is that the pairwise search path is effectively keyed by `PDB` rather than `PDB chain`, so it is not a true per-RNA-chain pipeline.
- A second important issue is that the runtime featurizer only consumes the first valid `.npz` listed for a sequence, so multiple template files in the index are not actually merged at runtime.
- A third issue is that training/inference sequences containing modified-base placeholders such as `X` can silently miss the RNA template index because the lookup normalization is weaker than the database-side normalization.

Net assessment:
- The current implementation is sufficient for a narrow validation pipeline.
- It is not yet a robust general RNA template pipeline for multi-chain or modified-base-heavy RNA data.

## 2. What The Pipeline Actually Calls

This section answers the core question: when the pipeline computes templates, which predefined scripts and compute functions are actually used?

### 2.1 Primary pipeline entrypoint

The main automated pipeline is:

- [`run_pipeline.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh)

Its steps are:

1. Run catalog extraction via [`01_extract_rna_catalog.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py)
2. Run template building via [`02_build_rna_templates.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py)
3. Run search/index building via [`03_search_and_index.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py)

Evidence:
- [`run_pipeline.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L102)
- [`run_pipeline.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L123)
- [`run_pipeline.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L164)

### 2.2 Scripts used in self mode

In `self` mode:

1. [`01_extract_rna_catalog.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py)
2. [`02_build_rna_templates.py --mode self`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py)
3. [`03_search_and_index.py --strategy self`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py)

Evidence:
- [`run_pipeline.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L128)
- [`run_pipeline.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L168)

### 2.3 Scripts used in pairwise mode

In `pairwise` mode:

1. [`01_extract_rna_catalog.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/01_extract_rna_catalog.py)
2. [`03_search_and_index.py --strategy pairwise`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py)
3. [`02_build_rna_templates.py --mode cross`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py)
4. [`03_search_and_index.py --strategy pairwise`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py) again to rebuild the final index

Evidence:
- [`run_pipeline.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L136)
- [`run_pipeline.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L141)
- [`run_pipeline.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L152)
- [`run_pipeline.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L177)

### 2.4 Compute functions actually used for template construction

`02_build_rna_templates.py` does not directly call `build_rna_template_protenix.py`.
Instead, it imports `rna_template_common.py` and uses its shared functions.

Direct imports:
- [`02_build_rna_templates.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L48)
- [`02_build_rna_templates.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L53)

Actual compute functions used in the main pipeline:
- [`load_structure_residues()`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/rna_template_common.py#L238)
- [`normalize_query_sequence()`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/rna_template_common.py#L176)
- [`build_minimal_template_arrays()`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/rna_template_common.py#L424)
- [`align_query_to_template()`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/rna_template_common.py#L294)
- [`compute_anchor()`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/rna_template_common.py#L327)
- [`compute_frame()`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/rna_template_common.py#L352)
- [`compute_distogram()`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/rna_template_common.py#L384)
- [`compute_unit_vectors()`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/rna_template_common.py#L400)
- [`stack_template_dicts()`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/rna_template_common.py#L481)
- [`save_npz_with_metadata()`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/rna_template_common.py#L529)

### 2.5 What is not on the primary path

The following path exists, but it is not the main `run_pipeline.sh` path:

- [`run_arena_and_template.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/run_arena_and_template.sh)
- [`build_rna_template_protenix.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/build_rna_template_protenix.py)

This path is a separate single-structure workflow:

1. Run Arena to complete atoms
2. Run `build_rna_template_protenix.py`
3. Emit one `.npz`

Evidence:
- [`run_arena_and_template.sh`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/run_arena_and_template.sh#L72)
- [`build_rna_template_protenix.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/build_rna_template_protenix.py#L58)

Conclusion:
- `build_rna_template_protenix.py` is a valid predefined script.
- But the full automated pipeline currently computes templates through `02_build_rna_templates.py` plus `rna_template_common.py`, not through `build_rna_template_protenix.py`.

## 3. Runtime Consumption Path Inside Protenix

Once `.npz` template files and the index are built, runtime consumption is:

1. Dataset factory creates `RNATemplateFeaturizer`
2. Dataset calls the featurizer during sample construction
3. `RNATemplateFeaturizer` loads `.npz` data and writes `rna_template_*` features
4. `Pairformer` reads those features and injects them through the RNA template projector

Evidence:
- [`dataset.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/pipeline/dataset.py#L1090)
- [`dataset.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/pipeline/dataset.py#L567)
- [`rna_template_featurizer.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L159)
- [`pairformer.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/model/modules/pairformer.py#L1083)
- [`pairformer.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/model/modules/pairformer.py#L1177)

## 4. Findings

Findings are listed in descending severity, with concrete code references.

### 4.1 High: Pairwise mode is keyed by PDB, not by RNA chain

Expected behavior from the design intent:
- A training/query RNA chain should map to its own matched templates.
- Query identity should therefore be chain-level or entity-level.

Actual behavior:
- `load_training_sequences_from_json()` claims to return `{pdb_id_chain: sequence}`.
- But the implementation uses only the PDB id as the query key.

Evidence:
- Comment says chain-level mapping: [`03_search_and_index.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py#L243)
- Implementation uses `key = f"{pdb_id.lower()}"`: [`03_search_and_index.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py#L257)

The self-exclusion logic then also operates only at PDB granularity:
- [`03_search_and_index.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py#L142)
- [`03_search_and_index.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py#L148)

Why this matters:
- If a structure contains multiple RNA chains, they cannot become distinct pairwise-mode queries through the `training_sequences` JSON path.
- The pairwise pipeline therefore does not truly implement a per-chain search-and-build workflow.
- The output cross-template file name is `query_id_template.npz`, so its identity semantics are already wrong at the file level when `query_id` is only a PDB id.

Impact:
- Incorrect abstraction boundary for multi-chain RNA structures
- Potential query collapsing
- Inability to represent chain-specific search behavior

Assessment:
- This is the single most important semantic mismatch in the current pipeline.

### 4.2 High: Runtime featurizer only consumes the first valid `.npz` for a sequence

The template index is built as:
- `sequence -> [template_a.npz, template_b.npz, ...]`

Evidence:
- Self index appends multiple paths: [`03_search_and_index.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py#L201)
- Cross index appends multiple paths: [`03_search_and_index.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py#L231)

But the runtime featurizer resolves the candidate paths and then loads only the first valid file:
- [`rna_template_featurizer.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L279)
- [`rna_template_featurizer.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L287)
- [`rna_template_featurizer.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L294)

There is a hard `break` after the first loadable `.npz`.

Why this matters:
- In self mode, each generated `.npz` typically contains only one real template and padded empties.
- Therefore multiple `.npz` entries in the index do not translate into multiple templates seen by the model.
- Effective template diversity is lower than the index suggests.

Additional evidence:
- Self mode writes a single-template stack: [`02_build_rna_templates.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L103)
- Runtime later pads the chosen file, but does not merge multiple files: [`rna_template_featurizer.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L299)

Impact:
- `max_templates` can be misleading at runtime
- Index cardinality overstates the actual number of templates used
- Search-side ranking improvements may not fully reach the model

Assessment:
- This is a major behavioral gap between data generation and model consumption.

### 4.3 Medium: Modified-base sequences can silently miss the RNA template index

Database and template generation normalize aggressively:
- `normalize_query_sequence()` converts unsupported characters to `N`: [`rna_template_common.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/rna_template_common.py#L176)
- `normalize_rna_sequence()` in search/index does the same: [`03_search_and_index.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py#L55)

But runtime index lookup is weaker:
- Exact match
- Then `T -> U`
- No `X -> N` or other normalization

Evidence:
- [`rna_template_featurizer.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L197)
- [`rna_template_featurizer.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L206)

Protenix core sequence construction can emit `X` for modified residues:
- [`ccd.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/core/ccd.py#L445)
- [`ccd.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/core/ccd.py#L448)

The RNA LM pipeline explicitly knows this and has a modified-base fallback:
- [`rnalm_featurizer.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rnalm/rnalm_featurizer.py#L330)
- [`rnalm_featurizer.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rnalm/rnalm_featurizer.py#L333)
- [`rnalm_featurizer.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rnalm/rnalm_featurizer.py#L553)

The RNA template pipeline does not.

Why this matters:
- A sequence can have embeddings but no templates, solely because one path understands modified bases and the other does not.
- This failure mode is likely silent: `_find_templates_for_sequence()` just returns `[]`, then the template path is skipped.

Impact:
- Lower template hit rate for modified RNA
- Mismatch between RNA LM and RNA template behavior
- Hard-to-debug silent coverage loss

### 4.4 Medium: Pairwise identity is optimistic for partial matches

The pairwise identity implementation:
- aligns globally
- then computes identity only on aligned blocks present in `alignment.aligned`
- divides by aligned block length, not by full query length or full alignment span

Evidence:
- [`03_search_and_index.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py#L85)
- [`03_search_and_index.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_search_and_index.py#L95)

Why this matters:
- Gap-heavy or partial matches can still receive a high identity value.
- Later, `build_minimal_template_arrays()` only populates aligned query positions and leaves the rest as gap or zero-mask.
- So ranking can favor a high-identity partial template over a broader but lower-identity full-chain template.

Evidence for later sparse filling:
- [`rna_template_common.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/rna_template_common.py#L446)
- [`rna_template_common.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/rna_template_common.py#L477)

Impact:
- Search quality may be acceptable for smoke tests
- Search quality is not yet robust enough for production or quantitative evaluation

Assessment:
- This is more of a ranking-quality issue than a wiring bug, but it directly affects template usefulness.

### 4.5 Low: There is a gap between documentation language and real primary-path implementation

Existing documentation repeatedly emphasizes `build_rna_template_protenix.py` as the builder.
That script is valid, but the main pipeline does not invoke it.

Evidence:
- Existing technical doc mentions `build_rna_template_protenix.py`: [`RNA_Template_Pipeline_Technical_Documentation.md`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/RNA_Template_Pipeline_Technical_Documentation.md)
- Main pipeline actually imports and calls `rna_template_common.py` through `02_build_rna_templates.py`: [`02_build_rna_templates.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L48)

Why this matters:
- The code path is still understandable.
- But it can confuse future maintenance, especially when debugging "which builder was actually used for this database?"

Impact:
- Documentation ambiguity
- Potential operator confusion

## 5. Detailed Call Graph

### 5.1 Self mode

Call graph:

`run_pipeline.sh`
-> `01_extract_rna_catalog.py`
-> `02_build_rna_templates.py --mode self`
-> `load_structure_residues()`
-> `build_minimal_template_arrays()`
-> `align_query_to_template()`
-> `compute_anchor()`
-> `compute_frame()`
-> `compute_distogram()`
-> `compute_unit_vectors()`
-> `stack_template_dicts()`
-> `save_npz_with_metadata()`
-> `03_search_and_index.py --strategy self`
-> `RNATemplateFeaturizer`
-> `Pairformer._single_rna_template_forward()`

### 5.2 Pairwise mode

Call graph:

`run_pipeline.sh`
-> `01_extract_rna_catalog.py`
-> `03_search_and_index.py --strategy pairwise`
-> `pairwise_search()`
-> `pairwise_identity()`
-> `02_build_rna_templates.py --mode cross`
-> `load_structure_residues()`
-> `build_minimal_template_arrays()`
-> `align_query_to_template()`
-> `compute_anchor()`
-> `compute_frame()`
-> `compute_distogram()`
-> `compute_unit_vectors()`
-> `stack_template_dicts()`
-> `save_npz_with_metadata()`
-> `03_search_and_index.py --strategy pairwise` again
-> `RNATemplateFeaturizer`
-> `Pairformer._single_rna_template_forward()`

## 6. Consistency Check Against Existing Report

The existing report at [`rna_template_search_pipeline_report_zh.md`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/code_review/rna_template_search_pipeline_report_zh.md) is broadly correct on the high-level architecture:

- It correctly identifies the main scripts.
- It correctly notes that `02_build_rna_templates.py` uses `rna_template_common.py`.
- It correctly states that the output flows into `RNATemplateFeaturizer` and then TemplateEmbedder/Pairformer.

However, the report does not sufficiently call out the following implementation realities:
- pairwise query identity is PDB-level, not chain-level, when driven by `rna_sequence_to_pdb_chains.json`
- the runtime featurizer does not merge multiple `.npz` files for one sequence
- modified-base normalization is inconsistent across template lookup vs RNA LM lookup
- pairwise identity is a partial-alignment-biased score, not a coverage-aware production metric

So the existing report is directionally right, but operationally optimistic.

## 7. Runtime Behavior Notes

### 7.1 What the model actually consumes

The model consumes these RNA template keys:
- `rna_template_aatype`
- `rna_template_distogram`
- `rna_template_pseudo_beta_mask`
- `rna_template_unit_vector`
- `rna_template_backbone_frame_mask`
- `rna_template_block_mask`

Evidence:
- [`rna_template_featurizer.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L247)
- [`pairformer.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/model/modules/pairformer.py#L1198)

### 7.2 What the builder writes beyond those core keys

The builder also writes extra metadata:
- `template_anchor_pos`
- `template_anchor_mask_1d`
- `template_frame_origin`
- `template_frame_axes`
- `template_frame_mask_1d`
- `template_mask`
- `template_names`
- `template_mapping_json`
- `query_length`

Evidence:
- [`rna_template_common.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/rna_template_common.py#L465)
- [`rna_template_common.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/compute/rna_template_common.py#L522)

But the runtime RNA featurizer currently does not consume `template_mask` or `template_mapping_json`.

Evidence:
- Core load list is only `RNA_TEMPLATE_FEATURES`: [`rna_template_featurizer.py`](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L30)

This matters because some information that could support smarter runtime assembly is already present in the files but unused.

## 8. Residual Risks

- Multi-chain RNA structures are not modeled cleanly in pairwise mode.
- Modified-base RNA may silently lose template coverage.
- Search ranking may overvalue short high-identity partial hits.
- Runtime template diversity is lower than the index structure suggests.
- Documentation can mislead operators into thinking the main pipeline is directly built around `build_rna_template_protenix.py`.

## 9. Final Assessment

Current status:
- The RNA template path is real and connected end to end.
- Template `.npz` generation, loading, and model injection are all present.
- For narrow self-template validation or limited smoke tests, the pipeline is serviceable.

Current limitations:
- The pairwise search pipeline is not yet a true chain-level RNA template search pipeline.
- The runtime consumer underuses the index structure.
- Sequence normalization behavior is not fully consistent across Protenix subsystems.

Bottom line:
- This is a functioning validation pipeline.
- It is not yet a production-grade or semantically complete RNA template pipeline.

## 10. No-Change Statement

This report is based on read-only inspection only.
No files were modified during the review itself.
