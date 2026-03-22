# Protein Template Leakage Guard vs RNA Template Pipeline

## Scope

This report reviews how Protenix prevents template leakage when `protein template` is enabled, then compares that mechanism to the current `RNA template` pipeline.

The review is based on code reading and light artifact inspection only. I did not modify any code. I used:

- `codex-code-review`: prioritize confirmed risks, regressions, and hidden failure modes.
- `codex-bio-ai-research`: keep the biological notion of "leakage" separate from acceptable template homology.

## Biological Framing

In template-based structure prediction, "no leakage" does **not** mean "no similar template exists in the database".

What the code usually tries to prevent is:

1. The target structure itself being used as a template for that same target.
2. Structures released after the target date being used during training or evaluation.
3. Extremely close duplicate / near-self hits that trivially reveal the answer.

What the code usually still allows by design is:

1. Older homologous templates.
2. High-identity but not identical structures released before the query cutoff.
3. Different structures with similar folds or sequences that are biologically plausible templates.

That distinction matters here: the protein pipeline is built to prevent **temporal/self leakage**, not to ban all high-overlap homologs.

## Findings

### 1. Medium: the protein pipeline has a real temporal/self-leakage guard, but it depends on `release_date` being present in the training sample

Confirmed behavior:

- During training, `TemplateFeaturizer` reads `bioassembly_dict["release_date"]` and converts it to `query_release_date`: [template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_featurizer.py#L441)
- It then sets `cutoff_date = min(global_max_template_date, query_release_date - 60 days)`: [template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_featurizer.py#L353)
- The 60-day guard is hardcoded as `DAYS_BEFORE_QUERY_DATE = 60`: [template_utils.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_utils.py#L72)
- Hits after the cutoff are rejected in prefilter: [template_utils.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_utils.py#L352)

Residual risk:

- If `query_release_date` is missing, ordinary training falls back to `max_template_date = 3000-01-01`, which effectively disables temporal filtering for that sample: [template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_featurizer.py#L268), [template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_featurizer.py#L354)
- The training-data doc says `release_date` exists in the bioassembly dict schema, which is the intended safeguard: [prepare_training_data.md](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/docs/prepare_training_data.md#L74)

Interpretation:

- For correctly prepared data, this guard is sound.
- For malformed or incomplete prepared data, the protein pipeline can silently lose its temporal leakage protection.

### 2. Medium: the current RNA pipeline does not replicate the protein pipeline's per-query temporal cutoff

Confirmed behavior:

- RNA leakage control is done during the offline search/build pipeline, not during runtime template loading.
- The RNA search script can filter the whole catalog by a single global `--release_date_cutoff`: [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L704), [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L970)
- `run_pipeline.sh` passes that cutoff through if configured: [run_pipeline.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L115), [run_pipeline.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L168)

Difference from protein:

- Protein computes a **query-specific** cutoff from the current sample's own release date minus 60 days.
- RNA applies at most a **single global cutoff** to the database before search.

Why this matters:

- A global cutoff can prevent gross future leakage for an entire benchmark split.
- It cannot express "for query A, use templates before date A; for query B, use templates before date B".
- So the RNA pipeline is weaker than the protein pipeline for mixed-date training data.

### 3. Medium: the RNA runtime loader trusts the prebuilt index and does no secondary leakage filtering

Confirmed behavior:

- `RNATemplateFeaturizer` loads a JSON index and does exact sequence lookup, then loads the first valid `.npz`: [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L188), [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L197), [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L275), [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L287)
- At runtime it does **not** re-check:
  - release date
  - self-hit status
  - identity threshold
  - duplicate / near-self status

Why this matters:

- The protein path enforces its leakage rules at template-selection time for the current sample.
- The RNA path assumes the offline artifacts were built correctly and remain provenance-aligned.
- If the index/search artifacts are stale, mixed, or built with a different cutoff, runtime cannot detect that.

### 4. Medium: the RNA pipeline preserves self-exclusion during search, but that protection is weaker after collapsing to a sequence-keyed runtime index

Confirmed behavior:

- RNA search excludes targets from the same base PDB as the query: [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L308), [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L1016)
- Training sequences loaded from `rna_sequence_to_pdb_chains.json` become `{pdb_id -> sequence}` keys, so self-exclusion is at base-PDB granularity: [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L769)
- Built cross-templates remain query-specific on disk, e.g. `query_id_template.npz`: [02_build_rna_templates.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L282), [02_build_rna_templates.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L290)
- But the runtime index is keyed only by `query_sequence`, not by `(query_id, sequence)`: [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L683)

Why this is architecturally weaker:

- Self-exclusion is computed per query during search.
- That query identity is later discarded when multiple template bundles are stored under one sequence key.
- Runtime then picks the first valid `.npz` for the sequence, without knowing which query it was built for.

Artifact check:

- The current checked-in `rna_template_index.json` contains `4204` sequence keys.
- `1466` of those keys map to more than one `.npz` path.
- One sequence key maps to `407` paths.

Interpretation:

- Even if the search step excluded self-hits correctly, the runtime representation is not query-aware.
- This is materially weaker than the protein path, which always computes template eligibility for the current query itself.

### 5. Low: neither pipeline tries to remove all highly similar older homologs, and that is mostly by design

Protein path:

- Prefilter rejects only large near-duplicates where template sequence is contained in the query and length ratio exceeds `0.95`: [template_utils.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_utils.py#L356)
- It also deduplicates by hit sequence after prefilter: [template_utils.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_utils.py#L907)

RNA path:

- Search keeps hits above `min_identity` and deduplicates only by `(pdb_id, chain_id)`: [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L295), [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L326)

Interpretation:

- Protein is stricter against exact/near-self duplication.
- RNA is looser and more artifact-dependent.
- But both systems still intentionally allow biologically related older templates.

## How Protein Template Prevents Leakage

### 1. The template subsystem is configured with release-date and obsolete-PDB metadata

The default data config enables protein templates and wires in both the release-date cache and obsolete-PDB mapping:

- [configs_data.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/configs/configs_data.py#L232)
- [configs_data.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/configs/configs_data.py#L243)
- [configs_data.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/configs/configs_data.py#L246)

This matters because date filtering only works if the hit PDB can be mapped to an actual release date, including obsolete entries.

### 2. Each dataset sample creates a `TemplateFeaturizer`, and each query gets its own cutoff

The dataset factory constructs a `TemplateFeaturizer` for the active dataset/stage:

- [dataset.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/pipeline/dataset.py#L957)

Inside `TemplateFeaturizer`:

- Non-training stages use a fixed max date `2021-09-30`.
- Training uses either a benchmark-specific fixed date (`2018-04-30` for distillation/openprotein) or `3000-01-01` as the global cap for ordinary training.
- The actual per-sample guard comes from the query sample's own `release_date`.

Relevant code:

- [template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_featurizer.py#L268)
- [template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_featurizer.py#L295)
- [template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_featurizer.py#L353)
- [template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_featurizer.py#L441)

### 3. Hit prefilter removes future structures, large duplicates, and weak alignments

Before any hit becomes a usable template, the protein path checks:

1. Release date must be on or before cutoff.
2. Alignment coverage must be above `0.1`.
3. Large duplicate / near-self hits are rejected.
4. Very short template sequences are rejected.

Relevant code:

- [template_utils.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_utils.py#L348)
- [template_utils.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_utils.py#L374)

This is the main reason the protein system can safely search a database that still contains the training targets themselves: the search space may contain the target, but the target is filtered back out before feature extraction.

### 4. Obsolete PDB handling closes a common leakage hole

If a hit PDB ID is obsolete, the protein code remaps it through the obsolete-PDB table before checking the release date:

- [template_utils.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_utils.py#L374)

And the parser can recover first release date from obsolete history:

- [parser.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/core/parser.py#L188)

This is important because a naive cutoff using only current IDs can miss renamed or superseded entries.

### 5. Final processing also deduplicates by sequence and checks actual mmCIF-chain compatibility

After prefilter:

- hits are sorted and deduplicated by `hit_sequence`
- the query is realigned to the actual mmCIF chain sequence
- if similarity to the real chain is too low, the hit is rejected

Relevant code:

- [template_utils.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_utils.py#L907)
- [template_utils.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_utils.py#L565)

So the protein path is not just date-based. It is also sequence-aware and structure-aware during final feature extraction.

### 6. Template copying across chains is restricted to truly identical entities

Within one assembly, Protenix can reuse one entity's template features for another chain only if all chains under that entity have exactly the same sequence:

- [template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_featurizer.py#L140)
- [template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_featurizer.py#L634)

This is not the main leakage guard, but it avoids accidental cross-chain feature reuse when entity definitions are heterogeneous.

## How the RNA Template Pipeline Prevents Leakage

### 1. Leakage control is mostly offline, in the build pipeline

The RNA pipeline runs:

1. extract RNA catalog
2. search for templates with MMseqs2
3. build cross-template `.npz` files from search results
4. build a sequence-to-template index

See:

- [run_pipeline.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L157)

This is already a major architectural difference from protein, where template eligibility is still decided online per query.

### 2. RNA self-hit prevention happens during search

The RNA search logic:

- normalizes the query sequence
- derives the base PDB ID from the query ID
- excludes hits whose target base PDB matches the query base PDB

Relevant code:

- [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L419)
- [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L424)
- [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L308)

This is a valid self-exclusion guard, but weaker than protein because it operates only at search time and only on base-PDB identity.

### 3. RNA temporal filtering exists, but only as an optional global prefilter

If `--release_date_cutoff` is supplied:

- the script loads RNA3DB metadata
- removes database entries newer than that date
- then runs MMseqs2 search on the filtered catalog

Relevant code:

- [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L704)
- [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L969)

This is useful, but it is still not equivalent to protein's query-specific date guard.

### 4. The runtime featurizer only loads precomputed artifacts by sequence

At train/inference time:

- `get_rna_template_featurizer()` creates `RNATemplateFeaturizer`: [dataset.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/pipeline/dataset.py#L1090)
- the featurizer loads the JSON index: [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L188)
- it looks up by sequence: [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L197)
- it resolves candidate paths and uses the first valid `.npz`: [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L279), [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L287)

That means RNA runtime does not independently know:

- which query ID produced the `.npz`
- what date cutoff was used
- whether self-exclusion was applied for the current sample

## Direct Comparison

| Topic | Protein template | RNA template |
|---|---|---|
| Where leakage guard lives | Online, per query | Mostly offline, during search/build |
| Self-hit guard | Duplicate + date + current-query filtering | Base-PDB self-exclusion during search |
| Temporal guard | `query_release_date - 60 days` | Optional global `--release_date_cutoff` |
| Obsolete-ID handling | Yes | No equivalent obsolete remap found |
| Runtime re-check | Yes, during hit processing | No, runtime trusts prebuilt index |
| Dedup granularity | Dedup by hit sequence | Dedup by `(pdb_id, chain_id)` |
| High-similarity policy | Older homologs allowed, near-self rejected | Older homologs allowed, weaker duplicate control |

## Answer to the Core Question

### Does protein template mode really prevent leakage even though the training target exists in the template database?

Yes, **for correctly prepared data**, the protein pipeline is explicitly designed for that situation.

It does **not** remove the training target from the database ahead of time. Instead, it keeps the database broad and rejects forbidden hits online:

1. future hits are removed by release-date cutoff
2. same/near-self duplicate hits are removed
3. bad or weak alignments are removed
4. obsolete IDs are resolved before date comparison

So the target may be present in the raw template source, but it should not survive into template features for that same sample.

### Does it guarantee "no highly similar structure"?

No. It guarantees something narrower and more realistic:

- no future leakage
- no trivial self / near-self leakage

It still allows older homologous templates, because that is the intended use of structural templates.

### How does your RNA template differ?

Your RNA system has real safeguards, but they are weaker and more artifact-dependent:

1. self-exclusion is done during search, not during runtime loading
2. temporal filtering is global, not per query
3. runtime only trusts the prebuilt sequence index and cannot re-verify provenance
4. sequence-keyed indexing discards query identity, which is a real architectural downgrade versus protein

## Artifact Notes From This Review

I also checked a few real files to validate the code assumptions:

- `rna_sequence_to_pdb_chains.json` currently contains `3460` sequences, and in that file each sequence maps to one PDB ID.
- RNA3DB `filter.json` currently contains `15441` metadata entries with `release_date`.
- The current checked-in `rna_template_index.json` contains `4204` sequence keys, with `1466` keys mapping to multiple `.npz` files. One sequence key maps to `407` paths.
- The current checked-in `search_results.json` contains only `143` query entries and uses base-PDB query IDs like `5el7`, `2uxc`, `9o3j`.

Interpretation:

- The current RNA runtime artifacts are not obviously one-to-one with the intended query-specific search process.
- That strengthens the conclusion that the RNA path depends heavily on artifact consistency, while the protein path enforces more logic online.

## Code Annotations

### Protein template path

- [configs_data.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/configs/configs_data.py#L232)
  Protein template is enabled by default and is given release-date / obsolete-PDB metadata.

- [dataset.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/pipeline/dataset.py#L957)
  Dataset factory constructs the protein `TemplateFeaturizer`.

- [template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_featurizer.py#L353)
  The actual query-specific cutoff is computed here.

- [template_utils.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_utils.py#L348)
  Quick prefilter removes date-violating hits, weak alignments, and near-self duplicates.

- [template_utils.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_utils.py#L907)
  Hits are deduplicated by sequence before final collection.

- [template_utils.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_utils.py#L565)
  Query is realigned to the actual mmCIF chain and rejected if too dissimilar.

### RNA template path

- [run_pipeline.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L157)
  RNA pipeline is offline: search first, build `.npz`, then build index.

- [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L308)
  Self-exclusion is implemented here by base-PDB comparison.

- [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L704)
  Optional global release-date filtering is implemented here.

- [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L683)
  The runtime index is built by sequence, which discards query identity.

- [02_build_rna_templates.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L282)
  Cross-template `.npz` files still carry query-specific metadata on disk.

- [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L197)
  Runtime lookup is by sequence only.

- [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L287)
  Runtime loads only the first valid `.npz`, with no provenance re-check.

## Final Judgement

### Protein template

I do **not** see evidence of a major leakage bug in the protein template path itself.

The mechanism is coherent:

1. query-aware temporal cutoff
2. obsolete-ID-aware date checking
3. near-self duplicate rejection
4. final sequence/structure consistency checks

The main caveat is that this protection depends on valid `release_date` in prepared training data.

### RNA template

Your RNA pipeline has improved a lot, but it is still **not equivalent** to the protein template leakage model.

It currently provides:

1. offline self-exclusion
2. optional global date cutoff

But it still lacks:

1. query-specific temporal filtering at runtime
2. query-aware runtime selection of template bundles
3. online secondary verification after loading the index

So the correct comparison is:

- `protein template`: stronger, online, query-aware leakage control
- `rna template`: weaker, offline, artifact-dependent leakage control
