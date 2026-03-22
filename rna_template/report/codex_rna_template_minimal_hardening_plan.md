# RNA Template Leakage Hardening Plan (Minimal-Change Version)

## Goal

This report proposes a **small-change** plan to make the current RNA template pipeline closer to the protein template leakage policy, while avoiding disruption to:

- the existing training loop
- the existing LLM finetune architecture
- `rna_template.enable=false` behavior
- checkpoint format and model structure

The design target is not to fully re-implement the protein template system online.  
The target is to patch the **highest-value leakage gaps** using the code you already have.

## Constraints

The plan below is designed to satisfy these constraints:

1. No change to model topology.
2. No change to loss, optimizer, trainer core, or checkpoint schema.
3. No change to the `RNA template disable -> old LLM finetune path` behavior.
4. Keep existing offline RNA database workflow.
5. Prefer changes limited to RNA template search/build/load code.

## Current Architecture Summary

Today the RNA pipeline works like this:

1. offline search builds `search_results.json`
2. offline builder writes `query_id_template.npz`
3. offline index maps `sequence -> [npz paths]`
4. runtime loader looks up by sequence and loads the first valid `.npz`

Relevant code:

- search/build entrypoint: [run_pipeline.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh#L157)
- search logic: [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L352)
- build output naming: [02_build_rna_templates.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L290)
- runtime sequence lookup: [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L197)
- runtime first-valid selection: [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py#L287)

The main weakness is that runtime knows the **current query sample**, but does not use that identity when choosing among multiple `.npz` files under the same sequence key.

## Recommended Plan

## Phase 1: Runtime Exact-Query Preference

### What to change

Modify only:

- [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py)

Add a lightweight candidate-ranking step before the current "load first valid `.npz`" behavior:

1. read current sample `pdb_id` from `bioassembly_dict`
2. for candidate template paths under the same sequence:
   - prefer the path whose `query_id` matches the current sample
   - if exact match is unavailable, prefer same base PDB
   - otherwise fall back to the current order

This can be done without touching model code.  
The runtime loader already receives `bioassembly_dict`, and training samples already have `pdb_id` injected:

- [dataset.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/pipeline/dataset.py#L361)

### How to derive query identity with minimal disruption

Use existing filename conventions first:

- `02_build_rna_templates.py` writes files as `query_id_template.npz`: [02_build_rna_templates.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/02_build_rna_templates.py#L290)

So the loader can parse `query_id` from filename without requiring a database rebuild.

If later you regenerate templates with `.meta.json` sidecars, the loader can optionally use metadata too, but that is not required for the first patch.

### Why this is the best first patch

This is the smallest high-value fix because it addresses the current architectural weakness directly:

- current index is sequence-keyed
- multiple `.npz` paths can exist per sequence
- runtime currently ignores the current sample identity

By preferring the current sample's own query bundle, you remove most of the ambiguity without changing the index format, database structure, or training loop.

### Compatibility impact

- `rna_template.enable=false`: no effect
- original LLM training path: no effect
- checkpoint loading: no effect
- model architecture: no effect
- inference: no effect unless the sequence key has multiple candidates, in which case behavior becomes more deterministic

### Safety recommendation

Do **not** make this strict at first.  
Use this policy:

1. exact query match if available
2. same-base-PDB match if available
3. fallback to current behavior

That keeps backward compatibility with older artifact layouts.

## Phase 2: Build a Canonical `pdb_id -> release_date` Table and Refresh `rna_bioassembly_indices.csv`

### What to change

Modify mainly:

- [rna_bioassembly_indices.csv](/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/protenix_prepared/indices/rna_bioassembly_indices.csv)
- [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py)
- optionally [run_pipeline.sh](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/run_pipeline.sh) if you want a CLI flag for margin days

The first step should be to construct one canonical release-date lookup for RNA training/query entries.

After that, use the canonical lookup to drive the **query-specific** date filter during MMseqs2 result parsing:

1. look up the current query's release date
2. compute `query_release_date - 60 days`
3. reject targets newer than that cutoff

This mirrors the protein policy conceptually:

- protein uses `query_release_date - 60 days`: [template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_featurizer.py#L357)
- protein defines the margin in one place: [template_utils.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/template/template_utils.py#L72)

### Why this is still a small patch

This still stays outside the model/trainer.  
It does **not** touch:

- trainer
- dataloader contract
- model
- template embedder
- finetune checkpoint behavior

It affects:

1. the canonical release-date values stored in `rna_bioassembly_indices.csv`
2. which hits are written into future `search_results.json`
3. which templates are written into future `.npz` bundles

### Suggested implementation detail

Use the existing RNA3DB metadata file already supported by the script as the **primary** date source, and treat the prepared-data CSV as the canonical training-set table to refresh:

- metadata filtering path already exists: [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L704)
- CLI already accepts `--rna3db_metadata`: [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py#L943)
- the current prepared-data index already contains `release_date` and should be treated as the place to normalize/fix values for downstream reuse: [rna_bioassembly_indices.csv](/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/protenix_prepared/indices/rna_bioassembly_indices.csv)

Recommended logic:

1. read [rna_bioassembly_indices.csv](/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/protenix_prepared/indices/rna_bioassembly_indices.csv)
2. normalize `pdb_id` to a canonical key, e.g. lowercase 4-char base ID
3. load RNA3DB `filter.json` and build `filter_lookup[pdb_id] -> release_date`
4. overwrite or fill the CSV `release_date` column from `filter_lookup` wherever a match exists
5. for PDB IDs not covered by `filter.json`, query the official RCSB PDB Data API online
6. extract `entry.rcsb_accession_info.initial_release_date` / `rcsb_accession_info.initial_release_date` from the response and fill the missing date
7. optionally query the official RCSB holdings status endpoint first for IDs that fail, to distinguish current / removed / unresolved entries
8. write out a refreshed, complete `rna_bioassembly_indices.csv`
9. build `query_date_lookup` from that refreshed CSV and use it inside `03_mmseqs2_search.py`
10. for targets, continue to prefer RNA3DB metadata first, then optionally the same refreshed CSV if a target is also present there
11. reject any target hit where `target_release_date > query_release_date - 60 days`
12. if query date is still missing after all sources, keep current behavior and log the unresolved PDB IDs to an audit file instead of failing the whole build

This gives a small but robust implementation:

- `filter.json` stays the main source because the script already depends on RNA3DB metadata format
- `rna_bioassembly_indices.csv` becomes the canonical training-set lookup consumed by later scripts
- online RCSB lookup only runs for the residual uncovered IDs
- unresolved IDs become explicit and auditable instead of silently skipping date logic

Concretely, the lookup shape should be:

```text
query_date_lookup[pdb_id] -> "YYYY-MM-DD"
target_date_lookup[pdb_id] -> "YYYY-MM-DD"
```

with:

- `query_date_lookup` built from the refreshed `indices/rna_bioassembly_indices.csv`
- `target_date_lookup` built from RNA3DB `filter.json` first, then optionally supplemented from the refreshed CSV

### Why update `rna_bioassembly_indices.csv` instead of keeping the lookup only in memory

Because `rna_bioassembly_indices.csv` is the natural place to persist the cleaned training-set release dates.

That gives you:

1. one reusable `pdb_id -> release_date` table for future RNA search builds
2. easier auditing of which entries came from RNA3DB vs online fallback
3. less hidden logic inside `03_mmseqs2_search.py`
4. a stable input for later validation scripts

### Evidence from the current data layout

In the current prepared data:

- `rna_bioassembly_indices.csv` already has a `release_date` column
- local inspection shows no nulls in the current file
- but RNA3DB `filter.json` does not cover all PDB IDs present in the prepared CSV

The implication is:

- this is not a "create the column from scratch" problem
- it is a "canonicalize, verify, and complete the release-date table" problem

That is exactly why the plan should center around refreshing `rna_bioassembly_indices.csv`.

### Official online fallback source

For online completion, use official RCSB PDB endpoints:

1. RCSB Data API as the main source for release date
2. RCSB holdings/status endpoint as an auxiliary status check

Official documentation:

- RCSB Search API docs explicitly state that if you need release date, you should use the RCSB Data API: https://search.rcsb.org/
- RCSB Data API docs describe the REST base path `https://data.rcsb.org/rest/v1/core`
- RCSB migration guide maps `release_date` to `entry.rcsb_accession_info.initial_release_date` and documents the holdings status endpoints: https://data.rcsb.org/migration-guide.html

Practical fallback order:

```text
RNA3DB filter.json
-> refreshed rna_bioassembly_indices.csv
-> RCSB Data API core entry lookup
-> RCSB holdings/status check for unresolved IDs
```

### Minimal implementation boundary

Even with this fuller plan, I would still keep the actual code boundary small:

1. one refresh script or utility for `rna_bioassembly_indices.csv`
2. one small hook in `03_mmseqs2_search.py` to load the refreshed CSV lookup
3. no changes to model / trainer / checkpoint logic

This is a better fit for your current data layout than relying on `rna_sequence_to_pdb_chains.json`, because that JSON only contains:

```text
sequence -> [pdb_id]
```

and does not carry release dates itself.

That preserves robustness while materially strengthening leakage control.

### Why not rely only on global `--release_date_cutoff`

Because global cutoff is too coarse.

It can protect a whole split, but it cannot express:

- query A must use templates older than date A
- query B must use templates older than date B

The protein system is query-aware.  
This phase is the closest small patch that moves RNA in that direction.

## Phase 3: Optional Strict Provenance Mode

### What to change

Add an optional RNA-template config flag, for example:

- `rna_template.strict_query_match: false`

When enabled:

1. if multiple `.npz` candidates exist for the sequence
2. and none matches the current query identity
3. return empty RNA template features instead of silently loading an unrelated candidate

### Why this should stay optional

This is a policy change, not just a bug fix.

For old databases or incomplete indexes, strict mode could reduce template coverage.  
That is acceptable for auditing or high-rigor experiments, but it should not be forced on all existing runs immediately.

### Why this does not affect the old architecture

When `rna_template.enable=false`, the flag is unused.  
When `rna_template.enable=true` but strict mode is off, behavior stays backward compatible.

## What I Do Not Recommend for a "Small Change" Patch

These ideas are valid in principle, but they are larger than necessary for your stated goal:

1. Rebuilding RNA templates online like the protein template pipeline.
2. Replacing the sequence-key index with a brand-new query-aware database schema.
3. Touching TemplateEmbedder, Pairformer, or checkpoint/projector logic.
4. Making RNA template mandatory-fail if any metadata is missing.

Those changes increase scope, compatibility risk, and review burden.  
They are not needed for a first hardening pass.

## Concrete Patch Scope

If you want the **smallest practical patch set**, I recommend this exact scope:

### Patch A

File:

- [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py)

Behavior:

- rank candidate `.npz` files using current `bioassembly_dict["pdb_id"]`
- prefer exact-query or same-base-PDB candidate
- otherwise fallback to existing order

Risk:

- very low

Training/inference compatibility:

- preserved

### Patch B

File:

- [03_mmseqs2_search.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_template/scripts/03_mmseqs2_search.py)

Behavior:

- add per-query temporal filtering during search result parsing
- use the same 60-day concept as the protein pipeline
- fallback gracefully when query date is unavailable

Risk:

- low to medium

Training/inference compatibility:

- preserved; only future RNA template artifacts change

### Optional Patch C

File:

- [rna_template_featurizer.py](/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/protenix/data/rna_template/rna_template_featurizer.py)

Behavior:

- optional strict provenance mode

Risk:

- low, if default stays off

Training/inference compatibility:

- preserved by default

## Why This Does Not Affect Your Original Training / LLM Path

These proposed changes stay outside:

- `runner/train.py`
- model definition
- RNA projector / LLM projector code
- checkpoint load semantics
- optimizer / EMA / loss

They only affect:

1. which RNA template artifact gets selected
2. which hits get written into future RNA template artifacts

So:

- `rna_template.disable` still returns to the original finetune architecture
- the LLM path does not change
- backward compatibility remains intact

## Suggested Validation Plan

After implementing the small patch set, I would validate in this order.

### Test 1: RNA disabled regression

Run the existing backward-compat smoke path with:

- `rna_template.enable=false`

Expected:

- identical model creation path
- identical checkpoint load path
- identical train/eval startup behavior

### Test 2: Runtime query preference

Construct a small synthetic index where one sequence maps to:

1. current-query `.npz`
2. unrelated-query `.npz`

Expected:

- loader selects the current-query candidate first

### Test 3: Fallback compatibility

Construct a small synthetic index where:

1. multiple candidates exist
2. none matches current query

Expected:

- with default mode, loader falls back to current behavior
- with strict mode off, no regression

### Test 4: Per-query date filter

Use MMseqs2 parse-unit or a mocked TSV result set:

1. query date = `2021-10-01`
2. target A date = `2021-07-01`
3. target B date = `2021-09-15`

Expected with 60-day margin:

- target A kept
- target B removed

### Test 5: End-to-end rebuild sanity

Rebuild a small RNA template database and confirm:

1. search still completes
2. `.npz` still build
3. index still loads
4. training with RNA template still starts

## Final Recommendation

If the goal is:

- smaller patch
- low regression risk
- no disruption to the old LLM finetune path

then the right order is:

1. **Phase 1 first**: runtime exact-query preference
2. **Phase 2 second**: per-query temporal cutoff in offline search
3. **Phase 3 optional**: strict provenance mode

This gives you the highest leakage-control improvement per line changed, while staying compatible with the current architecture.

## Bottom Line

The most practical minimal-change hardening plan is:

1. make runtime RNA template selection query-aware
2. make offline RNA search cutoff query-aware
3. keep fallback-compatible defaults so old training behavior does not break

That is the smallest path that materially improves RNA leakage control without changing your existing model/training/LLM architecture.
