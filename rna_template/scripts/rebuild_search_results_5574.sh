#!/usr/bin/env bash
set -euo pipefail

export PROJ=/inspire/ssd/project/sais-bio/public/ash_proj
export PROTENIX=$PROJ/code/protenix_new/Protenix
export PREP=$PROJ/data/stanford-rna-3d-folding/part2/protenix_prepared
export RNA_DB=$PROTENIX/rna_database
export FULL_CATALOG=$RNA_DB/rna_catalog.json
export TRAIN_QUERY_CATALOG=$RNA_DB/train_query_catalog_5574.json
export TRAIN_QUERY_JSON=$PREP/rna_msa/rna_sequence_to_pdb_chains_train5574.json
export SUMMARY_JSON=$RNA_DB/search_results_5574_summary.json
export SEARCH_JSON=$RNA_DB/search_results.json
export UNUSED_INDEX=$RNA_DB/rna_template_index.5574_unused.json

echo "[$(date -u)] waiting for current full catalog process if still alive"
while kill -0 40283 2>/dev/null; do
  sleep 60
done

set +e
python3 - <<'PY'
import json
import os
import sys

p = os.environ["FULL_CATALOG"]
ok = False
if os.path.exists(p):
    try:
        with open(p) as f:
            d = json.load(f)
        ok = isinstance(d, dict) and len(d) >= 1000
        print(f"catalog_check entries={len(d) if isinstance(d, dict) else 'bad'} ok={ok}")
    except Exception as e:
        print(f"catalog_check failed: {e}")
if not ok:
    sys.exit(1)
PY
catalog_ok=$?
set -e

if [ "$catalog_ok" -ne 0 ]; then
  echo "[$(date -u)] catalog invalid or still small, rebuilding full RNA3D catalog"
  python3 "$PROTENIX/rna_template/scripts/01_extract_rna_catalog.py" \
    --pdb_rna_dir "$PROJ/data/RNA3D/rna3db-mmcifs" \
    --output "$FULL_CATALOG" \
    --max_structures 0 \
    --min_length 10 \
    --max_length 2000 \
    --num_workers 8
fi

echo "[$(date -u)] extracting 5574 training-query catalog from part2/PDB_RNA"
python3 "$PROTENIX/rna_template/scripts/01_extract_rna_catalog.py" \
  --pdb_rna_dir "$PROJ/data/stanford-rna-3d-folding/part2/PDB_RNA" \
  --pdb_list "$PREP/rna_train_pdb_list_filtered.txt" \
  --output "$TRAIN_QUERY_CATALOG" \
  --max_structures 0 \
  --min_length 10 \
  --max_length 2000 \
  --num_workers 8

echo "[$(date -u)] building 5574-query training_sequences JSON"
python3 - <<'PY'
import json
import os
from pathlib import Path

catalog_path = Path(os.environ["TRAIN_QUERY_CATALOG"])
out_path = Path(os.environ["TRAIN_QUERY_JSON"])
with catalog_path.open() as f:
    catalog = json.load(f)

seq_to_pdbs = {}
for entry_id, chains in catalog.items():
    base_pdb = entry_id.split("_")[0].upper()
    best = None
    for chain in chains:
        seq = chain.get("sequence", "").upper().replace("T", "U")
        seq = "".join(c if c in "AGCUN" else "N" for c in seq)
        if len(seq) < 5:
            continue
        chain_id = str(chain.get("chain_id", ""))
        key = (len(seq), chain_id)
        if best is None or key > best[0]:
            best = (key, seq)
    if best is None:
        continue
    seq = best[1]
    seq_to_pdbs.setdefault(seq, [])
    if base_pdb not in seq_to_pdbs[seq]:
        seq_to_pdbs[seq].append(base_pdb)

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    json.dump(seq_to_pdbs, f, indent=2)

print(
    f"training_query_json unique_sequences={len(seq_to_pdbs)} "
    f"total_refs={sum(len(v) for v in seq_to_pdbs.values())}"
)
PY

echo "[$(date -u)] running MMseqs2 search for 5574-PDB training set"
python3 "$PROTENIX/rna_template/scripts/03_mmseqs2_search.py" \
  --catalog "$FULL_CATALOG" \
  --template_dir "$RNA_DB/templates" \
  --training_sequences "$TRAIN_QUERY_JSON" \
  --output_index "$UNUSED_INDEX" \
  --output_search "$SEARCH_JSON" \
  --strategy mmseqs2 \
  --min_identity 0.3 \
  --max_templates 4 \
  --sensitivity 7.5 \
  --evalue 1e-3 \
  --num_threads 8

echo "[$(date -u)] summarizing search results"
python3 - <<'PY'
import json
import os
import statistics
from pathlib import Path

search_path = Path(os.environ["SEARCH_JSON"])
with search_path.open() as f:
    data = json.load(f)

hit_counts = [len(v.get("templates", [])) for v in data.values()]
summary = {
    "query_entries_with_hits": len(data),
    "avg_templates_per_hit_query": round(sum(hit_counts) / max(len(hit_counts), 1), 4),
    "median_templates_per_hit_query": statistics.median(hit_counts) if hit_counts else 0,
    "max_templates_on_any_query": max(hit_counts) if hit_counts else 0,
    "search_results_path": str(search_path),
    "catalog_path": os.environ["FULL_CATALOG"],
    "training_query_json": os.environ["TRAIN_QUERY_JSON"],
}
with open(os.environ["SUMMARY_JSON"], "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
PY

echo "[$(date -u)] background rebuild finished"
