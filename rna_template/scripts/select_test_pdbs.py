#!/usr/bin/env python3
"""
Select a small subset of PDB IDs for end-to-end testing.

Picks PDB IDs from the training list that:
  - Have a bioassembly .pkl.gz file
  - Have a .cif file in the CIF directory
  - Have ≤ max_tokens tokens (small structures for fast testing)

---- CONFIGURABLE ----
Modify selection criteria by changing the scoring function or filters.
---- END CONFIGURABLE ----

Usage:
    python select_test_pdbs.py \
        --train_pdb_list /path/to/train_list.txt \
        --indices_file /path/to/indices.csv \
        --cif_dir /path/to/PDB_RNA \
        --bioassembly_dir /path/to/rna_bioassembly \
        --output /path/to/test_pdb_list.txt \
        --n_pdbs 5 \
        --max_tokens 100
"""

import argparse
import csv
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Select small PDBs for testing.")
    parser.add_argument("--train_pdb_list", required=True)
    parser.add_argument("--indices_file", required=True)
    parser.add_argument("--cif_dir", required=True)
    parser.add_argument("--bioassembly_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_pdbs", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--min_tokens", type=int, default=20)
    args = parser.parse_args()

    # Load training PDB list
    with open(args.train_pdb_list) as f:
        train_ids = {line.strip() for line in f if line.strip()}

    # Read indices for token counts
    pdb_tokens = {}
    with open(args.indices_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb = row["pdb_id"]
            tokens = int(row["num_tokens"])
            if pdb not in pdb_tokens or tokens < pdb_tokens[pdb]:
                pdb_tokens[pdb] = tokens

    # Find candidates
    candidates = []
    for pdb_id in sorted(train_ids):
        if pdb_id not in pdb_tokens:
            continue
        tokens = pdb_tokens[pdb_id]
        if tokens < args.min_tokens or tokens > args.max_tokens:
            continue
        cif = os.path.join(args.cif_dir, f"{pdb_id}.cif")
        bio = os.path.join(args.bioassembly_dir, f"{pdb_id}.pkl.gz")
        if os.path.exists(cif) and os.path.exists(bio):
            candidates.append((pdb_id, tokens))

    candidates.sort(key=lambda x: x[1])

    if not candidates:
        print(f"ERROR: No candidates found with {args.min_tokens}-{args.max_tokens} tokens")
        sys.exit(1)

    # Select diverse set: pick from different parts of the range
    selected = candidates[: args.n_pdbs]

    # Write output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for pdb_id, tokens in selected:
            f.write(f"{pdb_id}\n")

    print(f"Selected {len(selected)} PDBs (from {len(candidates)} candidates):")
    for pdb_id, tokens in selected:
        print(f"  {pdb_id}: {tokens} tokens")


if __name__ == "__main__":
    main()
