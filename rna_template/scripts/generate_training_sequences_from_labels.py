#!/usr/bin/env python3
"""
Build a Protenix training-sequence JSON from Stanford RNA labels CSV.

Input CSV format:
    ID,resname,resid,x_1,y_1,z_1,chain,copy

Output JSON format:
    {sequence: [PDB_ID, ...]}

For each PDB, this script reconstructs RNA sequences per (chain, copy) in the
original CSV row order, then selects one representative sequence:
  1. longest sequence
  2. lexicographically smallest (chain, copy) on ties
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


VALID_BASES = {"A", "C", "G", "U", "T", "N"}


def normalize_base(base: str) -> str:
    base = (base or "").strip().upper()
    if base == "T":
        return "U"
    if base in VALID_BASES:
        return base
    return "N"


def load_allowed_pdbs(path: str) -> List[str]:
    with open(path) as fh:
        return [line.strip().upper() for line in fh if line.strip()]


def reconstruct_sequences(
    labels_csv: str,
    allowed_pdbs: Iterable[str],
) -> Dict[str, Dict[Tuple[str, str], str]]:
    allowed = set(allowed_pdbs)
    residues: Dict[str, Dict[Tuple[str, str], List[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    with open(labels_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pdb_id = row["ID"].split("_", 1)[0].upper()
            if pdb_id not in allowed:
                continue
            key = (row["chain"], row["copy"])
            residues[pdb_id][key].append(normalize_base(row["resname"]))

    return {
        pdb_id: {key: "".join(seq) for key, seq in chain_map.items()}
        for pdb_id, chain_map in residues.items()
    }


def choose_representative(chain_map: Dict[Tuple[str, str], str]) -> str:
    best_key = min(chain_map, key=lambda k: (-len(chain_map[k]), k[0], k[1]))
    return chain_map[best_key]


def build_sequence_mapping(
    pdb_to_sequences: Dict[str, Dict[Tuple[str, str], str]],
    pdb_order: Iterable[str],
    min_length: int,
) -> Dict[str, List[str]]:
    sequence_to_pdbs: Dict[str, List[str]] = defaultdict(list)

    for pdb_id in pdb_order:
        chain_map = pdb_to_sequences.get(pdb_id, {})
        if not chain_map:
            continue
        sequence = choose_representative(chain_map)
        if len(sequence) < min_length:
            continue
        sequence_to_pdbs[sequence].append(pdb_id)

    return dict(sequence_to_pdbs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels_csv", required=True)
    parser.add_argument("--pdb_list", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--min_length", type=int, default=5)
    args = parser.parse_args()

    pdb_order = load_allowed_pdbs(args.pdb_list)
    pdb_to_sequences = reconstruct_sequences(args.labels_csv, pdb_order)
    missing = [p for p in pdb_order if p not in pdb_to_sequences]
    if missing:
        raise SystemExit(
            f"Missing {len(missing)} PDBs from labels CSV; first few: {missing[:10]}"
        )

    sequence_mapping = build_sequence_mapping(
        pdb_to_sequences=pdb_to_sequences,
        pdb_order=pdb_order,
        min_length=args.min_length,
    )

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(sequence_mapping, fh, indent=2, sort_keys=True)

    represented = sum(len(v) for v in sequence_mapping.values())
    print(f"pdbs_requested {len(pdb_order)}")
    print(f"pdbs_represented {represented}")
    print(f"unique_sequences {len(sequence_mapping)}")
    print(f"output_json {out_path}")


if __name__ == "__main__":
    main()
