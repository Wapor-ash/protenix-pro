#!/usr/bin/env python3
"""
Build a training-sequence mapping JSON for RNA template search from prepared
bioassembly pickles.

Output format matches the existing RNA MSA mapping schema:
    {sequence: [PDB_ID, ...]}

For each PDB, this script selects one representative RNA query sequence:
  1. Keep entities whose entity_poly_type contains "ribonucleotide".
  2. Choose the longest RNA sequence.
  3. Break ties by smaller entity_id for deterministic output.
"""

import argparse
import gzip
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def normalize_rna_sequence(seq: str) -> str:
    seq = seq.upper().replace("T", "U")
    return "".join(c if c in {"A", "G", "C", "U", "N"} else "N" for c in seq)


def is_rna_entity(poly_type: str) -> bool:
    return "ribonucleotide" in (poly_type or "").lower()


def entity_sort_key(entity_id: str) -> Tuple[int, str]:
    try:
        return (0, f"{int(entity_id):08d}")
    except (TypeError, ValueError):
        return (1, str(entity_id))


def select_query_sequence(bioassembly_path: Path) -> Tuple[str, int]:
    with gzip.open(bioassembly_path, "rb") as fh:
        obj = pickle.load(fh)

    sequences = obj.get("sequences", {})
    entity_poly_type = obj.get("entity_poly_type", {})

    rna_entities: List[Tuple[str, str]] = []
    for entity_id, seq in sequences.items():
        poly_type = entity_poly_type.get(str(entity_id), entity_poly_type.get(entity_id, ""))
        if not is_rna_entity(poly_type):
            continue
        norm_seq = normalize_rna_sequence(seq)
        if len(norm_seq) < 5:
            continue
        rna_entities.append((str(entity_id), norm_seq))

    if not rna_entities:
        return "", 0

    rna_entities.sort(key=lambda item: (-len(item[1]), entity_sort_key(item[0])))
    return rna_entities[0][1], len(rna_entities)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RNA template training_sequences JSON from prepared bioassembly pickles."
    )
    parser.add_argument(
        "--bioassembly_dir",
        default="/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/protenix_prepared/rna_bioassembly",
        help="Directory containing prepared *.pkl.gz bioassembly files.",
    )
    parser.add_argument(
        "--pdb_list",
        default="/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/protenix_prepared/rna_train_pdb_list_filtered.txt",
        help="Filtered training PDB list (one lowercase/uppercase PDB ID per line).",
    )
    parser.add_argument(
        "--output_json",
        default="/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/protenix_prepared/rna_msa/rna_sequence_to_pdb_chains_train5574.json",
        help="Output mapping JSON path.",
    )
    parser.add_argument(
        "--seed_json",
        default="",
        help="Optional existing mapping JSON to reuse first before filling missing PDBs from bioassembly.",
    )
    args = parser.parse_args()

    bioassembly_dir = Path(args.bioassembly_dir)
    pdb_list_path = Path(args.pdb_list)
    output_json = Path(args.output_json)

    pdb_ids = [line.strip().lower() for line in pdb_list_path.read_text().splitlines() if line.strip()]
    seq_to_pdbs: Dict[str, List[str]] = defaultdict(list)
    assigned_pdbs = set()

    if args.seed_json:
        with open(args.seed_json) as fh:
            seed_mapping = json.load(fh)
        for seq, pdb_list in seed_mapping.items():
            norm_seq = normalize_rna_sequence(seq)
            for pdb_id in pdb_list:
                pdb_id_lower = str(pdb_id).lower()
                if pdb_id_lower in assigned_pdbs:
                    continue
                if pdb_id_lower not in pdb_ids:
                    continue
                seq_to_pdbs[norm_seq].append(pdb_id_lower.upper())
                assigned_pdbs.add(pdb_id_lower)
        print(
            f"Seed mapping reused: {len(assigned_pdbs)} PDBs from {args.seed_json}"
        )

    missing_pickles = []
    skipped_no_rna = []
    multi_rna = 0

    for idx, pdb_id in enumerate(pdb_ids, start=1):
        if pdb_id in assigned_pdbs:
            continue
        if idx % 500 == 0:
            print(f"Progress: {idx}/{len(pdb_ids)}")

        bioassembly_path = bioassembly_dir / f"{pdb_id}.pkl.gz"
        if not bioassembly_path.exists():
            missing_pickles.append(pdb_id)
            continue

        query_seq, num_rna_entities = select_query_sequence(bioassembly_path)
        if not query_seq:
            skipped_no_rna.append(pdb_id)
            continue

        if num_rna_entities > 1:
            multi_rna += 1

        seq_to_pdbs[query_seq].append(pdb_id.upper())

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as fh:
        json.dump(dict(seq_to_pdbs), fh, indent=2)

    total_refs = sum(len(v) for v in seq_to_pdbs.values())
    print(f"Output written: {output_json}")
    print(f"Input PDBs: {len(pdb_ids)}")
    print(f"Selected PDBs: {total_refs}")
    print(f"Unique sequences: {len(seq_to_pdbs)}")
    print(f"Multi-RNA PDBs (resolved by longest sequence): {multi_rna}")
    print(f"Missing bioassembly pickles: {len(missing_pickles)}")
    print(f"Skipped with no RNA sequence: {len(skipped_no_rna)}")
    if missing_pickles:
        print(f"Sample missing pickles: {missing_pickles[:10]}")
    if skipped_no_rna:
        print(f"Sample skipped no-RNA: {skipped_no_rna[:10]}")


if __name__ == "__main__":
    main()
