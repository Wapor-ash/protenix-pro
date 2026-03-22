#!/usr/bin/env python3
"""
Step 1: Extract RNA chain sequences from mmCIF files in the PDB_RNA database.

Produces a JSON catalog mapping PDB IDs to their RNA chains with sequences.
This catalog is used by downstream scripts to build templates and search.

Output format:
{
    "pdb_id": [
        {"chain_id": "A", "sequence": "AGCU...", "num_residues": 76},
        ...
    ],
    ...
}

Usage:
    python 01_extract_rna_catalog.py \
        --pdb_rna_dir /path/to/PDB_RNA \
        --output /path/to/rna_catalog.json \
        --max_structures 0          # 0 = all
        --min_length 10             # skip very short chains
        --max_length 2000           # skip very long chains
        --num_workers 8
"""

import argparse
import glob
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _extract_chains_from_cif(cif_path: str, min_length: int = 10, max_length: int = 2000) -> List[dict]:
    """Extract RNA chains from a single mmCIF file.

    Returns list of {chain_id, sequence, num_residues} dicts.
    """
    from Bio.PDB import MMCIFParser

    # ---- CONFIGURABLE: modify parser or add custom logic here ----
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("rna", cif_path)
    except Exception:
        return []

    model = next(structure.get_models())
    results = []

    for chain in model:
        residues = []
        for res in chain:
            # Skip hetero residues that are water/ions
            hetfield = res.id[0]
            if hetfield == "W":
                continue

            atom_dict = {}
            for atom in res.get_atoms():
                name = atom.get_name().strip().replace("*", "'").upper()
                atom_dict[name] = True

            # Check if this looks like a nucleotide
            resname = res.resname.strip().upper()
            is_nuc = False
            if resname in {"A", "G", "C", "U", "DA", "DG", "DC", "DT"}:
                is_nuc = True
            elif "C1'" in atom_dict or "C1*" in atom_dict:
                is_nuc = True
            elif "P" in atom_dict and ("C4'" in atom_dict or "C4*" in atom_dict):
                is_nuc = True

            if not is_nuc:
                continue

            # Determine base letter
            base = _infer_base(resname, atom_dict)
            residues.append(base)

        if len(residues) < min_length or len(residues) > max_length:
            continue

        seq = "".join(residues)
        # Only keep if it looks like RNA (has U) or is all nucleotides
        # Convert T -> U for RNA-centric usage
        seq = seq.replace("T", "U")

        results.append({
            "chain_id": chain.id,
            "sequence": seq,
            "num_residues": len(seq),
        })

    return results


# ---- CONFIGURABLE: modified base mapping ----
_MODIFIED_BASE_MAP = {
    "PSU": "U", "H2U": "U", "5MU": "U", "OMU": "U", "UR3": "U",
    "5MC": "C", "OMC": "C", "DC": "C",
    "1MA": "A", "M2A": "A", "6MZ": "A", "DA": "A",
    "2MG": "G", "7MG": "G", "OMG": "G", "M2G": "G", "DG": "G",
    "DT": "U", "T": "U",
}


def _infer_base(resname: str, atom_dict: dict) -> str:
    rn = resname.strip().upper()
    if rn in _MODIFIED_BASE_MAP:
        return _MODIFIED_BASE_MAP[rn]
    if rn in {"A", "G", "C", "U"}:
        return rn
    if "N6" in atom_dict:
        return "A"
    if "O6" in atom_dict or "N2" in atom_dict:
        return "G"
    if "N4" in atom_dict:
        return "C"
    if "O4" in atom_dict:
        return "U"
    if rn.startswith("D") and len(rn) >= 2 and rn[1] in {"A", "G", "C", "T"}:
        return "U" if rn[1] == "T" else rn[1]
    return "N"


def _process_one(args: Tuple[str, int, int]) -> Tuple[str, List[dict]]:
    """Process a single CIF file. Returns (pdb_id, chains).

    Handles both flat naming (e.g. 1abc.cif) and rna3db naming (e.g. 1abc_B.cif).
    For rna3db naming, pdb_id is the part before the underscore.
    """
    cif_path, min_len, max_len = args
    stem = Path(cif_path).stem  # e.g. "1abc" or "1abc_B"
    try:
        chains = _extract_chains_from_cif(cif_path, min_len, max_len)
        # Store source path for downstream pipeline
        for ch in chains:
            ch["cif_path"] = cif_path
        return stem, chains
    except Exception as e:
        return stem, []


def main():
    parser = argparse.ArgumentParser(description="Extract RNA chain catalog from PDB_RNA CIF files.")
    parser.add_argument("--pdb_rna_dir",
                        default="/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/PDB_RNA",
                        help="Directory containing .cif files.")
    parser.add_argument("--output",
                        default="/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/rna_catalog.json",
                        help="Output JSON catalog path.")
    parser.add_argument("--max_structures", type=int, default=0,
                        help="Max structures to process (0 = all). For testing, use e.g. 100.")
    parser.add_argument("--min_length", type=int, default=10, help="Min RNA chain length.")
    parser.add_argument("--max_length", type=int, default=2000, help="Max RNA chain length.")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel workers.")
    # ---- CONFIGURABLE: add pdb_list filter ----
    parser.add_argument("--pdb_list", default="",
                        help="Optional file with PDB IDs to limit extraction (one per line).")
    args = parser.parse_args()

    # Collect CIF files: support both flat and nested (rna3db-mmcifs) layouts
    cif_files = sorted(glob.glob(os.path.join(args.pdb_rna_dir, "*.cif")))
    if not cif_files:
        # Try nested layout: rna3db-mmcifs/{train_set,test_set}/component_*/pdb_chain/*.cif
        cif_files = sorted(glob.glob(os.path.join(args.pdb_rna_dir, "**", "*.cif"), recursive=True))
    if not cif_files:
        print(f"ERROR: No .cif files found in {args.pdb_rna_dir} (checked flat and nested layouts)")
        sys.exit(1)

    # Optional PDB list filter
    if args.pdb_list and os.path.exists(args.pdb_list):
        with open(args.pdb_list) as f:
            allowed_ids = {line.strip().lower() for line in f if line.strip()}
        cif_files = [p for p in cif_files if Path(p).stem.lower() in allowed_ids]
        print(f"Filtered to {len(cif_files)} CIF files by PDB list.")

    if args.max_structures > 0:
        cif_files = cif_files[:args.max_structures]

    print(f"Processing {len(cif_files)} CIF files from {args.pdb_rna_dir}")

    # Process in parallel
    tasks = [(p, args.min_length, args.max_length) for p in cif_files]
    catalog = {}
    n_chains = 0

    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(_process_one, t): t[0] for t in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 500 == 0:
                print(f"  Progress: {done}/{len(tasks)}")
            pdb_id, chains = future.result()
            if chains:
                catalog[pdb_id] = chains
                n_chains += len(chains)

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"\nCatalog saved: {args.output}")
    print(f"  Structures with RNA chains: {len(catalog)}")
    print(f"  Total RNA chains: {n_chains}")

    # Print length distribution
    lengths = [c["num_residues"] for chains in catalog.values() for c in chains]
    if lengths:
        import statistics
        print(f"  Length range: {min(lengths)}-{max(lengths)}")
        print(f"  Median length: {statistics.median(lengths):.0f}")


if __name__ == "__main__":
    main()
