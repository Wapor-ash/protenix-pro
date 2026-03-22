"""
Check for missing RNA/DNA embeddings that would cause "key error" during training.

For each bioassembly .pkl.gz:
  - Extract entity_poly_type to classify entities as RNA, DNA, protein, or hybrid
  - Look up the sequence string in the corresponding embedding CSV manifest
  - Report any sequences not found in the manifest
"""

import gzip
import pickle
import os
import sys
import csv
from collections import defaultdict
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

OUTPUT_FILE = "/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/missing_embeddings_report.txt"

BIOASSEMBLY_DIR = "/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/protenix_prepared/rna_bioassembly/"
RNA_CSV = "/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/aido_embeddings/rna/rna_sequences.csv"
DNA_CSV = "/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/aido_embeddings/dna/dna_sequences.csv"


def load_manifest_sequences(csv_path):
    """Load the set of sequences from an embedding CSV manifest."""
    seqs = set()
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seqs.add(row["seq"])
    return seqs


def classify_entity(poly_type_str):
    """
    Classify entity_poly_type string into 'rna', 'dna', 'hybrid', 'protein', or 'other'.

    Common values:
      - 'polyribonucleotide' -> rna
      - 'polydeoxyribonucleotide' -> dna
      - 'polydeoxyribonucleotide/polyribonucleotide hybrid' -> hybrid
      - 'polypeptide(L)' or 'polypeptide(D)' -> protein
    """
    s = poly_type_str.lower()
    if "hybrid" in s:
        return "hybrid"
    elif "polyribonucleotide" in s and "deoxy" not in s:
        return "rna"
    elif "polydeoxyribonucleotide" in s:
        return "dna"
    elif "polypeptide" in s:
        return "protein"
    else:
        return "other"


def main():
    lines = []
    def log(msg=""):
        lines.append(msg)
        print(msg, flush=True)

    # Load manifests
    log("Loading RNA embedding manifest...")
    rna_seqs = load_manifest_sequences(RNA_CSV)
    log(f"  RNA manifest has {len(rna_seqs)} unique sequences")

    log("Loading DNA embedding manifest...")
    dna_seqs = load_manifest_sequences(DNA_CSV)
    log(f"  DNA manifest has {len(dna_seqs)} unique sequences")

    # Collect all bioassembly files
    pkl_files = sorted([f for f in os.listdir(BIOASSEMBLY_DIR) if f.endswith(".pkl.gz")])
    log(f"\nFound {len(pkl_files)} bioassembly files to check\n")

    # Track missing
    missing_rna = []  # (pdb_id, entity_id, seq)
    missing_dna = []
    missing_hybrid_rna = []
    missing_hybrid_dna = []
    load_errors = []

    # Track stats
    total_rna = 0
    total_dna = 0
    total_hybrid = 0
    total_protein = 0
    total_other = 0
    poly_type_values = defaultdict(int)

    for i, fname in enumerate(pkl_files):
        pdb_id = fname.replace(".pkl.gz", "")
        fpath = os.path.join(BIOASSEMBLY_DIR, fname)

        try:
            with gzip.open(fpath, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            load_errors.append((fname, str(e)))
            continue

        sequences = data.get("sequences", {})
        entity_poly_type = data.get("entity_poly_type", {})

        for entity_id, poly_type in entity_poly_type.items():
            poly_type_values[poly_type] += 1
            classification = classify_entity(poly_type)
            seq = sequences.get(entity_id, None)

            if seq is None:
                continue

            if classification == "rna":
                total_rna += 1
                if seq not in rna_seqs:
                    missing_rna.append((pdb_id, entity_id, seq))
            elif classification == "dna":
                total_dna += 1
                if seq not in dna_seqs:
                    missing_dna.append((pdb_id, entity_id, seq))
            elif classification == "hybrid":
                total_hybrid += 1
                # Hybrids could be looked up in either manifest
                in_rna = seq in rna_seqs
                in_dna = seq in dna_seqs
                if not in_rna:
                    missing_hybrid_rna.append((pdb_id, entity_id, seq))
                if not in_dna:
                    missing_hybrid_dna.append((pdb_id, entity_id, seq))
            elif classification == "protein":
                total_protein += 1
            else:
                total_other += 1

        if (i + 1) % 1000 == 0:
            log(f"  Processed {i+1}/{len(pkl_files)} files...")

    log(f"  Processed all {len(pkl_files)} files.")
    if load_errors:
        log(f"  ({len(load_errors)} files had load errors)")
        for fname, err in load_errors:
            log(f"    ERROR: {fname}: {err}")

    # Report
    log("=" * 80)
    log("SUMMARY")
    log("=" * 80)
    log(f"\nEntity poly_type distribution:")
    for pt, count in sorted(poly_type_values.items(), key=lambda x: -x[1]):
        log(f"  {pt}: {count}")

    log(f"\nTotal RNA entities: {total_rna}")
    log(f"Total DNA entities: {total_dna}")
    log(f"Total hybrid entities: {total_hybrid}")
    log(f"Total protein entities: {total_protein}")
    log(f"Total other entities: {total_other}")

    log(f"\n{'=' * 80}")
    log(f"MISSING RNA EMBEDDINGS: {len(missing_rna)} entities")
    log(f"{'=' * 80}")
    if missing_rna:
        for pdb_id, entity_id, seq in missing_rna:
            log(f"  PDB={pdb_id}  entity={entity_id}  seq={seq[:80]}{'...' if len(seq)>80 else ''} (len={len(seq)})")
    else:
        log("  (none)")

    log(f"\n{'=' * 80}")
    log(f"MISSING DNA EMBEDDINGS: {len(missing_dna)} entities")
    log(f"{'=' * 80}")
    if missing_dna:
        for pdb_id, entity_id, seq in missing_dna:
            log(f"  PDB={pdb_id}  entity={entity_id}  seq={seq[:80]}{'...' if len(seq)>80 else ''} (len={len(seq)})")
    else:
        log("  (none)")

    log(f"\n{'=' * 80}")
    log(f"HYBRID ENTITIES MISSING FROM RNA MANIFEST: {len(missing_hybrid_rna)} entities")
    log(f"{'=' * 80}")
    if missing_hybrid_rna:
        for pdb_id, entity_id, seq in missing_hybrid_rna:
            log(f"  PDB={pdb_id}  entity={entity_id}  seq={seq[:80]}{'...' if len(seq)>80 else ''} (len={len(seq)})")
    else:
        log("  (none)")

    log(f"\n{'=' * 80}")
    log(f"HYBRID ENTITIES MISSING FROM DNA MANIFEST: {len(missing_hybrid_dna)} entities")
    log(f"{'=' * 80}")
    if missing_hybrid_dna:
        for pdb_id, entity_id, seq in missing_hybrid_dna:
            log(f"  PDB={pdb_id}  entity={entity_id}  seq={seq[:80]}{'...' if len(seq)>80 else ''} (len={len(seq)})")
    else:
        log("  (none)")

    # Summary counts
    log(f"\n{'=' * 80}")
    log("FINAL COUNTS")
    log(f"{'=' * 80}")
    log(f"Missing RNA embeddings: {len(missing_rna)} / {total_rna} RNA entities")
    log(f"Missing DNA embeddings: {len(missing_dna)} / {total_dna} DNA entities")
    log(f"Hybrid missing from RNA manifest: {len(missing_hybrid_rna)} / {total_hybrid}")
    log(f"Hybrid missing from DNA manifest: {len(missing_hybrid_dna)} / {total_hybrid}")

    # Write to file
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\nReport written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
