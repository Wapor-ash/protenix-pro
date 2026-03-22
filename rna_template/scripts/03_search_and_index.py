#!/usr/bin/env python3
"""
Step 3: Search for RNA templates using pairwise alignment and build index.

For each training RNA sequence, searches the template database (catalog) for
similar sequences using pairwise alignment, then builds:
  1. search_results.json — maps query IDs to matched template structures
  2. rna_template_index.json — maps query sequences to template .npz paths

This script supports one strategy:
  A) Pairwise search index (--strategy pairwise): Uses pairwise sequence
     alignment to find the best matching templates from the database.

---- CONFIGURABLE INTERFACE ----
To swap out the search algorithm, replace `pairwise_search()` with your own
implementation that returns the same format:
    {query_id: {"query_sequence": str, "templates": [{"pdb_id": str, "chain_id": str, "identity": float}]}}

Future options: nhmmer, cmscan, BLAST, structural alignment
---- END CONFIGURABLE ----

Usage:
    # Pairwise search (legacy)
    python 03_search_and_index.py \
        --catalog /path/to/rna_catalog.json \
        --template_dir /path/to/rna_database/templates \
        --training_sequences /path/to/rna_sequence_to_pdb_chains.json \
        --output_index /path/to/rna_template_index.json \
        --output_search /path/to/search_results.json \
        --strategy pairwise \
        --min_identity 0.3 \
        --max_templates 4
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def normalize_rna_sequence(seq: str) -> str:
    """Normalize RNA sequence: uppercase, T→U, keep only AGCUN."""
    seq = seq.upper().replace("T", "U")
    return "".join(c if c in "AGCUN" else "N" for c in seq)


# ==================== CONFIGURABLE: Search Algorithm ====================
# Replace this function with nhmmer/cmscan/BLAST for production use.

def pairwise_identity(seq1: str, seq2: str) -> float:
    """Compute pairwise sequence identity between two RNA sequences.

    Uses BioPython PairwiseAligner for global alignment.
    Returns fraction of identical positions over aligned length.
    """
    if not seq1 or not seq2:
        return 0.0

    try:
        from Bio.Align import PairwiseAligner
        aligner = PairwiseAligner()
        aligner.mode = "global"
        aligner.match_score = 2.0
        aligner.mismatch_score = -1.0
        aligner.open_gap_score = -5.0
        aligner.extend_gap_score = -0.5
        alignments = aligner.align(seq1, seq2)
        if not alignments:
            return 0.0
        alignment = alignments[0]
        # Count matches in aligned blocks
        q_blocks, t_blocks = alignment.aligned
        total_matches = 0
        total_aligned = 0
        for (q_start, q_end), (t_start, t_end) in zip(q_blocks, t_blocks):
            block_len = q_end - q_start
            total_aligned += block_len
            for qi, ti in zip(range(q_start, q_end), range(t_start, t_end)):
                if seq1[qi] == seq2[ti]:
                    total_matches += 1
        return total_matches / max(total_aligned, 1)
    except ImportError:
        # Simple fallback without BioPython
        min_len = min(len(seq1), len(seq2))
        max_len = max(len(seq1), len(seq2))
        matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
        return matches / max(max_len, 1)


def pairwise_search(
    training_sequences: Dict[str, str],
    database_catalog: Dict[str, List[dict]],
    min_identity: float = 0.3,
    max_templates: int = 4,
    exclude_self: bool = True,
) -> Dict[str, dict]:
    """Search for templates using pairwise sequence alignment.

    ---- CONFIGURABLE: Replace this function for different search algorithms ----

    Args:
        training_sequences: {pdb_id_chain: sequence} for training data.
        database_catalog: {pdb_id: [{chain_id, sequence, ...}]} from catalog.
        min_identity: Minimum sequence identity threshold.
        max_templates: Maximum templates per query.
        exclude_self: Exclude the query's own structure as a template.

    Returns:
        {query_id: {"query_sequence": str, "templates": [{"pdb_id": str, "chain_id": str, "identity": float}]}}
    """
    # Flatten database into (pdb_id, chain_id, sequence) tuples
    db_entries = []
    for pdb_id, chains in database_catalog.items():
        for chain in chains:
            db_entries.append((pdb_id, chain["chain_id"], normalize_rna_sequence(chain["sequence"])))

    print(f"Pairwise search: {len(training_sequences)} queries × {len(db_entries)} database entries")

    results = {}
    for i, (query_id, query_seq) in enumerate(sorted(training_sequences.items())):
        if (i + 1) % 50 == 0:
            print(f"  Search progress: {i+1}/{len(training_sequences)}")

        query_seq_norm = normalize_rna_sequence(query_seq)
        if len(query_seq_norm) < 5:
            continue

        # Extract query pdb_id for self-exclusion
        query_pdb = query_id.split("_")[0] if "_" in query_id else query_id

        scored = []
        for db_pdb_id, db_chain_id, db_seq in db_entries:
            # Self-exclusion
            if exclude_self and db_pdb_id == query_pdb:
                continue

            # Quick length filter
            len_ratio = len(db_seq) / max(len(query_seq_norm), 1)
            if len_ratio < 0.3 or len_ratio > 3.0:
                continue

            identity = pairwise_identity(query_seq_norm, db_seq)
            if identity >= min_identity:
                scored.append({
                    "pdb_id": db_pdb_id,
                    "chain_id": db_chain_id,
                    "identity": round(identity, 4),
                })

        # Sort by identity descending, take top N
        scored.sort(key=lambda x: -x["identity"])
        best = scored[:max_templates]

        if best:
            results[query_id] = {
                "query_sequence": query_seq_norm,
                "templates": best,
            }

    print(f"Search complete: {len(results)}/{len(training_sequences)} queries have templates")
    return results

# ==================== END CONFIGURABLE ====================
def build_cross_index(
    search_results: Dict[str, dict],
    template_dir: str,
) -> Dict[str, List[str]]:
    """Build index for cross-template mode.

    Maps each query sequence to its cross-template .npz file.
    """
    index = {}
    found = 0
    missing = 0

    for query_id, info in sorted(search_results.items()):
        sequence = info["query_sequence"]
        npz_name = f"{query_id}_template.npz"
        npz_path = os.path.join(template_dir, npz_name)

        if os.path.exists(npz_path):
            rel_path = os.path.join("templates", npz_name)
            index.setdefault(sequence, []).append(rel_path)
            found += 1
        else:
            missing += 1

    print(f"Cross-index: {found} templates found, {missing} missing")
    print(f"  Unique sequences in index: {len(index)}")
    return index


def load_training_sequences_from_json(json_path: str) -> Dict[str, str]:
    """Load training RNA sequences from the sequence-to-PDB mapping JSON.

    Returns {pdb_id_chain: sequence} mapping.
    """
    with open(json_path) as f:
        data = json.load(f)

    sequences = {}
    for seq, pdb_list in data.items():
        norm_seq = normalize_rna_sequence(seq)
        if len(norm_seq) < 5:
            continue
        # Create entries for each PDB this sequence appears in
        for pdb_id in pdb_list:
            key = f"{pdb_id.lower()}"
            sequences[key] = norm_seq

    return sequences


def load_training_sequences_from_catalog(
    catalog: Dict[str, List[dict]],
    pdb_list_path: str = "",
) -> Dict[str, str]:
    """Load training sequences from catalog, optionally filtered by PDB list.

    Returns {pdb_id_chain: sequence} mapping.
    """
    sequences = {}

    # Optional PDB list filter
    allowed_ids = None
    if pdb_list_path and os.path.exists(pdb_list_path):
        with open(pdb_list_path) as f:
            allowed_ids = {line.strip().lower() for line in f if line.strip()}

    for pdb_id, chains in catalog.items():
        if allowed_ids and pdb_id.lower() not in allowed_ids:
            continue
        for chain in chains:
            key = f"{pdb_id}_{chain['chain_id']}"
            sequences[key] = normalize_rna_sequence(chain["sequence"])

    return sequences


def main():
    parser = argparse.ArgumentParser(description="Search for RNA templates and build index.")
    parser.add_argument("--catalog",
                        default="/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/rna_catalog.json")
    parser.add_argument("--template_dir",
                        default="/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/templates")
    parser.add_argument("--output_index",
                        default="/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/rna_template_index.json")
    parser.add_argument("--output_search", default="",
                        help="Output search results JSON (for cross mode).")
    parser.add_argument("--strategy", choices=["pairwise"], default="pairwise",
                        help="Search strategy. Only cross-template pairwise mode is supported.")
    # ---- CONFIGURABLE: training sequence sources ----
    parser.add_argument("--training_sequences", default="",
                        help="Path to rna_sequence_to_pdb_chains.json (for pairwise mode).")
    parser.add_argument("--training_pdb_list", default="",
                        help="Path to training PDB list file (one ID per line).")
    parser.add_argument("--min_identity", type=float, default=0.3,
                        help="Min sequence identity for pairwise search.")
    parser.add_argument("--max_templates", type=int, default=4)
    args = parser.parse_args()

    # Load catalog
    with open(args.catalog) as f:
        catalog = json.load(f)
    print(f"Loaded catalog: {len(catalog)} structures")

    # Load training sequences
    if args.training_sequences and os.path.exists(args.training_sequences):
        training_seqs = load_training_sequences_from_json(args.training_sequences)
    else:
        training_seqs = load_training_sequences_from_catalog(
            catalog, pdb_list_path=args.training_pdb_list
        )
    print(f"Training sequences: {len(training_seqs)}")

    # Self-hits are always excluded to avoid accidental leakage.
    search_results = pairwise_search(
        training_sequences=training_seqs,
        database_catalog=catalog,
        min_identity=args.min_identity,
        max_templates=args.max_templates,
        exclude_self=True,
    )

    # Save search results
    if args.output_search:
        os.makedirs(os.path.dirname(args.output_search), exist_ok=True)
        with open(args.output_search, "w") as f:
            json.dump(search_results, f, indent=2)
        print(f"Search results saved: {args.output_search}")

    # Build index from search results
    index = build_cross_index(search_results, args.template_dir)

    # Save index
    os.makedirs(os.path.dirname(args.output_index), exist_ok=True)
    with open(args.output_index, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nIndex saved: {args.output_index}")
    print(f"  Total sequences: {len(index)}")
    total_paths = sum(len(v) for v in index.values())
    print(f"  Total template paths: {total_paths}")


if __name__ == "__main__":
    main()
