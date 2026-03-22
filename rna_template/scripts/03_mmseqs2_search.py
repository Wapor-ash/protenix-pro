#!/usr/bin/env python3
"""
Step 3 (MMseqs2): Search for RNA templates using MMseqs2 and build index.

Replaces the pairwise alignment search in 03_search_and_index.py with MMseqs2
for scalable RNA template searching. MMseqs2 provides orders-of-magnitude
speedup over pairwise alignment while maintaining high sensitivity.

For each training RNA sequence, searches the template database (catalog) for
similar sequences using MMseqs2, then builds:
  1. search_results.json -- maps query IDs to matched template structures
  2. rna_template_index.json -- maps query sequences to template .npz paths

This script supports two strategies:
  A) MMseqs2 search index (--strategy mmseqs2): Uses MMseqs2 sequence search
     to find the best matching templates from the database. Production mode.
  B) Pairwise fallback (--strategy pairwise): Legacy pairwise alignment mode.
     Kept for backward compatibility; use mmseqs2 for production.

---- CONFIGURABLE INTERFACE ----
To swap out the search algorithm, replace `mmseqs2_search()` with your own
implementation that returns the same format:
    {query_id: {"query_sequence": str, "templates": [{"pdb_id": str, "chain_id": str, "identity": float}]}}

Future options: nhmmer, cmscan, BLAST, structural alignment
---- END CONFIGURABLE ----

Usage:
    # MMseqs2 search (production)
    python 03_mmseqs2_search.py \\
        --catalog /path/to/rna_catalog.json \\
        --template_dir /path/to/rna_database/templates \\
        --training_sequences /path/to/rna_sequence_to_pdb_chains.json \\
        --output_index /path/to/rna_template_index.json \\
        --output_search /path/to/search_results.json \\
        --strategy mmseqs2 \\
        --min_identity 0.3 \\
        --max_templates 4 \\
        --sensitivity 7.5 \\
        --evalue 1e-3
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VALID_BASES = frozenset("AGCUN")

# MMseqs2 tabular output columns for --format-output
_MMSEQS2_OUTPUT_COLUMNS = "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"


def extract_base_pdb_id(entry_id: str) -> str:
    """Extract the 4-character base PDB ID from a catalog key or query ID.

    RNA3DB catalog keys may include chain info, e.g.:
      '4tna_A' -> '4tna'
      '1jgp_1' -> '1jgp'
      '4tna'   -> '4tna'

    Standard PDB IDs are always 4 characters.
    """
    parts = entry_id.split("_")
    base = parts[0].lower()
    # Standard PDB IDs are 4 characters; handle edge cases
    if len(base) == 4:
        return base
    # Fallback: return the first part as-is
    return base


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def normalize_rna_sequence(seq: str) -> str:
    """Normalize RNA sequence: uppercase, T->U, keep only AGCUN."""
    seq = seq.upper().replace("T", "U")
    return "".join(c if c in _VALID_BASES else "N" for c in seq)


def _check_mmseqs2_available() -> bool:
    """Check whether mmseqs is on PATH."""
    try:
        result = subprocess.run(
            ["mmseqs", "version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            logger.info("MMseqs2 version: %s", result.stdout.strip())
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


# ---------------------------------------------------------------------------
# MMseqs2 database building
# ---------------------------------------------------------------------------

def build_mmseqs2_db(
    sequences: Dict[str, str],
    fasta_path: str,
    db_path: str,
    dbtype: int = 2,
) -> None:
    """Write sequences to FASTA, then create an MMseqs2 database.

    Args:
        sequences: Mapping of sequence ID -> sequence string.
        fasta_path: Path where the intermediate FASTA file will be written.
        db_path: Path for the resulting MMseqs2 database (prefix).
        dbtype: MMseqs2 database type. 2 = nucleotide.
    """
    # Write FASTA
    with open(fasta_path, "w") as fh:
        for seq_id, seq in sequences.items():
            fh.write(f">{seq_id}\n{seq}\n")

    logger.info(
        "Creating MMseqs2 database from %d sequences: %s", len(sequences), db_path
    )

    cmd = [
        "mmseqs", "createdb",
        fasta_path,
        db_path,
        "--dbtype", str(dbtype),
    ]
    _run_subprocess(cmd, "createdb")


# ---------------------------------------------------------------------------
# MMseqs2 search
# ---------------------------------------------------------------------------

def run_mmseqs2_search(
    query_db: str,
    target_db: str,
    result_db: str,
    tmp_dir: str,
    sensitivity: float = 7.5,
    evalue: float = 1e-3,
    num_threads: int = 8,
    search_type: int = 3,
    max_seqs: int = 300,
    extra_args: Optional[List[str]] = None,
) -> None:
    """Run MMseqs2 search.

    Args:
        query_db: Path to query MMseqs2 database.
        target_db: Path to target MMseqs2 database.
        result_db: Path for the result database (prefix).
        tmp_dir: Temporary directory for MMseqs2 intermediate files.
        sensitivity: Search sensitivity (1-7.5, higher = more sensitive).
        evalue: E-value threshold.
        num_threads: Number of threads to use.
        search_type: MMseqs2 search type. 3 = nucleotide.
        max_seqs: Maximum results per query to keep during pre-filter.
        extra_args: Additional command-line arguments for mmseqs search.
    """
    cmd = [
        "mmseqs", "search",
        query_db,
        target_db,
        result_db,
        tmp_dir,
        "-s", str(sensitivity),
        "-e", str(evalue),
        "--threads", str(num_threads),
        "--search-type", str(search_type),
        "--max-seqs", str(max_seqs),
    ]
    if extra_args:
        cmd.extend(extra_args)

    _run_subprocess(cmd, "search")


def convert_mmseqs2_results(
    query_db: str,
    target_db: str,
    result_db: str,
    output_tsv: str,
) -> None:
    """Convert MMseqs2 result database to human-readable tab-separated file.

    Args:
        query_db: Path to query MMseqs2 database.
        target_db: Path to target MMseqs2 database.
        result_db: Path to result database (prefix).
        output_tsv: Path for the output TSV file.
    """
    cmd = [
        "mmseqs", "convertalis",
        query_db,
        target_db,
        result_db,
        output_tsv,
        "--format-output", _MMSEQS2_OUTPUT_COLUMNS,
    ]
    _run_subprocess(cmd, "convertalis")


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def parse_mmseqs2_results(
    result_file: str,
    min_identity: float,
    max_templates: int,
    exclude_self: bool,
    query_to_pdb: Dict[str, str],
    db_id_to_info: Dict[str, dict],
    query_id_to_seq: Dict[str, str],
) -> Dict[str, dict]:
    """Parse MMseqs2 tabular output into search_results format.

    The tabular file has columns defined by _MMSEQS2_OUTPUT_COLUMNS:
        query, target, pident, alnlen, mismatch, gapopen,
        qstart, qend, tstart, tend, evalue, bits

    Args:
        result_file: Path to the TSV file from convertalis.
        min_identity: Minimum percent identity threshold (0-1 scale).
        max_templates: Maximum number of templates per query.
        exclude_self: Whether to exclude the query's own PDB as a template.
        query_to_pdb: Mapping from query ID -> PDB ID (for self-exclusion).
        db_id_to_info: Mapping from target database ID -> {"pdb_id": str, "chain_id": str}.
        query_id_to_seq: Mapping from query ID -> normalized query sequence.

    Returns:
        {query_id: {"query_sequence": str,
                     "templates": [{"pdb_id": str, "chain_id": str,
                                    "identity": float, "evalue": float,
                                    "bitscore": float}]}}
    """
    # Accumulate all hits per query
    raw_hits: Dict[str, List[dict]] = {}

    if not os.path.exists(result_file):
        logger.warning("Result file does not exist: %s", result_file)
        return {}

    with open(result_file) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 12:
                logger.warning("Skipping malformed line %d (only %d fields)", line_no, len(fields))
                continue

            query_id = fields[0]
            target_id = fields[1]
            try:
                pident = float(fields[2])  # MMseqs2 reports percent identity 0-100 or 0-1 depending on version
                evalue = float(fields[10])
                bitscore = float(fields[11])
            except (ValueError, IndexError) as exc:
                logger.warning("Skipping line %d due to parsing error: %s", line_no, exc)
                continue

            # MMseqs2 reports pident as a fraction (0.0-1.0) in convertalis by default
            # Normalize: if value > 1.0 assume it's a percentage
            identity = pident if pident <= 1.0 else pident / 100.0

            if identity < min_identity:
                continue

            # Resolve target info
            target_info = db_id_to_info.get(target_id)
            if target_info is None:
                logger.debug("Unknown target ID: %s", target_id)
                continue

            target_pdb_id = target_info["pdb_id"]
            target_chain_id = target_info["chain_id"]
            target_base_pdb = target_info.get("base_pdb_id", extract_base_pdb_id(target_pdb_id))

            # Self-exclusion: skip if query and target come from the same base PDB
            if exclude_self:
                query_pdb = query_to_pdb.get(query_id, "")
                if query_pdb and target_base_pdb == query_pdb.lower():
                    continue

            hit = {
                "pdb_id": target_pdb_id,
                "chain_id": target_chain_id,
                "identity": round(identity, 4),
                "evalue": evalue,
                "bitscore": round(bitscore, 2),
            }
            raw_hits.setdefault(query_id, []).append(hit)

    # Deduplicate, sort, and truncate per query
    results: Dict[str, dict] = {}
    for query_id, hits in raw_hits.items():
        # Deduplicate by (pdb_id, chain_id) -- keep highest identity
        seen: Dict[Tuple[str, str], dict] = {}
        for h in hits:
            key = (h["pdb_id"], h["chain_id"])
            if key not in seen or h["identity"] > seen[key]["identity"]:
                seen[key] = h
        unique_hits = list(seen.values())

        # Sort by identity descending, then bitscore descending
        unique_hits.sort(key=lambda x: (-x["identity"], -x["bitscore"]))
        best = unique_hits[:max_templates]

        if best:
            query_seq = query_id_to_seq.get(query_id, "")
            results[query_id] = {
                "query_sequence": query_seq,
                "templates": best,
            }

    return results


# ---------------------------------------------------------------------------
# Full MMseqs2 search pipeline
# ---------------------------------------------------------------------------

def mmseqs2_search(
    training_sequences: Dict[str, str],
    database_catalog: Dict[str, List[dict]],
    min_identity: float = 0.3,
    max_templates: int = 4,
    exclude_self: bool = True,
    sensitivity: float = 7.5,
    evalue: float = 1e-3,
    num_threads: int = 8,
    work_dir: Optional[str] = None,
    extra_mmseqs_args: Optional[List[str]] = None,
) -> Dict[str, dict]:
    """Search for templates using MMseqs2 sequence search.

    This function orchestrates the full MMseqs2 pipeline:
      1. Build target database from catalog
      2. Build query database from training sequences
      3. Run mmseqs search
      4. Convert and parse results

    Args:
        training_sequences: {query_id: sequence} for query/training data.
        database_catalog: {pdb_id: [{chain_id, sequence, ...}]} from catalog.
        min_identity: Minimum sequence identity threshold (0-1).
        max_templates: Maximum templates per query.
        exclude_self: Exclude the query's own structure as a template.
        sensitivity: MMseqs2 sensitivity parameter (1-7.5).
        evalue: E-value threshold for MMseqs2 search.
        num_threads: Number of threads for MMseqs2.
        work_dir: Working directory for intermediate files. If None, a
                  temporary directory is created and cleaned up automatically.
        extra_mmseqs_args: Additional arguments passed to mmseqs search.

    Returns:
        {query_id: {"query_sequence": str,
                     "templates": [{"pdb_id": str, "chain_id": str,
                                    "identity": float, "evalue": float,
                                    "bitscore": float}]}}
    """
    if not _check_mmseqs2_available():
        logger.error(
            "MMseqs2 is not available. Install it or activate the conda env "
            "(e.g., conda activate protenix). Falling back to empty results."
        )
        return {}

    # Flatten database into per-chain sequences and build lookup maps
    db_sequences: Dict[str, str] = {}       # db_id -> sequence
    db_id_to_info: Dict[str, dict] = {}     # db_id -> {"pdb_id", "chain_id", "base_pdb_id"}
    for pdb_id, chains in database_catalog.items():
        base_pdb = extract_base_pdb_id(pdb_id)
        for chain in chains:
            chain_id = chain["chain_id"]
            db_id = f"{pdb_id}_{chain_id}"
            seq = normalize_rna_sequence(chain["sequence"])
            if len(seq) < 5:
                continue
            db_sequences[db_id] = seq
            db_id_to_info[db_id] = {
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "base_pdb_id": base_pdb,
            }

    # Normalize query sequences
    query_sequences: Dict[str, str] = {}
    query_to_pdb: Dict[str, str] = {}
    for qid, qseq in training_sequences.items():
        norm_seq = normalize_rna_sequence(qseq)
        if len(norm_seq) < 5:
            continue
        query_sequences[qid] = norm_seq
        # Extract base 4-char PDB ID for self-exclusion
        query_to_pdb[qid] = extract_base_pdb_id(qid)

    logger.info(
        "MMseqs2 search: %d queries x %d database entries",
        len(query_sequences), len(db_sequences),
    )

    if not query_sequences:
        logger.warning("No valid query sequences after normalization.")
        return {}
    if not db_sequences:
        logger.warning("No valid database sequences after normalization.")
        return {}

    # Set up working directory
    cleanup_work_dir = False
    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="mmseqs2_rna_")
        cleanup_work_dir = True
    else:
        os.makedirs(work_dir, exist_ok=True)

    try:
        return _run_mmseqs2_pipeline(
            query_sequences=query_sequences,
            db_sequences=db_sequences,
            db_id_to_info=db_id_to_info,
            query_to_pdb=query_to_pdb,
            work_dir=work_dir,
            min_identity=min_identity,
            max_templates=max_templates,
            exclude_self=exclude_self,
            sensitivity=sensitivity,
            evalue=evalue,
            num_threads=num_threads,
            extra_mmseqs_args=extra_mmseqs_args,
        )
    finally:
        if cleanup_work_dir:
            logger.info("Cleaning up temporary directory: %s", work_dir)
            shutil.rmtree(work_dir, ignore_errors=True)


def _run_mmseqs2_pipeline(
    query_sequences: Dict[str, str],
    db_sequences: Dict[str, str],
    db_id_to_info: Dict[str, dict],
    query_to_pdb: Dict[str, str],
    work_dir: str,
    min_identity: float,
    max_templates: int,
    exclude_self: bool,
    sensitivity: float,
    evalue: float,
    num_threads: int,
    extra_mmseqs_args: Optional[List[str]],
) -> Dict[str, dict]:
    """Internal: run the full MMseqs2 search pipeline inside *work_dir*."""
    # Paths inside work_dir
    query_fasta = os.path.join(work_dir, "query.fasta")
    target_fasta = os.path.join(work_dir, "target.fasta")
    query_db = os.path.join(work_dir, "query_db")
    target_db = os.path.join(work_dir, "target_db")
    result_db = os.path.join(work_dir, "result_db")
    tmp_dir = os.path.join(work_dir, "tmp")
    result_tsv = os.path.join(work_dir, "results.tsv")

    os.makedirs(tmp_dir, exist_ok=True)

    # 1. Build databases
    logger.info("Building target database (%d sequences)...", len(db_sequences))
    build_mmseqs2_db(db_sequences, target_fasta, target_db)

    logger.info("Building query database (%d sequences)...", len(query_sequences))
    build_mmseqs2_db(query_sequences, query_fasta, query_db)

    # 2. Run search
    logger.info(
        "Running MMseqs2 search (sensitivity=%.1f, evalue=%g, threads=%d)...",
        sensitivity, evalue, num_threads,
    )
    run_mmseqs2_search(
        query_db=query_db,
        target_db=target_db,
        result_db=result_db,
        tmp_dir=tmp_dir,
        sensitivity=sensitivity,
        evalue=evalue,
        num_threads=num_threads,
        extra_args=extra_mmseqs_args,
    )

    # 3. Convert to tabular
    logger.info("Converting results to tabular format...")
    convert_mmseqs2_results(query_db, target_db, result_db, result_tsv)

    # 4. Parse
    logger.info("Parsing MMseqs2 results...")
    results = parse_mmseqs2_results(
        result_file=result_tsv,
        min_identity=min_identity,
        max_templates=max_templates,
        exclude_self=exclude_self,
        query_to_pdb=query_to_pdb,
        db_id_to_info=db_id_to_info,
        query_id_to_seq=query_sequences,
    )

    logger.info(
        "Search complete: %d/%d queries have templates",
        len(results), len(query_sequences),
    )
    return results


# ---------------------------------------------------------------------------
# Legacy pairwise search (kept for backward compatibility)
# ---------------------------------------------------------------------------

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
    """Search for templates using pairwise sequence alignment (legacy).

    This is the original O(N*M) search from 03_search_and_index.py.
    Kept for backward compatibility and small-scale testing.

    Args:
        training_sequences: {pdb_id_chain: sequence} for training data.
        database_catalog: {pdb_id: [{chain_id, sequence, ...}]} from catalog.
        min_identity: Minimum sequence identity threshold.
        max_templates: Maximum templates per query.
        exclude_self: Exclude the query's own structure as a template.

    Returns:
        {query_id: {"query_sequence": str,
                     "templates": [{"pdb_id": str, "chain_id": str, "identity": float}]}}
    """
    # Flatten database into (pdb_id, chain_id, sequence, base_pdb_id) tuples
    db_entries = []
    for pdb_id, chains in database_catalog.items():
        base_pdb = extract_base_pdb_id(pdb_id)
        for chain in chains:
            db_entries.append(
                (pdb_id, chain["chain_id"], normalize_rna_sequence(chain["sequence"]), base_pdb)
            )

    logger.info(
        "Pairwise search: %d queries x %d database entries",
        len(training_sequences), len(db_entries),
    )

    results = {}
    for i, (query_id, query_seq) in enumerate(sorted(training_sequences.items())):
        if (i + 1) % 50 == 0:
            logger.info("  Search progress: %d/%d", i + 1, len(training_sequences))

        query_seq_norm = normalize_rna_sequence(query_seq)
        if len(query_seq_norm) < 5:
            continue

        query_pdb = extract_base_pdb_id(query_id)

        scored = []
        for db_pdb_id, db_chain_id, db_seq, db_base_pdb in db_entries:
            if exclude_self and db_base_pdb == query_pdb:
                continue

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

        scored.sort(key=lambda x: -x["identity"])
        best = scored[:max_templates]

        if best:
            results[query_id] = {
                "query_sequence": query_seq_norm,
                "templates": best,
            }

    logger.info(
        "Search complete: %d/%d queries have templates",
        len(results), len(training_sequences),
    )
    return results


def build_cross_index(
    search_results: Dict[str, dict],
    template_dir: str,
) -> Dict[str, List[str]]:
    """Build index for cross-template mode.

    Maps each query sequence to its cross-template .npz file.
    The relative path prefix is derived from template_dir's basename
    (e.g. "templates" or "cross_templates") so the index matches the
    actual directory layout under template_database_dir.
    """
    index: Dict[str, List[str]] = {}
    found = 0
    missing = 0

    # Derive relative prefix from actual directory name instead of hardcoding
    dir_basename = os.path.basename(os.path.normpath(template_dir))

    for query_id, info in sorted(search_results.items()):
        sequence = info["query_sequence"]
        npz_name = f"{query_id}_template.npz"
        npz_path = os.path.join(template_dir, npz_name)

        if os.path.exists(npz_path):
            rel_path = os.path.join(dir_basename, npz_name)
            index.setdefault(sequence, []).append(rel_path)
            found += 1
        else:
            missing += 1

    logger.info("Cross-index: %d templates found, %d missing", found, missing)
    logger.info("  Unique sequences in index: %d", len(index))
    return index


# ---------------------------------------------------------------------------
# Training sequence loaders
# ---------------------------------------------------------------------------

def filter_catalog_by_release_date(
    catalog: Dict[str, List[dict]],
    metadata_path: str,
    cutoff_date: str,
) -> Dict[str, List[dict]]:
    """Filter catalog entries by release date using RNA3DB metadata.

    Removes entries whose release_date is after *cutoff_date*.

    Args:
        catalog: The RNA catalog {entry_id: [chain_info, ...]}.
        metadata_path: Path to RNA3DB metadata JSON (filter.json / parse.json)
                       with format {entry_id: {release_date: "YYYY-MM-DD", ...}}.
        cutoff_date: Cutoff date string in YYYY-MM-DD format.

    Returns:
        Filtered catalog with only entries released on or before cutoff_date.
    """
    from datetime import datetime

    cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d")

    with open(metadata_path) as fh:
        metadata = json.load(fh)

    # Build lookup: base_pdb_chain -> release_date
    # metadata keys are like "7zpi_B", matching catalog keys
    date_lookup: Dict[str, str] = {}
    for entry_id, info in metadata.items():
        if isinstance(info, dict) and "release_date" in info:
            date_lookup[entry_id.lower()] = info["release_date"]

    filtered = {}
    removed = 0
    kept = 0
    no_date = 0

    for entry_id, chains in catalog.items():
        release_str = date_lookup.get(entry_id.lower(), "")
        if not release_str:
            # No metadata found — keep the entry but log
            filtered[entry_id] = chains
            no_date += 1
            continue

        try:
            release = datetime.strptime(release_str, "%Y-%m-%d")
        except ValueError:
            filtered[entry_id] = chains
            no_date += 1
            continue

        if release <= cutoff:
            filtered[entry_id] = chains
            kept += 1
        else:
            removed += 1

    logger.info(
        "Release date filter (cutoff=%s): kept=%d, removed=%d, no_date=%d",
        cutoff_date, kept, removed, no_date,
    )
    return filtered


def load_training_sequences_from_json(json_path: str) -> Dict[str, str]:
    """Load training RNA sequences from the sequence-to-PDB mapping JSON.

    The expected format is: {sequence: [pdb_id, ...]}
    Returns {pdb_id: sequence} mapping.
    """
    with open(json_path) as fh:
        data = json.load(fh)

    sequences: Dict[str, str] = {}
    for seq, pdb_list in data.items():
        norm_seq = normalize_rna_sequence(seq)
        if len(norm_seq) < 5:
            continue
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
    sequences: Dict[str, str] = {}

    allowed_ids = None
    if pdb_list_path and os.path.exists(pdb_list_path):
        with open(pdb_list_path) as fh:
            allowed_ids = {line.strip().lower() for line in fh if line.strip()}

    for pdb_id, chains in catalog.items():
        if allowed_ids and pdb_id.lower() not in allowed_ids:
            continue
        for chain in chains:
            key = f"{pdb_id}_{chain['chain_id']}"
            sequences[key] = normalize_rna_sequence(chain["sequence"])

    return sequences


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------

def _run_subprocess(cmd: List[str], step_name: str) -> subprocess.CompletedProcess:
    """Run a subprocess command, logging and raising on failure."""
    logger.info("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout.strip():
            logger.debug("[%s stdout] %s", step_name, result.stdout.strip()[:500])
        return result
    except subprocess.CalledProcessError as exc:
        logger.error(
            "[%s] Command failed (exit %d):\n  cmd: %s\n  stderr: %s",
            step_name,
            exc.returncode,
            " ".join(cmd),
            exc.stderr.strip()[:1000] if exc.stderr else "(empty)",
        )
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search for RNA templates using MMseqs2 and build index.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input/output paths ---
    parser.add_argument(
        "--catalog",
        default="/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/rna_catalog.json",
        help="RNA catalog JSON from Step 1.",
    )
    parser.add_argument(
        "--template_dir",
        default="/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/templates",
        help="Directory containing .npz template files.",
    )
    parser.add_argument(
        "--output_index",
        default="/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/rna_template_index.json",
        help="Output template index JSON.",
    )
    parser.add_argument(
        "--output_search",
        default="",
        help="Output search results JSON (for mmseqs2/pairwise modes).",
    )

    # --- Strategy ---
    parser.add_argument(
        "--strategy",
        choices=["mmseqs2", "pairwise"],
        default="mmseqs2",
        help="Search strategy: mmseqs2 (production) or pairwise (legacy).",
    )

    # --- Training sequence sources ---
    parser.add_argument(
        "--training_sequences",
        default="",
        help="Path to rna_sequence_to_pdb_chains.json (for mmseqs2/pairwise mode).",
    )
    parser.add_argument(
        "--training_pdb_list",
        default="",
        help="Path to training PDB list file (one ID per line).",
    )

    # --- Search parameters ---
    parser.add_argument(
        "--min_identity",
        type=float,
        default=0.3,
        help="Minimum sequence identity threshold (0-1 scale). Default: 0.3.",
    )
    parser.add_argument(
        "--max_templates",
        type=int,
        default=4,
        help="Maximum number of templates per query. Default: 4.",
    )
    # --- MMseqs2-specific parameters ---
    parser.add_argument(
        "--sensitivity",
        type=float,
        default=7.5,
        help="MMseqs2 sensitivity (1-7.5, higher = more sensitive). Default: 7.5.",
    )
    parser.add_argument(
        "--evalue",
        type=float,
        default=1e-3,
        help="E-value threshold for MMseqs2 search. Default: 1e-3.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=8,
        help="Number of threads for MMseqs2. Default: 8.",
    )
    parser.add_argument(
        "--mmseqs_work_dir",
        default="",
        help="Working directory for MMseqs2 intermediate files. "
             "If empty, a temporary directory is created and cleaned up.",
    )

    # --- Data leakage prevention ---
    parser.add_argument(
        "--release_date_cutoff",
        default="",
        help="Release date cutoff (YYYY-MM-DD) for data leakage prevention. "
             "Templates released after this date are excluded from the database.",
    )
    parser.add_argument(
        "--rna3db_metadata",
        default="",
        help="Path to RNA3DB metadata JSON (e.g., filter.json or parse.json) "
             "that maps entry IDs to {release_date, ...}. Required when "
             "--release_date_cutoff is used.",
    )

    # --- Pipeline control ---
    parser.add_argument(
        "--skip_search",
        action="store_true",
        help="Skip search step and only rebuild index from existing search_results.json.",
    )
    parser.add_argument(
        "--pdb_list",
        default="",
        help="Optional PDB list file for filtering (one ID per line).",
    )

    args = parser.parse_args()

    # ---- Load catalog ----
    with open(args.catalog) as fh:
        catalog = json.load(fh)
    logger.info("Loaded catalog: %d structures", len(catalog))

    # ---- Release date cutoff filtering ----
    if args.release_date_cutoff:
        if not args.rna3db_metadata or not os.path.exists(args.rna3db_metadata):
            logger.error(
                "--release_date_cutoff requires --rna3db_metadata pointing to "
                "a valid RNA3DB metadata JSON (e.g., filter.json)."
            )
            sys.exit(1)
        catalog = filter_catalog_by_release_date(
            catalog, args.rna3db_metadata, args.release_date_cutoff
        )
        logger.info("Catalog after date filter: %d structures", len(catalog))

    # ---- MMseqs2 / Pairwise strategies ----
    if args.skip_search and args.output_search and os.path.exists(args.output_search):
        logger.info("Skipping search, loading existing results: %s", args.output_search)
        with open(args.output_search) as fh:
            search_results = json.load(fh)
    else:
        # Load training sequences
        if args.training_sequences and os.path.exists(args.training_sequences):
            training_seqs = load_training_sequences_from_json(args.training_sequences)
        else:
            training_seqs = load_training_sequences_from_catalog(
                catalog, pdb_list_path=args.training_pdb_list
            )

        # Filter training sequences by --pdb_list if provided
        if args.pdb_list and os.path.exists(args.pdb_list):
            with open(args.pdb_list) as fh:
                allowed_pdbs = {line.strip().lower() for line in fh if line.strip()}
            before = len(training_seqs)
            training_seqs = {
                k: v for k, v in training_seqs.items()
                if extract_base_pdb_id(k) in allowed_pdbs
            }
            logger.info(
                "Filtered training sequences by --pdb_list: %d -> %d",
                before, len(training_seqs),
            )

        logger.info("Training sequences: %d", len(training_seqs))

        if not training_seqs:
            logger.error("No training sequences found. Exiting.")
            sys.exit(1)

        # Self-hits are always excluded to avoid accidental leakage.
        exclude_self = True

        # Dispatch to selected search strategy
        if args.strategy == "mmseqs2":
            work_dir = args.mmseqs_work_dir if args.mmseqs_work_dir else None
            search_results = mmseqs2_search(
                training_sequences=training_seqs,
                database_catalog=catalog,
                min_identity=args.min_identity,
                max_templates=args.max_templates,
                exclude_self=exclude_self,
                sensitivity=args.sensitivity,
                evalue=args.evalue,
                num_threads=args.num_threads,
                work_dir=work_dir,
            )
        else:
            # Legacy pairwise
            search_results = pairwise_search(
                training_sequences=training_seqs,
                database_catalog=catalog,
                min_identity=args.min_identity,
                max_templates=args.max_templates,
                exclude_self=exclude_self,
            )

        # Save search results
        if args.output_search:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_search)), exist_ok=True)
            with open(args.output_search, "w") as fh:
                json.dump(search_results, fh, indent=2)
            logger.info("Search results saved: %s", args.output_search)

    # Build index from search results
    index = build_cross_index(search_results, args.template_dir)

    # ---- Save index ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output_index)), exist_ok=True)
    with open(args.output_index, "w") as fh:
        json.dump(index, fh, indent=2)

    total_paths = sum(len(v) for v in index.values())
    logger.info("Index saved: %s", args.output_index)
    logger.info("  Total sequences: %d", len(index))
    logger.info("  Total template paths: %d", total_paths)


if __name__ == "__main__":
    main()
