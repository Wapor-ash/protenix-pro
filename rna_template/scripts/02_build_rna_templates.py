#!/usr/bin/env python3
"""
Step 2: Batch-build RNA template .npz files from CIF structures.

For each query sequence (from the training set), uses the matched template
structures to build Protenix-compatible RNA template .npz tensors.

This script builds cross-template .npz tensors from search results.

Optionally, the Arena binary can be invoked to fill missing atoms before
template construction (--use_arena).  Arena operates on PDB files, so CIF
inputs are first converted to PDB via BioPython's PDBIO.

Input:
  - rna_catalog.json from Step 1
  - CIF files in pdb_rna_dir

Output:
  - cross-template .npz files in output_dir, one per query sequence

Usage:
    python 02_build_rna_templates.py \
        --catalog /path/to/rna_catalog.json \
        --pdb_rna_dir /path/to/PDB_RNA \
        --output_dir /path/to/rna_database/templates \
        --mode cross \
        --search_results /path/to/search_results.json

    # With Arena atom-filling
    python 02_build_rna_templates.py \
        --catalog /path/to/rna_catalog.json \
        --pdb_rna_dir /path/to/PDB_RNA \
        --output_dir /path/to/rna_database/templates \
        --mode cross \
        --use_arena \
        --arena_binary /path/to/Arena \
        --arena_option 5
"""

import argparse
import glob as glob_mod
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add compute dir to path for rna_template_common
SCRIPT_DIR = Path(__file__).resolve().parent
COMPUTE_DIR = SCRIPT_DIR.parent / "compute"
sys.path.insert(0, str(COMPUTE_DIR))

from rna_template_common import (
    build_minimal_template_arrays,
    load_structure_residues,
    normalize_query_sequence,
    save_npz_with_metadata,
    stack_template_dicts,
)


# ---------------------------------------------------------------------------
# CIF path lookup — supports flat and nested (rna3db-mmcifs) layouts
# ---------------------------------------------------------------------------

def find_cif_path(
    pdb_rna_dir: str,
    pdb_id: str,
    catalog_cif_path: Optional[str] = None,
) -> Optional[str]:
    """Locate a CIF file for *pdb_id* inside *pdb_rna_dir*.

    If *catalog_cif_path* is provided and the file exists, uses it directly
    (avoiding expensive recursive glob).  Otherwise falls back to:
      1. Flat layout: ``{pdb_rna_dir}/{pdb_id}.cif``
      2. Recursive search for nested dirs (rna3db-mmcifs).

    Returns the path as a string, or ``None`` if not found.
    """
    # 0. Use catalog-recorded path if available and valid
    if catalog_cif_path and os.path.exists(catalog_cif_path):
        return catalog_cif_path

    # 1. Flat layout
    flat = os.path.join(pdb_rna_dir, f"{pdb_id}.cif")
    if os.path.exists(flat):
        return flat

    # 2. Recursive search (handles nested dirs like rna3db-mmcifs/<subdir>/<pdb_id>.cif)
    pattern = os.path.join(pdb_rna_dir, "**", f"{pdb_id}.cif")
    matches = glob_mod.glob(pattern, recursive=True)
    if matches:
        return matches[0]

    return None


# ---------------------------------------------------------------------------
# Arena atom-filling
# ---------------------------------------------------------------------------

def run_arena_refine(
    cif_path: str,
    output_pdb_path: str,
    arena_binary: str,
    arena_option: int = 5,
) -> Optional[str]:
    """Convert CIF to PDB and run Arena to fill missing atoms.

    Args:
        cif_path: Path to the input .cif (or .pdb) structure file.
        output_pdb_path: Where the Arena-refined PDB should be written.
        arena_binary: Path to the Arena executable.
        arena_option: Arena mode (default 5 = fill all missing atoms).

    Returns:
        Path to the Arena-refined PDB on success, or ``None`` on failure.
    """
    try:
        from Bio.PDB import MMCIFParser, PDBParser, PDBIO

        suffix = Path(cif_path).suffix.lower()
        if suffix in {".cif", ".mmcif"}:
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)

        structure = parser.get_structure("arena_input", cif_path)

        # Write intermediate PDB that Arena can consume
        intermediate_pdb = output_pdb_path.replace(".pdb", "_input.pdb")
        io = PDBIO()
        io.set_structure(structure)
        io.save(intermediate_pdb)

        # Run Arena: Arena input.pdb output.pdb [option]
        cmd = [arena_binary, intermediate_pdb, output_pdb_path, str(arena_option)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            print(f"  Warning: Arena failed (rc={result.returncode}) for {cif_path}")
            if result.stderr:
                print(f"    stderr: {result.stderr[:500]}")
            return None

        if not os.path.exists(output_pdb_path):
            print(f"  Warning: Arena produced no output for {cif_path}")
            return None

        return output_pdb_path

    except subprocess.TimeoutExpired:
        print(f"  Warning: Arena timed out for {cif_path}")
        return None
    except Exception as e:
        print(f"  Warning: Arena refinement failed for {cif_path}: {e}")
        return None

def build_cross_template(
    query_seq: str,
    query_id: str,
    template_specs: List[dict],
    pdb_rna_dir: str,
    output_dir: str,
    max_templates: int = 4,
    anchor_mode: str = "base_center_fallback",
    arena_binary: str = "",
    arena_option: int = 5,
    arena_work_dir: str = "",
    catalog: Optional[Dict[str, List[dict]]] = None,
) -> Optional[str]:
    """Build a cross-template .npz using structures from search results.

    If *arena_binary* is set (non-empty), Arena is invoked on each template
    structure to fill missing atoms before building the template.  On Arena
    failure the original CIF is used as a fallback.

    Args:
        query_seq: The query RNA sequence.
        query_id: Identifier for the query (e.g., pdb_id_chain).
        template_specs: List of {"pdb_id": ..., "chain_id": ...} dicts.
        pdb_rna_dir: Path to CIF files.
        output_dir: Where to save .npz.
        max_templates: Max templates per .npz.
        anchor_mode: Anchor computation mode.
        arena_binary: Path to Arena binary (empty string = disabled).
        arena_option: Arena option (default 5 = fill all missing atoms).
        arena_work_dir: Working directory for Arena intermediates.

    Returns:
        Output path on success, None on failure.
    """
    query_seq = normalize_query_sequence(query_seq)
    if len(query_seq) < 5:
        return None

    template_dicts = []
    template_meta = []

    for spec in template_specs[:max_templates]:
        t_pdb_id = spec["pdb_id"]
        t_chain_id = spec["chain_id"]
        # Try to get cif_path from catalog if available
        catalog_cif = None
        if catalog and t_pdb_id in catalog:
            for ch in catalog[t_pdb_id]:
                if ch.get("cif_path"):
                    catalog_cif = ch["cif_path"]
                    break
        cif_path = find_cif_path(pdb_rna_dir, t_pdb_id, catalog_cif_path=catalog_cif)

        if cif_path is None:
            continue

        # Determine which structure to load
        structure_path = cif_path
        arena_used = False
        if arena_binary:
            work_dir = arena_work_dir or os.path.join(output_dir, "arena_tmp")
            os.makedirs(work_dir, exist_ok=True)
            arena_out = os.path.join(work_dir, f"{t_pdb_id}_{t_chain_id}_arena.pdb")
            refined = run_arena_refine(cif_path, arena_out, arena_binary, arena_option)
            if refined is not None:
                structure_path = refined
                arena_used = True
            else:
                print(f"  Info: Arena failed for {t_pdb_id}:{t_chain_id}, falling back to original CIF")

        try:
            residues = load_structure_residues(structure_path, chain_id=t_chain_id)
        except Exception as e:
            # Fallback to original CIF if Arena output couldn't be loaded
            if structure_path != cif_path:
                print(f"  Warning: Failed to load Arena output for {t_pdb_id}:{t_chain_id}: {e}, retrying with CIF")
                try:
                    residues = load_structure_residues(cif_path, chain_id=t_chain_id)
                    arena_used = False
                except Exception as e2:
                    print(f"  Warning: Also failed to load CIF {t_pdb_id}:{t_chain_id}: {e2}")
                    continue
            else:
                print(f"  Warning: Failed template {t_pdb_id}:{t_chain_id} for query {query_id}: {e}")
                continue

        template_name = f"{t_pdb_id}.cif:{t_chain_id}"
        if arena_used:
            template_name = f"{t_pdb_id}.arena.pdb:{t_chain_id}"

        try:
            td = build_minimal_template_arrays(
                query_seq=query_seq,
                residues=residues,
                template_name=template_name,
                anchor_mode=anchor_mode,
            )
            template_dicts.append(td)
            template_meta.append({
                "pdb_id": t_pdb_id,
                "chain_id": t_chain_id,
                "arena_refined": arena_used,
            })
        except Exception as e:
            print(f"  Warning: Failed template {t_pdb_id}:{t_chain_id} for query {query_id}: {e}")
            continue

    if not template_dicts:
        return None

    stacked = stack_template_dicts(template_dicts, max_templates=max_templates)
    stacked["query_sequence"] = np.asarray(list(query_seq), dtype="<U1")
    stacked["template_anchor_mode"] = np.asarray(anchor_mode)

    metadata = {
        "query_sequence": query_seq,
        "query_id": query_id,
        "mode": "cross",
        "anchor_mode": anchor_mode,
        "templates": template_meta,
    }

    output_path = os.path.join(output_dir, f"{query_id}_template.npz")
    save_npz_with_metadata(output_path, stacked, metadata)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Batch-build RNA template .npz files.")
    parser.add_argument("--catalog",
                        default="/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/rna_catalog.json",
                        help="RNA catalog JSON from Step 1.")
    parser.add_argument("--pdb_rna_dir",
                        default="/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/PDB_RNA",
                        help="Directory containing .cif files.")
    parser.add_argument("--output_dir",
                        default="/inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix/rna_database/templates",
                        help="Output directory for .npz files.")
    parser.add_argument("--mode", choices=["cross"], default="cross",
                        help="Template building mode. Only cross-template mode is supported.")
    parser.add_argument("--search_results", default="",
                        help="Search results JSON (required for cross mode).")
    parser.add_argument("--max_templates", type=int, default=4)
    parser.add_argument("--anchor_mode", default="base_center_fallback")
    # ---- CONFIGURABLE: limit for testing ----
    parser.add_argument("--max_structures", type=int, default=0,
                        help="Max structures to process (0 = all).")
    parser.add_argument("--pdb_list", default="",
                        help="Optional file with PDB IDs to process (one per line).")
    # ---- Arena atom-filling options ----
    parser.add_argument("--use_arena", action="store_true", default=False,
                        help="Enable Arena atom-filling before template construction.")
    parser.add_argument("--arena_binary",
                        default="/inspire/ssd/project/sais-bio/public/ash_proj/Arena/Arena",
                        help="Path to the Arena binary.")
    parser.add_argument("--arena_option", type=int, default=5,
                        help="Arena option (5 = fill all missing atoms).")
    parser.add_argument("--arena_work_dir", default="",
                        help="Working directory for Arena intermediate PDB files "
                             "(default: {output_dir}/arena_tmp).")
    args = parser.parse_args()

    # Resolve arena settings
    arena_binary = args.arena_binary if args.use_arena else ""
    arena_work_dir = args.arena_work_dir

    if args.use_arena:
        if not os.path.isfile(arena_binary):
            print(f"WARNING: Arena binary not found at {arena_binary}. "
                  f"Arena refinement will be skipped.")
            arena_binary = ""
        else:
            print(f"Arena atom-filling enabled: binary={arena_binary}, option={args.arena_option}")

    # Load catalog
    with open(args.catalog) as f:
        catalog = json.load(f)
    print(f"Loaded catalog with {len(catalog)} structures")

    # Optional PDB list filter
    if args.pdb_list and os.path.exists(args.pdb_list):
        with open(args.pdb_list) as f:
            allowed_ids = {line.strip().lower() for line in f if line.strip()}
        catalog = {k: v for k, v in catalog.items() if k.lower() in allowed_ids}
        print(f"Filtered to {len(catalog)} structures by PDB list.")

    if args.max_structures > 0:
        keys = sorted(catalog.keys())[:args.max_structures]
        catalog = {k: catalog[k] for k in keys}

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.search_results or not os.path.exists(args.search_results):
        print("ERROR: --search_results is required for cross mode")
        sys.exit(1)

    with open(args.search_results) as f:
        search_results = json.load(f)

    print(f"Building cross-templates for {len(search_results)} queries...")
    success = 0
    fail = 0
    for i, (query_id, info) in enumerate(sorted(search_results.items())):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(search_results)} (success={success}, fail={fail})")
        query_seq = info["query_sequence"]
        template_specs = info["templates"]
        result = build_cross_template(
            query_seq=query_seq,
            query_id=query_id,
            template_specs=template_specs,
            pdb_rna_dir=args.pdb_rna_dir,
            output_dir=args.output_dir,
            max_templates=args.max_templates,
            anchor_mode=args.anchor_mode,
            arena_binary=arena_binary,
            arena_option=args.arena_option,
            arena_work_dir=arena_work_dir,
            catalog=catalog,
        )
        if result:
            success += 1
        else:
            fail += 1

    summary = f"\nDone: {success} templates built, {fail} failures"
    if arena_binary:
        summary += f" (Arena enabled, option={args.arena_option})"
    print(summary)


if __name__ == "__main__":
    main()
