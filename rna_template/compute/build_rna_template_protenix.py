#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from rna_template_common import (
    build_minimal_template_arrays,
    load_structure_residues,
    parse_template_spec,
    read_text_sequence,
    save_npz_with_metadata,
    stack_template_dicts,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build Protenix-compatible RNA template features from mmCIF/PDB files. "
            "Output keys include template_aatype, template_distogram, "
            "template_pseudo_beta_mask, template_unit_vector, and "
            "template_backbone_frame_mask."
        )
    )
    parser.add_argument(
        "--query_seq",
        required=True,
        help="RNA query sequence, or a FASTA/plain-text file containing the sequence.",
    )
    parser.add_argument(
        "--template",
        dest="templates",
        action="append",
        required=True,
        help="Template spec in the form path or path:CHAIN. Can be repeated.",
    )
    parser.add_argument("--output", required=True, help="Output .npz path.")
    parser.add_argument("--max_templates", type=int, default=4)
    parser.add_argument(
        "--anchor_mode",
        choices=["base_center", "base_center_fallback", "c4p", "c4p_fallback"],
        default="base_center_fallback",
        help=(
            "Anchor used for the distogram. base_center_fallback is recommended: "
            "use base-center first, then fall back to C4' / C1'."
        ),
    )
    parser.add_argument("--min_bin", type=float, default=3.25)
    parser.add_argument("--max_bin", type=float, default=50.75)
    parser.add_argument("--num_bins", type=int, default=39)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    query_seq = read_text_sequence(args.query_seq)

    template_dicts: List[dict] = []
    template_meta: List[dict] = []

    for spec in args.templates:
        path_str, chain_id = parse_template_spec(spec)
        residues = load_structure_residues(path_str, chain_id=chain_id)
        template_name = f"{Path(path_str).name}:{chain_id or residues[0].chain_id}"
        td = build_minimal_template_arrays(
            query_seq=query_seq,
            residues=residues,
            template_name=template_name,
            anchor_mode=args.anchor_mode,
            min_bin=args.min_bin,
            max_bin=args.max_bin,
            num_bins=args.num_bins,
        )
        template_dicts.append(td)
        template_meta.append(
            {
                "template_name": template_name,
                "source_file": str(Path(path_str).resolve()),
                "chain_id": chain_id or residues[0].chain_id,
                "num_residues_in_template_chain": len(residues),
            }
        )

    stacked = stack_template_dicts(template_dicts, max_templates=args.max_templates)
    stacked["query_sequence"] = np.asarray(list(query_seq), dtype="<U1")
    stacked["template_anchor_mode"] = np.asarray(args.anchor_mode)

    metadata = {
        "query_sequence": query_seq,
        "output_description": "Protenix-compatible minimal RNA template features.",
        "anchor_mode": args.anchor_mode,
        "binning": {
            "min_bin": args.min_bin,
            "max_bin": args.max_bin,
            "num_bins": args.num_bins,
        },
        "templates": template_meta,
        "notes": [
            "template_aatype uses official Protenix RNA ids: A=21, G=22, C=23, U=24, N=25, gap=31.",
            "Unaligned query positions are filled with gap id 31, zero anchor/frame masks, and zero pairwise features.",
            "template_pseudo_beta_mask and template_backbone_frame_mask are emitted as pairwise [T, N, N] masks.",
            "If your exact Protenix checkout expects 1D masks instead, use template_anchor_mask_1d and template_frame_mask_1d.",
        ],
    }
    save_npz_with_metadata(args.output, stacked, metadata)


if __name__ == "__main__":
    main()
