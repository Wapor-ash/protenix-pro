#!/usr/bin/env python3
import csv
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

os.environ.setdefault("LAYERNORM_TYPE", "torch")

from protenix.data.rna_ss.rna_ss_featurizer import RNASSFeaturizer
from protenix.model.modules.embedders import ConstraintEmbedder
from protenix.utils.two_stage_adapter import (
    RNA_SS_REQUIRED_ADAPTER_SUBSTRINGS,
    parse_adapter_keywords,
    collect_required_adapter_param_substrings,
    validate_required_adapter_matches,
)

PASS_COUNT = 0
FAIL_COUNT = 0

RNA_VALUES = {"A": 21, "U": 24, "G": 23, "C": 22}


def print_section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check(condition, msg):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [PASS] {msg}")
    else:
        FAIL_COUNT += 1
        print(f"  [FAIL] {msg}")


def write_sequence_index(index_path: Path, sequence: str, prior_name: str):
    with open(index_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "path"])
        writer.writeheader()
        writer.writerow({"sequence": sequence, "path": prior_name})


class FakeTokenArray:
    def __init__(self, centre_atom_index, values):
        self._centre = np.asarray(centre_atom_index, dtype=np.int64)
        self._values = np.asarray(values, dtype=np.int64)

    def get_annotation(self, name):
        if name != "centre_atom_index":
            raise KeyError(name)
        return self._centre

    def __getitem__(self, idx):
        if isinstance(idx, (int, np.integer)):
            return SimpleNamespace(value=int(self._values[idx]))
        idx = np.asarray(idx)
        return FakeTokenArray(self._centre[idx], self._values[idx])

    def __len__(self):
        return len(self._centre)


class FakeAtomArray:
    def __init__(self, **kwargs):
        self.__dict__.update({k: np.asarray(v) for k, v in kwargs.items()})

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        return FakeAtomArray(**{k: v[idx] for k, v in self.__dict__.items()})


def test_sparse_prior_auto_symmetrizes():
    print_section("Test 1: Sparse Prior Auto-Symmetrizes One-Sided Edges")
    sequence = "AUGC"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        prior_path = tmpdir / "one_sided.npz"
        np.savez(
            prior_path,
            pair_i=np.asarray([0], dtype=np.int64),
            pair_j=np.asarray([3], dtype=np.int64),
            pair_p=np.asarray([0.9], dtype=np.float32),
            length=np.asarray(4, dtype=np.int64),
        )
        index_path = tmpdir / "sequence_to_prior.csv"
        write_sequence_index(index_path, sequence, prior_path.name)

        featurizer = RNASSFeaturizer(
            sequence_fpath=str(index_path),
            feature_dir=str(tmpdir),
            format="sparse_npz",
            strict=True,
        )
        prior = featurizer._load_prior(sequence)
        substructure = featurizer._build_chain_substructure(
            prior,
            np.asarray([0, 3], dtype=np.int64),
        )

        check(
            np.isclose(substructure[0, 1, 0], 0.9),
            f"Forward edge kept: {substructure[0, 1, 0]:.3f}",
        )
        check(
            np.isclose(substructure[1, 0, 0], 0.9),
            f"Reverse edge auto-filled: {substructure[1, 0, 0]:.3f}",
        )
        check(
            np.allclose(substructure[:, :, 0], substructure[:, :, 0].T),
            "Pair prior is symmetric after loading",
        )


def test_alignment_fallback_uses_prior_length():
    print_section("Test 2: Alignment Fallback Uses Prior Length")
    sequence = "AUGC"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        prior_path = tmpdir / "dense.npz"
        bpp = np.zeros((4, 4), dtype=np.float32)
        bpp[0, 3] = bpp[3, 0] = 0.8
        np.savez(prior_path, bpp=bpp)
        index_path = tmpdir / "sequence_to_prior.csv"
        write_sequence_index(index_path, sequence, prior_path.name)

        featurizer = RNASSFeaturizer(
            sequence_fpath=str(index_path),
            feature_dir=str(tmpdir),
            format="dense_npz",
            strict=False,
        )
        full_tokens = FakeTokenArray([0, 1, 2, 3], [RNA_VALUES[c] for c in sequence])
        full_atoms = FakeAtomArray(
            chain_mol_type=["rna", "rna", "rna", "rna"],
            label_asym_id=["A", "A", "A", "A"],
            label_entity_id=["1", "1", "1", "1"],
            res_id=[10, 11, 12, 13],
            chain_id=["A", "A", "A", "A"],
            entity_id=["1", "1", "1", "1"],
            mol_type=["rna", "rna", "rna", "rna"],
        )
        result = featurizer(
            full_token_array=full_tokens,
            full_atom_array=full_atoms,
            cropped_token_array=full_tokens,
            cropped_atom_array=full_atoms,
            # Keep the prior lookup exact, but make residue numbering incompatible
            # so strict=false must fall back to sequential chain order.
            entity_to_sequences={"1": sequence},
            selected_token_indices=np.arange(4),
        )
        substructure = result["substructure"].numpy()

        check(
            np.isclose(substructure[0, 3, 0], 0.8),
            f"Sequential fallback restored first pair: {substructure[0, 3, 0]:.3f}",
        )
        check(
            substructure[:, :, 5].sum() == 16.0,
            f"Valid mask preserved full chain coverage: {substructure[:, :, 5].sum():.0f}",
        )


def test_symmetric_copies_do_not_share_chain_groups():
    print_section("Test 3: Symmetric Assembly Copies Stay Isolated")
    sequence = "AUGC"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        prior_path = tmpdir / "dense_copy_isolation.npz"
        bpp = np.zeros((4, 4), dtype=np.float32)
        bpp[0, 3] = bpp[3, 0] = 0.8
        np.savez(prior_path, bpp=bpp)
        index_path = tmpdir / "sequence_to_prior.csv"
        write_sequence_index(index_path, sequence, prior_path.name)

        featurizer = RNASSFeaturizer(
            sequence_fpath=str(index_path),
            feature_dir=str(tmpdir),
            format="dense_npz",
            strict=True,
        )
        full_sequence = sequence * 2
        full_tokens = FakeTokenArray(np.arange(8), [RNA_VALUES[c] for c in full_sequence])
        full_atoms = FakeAtomArray(
            chain_mol_type=["rna"] * 8,
            asym_id_int=[0, 0, 0, 0, 1, 1, 1, 1],
            label_asym_id=["A"] * 8,
            label_entity_id=["1"] * 8,
            res_id=[1, 2, 3, 4, 1, 2, 3, 4],
            chain_id=["A", "A", "A", "A", "A.1", "A.1", "A.1", "A.1"],
            entity_id=["1"] * 8,
            mol_type=["rna"] * 8,
        )
        selected = np.asarray([0, 7], dtype=np.int64)
        result = featurizer(
            full_token_array=full_tokens,
            full_atom_array=full_atoms,
            cropped_token_array=full_tokens[selected],
            cropped_atom_array=full_atoms[selected],
            entity_to_sequences={"1": sequence},
            selected_token_indices=selected,
        )
        substructure = result["substructure"].numpy()

        check(
            np.isclose(substructure[0, 1, 0], 0.0),
            f"Cross-copy P_in stays zero: {substructure[0, 1, 0]:.3f}",
        )
        check(
            np.isclose(substructure[0, 1, 5], 0.0) and np.isclose(substructure[1, 0, 5], 0.0),
            "Cross-copy valid mask stays zero across symmetric copies",
        )
        check(
            np.isclose(substructure[0, 0, 5], 1.0) and np.isclose(substructure[1, 1, 5], 1.0),
            "Each copy still keeps its own same-chain diagonal validity",
        )


def test_constraint_embedder_supports_branch_specific_init():
    print_section("Test 4: ConstraintEmbedder Supports Branch-Specific Init")
    embedder = ConstraintEmbedder(
        pocket_embedder={"enable": True, "c_z_input": 1},
        contact_embedder={"enable": False, "c_z_input": 2},
        contact_atom_embedder={"enable": False, "c_z_input": 2},
        substructure_embedder={
            "enable": True,
            "n_classes": 6,
            "architecture": "mlp",
            "hidden_dim": 16,
            "n_layers": 3,
            "alpha_init": 1e-2,
            "initialize_method": "kaiming",
        },
        c_constraint_z=8,
        initialize_method="zero",
    )

    pocket_weight = embedder.pocket_z_embedder.weight.detach().cpu()
    sub_weight = embedder.substructure_z_embedder.network[0].weight.detach().cpu()

    check(
        torch.allclose(pocket_weight, torch.zeros_like(pocket_weight)),
        "Legacy pocket branch keeps zero init",
    )
    check(
        not torch.allclose(sub_weight, torch.zeros_like(sub_weight)),
        "Substructure branch can use independent non-zero init",
    )


def test_constraint_embedder_zero_init_skips_alpha_and_starts_zero():
    print_section("Test 5: ConstraintEmbedder Zero Init Skips Alpha Gate")
    embedder = ConstraintEmbedder(
        pocket_embedder={"enable": False, "c_z_input": 1},
        contact_embedder={"enable": False, "c_z_input": 2},
        contact_atom_embedder={"enable": False, "c_z_input": 2},
        substructure_embedder={
            "enable": True,
            "n_classes": 6,
            "architecture": "mlp",
            "hidden_dim": 16,
            "n_layers": 3,
            "initialize_method": "zero",
            "alpha_init": 1e-2,
        },
        c_constraint_z=8,
        initialize_method="zero",
    )

    substructure = torch.randn(4, 4, 6)
    output = embedder({"substructure": substructure})
    final_weight = embedder.substructure_z_embedder.network[-1].weight.detach().cpu()

    check(
        not hasattr(embedder, "substructure_log_alpha"),
        "Zero-init substructure branch does not create substructure_log_alpha",
    )
    check(
        torch.allclose(final_weight, torch.zeros_like(final_weight)),
        "Zero-init substructure branch zeroes the terminal projection",
    )
    check(
        torch.allclose(output, torch.zeros_like(output)),
        "Zero-init substructure branch starts with exact zero output",
    )


def test_required_adapter_validation_catches_missing_rna_ss_keywords():
    print_section("Test 6: Adapter Validation Catches Missing RNA SS Keywords")
    param_names = [
        "constraint_embedder.substructure_z_embedder.network.0.weight",
        "constraint_embedder.substructure_log_alpha",
    ]

    failed = False
    try:
        validate_required_adapter_matches(
            param_names=param_names,
            adapter_keywords=parse_adapter_keywords("rnalm_projection"),
            required_substrings=RNA_SS_REQUIRED_ADAPTER_SUBSTRINGS,
        )
    except RuntimeError:
        failed = True
    check(failed, "Validation rejects adapter keywords that miss RNA SS params")

    validate_required_adapter_matches(
        param_names=param_names,
        adapter_keywords=parse_adapter_keywords(
            "constraint_embedder.substructure_z_embedder,constraint_embedder.substructure_log_alpha"
        ),
        required_substrings=RNA_SS_REQUIRED_ADAPTER_SUBSTRINGS,
    )
    check(True, "Validation accepts adapter keywords that cover RNA SS params")


def test_required_adapter_validation_skips_alpha_for_zero_init():
    print_section("Test 7: Zero Init Adapter Validation Skips Alpha Requirement")
    config = {
        "rna_ss": {"enable": True},
        "model": {
            "constraint_embedder": {
                "initialize_method": "zero",
                "substructure_embedder": {
                    "enable": True,
                    "initialize_method": "zero",
                },
            }
        },
    }
    required_substrings = collect_required_adapter_param_substrings(config)
    check(
        required_substrings == ["constraint_embedder.substructure_z_embedder"],
        f"Zero-init required adapter params shrink to substructure embedder only: {required_substrings}",
    )

    validate_required_adapter_matches(
        param_names=["constraint_embedder.substructure_z_embedder.network.0.weight"],
        adapter_keywords=parse_adapter_keywords("constraint_embedder.substructure_z_embedder"),
        required_substrings=required_substrings,
    )
    check(True, "Zero-init adapter validation no longer requires substructure_log_alpha")


def main():
    tests = [
        test_sparse_prior_auto_symmetrizes,
        test_alignment_fallback_uses_prior_length,
        test_symmetric_copies_do_not_share_chain_groups,
        test_constraint_embedder_supports_branch_specific_init,
        test_constraint_embedder_zero_init_skips_alpha_and_starts_zero,
        test_required_adapter_validation_catches_missing_rna_ss_keywords,
        test_required_adapter_validation_skips_alpha_for_zero_init,
    ]

    for test_fn in tests:
        test_fn()

    print_section("Summary")
    print(f"  Total checks: {PASS_COUNT + FAIL_COUNT}")
    print(f"  Passed: {PASS_COUNT}")
    print(f"  Failed: {FAIL_COUNT}")

    if FAIL_COUNT:
        raise SystemExit(1)
    print("\n  ALL TESTS PASSED")


if __name__ == "__main__":
    main()
