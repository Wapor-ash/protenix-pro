#!/usr/bin/env python3
"""
Smoke test for RNA SS pair-only integration via constraint_feature["substructure"].

Usage:
    conda activate /inspire/ssd/project/sais-bio/public/ash_proj/conda/envs/tune_protenix
    cd /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_pro/Protenix
    python test_ss_integration.py
"""

import csv
import json
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PASS_COUNT = 0
FAIL_COUNT = 0


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


def write_sparse_prior(prior_path: Path, sequence: str, pairs: list[tuple[int, int, float]]):
    seq_len = len(sequence)
    pair_i = []
    pair_j = []
    pair_p = []
    row_sum = np.zeros(seq_len, dtype=np.float32)
    for i, j, p in pairs:
        pair_i.extend([i, j])
        pair_j.extend([j, i])
        pair_p.extend([p, p])
        row_sum[i] += p
        row_sum[j] += p
    np.savez(
        prior_path,
        pair_i=np.asarray(pair_i, dtype=np.int64),
        pair_j=np.asarray(pair_j, dtype=np.int64),
        pair_p=np.asarray(pair_p, dtype=np.float32),
        row_sum=row_sum,
        length=np.asarray(seq_len, dtype=np.int64),
    )


def write_sequence_index(index_path: Path, sequence: str, prior_name: str):
    with open(index_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "path"])
        writer.writeheader()
        writer.writerow({"sequence": sequence, "path": prior_name})


def test_rna_ss_featurizer_sparse():
    print_section("Test 1: RNASSFeaturizer Sparse Prior")

    from protenix.data.rna_ss.rna_ss_featurizer import RNASSFeaturizer

    sequence = "AUGCAUGC"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        prior_path = tmpdir / "sample_sparse.npz"
        index_path = tmpdir / "sequence_to_prior.csv"
        write_sparse_prior(prior_path, sequence, [(0, 7, 0.9), (1, 6, 0.7)])
        write_sequence_index(index_path, sequence, prior_path.name)

        featurizer = RNASSFeaturizer(
            sequence_fpath=str(index_path),
            feature_dir=str(tmpdir),
            format="sparse_npz",
            coverage_window=1,
        )
        prior = featurizer._load_prior(sequence)
        positions = np.asarray([0, 1, 6, 7], dtype=np.int64)
        substructure = featurizer._build_chain_substructure(prior, positions)

        check(substructure.shape == (4, 4, 6), f"substructure shape: {substructure.shape}")
        check(
            np.isclose(substructure[0, 3, 0], 0.9),
            f"P_in keeps sparse prior: {substructure[0, 3, 0]:.3f}",
        )
        check(
            np.isclose(substructure[1, 2, 0], 0.7),
            f"Second pair keeps sparse prior: {substructure[1, 2, 0]:.3f}",
        )
        check(
            np.allclose(substructure[..., 1], 0.0) and np.allclose(substructure[..., 2], 0.0),
            "Outside mass is zero for full-length chain",
        )
        check(
            np.allclose(substructure[..., 5], 1.0),
            "Valid same-chain RNA mask is one inside the chain block",
        )
        check(
            substructure[0, 0, 3] > 0.0,
            f"Reliability channel populated: {substructure[0, 0, 3]:.3f}",
        )

    return True


def test_constraint_embedder_substructure_gate():
    print_section("Test 2: ConstraintEmbedder Substructure Gate")

    from protenix.model.modules.embedders import ConstraintEmbedder

    embedder = ConstraintEmbedder(
        pocket_embedder={"enable": False, "c_z_input": 1},
        contact_embedder={"enable": False, "c_z_input": 2},
        contact_atom_embedder={"enable": False, "c_z_input": 2},
        substructure_embedder={
            "enable": True,
            "n_classes": 6,
            "architecture": "mlp",
            "hidden_dim": 32,
            "n_layers": 3,
            "alpha_init": 1e-2,
        },
        c_constraint_z=128,
        initialize_method="kaiming",
    ).to(DEVICE)

    check(embedder({}) is None, "Missing substructure key is skipped cleanly")

    substructure = torch.randn(6, 6, 6, device=DEVICE)
    output = embedder({"substructure": substructure})
    alpha = torch.exp(embedder.substructure_log_alpha).item()

    check(output.shape == (6, 6, 128), f"Output shape: {tuple(output.shape)}")
    check(0.0 < alpha < 0.1, f"Substructure gate alpha initialized small: {alpha:.5f}")
    check(output.abs().max().item() > 0.0, "Substructure branch produces non-zero output")

    return True


def test_inference_dataset_rna_ss():
    print_section("Test 3: InferenceDataset RNA SS Wiring")

    from ml_collections.config_dict import ConfigDict

    from protenix.data.inference.infer_dataloader import InferenceDataset
    import protenix.data.core.ccd as ccd

    sequence = "AUGCAUGC"
    sample = {
        "name": "rna_ss_inference_smoke",
        "sequences": [
            {
                "rnaSequence": {
                    "sequence": sequence,
                    "count": 1,
                }
            }
        ],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_json = tmpdir / "inputs.json"
        prior_path = tmpdir / "sample_sparse.npz"
        index_path = tmpdir / "sequence_to_prior.csv"
        input_json.write_text(json.dumps([sample]))
        write_sparse_prior(prior_path, sequence, [(0, 7, 0.9), (1, 6, 0.7)])
        write_sequence_index(index_path, sequence, prior_path.name)

        ccd.COMPONENTS_FILE = str(PROJECT_ROOT / "data/common/components.cif")
        ccd.RKDIT_MOL_PKL = PROJECT_ROOT / "data/common/components.cif.rdkit_mol.pkl"
        ccd.biotite_load_ccd_cif.cache_clear()
        ccd.get_component_atom_array.cache_clear()
        ccd.get_one_letter_code.cache_clear()
        ccd.get_mol_type.cache_clear()

        configs = ConfigDict()
        configs.input_json_path = str(input_json)
        configs.dump_dir = str(tmpdir)
        configs.use_msa = False
        configs.use_template = False
        configs.num_workers = 0
        configs.esm = ConfigDict({"enable": False, "model_name": "esm2_t33_650M_UR50D"})
        configs.rnalm = ConfigDict({"enable": False})
        configs.rna_template = ConfigDict({"enable": False})
        configs.ribonanzanet2 = ConfigDict({"enable": False})
        configs.rna_ss = ConfigDict(
            {
                "enable": True,
                "sequence_fpath": str(index_path),
                "feature_dir": str(tmpdir),
                "format": "sparse_npz",
                "n_classes": 6,
                "coverage_window": 2,
                "strict": True,
                "min_prob": 0.0,
            }
        )
        configs.data = ConfigDict(
            {
                "template": ConfigDict(
                    {
                        "prot_template_mmcif_dir": "",
                        "prot_template_cache_dir": "",
                        "kalign_binary_path": "",
                        "release_dates_path": "",
                        "obsolete_pdbs_path": "",
                    }
                )
            }
        )

        dataset = InferenceDataset(configs=configs)
        data, atom_array, time_tracker = dataset.process_one(sample)
        feat = data["input_feature_dict"]
        substructure = feat["constraint_feature"]["substructure"]

        check("constraint_feature" in feat, "Inference output contains constraint_feature")
        check("substructure" in feat["constraint_feature"], "Inference output contains substructure")
        check(
            substructure.shape == (len(sequence), len(sequence), 6),
            f"Inference substructure shape: {tuple(substructure.shape)}",
        )
        check(
            torch.isclose(substructure[0, 7, 0], torch.tensor(0.9), atol=1e-5),
            f"Inference keeps P_in=0.900: {substructure[0, 7, 0].item():.3f}",
        )
        check(
            substructure[..., 5].sum().item() == len(sequence) * len(sequence),
            f"Inference valid mask covers full RNA chain: {substructure[..., 5].sum().item():.0f}",
        )
        check(atom_array is not None, "Inference path returns atom_array")
        check("featurizer" in time_tracker, "Inference path returns featurizer timing")

    return True


def test_training_dataset_rna_ss():
    print_section("Test 4: Training Dataset RNA SS Wiring")

    from protenix.data.pipeline.dataset import BaseSingleDataset
    from protenix.data.rna_ss.rna_ss_featurizer import RNASSFeaturizer

    sequence = "CGCGAAUUAGCG"
    prepared_root = PROJECT_ROOT / "data/stanford-rna-3d-folding/part2/protenix_prepared"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        prior_path = tmpdir / "157d_sparse.npz"
        index_path = tmpdir / "sequence_to_prior.csv"
        write_sparse_prior(prior_path, sequence, [(0, 11, 0.9), (1, 10, 0.7)])
        write_sequence_index(index_path, sequence, prior_path.name)

        dataset = BaseSingleDataset(
            mmcif_dir=str(PROJECT_ROOT / "data/stanford-rna-3d-folding/part2/PDB_RNA"),
            bioassembly_dict_dir=str(prepared_root / "rna_bioassembly"),
            indices_fpath=str(prepared_root / "indices/rna_bioassembly_indices.csv"),
            cropping_configs={
                "crop_size": 0,
                "method_weights": [1.0, 0.0, 0.0],
                "contiguous_crop_complete_lig": True,
                "spatial_crop_complete_lig": True,
                "drop_last": True,
                "remove_metal": True,
            },
            msa_featurizer=None,
            template_featurizer=None,
            name="rna_ss_train_smoke",
            ref_pos_augment=False,
            pdb_list=["157d"],
            rna_ss_featurizer=RNASSFeaturizer(
                sequence_fpath=str(index_path),
                feature_dir=str(tmpdir),
                format="sparse_npz",
                coverage_window=2,
                strict=True,
            ),
            constraint={"enable": False},
        )

        data = dataset.process_one(0)
        substructure = data["input_feature_dict"]["constraint_feature"]["substructure"]

        check(
            substructure.shape[-1] == 6,
            f"Training substructure has 6 channels: {tuple(substructure.shape)}",
        )
        check(
            torch.isclose(substructure[0, 11, 0], torch.tensor(0.9), atol=1e-5),
            f"Training path keeps first-chain P_in=0.900: {substructure[0, 11, 0].item():.3f}",
        )
        check(
            torch.isclose(substructure[0, 12, 5], torch.tensor(0.0), atol=1e-5),
            "Inter-chain valid mask stays zero",
        )
        check(
            substructure[..., 5].sum().item() > 0,
            "Training path injects non-empty same-chain valid mask",
        )

    return True


def main():
    global PASS_COUNT, FAIL_COUNT
    print("=" * 60)
    print("  RNA SS Pair-Only Integration Smoke Test")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  PyTorch: {torch.__version__}")
    if DEVICE == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    tests = [
        ("RNASSFeaturizer Sparse Prior", test_rna_ss_featurizer_sparse),
        ("ConstraintEmbedder Substructure Gate", test_constraint_embedder_substructure_gate),
        ("InferenceDataset RNA SS Wiring", test_inference_dataset_rna_ss),
        ("Training Dataset RNA SS Wiring", test_training_dataset_rna_ss),
    ]

    results = {}
    for name, test_fn in tests:
        try:
            result = test_fn()
            results[name] = "PASS" if result else "FAIL"
        except Exception as exc:
            results[name] = f"ERROR: {exc}"
            traceback.print_exc()
            FAIL_COUNT += 1

    print_section("Summary")
    for name, result in results.items():
        status = "PASS" if "PASS" in str(result) else "FAIL"
        print(f"  [{status}] {name}: {result}")

    print(f"\n  Total checks: {PASS_COUNT + FAIL_COUNT}")
    print(f"  Passed: {PASS_COUNT}")
    print(f"  Failed: {FAIL_COUNT}")

    if FAIL_COUNT == 0:
        print("\n  ALL TESTS PASSED")
    else:
        print(f"\n  {FAIL_COUNT} TESTS FAILED")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
