#!/usr/bin/env python3
"""
End-to-end GPU smoke test for finetune/train_pipeline.sh and finetune/inference_pipeline.sh.

This validates:
1. One-step finetune startup with RNA SS enabled.
2. Inference startup with RNA SS enabled using the produced checkpoint.
3. Inference fallback when RNA SS is enabled but priors are unavailable.

Usage:
    conda activate /inspire/ssd/project/sais-bio/public/ash_proj/conda/envs/tune_protenix
    cd /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_pro/Protenix
    python finetune/test_ss_pipeline_gpu.py
"""

from __future__ import annotations

import copy
import csv
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path("/inspire/ssd/project/sais-bio/public/ash_proj")
PROTENIX_DIR = PROJECT_ROOT / "code/protenix_pro/Protenix"
if str(PROTENIX_DIR) not in sys.path:
    sys.path.insert(0, str(PROTENIX_DIR))

from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_model_type import model_configs
from protenix.config.config import parse_configs
from protenix.model.protenix import Protenix

FINETUNE_DIR = PROTENIX_DIR / "finetune"
ENV_PATH = PROJECT_ROOT / "conda/envs/tune_protenix"
MODEL_NAME = "protenix_base_20250630_v1.0.0"
TRAIN_PDB_ID = "157d"
TRAIN_SEQUENCE = "CGCGAAUUAGCG"
INFER_SEQUENCE = "AUGCAUGC"

PASS_COUNT = 0
FAIL_COUNT = 0


def print_section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")


def check(condition: bool, msg: str) -> None:
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [PASS] {msg}")
    else:
        FAIL_COUNT += 1
        print(f"  [FAIL] {msg}")


def deep_update(dst, src) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and key in dst and isinstance(dst[key], dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value


def run_cmd(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    print(f"  $ {' '.join(shlex.quote(part) for part in cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            f"{' '.join(shlex.quote(part) for part in cmd)}"
        )
    return result


def write_sparse_prior(prior_path: Path, sequence: str, pairs: list[tuple[int, int, float]]) -> None:
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


def write_sequence_index(index_path: Path, sequence: str, prior_name: str) -> None:
    with open(index_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sequence", "path"])
        writer.writeheader()
        writer.writerow({"sequence": sequence, "path": prior_name})


def write_pdb_list(path: Path, pdb_ids: list[str]) -> None:
    path.write_text("\n".join(pdb_ids) + "\n", encoding="utf-8")


def write_sample_json(path: Path, sequence: str, sample_name: str) -> None:
    sample = {
        "name": sample_name,
        "sequences": [
            {
                "rnaSequence": {
                    "sequence": sequence,
                    "count": 1,
                }
            }
        ],
    }
    path.write_text(json.dumps([sample]), encoding="utf-8")


def build_toy_config(tmpdir: Path) -> object:
    base = copy.deepcopy({**configs_base, "data": copy.deepcopy(data_configs)})
    deep_update(base, copy.deepcopy(model_configs[MODEL_NAME]))
    cfg = parse_configs(configs=base, arg_str="", fill_required_with_null=True)

    cfg.project = "ss_pipeline_gpu_smoke"
    cfg.run_name = "ss_pipeline_gpu_smoke"
    cfg.base_dir = str(tmpdir / "toy_base")
    cfg.eval_interval = 1
    cfg.log_interval = 1
    cfg.max_steps = 1
    cfg.use_wandb = False
    cfg.load_checkpoint_path = ""
    cfg.load_ema_checkpoint_path = ""
    cfg.load_strict = True
    cfg.diffusion_batch_size = 1
    cfg.dtype = "bf16"
    cfg.enable_tf32 = False
    cfg.enable_efficient_fusion = False
    cfg.enable_diffusion_shared_vars_cache = False
    cfg.triangle_attention = "torch"
    cfg.triangle_multiplicative = "torch"
    cfg.data.msa.enable_rna_msa = False
    cfg.data.msa.enable_prot_msa = False
    cfg.rnalm.enable = False
    cfg.ribonanzanet2.enable = False
    cfg.rna_template.enable = False
    cfg.rna_ss.enable = True
    cfg.rna_ss.sequence_fpath = str(tmpdir / "toy_sequence_index.csv")
    cfg.rna_ss.feature_dir = str(tmpdir)
    cfg.rna_ss.format = "sparse_npz"
    cfg.rna_ss.n_classes = 6
    cfg.rna_ss.coverage_window = 2
    cfg.rna_ss.strict = False
    cfg.rna_ss.min_prob = 0.0
    cfg.model.N_cycle = 1
    cfg.model.msa_module.n_blocks = 1
    cfg.model.pairformer.n_blocks = 1
    cfg.model.diffusion_module.atom_encoder.n_blocks = 1
    cfg.model.diffusion_module.transformer.n_blocks = 1
    cfg.model.diffusion_module.atom_decoder.n_blocks = 1
    cfg.sample_diffusion.N_step = 1
    cfg.sample_diffusion.N_sample = 1
    cfg.model.constraint_embedder.initialize_method = "kaiming"
    cfg.model.constraint_embedder.substructure_embedder.enable = True
    cfg.model.constraint_embedder.substructure_embedder.n_classes = 6
    cfg.model.constraint_embedder.substructure_embedder.architecture = "mlp"
    cfg.model.constraint_embedder.substructure_embedder.hidden_dim = 32
    cfg.model.constraint_embedder.substructure_embedder.n_layers = 2
    cfg.model.constraint_embedder.substructure_embedder.alpha_init = 1e-2
    return cfg


def create_toy_checkpoint(checkpoint_path: Path, tmpdir: Path) -> None:
    cfg = build_toy_config(tmpdir)
    model = Protenix(cfg).cpu()
    torch.save({"model": model.state_dict()}, checkpoint_path)


def write_temp_config(base_config: Path, path: Path, overrides: dict[str, str]) -> None:
    content = base_config.read_text(encoding="utf-8")
    override_lines = "\n".join(f'{key}="{value}"' for key, value in overrides.items())
    path.write_text(f"{content}\n\n# --- test overrides ---\n{override_lines}\n", encoding="utf-8")


def find_single_checkpoint(output_root: Path) -> Path:
    candidates = sorted(output_root.glob("ss_pipeline_gpu_smoke_*"))
    if not candidates:
        raise FileNotFoundError(f"No train run directories found under {output_root}")
    checkpoint_path = candidates[-1] / "checkpoints" / "0.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Expected checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def assert_cif_exists(output_dir: Path, label: str) -> None:
    cif_files = list(output_dir.rglob("*.cif"))
    check(bool(cif_files), f"{label} produced CIF output")


def main() -> None:
    global FAIL_COUNT

    print("=" * 72)
    print("  RNA SS Finetune / Inference Pipeline GPU Smoke Test")
    print("=" * 72)
    print(f"  PyTorch: {torch.__version__}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this smoke test.")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    base_config = FINETUNE_DIR / "train_config.sh"

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        output_root = tmpdir / "outputs"
        output_root.mkdir(parents=True, exist_ok=True)

        train_prior = tmpdir / "train_sparse.npz"
        train_index = tmpdir / "train_sequence_index.csv"
        infer_prior = tmpdir / "infer_sparse.npz"
        infer_index = tmpdir / "infer_sequence_index.csv"
        infer_json = tmpdir / "infer_inputs.json"
        train_pdb_list = tmpdir / "train_pdb_list.txt"
        val_pdb_list = tmpdir / "val_pdb_list.txt"
        toy_init_ckpt = tmpdir / "toy_init.pt"

        write_sparse_prior(train_prior, TRAIN_SEQUENCE, [(0, 11, 0.9), (1, 10, 0.7)])
        write_sequence_index(train_index, TRAIN_SEQUENCE, train_prior.name)
        write_sparse_prior(infer_prior, INFER_SEQUENCE, [(0, 7, 0.9), (1, 6, 0.7)])
        write_sequence_index(infer_index, INFER_SEQUENCE, infer_prior.name)
        write_sample_json(infer_json, INFER_SEQUENCE, "ss_pipeline_infer")
        write_pdb_list(train_pdb_list, [TRAIN_PDB_ID])
        write_pdb_list(val_pdb_list, [TRAIN_PDB_ID])
        create_toy_checkpoint(toy_init_ckpt, tmpdir)
        check(toy_init_ckpt.exists(), "Toy initialization checkpoint created")

        tiny_model_args = (
            "--model.N_cycle 1 "
            "--model.msa_module.n_blocks 1 "
            "--model.pairformer.n_blocks 1 "
            "--model.diffusion_module.atom_encoder.n_blocks 1 "
            "--model.diffusion_module.transformer.n_blocks 1 "
            "--model.diffusion_module.atom_decoder.n_blocks 1 "
            "--sample_diffusion.N_step 1 "
            "--sample_diffusion.N_sample 1"
        )

        train_cfg = tmpdir / "train_ss_enabled.sh"
        write_temp_config(
            base_config,
            train_cfg,
            {
                "CONDA_ENV": str(ENV_PATH),
                "PYTHON_BIN": "python",
                "USE_WANDB": "false",
                "PRINT_ONLY": "false",
                "RUN_NAME": "ss_pipeline_gpu_smoke",
                "OUTPUT_ROOT": str(output_root),
                "MODEL_NAME_ARG": MODEL_NAME,
                "CHECKPOINT_PATH": str(toy_init_ckpt),
                "LOAD_EMA_CHECKPOINT_PATH": "",
                "TRAIN_MODE": "1stage",
                "EVAL_INTERVAL": "-1",
                "CHECKPOINT_INTERVAL": "-1",
                "LOG_INTERVAL": "1",
                "MAX_STEPS": "1",
                "DIFFUSION_BATCH_SIZE": "1",
                "TRAIN_CROP_SIZE": "32",
                "DATA_NUM_DL_WORKERS": "0",
                "CUDA_VISIBLE_DEVICES_VALUE": "0",
                "TRIANGLE_ATTENTION_IMPL": "torch",
                "TRIANGLE_MULTIPLICATIVE_IMPL": "torch",
                "LAYERNORM_TYPE_VALUE": "torch",
                "USE_RNALM": "false",
                "USE_RIBONANZA": "false",
                "USE_RNA_TEMPLATE": "false",
                "USE_RNA_MSA": "false",
                "USE_PROT_MSA": "false",
                "USE_PROT_TEMPLATE": "false",
                "USE_RNA_SS": "true",
                "RNA_SS_SEQUENCE_FPATH": str(train_index),
                "RNA_SS_FEATURE_DIR": str(tmpdir),
                "RNA_SS_FORMAT": "sparse_npz",
                "RNA_SS_N_CLASSES": "6",
                "RNA_SS_COVERAGE_WINDOW": "2",
                "RNA_SS_STRICT": "true",
                "RNA_SS_ARCHITECTURE": "mlp",
                "RNA_SS_HIDDEN_DIM": "32",
                "RNA_SS_N_LAYERS": "2",
                "RNA_SS_ALPHA_INIT": "0.01",
                "RNA_SS_INIT_METHOD": "kaiming",
                "TRAIN_PDB_LIST": str(train_pdb_list),
                "VAL_PDB_LIST": str(val_pdb_list),
                "TRAIN_EXTRA_ARGS": tiny_model_args,
            },
        )

        print_section("Stage 1: One-Step Finetune Startup With RNA SS")
        run_cmd(["bash", str(FINETUNE_DIR / "train_pipeline.sh"), str(train_cfg)], PROTENIX_DIR)
        finetuned_ckpt = find_single_checkpoint(output_root)
        check(finetuned_ckpt.exists(), f"Finetune pipeline produced checkpoint: {finetuned_ckpt}")

        infer_cfg = tmpdir / "infer_ss_enabled.sh"
        infer_dump = tmpdir / "infer_ss_enabled_out"
        write_temp_config(
            base_config,
            infer_cfg,
            {
                "CONDA_ENV": str(ENV_PATH),
                "PYTHON_BIN": "python",
                "PRINT_ONLY": "false",
                "OUTPUT_ROOT": str(output_root),
                "MODEL_NAME_ARG": MODEL_NAME,
                "INFER_MODEL_NAME": MODEL_NAME,
                "INFER_CHECKPOINT_PATH": str(finetuned_ckpt),
                "INFER_INPUT_JSON": str(infer_json),
                "INFER_DUMP_DIR": str(infer_dump),
                "INFER_LOAD_STRICT": "false",
                "INFER_NUM_WORKERS": "0",
                "INFER_SEEDS": "101",
                "INFER_USE_SEEDS_IN_JSON": "false",
                "INFER_DTYPE": "bf16",
                "INFER_N_SAMPLE": "1",
                "INFER_N_STEP": "1",
                "INFER_N_CYCLE": "1",
                "INFER_USE_MSA": "false",
                "INFER_USE_TEMPLATE": "false",
                "INFER_USE_RNA_MSA": "false",
                "INFER_NEED_ATOM_CONFIDENCE": "false",
                "INFER_SORTED_BY_RANKING_SCORE": "true",
                "INFER_ENABLE_TF32": "false",
                "INFER_ENABLE_EFFICIENT_FUSION": "false",
                "INFER_ENABLE_DIFFUSION_SHARED_VARS_CACHE": "false",
                "CUDA_VISIBLE_DEVICES_VALUE": "0",
                "TRIANGLE_ATTENTION_IMPL": "torch",
                "TRIANGLE_MULTIPLICATIVE_IMPL": "torch",
                "LAYERNORM_TYPE_VALUE": "torch",
                "USE_RNALM": "false",
                "USE_RIBONANZA": "false",
                "USE_RNA_TEMPLATE": "false",
                "USE_RNA_SS": "true",
                "RNA_SS_SEQUENCE_FPATH": str(infer_index),
                "RNA_SS_FEATURE_DIR": str(tmpdir),
                "RNA_SS_FORMAT": "sparse_npz",
                "RNA_SS_N_CLASSES": "6",
                "RNA_SS_COVERAGE_WINDOW": "2",
                "RNA_SS_STRICT": "true",
                "RNA_SS_ARCHITECTURE": "mlp",
                "RNA_SS_HIDDEN_DIM": "32",
                "RNA_SS_N_LAYERS": "2",
                "RNA_SS_ALPHA_INIT": "0.01",
                "RNA_SS_INIT_METHOD": "kaiming",
                "INFER_EXTRA_ARGS": tiny_model_args,
            },
        )

        print_section("Stage 2: Inference Startup With RNA SS")
        run_cmd(["bash", str(FINETUNE_DIR / "inference_pipeline.sh"), str(infer_cfg)], PROTENIX_DIR)
        assert_cif_exists(infer_dump, "Inference pipeline with RNA SS")

        fallback_cfg = tmpdir / "infer_ss_fallback.sh"
        fallback_dump = tmpdir / "infer_ss_fallback_out"
        write_temp_config(
            base_config,
            fallback_cfg,
            {
                "CONDA_ENV": str(ENV_PATH),
                "PYTHON_BIN": "python",
                "PRINT_ONLY": "false",
                "OUTPUT_ROOT": str(output_root),
                "MODEL_NAME_ARG": MODEL_NAME,
                "INFER_MODEL_NAME": MODEL_NAME,
                "INFER_CHECKPOINT_PATH": str(finetuned_ckpt),
                "INFER_INPUT_JSON": str(infer_json),
                "INFER_DUMP_DIR": str(fallback_dump),
                "INFER_LOAD_STRICT": "false",
                "INFER_NUM_WORKERS": "0",
                "INFER_SEEDS": "101",
                "INFER_USE_SEEDS_IN_JSON": "false",
                "INFER_DTYPE": "bf16",
                "INFER_N_SAMPLE": "1",
                "INFER_N_STEP": "1",
                "INFER_N_CYCLE": "1",
                "INFER_USE_MSA": "false",
                "INFER_USE_TEMPLATE": "false",
                "INFER_USE_RNA_MSA": "false",
                "INFER_NEED_ATOM_CONFIDENCE": "false",
                "INFER_SORTED_BY_RANKING_SCORE": "true",
                "INFER_ENABLE_TF32": "false",
                "INFER_ENABLE_EFFICIENT_FUSION": "false",
                "INFER_ENABLE_DIFFUSION_SHARED_VARS_CACHE": "false",
                "CUDA_VISIBLE_DEVICES_VALUE": "0",
                "TRIANGLE_ATTENTION_IMPL": "torch",
                "TRIANGLE_MULTIPLICATIVE_IMPL": "torch",
                "LAYERNORM_TYPE_VALUE": "torch",
                "USE_RNALM": "false",
                "USE_RIBONANZA": "false",
                "USE_RNA_TEMPLATE": "false",
                "USE_RNA_SS": "true",
                "RNA_SS_SEQUENCE_FPATH": "",
                "RNA_SS_FEATURE_DIR": "",
                "RNA_SS_FORMAT": "sparse_npz",
                "RNA_SS_N_CLASSES": "6",
                "RNA_SS_COVERAGE_WINDOW": "2",
                "RNA_SS_STRICT": "false",
                "RNA_SS_ARCHITECTURE": "mlp",
                "RNA_SS_HIDDEN_DIM": "32",
                "RNA_SS_N_LAYERS": "2",
                "RNA_SS_ALPHA_INIT": "0.01",
                "RNA_SS_INIT_METHOD": "kaiming",
                "INFER_EXTRA_ARGS": tiny_model_args,
            },
        )

        print_section("Stage 3: Inference Fallback With Missing RNA SS Priors")
        run_cmd(["bash", str(FINETUNE_DIR / "inference_pipeline.sh"), str(fallback_cfg)], PROTENIX_DIR)
        assert_cif_exists(fallback_dump, "Inference pipeline fallback without priors")

    print_section("Summary")
    print(f"  Total checks: {PASS_COUNT + FAIL_COUNT}")
    print(f"  Passed: {PASS_COUNT}")
    print(f"  Failed: {FAIL_COUNT}")
    if FAIL_COUNT:
        raise SystemExit(1)
    print("\n  ALL TESTS PASSED")


if __name__ == "__main__":
    main()
