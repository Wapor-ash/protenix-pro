#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path("/inspire/ssd/project/sais-bio/public/ash_proj")
REPO_NEW = PROJECT_ROOT / "code/protenix_new/Protenix"
REPO_PRO = PROJECT_ROOT / "code/protenix_pro/Protenix"
DATA_ROOT = PROJECT_ROOT / "data/stanford-rna-3d-folding/part2"
PREPARED_ROOT = DATA_ROOT / "protenix_prepared"
INDEX_CSV = PREPARED_ROOT / "indices/rna_bioassembly_indices.csv"
CHECKPOINT_NAME = "protenix_base_20250630_v1.0.0.pt"
MODEL_NAME = "protenix_base_20250630_v1.0.0"
OUTPUT_ROOT = REPO_PRO / "artifacts" / "rnalm_repo_equivalence_20260324"

SAMPLES = [
    {
        "tag": "rna_157d_A",
        "pdb_id": "157d",
        "chain_1_id": "A",
        "type": "chain",
    },
    {
        "tag": "hybrid_165d_A",
        "pdb_id": "165d",
        "chain_1_id": "A",
        "type": "chain",
    },
]

MODES = [
    {
        "tag": "rnalm_off",
        "enable": False,
    },
    {
        "tag": "rnalm_on_diffusion",
        "enable": True,
    },
]

REPOS = [
    {"tag": "protenix_new", "path": REPO_NEW},
    {"tag": "protenix_pro", "path": REPO_PRO},
]

RNALM_PARAM_PATTERNS = [
    "rnalm_projection",
    "rna_projection",
    "dna_projection",
    "linear_rnalm",
    "linear_rna_llm",
    "linear_dna_llm",
    "rnalm_alpha_logit",
    "rnalm_gate_mlp",
]


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and key in dst and isinstance(dst[key], dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_for_save(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {key: clone_for_save(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clone_for_save(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(clone_for_save(value) for value in obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def clone_for_model(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, dict):
        return {key: clone_for_model(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [clone_for_model(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(clone_for_model(value) for value in obj)
    return copy.deepcopy(obj)


def flatten_tensors(obj: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    flat: dict[str, torch.Tensor] = {}
    if isinstance(obj, torch.Tensor):
        flat[prefix] = obj
        return flat
    if isinstance(obj, dict):
        for key in sorted(obj.keys()):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_tensors(obj[key], next_prefix))
        return flat
    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            next_prefix = f"{prefix}[{idx}]"
            flat.update(flatten_tensors(value, next_prefix))
    return flat


def make_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): make_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_jsonable(value) for value in obj]
    if isinstance(obj, tuple):
        return [make_jsonable(value) for value in obj]
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return {
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def tensor_tree_summary(obj: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, tensor in flatten_tensors(obj).items():
        summary[key] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
        }
    return summary


def compare_tensor_trees(lhs: Any, rhs: Any) -> dict[str, Any]:
    lhs_flat = flatten_tensors(lhs)
    rhs_flat = flatten_tensors(rhs)
    lhs_keys = set(lhs_flat.keys())
    rhs_keys = set(rhs_flat.keys())
    common_keys = sorted(lhs_keys & rhs_keys)
    max_abs_diff = 0.0
    exact_same = lhs_keys == rhs_keys
    per_key: list[dict[str, Any]] = []

    for key in common_keys:
        lt = lhs_flat[key]
        rt = rhs_flat[key]
        same_shape = tuple(lt.shape) == tuple(rt.shape)
        same_dtype = str(lt.dtype) == str(rt.dtype)
        key_exact = same_shape and same_dtype and torch.equal(lt, rt)
        if same_shape:
            if lt.numel() == 0:
                key_diff = 0.0
            elif lt.dtype == torch.bool or not (torch.is_floating_point(lt) or torch.is_floating_point(rt)):
                key_diff = float((lt.to(torch.int64) != rt.to(torch.int64)).to(torch.int64).max().item())
            else:
                key_diff = float((lt.detach().cpu().float() - rt.detach().cpu().float()).abs().max().item())
        else:
            key_diff = float("inf")
        max_abs_diff = max(max_abs_diff, key_diff if np.isfinite(key_diff) else 0.0)
        if not key_exact:
            exact_same = False
        per_key.append(
            {
                "key": key,
                "exact_same": key_exact,
                "same_shape": same_shape,
                "same_dtype": same_dtype,
                "max_abs_diff": key_diff,
            }
        )

    if lhs_keys != rhs_keys:
        exact_same = False

    return {
        "exact_same": exact_same,
        "lhs_only_keys": sorted(lhs_keys - rhs_keys),
        "rhs_only_keys": sorted(rhs_keys - lhs_keys),
        "max_abs_diff": max_abs_diff,
        "per_key": per_key,
    }


def compare_jsonable(lhs: Any, rhs: Any) -> bool:
    return json.dumps(lhs, sort_keys=True, ensure_ascii=False) == json.dumps(
        rhs, sort_keys=True, ensure_ascii=False
    )


def write_sample_index_csv(
    sample: dict[str, str],
    output_csv: Path,
) -> dict[str, Any]:
    df = pd.read_csv(INDEX_CSV)
    mask = (
        (df["pdb_id"] == sample["pdb_id"])
        & (df["chain_1_id"] == sample["chain_1_id"])
        & (df["type"] == sample["type"])
    )
    picked = df.loc[mask].copy()
    if len(picked) != 1:
        raise RuntimeError(
            f"Expected exactly one row for sample={sample}, got {len(picked)}"
        )
    picked.to_csv(output_csv, index=False)
    return make_jsonable(picked.iloc[0].to_dict())


def build_child_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--repo-tag")
    parser.add_argument("--repo-dir")
    parser.add_argument("--sample-tag")
    parser.add_argument("--sample-pdb-id")
    parser.add_argument("--sample-chain-1-id")
    parser.add_argument("--sample-type", default="chain")
    parser.add_argument("--rnalm-enabled", choices=["true", "false"])
    parser.add_argument("--output-dir")
    parser.add_argument("--benchmark-warmup", type=int, default=0)
    parser.add_argument("--benchmark-repeats", type=int, default=1)
    parser.add_argument("--benchmark-sample-tag", default="rna_157d_A")
    parser.add_argument("--benchmark-mode-tag", default="rnalm_off")
    return parser


def child_main(args: argparse.Namespace) -> None:
    repo_dir = Path(args.repo_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "payload.pt"
    summary_path = output_dir / "summary.json"

    os.environ["PROTENIX_ROOT_DIR"] = str(PROJECT_ROOT / "data")
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from configs.configs_base import configs as configs_base  # type: ignore
    from configs.configs_data import data_configs  # type: ignore
    from configs.configs_model_type import model_configs  # type: ignore
    from protenix.config.config import parse_configs  # type: ignore
    from protenix.data.pipeline.dataset import get_datasets  # type: ignore
    from protenix.model.protenix import Protenix, update_input_feature_dict  # type: ignore
    from protenix.utils.torch_utils import to_device  # type: ignore

    rnalm_enabled = args.rnalm_enabled == "true"
    sample = {
        "tag": args.sample_tag,
        "pdb_id": args.sample_pdb_id,
        "chain_1_id": args.sample_chain_1_id,
        "type": args.sample_type,
    }

    set_global_seed(20260324)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    with tempfile.TemporaryDirectory(prefix=f"rnalm_eq_{args.repo_tag}_") as tmp:
        tmpdir = Path(tmp)
        temp_index_csv = tmpdir / "sample_indices.csv"
        selected_row = write_sample_index_csv(sample, temp_index_csv)

        base = copy.deepcopy({**configs_base, "data": copy.deepcopy(data_configs)})
        deep_update(base, copy.deepcopy(model_configs[MODEL_NAME]))
        cfg = parse_configs(configs=base, arg_str="", fill_required_with_null=True)

        cfg.project = "rnalm_repo_equivalence_gpu"
        cfg.run_name = f"{args.repo_tag}_{sample['tag']}_{'on' if rnalm_enabled else 'off'}"
        cfg.base_dir = str(tmpdir / "run")
        cfg.eval_interval = 1
        cfg.log_interval = 1
        cfg.max_steps = 1
        cfg.use_wandb = False
        cfg.dtype = "fp32"
        cfg.load_strict = False
        cfg.enable_tf32 = False
        cfg.enable_efficient_fusion = False
        cfg.enable_diffusion_shared_vars_cache = False
        cfg.triangle_attention = "torch"
        cfg.triangle_multiplicative = "torch"
        cfg.diffusion_batch_size = 1
        cfg.infer_setting.chunk_size = None
        cfg.infer_setting.dynamic_chunk_size = False
        cfg.infer_setting.sample_diffusion_chunk_size = None
        cfg.data.num_dl_workers = 0
        cfg.data.epoch_size = 1
        cfg.data.train_ref_pos_augment = False
        cfg.data.test_ref_pos_augment = False
        cfg.data.train_sets = ["weightedPDB_before2109_wopb_nometalc_0925"]
        cfg.data.train_sampler.train_sample_weights = [1.0]
        cfg.data.test_sets = []

        train_name = cfg.data.train_sets[0]
        cfg.data[train_name].base_info.mmcif_dir = str(DATA_ROOT / "PDB_RNA")
        cfg.data[train_name].base_info.bioassembly_dict_dir = str(PREPARED_ROOT / "rna_bioassembly")
        cfg.data[train_name].base_info.indices_fpath = str(temp_index_csv)
        cfg.data[train_name].base_info.pdb_list = ""
        cfg.data[train_name].base_info.random_sample_if_failed = False
        cfg.data[train_name].base_info.max_n_token = -1
        cfg.data[train_name].base_info.use_reference_chains_only = False
        cfg.data[train_name].cropping_configs.crop_size = -1
        cfg.data[train_name].cropping_configs.method_weights = [0.0, 0.0, 1.0]
        cfg.data[train_name].sampler_configs.sampler_type = "uniform"

        cfg.data.msa.enable_prot_msa = False
        cfg.data.msa.enable_rna_msa = True
        cfg.data.msa.rna_msadir_raw_paths = [
            str(PREPARED_ROOT / "rna_msa/msas")
        ]
        cfg.data.msa.rna_seq_or_filename_to_msadir_jsons = [
            str(PREPARED_ROOT / "rna_msa/rna_sequence_to_pdb_chains.json")
        ]
        cfg.data.msa.rna_indexing_methods = ["sequence"]
        cfg.data.template.enable_prot_template = False

        cfg.esm.enable = False
        if "rna_loss" in cfg:
            cfg.rna_loss.enable = False
        if "two_stage" in cfg:
            cfg.two_stage.enable = False
        if "rna_template" in cfg:
            cfg.rna_template.enable = False
        if "ribonanzanet2" in cfg:
            cfg.ribonanzanet2.enable = False
        if "rna_ss" in cfg:
            cfg.rna_ss.enable = False

        cfg.rnalm.enable = rnalm_enabled
        cfg.rnalm.use_rna_embed = True
        cfg.rnalm.use_dna_embed = True
        cfg.rnalm.model_name = "aido"
        cfg.rnalm.embedding_dim = 2048
        cfg.rnalm.dna_embedding_dim = 1024
        cfg.rnalm.injection_mode = "diffusion"
        cfg.rnalm.separate_dna_projection = False
        cfg.rnalm.embedding_dir = str(DATA_ROOT / "aido_embeddings/rna")
        cfg.rnalm.sequence_fpath = str(DATA_ROOT / "aido_embeddings/rna/rna_sequences.csv")
        cfg.rnalm.dna_embedding_dir = str(DATA_ROOT / "aido_embeddings/dna")
        cfg.rnalm.dna_sequence_fpath = str(DATA_ROOT / "aido_embeddings/dna/dna_sequences.csv")

        cfg.model.N_cycle = 1
        cfg.model.N_model_seed = 1
        cfg.model.template_embedder.n_blocks = 0
        cfg.model.msa_module.n_blocks = 1
        cfg.model.pairformer.n_blocks = 2
        cfg.model.diffusion_module.atom_encoder.n_blocks = 1
        cfg.model.diffusion_module.transformer.n_blocks = 2
        cfg.model.diffusion_module.atom_decoder.n_blocks = 1
        cfg.model.confidence_head.n_blocks = 1
        cfg.sample_diffusion.N_step = 2
        cfg.sample_diffusion.N_sample = 1
        cfg.sample_diffusion.N_step_mini_rollout = 1
        cfg.sample_diffusion.N_sample_mini_rollout = 1

        checkpoint_path = repo_dir / "checkpoints" / CHECKPOINT_NAME
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("CUDA is required for this comparison.")
        torch.cuda.set_device(device)

        timings: dict[str, float] = {}

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        train_dataset, _ = get_datasets(cfg, error_dir=str(tmpdir / "errors"))
        torch.cuda.synchronize(device)
        timings["dataset_init_s"] = time.perf_counter() - t0

        ds = train_dataset.datasets[0]
        if len(ds) != 1:
            raise RuntimeError(f"Expected single-row dataset, got len={len(ds)}")

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        data = ds.process_one(0, return_atom_token_array=False)
        torch.cuda.synchronize(device)
        timings["dataset_item_load_s"] = time.perf_counter() - t0

        raw_input = clone_for_model(data["input_feature_dict"])
        raw_label = clone_for_model(data["label_dict"])
        raw_label_full = clone_for_model(data["label_full_dict"])

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        model = Protenix(cfg).to(device)
        model.eval()
        torch.cuda.synchronize(device)
        timings["model_init_s"] = time.perf_counter() - t0

        rnalm_params_pre = {}
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in RNALM_PARAM_PATTERNS):
                param_cpu = param.detach().cpu().float()
                rnalm_params_pre[name] = {
                    "shape": list(param_cpu.shape),
                    "sum": float(param_cpu.sum().item()),
                    "absmax": float(param_cpu.abs().max().item()) if param_cpu.numel() else 0.0,
                }

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"]
        sample_key = next(iter(state_dict.keys()))
        if sample_key.startswith("module."):
            state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        load_result = model.load_state_dict(state_dict=state_dict, strict=False)
        torch.cuda.synchronize(device)
        timings["checkpoint_load_s"] = time.perf_counter() - t0

        rnalm_params_post = {}
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in RNALM_PARAM_PATTERNS):
                param_cpu = param.detach().cpu().float()
                rnalm_params_post[name] = {
                    "shape": list(param_cpu.shape),
                    "sum": float(param_cpu.sum().item()),
                    "absmax": float(param_cpu.abs().max().item()) if param_cpu.numel() else 0.0,
                }

        prepared_input = clone_for_model(raw_input)
        prepared_input = model.relative_position_encoding.generate_relp(prepared_input)
        prepared_input = update_input_feature_dict(prepared_input)

        pair_input = to_device(clone_for_model(prepared_input), device)
        torch.cuda.reset_peak_memory_stats(device)
        set_global_seed(20260324)
        with torch.no_grad():
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            s_inputs, s_trunk, z_trunk = model.get_pairformer_output(
                input_feature_dict=pair_input,
                N_cycle=cfg.model.N_cycle,
                inplace_safe=True,
                chunk_size=cfg.infer_setting.chunk_size,
                mc_dropout=False,
                mc_dropout_rate=0.0,
            )
            s_rnalm = model._get_s_rnalm(
                input_feature_dict=pair_input,
                N_token=int(pair_input["residue_index"].shape[-1]),
                s_trunk=s_trunk,
            )
            torch.cuda.synchronize(device)
            timings["pairformer_s"] = time.perf_counter() - t0
        peak_pairformer_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        pred_input = to_device(clone_for_model(raw_input), device)
        pred_label = to_device(clone_for_model(raw_label), device)
        pred_label_full = to_device(clone_for_model(raw_label_full), device)
        torch.cuda.reset_peak_memory_stats(device)
        set_global_seed(20260324)
        with torch.no_grad():
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            prediction, _, log_dict = model(
                input_feature_dict=pred_input,
                label_full_dict=pred_label_full,
                label_dict=pred_label,
                mode="inference",
                mc_dropout_apply_rate=0.0,
            )
            torch.cuda.synchronize(device)
            timings["forward_single_s"] = time.perf_counter() - t0
        peak_forward_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

        perf_warmup = max(0, int(args.benchmark_warmup))
        perf_repeats = max(1, int(args.benchmark_repeats))

        for rep in range(perf_warmup):
            perf_input = to_device(clone_for_model(raw_input), device)
            perf_label = to_device(clone_for_model(raw_label), device)
            perf_label_full = to_device(clone_for_model(raw_label_full), device)
            set_global_seed(303000 + rep)
            with torch.no_grad():
                _ = model(
                    input_feature_dict=perf_input,
                    label_full_dict=perf_label_full,
                    label_dict=perf_label,
                    mode="inference",
                    mc_dropout_apply_rate=0.0,
                )
                torch.cuda.synchronize(device)

        perf_runs: list[float] = []
        for rep in range(perf_repeats):
            perf_input = to_device(clone_for_model(raw_input), device)
            perf_label = to_device(clone_for_model(raw_label), device)
            perf_label_full = to_device(clone_for_model(raw_label_full), device)
            set_global_seed(404000 + rep)
            with torch.no_grad():
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                _ = model(
                    input_feature_dict=perf_input,
                    label_full_dict=perf_label_full,
                    label_dict=perf_label,
                    mode="inference",
                    mc_dropout_apply_rate=0.0,
                )
                torch.cuda.synchronize(device)
                perf_runs.append(time.perf_counter() - t0)
        perf_arr = np.asarray(perf_runs, dtype=np.float64)
        timings["forward_avg_s"] = float(perf_arr.mean())
        timings["forward_std_s"] = float(perf_arr.std())
        timings["forward_median_s"] = float(np.median(perf_arr))
        timings["forward_min_s"] = float(perf_arr.min())
        timings["forward_max_s"] = float(perf_arr.max())
        timings["forward_p05_s"] = float(np.percentile(perf_arr, 5))
        timings["forward_p95_s"] = float(np.percentile(perf_arr, 95))

        payload = {
            "raw_input_feature_dict": clone_for_save(raw_input),
            "prepared_input_feature_dict": clone_for_save(prepared_input),
            "label_dict": clone_for_save(raw_label),
            "label_full_dict": clone_for_save(raw_label_full),
            "pairformer": {
                "s_inputs": clone_for_save(s_inputs),
                "s_trunk": clone_for_save(s_trunk),
                "z_trunk": clone_for_save(z_trunk),
                "s_rnalm": clone_for_save(s_rnalm) if s_rnalm is not None else None,
            },
            "prediction": {
                "coordinate": clone_for_save(prediction["coordinate"]),
                "plddt": clone_for_save(prediction["plddt"]),
                "pae": clone_for_save(prediction["pae"]),
                "pde": clone_for_save(prediction["pde"]),
                "resolved": clone_for_save(prediction["resolved"]),
                "contact_probs": clone_for_save(prediction["contact_probs"]),
            },
        }
        torch.save(payload, payload_path)

        summary = {
            "repo_tag": args.repo_tag,
            "repo_dir": str(repo_dir),
            "sample": sample,
            "rnalm_enabled": rnalm_enabled,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "python_executable": sys.executable,
            "checkpoint_path": str(checkpoint_path),
            "selected_row": selected_row,
            "timings": timings,
            "benchmark": {
                "warmup_runs": perf_warmup,
                "repeats": perf_repeats,
                "forward_runs_s": perf_runs,
            },
            "memory": {
                "pairformer_peak_mb": peak_pairformer_mb,
                "forward_peak_mb": peak_forward_mb,
            },
            "basic": make_jsonable(data["basic"]),
            "tensor_summary": {
                "raw_input_feature_dict": tensor_tree_summary(raw_input),
                "prepared_input_feature_dict": tensor_tree_summary(prepared_input),
                "label_dict": tensor_tree_summary(raw_label),
                "label_full_dict": tensor_tree_summary(raw_label_full),
                "pairformer": tensor_tree_summary(
                    {
                        "s_inputs": s_inputs,
                        "s_trunk": s_trunk,
                        "z_trunk": z_trunk,
                        "s_rnalm": s_rnalm if s_rnalm is not None else torch.empty(0),
                    }
                ),
                "prediction": tensor_tree_summary(payload["prediction"]),
            },
            "checkpoint_load": {
                "missing_keys": sorted(load_result.missing_keys),
                "unexpected_keys_count": len(load_result.unexpected_keys),
                "unexpected_keys_head": sorted(load_result.unexpected_keys)[:20],
                "unexpected_keys_sha256": hashlib.sha256(
                    "\n".join(sorted(load_result.unexpected_keys)).encode("utf-8")
                ).hexdigest(),
            },
            "rnalm_param_stats_pre": rnalm_params_pre,
            "rnalm_param_stats_post": rnalm_params_post,
            "log_dict": make_jsonable(log_dict),
            "payload_path": str(payload_path),
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def run_child_case(
    python_bin: str,
    repo_tag: str,
    repo_dir: Path,
    sample: dict[str, str],
    mode: dict[str, Any],
    output_root: Path,
    benchmark_warmup: int = 0,
    benchmark_repeats: int = 1,
) -> dict[str, Any]:
    case_dir = output_root / repo_tag / sample["tag"] / mode["tag"]
    case_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        str(Path(__file__).resolve()),
        "--child",
        "--repo-tag",
        repo_tag,
        "--repo-dir",
        str(repo_dir),
        "--sample-tag",
        sample["tag"],
        "--sample-pdb-id",
        sample["pdb_id"],
        "--sample-chain-1-id",
        sample["chain_1_id"],
        "--sample-type",
        sample["type"],
        "--rnalm-enabled",
        "true" if mode["enable"] else "false",
        "--output-dir",
        str(case_dir),
        "--benchmark-warmup",
        str(benchmark_warmup),
        "--benchmark-repeats",
        str(benchmark_repeats),
    ]
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        cwd=str(REPO_PRO),
    )
    wall_s = time.perf_counter() - start
    (case_dir / "stdout.log").write_text(proc.stdout, encoding="utf-8")
    (case_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Child case failed: repo={repo_tag} sample={sample['tag']} mode={mode['tag']}\n"
            f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
        )
    summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
    summary["child_wall_s"] = wall_s
    return summary


def load_payload(summary: dict[str, Any]) -> dict[str, Any]:
    return torch.load(summary["payload_path"], map_location="cpu", weights_only=False)


def compare_repo_pair(
    lhs_summary: dict[str, Any],
    rhs_summary: dict[str, Any],
) -> dict[str, Any]:
    lhs_payload = load_payload(lhs_summary)
    rhs_payload = load_payload(rhs_summary)
    return {
        "raw_input_feature_dict": compare_tensor_trees(
            lhs_payload["raw_input_feature_dict"], rhs_payload["raw_input_feature_dict"]
        ),
        "prepared_input_feature_dict": compare_tensor_trees(
            lhs_payload["prepared_input_feature_dict"],
            rhs_payload["prepared_input_feature_dict"],
        ),
        "label_dict": compare_tensor_trees(
            lhs_payload["label_dict"], rhs_payload["label_dict"]
        ),
        "label_full_dict": compare_tensor_trees(
            lhs_payload["label_full_dict"], rhs_payload["label_full_dict"]
        ),
        "pairformer": compare_tensor_trees(
            lhs_payload["pairformer"], rhs_payload["pairformer"]
        ),
        "prediction": compare_tensor_trees(
            lhs_payload["prediction"], rhs_payload["prediction"]
        ),
        "basic_exact_same": compare_jsonable(lhs_summary["basic"], rhs_summary["basic"]),
        "selected_row_exact_same": compare_jsonable(
            lhs_summary["selected_row"], rhs_summary["selected_row"]
        ),
        "checkpoint_missing_keys_exact_same": lhs_summary["checkpoint_load"]["missing_keys"]
        == rhs_summary["checkpoint_load"]["missing_keys"],
        "checkpoint_unexpected_keys_exact_same": (
            lhs_summary["checkpoint_load"]["unexpected_keys_count"]
            == rhs_summary["checkpoint_load"]["unexpected_keys_count"]
            and lhs_summary["checkpoint_load"]["unexpected_keys_sha256"]
            == rhs_summary["checkpoint_load"]["unexpected_keys_sha256"]
        ),
        "timings": {
            "lhs": lhs_summary["timings"],
            "rhs": rhs_summary["timings"],
        },
        "memory": {
            "lhs": lhs_summary["memory"],
            "rhs": rhs_summary["memory"],
        },
    }


def compare_within_repo(
    off_summary: dict[str, Any],
    on_summary: dict[str, Any],
) -> dict[str, Any]:
    off_payload = load_payload(off_summary)
    on_payload = load_payload(on_summary)
    raw_cmp = compare_tensor_trees(
        off_payload["raw_input_feature_dict"], on_payload["raw_input_feature_dict"]
    )
    prepared_cmp = compare_tensor_trees(
        off_payload["prepared_input_feature_dict"], on_payload["prepared_input_feature_dict"]
    )
    pairformer_cmp = compare_tensor_trees(
        off_payload["pairformer"], on_payload["pairformer"]
    )
    prediction_cmp = compare_tensor_trees(
        off_payload["prediction"], on_payload["prediction"]
    )
    return {
        "raw_input_feature_dict": raw_cmp,
        "prepared_input_feature_dict": prepared_cmp,
        "pairformer": pairformer_cmp,
        "prediction": prediction_cmp,
        "checkpoint_missing_keys_off": off_summary["checkpoint_load"]["missing_keys"],
        "checkpoint_missing_keys_on": on_summary["checkpoint_load"]["missing_keys"],
        "timings_off": off_summary["timings"],
        "timings_on": on_summary["timings"],
        "memory_off": off_summary["memory"],
        "memory_on": on_summary["memory"],
    }


def make_report(
    case_summaries: dict[str, dict[str, dict[str, dict[str, Any]]]],
    cross_repo: dict[str, dict[str, Any]],
    within_repo: dict[str, dict[str, dict[str, Any]]],
) -> str:
    lines: list[str] = []
    lines.append("# RNALM Repo Equivalence GPU Report")
    lines.append("")
    lines.append(f"- Date: 2026-03-24")
    lines.append(f"- GPU: {next(iter(next(iter(case_summaries.values())).values()))['protenix_new']['gpu_name']}")
    lines.append(f"- Python: {sys.executable}")
    lines.append(f"- Data root: `{DATA_ROOT}`")
    lines.append(f"- Model checkpoint: `{CHECKPOINT_NAME}`")
    lines.append("- Execution mode: GPU, fp32, `triangle_attention=torch`, `triangle_multiplicative=torch`, `N_cycle=1`, `N_step=2`, `N_sample=1`, crop disabled")
    lines.append("- Minimal-instance block counts: template=0, msa=1, pairformer=2, diffusion(atom/transformer/decoder)=1/2/1, confidence=1")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    for sample in SAMPLES:
        for mode in MODES:
            key = f"{sample['tag']}::{mode['tag']}"
            cmp_result = cross_repo[key]
            lines.append(f"### {sample['tag']} / {mode['tag']}")
            lines.append(
                f"- raw data equal: {cmp_result['raw_input_feature_dict']['exact_same']}"
            )
            lines.append(
                f"- prepared model input equal: {cmp_result['prepared_input_feature_dict']['exact_same']}"
            )
            lines.append(f"- pairformer tensors equal: {cmp_result['pairformer']['exact_same']}")
            lines.append(f"- final prediction equal: {cmp_result['prediction']['exact_same']}")
            lines.append(
                f"- checkpoint missing keys equal: {cmp_result['checkpoint_missing_keys_exact_same']}"
            )
            lines.append(
                f"- basic metadata equal: {cmp_result['basic_exact_same']}"
            )
            lines.append(
                f"- protenix_new forward_avg_s: {cmp_result['timings']['lhs']['forward_avg_s']:.6f}"
            )
            lines.append(
                f"- protenix_pro forward_avg_s: {cmp_result['timings']['rhs']['forward_avg_s']:.6f}"
            )
            lines.append("")
    lines.append("## RNALM On/Off Inside Each Repo")
    lines.append("")
    for sample in SAMPLES:
        sample_key = sample["tag"]
        lines.append(f"### {sample_key}")
        for repo in REPOS:
            cmp_result = within_repo[sample_key][repo["tag"]]
            lines.append(f"- {repo['tag']} raw input exact same: {cmp_result['raw_input_feature_dict']['exact_same']}")
            lines.append(f"- {repo['tag']} pairformer exact same: {cmp_result['pairformer']['exact_same']}")
            lines.append(f"- {repo['tag']} prediction exact same: {cmp_result['prediction']['exact_same']}")
            lines.append(f"- {repo['tag']} off missing keys: {cmp_result['checkpoint_missing_keys_off']}")
            lines.append(f"- {repo['tag']} on missing keys: {cmp_result['checkpoint_missing_keys_on']}")
        lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- If `rnalm_on_diffusion` reports extra missing keys, those are the RNALM projection parameters absent from the base checkpoint and left at their zero initialization.")
    lines.append("- Performance numbers are minimal-instance timings, useful for relative comparison only.")
    lines.append("")
    return "\n".join(lines)


def find_sample(sample_tag: str) -> dict[str, str]:
    for sample in SAMPLES:
        if sample["tag"] == sample_tag:
            return sample
    raise KeyError(f"Unknown sample tag: {sample_tag}")


def find_mode(mode_tag: str) -> dict[str, Any]:
    for mode in MODES:
        if mode["tag"] == mode_tag:
            return mode
    raise KeyError(f"Unknown mode tag: {mode_tag}")


def make_benchmark_report(
    sample: dict[str, str],
    mode: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
) -> str:
    new_timings = summaries["protenix_new"]["timings"]
    pro_timings = summaries["protenix_pro"]["timings"]
    median_delta_s = pro_timings["forward_median_s"] - new_timings["forward_median_s"]
    median_delta_pct = (
        (median_delta_s / new_timings["forward_median_s"]) * 100.0
        if new_timings["forward_median_s"] != 0
        else 0.0
    )

    lines: list[str] = []
    lines.append("# RNALM Fixed-Case Benchmark")
    lines.append("")
    lines.append(f"- Date: 2026-03-24")
    lines.append(f"- Sample: `{sample['tag']}`")
    lines.append(f"- RNALM mode: `{mode['tag']}`")
    lines.append(f"- Data root: `{DATA_ROOT}`")
    lines.append(f"- GPU: {next(iter(summaries.values()))['gpu_name']}")
    lines.append(f"- Warmup runs: {next(iter(summaries.values()))['benchmark']['warmup_runs']}")
    lines.append(f"- Timed runs: {next(iter(summaries.values()))['benchmark']['repeats']}")
    lines.append("")
    lines.append("## Timing Table")
    lines.append("")
    lines.append("| Repo | dataset_item_load_s | pairformer_s | forward_mean_s | forward_median_s | forward_std_s | forward_p05_s | forward_p95_s | forward_min_s | forward_max_s |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for repo in REPOS:
        summary = summaries[repo["tag"]]
        timings = summary["timings"]
        lines.append(
            f"| {repo['tag']} | "
            f"{timings['dataset_item_load_s']:.6f} | "
            f"{timings['pairformer_s']:.6f} | "
            f"{timings['forward_avg_s']:.6f} | "
            f"{timings['forward_median_s']:.6f} | "
            f"{timings['forward_std_s']:.6f} | "
            f"{timings['forward_p05_s']:.6f} | "
            f"{timings['forward_p95_s']:.6f} | "
            f"{timings['forward_min_s']:.6f} | "
            f"{timings['forward_max_s']:.6f} |"
        )
    lines.append("")
    lines.append("## Repo Delta")
    lines.append("")
    lines.append(f"- forward median delta (`protenix_pro - protenix_new`): {median_delta_s:.6f} s")
    lines.append(f"- forward median delta percent vs `protenix_new`: {median_delta_pct:.2f}%")
    lines.append("")
    lines.append("## Numerical Check")
    lines.append("")
    lines.append(f"- raw data equal: {comparison['raw_input_feature_dict']['exact_same']}")
    lines.append(f"- prepared model input equal: {comparison['prepared_input_feature_dict']['exact_same']}")
    lines.append(f"- pairformer exact same: {comparison['pairformer']['exact_same']}")
    lines.append(f"- pairformer max abs diff: {comparison['pairformer']['max_abs_diff']}")
    lines.append(f"- prediction exact same: {comparison['prediction']['exact_same']}")
    lines.append(f"- prediction max abs diff: {comparison['prediction']['max_abs_diff']}")
    lines.append("")
    lines.append("## Note")
    lines.append("")
    lines.append("- This benchmark fixes one case and repeats the same GPU forward path many times to reduce one-shot timing noise.")
    return "\n".join(lines)


def benchmark_main(
    sample_tag: str,
    mode_tag: str,
    benchmark_warmup: int,
    benchmark_repeats: int,
) -> None:
    sample = find_sample(sample_tag)
    mode = find_mode(mode_tag)
    bench_root = OUTPUT_ROOT / f"benchmark_{sample_tag}_{mode_tag}_r{benchmark_repeats}"
    bench_root.mkdir(parents=True, exist_ok=True)
    python_bin = sys.executable

    summaries: dict[str, dict[str, Any]] = {}
    for repo in REPOS:
        print(
            f"[BENCH] repo={repo['tag']} sample={sample_tag} mode={mode_tag} warmup={benchmark_warmup} repeats={benchmark_repeats}",
            flush=True,
        )
        summaries[repo["tag"]] = run_child_case(
            python_bin=python_bin,
            repo_tag=repo["tag"],
            repo_dir=repo["path"],
            sample=sample,
            mode=mode,
            output_root=bench_root,
            benchmark_warmup=benchmark_warmup,
            benchmark_repeats=benchmark_repeats,
        )

    comparison = compare_repo_pair(
        summaries["protenix_new"],
        summaries["protenix_pro"],
    )
    report = {
        "metadata": {
            "date": "2026-03-24",
            "sample": sample,
            "mode": mode,
            "benchmark_warmup": benchmark_warmup,
            "benchmark_repeats": benchmark_repeats,
        },
        "summaries": summaries,
        "comparison": comparison,
    }
    json_path = bench_root / f"rnalm_benchmark_{sample_tag}_{mode_tag}_r{benchmark_repeats}.json"
    md_path = bench_root / f"RNALM_BENCHMARK_{sample_tag}_{mode_tag}_r{benchmark_repeats}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(
        make_benchmark_report(sample, mode, summaries, comparison),
        encoding="utf-8",
    )
    print(f"[DONE] Benchmark JSON: {json_path}")
    print(f"[DONE] Benchmark Markdown: {md_path}")


def master_main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    python_bin = sys.executable

    case_summaries: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for sample in SAMPLES:
        sample_key = sample["tag"]
        case_summaries[sample_key] = {}
        for mode in MODES:
            mode_key = mode["tag"]
            case_summaries[sample_key][mode_key] = {}
            for repo in REPOS:
                print(
                    f"[RUN] repo={repo['tag']} sample={sample_key} mode={mode_key}",
                    flush=True,
                )
                summary = run_child_case(
                    python_bin=python_bin,
                    repo_tag=repo["tag"],
                    repo_dir=repo["path"],
                    sample=sample,
                    mode=mode,
                    output_root=OUTPUT_ROOT,
                )
                case_summaries[sample_key][mode_key][repo["tag"]] = summary

    cross_repo: dict[str, dict[str, Any]] = {}
    for sample in SAMPLES:
        sample_key = sample["tag"]
        for mode in MODES:
            mode_key = mode["tag"]
            key = f"{sample_key}::{mode_key}"
            cross_repo[key] = compare_repo_pair(
                case_summaries[sample_key][mode_key]["protenix_new"],
                case_summaries[sample_key][mode_key]["protenix_pro"],
            )

    within_repo: dict[str, dict[str, dict[str, Any]]] = {}
    for sample in SAMPLES:
        sample_key = sample["tag"]
        within_repo[sample_key] = {}
        for repo in REPOS:
            within_repo[sample_key][repo["tag"]] = compare_within_repo(
                case_summaries[sample_key]["rnalm_off"][repo["tag"]],
                case_summaries[sample_key]["rnalm_on_diffusion"][repo["tag"]],
            )

    report = {
        "metadata": {
            "date": "2026-03-24",
            "project_root": str(PROJECT_ROOT),
            "data_root": str(DATA_ROOT),
            "repos": {repo["tag"]: str(repo["path"]) for repo in REPOS},
            "model_name": MODEL_NAME,
            "checkpoint_name": CHECKPOINT_NAME,
            "python_executable": python_bin,
        },
        "case_summaries": case_summaries,
        "cross_repo": cross_repo,
        "within_repo": within_repo,
    }

    json_path = OUTPUT_ROOT / "rnalm_repo_equivalence_gpu_report_20260324.json"
    md_path = OUTPUT_ROOT / "RNALM_EQUIVALENCE_GPU_REPORT_20260324.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(
        make_report(case_summaries, cross_repo, within_repo),
        encoding="utf-8",
    )
    print(f"[DONE] JSON report: {json_path}")
    print(f"[DONE] Markdown report: {md_path}")


def main() -> None:
    parser = build_child_parser()
    args = parser.parse_args()
    if args.child:
        child_main(args)
    elif args.benchmark:
        benchmark_main(
            sample_tag=args.benchmark_sample_tag,
            mode_tag=args.benchmark_mode_tag,
            benchmark_warmup=args.benchmark_warmup,
            benchmark_repeats=args.benchmark_repeats,
        )
    else:
        master_main()


if __name__ == "__main__":
    main()
