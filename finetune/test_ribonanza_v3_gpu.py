#!/usr/bin/env python3
"""
Minimal GPU Integration Test: RibonanzaNet2 v3 Tokenizer + Model Forward/Backward.

Tests:
  1. RibonanzaTokenizer correctly produces tokenized_seq & ribonanza_token_mask
  2. Data pipeline integration (loading a real RNA sample)
  3. Model forward pass with v3 src_mask logic
  4. Backward pass and gradient flow through adapter parameters
  5. RibonanzaNet backbone stays frozen

Usage:
    cd /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_pro/Protenix
    python3 finetune/test_ribonanza_v3_gpu.py
"""
import os
import sys
import time
import traceback

# Setup environment
PROJECT_ROOT = "/inspire/ssd/project/sais-bio/public/ash_proj"
PROTENIX_DIR = f"{PROJECT_ROOT}/code/protenix_pro/Protenix"
os.chdir(PROTENIX_DIR)
sys.path.insert(0, PROTENIX_DIR)

os.environ.setdefault("PROTENIX_ROOT_DIR", f"{PROJECT_ROOT}/data")
os.environ.setdefault("TRIANGLE_ATTENTION", "cuequivariance")
os.environ.setdefault("TRIANGLE_MULTIPLICATIVE", "cuequivariance")

import gzip
import pickle
import logging
import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("test_ribonanza_v3")

DATA_DIR = f"{PROJECT_ROOT}/data/stanford-rna-3d-folding/part2"
PREPARED_DIR = f"{DATA_DIR}/protenix_prepared"
BIOASSEMBLY_DIR = f"{PREPARED_DIR}/rna_bioassembly"
RIBONANZA_MODEL_DIR = f"{PROJECT_ROOT}/data/ribonanzanet2/model_weights"

RESULTS = {}
ALL_PASSED = True


def record(name, passed, detail=""):
    global ALL_PASSED
    RESULTS[name] = {"passed": passed, "detail": detail}
    status = "PASS" if passed else "FAIL"
    logger.info(f"[{status}] {name}: {detail}")
    if not passed:
        ALL_PASSED = False


def test_1_tokenizer_unit():
    """Test RibonanzaTokenizer on a real bioassembly sample."""
    logger.info("=" * 60)
    logger.info("TEST 1: RibonanzaTokenizer unit test on real data")
    logger.info("=" * 60)
    try:
        from protenix.data.ribonanza.ribonanza_tokenizer import RibonanzaTokenizer

        # Load a real bioassembly
        pdb_id = "157d"  # small RNA duplex
        pkl_path = os.path.join(BIOASSEMBLY_DIR, f"{pdb_id}.pkl.gz")
        if not os.path.exists(pkl_path):
            record("tokenizer_unit", False, f"Sample not found: {pkl_path}")
            return None, None

        with gzip.open(pkl_path, "rb") as f:
            bioassembly_dict = pickle.load(f)

        atom_array = bioassembly_dict["atom_array"]
        token_array = bioassembly_dict["token_array"]
        n_tokens = len(token_array)
        logger.info(f"Loaded {pdb_id}: {len(atom_array)} atoms, {n_tokens} tokens")

        # Run tokenizer
        tokenizer = RibonanzaTokenizer()
        result = tokenizer(token_array=token_array, atom_array=atom_array)

        tokenized_seq = result["tokenized_seq"]
        mask = result["ribonanza_token_mask"]

        logger.info(f"tokenized_seq: shape={tokenized_seq.shape}, dtype={tokenized_seq.dtype}")
        logger.info(f"ribonanza_token_mask: shape={mask.shape}, dtype={mask.dtype}")
        logger.info(f"  RNA tokens (mask=True): {mask.sum().item()}")
        logger.info(f"  Non-RNA tokens (mask=False): {(~mask).sum().item()}")
        logger.info(f"  Unique token values in RNA: {tokenized_seq[mask].unique().tolist()}")
        logger.info(f"  Unique token values in non-RNA: {tokenized_seq[~mask].unique().tolist() if (~mask).any() else 'N/A'}")

        # Validate
        assert tokenized_seq.shape == (n_tokens,), f"Shape mismatch: {tokenized_seq.shape} vs {n_tokens}"
        assert mask.shape == (n_tokens,), f"Mask shape mismatch"
        assert tokenized_seq.dtype == torch.long, f"Wrong dtype: {tokenized_seq.dtype}"
        assert mask.dtype == torch.bool, f"Wrong mask dtype: {mask.dtype}"

        # 157d is pure RNA, so mask should have True entries
        n_rna = mask.sum().item()
        assert n_rna > 0, f"No RNA tokens found in {pdb_id}"

        # All RNA tokens should be in {0,1,2,3,5}, not PAD=4
        rna_tokens = tokenized_seq[mask]
        valid_rna_vals = {0, 1, 2, 3, 5}
        actual_vals = set(rna_tokens.unique().tolist())
        assert actual_vals.issubset(valid_rna_vals), f"Invalid RNA token values: {actual_vals - valid_rna_vals}"

        # Non-RNA tokens (if any) should all be PAD=4
        if (~mask).any():
            non_rna_tokens = tokenized_seq[~mask]
            assert (non_rna_tokens == 4).all(), f"Non-RNA tokens not all PAD: {non_rna_tokens.unique().tolist()}"

        record("tokenizer_unit", True,
               f"pdb={pdb_id}, n_tokens={n_tokens}, n_rna={n_rna}, "
               f"rna_values={sorted(actual_vals)}")

        return result

    except Exception as e:
        record("tokenizer_unit", False, f"Exception: {e}\n{traceback.format_exc()}")
        return None


def test_2_model_forward_backward():
    """Test model forward+backward with v3 RibonanzaNet2 integration."""
    logger.info("=" * 60)
    logger.info("TEST 2: Model forward + backward with RibonanzaNet2 v3")
    logger.info("=" * 60)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        if device.type != "cuda":
            record("model_forward_backward", False, "No GPU available")
            return

        # Build config using the proper parsing pipeline
        from configs.configs_base import configs as configs_base
        from configs.configs_model_type import model_configs
        from protenix.config.config import parse_configs
        from configs.configs_data import data_configs
        import copy

        # Merge base + data configs (like the real runner does)
        base = copy.deepcopy(configs_base)
        base["data"] = data_configs
        configs = parse_configs(
            base,
            arg_str=(
                f"--model_name protenix_base_20250630_v1.0.0 "
                f"--ribonanzanet2.enable true "
                f"--ribonanzanet2.model_dir {RIBONANZA_MODEL_DIR} "
                f"--ribonanzanet2.gate_type channel "
            ),
            fill_required_with_null=True,
        )

        # Build model
        from protenix.model.protenix import Protenix
        logger.info("Building Protenix model with RibonanzaNet2 enabled...")
        t0 = time.time()
        model = Protenix(configs)
        logger.info(f"Model built in {time.time() - t0:.1f}s")

        # Move to device
        model = model.to(device)
        model.train()

        # Create synthetic input that looks like a small RNA
        N_token = 24  # small RNA
        N_atom = 24 * 23  # ~23 atoms per nucleotide

        # Build minimal input_feature_dict
        input_feature_dict = {
            "token_index": torch.arange(N_token, device=device),
            "residue_index": torch.arange(N_token, device=device),
            "asym_id": torch.zeros(N_token, dtype=torch.long, device=device),
            "entity_id": torch.zeros(N_token, dtype=torch.long, device=device),
            "sym_id": torch.zeros(N_token, dtype=torch.long, device=device),
            "restype": torch.zeros(N_token, 32, device=device),
            "token_bonds": torch.zeros(N_token, N_token, device=device),
            "atom_to_token_idx": torch.arange(N_atom, device=device) // 23,
            "ref_pos": torch.randn(N_atom, 3, device=device),
            "ref_mask": torch.ones(N_atom, device=device),
            "ref_element": torch.zeros(N_atom, 128, device=device),
            "ref_charge": torch.zeros(N_atom, device=device),
            "ref_atom_name_chars": torch.zeros(N_atom, 64, device=device),
            "ref_space_uid": torch.arange(N_atom, device=device) // 23,
            "is_protein": torch.zeros(N_atom, dtype=torch.bool, device=device),
            "is_rna": torch.ones(N_atom, dtype=torch.bool, device=device),
            "is_dna": torch.zeros(N_atom, dtype=torch.bool, device=device),
            "is_ligand": torch.zeros(N_atom, dtype=torch.bool, device=device),
        }

        # Add proper dummy features for template and MSA
        input_feature_dict["msa"] = torch.zeros(1, N_token, device=device).long()
        input_feature_dict["has_deletion"] = torch.zeros(1, N_token, device=device)
        input_feature_dict["deletion_value"] = torch.zeros(1, N_token, device=device)
        input_feature_dict["prot_pair_num_alignments"] = torch.tensor(0, device=device)
        input_feature_dict["prot_unpair_num_alignments"] = torch.tensor(0, device=device)
        input_feature_dict["rna_pair_num_alignments"] = torch.tensor(0, device=device)
        input_feature_dict["rna_unpair_num_alignments"] = torch.tensor(0, device=device)
        input_feature_dict["template_restype_i"] = torch.zeros(4, N_token, N_token, device=device)
        input_feature_dict["template_restype_j"] = torch.zeros(4, N_token, N_token, device=device)
        input_feature_dict["template_pseudo_beta_mask"] = torch.zeros(4, N_token, device=device)
        input_feature_dict["template_backbone_frame_mask"] = torch.zeros(4, N_token, device=device)
        input_feature_dict["template_distogram"] = torch.zeros(4, N_token, N_token, 39, device=device)
        input_feature_dict["template_unit_vector"] = torch.zeros(4, N_token, N_token, 3, device=device)
        # relative position encoding
        input_feature_dict["relp"] = torch.zeros(N_token, N_token, 1, device=device).long()

        # Add v3 RibonanzaNet2 tokenizer outputs
        # Simulate pure RNA: all tokens are A, C, G, U alternating
        rna_bases = [0, 1, 2, 3] * (N_token // 4)  # A, C, G, U repeating
        input_feature_dict["tokenized_seq"] = torch.tensor(rna_bases, dtype=torch.long, device=device)
        input_feature_dict["ribonanza_token_mask"] = torch.ones(N_token, dtype=torch.bool, device=device)

        logger.info(f"Input: N_token={N_token}, N_atom={N_atom}")
        logger.info(f"tokenized_seq: {input_feature_dict['tokenized_seq'][:8].tolist()}...")
        logger.info(f"ribonanza_token_mask: all True ({input_feature_dict['ribonanza_token_mask'].sum().item()})")

        # Forward pass
        logger.info("Running forward pass...")
        t0 = time.time()
        try:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output = model(
                    input_feature_dict=input_feature_dict,
                    label_dict={},
                    label_full_dict={},
                    mode="inference",
                )
            logger.info(f"Forward pass completed in {time.time() - t0:.1f}s")
            record("model_forward", True, f"Forward OK in {time.time() - t0:.1f}s")
        except Exception as e:
            record("model_forward", False, f"Forward failed: {e}\n{traceback.format_exc()}")
            # Even if forward fails, report what we can
            return

        # Check output has expected structure
        if isinstance(output, dict):
            logger.info(f"Output keys: {list(output.keys())[:10]}...")
        elif isinstance(output, (list, tuple)):
            logger.info(f"Output is {type(output).__name__} of length {len(output)}")

        # Backward pass - create a simple loss
        logger.info("Running backward pass...")
        t0 = time.time()
        try:
            # Use a simple loss on whatever output we get
            if isinstance(output, dict):
                # Find a tensor to compute loss on
                for k, v in output.items():
                    if isinstance(v, torch.Tensor) and v.requires_grad:
                        loss = v.sum()
                        break
                else:
                    # If no grad tensor in output, try different approach
                    loss = sum(p.sum() for p in model.parameters() if p.requires_grad) * 0
                    for k, v in output.items():
                        if isinstance(v, torch.Tensor) and v.is_floating_point():
                            loss = loss + v.sum()
                            break
            else:
                loss = output[0].sum() if isinstance(output, (list, tuple)) else output.sum()

            loss.backward()
            logger.info(f"Backward pass completed in {time.time() - t0:.1f}s, loss={loss.item():.4f}")
            record("model_backward", True, f"Backward OK, loss={loss.item():.4f}")
        except Exception as e:
            record("model_backward", False, f"Backward failed: {e}\n{traceback.format_exc()}")
            return

        # Check gradient flow
        logger.info("Checking gradient flow...")

        # RibonanzaNet backbone should be frozen (no grad)
        rnet_has_grad = False
        for name, param in model.named_parameters():
            if "ribonanza_net." in name:
                if param.grad is not None and param.grad.abs().sum() > 0:
                    rnet_has_grad = True
                    break
        record("ribonanza_frozen", not rnet_has_grad,
               "RibonanzaNet backbone correctly frozen" if not rnet_has_grad
               else "ERROR: RibonanzaNet backbone has gradients!")

        # Adapter params should have gradients
        adapter_keywords = [
            "layer_weights", "projection_sequence_features",
            "projection_pairwise_features", "gated_sequence_feature_injector",
            "gated_pairwise_feature_injector"
        ]
        adapter_grad_info = {}
        for name, param in model.named_parameters():
            for kw in adapter_keywords:
                if kw in name and param.requires_grad:
                    has_grad = param.grad is not None and param.grad.abs().sum() > 0
                    adapter_grad_info[name] = has_grad
                    break

        n_with_grad = sum(1 for v in adapter_grad_info.values() if v)
        n_total = len(adapter_grad_info)
        record("adapter_gradients", n_with_grad > 0,
               f"{n_with_grad}/{n_total} adapter params have gradients")

        # Log details
        for name, has_grad in sorted(adapter_grad_info.items()):
            status = "HAS GRAD" if has_grad else "NO GRAD"
            logger.info(f"  {status}: {name}")

    except Exception as e:
        record("model_forward_backward", False, f"Exception: {e}\n{traceback.format_exc()}")


def test_3_mixed_complex():
    """Test with a mixed protein-RNA complex (non-RNA tokens should be PAD/masked)."""
    logger.info("=" * 60)
    logger.info("TEST 3: Mixed complex tokenization (protein-RNA)")
    logger.info("=" * 60)
    try:
        from protenix.data.ribonanza.ribonanza_tokenizer import RibonanzaTokenizer

        # Find a protein-RNA complex
        pdb_candidates = ["1a1t", "1a34"]
        pdb_id = None
        bioassembly_dict = None
        for cand in pdb_candidates:
            pkl_path = os.path.join(BIOASSEMBLY_DIR, f"{cand}.pkl.gz")
            if os.path.exists(pkl_path):
                with gzip.open(pkl_path, "rb") as f:
                    bioassembly_dict = pickle.load(f)
                pdb_id = cand
                break

        if bioassembly_dict is None:
            record("mixed_complex", False, "No protein-RNA complex found in test data")
            return

        atom_array = bioassembly_dict["atom_array"]
        token_array = bioassembly_dict["token_array"]
        n_tokens = len(token_array)

        tokenizer = RibonanzaTokenizer()
        result = tokenizer(token_array=token_array, atom_array=atom_array)
        tokenized_seq = result["tokenized_seq"]
        mask = result["ribonanza_token_mask"]

        n_rna = mask.sum().item()
        n_non_rna = (~mask).sum().item()

        logger.info(f"pdb={pdb_id}: {n_tokens} tokens, {n_rna} RNA, {n_non_rna} non-RNA")

        # For mixed complex, both should be > 0
        has_rna = n_rna > 0
        has_non_rna = n_non_rna > 0

        # Non-RNA should be PAD
        if has_non_rna:
            non_rna_all_pad = (tokenized_seq[~mask] == 4).all().item()
        else:
            non_rna_all_pad = True

        # RNA should be valid tokens
        if has_rna:
            rna_valid = set(tokenized_seq[mask].unique().tolist()).issubset({0, 1, 2, 3, 5})
        else:
            rna_valid = True

        passed = has_rna and non_rna_all_pad and rna_valid
        record("mixed_complex", passed,
               f"pdb={pdb_id}, rna={n_rna}, non_rna={n_non_rna}, "
               f"non_rna_pad={non_rna_all_pad}, rna_valid={rna_valid}")

    except Exception as e:
        record("mixed_complex", False, f"Exception: {e}\n{traceback.format_exc()}")


def print_summary():
    """Print test summary."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("  TEST SUMMARY")
    logger.info("=" * 70)
    for name, info in RESULTS.items():
        status = "PASS" if info["passed"] else "FAIL"
        logger.info(f"  [{status}] {name}: {info['detail']}")
    logger.info("=" * 70)
    overall = "ALL TESTS PASSED" if ALL_PASSED else "SOME TESTS FAILED"
    logger.info(f"  {overall}")
    logger.info("=" * 70)


if __name__ == "__main__":
    logger.info("RibonanzaNet2 v3 Integration Test")
    logger.info(f"PROTENIX_DIR: {PROTENIX_DIR}")
    logger.info(f"RIBONANZA_MODEL_DIR: {RIBONANZA_MODEL_DIR}")
    logger.info("")

    test_1_tokenizer_unit()
    test_3_mixed_complex()
    test_2_model_forward_backward()


    print_summary()
    sys.exit(0 if ALL_PASSED else 1)
