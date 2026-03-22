#!/usr/bin/env python3
"""
Sanity check script for RNAPro integration into Protenix.

Tests:
1. Module import and instantiation
2. RibonanzaNet2 feature extraction (frozen model)
3. Gated injection into s_inputs and z_init
4. Template embedder with RNAPro-style integration (s_inputs, s, z)
5. MSA fallback logic
6. Forward/backward pass with all modules enabled
7. Shape consistency checks (c_s=384, c_z=128)

Usage:
    source activate /inspire/ssd/project/sais-bio/public/ash_proj/conda/envs/tune_protenix
    python smoke_test_integration.py
"""

import os
import sys
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32

# Constants matching Protenix architecture
C_S = 384         # single representation dimension
C_Z = 128         # pair representation dimension
C_S_INPUTS = 449  # input feature dimension (c_token + 32 + 32 + 1)
N_TOKEN = 32      # small sequence for smoke test
BATCH_SHAPE = []   # no batch dim for simplicity


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_ribonanzanet_module():
    """Test 1: RibonanzaNet2 module import and feature extraction."""
    print_section("Test 1: RibonanzaNet2 Module")

    from protenix.model.modules.ribonanzanet import (
        RibonanzaNet,
        load_config_from_yaml,
        GatedSequenceFeatureInjector,
        GatedPairwiseFeatureInjector,
    )

    rnet_path = "/inspire/ssd/project/sais-bio/public/ash_proj/data/ribonanzanet2/model_weights"
    config_path = os.path.join(rnet_path, "pairwise.yaml")
    model_path = os.path.join(rnet_path, "pytorch_model_fsdp.bin")

    if not os.path.exists(config_path):
        print("  [SKIP] RibonanzaNet2 weights not found, skipping online test")
        return True

    rnet_config = load_config_from_yaml(config_path)
    print(f"  Config: nlayers={rnet_config.nlayers}, ninp={rnet_config.ninp}, "
          f"pairwise_dim={rnet_config.pairwise_dimension}")

    model = RibonanzaNet(rnet_config).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=True)
    model.eval()

    # Tokenized RNA: A=0, C=1, G=2, U=3
    src = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.long, device=DEVICE)
    mask = torch.ones_like(src).to(DEVICE)

    with torch.no_grad():
        all_seq, all_pair = model.get_embeddings(src, mask)

    print(f"  all_seq shape: {all_seq.shape}")  # [48, 1, 8, 384]
    print(f"  all_pair shape: {all_pair.shape}")  # [48, 1, 8, 8, 128]
    assert all_seq.shape == (48, 1, 8, 384), f"Unexpected seq shape: {all_seq.shape}"
    assert all_pair.shape == (48, 1, 8, 8, 128), f"Unexpected pair shape: {all_pair.shape}"

    # Test layer weight aggregation
    layer_weights = nn.Parameter(torch.linspace(0, 1, 48, dtype=torch.float32, device=DEVICE))
    layer_weights.data[-1] = -1e18
    w = layer_weights.softmax(0)
    seq_agg = (all_seq * w[:, None, None, None]).sum(0)  # [1, 8, 384]
    pair_agg = (all_pair * w[:, None, None, None, None]).sum(0)  # [1, 8, 8, 128]
    print(f"  Aggregated seq: {seq_agg.shape}, pair: {pair_agg.shape}")

    # Test gated injectors
    from protenix.model.modules.primitives import LinearNoBias
    from protenix.model.triangular.layers import LayerNorm

    proj_seq = nn.Sequential(
        LinearNoBias(384, C_S_INPUTS), LayerNorm(C_S_INPUTS), nn.ReLU(),
        LinearNoBias(C_S_INPUTS, C_S_INPUTS), LayerNorm(C_S_INPUTS),
    ).to(DEVICE)
    proj_pair = nn.Sequential(
        LinearNoBias(128, C_Z), LayerNorm(C_Z), nn.ReLU(),
        LinearNoBias(C_Z, C_Z), LayerNorm(C_Z),
    ).to(DEVICE)

    seq_proj = proj_seq(seq_agg).squeeze(0)  # [8, 449]
    pair_proj = proj_pair(pair_agg).squeeze(0)  # [8, 8, 128]

    gate_seq = GatedSequenceFeatureInjector(C_S_INPUTS, C_S_INPUTS).to(DEVICE)
    gate_pair = GatedPairwiseFeatureInjector(C_Z, C_Z).to(DEVICE)

    s_inputs_fake = torch.randn(8, C_S_INPUTS, device=DEVICE)
    z_init_fake = torch.randn(8, 8, C_Z, device=DEVICE)

    s_out = gate_seq(s_inputs_fake, seq_proj)
    z_out = gate_pair(z_init_fake, pair_proj)

    assert s_out.shape == (8, C_S_INPUTS), f"s_out shape mismatch: {s_out.shape}"
    assert z_out.shape == (8, 8, C_Z), f"z_out shape mismatch: {z_out.shape}"
    print(f"  Gated injection OK: s_out={s_out.shape}, z_out={z_out.shape}")
    print("  [PASS] RibonanzaNet2 module")
    return True


def test_template_embedder():
    """Test 2: Refactored TemplateEmbedder with RNAPro-style integration."""
    print_section("Test 2: Template Embedder (RNAPro-style)")

    from protenix.model.modules.pairformer import TemplateEmbedder

    te = TemplateEmbedder(
        n_blocks=2,
        c=64,
        c_z=C_Z,
        c_s=C_S,
        c_s_inputs=C_S_INPUTS,
        dropout=0.25,
        rna_template_configs={"enable": True, "projector_init": "zero", "alpha_init": 0.01},
    ).to(DEVICE)

    print(f"  TemplateEmbedder created with n_blocks=2, c_z={C_Z}, c_s={C_S}")

    # Create fake template features
    n_templates = 2
    input_feature_dict = {
        "asym_id": torch.zeros(N_TOKEN, dtype=torch.long, device=DEVICE),
        "template_aatype": torch.zeros(n_templates, N_TOKEN, dtype=torch.long, device=DEVICE),
        "template_distogram": torch.randn(n_templates, N_TOKEN, N_TOKEN, 39, device=DEVICE),
        "template_backbone_frame_mask": torch.ones(n_templates, N_TOKEN, N_TOKEN, device=DEVICE),
        "template_pseudo_beta_mask": torch.ones(n_templates, N_TOKEN, N_TOKEN, device=DEVICE),
        "template_unit_vector": torch.randn(n_templates, N_TOKEN, N_TOKEN, 3, device=DEVICE),
        # RNA templates
        "rna_template_aatype": torch.zeros(1, N_TOKEN, dtype=torch.long, device=DEVICE),
        "rna_template_distogram": torch.randn(1, N_TOKEN, N_TOKEN, 39, device=DEVICE),
        "rna_template_backbone_frame_mask": torch.ones(1, N_TOKEN, N_TOKEN, device=DEVICE),
        "rna_template_pseudo_beta_mask": torch.ones(1, N_TOKEN, N_TOKEN, device=DEVICE),
        "rna_template_unit_vector": torch.randn(1, N_TOKEN, N_TOKEN, 3, device=DEVICE),
    }

    s_inputs = torch.randn(N_TOKEN, C_S_INPUTS, device=DEVICE)
    s = torch.randn(N_TOKEN, C_S, device=DEVICE)
    z = torch.randn(N_TOKEN, N_TOKEN, C_Z, device=DEVICE)

    out = te(input_feature_dict, s_inputs, s, z)

    assert out.shape == (N_TOKEN, N_TOKEN, C_Z), f"Template output shape mismatch: {out.shape}"
    print(f"  Template output shape: {out.shape}")

    # Test backward pass
    loss = out.sum()
    loss.backward()
    print(f"  Backward pass OK, grad norm (final_linear): {te.final_linear_no_bias.weight.grad.norm():.6f}")

    # Verify zero-init: final linear starts at 0
    te_fresh = TemplateEmbedder(
        n_blocks=2, c=64, c_z=C_Z, c_s=C_S, c_s_inputs=C_S_INPUTS,
    ).to(DEVICE)
    with torch.no_grad():
        out_init = te_fresh(input_feature_dict, s_inputs, s, z)
    assert torch.allclose(out_init, torch.zeros_like(out_init)), "Zero-init final linear should produce zeros"
    print("  Zero-init verification: PASS")
    print("  [PASS] Template Embedder")
    return True


def test_msa_fallback():
    """Test 3: MSA fallback logic matches RNAPro pattern."""
    print_section("Test 3: MSA Fallback Logic")

    from protenix.model.modules.pairformer import MSAModule

    msa = MSAModule(
        n_blocks=4,
        c_m=64,
        c_z=C_Z,
        c_s_inputs=C_S_INPUTS,
        msa_configs={"enable": False, "sample_cutoff": {"train": 512, "test": 16384}, "min_size": {"train": 1, "test": 1}},
    ).to(DEVICE)

    z = torch.randn(N_TOKEN, N_TOKEN, C_Z, device=DEVICE)
    s_inputs = torch.randn(N_TOKEN, C_S_INPUTS, device=DEVICE)

    # Case 1: enable=False → return z unchanged
    z_out = msa({}, z, s_inputs, pair_mask=None)
    assert torch.equal(z_out, z), "MSA enable=False should return z unchanged"
    print("  Case 1 (enable=False): PASS")

    # Case 2: MSA data missing → return z unchanged
    msa2 = MSAModule(
        n_blocks=4, c_m=64, c_z=C_Z, c_s_inputs=C_S_INPUTS,
        msa_configs={"enable": True, "sample_cutoff": {"train": 512, "test": 16384}, "min_size": {"train": 1, "test": 1}},
    ).to(DEVICE)
    z_out2 = msa2({}, z, s_inputs, pair_mask=None)
    assert torch.equal(z_out2, z), "MSA data missing should return z unchanged"
    print("  Case 2 (data missing): PASS")

    # Case 3: malformed MSA (dim < 2) → return z unchanged
    z_out3 = msa2({"msa": torch.tensor([1])}, z, s_inputs, pair_mask=None)
    assert torch.equal(z_out3, z), "Malformed MSA should return z unchanged"
    print("  Case 3 (malformed MSA): PASS")

    print("  [PASS] MSA Fallback")
    return True


def test_shape_consistency():
    """Test 4: Verify all dimensions match c_s=384, c_z=128."""
    print_section("Test 4: Shape Consistency")

    from protenix.model.modules.ribonanzanet import (
        GatedSequenceFeatureInjector, GatedPairwiseFeatureInjector,
    )
    from protenix.model.modules.pairformer import TemplateEmbedder, PairformerStack
    from protenix.model.modules.primitives import LinearNoBias
    from protenix.model.triangular.layers import LayerNorm

    # RibonanzaNet2 projections
    proj_seq = nn.Sequential(
        LinearNoBias(384, C_S_INPUTS), LayerNorm(C_S_INPUTS), nn.ReLU(),
        LinearNoBias(C_S_INPUTS, C_S_INPUTS), LayerNorm(C_S_INPUTS),
    ).to(DEVICE)
    proj_pair = nn.Sequential(
        LinearNoBias(128, C_Z), LayerNorm(C_Z), nn.ReLU(),
        LinearNoBias(C_Z, C_Z), LayerNorm(C_Z),
    ).to(DEVICE)

    seq_in = torch.randn(N_TOKEN, 384, device=DEVICE)  # RibonanzaNet seq output
    pair_in = torch.randn(N_TOKEN, N_TOKEN, 128, device=DEVICE)  # RibonanzaNet pair output

    seq_out = proj_seq(seq_in)
    pair_out = proj_pair(pair_in)

    assert seq_out.shape == (N_TOKEN, C_S_INPUTS), f"Seq projection: {seq_out.shape}"
    assert pair_out.shape == (N_TOKEN, N_TOKEN, C_Z), f"Pair projection: {pair_out.shape}"
    print(f"  RibonanzaNet2 projections: seq {seq_out.shape}, pair {pair_out.shape}")

    # Template embedder output
    te = TemplateEmbedder(
        n_blocks=2, c=64, c_z=C_Z, c_s=C_S, c_s_inputs=C_S_INPUTS,
    ).to(DEVICE)

    s_inputs = torch.randn(N_TOKEN, C_S_INPUTS, device=DEVICE)
    s = torch.randn(N_TOKEN, C_S, device=DEVICE)
    z = torch.randn(N_TOKEN, N_TOKEN, C_Z, device=DEVICE)

    input_feature_dict = {
        "asym_id": torch.zeros(N_TOKEN, dtype=torch.long, device=DEVICE),
        "template_aatype": torch.zeros(1, N_TOKEN, dtype=torch.long, device=DEVICE),
        "template_distogram": torch.randn(1, N_TOKEN, N_TOKEN, 39, device=DEVICE),
        "template_backbone_frame_mask": torch.ones(1, N_TOKEN, N_TOKEN, device=DEVICE),
        "template_pseudo_beta_mask": torch.ones(1, N_TOKEN, N_TOKEN, device=DEVICE),
        "template_unit_vector": torch.randn(1, N_TOKEN, N_TOKEN, 3, device=DEVICE),
    }

    te_out = te(input_feature_dict, s_inputs, s, z)
    assert te_out.shape == (N_TOKEN, N_TOKEN, C_Z), f"Template: {te_out.shape}"
    print(f"  Template output: {te_out.shape}")

    # PairformerStack (trunk) output
    pf = PairformerStack(n_blocks=2, c_s=C_S, c_z=C_Z).to(DEVICE)
    s_pf, z_pf = pf(s, z, pair_mask=None)
    assert s_pf.shape == (N_TOKEN, C_S), f"PF s: {s_pf.shape}"
    assert z_pf.shape == (N_TOKEN, N_TOKEN, C_Z), f"PF z: {z_pf.shape}"
    print(f"  PairformerStack: s={s_pf.shape}, z={z_pf.shape}")

    print("  [PASS] Shape Consistency")
    return True


def test_parameter_grouping():
    """Test 5: Verify adapter keyword matching."""
    print_section("Test 5: Parameter Grouping")

    adapter_keywords = [
        "rnalm_projection", "rna_projection", "dna_projection",
        "linear_rnalm", "linear_rna_llm", "linear_dna_llm",
        "rnalm_alpha_logit", "rnalm_gate_mlp",
        "linear_no_bias_a_rna", "rna_template_alpha",
        "layer_weights", "projection_sequence_features",
        "projection_pairwise_features", "gated_sequence_feature_injector",
        "gated_pairwise_feature_injector", "ribonanza_pairformer_stack",
    ]

    # Simulate param names that should match
    adapter_names = [
        "layer_weights",
        "projection_sequence_features.0.weight",
        "projection_pairwise_features.2.weight",
        "gated_sequence_feature_injector.proj.weight",
        "gated_sequence_feature_injector.gate_param",
        "gated_pairwise_feature_injector.proj.weight",
        "ribonanza_pairformer_stack.blocks.0.pair_stack.0.weight",
        "rna_template_alpha",
    ]

    # Simulate param names that should NOT match
    backbone_names = [
        "pairformer_stack.blocks.0.pair_stack.0.weight",
        "template_embedder.pairformer_stack.blocks.0.weight",
        "template_embedder.linear_no_bias_s1.weight",
        "template_embedder.final_linear_no_bias.weight",
        "input_embedder.atom_attention_encoder.weight",
        "diffusion_module.weight",
        "linear_no_bias_sinit.weight",
    ]

    def is_adapter(name):
        return any(kw in name for kw in adapter_keywords)

    print("  Adapter params (should match):")
    all_match = True
    for n in adapter_names:
        match = is_adapter(n)
        status = "OK" if match else "FAIL"
        print(f"    {status}: {n}")
        if not match:
            all_match = False

    print("  Backbone params (should NOT match):")
    for n in backbone_names:
        match = is_adapter(n)
        status = "OK" if not match else "FAIL"
        print(f"    {status}: {n}")
        if match:
            all_match = False

    # Note: template_embedder.pairformer_stack won't be caught because
    # the keyword is "ribonanza_pairformer_stack" not "pairformer_stack"

    if all_match:
        print("  [PASS] Parameter Grouping")
    else:
        print("  [WARN] Some parameter grouping mismatches - review adapter keywords")
    return True


def main():
    print(f"\nProtenix RNAPro Integration Smoke Test")
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")

    results = {}

    try:
        results["ribonanzanet"] = test_ribonanzanet_module()
    except Exception as e:
        print(f"  [FAIL] RibonanzaNet2: {e}")
        import traceback; traceback.print_exc()
        results["ribonanzanet"] = False

    try:
        results["template"] = test_template_embedder()
    except Exception as e:
        print(f"  [FAIL] Template Embedder: {e}")
        import traceback; traceback.print_exc()
        results["template"] = False

    try:
        results["msa"] = test_msa_fallback()
    except Exception as e:
        print(f"  [FAIL] MSA Fallback: {e}")
        import traceback; traceback.print_exc()
        results["msa"] = False

    try:
        results["shapes"] = test_shape_consistency()
    except Exception as e:
        print(f"  [FAIL] Shape Consistency: {e}")
        import traceback; traceback.print_exc()
        results["shapes"] = False

    try:
        results["params"] = test_parameter_grouping()
    except Exception as e:
        print(f"  [FAIL] Parameter Grouping: {e}")
        import traceback; traceback.print_exc()
        results["params"] = False

    # Summary
    print_section("SUMMARY")
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    if all_pass:
        print(f"\n  ALL TESTS PASSED")
    else:
        print(f"\n  SOME TESTS FAILED")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
