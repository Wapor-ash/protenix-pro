# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import random
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from protenix.model import sample_confidence
from protenix.model.generator import (
    InferenceNoiseScheduler,
    sample_diffusion,
    sample_diffusion_training,
    TrainingNoiseSampler,
)
from protenix.model.modules.confidence import ConfidenceHead
from protenix.model.modules.diffusion import DiffusionModule
from protenix.model.modules.embedders import (
    ConstraintEmbedder,
    InputFeatureEmbedder,
    RelativePositionEncoding,
)
from protenix.model.modules.head import DistogramHead
from protenix.model.modules.pairformer import (
    MSAModule,
    PairformerStack,
    TemplateEmbedder,
)
from protenix.model.modules.primitives import LinearNoBias
from protenix.model.modules.ribonanzanet import (
    RibonanzaNet,
    load_config_from_yaml,
    GatedSequenceFeatureInjector,
    GatedPairwiseFeatureInjector,
)
from protenix.model.triangular.layers import LayerNorm
from protenix.model.utils import simple_merge_dict_list
from protenix.utils.logger import get_logger
from protenix.utils.permutation.permutation import SymmetricPermutation
from protenix.utils.torch_utils import autocasting_disable_decorator

logger = get_logger(__name__)


def update_input_feature_dict(input_feature_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Lines 1-3 of Algorithm 5 compute d_lm, v_lm, and pad_info utilized in the AtomAttentionEncoder.
    Args:
            input_feature_dict (dict[str, Any]): input features
    Returns:
            input_feature_dict (dict[str, Any]): input features
    """
    from protenix.model.modules.transformer import rearrange_qk_to_dense_trunk

    with torch.no_grad():
        # Prepare tensors in dense trunks for local operations
        q_trunked_list, k_trunked_list, pad_info = rearrange_qk_to_dense_trunk(
            q=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            k=[input_feature_dict["ref_pos"], input_feature_dict["ref_space_uid"]],
            dim_q=[-2, -1],
            dim_k=[-2, -1],
            n_queries=32,
            n_keys=128,
            compute_mask=True,
        )
        # Compute atom pair feature
        d_lm = (
            q_trunked_list[0][..., None, :] - k_trunked_list[0][..., None, :, :]
        )  # [..., n_blocks, n_queries, n_keys, 3]
        v_lm = (
            q_trunked_list[1][..., None].int() == k_trunked_list[1][..., None, :].int()
        ).unsqueeze(
            dim=-1
        )  # [..., n_blocks, n_queries, n_keys, 1]
        input_feature_dict["d_lm"] = d_lm
        input_feature_dict["v_lm"] = v_lm
        input_feature_dict["pad_info"] = pad_info
        return input_feature_dict


class Protenix(nn.Module):
    """
    Implements Algorithm 1 [Main Inference/Train Loop] in AF3
    """

    def __init__(self, configs: Any) -> None:
        super(Protenix, self).__init__()
        self.configs = configs
        torch.backends.cuda.matmul.allow_tf32 = self.configs.enable_tf32
        # Some constants
        self.enable_diffusion_shared_vars_cache = (
            self.configs.enable_diffusion_shared_vars_cache
        )
        self.enable_efficient_fusion = self.configs.enable_efficient_fusion
        self.N_cycle = self.configs.model.N_cycle
        self.N_model_seed = self.configs.model.N_model_seed
        self.train_confidence_only = configs.train_confidence_only
        if self.train_confidence_only:  # the final finetune stage
            assert configs.loss.weight.alpha_diffusion == 0.0
            assert configs.loss.weight.alpha_distogram == 0.0

        # Diffusion scheduler
        self.train_noise_sampler = TrainingNoiseSampler(**configs.train_noise_sampler)
        self.inference_noise_scheduler = InferenceNoiseScheduler(
            **configs.inference_noise_scheduler
        )
        self.diffusion_batch_size = self.configs.diffusion_batch_size

        # Model
        esm_configs = configs.get("esm", {})  # This is used in InputFeatureEmbedder
        rnalm_configs_for_embedder = configs.get("rnalm", {})
        self.input_embedder = InputFeatureEmbedder(
            **configs.model.input_embedder,
            esm_configs=esm_configs,
            rnalm_configs=rnalm_configs_for_embedder,
        )
        self.relative_position_encoding = RelativePositionEncoding(
            **configs.model.relative_position_encoding
        )
        # Pass RNA template configs to TemplateEmbedder if enabled
        rna_template_configs = configs.get("rna_template", {})
        if not rna_template_configs.get("enable", False):
            rna_template_configs = None
        self.template_embedder = TemplateEmbedder(
            **configs.model.template_embedder,
            rna_template_configs=rna_template_configs,
        )
        self.msa_module = MSAModule(
            **configs.model.msa_module,
            msa_configs=configs.data.get("msa", {}),
        )
        self.constraint_embedder = ConstraintEmbedder(
            **configs.model.constraint_embedder
        )
        self._validate_rna_ss_config()

        self.pairformer_stack = PairformerStack(**configs.model.pairformer)

        # === RNA LM (RiNALMo) Embedding Integration ===
        # injection_mode controls where RNA LLM embeddings are injected:
        #   "diffusion" - at DiffusionConditioning (add to s_trunk)
        #   "input"     - at InputFeatureEmbedder (add to s_inputs, like ESM)
        #   "both"      - at both locations
        rnalm_configs = configs.get("rnalm", {})
        self.rnalm_enable = rnalm_configs.get("enable", False)
        self.rnalm_configs = rnalm_configs
        self.rnalm_injection_mode = rnalm_configs.get("injection_mode", "diffusion")
        self.rnalm_separate_dna = rnalm_configs.get("separate_dna_projection", False)
        self.rnalm_use_rna = rnalm_configs.get("use_rna_embed", True)
        self.rnalm_use_dna = rnalm_configs.get("use_dna_embed", True)

        # If both use_rna and use_dna are False, data layer won't produce embeddings.
        # Sync model layer by disabling rnalm entirely to avoid creating orphan projection
        # layers that expect embedding keys missing from input_feature_dict.
        if self.rnalm_enable and not self.rnalm_use_rna and not self.rnalm_use_dna:
            logger.warning(
                "rnalm.enable=True but both use_rna_embed and use_dna_embed are False. "
                "Disabling rnalm at model layer to stay consistent with data layer."
            )
            self.rnalm_enable = False

        # Only pass rnalm_configs to DiffusionModule if diffusion injection is needed
        diffusion_module_kwargs = dict(configs.model.diffusion_module)
        if self.rnalm_enable and self.rnalm_injection_mode in ("diffusion", "both"):
            diffusion_module_kwargs["rnalm_configs"] = rnalm_configs
        else:
            diffusion_module_kwargs["rnalm_configs"] = None
        self.diffusion_module = DiffusionModule(**diffusion_module_kwargs)

        # Only create diffusion-level projection + gates when injection_mode includes diffusion
        if self.rnalm_enable and self.rnalm_injection_mode in ("diffusion", "both"):
            rnalm_embedding_dim = rnalm_configs.get("embedding_dim", 1280)

            if self.rnalm_separate_dna:
                # === Separate RNA/DNA projections — only create needed ones ===
                if self.rnalm_use_rna:
                    self.rna_projection = LinearNoBias(
                        in_features=rnalm_embedding_dim,
                        out_features=configs.c_s,
                    )
                    nn.init.zeros_(self.rna_projection.weight)
                if self.rnalm_use_dna:
                    dna_embedding_dim = rnalm_configs.get("dna_embedding_dim", 1024)
                    self.dna_projection = LinearNoBias(
                        in_features=dna_embedding_dim,
                        out_features=configs.c_s,
                    )
                    nn.init.zeros_(self.dna_projection.weight)
                logger.info(
                    f"Separate RNA/DNA diffusion projections: "
                    f"use_rna={self.rnalm_use_rna} ({rnalm_embedding_dim}->{configs.c_s}), "
                    f"use_dna={self.rnalm_use_dna} ({rnalm_configs.get('dna_embedding_dim', 1024)}->{configs.c_s})"
                )
            else:
                # === Combined projection (legacy) ===
                self.rnalm_projection = LinearNoBias(
                    in_features=rnalm_embedding_dim,
                    out_features=configs.c_s,
                )
                nn.init.zeros_(self.rnalm_projection.weight)

            # === Gate mechanism for controlled LLM injection ===
            self.rnalm_gate_mode = rnalm_configs.get("gate_mode", "none")
            gate_init = rnalm_configs.get("gate_init_logit", -3.0)
            if self.rnalm_gate_mode in ("scalar", "dual"):
                self.rnalm_alpha_logit = nn.Parameter(
                    torch.tensor(float(gate_init))
                )
            if self.rnalm_gate_mode in ("token", "dual"):
                c_s = configs.c_s
                self.rnalm_gate_mlp = nn.Sequential(
                    nn.Linear(c_s, c_s // 4),
                    nn.ReLU(),
                    nn.Linear(c_s // 4, 1),
                )
                # Conservative init: gate outputs ~sigmoid(gate_init) ≈ 0.047
                nn.init.zeros_(self.rnalm_gate_mlp[2].weight)
                nn.init.constant_(self.rnalm_gate_mlp[2].bias, gate_init)
            # === End gate mechanism ===

            logger.info(
                f"RNA LM diffusion injection enabled: "
                f"fusion_method=add, "
                f"gate_mode={self.rnalm_gate_mode}, "
                f"injection_mode={self.rnalm_injection_mode}, "
                f"separate_dna={self.rnalm_separate_dna}"
            )
        else:
            self.rnalm_gate_mode = "none"

        if self.rnalm_enable:
            logger.info(
                f"RNA LM injection_mode={self.rnalm_injection_mode} "
                f"(input={self.rnalm_injection_mode in ('input', 'both')}, "
                f"diffusion={self.rnalm_injection_mode in ('diffusion', 'both')}, "
                f"separate_dna={self.rnalm_separate_dna})"
            )
        # === End RNA LM ===

        # === RibonanzaNet2 Integration (following RNAPro pattern) ===
        rnet2_configs = configs.get("ribonanzanet2", {})
        self.rnet2_enable = rnet2_configs.get("enable", False)
        if self.rnet2_enable:
            import os
            rnet2_model_dir = rnet2_configs.get("model_dir", "")
            config_path = os.path.join(rnet2_model_dir, "pairwise.yaml")
            model_path = os.path.join(rnet2_model_dir, "pytorch_model_fsdp.bin")
            rnet_config = load_config_from_yaml(config_path)
            self.ribonanza_net = RibonanzaNet(rnet_config)
            self.ribonanza_net.load_state_dict(
                torch.load(model_path, map_location="cpu"), strict=True
            )
            # Freeze RibonanzaNet2 — never update its weights
            for p in self.ribonanza_net.parameters():
                p.requires_grad = False
            self.ribonanza_net.eval()

            # Learnable layer weights for aggregating 48 layers
            n_layers = rnet_config.nlayers  # 48
            self.layer_weights = nn.Parameter(
                torch.linspace(0, 1, n_layers, dtype=torch.float32)
            )
            # Mask last layer initially (like RNAPro)
            with torch.no_grad():
                self.layer_weights[-1] = -1e18

            # Projection MLPs: sequence features → c_s_inputs, pairwise → c_z
            s_input_dim = configs.c_s_inputs  # 449
            self.projection_sequence_features = nn.Sequential(
                LinearNoBias(rnet_config.ninp, s_input_dim),
                LayerNorm(s_input_dim),
                nn.ReLU(),
                LinearNoBias(s_input_dim, s_input_dim),
                LayerNorm(s_input_dim),
            )
            self.projection_pairwise_features = nn.Sequential(
                LinearNoBias(rnet_config.pairwise_dimension, configs.c_z),
                LayerNorm(configs.c_z),
                nn.ReLU(),
                LinearNoBias(configs.c_z, configs.c_z),
                LayerNorm(configs.c_z),
            )

            # Gated injectors
            gate_type = rnet2_configs.get("gate_type", "channel")
            self.gated_sequence_feature_injector = GatedSequenceFeatureInjector(
                c_s_new=s_input_dim, c_s=s_input_dim, gate_type=gate_type
            )
            self.gated_pairwise_feature_injector = GatedPairwiseFeatureInjector(
                c_pair=configs.c_z, c_z=configs.c_z, gate_type=gate_type
            )

            # Dedicated PairformerStack for RibonanzaNet2 pairwise features
            n_pf_blocks = rnet2_configs.get("n_pairformer_blocks", 4)
            self.ribonanza_pairformer_stack = PairformerStack(
                n_blocks=n_pf_blocks,
                n_heads=16,
                c_z=configs.c_z,
                c_s=configs.c_s,
            )

            logger.info(
                f"RibonanzaNet2 integration enabled: "
                f"n_layers={n_layers}, ninp={rnet_config.ninp}, "
                f"pairwise_dim={rnet_config.pairwise_dimension}, "
                f"gate_type={gate_type}, n_pairformer_blocks={n_pf_blocks}"
            )
        # === End RibonanzaNet2 ===

        self.distogram_head = DistogramHead(**configs.model.distogram_head)
        self.confidence_head = ConfidenceHead(**configs.model.confidence_head)

        self.c_s, self.c_z, self.c_s_inputs = (
            configs.c_s,
            configs.c_z,
            configs.c_s_inputs,
        )
        self.linear_no_bias_sinit = LinearNoBias(
            in_features=self.c_s_inputs, out_features=self.c_s
        )
        self.linear_no_bias_zinit1 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_zinit2 = LinearNoBias(
            in_features=self.c_s, out_features=self.c_z
        )
        self.linear_no_bias_token_bond = LinearNoBias(
            in_features=1, out_features=self.c_z
        )
        self.linear_no_bias_z_cycle = LinearNoBias(
            in_features=self.c_z, out_features=self.c_z
        )
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s, out_features=self.c_s
        )
        self.layernorm_z_cycle = LayerNorm(self.c_z)
        self.layernorm_s = LayerNorm(self.c_s)

        # Zero init the recycling layer
        nn.init.zeros_(self.linear_no_bias_z_cycle.weight)
        nn.init.zeros_(self.linear_no_bias_s.weight)

    def _validate_rna_ss_config(self) -> None:
        rna_ss_configs = self.configs.get("rna_ss", {})
        if not rna_ss_configs.get("enable", False):
            return

        substructure_configs = self.configs.model.constraint_embedder.substructure_embedder
        if not substructure_configs.enable:
            raise ValueError(
                "rna_ss.enable=True requires model.constraint_embedder.substructure_embedder.enable=True"
            )
        if substructure_configs.n_classes != rna_ss_configs.get("n_classes", 6):
            raise ValueError(
                "rna_ss.n_classes must match model.constraint_embedder.substructure_embedder.n_classes"
            )
        substructure_initialize_method = substructure_configs.get(
            "initialize_method",
            "inherit",
        )
        if substructure_initialize_method in (None, "", "inherit"):
            substructure_initialize_method = (
                self.configs.model.constraint_embedder.initialize_method
            )
        if substructure_initialize_method == "zero":
            raise ValueError(
                "rna_ss.enable=True requires substructure embedder initialize_method != 'zero'"
            )

    def reinit_rna_projector_from_protein(self, checkpoint_keys=None) -> str:
        """Conditionally re-initialize the RNA projector after checkpoint loading.

        Smart detection logic:
        1. If checkpoint already contains RNA projector weights
           (``linear_no_bias_a_rna``), they were loaded by ``load_state_dict``
           → do nothing, return "loaded_from_checkpoint".
        2. If checkpoint does NOT contain RNA projector weights (protein-only
           checkpoint), apply the configured init strategy:
           - ``projector_init="protein"``: copy from loaded protein projector
           - ``projector_init="zero"``: zero-init the projector

        Args:
            checkpoint_keys: set/list of keys present in the loaded checkpoint.
                Used to detect whether RNA projector weights were in the
                checkpoint. If None, falls back to always re-initializing.

        Returns:
            str: one of "loaded_from_checkpoint", "copied_from_protein",
                 "zero_initialized", or "skipped".
        """
        te = self.template_embedder
        if not getattr(te, "rna_template_enable", False):
            return "skipped"

        # Check if the checkpoint already contained RNA projector weights
        rna_projector_key = "template_embedder.linear_no_bias_a_rna.weight"
        # Also check with DDP module. prefix
        rna_projector_key_ddp = "module." + rna_projector_key

        if checkpoint_keys is not None:
            has_rna_weights = (
                rna_projector_key in checkpoint_keys
                or rna_projector_key_ddp in checkpoint_keys
            )
            if has_rna_weights:
                # Checkpoint already had trained RNA projector weights —
                # load_state_dict already loaded them. Don't overwrite.
                return "loaded_from_checkpoint"

        # Checkpoint did NOT have RNA projector weights — apply init strategy
        init_mode = getattr(te, "rna_projector_init", "protein")
        with torch.no_grad():
            if init_mode == "zero":
                nn.init.zeros_(te.linear_no_bias_a_rna.weight)
                return "zero_initialized"
            else:
                te.linear_no_bias_a_rna.weight.copy_(te.linear_no_bias_a.weight)
                return "copied_from_protein"

    def get_pairformer_output(
        self,
        input_feature_dict: dict[str, Any],
        N_cycle: int,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        mc_dropout: bool = False,
        mc_dropout_rate: float = 0.4,
    ) -> tuple[torch.Tensor, ...]:
        """
        The forward pass from the input to pairformer output

        Args:
            input_feature_dict (dict[str, Any]): input features
            N_cycle (int): number of cycles
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            Tuple[torch.Tensor, ...]: s_inputs, s, z
        """
        if self.train_confidence_only:
            self.input_embedder.eval()
            self.template_embedder.eval()
            self.msa_module.eval()
            self.pairformer_stack.eval()

        # Line 1-5
        s_inputs = self.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=chunk_size
        )  # [..., N_token, 449]
        z_constraint = None

        # === RibonanzaNet2: extract features and inject into s_inputs (v3) ===
        _rnet2_pairwise_features = None
        if self.rnet2_enable:
            mask = input_feature_dict.get("ribonanza_token_mask")  # [N_token] bool
            if mask is not None and mask.any():
                src = input_feature_dict["tokenized_seq"].unsqueeze(dim=0)  # [1, N]
                # v3: pass RNA mask as src_mask to prevent non-RNA tokens
                # from participating in attention inside RibonanzaNet
                src_mask = mask.unsqueeze(dim=0).long().to(src.device)  # [1, N]

                with torch.no_grad():
                    self.ribonanza_net.eval()
                    all_seq_feats, all_pair_feats = self.ribonanza_net.get_embeddings(
                        src, src_mask=src_mask
                    )

                # Aggregate across 48 layers using learnable weights
                w = self.layer_weights.softmax(0)  # [48]
                seq_feats = (all_seq_feats * w[:, None, None, None]).sum(0)  # [1, L, ninp]
                pair_feats = (all_pair_feats * w[:, None, None, None, None]).sum(0)  # [1, L, L, pair_dim]

                # Project features
                seq_feats_proj = self.projection_sequence_features(seq_feats).squeeze(dim=0)  # [L, c_s_inputs]
                pair_feats_proj = self.projection_pairwise_features(pair_feats).squeeze(dim=0)  # [L, L, c_z]

                # v3: output masking as second safety layer
                # src_mask prevents attention pollution; this prevents residual
                # feedforward leakage from non-RNA positions
                mask_f = mask.to(seq_feats_proj.dtype)  # [N]
                seq_feats_proj = seq_feats_proj * mask_f.unsqueeze(-1)  # [N, C]
                pair_feats_proj = pair_feats_proj * (
                    mask_f.unsqueeze(-1) * mask_f.unsqueeze(-2)
                ).unsqueeze(-1)  # [N, N, C]

                s_inputs = self.gated_sequence_feature_injector(s_inputs, seq_feats_proj)
                _rnet2_pairwise_features = pair_feats_proj
        # === End RibonanzaNet2 (v3) ===

        if "constraint_feature" in input_feature_dict:
            z_constraint = self.constraint_embedder(
                input_feature_dict["constraint_feature"]
            )

        s_init = self.linear_no_bias_sinit(s_inputs)  # [..., N_token, c_s]
        z_init = (
            self.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  # [..., N_token, N_token, c_z]
        if inplace_safe:
            z_init += self.relative_position_encoding(input_feature_dict["relp"])
            z_init += self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
            if z_constraint is not None:
                z_init += z_constraint
        else:
            z_init = z_init + self.relative_position_encoding(
                input_feature_dict["relp"]
            )
            z_init = z_init + self.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
            if z_constraint is not None:
                z_init = z_init + z_constraint

        # === RibonanzaNet2: inject pairwise features into z_init ===
        if self.rnet2_enable and _rnet2_pairwise_features is not None:
            z_init = self.gated_pairwise_feature_injector(z_init, _rnet2_pairwise_features)
        # === End RibonanzaNet2 pairwise injection ===

        # Line 6
        z = torch.zeros_like(z_init)
        s = torch.zeros_like(s_init)

        # Line 7-13 recycling
        for cycle_no in range(N_cycle):
            with torch.set_grad_enabled(
                self.training
                and (not self.train_confidence_only)
                and cycle_no == (N_cycle - 1)
            ):
                if mc_dropout:
                    z = z_init + F.dropout(
                        self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z)),
                        p=self.configs.mc_dropout_rate,
                    )
                else:
                    z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
                if inplace_safe:
                    if self.template_embedder.n_blocks > 0:
                        z += self.template_embedder(
                            input_feature_dict,
                            s_inputs,
                            s,
                            z,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                else:
                    if self.template_embedder.n_blocks > 0:
                        z = z + self.template_embedder(
                            input_feature_dict,
                            s_inputs,
                            s,
                            z,
                            triangle_multiplicative=self.configs.triangle_multiplicative,
                            triangle_attention=self.configs.triangle_attention,
                            inplace_safe=inplace_safe,
                            chunk_size=chunk_size,
                        )
                    z = self.msa_module(
                        input_feature_dict,
                        z,
                        s_inputs,
                        pair_mask=None,
                        triangle_multiplicative=self.configs.triangle_multiplicative,
                        triangle_attention=self.configs.triangle_attention,
                        inplace_safe=inplace_safe,
                        chunk_size=chunk_size,
                    )
                s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
                s, z = self.pairformer_stack(
                    s,
                    z,
                    pair_mask=None,
                    triangle_multiplicative=self.configs.triangle_multiplicative,
                    triangle_attention=self.configs.triangle_attention,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                )

        if self.train_confidence_only:
            self.input_embedder.train()
            self.template_embedder.train()
            self.msa_module.train()
            self.pairformer_stack.train()

        return s_inputs, s, z

    # === RNA LM: project RNA LM embeddings (fail-fast if missing, like ESM) ===
    def _get_s_rnalm(
        self,
        input_feature_dict: dict[str, Any],
        N_token: int,
        s_trunk: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Project RNA LM embeddings to c_s dimension, apply gating.
        Returns None when injection_mode is 'input' (input-only mode uses InputFeatureEmbedder).

        Raises RuntimeError if rnalm is enabled but embeddings are missing from
        input_feature_dict — following the same fail-fast pattern as ESM embeddings
        to prevent silent training with random noise.

        When separate_dna_projection=True:
            Uses rna_projection for RNA embeddings + dna_projection for DNA embeddings,
            combines them (they don't overlap by position).
        When separate_dna_projection=False (legacy):
            Uses single rnalm_projection on the combined tensor.
        """
        if not self.rnalm_enable:
            return None
        # Input-only injection is handled in InputFeatureEmbedder; no s_rnalm for diffusion
        if self.rnalm_injection_mode == "input":
            return None

        if self.rnalm_separate_dna:
            # === Separate RNA/DNA projections — only use enabled ones ===
            # Initialize lm_delta as zeros
            lm_delta = torch.zeros(
                [*input_feature_dict["token_index"].shape[:-1], self.configs.c_s],
                device=input_feature_dict["token_index"].device,
                dtype=next(self.parameters()).dtype,
            )

            if self.rnalm_use_rna:
                if "rna_llm_embedding" not in input_feature_dict:
                    raise RuntimeError(
                        "rnalm enabled with use_rna_embed=True but 'rna_llm_embedding' "
                        "missing from input features."
                    )
                lm_delta = lm_delta + self.rna_projection(input_feature_dict["rna_llm_embedding"])

            if self.rnalm_use_dna:
                if "dna_llm_embedding" not in input_feature_dict:
                    raise RuntimeError(
                        "rnalm enabled with use_dna_embed=True but 'dna_llm_embedding' "
                        "missing from input features."
                    )
                lm_delta = lm_delta + self.dna_projection(input_feature_dict["dna_llm_embedding"])
        else:
            # === Combined projection (legacy) ===
            if "rnalm_token_embedding" not in input_feature_dict:
                raise RuntimeError(
                    "rnalm.enable=True but 'rnalm_token_embedding' is missing from "
                    "input features. Ensure the RNA/DNA LLM embedding data pipeline is "
                    "correctly configured (check embedding_dir / sequence_fpath) and that "
                    "the bioassembly file is not corrupted. Remove corrupted entries from "
                    "training indices and regenerate embeddings if needed."
                )
            lm_delta = self.rnalm_projection(input_feature_dict["rnalm_token_embedding"])

        # Apply gating if enabled
        if self.rnalm_gate_mode == "scalar":
            g1 = torch.sigmoid(self.rnalm_alpha_logit)
            lm_delta = g1 * lm_delta
        elif self.rnalm_gate_mode == "token" and s_trunk is not None:
            g2 = torch.sigmoid(self.rnalm_gate_mlp(s_trunk.detach()))
            lm_delta = g2 * lm_delta
        elif self.rnalm_gate_mode == "dual" and s_trunk is not None:
            g1 = torch.sigmoid(self.rnalm_alpha_logit)
            g2 = torch.sigmoid(self.rnalm_gate_mlp(s_trunk.detach()))
            lm_delta = g1 * g2 * lm_delta

        return lm_delta
    # === End RNA LM ===

    def sample_diffusion(self, **kwargs: Any) -> torch.Tensor:
        """
        Samples diffusion process based on the provided configurations.

        Returns:
            torch.Tensor: The result of the diffusion sampling process.
        """
        _configs = {
            key: self.configs.sample_diffusion.get(key)
            for key in [
                "gamma0",
                "gamma_min",
                "noise_scale_lambda",
                "step_scale_eta",
            ]
        }
        _configs.update(
            {
                "attn_chunk_size": (
                    self.configs.infer_setting.chunk_size if not self.training else None
                ),
                "diffusion_chunk_size": (
                    self.configs.infer_setting.sample_diffusion_chunk_size
                    if not self.training
                    else None
                ),
            }
        )
        return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
            sample_diffusion
        )(**_configs, **kwargs)

    def run_confidence_head(self, *args: Any, **kwargs: Any) -> Any:
        """
        Runs the confidence head with optional automatic mixed precision (AMP) disabled.

        Returns:
            Any: The output of the confidence head.
        """
        return autocasting_disable_decorator(self.configs.skip_amp.confidence_head)(
            self.confidence_head
        )(*args, **kwargs)

    def main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        N_model_seed: int = 1,
        symmetric_permutation: SymmetricPermutation = None,
        mc_dropout_apply_rate: float = 0.4,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (multiple model seeds) for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_dict (dict[str, Any]): Label dictionary.
            N_cycle (int): Number of cycles.
            mode (str): Mode of operation (e.g., 'inference').
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to True.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to 4.
            N_model_seed (int): Number of model seeds. Defaults to 1.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.
            mc_dropout_apply_rate (float): Only for inference mode

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        # For backward compatibility, if N_model_seed > 1, process multiple seeds here
        # But in evaluation mode, this should be handled externally
        if N_model_seed > 1 and mode in ["inference"]:
            pred_dicts = []
            log_dicts = []
            time_trackers = []
            for _ in range(N_model_seed):
                pred_dict, log_dict, time_tracker = self._main_inference_loop(
                    input_feature_dict=(
                        copy.deepcopy(input_feature_dict)
                        if (N_model_seed > 1 and mode == "inference")
                        else input_feature_dict
                    ),  # the input_feature_dict is modified when mode is "inference"
                    label_dict=label_dict,
                    N_cycle=N_cycle,
                    mode=mode,
                    inplace_safe=inplace_safe,
                    chunk_size=chunk_size,
                    symmetric_permutation=symmetric_permutation,
                    mc_dropout=random.random() < mc_dropout_apply_rate,
                )
                pred_dicts.append(pred_dict)
                log_dicts.append(log_dict)
                time_trackers.append(time_tracker)

            # Combine outputs of multiple models
            def _cat(dict_list, key):
                return torch.cat([x[key] for x in dict_list], dim=0)

            def _list_join(dict_list, key):
                return sum([x[key] for x in dict_list], [])

            all_pred_dict = {
                "coordinate": _cat(pred_dicts, "coordinate"),
                "summary_confidence": _list_join(pred_dicts, "summary_confidence"),
                "full_data": _list_join(pred_dicts, "full_data"),
                "plddt": _cat(pred_dicts, "plddt"),
                "pae": _cat(pred_dicts, "pae"),
                "pde": _cat(pred_dicts, "pde"),
                "resolved": _cat(pred_dicts, "resolved"),
            }

            all_log_dict = simple_merge_dict_list(log_dicts)
            all_time_dict = simple_merge_dict_list(time_trackers)
            return all_pred_dict, all_log_dict, all_time_dict
        else:
            # Single seed inference - delegate to _main_inference_loop
            return self._main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
                symmetric_permutation=symmetric_permutation,
                mc_dropout=random.random() < mc_dropout_apply_rate,
            )

    def _get_dynamic_chunk_size(self, N_token: int) -> Optional[int]:
        """
        Get dynamic chunk_size based on token count

        Args:
            N_token (int): Number of tokens

        Returns:
            Optional[int]: Optimal chunk_size for the given token count
        """
        if not hasattr(self.configs.infer_setting, "chunk_size_thresholds"):
            return self.configs.infer_setting.chunk_size

        thresholds = self.configs.infer_setting.chunk_size_thresholds

        # Convert string keys to integers and sort in ascending order
        threshold_pairs = [(int(k), v) for k, v in thresholds.items()]
        sorted_thresholds = sorted(threshold_pairs, key=lambda x: x[0])

        # Find the appropriate chunk_size for the given token count
        for threshold, chunk_size in sorted_thresholds:
            if N_token <= threshold:
                return None if chunk_size == -1 else chunk_size

        # For token counts larger than the largest threshold, use smallest chunk_size
        return 32  # extreme case for very large proteins

    def _main_inference_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        mode: str,
        inplace_safe: bool = True,
        chunk_size: Optional[int] = 4,
        symmetric_permutation: SymmetricPermutation = None,
        mc_dropout: bool = False,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main inference loop (single model seed) for the Alphafold3 model.
        mc_dropout: do not use by default

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]: Prediction, log, and time dictionaries.
        """
        step_st = time.time()
        N_token = input_feature_dict["residue_index"].shape[-1]

        # Apply dynamic chunk_size if enabled (otherwise keep the passed chunk_size)
        if (
            hasattr(self.configs.infer_setting, "dynamic_chunk_size")
            and self.configs.infer_setting.dynamic_chunk_size
        ):
            chunk_size = self._get_dynamic_chunk_size(N_token)
        # If dynamic chunking is disabled, chunk_size keeps its original value from the function parameter

        log_dict = {}
        pred_dict = {}
        time_tracker = {}

        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            mc_dropout=mc_dropout,
        )

        # === RNA LM: project RNA LM embeddings for inference ===
        N_token = input_feature_dict["residue_index"].shape[-1]
        s_rnalm = self._get_s_rnalm(input_feature_dict, N_token, s_trunk=s)
        # === End RNA LM ===

        keys_to_delete = []
        for key in input_feature_dict.keys():
            if "template_" in key or key in [
                "msa",
                "has_deletion",
                "deletion_value",
                "profile",
                "deletion_mean",
                # "token_bonds",
            ]:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del input_feature_dict[key]
        step_trunk = time.time()
        time_tracker.update({"pairformer": step_trunk - step_st})
        # Sample diffusion
        # [..., N_sample, N_atom, 3]
        N_sample = self.configs.sample_diffusion["N_sample"]
        N_step = self.configs.sample_diffusion["N_step"]

        noise_schedule = self.inference_noise_scheduler(
            N_step=N_step, device=s_inputs.device, dtype=s_inputs.dtype
        )
        cache = dict()
        if self.enable_diffusion_shared_vars_cache:
            # line 1-5 of algorithm 21 calculate z in diffusion conditioning
            cache["pair_z"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.diffusion_conditioning.prepare_cache)(
                input_feature_dict["relp"], z, False
            )
            cache["p_lm/c_l"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.atom_attention_encoder.prepare_cache)(
                ref_pos=input_feature_dict["ref_pos"],
                ref_charge=input_feature_dict["ref_charge"],
                ref_mask=input_feature_dict["ref_mask"],
                ref_element=input_feature_dict["ref_element"],
                ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                d_lm=input_feature_dict["d_lm"],
                v_lm=input_feature_dict["v_lm"],
                pad_info=input_feature_dict["pad_info"],
                r_l=True,
                z=cache["pair_z"],
                inplace_safe=False,
            )
        else:
            cache["pair_z"] = None
            cache["p_lm/c_l"] = [None, None]
        # === RNA LM: pass s_rnalm to sample_diffusion ===
        pred_dict["coordinate"] = self.sample_diffusion(
            denoise_net=self.diffusion_module,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=None if cache["pair_z"] is not None else z,
            pair_z=cache["pair_z"],
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            N_sample=N_sample,
            noise_schedule=noise_schedule,
            inplace_safe=inplace_safe,
            enable_efficient_fusion=self.enable_efficient_fusion,
            s_rnalm=s_rnalm,
        )
        # === End RNA LM ===

        step_diffusion = time.time()
        time_tracker.update({"diffusion": step_diffusion - step_trunk})
        # Distogram logits: log contact_probs only, to reduce the dimension
        pred_dict["contact_probs"] = autocasting_disable_decorator(True)(
            sample_confidence.compute_contact_prob
        )(
            distogram_logits=self.distogram_head(z),
            **sample_confidence.get_bin_params(self.configs.loss.distogram),
        )  # [N_token, N_token]

        # Confidence logits
        (
            pred_dict["plddt"],
            pred_dict["pae"],
            pred_dict["pde"],
            pred_dict["resolved"],
        ) = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=pred_dict["coordinate"],
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        step_confidence = time.time()
        time_tracker.update({"confidence": step_confidence - step_diffusion})
        time_tracker.update({"model_forward": time.time() - step_st})

        # Permutation: when label is given, permute coordinates and other heads
        if label_dict is not None and symmetric_permutation is not None:
            pred_dict, log_dict = symmetric_permutation.permute_inference_pred_dict(
                input_feature_dict=input_feature_dict,
                pred_dict=pred_dict,
                label_dict=label_dict,
                permute_by_pocket=("pocket_mask" in label_dict)
                and ("interested_ligand_mask" in label_dict),
            )
            last_step_seconds = step_confidence
            time_tracker.update({"permutation": time.time() - last_step_seconds})

        # Summary Confidence & Full Data
        # Computed after coordinates and logits are permuted
        if label_dict is None:
            interested_atom_mask = None
        else:
            interested_atom_mask = label_dict.get("interested_ligand_mask", None)
        (
            pred_dict["summary_confidence"],
            pred_dict["full_data"],
        ) = autocasting_disable_decorator(True)(
            sample_confidence.compute_full_data_and_summary
        )(
            configs=self.configs,
            pae_logits=pred_dict["pae"],
            plddt_logits=pred_dict["plddt"],
            pde_logits=pred_dict["pde"],
            contact_probs=pred_dict.get(
                "per_sample_contact_probs", pred_dict["contact_probs"]
            ),
            token_asym_id=input_feature_dict["asym_id"],
            token_has_frame=input_feature_dict["has_frame"],
            atom_coordinate=pred_dict["coordinate"],
            atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
            atom_is_polymer=1 - input_feature_dict["is_ligand"],
            N_recycle=N_cycle,
            interested_atom_mask=interested_atom_mask,
            return_full_data=True,
            mol_id=(input_feature_dict["mol_id"] if mode != "inference" else None),
            elements_one_hot=(
                input_feature_dict["ref_element"] if mode != "inference" else None
            ),
        )

        return pred_dict, log_dict, time_tracker

    def main_train_loop(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        N_cycle: int,
        symmetric_permutation: SymmetricPermutation,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Main training loop for the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict): Label dictionary (cropped).
            N_cycle (int): Number of cycles.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object.
            inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        s_inputs, s, z = self.get_pairformer_output(
            input_feature_dict=input_feature_dict,
            N_cycle=N_cycle,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        # === RNA LM: project RNA LM embeddings for training ===
        N_token = input_feature_dict["residue_index"].shape[-1]
        s_rnalm = self._get_s_rnalm(input_feature_dict, N_token, s_trunk=s)
        # === End RNA LM ===

        log_dict = {}
        pred_dict = {}

        cache = dict()
        if self.enable_diffusion_shared_vars_cache:
            cache["pair_z"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.diffusion_conditioning.prepare_cache)(
                input_feature_dict["relp"], z, False
            )
            cache["p_lm/c_l"] = autocasting_disable_decorator(
                self.configs.skip_amp.sample_diffusion
            )(self.diffusion_module.atom_attention_encoder.prepare_cache)(
                ref_pos=input_feature_dict["ref_pos"],
                ref_charge=input_feature_dict["ref_charge"],
                ref_mask=input_feature_dict["ref_mask"],
                ref_element=input_feature_dict["ref_element"],
                ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
                atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
                d_lm=input_feature_dict["d_lm"],
                v_lm=input_feature_dict["v_lm"],
                pad_info=input_feature_dict["pad_info"],
                r_l=True,
                z=cache["pair_z"],
                inplace_safe=False,
            )
        else:
            cache["pair_z"] = None
            cache["p_lm/c_l"] = [None, None]
        # Mini-rollout: used for confidence and label permutation
        with torch.no_grad():
            # [..., 1, N_atom, 3]
            N_sample_mini_rollout = self.configs.sample_diffusion[
                "N_sample_mini_rollout"
            ]  # =1
            N_step_mini_rollout = self.configs.sample_diffusion["N_step_mini_rollout"]
            self.diffusion_module.eval()  # use eval mode for mini-rollout
            # === RNA LM: pass s_rnalm to mini-rollout ===
            coordinate_mini = self.sample_diffusion(
                denoise_net=self.diffusion_module,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs.detach(),
                s_trunk=s.detach(),
                z_trunk=None if cache["pair_z"] is not None else z.detach(),
                pair_z=None if cache["pair_z"] is None else cache["pair_z"].detach(),
                p_lm=(
                    None
                    if cache["p_lm/c_l"][0] is None
                    else cache["p_lm/c_l"][0].detach()
                ),
                c_l=(
                    None
                    if cache["p_lm/c_l"][1] is None
                    else cache["p_lm/c_l"][1].detach()
                ),
                N_sample=N_sample_mini_rollout,
                noise_schedule=self.inference_noise_scheduler(
                    N_step=N_step_mini_rollout,
                    device=s_inputs.device,
                    dtype=s_inputs.dtype,
                ),
                enable_efficient_fusion=self.enable_efficient_fusion,
                s_rnalm=s_rnalm.detach() if s_rnalm is not None else None,
            )
            # === End RNA LM ===
            self.diffusion_module.train()
            coordinate_mini.detach_()
            pred_dict["coordinate_mini"] = coordinate_mini

            # Permute ground truth to match mini-rollout prediction
            (
                label_dict,
                perm_log_dict,
            ) = symmetric_permutation.permute_label_to_match_mini_rollout(
                coordinate_mini,
                input_feature_dict,
                label_dict,
                label_full_dict,
            )
            log_dict.update(perm_log_dict)

        # Confidence: use mini-rollout prediction, and detach token embeddings
        drop_embedding = (
            random.random() < self.configs.model.confidence_embedding_drop_rate
        )
        plddt_pred, pae_pred, pde_pred, resolved_pred = self.run_confidence_head(
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=z,
            pair_mask=None,
            x_pred_coords=coordinate_mini,
            use_embedding=not drop_embedding,
            triangle_multiplicative=self.configs.triangle_multiplicative,
            triangle_attention=self.configs.triangle_attention,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )
        pred_dict.update(
            {
                "plddt": plddt_pred,
                "pae": pae_pred,
                "pde": pde_pred,
                "resolved": resolved_pred,
            }
        )

        if self.train_confidence_only:
            # Skip diffusion loss and distogram loss. Return now.
            return pred_dict, label_dict, log_dict

        # Denoising: use permuted coords to generate noisy samples and perform denoising
        # x_denoised: [..., N_sample, N_atom, 3]
        # x_noise_level: [..., N_sample]
        N_sample = self.diffusion_batch_size
        drop_conditioning = (
            random.random() < self.configs.model.condition_embedding_drop_rate
        )
        # === RNA LM: pass s_rnalm to sample_diffusion_training ===
        _, x_denoised, x_noise_level = autocasting_disable_decorator(
            self.configs.skip_amp.sample_diffusion_training
        )(sample_diffusion_training)(
            noise_sampler=self.train_noise_sampler,
            denoise_net=self.diffusion_module,
            label_dict=label_dict,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s,
            z_trunk=None if cache["pair_z"] is not None else z,
            pair_z=cache["pair_z"],
            p_lm=cache["p_lm/c_l"][0],
            c_l=cache["p_lm/c_l"][1],
            N_sample=N_sample,
            diffusion_chunk_size=self.configs.diffusion_chunk_size,
            use_conditioning=not drop_conditioning,
            enable_efficient_fusion=self.enable_efficient_fusion,
            s_rnalm=s_rnalm,
        )
        # === End RNA LM ===
        pred_dict.update(
            {
                "distogram": autocasting_disable_decorator(True)(self.distogram_head)(
                    z
                ),
                # [..., N_sample=48, N_atom, 3]: diffusion loss
                "coordinate": x_denoised,
                "noise_level": x_noise_level,
            }
        )

        # Permute symmetric atom/chain in each sample to match true structure
        # Note: currently chains cannot be permuted since label is cropped
        (
            pred_dict,
            perm_log_dict,
            _,
            _,
        ) = symmetric_permutation.permute_diffusion_sample_to_match_label(
            input_feature_dict, pred_dict, label_dict, stage="train"
        )
        log_dict.update(perm_log_dict)
        log_dict.update({"noise_level": x_noise_level})

        return pred_dict, label_dict, log_dict

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        label_full_dict: dict[str, Any],
        label_dict: dict[str, Any],
        mode: str = "inference",
        current_step: Optional[int] = None,
        symmetric_permutation: SymmetricPermutation = None,
        disable_inplace: bool = False,
        mc_dropout_apply_rate: float = 0.4,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        """
        Forward pass of the Alphafold3 model.

        Args:
            input_feature_dict (dict[str, Any]): Input features dictionary.
            label_full_dict (dict[str, Any]): Full label dictionary (uncropped).
            label_dict (dict[str, Any]): Label dictionary (cropped).
            mode (str): Mode of operation ('train', 'inference', 'eval'). Defaults to 'inference'.
            current_step (Optional[int]): Current training step. Defaults to None.
            symmetric_permutation (SymmetricPermutation): Symmetric permutation object. Defaults to None.

        Returns:
            tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
                Prediction, updated label, and log dictionaries.
        """

        assert mode in ["train", "eval", "inference"]
        not_use_gradient = not (self.training or torch.is_grad_enabled())
        inplace_safe = not_use_gradient and (not disable_inplace)

        input_feature_dict = self.relative_position_encoding.generate_relp(
            input_feature_dict
        )
        input_feature_dict = update_input_feature_dict(input_feature_dict)

        if mode == "train":
            nc_rng = np.random.RandomState(current_step)
            N_cycle = nc_rng.randint(1, self.N_cycle + 1)
            assert self.training
            assert label_dict is not None
            assert symmetric_permutation is not None

            pred_dict, label_dict, log_dict = self.main_train_loop(
                input_feature_dict=input_feature_dict,
                label_full_dict=label_full_dict,
                label_dict=label_dict,
                N_cycle=N_cycle,
                symmetric_permutation=symmetric_permutation,
                inplace_safe=inplace_safe,
                chunk_size=None,
            )
            log_dict["N_cycle"] = N_cycle
        elif mode == "inference":
            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=None,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=self.configs.infer_setting.chunk_size,
                N_model_seed=self.N_model_seed,
                symmetric_permutation=None,
                mc_dropout_apply_rate=mc_dropout_apply_rate,
            )
            log_dict.update({"time": time_tracker})
        elif mode == "eval":
            if label_dict is not None:
                assert (
                    label_dict["coordinate"].size()
                    == label_full_dict["coordinate"].size()
                )
                label_dict.update(label_full_dict)

            pred_dict, log_dict, time_tracker = self.main_inference_loop(
                input_feature_dict=input_feature_dict,
                label_dict=label_dict,
                N_cycle=self.N_cycle,
                mode=mode,
                inplace_safe=inplace_safe,
                chunk_size=self.configs.infer_setting.chunk_size,
                N_model_seed=1,
                symmetric_permutation=symmetric_permutation,
                mc_dropout_apply_rate=mc_dropout_apply_rate,
            )
            log_dict.update({"time": time_tracker})

        return pred_dict, label_dict, log_dict
