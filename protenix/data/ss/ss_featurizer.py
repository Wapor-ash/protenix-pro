# Secondary Structure (BPP) Featurizer
#
# Loads pre-computed full-length Base-Pair Probability (BPP) matrices and derives
# crop-aware pair + single features following the analysis report recommendations.
#
# Pair features (4-channel, [N_crop, N_crop, 4]):
#   ch0: P_in[i,j]      — crop-internal BPP (real pairing support within crop)
#   ch1: 1 - P_in[i,j]  — complement (non-pairing tendency within crop)
#   ch2: O[i,j]          — outside mass = (p_out(i) + p_out(j)) / 2
#   ch3: B[i,j]          — boundary mask (distance-based decay from crop edges)
#
# Single features (3-channel, [N_crop, 3]):
#   ch0: p_out(i)  — probability of pairing outside crop
#   ch1: p_in(i)   — probability of pairing inside crop
#   ch2: p_unp(i)  — probability of being unpaired
#
# Full-sequence lookup supports both:
#   - training-style mapping: bioassembly_dict["sequences"][entity_id] -> sequence
#   - inference-style list: [{"rnaSequence": {"sequence": ...}}, ...]
# Position mapping uses res_id (1-based) to index into the full-length BPP matrix.

import json
import os
from typing import Any, Optional

import numpy as np
import torch

from protenix.utils.logger import get_logger

logger = get_logger(__name__)


class SSFeaturizer:
    """Loads pre-computed BPP matrices and derives crop-aware SS features.

    Supports two input formats:
      1. Directory of .npz files (key="bpp") indexed by a JSON mapping file
      2. Directory of .npy files indexed by a JSON mapping file

    The JSON mapping file maps RNA entity sequences to BPP file paths:
      { "AUGCAUGC...": "path/to/bpp.npz", ... }

    Args:
        bpp_dir: Root directory containing BPP matrix files.
        index_path: Path to JSON mapping file (sequence -> relative path).
        n_pair_channels: Number of pair feature channels (default 4).
        n_single_channels: Number of single feature channels (default 3).
        boundary_margin: Number of tokens from crop edge for boundary mask decay.
    """

    def __init__(
        self,
        bpp_dir: str,
        index_path: str,
        n_pair_channels: int = 4,
        n_single_channels: int = 3,
        boundary_margin: int = 10,
    ):
        self.bpp_dir = bpp_dir
        self.n_pair_channels = n_pair_channels
        self.n_single_channels = n_single_channels
        self.boundary_margin = boundary_margin

        # Load index mapping
        self.seq_to_path = {}
        if index_path and os.path.exists(index_path):
            with open(index_path, "r") as f:
                self.seq_to_path = json.load(f)
            logger.info(
                f"SS featurizer: loaded {len(self.seq_to_path)} sequence mappings "
                f"from {index_path}"
            )
        else:
            logger.warning(
                f"SS featurizer: index_path '{index_path}' not found or empty. "
                "All RNA chains will get zero SS features."
            )

        # Cache loaded BPP matrices (sequence -> numpy array)
        self._cache = {}

    def _load_bpp(self, seq: str) -> Optional[np.ndarray]:
        """Load BPP matrix for a given RNA sequence.

        Returns:
            BPP matrix of shape [L, L] or None if not found.
        """
        if seq in self._cache:
            return self._cache[seq]

        rel_path = self.seq_to_path.get(seq)
        if rel_path is None:
            return None

        fpath = os.path.join(self.bpp_dir, rel_path) if self.bpp_dir else rel_path
        if not os.path.exists(fpath):
            logger.warning(f"SS BPP file not found: {fpath}")
            return None

        try:
            if fpath.endswith(".npz"):
                data = np.load(fpath)
                bpp = data["bpp"] if "bpp" in data else data[list(data.keys())[0]]
            elif fpath.endswith(".npy"):
                bpp = np.load(fpath)
            else:
                logger.warning(f"Unsupported BPP file format: {fpath}")
                return None

            bpp = bpp.astype(np.float32)
            # Symmetrize if needed
            if bpp.ndim == 2 and bpp.shape[0] == bpp.shape[1]:
                bpp = (bpp + bpp.T) / 2.0
            self._cache[seq] = bpp
            return bpp
        except Exception as e:
            logger.warning(f"Failed to load BPP from {fpath}: {e}")
            return None

    @staticmethod
    def _get_entity_id_to_sequence(
        bioassembly_dict: dict,
        inference_mode: bool,
    ) -> dict[str, str]:
        """Normalize entity-to-sequence lookup across training and inference inputs."""
        sequences = bioassembly_dict.get("sequences", {})

        if isinstance(sequences, dict):
            return {str(entity_id): seq for entity_id, seq in sequences.items()}

        if inference_mode and isinstance(sequences, list):
            entity_id_to_sequence = {}
            for i, entity_info_wrapper in enumerate(sequences):
                if not isinstance(entity_info_wrapper, dict) or len(entity_info_wrapper) != 1:
                    continue
                _, entity_info = next(iter(entity_info_wrapper.items()))
                if "sequence" in entity_info:
                    entity_id_to_sequence[str(i + 1)] = entity_info["sequence"]
            return entity_id_to_sequence

        return {}

    @staticmethod
    def _compute_boundary_mask(n_crop: int, margin: int) -> np.ndarray:
        """Compute boundary decay mask for crop positions.

        Tokens near the edge of the crop get values close to 1 (high uncertainty),
        tokens far from edges get values close to 0 (low uncertainty).
        """
        if margin <= 0 or n_crop == 0:
            return np.zeros(n_crop, dtype=np.float32)

        pos = np.arange(n_crop, dtype=np.float32)
        dist_from_edge = np.minimum(pos, n_crop - 1 - pos)
        boundary = np.clip(1.0 - dist_from_edge / margin, 0.0, 1.0)
        return boundary

    @staticmethod
    def _derive_crop_features(
        bpp: np.ndarray,
        crop_positions: np.ndarray,
        boundary_margin: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Derive crop-aware pair and single features from full BPP.

        Args:
            bpp: Full-length BPP matrix [L, L].
            crop_positions: 0-based indices of cropped tokens in full sequence.
            boundary_margin: Margin for boundary mask.

        Returns:
            pair_feat: [n_crop, n_crop, 4]
            single_feat: [n_crop, 3]
        """
        full_L = bpp.shape[0]
        n_crop = len(crop_positions)

        # W = set of crop positions
        W = set(crop_positions.tolist())

        # 1. P_in: crop-internal BPP submatrix
        P_in = bpp[np.ix_(crop_positions, crop_positions)]  # [n_crop, n_crop]

        # 2. p_out(i): sum of BPP with positions outside crop
        outside_mask = np.array([p not in W for p in range(full_L)], dtype=bool)
        p_out = np.zeros(n_crop, dtype=np.float32)
        for local_idx, global_pos in enumerate(crop_positions):
            p_out[local_idx] = bpp[global_pos, outside_mask].sum()

        # 3. p_in(i): sum of BPP within crop
        p_in_single = P_in.sum(axis=1)  # [n_crop]

        # 4. p_unp(i): unpaired probability = 1 - sum over ALL positions
        p_total = bpp[crop_positions, :].sum(axis=1)  # [n_crop]
        p_unp = np.clip(1.0 - p_total, 0.0, 1.0)

        # 5. O_ij: outside mass for pair
        O_ij = (p_out[:, None] + p_out[None, :]) / 2.0  # [n_crop, n_crop]

        # 6. B_ij: boundary mask
        boundary_1d = SSFeaturizer._compute_boundary_mask(n_crop, boundary_margin)
        B_ij = np.maximum(boundary_1d[:, None], boundary_1d[None, :])

        # Stack pair features
        pair_feat = np.stack([P_in, 1.0 - P_in, O_ij, B_ij], axis=-1)  # [n_crop, n_crop, 4]

        # Stack single features
        single_feat = np.stack([p_out, p_in_single, p_unp], axis=-1)  # [n_crop, 3]

        return pair_feat, single_feat

    def __call__(
        self,
        token_array,
        atom_array: Any,
        bioassembly_dict: dict,
        inference_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Generate crop-aware SS features for the given (cropped) token array.

        Uses bioassembly_dict["sequences"][entity_id] for full-sequence BPP lookup
        (same pattern as RiNALMo featurizer). Position mapping uses res_id.

        Args:
            token_array: The (cropped) token array.
            atom_array: The (cropped) atom array.
            bioassembly_dict: Bioassembly metadata with "sequences" dict.
            inference_mode: If True, uses entity_id_to_sequence for lookup.

        Returns:
            Dict with keys:
              - "ss_pair_feat": [N_token, N_token, n_pair_channels] float32
              - "ss_single_feat": [N_token, n_single_channels] float32
              - "ss_mask": [N_token] bool, True for RNA tokens with valid SS data
        """
        n_tokens = len(token_array)

        # Initialize zero features
        pair_feat = np.zeros(
            (n_tokens, n_tokens, self.n_pair_channels), dtype=np.float32
        )
        single_feat = np.zeros(
            (n_tokens, self.n_single_channels), dtype=np.float32
        )
        ss_mask = np.zeros(n_tokens, dtype=bool)

        # Get per-token annotations
        centre_atom_indices = token_array.get_annotation("centre_atom_index")
        centre_atoms = atom_array[centre_atom_indices]

        # Identify RNA tokens using mol_type from atom_array
        # (more reliable than restype, consistent with RibonanzaNet tokenizer)
        is_rna = np.array(
            [getattr(centre_atoms, 'mol_type', np.array([''] * n_tokens))[i] == 'rna'
             for i in range(n_tokens)],
            dtype=bool,
        ) if hasattr(centre_atoms, 'mol_type') else np.zeros(n_tokens, dtype=bool)

        # Fallback: use restype if mol_type not available
        if not is_rna.any():
            restypes = token_array.get_annotation("restype")
            RNA_RESTYPE_INDICES = {20, 21, 22, 23}
            is_rna = np.array([rt in RNA_RESTYPE_INDICES for rt in restypes], dtype=bool)

        if not is_rna.any():
            return {
                "ss_pair_feat": torch.from_numpy(pair_feat),
                "ss_single_feat": torch.from_numpy(single_feat),
                "ss_mask": torch.from_numpy(ss_mask),
            }

        # Group RNA tokens by label_entity_id (for full-seq lookup)
        entity_ids = centre_atoms.label_entity_id if hasattr(centre_atoms, 'label_entity_id') else centre_atoms.entity_id
        rna_entities = {}  # entity_id -> list of token indices
        for tok_idx in range(n_tokens):
            if is_rna[tok_idx]:
                eid = str(entity_ids[tok_idx])
                if eid not in rna_entities:
                    rna_entities[eid] = []
                rna_entities[eid].append(tok_idx)

        entity_id_to_sequence = self._get_entity_id_to_sequence(
            bioassembly_dict=bioassembly_dict,
            inference_mode=inference_mode,
        )

        for entity_id, tok_indices in rna_entities.items():
            tok_indices = sorted(tok_indices)

            # Get full sequence for BPP lookup
            full_seq = entity_id_to_sequence.get(entity_id)
            if full_seq is None:
                # Fallback: reconstruct from cropped tokens
                restypes = token_array.get_annotation("restype")
                idx_to_nt = {20: "A", 21: "G", 22: "C", 23: "U"}
                full_seq = "".join(idx_to_nt.get(restypes[ti], "N") for ti in tok_indices)

            # Load BPP matrix
            bpp = self._load_bpp(full_seq)
            if bpp is None:
                continue

            full_L = bpp.shape[0]

            # Map cropped tokens to positions in full BPP matrix using res_id
            res_ids = centre_atoms.res_id[tok_indices]
            # Convert to 0-based positions
            crop_positions = np.array(res_ids - 1, dtype=np.int64)
            # Clamp to valid range
            crop_positions = np.clip(crop_positions, 0, full_L - 1)

            # Derive crop-aware features
            chain_pair, chain_single = self._derive_crop_features(
                bpp, crop_positions, self.boundary_margin
            )

            # Fill into output arrays at correct token positions
            idx_arr = np.array(tok_indices)
            pair_feat[np.ix_(idx_arr, idx_arr)] = chain_pair
            single_feat[idx_arr] = chain_single
            ss_mask[idx_arr] = True

        return {
            "ss_pair_feat": torch.from_numpy(pair_feat),
            "ss_single_feat": torch.from_numpy(single_feat),
            "ss_mask": torch.from_numpy(ss_mask),
        }
