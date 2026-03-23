import csv
import os
from typing import Any, Optional

import numpy as np
import torch

from protenix.data.constants import RNA_STD_RESIDUES
from protenix.utils.logger import get_logger

logger = get_logger(__name__)

TOKEN_VALUE_TO_RNA_BASE = {
    value: base
    for base, value in RNA_STD_RESIDUES.items()
    if base in {"A", "G", "C", "U", "N"}
}
RNA_SEQUENCE_TYPES = {"rnaSequence", "rnaChain"}


class RNASSFeaturizer:
    """Load RNA secondary-structure priors and build pair-only substructure features.

    Output layout is fixed to six channels:
      0. P_in(i, j): crop-internal pair prior
      1. o_i: total outside mass broadcast on rows
      2. o_j: total outside mass broadcast on cols
      3. r_i: crop coverage reliability broadcast on rows
      4. r_j: crop coverage reliability broadcast on cols
      5. m_ij: valid same-chain RNA prior mask
    """

    def __init__(
        self,
        sequence_fpath: str,
        feature_dir: str = "",
        format: str = "sparse_npz",
        n_classes: int = 6,
        coverage_window: int = 8,
        strict: bool = False,
        min_prob: float = 0.0,
    ) -> None:
        if n_classes != 6:
            raise ValueError(f"RNASSFeaturizer expects n_classes=6, got {n_classes}")

        self.sequence_fpath = sequence_fpath
        self.feature_dir = feature_dir
        self.format = format
        self.n_classes = n_classes
        self.coverage_window = int(coverage_window)
        self.strict = bool(strict)
        self.min_prob = float(min_prob)
        self.seq_to_path = self._load_sequence_index(sequence_fpath)
        self._cache: dict[str, dict[str, np.ndarray]] = {}

    @staticmethod
    def _normalize_entity_sequences(entity_to_sequences: Any) -> dict[str, str]:
        if isinstance(entity_to_sequences, dict):
            return {str(entity_id): str(seq) for entity_id, seq in entity_to_sequences.items()}

        if isinstance(entity_to_sequences, list):
            normalized = {}
            for idx, entity_wrapper in enumerate(entity_to_sequences):
                if not isinstance(entity_wrapper, dict) or len(entity_wrapper) != 1:
                    continue
                entity_type, entity_info = next(iter(entity_wrapper.items()))
                if entity_type in RNA_SEQUENCE_TYPES and isinstance(entity_info, dict):
                    normalized[str(idx + 1)] = str(entity_info.get("sequence", ""))
            return normalized

        return {}

    @staticmethod
    def _sequence_from_tokens(token_array, token_indices: np.ndarray) -> str:
        sequence = []
        for token_idx in token_indices.tolist():
            base = TOKEN_VALUE_TO_RNA_BASE.get(token_array[token_idx].value, "N")
            sequence.append(base)
        return "".join(sequence)

    def _load_sequence_index(self, sequence_fpath: str) -> dict[str, str]:
        if not sequence_fpath:
            if self.strict:
                raise ValueError("rna_ss.sequence_fpath is required when rna_ss.enable=True")
            logger.warning("rna_ss.sequence_fpath is empty; RNA SS priors will fall back to zeros.")
            return {}

        if not os.path.exists(sequence_fpath):
            if self.strict:
                raise FileNotFoundError(f"rna_ss.sequence_fpath not found: {sequence_fpath}")
            logger.warning(
                f"rna_ss.sequence_fpath '{sequence_fpath}' does not exist; priors will fall back to zeros."
            )
            return {}

        seq_to_path = {}
        with open(sequence_fpath, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames and "sequence" in reader.fieldnames and "path" in reader.fieldnames:
                for row in reader:
                    seq = str(row.get("sequence", "")).strip()
                    path = str(row.get("path", "")).strip()
                    if seq and path:
                        seq_to_path[seq] = path
            else:
                handle.seek(0)
                raw_reader = csv.reader(handle)
                for row in raw_reader:
                    if len(row) < 2:
                        continue
                    seq = str(row[0]).strip()
                    path = str(row[1]).strip()
                    if seq and path and seq.lower() != "sequence":
                        seq_to_path[seq] = path

        logger.info(
            f"RNA SS featurizer loaded {len(seq_to_path)} sequence-path mappings from {sequence_fpath}"
        )
        return seq_to_path

    def _resolve_feature_path(self, feature_path: str) -> str:
        if os.path.isabs(feature_path):
            return feature_path
        if self.feature_dir:
            return os.path.join(self.feature_dir, feature_path)
        return feature_path

    def _load_prior(self, sequence: str) -> Optional[dict[str, np.ndarray]]:
        if sequence in self._cache:
            return self._cache[sequence]

        feature_path = self.seq_to_path.get(sequence)
        if feature_path is None:
            if self.strict:
                raise KeyError(f"No RNA SS prior path found for sequence '{sequence[:32]}...'")
            return None

        resolved_path = self._resolve_feature_path(feature_path)
        if not os.path.exists(resolved_path):
            if self.strict:
                raise FileNotFoundError(f"RNA SS prior file not found: {resolved_path}")
            logger.warning(f"RNA SS prior file not found: {resolved_path}")
            return None

        prior = self._load_npz_prior(resolved_path)
        self._cache[sequence] = prior
        return prior

    def _load_npz_prior(self, feature_path: str) -> dict[str, np.ndarray]:
        with np.load(feature_path, allow_pickle=False) as data:
            keys = set(data.keys())

            if self.format == "dense_npz" or "bpp" in keys:
                bpp = data["bpp"] if "bpp" in data else data[list(data.keys())[0]]
                bpp = np.asarray(bpp, dtype=np.float32)
                if bpp.ndim != 2 or bpp.shape[0] != bpp.shape[1]:
                    raise ValueError(f"Invalid dense RNA SS matrix shape: {bpp.shape}")
                bpp = (bpp + bpp.T) / 2.0
                if self.min_prob > 0:
                    bpp = np.where(bpp >= self.min_prob, bpp, 0.0)
                row_sum = np.asarray(
                    data["row_sum"] if "row_sum" in data else bpp.sum(axis=1),
                    dtype=np.float32,
                )
                if self.min_prob > 0 and "row_sum" in data:
                    row_sum = bpp.sum(axis=1).astype(np.float32)
                return {
                    "format": "dense",
                    "bpp": bpp.astype(np.float32),
                    "row_sum": np.clip(row_sum.astype(np.float32), 0.0, 1.0),
                }

            if self.format == "sparse_npz" or {"pair_i", "pair_j", "pair_p"}.issubset(keys):
                pair_i = np.asarray(data["pair_i"], dtype=np.int64)
                pair_j = np.asarray(data["pair_j"], dtype=np.int64)
                pair_p = np.asarray(data["pair_p"], dtype=np.float32)
                if self.min_prob > 0:
                    keep = pair_p >= self.min_prob
                    pair_i = pair_i[keep]
                    pair_j = pair_j[keep]
                    pair_p = pair_p[keep]

                if "row_sum" in data and self.min_prob <= 0:
                    row_sum = np.asarray(data["row_sum"], dtype=np.float32)
                else:
                    if "length" in data:
                        seq_len = int(data["length"])
                    elif len(pair_i) > 0:
                        seq_len = int(max(np.max(pair_i), np.max(pair_j)) + 1)
                    else:
                        seq_len = 0
                    row_sum = np.zeros(seq_len, dtype=np.float32)
                    np.add.at(row_sum, pair_i, pair_p)

                return {
                    "format": "sparse",
                    "pair_i": pair_i,
                    "pair_j": pair_j,
                    "pair_p": pair_p,
                    "row_sum": np.clip(row_sum.astype(np.float32), 0.0, 1.0),
                }

        raise ValueError(
            f"Unsupported RNA SS prior format at {feature_path}. Expected dense_npz or sparse_npz."
        )

    @staticmethod
    def _get_local_matrix(prior: dict[str, np.ndarray], positions: np.ndarray) -> np.ndarray:
        n_token = len(positions)
        if n_token == 0:
            return np.zeros((0, 0), dtype=np.float32)

        if prior["format"] == "dense":
            return np.asarray(prior["bpp"][np.ix_(positions, positions)], dtype=np.float32)

        local = np.zeros((n_token, n_token), dtype=np.float32)
        pos_to_local = {int(pos): idx for idx, pos in enumerate(positions.tolist())}
        pair_i = prior["pair_i"]
        pair_j = prior["pair_j"]
        pair_p = prior["pair_p"]
        for i, j, p in zip(pair_i.tolist(), pair_j.tolist(), pair_p.tolist()):
            if i in pos_to_local and j in pos_to_local:
                local[pos_to_local[i], pos_to_local[j]] = p
        return local

    def _compute_coverage(self, sequence_length: int, positions: np.ndarray) -> np.ndarray:
        n_token = len(positions)
        if n_token == 0:
            return np.zeros(0, dtype=np.float32)
        if self.coverage_window <= 0:
            return np.ones(n_token, dtype=np.float32)

        present = np.zeros(sequence_length, dtype=np.float32)
        present[positions] = 1.0
        prefix = np.pad(np.cumsum(present), (1, 0))
        reliability = np.zeros(n_token, dtype=np.float32)
        for idx, pos in enumerate(positions.tolist()):
            left = max(0, pos - self.coverage_window)
            right = min(sequence_length - 1, pos + self.coverage_window)
            covered = prefix[right + 1] - prefix[left]
            reliability[idx] = covered / max(right - left + 1, 1)
        return np.clip(reliability, 0.0, 1.0)

    def _build_chain_substructure(
        self,
        prior: dict[str, np.ndarray],
        positions: np.ndarray,
    ) -> np.ndarray:
        sequence_length = int(prior["row_sum"].shape[0])
        p_in = np.clip(self._get_local_matrix(prior, positions), 0.0, 1.0)
        row_sum = np.clip(prior["row_sum"][positions], 0.0, 1.0)
        outside_mass = np.clip(row_sum - p_in.sum(axis=1), 0.0, 1.0)
        reliability = self._compute_coverage(sequence_length, positions)
        valid_mask = np.ones((len(positions), len(positions)), dtype=np.float32)
        return np.stack(
            [
                p_in,
                np.broadcast_to(outside_mass[:, None], (len(positions), len(positions))),
                np.broadcast_to(outside_mass[None, :], (len(positions), len(positions))),
                np.broadcast_to(reliability[:, None], (len(positions), len(positions))),
                np.broadcast_to(reliability[None, :], (len(positions), len(positions))),
                valid_mask,
            ],
            axis=-1,
        ).astype(np.float32)

    def __call__(
        self,
        full_token_array,
        full_atom_array: Any,
        cropped_token_array,
        cropped_atom_array: Any,
        entity_to_sequences: Any,
        selected_token_indices: Optional[np.ndarray] = None,
    ) -> dict[str, torch.Tensor]:
        entity_to_sequences = self._normalize_entity_sequences(entity_to_sequences)
        n_token = len(cropped_token_array)
        substructure = np.zeros((n_token, n_token, self.n_classes), dtype=np.float32)

        full_centre_idx = np.asarray(full_token_array.get_annotation("centre_atom_index"), dtype=np.int64)
        full_centre_atoms = full_atom_array[full_centre_idx]
        selected_token_indices = (
            np.arange(len(full_token_array), dtype=np.int64)
            if selected_token_indices is None
            else np.asarray(selected_token_indices, dtype=np.int64)
        )
        full_to_crop = {int(full_idx): crop_idx for crop_idx, full_idx in enumerate(selected_token_indices.tolist())}

        chain_mol_type = getattr(full_centre_atoms, "chain_mol_type", None)
        if chain_mol_type is not None:
            is_rna = np.asarray(chain_mol_type == "rna", dtype=bool)
        else:
            mol_type = getattr(full_centre_atoms, "mol_type", None)
            is_rna = np.asarray(mol_type == "rna", dtype=bool) if mol_type is not None else np.zeros(len(full_centre_atoms), dtype=bool)

        if not is_rna.any():
            return {"substructure": torch.from_numpy(substructure)}

        chain_id_values = getattr(full_centre_atoms, "label_asym_id", None)
        if chain_id_values is None:
            chain_id_values = getattr(full_centre_atoms, "chain_id")
        entity_id_values = getattr(full_centre_atoms, "label_entity_id", None)
        if entity_id_values is None:
            entity_id_values = getattr(full_centre_atoms, "entity_id")

        chain_ids = np.asarray(chain_id_values, dtype=object)
        entity_ids = np.asarray(entity_id_values, dtype=object)
        res_ids = np.asarray(getattr(full_centre_atoms, "res_id"), dtype=np.int64)

        for chain_id in np.unique(chain_ids[is_rna]):
            chain_mask = is_rna & (chain_ids == chain_id)
            chain_full_token_indices = np.where(chain_mask)[0]
            if len(chain_full_token_indices) == 0:
                continue

            entity_id = str(entity_ids[chain_full_token_indices[0]])
            full_sequence = entity_to_sequences.get(entity_id)
            if not full_sequence:
                full_sequence = self._sequence_from_tokens(full_token_array, chain_full_token_indices)

            prior = self._load_prior(full_sequence)
            if prior is None:
                continue

            chain_positions = res_ids[chain_full_token_indices] - 1
            if (
                len(full_sequence) == len(chain_full_token_indices)
                and (
                    np.any(chain_positions < 0)
                    or np.any(chain_positions >= len(full_sequence))
                    or len(np.unique(chain_positions)) != len(chain_positions)
                )
            ):
                chain_positions = np.arange(len(chain_full_token_indices), dtype=np.int64)

            selected_mask = np.isin(chain_full_token_indices, selected_token_indices)
            crop_full_token_indices = chain_full_token_indices[selected_mask]
            crop_positions = chain_positions[selected_mask]

            valid_mask = (crop_positions >= 0) & (crop_positions < prior["row_sum"].shape[0])
            if not valid_mask.all():
                if self.strict:
                    raise ValueError(
                        f"RNA SS positions out of range for entity {entity_id}: "
                        f"{crop_positions[~valid_mask].tolist()}"
                    )
                crop_full_token_indices = crop_full_token_indices[valid_mask]
                crop_positions = crop_positions[valid_mask]

            if len(crop_positions) == 0:
                continue

            crop_local_indices = np.asarray(
                [full_to_crop[int(full_idx)] for full_idx in crop_full_token_indices.tolist()],
                dtype=np.int64,
            )
            chain_substructure = self._build_chain_substructure(prior, crop_positions.astype(np.int64))
            substructure[np.ix_(crop_local_indices, crop_local_indices)] = chain_substructure

        return {"substructure": torch.from_numpy(substructure)}
