"""
RiNALMo Featurizer - Analogous to ESMFeaturizer but for RNA/DNA Language Model embeddings.

Loads pre-computed RNA and/or DNA embeddings from disk and aligns them with cropped tokens.
RNA tokens receive RNA LM embeddings; DNA tokens receive DNA LM embeddings;
protein/ligand tokens get zeros.

Pipeline:
    1. Pre-compute embeddings for full RNA/DNA sequences: [N_full, embedding_dim]
    2. Data pipeline crops tokens (mixed protein/DNA/RNA/ligand): [N_crop, ...]
    3. Extract RNA/DNA token indices from cropped tokens
    4. Slice embeddings for the cropped residues: [N_crop_rna/dna, embedding_dim]
    5. Place into full-size tensor with zeros for non-RNA/DNA tokens: [N_crop, embedding_dim]
    6. Model projects to model dim: [N_crop, 384]

Supports two modes:
    - Combined (legacy): single tensor [N_crop, embedding_dim] with DNA zero-padded
    - Separate: dict with "rna_llm_embedding" [N_crop, rna_dim] and "dna_llm_embedding" [N_crop, dna_dim]
"""

import os
import traceback

import pandas as pd
import torch

from protenix.utils.logger import get_logger

logger = get_logger(__name__)


class RiNALMoFeaturizer:
    """
    Loads pre-computed RNA (and optionally DNA) embeddings and creates per-token
    embedding tensors.

    Analogous to ESMFeaturizer but filters for RNA/DNA tokens instead of protein tokens.
    Independent from ESMFeaturizer so both can be used simultaneously.

    Includes cross-manifest fallback: when a sequence is classified as DNA by the
    featurizer but only exists in the RNA manifest (or vice versa), the featurizer
    will try the other manifest with U<->T base conversion. This handles hybrid
    DNA/RNA entities and CCD classification mismatches.

    Args:
        embedding_dir: Directory containing pre-computed RNA embedding .pt files.
        sequence_fpath: CSV file mapping RNA sequences to embedding file paths.
        embedding_dim: Embedding dimension (1280 for RiNALMo, 2048 for AIDO.RNA).
        error_dir: Optional directory for saving error logs.
        dna_embedding_dir: Directory containing pre-computed DNA embedding .pt files.
        dna_sequence_fpath: CSV file mapping DNA sequences to embedding file paths.
        dna_embedding_dim: Native DNA embedding dimension (1024 for AIDO.DNA).
            Only used when return_separate=True in __call__.
    """

    def __init__(
        self,
        embedding_dir: str = "",
        sequence_fpath: str = "",
        embedding_dim: int = 1280,
        error_dir: str = None,
        dna_embedding_dir: str = "",
        dna_sequence_fpath: str = "",
        dna_embedding_dim: int = 1024,
        use_rna_embed: bool = True,
        use_dna_embed: bool = True,
    ):
        self.use_rna_embed = use_rna_embed
        self.use_dna_embed = use_dna_embed
        self.embedding_dim = embedding_dim
        self.dna_embedding_dim = dna_embedding_dim
        self.error_dir = error_dir
        if self.error_dir is not None:
            self.error_dir = os.path.join(self.error_dir, "rnalm_error")
            os.makedirs(self.error_dir, exist_ok=True)

        # === RNA Embedding Support (fail-fast: error if paths missing) ===
        self.rna_enable = False
        self.seq_to_filename = {}
        if use_rna_embed:
            if not embedding_dir or not sequence_fpath:
                raise ValueError(
                    "use_rna_embed=True but embedding_dir or sequence_fpath not configured. "
                    f"embedding_dir='{embedding_dir}', sequence_fpath='{sequence_fpath}'"
                )
            if not os.path.exists(embedding_dir):
                raise FileNotFoundError(
                    f"RNA embedding directory not found: {embedding_dir}"
                )
            if not os.path.exists(sequence_fpath):
                raise FileNotFoundError(
                    f"RNA sequence CSV not found: {sequence_fpath}"
                )
            self.embedding_dir = embedding_dir
            self.sequence_fpath = sequence_fpath
            self.seq_to_filename = self.get_seq_to_filename(sequence_fpath)
            self.rna_enable = True
            logger.info(
                f"RNA embedding enabled: dir={embedding_dir}, "
                f"entries={len(self.seq_to_filename)}"
            )
        else:
            logger.info("RNA embedding disabled by use_rna_embed=False")

        # === DNA Embedding Support (fail-fast: error if paths missing) ===
        self.dna_enable = False
        self.dna_embedding_dir = dna_embedding_dir
        self.dna_sequence_fpath = dna_sequence_fpath
        if use_dna_embed:
            if not dna_embedding_dir or not dna_sequence_fpath:
                raise ValueError(
                    "use_dna_embed=True but dna_embedding_dir or dna_sequence_fpath not configured. "
                    f"dna_embedding_dir='{dna_embedding_dir}', dna_sequence_fpath='{dna_sequence_fpath}'"
                )
            if not os.path.exists(dna_embedding_dir):
                raise FileNotFoundError(
                    f"DNA embedding directory not found: {dna_embedding_dir}"
                )
            if not os.path.exists(dna_sequence_fpath):
                raise FileNotFoundError(
                    f"DNA sequence CSV not found: {dna_sequence_fpath}"
                )
            self.dna_seq_to_filename = self.get_seq_to_filename(dna_sequence_fpath)
            self.dna_enable = True
            logger.info(
                f"DNA embedding enabled: dir={dna_embedding_dir}, "
                f"entries={len(self.dna_seq_to_filename)}"
            )
        else:
            logger.info("DNA embedding disabled by use_dna_embed=False")

    def get_seq_to_filename(self, sequence_fpath: str) -> dict[str, str]:
        """Build mapping from sequence string to embedding file path."""
        df = pd.read_csv(sequence_fpath)
        df["filename"] = (
            df["part_id"].astype(str) + "/" + df["seq_label"].astype(str) + ".pt"
        )
        return df.set_index("seq")["filename"].to_dict()

    def load_rnalm_embedding(self, sequence: str) -> torch.Tensor:
        """
        Load a pre-computed RNA embedding for a given RNA sequence.

        Args:
            sequence: The RNA sequence string.

        Returns:
            Tensor of shape [seq_len, embedding_dim].
        """
        x = torch.load(
            os.path.join(self.embedding_dir, self.seq_to_filename[sequence]),
            weights_only=True,
        )
        assert x.size(0) == len(sequence), (
            f"RNA embedding size {x.size(0)} not equal to sequence length {len(sequence)}. "
            "This may occur if embeddings were computed for a different sequence. "
            "Delete the embeddings directory and re-compute."
        )
        return x

    def load_dna_embedding(self, sequence: str) -> torch.Tensor:
        """
        Load a pre-computed DNA embedding for a given DNA sequence.
        If the DNA embedding dim is smaller than self.embedding_dim,
        zero-pad to match (e.g., AIDO.DNA 1024 -> AIDO.RNA 2048).

        Args:
            sequence: The DNA sequence string.

        Returns:
            Tensor of shape [seq_len, embedding_dim].
        """
        x = torch.load(
            os.path.join(self.dna_embedding_dir, self.dna_seq_to_filename[sequence]),
            weights_only=True,
        )
        assert x.size(0) == len(sequence), (
            f"DNA embedding size {x.size(0)} not equal to sequence length {len(sequence)}. "
            "This may occur if embeddings were computed for a different sequence. "
            "Delete the embeddings directory and re-compute."
        )
        # Zero-pad if DNA embedding dim < RNA embedding dim
        if x.size(1) < self.embedding_dim:
            pad = torch.zeros(x.size(0), self.embedding_dim - x.size(1))
            x = torch.cat([x, pad], dim=1)
        return x

    def load_dna_embedding_native(self, sequence: str) -> torch.Tensor:
        """
        Load a pre-computed DNA embedding WITHOUT zero-padding.
        Returns the native DNA embedding dimension (e.g., 1024 for AIDO.DNA).

        Args:
            sequence: The DNA sequence string.

        Returns:
            Tensor of shape [seq_len, native_dna_dim].
        """
        x = torch.load(
            os.path.join(self.dna_embedding_dir, self.dna_seq_to_filename[sequence]),
            weights_only=True,
        )
        assert x.size(0) == len(sequence), (
            f"DNA embedding size {x.size(0)} not equal to sequence length {len(sequence)}. "
            "This may occur if embeddings were computed for a different sequence. "
            "Delete the embeddings directory and re-compute."
        )
        return x

    # === Cross-manifest fallback helpers ===

    @staticmethod
    def _u_to_t(seq: str) -> str:
        """Convert RNA sequence (U) to DNA (T)."""
        return seq.replace("U", "T").replace("u", "t")

    @staticmethod
    def _t_to_u(seq: str) -> str:
        """Convert DNA sequence (T) to RNA (U)."""
        return seq.replace("T", "U").replace("t", "u")

    @staticmethod
    def _strip_modified(seq: str) -> str:
        """Strip modified nucleotide placeholders (X) from sequence for manifest lookup.
        Modified bases from CCD (pseudouridine, methylated bases, etc.) are converted
        to 'X' by the parser. Embeddings are generated from the original PDB sequence
        which uses standard bases, so we strip X to attempt a match."""
        return seq.replace("X", "").replace("x", "")

    def _try_load_with_stripped_modified(
        self, sequence: str, manifest: dict, emb_dir: str, target_dim: int
    ) -> torch.Tensor | None:
        """Try to load embedding after stripping modified base placeholders (X).
        Returns None if no match found or if stripping changes nothing."""
        stripped = self._strip_modified(sequence)
        if stripped == sequence or not stripped:
            return None

        # Try direct match of stripped sequence
        if stripped in manifest:
            logger.info(
                f"[Modified-Base] Loading after stripping X chars: "
                f"'{sequence[:30]}...' -> '{stripped[:30]}...'"
            )
            x = torch.load(
                os.path.join(emb_dir, manifest[stripped]),
                weights_only=True,
            )
            return x

        # Try T->U conversion of stripped sequence
        stripped_rna = self._t_to_u(stripped)
        if stripped_rna != stripped and stripped_rna in manifest:
            logger.info(
                f"[Modified-Base] Loading after stripping X + T->U: "
                f"'{sequence[:30]}...' -> '{stripped_rna[:30]}...'"
            )
            x = torch.load(
                os.path.join(emb_dir, manifest[stripped_rna]),
                weights_only=True,
            )
            return x

        return None

    def load_rnalm_embedding_with_fallback(self, sequence: str) -> torch.Tensor:
        """
        Load RNA embedding with RNA-First fallback strategy.

        Priority order:
            1. Direct match in RNA manifest (original sequence)
            2. T->U converted sequence in RNA manifest (for DNA-labeled seqs
               reclassified to RNA, e.g. hybrid chains)
            3. DNA manifest fallback (ONLY for pure ACGT sequences without uracil)
            4. Modified-base fallback: strip X chars and retry RNA/DNA manifests

        SAFETY: sequences containing uracil will NEVER fall back to DNA manifest.
        This ensures RNA structural priors (A-form helix, 2'-OH, pseudoknots)
        are preserved.
        """
        # Priority 1: Direct match in RNA manifest
        if sequence in self.seq_to_filename:
            return self.load_rnalm_embedding(sequence)

        # Priority 2: T->U conversion, check RNA manifest
        # This handles DNA-labeled sequences reclassified to RNA by RNA-First.
        # Their canonical sequence may use T (from CIF DNA convention) but the
        # RNA manifest stores the U version.
        rna_seq = self._t_to_u(sequence)
        if rna_seq != sequence and rna_seq in self.seq_to_filename:
            logger.info(
                f"[RNA-First] Loading from RNA manifest with T->U conversion: "
                f"{sequence[:30]}..."
            )
            x = torch.load(
                os.path.join(self.embedding_dir, self.seq_to_filename[rna_seq]),
                weights_only=True,
            )
            # Length assertion: T->U is 1:1 so lengths must match
            assert x.size(0) == len(rna_seq), (
                f"RNA embedding size {x.size(0)} != T->U seq length {len(rna_seq)}"
            )
            return x

        # Priority 3: DNA manifest fallback — ONLY for pure ACGT sequences
        # Sequences containing uracil must NEVER use the DNA model.
        if self.dna_enable and self._is_pure_dna(sequence):
            dna_seq = self._u_to_t(sequence)  # No-op for pure ACGT
            if dna_seq in self.dna_seq_to_filename:
                logger.info(
                    f"[RNA-First] Pure ACGT sequence, loading from DNA manifest "
                    f"(pad to RNA dim): {sequence[:30]}..."
                )
                x = torch.load(
                    os.path.join(self.dna_embedding_dir, self.dna_seq_to_filename[dna_seq]),
                    weights_only=True,
                )
                # Length assertion for cross-manifest load
                assert x.size(0) == len(dna_seq), (
                    f"DNA embedding size {x.size(0)} != seq length {len(dna_seq)}"
                )
                # Pad to RNA embedding dim if needed
                if x.size(1) < self.embedding_dim:
                    pad = torch.zeros(x.size(0), self.embedding_dim - x.size(1))
                    x = torch.cat([x, pad], dim=1)
                elif x.size(1) > self.embedding_dim:
                    x = x[:, :self.embedding_dim]
                return x

        # Priority 4: Modified-base fallback — strip X chars from modified nucleotides
        # Modified bases (pseudouridine PSU, methylated m5C, etc.) become 'X' in
        # the sequence but embeddings were generated from the standard-base version.
        if self._has_modified_base(sequence):
            x = self._try_load_with_stripped_modified(
                sequence, self.seq_to_filename, self.embedding_dir, self.embedding_dim
            )
            if x is not None:
                # Embedding length won't match sequence length (X chars stripped),
                # but that's handled by res_id-based indexing in _fill_entities.
                return x

            # Also try DNA manifest for stripped pure-ACGT sequences
            stripped = self._strip_modified(sequence)
            if self.dna_enable and stripped and self._is_pure_dna(stripped):
                dna_stripped = self._u_to_t(stripped)
                if dna_stripped in self.dna_seq_to_filename:
                    logger.info(
                        f"[Modified-Base] Loading from DNA manifest after stripping X: "
                        f"{sequence[:30]}..."
                    )
                    x = torch.load(
                        os.path.join(
                            self.dna_embedding_dir,
                            self.dna_seq_to_filename[dna_stripped],
                        ),
                        weights_only=True,
                    )
                    if x.size(1) < self.embedding_dim:
                        pad = torch.zeros(x.size(0), self.embedding_dim - x.size(1))
                        x = torch.cat([x, pad], dim=1)
                    elif x.size(1) > self.embedding_dim:
                        x = x[:, :self.embedding_dim]
                    return x

        raise KeyError(
            f"[RNA-First] Sequence not found in any manifest: {sequence[:50]}... "
            f"(has_uracil={self._has_uracil(sequence)}, "
            f"is_pure_dna={self._is_pure_dna(sequence)}, "
            f"has_modified={self._has_modified_base(sequence)})"
        )

    def load_dna_embedding_with_fallback(self, sequence: str) -> torch.Tensor:
        """
        Load DNA embedding (zero-padded) with fallback to RNA manifest.
        If the sequence is not in the DNA manifest, try T->U conversion in RNA manifest.

        SAFETY: Rejects sequences containing uracil — these should have been
        reclassified as RNA by _identify_entities (RNA-First strategy).
        """
        # Biological safety guard: uracil-containing sequences must not use DNA model
        if self._has_uracil(sequence):
            raise ValueError(
                f"[BIO SAFETY] Sequence contains uracil (U) but was routed to DNA "
                f"embedding loader. Uracil is an RNA-exclusive base — this sequence "
                f"should use the RNA model (AIDO.RNA) to preserve structural priors "
                f"(A-form helix, 2'-OH interactions). This indicates a bug in entity "
                f"classification. Seq: {sequence[:50]}..."
            )

        if self.dna_enable and sequence in self.dna_seq_to_filename:
            return self.load_dna_embedding(sequence)

        # Fallback: try RNA manifest with T->U (for pure ACGT sequences only)
        rna_seq = self._t_to_u(sequence)
        if rna_seq in self.seq_to_filename:
            logger.info(
                f"Fallback: DNA seq not found, loading from RNA manifest (T->U): "
                f"{sequence[:30]}..."
            )
            x = torch.load(
                os.path.join(self.embedding_dir, self.seq_to_filename[rna_seq]),
                weights_only=True,
            )
            # Length assertion for cross-manifest load
            assert x.size(0) == len(rna_seq), (
                f"RNA embedding size {x.size(0)} != T->U seq length {len(rna_seq)}"
            )
            # Ensure correct dim (truncate or pad to embedding_dim for combined mode)
            if x.size(1) < self.embedding_dim:
                pad = torch.zeros(x.size(0), self.embedding_dim - x.size(1))
                x = torch.cat([x, pad], dim=1)
            elif x.size(1) > self.embedding_dim:
                x = x[:, :self.embedding_dim]
            return x

        raise KeyError(f"Sequence not found in DNA or RNA manifest: {sequence[:50]}...")

    def load_dna_embedding_native_with_fallback(self, sequence: str) -> torch.Tensor:
        """
        Load DNA embedding (native dim) with fallback to RNA manifest.
        If the sequence is not in the DNA manifest, try T->U conversion in RNA manifest.

        SAFETY: Rejects sequences containing uracil — these should have been
        reclassified as RNA by _identify_entities (RNA-First strategy).
        """
        # Biological safety guard: uracil-containing sequences must not use DNA model
        if self._has_uracil(sequence):
            raise ValueError(
                f"[BIO SAFETY] Sequence contains uracil (U) but was routed to DNA "
                f"embedding loader (native). Uracil is an RNA-exclusive base — this "
                f"sequence should use the RNA model (AIDO.RNA). This indicates a bug "
                f"in entity classification. Seq: {sequence[:50]}..."
            )

        if self.dna_enable and sequence in self.dna_seq_to_filename:
            return self.load_dna_embedding_native(sequence)

        # Fallback: try RNA manifest with T->U (for pure ACGT sequences only)
        rna_seq = self._t_to_u(sequence)
        if rna_seq in self.seq_to_filename:
            logger.info(
                f"Fallback: DNA seq not found (native), loading from RNA manifest (T->U): "
                f"{sequence[:30]}..."
            )
            x = torch.load(
                os.path.join(self.embedding_dir, self.seq_to_filename[rna_seq]),
                weights_only=True,
            )
            # Length assertion for cross-manifest load
            assert x.size(0) == len(rna_seq), (
                f"RNA embedding size {x.size(0)} != T->U seq length {len(rna_seq)}"
            )
            # Truncate or pad to native DNA dim
            if x.size(1) < self.dna_embedding_dim:
                pad = torch.zeros(x.size(0), self.dna_embedding_dim - x.size(1))
                x = torch.cat([x, pad], dim=1)
            elif x.size(1) > self.dna_embedding_dim:
                x = x[:, :self.dna_embedding_dim]
            return x

        raise KeyError(f"Sequence not found in DNA or RNA manifest: {sequence[:50]}...")

    def save_error(self, error_sequences: list, pdb_id: str) -> None:
        """Save error information for debugging."""
        if (self.error_dir is None) or (len(error_sequences) == 0):
            return
        for error_data in error_sequences:
            fpath = os.path.join(
                self.error_dir, f"{pdb_id}_{error_data['entity_id']}.txt"
            )
            if os.path.exists(fpath):
                continue
            with open(fpath, "w") as f:
                f.write(error_data["error"])

    def _fill_entities(
        self,
        x: torch.Tensor,
        entity_ids: set,
        centre_atom_array,
        bioassembly_dict: dict,
        inference_mode: bool,
        entity_id_to_sequence: dict,
        load_fn,
        mol_type_label: str,
    ) -> list:
        """
        Fill embedding tensor for a set of entities using the given load function.

        Args:
            x: Output tensor [N_token, embedding_dim] to fill in-place.
            entity_ids: Set of entity IDs to process.
            centre_atom_array: Centre atom array for the cropped tokens.
            bioassembly_dict: Dictionary with sequence info.
            inference_mode: Whether in inference mode.
            entity_id_to_sequence: Pre-extracted sequence mapping (inference mode).
            load_fn: Function to load embedding for a sequence string.
            mol_type_label: 'RNA' or 'DNA' for logging.

        Returns:
            List of error dicts.
        """
        error_sequences = []
        for entity_id in entity_ids:
            try:
                if inference_mode:
                    sequence = entity_id_to_sequence[entity_id]
                else:
                    sequence = bioassembly_dict["sequences"][str(entity_id)]

                x_emb = load_fn(sequence)

                entity_mask = centre_atom_array.label_entity_id == entity_id
                res_index = centre_atom_array.res_id[entity_mask] - 1

                # Bounds check: ensure res_index values are within embedding range.
                # After cropping, res_id values may reference positions beyond the
                # embedding length if modified bases were stripped or if there's a
                # mismatch between the sequence and the structure.
                emb_len = x_emb.size(0)
                if res_index.max() >= emb_len or res_index.min() < 0:
                    pdb_id = bioassembly_dict.get("pdb_id", "unknown")
                    logger.warning(
                        f"[{pdb_id}] {mol_type_label} entity {entity_id}: "
                        f"res_index range [{res_index.min()}, {res_index.max()}] "
                        f"out of embedding bounds [0, {emb_len - 1}]. "
                        f"Clamping to valid range."
                    )
                    res_index = res_index.clamp(0, emb_len - 1)

                x[entity_mask] = x_emb[res_index]

            except Exception as e:
                pdb_id = bioassembly_dict.get("pdb_id", "unknown")
                raise RuntimeError(
                    f"[{pdb_id}] Failed to load {mol_type_label} embedding for "
                    f"entity {entity_id}: {e}"
                ) from e
        return error_sequences

    # === Sequence content analysis helpers ===

    @staticmethod
    def _has_uracil(seq: str) -> bool:
        """Check if sequence contains uracil (RNA base)."""
        return "U" in seq or "u" in seq

    @staticmethod
    def _is_pure_dna(seq: str) -> bool:
        """Check if sequence contains only standard DNA bases (A, C, G, T, N)."""
        return all(c in "ACGTNacgtn" for c in seq)

    @staticmethod
    def _has_modified_base(seq: str) -> bool:
        """Check if sequence contains modified base placeholders (X).
        Modified nucleotides from CCD (pseudouridine, methylated bases, etc.)
        are converted to 'X' by the parser's res_names_to_sequence()."""
        return "X" in seq or "x" in seq

    def _identify_entities(self, centre_atom_array, bioassembly_dict, inference_mode):
        """
        Identify RNA and DNA entity IDs from the cropped tokens.

        Uses RNA-First strategy: sequences containing uracil (U) are always
        classified as RNA regardless of the parser's chain_mol_type label.
        This prevents hybrid DNA/RNA entities from being incorrectly processed
        by the DNA language model, which lacks RNA structural priors (A-form helix,
        2'-OH interactions, pseudoknots, etc.).

        Also applies reverse check: RNA-labeled entities that are pure DNA
        (only ACGT, no U) and have DNA embeddings available are reclassified
        as DNA for more accurate structural priors.
        """
        rna_entity_id_to_seq = {}
        dna_entity_id_to_seq = {}

        if inference_mode:
            for i, entity_info_wrapper in enumerate(bioassembly_dict["sequences"]):
                entity_id = str(i + 1)
                entity_type = list(entity_info_wrapper.keys())[0]
                entity_info = entity_info_wrapper[entity_type]
                if entity_type in ("rnaSequence", "rnaChain"):
                    seq = entity_info["sequence"]
                    # Reverse RNA-First: reclassify RNA entities that are actually
                    # pure DNA (no uracil, only ACGT) when DNA model is available
                    if self.dna_enable and self._is_pure_dna(seq) and not self._has_uracil(seq):
                        logger.info(
                            f"[Reverse-RNA-First] Entity {entity_id} specified as "
                            f"{entity_type} but is pure ACGT (no uracil). "
                            f"Reclassifying as DNA. Seq: {seq[:40]}..."
                        )
                        dna_entity_id_to_seq[entity_id] = seq
                    else:
                        rna_entity_id_to_seq[entity_id] = seq
                elif entity_type in ("dnaSequence", "dnaChain") and self.dna_enable:
                    seq = entity_info["sequence"]
                    # RNA-First: reclassify DNA entities that contain uracil
                    if self._has_uracil(seq):
                        logger.warning(
                            f"[RNA-First] Entity {entity_id} specified as {entity_type} "
                            f"but contains uracil (RNA base). Reclassifying as RNA to "
                            f"preserve structural priors. Seq: {seq[:40]}..."
                        )
                        rna_entity_id_to_seq[entity_id] = seq
                    else:
                        dna_entity_id_to_seq[entity_id] = seq
            rna_entity_ids = set(rna_entity_id_to_seq.keys())
            dna_entity_ids = set(dna_entity_id_to_seq.keys())
        else:
            # Training mode: use chain_mol_type annotations as initial labels
            is_rna = centre_atom_array.chain_mol_type == "rna"
            rna_labeled_ids = set(centre_atom_array.label_entity_id[is_rna])

            dna_labeled_ids = set()
            if self.dna_enable:
                is_dna = centre_atom_array.chain_mol_type == "dna"
                dna_labeled_ids = set(centre_atom_array.label_entity_id[is_dna])

            # RNA-First reclassification: check actual sequence content.
            # Sequences containing uracil MUST use the RNA model because:
            #   1. Uracil is exclusive to RNA (DNA uses thymine)
            #   2. RNA models capture A-form helix, 2'-OH, pseudoknot priors
            #   3. DNA models (AIDO.DNA) trained only on ACGT vocabulary
            rna_entity_ids = set()
            dna_entity_ids = set()

            # Process DNA-labeled entities: reclassify those with uracil to RNA
            for entity_id in dna_labeled_ids:
                seq = bioassembly_dict["sequences"].get(str(entity_id), "")
                if self._has_uracil(seq):
                    logger.warning(
                        f"[RNA-First] Entity {entity_id} labeled as DNA by parser "
                        f"but contains uracil (RNA base). Reclassifying as RNA. "
                        f"Seq: {seq[:40]}..."
                    )
                    rna_entity_ids.add(entity_id)
                else:
                    dna_entity_ids.add(entity_id)

            # Process RNA-labeled entities: reclassify pure-ACGT ones to DNA
            for entity_id in rna_labeled_ids:
                seq = bioassembly_dict["sequences"].get(str(entity_id), "")
                if self.dna_enable and self._is_pure_dna(seq) and not self._has_uracil(seq):
                    logger.info(
                        f"[Reverse-RNA-First] Entity {entity_id} labeled as RNA "
                        f"but is pure ACGT (no uracil). Reclassifying as DNA. "
                        f"Seq: {seq[:40]}..."
                    )
                    dna_entity_ids.add(entity_id)
                else:
                    rna_entity_ids.add(entity_id)

        # Apply use_rna_embed / use_dna_embed filters
        if not self.use_rna_embed:
            rna_entity_ids = set()
            rna_entity_id_to_seq = {}
        if not self.use_dna_embed:
            dna_entity_ids = set()
            dna_entity_id_to_seq = {}

        return rna_entity_ids, dna_entity_ids, rna_entity_id_to_seq, dna_entity_id_to_seq

    def __call__(
        self,
        token_array,
        atom_array,
        bioassembly_dict: dict,
        inference_mode: bool = False,
        return_separate: bool = False,
    ):
        """
        Create per-token embedding tensor(s) for cropped tokens.

        RNA tokens receive RNA LM embeddings.
        DNA tokens receive DNA LM embeddings (if enabled).
        Protein/ligand tokens receive zeros.

        Args:
            token_array: Cropped token array with centre_atom_index annotations.
            atom_array: Cropped atom array with chain_mol_type, label_entity_id, res_id.
            bioassembly_dict: Dictionary with sequence information per entity.
            inference_mode: If True, extract entities from bioassembly_dict["sequences"]
                           list format. If False, use atom_array annotations.
            return_separate: If True, return a dict with separate RNA and DNA tensors:
                {"rna_llm_embedding": [N, rna_dim], "dna_llm_embedding": [N, dna_dim]}
                If False (default), return a single combined tensor [N, embedding_dim]
                with DNA zero-padded to match RNA dim (legacy behavior).

        Returns:
            If return_separate=False: Tensor [N_token, embedding_dim]
            If return_separate=True: dict with "rna_llm_embedding" and "dna_llm_embedding"
        """
        N_token = len(token_array)

        centre_atoms_indices = token_array.get_annotation("centre_atom_index")
        centre_atom_array = atom_array[centre_atoms_indices]

        # Identify entities
        rna_entity_ids, dna_entity_ids, rna_entity_id_to_seq, dna_entity_id_to_seq = (
            self._identify_entities(centre_atom_array, bioassembly_dict, inference_mode)
        )

        if return_separate:
            # === Separate mode: produce independent RNA and DNA tensors ===
            error_sequences = []

            # RNA tensor: only create and fill if use_rna_embed=True
            if self.use_rna_embed:
                rna_x = torch.zeros([N_token, self.embedding_dim])
                rna_errors = self._fill_entities(
                    rna_x, rna_entity_ids, centre_atom_array, bioassembly_dict,
                    inference_mode, rna_entity_id_to_seq,
                    self.load_rnalm_embedding_with_fallback, "RNA",
                )
                error_sequences.extend(rna_errors)

            # DNA tensor: only create and fill if use_dna_embed=True
            if self.use_dna_embed:
                dna_x = torch.zeros([N_token, self.dna_embedding_dim])
                if dna_entity_ids:
                    dna_errors = self._fill_entities(
                        dna_x, dna_entity_ids, centre_atom_array, bioassembly_dict,
                        inference_mode, dna_entity_id_to_seq,
                        self.load_dna_embedding_native_with_fallback, "DNA",
                    )
                    error_sequences.extend(dna_errors)

            id_key = "name" if inference_mode else "pdb_id"
            self.save_error(error_sequences, pdb_id=bioassembly_dict.get(id_key, "unknown"))

            result = {}
            if self.use_rna_embed:
                result["rna_llm_embedding"] = rna_x
            if self.use_dna_embed:
                result["dna_llm_embedding"] = dna_x
            return result
        else:
            # === Combined mode (legacy): single tensor with DNA zero-padded ===
            x = torch.zeros([N_token, self.embedding_dim])

            # Fill RNA embeddings (with fallback to DNA manifest)
            error_sequences = self._fill_entities(
                x, rna_entity_ids, centre_atom_array, bioassembly_dict,
                inference_mode, rna_entity_id_to_seq,
                self.load_rnalm_embedding_with_fallback, "RNA",
            )

            # Fill DNA embeddings (zero-padded, with fallback to RNA manifest)
            if dna_entity_ids:
                dna_errors = self._fill_entities(
                    x, dna_entity_ids, centre_atom_array, bioassembly_dict,
                    inference_mode, dna_entity_id_to_seq,
                    self.load_dna_embedding_with_fallback, "DNA",
                )
                error_sequences.extend(dna_errors)

            id_key = "name" if inference_mode else "pdb_id"
            self.save_error(error_sequences, pdb_id=bioassembly_dict.get(id_key, "unknown"))

            return x
