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

"""
Data-stage tokenizer for RibonanzaNet2 (v3).

Generates tokenized_seq and ribonanza_token_mask from parser annotations.
Does not depend on RNALM, model, restype, or sequence strings.
Sole dependency: atom_array.mol_type (from parser via CCD _chem_comp.type)
and atom_array.res_name (from parser/CCD).
"""

import logging

import torch

from protenix.data.core.ccd import get_one_letter_code

logger = logging.getLogger(__name__)


class RibonanzaTokenizer:
    """
    Data-stage tokenizer for RibonanzaNet2.

    Generates tokenized_seq and ribonanza_token_mask from parser annotations.
    Does not depend on RNALM, model, restype, or sequence strings.
    Sole dependency: atom_array.mol_type (from parser via CCD _chem_comp.type)
    and atom_array.res_name (from parser/CCD).
    """

    VOCAB = {"A": 0, "C": 1, "G": 2, "U": 3, "PAD": 4, "X": 5}
    PAD = 4

    @staticmethod
    def normalize_base(base: str) -> str:
        """Normalize a one-letter base code to RibonanzaNet vocabulary."""
        base = base.upper()
        if base in ("A", "C", "G", "U"):
            return base
        return "X"

    def __call__(
        self,
        token_array,
        atom_array,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            token_array: cropped TokenArray with centre_atom_index annotation
            atom_array: cropped AtomArray with mol_type and res_name annotations

        Returns:
            {
                "tokenized_seq": torch.long [N_token],
                "ribonanza_token_mask": torch.bool [N_token],
            }
        """
        centre_atom_indices = token_array.get_annotation("centre_atom_index")
        centre_atom_array = atom_array[centre_atom_indices]
        n_tokens = len(centre_atom_indices)

        tokenized_seq = torch.full((n_tokens,), self.PAD, dtype=torch.long)
        is_rna = centre_atom_array.mol_type == "rna"
        ribonanza_token_mask = torch.from_numpy(is_rna.copy())

        for i in range(n_tokens):
            if not is_rna[i]:
                continue

            res_name = centre_atom_array.res_name[i]

            # Defensive: empty or missing res_name -> mask out
            if not res_name or not res_name.strip():
                logger.warning(
                    f"RibonanzaTokenizer: token {i} is RNA but has "
                    f"empty res_name, setting mask to False"
                )
                ribonanza_token_mask[i] = False
                continue

            # CCD lookup -> one-letter code -> normalize
            one_letter = get_one_letter_code(res_name)
            if one_letter is None or len(one_letter) > 1:
                one_letter = "X"
            base = self.normalize_base(one_letter)
            tokenized_seq[i] = self.VOCAB[base]

        return {
            "tokenized_seq": tokenized_seq,
            "ribonanza_token_mask": ribonanza_token_mask,
        }
