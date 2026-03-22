from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from Bio.Align import PairwiseAligner
from Bio.PDB import MMCIFParser, PDBParser

# Official Protenix v1 residue ids from bytedance/Protenix main.
PROTENIX_TEMPLATE_RNA_SEQ_TO_ID: Dict[str, int] = {
    "A": 21,
    "G": 22,
    "C": 23,
    "U": 24,
    "N": 25,
    "-": 31,
}

# DNA ids are included for safety when a mixed template accidentally appears.
PROTENIX_TEMPLATE_DNA_SEQ_TO_ID: Dict[str, int] = {
    "A": 26,
    "G": 27,
    "C": 28,
    "T": 29,
    "N": 30,
    "-": 31,
}

BACKBONE_SUGAR_ATOMS = {
    "P",
    "OP1",
    "OP2",
    "OP3",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
    # Common legacy/alternative atom names
    "O5*",
    "C5*",
    "C4*",
    "O4*",
    "C3*",
    "O3*",
    "C2*",
    "O2*",
    "C1*",
}

COMMON_MODIFIED_BASE_MAP = {
    # U-like
    "PSU": "U",
    "H2U": "U",
    "5MU": "U",
    "OMU": "U",
    "UR3": "U",
    # C-like
    "5MC": "C",
    "OMC": "C",
    "DC": "C",
    # A-like
    "1MA": "A",
    "M2A": "A",
    "6MZ": "A",
    "DA": "A",
    # G-like
    "2MG": "G",
    "7MG": "G",
    "OMG": "G",
    "M2G": "G",
    "DG": "G",
    # T-like, folded back to U for RNA-centric use
    "DT": "U",
    "T": "U",
}

DEFAULT_DENSE_ATOM_NAMES: List[str] = [
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
    "N9",
    "C8",
    "N7",
    "C5",
    "C6",
    "N6",
    "O6",
    "N1",
    "C2",
    "N2",
    "O2",
    "N3",
    "C4",
    "N4",
    "O4",
]

LW_CLASSES: List[str] = [
    "cWW",
    "tWW",
    "cWH",
    "tWH",
    "cWS",
    "tWS",
    "cHW",
    "tHW",
    "cHH",
    "tHH",
    "cHS",
    "tHS",
    "cSW",
    "tSW",
    "cSH",
    "tSH",
    "cSS",
    "tSS",
]


def _clean_atom_name(name: str) -> str:
    return name.strip().replace("*", "'").upper()


@dataclass
class ResidueRecord:
    chain_id: str
    resseq: int
    icode: str
    resname: str
    base_letter: str
    atoms: Dict[str, np.ndarray]

    @property
    def residue_uid(self) -> str:
        suffix = self.icode if self.icode else ""
        return f"{self.chain_id}.{self.base_letter}{self.resseq}{suffix}"

    def aliases(self) -> set[Tuple[str, int, str]]:
        return {
            (self.chain_id, self.resseq, self.icode),
            (self.chain_id, self.resseq, ""),
        }


def read_text_sequence(text_or_path: str) -> str:
    path = Path(text_or_path)
    if path.exists():
        text = path.read_text().strip()
    else:
        text = text_or_path.strip()
    if text.startswith(">"):
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith(">")]
        text = "".join(lines)
    return normalize_query_sequence(text)


def normalize_query_sequence(seq: str) -> str:
    seq = re.sub(r"\s+", "", seq).upper().replace("T", "U")
    return "".join(ch if ch in {"A", "G", "C", "U", "N", "-"} else "N" for ch in seq)


def parse_template_spec(spec: str) -> Tuple[str, Optional[str]]:
    if ":" in spec and Path(spec).suffix.lower() not in {".pdb", ".ent", ".cif", ".mmcif"}:
        # Handles paths like file.cif:A by splitting on the last colon.
        path_str, chain_id = spec.rsplit(":", 1)
        return path_str, chain_id
    path = Path(spec)
    if path.exists():
        return spec, None
    if ":" in spec:
        path_str, chain_id = spec.rsplit(":", 1)
        return path_str, chain_id
    return spec, None


def _select_alt_atom(atom):
    if not atom.is_disordered():
        return atom
    if hasattr(atom, "selected_child") and atom.selected_child is not None:
        return atom.selected_child
    children = list(atom.child_dict.values())
    if not children:
        return atom
    return max(children, key=lambda child: (child.get_occupancy() or 0.0, child.get_bfactor() or 0.0))


def _infer_base_from_atoms(resname: str, atoms: Dict[str, np.ndarray]) -> str:
    rn = resname.strip().upper()
    if rn in COMMON_MODIFIED_BASE_MAP:
        return COMMON_MODIFIED_BASE_MAP[rn]
    if rn in {"A", "G", "C", "U"}:
        return rn
    if "N6" in atoms:
        return "A"
    if "O6" in atoms or "N2" in atoms:
        return "G"
    if "N4" in atoms:
        return "C"
    if "O4" in atoms:
        return "U"
    if rn.startswith("D") and len(rn) >= 2 and rn[1] in {"A", "G", "C", "T"}:
        return "U" if rn[1] == "T" else rn[1]
    if rn and rn[-1] in {"A", "G", "C", "U", "T"}:
        return "U" if rn[-1] == "T" else rn[-1]
    return "N"


def _looks_like_nucleotide(atoms: Dict[str, np.ndarray], resname: str) -> bool:
    clean = resname.strip().upper()
    if clean in {"A", "G", "C", "U", "DA", "DG", "DC", "DT"}:
        return True
    if "C1'" in atoms or "C1*" in atoms:
        return True
    if "P" in atoms and ("C4'" in atoms or "C4*" in atoms):
        return True
    return False


def load_structure_residues(structure_path: str, chain_id: Optional[str] = None) -> List[ResidueRecord]:
    path = Path(structure_path)
    suffix = path.suffix.lower()
    if suffix in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rna_template", str(path))
    model = next(structure.get_models())

    if chain_id is None:
        chains = [chain for chain in model]
        if len(chains) != 1:
            raise ValueError(
                f"{structure_path} has {len(chains)} chains in model 0. Please provide file:CHAIN explicitly."
            )
        chain = chains[0]
    else:
        chain = model[chain_id]

    records: List[ResidueRecord] = []
    for residue in chain:
        atom_dict: Dict[str, np.ndarray] = {}
        for atom in residue.get_atoms():
            sel = _select_alt_atom(atom)
            atom_name = _clean_atom_name(sel.get_name())
            element = (sel.element or atom_name[:1]).upper()
            if element == "H" or atom_name.startswith("H"):
                continue
            atom_dict[atom_name] = np.asarray(sel.get_coord(), dtype=np.float32)
        if not atom_dict:
            continue
        if not _looks_like_nucleotide(atom_dict, residue.resname):
            continue
        _, resseq, icode = residue.id
        icode = (icode or "").strip()
        base_letter = _infer_base_from_atoms(residue.resname, atom_dict)
        records.append(
            ResidueRecord(
                chain_id=chain.id,
                resseq=int(resseq),
                icode=icode,
                resname=residue.resname.strip().upper(),
                base_letter=base_letter,
                atoms=atom_dict,
            )
        )
    if not records:
        raise ValueError(f"No nucleotide-like residues found in {structure_path} chain {chain.id}.")
    return records


def residues_to_sequence(residues: Sequence[ResidueRecord]) -> str:
    return "".join(res.base_letter for res in residues)


def align_query_to_template(query_seq: str, template_seq: str) -> Dict[int, int]:
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(query_seq, template_seq)[0]
    mapping: Dict[int, int] = {}
    query_blocks, template_blocks = alignment.aligned
    for q_block, t_block in zip(query_blocks, template_blocks):
        q_start, q_end = q_block
        t_start, t_end = t_block
        for q_idx, t_idx in zip(range(q_start, q_end), range(t_start, t_end)):
            mapping[q_idx] = t_idx
    return mapping


def _get_atom(atoms: Dict[str, np.ndarray], *names: str) -> Optional[np.ndarray]:
    for name in names:
        clean = _clean_atom_name(name)
        if clean in atoms:
            return atoms[clean]
    return None


def _base_center(record: ResidueRecord) -> Optional[np.ndarray]:
    base_coords = [coord for atom_name, coord in record.atoms.items() if atom_name not in BACKBONE_SUGAR_ATOMS]
    if not base_coords:
        return None
    return np.mean(np.asarray(base_coords, dtype=np.float32), axis=0)


def compute_anchor(record: ResidueRecord, anchor_mode: str = "base_center_fallback") -> Tuple[np.ndarray, float]:
    mode = anchor_mode.lower()
    if mode in {"base_center", "base_center_fallback"}:
        center = _base_center(record)
        if center is not None:
            return center.astype(np.float32), 1.0
        if mode == "base_center":
            return np.zeros(3, dtype=np.float32), 0.0
    if mode in {"c4p", "base_center_fallback", "c4p_fallback"}:
        c4p = _get_atom(record.atoms, "C4'", "C4*")
        if c4p is not None:
            return c4p.astype(np.float32), 1.0
    c1p = _get_atom(record.atoms, "C1'", "C1*")
    if c1p is not None:
        return c1p.astype(np.float32), 1.0
    return np.zeros(3, dtype=np.float32), 0.0


def _normalize(vec: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return np.zeros_like(vec, dtype=np.float32), 0.0
    return (vec / norm).astype(np.float32), norm


def compute_frame(record: ResidueRecord) -> Tuple[np.ndarray, np.ndarray, float]:
    c4p = _get_atom(record.atoms, "C4'", "C4*")
    c1p = _get_atom(record.atoms, "C1'", "C1*")
    if c4p is None or c1p is None:
        return np.zeros(3, dtype=np.float32), np.zeros((3, 3), dtype=np.float32), 0.0

    origin = c4p.astype(np.float32)
    v1 = _get_atom(record.atoms, "P")
    if v1 is None:
        v1 = _get_atom(record.atoms, "O5'", "O5*", "C3'", "C3*")
    if v1 is None:
        return origin, np.zeros((3, 3), dtype=np.float32), 0.0

    e1, norm1 = _normalize(v1 - origin)
    if norm1 == 0.0:
        return origin, np.zeros((3, 3), dtype=np.float32), 0.0

    e2_raw = c1p.astype(np.float32) - origin
    e2_raw = e2_raw - np.dot(e2_raw, e1) * e1
    e2, norm2 = _normalize(e2_raw)
    if norm2 == 0.0:
        return origin, np.zeros((3, 3), dtype=np.float32), 0.0

    e3 = np.cross(e1, e2).astype(np.float32)
    e3, norm3 = _normalize(e3)
    if norm3 == 0.0:
        return origin, np.zeros((3, 3), dtype=np.float32), 0.0

    axes = np.stack([e1, e2, e3], axis=0)
    return origin, axes.astype(np.float32), 1.0


def compute_distogram(
    anchor_pos: np.ndarray,
    anchor_mask: np.ndarray,
    min_bin: float = 3.25,
    max_bin: float = 50.75,
    num_bins: int = 39,
) -> np.ndarray:
    lower_breaks = np.linspace(min_bin, max_bin, num_bins, dtype=np.float32)
    upper_breaks = np.concatenate([lower_breaks[1:], np.array([1e8], dtype=np.float32)])
    diffs = anchor_pos[:, None, :] - anchor_pos[None, :, :]
    sq_dists = np.sum(diffs * diffs, axis=-1)
    dgram = ((sq_dists[..., None] > lower_breaks[None, None, :] ** 2) & (sq_dists[..., None] <= upper_breaks[None, None, :] ** 2)).astype(np.float32)
    pair_mask = (anchor_mask[:, None] * anchor_mask[None, :]).astype(np.float32)
    return dgram * pair_mask[..., None]


def compute_unit_vectors(
    anchor_pos: np.ndarray,
    anchor_mask: np.ndarray,
    frame_origin: np.ndarray,
    frame_axes: np.ndarray,
    frame_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n = anchor_pos.shape[0]
    uv = np.zeros((n, n, 3), dtype=np.float32)
    mask = (frame_mask[:, None] * anchor_mask[None, :]).astype(np.float32)
    for i in range(n):
        if frame_mask[i] == 0:
            continue
        rel = anchor_pos - frame_origin[i][None, :]
        proj = rel @ frame_axes[i].T
        norms = np.linalg.norm(proj, axis=-1, keepdims=True)
        good = (norms[:, 0] > 1e-8).astype(np.float32)
        proj = proj / np.maximum(norms, 1e-8)
        uv[i] = proj.astype(np.float32)
        mask[i] *= good
    uv *= mask[..., None]
    return uv, mask


def build_minimal_template_arrays(
    query_seq: str,
    residues: Sequence[ResidueRecord],
    template_name: str,
    anchor_mode: str = "base_center_fallback",
    min_bin: float = 3.25,
    max_bin: float = 50.75,
    num_bins: int = 39,
) -> Dict[str, np.ndarray]:
    query_seq = normalize_query_sequence(query_seq)
    template_seq = residues_to_sequence(residues)
    mapping = align_query_to_template(query_seq, template_seq)
    n = len(query_seq)

    aatype = np.full((n,), PROTENIX_TEMPLATE_RNA_SEQ_TO_ID["-"], dtype=np.int32)
    anchor_pos = np.zeros((n, 3), dtype=np.float32)
    anchor_mask = np.zeros((n,), dtype=np.float32)
    frame_origin = np.zeros((n, 3), dtype=np.float32)
    frame_axes = np.zeros((n, 3, 3), dtype=np.float32)
    frame_mask = np.zeros((n,), dtype=np.float32)
    residue_meta: List[dict] = []

    for q_idx, t_idx in mapping.items():
        rec = residues[t_idx]
        aatype[q_idx] = PROTENIX_TEMPLATE_RNA_SEQ_TO_ID.get(rec.base_letter, PROTENIX_TEMPLATE_RNA_SEQ_TO_ID["N"])
        anchor_pos[q_idx], anchor_mask[q_idx] = compute_anchor(rec, anchor_mode=anchor_mode)
        frame_origin[q_idx], frame_axes[q_idx], frame_mask[q_idx] = compute_frame(rec)
        residue_meta.append(
            {
                "query_idx": int(q_idx),
                "template_idx": int(t_idx),
                "residue_uid": rec.residue_uid,
                "resname": rec.resname,
                "base_letter": rec.base_letter,
            }
        )

    distogram = compute_distogram(anchor_pos, anchor_mask, min_bin=min_bin, max_bin=max_bin, num_bins=num_bins)
    pair_anchor_mask = (anchor_mask[:, None] * anchor_mask[None, :]).astype(np.float32)
    unit_vector, pair_frame_mask = compute_unit_vectors(anchor_pos, anchor_mask, frame_origin, frame_axes, frame_mask)

    return {
        "template_name": np.asarray(template_name),
        "template_aatype": aatype,
        "template_anchor_pos": anchor_pos,
        "template_anchor_mask_1d": anchor_mask,
        "template_frame_origin": frame_origin,
        "template_frame_axes": frame_axes,
        "template_frame_mask_1d": frame_mask,
        "template_distogram": distogram,
        "template_pseudo_beta_mask": pair_anchor_mask,
        "template_unit_vector": unit_vector,
        "template_backbone_frame_mask": pair_frame_mask,
        "mapping_json": np.asarray(json.dumps(residue_meta, ensure_ascii=False)),
    }


def stack_template_dicts(template_dicts: Sequence[Dict[str, np.ndarray]], max_templates: int) -> Dict[str, np.ndarray]:
    if not template_dicts:
        raise ValueError("No template dictionaries to stack.")

    keys_to_stack = [
        "template_aatype",
        "template_anchor_pos",
        "template_anchor_mask_1d",
        "template_frame_origin",
        "template_frame_axes",
        "template_frame_mask_1d",
        "template_distogram",
        "template_pseudo_beta_mask",
        "template_unit_vector",
        "template_backbone_frame_mask",
    ]

    n = template_dicts[0]["template_aatype"].shape[0]
    out: Dict[str, np.ndarray] = {}
    limit = min(len(template_dicts), max_templates)
    template_mask = np.zeros((max_templates,), dtype=np.float32)
    template_names = np.full((max_templates,), "", dtype="<U256")
    mapping_json = np.full((max_templates,), "[]", dtype="<U1048576")

    for key in keys_to_stack:
        exemplar = template_dicts[0][key]
        shape = (max_templates,) + exemplar.shape
        out[key] = np.zeros(shape, dtype=exemplar.dtype)

    for idx in range(limit):
        td = template_dicts[idx]
        template_mask[idx] = 1.0
        template_names[idx] = str(td["template_name"])
        mapping_json[idx] = str(td["mapping_json"])
        for key in keys_to_stack:
            out[key][idx] = td[key]

    # Fill padded template aatype with gap ids to be explicit.
    if limit < max_templates:
        out["template_aatype"][limit:] = PROTENIX_TEMPLATE_RNA_SEQ_TO_ID["-"]

    out["template_mask"] = template_mask
    out["template_names"] = template_names
    out["template_mapping_json"] = mapping_json
    out["query_length"] = np.asarray(n, dtype=np.int32)
    return out


def save_npz_with_metadata(output_path: str, arrays: Dict[str, np.ndarray], metadata: dict) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    meta_path = output.with_suffix(output.suffix + ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))


def load_dssr_json(path: str) -> dict:
    return json.loads(Path(path).read_text())


def parse_dssr_nt_id(value: object) -> Optional[Tuple[str, int, str]]:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("nt_id") or value.get("nt_name") or value.get("id")
    text = str(value).strip()
    if not text:
        return None
    text = text.replace(" ", "")
    m = re.match(r"([^.:]+)[.:](.+)", text)
    if not m:
        return None
    chain_id = m.group(1)
    body = m.group(2)
    num_match = re.search(r"(-?\d+)([A-Za-z]?)", body)
    if not num_match:
        return None
    resseq = int(num_match.group(1))
    icode = (num_match.group(2) or "").upper()
    return chain_id, resseq, icode


def build_residue_lookup(residues: Sequence[ResidueRecord], mapping: Dict[int, int]) -> Dict[Tuple[str, int, str], int]:
    lookup: Dict[Tuple[str, int, str], int] = {}
    for q_idx, t_idx in mapping.items():
        rec = residues[t_idx]
        for alias in rec.aliases():
            lookup[alias] = q_idx
    return lookup


def canonical_pair_category(base_i: str, base_j: str, lw: str) -> str:
    pair = {base_i, base_j}
    lw_norm = lw.replace(" ", "")
    if lw_norm == "cWW" and pair in ({"A", "U"}, {"G", "C"}):
        return "canonical"
    if lw_norm == "cWW" and pair == {"G", "U"}:
        return "wobble"
    return "noncanonical"


def collect_dssr_pair_edges(dssr: dict, residue_lookup: Dict[Tuple[str, int, str], int], query_seq: str) -> List[dict]:
    entries = []
    pair_lists: List[list] = []
    for key in ("pairs", "basePairs", "bps"):
        if isinstance(dssr.get(key), list):
            pair_lists.append(dssr[key])
    for pair_list in pair_lists:
        for item in pair_list:
            nt1 = parse_dssr_nt_id(item.get("nt1") or item.get("nt1_id") or item.get("res1") or item.get("residue1"))
            nt2 = parse_dssr_nt_id(item.get("nt2") or item.get("nt2_id") or item.get("res2") or item.get("residue2"))
            if nt1 is None or nt2 is None:
                continue
            if nt1 not in residue_lookup or nt2 not in residue_lookup:
                continue
            i = residue_lookup[nt1]
            j = residue_lookup[nt2]
            if i == j:
                continue
            lw = str(item.get("LW") or item.get("lw") or item.get("bp") or item.get("name") or "").replace(" ", "")
            if lw and lw not in LW_CLASSES:
                # Keep only the standard 18-class alphabet when possible.
                if len(lw) >= 3 and lw[:3] in LW_CLASSES:
                    lw = lw[:3]
            category = canonical_pair_category(query_seq[i], query_seq[j], lw)
            entries.append({
                "i": i,
                "j": j,
                "lw": lw,
                "category": category,
                "raw": item,
            })
    return entries


def _iter_stack_members(item: dict) -> List[Tuple[str, int, str]]:
    if "nts_long" in item:
        values = str(item["nts_long"]).split(",")
    elif "nts" in item:
        nts_value = item["nts"]
        if isinstance(nts_value, list):
            values = [str(v) for v in nts_value]
        else:
            values = str(nts_value).split(",")
    else:
        values = []
        nt1 = item.get("nt1")
        nt2 = item.get("nt2")
        if nt1 is not None:
            values.append(str(nt1))
        if nt2 is not None:
            values.append(str(nt2))
    out: List[Tuple[str, int, str]] = []
    for value in values:
        parsed = parse_dssr_nt_id(value)
        if parsed is not None:
            out.append(parsed)
    return out


def collect_dssr_stack_edges(dssr: dict, residue_lookup: Dict[Tuple[str, int, str], int]) -> List[dict]:
    entries = []
    for key, value in dssr.items():
        if "stack" not in key.lower():
            continue
        if not isinstance(value, list):
            continue
        for item in value:
            nts = _iter_stack_members(item)
            if len(nts) < 2:
                continue
            q_indices = [residue_lookup[nt] for nt in nts if nt in residue_lookup]
            if len(q_indices) < 2:
                continue
            for a, b in zip(q_indices[:-1], q_indices[1:]):
                if a == b:
                    continue
                entries.append(
                    {
                        "i": a,
                        "j": b,
                        "consecutive_in_stack": 1.0,
                        "raw": item,
                    }
                )
    return entries


def dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[float]:
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1)
    if b1_norm < 1e-8:
        return None
    b1 = b1 / b1_norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)
    if v_norm < 1e-8 or w_norm < 1e-8:
        return None
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))


def compute_chi_angle(record: ResidueRecord) -> Optional[float]:
    o4p = _get_atom(record.atoms, "O4'", "O4*")
    c1p = _get_atom(record.atoms, "C1'", "C1*")
    if o4p is None or c1p is None:
        return None
    if record.base_letter in {"A", "G"}:
        n = _get_atom(record.atoms, "N9")
        c = _get_atom(record.atoms, "C4")
    else:
        n = _get_atom(record.atoms, "N1")
        c = _get_atom(record.atoms, "C2")
    if n is None or c is None:
        return None
    return dihedral_angle(o4p, c1p, n, c)


def compute_delta_angle(record: ResidueRecord) -> Optional[float]:
    c5p = _get_atom(record.atoms, "C5'", "C5*")
    c4p = _get_atom(record.atoms, "C4'", "C4*")
    c3p = _get_atom(record.atoms, "C3'", "C3*")
    o3p = _get_atom(record.atoms, "O3'", "O3*")
    if c5p is None or c4p is None or c3p is None or o3p is None:
        return None
    return dihedral_angle(c5p, c4p, c3p, o3p)


def compute_eta_theta(
    prev_record: Optional[ResidueRecord],
    record: ResidueRecord,
    next_record: Optional[ResidueRecord],
) -> Tuple[Optional[float], Optional[float]]:
    eta = None
    theta = None
    if prev_record is not None and next_record is not None:
        prev_c4p = _get_atom(prev_record.atoms, "C4'", "C4*")
        p = _get_atom(record.atoms, "P")
        c4p = _get_atom(record.atoms, "C4'", "C4*")
        next_p = _get_atom(next_record.atoms, "P")
        if prev_c4p is not None and p is not None and c4p is not None and next_p is not None:
            eta = dihedral_angle(prev_c4p, p, c4p, next_p)
    if next_record is not None:
        p = _get_atom(record.atoms, "P")
        c4p = _get_atom(record.atoms, "C4'", "C4*")
        next_p = _get_atom(next_record.atoms, "P")
        next_c4p = _get_atom(next_record.atoms, "C4'", "C4*")
        if p is not None and c4p is not None and next_p is not None and next_c4p is not None:
            theta = dihedral_angle(p, c4p, next_p, next_c4p)
    return eta, theta


def angle_to_sin_cos(angle: Optional[float]) -> Tuple[float, float]:
    if angle is None:
        return 0.0, 0.0
    return float(math.sin(angle)), float(math.cos(angle))


def save_json(path: str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
