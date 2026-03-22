"""
RNA Template Featurizer for Protenix — v3 (Online Mode).

Supports two modes:
  **Online mode** (default when search_results_path + cif_database_dir provided):
    Per-hit filtering and 3D feature construction happen at training time,
    mirroring the protein template pipeline:
      1. Look up query sequence → hit list from search_results.json
      2. Per-hit filter: self-hit exclusion, temporal filtering (with PDB API fallback)
      3. Per-hit build: load CIF → load_structure_residues → build_minimal_template_arrays
      4. Collect up to max_templates successful builds → stack

  **Offline mode** (legacy, when template_index_path provided):
    Loads pre-computed NPZ tensors, same as v2.

Data-leakage prevention (mirrors protein pipeline):
  - Per-query temporal filtering: rejects templates released after
    (query_release_date − DAYS_BEFORE_QUERY_DATE).
  - Self-hit exclusion: rejects templates from the same base PDB as the query.
  - PDB API fallback: when RNA3DB metadata lacks a release date, queries RCSB
    PDB REST API and caches the result. Unknown → REJECT (conservative).
  - Both safeguards are active only during training; inference uses all templates.
"""

import glob as glob_mod
import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from protenix.data.constants import RNA_CHAIN
from protenix.data.template.template_utils import DAYS_BEFORE_QUERY_DATE
from protenix.utils.logger import get_logger

logger = get_logger(__name__)

# RNA residue type IDs matching Protenix codec (from rna_template_common.py)
RNA_SEQ_TO_ID = {"A": 21, "G": 22, "C": 23, "U": 24, "N": 25, "-": 31}

# Feature keys expected by TemplateEmbedder (same as protein template)
RNA_TEMPLATE_FEATURES = (
    "template_aatype",
    "template_distogram",
    "template_pseudo_beta_mask",
    "template_unit_vector",
    "template_backbone_frame_mask",
)

# ── Release-date cache (one load per metadata path per process) ──────────
_RNA_RELEASE_DATES_CACHE: Dict[str, Dict[str, datetime]] = {}

# ── PDB API release date cache (thread-safe, persistent) ─────────────────
_PDB_API_DATE_CACHE: Dict[str, Optional[datetime]] = {}
_PDB_API_CACHE_LOCK = threading.Lock()
_PDB_API_CACHE_PATH: Optional[str] = None


def _load_rna_release_dates(metadata_path: str) -> Dict[str, datetime]:
    """Load RNA3DB metadata and build base_pdb → earliest_release_date mapping.

    RNA3DB keys are like '7zpi_B' (pdb_chain). We extract the 4-char base PDB
    and keep the *earliest* release date across all chains of the same PDB.
    """
    if metadata_path in _RNA_RELEASE_DATES_CACHE:
        return _RNA_RELEASE_DATES_CACHE[metadata_path]

    if not metadata_path or not os.path.exists(metadata_path):
        logger.warning(
            f"RNA3DB metadata not found at '{metadata_path}'. "
            f"Temporal filtering for RNA templates will be disabled."
        )
        _RNA_RELEASE_DATES_CACHE[metadata_path] = {}
        return {}

    with open(metadata_path, "r") as f:
        raw = json.load(f)

    dates: Dict[str, datetime] = {}
    for key, entry in raw.items():
        rd = entry.get("release_date")
        if not rd:
            continue
        base_pdb = key.split("_")[0].lower()
        try:
            dt = datetime.strptime(rd, "%Y-%m-%d")
        except ValueError:
            continue
        # Keep earliest date for multi-chain PDBs
        if base_pdb not in dates or dt < dates[base_pdb]:
            dates[base_pdb] = dt

    _RNA_RELEASE_DATES_CACHE[metadata_path] = dates
    logger.info(
        f"RNA release dates loaded: {len(dates)} PDBs from {metadata_path}"
    )
    return dates


# ── PDB API release-date lookup with persistent cache ────────────────────

def _init_pdb_api_cache(cache_dir: str) -> None:
    """Initialize the PDB API date cache from a persistent JSON file."""
    global _PDB_API_CACHE_PATH, _PDB_API_DATE_CACHE
    _PDB_API_CACHE_PATH = os.path.join(cache_dir, "pdb_release_dates_cache.json")
    if os.path.exists(_PDB_API_CACHE_PATH):
        try:
            with open(_PDB_API_CACHE_PATH, "r") as f:
                raw = json.load(f)
            for pdb_id, date_str in raw.items():
                if date_str is None:
                    _PDB_API_DATE_CACHE[pdb_id] = None
                else:
                    try:
                        _PDB_API_DATE_CACHE[pdb_id] = datetime.strptime(date_str, "%Y-%m-%d")
                    except ValueError:
                        _PDB_API_DATE_CACHE[pdb_id] = None
            logger.info(f"PDB API date cache loaded: {len(_PDB_API_DATE_CACHE)} entries from {_PDB_API_CACHE_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load PDB API date cache: {e}")


def _save_pdb_api_cache() -> None:
    """Persist the PDB API date cache to disk."""
    if _PDB_API_CACHE_PATH is None:
        return
    try:
        serialized = {}
        for pdb_id, dt in _PDB_API_DATE_CACHE.items():
            serialized[pdb_id] = dt.strftime("%Y-%m-%d") if dt is not None else None
        os.makedirs(os.path.dirname(_PDB_API_CACHE_PATH), exist_ok=True)
        with open(_PDB_API_CACHE_PATH, "w") as f:
            json.dump(serialized, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save PDB API date cache: {e}")


def _fetch_pdb_release_date(pdb_id_4char: str) -> Optional[datetime]:
    """Query RCSB PDB REST API for the release date of a 4-char PDB ID.

    Uses https://data.rcsb.org/rest/v1/core/entry/{pdb_id} and extracts
    rcsb_accession_info.initial_release_date.

    Returns datetime on success, None on failure. Results are cached.
    """
    pdb_id_4char = pdb_id_4char.lower()

    with _PDB_API_CACHE_LOCK:
        if pdb_id_4char in _PDB_API_DATE_CACHE:
            return _PDB_API_DATE_CACHE[pdb_id_4char]

    try:
        import urllib.request
        import urllib.error

        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id_4char}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Try multiple date fields
        release_date_str = None
        accession = data.get("rcsb_accession_info", {})
        release_date_str = accession.get("initial_release_date")
        if not release_date_str:
            release_date_str = accession.get("deposit_date")

        if release_date_str:
            # RCSB dates are ISO 8601, e.g. "2022-06-15T00:00:00+00:00"
            dt = datetime.strptime(release_date_str[:10], "%Y-%m-%d")
            with _PDB_API_CACHE_LOCK:
                _PDB_API_DATE_CACHE[pdb_id_4char] = dt
                _save_pdb_api_cache()
            logger.debug(f"PDB API: {pdb_id_4char} release_date={dt.strftime('%Y-%m-%d')}")
            return dt

    except urllib.error.HTTPError as e:
        logger.debug(f"PDB API HTTP error for {pdb_id_4char}: {e.code}")
    except Exception as e:
        logger.debug(f"PDB API error for {pdb_id_4char}: {e}")

    # Cache the failure too so we don't re-query
    with _PDB_API_CACHE_LOCK:
        _PDB_API_DATE_CACHE[pdb_id_4char] = None
        _save_pdb_api_cache()
    return None


# ── Utility functions ────────────────────────────────────────────────────

def _extract_base_pdb_id(identifier: str) -> str:
    """Extract 4-char base PDB ID from various identifier formats.

    Examples:
        '1asy_R_R_template.npz' → '1asy'  (NPZ filename)
        '1g1x_I'                → '1g1x'  (search result pdb_id)
        '7zpi'                  → '7zpi'  (plain PDB ID)
    """
    parts = identifier.split("_")
    base = parts[0].lower()
    if len(base) == 4:
        return base
    # For NPZ filenames, extract from basename first
    basename = os.path.basename(identifier)
    parts = basename.split("_")
    if parts:
        return parts[0].lower()
    return base


def _extract_pdb_from_npz_path(npz_path: str) -> str:
    """Extract the 4-char base PDB ID from an NPZ filename (backward compat)."""
    return _extract_base_pdb_id(os.path.basename(npz_path))


def _normalize_rna_sequence(seq: str) -> str:
    """Normalize RNA sequence: uppercase, T→U, keep only AGCUN."""
    seq = seq.upper().replace("T", "U")
    return "".join(c if c in {"A", "G", "C", "U", "N"} else "N" for c in seq)


def _load_rna_template_index(index_path: str, fail_fast: bool = False) -> Dict[str, List[str]]:
    """Load the RNA template index JSON.

    The index maps RNA sequences to lists of template .npz file paths.
    Format: { "AGCUAGCU...": ["path/to/template1.npz", "path/to/template2.npz"], ... }
    """
    if not index_path or not os.path.exists(index_path):
        if fail_fast:
            raise FileNotFoundError(
                f"RNA template index not found at '{index_path}'. "
                f"Either provide a valid template_index_path or set rna_template.enable=false."
            )
        return {}
    with open(index_path, "r") as f:
        index = json.load(f)
    if not index and fail_fast:
        raise ValueError(
            f"RNA template index at '{index_path}' is empty (0 sequences). "
            f"Populate the index or set rna_template.enable=false."
        )
    return index


def _empty_rna_template_features(num_tokens: int, max_templates: int = 4) -> Dict[str, np.ndarray]:
    """Create empty (masked-out) RNA template features."""
    return {
        "rna_template_aatype": np.full((max_templates, num_tokens), RNA_SEQ_TO_ID["-"], dtype=np.int32),
        "rna_template_distogram": np.zeros((max_templates, num_tokens, num_tokens, 39), dtype=np.float32),
        "rna_template_pseudo_beta_mask": np.zeros((max_templates, num_tokens, num_tokens), dtype=np.float32),
        "rna_template_unit_vector": np.zeros((max_templates, num_tokens, num_tokens, 3), dtype=np.float32),
        "rna_template_backbone_frame_mask": np.zeros((max_templates, num_tokens, num_tokens), dtype=np.float32),
        "rna_template_block_mask": np.zeros((num_tokens, num_tokens), dtype=np.float32),
    }


def _load_and_crop_rna_template(
    npz_path: str,
    query_seq: str,
    max_templates: int = 4,
) -> Optional[Dict[str, np.ndarray]]:
    """Load a pre-computed RNA template .npz and crop/align to query sequence."""
    if not os.path.exists(npz_path):
        logger.warning(f"RNA template file not found: {npz_path}")
        return None

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        logger.warning(f"Failed to load RNA template {npz_path}: {e}")
        return None

    features = {}
    for key in RNA_TEMPLATE_FEATURES:
        if key not in data:
            logger.warning(f"Missing key '{key}' in RNA template {npz_path}")
            return None
        features[key] = data[key]

    num_t = features["template_aatype"].shape[0]
    if num_t > max_templates:
        for key in features:
            features[key] = features[key][:max_templates]

    template_n = features["template_aatype"].shape[1]
    query_n = len(query_seq)
    if template_n != query_n:
        logger.warning(
            f"RNA template sequence length mismatch: template={template_n}, "
            f"query={query_n} in {npz_path}. Skipping."
        )
        return None

    return features


def _pad_rna_template_features(
    features: Dict[str, np.ndarray],
    max_templates: int = 4,
) -> Dict[str, np.ndarray]:
    """Pad template features to max_templates along the first dimension."""
    num_t = features["template_aatype"].shape[0]
    if num_t >= max_templates:
        return features

    padded = {}
    for key, arr in features.items():
        pad_shape = list(arr.shape)
        pad_shape[0] = max_templates - num_t
        if key == "template_aatype":
            pad_arr = np.full(pad_shape, RNA_SEQ_TO_ID["-"], dtype=arr.dtype)
        else:
            pad_arr = np.zeros(pad_shape, dtype=arr.dtype)
        padded[key] = np.concatenate([arr, pad_arr], axis=0)
    return padded


# ── CIF path resolution (reused from 02_build_rna_templates.py) ─────────

def _find_cif_path(
    cif_database_dir: str,
    pdb_id: str,
) -> Optional[str]:
    """Locate a CIF file for *pdb_id* inside *cif_database_dir*.

    Tries:
      1. Flat layout: {cif_database_dir}/{pdb_id}.cif
      2. Recursive search for nested dirs (rna3db-mmcifs).

    Returns the path as a string, or None if not found.
    """
    # 1. Flat layout
    flat = os.path.join(cif_database_dir, f"{pdb_id}.cif")
    if os.path.exists(flat):
        return flat

    # 2. Recursive search
    pattern = os.path.join(cif_database_dir, "**", f"{pdb_id}.cif")
    matches = glob_mod.glob(pattern, recursive=True)
    if matches:
        return matches[0]

    return None


# ── CIF path cache (avoid repeated glob) ─────────────────────────────────
_CIF_PATH_CACHE: Dict[str, Optional[str]] = {}
_CIF_PATH_CACHE_LOCK = threading.Lock()


def _find_cif_path_cached(cif_database_dir: str, pdb_id: str) -> Optional[str]:
    """Thread-safe cached CIF path lookup."""
    cache_key = f"{cif_database_dir}::{pdb_id}"
    with _CIF_PATH_CACHE_LOCK:
        if cache_key in _CIF_PATH_CACHE:
            return _CIF_PATH_CACHE[cache_key]

    result = _find_cif_path(cif_database_dir, pdb_id)
    with _CIF_PATH_CACHE_LOCK:
        _CIF_PATH_CACHE[cache_key] = result
    return result


# ── Online template building ─────────────────────────────────────────────

def _build_single_template_online(
    query_seq: str,
    hit: dict,
    cif_database_dir: str,
    anchor_mode: str = "base_center_fallback",
) -> Optional[Dict[str, np.ndarray]]:
    """Build template features for a single hit from CIF file online.

    This replicates the offline build_cross_template logic per-hit,
    using the same functions from rna_template_common.py.

    Args:
        query_seq: Normalized query RNA sequence.
        hit: Dict with pdb_id, chain_id, identity, etc.
        cif_database_dir: Root directory to search for CIF files.
        anchor_mode: Anchor computation mode.

    Returns:
        Dict of template features [N_query, ...] or None on failure.
    """
    # Import rna_template_common lazily to avoid circular imports at module level
    try:
        from protenix.data.rna_template.rna_template_common_online import (
            load_structure_residues,
            build_minimal_template_arrays,
            normalize_query_sequence,
        )
    except ImportError:
        # Fallback: try the compute directory path
        compute_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "rna_template", "compute"
        )
        if compute_dir not in sys.path:
            sys.path.insert(0, compute_dir)
        from rna_template_common import (
            load_structure_residues,
            build_minimal_template_arrays,
            normalize_query_sequence,
        )

    t_pdb_id_raw = hit["pdb_id"]
    t_chain_id = hit["chain_id"]
    # Extract 4-char base PDB ID (search results have "1et4_D" format)
    t_pdb_base = _extract_base_pdb_id(t_pdb_id_raw)

    # Find CIF file using base PDB ID
    cif_path = _find_cif_path_cached(cif_database_dir, t_pdb_base)
    if cif_path is None:
        logger.debug(f"Online build: CIF not found for {t_pdb_base} (raw={t_pdb_id_raw})")
        return None

    try:
        residues = load_structure_residues(cif_path, chain_id=t_chain_id)
    except Exception as e:
        logger.debug(f"Online build: failed to load residues {t_pdb_base}:{t_chain_id}: {e}")
        return None

    try:
        template_name = f"{t_pdb_base}.cif:{t_chain_id}"
        td = build_minimal_template_arrays(
            query_seq=query_seq,
            residues=residues,
            template_name=template_name,
            anchor_mode=anchor_mode,
        )
        # Return only the features needed by TemplateEmbedder
        return {
            "template_aatype": td["template_aatype"],
            "template_distogram": td["template_distogram"],
            "template_pseudo_beta_mask": td["template_pseudo_beta_mask"],
            "template_unit_vector": td["template_unit_vector"],
            "template_backbone_frame_mask": td["template_backbone_frame_mask"],
        }
    except Exception as e:
        logger.debug(f"Online build: failed to build template {t_pdb_base}:{t_chain_id}: {e}")
        return None


def _stack_online_templates(
    template_dicts: List[Dict[str, np.ndarray]],
    max_templates: int,
    query_n: int,
) -> Dict[str, np.ndarray]:
    """Stack multiple single-template dicts into [T, N, ...] format."""
    features = {
        "template_aatype": np.full((max_templates, query_n), RNA_SEQ_TO_ID["-"], dtype=np.int32),
        "template_distogram": np.zeros((max_templates, query_n, query_n, 39), dtype=np.float32),
        "template_pseudo_beta_mask": np.zeros((max_templates, query_n, query_n), dtype=np.float32),
        "template_unit_vector": np.zeros((max_templates, query_n, query_n, 3), dtype=np.float32),
        "template_backbone_frame_mask": np.zeros((max_templates, query_n, query_n), dtype=np.float32),
    }

    for i, td in enumerate(template_dicts[:max_templates]):
        for key in features:
            features[key][i] = td[key]

    return features


# ── Manual template building (from user-specified CIF/PDB) ───────────

def _build_single_template_from_structure(
    query_seq: str,
    structure_path: str,
    chain_id: str = "",
    anchor_mode: str = "base_center_fallback",
) -> Optional[Dict[str, np.ndarray]]:
    """Build template features from a manually specified structure file.

    Unlike _build_single_template_online which resolves CIF from a database
    directory using a search hit dict, this function directly reads from the
    given file path.  Supports .cif, .mmcif, and .pdb formats (via BioPython).

    Args:
        query_seq: Normalized query RNA sequence.
        structure_path: Absolute path to the structure file.
        chain_id: Chain to extract.  If empty, auto-detects (single-chain only).
        anchor_mode: Anchor computation mode.

    Returns:
        Dict of template features ``[N_query, ...]`` or ``None`` on failure.
    """
    try:
        from protenix.data.rna_template.rna_template_common_online import (
            load_structure_residues,
            build_minimal_template_arrays,
        )
    except ImportError:
        compute_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
            "rna_template", "compute",
        )
        if compute_dir not in sys.path:
            sys.path.insert(0, compute_dir)
        from rna_template_common import (  # noqa: F811
            load_structure_residues,
            build_minimal_template_arrays,
        )

    if not os.path.exists(structure_path):
        logger.warning(f"Manual template structure not found: {structure_path}")
        return None

    try:
        residues = load_structure_residues(
            structure_path, chain_id=chain_id if chain_id else None
        )
    except Exception as e:
        logger.warning(
            f"Manual template: failed to load residues from "
            f"{structure_path} (chain={chain_id or 'auto'}): {e}"
        )
        return None

    try:
        template_name = f"manual:{os.path.basename(structure_path)}"
        if chain_id:
            template_name += f":{chain_id}"
        td = build_minimal_template_arrays(
            query_seq=query_seq,
            residues=residues,
            template_name=template_name,
            anchor_mode=anchor_mode,
        )
        return {
            "template_aatype": td["template_aatype"],
            "template_distogram": td["template_distogram"],
            "template_pseudo_beta_mask": td["template_pseudo_beta_mask"],
            "template_unit_vector": td["template_unit_vector"],
            "template_backbone_frame_mask": td["template_backbone_frame_mask"],
        }
    except Exception as e:
        logger.warning(
            f"Manual template: failed to build template from "
            f"{structure_path}: {e}"
        )
        return None


# ═════════════════════════════════════════════════════════════════════════
# Main Featurizer Class
# ═════════════════════════════════════════════════════════════════════════

class RNATemplateFeaturizer:
    """
    Featurizer for RNA structural templates in Protenix.

    Supports two modes:
      **Online mode** (search_results_path + cif_database_dir provided):
        Per-hit filtering and 3D feature building at training time from CIF files.
        Mirrors the protein template pipeline's online approach.

      **Offline mode** (template_index_path provided, legacy):
        Loads pre-computed .npz template tensors.

    Data-leakage prevention (active during training only):
      1. Self-hit exclusion: templates whose base PDB matches the query.
      2. Per-query temporal filtering: templates released after
         (query_release_date − 60 days) are rejected.
      3. PDB API fallback: when RNA3DB metadata lacks a date, queries RCSB PDB
         API and caches. Still unknown → REJECT (conservative).

    Args:
        template_database_dir: Directory with pre-computed .npz files (offline mode).
        template_index_path: JSON index for offline mode.
        max_templates: Maximum templates per RNA chain.
        rna3db_metadata_path: RNA3DB filter.json for release-date lookup.
        search_results_path: Path to search_results.json (online mode).
        cif_database_dir: Root directory of CIF files (online mode).
    """

    def __init__(
        self,
        template_database_dir: str = "",
        template_index_path: str = "",
        max_templates: int = 4,
        rna3db_metadata_path: str = "",
        search_results_path: str = "",
        cif_database_dir: str = "",
        manual_template_hints_path: str = "",
    ):
        self.template_database_dir = template_database_dir
        self.max_templates = max_templates
        self.cif_database_dir = cif_database_dir

        # ── Determine mode ──
        self.online_mode = bool(search_results_path and cif_database_dir)

        if self.online_mode:
            # Online mode: load search_results.json and build sequence → hits mapping
            self._index = {}  # Not used in online mode
            self._search_hits = self._load_search_results(search_results_path)
            logger.info(
                f"RNA template featurizer: ONLINE mode enabled "
                f"({len(self._search_hits)} sequences, cif_dir={cif_database_dir})"
            )
        else:
            # Offline mode: load NPZ index
            self._search_hits = {}
            self._index = _load_rna_template_index(template_index_path, fail_fast=True)
            if self._index:
                logger.info(
                    f"RNA template featurizer: OFFLINE mode "
                    f"({len(self._index)} sequences from {template_index_path})"
                )
            else:
                logger.info("RNA template index is empty or not configured.")

        # ── Release-date lookup for temporal filtering ──
        self._release_dates: Dict[str, datetime] = {}
        if rna3db_metadata_path:
            self._release_dates = _load_rna_release_dates(rna3db_metadata_path)

        # ── Initialize PDB API cache ──
        cache_dir = template_database_dir or os.path.dirname(search_results_path or "")
        if cache_dir:
            _init_pdb_api_cache(cache_dir)

        # ── Load training-time manual template hints (v6) ──
        # JSON mapping pdb_id → {entity_id → templateHints dict}
        self._training_manual_hints: Dict[str, Dict[str, dict]] = {}
        if manual_template_hints_path:
            if os.path.exists(manual_template_hints_path):
                with open(manual_template_hints_path, "r") as f:
                    self._training_manual_hints = json.load(f)
                logger.info(
                    f"Training manual template hints loaded: "
                    f"{len(self._training_manual_hints)} PDB entries "
                    f"from {manual_template_hints_path}"
                )
            else:
                logger.warning(
                    f"manual_template_hints_path='{manual_template_hints_path}' "
                    f"does not exist — training manual hints disabled."
                )

    @staticmethod
    def _load_search_results(search_results_path: str) -> Dict[str, List[dict]]:
        """Load search_results.json and build sequence → [hit list] mapping.

        search_results.json format:
            {query_id: {query_sequence: str, templates: [{pdb_id, chain_id, identity, ...}]}}

        Returns:
            {normalized_sequence: [hit_dicts]} with hits sorted by identity desc.
        """
        if not os.path.exists(search_results_path):
            raise FileNotFoundError(
                f"search_results.json not found at '{search_results_path}'."
            )

        with open(search_results_path, "r") as f:
            raw = json.load(f)

        seq_to_hits: Dict[str, List[dict]] = {}
        for query_id, info in raw.items():
            seq = _normalize_rna_sequence(info.get("query_sequence", ""))
            if len(seq) < 5:
                continue
            templates = info.get("templates", [])
            if not templates:
                continue
            # Merge hits for the same sequence (from different query_ids)
            existing = seq_to_hits.get(seq, [])
            seen_keys = {(h["pdb_id"], h["chain_id"]) for h in existing}
            for t in templates:
                key = (t["pdb_id"], t["chain_id"])
                if key not in seen_keys:
                    existing.append(t)
                    seen_keys.add(key)
            seq_to_hits[seq] = existing

        # Sort each hit list by identity descending
        for seq in seq_to_hits:
            seq_to_hits[seq].sort(key=lambda x: -x.get("identity", 0))

        logger.info(
            f"Search results loaded: {len(seq_to_hits)} unique sequences, "
            f"{sum(len(v) for v in seq_to_hits.values())} total hits"
        )
        return seq_to_hits

    def _find_templates_for_sequence(self, sequence: str) -> List[str]:
        """Look up template .npz paths for a given RNA sequence (offline mode)."""
        if sequence in self._index:
            return self._index[sequence]

        seq_u = sequence.replace("T", "U").replace("t", "u")
        if seq_u in self._index:
            return self._index[seq_u]

        return []

    def _find_hits_for_sequence(self, sequence: str) -> List[dict]:
        """Look up template hits for a given RNA sequence (online mode)."""
        seq_norm = _normalize_rna_sequence(sequence)
        if seq_norm in self._search_hits:
            return self._search_hits[seq_norm]

        # Try with T→U
        seq_u = sequence.replace("T", "U").replace("t", "u").upper()
        seq_u = "".join(c if c in {"A", "G", "C", "U", "N"} else "N" for c in seq_u)
        if seq_u in self._search_hits:
            return self._search_hits[seq_u]

        return []

    def _get_release_date(self, pdb_id_4char: str) -> Optional[datetime]:
        """Get release date from RNA3DB metadata, with PDB API fallback.

        Returns datetime if found, None if unknown from all sources.
        """
        # 1. Check RNA3DB metadata
        tpl_date = self._release_dates.get(pdb_id_4char)
        if tpl_date is not None:
            return tpl_date

        # 2. Check PDB API cache
        with _PDB_API_CACHE_LOCK:
            if pdb_id_4char in _PDB_API_DATE_CACHE:
                return _PDB_API_DATE_CACHE[pdb_id_4char]

        # 3. Query PDB API
        api_date = _fetch_pdb_release_date(pdb_id_4char)
        if api_date is not None:
            # Also update local release_dates for future lookups within same session
            self._release_dates[pdb_id_4char] = api_date
        return api_date

    def _filter_hits_online(
        self,
        hits: List[dict],
        query_pdb_id: Optional[str],
        cutoff_date: Optional[datetime],
    ) -> Tuple[List[dict], Dict[str, int]]:
        """Filter hit list by self-hit exclusion and temporal cutoff (online mode).

        Unlike offline mode which filters NPZ paths, this filters the raw hit dicts
        and uses PDB API fallback for unknown dates.

        Returns:
            (filtered_hits, stats)
        """
        if not query_pdb_id and cutoff_date is None:
            return hits, {"self_hit": 0, "future": 0, "no_date": 0}

        filtered = []
        stats = {"self_hit": 0, "future": 0, "no_date": 0}

        for hit in hits:
            tpl_pdb = _extract_base_pdb_id(hit["pdb_id"])

            # 1) Self-hit exclusion
            if query_pdb_id and tpl_pdb == query_pdb_id:
                stats["self_hit"] += 1
                continue

            # 2) Temporal filtering with PDB API fallback
            if cutoff_date is not None:
                tpl_date = self._get_release_date(tpl_pdb)
                if tpl_date is not None:
                    if tpl_date > cutoff_date:
                        stats["future"] += 1
                        continue
                else:
                    # Unknown date even after PDB API → REJECT (conservative)
                    stats["no_date"] += 1
                    continue

            filtered.append(hit)

        return filtered, stats

    def _filter_candidates(
        self,
        candidate_paths: List[str],
        query_pdb_id: Optional[str],
        cutoff_date: Optional[datetime],
    ) -> Tuple[List[str], Dict[str, int]]:
        """Filter candidate NPZ paths (offline mode) with PDB API fallback.

        Now rejects unknown-date templates (conservative) after trying PDB API.
        """
        if not query_pdb_id and cutoff_date is None:
            return candidate_paths, {"self_hit": 0, "future": 0, "no_date": 0}

        filtered = []
        stats = {"self_hit": 0, "future": 0, "no_date": 0}

        for p in candidate_paths:
            tpl_pdb = _extract_pdb_from_npz_path(p)

            # 1) Self-hit exclusion
            if query_pdb_id and tpl_pdb == query_pdb_id:
                stats["self_hit"] += 1
                continue

            # 2) Temporal filtering with PDB API fallback
            if cutoff_date is not None:
                tpl_date = self._get_release_date(tpl_pdb)
                if tpl_date is not None:
                    if tpl_date > cutoff_date:
                        stats["future"] += 1
                        continue
                else:
                    # Unknown date even after PDB API → REJECT (conservative)
                    stats["no_date"] += 1
                    continue

            filtered.append(p)

        return filtered, stats

    # ── Online feature building ──────────────────────────────────────────

    def _build_chain_features_online(
        self,
        sequence: str,
        hits: List[dict],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Build template features for one chain from CIF files online.

        Iterates through hits, building features per-hit until max_templates
        successful builds are collected. Failed hits are skipped.

        Returns:
            Dict of stacked features [T, N, ...] or None if no successful builds.
        """
        query_seq = _normalize_rna_sequence(sequence)
        successful_templates = []

        for hit in hits:
            if len(successful_templates) >= self.max_templates:
                break

            td = _build_single_template_online(
                query_seq=query_seq,
                hit=hit,
                cif_database_dir=self.cif_database_dir,
            )
            if td is not None:
                successful_templates.append(td)

        if not successful_templates:
            return None

        # Stack into [T, N, ...] format
        query_n = len(query_seq)
        return _stack_online_templates(successful_templates, self.max_templates, query_n)

    # ── Manual template resolution ──────────────────────────────────────

    def _build_from_manual_spec(
        self,
        query_seq: str,
        spec: dict,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Build a single template from a manual specification dict.

        Supported ``spec["type"]`` values:

        * ``"structure"`` — CIF/PDB file at ``spec["path"]``, optional
          ``spec["chain_id"]``.
        * ``"npz"`` — Pre-computed NPZ at ``spec["path"]`` (first template
          slot is used).
        """
        hint_type = spec.get("type", "structure")
        path = spec.get("path", "")

        if not path:
            logger.warning("Manual template spec missing 'path'")
            return None
        if not os.path.exists(path):
            logger.warning(f"Manual template path not found: {path}")
            return None

        if hint_type == "npz":
            features = _load_and_crop_rna_template(path, query_seq, max_templates=1)
            if features is None:
                return None
            # Extract first template slot as individual dict
            return {key: features[key][0] for key in RNA_TEMPLATE_FEATURES}

        elif hint_type == "structure":
            chain_id = spec.get("chain_id", "")
            return _build_single_template_from_structure(
                query_seq=query_seq,
                structure_path=path,
                chain_id=chain_id,
            )
        else:
            logger.warning(f"Unknown manual template type: {hint_type}")
            return None

    def _collect_search_templates(
        self,
        sequence: str,
        query_pdb_id: Optional[str],
        cutoff_date: Optional[datetime],
        max_count: int,
    ) -> Tuple[List[Dict[str, np.ndarray]], Dict[str, int]]:
        """Collect individual template dicts from search (online or offline).

        This is a helper used by :meth:`_resolve_with_manual` to obtain
        search-based templates that can be merged with manual templates at
        the slot level.

        Returns:
            ``(list_of_individual_template_dicts, filter_stats_dict)``
        """
        stats: Dict[str, int] = {"self_hit": 0, "future": 0, "no_date": 0}

        if self.online_mode:
            hits = self._find_hits_for_sequence(sequence)
            if not hits:
                return [], stats

            hits, stats = self._filter_hits_online(hits, query_pdb_id, cutoff_date)
            if not hits:
                return [], stats

            query_seq = _normalize_rna_sequence(sequence)
            templates: List[Dict[str, np.ndarray]] = []
            for hit in hits:
                if len(templates) >= max_count:
                    break
                td = _build_single_template_online(
                    query_seq, hit, self.cif_database_dir,
                )
                if td is not None:
                    templates.append(td)
            return templates, stats

        else:
            # Offline mode: load first valid NPZ → extract individual templates
            template_paths = self._find_templates_for_sequence(sequence)
            if not template_paths:
                return [], stats

            template_paths, stats = self._filter_candidates(
                template_paths, query_pdb_id, cutoff_date,
            )
            if not template_paths:
                return [], stats

            for p in template_paths:
                resolved = p if os.path.isabs(p) else os.path.join(self.template_database_dir, p)
                features = _load_and_crop_rna_template(resolved, sequence, max_count)
                if features is not None:
                    templates = []
                    num_t = features["template_aatype"].shape[0]
                    for t in range(min(num_t, max_count)):
                        td = {key: features[key][t] for key in RNA_TEMPLATE_FEATURES}
                        templates.append(td)
                    if templates:
                        return templates, stats
            return [], stats

    def _resolve_with_manual(
        self,
        entity_id,
        sequence: str,
        hints: dict,
        query_pdb_id: Optional[str],
        cutoff_date: Optional[datetime],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Resolve templates for an entity using manual hints + optional fallback.

        Modes (``hints["mode"]``):

        * ``manual_only``    — Only manual templates; no search fallback.
        * ``prefer_manual``  — Use manual if any succeed; full search fallback
          only when **all** manual templates fail.
        * ``hybrid``         — Manual templates fill front slots; remaining
          slots filled by search.  *(recommended default)*
        * ``default_only``   — Ignore manual specs, use search only.

        Returns:
            Stacked features ``[T, N, ...]`` or ``None``.
        """
        mode = hints.get("mode", "prefer_manual")
        manual_specs = hints.get("manual_templates", [])
        query_seq = _normalize_rna_sequence(sequence)

        # ── Build manual templates ──
        manual_tds: List[Dict[str, np.ndarray]] = []
        if mode != "default_only":
            for spec in manual_specs:
                if len(manual_tds) >= self.max_templates:
                    break
                td = self._build_from_manual_spec(query_seq, spec)
                if td is not None:
                    manual_tds.append(td)
                    label = spec.get("label", os.path.basename(spec.get("path", "?")))
                    logger.info(
                        f"Manual template built for entity {entity_id}: {label}"
                    )

        # ── Determine search fallback ──
        remaining_slots = self.max_templates - len(manual_tds)
        search_tds: List[Dict[str, np.ndarray]] = []

        if mode == "manual_only":
            pass  # No search fallback
        elif mode == "prefer_manual":
            # Fallback to search only when ALL manual templates failed
            if not manual_tds:
                search_tds, _ = self._collect_search_templates(
                    sequence, query_pdb_id, cutoff_date, self.max_templates,
                )
        elif mode == "hybrid":
            # Fill remaining slots with search results
            if remaining_slots > 0:
                search_tds, _ = self._collect_search_templates(
                    sequence, query_pdb_id, cutoff_date, remaining_slots,
                )
        elif mode == "default_only":
            search_tds, _ = self._collect_search_templates(
                sequence, query_pdb_id, cutoff_date, self.max_templates,
            )

        # ── Merge: manual first, then search ──
        all_tds = manual_tds[: self.max_templates]
        remain = self.max_templates - len(all_tds)
        if remain > 0:
            all_tds.extend(search_tds[:remain])

        if not all_tds:
            logger.info(
                f"No templates available for entity {entity_id} "
                f"(mode={mode}, manual={len(manual_tds)}, search={len(search_tds)})"
            )
            return None

        logger.info(
            f"Template resolution for entity {entity_id}: "
            f"mode={mode}, manual={len(manual_tds)}, search={len(search_tds)}, "
            f"total={len(all_tds)}/{self.max_templates}"
        )
        return _stack_online_templates(all_tds, self.max_templates, len(query_seq))

    # ── Main feature assembly ────────────────────────────────────────────

    def get_rna_template_features(
        self,
        rna_sequences: Dict[int, str],
        token_entity_ids: np.ndarray,
        token_res_ids: np.ndarray,
        num_tokens: int,
        query_pdb_id: Optional[str] = None,
        query_release_date: Optional[datetime] = None,
        manual_template_hints: Optional[Dict[str, dict]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Load and assemble RNA template features for all RNA chains.

        When *manual_template_hints* is provided for a given entity, the
        resolver-with-manual path is used (manual templates first, then
        optional search fallback depending on mode).  Entities without
        manual hints use the existing online/offline pipeline unchanged.

        Online mode:
        1. Look up query sequence → hit list from search_results.json
        2. Per-hit filter (self-hit, temporal with PDB API fallback)
        3. Per-hit build from CIF → collect up to max_templates
        4. Place features at correct token positions

        Offline mode (legacy):
        1. Look up sequence → NPZ paths from index
        2. Filter NPZ paths
        3. Load first valid NPZ
        4. Place features at correct token positions
        """
        t_start = time.time()

        # Compute temporal cutoff
        cutoff_date: Optional[datetime] = None
        if query_release_date is not None:
            cutoff_date = query_release_date - timedelta(days=DAYS_BEFORE_QUERY_DATE)

        q_pdb = query_pdb_id.lower().strip() if query_pdb_id else None

        # Initialize output tensors
        rna_features = {
            "rna_template_aatype": np.full(
                (self.max_templates, num_tokens), RNA_SEQ_TO_ID["-"], dtype=np.int32
            ),
            "rna_template_distogram": np.zeros(
                (self.max_templates, num_tokens, num_tokens, 39), dtype=np.float32
            ),
            "rna_template_pseudo_beta_mask": np.zeros(
                (self.max_templates, num_tokens, num_tokens), dtype=np.float32
            ),
            "rna_template_unit_vector": np.zeros(
                (self.max_templates, num_tokens, num_tokens, 3), dtype=np.float32
            ),
            "rna_template_backbone_frame_mask": np.zeros(
                (self.max_templates, num_tokens, num_tokens), dtype=np.float32
            ),
        }

        rna_token_mask = np.zeros(num_tokens, dtype=np.float32)
        has_any_template = False
        total_stats = {"self_hit": 0, "future": 0, "no_date": 0}

        for entity_id, sequence in rna_sequences.items():
            if len(sequence) <= 4:
                continue

            # ── Manual template override (if hints provided for this entity) ──
            entity_hints = (manual_template_hints or {}).get(str(entity_id))
            if entity_hints and entity_hints.get("manual_templates"):
                chain_features = self._resolve_with_manual(
                    entity_id, sequence, entity_hints, q_pdb, cutoff_date,
                )

            elif self.online_mode:
                # ── ONLINE PATH (unchanged) ──
                hits = self._find_hits_for_sequence(sequence)
                if not hits:
                    continue

                # Per-hit filtering
                hits, stats = self._filter_hits_online(hits, q_pdb, cutoff_date)
                for k in total_stats:
                    total_stats[k] += stats[k]

                if not hits:
                    continue

                # Build features from CIF online
                chain_features = self._build_chain_features_online(sequence, hits)

            else:
                # ── OFFLINE PATH (legacy, unchanged) ──
                template_paths = self._find_templates_for_sequence(sequence)
                if not template_paths:
                    continue

                template_paths, stats = self._filter_candidates(
                    template_paths, q_pdb, cutoff_date
                )
                for k in total_stats:
                    total_stats[k] += stats[k]

                if not template_paths:
                    continue

                # Resolve paths
                resolved_paths = []
                for p in template_paths[:self.max_templates]:
                    if os.path.isabs(p):
                        resolved_paths.append(p)
                    else:
                        resolved_paths.append(os.path.join(self.template_database_dir, p))

                chain_features = None
                for npz_path in resolved_paths:
                    chain_features = _load_and_crop_rna_template(
                        npz_path, sequence, self.max_templates
                    )
                    if chain_features is not None:
                        break

            if chain_features is None:
                continue

            # Pad to max_templates
            chain_features = _pad_rna_template_features(chain_features, self.max_templates)

            # Find token indices for this entity
            entity_mask = token_entity_ids == entity_id
            entity_token_indices = np.where(entity_mask)[0]

            if len(entity_token_indices) == 0:
                continue

            # Map residue IDs to positions in the chain template
            res_ids = token_res_ids[entity_token_indices]
            chain_indices = res_ids - res_ids.min()

            template_n = chain_features["template_aatype"].shape[1]
            valid_mask = (chain_indices >= 0) & (chain_indices < template_n)
            if not valid_mask.all():
                logger.warning(
                    f"RNA template: some residue indices out of bounds for entity {entity_id}. "
                    f"range=[{chain_indices.min()}, {chain_indices.max()}], template_n={template_n}. "
                    f"Clamping."
                )
                chain_indices = np.clip(chain_indices, 0, template_n - 1)

            token_idx = entity_token_indices
            chain_idx = chain_indices

            # 1D features: aatype [T, N]
            for t in range(self.max_templates):
                rna_features["rna_template_aatype"][t, token_idx] = \
                    chain_features["template_aatype"][t, chain_idx]

            # 2D features: distogram, masks, unit_vector [T, N, N, ...]
            ix = np.ix_(token_idx, token_idx)
            for t in range(self.max_templates):
                chain_ix = np.ix_(chain_idx, chain_idx)
                rna_features["rna_template_distogram"][t][ix] = \
                    chain_features["template_distogram"][t][chain_ix]
                rna_features["rna_template_pseudo_beta_mask"][t][ix] = \
                    chain_features["template_pseudo_beta_mask"][t][chain_ix]
                rna_features["rna_template_unit_vector"][t][ix] = \
                    chain_features["template_unit_vector"][t][chain_ix]
                rna_features["rna_template_backbone_frame_mask"][t][ix] = \
                    chain_features["template_backbone_frame_mask"][t][chain_ix]

            rna_token_mask[token_idx] = 1.0
            has_any_template = True

        # Build block mask
        rna_features["rna_template_block_mask"] = (
            rna_token_mask[:, None] * rna_token_mask[None, :]
        ).astype(np.float32)

        elapsed = time.time() - t_start
        mode_str = "ONLINE" if self.online_mode else "OFFLINE"
        if has_any_template:
            logger.info(
                f"RNA template features [{mode_str}] loaded for {len(rna_sequences)} chains "
                f"in {elapsed:.2f}s (filtered: self_hit={total_stats['self_hit']}, "
                f"future={total_stats['future']}, no_date={total_stats['no_date']})"
            )
        elif any(v > 0 for v in total_stats.values()):
            logger.info(
                f"RNA templates [{mode_str}]: all candidates filtered out for query "
                f"pdb={q_pdb} (self_hit={total_stats['self_hit']}, "
                f"future={total_stats['future']}, no_date={total_stats['no_date']})"
            )

        return rna_features

    def __call__(
        self,
        token_array,
        atom_array,
        bioassembly_dict: dict,
        inference_mode: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Main entry point: extract RNA template features for a sample.

        Identifies RNA chains using the same RNA-First strategy as RiNALMoFeaturizer,
        then loads and assembles template features.

        When an ``rnaSequence`` entity in the input JSON includes a
        ``templateHints`` dict, manual template resolution is used for that
        entity (with optional search fallback depending on the mode).
        Entities without ``templateHints`` use the default search pipeline.

        During training, per-query temporal filtering and self-hit exclusion are
        applied using ``bioassembly_dict["pdb_id"]`` and
        ``bioassembly_dict["release_date"]``.
        """
        centre_atom_indices = token_array.get_annotation("centre_atom_index")
        centre_atom_array = atom_array[centre_atom_indices]
        num_tokens = len(token_array)

        # Extract query metadata for leakage prevention
        query_pdb_id: Optional[str] = None
        query_release_date: Optional[datetime] = None
        if not inference_mode:
            query_pdb_id = bioassembly_dict.get("pdb_id")
            rd_str = bioassembly_dict.get("release_date")
            if rd_str:
                try:
                    query_release_date = datetime.strptime(str(rd_str), "%Y-%m-%d")
                except (ValueError, TypeError):
                    pass

        # Identify RNA entities and extract manual template hints.
        # Uses RNA-First strategy consistent with RNALM featurizer:
        #   1. All rnaSequence/rnaChain entities → accepted as RNA
        #   2. dnaSequence/dnaChain entities containing uracil (U) → reclassified as RNA
        #      (uracil is exclusive to RNA; preserves A-form helix, 2'-OH structural priors)
        rna_sequences = {}
        manual_template_hints: Dict[str, dict] = {}

        if inference_mode:
            for i, entity_info_wrapper in enumerate(bioassembly_dict["sequences"]):
                entity_id = str(i + 1)
                entity_type = list(entity_info_wrapper.keys())[0]
                if entity_type in ("rnaSequence", "rnaChain"):
                    # RNA-labeled entities are always accepted for RNA template search
                    entity_info = entity_info_wrapper[entity_type]
                    seq = entity_info["sequence"]
                    rna_sequences[entity_id] = seq
                    # Extract manual template hints if present
                    hints = entity_info.get("templateHints")
                    if hints:
                        manual_template_hints[entity_id] = hints
                        logger.info(
                            f"Entity {entity_id}: templateHints found "
                            f"(mode={hints.get('mode', 'prefer_manual')}, "
                            f"manual_templates={len(hints.get('manual_templates', []))})"
                        )
                elif entity_type in ("dnaSequence", "dnaChain"):
                    # RNA-First: reclassify DNA entities that contain uracil as RNA
                    entity_info = entity_info_wrapper[entity_type]
                    seq = entity_info["sequence"]
                    if "U" in seq or "u" in seq:
                        logger.info(
                            f"[RNA-First] Entity {entity_id} specified as {entity_type} "
                            f"but contains uracil (RNA base). Including in RNA template "
                            f"search. Seq: {seq[:40]}..."
                        )
                        rna_sequences[entity_id] = seq
                        # Also extract templateHints for reclassified DNA-with-U entities
                        hints = entity_info.get("templateHints")
                        if hints:
                            manual_template_hints[entity_id] = hints
                            logger.info(
                                f"Entity {entity_id} (reclassified DNA→RNA): "
                                f"templateHints found (mode={hints.get('mode', 'prefer_manual')})"
                            )
        else:
            # Training mode: use chain_mol_type annotations as initial labels,
            # then apply RNA-First reclassification (consistent with RNALM)
            is_rna = centre_atom_array.chain_mol_type == "rna"
            rna_entity_ids = set(centre_atom_array.label_entity_id[is_rna])

            is_dna = centre_atom_array.chain_mol_type == "dna"
            dna_entity_ids = set(centre_atom_array.label_entity_id[is_dna])

            # All RNA-labeled entities are accepted
            for entity_id in rna_entity_ids:
                seq = bioassembly_dict["sequences"].get(str(entity_id), "")
                if seq:
                    rna_sequences[entity_id] = seq

            # RNA-First: DNA-labeled entities with uracil → reclassify as RNA
            for entity_id in dna_entity_ids:
                seq = bioassembly_dict["sequences"].get(str(entity_id), "")
                if seq and ("U" in seq or "u" in seq):
                    logger.info(
                        f"[RNA-First] Entity {entity_id} labeled as DNA but contains "
                        f"uracil. Including in RNA template search. Seq: {seq[:40]}..."
                    )
                    rna_sequences[entity_id] = seq

            # ── Training manual template hints (v6) ──
            # Look up per-PDB, per-entity hints from the external JSON config.
            # Only triggered when manual_template_hints_path was configured.
            if self._training_manual_hints and query_pdb_id:
                pdb_hints = self._training_manual_hints.get(
                    query_pdb_id,
                    self._training_manual_hints.get(query_pdb_id.upper(), {}),
                )
                if pdb_hints:
                    for entity_id in rna_sequences:
                        entity_hints = pdb_hints.get(
                            str(entity_id),
                            pdb_hints.get("*", None),  # "*" = wildcard: apply to all entities
                        )
                        if entity_hints:
                            manual_template_hints[str(entity_id)] = entity_hints
                            logger.info(
                                f"Training manual hints for pdb={query_pdb_id} "
                                f"entity={entity_id}: mode={entity_hints.get('mode', 'prefer_manual')}, "
                                f"manual_templates={len(entity_hints.get('manual_templates', []))}"
                            )

        if not rna_sequences:
            return _empty_rna_template_features(num_tokens, self.max_templates)

        token_entity_ids = centre_atom_array.label_entity_id
        token_res_ids = centre_atom_array.res_id

        return self.get_rna_template_features(
            rna_sequences=rna_sequences,
            token_entity_ids=token_entity_ids,
            token_res_ids=token_res_ids,
            num_tokens=num_tokens,
            query_pdb_id=query_pdb_id,
            query_release_date=query_release_date,
            manual_template_hints=manual_template_hints if manual_template_hints else None,
        )
