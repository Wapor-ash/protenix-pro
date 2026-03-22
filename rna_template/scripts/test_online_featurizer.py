#!/usr/bin/env python3
"""
Test script for RNA Template Featurizer v3 (online mode).

Tests:
  1. PDB API release date lookup + caching (Problem 1)
  2. Online template building from CIF files (Problem 2)
  3. Temporal filtering with PDB API fallback (conservative rejection)
  4. Self-hit exclusion
  5. Backward compatibility with offline mode
  6. Feature shape validation

Usage:
    cd /inspire/ssd/project/sais-bio/public/ash_proj/code/protenix_new/Protenix
    python rna_template/scripts/test_online_featurizer.py
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta

import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Also add compute dir
COMPUTE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "compute")
sys.path.insert(0, COMPUTE_DIR)

# Paths
RNA_DATABASE_DIR = os.path.join(PROJECT_ROOT, "rna_database")
SEARCH_RESULTS_PATH = os.path.join(RNA_DATABASE_DIR, "search_results.json")
TEMPLATE_INDEX_PATH = os.path.join(RNA_DATABASE_DIR, "rna_template_index.json")
CIF_DATABASE_DIR = "/inspire/ssd/project/sais-bio/public/ash_proj/data/stanford-rna-3d-folding/part2/PDB_RNA"
RNA3DB_METADATA_PATH = "/inspire/ssd/project/sais-bio/public/ash_proj/data/RNA3D/rna3db-jsons/filter.json"

PASSED = 0
FAILED = 0


def test_result(name, passed, detail=""):
    global PASSED, FAILED
    if passed:
        PASSED += 1
        print(f"  ✓ {name}")
    else:
        FAILED += 1
        print(f"  ✗ {name}")
    if detail:
        print(f"    {detail}")


def test_pdb_api_lookup():
    """Test 1: PDB API release date lookup."""
    print("\n=== Test 1: PDB API Release Date Lookup ===")

    from protenix.data.rna_template.rna_template_featurizer import (
        _fetch_pdb_release_date,
        _PDB_API_DATE_CACHE,
    )

    # Test with a known PDB ID
    dt = _fetch_pdb_release_date("1g1x")
    test_result(
        "Known PDB (1g1x) returns valid date",
        dt is not None,
        f"date={dt.strftime('%Y-%m-%d') if dt else 'None'}"
    )

    # Test caching - second call should be instant
    t0 = time.time()
    dt2 = _fetch_pdb_release_date("1g1x")
    elapsed = time.time() - t0
    test_result(
        "Cached lookup is fast (<0.01s)",
        elapsed < 0.01 and dt2 == dt,
        f"elapsed={elapsed:.4f}s"
    )

    # Test with another PDB
    dt3 = _fetch_pdb_release_date("4tna")
    test_result(
        "Another PDB (4tna) returns valid date",
        dt3 is not None,
        f"date={dt3.strftime('%Y-%m-%d') if dt3 else 'None'}"
    )

    # Test with invalid PDB
    dt4 = _fetch_pdb_release_date("zzzz")
    test_result(
        "Invalid PDB (zzzz) returns None",
        dt4 is None,
        f"result={dt4}"
    )

    # Verify cache has entries
    test_result(
        "Cache populated",
        len(_PDB_API_DATE_CACHE) >= 3,
        f"cache_size={len(_PDB_API_DATE_CACHE)}"
    )


def test_online_featurizer_init():
    """Test 2: Online featurizer initialization."""
    print("\n=== Test 2: Online Featurizer Initialization ===")

    from protenix.data.rna_template.rna_template_featurizer import RNATemplateFeaturizer

    # Online mode
    featurizer = RNATemplateFeaturizer(
        template_database_dir=RNA_DATABASE_DIR,
        search_results_path=SEARCH_RESULTS_PATH,
        cif_database_dir=CIF_DATABASE_DIR,
        rna3db_metadata_path=RNA3DB_METADATA_PATH,
        max_templates=4,
    )

    test_result(
        "Online mode enabled",
        featurizer.online_mode is True,
    )
    test_result(
        "Search hits loaded",
        len(featurizer._search_hits) > 0,
        f"num_sequences={len(featurizer._search_hits)}"
    )
    test_result(
        "Release dates loaded",
        len(featurizer._release_dates) > 0,
        f"num_dates={len(featurizer._release_dates)}"
    )

    return featurizer


def test_online_template_building(featurizer):
    """Test 3: Online template building from CIF."""
    print("\n=== Test 3: Online Template Building from CIF ===")

    from protenix.data.rna_template.rna_template_featurizer import (
        _build_single_template_online,
        _normalize_rna_sequence,
    )

    # Get a test sequence + hit
    test_seq = None
    test_hit = None
    for seq, hits in featurizer._search_hits.items():
        if len(seq) < 200 and len(hits) > 0:  # Pick a short sequence for speed
            test_seq = seq
            test_hit = hits[0]
            break

    if test_seq is None:
        # Fallback: use any sequence
        test_seq = list(featurizer._search_hits.keys())[0]
        test_hit = featurizer._search_hits[test_seq][0]

    print(f"  Test sequence length: {len(test_seq)}")
    print(f"  Test hit: pdb={test_hit['pdb_id']}, chain={test_hit['chain_id']}, identity={test_hit.get('identity', 'N/A')}")

    # Build single template
    t0 = time.time()
    td = _build_single_template_online(
        query_seq=test_seq,
        hit=test_hit,
        cif_database_dir=CIF_DATABASE_DIR,
    )
    elapsed = time.time() - t0

    test_result(
        "Single template build succeeds",
        td is not None,
        f"elapsed={elapsed:.2f}s"
    )

    if td is not None:
        # Check feature shapes
        n = len(test_seq)
        test_result(
            "template_aatype shape",
            td["template_aatype"].shape == (n,),
            f"expected=({n},), got={td['template_aatype'].shape}"
        )
        test_result(
            "template_distogram shape",
            td["template_distogram"].shape == (n, n, 39),
            f"expected=({n},{n},39), got={td['template_distogram'].shape}"
        )
        test_result(
            "template_pseudo_beta_mask shape",
            td["template_pseudo_beta_mask"].shape == (n, n),
            f"expected=({n},{n}), got={td['template_pseudo_beta_mask'].shape}"
        )
        test_result(
            "template_unit_vector shape",
            td["template_unit_vector"].shape == (n, n, 3),
            f"expected=({n},{n},3), got={td['template_unit_vector'].shape}"
        )

        # Check non-trivial values
        mask_sum = float(td["template_pseudo_beta_mask"].sum())
        test_result(
            "Non-zero anchor mask",
            mask_sum > 0,
            f"mask_sum={mask_sum}"
        )

    return test_seq, test_hit


def test_chain_features_online(featurizer, test_seq):
    """Test 4: Full chain feature building (multiple templates stacked)."""
    print("\n=== Test 4: Chain Feature Stacking (Online) ===")

    hits = featurizer._find_hits_for_sequence(test_seq)
    test_result(
        "Hits found for sequence",
        len(hits) > 0,
        f"num_hits={len(hits)}"
    )

    if not hits:
        return

    t0 = time.time()
    chain_features = featurizer._build_chain_features_online(test_seq, hits)
    elapsed = time.time() - t0

    test_result(
        "Chain features built",
        chain_features is not None,
        f"elapsed={elapsed:.2f}s"
    )

    if chain_features is not None:
        n = len(test_seq)
        max_t = featurizer.max_templates
        test_result(
            "Stacked aatype shape [T, N]",
            chain_features["template_aatype"].shape == (max_t, n),
            f"expected=({max_t},{n}), got={chain_features['template_aatype'].shape}"
        )
        test_result(
            "Stacked distogram shape [T, N, N, 39]",
            chain_features["template_distogram"].shape == (max_t, n, n, 39),
            f"got={chain_features['template_distogram'].shape}"
        )


def test_temporal_filtering_with_pdb_api(featurizer):
    """Test 5: Temporal filtering uses PDB API fallback."""
    print("\n=== Test 5: Temporal Filtering with PDB API Fallback ===")

    # Create hits with a PDB ID that's NOT in RNA3DB metadata
    # but IS a valid PDB ID (so PDB API should find it)
    test_hits = [
        {"pdb_id": "1g1x_I", "chain_id": "I", "identity": 0.95},
    ]

    # Use a very old cutoff date → should reject (template is newer)
    old_cutoff = datetime(1990, 1, 1)
    filtered, stats = featurizer._filter_hits_online(
        test_hits,
        query_pdb_id=None,
        cutoff_date=old_cutoff,
    )
    test_result(
        "Old cutoff rejects template (future filter)",
        len(filtered) == 0 and stats["future"] > 0,
        f"filtered={len(filtered)}, stats={stats}"
    )

    # Use a very far future cutoff → should keep
    future_cutoff = datetime(2099, 1, 1)
    filtered2, stats2 = featurizer._filter_hits_online(
        test_hits,
        query_pdb_id=None,
        cutoff_date=future_cutoff,
    )
    test_result(
        "Far future cutoff keeps template",
        len(filtered2) == 1,
        f"filtered={len(filtered2)}, stats={stats2}"
    )


def test_self_hit_exclusion(featurizer):
    """Test 6: Self-hit exclusion."""
    print("\n=== Test 6: Self-Hit Exclusion ===")

    test_hits = [
        {"pdb_id": "1g1x_I", "chain_id": "I", "identity": 0.95},
        {"pdb_id": "4tna_A", "chain_id": "A", "identity": 0.90},
    ]

    # Exclude 1g1x as self-hit
    filtered, stats = featurizer._filter_hits_online(
        test_hits,
        query_pdb_id="1g1x",
        cutoff_date=None,
    )
    test_result(
        "Self-hit excluded (1g1x)",
        len(filtered) == 1 and stats["self_hit"] == 1,
        f"filtered={len(filtered)}, remaining_pdb={filtered[0]['pdb_id'] if filtered else 'none'}"
    )


def test_unknown_date_rejection(featurizer):
    """Test 7: Unknown date templates are REJECTED (conservative)."""
    print("\n=== Test 7: Unknown Date → REJECT (Conservative) ===")

    # Use a fake PDB ID that doesn't exist in RNA3DB or RCSB
    test_hits = [
        {"pdb_id": "zzzz_A", "chain_id": "A", "identity": 0.95},
    ]

    cutoff = datetime(2025, 1, 1)
    filtered, stats = featurizer._filter_hits_online(
        test_hits,
        query_pdb_id=None,
        cutoff_date=cutoff,
    )
    test_result(
        "Unknown date template rejected",
        len(filtered) == 0 and stats["no_date"] == 1,
        f"filtered={len(filtered)}, stats={stats}"
    )


def test_offline_backward_compat():
    """Test 8: Offline mode backward compatibility."""
    print("\n=== Test 8: Offline Mode Backward Compatibility ===")

    from protenix.data.rna_template.rna_template_featurizer import RNATemplateFeaturizer

    featurizer = RNATemplateFeaturizer(
        template_database_dir=RNA_DATABASE_DIR,
        template_index_path=TEMPLATE_INDEX_PATH,
        max_templates=4,
        rna3db_metadata_path=RNA3DB_METADATA_PATH,
    )

    test_result(
        "Offline mode (online_mode=False)",
        featurizer.online_mode is False,
    )
    test_result(
        "NPZ index loaded",
        len(featurizer._index) > 0,
        f"num_sequences={len(featurizer._index)}"
    )


def test_pdb_api_cache_persistence():
    """Test 9: PDB API cache persists to disk."""
    print("\n=== Test 9: PDB API Cache Persistence ===")

    from protenix.data.rna_template.rna_template_featurizer import _PDB_API_CACHE_PATH

    test_result(
        "Cache file path set",
        _PDB_API_CACHE_PATH is not None,
        f"path={_PDB_API_CACHE_PATH}"
    )

    if _PDB_API_CACHE_PATH:
        exists = os.path.exists(_PDB_API_CACHE_PATH)
        test_result(
            "Cache file exists on disk",
            exists,
        )

        if exists:
            with open(_PDB_API_CACHE_PATH) as f:
                cache_data = json.load(f)
            test_result(
                "Cache file has entries",
                len(cache_data) > 0,
                f"num_entries={len(cache_data)}"
            )


def test_get_rna_template_features_online(featurizer, test_seq):
    """Test 10: Full get_rna_template_features with online mode."""
    print("\n=== Test 10: Full Feature Assembly (Online Mode) ===")

    # Simulate a minimal token array
    n = min(len(test_seq), 100)  # Crop for speed
    seq_cropped = test_seq[:n]
    token_entity_ids = np.ones(n, dtype=object)  # All entity_id = 1
    token_res_ids = np.arange(1, n + 1)  # 1-based

    rna_sequences = {1: seq_cropped}

    t0 = time.time()
    features = featurizer.get_rna_template_features(
        rna_sequences=rna_sequences,
        token_entity_ids=token_entity_ids,
        token_res_ids=token_res_ids,
        num_tokens=n,
        query_pdb_id="xxxx",  # Fake PDB, won't match any template
        query_release_date=datetime(2026, 1, 1),
    )
    elapsed = time.time() - t0

    test_result(
        "Features returned",
        features is not None,
        f"elapsed={elapsed:.2f}s"
    )

    if features:
        max_t = featurizer.max_templates
        test_result(
            "rna_template_aatype shape",
            features["rna_template_aatype"].shape == (max_t, n),
        )
        test_result(
            "rna_template_block_mask shape",
            features["rna_template_block_mask"].shape == (n, n),
        )

        # Check if any template was actually loaded (block mask > 0)
        block_sum = float(features["rna_template_block_mask"].sum())
        test_result(
            "Block mask populated (templates found)",
            block_sum > 0,
            f"block_mask_sum={block_sum}"
        )


def main():
    global PASSED, FAILED

    print("=" * 60)
    print("RNA Template Featurizer v3 — Online Mode Test Suite")
    print("=" * 60)

    try:
        # Test 1: PDB API lookup
        test_pdb_api_lookup()

        # Test 2: Online featurizer init
        featurizer = test_online_featurizer_init()

        # Test 3: Single template building
        test_seq, test_hit = test_online_template_building(featurizer)

        # Test 4: Chain stacking
        test_chain_features_online(featurizer, test_seq)

        # Test 5: Temporal filtering
        test_temporal_filtering_with_pdb_api(featurizer)

        # Test 6: Self-hit exclusion
        test_self_hit_exclusion(featurizer)

        # Test 7: Unknown date rejection
        test_unknown_date_rejection(featurizer)

        # Test 8: Offline backward compat
        test_offline_backward_compat()

        # Test 9: Cache persistence
        test_pdb_api_cache_persistence()

        # Test 10: Full feature assembly
        test_get_rna_template_features_online(featurizer, test_seq)

    except Exception as e:
        FAILED += 1
        print(f"\n  ✗ UNEXPECTED ERROR: {e}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"Results: {PASSED} passed, {FAILED} failed, {PASSED + FAILED} total")
    print("=" * 60)

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
