import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.beta6.data_manifest import (
    CorpusManifestPolicy,
    build_pretrain_corpus_manifest,
    load_feature_matrix,
)
from ml.beta6.feature_store import build_feature_sequence_cache_key


def _write_csv(path: Path, matrix: np.ndarray) -> None:
    frame = pd.DataFrame(matrix, columns=[f"f{i}" for i in range(matrix.shape[1])])
    frame.to_csv(path, index=False)


def _write_labeled_frame(path: Path, *, elder_id: str = "HK0011_jessica") -> None:
    frame = pd.DataFrame(
        {
            "elder_id": [elder_id, elder_id, elder_id, elder_id],
            "timestamp": [
                "2025-12-04T07:00:00",
                "2025-12-04T07:05:00",
                "2025-12-05T12:00:00",
                "2025-12-05T12:15:00",
            ],
            "room": ["Bedroom", "Bedroom", "Kitchen", "Kitchen"],
            "activity": ["sleep", "bedroom_normal_use", "meal_preparation", "meal_preparation"],
            "pir": [1.0, 0.0, 1.0, 1.0],
            "door": [0.0, 0.0, 1.0, 0.0],
        }
    )
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def test_manifest_fingerprint_is_stable_across_reruns(tmp_path: Path):
    matrix = np.arange(80, dtype=np.float32).reshape(20, 4)
    _write_csv(tmp_path / "a.csv", matrix)
    _write_csv(tmp_path / "b.csv", matrix + 1.0)
    policy = CorpusManifestPolicy(min_rows=8, min_features=2, max_missing_ratio=0.2)

    manifest_a = build_pretrain_corpus_manifest(corpus_roots=[tmp_path], policy=policy)
    manifest_b = build_pretrain_corpus_manifest(corpus_roots=[tmp_path], policy=policy)

    assert manifest_a["fingerprint"]["value"] == manifest_b["fingerprint"]["value"]
    assert manifest_a["stats"]["records_kept"] == 2
    assert manifest_a["stats"]["p0_violations"] == 0


def test_manifest_dedupes_by_content_hash(tmp_path: Path):
    matrix = np.arange(40, dtype=np.float32).reshape(10, 4)
    _write_csv(tmp_path / "x.csv", matrix)
    _write_csv(tmp_path / "x_copy.csv", matrix)

    manifest = build_pretrain_corpus_manifest(
        corpus_roots=[tmp_path],
        policy=CorpusManifestPolicy(min_rows=4, min_features=2, max_missing_ratio=0.2),
    )
    assert manifest["stats"]["records_kept"] == 1
    assert manifest["stats"]["duplicates_removed"] == 1


def test_manifest_flags_missing_ratio_violation(tmp_path: Path):
    matrix = np.array(
        [
            [1.0, np.nan, np.nan],
            [2.0, np.nan, np.nan],
            [3.0, np.nan, np.nan],
            [4.0, np.nan, np.nan],
        ],
        dtype=np.float32,
    )
    _write_csv(tmp_path / "noisy.csv", matrix)
    manifest = build_pretrain_corpus_manifest(
        corpus_roots=[tmp_path],
        policy=CorpusManifestPolicy(min_rows=2, min_features=2, max_missing_ratio=0.3),
    )
    assert manifest["stats"]["records_kept"] == 0
    assert manifest["stats"]["p0_violations"] >= 1
    reasons = {item["reason_code"] for item in manifest["violations"]}
    assert "manifest_missing_ratio_violation" in reasons


def test_manifest_empty_corpus_is_p0_violation(tmp_path: Path):
    manifest = build_pretrain_corpus_manifest(
        corpus_roots=[tmp_path],
        policy=CorpusManifestPolicy(min_rows=2, min_features=2, max_missing_ratio=0.3),
    )
    assert manifest["stats"]["files_scanned"] == 0
    assert manifest["stats"]["records_kept"] == 0
    assert manifest["stats"]["p0_violations"] >= 1
    assert any(v["reason_code"] == "manifest_empty_corpus" for v in manifest["violations"])


def test_intake_manifest_tracks_user_and_date_tags(tmp_path: Path):
    _write_labeled_frame(tmp_path / "jessica_4_5dec.csv")

    manifest = build_pretrain_corpus_manifest(
        corpus_roots=[tmp_path],
        policy=CorpusManifestPolicy(min_rows=2, min_features=2, max_missing_ratio=0.2),
    )

    assert manifest["gate"]["approved"] is True
    assert manifest["summary"]["user_tags"] == ["HK0011_jessica"]
    assert manifest["summary"]["date_tags"] == ["2025-12-04", "2025-12-05"]
    entry = manifest["entries"][0]
    assert entry["source_tags"]["user_tags"] == ["HK0011_jessica"]
    assert entry["source_tags"]["date_tags"] == ["2025-12-04", "2025-12-05"]


def test_intake_summary_includes_per_room_per_date_label_counts(tmp_path: Path):
    _write_labeled_frame(tmp_path / "jessica_4_5dec.parquet")

    manifest = build_pretrain_corpus_manifest(
        corpus_roots=[tmp_path],
        policy=CorpusManifestPolicy(min_rows=2, min_features=2, max_missing_ratio=0.2),
    )

    counts = manifest["summary"]["per_room_per_date_label_counts"]
    assert {"room": "bedroom", "date": "2025-12-04", "activity": "sleep", "count": 1} in counts
    assert {
        "room": "kitchen",
        "date": "2025-12-05",
        "activity": "meal_preparation",
        "count": 2,
    } in counts


def test_intake_summary_uses_filename_date_tags_for_labeled_daily_files(tmp_path: Path):
    daily = pd.DataFrame(
        {
            "elder_id": ["HK0011_jessica", "HK0011_jessica"],
            "room": ["Bedroom", "Bedroom"],
            "activity": ["sleep", "bedroom_normal_use"],
            "pir": [1.0, 0.0],
            "door": [0.0, 0.0],
        }
    )
    daily.to_csv(tmp_path / "HK0011_jessica_2025-12-04.csv", index=False)

    manifest = build_pretrain_corpus_manifest(
        corpus_roots=[tmp_path],
        policy=CorpusManifestPolicy(min_rows=2, min_features=2, max_missing_ratio=0.2),
    )

    assert manifest["gate"]["approved"] is True
    entry_counts = manifest["entries"][0]["label_summary"]["per_room_per_date_label_counts"]
    assert {
        "room": "bedroom",
        "date": "2025-12-04",
        "activity": "sleep",
        "count": 1,
    } in entry_counts
    assert {
        "room": "bedroom",
        "date": "2025-12-04",
        "activity": "bedroom_normal_use",
        "count": 1,
    } in manifest["summary"]["per_room_per_date_label_counts"]


def test_manifest_duplicate_governance_is_not_path_order_dependent(tmp_path: Path):
    daily = pd.DataFrame(
        {
            "elder_id": ["HK0011_jessica", "HK0011_jessica"],
            "room": ["Bedroom", "Bedroom"],
            "activity": ["sleep", "sleep"],
            "pir": [1.0, 0.0],
            "door": [0.0, 1.0],
        }
    )
    daily.to_csv(tmp_path / "a_bad.csv", index=False)
    daily.to_csv(tmp_path / "z_HK0011_jessica_2025-12-04.csv", index=False)

    manifest = build_pretrain_corpus_manifest(
        corpus_roots=[tmp_path],
        policy=CorpusManifestPolicy(min_rows=2, min_features=2, max_missing_ratio=0.2),
    )

    assert manifest["gate"]["approved"] is True
    assert manifest["stats"]["records_kept"] == 1
    assert manifest["stats"]["quarantined"] == 0
    assert manifest["entries"][0]["source_tags"]["date_tags"] == ["2025-12-04"]
    assert manifest["stats"]["duplicates_removed"] == 1


def test_load_feature_matrix_ignores_metadata_columns_in_labeled_tables(tmp_path: Path):
    csv_path = tmp_path / "labeled.csv"
    _write_labeled_frame(csv_path)

    matrix = load_feature_matrix(csv_path)

    assert matrix.shape == (4, 2)
    assert matrix.dtype == np.float32
    assert np.array_equal(matrix[:, 0], np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32))


def test_cache_key_builder_is_stable_for_manifest_and_policy_fingerprint():
    key_a = build_feature_sequence_cache_key(
        manifest_fingerprint="manifest-1",
        policy_fingerprint="policy-1",
        stage="pretrain_matrix",
    )
    key_b = build_feature_sequence_cache_key(
        manifest_fingerprint="manifest-1",
        policy_fingerprint="policy-1",
        stage="pretrain_matrix",
    )
    key_c = build_feature_sequence_cache_key(
        manifest_fingerprint="manifest-2",
        policy_fingerprint="policy-1",
        stage="pretrain_matrix",
    )

    assert key_a == key_b
    assert key_a != key_c
