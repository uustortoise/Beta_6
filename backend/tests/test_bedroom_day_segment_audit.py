from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "bedroom_day_segment_audit.py"
    spec = importlib.util.spec_from_file_location("bedroom_day_segment_audit", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample_day_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-12-05 00:00:00",
                    "2025-12-05 00:10:00",
                    "2025-12-05 00:20:00",
                    "2025-12-05 00:30:00",
                    "2025-12-05 01:00:00",
                    "2025-12-05 01:40:00",
                ]
            ),
            "activity": [
                "sleep",
                "sleep",
                "bedroom_normal_use",
                "bedroom_normal_use",
                "unoccupied",
                "sleep",
            ],
            "motion": [1.0, 1.0, 0.0, 0.0, None, None],
            "temperature": [22.0, 22.1, 22.2, 22.3, 22.4, 22.5],
        }
    )


def test_audit_day_frame_splits_into_bounded_time_blocks():
    module = _load_module()

    blocks = module.audit_day_frame(
        _sample_day_frame(),
        block_minutes=60,
        expected_interval_seconds=600,
    )["blocks"]

    assert [block["block_label"] for block in blocks] == ["00:00-01:00", "01:00-02:00"]
    assert blocks[0]["row_count"] == 4
    assert blocks[1]["row_count"] == 2


def test_audit_day_frame_reports_label_counts_transitions_and_run_lengths():
    module = _load_module()

    first_block = module.audit_day_frame(
        _sample_day_frame(),
        block_minutes=60,
        expected_interval_seconds=600,
    )["blocks"][0]

    assert first_block["label_counts"] == {"bedroom_normal_use": 2, "sleep": 2}
    assert first_block["transition_counts"] == {"sleep -> bedroom_normal_use": 1}
    assert first_block["run_length_seconds_by_label"]["sleep"]["max"] == 1200.0
    assert first_block["run_length_seconds_by_label"]["bedroom_normal_use"]["mean"] == 1200.0


def test_audit_day_frame_flags_sparse_and_missing_blocks():
    module = _load_module()

    second_block = module.audit_day_frame(
        _sample_day_frame(),
        block_minutes=60,
        expected_interval_seconds=600,
    )["blocks"][1]

    assert "sparse_rows" in second_block["flags"]
    assert "large_gap" in second_block["flags"]
    assert "sensor_missingness" in second_block["flags"]
    assert second_block["largest_gap_seconds"] == 2400.0


def test_audit_day_frame_compares_target_block_against_reference_blocks():
    module = _load_module()

    reference_frame = _sample_day_frame().assign(
        activity=[
            "sleep",
            "sleep",
            "sleep",
            "bedroom_normal_use",
            "sleep",
            "sleep",
        ]
    )
    audit = module.audit_day_frame(
        _sample_day_frame(),
        reference_frames=[reference_frame],
        block_minutes=60,
        expected_interval_seconds=600,
    )

    first_block = audit["blocks"][0]
    delta = first_block["reference_delta"]["label_share_delta"]
    assert delta["sleep"] == -0.25
    assert delta["bedroom_normal_use"] == 0.25
