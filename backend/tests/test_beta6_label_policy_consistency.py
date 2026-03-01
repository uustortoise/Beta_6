from __future__ import annotations

import json
from pathlib import Path

from ml.beta6.label_policy_consistency import validate_label_policy_consistency


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_decision_trace(path: Path, labels: dict[str, int]) -> None:
    payload = {"metrics": {"per_label_support": labels}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_label_policy_consistency_passes_with_canonical_labels(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(
        config_dir / "beta6_critical_labels.yaml",
        """
version: v1
critical_labels_by_room:
  bedroom:
    - sleep
    - unoccupied
defaults:
  alias_to_canonical:
    sleeping: sleep
""".strip()
        + "\n",
    )
    _write_yaml(
        config_dir / "beta6_lane_b_event_labels.yaml",
        """
version: v1
lane_b_event_labels_by_room:
  bedroom:
    sleep_duration:
      - sleep
""".strip()
        + "\n",
    )

    report = validate_label_policy_consistency(
        config_dir=config_dir,
        models_dir=tmp_path / "models",
    )
    assert report.status == "pass"
    assert list(report.errors) == []


def test_label_policy_consistency_rejects_non_canonical_critical_label(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(
        config_dir / "beta6_critical_labels.yaml",
        """
version: v1
critical_labels_by_room:
  bedroom:
    - sleeping
    - unoccupied
defaults:
  alias_to_canonical:
    sleeping: sleep
""".strip()
        + "\n",
    )
    _write_yaml(
        config_dir / "beta6_lane_b_event_labels.yaml",
        """
version: v1
lane_b_event_labels_by_room:
  bedroom:
    sleep_duration:
      - sleep
""".strip()
        + "\n",
    )

    report = validate_label_policy_consistency(
        config_dir=config_dir,
        models_dir=tmp_path / "models",
    )
    assert report.status == "fail"
    assert any(issue.code == "non_canonical_critical_label" for issue in report.errors)


def test_label_policy_consistency_rejects_lane_b_label_not_in_critical(tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(
        config_dir / "beta6_critical_labels.yaml",
        """
version: v1
critical_labels_by_room:
  bedroom:
    - sleep
defaults:
  alias_to_canonical: {}
""".strip()
        + "\n",
    )
    _write_yaml(
        config_dir / "beta6_lane_b_event_labels.yaml",
        """
version: v1
lane_b_event_labels_by_room:
  bedroom:
    sleep_duration:
      - nap
""".strip()
        + "\n",
    )

    report = validate_label_policy_consistency(
        config_dir=config_dir,
        models_dir=tmp_path / "models",
    )
    assert report.status == "fail"
    assert any(issue.code == "lane_b_label_not_in_critical" for issue in report.errors)


def test_label_policy_consistency_emits_observed_data_warning(tmp_path: Path):
    config_dir = tmp_path / "config"
    models_dir = tmp_path / "models"
    config_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml(
        config_dir / "beta6_critical_labels.yaml",
        """
version: v1
critical_labels_by_room:
  bedroom:
    - sleep
    - unoccupied
defaults:
  alias_to_canonical:
    sleeping: sleep
""".strip()
        + "\n",
    )
    _write_yaml(
        config_dir / "beta6_lane_b_event_labels.yaml",
        """
version: v1
lane_b_event_labels_by_room:
  bedroom:
    sleep_duration:
      - sleep
""".strip()
        + "\n",
    )

    _write_decision_trace(
        models_dir / "HK001" / "Bedroom_v1_decision_trace.json",
        {"sleep": 12},
    )

    report = validate_label_policy_consistency(
        config_dir=config_dir,
        models_dir=models_dir,
    )
    assert report.status == "pass"
    assert any(issue.code == "critical_labels_missing_in_observed_data" for issue in report.warnings)

