from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_beta6_training_lineage.py"
    spec = importlib.util.spec_from_file_location("check_beta6_training_lineage", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_lineage_match_numeric_suffix_drift_accepted():
    module = _load_module()
    assert module._elder_id_lineage_matches("HK001_jessica", "HK0011_jessica") is True
    assert module._elder_id_lineage_matches("HK0011_jessica", "HK001_jessica") is True


def test_lineage_match_different_prefix_rejected():
    module = _load_module()
    assert module._elder_id_lineage_matches("HK001_jessica", "AB001_jessica") is False


def test_build_lineage_findings_flags_incompatible_name_collisions(tmp_path):
    module = _load_module()

    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "HK001_jessica_train_4dec2025.xlsx").write_text("x")
    (raw / "AB001_jessica_train_5dec2025.xlsx").write_text("x")

    findings = module._build_lineage_findings([p for p in raw.iterdir()])
    assert findings
    assert any(f.code == "elder_id_suffix_collision" for f in findings)


def test_build_lineage_findings_no_error_for_same_lineage_alias_ids(tmp_path):
    module = _load_module()

    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "HK001_jessica_train_4dec2025.xlsx").write_text("x")
    (raw / "HK0011_jessica_train_5dec2025.xlsx").write_text("x")

    findings = module._build_lineage_findings([p for p in raw.iterdir()])
    assert findings == []
