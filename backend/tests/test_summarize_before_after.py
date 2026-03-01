import json

import pytest

from scripts.summarize_before_after import _to_markdown, summarize_before_after


def test_summarize_before_after_builds_room_deltas_and_gate_deltas():
    before_rolling = {
        "classification_summary": {
            "Bedroom": {
                "accuracy": {"mean": 0.80},
                "macro_f1": {"mean": 0.60},
                "occupied_f1": {"mean": 0.50},
                "occupied_recall": {"mean": 0.45},
            }
        },
        "timeline_summary": {"Bedroom": {"fragmentation_score": {"mean": 0.40}}},
    }
    after_rolling = {
        "classification_summary": {
            "Bedroom": {
                "accuracy": {"mean": 0.84},
                "macro_f1": {"mean": 0.66},
                "occupied_f1": {"mean": 0.58},
                "occupied_recall": {"mean": 0.52},
            }
        },
        "timeline_summary": {"Bedroom": {"fragmentation_score": {"mean": 0.48}}},
    }
    before_signoff = {"gate_summary": {"hard_gate_checks_passed": 24, "hard_gate_checks_total": 30}, "gate_decision": "FAIL"}
    after_signoff = {"gate_summary": {"hard_gate_checks_passed": 26, "hard_gate_checks_total": 30}, "gate_decision": "PASS"}

    summary = summarize_before_after(
        before_rolling=before_rolling,
        before_signoff=before_signoff,
        after_rolling=after_rolling,
        after_signoff=after_signoff,
    )
    row = summary["rooms"][0]
    assert row["room"] == "Bedroom"
    assert row["delta_accuracy"] == pytest.approx(0.04, abs=1e-9)
    assert row["delta_occupied_recall"] == pytest.approx(0.07, abs=1e-9)
    assert row["delta_fragmentation_score"] == pytest.approx(0.08, abs=1e-9)
    assert summary["global"]["delta_hard_gate_checks_passed"] == 2
    assert summary["global"]["after_gate_decision"] == "PASS"


def test_to_markdown_contains_room_and_gate_sections():
    summary = {
        "rooms": [
            {
                "room": "LivingRoom",
                "before_accuracy": 0.7,
                "after_accuracy": 0.8,
                "delta_accuracy": 0.1,
                "before_macro_f1": 0.5,
                "after_macro_f1": 0.6,
                "delta_macro_f1": 0.1,
                "before_occupied_f1": 0.4,
                "after_occupied_f1": 0.5,
                "delta_occupied_f1": 0.1,
                "before_occupied_recall": 0.3,
                "after_occupied_recall": 0.4,
                "delta_occupied_recall": 0.1,
                "before_fragmentation_score": 0.45,
                "after_fragmentation_score": 0.5,
                "delta_fragmentation_score": 0.05,
            }
        ],
        "global": {
            "before_gate_decision": "FAIL",
            "after_gate_decision": "PASS",
            "before_gate": {"hard_gate_checks_passed": 24, "hard_gate_checks_total": 30},
            "after_gate": {"hard_gate_checks_passed": 26, "hard_gate_checks_total": 30},
        },
    }
    md = _to_markdown(summary)
    assert "# Before/After Model Runtime Summary" in md
    assert "| LivingRoom |" in md
    assert "after_gate_decision: `PASS`" in md


def test_summarize_before_after_uses_split_level_fragmentation_fallback():
    before_rolling = {
        "classification_summary": {"LivingRoom": {"accuracy": {"mean": 0.7}}},
        "splits": [{"rooms": {"LivingRoom": {"fragmentation_score": 0.40}}}],
    }
    after_rolling = {
        "classification_summary": {"LivingRoom": {"accuracy": {"mean": 0.71}}},
        "splits": [{"rooms": {"LivingRoom": {"fragmentation_score": 0.52}}}],
    }
    before_signoff = {"gate_summary": {}, "gate_decision": "FAIL"}
    after_signoff = {"gate_summary": {}, "gate_decision": "FAIL"}

    summary = summarize_before_after(
        before_rolling=before_rolling,
        before_signoff=before_signoff,
        after_rolling=after_rolling,
        after_signoff=after_signoff,
    )
    row = summary["rooms"][0]
    assert row["before_fragmentation_score"] == pytest.approx(0.40, abs=1e-9)
    assert row["after_fragmentation_score"] == pytest.approx(0.52, abs=1e-9)
    assert row["delta_fragmentation_score"] == pytest.approx(0.12, abs=1e-9)


def test_summarize_before_after_uses_seed_report_path_fragmentation_fallback(tmp_path):
    before_seed = tmp_path / "before_seed.json"
    after_seed = tmp_path / "after_seed.json"
    before_seed.write_text(
        json.dumps(
            {
                "splits": [
                    {"rooms": {"LivingRoom": {"fragmentation_score": 0.41}}},
                    {"rooms": {"LivingRoom": {"fragmentation_score": 0.45}}},
                ]
            }
        )
    )
    after_seed.write_text(
        json.dumps(
            {
                "splits": [
                    {"rooms": {"LivingRoom": {"fragmentation_score": 0.50}}},
                    {"rooms": {"LivingRoom": {"fragmentation_score": 0.54}}},
                ]
            }
        )
    )

    before_rolling = {
        "classification_summary": {"LivingRoom": {"accuracy": {"mean": 0.7}}},
        "splits": [{"path": str(before_seed)}],
    }
    after_rolling = {
        "classification_summary": {"LivingRoom": {"accuracy": {"mean": 0.71}}},
        "splits": [{"path": str(after_seed)}],
    }
    summary = summarize_before_after(
        before_rolling=before_rolling,
        before_signoff={"gate_summary": {}, "gate_decision": "FAIL"},
        after_rolling=after_rolling,
        after_signoff={"gate_summary": {}, "gate_decision": "FAIL"},
    )
    row = summary["rooms"][0]
    assert row["before_fragmentation_score"] == pytest.approx(0.43, abs=1e-9)
    assert row["after_fragmentation_score"] == pytest.approx(0.52, abs=1e-9)


def test_summarize_before_after_uses_seed_split_stability_when_gate_summary_missing():
    before_rolling = {"classification_summary": {"Bedroom": {"accuracy": {"mean": 0.7}}}}
    after_rolling = {"classification_summary": {"Bedroom": {"accuracy": {"mean": 0.7}}}}
    before_signoff = {
        "gate_summary": {},
        "seed_split_stability": {"hard_gate_splits_passed": 16, "hard_gate_splits_total": 20},
        "gate_decision": "FAIL",
    }
    after_signoff = {
        "seed_split_stability": {"hard_gate_splits_passed": 18, "hard_gate_splits_total": 20},
        "gate_decision": "FAIL",
    }

    summary = summarize_before_after(
        before_rolling=before_rolling,
        before_signoff=before_signoff,
        after_rolling=after_rolling,
        after_signoff=after_signoff,
    )
    assert summary["global"]["before_gate"]["hard_gate_checks_passed"] == 16
    assert summary["global"]["after_gate"]["hard_gate_checks_passed"] == 18
