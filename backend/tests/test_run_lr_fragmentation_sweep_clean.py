import json
from pathlib import Path

from scripts.run_lr_fragmentation_sweep_clean import _valid_json, _write_ranking


def test_valid_json_helper(tmp_path: Path):
    ok = tmp_path / "ok.json"
    ok.write_text('{"a": 1}')
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    missing = tmp_path / "missing.json"
    assert _valid_json(ok) is True
    assert _valid_json(bad) is False
    assert _valid_json(missing) is False


def test_write_ranking_sorts_by_status_then_eligible(tmp_path: Path):
    summaries = {
        "variant_b": {
            "go_no_go": {
                "status": "fail",
                "blocking_reasons": ["x"],
                "counters": {
                    "passed_eligible": 10,
                    "total_eligible": 20,
                    "room_eligible_passed": {"livingroom": 1},
                },
            }
        },
        "variant_a": {
            "go_no_go": {
                "status": "pass",
                "blocking_reasons": [],
                "counters": {
                    "passed_eligible": 8,
                    "total_eligible": 20,
                    "room_eligible_passed": {"livingroom": 2},
                },
            }
        },
        "variant_c": {
            "go_no_go": {
                "status": "fail",
                "blocking_reasons": ["x", "y"],
                "counters": {
                    "passed_eligible": 12,
                    "total_eligible": 20,
                    "room_eligible_passed": {"livingroom": 0},
                },
            }
        },
    }

    csv_path = tmp_path / "ranking.csv"
    md_path = tmp_path / "ranking.md"
    _write_ranking(summaries, csv_path=csv_path, md_path=md_path)

    lines = csv_path.read_text().strip().splitlines()
    assert lines[0].startswith("variant,status,eligible_passed")
    # pass variant first
    assert lines[1].startswith("variant_a,pass,8,20,2")
    # then fail variants by eligible desc
    assert lines[2].startswith("variant_c,fail,12,20,0")
    assert lines[3].startswith("variant_b,fail,10,20,1")

    md = md_path.read_text()
    assert "| variant_a | pass | 8/20 | 2 |  |" in md
    assert "| variant_c | fail | 12/20 | 0 | x;y |" in md
