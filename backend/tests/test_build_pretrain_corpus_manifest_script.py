import json
from pathlib import Path

from scripts import build_pretrain_corpus_manifest as script


def test_manifest_script_fails_on_empty_corpus(tmp_path: Path, monkeypatch):
    out = tmp_path / "manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_pretrain_corpus_manifest.py",
            "--corpus-root",
            str(tmp_path / "empty"),
            "--output",
            str(out),
        ],
    )
    rc = script.main()
    assert rc == 2
    assert out.exists()


def test_manifest_script_auto_approves_clean_files_and_quarantines_red_flags(tmp_path: Path, monkeypatch):
    clean = tmp_path / "clean.csv"
    clean.write_text(
        "\n".join(
            [
                "elder_id,timestamp,room,activity,f1,f2",
                "HK0011_jessica,2025-12-04T07:00:00,Bedroom,sleep,1,0",
                "HK0011_jessica,2025-12-04T07:05:00,Bedroom,bedroom_normal_use,0,0",
            ]
        ),
        encoding="utf-8",
    )
    bad = tmp_path / "bad.csv"
    bad.write_text(
        "\n".join(
            [
                "elder_id,room,activity,f1,f2",
                "HK0011_jessica,Kitchen,meal_preparation,1,0",
                "HK0011_jessica,Kitchen,meal_preparation,1,1",
            ]
        ),
        encoding="utf-8",
    )
    out = tmp_path / "manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_pretrain_corpus_manifest.py",
            "--corpus-root",
            str(tmp_path),
            "--output",
            str(out),
            "--min-rows",
            "2",
            "--min-features",
            "2",
            "--max-missing-ratio",
            "0.2",
        ],
    )

    rc = script.main()
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert rc == 0
    assert payload["gate"]["approved"] is True
    assert payload["stats"]["records_kept"] == 1
    assert payload["stats"]["quarantined"] == 1
