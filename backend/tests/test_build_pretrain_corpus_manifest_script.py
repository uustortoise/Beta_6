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
