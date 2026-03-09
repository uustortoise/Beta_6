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


def test_manifest_script_writes_beta62_contract_summary(tmp_path: Path, monkeypatch):
    data = tmp_path / "HK001_shadow.csv"
    data.write_text("f0,f1\n1,2\n3,4\n5,6\n7,8\n9,10\n11,12\n13,14\n15,16\n", encoding="utf-8")
    (tmp_path / "HK001_shadow.csv.meta.json").write_text(
        '{"resident_id":"HK001","days_covered":14,"views":["shadow_cohort","unlabeled_pretrain"],'
        '"resident_home_context":{"status":"ready","missing_fields":[]}}',
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
            "--required-residents",
            "1",
            "--required-days",
            "14",
        ],
    )
    rc = script.main()
    assert out.exists()
    assert rc == 2  # labeled view still missing, so Beta 6.2 contract should fail closed
