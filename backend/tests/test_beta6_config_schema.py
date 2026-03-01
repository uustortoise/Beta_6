from pathlib import Path
import shutil

from ml.beta6.beta6_schema import EXPECTED_BETA6_CONFIG_FILES, validate_all_beta6_configs


def test_validate_all_beta6_configs_passes_for_repo_defaults():
    config_dir = Path(__file__).resolve().parent.parent / "config"
    report = validate_all_beta6_configs(config_dir)
    assert report.expected_files >= 16
    assert report.expected_files == len(EXPECTED_BETA6_CONFIG_FILES)
    assert report.checked_files >= 16
    assert report.status == "pass"
    assert report.errors == []


def test_validate_all_beta6_configs_rejects_non_boolean_canary_toggle(tmp_path: Path):
    source_config_dir = Path(__file__).resolve().parent.parent / "config"
    copied_config_dir = tmp_path / "config"
    shutil.copytree(source_config_dir, copied_config_dir)

    canary_config = copied_config_dir / "beta6_canary_gate.yaml"
    text = canary_config.read_text(encoding="utf-8")
    canary_config.write_text(
        text.replace("require_real_data_evidence: true", "require_real_data_evidence: \"false\""),
        encoding="utf-8",
    )

    report = validate_all_beta6_configs(copied_config_dir)
    assert report.status == "fail"
    assert any(
        "canary.require_real_data_evidence" in err and "expected boolean" in err
        for err in report.errors
    )
