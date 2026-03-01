from pathlib import Path
from subprocess import TimeoutExpired

import yaml

from scripts.run_event_first_matrix import _build_backtest_command, _run_command, run_matrix


def test_build_backtest_command_includes_defaults_and_variant_args(tmp_path: Path):
    cmd = _build_backtest_command(
        data_dir=tmp_path / "data",
        elder_id="HK0011_jessica",
        seed=11,
        output_path=tmp_path / "seed11.json",
        defaults={
            "min_day": 4,
            "max_day": 10,
            "occupancy_threshold": 0.35,
            "calibration_method": "isotonic",
            "calib_fraction": 0.2,
            "min_calib_samples": 500,
            "min_calib_label_support": 30,
        },
        variant_args=["--enable-bedroom-livingroom-stage-a-hgb", "--hard-gate-min-train-days=3"],
        min_day_override=None,
        max_day_override=None,
    )
    joined = " ".join(cmd)
    assert "--min-day 4" in joined
    assert "--max-day 10" in joined
    assert "--seed 11" in joined
    assert "--enable-bedroom-livingroom-stage-a-hgb" in joined
    assert "--hard-gate-min-train-days=3" in joined


def test_run_matrix_dry_run_builds_manifest_without_execution(tmp_path: Path):
    profiles_yaml = tmp_path / "profiles.yaml"
    profiles_yaml.write_text(
        yaml.safe_dump(
            {
                "defaults": {
                    "min_day": 4,
                    "max_day": 10,
                    "seeds": [11],
                    "occupancy_threshold": 0.35,
                    "calibration_method": "isotonic",
                    "calib_fraction": 0.2,
                    "min_calib_samples": 500,
                    "min_calib_label_support": 30,
                    "comparison_window": "dec4_to_dec10",
                    "required_split_pass_ratio": 1.0,
                },
                "variants": {
                    "anchor": {"description": "anchor", "args": ["--hard-gate-min-train-days=3"]},
                },
                "profiles": {
                    "quick": {"seeds": [11], "variants": ["anchor"]},
                },
            }
        )
    )

    manifest = run_matrix(
        profiles_yaml=profiles_yaml,
        profile="quick",
        data_dir=tmp_path / "data",
        elder_id="HK0011_jessica",
        output_dir=tmp_path / "matrix",
        max_workers=1,
        dry_run=True,
        go_no_go_config=None,
        min_day_override=None,
        max_day_override=None,
        seed_timeout_seconds=900,
        seed_retries=1,
        force_single_process=True,
        recover_written_output=True,
    )

    assert manifest["status"] == "dry_run"
    assert manifest["dry_run"] is True
    assert "anchor" in manifest["variants"]
    seed_run = manifest["variants"]["anchor"]["seed_runs"]["11"]
    assert seed_run["result"]["status"] == "dry_run"
    assert seed_run["result"]["returncode"] == 0


def test_run_command_recovers_when_timeout_but_output_json_exists(tmp_path: Path, monkeypatch):
    out_json = tmp_path / "seed11.json"
    out_json.write_text('{"ok": true}')

    def _fake_run(*args, **kwargs):  # noqa: ANN001
        raise TimeoutExpired(cmd=args[0], timeout=1)

    monkeypatch.setattr("scripts.run_event_first_matrix.subprocess.run", _fake_run)

    result = _run_command(
        ["/usr/bin/python3", "x.py", "--output", str(out_json)],
        dry_run=False,
        timeout_seconds=1,
        retries=0,
        force_single_process=True,
        recover_written_output=True,
    )

    assert result["status"] == "ok"
    assert result["returncode"] == 0
    assert result["attempt_count"] == 1
    assert result["attempts"][0]["timed_out"] is True
    assert result["attempts"][0]["recovered_output"] is True
