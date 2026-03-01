import json
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "deterministic_retrain.py"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def test_run_fails_when_no_valid_data_files():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        manifest = _write_manifest(
            tmp,
            {
                "elder_id": "HK001",
                "data_files": [],
                "random_seed": 42,
                "code_version": {},
                "policy": {},
            },
        )

        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "run", "--manifest", str(manifest), "--output-dir", str(tmp)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )

        assert proc.returncode == 1
        output = f"{proc.stdout}\n{proc.stderr}"
        assert "No valid data files available after manifest verification" in output


def test_verify_reports_missing_manifest_data_file():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        missing_path = tmp / "missing.parquet"
        manifest = _write_manifest(
            tmp,
            {
                "elder_id": "HK001",
                "data_files": [{"path": str(missing_path), "hash": "deadbeef"}],
                "random_seed": 42,
                "code_version": {},
                "policy": {},
            },
        )

        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "verify", "--manifest", str(manifest)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )

        assert proc.returncode == 1
        output = f"{proc.stdout}\n{proc.stderr}"
        assert "Missing:" in output
