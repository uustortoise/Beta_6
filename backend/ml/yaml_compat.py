"""YAML loader with fallback when PyYAML is unavailable in the active interpreter."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def _safe_load_via_system_python(path: Path) -> Any:
    script = (
        "import json, pathlib, sys\n"
        "import yaml\n"
        "p = pathlib.Path(sys.argv[1])\n"
        "obj = yaml.safe_load(p.read_text(encoding='utf-8'))\n"
        "print(json.dumps(obj))\n"
    )
    out = subprocess.check_output(
        ["python3", "-c", script, str(path)],
        text=True,
        timeout=10,
    )
    return json.loads(out) if str(out).strip() else None


def load_yaml_file(path: str | Path) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    try:
        import yaml  # type: ignore

        return yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except ModuleNotFoundError:
        return _safe_load_via_system_python(file_path)
