#!/usr/bin/env python3
"""Global Beta 6 training lineage + policy consistency check (fail-closed)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.beta6.label_policy_consistency import validate_label_policy_consistency  # noqa: E402
from utils.elder_id_utils import (  # noqa: E402
    elder_id_lineage_matches as _elder_id_lineage_matches,
    parse_elder_id_from_filename as _elder_id_from_filename,
)

TRAINING_EXTENSIONS = {".xlsx", ".xls", ".parquet"}


@dataclass
class Finding:
    severity: str
    code: str
    message: str
    context: dict

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "context": dict(self.context),
        }


def _is_training_file(path: Path) -> bool:
    name = path.name.lower()
    return "_train" in name and "_manual_" not in name and path.suffix.lower() in TRAINING_EXTENSIONS


def _split_elder_id_code_and_name(elder_id: str) -> tuple[str, str]:
    txt = str(elder_id or "").strip()
    if "_" not in txt:
        return "", ""
    code, name = txt.split("_", 1)
    return code.strip(), name.strip().lower()


def _scan_training_files(*roots: Path) -> List[Path]:
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for ext in TRAINING_EXTENSIONS:
            for path in root.rglob(f"*{ext}"):
                if not path.is_file() or not _is_training_file(path):
                    continue
                out.append(path)
    return sorted(out, key=lambda p: str(p))


def _build_lineage_findings(paths: List[Path]) -> List[Finding]:
    findings: List[Finding] = []

    by_name: Dict[str, set[str]] = {}
    for path in paths:
        elder_id = _elder_id_from_filename(path.name)
        _, name = _split_elder_id_code_and_name(elder_id)
        if not name:
            continue
        by_name.setdefault(name, set()).add(elder_id)

    for resident_name, elder_ids in sorted(by_name.items()):
        sorted_ids = sorted(elder_ids)
        incompatible: List[list[str]] = []
        for i, left in enumerate(sorted_ids):
            for right in sorted_ids[i + 1 :]:
                if not _elder_id_lineage_matches(left, right):
                    incompatible.append([left, right])
        if incompatible:
            findings.append(
                Finding(
                    severity="error",
                    code="elder_id_suffix_collision",
                    message="Detected resident-name collisions with incompatible elder IDs.",
                    context={
                        "resident_name": resident_name,
                        "elder_ids": sorted_ids,
                        "incompatible_pairs": incompatible,
                    },
                )
            )

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Beta 6 lineage and label-policy consistency.")
    parser.add_argument("--config-dir", default=str(BACKEND_DIR / "config"))
    parser.add_argument("--models-dir", default=str(BACKEND_DIR / "models"))
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--archive-dir", default=str(PROJECT_ROOT / "data" / "archive"))
    parser.add_argument("--fail-on-warnings", action="store_true")
    args = parser.parse_args()

    findings: List[Finding] = []

    # 1) Reuse existing fail-closed label policy consistency guard.
    policy = validate_label_policy_consistency(
        config_dir=Path(args.config_dir),
        models_dir=Path(args.models_dir),
        fail_on_warnings=bool(args.fail_on_warnings),
    )
    for issue in policy.errors:
        findings.append(
            Finding(
                severity="error",
                code=str(issue.code),
                message=str(issue.message),
                context=dict(issue.context),
            )
        )
    for issue in policy.warnings:
        findings.append(
            Finding(
                severity="warning",
                code=str(issue.code),
                message=str(issue.message),
                context=dict(issue.context),
            )
        )

    # 2) Scan lineage consistency globally.
    training_files = _scan_training_files(Path(args.raw_dir), Path(args.archive_dir))
    findings.extend(_build_lineage_findings(training_files))

    error_count = sum(1 for f in findings if f.severity == "error")
    warning_count = sum(1 for f in findings if f.severity == "warning")
    status = "pass" if error_count == 0 and (warning_count == 0 or not args.fail_on_warnings) else "fail"

    payload = {
        "status": status,
        "training_files_scanned": len(training_files),
        "error_count": error_count,
        "warning_count": warning_count,
        "findings": [f.to_dict() for f in findings],
    }
    print(json.dumps(payload, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
