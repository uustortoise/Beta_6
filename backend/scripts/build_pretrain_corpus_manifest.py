#!/usr/bin/env python3
"""Build deterministic Beta 6 pretraining corpus manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.beta6.data_manifest import CorpusManifestPolicy, build_pretrain_corpus_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Beta 6 pretraining corpus manifest")
    parser.add_argument(
        "--corpus-root",
        action="append",
        dest="corpus_roots",
        required=True,
        help="Corpus root directory or file (repeatable)",
    )
    parser.add_argument("--output", required=True, help="Output manifest JSON path")
    parser.add_argument(
        "--include-ext",
        action="append",
        default=None,
        help="Allowed extension (repeatable, default: .csv,.parquet,.npy)",
    )
    parser.add_argument("--max-missing-ratio", type=float, default=0.4)
    parser.add_argument("--min-rows", type=int, default=8)
    parser.add_argument("--min-features", type=int, default=2)
    args = parser.parse_args()

    include_ext = tuple(args.include_ext) if args.include_ext else (".csv", ".parquet", ".npy")
    policy = CorpusManifestPolicy(
        include_extensions=include_ext,
        max_missing_ratio=float(args.max_missing_ratio),
        min_rows=max(int(args.min_rows), 1),
        min_features=max(int(args.min_features), 1),
    )
    manifest = build_pretrain_corpus_manifest(
        corpus_roots=[Path(path).resolve() for path in args.corpus_roots],
        policy=policy,
    )

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    stats = manifest.get("stats", {})
    fingerprint = manifest.get("fingerprint", {}).get("value")
    print(f"Wrote manifest: {output_path}")
    print(f"Fingerprint: {fingerprint}")
    print(
        "Stats: "
        f"files_scanned={stats.get('files_scanned', 0)} "
        f"records_kept={stats.get('records_kept', 0)} "
        f"quarantined={stats.get('quarantined', 0)} "
        f"p0_violations={stats.get('p0_violations', 0)}"
    )
    gate = manifest.get("gate", {}) if isinstance(manifest, dict) else {}
    if not bool(gate.get("approved", False)):
        print(f"Manifest gate failed: {gate.get('blocking_reasons', [])}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
