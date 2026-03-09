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

from ml.beta6.data_manifest import (  # noqa: E402
    CorpusManifestPolicy,
    build_pretrain_corpus_manifest,
    evaluate_beta62_corpus_contract,
)


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
    parser.add_argument("--required-residents", type=int, default=20)
    parser.add_argument("--required-days", type=int, default=14)
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
    manifest["beta62_corpus_contract"] = evaluate_beta62_corpus_contract(
        manifest,
        required_residents=max(int(args.required_residents), 1),
        required_days=max(int(args.required_days), 1),
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
        f"p0_violations={stats.get('p0_violations', 0)}"
    )
    contract = manifest.get("beta62_corpus_contract", {})
    print(
        "Beta62 contract: "
        f"pass={bool(contract.get('pass'))} "
        f"reasons={','.join(contract.get('reason_codes', [])) or 'none'}"
    )
    p0_violations = int(stats.get("p0_violations", 0) or 0)
    records_kept = int(stats.get("records_kept", 0) or 0)
    contract_pass = bool(contract.get("pass", False))
    if p0_violations > 0 or records_kept <= 0:
        if records_kept <= 0 and p0_violations == 0:
            print("Manifest gate failed: no usable records kept")
        return 2
    if not contract_pass:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
