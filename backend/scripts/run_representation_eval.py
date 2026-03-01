#!/usr/bin/env python3
"""Run Beta 6 representation quality evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ml.beta6.representation_eval import evaluate_representation_quality  # noqa: E402
from ml.beta6.self_supervised_pretrain import (  # noqa: E402
    encode_with_checkpoint,
    load_pretrain_checkpoint,
)


def _load_eval_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(csv_path)
    required = {"resident_id", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"dataset missing required columns: {sorted(missing)}")
    feature_frame = frame.drop(columns=["resident_id", "label"]).select_dtypes(include=[np.number])
    if feature_frame.empty:
        raise ValueError("dataset must include numeric feature columns")
    features = feature_frame.to_numpy(dtype=np.float32)
    labels = frame["label"].astype(str).to_numpy()
    resident_ids = frame["resident_id"].astype(str).to_numpy()
    return features, labels, resident_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Beta 6 representation evaluation")
    parser.add_argument("--dataset-csv", required=True, help="CSV with resident_id,label,numeric features")
    parser.add_argument("--checkpoint", required=True, help="Pretrain checkpoint .npz path")
    parser.add_argument("--output", required=True, help="Output JSON report path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--knn-k", type=int, default=5)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_csv).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()

    features, labels, resident_ids = _load_eval_dataset(dataset_path)
    checkpoint = load_pretrain_checkpoint(checkpoint_path)
    embeddings = encode_with_checkpoint(features, checkpoint)

    result = evaluate_representation_quality(
        embeddings=embeddings,
        labels=labels,
        resident_ids=resident_ids,
        seed=int(args.seed),
        test_fraction=float(args.test_fraction),
        knn_k=max(int(args.knn_k), 1),
    )
    report = {
        "status": "pass" if result.improvement_margin > 0 else "fail",
        "dataset_csv": str(dataset_path),
        "checkpoint": str(checkpoint_path),
        "metrics": result.to_dict(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote representation report: {output_path}")
    print(
        f"linear_probe={result.linear_probe_accuracy:.4f} "
        f"random_probe={result.random_probe_accuracy:.4f} "
        f"margin={result.improvement_margin:.4f} "
        f"knn_purity={result.knn_purity:.4f}"
    )
    return 0 if result.improvement_margin > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
