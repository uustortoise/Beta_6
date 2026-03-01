import json
from pathlib import Path

import numpy as np
import pytest

from ml.beta6.fine_tune_safe_classes import run_safe_class_finetune


def _write_dataset(path: Path, *, residents: int = 6, per_class_per_resident: int = 6) -> Path:
    rng = np.random.default_rng(7)
    samples = []
    classes = ["sleep", "nap"]
    for ridx in range(residents):
        elder = f"R{ridx:02d}"
        for label_idx, label in enumerate(classes):
            for i in range(per_class_per_resident):
                base = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
                if label_idx == 1:
                    base[0] += 2.0
                noise = rng.normal(scale=0.03, size=4).astype(np.float32)
                vec = base + noise
                samples.append(
                    {
                        "elder_id": elder,
                        "room": "bedroom",
                        "timestamp": f"2026-02-{(i % 20) + 1:02d}T00:{i:02d}:00",
                        "activity": label,
                        "sensor_features": {
                            "motion_mean": float(vec[0]),
                            "light_mean": float(vec[1]),
                            "motion_auc": float(vec[2]),
                            "light_auc": float(vec[3]),
                        },
                    }
                )
    payload = {"metadata": {"sample_count": len(samples)}, "samples": samples}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_config(path: Path, *, seed: int = 42, min_accuracy: float = 0.85) -> Path:
    path.write_text(
        (
            "version: v1\n"
            "fine_tune:\n"
            "  safe_classes: [sleep, nap]\n"
            "  min_samples_per_class: 8\n"
            "  min_unique_residents: 2\n"
            "  holdout_fraction: 0.3\n"
            f"  random_seed: {seed}\n"
            f"  min_accuracy: {min_accuracy}\n"
            "  max_split_attempts: 40\n"
        ),
        encoding="utf-8",
    )
    return path


def test_safe_finetune_passes_with_resident_disjoint_holdout(tmp_path: Path):
    dataset = _write_dataset(tmp_path / "golden_dataset_1.json")
    config = _write_config(tmp_path / "cfg.yaml", seed=11, min_accuracy=0.8)
    report = run_safe_class_finetune(
        dataset_path=dataset,
        config_path=config,
        output_dir=tmp_path / "out",
    )

    assert report["status"] == "pass"
    assert report["metrics"]["heldout_accuracy"] >= 0.8
    assert report["leakage"]["warnings"] == []
    assert Path(report["artifacts"]["model_joblib"]).exists()
    assert Path(report["artifacts"]["report_json"]).exists()


def test_safe_finetune_reproducible_with_fixed_seed(tmp_path: Path):
    dataset = _write_dataset(tmp_path / "golden_dataset_1.json")
    config = _write_config(tmp_path / "cfg.yaml", seed=17, min_accuracy=0.8)

    run_a = run_safe_class_finetune(
        dataset_path=dataset,
        config_path=config,
        output_dir=tmp_path / "a",
    )
    run_b = run_safe_class_finetune(
        dataset_path=dataset,
        config_path=config,
        output_dir=tmp_path / "b",
    )
    assert run_a["metrics"]["heldout_accuracy"] == run_b["metrics"]["heldout_accuracy"]


def test_safe_finetune_fails_on_low_support(tmp_path: Path):
    dataset = _write_dataset(tmp_path / "golden_dataset_1.json", residents=2, per_class_per_resident=1)
    config = _write_config(tmp_path / "cfg.yaml", seed=5, min_accuracy=0.8)
    with pytest.raises(ValueError, match="insufficient support"):
        run_safe_class_finetune(
            dataset_path=dataset,
            config_path=config,
            output_dir=tmp_path / "out",
        )
