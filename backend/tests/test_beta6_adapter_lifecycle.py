from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.beta6.adapters.adapter_store import AdapterStore, AdapterStorePolicy
from ml.beta6.adapters.lora_adapter import LoRAAdapterConfig, train_lora_adapter_from_frame
from ml.beta6.orchestrator import Beta6Orchestrator


def _training_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(60):
        rows.append(
            {
                "motion_auc": float(rng.normal(1.2, 0.08)),
                "light_auc": float(rng.normal(0.2, 0.05)),
                "activity": "sleep",
            }
        )
    for _ in range(60):
        rows.append(
            {
                "motion_auc": float(rng.normal(0.3, 0.05)),
                "light_auc": float(rng.normal(1.1, 0.08)),
                "activity": "out",
            }
        )
    return pd.DataFrame(rows)


def test_adapter_store_lifecycle_create_warmup_promote_and_rollback(tmp_path: Path):
    frame = _training_frame()
    config = LoRAAdapterConfig(rank=2, alpha=4.0, min_rows=20)
    store = AdapterStore(
        root=tmp_path / "adapters",
        policy=AdapterStorePolicy(
            max_versions_per_resident=3,
            min_warmup_accuracy=0.5,
            retirement_inactive_days=30,
            enable_auto_retire=False,
        ),
    )

    artifact_v1 = train_lora_adapter_from_frame(
        frame,
        resident_id="HK001",
        room="bedroom",
        backbone_id="bb_v1",
        config=config,
    )
    store.create_adapter(artifact=artifact_v1, run_id="run_1")
    warmed_v1 = store.warmup_adapter(
        resident_id="HK001",
        room="bedroom",
        adapter_id=artifact_v1.adapter_id,
        warmup_accuracy=artifact_v1.warmup_accuracy,
        run_id="run_1",
    )
    assert warmed_v1["warmup_pass"] is True
    active_v1 = store.promote_adapter(
        resident_id="HK001",
        room="bedroom",
        adapter_id=artifact_v1.adapter_id,
        run_id="run_1",
    )
    assert active_v1["adapter_id"] == artifact_v1.adapter_id

    artifact_v2 = train_lora_adapter_from_frame(
        frame.sample(frac=1.0, random_state=7),
        resident_id="HK001",
        room="bedroom",
        backbone_id="bb_v1",
        config=config,
    )
    store.create_adapter(artifact=artifact_v2, run_id="run_2")
    store.warmup_adapter(
        resident_id="HK001",
        room="bedroom",
        adapter_id=artifact_v2.adapter_id,
        warmup_accuracy=artifact_v2.warmup_accuracy,
        run_id="run_2",
    )
    active_v2 = store.promote_adapter(
        resident_id="HK001",
        room="bedroom",
        adapter_id=artifact_v2.adapter_id,
        run_id="run_2",
    )
    assert active_v2["adapter_id"] == artifact_v2.adapter_id

    rollback = store.rollback_adapter(
        resident_id="HK001",
        room="bedroom",
        run_id="run_3",
        target_adapter_id=artifact_v1.adapter_id,
    )
    assert rollback["adapter_id"] == artifact_v1.adapter_id


def test_orchestrator_phase5_adapter_lifecycle_passes(tmp_path: Path):
    frame = _training_frame()
    orchestrator = Beta6Orchestrator(require_intake_artifact=False)
    result = orchestrator.run_phase5_adapter_lifecycle(
        training_frame=frame,
        resident_id="HK002",
        room="livingroom",
        backbone_id="bb_v1",
        run_id="run_phase5_adapter",
        adapter_store_root=tmp_path / "adapter_store",
        label_col="activity",
        auto_promote=True,
    )
    assert result.status == "pass"
    assert result.warmup_pass is True
    assert result.promoted is True
    assert result.active_adapter_id == result.adapter_id


def test_rollback_rejects_unvalidated_target_adapter(tmp_path: Path):
    frame = _training_frame()
    config = LoRAAdapterConfig(rank=2, alpha=4.0, min_rows=20)
    store = AdapterStore(
        root=tmp_path / "adapters",
        policy=AdapterStorePolicy(
            max_versions_per_resident=5,
            min_warmup_accuracy=0.5,
            retirement_inactive_days=30,
            enable_auto_retire=False,
        ),
    )

    validated = train_lora_adapter_from_frame(
        frame,
        resident_id="HK001",
        room="bedroom",
        backbone_id="bb_v1",
        config=config,
        adapter_id="adapter_validated",
    )
    store.create_adapter(artifact=validated, run_id="run_1")
    store.warmup_adapter(
        resident_id="HK001",
        room="bedroom",
        adapter_id=validated.adapter_id,
        warmup_accuracy=validated.warmup_accuracy,
        run_id="run_1",
    )
    store.promote_adapter(
        resident_id="HK001",
        room="bedroom",
        adapter_id=validated.adapter_id,
        run_id="run_1",
    )

    created_only = train_lora_adapter_from_frame(
        frame.sample(frac=1.0, random_state=11),
        resident_id="HK001",
        room="bedroom",
        backbone_id="bb_v1",
        config=config,
        adapter_id="adapter_created_only",
    )
    store.create_adapter(artifact=created_only, run_id="run_2")

    with pytest.raises(ValueError, match="rollback-eligible"):
        store.rollback_adapter(
            resident_id="HK001",
            room="bedroom",
            run_id="run_3",
            target_adapter_id=created_only.adapter_id,
        )


def test_enforce_retention_prunes_non_active_overflow(tmp_path: Path):
    frame = _training_frame()
    config = LoRAAdapterConfig(rank=2, alpha=4.0, min_rows=20)
    store = AdapterStore(
        root=tmp_path / "adapters",
        policy=AdapterStorePolicy(
            max_versions_per_resident=1,
            min_warmup_accuracy=0.5,
            retirement_inactive_days=30,
            enable_auto_retire=False,
        ),
    )

    adapter_a = train_lora_adapter_from_frame(
        frame,
        resident_id="HK001",
        room="bedroom",
        backbone_id="bb_v1",
        config=config,
        adapter_id="adapter_a",
    )
    adapter_b = train_lora_adapter_from_frame(
        frame.sample(frac=1.0, random_state=17),
        resident_id="HK001",
        room="bedroom",
        backbone_id="bb_v1",
        config=config,
        adapter_id="adapter_b",
    )
    store.create_adapter(artifact=adapter_a, run_id="run_1")
    store.create_adapter(artifact=adapter_b, run_id="run_2")

    assert len(store.list_adapters("HK001", "bedroom")) == 2
    retention = store.enforce_retention(resident_id="HK001", room="bedroom")
    assert len(retention["removed"]) == 1
    assert len(store.list_adapters("HK001", "bedroom")) == 1
    removed_id = retention["removed"][0]
    removed_dir = tmp_path / "adapters" / "HK001" / "bedroom" / "adapters" / removed_id
    assert not removed_dir.exists()
