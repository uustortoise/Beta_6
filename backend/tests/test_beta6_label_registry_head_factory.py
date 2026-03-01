import pandas as pd

from ml.beta6.contracts import label_registry as label_registry_module
from ml.beta6.contracts.label_registry import (
    DEFAULT_MANDATORY_CLASSES,
    build_label_registry,
    build_label_registry_from_training_frame,
)
from ml.beta6.head_factory import build_dynamic_head_specs, summarize_head_specs


def test_label_registry_merges_training_backend_and_manual(monkeypatch):
    monkeypatch.setattr(
        label_registry_module,
        "_collect_backend_labels_by_room",
        lambda: {"Bedroom": {"sleeping", "asleep"}},
    )
    registry = build_label_registry(
        training_labels_by_room={"bedroom": ["resting", "active_use"]},
        manual_additions_by_room={"Bedroom": ["toilet_visit"]},
        include_backend_registry=True,
    )

    labels = set(registry.labels_for_room("BEDROOM"))
    assert {"sleeping", "asleep", "resting", "toilet_visit", "active_use", "unoccupied"} <= labels
    assert registry.sources["bedroom"]["backend_count"] == 2
    assert registry.sources["bedroom"]["training_count"] == 2
    assert registry.sources["bedroom"]["manual_count"] == 1


def test_label_registry_from_training_frame_is_room_normalized(monkeypatch):
    monkeypatch.setattr(
        label_registry_module,
        "_collect_backend_labels_by_room",
        lambda: {},
    )
    frame = pd.DataFrame(
        {
            "room": ["Living Room", "livingroom", "Bedroom"],
            "activity": ["relaxing", "tv", "sleep"],
        }
    )
    registry = build_label_registry_from_training_frame(
        frame,
        include_backend_registry=False,
        mandatory_classes=DEFAULT_MANDATORY_CLASSES,
    )
    assert set(registry.room_to_labels.keys()) == {"livingroom", "bedroom"}


def test_dynamic_head_specs_are_deterministic_and_include_mandatory(monkeypatch):
    monkeypatch.setattr(
        label_registry_module,
        "_collect_backend_labels_by_room",
        lambda: {},
    )
    registry = build_label_registry(
        training_labels_by_room={
            "bedroom": ["sleep", "active_use"],
            "livingroom": ["relaxing", "unoccupied"],
        },
        include_backend_registry=False,
    )

    specs = build_dynamic_head_specs(label_registry=registry)
    summary = summarize_head_specs(specs)

    assert set(summary["rooms"]) == {"bedroom", "livingroom"}
    assert summary["room_output_dims"]["bedroom"] == len(registry.labels_for_room("bedroom"))
    assert summary["room_output_dims"]["livingroom"] == len(registry.labels_for_room("livingroom"))
    assert specs["bedroom"].mandatory_present["active_use"] is True
    assert specs["bedroom"].mandatory_present["unoccupied"] is True
    assert specs["livingroom"].mandatory_present["active_use"] is True
    assert specs["livingroom"].mandatory_present["unoccupied"] is True
