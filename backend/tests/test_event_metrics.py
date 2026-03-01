import pandas as pd

from ml.event_metrics import (
    IntervalEvent,
    compute_care_kpi_summary,
    compute_room_care_kpis,
    event_precision_recall_f1,
    interval_iou,
)


def test_interval_iou_overlap():
    a = IntervalEvent("sleep", pd.Timestamp("2025-12-04 00:00:00"), pd.Timestamp("2025-12-04 01:00:00"))
    b = IntervalEvent("sleep", pd.Timestamp("2025-12-04 00:30:00"), pd.Timestamp("2025-12-04 01:30:00"))

    iou = interval_iou(a, b)
    assert round(iou, 3) == 0.333


def test_event_precision_recall_f1():
    gt = [
        IntervalEvent("shower", pd.Timestamp("2025-12-04 07:00:00"), pd.Timestamp("2025-12-04 07:10:00")),
        IntervalEvent("shower", pd.Timestamp("2025-12-05 07:05:00"), pd.Timestamp("2025-12-05 07:15:00")),
    ]
    pred = [
        IntervalEvent("shower", pd.Timestamp("2025-12-04 07:02:00"), pd.Timestamp("2025-12-04 07:11:00")),
        IntervalEvent("shower", pd.Timestamp("2025-12-06 08:00:00"), pd.Timestamp("2025-12-06 08:10:00")),
    ]

    out = event_precision_recall_f1(gt, pred, label="shower", min_iou=0.1)
    assert out["precision"] == 0.5
    assert out["recall"] == 0.5
    assert out["f1"] == 0.5


def test_compute_care_kpi_summary():
    gt_days = [
        {"sleep_minutes": 420, "livingroom_active_minutes": 120, "kitchen_use_minutes": 60, "shower_day": 1},
        {"sleep_minutes": 410, "livingroom_active_minutes": 100, "kitchen_use_minutes": 55, "shower_day": 0},
    ]
    pred_days = [
        {"sleep_minutes": 390, "livingroom_active_minutes": 110, "kitchen_use_minutes": 70, "shower_day": 1},
        {"sleep_minutes": 430, "livingroom_active_minutes": 80, "kitchen_use_minutes": 45, "shower_day": 1},
    ]

    out = compute_care_kpi_summary(gt_days, pred_days)
    assert out["sleep_duration_mae_minutes"] == 25.0
    assert out["livingroom_active_mae_minutes"] == 15.0
    assert out["kitchen_use_mae_minutes"] == 10.0
    assert out["shower_day_precision"] == 0.5
    assert out["shower_day_recall"] == 1.0


def test_compute_room_care_kpis_scoped_fields():
    gt_days = [
        {"sleep_minutes": 420, "kitchen_use_minutes": 60, "livingroom_active_minutes": 120, "shower_day": 1, "bathroom_use_minutes": 45, "out_minutes": 10},
        {"sleep_minutes": 410, "kitchen_use_minutes": 55, "livingroom_active_minutes": 100, "shower_day": 0, "bathroom_use_minutes": 35, "out_minutes": 20},
    ]
    pred_days = [
        {"sleep_minutes": 390, "kitchen_use_minutes": 70, "livingroom_active_minutes": 110, "shower_day": 1, "bathroom_use_minutes": 55, "out_minutes": 5},
        {"sleep_minutes": 430, "kitchen_use_minutes": 45, "livingroom_active_minutes": 80, "shower_day": 1, "bathroom_use_minutes": 20, "out_minutes": 10},
    ]

    bedroom = compute_room_care_kpis("Bedroom", gt_days, pred_days)
    living = compute_room_care_kpis("LivingRoom", gt_days, pred_days)
    kitchen = compute_room_care_kpis("Kitchen", gt_days, pred_days)
    bathroom = compute_room_care_kpis("Bathroom", gt_days, pred_days)
    entrance = compute_room_care_kpis("Entrance", gt_days, pred_days)

    assert set(bedroom.keys()) == {"sleep_duration_mae_minutes"}
    assert set(living.keys()) == {"livingroom_active_mae_minutes"}
    assert set(kitchen.keys()) == {"kitchen_use_mae_minutes"}
    assert "shower_day_f1" in bathroom
    assert "bathroom_use_mae_minutes" in bathroom
    assert set(entrance.keys()) == {"out_minutes_mae"}
