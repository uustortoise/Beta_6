import pandas as pd

from ml.beta6.active_learning import ActiveLearningPolicy, build_active_learning_queue


def _candidates_frame() -> pd.DataFrame:
    rows = []
    cid = 0
    for room in ["bedroom", "livingroom", "kitchen"]:
        for activity in ["sleep", "nap", "out"]:
            for confidence in [0.15, 0.3, 0.55, 0.8]:
                rows.append(
                    {
                        "candidate_id": f"c{cid}",
                        "room": room,
                        "activity": activity,
                        "confidence": confidence,
                        "predicted_label": activity,
                        "baseline_label": activity if confidence > 0.3 else "other",
                    }
                )
                cid += 1
    return pd.DataFrame(rows)


def test_active_learning_queue_builds_policy_mix():
    frame = _candidates_frame()
    policy = ActiveLearningPolicy(
        queue_size=12,
        uncertainty_fraction=0.5,
        disagreement_fraction=0.3,
        diversity_fraction=0.2,
        uncertainty_percentile=35,
        max_share_per_room=0.5,
        max_share_per_class=0.5,
        random_seed=13,
    )
    result = build_active_learning_queue(frame, policy=policy)
    queue = pd.DataFrame(result["queue"])

    assert result["status"] == "pass"
    assert len(queue) <= policy.queue_size
    assert (queue["selection_reason"] == "uncertainty").any()
    assert (queue["selection_reason"] == "disagreement").any()
    assert (queue["selection_reason"] == "diversity").any()
    assert queue["confidence"].max() <= 0.8


def test_active_learning_prevents_room_or_class_domination():
    frame = _candidates_frame()
    policy = ActiveLearningPolicy(
        queue_size=10,
        uncertainty_fraction=0.7,
        disagreement_fraction=0.2,
        diversity_fraction=0.1,
        max_share_per_room=0.4,
        max_share_per_class=0.4,
        random_seed=42,
    )
    result = build_active_learning_queue(frame, policy=policy)
    queue = pd.DataFrame(result["queue"])
    room_counts = queue["room"].value_counts(normalize=True)
    class_counts = queue["activity"].value_counts(normalize=True)

    assert (room_counts <= policy.max_share_per_room + 1e-9).all()
    assert (class_counts <= policy.max_share_per_class + 1e-9).all()


def test_active_learning_fails_closed_on_missing_required_activity_column():
    frame = pd.DataFrame(
        [
            {"candidate_id": "c1", "room": "bedroom", "confidence": 0.2},
            {"candidate_id": "c2", "room": "livingroom", "confidence": 0.4},
        ]
    )
    policy = ActiveLearningPolicy(queue_size=5)
    result = build_active_learning_queue(frame, policy=policy)

    assert result["status"] == "fail"
    assert result["reason"] == "invalid_candidate_schema"
    assert "activity" in result["details"]["missing_columns"]


def test_active_learning_uncertainty_percentile_is_per_room_class():
    rows = []
    for idx, conf in enumerate([0.85, 0.87, 0.89, 0.91, 0.93, 0.95]):
        rows.append(
            {
                "candidate_id": f"b{idx}",
                "room": "bedroom",
                "activity": "sleep",
                "confidence": conf,
                "predicted_label": "sleep",
                "baseline_label": "sleep",
            }
        )
    for idx, conf in enumerate([0.10, 0.12, 0.14, 0.16, 0.18, 0.20]):
        rows.append(
            {
                "candidate_id": f"l{idx}",
                "room": "livingroom",
                "activity": "relaxing",
                "confidence": conf,
                "predicted_label": "relaxing",
                "baseline_label": "relaxing",
            }
        )
    frame = pd.DataFrame(rows)
    policy = ActiveLearningPolicy(
        queue_size=6,
        uncertainty_fraction=1.0,
        disagreement_fraction=0.0,
        diversity_fraction=0.0,
        uncertainty_percentile=25.0,
        max_share_per_room=1.0,
        max_share_per_class=1.0,
        random_seed=7,
    )
    result = build_active_learning_queue(frame, policy=policy)
    queue = pd.DataFrame(result["queue"])

    assert result["status"] == "pass"
    assert {"bedroom", "livingroom"} <= set(queue["room"].unique())


def test_active_learning_refills_after_caps_to_hit_queue_target():
    rows = []
    for idx in range(30):
        rows.append(
            {
                "candidate_id": f"b{idx}",
                "room": "bedroom",
                "activity": "sleep",
                "confidence": 0.05 + (idx * 0.002),
                "predicted_label": "sleep",
                "baseline_label": "other",
            }
        )
    for idx in range(30):
        rows.append(
            {
                "candidate_id": f"l{idx}",
                "room": "livingroom",
                "activity": "relaxing",
                "confidence": 0.30 + (idx * 0.002),
                "predicted_label": "relaxing",
                "baseline_label": "relaxing",
            }
        )
    for idx in range(30):
        rows.append(
            {
                "candidate_id": f"k{idx}",
                "room": "kitchen",
                "activity": "cook",
                "confidence": 0.45 + (idx * 0.002),
                "predicted_label": "cook",
                "baseline_label": "cook",
            }
        )
    frame = pd.DataFrame(rows)
    policy = ActiveLearningPolicy(
        queue_size=20,
        uncertainty_fraction=0.5,
        disagreement_fraction=0.5,
        diversity_fraction=0.0,
        uncertainty_percentile=25.0,
        max_share_per_room=0.35,
        max_share_per_class=1.0,
        random_seed=11,
    )
    result = build_active_learning_queue(frame, policy=policy)
    queue = pd.DataFrame(result["queue"])

    assert result["status"] == "pass"
    assert len(queue) == policy.queue_size
    assert queue["room"].value_counts().max() <= int(policy.queue_size * policy.max_share_per_room)


def test_accepted_corrections_emit_boundary_and_hard_negative_payloads():
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "corr-1",
                "room": "bedroom",
                "activity": "sleep",
                "confidence": 0.42,
                "predicted_label": "nap",
                "baseline_label": "sleep",
                "corrected_event": True,
                "boundary_start_target": 1,
                "boundary_end_target": 1,
                "hard_negative_flag": True,
                "hard_negative_label": "nap",
                "residual_review_flag": True,
                "residual_review_rows": 2,
            }
        ]
    )
    policy = ActiveLearningPolicy(
        queue_size=1,
        uncertainty_fraction=1.0,
        disagreement_fraction=0.0,
        diversity_fraction=0.0,
        max_share_per_room=1.0,
        max_share_per_class=1.0,
    )

    result = build_active_learning_queue(frame, policy=policy)
    queue = pd.DataFrame(result["queue"])

    assert result["status"] == "pass"
    assert bool(queue.iloc[0]["corrected_event"]) is True
    assert int(queue.iloc[0]["boundary_start_target"]) == 1
    assert int(queue.iloc[0]["boundary_end_target"]) == 1
    assert bool(queue.iloc[0]["hard_negative_flag"]) is True
    assert queue.iloc[0]["hard_negative_label"] == "nap"
    assert float(queue.iloc[0]["triage_priority_score"]) > float(queue.iloc[0]["uncertainty_score"])


def test_active_learning_triage_prioritizes_high_yield_segments():
    frame = pd.DataFrame(
        [
            {
                "candidate_id": "corr-1",
                "room": "bedroom",
                "activity": "sleep",
                "confidence": 0.48,
                "predicted_label": "nap",
                "baseline_label": "sleep",
                "corrected_event": True,
                "boundary_start_target": 1,
                "boundary_end_target": 1,
                "hard_negative_flag": True,
                "residual_review_flag": False,
            },
            {
                "candidate_id": "plain-1",
                "room": "bedroom",
                "activity": "sleep",
                "confidence": 0.30,
                "predicted_label": "sleep",
                "baseline_label": "sleep",
                "corrected_event": False,
                "boundary_start_target": 0,
                "boundary_end_target": 0,
                "hard_negative_flag": False,
                "residual_review_flag": False,
            },
        ]
    )
    policy = ActiveLearningPolicy(
        queue_size=2,
        uncertainty_fraction=1.0,
        disagreement_fraction=0.0,
        diversity_fraction=0.0,
        uncertainty_percentile=100.0,
        max_share_per_room=1.0,
        max_share_per_class=1.0,
    )

    result = build_active_learning_queue(frame, policy=policy)
    queue = pd.DataFrame(result["queue"])

    assert result["status"] == "pass"
    assert list(queue["candidate_id"]) == ["corr-1", "plain-1"]
    assert float(queue.iloc[0]["triage_priority_score"]) > float(queue.iloc[1]["triage_priority_score"])
