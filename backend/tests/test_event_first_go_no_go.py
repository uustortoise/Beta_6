from scripts.run_event_first_matrix import _evaluate_go_no_go


def _seed_report(
    *,
    lr_pass: bool,
    bedroom_pass: bool,
    day7_lr_recall: float,
    day8_sleep_recall: float,
    day8_lr_fragmentation: float,
    lr_episode_recall: float = 0.50,
    lr_episode_f1: float = 0.45,
    lr_active_mae_minutes: float = 120.0,
    bedroom_sleep_mae_minutes: float = 85.0,
    day8_sleep_support: float = 500.0,
):
    sleep_payload = {"recall": float(day8_sleep_recall), "support": float(day8_sleep_support)}
    return {
        "summary": {
            "LivingRoom": {"livingroom_active_mae_minutes": float(lr_active_mae_minutes)},
            "Bedroom": {"sleep_duration_mae_minutes": float(bedroom_sleep_mae_minutes)},
        },
        "splits": [
            {
                "test_day": 7,
                "rooms": {
                    "LivingRoom": {
                        "hard_gate": {"eligible": True, "pass": lr_pass},
                        "classification": {"occupied_recall": day7_lr_recall},
                        "episode_metrics": {"episode_recall": lr_episode_recall, "episode_f1": lr_episode_f1},
                    },
                    "Bedroom": {
                        "hard_gate": {"eligible": True, "pass": bedroom_pass},
                    },
                },
            },
            {
                "test_day": 8,
                "rooms": {
                    "LivingRoom": {
                        "hard_gate": {"eligible": True, "pass": lr_pass},
                        "fragmentation_score": day8_lr_fragmentation,
                        "episode_metrics": {"episode_recall": lr_episode_recall, "episode_f1": lr_episode_f1},
                    },
                    "Bedroom": {
                        "hard_gate": {"eligible": True, "pass": bedroom_pass},
                        "label_recall_summary": {"sleep": sleep_payload},
                    },
                },
            },
        ]
    }


def test_evaluate_go_no_go_passes_when_all_thresholds_met():
    cfg = {
        "go_no_go": {
            "overall_eligible_pass_count_min": 3,
            "livingroom_eligible_pass_count_min": 2,
            "bedroom_max_regression_splits": 1,
            "day7_livingroom_recall_min": 0.45,
            "day7_livingroom_episode_recall_min": 0.40,
            "day8_bedroom_sleep_recall_min": 0.40,
            "day8_livingroom_fragmentation_min": 0.45,
            "livingroom_episode_recall_min": 0.40,
            "livingroom_episode_f1_min": 0.35,
        }
    }
    result = _evaluate_go_no_go(
        [
            _seed_report(
                lr_pass=True,
                bedroom_pass=True,
                day7_lr_recall=0.50,
                day8_sleep_recall=0.55,
                day8_lr_fragmentation=0.48,
            )
        ],
        cfg,
    )
    assert result["status"] == "pass"
    assert result["blocking_reasons"] == []


def test_evaluate_go_no_go_fails_when_day7_recall_below_floor():
    cfg = {
        "go_no_go": {
            "overall_eligible_pass_count_min": 1,
            "livingroom_eligible_pass_count_min": 1,
            "bedroom_max_regression_splits": 2,
            "day7_livingroom_recall_min": 0.45,
            "day8_bedroom_sleep_recall_min": 0.40,
            "day8_livingroom_fragmentation_min": 0.45,
        }
    }
    result = _evaluate_go_no_go(
        [
            _seed_report(
                lr_pass=True,
                bedroom_pass=True,
                day7_lr_recall=0.35,
                day8_sleep_recall=0.60,
                day8_lr_fragmentation=0.50,
            )
        ],
        cfg,
    )
    assert result["status"] == "fail"
    assert "day7_livingroom_recall_min" in result["blocking_reasons"]


def test_evaluate_go_no_go_fails_when_episode_metrics_below_floor():
    cfg = {
        "go_no_go": {
            "overall_eligible_pass_count_min": 1,
            "livingroom_eligible_pass_count_min": 1,
            "bedroom_max_regression_splits": 2,
            "day7_livingroom_recall_min": 0.30,
            "day8_bedroom_sleep_recall_min": 0.40,
            "day8_livingroom_fragmentation_min": 0.20,
            "livingroom_episode_recall_min": 0.55,
            "livingroom_episode_f1_min": 0.50,
        }
    }
    result = _evaluate_go_no_go(
        [
            _seed_report(
                lr_pass=True,
                bedroom_pass=True,
                day7_lr_recall=0.60,
                day8_sleep_recall=0.60,
                day8_lr_fragmentation=0.60,
                lr_episode_recall=0.40,
                lr_episode_f1=0.30,
            )
        ],
        cfg,
    )
    assert result["status"] == "fail"
    assert "livingroom_episode_recall_min" in result["blocking_reasons"]
    assert "livingroom_episode_f1_min" in result["blocking_reasons"]


def test_evaluate_go_no_go_treats_configured_checks_as_informational():
    cfg = {
        "go_no_go": {
            "overall_eligible_pass_count_min": 1,
            "livingroom_eligible_pass_count_min": 1,
            "bedroom_max_regression_splits": 2,
            "day7_livingroom_recall_min": 0.30,
            "day8_bedroom_sleep_recall_min": 0.40,
            "day8_livingroom_fragmentation_min": 0.20,
            "livingroom_episode_recall_min": 0.55,
            "livingroom_episode_f1_min": 0.50,
            "informational_checks": [
                "livingroom_episode_recall_min",
                "livingroom_episode_f1_min",
            ],
        }
    }
    result = _evaluate_go_no_go(
        [
            _seed_report(
                lr_pass=True,
                bedroom_pass=True,
                day7_lr_recall=0.60,
                day8_sleep_recall=0.60,
                day8_lr_fragmentation=0.60,
                lr_episode_recall=0.40,
                lr_episode_f1=0.30,
            )
        ],
        cfg,
    )
    assert result["status"] == "pass"
    assert "livingroom_episode_recall_min" not in result["blocking_reasons"]
    assert "livingroom_episode_f1_min" not in result["blocking_reasons"]
    assert "livingroom_episode_recall_min" in result["informational_failures"]
    assert "livingroom_episode_f1_min" in result["informational_failures"]


def test_evaluate_go_no_go_can_treat_livingroom_eligible_pass_count_as_informational():
    cfg = {
        "go_no_go": {
            "overall_eligible_pass_count_min": 1,
            "livingroom_eligible_pass_count_min": 2,
            "bedroom_max_regression_splits": 2,
            "day7_livingroom_recall_min": 0.30,
            "day8_bedroom_sleep_recall_min": 0.40,
            "day8_livingroom_fragmentation_min": 0.20,
            "informational_checks": [
                "livingroom_eligible_pass_count_min",
            ],
        }
    }
    result = _evaluate_go_no_go(
        [
            _seed_report(
                lr_pass=False,
                bedroom_pass=True,
                day7_lr_recall=0.60,
                day8_sleep_recall=0.60,
                day8_lr_fragmentation=0.60,
            )
        ],
        cfg,
    )
    assert result["status"] == "pass"
    assert "livingroom_eligible_pass_count_min" not in result["blocking_reasons"]
    assert "livingroom_eligible_pass_count_min" in result["informational_failures"]


def test_evaluate_go_no_go_enforces_relative_mae_non_regression_guards():
    cfg = {
        "go_no_go": {
            "overall_eligible_pass_count_min": 1,
            "livingroom_eligible_pass_count_min": 1,
            "bedroom_max_regression_splits": 2,
            "day7_livingroom_recall_min": 0.30,
            "day8_bedroom_sleep_recall_min": 0.40,
            "day8_livingroom_fragmentation_min": 0.20,
            "livingroom_active_mae_baseline_minutes": 100.0,
            "livingroom_active_mae_max_regression_pct": 10.0,
            "bedroom_sleep_mae_baseline_minutes": 80.0,
            "bedroom_sleep_mae_max_regression_pct": 10.0,
        }
    }
    result = _evaluate_go_no_go(
        [
            _seed_report(
                lr_pass=True,
                bedroom_pass=True,
                day7_lr_recall=0.60,
                day8_sleep_recall=0.60,
                day8_lr_fragmentation=0.60,
                lr_active_mae_minutes=115.0,
                bedroom_sleep_mae_minutes=89.0,
            )
        ],
        cfg,
    )
    assert result["status"] == "fail"
    assert "livingroom_active_mae_max_regression_pct" in result["blocking_reasons"]
    assert "bedroom_sleep_mae_max_regression_pct" in result["blocking_reasons"]


def test_evaluate_go_no_go_skips_day8_sleep_floor_when_support_below_minimum():
    cfg = {
        "go_no_go": {
            "overall_eligible_pass_count_min": 1,
            "livingroom_eligible_pass_count_min": 1,
            "bedroom_max_regression_splits": 2,
            "day7_livingroom_recall_min": 0.30,
            "day8_bedroom_sleep_recall_min": 0.40,
            "day8_bedroom_sleep_recall_min_support": 100,
            "day8_livingroom_fragmentation_min": 0.20,
        }
    }
    result = _evaluate_go_no_go(
        [
            _seed_report(
                lr_pass=True,
                bedroom_pass=True,
                day7_lr_recall=0.60,
                day8_sleep_recall=0.0,
                day8_sleep_support=0.0,
                day8_lr_fragmentation=0.60,
            )
        ],
        cfg,
    )
    assert result["status"] == "pass"
    assert "day8_bedroom_sleep_recall_min" not in result["blocking_reasons"]
    sleep_check = next(c for c in result["checks"] if c["name"] == "day8_bedroom_sleep_recall_min")
    assert sleep_check["pass"] is True
    assert sleep_check["low_support_skip"] is True
