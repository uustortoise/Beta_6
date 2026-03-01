from scripts.generate_residual_review_pack import build_residual_pack


def test_build_residual_pack_aggregates_room_day_and_top_windows():
    reports = [
        {
            "seed": 11,
            "splits": [
                {
                    "test_day": 6,
                    "rooms": {
                        "Kitchen": {
                            "error_episodes": {
                                "positive_label": "kitchen_normal_use",
                                "fn_minutes_total": 120.0,
                                "fp_minutes_total": 20.0,
                                "fn_top": [{"start": "2025-12-06 09:00:00", "end": "2025-12-06 10:00:00", "duration_minutes": 60.0}],
                                "fp_top": [{"start": "2025-12-06 14:00:00", "end": "2025-12-06 14:10:00", "duration_minutes": 10.0}],
                            }
                        }
                    },
                }
            ],
        },
        {
            "seed": 22,
            "splits": [
                {
                    "test_day": 6,
                    "rooms": {
                        "Kitchen": {
                            "error_episodes": {
                                "positive_label": "kitchen_normal_use",
                                "fn_minutes_total": 90.0,
                                "fp_minutes_total": 30.0,
                                "fn_top": [{"start": "2025-12-06 12:00:00", "end": "2025-12-06 12:30:00", "duration_minutes": 30.0}],
                                "fp_top": [],
                            }
                        }
                    },
                }
            ],
        },
    ]
    pack = build_residual_pack(reports, day_file_map={6: "HK0011_jessica_train_6dec2025.xlsx"}, top_k=5)
    assert "Kitchen" in pack["room_day_summary"]
    row = pack["room_day_summary"]["Kitchen"][0]
    assert row["day"] == 6
    assert row["file"] == "HK0011_jessica_train_6dec2025.xlsx"
    assert row["fn_minutes_mean"] == 105.0
    assert row["fp_minutes_mean"] == 25.0
    assert len(pack["top_windows"]) >= 2

