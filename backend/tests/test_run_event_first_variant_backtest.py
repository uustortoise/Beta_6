from pathlib import Path

import yaml

from scripts.run_event_first_variant_backtest import _build_argv


def test_build_argv_uses_defaults_and_variant_args(tmp_path: Path):
    profiles_yaml = tmp_path / "profiles.yaml"
    profiles_yaml.write_text(
        yaml.safe_dump(
            {
                "defaults": {
                    "min_day": 4,
                    "max_day": 10,
                    "occupancy_threshold": 0.35,
                    "calibration_method": "isotonic",
                    "calib_fraction": 0.2,
                    "min_calib_samples": 500,
                    "min_calib_label_support": 30,
                },
                "variants": {
                    "anchor": {"args": ["--enable-bedroom-livingroom-stage-a-hgb"]},
                },
            }
        )
    )

    argv = _build_argv(
        profiles_yaml=profiles_yaml,
        variant="anchor",
        data_dir=tmp_path / "data",
        elder_id="HK0011_jessica",
        seed=11,
        output=tmp_path / "seed11.json",
        min_day_override=None,
        max_day_override=None,
    )
    joined = " ".join(argv)
    assert "--min-day 4" in joined
    assert "--max-day 10" in joined
    assert "--seed 11" in joined
    assert "--enable-bedroom-livingroom-stage-a-hgb" in joined
