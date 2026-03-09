from scripts.run_room_experiments import run_room_experiments


def test_room_experiments_can_replay_policy_sweep_candidates():
    report = run_room_experiments(["livingroom_fast_diagnosis"])

    assert report["status"] == "ok"
    assert report["candidate_count"] == 1
    profile = report["profiles"][0]
    assert profile["profile_name"] == "livingroom_fast_diagnosis"
    assert profile["room"] == "livingroom"
    assert profile["replay_mode"] == "replay_only"


def test_livingroom_replay_candidate_is_traceable_to_typed_policy_fields():
    report = run_room_experiments(["livingroom_fast_diagnosis"])
    typed_policy = report["profiles"][0]["typed_policy"]

    assert typed_policy["unoccupied_downsample"]["min_share"] == 0.12
    assert typed_policy["unoccupied_downsample"]["stride"] == 12
    assert typed_policy["minority_sampling"]["target_share"] == 0.15
    assert typed_policy["runtime"]["wf_min_minority_support"] == 5


def test_room_policy_sweeps_can_run_as_fast_replay_diagnostics():
    report = run_room_experiments(["livingroom_fast_diagnosis"], fast_replay_only=True)

    assert report["status"] == "ok"
    assert report["fast_replay_only"] is True
    assert report["profiles"][0]["fast_replay_eligible"] is True
