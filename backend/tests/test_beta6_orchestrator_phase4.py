import numpy as np
import pandas as pd

from ml.beta6.orchestrator import Beta6Orchestrator


def test_orchestrator_phase4_build_dynamic_heads(monkeypatch):
    from ml.beta6.contracts import label_registry as label_registry_module

    monkeypatch.setattr(
        label_registry_module,
        "_collect_backend_labels_by_room",
        lambda: {"bedroom": {"sleep"}},
    )
    frame = pd.DataFrame(
        {
            "room": ["bedroom", "livingroom", "livingroom"],
            "activity": ["sleep", "relaxing", "tv"],
        }
    )
    result = Beta6Orchestrator(require_intake_artifact=False).run_phase4_build_dynamic_heads(
        training_frame=frame,
        manual_additions_by_room={"livingroom": ["reading"]},
        include_backend_registry=True,
    )
    assert "bedroom" in result.label_registry["room_to_labels"]
    assert "livingroom" in result.label_registry["room_to_labels"]
    assert "active_use" in result.label_registry["room_to_labels"]["bedroom"]
    assert result.summary["room_output_dims"]["livingroom"] >= 3


def test_orchestrator_phase4_hmm_baseline():
    obs = np.array(
        [
            [0.0, -6.0],
            [-6.0, 0.0],
            [-6.0, 0.0],
        ],
        dtype=np.float64,
    )
    result = Beta6Orchestrator(require_intake_artifact=False).run_phase4_hmm_baseline(
        observation_log_probs=obs,
        labels=["a", "b"],
        disallowed_pairs=[("a", "b")],
    )
    forbidden = [pair for pair in zip(result.labels[:-1], result.labels[1:]) if pair == ("a", "b")]
    assert forbidden == []


def test_orchestrator_phase4_hmm_baseline_passes_room_context_to_transition_builder(monkeypatch):
    captured = {}

    def _fake_build_transition_log_matrix(labels, *, allowed_map=None, policy=None, room_name=None, resident_home_context=None):
        captured["room_name"] = room_name
        captured["resident_home_context"] = resident_home_context
        return np.zeros((len(labels), len(labels)), dtype=np.float64)

    monkeypatch.setattr(
        "ml.beta6.orchestrator.build_transition_log_matrix",
        _fake_build_transition_log_matrix,
    )

    obs = np.array([[0.0, -1.0], [-1.0, 0.0]], dtype=np.float64)
    context = {
        "household_type": "multi",
        "helper_presence": "present",
        "layout_topology": {"bedroom": ["entrance"]},
    }
    Beta6Orchestrator(require_intake_artifact=False).run_phase4_hmm_baseline(
        observation_log_probs=obs,
        labels=["a", "b"],
        room_name="bedroom",
        resident_home_context=context,
    )

    assert captured["room_name"] == "bedroom"
    assert captured["resident_home_context"] == context


def test_orchestrator_phase4_unknown_abstain_generates_triage():
    probs = np.array([[0.51, 0.49], [0.99, 0.01]], dtype=np.float64)
    result = Beta6Orchestrator(require_intake_artifact=False).run_phase4_unknown_abstain(
        probabilities=probs,
        labels=["active_use", "unoccupied"],
        room="bedroom",
        activity_hint="sleep",
    )
    assert len(result.inference["labels"]) == 2
    assert result.room_report["room"] == "bedroom"
    assert isinstance(result.triage_candidates, list)


def test_orchestrator_phase4_dynamic_gate_returns_signed_artifacts(tmp_path):
    room_reports = [
        {"room": "bedroom", "passed": True, "metrics_passed": True, "details": {}},
        {"room": "livingroom", "passed": False, "metrics_passed": False, "reason_code": "fail_gate_policy"},
    ]
    result = Beta6Orchestrator(require_intake_artifact=False).run_phase4_dynamic_gate(
        room_reports=room_reports,
        run_id="r1",
        elder_id="e1",
        signing_key="k",
        output_dir=tmp_path,
    )
    assert result["run_decision"]["passed"] is False
    assert result["evaluation_report"]["run_id"] == "r1"
