"""Beta 6 orchestrator for parity checks and Phase 2 pretrain/eval flow."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .contracts.label_registry import DEFAULT_MANDATORY_CLASSES, build_label_registry_from_training_frame
from .evaluation.evaluation_engine import build_room_evaluation_report
from .evaluation.shadow_compare import create_signed_shadow_compare_report
from .evaluation.representation_eval import evaluate_representation_quality
from .evaluation.runtime_eval_parity import DecoderPolicy, run_fixed_trace_parity
from .gates.intake_precheck import enforce_approved_intake_artifact
from .registry.gate_engine import GateEngine
from .serving.head_factory import build_dynamic_head_specs, summarize_head_specs
from .serving.prediction import (
    build_triage_candidates_from_inference,
    infer_with_unknown_path,
    load_unknown_policy,
)
from .sequence.hmm_decoder import decode_hmm_with_duration_priors
from .sequence.crf_decoder import decode_crf_with_duration_priors
from .sequence.transition_builder import (
    build_allowed_transition_map,
    build_transition_log_matrix,
    load_duration_prior_policy,
)
from .adapters.adapter_store import AdapterStore, load_adapter_store_policy
from .adapters.lora_adapter import load_adapter_config, train_lora_adapter_from_frame
from .training.active_learning import build_active_learning_queue, load_active_learning_policy
from .training.fine_tune_safe_classes import run_safe_class_finetune
from .training.self_supervised_pretrain import (
    encode_with_checkpoint,
    load_pretrain_checkpoint,
    run_self_supervised_pretraining,
)


REASON_RUNTIME_EVAL_PARITY_FAILED = "runtime_eval_parity_failed"
REASON_INTAKE_REQUIRED = "intake_gate_missing_artifact"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


class PhaseGateError(RuntimeError):
    def __init__(self, *, reason_code: str, detail: str):
        super().__init__(detail)
        self.reason_code = reason_code
        self.detail = detail


@dataclass(frozen=True)
class Phase1ParityResult:
    passed: bool
    mismatch_count: int


@dataclass(frozen=True)
class Phase2PretrainResult:
    status: str
    checkpoint_npz: str
    checkpoint_metadata: str
    metrics: Dict[str, Any]


@dataclass(frozen=True)
class Phase3FineTuneResult:
    status: str
    heldout_accuracy: float
    report_json: str
    model_joblib: str


@dataclass(frozen=True)
class Phase4HeadResult:
    label_registry: Dict[str, Any]
    head_specs: Dict[str, Any]
    summary: Dict[str, Any]


@dataclass(frozen=True)
class Phase4HMMResult:
    labels: Sequence[str]
    state_indices: Sequence[int]
    score: float
    ping_pong_rate: float


@dataclass(frozen=True)
class Phase4UnknownResult:
    inference: Dict[str, Any]
    triage_candidates: Sequence[Dict[str, Any]]
    room_report: Dict[str, Any]


@dataclass(frozen=True)
class Phase5AdapterResult:
    status: str
    adapter_id: str
    warmup_accuracy: float
    warmup_pass: bool
    promoted: bool
    active_adapter_id: Optional[str]
    store_root: str


@dataclass(frozen=True)
class Phase5CRFABResult:
    status: str
    pass_gate: bool
    hmm_ping_pong_rate: float
    crf_ping_pong_rate: float
    hmm_accuracy: Optional[float]
    crf_accuracy: Optional[float]
    decode_length: int
    details: Dict[str, Any]


@dataclass(frozen=True)
class Phase6ShadowCompareResult:
    status: str
    divergence_count: int
    unexplained_divergence_count: int
    divergence_rate: float
    unexplained_divergence_rate: float
    report_path: Optional[str]
    signature: str
    badges: Sequence[Dict[str, Any]]


class Beta6Orchestrator:
    """Single authority runner for Step 1.5 parity and Phase 2 execution."""

    def __init__(self, *, require_intake_artifact: bool = True):
        self.require_intake_artifact = bool(require_intake_artifact)

    def _enforce_intake(self, intake_artifact_path: Optional[str | Path]) -> Optional[Dict[str, Any]]:
        if not self.require_intake_artifact:
            return None
        if intake_artifact_path is None:
            raise PhaseGateError(
                reason_code=REASON_INTAKE_REQUIRED,
                detail="phase execution blocked: missing intake artifact path",
            )
        return enforce_approved_intake_artifact(Path(intake_artifact_path).resolve())

    def run_phase1_parity_gate(
        self,
        *,
        trace_steps: Sequence[Mapping[str, Any]],
        runtime_label_map: Optional[Mapping[str, str]] = None,
        eval_label_map: Optional[Mapping[str, str]] = None,
        runtime_policy: DecoderPolicy = DecoderPolicy(),
        eval_policy: DecoderPolicy = DecoderPolicy(),
    ) -> Phase1ParityResult:
        report = run_fixed_trace_parity(
            trace_steps=trace_steps,
            runtime_label_map=runtime_label_map,
            eval_label_map=eval_label_map,
            runtime_policy=runtime_policy,
            eval_policy=eval_policy,
        )
        if not report.passed:
            detail = "; ".join(
                [
                    f"{m.step_index}:{m.field}:{m.runtime_value}!={m.eval_value}"
                    for m in report.mismatches[:5]
                ]
            )
            raise PhaseGateError(
                reason_code=REASON_RUNTIME_EVAL_PARITY_FAILED,
                detail=detail or "runtime/eval parity mismatches detected",
            )
        return Phase1ParityResult(passed=True, mismatch_count=0)

    def run_phase2_pretraining(
        self,
        *,
        manifest_path: str | Path,
        output_dir: str | Path,
        config_path: str | Path | None = None,
        intake_artifact_path: str | Path | None = None,
        max_files: Optional[int] = None,
    ) -> Phase2PretrainResult:
        self._enforce_intake(intake_artifact_path)
        summary = run_self_supervised_pretraining(
            manifest_path=manifest_path,
            config_path=config_path,
            output_dir=output_dir,
            max_files=max_files,
        )
        artifacts = dict(summary.get("artifacts", {}))
        return Phase2PretrainResult(
            status=str(summary.get("status", "fail")),
            checkpoint_npz=str(artifacts.get("checkpoint_npz", "")),
            checkpoint_metadata=str(artifacts.get("checkpoint_metadata", "")),
            metrics=dict(summary.get("metrics", {})),
        )

    def run_phase2_representation_eval(
        self,
        *,
        dataset_csv: str | Path,
        checkpoint_npz: str | Path,
        output_path: str | Path,
        seed: int = 42,
        test_fraction: float = 0.2,
        knn_k: int = 5,
    ) -> Dict[str, Any]:
        frame = pd.read_csv(Path(dataset_csv).resolve())
        for column in ("resident_id", "label"):
            if column not in frame.columns:
                raise ValueError(f"representation dataset missing required column: {column}")
        feature_matrix = (
            frame.drop(columns=["resident_id", "label"])
            .select_dtypes(include=[np.number])
            .to_numpy(dtype=np.float32)
        )
        if feature_matrix.size == 0:
            raise ValueError("representation dataset must include numeric features")

        checkpoint = load_pretrain_checkpoint(checkpoint_npz)
        embeddings = encode_with_checkpoint(feature_matrix, checkpoint)
        result = evaluate_representation_quality(
            embeddings=embeddings,
            labels=frame["label"].astype(str).to_numpy(),
            resident_ids=frame["resident_id"].astype(str).to_numpy(),
            seed=int(seed),
            test_fraction=float(test_fraction),
            knn_k=max(int(knn_k), 1),
        )
        report = {
            "status": "pass" if result.improvement_margin > 0 else "fail",
            "metrics": result.to_dict(),
            "checkpoint_npz": str(Path(checkpoint_npz).resolve()),
            "dataset_csv": str(Path(dataset_csv).resolve()),
        }
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def run_phase3_safe_finetune(
        self,
        *,
        dataset_path: str | Path,
        output_dir: str | Path,
        config_path: str | Path | None = None,
        pretrain_checkpoint: str | Path | None = None,
        intake_artifact_path: str | Path | None = None,
    ) -> Phase3FineTuneResult:
        self._enforce_intake(intake_artifact_path)
        report = run_safe_class_finetune(
            dataset_path=dataset_path,
            config_path=config_path,
            output_dir=output_dir,
            pretrain_checkpoint=pretrain_checkpoint,
        )
        artifacts = _as_mapping(report.get("artifacts"))
        metrics = _as_mapping(report.get("metrics"))
        return Phase3FineTuneResult(
            status=str(report.get("status", "fail")),
            heldout_accuracy=float(metrics.get("heldout_accuracy", 0.0)),
            report_json=str(artifacts.get("report_json", "")),
            model_joblib=str(artifacts.get("model_joblib", "")),
        )

    def run_phase3_active_learning_triage(
        self,
        *,
        candidates_csv: str | Path,
        output_csv: str | Path,
        report_json: str | Path,
        policy_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        frame = pd.read_csv(Path(candidates_csv).resolve())
        policy = load_active_learning_policy(policy_path)
        result = build_active_learning_queue(frame, policy=policy)
        queue = pd.DataFrame(result.get("queue", []))

        out_csv = Path(output_csv).resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        queue.to_csv(out_csv, index=False)

        out_report = Path(report_json).resolve()
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(result, indent=2), encoding="utf-8")
        result["artifacts"] = {"queue_csv": str(out_csv), "report_json": str(out_report)}
        return result

    def run_phase4_build_dynamic_heads(
        self,
        *,
        training_frame: pd.DataFrame,
        room_col: str = "room",
        activity_col: str = "activity",
        manual_additions_by_room: Optional[Mapping[str, Sequence[str]]] = None,
        mandatory_classes: Sequence[str] = DEFAULT_MANDATORY_CLASSES,
        include_backend_registry: bool = True,
    ) -> Phase4HeadResult:
        label_registry = build_label_registry_from_training_frame(
            training_frame,
            room_col=room_col,
            activity_col=activity_col,
            manual_additions_by_room=manual_additions_by_room,
            mandatory_classes=mandatory_classes,
            include_backend_registry=include_backend_registry,
        )
        head_specs = build_dynamic_head_specs(
            label_registry=label_registry,
            mandatory_classes=mandatory_classes,
        )
        return Phase4HeadResult(
            label_registry=label_registry.to_dict(),
            head_specs={room: spec.to_dict() for room, spec in head_specs.items()},
            summary=summarize_head_specs(head_specs),
        )

    def run_phase4_hmm_baseline(
        self,
        *,
        observation_log_probs: np.ndarray,
        labels: Sequence[str],
        duration_policy_path: str | Path | None = None,
        disallowed_pairs: Optional[Sequence[tuple[str, str]]] = None,
        room_name: str | None = None,
        resident_home_context: Optional[Mapping[str, Any]] = None,
    ) -> Phase4HMMResult:
        duration_policy = load_duration_prior_policy(duration_policy_path)
        allowed_map = build_allowed_transition_map(
            labels,
            disallowed_pairs=disallowed_pairs,
        )
        transition_matrix = build_transition_log_matrix(
            labels,
            allowed_map=allowed_map,
            policy=duration_policy.transition,
            room_name=room_name,
            resident_home_context=resident_home_context,
        )
        result = decode_hmm_with_duration_priors(
            observation_log_probs=observation_log_probs,
            labels=labels,
            transition_log_matrix=transition_matrix,
            duration_policy=duration_policy,
            room_name=room_name,
            resident_home_context=resident_home_context,
        )
        return Phase4HMMResult(
            labels=result.labels,
            state_indices=result.state_indices,
            score=float(result.score),
            ping_pong_rate=float(result.ping_pong_rate),
        )

    def run_phase4_unknown_abstain(
        self,
        *,
        probabilities: np.ndarray,
        labels: Sequence[str],
        room: str,
        activity_hint: str,
        unknown_policy_path: str | Path | None = None,
        outside_sensed_space_scores: Optional[np.ndarray] = None,
        true_labels: Optional[Sequence[str]] = None,
        true_indices: Optional[Sequence[int]] = None,
    ) -> Phase4UnknownResult:
        unknown_policy = load_unknown_policy(unknown_policy_path)
        inference = infer_with_unknown_path(
            probabilities=probabilities,
            labels=labels,
            policy=unknown_policy,
            outside_sensed_space_scores=outside_sensed_space_scores,
        )
        triage = build_triage_candidates_from_inference(
            inference=inference,
            room=room,
            activity_hint=activity_hint,
        )
        room_report = build_room_evaluation_report(
            room=room,
            y_true=true_labels,
            y_pred=inference.get("labels"),
            probabilities=probabilities if true_indices is not None else None,
            true_indices=true_indices,
            uncertainty_states=inference.get("uncertainty_states"),
            unknown_policy=unknown_policy,
        )
        return Phase4UnknownResult(
            inference=inference,
            triage_candidates=triage,
            room_report=room_report,
        )

    def run_phase4_dynamic_gate(
        self,
        *,
        room_reports: Sequence[Mapping[str, Any]],
        run_id: str,
        elder_id: str,
        signing_key: str,
        output_dir: str | Path | None = None,
    ) -> Dict[str, Any]:
        engine = GateEngine()
        return engine.decide_run_from_reports(
            room_reports=room_reports,
            run_id=run_id,
            elder_id=elder_id,
            signing_key=signing_key,
            output_dir=output_dir,
        )

    def run_phase5_adapter_lifecycle(
        self,
        *,
        training_frame: pd.DataFrame,
        resident_id: str,
        room: str,
        backbone_id: str,
        run_id: str,
        adapter_store_root: str | Path,
        adapter_config_path: str | Path | None = None,
        policy_path: str | Path | None = None,
        label_col: str = "activity",
        auto_promote: bool = True,
    ) -> Phase5AdapterResult:
        adapter_config = load_adapter_config(adapter_config_path)
        store = AdapterStore(
            root=adapter_store_root,
            policy=load_adapter_store_policy(policy_path),
        )
        artifact = train_lora_adapter_from_frame(
            training_frame,
            resident_id=resident_id,
            room=room,
            backbone_id=backbone_id,
            config=adapter_config,
            label_col=label_col,
        )
        store.create_adapter(artifact=artifact, run_id=run_id)
        warmed = store.warmup_adapter(
            resident_id=resident_id,
            room=room,
            adapter_id=artifact.adapter_id,
            warmup_accuracy=artifact.warmup_accuracy,
            run_id=run_id,
        )
        warmup_pass = bool(warmed.get("warmup_pass", False))
        promoted = False
        active = store.get_active_adapter(resident_id, room)
        if auto_promote and warmup_pass:
            active = store.promote_adapter(
                resident_id=resident_id,
                room=room,
                adapter_id=artifact.adapter_id,
                run_id=run_id,
                metadata={"phase": "phase5_step5_1"},
            )
            promoted = True
        store.enforce_retention(resident_id=resident_id, room=room)
        store.auto_retire_inactive(resident_id=resident_id, room=room, run_id=run_id)
        return Phase5AdapterResult(
            status="pass" if (warmup_pass or not auto_promote) else "fail",
            adapter_id=artifact.adapter_id,
            warmup_accuracy=float(artifact.warmup_accuracy),
            warmup_pass=warmup_pass,
            promoted=promoted,
            active_adapter_id=str((active or {}).get("adapter_id")) if isinstance(active, Mapping) else None,
            store_root=str(Path(adapter_store_root).resolve()),
        )

    def run_phase5_crf_ab_gate(
        self,
        *,
        observation_log_probs: np.ndarray,
        labels: Sequence[str],
        true_labels: Optional[Sequence[str]] = None,
        duration_policy_path: str | Path | None = None,
        disallowed_pairs: Optional[Sequence[tuple[str, str]]] = None,
        label_sequences_for_fit: Optional[Sequence[Sequence[str]]] = None,
        room_name: str | None = None,
        resident_home_context: Optional[Mapping[str, Any]] = None,
    ) -> Phase5CRFABResult:
        duration_policy = load_duration_prior_policy(duration_policy_path)
        allowed_map = build_allowed_transition_map(
            labels,
            disallowed_pairs=disallowed_pairs,
        )
        transition_matrix = build_transition_log_matrix(
            labels,
            allowed_map=allowed_map,
            policy=duration_policy.transition,
            room_name=room_name,
            resident_home_context=resident_home_context,
        )
        hmm = decode_hmm_with_duration_priors(
            observation_log_probs=observation_log_probs,
            labels=labels,
            transition_log_matrix=transition_matrix,
            duration_policy=duration_policy,
            room_name=room_name,
            resident_home_context=resident_home_context,
        )
        crf = decode_crf_with_duration_priors(
            observation_log_probs=observation_log_probs,
            labels=labels,
            duration_policy=duration_policy,
            transition_log_matrix=None,
            label_sequences_for_fit=label_sequences_for_fit,
            disallowed_pairs=disallowed_pairs,
            room_name=room_name,
            resident_home_context=resident_home_context,
        )

        hmm_accuracy: Optional[float] = None
        crf_accuracy: Optional[float] = None
        if true_labels is not None:
            target = np.asarray([str(v).strip().lower() for v in true_labels], dtype=object)
            if len(target) != len(hmm.labels):
                raise ValueError("true_labels length must match decode sequence length")
            hmm_accuracy = float(np.mean(target == np.asarray(hmm.labels, dtype=object)))
            crf_accuracy = float(np.mean(target == np.asarray(crf.labels, dtype=object)))

        if hmm_accuracy is not None and crf_accuracy is not None:
            pass_gate = bool(crf_accuracy >= hmm_accuracy)
        else:
            pass_gate = bool(crf.ping_pong_rate <= hmm.ping_pong_rate + 1e-9)
        status = "pass" if pass_gate else "fail"
        details = {
            "criterion": "accuracy_non_regression" if hmm_accuracy is not None else "ping_pong_non_regression",
            "hmm_score": float(hmm.score),
            "crf_score": float(crf.score),
        }
        return Phase5CRFABResult(
            status=status,
            pass_gate=pass_gate,
            hmm_ping_pong_rate=float(hmm.ping_pong_rate),
            crf_ping_pong_rate=float(crf.ping_pong_rate),
            hmm_accuracy=hmm_accuracy,
            crf_accuracy=crf_accuracy,
            decode_length=int(len(hmm.labels)),
            details=details,
        )

    def run_phase6_shadow_compare(
        self,
        *,
        room_rows: Sequence[Mapping[str, Any]],
        run_id: str,
        elder_id: str,
        signing_key: str,
        output_path: str | Path | None = None,
        unexplained_divergence_rate_max: float = 0.05,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Phase6ShadowCompareResult:
        artifact = create_signed_shadow_compare_report(
            run_id=run_id,
            elder_id=elder_id,
            room_rows=room_rows,
            signing_key=signing_key,
            output_path=output_path,
            unexplained_divergence_rate_max=float(unexplained_divergence_rate_max),
            metadata=metadata,
        )
        summary = _as_mapping(artifact.get("summary"))
        return Phase6ShadowCompareResult(
            status=str(summary.get("status", "insufficient_data")),
            divergence_count=int(summary.get("divergence_count", 0) or 0),
            unexplained_divergence_count=int(summary.get("unexplained_divergence_count", 0) or 0),
            divergence_rate=float(summary.get("divergence_rate", 0.0) or 0.0),
            unexplained_divergence_rate=float(summary.get("unexplained_divergence_rate", 0.0) or 0.0),
            report_path=str(Path(output_path).resolve()) if output_path is not None else None,
            signature=str(artifact.get("signature", "")),
            badges=list(artifact.get("badges", [])) if isinstance(artifact.get("badges"), list) else [],
        )


__all__ = [
    "Beta6Orchestrator",
    "Phase1ParityResult",
    "Phase4HeadResult",
    "Phase4HMMResult",
    "Phase4UnknownResult",
    "Phase5AdapterResult",
    "Phase5CRFABResult",
    "Phase6ShadowCompareResult",
    "Phase3FineTuneResult",
    "Phase2PretrainResult",
    "PhaseGateError",
    "REASON_INTAKE_REQUIRED",
    "REASON_RUNTIME_EVAL_PARITY_FAILED",
]
