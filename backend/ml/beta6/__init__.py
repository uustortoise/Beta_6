"""
Beta 6 Universal Backbone Module.

This package contains the training and inference code for the
Universal Backbone model used in Beta 6's Transfer Learning architecture.
"""

from .gates.intake_gate import (
    APPROVED_STATUS,
    INTAKE_ARTIFACT_VERSION,
    REJECTED_STATUS,
    assert_intake_artifact_approved,
    load_intake_artifact,
    validate_intake_artifact,
)
from .gates.intake_precheck import (
    REASON_INTAKE_INVALID_ARTIFACT,
    REASON_INTAKE_MISSING_ARTIFACT,
    REASON_INTAKE_NOT_APPROVED,
    IntakeGateBlockedError,
    enforce_approved_intake_artifact,
)
from .evaluation.runtime_eval_parity import (
    DEFAULT_LABEL_MAP,
    DecoderPolicy,
    assert_fixed_trace_parity,
    compare_runtime_eval_steps,
    decode_eval_trace,
    decode_runtime_trace,
    run_fixed_trace_parity,
)
from .orchestrator import (
    Beta6Orchestrator,
    Phase1ParityResult,
    Phase4HeadResult,
    Phase4HMMResult,
    Phase4UnknownResult,
    Phase5AdapterResult,
    Phase5CRFABResult,
    Phase6ShadowCompareResult,
    Phase3FineTuneResult,
    Phase2PretrainResult,
    PhaseGateError,
)
from .serving.head_factory import (
    HeadSpec,
    build_dynamic_head_specs,
    summarize_head_specs,
)
from .serving.prediction import (
    UnknownPolicy,
    build_triage_candidates_from_inference,
    infer_with_unknown_path,
    load_unknown_policy,
)
from .evaluation.calibration import (
    CalibrationReport,
    evaluate_calibration,
    expected_calibration_error,
    multiclass_brier_score,
)
from .evaluation.evaluation_engine import (
    build_room_evaluation_report,
    create_signed_evaluation_report,
    verify_evaluation_report_signature,
)
from .evaluation.evaluation_metrics import (
    build_room_status,
    derive_room_confidence,
    parse_room_override_map,
    resolve_float_threshold_with_source,
    resolve_int_threshold_with_source,
)
from .registry.rejection_artifact import (
    create_signed_rejection_artifact,
    verify_rejection_artifact_signature,
)
from .sequence.hmm_decoder import (
    HMMDecodeResult,
    decode_hmm_with_duration_priors,
)
from .sequence.crf_decoder import (
    CRFDecodeResult,
    decode_crf_with_duration_priors,
    fit_transition_log_matrix_from_sequences,
)
from .sequence.transition_builder import (
    DurationPrior,
    DurationPriorPolicy,
    TransitionPolicy,
    build_allowed_transition_map,
    build_transition_log_matrix,
    duration_log_penalty,
    load_duration_prior_policy,
)
from .contracts.label_registry import (
    DEFAULT_MANDATORY_CLASSES,
    LabelRegistry,
    build_label_registry,
    build_label_registry_from_training_frame,
)
from .training.fine_tune_safe_classes import (
    SafeFineTuneConfig,
    load_safe_finetune_config,
    run_safe_class_finetune,
)
from .training.active_learning import (
    ActiveLearningPolicy,
    build_active_learning_queue,
    load_active_learning_policy,
)
from .training.self_supervised_pretrain import (
    PretrainConfig,
    load_pretrain_config,
    run_self_supervised_pretraining,
)
from .adapters.adapter_store import (
    AdapterStore,
    AdapterStorePolicy,
    load_adapter_store_policy,
)
from .adapters.lora_adapter import (
    LoRAAdapterArtifact,
    LoRAAdapterConfig,
    load_adapter_artifact,
    load_adapter_config,
    save_adapter_artifact,
    score_with_adapter,
    train_lora_adapter_from_frame,
)
from .evaluation.representation_eval import (
    RepresentationEvalResult,
    evaluate_representation_quality,
)
from .evaluation.shadow_compare import (
    build_shadow_compare_report,
    create_signed_shadow_compare_report,
    verify_shadow_compare_signature,
)
from .evaluation.slo_observability import (
    DEFAULT_SLO_POLICY,
    SLOSeverity,
    evaluate_model_behavior_slo,
    generate_daily_slo_report,
)

__all__ = [
    "APPROVED_STATUS",
    "INTAKE_ARTIFACT_VERSION",
    "IntakeGateBlockedError",
    "REASON_INTAKE_INVALID_ARTIFACT",
    "REASON_INTAKE_MISSING_ARTIFACT",
    "REASON_INTAKE_NOT_APPROVED",
    "REJECTED_STATUS",
    "DEFAULT_LABEL_MAP",
    "DEFAULT_SLO_POLICY",
    "Beta6Orchestrator",
    "ActiveLearningPolicy",
    "SafeFineTuneConfig",
    "HeadSpec",
    "DecoderPolicy",
    "UnknownPolicy",
    "CalibrationReport",
    "DurationPrior",
    "DurationPriorPolicy",
    "TransitionPolicy",
    "LabelRegistry",
    "Phase1ParityResult",
    "Phase4HeadResult",
    "Phase4HMMResult",
    "Phase4UnknownResult",
    "Phase3FineTuneResult",
    "Phase2PretrainResult",
    "PhaseGateError",
    "PretrainConfig",
    "HMMDecodeResult",
    "CRFDecodeResult",
    "RepresentationEvalResult",
    "SLOSeverity",
    "AdapterStore",
    "AdapterStorePolicy",
    "LoRAAdapterArtifact",
    "LoRAAdapterConfig",
    "DEFAULT_MANDATORY_CLASSES",
    "assert_intake_artifact_approved",
    "assert_fixed_trace_parity",
    "compare_runtime_eval_steps",
    "build_dynamic_head_specs",
    "decode_eval_trace",
    "decode_runtime_trace",
    "decode_hmm_with_duration_priors",
    "decode_crf_with_duration_priors",
    "infer_with_unknown_path",
    "build_triage_candidates_from_inference",
    "build_room_evaluation_report",
    "build_room_status",
    "create_signed_evaluation_report",
    "verify_evaluation_report_signature",
    "create_signed_rejection_artifact",
    "verify_rejection_artifact_signature",
    "evaluate_calibration",
    "expected_calibration_error",
    "multiclass_brier_score",
    "build_allowed_transition_map",
    "build_transition_log_matrix",
    "duration_log_penalty",
    "fit_transition_log_matrix_from_sequences",
    "build_label_registry",
    "build_label_registry_from_training_frame",
    "evaluate_model_behavior_slo",
    "enforce_approved_intake_artifact",
    "generate_daily_slo_report",
    "build_active_learning_queue",
    "evaluate_representation_quality",
    "derive_room_confidence",
    "load_intake_artifact",
    "load_active_learning_policy",
    "load_unknown_policy",
    "load_duration_prior_policy",
    "load_pretrain_config",
    "load_safe_finetune_config",
    "load_adapter_artifact",
    "load_adapter_config",
    "load_adapter_store_policy",
    "parse_room_override_map",
    "resolve_float_threshold_with_source",
    "resolve_int_threshold_with_source",
    "run_fixed_trace_parity",
    "run_safe_class_finetune",
    "run_self_supervised_pretraining",
    "save_adapter_artifact",
    "score_with_adapter",
    "summarize_head_specs",
    "train_lora_adapter_from_frame",
    "validate_intake_artifact",
    "Phase5AdapterResult",
    "Phase5CRFABResult",
    "Phase6ShadowCompareResult",
    "build_shadow_compare_report",
    "create_signed_shadow_compare_report",
    "verify_shadow_compare_signature",
]
