"""Strict RunSpec v1 contract and schema/hash policy for Beta 6."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

RUN_SPEC_VERSION = "v1"
HASH_ALGORITHM = "sha256"
HASH_POLICY_VERSION = "run_spec_hash_policy_v1"
HASH_EXCLUDED_TOP_LEVEL_FIELDS: Tuple[str, ...] = ("run_id",)

_TOP_LEVEL_KEYS = (
    "run_spec_version",
    "run_id",
    "elder_id",
    "mode",
    "data",
    "features",
    "training",
    "evaluation",
    "gating",
)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    encoded = _canonical_json(payload).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return f"{HASH_ALGORITHM}:{digest}"


def _ensure_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _ensure_exact_keys(name: str, payload: Mapping[str, Any], required_keys: Sequence[str]) -> None:
    expected = set(required_keys)
    provided = set(payload.keys())
    missing = sorted(expected - provided)
    extra = sorted(provided - expected)
    errors = []
    if missing:
        errors.append(f"missing: {', '.join(missing)}")
    if extra:
        errors.append(f"unknown: {', '.join(extra)}")
    if errors:
        raise ValueError(f"{name} has invalid fields ({'; '.join(errors)})")


def _ensure_nonempty_str(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty string")
    return normalized


def _ensure_int(name: str, value: Any, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _ensure_float(name: str, value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float")
    normalized = float(value)
    if minimum is not None and normalized < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return normalized


@dataclass(frozen=True)
class RunSpecData:
    manifest_paths: Tuple[str, ...]
    time_zone: str
    max_ffill_gap_seconds: int
    duplicate_resolution_policy: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunSpecData":
        _ensure_exact_keys(
            "data",
            payload,
            (
                "manifest_paths",
                "time_zone",
                "max_ffill_gap_seconds",
                "duplicate_resolution_policy",
            ),
        )
        paths = payload["manifest_paths"]
        if isinstance(paths, (str, bytes)) or not isinstance(paths, Sequence):
            raise TypeError("data.manifest_paths must be a sequence of paths")
        normalized_paths = tuple(_ensure_nonempty_str("data.manifest_paths[]", path) for path in paths)
        if not normalized_paths:
            raise ValueError("data.manifest_paths must not be empty")
        return cls(
            manifest_paths=normalized_paths,
            time_zone=_ensure_nonempty_str("data.time_zone", payload["time_zone"]),
            max_ffill_gap_seconds=_ensure_int(
                "data.max_ffill_gap_seconds",
                payload["max_ffill_gap_seconds"],
                minimum=0,
            ),
            duplicate_resolution_policy=_ensure_nonempty_str(
                "data.duplicate_resolution_policy",
                payload["duplicate_resolution_policy"],
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "manifest_paths": list(self.manifest_paths),
            "time_zone": self.time_zone,
            "max_ffill_gap_seconds": self.max_ffill_gap_seconds,
            "duplicate_resolution_policy": self.duplicate_resolution_policy,
        }


@dataclass(frozen=True)
class RunSpecFeatures:
    sequence_window_seconds: int
    stride_seconds: int
    feature_version: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunSpecFeatures":
        _ensure_exact_keys(
            "features",
            payload,
            (
                "sequence_window_seconds",
                "stride_seconds",
                "feature_version",
            ),
        )
        sequence_window = _ensure_int(
            "features.sequence_window_seconds",
            payload["sequence_window_seconds"],
            minimum=1,
        )
        stride = _ensure_int("features.stride_seconds", payload["stride_seconds"], minimum=1)
        if stride > sequence_window:
            raise ValueError("features.stride_seconds must be <= features.sequence_window_seconds")
        return cls(
            sequence_window_seconds=sequence_window,
            stride_seconds=stride,
            feature_version=_ensure_nonempty_str("features.feature_version", payload["feature_version"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_window_seconds": self.sequence_window_seconds,
            "stride_seconds": self.stride_seconds,
            "feature_version": self.feature_version,
        }


@dataclass(frozen=True)
class RunSpecTraining:
    architecture_family: str
    random_seed: int
    profile: str
    optimizer: str
    learning_rate: float
    epochs: int

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunSpecTraining":
        _ensure_exact_keys(
            "training",
            payload,
            (
                "architecture_family",
                "random_seed",
                "profile",
                "optimizer",
                "learning_rate",
                "epochs",
            ),
        )
        return cls(
            architecture_family=_ensure_nonempty_str(
                "training.architecture_family",
                payload["architecture_family"],
            ),
            random_seed=_ensure_int("training.random_seed", payload["random_seed"], minimum=0),
            profile=_ensure_nonempty_str("training.profile", payload["profile"]),
            optimizer=_ensure_nonempty_str("training.optimizer", payload["optimizer"]),
            learning_rate=_ensure_float("training.learning_rate", payload["learning_rate"], minimum=1e-12),
            epochs=_ensure_int("training.epochs", payload["epochs"], minimum=1),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture_family": self.architecture_family,
            "random_seed": self.random_seed,
            "profile": self.profile,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
        }


@dataclass(frozen=True)
class RunSpecWalkForward:
    lookback_days: int
    min_train_days: int
    valid_days: int
    step_days: int
    max_folds: int

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunSpecWalkForward":
        _ensure_exact_keys(
            "evaluation.walk_forward",
            payload,
            (
                "lookback_days",
                "min_train_days",
                "valid_days",
                "step_days",
                "max_folds",
            ),
        )
        min_train_days = _ensure_int(
            "evaluation.walk_forward.min_train_days",
            payload["min_train_days"],
            minimum=1,
        )
        lookback_days = _ensure_int(
            "evaluation.walk_forward.lookback_days",
            payload["lookback_days"],
            minimum=min_train_days,
        )
        return cls(
            lookback_days=lookback_days,
            min_train_days=min_train_days,
            valid_days=_ensure_int("evaluation.walk_forward.valid_days", payload["valid_days"], minimum=1),
            step_days=_ensure_int("evaluation.walk_forward.step_days", payload["step_days"], minimum=1),
            max_folds=_ensure_int("evaluation.walk_forward.max_folds", payload["max_folds"], minimum=1),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lookback_days": self.lookback_days,
            "min_train_days": self.min_train_days,
            "valid_days": self.valid_days,
            "step_days": self.step_days,
            "max_folds": self.max_folds,
        }


@dataclass(frozen=True)
class RunSpecEvaluation:
    walk_forward: RunSpecWalkForward

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunSpecEvaluation":
        _ensure_exact_keys("evaluation", payload, ("walk_forward",))
        walk_forward = RunSpecWalkForward.from_dict(
            _ensure_mapping("evaluation.walk_forward", payload["walk_forward"])
        )
        return cls(walk_forward=walk_forward)

    def to_dict(self) -> Dict[str, Any]:
        return {"walk_forward": self.walk_forward.to_dict()}


@dataclass(frozen=True)
class RunSpecGating:
    room_policy_ref: str
    run_policy_ref: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunSpecGating":
        _ensure_exact_keys("gating", payload, ("room_policy_ref", "run_policy_ref"))
        return cls(
            room_policy_ref=_ensure_nonempty_str("gating.room_policy_ref", payload["room_policy_ref"]),
            run_policy_ref=_ensure_nonempty_str("gating.run_policy_ref", payload["run_policy_ref"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "room_policy_ref": self.room_policy_ref,
            "run_policy_ref": self.run_policy_ref,
        }


@dataclass(frozen=True)
class RunSpec:
    """Strict typed RunSpec v1 used by Beta 6 orchestration."""

    run_spec_version: str
    run_id: str
    elder_id: str
    mode: str
    data: RunSpecData
    features: RunSpecFeatures
    training: RunSpecTraining
    evaluation: RunSpecEvaluation
    gating: RunSpecGating

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RunSpec":
        top_level = _ensure_mapping("run_spec", payload)
        _ensure_exact_keys("run_spec", top_level, _TOP_LEVEL_KEYS)
        run_spec_version = _ensure_nonempty_str("run_spec_version", top_level["run_spec_version"])
        if run_spec_version != RUN_SPEC_VERSION:
            raise ValueError(
                f"Unsupported run_spec_version: {run_spec_version}. Expected {RUN_SPEC_VERSION}"
            )

        return cls(
            run_spec_version=run_spec_version,
            run_id=_ensure_nonempty_str("run_id", top_level["run_id"]),
            elder_id=_ensure_nonempty_str("elder_id", top_level["elder_id"]),
            mode=_ensure_nonempty_str("mode", top_level["mode"]),
            data=RunSpecData.from_dict(_ensure_mapping("data", top_level["data"])),
            features=RunSpecFeatures.from_dict(_ensure_mapping("features", top_level["features"])),
            training=RunSpecTraining.from_dict(_ensure_mapping("training", top_level["training"])),
            evaluation=RunSpecEvaluation.from_dict(
                _ensure_mapping("evaluation", top_level["evaluation"])
            ),
            gating=RunSpecGating.from_dict(_ensure_mapping("gating", top_level["gating"])),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_spec_version": self.run_spec_version,
            "run_id": self.run_id,
            "elder_id": self.elder_id,
            "mode": self.mode,
            "data": self.data.to_dict(),
            "features": self.features.to_dict(),
            "training": self.training.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "gating": self.gating.to_dict(),
        }

    @classmethod
    def schema_descriptor(cls) -> Dict[str, Any]:
        """Schema descriptor used to freeze and publish RunSpec v1 contract hash."""
        return {
            "schema_name": "RunSpec",
            "schema_version": RUN_SPEC_VERSION,
            "hash_policy_version": HASH_POLICY_VERSION,
            "hash_algorithm": HASH_ALGORITHM,
            "hash_excluded_top_level_fields": list(HASH_EXCLUDED_TOP_LEVEL_FIELDS),
            "required_top_level_fields": list(_TOP_LEVEL_KEYS),
            "sections": {
                "data": [
                    "manifest_paths",
                    "time_zone",
                    "max_ffill_gap_seconds",
                    "duplicate_resolution_policy",
                ],
                "features": [
                    "sequence_window_seconds",
                    "stride_seconds",
                    "feature_version",
                ],
                "training": [
                    "architecture_family",
                    "random_seed",
                    "profile",
                    "optimizer",
                    "learning_rate",
                    "epochs",
                ],
                "evaluation.walk_forward": [
                    "lookback_days",
                    "min_train_days",
                    "valid_days",
                    "step_days",
                    "max_folds",
                ],
                "gating": [
                    "room_policy_ref",
                    "run_policy_ref",
                ],
            },
        }

    @classmethod
    def schema_hash(cls) -> str:
        return _hash_payload(cls.schema_descriptor())

    def hash_payload(self) -> Dict[str, Any]:
        payload = self.to_dict()
        for field_name in HASH_EXCLUDED_TOP_LEVEL_FIELDS:
            payload.pop(field_name, None)
        return payload

    def run_spec_hash(self) -> str:
        """Deterministic spec hash using RunSpec hash policy v1."""
        return _hash_payload(self.hash_payload())


def schema_metadata() -> Dict[str, Any]:
    """Published metadata for RunSpec v1 schema freeze."""
    return {
        "schema_name": "RunSpec",
        "schema_version": RUN_SPEC_VERSION,
        "schema_hash": RunSpec.schema_hash(),
        "hash_policy_version": HASH_POLICY_VERSION,
        "hash_algorithm": HASH_ALGORITHM,
        "hash_excluded_top_level_fields": list(HASH_EXCLUDED_TOP_LEVEL_FIELDS),
    }
