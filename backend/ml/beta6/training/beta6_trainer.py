#!/usr/bin/env python3
"""
Authoritative Beta 6.2 training entrypoint.

Beta 6 Universal Backbone Trainer.

This module provides the training pipeline for the Universal Backbone model,
which learns from Golden Samples collected across multiple residents.

Usage:
    python beta6_trainer.py --dataset ./data/golden_samples/golden_dataset_*.json
    python beta6_trainer.py --dataset ./data/golden_samples/ --epochs 10
    python beta6_trainer.py --validate-only  # Load and validate without training

Architecture:
    - Uses TransformerBackbone from ml/transformer_backbone.py
    - Input: (seq_length, num_features) sensor windows
    - Output: Activity classification
    
The trained backbone can be frozen and used with resident-specific heads in production.
"""

BETA62_SURFACE_NAME = "trainer"
BETA62_CANONICAL_IMPORT = "ml.beta6.training.beta6_trainer"
BETA62_COMPAT_SHIMS = ("ml.beta6.beta6_trainer",)

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Mapping
import numpy as np

# Path setup
_script_dir = Path(__file__).resolve().parent
_ml_dir = _script_dir.parent
_backend_dir = _ml_dir.parent
sys.path.insert(0, str(_backend_dir))

from elderlycare_v1_16.config.settings import (
    PROJECT_ROOT, MODELS_DIR, DEFAULT_SENSOR_COLUMNS,
    DEFAULT_EPOCHS
)
from ml.beta6.sequence.transition_builder import (
    TransitionPolicy,
    resolve_transition_policy_for_context,
)
from ml.beta6.data.data_manifest import load_manifest
from ml.beta6.data.feature_fingerprint import hash_json_payload
from ml.beta6.feature_store import build_feature_sequence_cache_key
from ml.home_empty_fusion import ResidentHomeContext
from ml.household_analyzer import (
    build_resident_home_context_contract,
    identify_deferred_demographic_fields,
)
from ml.timeline_targets import build_event_native_targets
from ml.beta6.gates.intake_precheck import IntakeGateBlockedError, enforce_approved_intake_artifact
from ml.beta6.training.fine_tune_safe_classes import run_safe_class_finetune
from ml.beta6.training.self_supervised_pretrain import (
    build_pretrain_policy_fingerprint,
    load_corpus_matrix,
    load_pretrain_config,
    run_self_supervised_pretraining,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_EXCLUDED_DEMOGRAPHIC_FIELDS = ("age", "gender", "sex")


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _select_resident_home_context_payload(
    *,
    profile_context: Mapping[str, Any] | None = None,
    resident_home_context: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    if isinstance(resident_home_context, Mapping):
        return resident_home_context
    profile = _as_mapping(profile_context)
    nested = profile.get("resident_home_context")
    if isinstance(nested, Mapping):
        return nested
    return profile


# =============================================================================
# Data Loading
# =============================================================================

def load_golden_dataset(path: Path) -> Dict[str, Any]:
    """Load a single golden dataset JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded dataset: {path.name}")
    logger.info(f"  Samples: {data['metadata']['sample_count']}")
    logger.info(f"  Elders: {data['metadata']['unique_elders']}")
    logger.info(f"  Activities: {data['metadata']['activities']}")
    
    return data


def load_all_datasets(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load all golden datasets from a directory or single file."""
    datasets = []
    
    if dataset_path.is_file():
        datasets.append(load_golden_dataset(dataset_path))
    elif dataset_path.is_dir():
        for json_file in sorted(dataset_path.glob('golden_dataset_*.json')):
            datasets.append(load_golden_dataset(json_file))
    else:
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    if not datasets:
        raise ValueError(f"No golden datasets found in: {dataset_path}")
    
    return datasets


def prepare_training_data(
    datasets: List[Dict[str, Any]],
    label_map: Optional[Dict[str, int]] = None,
    seq_length: int = 60
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Prepare training data from loaded datasets.
    
    Supports two formats:
    1. Sequence windows: sample['sensor_window'] = list of seq_length timesteps
    2. Legacy single-timestep: sample['sensor_features'] = single timestep dict
    
    Returns:
        X: Feature array (num_samples, seq_length, num_features) for sequence data
           OR (num_samples, num_features) for legacy single-timestep
        y: Label array (num_samples,)
        label_map: Mapping from activity name to integer
    """
    all_samples = []
    for dataset in datasets:
        all_samples.extend(dataset['samples'])
    
    if not all_samples:
        raise ValueError("No samples found in datasets")
    
    # Build label map if not provided
    if label_map is None:
        unique_activities = sorted(set(s['activity'] for s in all_samples))
        label_map = {act: i for i, act in enumerate(unique_activities)}
    
    # Detect format: sequence window vs single timestep
    first_sample = all_samples[0]
    is_sequence_format = 'sensor_window' in first_sample
    
    X_list = []
    y_list = []
    skipped = 0
    
    for sample in all_samples:
        activity = sample['activity']
        if activity not in label_map:
            skipped += 1
            continue
        
        if is_sequence_format:
            # New format: sequence window (list of timestep dicts)
            sensor_window = sample.get('sensor_window', [])
            if len(sensor_window) < seq_length * 0.8:
                skipped += 1
                continue
            
            # Build sequence array
            sequence = []
            for timestep in sensor_window[-seq_length:]:  # Take last seq_length
                feature_vec = []
                for col in DEFAULT_SENSOR_COLUMNS:
                    val = timestep.get(col, 0.0)
                    if val is None:
                        val = 0.0
                    feature_vec.append(float(val))
                sequence.append(feature_vec)
            
            # Pad if needed
            while len(sequence) < seq_length:
                sequence.insert(0, sequence[0] if sequence else [0.0] * len(DEFAULT_SENSOR_COLUMNS))
            
            X_list.append(sequence)
        else:
            # Legacy format: single timestep
            sensor_features = sample['sensor_features']
            feature_vec = []
            for col in DEFAULT_SENSOR_COLUMNS:
                val = sensor_features.get(col, 0.0)
                if val is None:
                    val = 0.0
                feature_vec.append(float(val))
            X_list.append(feature_vec)
        
        y_list.append(label_map[activity])
    
    if skipped > 0:
        logger.warning(f"Skipped {skipped} samples with unknown activities or insufficient data")
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    format_type = "sequence windows" if is_sequence_format else "single timesteps"
    logger.info(f"Prepared {len(X)} {format_type} with {len(label_map)} classes, shape={X.shape}")
    
    return X, y, label_map


def build_event_native_supervision_bundle(
    *,
    timestamps: np.ndarray,
    labels: np.ndarray,
    room_name: str,
    min_episode_duration_seconds: float = 30.0,
) -> Dict[str, Any]:
    """
    Build offline event-native supervision targets for Beta 6.2 experiments.
    """
    event_targets = build_event_native_targets(
        timestamps=timestamps,
        labels=labels,
        room_name=room_name,
        min_episode_duration_seconds=min_episode_duration_seconds,
    )
    return {
        "boundary_start_labels": event_targets.boundary_targets.start_flags.astype(np.float32),
        "boundary_end_labels": event_targets.boundary_targets.end_flags.astype(np.float32),
        "continuity_labels": event_targets.continuity_flags.astype(np.float32),
        "event_duration_minutes": event_targets.duration_targets_minutes.astype(np.float32),
        "daily_duration": float(event_targets.daily_duration_minutes),
        "daily_count": float(event_targets.daily_episode_count),
        "timestamps": event_targets.timestamps,
    }


def build_context_conditioning_bundle(
    *,
    room_name: str,
    profile_context: Mapping[str, Any] | None = None,
    resident_home_context: Mapping[str, Any] | None = None,
    base_transition_policy: TransitionPolicy = TransitionPolicy(),
) -> Dict[str, Any]:
    """
    Build offline-only context conditioning features for Beta 6.2 experiments.

    This path intentionally excludes demographic metadata from default model inputs.
    """
    raw_profile = _as_mapping(profile_context)
    context_payload = _select_resident_home_context_payload(
        profile_context=raw_profile,
        resident_home_context=resident_home_context,
    )
    contract = build_resident_home_context_contract(
        profile_context=context_payload,
        config_context={},
    )
    typed_context = ResidentHomeContext.from_payload(contract)
    runtime_payload = typed_context.to_runtime_payload()
    feature_values = typed_context.to_model_feature_payload(room_name=room_name)
    feature_names = list(feature_values.keys())
    effective_policy = resolve_transition_policy_for_context(
        policy=base_transition_policy,
        room_name=room_name,
        resident_home_context=runtime_payload,
    )
    demographic_fields_present = list(identify_deferred_demographic_fields(raw_profile))
    excluded_fields = [
        field for field in DEFAULT_EXCLUDED_DEMOGRAPHIC_FIELDS if field in demographic_fields_present
    ]
    return {
        "room_name": str(room_name or "").strip().lower(),
        "offline_only": True,
        "context_status": typed_context.status,
        "context_contract_status": str(contract.get("status") or typed_context.status),
        "resident_home_context": runtime_payload,
        "context_feature_names": feature_names,
        "context_feature_values": feature_values,
        "context_features": [float(feature_values[name]) for name in feature_names],
        "effective_transition_policy": {
            "switch_penalty": float(effective_policy.switch_penalty),
            "self_transition_bias": float(effective_policy.self_transition_bias),
            "impossible_transition_penalty": float(effective_policy.impossible_transition_penalty),
            "step_minutes": float(effective_policy.step_minutes),
        },
        "demographic_fields_present": demographic_fields_present,
        "excluded_default_input_fields": excluded_fields,
        "uses_demographic_inputs": False,
    }


def build_pretrain_cache_context(
    *,
    manifest_path: str | Path,
    config_path: str | Path | None,
    max_files: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve the canonical pretrain cache key from manifest + policy fingerprints."""
    manifest = load_manifest(manifest_path)
    manifest_fingerprint = str(
        _as_mapping(manifest.get("fingerprint")).get("value")
        or hash_json_payload({"entries": manifest.get("entries", [])})
    )
    config = load_pretrain_config(config_path)
    policy_fingerprint = build_pretrain_policy_fingerprint(config, max_files=max_files)
    cache_key = build_feature_sequence_cache_key(
        manifest_fingerprint=manifest_fingerprint,
        policy_fingerprint=policy_fingerprint,
        stage="pretrain_matrix",
        extra={"max_files": None if max_files is None else int(max_files)},
    )
    return {
        "manifest_fingerprint": manifest_fingerprint,
        "policy_fingerprint": policy_fingerprint,
        "cache_key": cache_key,
        "cache_dir": str(Path(manifest_path).resolve().parent / ".beta6_cache"),
    }


# =============================================================================
# Model Building
# =============================================================================

def build_backbone_model(
    num_features: int,
    num_classes: int,
    seq_length: int = 60
) -> 'tf.keras.Model':
    """
    Build the Universal Backbone model using Hybrid CNN-Transformer.
    
    This uses the same architecture as Beta 5.5's per-resident models,
    enabling transfer learning and fast head adaptation.
    
    Args:
        num_features: Number of sensor features per timestep
        num_classes: Number of activity classes
        seq_length: Sequence length (timesteps per window)
        
    Returns:
        Compiled Keras model with Transformer architecture
    """
    # Import the actual Transformer builder
    import sys
    _script_dir = Path(__file__).resolve().parent
    _ml_dir = _script_dir.parent
    if str(_ml_dir) not in sys.path:
        sys.path.insert(0, str(_ml_dir))
    
    from transformer_backbone import build_transformer_model
    
    model = build_transformer_model(
        input_shape=(seq_length, num_features),
        num_classes=num_classes,
        d_model=64,
        num_heads=4,
        ff_dim=128,
        num_transformer_blocks=2,
        dropout_rate=0.2,
        positional_encoding_type='sinusoidal',
        use_cnn_embedding=True,
        use_alibi=False
    )
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info(f"Built Transformer backbone: input=({seq_length}, {num_features}), classes={num_classes}")
    
    return model


# =============================================================================
# Training
# =============================================================================

def train_backbone(
    X: np.ndarray,
    y: np.ndarray,
    label_map: Dict[str, int],
    epochs: int = DEFAULT_EPOCHS,
    validation_split: float = 0.2,
    output_dir: Optional[Path] = None,
    seq_length: int = 60
) -> 'tf.keras.Model':
    """
    Train the Universal Backbone model.
    
    Args:
        X: Feature array - either (num_samples, seq_length, num_features) for sequence
           or (num_samples, num_features) for single-timestep (legacy)
        y: Label array
        label_map: Activity to integer mapping
        epochs: Number of training epochs
        validation_split: Fraction for validation
        output_dir: Where to save the trained model
        seq_length: Sequence length (used if X is 3D)
        
    Returns:
        Trained Keras model
    """
    num_classes = len(label_map)
    
    # Detect input shape
    if len(X.shape) == 3:
        # Sequence format: (samples, seq_length, features)
        seq_len, num_features = X.shape[1], X.shape[2]
        logger.info(f"Using sequence format: {X.shape}")
    else:
        # Legacy single-timestep: (samples, features)
        # Reshape to (samples, 1, features) for Transformer
        num_features = X.shape[1]
        seq_len = 1
        X = X.reshape(-1, 1, num_features)
        logger.warning(f"Legacy format detected, reshaping to: {X.shape}")
    
    model = build_backbone_model(num_features, num_classes, seq_length=seq_len)
    
    logger.info(f"Starting training: {len(X)} samples, {epochs} epochs")
    
    history = model.fit(
        X, y,
        epochs=epochs,
        validation_split=validation_split,
        batch_size=32,
        verbose=1
    )
    
    # Log final metrics
    final_acc = history.history['accuracy'][-1]
    final_val_acc = history.history.get('val_accuracy', [0])[-1]
    logger.info(f"Training complete: acc={final_acc:.4f}, val_acc={final_val_acc:.4f}")
    
    # Save model
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = output_dir / 'universal_backbone.keras'
        model.save(model_path)
        logger.info(f"Saved backbone model: {model_path}")
        
        # Save label map
        label_map_path = output_dir / 'label_map.json'
        with open(label_map_path, 'w') as f:
            json.dump(label_map, f, indent=2)
        logger.info(f"Saved label map: {label_map_path}")
    
    return model


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Universal Backbone for Beta 6 from Golden Samples"
    )
    parser.add_argument(
        '--dataset', '-d',
        type=Path,
        required=False,
        help='Path to golden_dataset JSON file or directory containing them'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=MODELS_DIR / 'backbone',
        help='Output directory for trained model (default: backend/models/backbone/)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=DEFAULT_EPOCHS,
        help=f'Number of training epochs (default: {DEFAULT_EPOCHS})'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate dataset without training'
    )
    parser.add_argument(
        '--intake-artifact',
        type=Path,
        default=None,
        help='Approved Beta 6 intake artifact path (required for non-validate runs)'
    )
    parser.add_argument(
        '--pretrain-manifest',
        type=Path,
        default=None,
        help='Pretraining manifest JSON path (Phase 2 self-supervised mode)'
    )
    parser.add_argument(
        '--pretrain-config',
        type=Path,
        default=PROJECT_ROOT / 'backend' / 'config' / 'beta6_pretrain.yaml',
        help='Pretraining config YAML path'
    )
    parser.add_argument(
        '--pretrain-output',
        type=Path,
        default=MODELS_DIR / 'beta6_pretrain',
        help='Output directory for self-supervised pretrain checkpoints'
    )
    parser.add_argument(
        '--pretrain-max-files',
        type=int,
        default=None,
        help='Optional cap on number of manifest files used during pretraining'
    )
    parser.add_argument(
        '--safe-finetune-dataset',
        type=Path,
        default=None,
        help='Golden dataset file/dir for Phase 3 safe-class fine-tuning'
    )
    parser.add_argument(
        '--safe-finetune-config',
        type=Path,
        default=PROJECT_ROOT / 'backend' / 'config' / 'beta6_golden_safe_finetune.yaml',
        help='Safe-class fine-tune config YAML path'
    )
    parser.add_argument(
        '--safe-finetune-output',
        type=Path,
        default=MODELS_DIR / 'beta6_safe_finetune',
        help='Output directory for safe-class fine-tune artifacts'
    )
    parser.add_argument(
        '--safe-finetune-pretrain-checkpoint',
        type=Path,
        default=None,
        help='Optional pretrain checkpoint NPZ for embedding transfer'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("BETA 6 UNIVERSAL BACKBONE TRAINER")
    logger.info("=" * 60)

    is_pretrain = args.pretrain_manifest is not None
    is_safe_finetune = args.safe_finetune_dataset is not None
    selected_modes = int(is_pretrain) + int(is_safe_finetune)
    if selected_modes > 1:
        logger.error("Only one mode may be selected: --pretrain-manifest or --safe-finetune-dataset")
        return 2
    if args.dataset is None and not is_pretrain and not is_safe_finetune:
        logger.error("Provide one mode: --dataset, --pretrain-manifest, or --safe-finetune-dataset")
        return 2

    if not args.validate_only:
        intake_artifact = args.intake_artifact
        if intake_artifact is None:
            env_path = os.environ.get("BETA6_INTAKE_ARTIFACT")
            if env_path:
                intake_artifact = Path(env_path)
        if intake_artifact is None:
            logger.error(
                "Blocked by intake gate: reason_code=intake_gate_missing_artifact "
                "(set --intake-artifact or BETA6_INTAKE_ARTIFACT)"
            )
            return 2
        try:
            artifact = enforce_approved_intake_artifact(intake_artifact)
        except IntakeGateBlockedError as exc:
            logger.error(
                "Blocked by intake gate: reason_code=%s detail=%s",
                exc.reason_code,
                exc.detail,
            )
            return 2
        logger.info(
            "Intake gate approved: artifact=%s generated_at=%s",
            Path(intake_artifact).resolve(),
            artifact.get("generated_at"),
        )

    if is_pretrain:
        manifest_path = Path(args.pretrain_manifest).resolve()
        if args.validate_only:
            matrix = load_corpus_matrix(
                manifest_path,
                max_files=args.pretrain_max_files,
            )
            logger.info(
                "Pretrain manifest validation complete: rows=%s features=%s",
                matrix.shape[0],
                matrix.shape[1],
            )
            return 0

        result = run_self_supervised_pretraining(
            manifest_path=manifest_path,
            config_path=args.pretrain_config,
            output_dir=args.pretrain_output,
            max_files=args.pretrain_max_files,
        )
        logger.info(
            "Pretraining complete: checkpoint=%s mse=%.6f",
            result.get("artifacts", {}).get("checkpoint_npz"),
            float(result.get("metrics", {}).get("final_reconstruction_mse", 0.0)),
        )
        return 0

    if is_safe_finetune:
        dataset_path = Path(args.safe_finetune_dataset).resolve()
        if args.validate_only:
            datasets = load_all_datasets(dataset_path)
            X, y, label_map = prepare_training_data(datasets)
            logger.info(
                "Safe fine-tune dataset validation complete: rows=%s classes=%s",
                X.shape[0],
                len(label_map),
            )
            return 0
        report = run_safe_class_finetune(
            dataset_path=dataset_path,
            config_path=args.safe_finetune_config,
            output_dir=args.safe_finetune_output,
            pretrain_checkpoint=args.safe_finetune_pretrain_checkpoint,
        )
        logger.info(
            "Safe fine-tune complete: status=%s heldout_accuracy=%.4f report=%s",
            report.get("status"),
            float(report.get("metrics", {}).get("heldout_accuracy", 0.0)),
            report.get("artifacts", {}).get("report_json"),
        )
        return 0 if str(report.get("status")) == "pass" else 2

    # Load datasets
    datasets = load_all_datasets(args.dataset)
    
    # Prepare data
    X, y, label_map = prepare_training_data(datasets)
    
    if args.validate_only:
        logger.info("Validation complete. Exiting without training.")
        return 0
    
    # Train
    model = train_backbone(
        X, y, label_map,
        epochs=args.epochs,
        output_dir=args.output
    )
    
    logger.info("=" * 60)
    logger.info("✅ Beta 6 Backbone training complete!")
    logger.info("=" * 60)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
