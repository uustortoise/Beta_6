from __future__ import annotations

import ast
import tokenize
from pathlib import Path


POLICY_ENV_KEYS = {
    "CLASS_WEIGHT_CAP",
    "CLASS_WEIGHT_FLOOR",
    "CLINICAL_PRIORITY_MULTIPLIERS",
    "UNOCCUPIED_DOWNSAMPLE_MIN_SHARE",
    "UNOCCUPIED_DOWNSAMPLE_STRIDE",
    "UNOCCUPIED_BOUNDARY_KEEP",
    "UNOCCUPIED_MIN_RUN_LENGTH",
    "UNOCCUPIED_DOWNSAMPLE_MIN_SHARE_BY_ROOM",
    "UNOCCUPIED_DOWNSAMPLE_STRIDE_BY_ROOM",
    "UNOCCUPIED_BOUNDARY_KEEP_BY_ROOM",
    "UNOCCUPIED_MIN_RUN_LENGTH_BY_ROOM",
    "ENABLE_MINORITY_CLASS_SAMPLING",
    "MINORITY_TARGET_SHARE",
    "MINORITY_MAX_MULTIPLIER",
    "MINORITY_TARGET_SHARE_BY_ROOM",
    "MINORITY_MAX_MULTIPLIER_BY_ROOM",
    "CALIBRATION_FRACTION_OF_HOLDOUT",
    "CALIBRATION_MIN_SAMPLES",
    "SEPARATE_CALIBRATION_MIN_HOLDOUT",
    "CALIBRATION_MIN_SUPPORT_PER_CLASS",
    "THRESHOLD_FLOOR",
    "THRESHOLD_CAP",
    "DEFAULT_PRECISION_TARGET",
    "DEFAULT_RECALL_FLOOR",
    "PRECISION_TARGETS_BY_LABEL",
    "RECALL_FLOOR_BY_LABEL",
    "MAX_RESAMPLE_FFILL_GAP_SECONDS",
    "RELEASE_GATE_MIN_TRAINING_DAYS",
    "RELEASE_GATE_MIN_SAMPLES",
    "RELEASE_GATE_MIN_CALIBRATION_SUPPORT",
    "RELEASE_GATE_MIN_VALIDATION_CLASS_SUPPORT",
    "RELEASE_GATE_MIN_OBSERVED_DAYS",
    "RELEASE_GATE_MIN_RETAINED_SAMPLE_RATIO",
    "RELEASE_GATE_MAX_DROPPED_RATIO",
    "ALLOW_GATE_CONFIG_FALLBACK_PASS",
    "RELEASE_GATE_BLOCK_ON_LOW_SUPPORT_FALLBACK",
    "RELEASE_GATE_BLOCK_ON_TRAIN_FALLBACK_METRICS",
    "DATA_VIABILITY_MIN_OBSERVED_DAYS",
    "DATA_VIABILITY_MIN_POST_GAP_ROWS",
    "DATA_VIABILITY_MAX_UNRESOLVED_DROP_RATIO",
    "DATA_VIABILITY_MIN_TRAINING_WINDOWS",
    "DATA_VIABILITY_MIN_OBSERVED_DAYS_BY_ROOM",
    "DATA_VIABILITY_MIN_POST_GAP_ROWS_BY_ROOM",
    "DATA_VIABILITY_MAX_UNRESOLVED_DROP_RATIO_BY_ROOM",
    "DATA_VIABILITY_MIN_TRAINING_WINDOWS_BY_ROOM",
    "TRAINING_RANDOM_SEED",
    "SKIP_RETRAIN_IF_SAME_DATA_AND_POLICY",
    "PROMOTION_MIN_TRAINING_DAYS_WITH_CHAMPION",
}


class _EnvReadVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.keys: set[str] = set()

    def _record_literal(self, node: ast.AST | None) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            self.keys.add(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        # os.getenv("KEY")
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "getenv"
            and isinstance(func.value, ast.Name)
            and func.value.id == "os"
            and node.args
        ):
            self._record_literal(node.args[0])
        # os.environ.get("KEY")
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "os"
            and func.value.attr == "environ"
            and node.args
        ):
            self._record_literal(node.args[0])
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # os.environ["KEY"]
        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "os"
            and node.value.attr == "environ"
        ):
            self._record_literal(node.slice)
        self.generic_visit(node)


def _scan_env_reads(py_file: Path) -> set[str]:
    with tokenize.open(py_file) as handle:
        source = handle.read()
    tree = ast.parse(source, filename=str(py_file))
    visitor = _EnvReadVisitor()
    visitor.visit(tree)
    return visitor.keys


def test_policy_env_keys_are_only_read_from_policy_module():
    backend_root = Path(__file__).resolve().parents[1]
    # Explicit allowlist for policy env reads.
    allowed_policy_read_files = {
        backend_root / "ml" / "policy_config.py",
    }

    violations: list[str] = []
    excluded_dirs = {"venv", ".venv", "__pycache__", ".git", ".pytest_cache"}
    files_to_scan = [
        path
        for path in sorted(backend_root.rglob("*.py"))
        if not any(part in excluded_dirs for part in path.parts)
    ]

    for py_file in files_to_scan:
        if py_file in allowed_policy_read_files:
            continue
        keys = _scan_env_reads(py_file)
        forbidden = sorted(k for k in keys if k in POLICY_ENV_KEYS)
        if forbidden:
            violations.append(f"{py_file}: {forbidden}")

    assert not violations, (
        "Policy SSoT violation: policy env key read outside allowlist "
        f"{sorted(str(p) for p in allowed_policy_read_files)}\n"
        + "\n".join(violations)
    )
