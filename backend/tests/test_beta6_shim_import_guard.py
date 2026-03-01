import ast
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent.parent
SHIM_MODULE_NAMES = {
    "active_learning",
    "beta6_trainer",
    "calibration",
    "capability_profiles",
    "data_manifest",
    "evaluation_engine",
    "feature_fingerprint",
    "feature_store",
    "fine_tune_safe_classes",
    "gate_engine",
    "head_factory",
    "intake_gate",
    "intake_precheck",
    "prediction",
    "registry_events",
    "registry_v2",
    "rejection_artifact",
    "representation_eval",
    "runtime_eval_parity",
    "self_supervised_pretrain",
    "slo_observability",
    "timeline_hard_gates",
}
SHIM_FULL_MODULES = {f"ml.beta6.{name}" for name in SHIM_MODULE_NAMES}

# Temporary allowance for migration period only.
ALLOWED_IMPORTER_PREFIXES = (
    "tests/",
    "scripts/",
)


def _extract_shim_imports(py_path: Path) -> list[str]:
    source = py_path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(source, filename=str(py_path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = str(alias.name)
                if module in SHIM_FULL_MODULES:
                    hits.append(module)
        elif isinstance(node, ast.ImportFrom):
            module = str(node.module or "")
            if module in SHIM_FULL_MODULES:
                hits.append(module)
                continue
            if module == "ml.beta6":
                for alias in node.names:
                    name = str(alias.name)
                    if name in SHIM_MODULE_NAMES:
                        hits.append(f"ml.beta6.{name}")
    return hits


def test_no_new_non_test_imports_from_beta6_shim_paths():
    violations: list[str] = []
    for py_path in sorted(BACKEND_ROOT.rglob("*.py")):
        rel = py_path.relative_to(BACKEND_ROOT).as_posix()
        if rel.startswith("venv/"):
            continue
        if rel.startswith("ml/beta6/") and py_path.name in {f"{name}.py" for name in SHIM_MODULE_NAMES}:
            continue
        shim_imports = _extract_shim_imports(py_path)
        if not shim_imports:
            continue
        if rel.startswith(ALLOWED_IMPORTER_PREFIXES):
            continue
        violations.append(f"{rel}: {', '.join(sorted(set(shim_imports)))}")

    assert not violations, (
        "Non-test/script imports from deprecated ml.beta6 shim paths are blocked:\n"
        + "\n".join(violations)
    )
