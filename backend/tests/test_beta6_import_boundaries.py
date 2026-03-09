import ast
from pathlib import Path
import importlib


BETA6_ROOT = Path(__file__).resolve().parent.parent / "ml" / "beta6"
FORBIDDEN_LEGACY_IMPORT_PREFIXES = (
    "ml.prediction",
    "ml.rejection_artifact",
    "ml.calibration",
    "ml.registry",
)

# Path -> fully-qualified modules allowed despite forbidden prefix rules.
BETA6_APPROVED_LEGACY_IMPORTS: dict[str, set[str]] = {}


def _is_forbidden(module_name: str) -> bool:
    return any(
        module_name == prefix or module_name.startswith(prefix + ".")
        for prefix in FORBIDDEN_LEGACY_IMPORT_PREFIXES
    )


def _collect_import_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(str(alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(str(node.module))
    return modules


def test_beta6_import_boundaries_block_legacy_direct_imports():
    violations: list[str] = []
    for py_path in sorted(BETA6_ROOT.rglob("*.py")):
        if "__pycache__" in py_path.parts:
            continue
        rel = str(py_path.relative_to(BETA6_ROOT))
        approved = BETA6_APPROVED_LEGACY_IMPORTS.get(rel, set())
        for module_name in _collect_import_modules(py_path):
            if not _is_forbidden(module_name):
                continue
            if module_name in approved:
                continue
            violations.append(f"{rel}: {module_name}")

    assert not violations, "Forbidden legacy imports found in ml.beta6:\\n" + "\\n".join(violations)


def test_beta62_training_namespace_has_single_authoritative_entrypoint():
    shim = importlib.import_module("ml.beta6.beta6_trainer")
    impl = importlib.import_module("ml.beta6.training.beta6_trainer")

    assert getattr(shim, "BETA62_AUTHORITATIVE_MODULE") == "ml.beta6.training.beta6_trainer"
    assert getattr(shim, "BETA62_SHIM_DEPRECATED") is True
    assert getattr(impl, "BETA62_AUTHORITATIVE_MODULE") == "ml.beta6.training.beta6_trainer"
    assert getattr(impl, "BETA62_MODULE_SURFACE") == "training"


def test_duplicate_registry_and_gate_paths_are_explicit_shims_or_blocked():
    registry_shim = importlib.import_module("ml.beta6.registry_v2")
    registry_impl = importlib.import_module("ml.beta6.registry.registry_v2")
    gates_shim = importlib.import_module("ml.beta6.timeline_hard_gates")
    gates_impl = importlib.import_module("ml.beta6.gates.timeline_hard_gates")

    assert getattr(registry_shim, "BETA62_SHIM_TARGET") == "ml.beta6.registry.registry_v2"
    assert getattr(registry_shim, "BETA62_SHIM_DEPRECATED") is True
    assert getattr(registry_impl, "BETA62_AUTHORITATIVE_MODULE") == "ml.beta6.registry.registry_v2"

    assert getattr(gates_shim, "BETA62_SHIM_TARGET") == "ml.beta6.gates.timeline_hard_gates"
    assert getattr(gates_shim, "BETA62_SHIM_DEPRECATED") is True
    assert getattr(gates_impl, "BETA62_AUTHORITATIVE_MODULE") == "ml.beta6.gates.timeline_hard_gates"
