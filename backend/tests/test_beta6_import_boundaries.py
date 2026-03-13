import ast
from pathlib import Path


BETA6_ROOT = Path(__file__).resolve().parent.parent / "ml" / "beta6"
FORBIDDEN_LEGACY_IMPORT_PREFIXES = (
    "ml.prediction",
    "ml.rejection_artifact",
    "ml.calibration",
    "ml.registry",
)

# Path -> fully-qualified modules allowed despite forbidden prefix rules.
BETA6_APPROVED_LEGACY_IMPORTS: dict[str, set[str]] = {}
EXPECTED_SSOT_NAMESPACE = {
    "trainer": {
        "canonical_path": Path("training/beta6_trainer.py"),
        "shim_path": Path("beta6_trainer.py"),
        "canonical_import": "ml.beta6.training.beta6_trainer",
        "compat_shims": ("ml.beta6.beta6_trainer",),
        "shim_import_module": "training",
        "shim_import_name": "beta6_trainer",
    },
    "registry_v2": {
        "canonical_path": Path("registry/registry_v2.py"),
        "shim_path": Path("registry_v2.py"),
        "canonical_import": "ml.beta6.registry.registry_v2",
        "compat_shims": ("ml.beta6.registry_v2",),
        "shim_import_module": "registry",
        "shim_import_name": "registry_v2",
    },
    "timeline_hard_gates": {
        "canonical_path": Path("gates/timeline_hard_gates.py"),
        "shim_path": Path("timeline_hard_gates.py"),
        "canonical_import": "ml.beta6.gates.timeline_hard_gates",
        "compat_shims": ("ml.beta6.timeline_hard_gates",),
        "shim_import_module": "gates",
        "shim_import_name": "timeline_hard_gates",
    },
}


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


def _literal_assignments(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assignments: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        try:
            assignments[node.targets[0].id] = ast.literal_eval(node.value)
        except Exception:
            continue
    return assignments


def _docstring(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return ast.get_docstring(tree) or ""


def _has_shim_import(path: Path, module_name: str, import_name: str) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != module_name or node.level != 1:
            continue
        for alias in node.names:
            if alias.name == import_name and alias.asname == "_impl":
                return True
    return False


def _has_sys_modules_alias(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Subscript):
            continue
        if not (
            isinstance(target.value, ast.Attribute)
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "_sys"
            and target.value.attr == "modules"
        ):
            continue
        slice_node = target.slice
        if isinstance(slice_node, ast.Name) and slice_node.id == "__name__":
            pass
        elif (
            isinstance(slice_node, ast.Constant)
            and slice_node.value == "__name__"
        ):
            pass
        else:
            continue
        if isinstance(node.value, ast.Name) and node.value.id == "_impl":
            return True
    return False


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
    contract = EXPECTED_SSOT_NAMESPACE["trainer"]
    canonical_path = BETA6_ROOT / contract["canonical_path"]
    shim_path = BETA6_ROOT / contract["shim_path"]

    assignments = _literal_assignments(canonical_path)
    assert assignments["BETA62_SURFACE_NAME"] == "trainer"
    assert assignments["BETA62_CANONICAL_IMPORT"] == contract["canonical_import"]
    assert tuple(assignments["BETA62_COMPAT_SHIMS"]) == contract["compat_shims"]

    shim_docstring = _docstring(shim_path)
    assert contract["canonical_import"] in shim_docstring
    assert _has_shim_import(
        shim_path,
        str(contract["shim_import_module"]),
        str(contract["shim_import_name"]),
    )
    assert _has_sys_modules_alias(shim_path)


def test_duplicate_registry_and_gate_paths_are_explicit_shims_or_blocked():
    for surface_name in ("registry_v2", "timeline_hard_gates"):
        contract = EXPECTED_SSOT_NAMESPACE[surface_name]
        canonical_path = BETA6_ROOT / contract["canonical_path"]
        shim_path = BETA6_ROOT / contract["shim_path"]

        assignments = _literal_assignments(canonical_path)
        assert assignments["BETA62_SURFACE_NAME"] == surface_name
        assert assignments["BETA62_CANONICAL_IMPORT"] == contract["canonical_import"]
        assert tuple(assignments["BETA62_COMPAT_SHIMS"]) == contract["compat_shims"]

        shim_docstring = _docstring(shim_path)
        assert contract["canonical_import"] in shim_docstring
        assert _has_shim_import(
            shim_path,
            str(contract["shim_import_module"]),
            str(contract["shim_import_name"]),
        )
        assert _has_sys_modules_alias(shim_path)
