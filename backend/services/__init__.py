"""Service package bootstrap helpers."""

import importlib
import sys


def _ensure_backend_utils_alias() -> None:
    """Map legacy `utils.*` imports to `backend.utils` when shadowed."""
    existing = sys.modules.get("utils")
    if existing is not None and getattr(existing, "__path__", None):
        return
    try:
        sys.modules["utils"] = importlib.import_module("backend.utils")
    except Exception:
        # Keep import behavior unchanged if backend package is unavailable.
        pass


_ensure_backend_utils_alias()
