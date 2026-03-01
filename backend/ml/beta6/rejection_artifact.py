"""DEPRECATED shim: import from ml.beta6.registry.rejection_artifact.

Removal target: 2026-04-30 (end of Phase 5).
"""

from .registry import rejection_artifact as _impl
import sys as _sys

_sys.modules[__name__] = _impl
