"""DEPRECATED shim: import from ml.beta6.registry.gate_engine.

Removal target: 2026-04-30 (end of Phase 5).
"""

from .registry import gate_engine as _impl
import sys as _sys

_sys.modules[__name__] = _impl
