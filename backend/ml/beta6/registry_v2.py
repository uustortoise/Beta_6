"""DEPRECATED shim: import from ml.beta6.registry.registry_v2.

Removal target: 2026-04-30 (end of Phase 5).
"""

from .registry import registry_v2 as _impl
import sys as _sys

_sys.modules[__name__] = _impl
