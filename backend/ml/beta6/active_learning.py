"""DEPRECATED shim: import from ml.beta6.training.active_learning.

Removal target: 2026-04-30 (end of Phase 5).
"""

from .training import active_learning as _impl
import sys as _sys

_sys.modules[__name__] = _impl
