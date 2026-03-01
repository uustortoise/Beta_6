"""DEPRECATED shim: import from ml.beta6.evaluation.slo_observability.

Removal target: 2026-04-30 (end of Phase 5).
"""

from .evaluation import slo_observability as _impl
import sys as _sys

_sys.modules[__name__] = _impl
