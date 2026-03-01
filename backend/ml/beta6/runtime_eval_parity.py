"""DEPRECATED shim: import from ml.beta6.evaluation.runtime_eval_parity.

Removal target: 2026-04-30 (end of Phase 5).
"""

from .evaluation import runtime_eval_parity as _impl
import sys as _sys

_sys.modules[__name__] = _impl
