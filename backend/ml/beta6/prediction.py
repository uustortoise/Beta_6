"""DEPRECATED shim: import from ml.beta6.serving.prediction.

Removal target: 2026-04-30 (end of Phase 5).
"""

from .serving import prediction as _impl
import sys as _sys

_sys.modules[__name__] = _impl
