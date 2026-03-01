"""DEPRECATED shim: import from ml.beta6.data.feature_fingerprint.

Removal target: 2026-04-30 (end of Phase 5).
"""

from .data import feature_fingerprint as _impl
import sys as _sys

_sys.modules[__name__] = _impl
