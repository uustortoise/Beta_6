"""DEPRECATED shim: import from ml.beta6.data.data_manifest.

Removal target: 2026-04-30 (end of Phase 5).
"""

from .data import data_manifest as _impl
import sys as _sys

_sys.modules[__name__] = _impl
