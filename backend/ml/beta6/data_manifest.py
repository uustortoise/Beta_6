"""DEPRECATED shim: import from ml.beta6.data.data_manifest.

Removal target: 2026-04-30 (end of Phase 5).
"""

from .data import data_manifest as _impl
import sys as _sys

BETA62_AUTHORITATIVE_MODULE = "ml.beta6.data.data_manifest"
BETA62_SHIM_TARGET = BETA62_AUTHORITATIVE_MODULE
BETA62_SHIM_DEPRECATED = True

_sys.modules[__name__] = _impl
