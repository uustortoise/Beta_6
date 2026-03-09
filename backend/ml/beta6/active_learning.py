"""DEPRECATED shim: import from ml.beta6.training.active_learning.

Removal target: 2026-04-30 (end of Phase 5).
"""

from .training import active_learning as _impl
import sys as _sys

BETA62_AUTHORITATIVE_MODULE = "ml.beta6.training.active_learning"
BETA62_SHIM_TARGET = BETA62_AUTHORITATIVE_MODULE
BETA62_SHIM_DEPRECATED = True
_impl.BETA62_AUTHORITATIVE_MODULE = BETA62_AUTHORITATIVE_MODULE
_impl.BETA62_SHIM_TARGET = BETA62_SHIM_TARGET
_impl.BETA62_SHIM_DEPRECATED = BETA62_SHIM_DEPRECATED
_impl.BETA62_SHIM_ALIAS = "ml.beta6.active_learning"

_sys.modules[__name__] = _impl
