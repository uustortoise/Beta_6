"""DEPRECATED shim: import from ml.beta6.serving.head_factory.

Removal target: 2026-04-30 (end of Phase 5).
"""

from .serving import head_factory as _impl
import sys as _sys

_sys.modules[__name__] = _impl
