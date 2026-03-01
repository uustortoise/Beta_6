"""DEPRECATED compatibility wrapper.

Use `ml.legacy.rejection_artifact` for legacy behavior.
For new Beta 6 development, use `ml.beta6.registry.rejection_artifact`.
"""

from .legacy import rejection_artifact as _impl
import sys as _sys

_sys.modules[__name__] = _impl
