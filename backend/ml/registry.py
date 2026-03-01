"""DEPRECATED compatibility wrapper.

Use `ml.legacy.registry` for legacy behavior.
For new Beta 6 development, use `ml.beta6.registry.registry_v2`.
"""

from .legacy import registry as _impl
import sys as _sys

_sys.modules[__name__] = _impl
