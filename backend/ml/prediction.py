"""DEPRECATED compatibility wrapper.

Use `ml.legacy.prediction` for legacy behavior.
For new Beta 6 development, use `ml.beta6.serving.prediction`.
"""

from .legacy import prediction as _impl
import sys as _sys

_sys.modules[__name__] = _impl
