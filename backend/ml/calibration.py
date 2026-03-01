"""DEPRECATED compatibility wrapper.

Use `ml.legacy.calibration` for legacy behavior.
For new Beta 6 development, use `ml.beta6.evaluation.calibration`.
"""

from .legacy import calibration as _impl
import sys as _sys

_sys.modules[__name__] = _impl
