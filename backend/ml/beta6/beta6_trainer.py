"""DEPRECATED compatibility shim for ml.beta6.training.beta6_trainer.

Removal target: 2026-04-30 (end of Phase 5).
"""

from .training import beta6_trainer as _impl
import sys as _sys

# Keep old imports bit-equivalent while the Beta 6.2 canonical surface settles.
_sys.modules[__name__] = _impl
