"""DEPRECATED compatibility shim for ml.beta6.gates.timeline_hard_gates.

Removal target: 2026-04-30 (end of Phase 5).
"""

from .gates import timeline_hard_gates as _impl
import sys as _sys

# Keep old imports bit-equivalent while the Beta 6.2 canonical surface settles.
_sys.modules[__name__] = _impl
