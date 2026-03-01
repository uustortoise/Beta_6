"""
ICOPE (Integrated Care for Older People) Module
Aging assessment framework based on WHO ICOPE guidelines
"""

from .icope_scoring import ICOPE_Scoring
from .icope_visualizer import ICOPE_Visualizer, display_icope_dashboard
from .icope_interpreter import ICOPE_Interpreter
from .icope_processor import ICOPE_Processor, test_icope_processor

__version__ = "1.0.0"
__author__ = "Elderly Care System"
__description__ = "ICOPE-based aging assessment module"

__all__ = [
    "ICOPE_Scoring",
    "ICOPE_Visualizer",
    "ICOPE_Interpreter",
    "ICOPE_Processor",
    "display_icope_dashboard",
    "test_icope_processor"
]
