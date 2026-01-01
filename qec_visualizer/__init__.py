"""
Quantum Error Correction Visualization Tool

An interactive, educational tool for visualizing quantum error correction codes.
"""

__version__ = "1.0.0"

from .qec_codes import BitFlipCode, PerfectCode
from .error_injection import ErrorInjector, ErrorType
from .visualizer import QECVisualizer
from .backend import QECBackend
from .spatial_layout import SpatialLayout, BitFlipCodeLayout, PerfectCodeLayout, get_layout_for_code
from .interactive_visualizer import InteractiveVisualizer

__all__ = [
    "BitFlipCode",
    "PerfectCode",
    "ErrorInjector",
    "ErrorType",
    "QECVisualizer",
    "QECBackend",
    "SpatialLayout",
    "BitFlipCodeLayout",
    "PerfectCodeLayout",
    "get_layout_for_code",
    "InteractiveVisualizer",
]

