"""
Interactive Visualization Module

This module provides interactive 2D and 3D visualizations of QEC codes
using Plotly, based on PanQEC's visualization architecture and style.
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import plotly.graph_objects as go

from .spatial_layout import SpatialLayout, get_layout_for_code
from .panqec_style_visualizer import PanQECStyleVisualizer


class InteractiveVisualizer:
    """
    Interactive 2D and 3D visualizer for QEC codes.
    Uses PanQEC-style visualization architecture.
    """
    
    def __init__(self):
        """Initialize with PanQEC-style visualizer."""
        self.panqec_vis = PanQECStyleVisualizer()
    
    def visualize_code_2d(
        self,
        code,
        error_qubits: Optional[Set[int]] = None,
        error_types: Optional[Dict[int, str]] = None,
        syndrome_values: Optional[Dict[int, int]] = None,
        corrected_qubits: Optional[Set[int]] = None,
        show_stabilizers: bool = True,
        show_connections: bool = True
    ) -> go.Figure:
        """
        Create PanQEC-style 2D visualization.
        
        This method uses the PanQEC visualization architecture.
        """
        return self.panqec_vis.visualize_code_2d(
            code, error_qubits, error_types, syndrome_values,
            corrected_qubits, show_stabilizers, show_connections
        )
    
    def visualize_code_3d(
        self,
        code,
        error_qubits: Optional[Set[int]] = None,
        error_types: Optional[Dict[int, str]] = None,
        syndrome_values: Optional[Dict[int, int]] = None,
        corrected_qubits: Optional[Set[int]] = None,
        show_stabilizers: bool = True,
        show_connections: bool = True
    ) -> go.Figure:
        """
        Create PanQEC-style 3D visualization.
        
        This method uses the PanQEC visualization architecture.
        Mimics PanQEC's 3D Toric code visualization style.
        """
        return self.panqec_vis.visualize_code_3d(
            code, error_qubits, error_types, syndrome_values,
            corrected_qubits, show_stabilizers, show_connections
        )
