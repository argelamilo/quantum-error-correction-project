"""
PanQEC-Style Visualization Module

This module provides visualizations that closely match PanQEC's architecture and style.
Adapted from PanQEC's visualization system for use with Plotly in Streamlit.
"""

from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# PanQEC-style color map
PANQEC_COLORMAP = {
    'pink': '#FFC0CB',
    'white': '#FFFFFF',
    'red': '#FF0000',
    'blue': '#0000FF',
    'green': '#00FF00',
    'gold': '#FFD700',
    'light-yellow': '#FFFFE0',
    'light-orange': '#FFA500',
    'salmon': '#FA8072',
    'orange': '#FFA500',
    'light-gray': '#D3D3D3',
    'gray': '#808080',
}

# PanQEC-style visualization config (adapted from gui-config.json)
PANQEC_VIS_CONFIG = {
    'qubits': {
        'object': 'sphere',  # Use sphere for Plotly (simplified from cylinder/sphere)
        'color': {
            'I': 'pink',      # No error - PanQEC uses pink for I
            'X': 'red',       # X error
            'Y': 'green',     # Y error  
            'Z': 'blue',      # Z error
            'corrected': 'green'  # Corrected qubits
        },
        'opacity': {
            'activated': {'min': 1.0, 'max': 1.0},      # Errors visible
            'deactivated': {'min': 0.3, 'max': 0.6}     # No errors, semi-transparent
        },
        'params': {'radius': 0.2}
    },
    'stabilizers': {
        'vertex': {
            'object': 'sphere',
            'color': {
                'activated': 'gold',      # Error detected
                'deactivated': 'white'    # No error
            },
            'opacity': {
                'activated': {'min': 1.0, 'max': 1.0},
                'deactivated': {'min': 0.3, 'max': 0.6}
            },
            'params': {'radius': 0.2}
        },
        'face': {
            'object': 'sphere',  # Simplified for Plotly
            'color': {
                'activated': 'gold',
                'deactivated': 'light-gray'
            },
            'opacity': {
                'activated': {'min': 0.8, 'max': 0.8},
                'deactivated': {'min': 0.2, 'max': 0.4}
            },
            'params': {'radius': 0.15}
        }
    }
}


class PanQECStyleVisualizer:
    """
    PanQEC-style visualizer that mimics PanQEC's architecture.
    Uses the same color scheme, coordinate system, and visualization approach.
    """
    
    def __init__(self):
        self.colormap = PANQEC_COLORMAP
        self.config = PANQEC_VIS_CONFIG
        self.error_qubits: Set[int] = set()
        self.error_types: Dict[int, str] = {}  # Maps qubit index to 'X', 'Y', 'Z'
        self.syndrome_values: Dict[int, int] = {}  # Maps stabilizer index to 0 or 1
        self.corrected_qubits: Set[int] = set()
    
    def qubit_representation(
        self,
        qubit_index: int,
        location: Tuple[float, float, float],
        error_type: Optional[str] = None,
        is_corrected: bool = False
    ) -> Dict:
        """
        Get PanQEC-style representation for a qubit.
        Similar to PanQEC's qubit_representation method.
        
        Args:
            qubit_index: Index of the qubit
            location: (x, y, z) coordinates
            error_type: 'X', 'Y', 'Z', or None for 'I'
            is_corrected: Whether this qubit has been corrected
            
        Returns:
            Dictionary with visualization parameters
        """
        config = self.config['qubits']
        
        # Determine color based on error type (PanQEC style)
        if is_corrected:
            color_name = 'corrected'
        elif error_type is None:
            color_name = 'I'
        else:
            color_name = error_type
        
        color_hex = self.colormap[config['color'][color_name]]
        
        # Determine opacity
        if error_type is not None or is_corrected:
            opacity = config['opacity']['activated']['max']
        else:
            opacity = config['opacity']['deactivated']['max']
        
        return {
            'index': qubit_index,
            'location': location,
            'object': config['object'],
            'color': color_hex,
            'opacity': opacity,
            'params': config['params'],
            'error_type': error_type or 'I',
            'is_corrected': is_corrected
        }
    
    def stabilizer_representation(
        self,
        stabilizer_index: int,
        location: Tuple[float, float, float],
        stabilizer_type: str = 'vertex',
        is_activated: bool = False
    ) -> Dict:
        """
        Get PanQEC-style representation for a stabilizer.
        Similar to PanQEC's stabilizer_representation method.
        
        Args:
            stabilizer_index: Index of the stabilizer
            location: (x, y, z) coordinates
            stabilizer_type: 'vertex' or 'face'
            is_activated: Whether error was detected (syndrome = 1)
            
        Returns:
            Dictionary with visualization parameters
        """
        config = self.config['stabilizers'][stabilizer_type]
        
        # Determine color (PanQEC style: gold for activated, white/gray for deactivated)
        activation = 'activated' if is_activated else 'deactivated'
        color_name = config['color'][activation]
        color_hex = self.colormap[color_name]
        
        # Determine opacity
        opacity = config['opacity'][activation]['max']
        
        return {
            'index': stabilizer_index,
            'location': location,
            'type': stabilizer_type,
            'object': config['object'],
            'color': color_hex,
            'opacity': opacity,
            'params': config['params'],
            'is_activated': is_activated
        }
    
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
        Create PanQEC-style 2D visualization using Plotly.
        
        Args:
            code: QEC code instance
            error_qubits: Set of qubit indices with errors
            error_types: Dict mapping qubit index to error type ('X', 'Y', 'Z')
            syndrome_values: Dict mapping stabilizer index to syndrome value (0 or 1)
            corrected_qubits: Set of qubit indices that have been corrected
            show_stabilizers: Whether to show stabilizers
            show_connections: Whether to show connections
            
        Returns:
            Plotly figure object
        """
        from .spatial_layout import get_layout_for_code
        
        # Update state
        self.error_qubits = error_qubits or set()
        self.error_types = error_types or {}
        self.syndrome_values = syndrome_values or {}
        self.corrected_qubits = corrected_qubits or set()
        
        # Get layout
        layout = get_layout_for_code(code)
        qubit_positions = layout.get_qubit_positions('2d')
        stabilizer_positions = layout.get_stabilizer_positions('2d')
        connections = layout.get_qubit_stabilizer_connections()
        
        # Create figure
        fig = go.Figure()
        
        # Draw connections first (so they appear behind)
        if show_connections and show_stabilizers:
            for q_idx, s_idx in connections:
                if q_idx in qubit_positions and s_idx in stabilizer_positions:
                    q_pos = qubit_positions[q_idx]
                    s_pos = stabilizer_positions[s_idx]
                    
                    fig.add_trace(go.Scatter(
                        x=[q_pos[0], s_pos[0]],
                        y=[q_pos[1], s_pos[1]],
                        mode='lines',
                        line=dict(color='rgba(80, 80, 80, 0.7)', width=3),  # Darker, more visible connections
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Draw qubits (PanQEC style)
        qubit_x, qubit_y = [], []
        qubit_colors, qubit_sizes, qubit_texts, qubit_opacities = [], [], [], []
        
        for q_idx in range(layout.n_qubits):
            if q_idx in qubit_positions:
                pos = qubit_positions[q_idx]
                qubit_x.append(pos[0])
                qubit_y.append(pos[1])
                
                # Get representation
                error_type = self.error_types.get(q_idx)
                is_corrected = q_idx in self.corrected_qubits
                repr_dict = self.qubit_representation(q_idx, pos, error_type, is_corrected)
                
                qubit_colors.append(repr_dict['color'])
                qubit_opacities.append(repr_dict['opacity'])
                qubit_sizes.append(15)  # Marker size
                
                # Create hover text
                error_str = f"Error: {error_type}" if error_type else "No error"
                corrected_str = " (Corrected)" if is_corrected else ""
                qubit_texts.append(f"Qubit {q_idx}<br>{error_str}{corrected_str}")
        
        # Add qubits trace
        fig.add_trace(go.Scatter(
            x=qubit_x,
            y=qubit_y,
            mode='markers+text',
            marker=dict(
                size=qubit_sizes,
                color=qubit_colors,
                opacity=0.9,  # Use constant opacity
                line=dict(width=2, color='black')
            ),
            text=[f"Q{i}" for i in range(len(qubit_x))],
            textposition="middle center",
            textfont=dict(size=10, color='black'),
            name='Qubits',
            hovertemplate='%{text}<extra></extra>',
            customdata=qubit_texts
        ))
        
        # Draw stabilizers (PanQEC style)
        if show_stabilizers:
            stab_x, stab_y = [], []
            stab_colors, stab_sizes, stab_texts, stab_opacities = [], [], [], []
            
            for s_idx in range(layout.n_stabilizers):
                if s_idx in stabilizer_positions:
                    pos = stabilizer_positions[s_idx]
                    stab_x.append(pos[0])
                    stab_y.append(pos[1])
                    
                    # Get representation
                    is_activated = self.syndrome_values.get(s_idx, 0) == 1
                    stab_type = 'vertex'  # Default, can be enhanced
                    repr_dict = self.stabilizer_representation(s_idx, pos, stab_type, is_activated)
                    
                    stab_colors.append(repr_dict['color'])
                    stab_opacities.append(repr_dict['opacity'])
                    stab_sizes.append(12)
                    
                    status = "Activated" if is_activated else "Deactivated"
                    stab_texts.append(f"Stabilizer {s_idx}<br>Status: {status}")
            
            # Add stabilizers trace (using diamond marker for distinction)
            fig.add_trace(go.Scatter(
                x=stab_x,
                y=stab_y,
                mode='markers+text',
                marker=dict(
                    size=stab_sizes,
                    color=stab_colors,
                    opacity=0.8,  # Use constant opacity
                    symbol='diamond',
                    line=dict(width=2, color='black')
                ),
                text=[f"S{i}" for i in range(len(stab_x))],
                textposition="middle center",
                textfont=dict(size=9, color='black'),
                name='Stabilizers',
                hovertemplate='%{text}<extra></extra>',
                customdata=stab_texts
            ))
        
        # Update layout (PanQEC style: dark background)
        fig.update_layout(
            title='',
            xaxis=dict(title='', showgrid=True, gridcolor='rgba(120, 120, 120, 0.5)', zeroline=False, zerolinecolor='rgba(100, 100, 100, 0.8)'),
            yaxis=dict(title='', showgrid=True, gridcolor='rgba(120, 120, 120, 0.5)', zeroline=False, scaleanchor='x', scaleratio=1, zerolinecolor='rgba(100, 100, 100, 0.8)'),
            plot_bgcolor='rgba(250, 250, 250, 1)',  # Slightly off-white
            paper_bgcolor='white',
            hovermode='closest',
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255, 255, 255, 0.8)'),
            width=700,
            height=600,
            margin=dict(l=50, r=50, t=20, b=50)
        )
        
        return fig
    
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
        Create PanQEC-style 3D visualization using Plotly.
        Mimics PanQEC's 3D Toric code visualization style.
        
        Args:
            code: QEC code instance
            error_qubits: Set of qubit indices with errors
            error_types: Dict mapping qubit index to error type ('X', 'Y', 'Z')
            syndrome_values: Dict mapping stabilizer index to syndrome value (0 or 1)
            corrected_qubits: Set of qubit indices that have been corrected
            show_stabilizers: Whether to show stabilizers
            show_connections: Whether to show connections
            
        Returns:
            Plotly figure object
        """
        from .spatial_layout import get_layout_for_code
        
        # Update state
        self.error_qubits = error_qubits or set()
        self.error_types = error_types or {}
        self.syndrome_values = syndrome_values or {}
        self.corrected_qubits = corrected_qubits or set()
        
        # Get layout
        layout = get_layout_for_code(code)
        qubit_positions = layout.get_qubit_positions('3d')
        stabilizer_positions = layout.get_stabilizer_positions('3d')
        connections = layout.get_qubit_stabilizer_connections()
        
        # Create figure
        fig = go.Figure()
        
        # Draw connections first (PanQEC style: gray, semi-transparent)
        if show_connections and show_stabilizers:
            for q_idx, s_idx in connections:
                if q_idx in qubit_positions and s_idx in stabilizer_positions:
                    q_pos = qubit_positions[q_idx]
                    s_pos = stabilizer_positions[s_idx]
                    
                    fig.add_trace(go.Scatter3d(
                        x=[q_pos[0], s_pos[0]],
                        y=[q_pos[1], s_pos[1]],
                        z=[q_pos[2], s_pos[2]],
                        mode='lines',
                        line=dict(color='rgba(80, 80, 80, 0.6)', width=4),  # Darker, more visible connections
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Draw qubits (PanQEC style: spheres with color coding)
        qubit_x, qubit_y, qubit_z = [], [], []
        qubit_colors, qubit_sizes, qubit_texts, qubit_opacities = [], [], [], []
        
        for q_idx in range(layout.n_qubits):
            if q_idx in qubit_positions:
                pos = qubit_positions[q_idx]
                qubit_x.append(pos[0])
                qubit_y.append(pos[1])
                qubit_z.append(pos[2])
                
                # Get PanQEC-style representation
                error_type = self.error_types.get(q_idx)
                is_corrected = q_idx in self.corrected_qubits
                repr_dict = self.qubit_representation(q_idx, pos, error_type, is_corrected)
                
                qubit_colors.append(repr_dict['color'])
                qubit_opacities.append(repr_dict['opacity'])
                qubit_sizes.append(12)  # Marker size
                
                # Create hover text
                error_str = f"Error: {error_type}" if error_type else "No error"
                corrected_str = " (Corrected)" if is_corrected else ""
                qubit_texts.append(f"Qubit {q_idx}<br>{error_str}{corrected_str}")
        
        # Add qubits trace - Plotly needs single opacity value, so we'll use a constant
        # Individual opacities would need to be handled differently
        fig.add_trace(go.Scatter3d(
            x=qubit_x,
            y=qubit_y,
            z=qubit_z,
            mode='markers+text',
            marker=dict(
                size=qubit_sizes,
                color=qubit_colors,
                opacity=0.9,  # Use constant opacity for all qubits
                line=dict(width=2, color='black')
            ),
            text=[f"Q{i}" for i in range(len(qubit_x))],
            textposition="middle center",
            textfont=dict(size=10, color='black'),
            name='Qubits',
            hovertemplate='%{text}<extra></extra>',
            customdata=qubit_texts
        ))
        
        # Draw stabilizers (PanQEC style: gold when activated, white when deactivated)
        if show_stabilizers:
            stab_x, stab_y, stab_z = [], [], []
            stab_colors, stab_sizes, stab_texts, stab_opacities = [], [], [], []
            
            for s_idx in range(layout.n_stabilizers):
                if s_idx in stabilizer_positions:
                    pos = stabilizer_positions[s_idx]
                    stab_x.append(pos[0])
                    stab_y.append(pos[1])
                    stab_z.append(pos[2])
                    
                    # Get PanQEC-style representation
                    is_activated = self.syndrome_values.get(s_idx, 0) == 1
                    stab_type = 'vertex'  # Default
                    repr_dict = self.stabilizer_representation(s_idx, pos, stab_type, is_activated)
                    
                    stab_colors.append(repr_dict['color'])
                    stab_opacities.append(repr_dict['opacity'])
                    stab_sizes.append(10)
                    
                    status = "Activated (Error detected)" if is_activated else "Deactivated (No error)"
                    stab_texts.append(f"Stabilizer {s_idx}<br>{status}")
            
            # Add stabilizers trace (using diamond marker)
            fig.add_trace(go.Scatter3d(
                x=stab_x,
                y=stab_y,
                z=stab_z,
                mode='markers+text',
                marker=dict(
                    size=stab_sizes,
                    color=stab_colors,
                    opacity=0.8,  # Use constant opacity
                    symbol='diamond',
                    line=dict(width=2, color='black')
                ),
                text=[f"S{i}" for i in range(len(stab_x))],
                textposition="middle center",
                textfont=dict(size=9, color='black'),
                name='Stabilizers',
                hovertemplate='%{text}<extra></extra>',
                customdata=stab_texts
            ))
        
        # Update layout (PanQEC style: dark blue background similar to PanQEC)
        fig.update_layout(
            title='',
            scene=dict(
                xaxis=dict(title='X', backgroundcolor='rgba(240, 240, 240, 1)', gridcolor='rgba(100, 100, 100, 0.5)', showgrid=True),
                yaxis=dict(title='Y', backgroundcolor='rgba(240, 240, 240, 1)', gridcolor='rgba(100, 100, 100, 0.5)', showgrid=True),
                zaxis=dict(title='Z', backgroundcolor='rgba(240, 240, 240, 1)', gridcolor='rgba(100, 100, 100, 0.5)', showgrid=True),
                bgcolor='rgba(245, 245, 245, 1)',  # Slightly darker background
                aspectmode='data'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='closest',
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255, 255, 255, 0.8)'),
            width=800,
            height=700,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        return fig

