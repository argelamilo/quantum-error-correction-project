"""
Spatial Layout Module

This module generates 2D and 3D spatial coordinates for QEC codes,
enabling interactive visualization of qubits, stabilizers, and their relationships.
"""

from typing import List, Tuple, Dict, Optional
import numpy as np


class SpatialLayout:
    """Base class for spatial layouts of QEC codes."""
    
    def __init__(self, name: str):
        self.name = name
    
    def get_qubit_positions(self, dimension: str = '2d') -> Dict[int, Tuple[float, float, float]]:
        """
        Get 3D positions for qubits.
        
        Args:
            dimension: '2d' or '3d'
            
        Returns:
            Dictionary mapping qubit index to (x, y, z) coordinates
        """
        raise NotImplementedError
    
    def get_stabilizer_positions(self, dimension: str = '2d') -> Dict[int, Tuple[float, float, float]]:
        """
        Get 3D positions for stabilizers.
        
        Args:
            dimension: '2d' or '3d'
            
        Returns:
            Dictionary mapping stabilizer index to (x, y, z) coordinates
        """
        raise NotImplementedError
    
    def get_qubit_stabilizer_connections(self) -> List[Tuple[int, int]]:
        """
        Get connections between qubits and stabilizers.
        
        Returns:
            List of (qubit_index, stabilizer_index) tuples
        """
        raise NotImplementedError


class BitFlipCodeLayout(SpatialLayout):
    """Spatial layout for 3-qubit bit-flip code."""
    
    def __init__(self):
        super().__init__("3-qubit Bit-Flip Code")
        self.n_qubits = 3
        self.n_stabilizers = 2
        
        # For bit-flip code:
        # Stabilizer 0: Z_0 Z_1 (measures parity of qubits 0 and 1)
        # Stabilizer 1: Z_1 Z_2 (measures parity of qubits 1 and 2)
        self.stabilizer_qubits = {
            0: [0, 1],  # Stabilizer 0 connects to qubits 0 and 1
            1: [1, 2],  # Stabilizer 1 connects to qubits 1 and 2
        }
    
    def get_qubit_positions(self, dimension: str = '2d') -> Dict[int, Tuple[float, float, float]]:
        """Get positions for 3 qubits - Toric 3D style structure."""
        if dimension == '2d':
            # Simple 2D layout: qubits in a horizontal line
            positions = {
                0: (-1.0, 0.0, 0.0),
                1: (0.0, 0.0, 0.0),
                2: (1.0, 0.0, 0.0),
            }
        else:
            # 3D Toric-style layout: Qubits positioned on edges of a cubic lattice
            # Similar to PanQEC's Toric 3D code where qubits live on edges of a cube
            # Creates a proper 3D cubic lattice visualization with proper spacing
            scale = 1.2
            positions = {
                0: (-scale, 0.0, 0.0),   # Qubit 0: on edge along x-axis (left edge)
                1: (0.0, 0.0, scale),    # Qubit 1: on edge along z-axis (front vertical edge)
                2: (scale, 0.0, 0.0),    # Qubit 2: on edge along x-axis (right edge)
            }
        return positions
    
    def get_stabilizer_positions(self, dimension: str = '2d') -> Dict[int, Tuple[float, float, float]]:
        """Get positions for 2 stabilizers."""
        qubit_positions = self.get_qubit_positions(dimension)
        
        # Position stabilizers in 3D space (like vertices in a 3D structure)
        positions = {}
        for stabilizer_idx, qubit_indices in self.stabilizer_qubits.items():
            # Calculate midpoint of connected qubits
            qubit_coords = [qubit_positions[q] for q in qubit_indices]
            midpoint = (
                sum(c[0] for c in qubit_coords) / len(qubit_coords),
                sum(c[1] for c in qubit_coords) / len(qubit_coords),
                sum(c[2] for c in qubit_coords) / len(qubit_coords)
            )
            if dimension == '2d':
                # 2D: position above
                positions[stabilizer_idx] = (midpoint[0], midpoint[1] + 0.5, midpoint[2])
            else:
                # 3D: position above in Y direction (like a vertex above the edge)
                positions[stabilizer_idx] = (midpoint[0], midpoint[1] + 0.8, midpoint[2])
        
        return positions
    
    def get_qubit_stabilizer_connections(self) -> List[Tuple[int, int]]:
        """Get connections between qubits and stabilizers."""
        connections = []
        for stabilizer_idx, qubit_indices in self.stabilizer_qubits.items():
            for qubit_idx in qubit_indices:
                connections.append((qubit_idx, stabilizer_idx))
        return connections


class PerfectCodeLayout(SpatialLayout):
    """Spatial layout for 5-qubit perfect code."""
    
    def __init__(self):
        super().__init__("5-qubit Perfect Code")
        self.n_qubits = 5
        self.n_stabilizers = 4
        
        # For 5-qubit perfect code, stabilizers are:
        # S1 = X Z Z X I (qubits 0, 1, 2, 3)
        # S2 = I X Z Z X (qubits 1, 2, 3, 4)
        # S3 = X I X Z Z (qubits 0, 2, 3, 4)
        # S4 = Z X I X Z (qubits 0, 1, 3, 4)
        self.stabilizer_qubits = {
            0: [0, 1, 2, 3],
            1: [1, 2, 3, 4],
            2: [0, 2, 3, 4],
            3: [0, 1, 3, 4],
        }
    
    def get_qubit_positions(self, dimension: str = '2d') -> Dict[int, Tuple[float, float, float]]:
        """Get positions for 5 qubits - Toric 3D style cubic lattice structure."""
        if dimension == '2d':
            # Pentagonal layout in 2D
            angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
            radius = 1.0
            positions = {}
            for i, angle in enumerate(angles):
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                positions[i] = (x, y, 0.0)
        else:
            # 3D Toric-style layout: Qubits on edges of a cubic lattice
            # Creates a proper 3D lattice visualization like PanQEC's Toric 3D code
            # Qubits positioned on edges of a cube (like toric code edges)
            scale = 1.1
            positions = {
                0: (-scale, -scale, 0.0),   # Qubit 0: bottom-left-front edge (xy plane, z=0)
                1: (scale, -scale, 0.0),    # Qubit 1: bottom-right-front edge (xy plane, z=0)
                2: (scale, scale, 0.0),     # Qubit 2: top-right-front edge (xy plane, z=0)
                3: (-scale, 0.0, scale),    # Qubit 3: left-front-vertical edge (xz plane)
                4: (0.0, scale, scale),     # Qubit 4: top-front-vertical edge (yz plane)
            }
        return positions
    
    def get_stabilizer_positions(self, dimension: str = '2d') -> Dict[int, Tuple[float, float, float]]:
        """Get positions for 4 stabilizers."""
        qubit_positions = self.get_qubit_positions(dimension)
        
        # Position stabilizers in 3D space (like vertices/faces in a 3D structure)
        positions = {}
        for stabilizer_idx, qubit_indices in self.stabilizer_qubits.items():
            # Calculate centroid of connected qubits
            qubit_coords = [qubit_positions[q] for q in qubit_indices]
            centroid = (
                sum(c[0] for c in qubit_coords) / len(qubit_coords),
                sum(c[1] for c in qubit_coords) / len(qubit_coords),
                sum(c[2] for c in qubit_coords) / len(qubit_coords)
            )
            if dimension == '2d':
                # 2D: offset upward
                positions[stabilizer_idx] = (centroid[0], centroid[1] + 0.3, centroid[2])
            else:
                # 3D: Position stabilizers at strategic 3D positions
                # Place them at different heights to create a 3D structure
                offset_y = 0.4 + (stabilizer_idx * 0.2)  # Stagger them in Y direction
                positions[stabilizer_idx] = (centroid[0], centroid[1] + offset_y, centroid[2])
        
        return positions
    
    def get_qubit_stabilizer_connections(self) -> List[Tuple[int, int]]:
        """Get connections between qubits and stabilizers."""
        connections = []
        for stabilizer_idx, qubit_indices in self.stabilizer_qubits.items():
            for qubit_idx in qubit_indices:
                connections.append((qubit_idx, stabilizer_idx))
        return connections


def get_layout_for_code(code) -> SpatialLayout:
    """Get appropriate spatial layout for a QEC code."""
    from .qec_codes import BitFlipCode, PerfectCode
    
    if isinstance(code, BitFlipCode):
        return BitFlipCodeLayout()
    elif isinstance(code, PerfectCode):
        return PerfectCodeLayout()
    else:
        # For unknown codes, return a default layout
        return BitFlipCodeLayout()  # Fallback

