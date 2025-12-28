"""
Error Injection Module

This module handles injection of various types of errors into quantum circuits,
including bit-flip, phase-flip, depolarizing, and rotation errors (Rx gate).
"""

from enum import Enum
from typing import Optional, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


class ErrorType(Enum):
    """Types of errors that can be injected."""
    BIT_FLIP = "bit_flip"  # X error
    PHASE_FLIP = "phase_flip"  # Z error
    DEPOLARIZING = "depolarizing"  # Random Pauli error
    ROTATION_X = "rotation_x"  # Rx gate error (as per professor feedback)
    ROTATION_Y = "rotation_y"  # Ry gate error
    ROTATION_Z = "rotation_z"  # Rz gate error
    Y_ERROR = "y_error"  # Y error (combination of X and Z)


class ErrorInjector:
    """
    Handles injection of errors into quantum circuits.
    """
    
    def __init__(self, n_qubits: int):
        """
        Initialize error injector.
        
        Args:
            n_qubits: Number of qubits in the circuit
        """
        self.n_qubits = n_qubits
    
    def inject_error(
        self,
        circuit: QuantumCircuit,
        error_type: ErrorType,
        qubit: int,
        error_probability: float = 1.0,
        rotation_angle: float = None
    ) -> QuantumCircuit:
        """
        Inject an error into a quantum circuit.
        
        Args:
            circuit: Quantum circuit to inject error into
            error_type: Type of error to inject
            qubit: Index of qubit where error occurs
            error_probability: Probability of error occurring (0.0 to 1.0)
            rotation_angle: Angle for rotation errors (in radians)
            
        Returns:
            Circuit with error injected
        """
        if qubit < 0 or qubit >= self.n_qubits:
            raise ValueError(f"Qubit index {qubit} out of range [0, {self.n_qubits})")
        
        if error_probability < 0.0 or error_probability > 1.0:
            raise ValueError("Error probability must be between 0.0 and 1.0")
        
        # Create a copy of the circuit
        error_circuit = circuit.copy()
        
        # Apply error with given probability
        # If probability is 1.0 (or very close), always apply
        if error_probability >= 1.0 - 1e-10:
            should_apply = True
        else:
            should_apply = np.random.random() < error_probability
        
        if should_apply:
            # Get the actual qubit from the circuit's quantum register
            # This ensures we're using the correct qubit object
            if error_circuit.qregs:
                qreg = error_circuit.qregs[0]
                if qubit < len(qreg):
                    target_qubit = qreg[qubit]
                else:
                    raise ValueError(f"Qubit index {qubit} out of range for register with {len(qreg)} qubits")
            else:
                # Fallback: use qubit index directly
                target_qubit = qubit
            
            if error_type == ErrorType.BIT_FLIP:
                error_circuit.x(target_qubit)
            
            elif error_type == ErrorType.PHASE_FLIP:
                error_circuit.z(target_qubit)
            
            elif error_type == ErrorType.Y_ERROR:
                error_circuit.y(target_qubit)
            
            elif error_type == ErrorType.DEPOLARIZING:
                # Randomly choose X, Y, or Z error
                pauli_error = np.random.choice(['X', 'Y', 'Z'])
                if pauli_error == 'X':
                    error_circuit.x(target_qubit)
                elif pauli_error == 'Y':
                    error_circuit.y(target_qubit)
                else:
                    error_circuit.z(target_qubit)
            
            elif error_type == ErrorType.ROTATION_X:
                # Rx gate error (as per professor feedback)
                angle = rotation_angle if rotation_angle is not None else np.pi / 4
                error_circuit.rx(angle, target_qubit)
            
            elif error_type == ErrorType.ROTATION_Y:
                angle = rotation_angle if rotation_angle is not None else np.pi / 4
                error_circuit.ry(angle, target_qubit)
            
            elif error_type == ErrorType.ROTATION_Z:
                angle = rotation_angle if rotation_angle is not None else np.pi / 4
                error_circuit.rz(angle, target_qubit)
        
        return error_circuit
    
    def inject_multiple_errors(
        self,
        circuit: QuantumCircuit,
        errors: List[dict]
    ) -> QuantumCircuit:
        """
        Inject multiple errors into a circuit.
        
        Args:
            circuit: Quantum circuit to inject errors into
            errors: List of error dictionaries, each containing:
                - error_type: ErrorType enum
                - qubit: int
                - error_probability: float (optional, default 1.0)
                - rotation_angle: float (optional, for rotation errors)
                
        Returns:
            Circuit with all errors injected
        """
        error_circuit = circuit.copy()
        
        for error in errors:
            error_type = error['error_type']
            qubit = error['qubit']
            error_probability = error.get('error_probability', 1.0)
            rotation_angle = error.get('rotation_angle', None)
            
            error_circuit = self.inject_error(
                error_circuit,
                error_type,
                qubit,
                error_probability,
                rotation_angle
            )
        
        return error_circuit
    
    @staticmethod
    def get_error_description(error_type: ErrorType) -> str:
        """
        Get a human-readable description of an error type.
        
        Args:
            error_type: Type of error
            
        Returns:
            Description string
        """
        descriptions = {
            ErrorType.BIT_FLIP: "Bit-flip error (X gate) - flips |0⟩ ↔ |1⟩",
            ErrorType.PHASE_FLIP: "Phase-flip error (Z gate) - flips phase |+⟩ ↔ |-⟩",
            ErrorType.DEPOLARIZING: "Depolarizing error - random Pauli (X, Y, or Z)",
            ErrorType.ROTATION_X: "Rotation error around X-axis (Rx gate)",
            ErrorType.ROTATION_Y: "Rotation error around Y-axis (Ry gate)",
            ErrorType.ROTATION_Z: "Rotation error around Z-axis (Rz gate)",
            ErrorType.Y_ERROR: "Y error - combination of bit-flip and phase-flip",
        }
        return descriptions.get(error_type, "Unknown error type")

