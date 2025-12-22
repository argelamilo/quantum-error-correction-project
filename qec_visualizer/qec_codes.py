"""
Quantum Error Correction Code Implementations

This module contains implementations of various QEC codes:
- 3-qubit bit-flip code
- 5-qubit perfect code
"""

from typing import List, Tuple, Optional
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator


class QECCode:
    """Base class for quantum error correction codes."""
    
    def __init__(self, name: str, n_qubits: int, n_logical: int):
        """
        Initialize a QEC code.
        
        Args:
            name: Name of the code
            n_qubits: Total number of physical qubits
            n_logical: Number of logical qubits
        """
        self.name = name
        self.n_qubits = n_qubits
        self.n_logical = n_logical
    
    def encode(self, logical_state: int = 0) -> QuantumCircuit:
        """
        Create encoding circuit.
        
        Args:
            logical_state: Logical state to encode (0 or 1)
            
        Returns:
            Quantum circuit that encodes the logical state
        """
        raise NotImplementedError
    
    def syndrome_measurement(self) -> QuantumCircuit:
        """
        Create syndrome measurement circuit.
        
        Returns:
            Quantum circuit that measures the syndrome
        """
        raise NotImplementedError
    
    def correct(self, syndrome: List[int]) -> QuantumCircuit:
        """
        Create correction circuit based on syndrome.
        
        Args:
            syndrome: Syndrome measurement results
            
        Returns:
            Quantum circuit that corrects errors
        """
        raise NotImplementedError
    
    def decode(self) -> QuantumCircuit:
        """
        Create decoding circuit.
        
        Returns:
            Quantum circuit that decodes the logical qubit
        """
        raise NotImplementedError


class BitFlipCode(QECCode):
    """
    3-qubit bit-flip code.
    
    Encodes one logical qubit into three physical qubits.
    Can detect and correct single bit-flip errors.
    """
    
    def __init__(self):
        super().__init__("3-qubit Bit-Flip Code", n_qubits=3, n_logical=1)
        self.n_syndromes = 2  # Two stabilizer measurements
    
    def encode(self, logical_state: int = 0) -> QuantumCircuit:
        """
        Encode a logical qubit into three physical qubits.
        
        Args:
            logical_state: 0 for |0⟩, 1 for |1⟩
            
        Returns:
            Encoding circuit
        """
        qreg = QuantumRegister(3, 'q')
        creg = ClassicalRegister(3, 'c')
        circuit = QuantumCircuit(qreg, creg)
        
        # Prepare logical state
        if logical_state == 1:
            circuit.x(qreg[0])
        
        # Encoding: |0⟩ -> |000⟩, |1⟩ -> |111⟩
        circuit.cx(qreg[0], qreg[1])
        circuit.cx(qreg[0], qreg[2])
        
        return circuit
    
    def syndrome_measurement(self, qreg: Optional[QuantumRegister] = None, creg: Optional[ClassicalRegister] = None) -> QuantumCircuit:
        """
        Measure stabilizers: Z_0 Z_1 and Z_1 Z_2.
        
        Args:
            qreg: Optional quantum register (if None, creates new one)
            creg: Optional classical register for syndrome (if None, creates new one)
        
        Returns:
            Syndrome measurement circuit
        """
        if qreg is None:
            qreg = QuantumRegister(3, 'q')
        if creg is None:
            creg = ClassicalRegister(2, 'syndrome')
        
        circuit = QuantumCircuit(qreg, creg)
        
        # Measure Z_0 Z_1 (parity of qubits 0 and 1)
        # Use ancilla approach: CNOT into ancilla, measure ancilla
        # For simplicity, we'll use the parity measurement approach
        circuit.cx(qreg[0], qreg[1])
        circuit.measure(qreg[1], creg[0])
        circuit.cx(qreg[0], qreg[1])  # Uncompute
        
        # Measure Z_1 Z_2 (parity of qubits 1 and 2)
        circuit.cx(qreg[1], qreg[2])
        circuit.measure(qreg[2], creg[1])
        circuit.cx(qreg[1], qreg[2])  # Uncompute
        
        return circuit
    
    def correct(self, syndrome: List[int]) -> QuantumCircuit:
        """
        Apply correction based on syndrome.
        
        Syndrome interpretation:
        - [0, 0]: No error
        - [1, 0]: Error on qubit 0
        - [0, 1]: Error on qubit 2
        - [1, 1]: Error on qubit 1
        
        Args:
            syndrome: Two-bit syndrome measurement
            
        Returns:
            Correction circuit
        """
        qreg = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qreg)
        
        if len(syndrome) != 2:
            raise ValueError("Syndrome must be a 2-bit value")
        
        # Determine error location
        if syndrome == [1, 0]:
            circuit.x(qreg[0])  # Error on qubit 0
        elif syndrome == [0, 1]:
            circuit.x(qreg[2])  # Error on qubit 2
        elif syndrome == [1, 1]:
            circuit.x(qreg[1])  # Error on qubit 1
        # [0, 0] means no error, no correction needed
        
        return circuit
    
    def decode(self) -> QuantumCircuit:
        """
        Decode three physical qubits back to one logical qubit.
        
        Returns:
            Decoding circuit
        """
        qreg = QuantumRegister(3, 'q')
        creg = ClassicalRegister(1, 'logical')
        circuit = QuantumCircuit(qreg, creg)
        
        # Decoding: inverse of encoding
        circuit.cx(qreg[0], qreg[2])
        circuit.cx(qreg[0], qreg[1])
        circuit.measure(qreg[0], creg[0])
        
        return circuit


class PerfectCode(QECCode):
    """
    5-qubit perfect code.
    
    The smallest code that can correct arbitrary single-qubit errors.
    Encodes one logical qubit into five physical qubits.
    """
    
    def __init__(self):
        super().__init__("5-qubit Perfect Code", n_qubits=5, n_logical=1)
        self.n_syndromes = 4  # Four stabilizer measurements
    
    def encode(self, logical_state: int = 0) -> QuantumCircuit:
        """
        Encode a logical qubit into five physical qubits.
        
        Args:
            logical_state: 0 for |0⟩, 1 for |1⟩
            
        Returns:
            Encoding circuit
        """
        qreg = QuantumRegister(5, 'q')
        creg = ClassicalRegister(5, 'c')
        circuit = QuantumCircuit(qreg, creg)
        
        # Prepare logical state
        if logical_state == 1:
            circuit.x(qreg[0])
        
        # Encoding circuit for 5-qubit perfect code
        # This is a standard encoding sequence
        circuit.h(qreg[1])
        circuit.h(qreg[2])
        circuit.h(qreg[3])
        circuit.h(qreg[4])
        
        circuit.cx(qreg[0], qreg[1])
        circuit.cx(qreg[0], qreg[2])
        circuit.cx(qreg[0], qreg[3])
        circuit.cx(qreg[0], qreg[4])
        
        circuit.cz(qreg[1], qreg[2])
        circuit.cz(qreg[2], qreg[3])
        circuit.cz(qreg[3], qreg[4])
        circuit.cz(qreg[4], qreg[1])
        
        circuit.h(qreg[1])
        circuit.h(qreg[2])
        circuit.h(qreg[3])
        circuit.h(qreg[4])
        
        return circuit
    
    def syndrome_measurement(self) -> QuantumCircuit:
        """
        Measure the four stabilizer generators.
        
        Returns:
            Syndrome measurement circuit
        """
        qreg = QuantumRegister(5, 'q')
        creg = ClassicalRegister(4, 'syndrome')
        circuit = QuantumCircuit(qreg, creg)
        
        # Stabilizer generators for 5-qubit code:
        # S1 = X Z Z X I
        # S2 = I X Z Z X
        # S3 = X I X Z Z
        # S4 = Z X I X Z
        
        # Measure S1 = X Z Z X I
        circuit.h(qreg[0])
        circuit.cz(qreg[0], qreg[1])
        circuit.cz(qreg[1], qreg[2])
        circuit.h(qreg[3])
        circuit.cz(qreg[0], qreg[3])
        circuit.measure(qreg[3], creg[0])
        circuit.h(qreg[3])
        circuit.cz(qreg[0], qreg[3])
        circuit.cz(qreg[1], qreg[2])
        circuit.cz(qreg[0], qreg[1])
        circuit.h(qreg[0])
        
        # Measure S2 = I X Z Z X
        circuit.h(qreg[1])
        circuit.cz(qreg[1], qreg[2])
        circuit.cz(qreg[2], qreg[3])
        circuit.h(qreg[4])
        circuit.cz(qreg[1], qreg[4])
        circuit.measure(qreg[4], creg[1])
        circuit.h(qreg[4])
        circuit.cz(qreg[1], qreg[4])
        circuit.cz(qreg[2], qreg[3])
        circuit.cz(qreg[1], qreg[2])
        circuit.h(qreg[1])
        
        # Measure S3 = X I X Z Z
        circuit.h(qreg[0])
        circuit.h(qreg[2])
        circuit.cz(qreg[2], qreg[3])
        circuit.cz(qreg[3], qreg[4])
        circuit.cz(qreg[0], qreg[2])
        circuit.measure(qreg[2], creg[2])
        circuit.cz(qreg[0], qreg[2])
        circuit.cz(qreg[3], qreg[4])
        circuit.cz(qreg[2], qreg[3])
        circuit.h(qreg[2])
        circuit.h(qreg[0])
        
        # Measure S4 = Z X I X Z
        circuit.h(qreg[1])
        circuit.h(qreg[3])
        circuit.cz(qreg[0], qreg[1])
        circuit.cz(qreg[3], qreg[4])
        circuit.cz(qreg[1], qreg[3])
        circuit.measure(qreg[3], creg[3])
        circuit.cz(qreg[1], qreg[3])
        circuit.cz(qreg[3], qreg[4])
        circuit.cz(qreg[0], qreg[1])
        circuit.h(qreg[3])
        circuit.h(qreg[1])
        
        return circuit
    
    def correct(self, syndrome: List[int]) -> QuantumCircuit:
        """
        Apply correction based on syndrome.
        
        The 5-qubit code can correct any single-qubit error.
        Syndrome determines both error type and location.
        
        Args:
            syndrome: Four-bit syndrome measurement
            
        Returns:
            Correction circuit
        """
        qreg = QuantumRegister(5, 'q')
        circuit = QuantumCircuit(qreg)
        
        if len(syndrome) != 4:
            raise ValueError("Syndrome must be a 4-bit value")
        
        # Convert syndrome to integer for lookup
        syndrome_int = sum(syndrome[i] * (2 ** i) for i in range(4))
        
        # Lookup table for corrections
        # This is a simplified version - full implementation would
        # decode all possible single-qubit errors
        correction_table = {
            0: None,  # No error
            # Bit-flip errors
            1: (0, 'X'),  # X error on qubit 0
            2: (1, 'X'),  # X error on qubit 1
            4: (2, 'X'),  # X error on qubit 2
            8: (3, 'X'),  # X error on qubit 3
            # Phase-flip errors
            3: (0, 'Z'),  # Z error on qubit 0
            6: (1, 'Z'),  # Z error on qubit 1
            12: (2, 'Z'),  # Z error on qubit 2
            # Y errors (combination)
            5: (0, 'Y'),  # Y error on qubit 0
            7: (1, 'Y'),  # Y error on qubit 1
            # Additional corrections for other syndromes
        }
        
        if syndrome_int in correction_table and correction_table[syndrome_int] is not None:
            qubit_idx, error_type = correction_table[syndrome_int]
            if error_type == 'X':
                circuit.x(qreg[qubit_idx])
            elif error_type == 'Z':
                circuit.z(qreg[qubit_idx])
            elif error_type == 'Y':
                circuit.y(qreg[qubit_idx])
        
        return circuit
    
    def decode(self) -> QuantumCircuit:
        """
        Decode five physical qubits back to one logical qubit.
        
        Returns:
            Decoding circuit (inverse of encoding)
        """
        qreg = QuantumRegister(5, 'q')
        creg = ClassicalRegister(1, 'logical')
        circuit = QuantumCircuit(qreg, creg)
        
        # Inverse of encoding
        circuit.h(qreg[4])
        circuit.h(qreg[3])
        circuit.h(qreg[2])
        circuit.h(qreg[1])
        
        circuit.cz(qreg[4], qreg[1])
        circuit.cz(qreg[3], qreg[4])
        circuit.cz(qreg[2], qreg[3])
        circuit.cz(qreg[1], qreg[2])
        
        circuit.cx(qreg[0], qreg[4])
        circuit.cx(qreg[0], qreg[3])
        circuit.cx(qreg[0], qreg[2])
        circuit.cx(qreg[0], qreg[1])
        
        circuit.h(qreg[4])
        circuit.h(qreg[3])
        circuit.h(qreg[2])
        circuit.h(qreg[1])
        
        circuit.measure(qreg[0], creg[0])
        
        return circuit

