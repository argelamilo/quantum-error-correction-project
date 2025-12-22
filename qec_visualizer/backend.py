"""
Backend Logic Module

This module handles state vector calculations, error simulation,
syndrome extraction, error correction logic, and fidelity calculations.
"""

from typing import List, Tuple, Optional, Dict
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.quantum_info import Statevector, Operator, partial_trace
from qiskit_aer import AerSimulator


class QECBackend:
    """
    Backend for quantum error correction simulations.
    Handles state calculations, syndrome extraction, and fidelity metrics.
    """
    
    def __init__(self, shots: int = 1024):
        """
        Initialize QEC backend.
        
        Args:
            shots: Number of measurement shots for statistical simulations
        """
        self.shots = shots
        self.simulator = AerSimulator()
    
    def get_statevector(self, circuit: QuantumCircuit) -> Statevector:
        """
        Calculate the state vector of a quantum circuit.
        
        Args:
            circuit: Quantum circuit
            
        Returns:
            Statevector object
        """
        # Create a copy without measurements for statevector calculation
        # Statevector cannot handle measurement operations
        circuit_no_measure = circuit.copy()
        
        # Remove measurements - try different methods for compatibility
        try:
            # Newer Qiskit versions
            circuit_no_measure.remove_final_measurements(inplace=True)
        except (AttributeError, TypeError):
            # Fallback: manually remove measurement operations
            new_circuit = QuantumCircuit(*circuit_no_measure.qregs)
            for instruction in circuit_no_measure.data:
                if instruction.operation.name != 'measure':
                    new_circuit.append(instruction.operation, instruction.qubits)
            circuit_no_measure = new_circuit
        
        return Statevector(circuit_no_measure)
    
    def get_probability_distribution(
        self,
        circuit: QuantumCircuit,
        shots: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get probability distribution of measurement outcomes.
        
        Args:
            circuit: Quantum circuit with measurements
            shots: Number of shots (defaults to self.shots)
            
        Returns:
            Dictionary mapping bitstrings to probabilities
        """
        if shots is None:
            shots = self.shots
        
        # Ensure circuit has measurements
        if not circuit.clbits:
            temp_circuit = circuit.copy()
            temp_circuit.measure_all()
        else:
            temp_circuit = circuit
        
        # Run simulation
        job = self.simulator.run(temp_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Convert to probabilities
        total = sum(counts.values())
        probabilities = {k: v / total for k, v in counts.items()}
        
        return probabilities
    
    def extract_syndrome(
        self,
        circuit: QuantumCircuit,
        syndrome_measurement_circuit: QuantumCircuit
    ) -> List[int]:
        """
        Extract syndrome from a circuit with error.
        
        Args:
            circuit: Circuit with encoded state and error
            syndrome_measurement_circuit: Circuit that measures syndrome
            
        Returns:
            List of syndrome bits
        """
        # Create a combined circuit
        full_circuit = circuit.copy()
        
        # Get quantum register from original circuit
        if not full_circuit.qregs:
            raise ValueError("Circuit must have quantum registers")
        
        qreg = full_circuit.qregs[0]
        n_qubits = len(qreg)
        
        # Create classical register for syndrome
        n_syndromes = syndrome_measurement_circuit.num_clbits
        creg = ClassicalRegister(n_syndromes, 'syndrome')
        full_circuit.add_register(creg)
        
        # Rebuild syndrome measurement using the same quantum register
        # Extract the syndrome measurement logic
        syndrome_ops = []
        for instruction in syndrome_measurement_circuit.data:
            op = instruction.operation
            qubits = instruction.qubits
            clbits = instruction.clbits
            
            # Map qubits and clbits to our circuit
            new_qubits = [qreg[q._index] for q in qubits if q._index < n_qubits]
            new_clbits = [creg[c._index] for c in clbits if c._index < n_syndromes]
            
            if new_qubits and new_clbits:
                full_circuit.append(op, new_qubits, new_clbits)
            elif new_qubits:
                full_circuit.append(op, new_qubits)
        
        # Run simulation
        job = self.simulator.run(full_circuit, shots=1)
        result = job.result()
        counts = result.get_counts()
        
        # Extract syndrome from measurement result
        if counts:
            bitstring = list(counts.keys())[0]
            # Remove all whitespace and convert to list of bits
            # Qiskit bitstrings may have spaces between registers
            bitstring_clean = ''.join(bitstring.split())
            # Filter out only '0' and '1' characters
            bitstring_clean = ''.join(c for c in bitstring_clean if c in '01')
            # Qiskit uses little-endian, so reverse to get correct order
            all_bits = [int(bit) for bit in reversed(bitstring_clean)]
            
            # Extract syndrome bits (they are in the syndrome register)
            # Since we added the syndrome register last, it should be in the first n_syndromes bits
            # (after reversing, which makes them the last n_syndromes in original order)
            if len(all_bits) >= n_syndromes:
                # Take the first n_syndromes bits (which are the syndrome register after reverse)
                syndrome = all_bits[:n_syndromes]
            else:
                # Pad with zeros if needed
                syndrome = all_bits + [0] * (n_syndromes - len(all_bits))
            
            return syndrome
        
        return []
    
    def calculate_fidelity(
        self,
        ideal_state: Statevector,
        noisy_state: Statevector
    ) -> float:
        """
        Calculate fidelity between ideal and noisy states.
        
        Fidelity = |⟨ψ_ideal|ψ_noisy⟩|²
        
        Args:
            ideal_state: Ideal state vector
            noisy_state: Noisy state vector
            
        Returns:
            Fidelity value between 0 and 1
        """
        # Calculate overlap
        overlap = ideal_state.inner(noisy_state)
        fidelity = abs(overlap) ** 2
        
        return fidelity
    
    def simulate_full_qec_process(
        self,
        encoding_circuit: QuantumCircuit,
        error_circuit: QuantumCircuit,
        syndrome_circuit: QuantumCircuit,
        correction_circuit: QuantumCircuit,
        decoding_circuit: QuantumCircuit
    ) -> Dict:
        """
        Simulate the full QEC process: encode -> error -> syndrome -> correct -> decode.
        
        Args:
            encoding_circuit: Circuit that encodes logical qubit
            error_circuit: Circuit with error injected
            syndrome_circuit: Circuit that measures syndrome
            correction_circuit: Circuit that corrects error
            decoding_circuit: Circuit that decodes logical qubit
            
        Returns:
            Dictionary containing:
                - initial_state: Statevector after encoding
                - error_state: Statevector after error
                - corrected_state: Statevector after correction
                - final_state: Statevector after decoding
                - syndrome: Extracted syndrome
                - fidelity_before: Fidelity before correction
                - fidelity_after: Fidelity after correction
                - success: Whether logical qubit was recovered correctly
        """
        # Step 1: Encoding
        initial_state = self.get_statevector(encoding_circuit)
        
        # Step 2: Error
        error_state = self.get_statevector(error_circuit)
        fidelity_before = self.calculate_fidelity(initial_state, error_state)
        
        # Step 3: Syndrome extraction
        syndrome = self.extract_syndrome(error_circuit, syndrome_circuit)
        
        # Step 4: Correction
        corrected_circuit = error_circuit.copy()
        corrected_circuit = corrected_circuit.compose(correction_circuit)
        corrected_state = self.get_statevector(corrected_circuit)
        fidelity_after = self.calculate_fidelity(initial_state, corrected_state)
        
        # Step 5: Decoding
        final_circuit = corrected_circuit.copy()
        final_circuit = final_circuit.compose(decoding_circuit)
        
        # Check if logical qubit was recovered
        # This is a simplified check - in practice, we'd measure and compare
        final_state = self.get_statevector(final_circuit)
        
        # Calculate success probability
        # For a perfect recovery, the final state should match the initial encoded state
        # (up to decoding)
        success_probability = abs(initial_state.inner(final_state)) ** 2
        
        return {
            'initial_state': initial_state,
            'error_state': error_state,
            'corrected_state': corrected_state,
            'final_state': final_state,
            'syndrome': syndrome,
            'fidelity_before': fidelity_before,
            'fidelity_after': fidelity_after,
            'success_probability': success_probability,
            'success': success_probability > 0.99  # Threshold for success
        }
    
    def calculate_error_rate(
        self,
        encoding_circuit: QuantumCircuit,
        error_type: str,
        error_probability: float,
        n_trials: int = 100
    ) -> Dict[str, float]:
        """
        Calculate error rates and correction success rates over multiple trials.
        
        Args:
            encoding_circuit: Encoding circuit
            error_type: Type of error
            error_probability: Probability of error occurring
            n_trials: Number of trials to run
            
        Returns:
            Dictionary with error statistics
        """
        # This would be implemented based on specific QEC code
        # For now, return placeholder
        return {
            'error_rate': error_probability,
            'detection_rate': 0.0,
            'correction_rate': 0.0,
            'logical_error_rate': 0.0
        }
    
    def get_bloch_sphere_coordinates(
        self,
        statevector: Statevector,
        qubit_index: int = 0
    ) -> Tuple[float, float, float]:
        """
        Get Bloch sphere coordinates (x, y, z) for a single qubit.
        
        Args:
            statevector: Full state vector
            qubit_index: Index of qubit to extract
            
        Returns:
            Tuple of (x, y, z) coordinates on Bloch sphere
        """
        # For multi-qubit states, we need to trace out other qubits
        if statevector.num_qubits > 1:
            # Partial trace to get single-qubit density matrix
            density_matrix = partial_trace(statevector, [i for i in range(statevector.num_qubits) if i != qubit_index])
        else:
            density_matrix = statevector.to_operator()
        
        # Convert density matrix to Bloch coordinates
        # ρ = (I + x*X + y*Y + z*Z) / 2
        # x = Tr(ρ*X), y = Tr(ρ*Y), z = Tr(ρ*Z)
        pauli_x = Operator.from_label('X')
        pauli_y = Operator.from_label('Y')
        pauli_z = Operator.from_label('Z')
        
        x = np.real(np.trace(density_matrix @ pauli_x))
        y = np.real(np.trace(density_matrix @ pauli_y))
        z = np.real(np.trace(density_matrix @ pauli_z))
        
        return (x, y, z)

