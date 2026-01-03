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
        # Create a clean combined circuit - remove all classical registers first
        from qiskit import QuantumRegister, ClassicalRegister
        
        # Get quantum register from original circuit
        if not circuit.qregs:
            raise ValueError("Circuit must have quantum registers")
        
        qreg = circuit.qregs[0]
        n_qubits = len(qreg)
        
        # Create a fresh circuit with only quantum register (no classical registers from encoding)
        full_circuit = QuantumCircuit(qreg)
        
        # Copy all quantum operations from the error circuit (skip measurements)
        for instruction in circuit.data:
            if instruction.operation.name != 'measure':
                full_circuit.append(instruction.operation, instruction.qubits)
        
        # Create classical register for syndrome
        n_syndromes = syndrome_measurement_circuit.num_clbits
        creg = ClassicalRegister(n_syndromes, 'syndrome')
        full_circuit.add_register(creg)
        
        # Rebuild syndrome measurement using the same quantum register
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
        
        # Run simulation with multiple shots for reliability
        job = self.simulator.run(full_circuit, shots=1024)
        result = job.result()
        
        # Extract syndrome - the syndrome register is the only classical register now
        syndrome_creg = full_circuit.cregs[0]  # First (and only) classical register
        
        try:
            # Method 1: Get counts for just the syndrome register (most reliable)
            syndrome_counts = result.get_counts(creg=syndrome_creg)
            if syndrome_counts:
                syndrome_bitstring = max(syndrome_counts, key=syndrome_counts.get)
                # Qiskit returns bitstrings - need to check the format
                # Sometimes it's a string like "00" or "01", sometimes it's formatted differently
                if isinstance(syndrome_bitstring, str):
                    # Remove any spaces
                    syndrome_bitstring = syndrome_bitstring.replace(' ', '')
                    # Extract only 0/1
                    syndrome_bitstring = ''.join(c for c in syndrome_bitstring if c in '01')
                    
                    if len(syndrome_bitstring) >= n_syndromes:
                        # Qiskit typically uses little-endian for individual registers
                        # But we want [creg[0], creg[1], ...] which is measurement order
                        # Try both orders to be safe
                        syndrome = [int(bit) for bit in syndrome_bitstring[:n_syndromes]]
                        # Reverse if needed (Qiskit often uses little-endian)
                        syndrome = syndrome[::-1] if len(syndrome) == n_syndromes else syndrome
                        if len(syndrome) == n_syndromes:
                            return syndrome
        except Exception as e:
            # Fall through to method 2
            pass
        
        # Method 2: Parse from full counts
        counts = result.get_counts()
        if counts:
            bitstring = max(counts, key=counts.get)
            # Remove spaces and extract only 0/1
            bitstring_clean = ''.join(c for c in bitstring if c in '01')
            
            # Since syndrome register is the only classical register, 
            # the bitstring should just be the syndrome bits
            if len(bitstring_clean) >= n_syndromes:
                # Qiskit uses little-endian, so reverse to get measurement order
                syndrome_bits_str = bitstring_clean[:n_syndromes]
                syndrome = [int(bit) for bit in reversed(syndrome_bits_str)]
                return syndrome
        
        # Default: return all zeros (no error detected)
        return [0] * n_syndromes
    
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
        
        # Check if correction is imperfect (fidelity after correction is low)
        is_imperfect = fidelity_after < 0.99
        correction_perfect = fidelity_after > 0.99
        
        return {
            'initial_state': initial_state,
            'error_state': error_state,
            'corrected_state': corrected_state,
            'final_state': final_state,
            'syndrome': syndrome,
            'fidelity_before': fidelity_before,
            'fidelity_after': fidelity_after,
            'success_probability': success_probability,
            'success': correction_perfect,
            'is_imperfect': is_imperfect,  # True if correction didn't fully recover the state
            'correction_perfect': correction_perfect  # True if correction was successful
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

