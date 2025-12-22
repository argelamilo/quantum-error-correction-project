"""
Interactive QEC Visualization Tool

A user-friendly, step-by-step interactive tool for exploring quantum error correction.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from qec_visualizer import BitFlipCode, PerfectCode, ErrorInjector, ErrorType, QECVisualizer, QECBackend


class InteractiveQECTool:
    """Interactive tool for exploring QEC codes step-by-step."""
    
    def __init__(self):
        self.visualizer = QECVisualizer()
        self.backend = QECBackend()
        self.current_code = None
        self.encoding_circuit = None
        self.error_circuit = None
        self.syndrome = None
        
    def print_header(self, title):
        """Print a formatted header."""
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)
    
    def print_step(self, step_num, description):
        """Print a step description."""
        print(f"\n[Step {step_num}] {description}")
        print("-" * 70)
    
    def select_code(self):
        """Let user select a QEC code."""
        self.print_header("Quantum Error Correction - Code Selection")
        print("\nAvailable QEC Codes:")
        print("  1. 3-qubit Bit-Flip Code (recommended for beginners)")
        print("     - Encodes 1 logical qubit into 3 physical qubits")
        print("     - Detects and corrects single bit-flip errors")
        print("\n  2. 5-qubit Perfect Code (advanced)")
        print("     - Encodes 1 logical qubit into 5 physical qubits")
        print("     - Corrects any single-qubit error")
        
        while True:
            choice = input("\nSelect code (1 or 2): ").strip()
            if choice == "1":
                self.current_code = BitFlipCode()
                print(f"\n✓ Selected: {self.current_code.name}")
                return
            elif choice == "2":
                self.current_code = PerfectCode()
                print(f"\n✓ Selected: {self.current_code.name}")
                return
            else:
                print("Invalid choice. Please enter 1 or 2.")
    
    def encode_logical_qubit(self):
        """Let user encode a logical qubit."""
        self.print_step(1, "Encoding Logical Qubit")
        print("\nLogical qubits can be in state |0⟩ or |1⟩.")
        print("We will encode this logical qubit into multiple physical qubits for error protection.")
        
        while True:
            state_str = input("\nEnter logical state (0 or 1): ").strip()
            if state_str in ["0", "1"]:
                logical_state = int(state_str)
                break
            else:
                print("Invalid input. Please enter 0 or 1.")
        
        self.encoding_circuit = self.current_code.encode(logical_state=logical_state)
        
        print(f"\n✓ Encoded logical |{logical_state}⟩")
        print(f"  Circuit has {self.encoding_circuit.num_qubits} physical qubits")
        
        # Show encoding circuit
        print("\nEncoding Circuit:")
        print(self.encoding_circuit)
        
        # Show state vector
        print("\nState Vector (Probability Distribution):")
        state = self.backend.get_statevector(self.encoding_circuit)
        probs = np.abs(state.data) ** 2
        n_qubits = state.num_qubits
        for i, prob in enumerate(probs):
            if prob > 0.001:
                state_str = format(i, f'0{n_qubits}b')
                print(f"  |{state_str}⟩: {prob:.4f} ({prob*100:.2f}%)")
        
        input("\nPress Enter to continue...")
    
    def inject_error(self):
        """Let user inject an error."""
        self.print_step(2, "Error Injection")
        
        print("\nAvailable Error Types:")
        error_types = [
            ("1", ErrorType.BIT_FLIP, "Bit-flip (X gate) - flips |0⟩ ↔ |1⟩"),
            ("2", ErrorType.PHASE_FLIP, "Phase-flip (Z gate) - flips phase"),
            ("3", ErrorType.DEPOLARIZING, "Depolarizing - random Pauli error"),
            ("4", ErrorType.ROTATION_X, "Rotation around X-axis (Rx gate)"),
        ]
        
        for num, _, desc in error_types:
            print(f"  {num}. {desc}")
        
        while True:
            error_choice = input("\nSelect error type (1-4): ").strip()
            error_type_map = {num: et for num, et, _ in error_types}
            if error_choice in error_type_map:
                selected_error = error_type_map[error_choice]
                break
            else:
                print("Invalid choice. Please enter 1-4.")
        
        n_qubits = self.current_code.n_qubits
        while True:
            try:
                qubit = int(input(f"\nSelect qubit to apply error (0-{n_qubits-1}): ").strip())
                if 0 <= qubit < n_qubits:
                    break
                else:
                    print(f"Qubit must be between 0 and {n_qubits-1}.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        # For rotation errors, ask for angle
        rotation_angle = None
        if selected_error in [ErrorType.ROTATION_X, ErrorType.ROTATION_Y, ErrorType.ROTATION_Z]:
            while True:
                try:
                    angle_deg = float(input("\nEnter rotation angle in degrees (e.g., 45): ").strip())
                    rotation_angle = np.deg2rad(angle_deg)
                    break
                except ValueError:
                    print("Invalid input. Please enter a number.")
        
        # Inject error
        error_injector = ErrorInjector(n_qubits=n_qubits)
        if rotation_angle is not None:
            self.error_circuit = error_injector.inject_error(
                self.encoding_circuit,
                selected_error,
                qubit=qubit,
                error_probability=1.0,
                rotation_angle=rotation_angle
            )
        else:
            self.error_circuit = error_injector.inject_error(
                self.encoding_circuit,
                selected_error,
                qubit=qubit,
                error_probability=1.0
            )
        
        # Get error description
        error_descriptions = {
            ErrorType.BIT_FLIP: "Bit-flip error (X gate)",
            ErrorType.PHASE_FLIP: "Phase-flip error (Z gate)",
            ErrorType.DEPOLARIZING: "Depolarizing error",
            ErrorType.ROTATION_X: "Rotation error (Rx gate)",
            ErrorType.ROTATION_Y: "Rotation error (Ry gate)",
            ErrorType.ROTATION_Z: "Rotation error (Rz gate)",
        }
        error_name = error_descriptions.get(selected_error, "Unknown error")
        print(f"\n✓ Error injected: {error_name} on qubit {qubit}")
        
        # Show error circuit
        print("\nCircuit with Error:")
        print(self.error_circuit)
        
        # Compare states
        print("\nState Comparison (Before vs After Error):")
        initial_state = self.backend.get_statevector(self.encoding_circuit)
        error_state = self.backend.get_statevector(self.error_circuit)
        fidelity = self.backend.calculate_fidelity(initial_state, error_state)
        
        print(f"  Fidelity: {fidelity:.4f} ({fidelity*100:.2f}%)")
        if fidelity < 0.99:
            print("  ⚠ Error has affected the quantum state!")
        else:
            print("  ✓ State is still intact")
        
        input("\nPress Enter to continue...")
    
    def measure_syndrome(self):
        """Measure the syndrome."""
        self.print_step(3, "Syndrome Measurement")
        
        print("\nSyndrome measurement detects errors by measuring stabilizer operators.")
        print("The syndrome is a bit pattern that tells us where and what type of error occurred.")
        
        syndrome_circuit = self.current_code.syndrome_measurement()
        print("\nSyndrome Measurement Circuit:")
        print(syndrome_circuit)
        
        self.syndrome = self.backend.extract_syndrome(self.error_circuit, syndrome_circuit)
        
        print(f"\n✓ Syndrome measured: {self.syndrome}")
        print(f"  Syndrome value: {''.join(map(str, self.syndrome))}")
        
        input("\nPress Enter to continue...")
    
    def apply_correction(self):
        """Apply error correction."""
        self.print_step(4, "Error Correction")
        
        print(f"\nBased on syndrome {self.syndrome}, we determine which correction to apply.")
        
        correction_circuit = self.current_code.correct(self.syndrome)
        print("\nCorrection Circuit:")
        print(correction_circuit)
        
        # Apply correction
        corrected_circuit = self.error_circuit.copy()
        corrected_circuit = corrected_circuit.compose(correction_circuit)
        
        # Compare states
        print("\nState Comparison (After Error vs After Correction):")
        initial_state = self.backend.get_statevector(self.encoding_circuit)
        error_state = self.backend.get_statevector(self.error_circuit)
        corrected_state = self.backend.get_statevector(corrected_circuit)
        
        fidelity_before = self.backend.calculate_fidelity(initial_state, error_state)
        fidelity_after = self.backend.calculate_fidelity(initial_state, corrected_state)
        
        print(f"  Fidelity before correction: {fidelity_before:.4f} ({fidelity_before*100:.2f}%)")
        print(f"  Fidelity after correction:  {fidelity_after:.4f} ({fidelity_after*100:.2f}%)")
        
        if fidelity_after > 0.99:
            print("\n  ✓✓✓ Error successfully corrected! ✓✓✓")
        else:
            print("\n  ⚠ Correction was not perfect. This may be expected for some error types.")
        
        input("\nPress Enter to continue...")
    
    def decode_and_summarize(self):
        """Decode and show final summary."""
        self.print_step(5, "Decoding and Summary")
        
        print("\nFinally, we decode the corrected state back to the logical qubit.")
        
        decoding_circuit = self.current_code.decode()
        print("\nDecoding Circuit:")
        print(decoding_circuit)
        
        # Full process summary
        self.print_header("Complete QEC Process Summary")
        
        initial_state = self.backend.get_statevector(self.encoding_circuit)
        error_state = self.backend.get_statevector(self.error_circuit)
        
        # Apply correction
        corrected_circuit = self.error_circuit.copy()
        correction_circuit = self.current_code.correct(self.syndrome)
        corrected_circuit = corrected_circuit.compose(correction_circuit)
        corrected_state = self.backend.get_statevector(corrected_circuit)
        
        # Final state after decoding
        final_circuit = corrected_circuit.copy()
        final_circuit = final_circuit.compose(decoding_circuit)
        final_state = self.backend.get_statevector(final_circuit)
        
        fidelities = [
            ("After Encoding", 1.0),
            ("After Error", self.backend.calculate_fidelity(initial_state, error_state)),
            ("After Correction", self.backend.calculate_fidelity(initial_state, corrected_state)),
            ("After Decoding", abs(initial_state.inner(final_state)) ** 2)
        ]
        
        print("\nFidelity at Each Stage:")
        for stage, fidelity in fidelities:
            bar_length = int(fidelity * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"  {stage:20s} {bar} {fidelity:.4f} ({fidelity*100:.2f}%)")
        
        print("\n" + "=" * 70)
        print("  Process Complete!")
        print("=" * 70)
    
    def run(self):
        """Run the interactive tool."""
        print("\n" + "=" * 70)
        print("  Quantum Error Correction - Interactive Visualization Tool")
        print("=" * 70)
        print("\nThis tool will guide you through the QEC process step-by-step.")
        print("At each step, you'll see what's happening and can make choices.")
        
        input("\nPress Enter to start...")
        
        self.select_code()
        self.encode_logical_qubit()
        self.inject_error()
        self.measure_syndrome()
        self.apply_correction()
        self.decode_and_summarize()
        
        print("\nThank you for using the QEC Visualization Tool!")
        print("Run this script again to try different codes or error types.")


if __name__ == "__main__":
    tool = InteractiveQECTool()
    tool.run()

