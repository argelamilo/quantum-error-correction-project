"""
Simple QEC Demo with Clear Step-by-Step Output

A simplified demonstration that shows the QEC process clearly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from qec_visualizer import BitFlipCode, ErrorInjector, ErrorType, QECVisualizer, QECBackend


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_step(num, description):
    """Print a step header."""
    print(f"\n[{num}] {description}")
    print("-" * 80)


def print_state_info(state, title):
    """Print state vector information in a readable format."""
    probs = np.abs(state.data) ** 2
    n_qubits = state.num_qubits
    
    print(f"\n{title}:")
    print("  State Vector Probabilities:")
    for i, prob in enumerate(probs):
        if prob > 0.0001:  # Only show significant probabilities
            state_str = format(i, f'0{n_qubits}b')
            bar = "█" * int(prob * 30)
            print(f"    |{state_str}⟩: {bar} {prob:.4f} ({prob*100:.2f}%)")


def simple_demo():
    """Run a simple, clear demonstration."""
    
    print_section("Quantum Error Correction - Simple Demonstration")
    print("\nThis demo shows how the 3-qubit bit-flip code works.")
    print("We'll encode |1⟩, inject an error, detect it, and correct it.")
    
    # Initialize
    code = BitFlipCode()
    error_injector = ErrorInjector(n_qubits=3)
    backend = QECBackend()
    
    # Step 1: Encoding
    print_step(1, "ENCODING: Logical |1⟩ → Physical |111⟩")
    encoding_circuit = code.encode(logical_state=1)
    print("\nEncoding Circuit:")
    print(encoding_circuit)
    
    initial_state = backend.get_statevector(encoding_circuit)
    print_state_info(initial_state, "Encoded State")
    
    # Step 2: Error
    print_step(2, "ERROR INJECTION: Bit-flip on qubit 0")
    print("\nAn X error (bit-flip) occurs on qubit 0.")
    print("This changes |111⟩ → |011⟩")
    
    error_circuit = error_injector.inject_error(
        encoding_circuit, ErrorType.BIT_FLIP, qubit=0, error_probability=1.0
    )
    
    error_state = backend.get_statevector(error_circuit)
    print_state_info(error_state, "State After Error")
    
    fidelity = backend.calculate_fidelity(initial_state, error_state)
    print(f"\n  Fidelity: {fidelity:.4f} → Error has corrupted the state!")
    
    # Step 3: Syndrome
    print_step(3, "SYNDROME MEASUREMENT: Detect the Error")
    print("\nWe measure stabilizers to detect where the error occurred.")
    
    syndrome_circuit = code.syndrome_measurement()
    print("\nSyndrome Measurement Circuit:")
    print(syndrome_circuit)
    
    syndrome = backend.extract_syndrome(error_circuit, syndrome_circuit)
    print(f"\n  Syndrome measured: {syndrome}")
    print(f"  Syndrome value: {''.join(map(str, syndrome))}")
    
    # Interpret syndrome
    if syndrome == [0, 0]:
        print("  Interpretation: No error detected")
    elif syndrome == [1, 0]:
        print("  Interpretation: Error on qubit 0 → Apply X on qubit 0")
    elif syndrome == [0, 1]:
        print("  Interpretation: Error on qubit 2 → Apply X on qubit 2")
    elif syndrome == [1, 1]:
        print("  Interpretation: Error on qubit 1 → Apply X on qubit 1")
    
    # Step 4: Correction
    print_step(4, "CORRECTION: Fix the Error")
    correction_circuit = code.correct(syndrome)
    print("\nCorrection Circuit:")
    print(correction_circuit)
    
    corrected_circuit = error_circuit.copy()
    corrected_circuit = corrected_circuit.compose(correction_circuit)
    corrected_state = backend.get_statevector(corrected_circuit)
    
    print_state_info(corrected_state, "State After Correction")
    
    fidelity_after = backend.calculate_fidelity(initial_state, corrected_state)
    print(f"\n  Fidelity: {fidelity_after:.4f}")
    
    if fidelity_after > 0.99:
        print("  ✓✓✓ SUCCESS! Error has been corrected! ✓✓✓")
    else:
        print("  ⚠ Correction incomplete")
    
    # Step 5: Summary
    print_step(5, "SUMMARY: Complete Process")
    
    fidelities = [
        ("Initial (Encoded)", 1.0000),
        ("After Error", fidelity),
        ("After Correction", fidelity_after),
    ]
    
    print("\nFidelity Evolution:")
    print("  Stage                 Fidelity    Progress")
    print("  " + "-" * 60)
    for stage, fid in fidelities:
        bar = "█" * int(fid * 40) + "░" * (40 - int(fid * 40))
        print(f"  {stage:20s} {fid:.4f}     {bar}")
    
    print_section("Demo Complete!")
    print("\nKey Takeaways:")
    print("  • Encoding protects logical information across multiple qubits")
    print("  • Errors can be detected through syndrome measurement")
    print("  • The syndrome tells us exactly how to correct the error")
    print("  • After correction, the original state is recovered!")


if __name__ == "__main__":
    simple_demo()

