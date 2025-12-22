"""
Basic Demo Script

This script demonstrates the basic usage of the QEC visualization tool.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qec_visualizer import BitFlipCode, PerfectCode, ErrorInjector, ErrorType, QECVisualizer, QECBackend


def demo_bit_flip_code():
    """Demonstrate 3-qubit bit-flip code."""
    print("=" * 60)
    print("Demo: 3-qubit Bit-Flip Code")
    print("=" * 60)
    
    # Initialize components
    code = BitFlipCode()
    error_injector = ErrorInjector(n_qubits=3)
    visualizer = QECVisualizer()
    backend = QECBackend()
    
    # Step 1: Encode logical |1⟩
    print("\n1. Encoding logical |1⟩ into 3 physical qubits...")
    encoding_circuit = code.encode(logical_state=1)
    print(f"   Encoding circuit created with {encoding_circuit.num_qubits} qubits")
    
    # Step 2: Inject bit-flip error on qubit 0
    print("\n2. Injecting bit-flip error on qubit 0...")
    error_circuit = error_injector.inject_error(
        encoding_circuit,
        ErrorType.BIT_FLIP,
        qubit=0,
        error_probability=1.0
    )
    print("   Error injected successfully")
    
    # Step 3: Measure syndrome
    print("\n3. Measuring syndrome...")
    syndrome_circuit = code.syndrome_measurement()
    syndrome = backend.extract_syndrome(error_circuit, syndrome_circuit)
    print(f"   Syndrome measured: {syndrome}")
    
    # Step 4: Apply correction
    print("\n4. Applying correction based on syndrome...")
    correction_circuit = code.correct(syndrome)
    print("   Correction circuit created")
    
    # Step 5: Decode
    print("\n5. Decoding to recover logical qubit...")
    decoding_circuit = code.decode()
    
    # Visualize
    print("\n6. Generating visualizations...")
    visualizer.plot_circuit_diagram(encoding_circuit, title="Encoding Circuit")
    
    # Full process simulation
    print("\n7. Running full QEC simulation...")
    result = backend.simulate_full_qec_process(
        encoding_circuit,
        error_circuit,
        syndrome_circuit,
        correction_circuit,
        decoding_circuit
    )
    
    print(f"\nResults:")
    print(f"  Fidelity before correction: {result['fidelity_before']:.4f}")
    print(f"  Fidelity after correction: {result['fidelity_after']:.4f}")
    print(f"  Success: {result['success']}")
    
    # Plot fidelity evolution
    visualizer.plot_fidelity_evolution(
        [result['fidelity_before'], result['fidelity_after']],
        ['After Error', 'After Correction'],
        title="Fidelity Evolution - Bit-Flip Code"
    )


def demo_rx_error():
    """Demonstrate Rx gate error (as per professor feedback)."""
    print("\n" + "=" * 60)
    print("Demo: Rx Gate Error (Professor Feedback)")
    print("=" * 60)
    
    code = BitFlipCode()
    error_injector = ErrorInjector(n_qubits=3)
    visualizer = QECVisualizer()
    backend = QECBackend()
    
    # Encode
    encoding_circuit = code.encode(logical_state=0)
    
    # Inject Rx error
    print("\nInjecting Rx(π/4) error on qubit 1...")
    error_circuit = error_injector.inject_error(
        encoding_circuit,
        ErrorType.ROTATION_X,
        qubit=1,
        error_probability=1.0,
        rotation_angle=np.pi / 4
    )
    
    # Visualize states
    initial_state = backend.get_statevector(encoding_circuit)
    error_state = backend.get_statevector(error_circuit)
    
    print("\nComparing states before and after Rx error...")
    visualizer.plot_probability_comparison(
        [initial_state, error_state],
        ['Initial State', 'After Rx Error'],
        title="Effect of Rx Gate Error"
    )
    
    # Calculate fidelity
    fidelity = backend.calculate_fidelity(initial_state, error_state)
    print(f"\nFidelity after Rx error: {fidelity:.4f}")


def demo_perfect_code():
    """Demonstrate 5-qubit perfect code."""
    print("\n" + "=" * 60)
    print("Demo: 5-qubit Perfect Code")
    print("=" * 60)
    
    code = PerfectCode()
    error_injector = ErrorInjector(n_qubits=5)
    visualizer = QECVisualizer()
    backend = QECBackend()
    
    # Encode
    print("\n1. Encoding logical |0⟩ into 5 physical qubits...")
    encoding_circuit = code.encode(logical_state=0)
    
    # Inject error
    print("\n2. Injecting phase-flip error on qubit 2...")
    error_circuit = error_injector.inject_error(
        encoding_circuit,
        ErrorType.PHASE_FLIP,
        qubit=2,
        error_probability=1.0
    )
    
    # Measure syndrome
    print("\n3. Measuring syndrome...")
    syndrome_circuit = code.syndrome_measurement()
    syndrome = backend.extract_syndrome(error_circuit, syndrome_circuit)
    print(f"   Syndrome: {syndrome}")
    
    # Correct
    print("\n4. Applying correction...")
    correction_circuit = code.correct(syndrome)
    
    # Simulate
    result = backend.simulate_full_qec_process(
        encoding_circuit,
        error_circuit,
        syndrome_circuit,
        correction_circuit,
        code.decode()
    )
    
    print(f"\nResults:")
    print(f"  Fidelity before: {result['fidelity_before']:.4f}")
    print(f"  Fidelity after: {result['fidelity_after']:.4f}")
    print(f"  Success: {result['success']}")


if __name__ == "__main__":
    import numpy as np
    
    print("Quantum Error Correction Visualization Tool - Basic Demo")
    print("=" * 60)
    
    # Run demos
    demo_bit_flip_code()
    demo_rx_error()
    demo_perfect_code()
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)

