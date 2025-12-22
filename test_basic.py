"""
Basic test script to verify the installation and basic functionality.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from qec_visualizer import BitFlipCode, ErrorInjector, ErrorType, QECBackend
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

try:
    # Test basic functionality
    code = BitFlipCode()
    print(f"✓ Created {code.name}")
    
    encoding_circuit = code.encode(logical_state=0)
    print(f"✓ Encoding circuit created: {encoding_circuit.num_qubits} qubits")
    
    error_injector = ErrorInjector(n_qubits=3)
    error_circuit = error_injector.inject_error(
        encoding_circuit, ErrorType.BIT_FLIP, qubit=0, error_probability=1.0
    )
    print("✓ Error injection successful")
    
    backend = QECBackend()
    initial_state = backend.get_statevector(encoding_circuit)
    error_state = backend.get_statevector(error_circuit)
    fidelity = backend.calculate_fidelity(initial_state, error_state)
    print(f"✓ Fidelity calculation: {fidelity:.4f}")
    
    print("\n✓ All basic tests passed!")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

