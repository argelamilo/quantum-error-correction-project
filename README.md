# Quantum Error Correction Visualization Tool

An interactive, educational visualization tool that demonstrates quantum error correction codes, allowing users to inject errors, observe error detection, and visualize the correction process in real-time.

## Project Overview

This project implements a comprehensive quantum error correction (QEC) visualization tool designed for educational purposes. It provides:

- **Multiple QEC Codes**: 3-qubit bit-flip code and 5-qubit perfect code
- **Error Injection**: Support for various error types including bit-flip, phase-flip, depolarizing, and rotation errors (Rx, Ry, Rz gates)
- **Interactive Visualizations**: Circuit diagrams, state vector plots, probability distributions, and fidelity analysis
- **Step-by-Step Process**: Animated visualization of the complete QEC workflow
- **Educational Content**: Clear explanations and tooltips for each step

## Project Members

- Serxhio Dosku
- Argela Milo

## Grading Scheme

**Balanced** - A comprehensive project demonstrating understanding of quantum error correction fundamentals with practical implementation and visualization.

## Features

### Implemented QEC Codes

1. **3-qubit Bit-Flip Code**
   - Simplest QEC code, ideal for beginners
   - Encodes one logical qubit into three physical qubits
   - Detects and corrects single bit-flip errors

2. **5-qubit Perfect Code**
   - The smallest code that can correct arbitrary single-qubit errors
   - Encodes one logical qubit into five physical qubits
   - Demonstrates full error correction capabilities

### Error Types

- **Bit-Flip (X)**: Flips |0⟩ ↔ |1⟩
- **Phase-Flip (Z)**: Flips phase |+⟩ ↔ |-⟩
- **Y Error**: Combination of bit-flip and phase-flip
- **Depolarizing**: Random Pauli error (X, Y, or Z)
- **Rotation Errors**: Rx, Ry, Rz gates (as per professor feedback)

### Visualization Components

- **Circuit Diagrams**: Visual representation of encoding, error injection, syndrome measurement, correction, and decoding circuits
- **State Vector Visualization**: Probability distributions showing the quantum state at each stage
- **Fidelity Metrics**: Quantitative analysis of error correction effectiveness
- **Step-by-Step Animation**: Complete workflow visualization
- **Bloch Sphere**: Single-qubit state visualization (for multi-qubit states, individual qubits can be traced out)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd quantum-error-correction-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import qiskit; print('Qiskit version:', qiskit.__version__)"
```

## Usage

### Recommended: Interactive Tool (Easiest to Understand)

For the clearest, step-by-step experience:

```bash
python examples/interactive_qec_tool.py
```

This interactive tool:
- Guides you through each step with clear explanations
- Lets you choose error types and locations
- Shows state vectors and fidelity at each stage
- Explains what's happening in plain language

### Simple Demo (Quick Overview)

For a quick, clear demonstration:

```bash
python examples/simple_demo.py
```

This shows a complete QEC process with:
- Clear step-by-step output
- State vector visualizations (text-based)
- Fidelity progress bars
- Easy-to-understand explanations

### Advanced: Basic Demo Script

For multiple examples:

```bash
python examples/basic_demo.py
```

This will demonstrate:
- 3-qubit bit-flip code with bit-flip error
- Rx gate error (as per professor feedback)
- 5-qubit perfect code with phase-flip error

### Interactive Jupyter Notebook

For an interactive experience, use the Jupyter notebook:

```bash
jupyter notebook examples/interactive_demo.ipynb
```

The notebook provides:
- Interactive widgets to control error types, locations, and probabilities
- Real-time visualization updates
- Step-by-step QEC process visualization
- Fidelity analysis for different error types

### Programmatic Usage

```python
from qec_visualizer import BitFlipCode, ErrorInjector, ErrorType, QECVisualizer, QECBackend

# Initialize components
code = BitFlipCode()
error_injector = ErrorInjector(n_qubits=3)
visualizer = QECVisualizer()
backend = QECBackend()

# Encode logical |1⟩
encoding_circuit = code.encode(logical_state=1)

# Inject bit-flip error on qubit 0
error_circuit = error_injector.inject_error(
    encoding_circuit,
    ErrorType.BIT_FLIP,
    qubit=0,
    error_probability=1.0
)

# Measure syndrome
syndrome_circuit = code.syndrome_measurement()
syndrome = backend.extract_syndrome(error_circuit, syndrome_circuit)

# Apply correction
correction_circuit = code.correct(syndrome)

# Visualize
visualizer.plot_circuit_diagram(encoding_circuit, title="Encoding Circuit")

# Full simulation
result = backend.simulate_full_qec_process(
    encoding_circuit,
    error_circuit,
    syndrome_circuit,
    correction_circuit,
    code.decode()
)

print(f"Fidelity before correction: {result['fidelity_before']:.4f}")
print(f"Fidelity after correction: {result['fidelity_after']:.4f}")
```

## Project Structure

```
quantum-error-correction-project/
├── qec_visualizer/          # Main package
│   ├── __init__.py          # Package initialization
│   ├── qec_codes.py         # QEC code implementations
│   ├── error_injection.py   # Error injection system
│   ├── backend.py           # Backend logic and simulations
│   └── visualizer.py        # Visualization components
├── examples/                # Example scripts and notebooks
│   ├── basic_demo.py        # Basic demonstration script
│   └── interactive_demo.ipynb # Interactive Jupyter notebook
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore file
└── README.md               # This file
```

## Normal Workflow

1. **Code Selection**: Choose between 3-qubit bit-flip code or 5-qubit perfect code
2. **Encoding**: Encode a logical qubit (|0⟩ or |1⟩) into physical qubits
3. **Error Injection**:
   - Select error type (bit-flip, phase-flip, depolarizing, rotation)
   - Choose error location (which qubit)
   - Set error probability
4. **Syndrome Measurement**: Measure stabilizers to detect errors
5. **Error Correction**: Apply correction based on syndrome
6. **Decoding**: Decode to recover the logical qubit
7. **Visualization**: View circuit diagrams, state vectors, and fidelity metrics

## Key Components

### QEC Codes (`qec_codes.py`)

- `QECCode`: Base class for QEC codes
- `BitFlipCode`: 3-qubit bit-flip code implementation
- `PerfectCode`: 5-qubit perfect code implementation

Each code provides:
- `encode()`: Encoding circuit
- `syndrome_measurement()`: Syndrome measurement circuit
- `correct()`: Error correction circuit
- `decode()`: Decoding circuit

### Error Injection (`error_injection.py`)

- `ErrorType`: Enumeration of error types
- `ErrorInjector`: Handles error injection into circuits

Supports:
- Single error injection
- Multiple error injection
- Probabilistic errors
- Rotation errors with custom angles

### Backend (`backend.py`)

- `QECBackend`: Simulation and calculation backend

Features:
- State vector calculations
- Syndrome extraction
- Fidelity calculations
- Full QEC process simulation
- Bloch sphere coordinate extraction

### Visualizer (`visualizer.py`)

- `QECVisualizer`: Visualization tools

Capabilities:
- Circuit diagram plotting
- State vector probability distributions
- Multi-state comparisons
- Fidelity evolution plots
- Bloch sphere visualization
- Step-by-step process visualization
- Syndrome lookup tables

## Example Scenarios

### Scenario 1: Bit-Flip Error Correction

```python
code = BitFlipCode()
# Encode |1⟩ → |111⟩
# Inject X error on qubit 0 → |011⟩
# Measure syndrome → [1, 0]
# Apply X correction on qubit 0 → |111⟩
# Decode → |1⟩ ✓
```

### Scenario 2: Rx Gate Error (Professor Feedback)

```python
code = BitFlipCode()
# Encode |0⟩ → |000⟩
# Inject Rx(π/4) error on qubit 1
# Observe state evolution
# Note: Bit-flip code doesn't correct rotation errors perfectly
```

### Scenario 3: Perfect Code with Phase-Flip

```python
code = PerfectCode()
# Encode |0⟩ into 5 qubits
# Inject Z error on qubit 2
# Measure 4-bit syndrome
# Apply correction based on syndrome
# Decode successfully
```

## Dependencies

- **qiskit** (≥0.45.0): Quantum circuit framework
- **qiskit-aer** (≥0.13.0): Quantum simulator
- **qiskit-visualization** (≥0.7.0): Circuit visualization
- **numpy** (≥1.24.0): Numerical computations
- **matplotlib** (≥3.7.0): Plotting and visualization
- **ipywidgets** (≥8.0.0): Interactive widgets for Jupyter
- **scipy** (≥1.10.0): Scientific computing

## Future Enhancements

Potential improvements for future work:

- Additional QEC codes (Shor code, Steane code, etc.)
- Real hardware integration (IBM Quantum, etc.)
- More sophisticated error models
- Performance benchmarking
- Web-based interface
- Educational tutorials and explanations
- Error threshold calculations

## References

- Nielsen & Chuang, "Quantum Computation and Quantum Information"
- Qiskit Textbook: Quantum Error Correction
- Various QEC code implementations and research papers

## License

This project is created for educational purposes as part of a quantum computing course.

## Contact

For questions or feedback, please contact the project members:
- Serxhio Dosku
- Argela Milo

---

**Note**: This project is designed for educational purposes to demonstrate understanding of quantum error correction concepts. The implementation focuses on clarity and educational value rather than production-level optimization.
