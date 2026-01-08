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

## Grading Scheme - **Balanced** 

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

### 🌟 **NEW: Web Frontend (Most Beginner-Friendly!)**

For the best visual experience with a modern web interface:

```bash
python -m streamlit run frontend.py
```

Or simply:
```bash
streamlit run frontend.py
```

Or use the convenience scripts:
- **Windows:** `run_frontend.bat`
- **Linux/Mac:** `./run_frontend.sh`

The web frontend provides:
- 📊 Real-time visualizations (circuit diagrams, state vectors, fidelity charts)
- 🎯 Step-by-step guided workflow
- 💡 Interactive widgets and clear explanations
- 📈 Fidelity tracking and progress indicators


## Step-by-Step Usage Guide

The web frontend guides you through a complete quantum error correction workflow in six intuitive steps. 

**Step 1: Select Code** - Choose between the 3-qubit Bit-Flip Code (beginner-friendly) or the 5-qubit Perfect Code (advanced), with interactive 2D/3D previews showing the qubit and stabilizer structure. 

**Step 2: Encode** - Select a logical qubit state (|0⟩ or |1⟩) and click "Encode Qubit" to create the encoded state, visualizing how your logical information is distributed across multiple physical qubits. 

**Step 3: Inject Error** - Choose an error type (bit-flip, phase-flip, rotation errors, etc.), select which qubit to affect, and optionally set an error probability, then click "Inject Error" to simulate quantum noise. 

**Step 4: Measure Syndrome** - Click "Measure Syndrome" to detect the error location; the interface displays the syndrome bit pattern and highlights which stabilizers detected the error in the visualization. 

**Step 5: Apply Correction** - Review the detected syndrome and click "Apply Correction" to automatically fix the error based on the syndrome lookup table. 

**Step 6: View Results** - Examine comprehensive results including fidelity metrics (before and after correction), state vector probability distributions, 2D/3D visualizations showing corrected qubits in green, and success indicators confirming whether the error was successfully corrected. Throughout the process, the sidebar shows your progress, and each step includes educational explanations and visualizations to help you understand quantum error correction principles.


## Normal Workflow (Shorter Version)

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

### Scenario 2: Rx Gate Error

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


## License

This project is created for educational purposes as part of a quantum computing course.

---

**Note**: This project is designed for educational purposes to demonstrate understanding of quantum error correction concepts. The implementation focuses on clarity and educational value rather than production-level optimization.
