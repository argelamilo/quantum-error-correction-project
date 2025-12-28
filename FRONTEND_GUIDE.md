# Quantum Error Correction - Web Frontend Guide

## 🚀 Quick Start

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the web frontend:**
   ```bash
   python -m streamlit run frontend.py
   ```
   
   Or simply use:
   ```bash
   streamlit run frontend.py
   ```

3. **Open your browser:**
   The app will automatically open at `http://localhost:8501`

## 🎯 Features

### Beginner-Friendly Interface
- **Step-by-step workflow** - Guided process through all QEC stages
- **Visual explanations** - Clear descriptions at each step
- **Interactive widgets** - Easy-to-use controls for all parameters
- **Real-time visualizations** - See results instantly

### Complete QEC Workflow
1. **Select QEC Code** - Choose between 3-qubit bit-flip or 5-qubit perfect code
2. **Encode Logical Qubit** - Encode |0⟩ or |1⟩ into physical qubits
3. **Inject Error** - Choose error type, location, and probability
4. **Measure Syndrome** - Detect the error location
5. **Apply Correction** - Fix the error based on syndrome
6. **View Results** - See fidelity metrics and state evolution

### Visualizations
- **State Vector Bars** - Probability distributions at each stage
- **Fidelity Gauges** - Visual fidelity indicators
- **Fidelity Evolution** - Track fidelity through the process
- **Circuit Diagrams** - View quantum circuits at each step

## 📖 Usage Tips

### For Beginners
1. Start with the **3-qubit Bit-Flip Code** - it's simpler to understand
2. Try a **Bit-Flip error** first - it's the easiest to visualize
3. Use **error probability 1.0** to see the full effect
4. Watch how fidelity changes at each step

### For Advanced Users
1. Try the **5-qubit Perfect Code** for full error correction
2. Experiment with **rotation errors** (Rx, Ry, Rz)
3. Adjust **error probabilities** to see partial errors
4. Compare different error types on the same code

## 🎨 Interface Overview

### Sidebar
- **Navigation** - See current step and progress
- **Reset Button** - Start a new experiment
- **About Section** - Quick reference guide

### Main Area
- **Step Instructions** - Clear guidance for each step
- **Interactive Controls** - Sliders, dropdowns, buttons
- **Visualizations** - Charts and graphs
- **Results** - Metrics and summaries

## 🔧 Troubleshooting

### Port Already in Use
If port 8501 is busy, Streamlit will automatically use the next available port.

### Import Errors
Make sure you're in the project root directory:
```bash
cd "path/to/quantum-error-correction-project/Code"
streamlit run frontend.py
```

### Visualization Issues
If plots don't display:
- Make sure matplotlib is installed: `pip install matplotlib`
- Check that your browser supports JavaScript

## 💡 Tips for Best Experience

1. **Use a modern browser** - Chrome, Firefox, or Edge recommended
2. **Full screen mode** - Better for viewing visualizations
3. **Follow the steps** - Don't skip ahead for best understanding
4. **Experiment** - Try different combinations to see how QEC works

## 🎓 Educational Value

This frontend is designed to help you:
- Understand the QEC workflow visually
- See how errors affect quantum states
- Learn how syndrome measurement works
- Observe error correction in action
- Compare different QEC codes

## 📝 Notes

- The frontend uses Streamlit for easy deployment
- All visualizations are generated in real-time
- Session state is maintained throughout your experiment
- You can reset and start over at any time

Enjoy exploring Quantum Error Correction! ⚛️

