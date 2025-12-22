# Quick Start Guide

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify installation:**
```bash
python test_basic.py
```

You should see:
```
✓ Imports successful
✓ Created 3-qubit Bit-Flip Code
✓ Encoding circuit created: 3 qubits
✓ Error injection successful
✓ Fidelity calculation: 0.0000
✓ All basic tests passed!
```

## Which Tool Should I Use?

### 🎯 **For First-Time Users** (Recommended)

**Interactive Step-by-Step Tool:**
```bash
python examples/interactive_qec_tool.py
```

**Why use this?**
- ✅ Guided step-by-step process
- ✅ Explains what's happening at each stage
- ✅ You choose the error type and location
- ✅ Clear explanations in plain language
- ✅ Shows state vectors and fidelity

**What you'll see:**
- Code selection menu
- Encoding explanation
- Error injection with choices
- Syndrome measurement explanation
- Correction demonstration
- Complete summary with fidelity

---

### 📊 **For Quick Understanding**

**Simple Demo:**
```bash
python examples/simple_demo.py
```

**Why use this?**
- ✅ Fast overview of the complete process
- ✅ Text-based visualizations (state vectors as bars)
- ✅ No user input required
- ✅ Clear progress indicators
- ✅ Perfect for understanding the workflow

**What you'll see:**
- Complete QEC process in one run
- State vector probabilities as progress bars
- Fidelity evolution chart
- Key takeaways summary

---

### 🔬 **For Exploring Multiple Examples**

**Basic Demo:**
```bash
python examples/basic_demo.py
```

**Why use this?**
- ✅ Multiple examples in one run
- ✅ Shows different error types
- ✅ Demonstrates both QEC codes
- ✅ Includes Rx gate error (professor feedback)

---

### 📓 **For Interactive Experimentation**

**Jupyter Notebook:**
```bash
jupyter notebook examples/interactive_demo.ipynb
```

**Why use this?**
- ✅ Interactive widgets (sliders, dropdowns)
- ✅ Real-time visualization updates
- ✅ Experiment with different parameters
- ✅ See plots and graphs
- ✅ Great for presentations

## Understanding the Output

### State Vector Display

When you see something like:
```
|000⟩: 0.0000 (0.00%)
|111⟩: 1.0000 (100.00%)
```

This shows:
- **|000⟩**: The quantum state where all three qubits are |0⟩
- **0.0000**: Probability amplitude (0 means impossible)
- **(0.00%)**: Percentage probability
- **|111⟩: 1.0000 (100.00%)**: The state |111⟩ has 100% probability

### Fidelity

Fidelity measures how close two quantum states are:
- **1.0 (100%)**: Perfect match
- **0.0 (0%)**: Completely different
- **0.9 (90%)**: Very similar

In QEC:
- After encoding: Fidelity = 1.0 ✓
- After error: Fidelity < 1.0 ✗
- After correction: Fidelity should return to ~1.0 ✓

### Syndrome

The syndrome is a bit pattern that tells us:
- **Where** the error occurred (which qubit)
- **What type** of error it was (sometimes)

Examples:
- `[0, 0]`: No error detected
- `[1, 0]`: Error on qubit 0
- `[1, 1]`: Error on qubit 1

## Common Questions

**Q: Which error should I try first?**  
A: Start with "Bit-Flip" error - it's the easiest to understand.

**Q: What's the difference between the codes?**  
A: 
- **3-qubit Bit-Flip Code**: Simpler, only corrects bit-flip errors
- **5-qubit Perfect Code**: More complex, corrects any single-qubit error

**Q: Why does fidelity sometimes not return to 1.0?**  
A: Some error types (like rotation errors) can't be perfectly corrected by all codes. The bit-flip code only corrects bit-flip errors perfectly.

**Q: Can I see the actual quantum circuits?**  
A: Yes! The tools print the circuit diagrams. The Jupyter notebook also shows visual circuit diagrams.

## Troubleshooting

### Import Errors

If you get import errors, make sure you're in the project root directory:
```bash
cd quantum-error-correction-project
python examples/interactive_qec_tool.py
```

### Qiskit Version Issues

If you encounter Qiskit-related errors:
```bash
pip install --upgrade qiskit[visualization] qiskit-aer
```

### Matplotlib Display Issues

If plots don't show:
- On Linux: Make sure you have a display server running
- On Windows/Mac: Should work out of the box
- In Jupyter: Plots should display automatically

## Next Steps

1. **Try the Interactive Tool** - Best starting point
2. **Read the README.md** - Detailed documentation
3. **Experiment with the Notebook** - Interactive widgets
4. **Try Different Errors** - See how each code handles them
5. **Modify the Code** - Add your own experiments!

## Tips for Understanding QEC

1. **Start Simple**: Use the 3-qubit bit-flip code first
2. **Follow the Steps**: Encoding → Error → Syndrome → Correction → Decoding
3. **Watch the Fidelity**: It tells you if correction worked
4. **Check the Syndrome**: It tells you what correction to apply
5. **Compare States**: See how states change at each step

Happy exploring! 🚀
