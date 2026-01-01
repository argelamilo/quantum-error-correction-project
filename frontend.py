"""
Quantum Error Correction - Beginner-Friendly Web Interface

A beautiful, interactive web frontend for exploring quantum error correction.
Built with Streamlit for an intuitive, educational experience.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from qec_visualizer import (
    BitFlipCode, PerfectCode, ErrorInjector, ErrorType,
    QECVisualizer, QECBackend, InteractiveVisualizer
)

# Page configuration
st.set_page_config(
    page_title="Quantum Error Correction Explorer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Modern Header Styles */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -1px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .step-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        color: #1a1a1a;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .step-box h2 {
        color: #1565c0;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    .step-box p {
        color: #1a1a1a;
        margin: 0.5rem 0;
    }
    .step-box strong {
        color: #0d47a1;
    }
    .success-box {
        background: linear-gradient(135deg, #c8e6c9 0%, #a5d6a7 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
        color: #1a1a1a;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .success-box h3 {
        color: #1b5e20;
        margin-top: 0;
    }
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
        color: #1a1a1a;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .info-box strong {
        color: #01579b;
    }
    .info-box h3 {
        color: #01579b;
        margin-top: 0;
    }
    .info-box code {
        background-color: #0277bd;
        color: #ffffff;
        padding: 0.2rem 0.4rem;
        border-radius: 3px;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .circuit-container {
        width: 100%;
        max-width: 100%;
        overflow-x: auto;
        padding: 1rem 0;
        margin: 1rem 0;
    }
    .circuit-container img {
        max-width: 100%;
        height: auto;
        display: block;
        margin: 0 auto;
    }
    /* Ensure Streamlit images are responsive */
    .stImage > img {
        max-width: 100% !important;
        height: auto !important;
    }
    /* Style for circuit tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.75rem 1.5rem;
    }
    
    /* Modern sidebar styling */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Better button styling with hover effects */
    .stButton > button {
        border-radius: 8px;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Better caption styling */
    .stCaption {
        font-size: 0.85rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'code' not in st.session_state:
    st.session_state.code = None
if 'encoding_circuit' not in st.session_state:
    st.session_state.encoding_circuit = None
if 'error_circuit' not in st.session_state:
    st.session_state.error_circuit = None
if 'syndrome' not in st.session_state:
    st.session_state.syndrome = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'last_error_type' not in st.session_state:
    st.session_state.last_error_type = None
if 'last_error_qubit' not in st.session_state:
    st.session_state.last_error_qubit = None
if 'last_error_prob' not in st.session_state:
    st.session_state.last_error_prob = 1.0
if 'error_was_applied' not in st.session_state:
    st.session_state.error_was_applied = True

# Initialize backend and visualizer
@st.cache_resource
def get_backend():
    return QECBackend()

@st.cache_resource
def get_visualizer():
    return QECVisualizer()

@st.cache_resource
def get_interactive_visualizer():
    return InteractiveVisualizer()

backend = get_backend()
visualizer = get_visualizer()
interactive_vis = get_interactive_visualizer()

# Helper function to convert error names to PanQEC format (X, Y, Z)
def convert_error_name_to_panqec(error_name: str) -> Optional[str]:
    """
    Convert error name from frontend format to PanQEC format.
    
    Args:
        error_name: Error name like "Bit-Flip (X)", "Phase-Flip (Z)", etc.
        
    Returns:
        PanQEC format error type: 'X', 'Y', 'Z', or None
    """
    if not error_name:
        return None
    error_name_lower = error_name.lower()
    if "bit-flip" in error_name_lower or error_name_lower.endswith("(x)") or error_name_lower == "x":
        return "X"
    elif "phase-flip" in error_name_lower or error_name_lower.endswith("(z)") or error_name_lower == "z":
        return "Z"
    elif error_name_lower.endswith("(y)") or error_name_lower == "y":
        return "Y"
    elif "rotation" in error_name_lower:
        # Extract axis from rotation error
        if "rx" in error_name_lower or "rotation x" in error_name_lower:
            return "X"
        elif "ry" in error_name_lower or "rotation y" in error_name_lower:
            return "Y"
        elif "rz" in error_name_lower or "rotation z" in error_name_lower:
            return "Z"
        else:
            return "X"  # Default to X for rotation errors
    elif "depolarizing" in error_name_lower:
        return "X"  # Default to X for depolarizing (it's random anyway)
    return None

def reset_session():
    """Reset the session to start over."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.step = 1

def plot_fidelity_gauge(fidelity, title="Fidelity"):
    """Create a gauge chart for fidelity."""
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(projection='polar'))
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)
    
    # Color based on fidelity
    if fidelity > 0.9:
        color = 'green'
    elif fidelity > 0.5:
        color = 'orange'
    else:
        color = 'red'
    
    ax.fill_between(theta, 0, r, alpha=0.3, color=color)
    ax.plot(theta, r, 'k-', linewidth=2)
    
    # Add needle
    needle_angle = np.pi * (1 - fidelity)
    ax.plot([needle_angle, needle_angle], [0, 1], 'k-', linewidth=3)
    
    # Add labels
    ax.set_ylim(0, 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0, 0, f'{fidelity:.3f}', ha='center', va='center', fontsize=20, fontweight='bold')
    ax.set_title(title, fontsize=14, pad=20)
    
    plt.tight_layout()
    return fig

def plot_state_vector_bars(state, title="State Vector"):
    """Plot state vector as a bar chart."""
    probs = np.abs(state.data) ** 2
    n_qubits = state.num_qubits
    labels = [format(i, f'0{n_qubits}b') for i in range(len(probs))]
    
    fig, ax = plt.subplots(figsize=(7, 4))  # Smaller default size
    bars = ax.bar(labels, probs, color='steelblue', alpha=0.7, edgecolor='navy')
    
    # Add value labels
    for bar, prob in zip(bars, probs):
        if prob > 0.01:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Basis State', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_circuit(circuit, title="Quantum Circuit", figsize_scale=1.0, max_width=12, responsive=True, show_title=False):
    """Plot quantum circuit using Qiskit's visualization with high quality and responsive sizing."""
    try:
        # Calculate appropriate figure size based on circuit complexity
        n_qubits = circuit.num_qubits
        n_gates = circuit.size()
        
        # Responsive sizing: smaller for better display in web interface
        if responsive:
            # Calculate width based on number of gates, but cap it
            estimated_width = min(max(4, n_gates * 0.3), max_width) * figsize_scale
            # Height based on qubits, but keep it reasonable
            estimated_height = min(1.5, min(n_qubits * 0.4, 4)) * figsize_scale
        else:
            estimated_width = 8 * figsize_scale
            estimated_height = min(2, n_qubits * 0.5) * figsize_scale
        
        # Set high DPI for better quality
        plt.rcParams['figure.dpi'] = 350  # High DPI for crisp images
        plt.rcParams['savefig.dpi'] = 350
        plt.rcParams['figure.facecolor'] = 'white'
        
        # Try to use Qiskit's matplotlib drawer with better settings
        try:
            # Use fold parameter to wrap long circuits (fold every 20 gates)
            fold_value = 20 if n_gates > 20 else -1
            
            # Scale for compact but readable display
            draw_scale = max(0.9, 1 * figsize_scale)  # Increased scale for better readability
            
            # Newer Qiskit versions
            fig = circuit.draw(
                output='mpl', 
                style='iqp', 
                fold=fold_value, 
                scale=draw_scale,
                cregbundle=False,  # Don't bundle classical registers
                vertical_compression='medium'  # Medium compression for readability
            )
        except (TypeError, ValueError, AttributeError):
            # Older Qiskit versions or fallback
            try:
                fig = circuit.draw(
                    'mpl', 
                    style='iqp', 
                    fold=fold_value if 'fold_value' in locals() else -1, 
                    scale=draw_scale if 'draw_scale' in locals() else 0.7
                )
            except:
                # If all else fails, create a text representation
                fig, ax = plt.subplots(figsize=(estimated_width, estimated_height), dpi=150)
                ax.axis('off')
                circuit_text = str(circuit)
                ax.text(0.5, 0.5, circuit_text, 
                       ha='left', va='top', fontsize=9, 
                       family='monospace', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                if show_title:
                    fig.suptitle(title, fontsize=10, fontweight='bold')
                return fig
        
        # Set high DPI on the figure
        if hasattr(fig, 'set_dpi'):
            fig.set_dpi(150)
        
        # Adjust figure size to be responsive
        if hasattr(fig, 'set_size_inches'):
            # Ensure the figure fits within reasonable bounds
            current_size = fig.get_size_inches()
            # Scale down if too large
            if current_size[0] > max_width:
                scale_factor = max_width / current_size[0]
                new_width = max_width
                new_height = current_size[1] * scale_factor
                fig.set_size_inches(new_width, new_height)
            else:
                # Use calculated size but ensure it's not too small
                fig.set_size_inches(
                    max(estimated_width, current_size[0] * 0.9), 
                    max(estimated_height, current_size[1] * 0.9)
                )
        
        # Only add title if explicitly requested (removed by default)
        if show_title:
            if hasattr(fig, 'suptitle'):
                fig.suptitle(title, fontsize=10, fontweight='bold', y=0.995)
            elif hasattr(fig, 'canvas'):
                fig.text(0.5, 0.98, title, ha='center', fontsize=10, fontweight='bold')
        
        # Tight layout with padding to prevent cutoff
        plt.tight_layout(pad=1.0)
        return fig
    except Exception as e:
        # Fallback: create a simple text representation
        # Calculate fallback dimensions
        n_qubits = circuit.num_qubits
        fallback_width = min(8, max_width)
        fallback_height = max(1.5, min(n_qubits * 0.4, 4))
        
        fig, ax = plt.subplots(figsize=(fallback_width, fallback_height), dpi=150)
        ax.axis('off')
        circuit_text = str(circuit)
        if not circuit_text.strip() or circuit_text.strip() == '\n'.join([f'q_{i}:' for i in range(circuit.num_qubits)]):
            circuit_text = "Empty circuit (no operations)"
        ax.text(0.05, 0.95, circuit_text, 
               ha='left', va='top', fontsize=9, 
               family='monospace', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        if show_title:
            fig.suptitle(title, fontsize=10, fontweight='bold')
        plt.tight_layout(pad=1.0)
        return fig

def main():
    # Header
    st.markdown('<div class="main-header">⚛️ Quantum Error Correction Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Learn QEC through interactive visualization</div>', unsafe_allow_html=True)
    
    # Modern Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1rem 0;'>
            <h1 style='font-size: 1.8rem; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
                ⚛️ QEC Explorer
            </h1>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        # Modern step navigation with progress indicator
        st.markdown("### 🎯 Progress")
        progress_value = st.session_state.step / 6.0
        st.progress(progress_value)
        st.caption(f"Step {st.session_state.step} of 6")
        st.markdown("---")
        
        # Interactive step buttons
        steps = [
            ("1️⃣", "Select Code", "Choose your QEC code"),
            ("2️⃣", "Encode", "Encode logical qubit"),
            ("3️⃣", "Inject Error", "Simulate an error"),
            ("4️⃣", "Measure", "Detect the error"),
            ("5️⃣", "Correct", "Apply correction"),
            ("6️⃣", "Results", "View outcomes")
        ]
        
        for i, (emoji, name, desc) in enumerate(steps, 1):
            if i == st.session_state.step:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0;
                            color: white; font-weight: bold;'>
                    {emoji} {name} ← Current
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"**{emoji} {name}**")
                st.caption(f"_{desc}_")
        
        st.markdown("---")
        
        # Action buttons
        col_reset, col_info = st.columns(2)
        with col_reset:
            if st.button("🔄 Reset", use_container_width=True, help="Start a new session"):
                reset_session()
                st.rerun()
        
        with col_info:
            if st.button("ℹ️ Help", use_container_width=True, help="Show help information"):
                st.info("💡 Navigate through steps using the progress bar above. Each step builds on the previous one!")
        
        st.markdown("---")
        
        # Quick info section
        with st.expander("📚 Quick Guide", expanded=False):
            st.markdown("""
            **QEC Workflow:**
            1. **Select Code** → Choose your error correction scheme
            2. **Encode** → Protect your quantum information
            3. **Inject Error** → Simulate quantum noise
            4. **Measure Syndrome** → Detect where errors occurred
            5. **Correct** → Fix the errors automatically
            6. **View Results** → See success metrics!
            
            **Tips:**
            - Start with 3-qubit Bit-Flip code for beginners
            - Use 2D view first, then try 3D for better visualization
            - Check the color legend on each page
            """)
    
    # Main content area
    if st.session_state.step == 1:
        step_1_select_code()
    elif st.session_state.step == 2:
        step_2_encode()
    elif st.session_state.step == 3:
        step_3_inject_error()
    elif st.session_state.step == 4:
        step_4_measure_syndrome()
    elif st.session_state.step == 5:
        step_5_apply_correction()
    elif st.session_state.step == 6:
        step_6_view_results()

def step_1_select_code():
    """Step 1: Select QEC Code"""
    st.markdown('<div class="step-box"><h2>Step 1: Select a Quantum Error Correction Code</h2></div>', unsafe_allow_html=True)
    
    # Show code structure previews
    st.markdown("### 📊 Code Structure Preview")
    preview_mode = st.radio("View mode:", ["2D", "3D"], horizontal=True, key="preview_mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔵 3-Qubit Bit-Flip Code")
        st.markdown("""
        **Best for beginners!**
        
        - Encodes 1 logical qubit into 3 physical qubits
        - Detects and corrects single bit-flip errors
        - Simple and easy to understand
        - Perfect for learning the basics
        
        **Use case:** Understanding how QEC works
        """)
        
        # Show 2D/3D preview
        preview_code_1 = BitFlipCode()
        if preview_mode == "2D":
            fig_preview_1 = interactive_vis.visualize_code_2d(preview_code_1, show_stabilizers=True, show_connections=True)
        else:
            fig_preview_1 = interactive_vis.visualize_code_3d(preview_code_1, show_stabilizers=True, show_connections=True)
        st.plotly_chart(fig_preview_1, use_container_width=True, key="preview_1")
        
        if st.button("Select Bit-Flip Code", use_container_width=True, type="primary"):
            st.session_state.code = BitFlipCode()
            st.session_state.step = 2
            st.rerun()
    
    with col2:
        st.markdown("### 🟢 5-Qubit Perfect Code")
        st.markdown("""
        **Advanced code**
        
        - Encodes 1 logical qubit into 5 physical qubits
        - Corrects ANY single-qubit error
        - More powerful but more complex
        - Demonstrates full error correction
        
        **Use case:** Advanced error correction
        """)
        
        # Show 2D/3D preview
        preview_code_2 = PerfectCode()
        if preview_mode == "2D":
            fig_preview_2 = interactive_vis.visualize_code_2d(preview_code_2, show_stabilizers=True, show_connections=True)
        else:
            fig_preview_2 = interactive_vis.visualize_code_3d(preview_code_2, show_stabilizers=True, show_connections=True)
        st.plotly_chart(fig_preview_2, use_container_width=True, key="preview_2")
        
        if st.button("Select Perfect Code", use_container_width=True):
            st.session_state.code = PerfectCode()
            st.session_state.step = 2
            st.rerun()
    
    st.markdown("---")
    
    # Beginner-friendly color legend
    st.markdown("### 🎨 Color Guide (PanQEC Style)")
    col_legend1, col_legend2, col_legend3 = st.columns(3)
    
    with col_legend1:
        st.markdown("""
        **Qubits:**
        - ⚪ Light Gray = No error
        - 🔴 Red = X error (Bit-flip)
        - 🔵 Blue = Z error (Phase-flip)
        - 🟢 Green = Y error or Corrected
        - 🟠 Orange = Rotation error
        """)
    
    with col_legend2:
        st.markdown("""
        **Stabilizers:**
        - ⚪ White = No error detected
        - 🟡 Gold = Error detected!
        - ⚫ Gray = Not measured yet
        """)
    
    with col_legend3:
        st.markdown("""
        **Visualization Tips:**
        - 🔄 **2D Mode**: Click and drag to pan
        - 🌐 **3D Mode**: Click and drag to rotate, scroll to zoom
        - 💡 Hover over elements for details
        """)
    
    st.markdown('<div class="info-box"><strong>💡 Beginner Tip:</strong> Start with the 3-qubit Bit-Flip Code! It\'s the simplest and perfect for learning. Try the <strong>2D view</strong> first to see the structure clearly, then switch to <strong>3D</strong> for a more immersive experience!</div>', unsafe_allow_html=True)

def step_2_encode():
    """Step 2: Encode Logical Qubit"""
    if st.session_state.code is None:
        st.error("Please select a code first!")
        st.session_state.step = 1
        st.rerun()
        return
    
    st.markdown(f'<div class="step-box"><h2>Step 2: Encode Logical Qubit</h2><p>Using: <strong>{st.session_state.code.name}</strong></p></div>', unsafe_allow_html=True)
    
    # Show code structure visualization
    col_vis1, col_vis2 = st.columns(2)
    view_mode = st.radio("Visualization mode:", ["2D", "3D"], horizontal=True, key="encode_view_mode")
    
    with col_vis1:
        if view_mode == "2D":
            fig_code = interactive_vis.visualize_code_2d(st.session_state.code, show_stabilizers=True, show_connections=True)
        else:
            fig_code = interactive_vis.visualize_code_3d(st.session_state.code, show_stabilizers=True, show_connections=True)
        st.plotly_chart(fig_code, use_container_width=True, key="code_structure_encode")
    
    with col_vis2:
        st.markdown("""
        ### 🎓 Understanding the Visualization:
        
        **What you're seeing:**
        - ⚪ **Light circles** = Physical qubits (the quantum bits that store information)
        - 💎 **Diamond shapes** = Stabilizers (special "error detectors" that monitor qubits)
        - **Gray lines** = Connections showing which qubits each stabilizer watches
        
        **How QEC works:**
        1. We encode 1 logical qubit into multiple physical qubits
        2. Stabilizers continuously monitor the qubits
        3. If an error occurs, stabilizers detect it (turn gold)
        4. We use this information to correct the error
        
        **💡 Beginner Tip:** Think of stabilizers like security cameras watching the qubits!
        """)
    
    st.markdown("""
    ### 🎓 What is Encoding? (Beginner-Friendly Explanation)
    
    **Simple Analogy:** Think of encoding like making multiple copies of an important document. If one copy gets damaged, you still have the others!
    
    **In Quantum Terms:**
    - **Logical Qubit**: The original information you want to protect (like a message)
    - **Physical Qubits**: Multiple copies/redundant storage of that information
    - **Encoding**: The process of converting 1 logical qubit into multiple physical qubits
    
    **Why Encoding Matters:**
    - Quantum information is fragile and can easily be corrupted
    - By spreading it across multiple qubits, we create redundancy
    - This redundancy lets us detect and fix errors without losing our information!
    
    **💡 Example:** For the 3-qubit code:
    - Logical |0⟩ becomes |000⟩ (all three qubits in state 0)
    - Logical |1⟩ becomes |111⟩ (all three qubits in state 1)
    - If one qubit flips (0→1), we can detect and fix it!
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Choose Logical State")
        logical_state = st.radio(
            "Select the logical qubit state to encode:",
            options=[0, 1],
            format_func=lambda x: f"|{x}⟩",
            index=0
        )
        
        if st.button("🚀 Encode Qubit", use_container_width=True, type="primary"):
            st.session_state.encoding_circuit = st.session_state.code.encode(logical_state=logical_state)
            st.session_state.step = 3
            st.rerun()
    
    with col2:
        st.markdown("### 📖 Understanding Encoding")
        if st.session_state.code.name == "3-qubit Bit-Flip Code":
            st.markdown("""
            **3-Qubit Bit-Flip Code Encoding:**
            - |0⟩ → |000⟩ (all three qubits in state 0)
            - |1⟩ → |111⟩ (all three qubits in state 1)
            
            This creates redundancy: if one qubit flips, we can detect and correct it!
            """)
        else:
            st.markdown("""
            **5-Qubit Perfect Code Encoding:**
            - Encodes |0⟩ or |1⟩ into a superposition across 5 qubits
            - More complex encoding that protects against all error types
            - Uses stabilizer generators for error detection
            """)
    
    # Show encoding circuit if available - compact but high quality
    if st.session_state.encoding_circuit is not None:
        st.markdown("---")
        st.markdown("### Encoding Circuit")
        fig = plot_circuit(st.session_state.encoding_circuit, "Encoding Circuit", figsize_scale=0.5, max_width=8, show_title=False)
        if hasattr(fig, 'set_size_inches'):
            current_size = fig.get_size_inches()
            fig.set_size_inches(min(current_size[0], 8), min(current_size[1], 3))
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, dpi=200)
        plt.close()
        
        # Show state vector - compact
        state = backend.get_statevector(st.session_state.encoding_circuit)
        fig = plot_state_vector_bars(state, "Encoded State Vector")
        fig.set_size_inches(6, 3)  # Compact size
        plt.tight_layout()
        st.pyplot(fig, dpi=200)
        plt.close()

def step_3_inject_error():
    """Step 3: Inject Error"""
    if st.session_state.encoding_circuit is None:
        st.error("Please encode a qubit first!")
        st.session_state.step = 2
        st.rerun()
        return
    
    st.markdown('<div class="step-box"><h2>Step 3: Inject an Error</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎓 What is Error Injection? (Beginner-Friendly Explanation)
    
    **Why Errors Happen:**
    Quantum computers are extremely sensitive! Errors can occur from:
    - **Noise** from the environment (temperature, electromagnetic fields)
    - **Decoherence** (quantum states naturally "leak" information)
    - **Imperfect gates** (operations aren't 100% accurate)
    - **Interactions** with other qubits
    
    **What We're Doing:**
    We'll **intentionally inject** an error (like flipping a bit) to simulate what happens in real quantum computers. This lets us test if our error correction code works!
    
    **Types of Errors:**
    - **X (Bit-flip)**: Changes |0⟩ to |1⟩ and vice versa (most common)
    - **Z (Phase-flip)**: Flips the phase of quantum states
    - **Y**: Combination of X and Z errors
    - **Rotation errors**: Gradual changes (more realistic, but harder to correct)
    
    **💡 Think of it like:** Testing a car's brakes by intentionally hitting the brake pedal - we want to see if the safety systems work!
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Error Type")
        error_type_map = {
            "Bit-Flip (X)": ErrorType.BIT_FLIP,
            "Phase-Flip (Z)": ErrorType.PHASE_FLIP,
            "Depolarizing": ErrorType.DEPOLARIZING,
            "Rotation X (Rx)": ErrorType.ROTATION_X,
            "Rotation Y (Ry)": ErrorType.ROTATION_Y,
            "Rotation Z (Rz)": ErrorType.ROTATION_Z,
        }
        
        error_name = st.selectbox(
            "Select error type:",
            options=list(error_type_map.keys()),
            help="Choose the type of error to inject"
        )
        error_type = error_type_map[error_name]
        
        # Rotation angle for rotation errors
        rotation_angle = None
        if "Rotation" in error_name:
            angle_deg = st.slider("Rotation angle (degrees):", 0, 180, 45)
            rotation_angle = np.deg2rad(angle_deg)
    
    with col2:
        st.markdown("### Error Location")
        n_qubits = st.session_state.code.n_qubits
        qubit = st.selectbox(
            f"Select qubit (0 to {n_qubits-1}):",
            options=list(range(n_qubits)),
            help="Which qubit should the error affect?"
        )
        
        st.markdown("### Error Probability")
        error_prob = st.slider(
            "Error probability:",
            0.0, 1.0, 1.0, 0.1,
            help="Probability that the error occurs (1.0 = always)"
        )
    
    # Error descriptions
    st.markdown("---")
    error_descriptions = {
        ErrorType.BIT_FLIP: "**Bit-Flip (X)**: Flips |0⟩ ↔ |1⟩. The most common type of error.",
        ErrorType.PHASE_FLIP: "**Phase-Flip (Z)**: Flips the phase |+⟩ ↔ |-⟩. Affects superposition states.",
        ErrorType.DEPOLARIZING: "**Depolarizing**: Randomly applies X, Y, or Z error. Simulates general noise.",
        ErrorType.ROTATION_X: "**Rotation X (Rx)**: Rotates the qubit around the X-axis. Continuous error.",
        ErrorType.ROTATION_Y: "**Rotation Y (Ry)**: Rotates the qubit around the Y-axis. Continuous error.",
        ErrorType.ROTATION_Z: "**Rotation Z (Rz)**: Rotates the qubit around the Z-axis. Phase rotation.",
    }
    st.markdown(f'<div class="info-box">{error_descriptions[error_type]}</div>', unsafe_allow_html=True)
    
    # Show visualization with error preview
    st.markdown("---")
    st.markdown("### 📊 Code Visualization with Error Preview")
    error_view_mode = st.radio("View mode:", ["2D", "3D"], horizontal=True, key="error_view_mode")
    
    # Preview error on visualization - convert error name to PanQEC format (X, Y, Z)
    error_qubits_preview = {qubit} if error_prob >= 1.0 else set()
    error_type_panqec = convert_error_name_to_panqec(error_name)
    error_types_preview = {qubit: error_type_panqec} if error_prob >= 1.0 and error_type_panqec else {}
    
    if error_view_mode == "2D":
        fig_error_preview = interactive_vis.visualize_code_2d(
            st.session_state.code,
            error_qubits=error_qubits_preview,
            error_types=error_types_preview,
            show_stabilizers=True,
            show_connections=True
        )
    else:
        fig_error_preview = interactive_vis.visualize_code_3d(
            st.session_state.code,
            error_qubits=error_qubits_preview,
            error_types=error_types_preview,
            show_stabilizers=True,
            show_connections=True
        )
    st.plotly_chart(fig_error_preview, use_container_width=True, key="error_preview")
    
    if error_prob >= 1.0:
        st.info(f"💡 Preview: Error will be injected on qubit {qubit} ({error_name})")
    
    if st.button("💥 Inject Error", use_container_width=True, type="primary"):
        error_injector = ErrorInjector(n_qubits=n_qubits)
        
        # Store error info for later diagnostics
        st.session_state.last_error_type = error_name
        st.session_state.last_error_qubit = qubit
        st.session_state.last_error_prob = error_prob
        
        # Force error application if probability is 1.0 or very close
        actual_prob = error_prob if error_prob < 1.0 else 1.0
        
        if rotation_angle is not None:
            st.session_state.error_circuit = error_injector.inject_error(
                st.session_state.encoding_circuit,
                error_type,
                qubit=qubit,
                error_probability=actual_prob,
                rotation_angle=rotation_angle
            )
        else:
            st.session_state.error_circuit = error_injector.inject_error(
                st.session_state.encoding_circuit,
                error_type,
                qubit=qubit,
                error_probability=actual_prob
            )
        
        # Verify error was applied by checking the circuit
        error_applied = False
        error_gate_name = None
        
        # Determine what gate name to look for
        if error_type == ErrorType.BIT_FLIP:
            error_gate_name = 'x'
        elif error_type == ErrorType.PHASE_FLIP:
            error_gate_name = 'z'
        elif error_type == ErrorType.Y_ERROR:
            error_gate_name = 'y'
        elif error_type == ErrorType.ROTATION_X:
            error_gate_name = 'rx'
        elif error_type == ErrorType.ROTATION_Y:
            error_gate_name = 'ry'
        elif error_type == ErrorType.ROTATION_Z:
            error_gate_name = 'rz'
        elif error_type == ErrorType.DEPOLARIZING:
            # Depolarizing can be X, Y, or Z
            error_gate_name = ['x', 'y', 'z']
        
        # Check if error gate was added
        if error_gate_name and actual_prob >= 1.0:
            if isinstance(error_gate_name, list):
                # For depolarizing, check for any of the gates
                for instruction in st.session_state.error_circuit.data:
                    if instruction.operation.name in error_gate_name:
                        # Check if it's on the correct qubit
                        for q in instruction.qubits:
                            if hasattr(q, '_index') and q._index == qubit:
                                error_applied = True
                                break
                        if error_applied:
                            break
            else:
                # For specific error types
                for instruction in st.session_state.error_circuit.data:
                    if instruction.operation.name == error_gate_name:
                        # Check if it's on the correct qubit
                        for q in instruction.qubits:
                            if hasattr(q, '_index') and q._index == qubit:
                                error_applied = True
                                break
                        if error_applied:
                            break
        
        # Also verify by checking fidelity (backup method)
        if not error_applied or actual_prob < 1.0:
            initial_state = backend.get_statevector(st.session_state.encoding_circuit)
            error_state = backend.get_statevector(st.session_state.error_circuit)
            fidelity = backend.calculate_fidelity(initial_state, error_state)
            # If fidelity dropped significantly, error was likely applied
            if fidelity < 0.99:
                error_applied = True
        
        st.session_state.error_was_applied = error_applied
        st.session_state.error_gate_found = error_gate_name
        
        # Calculate fidelity
        initial_state = backend.get_statevector(st.session_state.encoding_circuit)
        error_state = backend.get_statevector(st.session_state.error_circuit)
        fidelity = backend.calculate_fidelity(initial_state, error_state)
        
        st.session_state.step = 4
        st.rerun()
    
    # Show comparison if error already injected
    if st.session_state.error_circuit is not None:
        # Show error visualization
        st.markdown("---")
        st.markdown("### 📊 Error Visualization")
        injected_error_view = st.radio("View mode:", ["2D", "3D"], horizontal=True, key="injected_error_view")
        
        error_qubits_injected = {st.session_state.last_error_qubit} if st.session_state.error_was_applied else set()
        error_type_panqec = convert_error_name_to_panqec(st.session_state.last_error_type) if st.session_state.error_was_applied else None
        error_types_injected = {st.session_state.last_error_qubit: error_type_panqec} if st.session_state.error_was_applied and error_type_panqec else {}
        
        if injected_error_view == "2D":
            fig_error_injected = interactive_vis.visualize_code_2d(
                st.session_state.code,
                error_qubits=error_qubits_injected,
                error_types=error_types_injected,
                show_stabilizers=True,
                show_connections=True
            )
        else:
            fig_error_injected = interactive_vis.visualize_code_3d(
                st.session_state.code,
                error_qubits=error_qubits_injected,
                error_types=error_types_injected,
                show_stabilizers=True,
                show_connections=True
            )
        st.plotly_chart(fig_error_injected, use_container_width=True, key="error_injected")
        st.markdown("---")
        st.markdown("### State Comparison")
        
        # Debug: Show what gates are in the error circuit
        with st.expander("🔍 Debug: Error Circuit Analysis", expanded=False):
            st.text("Error Circuit Operations:")
            error_gates = {}
            for i, instruction in enumerate(st.session_state.error_circuit.data):
                gate_name = instruction.operation.name
                qubit_indices = [q._index for q in instruction.qubits if hasattr(q, '_index')]
                if gate_name not in error_gates:
                    error_gates[gate_name] = []
                error_gates[gate_name].extend(qubit_indices)
                st.text(f"  Instruction {i}: {gate_name} on qubits {qubit_indices}")
            
            st.text(f"\nSummary of gates by type:")
            for gate_name, qubits in error_gates.items():
                st.text(f"  {gate_name}: {len(qubits)} occurrences on qubits {set(qubits)}")
            
            # Check if expected error gate is present
            if hasattr(st.session_state, 'error_gate_found') and st.session_state.error_gate_found:
                expected_gate = st.session_state.error_gate_found
                if isinstance(expected_gate, list):
                    found = any(g in error_gates for g in expected_gate)
                else:
                    found = expected_gate in error_gates
                if found:
                    st.success(f"✅ Expected error gate ({expected_gate}) found in circuit!")
                else:
                    st.error(f"❌ Expected error gate ({expected_gate}) NOT found in circuit!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            initial_state = backend.get_statevector(st.session_state.encoding_circuit)
            fig1 = plot_state_vector_bars(initial_state, "Before Error")
            fig1.set_size_inches(5, 2.5)  # Compact
            plt.tight_layout()
            st.pyplot(fig1, dpi=200)
            plt.close()
        
        with col2:
            error_state = backend.get_statevector(st.session_state.error_circuit)
            fig2 = plot_state_vector_bars(error_state, "After Error")
            fig2.set_size_inches(5, 2.5)  # Compact
            plt.tight_layout()
            st.pyplot(fig2, dpi=200)
            plt.close()
        
        fidelity = backend.calculate_fidelity(initial_state, error_state)
        st.metric("Fidelity", f"{fidelity:.4f}", f"{(fidelity-1)*100:.2f}%")
        
        if fidelity > 0.99:
            st.warning("⚠️ Fidelity is very high - the error may not have been applied correctly!")
        elif fidelity < 0.01:
            st.info("ℹ️ Fidelity is very low - error was definitely applied!")

def step_4_measure_syndrome():
    """Step 4: Measure Syndrome"""
    if st.session_state.error_circuit is None:
        st.error("Please inject an error first!")
        st.session_state.step = 3
        st.rerun()
        return
    
    st.markdown('<div class="step-box"><h2>Step 4: Measure Syndrome</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### What is Syndrome Measurement?
    
    **Syndrome measurement** is the process of detecting errors in quantum error correction without destroying the encoded quantum information. Here's how it works:
    
    #### Key Concepts:
    
    1. **Stabilizer Operators**: These are special quantum operators that leave valid encoded states unchanged. 
       - For the 3-qubit bit-flip code, we measure **Z₀Z₁** and **Z₁Z₂** (parity checks)
       - These operators detect bit-flip errors by checking if qubits have the same value
    
    2. **Syndrome Bits**: The measurement results give us a syndrome (bit pattern)
       - Each bit tells us whether a stabilizer detected an error
       - **00** = No error detected
       - **10** = Error detected on qubit 0
       - **01** = Error detected on qubit 2  
       - **11** = Error detected on qubit 1
    
    3. **Non-Destructive**: The syndrome measurement doesn't collapse the quantum state - it only tells us about errors
    
    #### How It Works:
    - We use **CNOT gates** to copy parity information into ancilla qubits
    - Measure the ancilla qubits to get the syndrome
    - The syndrome uniquely identifies which qubit has an error
    
    #### Important Note:
    The **3-qubit bit-flip code can only detect bit-flip (X) errors**. 
    - ✅ **Detects**: Bit-flip errors (X gates)
    - ❌ **Cannot detect**: Phase-flip errors (Z gates), rotation errors, or other error types
    
    If you injected a non-bit-flip error, the syndrome might show "00" (no error) even though an error occurred!
    """)
    
    # Show compatibility warning
    if st.session_state.last_error_type and st.session_state.code.name == "3-qubit Bit-Flip Code":
        if "Phase-Flip" in st.session_state.last_error_type or "Rotation" in st.session_state.last_error_type or "Depolarizing" in st.session_state.last_error_type:
            st.warning("⚠️ **Warning**: The 3-qubit bit-flip code can only detect **bit-flip (X) errors**. "
                      f"You injected a **{st.session_state.last_error_type}** error, which this code cannot detect. "
                      "The syndrome may show 'no error' even though an error occurred. Try using a bit-flip error or the 5-qubit perfect code instead.")
    
    if st.button("🔍 Measure Syndrome", use_container_width=True, type="primary"):
        syndrome_circuit = st.session_state.code.syndrome_measurement()
        
        # Debug: Show the syndrome circuit
        with st.expander("🔧 Debug: Syndrome Measurement Circuit", expanded=False):
            st.text("Syndrome Measurement Circuit:")
            st.text(str(syndrome_circuit))
            st.text(f"\nError Circuit (first 20 instructions):")
            error_circuit_str = str(st.session_state.error_circuit)
            st.text('\n'.join(error_circuit_str.split('\n')[:20]))
            
            # Check for all error gates
            error_gates_found = {}
            for i, inst in enumerate(st.session_state.error_circuit.data):
                gate_name = inst.operation.name
                if gate_name in ['x', 'y', 'z', 'rx', 'ry', 'rz']:
                    qubits = [q._index for q in inst.qubits if hasattr(q, '_index')]
                    if gate_name not in error_gates_found:
                        error_gates_found[gate_name] = []
                    error_gates_found[gate_name].extend(qubits)
            
            if error_gates_found:
                st.text(f"\n✓ Found error gates:")
                for gate, qubits in error_gates_found.items():
                    st.text(f"  {gate.upper()} gates on qubits: {set(qubits)}")
            else:
                st.error("❌ No error gates (X, Y, Z, Rx, Ry, Rz) found in error circuit!")
            
            # Show expected error
            if hasattr(st.session_state, 'last_error_type'):
                st.text(f"\nExpected error: {st.session_state.last_error_type} on qubit {st.session_state.last_error_qubit}")
        
        syndrome = backend.extract_syndrome(
            st.session_state.error_circuit,
            syndrome_circuit
        )
        st.session_state.syndrome = syndrome
        st.session_state.step = 5
        st.rerun()
    
    if st.session_state.syndrome is not None:
        st.markdown("---")
        st.markdown("### Syndrome Result")
        
        syndrome_str = ''.join(map(str, st.session_state.syndrome))
        st.markdown(f'<div class="info-box"><h3>Syndrome: <code>{syndrome_str}</code></h3></div>', unsafe_allow_html=True)
        
        # Show syndrome visualization
        st.markdown("### 📊 Syndrome Visualization")
        syndrome_view_mode = st.radio("View mode:", ["2D", "3D"], horizontal=True, key="syndrome_view_mode")
        
        # Convert syndrome list to dict for visualization
        syndrome_dict = {i: st.session_state.syndrome[i] for i in range(len(st.session_state.syndrome))}
        error_qubits_syndrome = {st.session_state.last_error_qubit} if st.session_state.error_was_applied else set()
        error_type_panqec = convert_error_name_to_panqec(st.session_state.last_error_type) if st.session_state.error_was_applied else None
        error_types_syndrome = {st.session_state.last_error_qubit: error_type_panqec} if st.session_state.error_was_applied and error_type_panqec else {}
        
        if syndrome_view_mode == "2D":
            fig_syndrome = interactive_vis.visualize_code_2d(
                st.session_state.code,
                error_qubits=error_qubits_syndrome,
                error_types=error_types_syndrome,
                syndrome_values=syndrome_dict,
                show_stabilizers=True,
                show_connections=True
            )
        else:
            fig_syndrome = interactive_vis.visualize_code_3d(
                st.session_state.code,
                error_qubits=error_qubits_syndrome,
                error_types=error_types_syndrome,
                syndrome_values=syndrome_dict,
                show_stabilizers=True,
                show_connections=True
            )
        st.plotly_chart(fig_syndrome, use_container_width=True, key="syndrome_viz")
        
        st.markdown("""
        **🎨 What the Colors Mean:**
        - 🔴 **Red qubits** = X (bit-flip) errors detected
        - 🔵 **Blue qubits** = Z (phase-flip) errors detected  
        - 🟢 **Green qubits** = Y errors or successfully corrected qubits
        - 🟡 **Gold stabilizers** = Error detected! (Syndrome = 1)
        - ⚪ **White stabilizers** = No error detected (Syndrome = 0)
        
        **💡 Beginner Tip:** When a stabilizer turns gold, it means it "saw" an error on one of its connected qubits!
        """)
        
        # Interpret syndrome
        if st.session_state.code.name == "3-qubit Bit-Flip Code":
            interpretations = {
                (0, 0): "No error detected",
                (1, 0): "Error on qubit 0 → Apply X correction on qubit 0",
                (0, 1): "Error on qubit 2 → Apply X correction on qubit 2",
                (1, 1): "Error on qubit 1 → Apply X correction on qubit 1",
            }
            syndrome_tuple = tuple(st.session_state.syndrome)
            if syndrome_tuple in interpretations:
                st.success(f"**Interpretation:** {interpretations[syndrome_tuple]}")
            
            # Show syndrome lookup table
            st.markdown("---")
            st.markdown("### 📋 Syndrome Lookup Table (3-Qubit Bit-Flip Code)")
            st.markdown("""
            | Syndrome | Meaning | Correction |
            |----------|---------|------------|
            | **00** | No error detected | No correction needed |
            | **10** | Error on qubit 0 | Apply X gate to qubit 0 |
            | **01** | Error on qubit 2 | Apply X gate to qubit 2 |
            | **11** | Error on qubit 1 | Apply X gate to qubit 1 |
            """)
        
        st.markdown("---")
        st.markdown("### Syndrome Measurement Circuit")
        syndrome_circuit = st.session_state.code.syndrome_measurement()
        fig = plot_circuit(syndrome_circuit, "Syndrome Measurement Circuit", figsize_scale=0.5, max_width=8, show_title=False)
        if hasattr(fig, 'set_size_inches'):
            current_size = fig.get_size_inches()
            fig.set_size_inches(min(current_size[0], 8), min(current_size[1], 3))
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, dpi=200)
        plt.close()
        
        # Explain the circuit
        if st.session_state.code.name == "3-qubit Bit-Flip Code":
            st.markdown("""
            **How this circuit works:**
            1. **First measurement (syndrome bit 0)**: Checks parity of qubits 0 and 1 using CNOT gates
            2. **Second measurement (syndrome bit 1)**: Checks parity of qubits 1 and 2 using CNOT gates
            3. The CNOT gates copy parity information, then we measure it
            4. After measurement, we "uncompute" (reverse the CNOTs) to restore the state
            """)

def step_5_apply_correction():
    """Step 5: Apply Correction"""
    if st.session_state.syndrome is None:
        st.error("Please measure syndrome first!")
        st.session_state.step = 4
        st.rerun()
        return
    
    st.markdown('<div class="step-box"><h2>Step 5: Apply Error Correction</h2></div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎓 What is Error Correction? (Beginner-Friendly Explanation)
    
    **Simple Analogy:** If encoding is like making backup copies, error correction is like using those backups to restore your original document when one copy gets damaged!
    
    **The Correction Process:**
    1. **Read the Syndrome**: Like reading GPS coordinates - tells us where the error is
    2. **Apply the Fix**: We apply the **inverse** operation to undo the error
    3. **Verify**: Check if the correction worked (fidelity should increase!)
    
    **How It Works (Simple Version):**
    - If an **X error** flipped a qubit from |0⟩ to |1⟩, we apply **X again** to flip it back to |0⟩
    - Think of it like: "Oops, flipped it the wrong way? Flip it again to fix it!"
    
    **Correction Table (3-Qubit Code):**
    | Syndrome | Meaning | Action |
    |----------|---------|--------|
    | **00** | No error | Do nothing ✓ |
    | **10** | Error on qubit 0 | Apply X to qubit 0 |
    | **01** | Error on qubit 2 | Apply X to qubit 2 |
    | **11** | Error on qubit 1 | Apply X to qubit 1 |
    
    **💡 Why "No Correction Needed"?**
    This happens when:
    - ✅ No error actually occurred (error probability was < 100%)
    - ❌ Wrong error type (code can't detect it - try X error instead!)
    - ⚠️ Error didn't affect stabilizers
    
    **🎯 Success Criteria:**
    After correction, your qubits should turn **green** in the visualization, and fidelity should be close to 1.0!
    """)
    
    # Show syndrome and what correction will be applied
    if st.session_state.syndrome is not None:
        syndrome_str = ''.join(map(str, st.session_state.syndrome))
        st.markdown(f'**Current Syndrome:** `{syndrome_str}`')
        
        # Show what error was injected for context
        if st.session_state.last_error_type:
            error_info = f'**Injected Error:** {st.session_state.last_error_type} on qubit {st.session_state.last_error_qubit}'
            if hasattr(st.session_state, 'last_error_prob'):
                error_info += f' (probability: {st.session_state.last_error_prob})'
            st.markdown(error_info)
            
            # Show if error was actually applied
            if hasattr(st.session_state, 'error_was_applied'):
                if st.session_state.error_was_applied:
                    st.success("✅ Error was successfully applied to the circuit")
                else:
                    st.warning("⚠️ Error may not have been applied (check error probability)")
        
        # Preview correction circuit
        preview_correction = st.session_state.code.correct(st.session_state.syndrome)
        if preview_correction.size() == 0:
            st.info("**Preview:** No correction needed - syndrome indicates no error was detected.")
            
            # Diagnostic information
            st.markdown("#### 🔍 Why No Correction?")
            diagnostic_info = []
            
            if st.session_state.last_error_type:
                if st.session_state.code.name == "3-qubit Bit-Flip Code":
                    if "Bit-Flip" not in st.session_state.last_error_type:
                        diagnostic_info.append(
                            f"❌ **Error Type Mismatch**: You injected a **{st.session_state.last_error_type}** error, "
                            "but the 3-qubit bit-flip code can only detect **bit-flip (X) errors**. "
                            "This is why the syndrome shows no error - the code cannot detect this error type!"
                        )
                    else:
                        # Bit-flip error was used
                        error_prob = getattr(st.session_state, 'last_error_prob', 1.0)
                        if error_prob < 1.0:
                            diagnostic_info.append(
                                f"⚠️ **Error Probability**: You set error probability to {error_prob}. "
                                "With probability < 1.0, sometimes no error occurs. Try setting it to 1.0."
                            )
                        
                        if hasattr(st.session_state, 'error_was_applied') and not st.session_state.error_was_applied:
                            diagnostic_info.append(
                                "❌ **Error Not Applied**: The error may not have been applied to the circuit. "
                                "This could happen if error probability < 1.0 and the random check failed."
                            )
                        else:
                            diagnostic_info.append(
                                "✅ Error was applied, but syndrome shows [0,0]. This might indicate:"
                            )
                            diagnostic_info.append("  - An issue with syndrome measurement (check circuit composition)")
                            diagnostic_info.append("  - The error didn't affect the stabilizer measurements")
                        
                else:
                    diagnostic_info.append(
                        "The 5-qubit perfect code should detect most errors. "
                        "If syndrome is [0,0,0,0], it might mean no error occurred or there's a measurement issue."
                    )
            else:
                diagnostic_info.append("Unable to determine error type - please check the error injection step.")
            
            if diagnostic_info:
                st.markdown('\n'.join(diagnostic_info))
            
            st.markdown("**💡 Suggestion**: Try injecting a **bit-flip (X) error** with **error probability = 1.0** to ensure an error occurs and is detected.")
        else:
            st.markdown("**Preview of correction to be applied:**")
            fig_preview = plot_circuit(preview_correction, "Correction Circuit Preview", figsize_scale=0.4, max_width=7, show_title=False)
            if hasattr(fig_preview, 'set_size_inches'):
                current_size = fig_preview.get_size_inches()
                fig_preview.set_size_inches(min(current_size[0], 7), min(current_size[1], 2.5))
            plt.tight_layout(pad=0.5)
            st.pyplot(fig_preview, dpi=200)
            plt.close()
            st.markdown(f"**Operations:** {preview_correction.count_ops()}")
            
            # Show what the correction will do
            if st.session_state.code.name == "3-qubit Bit-Flip Code":
                syndrome_tuple = tuple(st.session_state.syndrome)
                correction_map = {
                    (1, 0): "Apply X gate to qubit 0 to flip it back",
                    (0, 1): "Apply X gate to qubit 2 to flip it back",
                    (1, 1): "Apply X gate to qubit 1 to flip it back",
                }
                if syndrome_tuple in correction_map:
                    st.success(f"**Correction Action:** {correction_map[syndrome_tuple]}")
    
    # Show correction preview visualization
    if st.session_state.syndrome is not None:
        st.markdown("---")
        st.markdown("### 📊 Correction Preview Visualization")
        correction_preview_mode = st.radio("View mode:", ["2D", "3D"], horizontal=True, key="correction_preview_mode")
        
        # Determine which qubits need correction based on syndrome
        correction_circuit_preview = st.session_state.code.correct(st.session_state.syndrome)
        corrected_qubits_preview = set()
        # Extract qubit indices from correction circuit
        for instruction in correction_circuit_preview.data:
            for q in instruction.qubits:
                if hasattr(q, '_index'):
                    corrected_qubits_preview.add(q._index)
        
        syndrome_dict_preview = {i: st.session_state.syndrome[i] for i in range(len(st.session_state.syndrome))}
        error_qubits_preview = {st.session_state.last_error_qubit} if st.session_state.error_was_applied else set()
        error_type_panqec = convert_error_name_to_panqec(st.session_state.last_error_type) if st.session_state.error_was_applied else None
        error_types_preview = {st.session_state.last_error_qubit: error_type_panqec} if st.session_state.error_was_applied and error_type_panqec else {}
        
        if correction_preview_mode == "2D":
            fig_correction_preview = interactive_vis.visualize_code_2d(
                st.session_state.code,
                error_qubits=error_qubits_preview,
                error_types=error_types_preview,
                syndrome_values=syndrome_dict_preview,
                corrected_qubits=corrected_qubits_preview,
                show_stabilizers=True,
                show_connections=True
            )
        else:
            fig_correction_preview = interactive_vis.visualize_code_3d(
                st.session_state.code,
                error_qubits=error_qubits_preview,
                error_types=error_types_preview,
                syndrome_values=syndrome_dict_preview,
                corrected_qubits=corrected_qubits_preview,
                show_stabilizers=True,
                show_connections=True
            )
        st.plotly_chart(fig_correction_preview, use_container_width=True, key="correction_preview")
        
        st.markdown("""
        **🎨 Correction Preview Guide:**
        - 🔴 **Red qubits** = Qubits with errors (need correction)
        - 🟢 **Green qubits** = Qubits that will be corrected (correction will be applied here)
        - 🟡 **Gold stabilizers** = Error detected (these triggered the correction)
        - ⚪ **White stabilizers** = No error detected
        
        **💡 Beginner Tip:** The correction circuit will apply gates (like X gates) to the green qubits to fix the errors!
        """)
    
    if st.button("🔧 Apply Correction", use_container_width=True, type="primary"):
        correction_circuit = st.session_state.code.correct(st.session_state.syndrome)
        
        # Apply correction - need to ensure registers match
        corrected_circuit = st.session_state.error_circuit.copy()
        
        # Compose correction circuit, ensuring register compatibility
        try:
            corrected_circuit = corrected_circuit.compose(correction_circuit, qubits=range(corrected_circuit.num_qubits))
        except:
            # Fallback: manually add gates
            for instruction in correction_circuit.data:
                op = instruction.operation
                qubits = [q._index for q in instruction.qubits]
                if all(q < corrected_circuit.num_qubits for q in qubits):
                    corrected_circuit.append(op, qubits)
        
        # Calculate results
        initial_state = backend.get_statevector(st.session_state.encoding_circuit)
        error_state = backend.get_statevector(st.session_state.error_circuit)
        corrected_state = backend.get_statevector(corrected_circuit)
        
        # Decode
        decoding_circuit = st.session_state.code.decode()
        final_circuit = corrected_circuit.copy()
        
        # Compose decoding circuit (without measurements for statevector calculation)
        try:
            # Remove measurements from decoding circuit for statevector
            decoding_no_measure = decoding_circuit.copy()
            decoding_no_measure.remove_final_measurements(inplace=True)
            final_circuit = final_circuit.compose(decoding_no_measure, qubits=range(final_circuit.num_qubits))
        except:
            # Fallback: manually add gates (excluding measurements)
            for instruction in decoding_circuit.data:
                if instruction.operation.name != 'measure':
                    op = instruction.operation
                    qubits = [q._index for q in instruction.qubits]
                    if all(q < final_circuit.num_qubits for q in qubits):
                        final_circuit.append(op, qubits)
        
        final_state = backend.get_statevector(final_circuit)
        
        st.session_state.results = {
            'initial_state': initial_state,
            'error_state': error_state,
            'corrected_state': corrected_state,
            'final_state': final_state,
            'fidelity_before': backend.calculate_fidelity(initial_state, error_state),
            'fidelity_after': backend.calculate_fidelity(initial_state, corrected_state),
            'correction_circuit': correction_circuit,
            'decoding_circuit': decoding_circuit,
        }
        
        st.session_state.step = 6
        st.rerun()

def step_6_view_results():
    """Step 6: View Results"""
    if st.session_state.results is None:
        st.error("Please apply correction first!")
        st.session_state.step = 5
        st.rerun()
        return
    
    st.markdown('<div class="step-box"><h2>Step 6: Results & Summary</h2></div>', unsafe_allow_html=True)
    
    results = st.session_state.results
    
    # Fidelity metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Fidelity (After Error)",
            f"{results['fidelity_before']:.4f}",
            f"{(results['fidelity_before']-1)*100:.2f}%"
        )
    
    with col2:
        st.metric(
            "Fidelity (After Correction)",
            f"{results['fidelity_after']:.4f}",
            f"{(results['fidelity_after']-1)*100:.2f}%"
        )
    
    with col3:
        improvement = results['fidelity_after'] - results['fidelity_before']
        st.metric(
            "Improvement",
            f"{improvement:.4f}",
            f"{improvement*100:.2f}%"
        )
    
    # Success indicator
    if results['fidelity_after'] > 0.99:
        st.markdown('<div class="success-box"><h3>✅ Success! Error has been corrected!</h3></div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Correction was not perfect. This may be expected for some error types.")
    
    st.markdown("---")
    
    # Show final visualization with corrected state
    st.markdown("### 📊 Final Code State Visualization")
    final_view_mode = st.radio("View mode:", ["2D", "3D"], horizontal=True, key="final_view_mode")
    
    # Determine corrected qubits from correction circuit
    correction_circuit = results['correction_circuit']
    corrected_qubits_final = set()
    for instruction in correction_circuit.data:
        for q in instruction.qubits:
            if hasattr(q, '_index'):
                corrected_qubits_final.add(q._index)
    
    syndrome_dict_final = {i: st.session_state.syndrome[i] for i in range(len(st.session_state.syndrome))} if st.session_state.syndrome else {}
    
    if final_view_mode == "2D":
        fig_final = interactive_vis.visualize_code_2d(
            st.session_state.code,
            error_qubits=set(),  # Errors are now corrected
            error_types={},
            syndrome_values=syndrome_dict_final,
            corrected_qubits=corrected_qubits_final,
            show_stabilizers=True,
            show_connections=True
        )
    else:
        fig_final = interactive_vis.visualize_code_3d(
            st.session_state.code,
            error_qubits=set(),  # Errors are now corrected
            error_types={},
            syndrome_values=syndrome_dict_final,
            corrected_qubits=corrected_qubits_final,
            show_stabilizers=True,
            show_connections=True
        )
    st.plotly_chart(fig_final, use_container_width=True, key="final_viz")
    
    st.markdown("""
    **🎨 Final State Guide:**
    - 🟢 **Green qubits** = Successfully corrected! Errors have been fixed
    - ⚪ **Light gray qubits** = Normal qubits (no errors throughout)
    - 🟡 **Gold stabilizers** = Previously detected errors (now corrected)
    - ⚪ **White stabilizers** = No errors detected
        
    **💡 Success!** If qubits are green, it means the correction worked and your quantum information is safe!
    """)
    
    st.markdown("---")
    
    # State vector visualizations - compact but high quality
    st.markdown("### State Vector Evolution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = plot_state_vector_bars(results['error_state'], "After Error")
        fig1.set_size_inches(5, 2.5)  # Smaller, more compact
        plt.tight_layout()
        st.pyplot(fig1, dpi=200)  # High DPI for quality
        plt.close()
    
    with col2:
        fig2 = plot_state_vector_bars(results['corrected_state'], "After Correction")
        fig2.set_size_inches(5, 2.5)  # Smaller, more compact
        plt.tight_layout()
        st.pyplot(fig2, dpi=200)  # High DPI for quality
        plt.close()
    
    # Fidelity gauge - compact but high quality
    st.markdown("---")
    st.markdown("### Fidelity Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig3 = plot_fidelity_gauge(results['fidelity_before'], "Fidelity (After Error)")
        fig3.set_size_inches(4, 2.5)  # Compact size
        plt.tight_layout()
        st.pyplot(fig3, dpi=200)  # High DPI for quality
        plt.close()
    
    with col2:
        fig4 = plot_fidelity_gauge(results['fidelity_after'], "Fidelity (After Correction)")
        fig4.set_size_inches(4, 2.5)  # Compact size
        plt.tight_layout()
        st.pyplot(fig4, dpi=200)  # High DPI for quality
        plt.close()
    
    # Fidelity evolution chart - compact but high quality
    st.markdown("---")
    st.markdown("### Fidelity Evolution")
    
    stages = ["After Encoding", "After Error", "After Correction"]
    fidelities = [1.0, results['fidelity_before'], results['fidelity_after']]
    
    fig5, ax = plt.subplots(figsize=(7, 3.5))  # Compact chart
    colors = ['green' if f > 0.9 else 'orange' if f > 0.5 else 'red' for f in fidelities]
    bars = ax.bar(stages, fidelities, color=colors, alpha=0.7, edgecolor='black')
    
    for bar, fid in zip(bars, fidelities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{fid:.3f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Fidelity', fontsize=11)
    ax.set_title('Fidelity at Each Stage', fontsize=13, fontweight='bold')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Fidelity')
    ax.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig5, dpi=200)  # High DPI for quality
    plt.close()
    
    # Circuits
    st.markdown("---")
    st.markdown("### 🔌 Complete Process Circuits")
    st.markdown("Visualize the quantum circuits used in each step of the error correction process.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📐 Correction Circuit", "🔄 Decoding Circuit", "🔗 Full Process", "📊 Summary"])
    
    with tab1:
        st.markdown("#### 📐 Correction Circuit")
        st.markdown("The correction circuit applies the necessary gates to fix the detected error.")
        
        correction_circuit = results['correction_circuit']
        
        # Use container to center and constrain the image
        with st.container():
            # Create compact circuit visualization with high quality
            fig = plot_circuit(
                correction_circuit, 
                "Correction Circuit", 
                figsize_scale=0.35,  # Even smaller scale for compact display
                max_width=7,  # Smaller max width
                responsive=True,
                show_title=False
            )
            # Compact display with high quality
            if hasattr(fig, 'set_size_inches'):
                current_size = fig.get_size_inches()
                fig.set_size_inches(min(current_size[0], 7), min(current_size[1], 3))
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, dpi=200)  # High DPI for quality
            plt.close(fig)
        
        st.markdown("---")
        if correction_circuit.size() == 0:
            st.info("**No correction needed** - The syndrome indicated no error was detected.")
        else:
            st.markdown("**Operations in correction circuit:**")
            ops_list = []
            for gate, count in correction_circuit.count_ops().items():
                ops_list.append(f"`{gate}`: {count} time{'s' if count > 1 else ''}")
            st.markdown(" • ".join(ops_list))
    
    with tab2:
        st.markdown("#### 🔄 Decoding Circuit")
        st.markdown("The decoding circuit reverses the encoding process to recover the logical qubit.")
        
        decoding_circuit = results['decoding_circuit']
        
        # Use container to center and constrain the image
        with st.container():
            # Create responsive circuit visualization with high quality, no title, smaller size
            fig = plot_circuit(
                decoding_circuit, 
                "Decoding Circuit", 
                figsize_scale=0.4,  # Smaller scale
                max_width=8,  # Smaller max width
                responsive=True,
                show_title=False  # No title on image
            )
            # Compact display with high quality
            if hasattr(fig, 'set_size_inches'):
                current_size = fig.get_size_inches()
                fig.set_size_inches(min(current_size[0], 7), min(current_size[1], 3))
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, dpi=200)  # High DPI for quality
            plt.close(fig)
        
        st.markdown("---")
        st.markdown("**Operations in decoding circuit:**")
        ops_list = []
        for gate, count in decoding_circuit.count_ops().items():
            ops_list.append(f"`{gate}`: {count} time{'s' if count > 1 else ''}")
        st.markdown(" • ".join(ops_list))
    
    with tab3:
        st.markdown("#### 🔗 Complete Process Circuit")
        st.markdown("This shows the full circuit sequence: **Encoding → Error → Correction**")
        st.markdown("*Note: Decoding is shown separately as it includes measurements.*")
        
        # Create a combined circuit for visualization
        from qiskit import QuantumCircuit
        full_circuit = st.session_state.encoding_circuit.copy()
        full_circuit = full_circuit.compose(st.session_state.error_circuit)
        full_circuit = full_circuit.compose(results['correction_circuit'])
        
        # Use container with scrollable option for very long circuits
        with st.container():
            # For longer circuits, use smaller scale and enable folding
            circuit_size = full_circuit.size()
            scale_factor = 0.3 if circuit_size > 20 else (0.35 if circuit_size > 15 else 0.4)  # Smaller
            max_w = 8 if circuit_size > 25 else (7 if circuit_size > 15 else 7)  # Smaller max width
            
            fig = plot_circuit(
                full_circuit, 
                "Full Process: Encoding + Error + Correction", 
                figsize_scale=scale_factor,
                max_width=max_w,
                responsive=True,
                show_title=False  # No title on image
            )
            # Compact display with high quality
            if hasattr(fig, 'set_size_inches'):
                current_size = fig.get_size_inches()
                fig.set_size_inches(min(current_size[0], 7), min(current_size[1], 4))
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, dpi=200)  # High DPI for quality
            plt.close(fig)
        
        st.markdown("---")
        st.markdown("**Circuit Statistics:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Qubits", full_circuit.num_qubits)
        with col2:
            st.metric("Total Gates", full_circuit.size())
        with col3:
            st.metric("Depth", full_circuit.depth())
    
    with tab4:
        st.markdown("### 📊 Complete Summary")
        st.markdown(f"""
        **QEC Code Used:** {st.session_state.code.name}
        
        **Syndrome Measured:** {''.join(map(str, st.session_state.syndrome))}
        
        **Fidelity Results:**
        - After encoding: 1.0000 (perfect)
        - After error: {results['fidelity_before']:.4f}
        - After correction: {results['fidelity_after']:.4f}
        
        **Success:** {'✅ Yes' if results['fidelity_after'] > 0.99 else '⚠️ Partial'}
        """)
    
    st.markdown("---")
    if st.button("🔄 Start New Experiment", use_container_width=True, type="primary"):
        reset_session()
        st.rerun()

if __name__ == "__main__":
    main()

