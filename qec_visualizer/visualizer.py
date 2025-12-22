"""
Visualization Module

This module handles visualization of quantum circuits, state vectors,
probability distributions, and step-by-step animations of the QEC process.
"""

from typing import List, Optional, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import matplotlib.patches as mpatches

# Try to import Qiskit visualization functions (may not all be available)
try:
    from qiskit.visualization import plot_bloch_multivector
except ImportError:
    plot_bloch_multivector = None


class QECVisualizer:
    """
    Handles visualization of quantum error correction processes.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.current_figure = None
    
    def plot_circuit_diagram(
        self,
        circuit: QuantumCircuit,
        title: str = "Quantum Circuit",
        filename: Optional[str] = None
    ) -> None:
        """
        Plot quantum circuit diagram.
        
        Args:
            circuit: Quantum circuit to visualize
            title: Title for the plot
            filename: Optional filename to save the plot
        """
        try:
            # Use Qiskit's built-in circuit visualization
            # Try different methods for different Qiskit versions
            try:
                # Newer Qiskit versions
                fig = circuit.draw(output='mpl', style='iqp')
            except (TypeError, ValueError):
                # Older Qiskit versions
                fig = circuit.draw('mpl', style='iqp')
            
            if hasattr(fig, 'suptitle'):
                fig.suptitle(title, fontsize=14, fontweight='bold')
            
            if filename:
                fig.savefig(filename, dpi=150, bbox_inches='tight')
            
            plt.show()
        except Exception as e:
            print(f"Error plotting circuit: {e}")
            # Fallback: print text representation
            print(circuit)
    
    def plot_state_vector(
        self,
        statevector: Statevector,
        title: str = "State Vector",
        filename: Optional[str] = None
    ) -> None:
        """
        Plot state vector as probability distribution.
        
        Args:
            statevector: State vector to visualize
            title: Title for the plot
            filename: Optional filename to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Get probabilities
        probabilities = np.abs(statevector.data) ** 2
        
        # Create labels for basis states
        n_qubits = statevector.num_qubits
        labels = [format(i, f'0{n_qubits}b') for i in range(len(probabilities))]
        
        # Plot bar chart
        bars = ax.bar(labels, probabilities, color='steelblue', alpha=0.7)
        ax.set_xlabel('Basis State', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        
        # Add probability values on bars
        for bar, prob in zip(bars, probabilities):
            if prob > 0.01:  # Only show if significant
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{prob:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_probability_comparison(
        self,
        states: List[Statevector],
        labels: List[str],
        title: str = "State Comparison",
        filename: Optional[str] = None
    ) -> None:
        """
        Compare probability distributions of multiple states.
        
        Args:
            states: List of state vectors to compare
            labels: Labels for each state
            title: Title for the plot
            filename: Optional filename to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        n_qubits = states[0].num_qubits
        basis_labels = [format(i, f'0{n_qubits}b') for i in range(2 ** n_qubits)]
        
        x = np.arange(len(basis_labels))
        width = 0.8 / len(states)
        
        for i, (state, label) in enumerate(zip(states, labels)):
            probabilities = np.abs(state.data) ** 2
            offset = (i - len(states) / 2 + 0.5) * width
            ax.bar(x + offset, probabilities, width, label=label, alpha=0.7)
        
        ax.set_xlabel('Basis State', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(basis_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_fidelity_evolution(
        self,
        fidelities: List[float],
        labels: List[str],
        title: str = "Fidelity Evolution",
        filename: Optional[str] = None
    ) -> None:
        """
        Plot fidelity at different stages of QEC process.
        
        Args:
            fidelities: List of fidelity values
            labels: Labels for each stage
            title: Title for the plot
            filename: Optional filename to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(len(labels))
        bars = ax.bar(x, fidelities, color=['green' if f > 0.9 else 'orange' if f > 0.5 else 'red' for f in fidelities], alpha=0.7)
        
        ax.set_xlabel('Stage', fontsize=12)
        ax.set_ylabel('Fidelity', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, fid in zip(bars, fidelities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{fid:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        # Add horizontal line at fidelity = 1.0
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Fidelity')
        ax.legend()
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_bloch_sphere(
        self,
        statevector: Statevector,
        qubit_index: int = 0,
        title: str = "Bloch Sphere",
        filename: Optional[str] = None
    ) -> None:
        """
        Plot state on Bloch sphere.
        
        Args:
            statevector: State vector to visualize
            qubit_index: Index of qubit to plot (for multi-qubit states)
            title: Title for the plot
            filename: Optional filename to save the plot
        """
        try:
            if plot_bloch_multivector is not None:
                if statevector.num_qubits == 1:
                    fig = plot_bloch_multivector(statevector, title=title)
                else:
                    # For multi-qubit, plot the specific qubit
                    # This is a simplified version - full implementation would
                    # properly trace out other qubits
                    fig = plot_bloch_multivector(statevector, title=f"{title} - Qubit {qubit_index}")
                
                if filename:
                    fig.savefig(filename, dpi=150, bbox_inches='tight')
                
                plt.show()
            else:
                print("Bloch sphere visualization not available. Install qiskit[visualization] for full support.")
        except Exception as e:
            print(f"Error plotting Bloch sphere: {e}")
    
    def create_step_by_step_visualization(
        self,
        circuits: List[QuantumCircuit],
        statevectors: List[Statevector],
        stage_names: List[str],
        title: str = "QEC Process Step-by-Step"
    ) -> None:
        """
        Create a comprehensive step-by-step visualization of the QEC process.
        
        Args:
            circuits: List of circuits at each stage
            statevectors: List of state vectors at each stage
            stage_names: Names for each stage
            title: Overall title
        """
        n_stages = len(circuits)
        fig = plt.figure(figsize=(16, 4 * n_stages))
        
        for i, (circuit, state, name) in enumerate(zip(circuits, statevectors, stage_names)):
            # Circuit diagram
            ax1 = plt.subplot(n_stages, 2, 2*i + 1)
            try:
                # Try different draw methods for compatibility
                try:
                    circuit.draw(output='mpl', ax=ax1, style='iqp')
                except (TypeError, ValueError):
                    circuit.draw('mpl', ax=ax1, style='iqp')
            except:
                ax1.text(0.5, 0.5, f'Circuit: {name}', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title(f'{name} - Circuit', fontweight='bold')
            
            # State vector
            ax2 = plt.subplot(n_stages, 2, 2*i + 2)
            probabilities = np.abs(state.data) ** 2
            n_qubits = state.num_qubits
            labels = [format(j, f'0{n_qubits}b') for j in range(len(probabilities))]
            ax2.bar(labels, probabilities, color='steelblue', alpha=0.7)
            ax2.set_xlabel('Basis State')
            ax2.set_ylabel('Probability')
            ax2.set_title(f'{name} - State Vector', fontweight='bold')
            ax2.set_ylim([0, 1.1])
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.show()
    
    def plot_syndrome_table(
        self,
        syndromes: List[List[int]],
        corrections: List[str],
        title: str = "Syndrome Lookup Table",
        filename: Optional[str] = None
    ) -> None:
        """
        Plot a table showing syndrome values and corresponding corrections.
        
        Args:
            syndromes: List of syndrome bit patterns
            corrections: List of correction descriptions
            title: Title for the plot
            filename: Optional filename to save the plot
        """
        fig, ax = plt.subplots(figsize=(10, len(syndromes) * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table_data = []
        for syndrome, correction in zip(syndromes, corrections):
            syndrome_str = ''.join(map(str, syndrome))
            table_data.append([syndrome_str, correction])
        
        table = ax.table(cellText=table_data,
                        colLabels=['Syndrome', 'Correction'],
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(2):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=150, bbox_inches='tight')
        
        plt.show()

