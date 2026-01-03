"""
Test Suite for 5-Qubit Perfect Code with Various Error Types

This test suite tests the 5-qubit perfect code's ability to detect and correct
different types of errors. Each error type is tested with 1-2 test cases.
"""

import sys
import os

# Add parent directory to path so we can import qec_visualizer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qec_visualizer import PerfectCode, ErrorInjector, ErrorType, QECBackend
import numpy as np
from typing import Dict, List, Tuple


class PerfectCodeErrorTests:
    """Test suite for 5-qubit perfect code error correction."""
    
    def __init__(self):
        self.code = PerfectCode()
        self.backend = QECBackend()
        self.error_injector = ErrorInjector(n_qubits=5)
        self.results: List[Dict] = []
    
    def run_single_test(
        self,
        error_type: ErrorType,
        qubit: int,
        logical_state: int = 0,
        test_name: str = "",
        rotation_angle: float = None
    ) -> Dict:
        """
        Run a single test case.
        
        Returns:
            Dictionary with test results
        """
        # Encode
        encoding_circuit = self.code.encode(logical_state=logical_state)
        
        # Inject error
        error_circuit = self.error_injector.inject_error(
            encoding_circuit,
            error_type,
            qubit=qubit,
            error_probability=1.0,
            rotation_angle=rotation_angle
        )
        
        # Measure syndrome
        syndrome_circuit = self.code.syndrome_measurement()
        syndrome = self.backend.extract_syndrome(error_circuit, syndrome_circuit)
        
        # Apply correction
        correction_circuit = self.code.correct(syndrome)
        
        # Simulate full process
        result = self.backend.simulate_full_qec_process(
            encoding_circuit,
            error_circuit,
            syndrome_circuit,
            correction_circuit,
            self.code.decode()
        )
        
        # Build test result
        test_result = {
            'test_name': test_name or f"{error_type.name} on qubit {qubit}",
            'error_type': error_type.name,
            'qubit': qubit,
            'logical_state': logical_state,
            'syndrome': syndrome,
            'fidelity_before': result['fidelity_before'],
            'fidelity_after': result['fidelity_after'],
            'success': result['success'],
            'success_probability': result['success_probability'],
            'syndrome_int': sum(syndrome[i] * (2 ** i) for i in range(4))
        }
        
        return test_result
    
    def test_bit_flip_errors(self):
        """Test Bit-Flip (X) errors."""
        print("\n" + "="*70)
        print("Testing BIT-FLIP (X) Errors")
        print("="*70)
        
        # Test 1: X error on qubit 0
        result1 = self.run_single_test(
            ErrorType.BIT_FLIP, qubit=0, logical_state=0,
            test_name="X error on qubit 0 (logical |0⟩)"
        )
        self.results.append(result1)
        self._print_result(result1)
        
        # Test 2: X error on qubit 2
        result2 = self.run_single_test(
            ErrorType.BIT_FLIP, qubit=2, logical_state=1,
            test_name="X error on qubit 2 (logical |1⟩)"
        )
        self.results.append(result2)
        self._print_result(result2)
    
    def test_phase_flip_errors(self):
        """Test Phase-Flip (Z) errors."""
        print("\n" + "="*70)
        print("Testing PHASE-FLIP (Z) Errors")
        print("="*70)
        
        # Test 1: Z error on qubit 0
        result1 = self.run_single_test(
            ErrorType.PHASE_FLIP, qubit=0, logical_state=0,
            test_name="Z error on qubit 0 (logical |0⟩)"
        )
        self.results.append(result1)
        self._print_result(result1)
        
        # Test 2: Z error on qubit 2
        result2 = self.run_single_test(
            ErrorType.PHASE_FLIP, qubit=2, logical_state=1,
            test_name="Z error on qubit 2 (logical |1⟩)"
        )
        self.results.append(result2)
        self._print_result(result2)
    
    def test_y_errors(self):
        """Test Y errors (combination of X and Z)."""
        print("\n" + "="*70)
        print("Testing Y Errors")
        print("="*70)
        
        # Test 1: Y error on qubit 0
        result1 = self.run_single_test(
            ErrorType.Y_ERROR, qubit=0, logical_state=0,
            test_name="Y error on qubit 0 (logical |0⟩)"
        )
        self.results.append(result1)
        self._print_result(result1)
        
        # Test 2: Y error on qubit 1
        result2 = self.run_single_test(
            ErrorType.Y_ERROR, qubit=1, logical_state=1,
            test_name="Y error on qubit 1 (logical |1⟩)"
        )
        self.results.append(result2)
        self._print_result(result2)
    
    def test_depolarizing_errors(self):
        """Test Depolarizing errors (random X, Y, or Z)."""
        print("\n" + "="*70)
        print("Testing DEPOLARIZING Errors")
        print("="*70)
        print("Note: Depolarizing errors are probabilistic (random X, Y, or Z)")
        print("Running multiple tests to see different outcomes...\n")
        
        # Test 1: Depolarizing error on qubit 0
        # Note: We set a seed for reproducibility, but depolarizing is still random
        np.random.seed(42)
        result1 = self.run_single_test(
            ErrorType.DEPOLARIZING, qubit=0, logical_state=0,
            test_name="Depolarizing error on qubit 0 (logical |0⟩) - Test 1"
        )
        self.results.append(result1)
        self._print_result(result1)
        
        # Test 2: Depolarizing error on qubit 2 (different seed)
        np.random.seed(123)
        result2 = self.run_single_test(
            ErrorType.DEPOLARIZING, qubit=2, logical_state=1,
            test_name="Depolarizing error on qubit 2 (logical |1⟩) - Test 2"
        )
        self.results.append(result2)
        self._print_result(result2)
    
    def test_rotation_x_errors(self):
        """Test Rotation X (Rx) errors."""
        print("\n" + "="*70)
        print("Testing ROTATION X (Rx) Errors")
        print("="*70)
        print("Note: Rotation errors are continuous and may not be perfectly correctable")
        print("by stabilizer codes (they're approximations).\n")
        
        # Test 1: Rx error (small angle) on qubit 0
        result1 = self.run_single_test(
            ErrorType.ROTATION_X, qubit=0, logical_state=0,
            test_name="Rx(π/4) error on qubit 0 (logical |0⟩)",
            rotation_angle=np.pi / 4
        )
        self.results.append(result1)
        self._print_result(result1)
        
        # Test 2: Rx error (small angle) on qubit 2
        result2 = self.run_single_test(
            ErrorType.ROTATION_X, qubit=2, logical_state=1,
            test_name="Rx(π/4) error on qubit 2 (logical |1⟩)",
            rotation_angle=np.pi / 4
        )
        self.results.append(result2)
        self._print_result(result2)
    
    def test_rotation_y_errors(self):
        """Test Rotation Y (Ry) errors."""
        print("\n" + "="*70)
        print("Testing ROTATION Y (Ry) Errors")
        print("="*70)
        print("Note: Rotation errors are continuous and may not be perfectly correctable")
        print("by stabilizer codes (they're approximations).\n")
        
        # Test 1: Ry error (small angle) on qubit 0
        result1 = self.run_single_test(
            ErrorType.ROTATION_Y, qubit=0, logical_state=0,
            test_name="Ry(π/4) error on qubit 0 (logical |0⟩)",
            rotation_angle=np.pi / 4
        )
        self.results.append(result1)
        self._print_result(result1)
        
        # Test 2: Ry error (small angle) on qubit 2
        result2 = self.run_single_test(
            ErrorType.ROTATION_Y, qubit=2, logical_state=1,
            test_name="Ry(π/4) error on qubit 2 (logical |1⟩)",
            rotation_angle=np.pi / 4
        )
        self.results.append(result2)
        self._print_result(result2)
    
    def test_rotation_z_errors(self):
        """Test Rotation Z (Rz) errors."""
        print("\n" + "="*70)
        print("Testing ROTATION Z (Rz) Errors")
        print("="*70)
        print("Note: Rotation errors are continuous and may not be perfectly correctable")
        print("by stabilizer codes (they're approximations).\n")
        
        # Test 1: Rz error (small angle) on qubit 0
        result1 = self.run_single_test(
            ErrorType.ROTATION_Z, qubit=0, logical_state=0,
            test_name="Rz(π/4) error on qubit 0 (logical |0⟩)",
            rotation_angle=np.pi / 4
        )
        self.results.append(result1)
        self._print_result(result1)
        
        # Test 2: Rz error (small angle) on qubit 2
        result2 = self.run_single_test(
            ErrorType.ROTATION_Z, qubit=2, logical_state=1,
            test_name="Rz(π/4) error on qubit 2 (logical |1⟩)",
            rotation_angle=np.pi / 4
        )
        self.results.append(result2)
        self._print_result(result2)
    
    def _print_result(self, result: Dict):
        """Print a single test result in a formatted way."""
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"\n{status}: {result['test_name']}")
        print(f"  Error Type: {result['error_type']}")
        print(f"  Qubit: {result['qubit']}")
        print(f"  Logical State: |{result['logical_state']}⟩")
        print(f"  Syndrome: {result['syndrome']} (decimal: {result['syndrome_int']})")
        print(f"  Fidelity Before Correction: {result['fidelity_before']:.6f}")
        print(f"  Fidelity After Correction: {result['fidelity_after']:.6f}")
        print(f"  Success Probability: {result['success_probability']:.6f}")
        print(f"  Correction Successful: {result['success']}")
    
    def run_all_tests(self):
        """Run all test suites."""
        print("\n" + "="*70)
        print("5-QUBIT PERFECT CODE ERROR CORRECTION TEST SUITE")
        print("="*70)
        print("\nThis test suite tests the 5-qubit perfect code's ability to")
        print("detect and correct various error types.")
        print("\nEach error type is tested with 1-2 test cases.")
        
        # Run all test suites
        self.test_bit_flip_errors()
        self.test_phase_flip_errors()
        self.test_y_errors()
        self.test_depolarizing_errors()
        self.test_rotation_x_errors()
        self.test_rotation_y_errors()
        self.test_rotation_z_errors()
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print a summary of all test results."""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r['success'])
        failed_tests = total_tests - passed_tests
        
        # Group by error type
        error_type_stats = {}
        for result in self.results:
            error_type = result['error_type']
            if error_type not in error_type_stats:
                error_type_stats[error_type] = {'total': 0, 'passed': 0, 'avg_fidelity_after': []}
            error_type_stats[error_type]['total'] += 1
            if result['success']:
                error_type_stats[error_type]['passed'] += 1
            error_type_stats[error_type]['avg_fidelity_after'].append(result['fidelity_after'])
        
        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        print("\n" + "-"*70)
        print("Results by Error Type:")
        print("-"*70)
        
        for error_type, stats in sorted(error_type_stats.items()):
            avg_fidelity = np.mean(stats['avg_fidelity_after'])
            pass_rate = stats['passed'] / stats['total'] * 100
            status_icon = "✅" if stats['passed'] == stats['total'] else "⚠️"
            print(f"\n{status_icon} {error_type}:")
            print(f"   Tests: {stats['passed']}/{stats['total']} passed ({pass_rate:.1f}%)")
            print(f"   Average Fidelity After Correction: {avg_fidelity:.6f}")
        
        print("\n" + "="*70)
        print("Interpretation:")
        print("="*70)
        print("• ✅ PASS: Fidelity after correction > 0.99 (successful correction)")
        print("• ❌ FAIL: Fidelity after correction ≤ 0.99 (correction not perfect)")
        print("\nNote: Rotation errors (Rx, Ry, Rz) are continuous errors and")
        print("may not be perfectly correctable by stabilizer codes, which")
        print("are designed for discrete Pauli errors (X, Y, Z).")


def main():
    """Main function to run the test suite."""
    tester = PerfectCodeErrorTests()
    tester.run_all_tests()


if __name__ == "__main__":
    main()

