# Quantum Error Correction - Error Type Analysis

## Summary

The error message you're seeing is **CORRECT** - the 3-qubit bit-flip code cannot detect Rotation Z (Rz) errors. This is expected behavior based on quantum error correction theory.

---

## 1. 3-Qubit Bit-Flip Code

### ✅ Errors it CAN detect and correct:
- **Bit-Flip (X errors)**: YES - This is the ONLY error type this code can handle
  - The code uses Z-stabilizers (Z₀Z₁, Z₁Z₂) which only detect X errors
  - Encoding: |0⟩ → |000⟩, |1⟩ → |111⟩
  - Detects which qubit flipped (0→1 or 1→0)

### ❌ Errors it CANNOT detect:
- **Phase-Flip (Z errors)**: NO - Z errors commute with Z-stabilizers, so they're invisible
- **Y errors**: NO - Y = iXZ, not detectable by this code
- **Rotation X (Rx)**: NO - Continuous error, not a discrete Pauli error
- **Rotation Y (Ry)**: NO - Continuous error, not a discrete Pauli error
- **Rotation Z (Rz)**: NO - Continuous error, not a discrete Pauli error
- **Depolarizing**: NO - This is a mixture of X, Y, Z errors, but the code can only handle X

### Recommendation for 3-Qubit Bit-Flip Code:
**Keep ONLY:**
- ✅ Bit-Flip (X) error

**Remove:**
- ❌ All other error types (Z, Y, Rx, Ry, Rz, Depolarizing)

---

## 2. 5-Qubit Perfect Code

### ✅ Errors it CAN detect and correct:
- **Bit-Flip (X errors)**: YES - Full correction capability
- **Phase-Flip (Z errors)**: YES - Full correction capability
- **Y errors**: YES - Y = iXZ, can be corrected
- **Depolarizing**: YES (with caveat) - Since it's a probabilistic mixture of X, Y, Z, and the code can correct any single Pauli error, it can handle depolarizing errors when treated as discrete Pauli errors

### ⚠️ Errors it CANNOT perfectly detect/correct:
- **Rotation X (Rx)**: PARTIAL - Rotation errors are continuous, not discrete. Stabilizer codes work with discrete Pauli errors (X, Y, Z). Small rotation angles can be approximated, but perfect correction is not guaranteed
- **Rotation Y (Ry)**: PARTIAL - Same as Rx
- **Rotation Z (Rz)**: PARTIAL - Same as Rx/Ry

### Technical Note on Rotation Errors:
- Rotation errors (Rx, Ry, Rz) are **continuous errors** (they depend on an angle parameter)
- Stabilizer codes are designed for **discrete Pauli errors** (X, Y, Z)
- Small rotation errors can be approximated as Pauli errors, but:
  - They don't perfectly fit the stabilizer formalism
  - Perfect correction is not guaranteed
  - Large rotation angles cause more problems

### Recommendation for 5-Qubit Perfect Code:
**Keep:**
- ✅ Bit-Flip (X) error
- ✅ Phase-Flip (Z) error
- ✅ Y error (combination of X and Z)
- ✅ Depolarizing error (probabilistic X, Y, Z mixture)

**Consider removing or adding warning:**
- ⚠️ Rotation X (Rx) - Can be kept but add educational note that it's an approximation
- ⚠️ Rotation Y (Ry) - Can be kept but add educational note that it's an approximation
- ⚠️ Rotation Z (Rz) - Can be kept but add educational note that it's an approximation

---

## 3. Current Error Type Definitions

Looking at your code, you have these error types defined:
1. **BIT_FLIP** (X) - Discrete Pauli error
2. **PHASE_FLIP** (Z) - Discrete Pauli error
3. **Y_ERROR** (Y) - Discrete Pauli error
4. **DEPOLARIZING** - Probabilistic mixture of X, Y, Z
5. **ROTATION_X** (Rx) - Continuous rotation error
6. **ROTATION_Y** (Ry) - Continuous rotation error
7. **ROTATION_Z** (Rz) - Continuous rotation error

---

## 4. Recommended Implementation Strategy

### Option A: Filter errors by code type (Recommended for educational clarity)

**For 3-qubit Bit-Flip Code:**
- Show only: Bit-Flip (X)
- Hide: All others
- This prevents user confusion and shows the code's true limitation

**For 5-qubit Perfect Code:**
- Show: Bit-Flip (X), Phase-Flip (Z), Y Error, Depolarizing
- Show with warning: Rotation X, Rotation Y, Rotation Z
  - Add a note: "Rotation errors are continuous and not perfectly correctable by stabilizer codes. This is an approximation for small angles."

### Option B: Keep all errors but show warnings (Current approach)

- Keep all error types available
- Show clear warnings when incompatible errors are selected
- This is more flexible but may confuse beginners

---

## 5. Why Rotation Errors Are Problematic

**Stabilizer Codes Theory:**
- Stabilizer codes work with the **Pauli group**: {I, X, Y, Z} (discrete operations)
- Errors are detected by measuring **stabilizer operators** (products of Pauli operators)
- Rotation errors (Rx, Ry, Rz) are **continuous unitary operations**, not discrete Pauli errors

**Mathematical Issue:**
- Rz(θ) = exp(-iθZ/2) = cos(θ/2)I - i sin(θ/2)Z
- This is a superposition of I and Z, not a pure Z error
- For small angles, it can be approximated as Z, but it's not exact
- The stabilizer measurement may not perfectly detect it

**Why Your Error Message is Correct:**
- The 3-qubit bit-flip code uses Z-stabilizers
- Rz errors don't trigger these stabilizers the same way X errors do
- The code cannot detect Rz errors, so the syndrome shows [0,0] (no error detected)

---

## 6. Conclusion

**Your current error message is correct!** The 3-qubit bit-flip code cannot detect Rz errors, and your code is correctly warning users about this.

**Recommended Changes:**
1. **3-qubit Bit-Flip Code**: Filter to show ONLY Bit-Flip (X) errors
2. **5-qubit Perfect Code**: Show X, Z, Y, Depolarizing clearly, and Rotation errors with educational warnings

This will make your tool more educational and prevent user confusion while maintaining theoretical correctness.

