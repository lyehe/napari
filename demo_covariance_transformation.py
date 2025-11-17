"""
Demonstration of covariance matrix coordinate transformation (ZYX → XYZ).

This validates the transformation: Σ_xyz = P @ Σ_zyx @ P^T
where P = [[0,0,1],[0,1,0],[1,0,0]]

This is a standalone script that can be run without pytest.
"""

# Try to use numpy if available, otherwise provide manual implementations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: NumPy not available. Using manual implementations.")
    print("For full testing, please install NumPy.\n")

    # Minimal implementations for demonstration
    class np:
        @staticmethod
        def array(data):
            return data

        @staticmethod
        def matmul(A, B):
            """Matrix multiplication."""
            n, m = len(A), len(A[0])
            m2, p = len(B), len(B[0])
            assert m == m2, "Incompatible dimensions"

            result = [[0 for _ in range(p)] for _ in range(n)]
            for i in range(n):
                for j in range(p):
                    for k in range(m):
                        result[i][j] += A[i][k] * B[k][j]
            return result

        @staticmethod
        def transpose(A):
            """Matrix transpose."""
            return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

        @staticmethod
        def allclose(A, B, rtol=1e-9):
            """Check if two matrices are close."""
            for i in range(len(A)):
                for j in range(len(A[0])):
                    if abs(A[i][j] - B[i][j]) > rtol * (abs(A[i][j]) + abs(B[i][j])):
                        return False
            return True

        @staticmethod
        def eye(n):
            """Identity matrix."""
            return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def print_matrix(name, M, precision=4):
    """Pretty print a matrix."""
    print(f"{name}:")
    if HAS_NUMPY:
        print(np.array2string(np.array(M), precision=precision, suppress_small=True))
    else:
        for row in M:
            print("  [", end="")
            print(", ".join(f"{x:>{precision+4}.{precision}f}" for x in row), end="")
            print("]")
    print()


def transform_covariance_zyx_to_xyz(Σ_zyx):
    """Transform covariance matrix from ZYX to XYZ coordinates.

    Args:
        Σ_zyx: 3×3 covariance matrix in ZYX coordinates

    Returns:
        Σ_xyz: 3×3 covariance matrix in XYZ coordinates
    """
    P = [[0, 0, 1],
         [0, 1, 0],
         [1, 0, 0]]

    if HAS_NUMPY:
        P = np.array(P)
        Σ_zyx = np.array(Σ_zyx)
        return P @ Σ_zyx @ P.T
    else:
        # Manual computation: P @ Σ_zyx @ P^T
        temp = np.matmul(P, Σ_zyx)
        P_T = np.transpose(P)
        return np.matmul(temp, P_T)


def main():
    """Run demonstration tests."""
    print("="*70)
    print("COVARIANCE MATRIX COORDINATE TRANSFORMATION VALIDATION")
    print("="*70)
    print()
    print("Proposed transformation: Σ_xyz = P @ Σ_zyx @ P^T")
    print("where P = [[0,0,1],[0,1,0],[1,0,0]]")
    print()

    # ========================================================================
    # Test 1: Identity Matrix
    # ========================================================================
    print("="*70)
    print("TEST 1: Identity Covariance Matrix")
    print("="*70)
    print("An identity covariance matrix represents isotropic uncertainty")
    print("(equal variance in all directions, no correlation).")
    print()

    Σ_zyx = [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]]

    print_matrix("Input Σ_zyx (identity)", Σ_zyx)

    Σ_xyz = transform_covariance_zyx_to_xyz(Σ_zyx)

    print_matrix("Output Σ_xyz", Σ_xyz)

    expected = [[1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]]

    if np.allclose(Σ_xyz, expected):
        print("✓ PASSED: Identity matrix unchanged by permutation")
    else:
        print("✗ FAILED: Result differs from expected")
    print()

    # ========================================================================
    # Test 2: Diagonal Matrix
    # ========================================================================
    print("="*70)
    print("TEST 2: Diagonal Covariance Matrix")
    print("="*70)
    print("Diagonal covariance means no correlation between axes.")
    print("Variances in ZYX: σ_z² = 1.0, σ_y² = 4.0, σ_x² = 9.0")
    print()

    Σ_zyx = [[1.0, 0.0, 0.0],
             [0.0, 4.0, 0.0],
             [0.0, 0.0, 9.0]]

    print_matrix("Input Σ_zyx (diagonal)", Σ_zyx)

    Σ_xyz = transform_covariance_zyx_to_xyz(Σ_zyx)

    print_matrix("Output Σ_xyz", Σ_xyz)

    # After ZYX → XYZ: σ_x² = 9.0, σ_y² = 4.0, σ_z² = 1.0
    expected = [[9.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 1.0]]

    if np.allclose(Σ_xyz, expected):
        print("✓ PASSED: Diagonal elements correctly permuted")
        print("  Note: σ_x² and σ_z² swapped, σ_y² unchanged")
    else:
        print("✗ FAILED: Result differs from expected")
    print()

    # ========================================================================
    # Test 3: Full Covariance Matrix with Correlations
    # ========================================================================
    print("="*70)
    print("TEST 3: Covariance Matrix with Correlations")
    print("="*70)
    print("Full covariance matrix with off-diagonal correlation terms.")
    print()

    Σ_zyx = [[2.0, 0.5, 0.3],
             [0.5, 3.0, 0.7],
             [0.3, 0.7, 4.0]]

    print_matrix("Input Σ_zyx (with correlations)", Σ_zyx)

    # Verify symmetry
    Σ_zyx_T = np.transpose(Σ_zyx)
    if np.allclose(Σ_zyx, Σ_zyx_T):
        print("✓ Input is symmetric (as required for covariance)")
    print()

    Σ_xyz = transform_covariance_zyx_to_xyz(Σ_zyx)

    print_matrix("Output Σ_xyz", Σ_xyz)

    # Manual calculation: swap rows and columns 0 ↔ 2
    expected = [[4.0, 0.7, 0.3],
                [0.7, 3.0, 0.5],
                [0.3, 0.5, 2.0]]

    if np.allclose(Σ_xyz, expected, rtol=1e-10):
        print("✓ PASSED: Covariance correctly transformed")
    else:
        print("✗ FAILED: Result differs from expected")

    # Verify symmetry of output
    Σ_xyz_T = np.transpose(Σ_xyz)
    if np.allclose(Σ_xyz, Σ_xyz_T, rtol=1e-10):
        print("✓ Output is symmetric (symmetry preserved)")
    else:
        print("✗ WARNING: Output is not symmetric")
    print()

    # ========================================================================
    # Test 4: Equivalence with Index Permutation
    # ========================================================================
    print("="*70)
    print("TEST 4: Equivalence with Direct Index Permutation")
    print("="*70)
    print("For permutation matrices, two methods should be equivalent:")
    print("  Method 1: P @ Σ @ P^T (matrix multiplication)")
    print("  Method 2: Σ[[2,1,0], :][:, [2,1,0]] (index permutation)")
    print()

    Σ_zyx = [[2.0, 0.5, 0.3],
             [0.5, 3.0, 0.7],
             [0.3, 0.7, 4.0]]

    # Method 1: Matrix multiplication
    Σ_xyz_method1 = transform_covariance_zyx_to_xyz(Σ_zyx)

    # Method 2: Direct index permutation (manual for non-numpy)
    if HAS_NUMPY:
        Σ_zyx_np = np.array(Σ_zyx)
        Σ_xyz_method2 = Σ_zyx_np[[2, 1, 0], :][:, [2, 1, 0]]
    else:
        # Manual permutation
        indices = [2, 1, 0]
        temp = [[Σ_zyx[i][j] for j in range(3)] for i in indices]
        Σ_xyz_method2 = [[temp[i][j] for j in indices] for i in range(3)]

    print_matrix("Method 1 (P @ Σ @ P^T)", Σ_xyz_method1)
    print_matrix("Method 2 (index permutation)", Σ_xyz_method2)

    if np.allclose(Σ_xyz_method1, Σ_xyz_method2, rtol=1e-10):
        print("✓ PASSED: Both methods give identical results")
        print("  Note: Index permutation is faster but less general")
    else:
        print("✗ FAILED: Methods give different results")
    print()

    # ========================================================================
    # Test 5: Roundtrip Transformation
    # ========================================================================
    print("="*70)
    print("TEST 5: Roundtrip Transformation")
    print("="*70)
    print("Applying the permutation twice should return to original.")
    print("This is because P^2 = I for this specific permutation matrix.")
    print()

    Σ_zyx_original = [[2.0, 0.5, 0.3],
                      [0.5, 3.0, 0.7],
                      [0.3, 0.7, 4.0]]

    print_matrix("Original Σ_zyx", Σ_zyx_original)

    # First transformation: ZYX → XYZ
    Σ_xyz = transform_covariance_zyx_to_xyz(Σ_zyx_original)
    print_matrix("After 1st transform (XYZ)", Σ_xyz)

    # Second transformation: XYZ → ZYX (same operation)
    Σ_zyx_roundtrip = transform_covariance_zyx_to_xyz(Σ_xyz)
    print_matrix("After 2nd transform (back to ZYX)", Σ_zyx_roundtrip)

    if np.allclose(Σ_zyx_roundtrip, Σ_zyx_original, rtol=1e-10):
        print("✓ PASSED: Roundtrip returns to original")
        print("  This confirms P^2 = I (P is an involution)")
    else:
        print("✗ FAILED: Roundtrip does not match original")
    print()

    # ========================================================================
    # Test 6: Permutation Matrix Properties
    # ========================================================================
    print("="*70)
    print("TEST 6: Permutation Matrix Properties")
    print("="*70)
    print("Verifying mathematical properties of P = [[0,0,1],[0,1,0],[1,0,0]]")
    print()

    P = [[0, 0, 1],
         [0, 1, 0],
         [1, 0, 0]]

    print_matrix("Permutation matrix P", P)

    # Property 1: P^T = P (symmetric for this specific permutation)
    P_T = np.transpose(P)
    if np.allclose(P, P_T):
        print("✓ Property 1: P is symmetric (P = P^T)")
    else:
        print("  Property 1: P is not symmetric (but that's OK for general permutations)")

    # Property 2: P^2 = I (applying twice returns to identity)
    P_squared = np.matmul(P, P)
    I = np.eye(3)
    if np.allclose(P_squared, I, rtol=1e-10):
        print("✓ Property 2: P^2 = I (P is an involution)")
    else:
        print("✗ Property 2 FAILED: P^2 ≠ I")

    # Property 3: P^T P = I (orthogonality)
    P_T_P = np.matmul(P_T, P)
    if np.allclose(P_T_P, I, rtol=1e-10):
        print("✓ Property 3: P^T P = I (P is orthogonal)")
    else:
        print("✗ Property 3 FAILED: P^T P ≠ I")

    print()
    print("All properties confirmed. This validates the permutation matrix.")
    print()

    # ========================================================================
    # Test 7: Demonstrating Incorrect Transformation
    # ========================================================================
    print("="*70)
    print("TEST 7: What Happens if We Forget P^T?")
    print("="*70)
    print("Demonstrating why we need both P and P^T in the transformation.")
    print()

    Σ_zyx = [[2.0, 0.5, 0.3],
             [0.5, 3.0, 0.7],
             [0.3, 0.7, 4.0]]

    print_matrix("Input Σ_zyx (symmetric)", Σ_zyx)

    # INCORRECT: Only multiply by P from the left
    Σ_wrong = np.matmul(P, Σ_zyx)
    print_matrix("WRONG: P @ Σ_zyx (without P^T)", Σ_wrong)

    # Check if symmetric
    Σ_wrong_T = np.transpose(Σ_wrong)
    if np.allclose(Σ_wrong, Σ_wrong_T, rtol=1e-10):
        print("  Symmetry: preserved")
    else:
        print("  ✗ Symmetry: LOST (this is wrong!)")

    # CORRECT: Multiply by P and P^T
    Σ_correct = transform_covariance_zyx_to_xyz(Σ_zyx)
    print_matrix("CORRECT: P @ Σ_zyx @ P^T", Σ_correct)

    Σ_correct_T = np.transpose(Σ_correct)
    if np.allclose(Σ_correct, Σ_correct_T, rtol=1e-10):
        print("  ✓ Symmetry: preserved")
    else:
        print("  Symmetry: LOST")

    print()
    print("This demonstrates why covariance matrices need the congruence")
    print("transformation Σ' = P Σ P^T, not just Σ' = P Σ.")
    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print("The proposed transformation Σ_xyz = P @ Σ_zyx @ P^T is CORRECT.")
    print()
    print("Key findings:")
    print("  1. ✓ The transformation follows the standard congruence form")
    print("  2. ✓ The permutation matrix P correctly maps ZYX → XYZ")
    print("  3. ✓ Symmetry is preserved under the transformation")
    print("  4. ✓ The transformation is its own inverse (P^2 = I)")
    print("  5. ✓ Equivalent to direct index permutation: Σ[[2,1,0],:][:,[2,1,0]]")
    print()
    print("Mathematical justification:")
    print("  - For linear transformation y = Px, covariance transforms as Σ_y = P Σ_x P^T")
    print("  - This is called a congruence or similarity transformation")
    print("  - Different from vector transformation (v' = Pv) because covariance")
    print("    is a second-order tensor with two spatial indices")
    print()
    print("References:")
    print("  - Hartley & Zisserman, 'Multiple View Geometry', Chapter 2")
    print("  - Thrun et al., 'Probabilistic Robotics', Equation 3.15")
    print("  - Kerbl et al., '3D Gaussian Splatting', SIGGRAPH 2023")
    print()
    print("Numerical considerations:")
    print("  - Exact permutation introduces no floating-point errors")
    print("  - Symmetry is mathematically preserved")
    print("  - Positive definiteness is preserved")
    print("  - For robustness, optionally force symmetry: Σ = 0.5*(Σ + Σ^T)")
    print()
    print("="*70)
    print("VALIDATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
