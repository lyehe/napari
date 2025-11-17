"""
Test to verify that NumPy's @ operator handles (3,3) @ (N,3,3) @ (3,3) correctly.

This test validates the rotation transformation used in the Gaussian Splatting implementation.
"""

import numpy as np

def test_rotation_broadcasting_manual():
    """
    Verify that P @ R @ P.T works correctly for (N, 3, 3) arrays.

    According to NumPy documentation, the @ operator (matmul) broadcasts
    by treating extra dimensions as stacks of matrices.

    For (3,3) @ (N,3,3):
    - Result shape: (N, 3, 3)
    - Each result[i] = (3,3) @ R[i]

    For (N,3,3) @ (3,3):
    - Result shape: (N, 3, 3)
    - Each result[i] = result_temp[i] @ (3,3)
    """
    print("Testing NumPy matrix multiplication broadcasting...")
    print("=" * 60)

    # Create test data
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
    print(f"\nPermutation matrix P shape: {P.shape}")
    print(f"P =\n{P}")

    # Create batch of rotation matrices
    N = 5
    np.random.seed(42)
    R = np.random.rand(N, 3, 3).astype(np.float32)
    print(f"\nRotation matrices R shape: {R.shape}")
    print(f"R[0] =\n{R[0]}")

    # Test 1: Direct broadcasting
    print("\n" + "-" * 60)
    print("Test 1: Direct broadcasting with @ operator")
    print("-" * 60)
    try:
        result_broadcast = P @ R @ P.T
        print(f"✓ Broadcasting succeeded!")
        print(f"  Result shape: {result_broadcast.shape}")
        print(f"  Result[0] =\n{result_broadcast[0]}")
    except Exception as e:
        print(f"✗ Broadcasting failed: {e}")
        return False

    # Test 2: Manual element-wise computation
    print("\n" + "-" * 60)
    print("Test 2: Manual element-wise computation (ground truth)")
    print("-" * 60)
    result_manual = np.array([P @ R[i] @ P.T for i in range(N)])
    print(f"  Result shape: {result_manual.shape}")
    print(f"  Result[0] =\n{result_manual[0]}")

    # Test 3: Compare results
    print("\n" + "-" * 60)
    print("Test 3: Comparing broadcast vs manual results")
    print("-" * 60)
    if np.allclose(result_broadcast, result_manual):
        print("✓ Results match! Broadcasting works correctly.")
        max_diff = np.max(np.abs(result_broadcast - result_manual))
        print(f"  Max absolute difference: {max_diff:.2e}")
    else:
        print("✗ Results do NOT match!")
        max_diff = np.max(np.abs(result_broadcast - result_manual))
        print(f"  Max absolute difference: {max_diff:.2e}")
        return False

    # Test 4: Verify properties are preserved
    print("\n" + "-" * 60)
    print("Test 4: Verify transformation properties")
    print("-" * 60)

    # Check if transformation preserves determinant magnitude
    det_original = np.linalg.det(R)
    det_transformed = np.linalg.det(result_broadcast)
    print(f"  Original determinants: {det_original}")
    print(f"  Transformed determinants: {det_transformed}")
    if np.allclose(np.abs(det_original), np.abs(det_transformed)):
        print("  ✓ Determinant magnitude preserved")
    else:
        print("  ✗ Determinant magnitude NOT preserved!")
        return False

    # Test 5: Test with scipy Rotation objects
    print("\n" + "-" * 60)
    print("Test 5: Test with actual rotation matrices from scipy")
    print("-" * 60)
    try:
        from scipy.spatial.transform import Rotation as R_scipy

        # Create random rotation matrices
        rotations = R_scipy.random(N, random_state=42)
        R_matrices = rotations.as_matrix().astype(np.float32)

        # Transform
        R_transformed = P @ R_matrices @ P.T

        # Verify result is still valid rotation matrices
        for i in range(N):
            det = np.linalg.det(R_transformed[i])
            orth_check = np.allclose(R_transformed[i] @ R_transformed[i].T, np.eye(3))
            if abs(det - 1.0) < 1e-5 and orth_check:
                pass  # Good
            else:
                print(f"  ✗ Transformed matrix {i} is not a valid rotation!")
                print(f"    det = {det}, orthogonal = {orth_check}")
                return False

        print("  ✓ All transformed matrices are valid rotations")

    except ImportError:
        print("  ⚠ scipy not available, skipping rotation validation")

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nConclusion: NumPy's @ operator CORRECTLY handles")
    print("the broadcasting for (3,3) @ (N,3,3) @ (3,3)")
    print("\nThe implementation in gaussians.py is CORRECT:")
    print("  R_matrices_xyz = P.T @ R_matrices_zyx @ P")

    return True


if __name__ == '__main__':
    success = test_rotation_broadcasting_manual()
    exit(0 if success else 1)
