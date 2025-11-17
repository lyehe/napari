"""
Test suite for covariance matrix coordinate transformation (ZYX → XYZ).

This validates the transformation: Σ_xyz = P @ Σ_zyx @ P^T
where P = [[0,0,1],[0,1,0],[1,0,0]]
"""

import numpy as np
import pytest


def transform_covariance_zyx_to_xyz(Σ_zyx, force_symmetric=True, min_eigenvalue=None):
    """Transform covariance matrix from ZYX to XYZ coordinates.

    Args:
        Σ_zyx: 3×3 covariance matrix in ZYX coordinates
        force_symmetric: Enforce symmetry after transformation
        min_eigenvalue: Minimum eigenvalue for positive definiteness

    Returns:
        Σ_xyz: 3×3 covariance matrix in XYZ coordinates
    """
    P = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]], dtype=Σ_zyx.dtype)

    Σ_xyz = P @ Σ_zyx @ P.T

    if force_symmetric:
        Σ_xyz = 0.5 * (Σ_xyz + Σ_xyz.T)

    # Optional: ensure positive definiteness
    if min_eigenvalue is not None:
        eigenvalues, eigenvectors = np.linalg.eigh(Σ_xyz)
        if np.any(eigenvalues < min_eigenvalue):
            eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
            Σ_xyz = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    return Σ_xyz


def verify_covariance_properties(Σ, name="Σ", rtol=1e-10):
    """Verify that a matrix is a valid covariance matrix.

    Args:
        Σ: Matrix to verify
        name: Name for error messages
        rtol: Relative tolerance for numerical checks

    Returns:
        True if all checks pass

    Raises:
        AssertionError if any check fails
    """
    # Test 1: Must be square and 3×3
    assert Σ.shape == (3, 3), f"{name} must be 3×3, got {Σ.shape}"

    # Test 2: Symmetry
    assert np.allclose(Σ, Σ.T, rtol=rtol), f"{name} not symmetric:\n{Σ}"

    # Test 3: Positive semi-definiteness
    eigenvalues = np.linalg.eigvalsh(Σ)
    assert np.all(eigenvalues > -rtol), \
        f"{name} not positive semi-definite. Eigenvalues: {eigenvalues}"

    # Test 4: Real values
    assert np.all(np.isfinite(Σ)), f"{name} contains inf or nan"

    return True


# ============================================================================
# Test Case 1: Identity Matrix
# ============================================================================
def test_identity_covariance():
    """Test with identity covariance (isotropic, no correlation)."""
    Σ_zyx = np.eye(3)
    Σ_xyz = transform_covariance_zyx_to_xyz(Σ_zyx)

    # Identity should be unchanged by coordinate permutation
    expected = np.eye(3)
    np.testing.assert_allclose(Σ_xyz, expected, rtol=1e-15)
    verify_covariance_properties(Σ_xyz, "Σ_xyz (identity)")


# ============================================================================
# Test Case 2: Diagonal Matrix (No Correlation)
# ============================================================================
def test_diagonal_covariance():
    """Test with diagonal covariance (no correlation between axes)."""
    # Variances: σ_z² = 1.0, σ_y² = 4.0, σ_x² = 9.0
    Σ_zyx = np.diag([1.0, 4.0, 9.0])

    Σ_xyz = transform_covariance_zyx_to_xyz(Σ_zyx)

    # After ZYX → XYZ permutation: σ_x² = 9.0, σ_y² = 4.0, σ_z² = 1.0
    expected = np.diag([9.0, 4.0, 1.0])

    np.testing.assert_allclose(Σ_xyz, expected, rtol=1e-15)
    verify_covariance_properties(Σ_xyz, "Σ_xyz (diagonal)")


# ============================================================================
# Test Case 3: Covariance with Correlations
# ============================================================================
def test_correlated_covariance():
    """Test with full covariance matrix (with correlations)."""
    # Create a symmetric positive definite matrix with correlations
    Σ_zyx = np.array([
        [2.0, 0.5, 0.3],  # z-z, z-y, z-x
        [0.5, 3.0, 0.7],  # y-z, y-y, y-x
        [0.3, 0.7, 4.0],  # x-z, x-y, x-x
    ])

    verify_covariance_properties(Σ_zyx, "Σ_zyx (input)")

    Σ_xyz = transform_covariance_zyx_to_xyz(Σ_zyx)

    # Manual calculation of expected result:
    # P @ Σ_zyx @ P^T where P = [[0,0,1],[0,1,0],[1,0,0]]
    #
    # The transformation swaps rows 0↔2 and columns 0↔2:
    expected = np.array([
        [4.0, 0.7, 0.3],  # x-x, x-y, x-z (from row/col 2 of Σ_zyx)
        [0.7, 3.0, 0.5],  # y-x, y-y, y-z (from row/col 1 of Σ_zyx)
        [0.3, 0.5, 2.0],  # z-x, z-y, z-z (from row/col 0 of Σ_zyx)
    ])

    np.testing.assert_allclose(Σ_xyz, expected, rtol=1e-15)
    verify_covariance_properties(Σ_xyz, "Σ_xyz (correlated)")


# ============================================================================
# Test Case 4: Comparison with Alternative Implementation
# ============================================================================
def test_equivalence_with_index_permutation():
    """Verify equivalence with direct index permutation."""
    Σ_zyx = np.array([
        [2.0, 0.5, 0.3],
        [0.5, 3.0, 0.7],
        [0.3, 0.7, 4.0],
    ])

    # Method 1: Matrix multiplication
    P = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]])
    Σ_xyz_method1 = P @ Σ_zyx @ P.T

    # Method 2: Direct index permutation (ZYX → XYZ means indices [2,1,0])
    Σ_xyz_method2 = Σ_zyx[[2, 1, 0], :][:, [2, 1, 0]]

    # Both methods should give identical results
    np.testing.assert_allclose(Σ_xyz_method1, Σ_xyz_method2, rtol=1e-15)


# ============================================================================
# Test Case 5: Determinant Preservation
# ============================================================================
def test_determinant_preserved():
    """Verify that determinant (volume of uncertainty ellipsoid) is preserved."""
    Σ_zyx = np.array([
        [2.0, 0.5, 0.3],
        [0.5, 3.0, 0.7],
        [0.3, 0.7, 4.0],
    ])

    Σ_xyz = transform_covariance_zyx_to_xyz(Σ_zyx)

    det_zyx = np.linalg.det(Σ_zyx)
    det_xyz = np.linalg.det(Σ_xyz)

    # For orthogonal transformations, det(P Σ P^T) = det(Σ) * det(P)^2
    # Since det(P) = ±1, we have det(P)^2 = 1
    np.testing.assert_allclose(det_xyz, det_zyx, rtol=1e-14)


# ============================================================================
# Test Case 6: Eigenvalues Preserved
# ============================================================================
def test_eigenvalues_preserved():
    """Verify that eigenvalues (uncertainty magnitudes) are preserved."""
    Σ_zyx = np.array([
        [2.0, 0.5, 0.3],
        [0.5, 3.0, 0.7],
        [0.3, 0.7, 4.0],
    ])

    Σ_xyz = transform_covariance_zyx_to_xyz(Σ_zyx)

    eig_zyx = np.sort(np.linalg.eigvalsh(Σ_zyx))
    eig_xyz = np.sort(np.linalg.eigvalsh(Σ_xyz))

    # Eigenvalues should be identical (up to numerical precision)
    np.testing.assert_allclose(eig_xyz, eig_zyx, rtol=1e-14)


# ============================================================================
# Test Case 7: Roundtrip Transformation
# ============================================================================
def test_roundtrip_transformation():
    """Applying the transformation twice should return to original."""
    Σ_zyx_original = np.array([
        [2.0, 0.5, 0.3],
        [0.5, 3.0, 0.7],
        [0.3, 0.7, 4.0],
    ])

    # ZYX → XYZ
    Σ_xyz = transform_covariance_zyx_to_xyz(Σ_zyx_original)

    # XYZ → ZYX (apply same transformation again, as P^2 = I for this P)
    Σ_zyx_roundtrip = transform_covariance_zyx_to_xyz(Σ_xyz)

    # Should return to original
    np.testing.assert_allclose(Σ_zyx_roundtrip, Σ_zyx_original, rtol=1e-14)


# ============================================================================
# Test Case 8: Anisotropic Gaussian (Ellipsoid)
# ============================================================================
def test_anisotropic_gaussian():
    """Test with an elongated Gaussian (common in 3D Gaussian splatting)."""
    # Elongated along z-axis, thin in x-y
    Σ_zyx = np.array([
        [100.0,  0.0,   0.0],   # Large variance in z
        [0.0,    1.0,   0.5],   # Small variance in y, correlation with x
        [0.0,    0.5,   1.0],   # Small variance in x
    ])

    Σ_xyz = transform_covariance_zyx_to_xyz(Σ_zyx)

    # After transformation, large variance should be in z (was in z before)
    expected = np.array([
        [1.0,    0.5,   0.0],   # x variance (was x)
        [0.5,    1.0,   0.0],   # y variance (was y)
        [0.0,    0.0,   100.0], # z variance (was z)
    ])

    np.testing.assert_allclose(Σ_xyz, expected, rtol=1e-15)
    verify_covariance_properties(Σ_xyz, "Σ_xyz (anisotropic)")


# ============================================================================
# Test Case 9: Numerical Stability with Small Values
# ============================================================================
def test_numerical_stability_small_values():
    """Test with very small covariance values."""
    epsilon = 1e-10
    Σ_zyx = np.eye(3) * epsilon

    Σ_xyz = transform_covariance_zyx_to_xyz(Σ_zyx)

    expected = np.eye(3) * epsilon
    np.testing.assert_allclose(Σ_xyz, expected, rtol=1e-6, atol=1e-15)
    verify_covariance_properties(Σ_xyz, "Σ_xyz (small)", rtol=epsilon * 10)


# ============================================================================
# Test Case 10: Numerical Stability with Large Values
# ============================================================================
def test_numerical_stability_large_values():
    """Test with very large covariance values."""
    scale = 1e10
    Σ_zyx = np.eye(3) * scale

    Σ_xyz = transform_covariance_zyx_to_xyz(Σ_zyx)

    expected = np.eye(3) * scale
    np.testing.assert_allclose(Σ_xyz, expected, rtol=1e-6)
    verify_covariance_properties(Σ_xyz, "Σ_xyz (large)", rtol=1e-6)


# ============================================================================
# Test Case 11: Realistic 3D Gaussian Splatting Example
# ============================================================================
def test_gaussian_splatting_realistic():
    """Test with parameters typical in 3D Gaussian splatting.

    In Gaussian splatting, covariances are often:
    - Constructed from scale and rotation: Σ = R S S^T R^T
    - Can have extreme anisotropy (thin splats)
    - May be stored in camera/depth coordinate system (ZYX)
    - Need to be rendered in world coordinate system (XYZ)
    """
    # Simulate a thin splat oriented at an angle
    # Scale: very thin in one direction, wider in others
    scale = np.array([2.0, 1.5, 0.1])  # [sx, sy, sz]
    S = np.diag(scale)

    # Rotation: 45 degrees around Y-axis
    angle = np.pi / 4
    R_y = np.array([
        [np.cos(angle),  0, np.sin(angle)],
        [0,              1, 0            ],
        [-np.sin(angle), 0, np.cos(angle)],
    ])

    # Construct covariance in ZYX coordinates
    # Note: This is in ZYX order, so we permute to match
    Σ_zyx_construct = R_y @ (S @ S.T) @ R_y.T
    # Permute to actual ZYX order for this test
    Σ_zyx = Σ_zyx_construct[[2,1,0], :][:, [2,1,0]]

    verify_covariance_properties(Σ_zyx, "Σ_zyx (splat)")

    Σ_xyz = transform_covariance_zyx_to_xyz(Σ_zyx)

    verify_covariance_properties(Σ_xyz, "Σ_xyz (splat)")

    # Check that the splat is still thin (smallest eigenvalue is small)
    eigenvalues = np.sort(np.linalg.eigvalsh(Σ_xyz))
    assert eigenvalues[0] < 0.1, "Splat should still be thin"
    assert eigenvalues[2] > 1.0, "Splat should be elongated"


# ============================================================================
# Test Case 12: Symmetry Forcing
# ============================================================================
def test_symmetry_forcing():
    """Test that symmetry forcing works correctly."""
    # Create a slightly asymmetric matrix (simulating numerical error)
    Σ_zyx = np.array([
        [2.0,       0.5 + 1e-14, 0.3        ],
        [0.5,       3.0,         0.7 + 1e-14],
        [0.3 - 1e-14, 0.7,       4.0        ],
    ])

    # Without forcing symmetry, might accumulate errors
    Σ_xyz_no_force = transform_covariance_zyx_to_xyz(Σ_zyx, force_symmetric=False)

    # With forcing symmetry
    Σ_xyz_forced = transform_covariance_zyx_to_xyz(Σ_zyx, force_symmetric=True)

    # Forced version should be exactly symmetric
    assert np.allclose(Σ_xyz_forced, Σ_xyz_forced.T), "Should be exactly symmetric"

    # Both should be very close
    np.testing.assert_allclose(Σ_xyz_no_force, Σ_xyz_forced, rtol=1e-12)


# ============================================================================
# Test Case 13: Permutation Matrix Properties
# ============================================================================
def test_permutation_matrix_properties():
    """Verify that P has the expected properties."""
    P = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]])

    # Property 1: P is orthogonal (P^T P = I)
    assert np.allclose(P.T @ P, np.eye(3)), "P should be orthogonal"

    # Property 2: P^T = P^{-1}
    assert np.allclose(P.T @ P, np.eye(3)), "P^T should equal P^{-1}"

    # Property 3: P^2 should be identity (applying twice returns to original)
    assert np.allclose(P @ P, np.eye(3)), "P^2 should equal I"

    # Property 4: det(P) = ±1
    det_P = np.linalg.det(P)
    assert np.isclose(abs(det_P), 1.0), f"det(P) should be ±1, got {det_P}"

    # Property 5: P is its own inverse (for this specific permutation)
    assert np.allclose(P, P.T), "For this P, P = P^T"


# ============================================================================
# Test Case 14: Incorrect Transformations (Should Fail)
# ============================================================================
def test_incorrect_transformation_without_transpose():
    """Demonstrate that forgetting P^T gives wrong results."""
    P = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]])

    Σ_zyx = np.array([
        [2.0, 0.5, 0.3],
        [0.5, 3.0, 0.7],
        [0.3, 0.7, 4.0],
    ])

    # INCORRECT: Only apply P from the left
    Σ_wrong = P @ Σ_zyx

    # This should NOT be symmetric
    assert not np.allclose(Σ_wrong, Σ_wrong.T), \
        "P @ Σ (without P^T) should NOT be symmetric"

    # Correct transformation
    Σ_correct = P @ Σ_zyx @ P.T

    # Should be different
    assert not np.allclose(Σ_wrong, Σ_correct), \
        "Correct and incorrect transformations should differ"


# ============================================================================
# Benchmarking and Performance Tests
# ============================================================================
def test_performance_matrix_multiplication():
    """Compare performance of matrix multiplication vs index permutation."""
    import timeit

    Σ_zyx = np.random.randn(3, 3)
    Σ_zyx = Σ_zyx @ Σ_zyx.T  # Make it positive definite

    P = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]])

    # Method 1: Matrix multiplication
    def method1():
        return P @ Σ_zyx @ P.T

    # Method 2: Index permutation
    def method2():
        return Σ_zyx[[2, 1, 0], :][:, [2, 1, 0]]

    # Verify equivalence first
    assert np.allclose(method1(), method2())

    # Time both methods
    n_iterations = 10000
    time1 = timeit.timeit(method1, number=n_iterations)
    time2 = timeit.timeit(method2, number=n_iterations)

    print(f"\nPerformance comparison ({n_iterations} iterations):")
    print(f"  Matrix multiplication: {time1:.4f}s")
    print(f"  Index permutation:     {time2:.4f}s")
    print(f"  Speedup: {time1/time2:.2f}x")

    # Note: Index permutation is usually faster for small matrices
    # but matrix multiplication is more general and clearer


# ============================================================================
# Edge Cases
# ============================================================================
def test_near_singular_covariance():
    """Test with nearly singular covariance matrix."""
    # Create a nearly rank-deficient matrix
    Σ_zyx = np.array([
        [1.0,   1.0,   1.0],
        [1.0,   1.0,   1.0],
        [1.0,   1.0,   1.0 + 1e-10],
    ])

    # This is nearly singular but still valid
    Σ_xyz = transform_covariance_zyx_to_xyz(Σ_zyx, min_eigenvalue=1e-12)

    # Should still be symmetric and positive semi-definite
    verify_covariance_properties(Σ_xyz, "Σ_xyz (near-singular)", rtol=1e-6)

    # Condition number should be very large
    cond = np.linalg.cond(Σ_xyz)
    assert cond > 1e8, f"Should be ill-conditioned, got cond={cond}"


if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "-s"])

    # Or run individual tests for demonstration
    print("\n" + "="*70)
    print("Running demonstration tests...")
    print("="*70)

    print("\n[Test 1] Identity covariance...")
    test_identity_covariance()
    print("✓ Passed")

    print("\n[Test 2] Diagonal covariance...")
    test_diagonal_covariance()
    print("✓ Passed")

    print("\n[Test 3] Correlated covariance...")
    test_correlated_covariance()
    print("✓ Passed")

    print("\n[Test 4] Equivalence with index permutation...")
    test_equivalence_with_index_permutation()
    print("✓ Passed")

    print("\n[Test 5] Determinant preservation...")
    test_determinant_preserved()
    print("✓ Passed")

    print("\n[Test 6] Eigenvalue preservation...")
    test_eigenvalues_preserved()
    print("✓ Passed")

    print("\n[Test 7] Roundtrip transformation...")
    test_roundtrip_transformation()
    print("✓ Passed")

    print("\n[Test 8] Anisotropic Gaussian...")
    test_anisotropic_gaussian()
    print("✓ Passed")

    print("\n[Test 11] Realistic Gaussian splatting...")
    test_gaussian_splatting_realistic()
    print("✓ Passed")

    print("\n[Test 13] Permutation matrix properties...")
    test_permutation_matrix_properties()
    print("✓ Passed")

    print("\n[Test 14] Incorrect transformation (should fail)...")
    test_incorrect_transformation_without_transpose()
    print("✓ Passed (correctly identified as wrong)")

    print("\n" + "="*70)
    print("All demonstration tests passed! ✓")
    print("="*70)
