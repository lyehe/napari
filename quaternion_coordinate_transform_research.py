"""
Quaternion Coordinate Transformation Research and Validation
=============================================================

This module demonstrates and validates various approaches to transforming
quaternions when changing coordinate systems, with specific focus on
3D Gaussian Splatting applications.

Author: Research for napari Gaussian Splatting integration
Date: 2025-11-17

Key Findings:
1. Quaternions represent rotations, not coordinate systems
2. When changing coordinate systems, you need to transform the rotation itself
3. Simple component swapping DOES NOT work (changes chirality)
4. Two valid approaches:
   a) Convert to rotation matrix, apply permutation, convert back
   b) Compose with appropriate rotation quaternions

References:
- scipy.spatial.transform.Rotation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
- 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Math Stack Exchange on axis permutations: https://math.stackexchange.com/questions/1795351/
- GitHub Gaussian Splatting Issue #176: https://github.com/graphdeco-inria/gaussian-splatting/issues/176
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import warnings


# ==============================================================================
# PART 1: Understanding Quaternion Format and Coordinate Systems
# ==============================================================================

def demonstrate_quaternion_formats():
    """
    Demonstrate the difference between scalar-first and scalar-last quaternion formats.

    SciPy uses scalar-last (x, y, z, w) by default.
    Gaussian Splatting often uses scalar-first (w, x, y, z).
    """
    print("=" * 80)
    print("QUATERNION FORMAT CONVENTIONS")
    print("=" * 80)

    # Create a 90-degree rotation around Z-axis
    angle = np.pi / 2  # 90 degrees
    axis = np.array([0, 0, 1])  # Z-axis

    # Quaternion components: q = w + xi + yj + zk
    # For axis-angle: w = cos(θ/2), (x,y,z) = sin(θ/2) * axis
    w = np.cos(angle / 2)
    xyz = np.sin(angle / 2) * axis

    print("\n90° rotation around Z-axis:")
    print(f"  Scalar-first (w, x, y, z): ({w:.4f}, {xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f})")
    print(f"  Scalar-last  (x, y, z, w): ({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}, {w:.4f})")

    # SciPy uses scalar-last by default
    quat_scipy = np.array([xyz[0], xyz[1], xyz[2], w])
    rot = R.from_quat(quat_scipy)

    # Verify by rotating a point
    point = np.array([1, 0, 0])
    rotated = rot.apply(point)
    expected = np.array([0, 1, 0])  # X -> Y for 90° around Z

    print(f"\nVerification:")
    print(f"  Input point:    {point}")
    print(f"  Rotated point:  {rotated}")
    print(f"  Expected:       {expected}")
    print(f"  Match: {np.allclose(rotated, expected)}")

    return rot


# ==============================================================================
# PART 2: The Wrong Approach - Direct Component Swapping
# ==============================================================================

def demonstrate_wrong_approach():
    """
    Demonstrate why you CANNOT simply swap quaternion components to change
    coordinate systems. This changes the chirality (handedness).
    """
    print("\n" + "=" * 80)
    print("WHY DIRECT COMPONENT SWAPPING FAILS")
    print("=" * 80)

    # Create a rotation: 45 degrees around the vector (1, 1, 1)
    axis = np.array([1, 1, 1]) / np.sqrt(3)  # Normalized
    angle = np.deg2rad(45)

    rot = R.from_rotvec(angle * axis)
    quat_original = rot.as_quat()  # scalar-last format [x, y, z, w]

    print(f"\nOriginal quaternion (x,y,z,w): {quat_original}")
    print(f"Original rotation axis: {axis}")
    print(f"Original rotation angle: {np.rad2deg(angle):.2f}°")

    # WRONG: Try to swap Y and Z by swapping quaternion components
    quat_swapped = np.array([
        quat_original[0],  # x stays
        quat_original[2],  # z -> y
        quat_original[1],  # y -> z
        quat_original[3]   # w stays
    ])

    rot_swapped = R.from_quat(quat_swapped)

    # Test on a point
    test_point = np.array([1, 2, 3])
    result_original = rot.apply(test_point)
    result_swapped = rot_swapped.apply(test_point)

    # What we MIGHT expect (swap Y and Z in result)
    expected_if_correct = np.array([
        result_original[0],
        result_original[2],
        result_original[1]
    ])

    print(f"\nTesting on point {test_point}:")
    print(f"  Original rotation result:        {result_original}")
    print(f"  Swapped quaternion result:       {result_swapped}")
    print(f"  Expected (if swapping worked):   {expected_if_correct}")
    print(f"  Does it match? {np.allclose(result_swapped, expected_if_correct)}")

    print("\n⚠️  CONCLUSION: Direct component swapping does NOT correctly transform")
    print("    the rotation to a new coordinate system!")


# ==============================================================================
# PART 3: Correct Approach 1 - Rotation Matrix Method
# ==============================================================================

def correct_approach_rotation_matrix(quat_input, coord_transform_matrix):
    """
    Transform a quaternion to a new coordinate system using rotation matrices.

    This is the mathematically rigorous approach:
    1. Convert quaternion to rotation matrix R
    2. Apply coordinate transformation: R' = P @ R @ P^T
       where P is the permutation/transformation matrix
    3. Convert back to quaternion

    Parameters
    ----------
    quat_input : array-like, shape (4,)
        Input quaternion in scalar-last format [x, y, z, w]
    coord_transform_matrix : array-like, shape (3, 3)
        Coordinate transformation matrix (e.g., permutation matrix)

    Returns
    -------
    quat_output : ndarray, shape (4,)
        Transformed quaternion in scalar-last format
    """
    # Step 1: Convert to rotation matrix
    rot = R.from_quat(quat_input)
    R_matrix = rot.as_matrix()

    # Step 2: Transform the rotation matrix
    # The rotation in the new coordinate system is: R' = P @ R @ P^T
    R_transformed = coord_transform_matrix @ R_matrix @ coord_transform_matrix.T

    # Step 3: Convert back to quaternion
    rot_transformed = R.from_matrix(R_transformed)
    quat_output = rot_transformed.as_quat()

    return quat_output


def demonstrate_rotation_matrix_approach():
    """
    Demonstrate the rotation matrix approach for coordinate transformation.
    """
    print("\n" + "=" * 80)
    print("CORRECT APPROACH 1: ROTATION MATRIX METHOD")
    print("=" * 80)

    # Example: Transform from ZYX coordinate system to XYZ
    # This means: old_Z -> new_X, old_Y -> new_Y, old_X -> new_Z

    print("\nExample: Transform from (Z,Y,X) to (X,Y,Z)")
    print("This permutation swaps the first and third axes")

    # Permutation matrix: [Z, Y, X] -> [X, Y, Z]
    # Old coordinates: (Z, Y, X)
    # New coordinates: (X, Y, Z)
    # So: new_X = old_X, new_Y = old_Y, new_Z = old_Z
    # Actually, if old coords are ordered as (Z,Y,X) and we want (X,Y,Z):
    # new_X = old_Z, new_Y = old_Y, new_Z = old_X

    P = np.array([
        [0, 0, 1],  # new_X comes from old_Z (3rd position when indexed as Z,Y,X)
        [0, 1, 0],  # new_Y comes from old_Y
        [1, 0, 0]   # new_Z comes from old_Z (1st position when indexed as Z,Y,X)
    ])

    # Wait, let me reconsider. If we have axes in order (X, Y, Z) and want to
    # permute to (Z, Y, X), the permutation matrix is:
    P_ZYX_to_XYZ = np.array([
        [0, 0, 1],  # X_new = Z_old
        [0, 1, 0],  # Y_new = Y_old
        [1, 0, 0]   # Z_new = X_old
    ])

    print(f"\nPermutation matrix P:")
    print(P_ZYX_to_XYZ)

    # Create a test rotation: 45° around old X-axis
    angle = np.deg2rad(45)
    axis_old = np.array([1, 0, 0])  # X-axis in old system
    rot_old = R.from_rotvec(angle * axis_old)
    quat_old = rot_old.as_quat()

    print(f"\nOriginal rotation:")
    print(f"  Axis (in old XYZ):    {axis_old}")
    print(f"  Angle:                {np.rad2deg(angle):.2f}°")
    print(f"  Quaternion (x,y,z,w): {quat_old}")

    # Transform to new coordinate system
    quat_new = correct_approach_rotation_matrix(quat_old, P_ZYX_to_XYZ)
    rot_new = R.from_matrix(P_ZYX_to_XYZ @ rot_old.as_matrix() @ P_ZYX_to_XYZ.T)

    print(f"\nTransformed rotation:")
    print(f"  Quaternion (x,y,z,w): {quat_new}")

    # Verify correctness
    test_point_old = np.array([0, 1, 0])  # Y-axis in old system
    result_old = rot_old.apply(test_point_old)

    # Transform test point to new coordinate system
    test_point_new = P_ZYX_to_XYZ @ test_point_old
    result_new = rot_new.apply(test_point_new)

    # Expected result in new system
    expected_new = P_ZYX_to_XYZ @ result_old

    print(f"\nVerification:")
    print(f"  Test point (old system):    {test_point_old}")
    print(f"  Test point (new system):    {test_point_new}")
    print(f"  Result (new system):        {result_new}")
    print(f"  Expected (new system):      {expected_new}")
    print(f"  Match: {np.allclose(result_new, expected_new)}")


# ==============================================================================
# PART 4: Correct Approach 2 - Quaternion Composition
# ==============================================================================

def correct_approach_quaternion_composition(quat_input, coord_rotation):
    """
    Transform a quaternion to a new coordinate system using quaternion composition.

    This approach composes the input quaternion with a quaternion representing
    the coordinate system transformation.

    Parameters
    ----------
    quat_input : array-like, shape (4,)
        Input quaternion in scalar-last format [x, y, z, w]
    coord_rotation : Rotation object
        Rotation representing the coordinate system transformation

    Returns
    -------
    quat_output : ndarray, shape (4,)
        Transformed quaternion in scalar-last format
    """
    rot_input = R.from_quat(quat_input)

    # Compose rotations: R' = coord_rotation * R * coord_rotation^(-1)
    rot_transformed = coord_rotation * rot_input * coord_rotation.inv()

    return rot_transformed.as_quat()


def demonstrate_quaternion_composition():
    """
    Demonstrate the quaternion composition approach.
    """
    print("\n" + "=" * 80)
    print("CORRECT APPROACH 2: QUATERNION COMPOSITION")
    print("=" * 80)

    print("\nExample: Convert from Y-up to Z-up coordinate system")
    print("This requires a -90° rotation around X-axis")

    # The coordinate transformation: Y-up to Z-up
    # This is equivalent to rotating the coordinate frame by -90° around X
    coord_transform = R.from_euler('x', -90, degrees=True)

    # Test rotation: 30° around the old Y-axis (up)
    test_angle = 30
    rot_old = R.from_euler('y', test_angle, degrees=True)
    quat_old = rot_old.as_quat()

    print(f"\nOriginal rotation (in Y-up system):")
    print(f"  30° around Y-axis (up)")
    print(f"  Quaternion: {quat_old}")

    # Transform to new coordinate system
    quat_new = correct_approach_quaternion_composition(quat_old, coord_transform)
    rot_new = R.from_quat(quat_new)

    # In Z-up system, this should be 30° around Z-axis
    print(f"\nTransformed rotation (in Z-up system):")
    print(f"  Should be 30° around Z-axis (up)")
    print(f"  Quaternion: {quat_new}")

    # Verify: rotating around Z in new system
    expected_rot = R.from_euler('z', test_angle, degrees=True)
    expected_quat = expected_rot.as_quat()

    # Note: quaternions q and -q represent the same rotation
    match = np.allclose(quat_new, expected_quat) or np.allclose(quat_new, -expected_quat)
    print(f"  Expected quaternion: {expected_quat}")
    print(f"  Match: {match}")


# ==============================================================================
# PART 5: Edge Cases and Special Considerations
# ==============================================================================

def demonstrate_edge_cases():
    """
    Demonstrate important edge cases and special considerations.
    """
    print("\n" + "=" * 80)
    print("EDGE CASES AND SPECIAL CONSIDERATIONS")
    print("=" * 80)

    # Edge Case 1: Quaternion sign ambiguity (double cover)
    print("\n1. QUATERNION SIGN AMBIGUITY (Double Cover)")
    print("-" * 80)

    angle = np.deg2rad(45)
    axis = np.array([1, 0, 0])
    rot = R.from_rotvec(angle * axis)

    quat_pos = rot.as_quat()
    quat_neg = -quat_pos

    rot_pos = R.from_quat(quat_pos)
    rot_neg = R.from_quat(quat_neg)

    test_point = np.array([0, 1, 0])
    result_pos = rot_pos.apply(test_point)
    result_neg = rot_neg.apply(test_point)

    print(f"Quaternion:     {quat_pos}")
    print(f"Negated:        {quat_neg}")
    print(f"Result with q:  {result_pos}")
    print(f"Result with -q: {result_neg}")
    print(f"Same rotation:  {np.allclose(result_pos, result_neg)}")
    print("\n⚠️  Both q and -q represent the SAME rotation!")
    print("    When comparing quaternions, account for sign ambiguity.")

    # Edge Case 2: Gimbal lock (only relevant for Euler angles)
    print("\n\n2. GIMBAL LOCK")
    print("-" * 80)
    print("Gimbal lock occurs with Euler angles, NOT with quaternions.")
    print("However, conversions between quaternions and Euler angles")
    print("can encounter singularities at gimbal lock positions.")

    # Create a rotation at gimbal lock position (pitch = 90°)
    euler_gimbal = [0, 90, 45]  # Roll, Pitch, Yaw in degrees
    rot_gimbal = R.from_euler('xyz', euler_gimbal, degrees=True)
    quat_gimbal = rot_gimbal.as_quat()

    # Convert back to Euler angles
    euler_recovered = rot_gimbal.as_euler('xyz', degrees=True)

    print(f"\nOriginal Euler angles:   {euler_gimbal}")
    print(f"Quaternion:              {quat_gimbal}")
    print(f"Recovered Euler angles:  {euler_recovered}")
    print(f"\n⚠️  Recovered angles may differ due to Euler angle degeneracy,")
    print("    but they represent the same rotation. Quaternions avoid this!")

    # Edge Case 3: Near-zero rotations
    print("\n\n3. NEAR-ZERO ROTATIONS")
    print("-" * 80)

    # Very small rotation
    tiny_angle = 1e-10
    axis = np.array([1, 0, 0])
    rot_tiny = R.from_rotvec(tiny_angle * axis)
    quat_tiny = rot_tiny.as_quat()

    print(f"Tiny rotation: {tiny_angle} radians around X")
    print(f"Quaternion: {quat_tiny}")
    print(f"Near identity: {np.allclose(quat_tiny, [0, 0, 0, 1], atol=1e-6)}")

    # Edge Case 4: 180-degree rotations
    print("\n\n4. 180-DEGREE ROTATIONS")
    print("-" * 80)

    rot_180 = R.from_euler('z', 180, degrees=True)
    quat_180 = rot_180.as_quat()

    print(f"180° rotation around Z-axis")
    print(f"Quaternion: {quat_180}")
    print(f"Note: w component ≈ 0 for 180° rotations")
    print(f"The axis components determine the rotation axis.")


# ==============================================================================
# PART 6: Gaussian Splatting Specific Transformations
# ==============================================================================

def gaussian_splatting_coordinate_transform():
    """
    Demonstrate coordinate transformation for 3D Gaussian Splatting.

    Key formulas from the paper:
    - Covariance: Σ = R S S^T R^T
    - Where R is rotation matrix from quaternion
    - Where S is diagonal scale matrix
    """
    print("\n" + "=" * 80)
    print("GAUSSIAN SPLATTING COORDINATE TRANSFORMATIONS")
    print("=" * 80)

    print("\nIn 3D Gaussian Splatting:")
    print("- Each Gaussian has: position (3), rotation (4 quaternion), scale (3)")
    print("- Covariance matrix: Σ = R S S^T R^T")
    print("- Quaternions use WXYZ format in some implementations!")

    # Example Gaussian parameters
    position = np.array([1.0, 2.0, 3.0])
    quaternion_xyzw = np.array([0.0, 0.0, 0.7071, 0.7071])  # 90° around Z
    scale = np.array([2.0, 0.5, 0.3])

    print(f"\nExample Gaussian (in ZYX coordinate system):")
    print(f"  Position:             {position}")
    print(f"  Quaternion (x,y,z,w): {quaternion_xyzw}")
    print(f"  Scale:                {scale}")

    # Compute covariance in original system
    rot = R.from_quat(quaternion_xyzw)
    R_matrix = rot.as_matrix()
    S_matrix = np.diag(scale)

    covariance_old = R_matrix @ S_matrix @ S_matrix.T @ R_matrix.T

    print(f"\nCovariance matrix (old system):")
    print(covariance_old)

    # Transform to XYZ coordinate system
    # Permutation: (Z,Y,X) -> (X,Y,Z)
    P = np.array([
        [0, 0, 1],  # new_X = old_Z
        [0, 1, 0],  # new_Y = old_Y
        [1, 0, 0]   # new_Z = old_X
    ])

    # Transform position
    position_new = P @ position

    # Transform rotation (using rotation matrix method)
    R_new = P @ R_matrix @ P.T
    rot_new = R.from_matrix(R_new)
    quaternion_new = rot_new.as_quat()

    # Scale doesn't change in magnitude, but corresponds to different axes
    scale_new = P @ scale

    # Verify: compute covariance in new system
    covariance_new = R_new @ np.diag(scale_new) @ np.diag(scale_new).T @ R_new.T

    # Also compute by directly transforming old covariance
    covariance_transformed = P @ covariance_old @ P.T

    print(f"\nTransformed Gaussian (in XYZ coordinate system):")
    print(f"  Position:             {position_new}")
    print(f"  Quaternion (x,y,z,w): {quaternion_new}")
    print(f"  Scale:                {scale_new}")

    print(f"\nCovariance matrix (new system, computed from R',S'):")
    print(covariance_new)

    print(f"\nCovariance matrix (new system, transformed directly):")
    print(covariance_transformed)

    print(f"\nCovariances match: {np.allclose(covariance_new, covariance_transformed)}")

    print("\n⚠️  IMPORTANT NOTES:")
    print("    1. Check quaternion format: XYZW (SciPy) vs WXYZ (some implementations)")
    print("    2. Scale components are permuted along with axes")
    print("    3. Covariance transformation: Σ' = P Σ P^T")
    print("    4. Spherical harmonics (if present) need separate Wigner-D rotation!")


# ==============================================================================
# PART 7: Comprehensive Validation Tests
# ==============================================================================

def comprehensive_validation():
    """
    Run comprehensive validation tests.
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VALIDATION TESTS")
    print("=" * 80)

    tests_passed = 0
    tests_total = 0

    # Test 1: Rotation matrix method preserves rotation magnitude
    print("\nTest 1: Rotation matrix method preserves rotation magnitude")
    tests_total += 1

    quat1 = R.from_euler('xyz', [30, 45, 60], degrees=True).as_quat()
    P = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])  # Cyclic permutation
    quat1_transformed = correct_approach_rotation_matrix(quat1, P)

    # Both should be unit quaternions
    norm1 = np.linalg.norm(quat1)
    norm2 = np.linalg.norm(quat1_transformed)

    if np.allclose(norm1, 1.0) and np.allclose(norm2, 1.0):
        print("  ✓ PASS: Both quaternions are unit quaternions")
        tests_passed += 1
    else:
        print(f"  ✗ FAIL: Norms are {norm1:.6f} and {norm2:.6f}")

    # Test 2: Identity transformation
    print("\nTest 2: Identity transformation preserves quaternion")
    tests_total += 1

    quat2 = R.from_euler('xyz', [10, 20, 30], degrees=True).as_quat()
    I = np.eye(3)
    quat2_transformed = correct_approach_rotation_matrix(quat2, I)

    # Should be the same (or negated)
    match = np.allclose(quat2, quat2_transformed) or np.allclose(quat2, -quat2_transformed)
    if match:
        print("  ✓ PASS: Identity transformation preserves quaternion")
        tests_passed += 1
    else:
        print(f"  ✗ FAIL: Original {quat2} vs Transformed {quat2_transformed}")

    # Test 3: Orthogonal permutation preserves determinant magnitude
    print("\nTest 3: Orthogonal matrix preserves rotation matrix determinant")
    tests_total += 1

    quat3 = R.from_euler('x', 45, degrees=True).as_quat()
    P = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

    R1 = R.from_quat(quat3).as_matrix()
    R2 = P @ R1 @ P.T

    det1 = np.linalg.det(R1)
    det2 = np.linalg.det(R2)

    if np.allclose(abs(det1), 1.0) and np.allclose(abs(det2), 1.0):
        print(f"  ✓ PASS: Both determinants have magnitude 1 ({det1:.6f}, {det2:.6f})")
        tests_passed += 1
    else:
        print(f"  ✗ FAIL: Determinants are {det1:.6f} and {det2:.6f}")

    # Test 4: Round-trip transformation
    print("\nTest 4: Round-trip transformation (P then P^T) recovers original")
    tests_total += 1

    quat4 = R.from_euler('zyx', [15, 25, 35], degrees=True).as_quat()
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    quat4_step1 = correct_approach_rotation_matrix(quat4, P)
    quat4_step2 = correct_approach_rotation_matrix(quat4_step1, P.T)

    match = np.allclose(quat4, quat4_step2) or np.allclose(quat4, -quat4_step2)
    if match:
        print("  ✓ PASS: Round-trip recovers original quaternion")
        tests_passed += 1
    else:
        print(f"  ✗ FAIL: Original {quat4} vs Recovered {quat4_step2}")

    # Test 5: Consistency between rotation matrix and quaternion composition methods
    print("\nTest 5: Both methods produce equivalent results")
    tests_total += 1

    quat5 = R.from_euler('y', 60, degrees=True).as_quat()
    coord_rot = R.from_euler('x', 90, degrees=True)
    P = coord_rot.as_matrix()

    quat5_method1 = correct_approach_rotation_matrix(quat5, P)
    quat5_method2 = correct_approach_quaternion_composition(quat5, coord_rot)

    match = np.allclose(quat5_method1, quat5_method2) or np.allclose(quat5_method1, -quat5_method2)
    if match:
        print("  ✓ PASS: Both methods produce equivalent results")
        tests_passed += 1
    else:
        print(f"  ✗ FAIL: Method 1: {quat5_method1} vs Method 2: {quat5_method2}")

    # Summary
    print("\n" + "=" * 80)
    print(f"VALIDATION SUMMARY: {tests_passed}/{tests_total} tests passed")
    print("=" * 80)

    return tests_passed == tests_total


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("QUATERNION COORDINATE TRANSFORMATION RESEARCH")
    print("=" * 80)
    print("\nThis script demonstrates and validates methods for transforming")
    print("quaternions when changing coordinate systems.")
    print("\nKey References:")
    print("- SciPy Rotation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html")
    print("- Gaussian Splatting: https://github.com/graphdeco-inria/gaussian-splatting")
    print("- Math StackExchange: https://math.stackexchange.com/questions/1795351/")

    # Run all demonstrations
    demonstrate_quaternion_formats()
    demonstrate_wrong_approach()
    demonstrate_rotation_matrix_approach()
    demonstrate_quaternion_composition()
    demonstrate_edge_cases()
    gaussian_splatting_coordinate_transform()

    # Run validation
    all_passed = comprehensive_validation()

    if all_passed:
        print("\n✓ All validation tests passed!")
    else:
        print("\n⚠ Some validation tests failed. Review the output above.")

    print("\n" + "=" * 80)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 80)
    print("""
1. NEVER swap quaternion components directly - it changes chirality!

2. Use the rotation matrix method for general coordinate transformations:
   - Convert quaternion → rotation matrix
   - Apply transformation: R' = P @ R @ P^T
   - Convert back: rotation matrix → quaternion

3. For specific axis-aligned transformations, quaternion composition works well:
   - Create a rotation representing the coordinate change
   - Compose: R' = Q_coord * R * Q_coord^(-1)

4. Remember edge cases:
   - Quaternion sign ambiguity: q and -q represent the same rotation
   - Gimbal lock doesn't affect quaternions directly
   - Near-zero rotations are stable with quaternions
   - Check quaternion format: XYZW (SciPy) vs WXYZ (some implementations)

5. For Gaussian Splatting:
   - Transform position: p' = P @ p
   - Transform rotation: use rotation matrix method
   - Transform scale: s' = P @ s (permute components)
   - Verify with covariance: Σ' = P @ Σ @ P^T
   - Don't forget Spherical Harmonics need Wigner-D rotation!

6. Validation is crucial:
   - Check unit norm is preserved
   - Verify round-trip transformations
   - Test on known cases
   - Compare multiple methods for consistency
""")
