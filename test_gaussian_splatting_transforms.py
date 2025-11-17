"""
Comprehensive Test Suite for Gaussian Splatting Coordinate Transformations

This module contains test cases to validate coordinate transformations needed
for 3D Gaussian Splatting in napari. These tests ensure correctness of:
1. Vector (position) transformations
2. Rotation matrix transformations
3. Quaternion transformations
4. Covariance matrix transformations
5. Roundtrip transformations

Mathematical Background:
-----------------------
Gaussian splatting represents each 3D Gaussian with:
- Position: μ ∈ ℝ³ (3D vector)
- Rotation: R ∈ SO(3) (3x3 rotation matrix) or quaternion q
- Scale: s ∈ ℝ³ (3D scale vector)
- Covariance: Σ = R·S·Sᵀ·Rᵀ where S = diag(s)

Under an affine transformation T with rotation R_t, scale S_t, translation t:
- Position: μ' = R_t·S_t·μ + t
- Covariance: Σ' = (R_t·S_t)·Σ·(R_t·S_t)ᵀ
- Rotation: R' = R_t·R (when no scale, otherwise extract from decomposition)
"""

import numpy as np
import numpy.testing as npt
from scipy.spatial.transform import Rotation as R


# ============================================================================
# Helper Functions for Gaussian Splatting Transformations
# ============================================================================

def rotation_matrix_to_quaternion(rotation_matrix):
    """
    Convert a 3x3 rotation matrix to a quaternion [w, x, y, z].

    Mathematical reasoning:
    Quaternions provide a compact, singularity-free representation of 3D rotations.
    The conversion preserves the rotation while avoiding gimbal lock.

    Parameters
    ----------
    rotation_matrix : np.ndarray, shape (3, 3)
        Orthogonal rotation matrix with det(R) = +1

    Returns
    -------
    quaternion : np.ndarray, shape (4,)
        Unit quaternion [w, x, y, z] where w is the scalar part
    """
    rot = R.from_matrix(rotation_matrix)
    quat = rot.as_quat()  # Returns [x, y, z, w]
    return np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix.

    Parameters
    ----------
    quaternion : np.ndarray, shape (4,)
        Unit quaternion [w, x, y, z]

    Returns
    -------
    rotation_matrix : np.ndarray, shape (3, 3)
        Orthogonal rotation matrix
    """
    w, x, y, z = quaternion
    # Convert to scipy format [x, y, z, w]
    quat_scipy = np.array([x, y, z, w])
    rot = R.from_quat(quat_scipy)
    return rot.as_matrix()


def build_covariance_matrix(rotation, scale):
    """
    Build a 3D covariance matrix from rotation and scale.

    Mathematical reasoning:
    Σ = R·S·Sᵀ·Rᵀ where R is rotation matrix, S = diag(scale)
    This represents an anisotropic Gaussian with principal axes along R
    and standard deviations given by scale.

    Parameters
    ----------
    rotation : np.ndarray, shape (3, 3)
        Rotation matrix
    scale : np.ndarray, shape (3,)
        Scale factors (standard deviations) along each principal axis

    Returns
    -------
    covariance : np.ndarray, shape (3, 3)
        Symmetric positive definite covariance matrix
    """
    S = np.diag(scale)
    return rotation @ S @ S.T @ rotation.T


def transform_covariance_matrix(covariance, transformation_matrix):
    """
    Transform a covariance matrix under an affine transformation.

    Mathematical reasoning:
    Given covariance Σ and linear transform A (the linear part of affine transform),
    the transformed covariance is: Σ' = A·Σ·Aᵀ

    This follows from the transformation of a multivariate Gaussian:
    If X ~ N(μ, Σ), then A·X ~ N(A·μ, A·Σ·Aᵀ)

    Parameters
    ----------
    covariance : np.ndarray, shape (3, 3)
        Original covariance matrix
    transformation_matrix : np.ndarray, shape (3, 3)
        Linear transformation matrix (rotation + scale part of affine)

    Returns
    -------
    transformed_covariance : np.ndarray, shape (3, 3)
        Transformed covariance matrix
    """
    return transformation_matrix @ covariance @ transformation_matrix.T


# ============================================================================
# Test 1: Vector Transformation (Simple Case)
# ============================================================================

class TestVectorTransformation:
    """Test basic vector transformations for Gaussian positions."""

    def test_identity_transform(self):
        """
        Test Case: Identity transformation should leave vectors unchanged.

        Mathematical reasoning:
        I·v = v for any vector v

        Input: position = [1.0, 2.0, 3.0]
        Expected output: [1.0, 2.0, 3.0]
        Tolerance: 1e-10 (numerical precision)
        """
        position = np.array([1.0, 2.0, 3.0])
        identity = np.eye(3)
        translation = np.zeros(3)

        # Apply transformation: p' = I·p + 0
        transformed = identity @ position + translation

        npt.assert_allclose(transformed, position, rtol=1e-10, atol=1e-10)
        print("✓ Identity transform test passed")

    def test_translation_only(self):
        """
        Test Case: Pure translation (no rotation or scale).

        Mathematical reasoning:
        v' = v + t

        Input: position = [1.0, 2.0, 3.0], translation = [10.0, 20.0, 30.0]
        Expected output: [11.0, 22.0, 33.0]
        """
        position = np.array([1.0, 2.0, 3.0])
        translation = np.array([10.0, 20.0, 30.0])

        transformed = position + translation
        expected = np.array([11.0, 22.0, 33.0])

        npt.assert_allclose(transformed, expected, rtol=1e-10)
        print("✓ Translation-only test passed")

    def test_uniform_scale(self):
        """
        Test Case: Uniform scaling (isotropic).

        Mathematical reasoning:
        v' = s·v where s is a scalar

        Input: position = [1.0, 2.0, 3.0], scale = 2.0
        Expected output: [2.0, 4.0, 6.0]
        """
        position = np.array([1.0, 2.0, 3.0])
        scale = 2.0

        transformed = scale * position
        expected = np.array([2.0, 4.0, 6.0])

        npt.assert_allclose(transformed, expected, rtol=1e-10)
        print("✓ Uniform scale test passed")

    def test_anisotropic_scale(self):
        """
        Test Case: Non-uniform scaling (anisotropic).

        Mathematical reasoning:
        v' = S·v where S = diag(sx, sy, sz)

        Input: position = [1.0, 2.0, 3.0], scale = [2.0, 3.0, 4.0]
        Expected output: [2.0, 6.0, 12.0]
        """
        position = np.array([1.0, 2.0, 3.0])
        scale = np.array([2.0, 3.0, 4.0])

        transformed = scale * position
        expected = np.array([2.0, 6.0, 12.0])

        npt.assert_allclose(transformed, expected, rtol=1e-10)
        print("✓ Anisotropic scale test passed")

    def test_combined_transform(self):
        """
        Test Case: Combined rotation, scale, and translation.

        Mathematical reasoning:
        v' = R·S·v + t where R is rotation, S is scale, t is translation

        Input: position = [1.0, 0.0, 0.0]
        Rotation: 90° around Z-axis
        Scale: [2.0, 2.0, 2.0]
        Translation: [1.0, 1.0, 1.0]

        Expected:
        - After scale: [2.0, 0.0, 0.0]
        - After rotation: [0.0, 2.0, 0.0] (90° rotation moves x to y)
        - After translation: [1.0, 3.0, 1.0]
        """
        position = np.array([1.0, 0.0, 0.0])

        # 90-degree rotation around Z-axis
        rotation = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ], dtype=float)

        scale = np.array([2.0, 2.0, 2.0])
        translation = np.array([1.0, 1.0, 1.0])

        # Apply transformations in order: scale, rotate, translate
        transformed = rotation @ (scale * position) + translation
        expected = np.array([1.0, 3.0, 1.0])

        npt.assert_allclose(transformed, expected, rtol=1e-10, atol=1e-10)
        print("✓ Combined transform test passed")


# ============================================================================
# Test 2: Rotation Matrix Transformation
# ============================================================================

class TestRotationMatrix:
    """Test rotation matrix properties and transformations."""

    def test_rotation_matrix_orthogonality(self):
        """
        Test Case: Rotation matrices must be orthogonal (R·Rᵀ = I).

        Mathematical reasoning:
        R ∈ SO(3) requires R·Rᵀ = I and det(R) = +1
        This ensures the transformation preserves lengths and angles.
        """
        # 45-degree rotation around Z-axis
        angle = np.deg2rad(45)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1]
        ])

        # Check orthogonality: R·Rᵀ = I
        result = rotation @ rotation.T
        npt.assert_allclose(result, np.eye(3), rtol=1e-10, atol=1e-10)

        # Check determinant = +1
        det = np.linalg.det(rotation)
        npt.assert_allclose(det, 1.0, rtol=1e-10)

        print("✓ Rotation matrix orthogonality test passed")

    def test_90_degree_rotation_x(self):
        """
        Test Case: 90-degree rotation around X-axis.

        Mathematical reasoning:
        Rotation matrix for θ degrees around X-axis:
        R_x(θ) = [1    0         0     ]
                 [0  cos(θ)  -sin(θ)]
                 [0  sin(θ)   cos(θ)]

        For θ = 90°:
        R_x(90°) = [1   0   0]
                   [0   0  -1]
                   [0   1   0]

        Input: v = [0, 1, 0] (unit vector along Y)
        Expected: v' = [0, 0, 1] (rotates to Z)
        """
        rotation = np.array([
            [1,  0,  0],
            [0,  0, -1],
            [0,  1,  0]
        ], dtype=float)

        vector = np.array([0.0, 1.0, 0.0])
        transformed = rotation @ vector
        expected = np.array([0.0, 0.0, 1.0])

        npt.assert_allclose(transformed, expected, rtol=1e-10, atol=1e-10)
        print("✓ 90-degree X rotation test passed")

    def test_90_degree_rotation_y(self):
        """
        Test Case: 90-degree rotation around Y-axis.

        Input: v = [1, 0, 0] (unit vector along X)
        Expected: v' = [0, 0, -1] (rotates to -Z)
        """
        rotation = np.array([
            [ 0,  0,  1],
            [ 0,  1,  0],
            [-1,  0,  0]
        ], dtype=float)

        vector = np.array([1.0, 0.0, 0.0])
        transformed = rotation @ vector
        expected = np.array([0.0, 0.0, -1.0])

        npt.assert_allclose(transformed, expected, rtol=1e-10, atol=1e-10)
        print("✓ 90-degree Y rotation test passed")

    def test_90_degree_rotation_z(self):
        """
        Test Case: 90-degree rotation around Z-axis.

        Input: v = [1, 0, 0] (unit vector along X)
        Expected: v' = [0, 1, 0] (rotates to Y)
        """
        rotation = np.array([
            [0, -1,  0],
            [1,  0,  0],
            [0,  0,  1]
        ], dtype=float)

        vector = np.array([1.0, 0.0, 0.0])
        transformed = rotation @ vector
        expected = np.array([0.0, 1.0, 0.0])

        npt.assert_allclose(transformed, expected, rtol=1e-10, atol=1e-10)
        print("✓ 90-degree Z rotation test passed")

    def test_rotation_composition(self):
        """
        Test Case: Composition of rotations R2·R1 should equal combined rotation.

        Mathematical reasoning:
        Rotations compose via matrix multiplication: (R2·R1)·v = R2·(R1·v)

        Test: 90° around X, then 90° around Z
        """
        # 90° around X
        R_x = np.array([
            [1,  0,  0],
            [0,  0, -1],
            [0,  1,  0]
        ], dtype=float)

        # 90° around Z
        R_z = np.array([
            [0, -1,  0],
            [1,  0,  0],
            [0,  0,  1]
        ], dtype=float)

        vector = np.array([1.0, 0.0, 0.0])

        # Method 1: Apply sequentially
        v1 = R_x @ vector  # [1, 0, 0] -> [1, 0, 0]
        v2 = R_z @ v1      # [1, 0, 0] -> [0, 1, 0]

        # Method 2: Compose rotations first
        R_combined = R_z @ R_x
        v3 = R_combined @ vector

        npt.assert_allclose(v2, v3, rtol=1e-10, atol=1e-10)
        print("✓ Rotation composition test passed")


# ============================================================================
# Test 3: Quaternion Transformation
# ============================================================================

class TestQuaternion:
    """Test quaternion representations and conversions."""

    def test_identity_quaternion(self):
        """
        Test Case: Identity quaternion [1, 0, 0, 0] produces identity rotation.

        Mathematical reasoning:
        The identity quaternion q = (1, 0, 0, 0) represents no rotation.
        Its rotation matrix should be the identity matrix.
        """
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])
        rotation = quaternion_to_rotation_matrix(identity_quat)

        npt.assert_allclose(rotation, np.eye(3), rtol=1e-10, atol=1e-10)
        print("✓ Identity quaternion test passed")

    def test_quaternion_normalization(self):
        """
        Test Case: Quaternions must be unit quaternions (||q|| = 1).

        Mathematical reasoning:
        Only unit quaternions represent valid rotations.
        Non-unit quaternions should be normalized: q' = q / ||q||
        """
        # Non-normalized quaternion
        quat = np.array([2.0, 0.0, 0.0, 0.0])
        norm = np.linalg.norm(quat)

        # Normalize
        quat_normalized = quat / norm
        norm_after = np.linalg.norm(quat_normalized)

        npt.assert_allclose(norm_after, 1.0, rtol=1e-10)
        npt.assert_allclose(quat_normalized, [1, 0, 0, 0], rtol=1e-10)
        print("✓ Quaternion normalization test passed")

    def test_quaternion_to_matrix_90_degree_x(self):
        """
        Test Case: Quaternion for 90° rotation around X-axis.

        Mathematical reasoning:
        For rotation by θ around axis (x, y, z):
        q = [cos(θ/2), x·sin(θ/2), y·sin(θ/2), z·sin(θ/2)]

        For 90° around X (axis = [1, 0, 0]):
        q = [cos(45°), sin(45°), 0, 0] = [√2/2, √2/2, 0, 0]
        """
        # Quaternion for 90° rotation around X-axis
        sqrt2_2 = np.sqrt(2) / 2
        quat = np.array([sqrt2_2, sqrt2_2, 0.0, 0.0])

        rotation = quaternion_to_rotation_matrix(quat)

        expected = np.array([
            [1,  0,  0],
            [0,  0, -1],
            [0,  1,  0]
        ], dtype=float)

        npt.assert_allclose(rotation, expected, rtol=1e-10, atol=1e-10)
        print("✓ Quaternion 90° X rotation test passed")

    def test_quaternion_to_matrix_90_degree_z(self):
        """
        Test Case: Quaternion for 90° rotation around Z-axis.

        For 90° around Z (axis = [0, 0, 1]):
        q = [cos(45°), 0, 0, sin(45°)] = [√2/2, 0, 0, √2/2]
        """
        sqrt2_2 = np.sqrt(2) / 2
        quat = np.array([sqrt2_2, 0.0, 0.0, sqrt2_2])

        rotation = quaternion_to_rotation_matrix(quat)

        expected = np.array([
            [0, -1,  0],
            [1,  0,  0],
            [0,  0,  1]
        ], dtype=float)

        npt.assert_allclose(rotation, expected, rtol=1e-10, atol=1e-10)
        print("✓ Quaternion 90° Z rotation test passed")

    def test_matrix_to_quaternion_to_matrix(self):
        """
        Test Case: Roundtrip conversion matrix -> quaternion -> matrix.

        Mathematical reasoning:
        Converting R -> q -> R' should yield R' = R (up to numerical precision).

        Note: Quaternions have double cover (q and -q represent same rotation),
        so we test the rotation matrix equality, not quaternion equality.
        """
        # 45-degree rotation around arbitrary axis [1, 1, 1]
        axis = np.array([1, 1, 1])
        axis = axis / np.linalg.norm(axis)
        angle = np.deg2rad(45)

        original_rotation = R.from_rotvec(angle * axis).as_matrix()

        # Convert to quaternion and back
        quat = rotation_matrix_to_quaternion(original_rotation)
        reconstructed_rotation = quaternion_to_rotation_matrix(quat)

        npt.assert_allclose(reconstructed_rotation, original_rotation, rtol=1e-10, atol=1e-10)
        print("✓ Matrix->Quaternion->Matrix roundtrip test passed")


# ============================================================================
# Test 4: Covariance Matrix Transformation
# ============================================================================

class TestCovarianceTransformation:
    """Test covariance matrix transformations for Gaussian splatting."""

    def test_isotropic_gaussian_identity(self):
        """
        Test Case: Isotropic (spherical) Gaussian with identity transform.

        Mathematical reasoning:
        For isotropic Gaussian: Σ = σ²·I
        Under identity transform: Σ' = I·Σ·Iᵀ = Σ

        Input: Σ = diag([1, 1, 1])
        Transform: I (identity)
        Expected: Σ' = diag([1, 1, 1])
        """
        covariance = np.eye(3)
        transform = np.eye(3)

        transformed = transform_covariance_matrix(covariance, transform)

        npt.assert_allclose(transformed, covariance, rtol=1e-10, atol=1e-10)
        print("✓ Isotropic Gaussian identity test passed")

    def test_isotropic_gaussian_uniform_scale(self):
        """
        Test Case: Isotropic Gaussian with uniform scaling.

        Mathematical reasoning:
        Σ = σ²·I, transform T = s·I (uniform scale)
        Σ' = s²·σ²·I

        Input: Σ = diag([1, 1, 1]), scale = 2
        Expected: Σ' = diag([4, 4, 4])
        """
        covariance = np.eye(3)
        scale = 2.0
        transform = scale * np.eye(3)

        transformed = transform_covariance_matrix(covariance, transform)
        expected = (scale ** 2) * np.eye(3)

        npt.assert_allclose(transformed, expected, rtol=1e-10, atol=1e-10)
        print("✓ Isotropic Gaussian uniform scale test passed")

    def test_anisotropic_gaussian_scale(self):
        """
        Test Case: Anisotropic Gaussian with non-uniform scaling.

        Mathematical reasoning:
        Σ = diag([σx², σy², σz²])
        Transform: S = diag([sx, sy, sz])
        Σ' = S·Σ·Sᵀ = diag([sx²·σx², sy²·σy², sz²·σz²])

        Input: Σ = diag([1, 2, 3]), S = diag([2, 3, 4])
        Expected: Σ' = diag([4, 18, 48])
        """
        covariance = np.diag([1.0, 2.0, 3.0])
        scale = np.diag([2.0, 3.0, 4.0])

        transformed = transform_covariance_matrix(covariance, scale)
        expected = np.diag([4.0, 18.0, 48.0])

        npt.assert_allclose(transformed, expected, rtol=1e-10, atol=1e-10)
        print("✓ Anisotropic Gaussian scale test passed")

    def test_covariance_rotation_invariant_trace(self):
        """
        Test Case: Trace of covariance is invariant under rotation.

        Mathematical reasoning:
        For rotation R: tr(R·Σ·Rᵀ) = tr(Σ)
        This is because rotation preserves the sum of variances.

        Proof: tr(R·Σ·Rᵀ) = tr(Rᵀ·R·Σ) = tr(I·Σ) = tr(Σ)
        """
        # Create arbitrary covariance matrix
        rotation_init = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
        scale = np.array([1.0, 2.0, 3.0])
        covariance = build_covariance_matrix(rotation_init, scale)

        # Rotate by 90° around Z
        rotation_transform = np.array([
            [0, -1,  0],
            [1,  0,  0],
            [0,  0,  1]
        ], dtype=float)

        transformed = transform_covariance_matrix(covariance, rotation_transform)

        # Trace should be preserved
        trace_original = np.trace(covariance)
        trace_transformed = np.trace(transformed)

        npt.assert_allclose(trace_transformed, trace_original, rtol=1e-10, atol=1e-10)
        print("✓ Covariance rotation trace invariance test passed")

    def test_covariance_symmetry(self):
        """
        Test Case: Covariance matrices must remain symmetric.

        Mathematical reasoning:
        Σ is symmetric (Σ = Σᵀ)
        Under transform A: Σ' = A·Σ·Aᵀ
        Σ'ᵀ = (A·Σ·Aᵀ)ᵀ = A·Σᵀ·Aᵀ = A·Σ·Aᵀ = Σ'
        """
        # Create arbitrary covariance
        rotation = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
        scale = np.array([1.0, 2.0, 3.0])
        covariance = build_covariance_matrix(rotation, scale)

        # Apply arbitrary transformation
        transform = R.from_euler('xyz', [15, 25, 35], degrees=True).as_matrix()
        transform = transform @ np.diag([1.5, 2.0, 2.5])

        transformed = transform_covariance_matrix(covariance, transform)

        # Check symmetry
        npt.assert_allclose(transformed, transformed.T, rtol=1e-10, atol=1e-10)
        print("✓ Covariance symmetry test passed")

    def test_covariance_positive_definite(self):
        """
        Test Case: Covariance matrices must remain positive definite.

        Mathematical reasoning:
        If Σ is positive definite and A is invertible, then A·Σ·Aᵀ is positive definite.

        Verification: All eigenvalues must be positive.
        """
        # Create positive definite covariance
        rotation = np.eye(3)
        scale = np.array([1.0, 2.0, 3.0])
        covariance = build_covariance_matrix(rotation, scale)

        # Apply invertible transformation
        transform = R.from_euler('z', 45, degrees=True).as_matrix()
        transform = transform @ np.diag([2.0, 2.0, 2.0])

        transformed = transform_covariance_matrix(covariance, transform)

        # Check all eigenvalues are positive
        eigenvalues = np.linalg.eigvals(transformed)
        assert np.all(eigenvalues > 0), f"Eigenvalues should be positive, got {eigenvalues}"
        print("✓ Covariance positive definiteness test passed")

    def test_full_gaussian_transformation(self):
        """
        Test Case: Complete transformation of a 3D Gaussian.

        Mathematical reasoning:
        Transform a Gaussian with position μ, rotation R, scale s:
        - Covariance: Σ = R·S·Sᵀ·Rᵀ where S = diag(s)
        - Under transform with matrix A and translation t:
          - μ' = A·μ + t
          - Σ' = A·Σ·Aᵀ

        This test verifies the complete pipeline.
        """
        # Original Gaussian parameters
        position = np.array([1.0, 2.0, 3.0])
        rotation = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
        scale = np.array([0.5, 1.0, 1.5])
        covariance = build_covariance_matrix(rotation, scale)

        # Transformation: 90° rotation around Z + scale [2, 2, 2] + translation
        transform_rotation = np.array([
            [0, -1,  0],
            [1,  0,  0],
            [0,  0,  1]
        ], dtype=float)
        transform_scale = np.diag([2.0, 2.0, 2.0])
        transform_matrix = transform_rotation @ transform_scale
        translation = np.array([5.0, 10.0, 15.0])

        # Apply transformations
        new_position = transform_matrix @ position + translation
        new_covariance = transform_covariance_matrix(covariance, transform_matrix)

        # Verify covariance is still symmetric and positive definite
        npt.assert_allclose(new_covariance, new_covariance.T, rtol=1e-10, atol=1e-10)
        eigenvalues = np.linalg.eigvals(new_covariance)
        assert np.all(eigenvalues > 0)

        print("✓ Full Gaussian transformation test passed")


# ============================================================================
# Test 5: Roundtrip Transformations
# ============================================================================

class TestRoundtripTransformations:
    """Test that transformation roundtrips preserve data (A→B→A = A)."""

    def test_position_roundtrip(self):
        """
        Test Case: Position transformation roundtrip.

        Mathematical reasoning:
        If T·v + t is the forward transform, then:
        T⁻¹·(v' - t) should give back v

        For invertible transform: T⁻¹·T = I
        """
        position = np.array([1.5, 2.3, 3.7])

        # Forward transform
        transform = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
        transform = transform @ np.diag([2.0, 3.0, 4.0])
        translation = np.array([10.0, 20.0, 30.0])

        # Forward
        transformed = transform @ position + translation

        # Inverse
        transform_inv = np.linalg.inv(transform)
        recovered = transform_inv @ (transformed - translation)

        npt.assert_allclose(recovered, position, rtol=1e-10, atol=1e-10)
        print("✓ Position roundtrip test passed")

    def test_rotation_roundtrip(self):
        """
        Test Case: Rotation matrix roundtrip.

        Mathematical reasoning:
        R·Rᵀ = I for any rotation matrix R
        Therefore: Rᵀ·R·v = v
        """
        vector = np.array([1.0, 2.0, 3.0])
        rotation = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()

        # Forward
        rotated = rotation @ vector

        # Inverse (transpose for rotation matrices)
        recovered = rotation.T @ rotated

        npt.assert_allclose(recovered, vector, rtol=1e-10, atol=1e-10)
        print("✓ Rotation roundtrip test passed")

    def test_covariance_roundtrip(self):
        """
        Test Case: Covariance transformation roundtrip.

        Mathematical reasoning:
        Forward: Σ' = A·Σ·Aᵀ
        Inverse: Σ = A⁻¹·Σ'·(A⁻¹)ᵀ

        Therefore: A⁻¹·A·Σ·Aᵀ·(A⁻¹)ᵀ = Σ
        """
        # Create initial covariance
        rotation = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
        scale = np.array([1.0, 2.0, 3.0])
        covariance = build_covariance_matrix(rotation, scale)

        # Transform
        transform = R.from_euler('xyz', [15, 25, 35], degrees=True).as_matrix()
        transform = transform @ np.diag([2.0, 3.0, 4.0])

        # Forward
        transformed = transform_covariance_matrix(covariance, transform)

        # Inverse
        transform_inv = np.linalg.inv(transform)
        recovered = transform_covariance_matrix(transformed, transform_inv)

        npt.assert_allclose(recovered, covariance, rtol=1e-9, atol=1e-9)
        print("✓ Covariance roundtrip test passed")

    def test_quaternion_roundtrip(self):
        """
        Test Case: Quaternion conversion roundtrip.

        Mathematical reasoning:
        q -> R -> q' should give q' = ±q (double cover)
        But R -> q -> R' should give R' = R

        We test the matrix equality.
        """
        # Start with a quaternion
        original_quat = np.array([0.5, 0.5, 0.5, 0.5])  # Must be normalized
        original_quat = original_quat / np.linalg.norm(original_quat)

        # Convert to matrix
        matrix = quaternion_to_rotation_matrix(original_quat)

        # Convert back to quaternion
        recovered_quat = rotation_matrix_to_quaternion(matrix)

        # Convert both to matrices to compare (handles double cover)
        recovered_matrix = quaternion_to_rotation_matrix(recovered_quat)

        npt.assert_allclose(recovered_matrix, matrix, rtol=1e-10, atol=1e-10)
        print("✓ Quaternion roundtrip test passed")

    def test_multiple_transforms_roundtrip(self):
        """
        Test Case: Chain of transformations with final inverse.

        Mathematical reasoning:
        T3⁻¹·T2⁻¹·T1⁻¹·T1·T2·T3·v = v
        """
        position = np.array([1.0, 2.0, 3.0])

        # Create three different transforms
        T1 = R.from_euler('x', 30, degrees=True).as_matrix()
        T2 = R.from_euler('y', 45, degrees=True).as_matrix()
        T3 = R.from_euler('z', 60, degrees=True).as_matrix()

        # Forward chain
        v1 = T1 @ position
        v2 = T2 @ v1
        v3 = T3 @ v2

        # Backward chain
        v4 = T3.T @ v3
        v5 = T2.T @ v4
        v6 = T1.T @ v5

        npt.assert_allclose(v6, position, rtol=1e-10, atol=1e-10)
        print("✓ Multiple transforms roundtrip test passed")


# ============================================================================
# Edge Cases and Numerical Precision Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and numerical precision limits."""

    def test_degenerate_zero_scale(self):
        """
        Test Case: Zero scale factor (degenerate case).

        Mathematical reasoning:
        Scale of 0 is mathematically valid but creates a degenerate transformation.
        The transformation is non-invertible.

        Edge case: We should handle this without errors, but note singularity.
        """
        position = np.array([1.0, 2.0, 3.0])
        scale = np.array([0.0, 1.0, 1.0])  # Zero scale in X

        result = scale * position
        expected = np.array([0.0, 2.0, 3.0])

        npt.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

        # Verify transformation is singular (non-invertible)
        transform_matrix = np.diag(scale)
        det = np.linalg.det(transform_matrix)
        assert det == 0.0, "Zero scale should create singular matrix"

        print("✓ Zero scale edge case test passed")

    def test_near_singular_matrix(self):
        """
        Test Case: Near-singular matrix (numerical stability).

        Mathematical reasoning:
        Very small scale factors can cause numerical instability.
        Test with scale = [1e-10, 1, 1] to check stability.
        """
        position = np.array([1.0, 2.0, 3.0])
        small_scale = 1e-10
        scale = np.array([small_scale, 1.0, 1.0])

        transform = np.diag(scale)
        result = transform @ position
        expected = np.array([small_scale, 2.0, 3.0])

        # Use absolute tolerance for very small values
        npt.assert_allclose(result, expected, rtol=1e-8, atol=1e-15)
        print("✓ Near-singular matrix test passed")

    def test_numerical_precision_accumulation(self):
        """
        Test Case: Numerical precision accumulation in repeated transformations.

        Mathematical reasoning:
        Applying the same transformation N times then its inverse N times
        should return to original (with accumulated error).

        Error typically grows as O(ε·N) where ε is machine epsilon.
        """
        position = np.array([1.0, 2.0, 3.0])
        rotation = R.from_euler('xyz', [1, 2, 3], degrees=True).as_matrix()

        # Apply rotation 100 times
        result = position.copy()
        for _ in range(100):
            result = rotation @ result

        # Apply inverse 100 times
        rotation_inv = rotation.T
        for _ in range(100):
            result = rotation_inv @ result

        # Should be close to original, with some accumulated error
        # Allow larger tolerance due to error accumulation
        npt.assert_allclose(result, position, rtol=1e-8, atol=1e-8)
        print("✓ Numerical precision accumulation test passed")

    def test_very_large_scale(self):
        """
        Test Case: Very large scale factors.

        Mathematical reasoning:
        Large scales should work correctly within numerical limits.
        Test with scale = 1e10 to check for overflow.
        """
        position = np.array([1.0, 2.0, 3.0])
        large_scale = 1e10

        result = large_scale * position
        expected = np.array([1e10, 2e10, 3e10])

        npt.assert_allclose(result, expected, rtol=1e-10)
        print("✓ Very large scale test passed")

    def test_orthogonality_after_many_operations(self):
        """
        Test Case: Rotation matrix orthogonality after many operations.

        Mathematical reasoning:
        Repeated matrix operations can accumulate numerical errors,
        potentially violating orthogonality constraint R·Rᵀ = I.
        """
        # Start with identity
        rotation = np.eye(3)

        # Apply many small rotations
        small_rotation = R.from_euler('z', 1, degrees=True).as_matrix()
        for _ in range(360):  # 360 rotations of 1° = 360° total
            rotation = small_rotation @ rotation

        # Should be close to identity (full rotation)
        npt.assert_allclose(rotation, np.eye(3), rtol=1e-8, atol=1e-8)

        # Check orthogonality is maintained
        orthogonality_check = rotation @ rotation.T
        npt.assert_allclose(orthogonality_check, np.eye(3), rtol=1e-8, atol=1e-8)

        print("✓ Orthogonality preservation test passed")

    def test_gimbal_lock_quaternion(self):
        """
        Test Case: Gimbal lock situation with quaternions.

        Mathematical reasoning:
        Euler angles suffer from gimbal lock at pitch = ±90°.
        Quaternions should handle this gracefully.

        Test: 90° pitch rotation
        """
        # Create rotation with gimbal lock condition (90° pitch)
        rotation = R.from_euler('xyz', [30, 90, 60], degrees=True).as_matrix()

        # Convert to quaternion and back
        quat = rotation_matrix_to_quaternion(rotation)
        recovered = quaternion_to_rotation_matrix(quat)

        # Should recover the same rotation despite gimbal lock
        npt.assert_allclose(recovered, rotation, rtol=1e-10, atol=1e-10)
        print("✓ Gimbal lock quaternion test passed")


# ============================================================================
# Main Test Runner
# ============================================================================

def run_all_tests():
    """
    Run all test suites and print summary.

    Returns
    -------
    bool
        True if all tests passed, False otherwise
    """
    print("\n" + "=" * 80)
    print("GAUSSIAN SPLATTING COORDINATE TRANSFORMATION TEST SUITE")
    print("=" * 80 + "\n")

    test_classes = [
        ("Vector Transformations", TestVectorTransformation),
        ("Rotation Matrix", TestRotationMatrix),
        ("Quaternion", TestQuaternion),
        ("Covariance Transformation", TestCovarianceTransformation),
        ("Roundtrip Transformations", TestRoundtripTransformations),
        ("Edge Cases", TestEdgeCases),
    ]

    all_passed = True

    for suite_name, test_class in test_classes:
        print(f"\n{'─' * 80}")
        print(f"Test Suite: {suite_name}")
        print(f"{'─' * 80}")

        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith('test_')]

        for method_name in test_methods:
            try:
                method = getattr(instance, method_name)
                method()
            except AssertionError as e:
                print(f"✗ {method_name} FAILED: {e}")
                all_passed = False
            except Exception as e:
                print(f"✗ {method_name} ERROR: {e}")
                all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
