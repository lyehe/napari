"""
Practical Example: Using Coordinate Transformations for Gaussian Splatting

This script demonstrates how to use the transformation test suite to validate
a real Gaussian splatting implementation in napari.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


# ============================================================================
# Gaussian Splat Data Structure
# ============================================================================

class GaussianSplat:
    """
    Represents a single 3D Gaussian splat.

    Attributes
    ----------
    position : np.ndarray, shape (3,)
        3D position (mean) of the Gaussian
    rotation : np.ndarray, shape (3, 3)
        Rotation matrix defining principal axes
    scale : np.ndarray, shape (3,)
        Scale factors (standard deviations) along principal axes
    opacity : float
        Opacity value in [0, 1]
    color : np.ndarray, shape (3,)
        RGB color in [0, 1]
    """

    def __init__(self, position, rotation=None, scale=None, opacity=1.0, color=None):
        self.position = np.asarray(position, dtype=float)

        if rotation is None:
            rotation = np.eye(3)
        self.rotation = np.asarray(rotation, dtype=float)

        if scale is None:
            scale = np.ones(3)
        self.scale = np.asarray(scale, dtype=float)

        self.opacity = opacity

        if color is None:
            color = np.array([1.0, 1.0, 1.0])
        self.color = np.asarray(color, dtype=float)

    @property
    def covariance(self):
        """Compute the 3D covariance matrix."""
        S = np.diag(self.scale)
        return self.rotation @ S @ S.T @ self.rotation.T

    @property
    def quaternion(self):
        """Get rotation as quaternion [w, x, y, z]."""
        rot = R.from_matrix(self.rotation)
        quat = rot.as_quat()  # Returns [x, y, z, w]
        return np.array([quat[3], quat[0], quat[1], quat[2]])

    def transform(self, affine_matrix, translation):
        """
        Apply an affine transformation to this Gaussian.

        Parameters
        ----------
        affine_matrix : np.ndarray, shape (3, 3)
            Linear part of transformation (rotation + scale)
        translation : np.ndarray, shape (3,)
            Translation vector

        Returns
        -------
        GaussianSplat
            New transformed Gaussian
        """
        # Transform position
        new_position = affine_matrix @ self.position + translation

        # Transform covariance
        new_covariance = affine_matrix @ self.covariance @ affine_matrix.T

        # Decompose new covariance to get rotation and scale
        # Using eigendecomposition: Σ = V·Λ·V^T where V is rotation, Λ is scale²
        eigenvalues, eigenvectors = np.linalg.eigh(new_covariance)

        # Ensure positive eigenvalues and sort in descending order
        eigenvalues = np.abs(eigenvalues)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        new_scale = np.sqrt(eigenvalues)
        new_rotation = eigenvectors

        # Ensure proper rotation (det = +1)
        if np.linalg.det(new_rotation) < 0:
            new_rotation[:, -1] *= -1

        return GaussianSplat(
            position=new_position,
            rotation=new_rotation,
            scale=new_scale,
            opacity=self.opacity,
            color=self.color
        )

    def __repr__(self):
        return (f"GaussianSplat(position={self.position}, "
                f"scale={self.scale}, opacity={self.opacity})")


# ============================================================================
# Example 1: Creating and Transforming a Single Gaussian
# ============================================================================

def example_single_gaussian():
    """Demonstrate transformation of a single Gaussian."""
    print("\n" + "=" * 80)
    print("Example 1: Single Gaussian Transformation")
    print("=" * 80 + "\n")

    # Create a Gaussian at the origin with specific orientation
    gaussian = GaussianSplat(
        position=[0.0, 0.0, 0.0],
        rotation=R.from_euler('xyz', [0, 0, 45], degrees=True).as_matrix(),
        scale=[1.0, 2.0, 0.5],  # Elongated along Y
        opacity=0.8,
        color=[1.0, 0.0, 0.0]  # Red
    )

    print(f"Original Gaussian:")
    print(f"  Position: {gaussian.position}")
    print(f"  Scale: {gaussian.scale}")
    print(f"  Rotation (Euler XYZ): {R.from_matrix(gaussian.rotation).as_euler('xyz', degrees=True)}")
    print(f"  Covariance:\n{gaussian.covariance}")

    # Create a transformation: 90° rotation around Z + translation
    transform_matrix = R.from_euler('z', 90, degrees=True).as_matrix()
    transform_matrix = transform_matrix @ np.diag([2.0, 2.0, 2.0])  # Add 2x scale
    translation = np.array([5.0, 10.0, 15.0])

    print(f"\nTransformation:")
    print(f"  Matrix:\n{transform_matrix}")
    print(f"  Translation: {translation}")

    # Apply transformation
    transformed = gaussian.transform(transform_matrix, translation)

    print(f"\nTransformed Gaussian:")
    print(f"  Position: {transformed.position}")
    print(f"  Scale: {transformed.scale}")
    print(f"  Rotation (Euler XYZ): {R.from_matrix(transformed.rotation).as_euler('xyz', degrees=True)}")
    print(f"  Covariance:\n{transformed.covariance}")

    # Verify properties are preserved
    print(f"\nVerification:")
    print(f"  Original trace(Σ): {np.trace(gaussian.covariance):.6f}")
    print(f"  Transformed trace(Σ): {np.trace(transformed.covariance):.6f}")
    print(f"  Ratio: {np.trace(transformed.covariance) / np.trace(gaussian.covariance):.6f}")
    print(f"  Expected ratio (for 2x scale): {4.0:.6f}")  # scale² = 4


# ============================================================================
# Example 2: Batch Transformation of Multiple Gaussians
# ============================================================================

def example_batch_gaussians():
    """Demonstrate efficient batch transformation of multiple Gaussians."""
    print("\n" + "=" * 80)
    print("Example 2: Batch Gaussian Transformation")
    print("=" * 80 + "\n")

    # Create a grid of Gaussians
    n_gaussians = 27  # 3x3x3 grid
    gaussians = []

    print(f"Creating {n_gaussians} Gaussians in a 3×3×3 grid...")

    for x in range(3):
        for y in range(3):
            for z in range(3):
                position = np.array([x * 2.0, y * 2.0, z * 2.0])

                # Random rotation
                rotation = R.from_euler('xyz',
                                       np.random.rand(3) * 360,
                                       degrees=True).as_matrix()

                # Random scale
                scale = np.random.rand(3) * 0.5 + 0.5  # [0.5, 1.0]

                # Random color
                color = np.random.rand(3)

                gaussians.append(GaussianSplat(
                    position=position,
                    rotation=rotation,
                    scale=scale,
                    color=color
                ))

    # Vectorized transformation
    transform_matrix = R.from_euler('xyz', [30, 45, 60], degrees=True).as_matrix()
    transform_matrix = transform_matrix @ np.diag([1.5, 1.5, 1.5])
    translation = np.array([10.0, 10.0, 10.0])

    print(f"\nApplying transformation to all Gaussians...")

    transformed_gaussians = [
        g.transform(transform_matrix, translation)
        for g in gaussians
    ]

    # Statistics
    original_positions = np.array([g.position for g in gaussians])
    transformed_positions = np.array([g.position for g in transformed_gaussians])

    print(f"\nStatistics:")
    print(f"  Original center: {original_positions.mean(axis=0)}")
    print(f"  Transformed center: {transformed_positions.mean(axis=0)}")
    print(f"  Original extent: {original_positions.ptp(axis=0)}")
    print(f"  Transformed extent: {transformed_positions.ptp(axis=0)}")


# ============================================================================
# Example 3: Roundtrip Transformation Validation
# ============================================================================

def example_roundtrip_validation():
    """Validate that forward + inverse transformation recovers original."""
    print("\n" + "=" * 80)
    print("Example 3: Roundtrip Transformation Validation")
    print("=" * 80 + "\n")

    # Create a Gaussian with complex properties
    original = GaussianSplat(
        position=[3.5, -2.1, 7.8],
        rotation=R.from_euler('xyz', [23, 67, -15], degrees=True).as_matrix(),
        scale=[0.7, 1.3, 0.9],
        opacity=0.6,
        color=[0.8, 0.3, 0.5]
    )

    print("Original Gaussian:")
    print(f"  Position: {original.position}")
    print(f"  Scale: {original.scale}")
    print(f"  Covariance determinant: {np.linalg.det(original.covariance):.6e}")

    # Create an invertible transformation
    transform_matrix = R.from_euler('xyz', [45, 30, 60], degrees=True).as_matrix()
    transform_matrix = transform_matrix @ np.diag([2.0, 3.0, 4.0])
    translation = np.array([5.0, -3.0, 8.0])

    print(f"\nTransformation determinant: {np.linalg.det(transform_matrix):.6f}")

    # Forward transformation
    transformed = original.transform(transform_matrix, translation)

    print("\nTransformed Gaussian:")
    print(f"  Position: {transformed.position}")
    print(f"  Scale: {transformed.scale}")
    print(f"  Covariance determinant: {np.linalg.det(transformed.covariance):.6e}")

    # Inverse transformation
    transform_inv = np.linalg.inv(transform_matrix)
    translation_inv = -transform_inv @ translation

    recovered = transformed.transform(transform_inv, translation_inv)

    print("\nRecovered Gaussian:")
    print(f"  Position: {recovered.position}")
    print(f"  Scale: {recovered.scale}")
    print(f"  Covariance determinant: {np.linalg.det(recovered.covariance):.6e}")

    # Compute errors
    position_error = np.linalg.norm(recovered.position - original.position)
    scale_error = np.linalg.norm(recovered.scale - original.scale)
    covariance_error = np.linalg.norm(recovered.covariance - original.covariance)

    print("\nRoundtrip Errors:")
    print(f"  Position error: {position_error:.2e}")
    print(f"  Scale error: {scale_error:.2e}")
    print(f"  Covariance error: {covariance_error:.2e}")

    # Validate
    tolerance = 1e-9
    success = (position_error < tolerance and
               scale_error < tolerance and
               covariance_error < tolerance)

    print(f"\nValidation: {'✓ PASSED' if success else '✗ FAILED'}")
    print(f"  (tolerance = {tolerance:.2e})")


# ============================================================================
# Example 4: Numerical Stability Test
# ============================================================================

def example_numerical_stability():
    """Test numerical stability under extreme transformations."""
    print("\n" + "=" * 80)
    print("Example 4: Numerical Stability Test")
    print("=" * 80 + "\n")

    gaussian = GaussianSplat(
        position=[1.0, 1.0, 1.0],
        rotation=np.eye(3),
        scale=[1.0, 1.0, 1.0]
    )

    print("Testing numerical stability under extreme conditions...\n")

    # Test 1: Very large scale
    print("Test 1: Very large scale (1e6)")
    transform = np.diag([1e6, 1e6, 1e6])
    translation = np.zeros(3)
    transformed = gaussian.transform(transform, translation)
    print(f"  Scale: {transformed.scale}")
    print(f"  Covariance is symmetric: {np.allclose(transformed.covariance, transformed.covariance.T)}")
    print(f"  Covariance is positive definite: {all(np.linalg.eigvals(transformed.covariance) > 0)}")

    # Test 2: Very small scale
    print("\nTest 2: Very small scale (1e-6)")
    transform = np.diag([1e-6, 1e-6, 1e-6])
    transformed = gaussian.transform(transform, translation)
    print(f"  Scale: {transformed.scale}")
    print(f"  Covariance is symmetric: {np.allclose(transformed.covariance, transformed.covariance.T)}")
    print(f"  Covariance is positive definite: {all(np.linalg.eigvals(transformed.covariance) > 0)}")

    # Test 3: Extreme anisotropic scale
    print("\nTest 3: Extreme anisotropic scale [1e6, 1, 1e-6]")
    transform = np.diag([1e6, 1.0, 1e-6])
    transformed = gaussian.transform(transform, translation)
    print(f"  Scale: {transformed.scale}")
    print(f"  Condition number: {np.linalg.cond(transformed.covariance):.2e}")

    # Test 4: Many sequential transformations
    print("\nTest 4: 100 sequential rotations")
    current = gaussian
    small_rotation = R.from_euler('z', 3.6, degrees=True).as_matrix()  # 3.6° × 100 = 360°

    for i in range(100):
        current = current.transform(small_rotation, np.zeros(3))

    print(f"  Final position: {current.position}")
    print(f"  Position error: {np.linalg.norm(current.position - gaussian.position):.2e}")
    print(f"  Final rotation (should be ~identity):")
    print(f"    {R.from_matrix(current.rotation).as_euler('xyz', degrees=True)}")


# ============================================================================
# Example 5: Integration with napari Transforms
# ============================================================================

def example_napari_integration():
    """Show how to integrate with napari's transform system."""
    print("\n" + "=" * 80)
    print("Example 5: napari Integration Pattern")
    print("=" * 80 + "\n")

    print("Pseudo-code for napari integration:\n")

    code = '''
# In your Gaussian splatting layer:

class GaussianSplattingLayer:
    def __init__(self, gaussians, **kwargs):
        self.gaussians = gaussians
        self._transform = Affine()  # napari's Affine transform

    def _on_transform_change(self, event):
        """Called when layer transform changes."""
        # Get the affine transform
        affine = self._transform.affine_matrix

        # Extract components
        linear_matrix = affine[:-1, :-1]  # Rotation + scale
        translation = affine[:-1, -1]

        # Transform all Gaussians
        self._transformed_gaussians = [
            g.transform(linear_matrix, translation)
            for g in self.gaussians
        ]

        # Trigger re-render
        self.events.data()

    def _get_transformed_data(self):
        """Get Gaussians in view coordinates."""
        # Combine layer transform with view transform
        total_transform = self._view_transform @ self._transform

        # Apply to Gaussians
        return [
            g.transform(total_transform.linear_matrix,
                       total_transform.translate)
            for g in self.gaussians
        ]
'''

    print(code)

    print("\nKey Integration Points:")
    print("  1. Use napari.utils.transforms.Affine for layer transforms")
    print("  2. Extract linear_matrix and translate from affine_matrix")
    print("  3. Transform Gaussians when transform changes")
    print("  4. Cache transformed Gaussians for performance")
    print("  5. Combine multiple transforms via composition")


# ============================================================================
# Example 6: Performance Benchmark
# ============================================================================

def example_performance_benchmark():
    """Benchmark transformation performance."""
    print("\n" + "=" * 80)
    print("Example 6: Performance Benchmark")
    print("=" * 80 + "\n")

    import time

    # Test different sizes
    sizes = [100, 1000, 10000, 100000]

    print("Benchmarking Gaussian transformation performance...\n")
    print(f"{'N Gaussians':<15} {'Time (ms)':<15} {'Gaussians/sec':<20}")
    print("-" * 50)

    for n in sizes:
        # Create Gaussians
        gaussians = [
            GaussianSplat(
                position=np.random.randn(3),
                rotation=R.random().as_matrix(),
                scale=np.random.rand(3) + 0.5
            )
            for _ in range(n)
        ]

        # Create transform
        transform = R.random().as_matrix() @ np.diag([2.0, 2.0, 2.0])
        translation = np.random.randn(3)

        # Benchmark
        start = time.time()
        transformed = [g.transform(transform, translation) for g in gaussians]
        elapsed = time.time() - start

        throughput = n / elapsed

        print(f"{n:<15} {elapsed*1000:<15.2f} {throughput:<20.1f}")

    print("\nNote: Performance depends on hardware and Python environment")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  GAUSSIAN SPLATTING COORDINATE TRANSFORMATION EXAMPLES".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")

    examples = [
        ("Single Gaussian", example_single_gaussian),
        ("Batch Transformation", example_batch_gaussians),
        ("Roundtrip Validation", example_roundtrip_validation),
        ("Numerical Stability", example_numerical_stability),
        ("napari Integration", example_napari_integration),
        ("Performance Benchmark", example_performance_benchmark),
    ]

    for i, (name, func) in enumerate(examples, 1):
        try:
            func()
        except Exception as e:
            print(f"\n✗ Example {i} ({name}) failed: {e}")

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
