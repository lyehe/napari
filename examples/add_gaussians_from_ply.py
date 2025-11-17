"""
Example of loading Gaussian splat data from a PLY file.

This demonstrates loading real 3D Gaussian Splatting data (e.g., from a trained model)
and visualizing it in napari.

Usage:
    python add_gaussians_from_ply.py path/to/point_cloud.ply
"""

import sys
import napari

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python add_gaussians_from_ply.py path/to/point_cloud.ply")
        print("\nCreating a synthetic example instead...")

        # Create synthetic data for demonstration
        import numpy as np
        from scipy.spatial.transform import Rotation
        from napari.layers.gaussians._gaussians_utils import GaussianData, write_ply_gaussians

        # Create a simple scene with a few oriented Gaussians
        positions = np.array([
            [50, 50, 50],  # Center
            [30, 50, 50],  # Left
            [70, 50, 50],  # Right
            [50, 30, 50],  # Front
            [50, 70, 50],  # Back
        ], dtype=np.float32)

        # Create oriented rotations
        rotations = np.array([
            [0, 0, 0, 1],  # Identity
            [0, 0, 0, 1],  # Identity
            [0, 0, 0, 1],  # Identity
            Rotation.from_euler('z', 45, degrees=True).as_quat(),  # Rotated
            Rotation.from_euler('x', 45, degrees=True).as_quat(),  # Rotated
        ], dtype=np.float32)

        # Anisotropic scales
        scales = np.array([
            [5, 5, 5],    # Isotropic
            [10, 2, 2],   # Elongated along Z
            [2, 10, 2],   # Elongated along Y
            [2, 2, 10],   # Elongated along X
            [8, 3, 3],    # Moderate anisotropy
        ], dtype=np.float32)

        # Different colors
        colors = np.array([
            [1, 0, 0],    # Red
            [0, 1, 0],    # Green
            [0, 0, 1],    # Blue
            [1, 1, 0],    # Yellow
            [1, 0, 1],    # Magenta
        ], dtype=np.float32)

        opacities = np.array([0.9, 0.8, 0.8, 0.7, 0.7], dtype=np.float32)

        # Save to PLY file
        gaussian_data = GaussianData(
            positions=positions,
            rotations=rotations,
            scales=scales,
            opacities=opacities,
            colors=colors,
        )

        ply_path = '/tmp/example_gaussians.ply'
        write_ply_gaussians(ply_path, gaussian_data)
        print(f"Created example PLY file: {ply_path}")

        ply_file = ply_path
    else:
        ply_file = sys.argv[1]

    # Create viewer
    viewer = napari.Viewer(ndisplay=3)

    # Load Gaussians from PLY file
    print(f"Loading Gaussians from {ply_file}...")
    gaussians_layer = viewer.add_layer(
        napari.layers.Gaussians(
            data=ply_file,
            name='Gaussian Splats',
            point_size=1.0,
            blending='translucent',
        )
    )

    print(f"Loaded {len(gaussians_layer.data)} Gaussian splats")
    print(f"Data shape: {gaussians_layer.data.shape}")
    print(f"Rotations shape: {gaussians_layer.rotations.shape}")
    print(f"Scales shape: {gaussians_layer.scales.shape}")
    print(f"Opacities shape: {gaussians_layer.opacities.shape}")
    print(f"Colors shape: {gaussians_layer.colors.shape}")

    # Set camera
    viewer.camera.angles = (45, 45, 0)
    viewer.camera.zoom = 1.5
    viewer.camera.center = (50, 50, 50)

    napari.run()
