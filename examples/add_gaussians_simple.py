"""
Simple example of creating and visualizing a Gaussians layer in napari.

This demonstrates basic usage of the new Gaussian Splatting layer.
"""

import numpy as np
import napari

# Create some sample Gaussian splat data
n_gaussians = 100

# Random positions in 3D space (ZYX order)
positions = np.random.rand(n_gaussians, 3) * 100

# Random rotations (identity quaternions as default)
rotations = np.zeros((n_gaussians, 4))
rotations[:, 3] = 1.0  # W component = 1 for identity
# Add some random rotations
random_angles = np.random.rand(n_gaussians, 3) * 2 * np.pi
from scipy.spatial.transform import Rotation
rotations = Rotation.from_euler('zyx', random_angles).as_quat()

# Random scales (anisotropic)
scales = np.random.rand(n_gaussians, 3) * 3 + 0.5

# Random opacities
opacities = np.random.rand(n_gaussians) * 0.5 + 0.5

# Random colors
colors = np.random.rand(n_gaussians, 3)

# Create viewer and add Gaussians layer
viewer = napari.Viewer(ndisplay=3)

# Add Gaussians layer
gaussians_layer = viewer.add_layer(
    napari.layers.Gaussians(
        data=positions,
        rotations=rotations,
        scales=scales,
        opacities=opacities,
        colors=colors,
        name='Random Gaussians',
        point_size=1.0,
        blending='translucent',
    )
)

print(f"Created Gaussians layer with {len(positions)} splats")
print(f"Layer properties:")
print(f"  - Point size: {gaussians_layer.point_size}")
print(f"  - SH degree: {gaussians_layer.sh_degree}")
print(f"  - Color mode: {gaussians_layer.color_mode}")

# Set camera to see the data
viewer.camera.angles = (45, 45, 45)
viewer.camera.zoom = 2.0

napari.run()
