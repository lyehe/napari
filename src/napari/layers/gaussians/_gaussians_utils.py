"""Utilities for Gaussian Splatting layer including PLY I/O and data structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class GaussianData:
    """
    Data structure for 3D Gaussian Splats.

    Attributes
    ----------
    positions : NDArray
        (N, 3) array of Gaussian centers in ZYX order (napari convention)
    rotations : NDArray
        (N, 4) array of rotation quaternions in XYZW format (scipy convention)
    scales : NDArray
        (N, 3) array of scale parameters (axis-aligned standard deviations)
    opacities : NDArray
        (N,) array of opacity values [0, 1]
    colors : NDArray, optional
        (N, 3) array of RGB colors [0, 1] for SH degree 0, or
        (N, K, 3) array of spherical harmonics coefficients where K depends on degree
    features : NDArray, optional
        (N, D) array of additional feature dimensions
    """

    positions: NDArray[np.float32]
    rotations: NDArray[np.float32]
    scales: NDArray[np.float32]
    opacities: NDArray[np.float32]
    colors: NDArray[np.float32] | None = None
    features: NDArray[np.float32] | None = None

    def __post_init__(self):
        """Validate data shapes and types."""
        n = len(self.positions)

        if self.positions.shape != (n, 3):
            raise ValueError(f'positions must be (N, 3), got {self.positions.shape}')

        if self.rotations.shape != (n, 4):
            raise ValueError(f'rotations must be (N, 4), got {self.rotations.shape}')

        if self.scales.shape != (n, 3):
            raise ValueError(f'scales must be (N, 3), got {self.scales.shape}')

        if self.opacities.shape != (n,):
            raise ValueError(f'opacities must be (N,), got {self.opacities.shape}')

        # Normalize quaternions
        norms = np.linalg.norm(self.rotations, axis=1, keepdims=True)
        self.rotations = self.rotations / norms


def read_ply_gaussians(path: str | Path) -> GaussianData:
    """
    Read Gaussian splat data from PLY file.

    PLY files from 3D Gaussian Splatting typically use:
    - XYZ coordinate ordering (not ZYX)
    - WXYZ quaternion format (scalar-first, not scipy's XYZW)

    This function converts to napari conventions:
    - ZYX coordinate ordering
    - XYZW quaternion format (scipy convention)

    Parameters
    ----------
    path : str or Path
        Path to PLY file

    Returns
    -------
    GaussianData
        Loaded Gaussian splat data in napari conventions
    """
    from plyfile import PlyData

    plydata = PlyData.read(str(path))
    vertex = plydata['vertex']
    n = len(vertex)

    # Read positions (XYZ in file -> ZYX for napari)
    x = np.asarray(vertex['x'], dtype=np.float32)
    y = np.asarray(vertex['y'], dtype=np.float32)
    z = np.asarray(vertex['z'], dtype=np.float32)
    positions = np.stack([z, y, x], axis=1)  # Convert XYZ -> ZYX

    # Read rotations (WXYZ in file -> XYZW for scipy)
    # PLY format uses WXYZ (scalar-first) quaternion format
    rot_0 = np.asarray(vertex['rot_0'], dtype=np.float32)  # W component
    rot_1 = np.asarray(vertex['rot_1'], dtype=np.float32)  # X component
    rot_2 = np.asarray(vertex['rot_2'], dtype=np.float32)  # Y component
    rot_3 = np.asarray(vertex['rot_3'], dtype=np.float32)  # Z component

    # Convert WXYZ -> XYZW (scipy convention)
    rotations_wxyz = np.stack([rot_0, rot_1, rot_2, rot_3], axis=1)
    rotations_xyzw = np.stack([rot_1, rot_2, rot_3, rot_0], axis=1)

    # Transform rotation from XYZ space to ZYX space via rotation matrix
    # This preserves the actual rotation while accounting for coordinate swap
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
    R_matrices_xyz = Rotation.from_quat(rotations_xyzw).as_matrix()
    R_matrices_zyx = P @ R_matrices_xyz @ P.T
    rotations = Rotation.from_matrix(R_matrices_zyx).as_quat()  # Back to XYZW

    # Read scales (logarithmic in file)
    scale_0 = np.asarray(vertex['scale_0'], dtype=np.float32)
    scale_1 = np.asarray(vertex['scale_1'], dtype=np.float32)
    scale_2 = np.asarray(vertex['scale_2'], dtype=np.float32)

    # Exponentiate and swap from XYZ -> ZYX ordering
    scales = np.exp(np.stack([scale_2, scale_1, scale_0], axis=1))

    # Read opacity (logit in file)
    opacity_logit = np.asarray(vertex['opacity'], dtype=np.float32)
    opacities = 1.0 / (1.0 + np.exp(-opacity_logit))  # Sigmoid

    # Read spherical harmonics (SH) coefficients
    # DC component (degree 0) - 3 values for RGB
    f_dc_0 = np.asarray(vertex['f_dc_0'], dtype=np.float32)
    f_dc_1 = np.asarray(vertex['f_dc_1'], dtype=np.float32)
    f_dc_2 = np.asarray(vertex['f_dc_2'], dtype=np.float32)
    colors_sh0 = np.stack([f_dc_0, f_dc_1, f_dc_2], axis=1)

    # Check for higher degree SH coefficients
    sh_degree = 0
    sh_coeffs_list = [colors_sh0]

    # Try to load SH degree 1 (9 coefficients: 3 RGB × 3 basis functions)
    try:
        sh_rest_names = [f'f_rest_{i}' for i in range(9)]
        sh_rest = np.stack([np.asarray(vertex[name], dtype=np.float32)
                           for name in sh_rest_names], axis=1)
        sh_rest = sh_rest.reshape(n, 3, 3)  # (N, 3_basis, 3_RGB)
        sh_coeffs_list.append(sh_rest)
        sh_degree = 1

        # Try degree 2 (25 = 9 + 15 coefficients)
        try:
            sh_rest2_names = [f'f_rest_{i}' for i in range(9, 24)]
            sh_rest2 = np.stack([np.asarray(vertex[name], dtype=np.float32)
                                for name in sh_rest2_names], axis=1)
            sh_rest2 = sh_rest2.reshape(n, 5, 3)  # (N, 5_basis, 3_RGB)
            sh_coeffs_list.append(sh_rest2)
            sh_degree = 2

            # Try degree 3 (49 = 25 + 21 coefficients)
            try:
                sh_rest3_names = [f'f_rest_{i}' for i in range(24, 45)]
                sh_rest3 = np.stack([np.asarray(vertex[name], dtype=np.float32)
                                    for name in sh_rest3_names], axis=1)
                sh_rest3 = sh_rest3.reshape(n, 7, 3)  # (N, 7_basis, 3_RGB)
                sh_coeffs_list.append(sh_rest3)
                sh_degree = 3
            except (KeyError, ValueError):
                pass
        except (KeyError, ValueError):
            pass
    except (KeyError, ValueError):
        pass

    # Concatenate all SH coefficients
    # Shape: (N, num_coeffs, 3) where num_coeffs = (sh_degree+1)^2
    if sh_degree > 0:
        colors = np.concatenate([
            colors_sh0.reshape(n, 1, 3),  # Degree 0: 1 coeff
            *[sh_coeffs_list[i] for i in range(1, len(sh_coeffs_list))]
        ], axis=1)
    else:
        # Only DC component - convert to simple RGB colors
        colors = colors_sh0

    return GaussianData(
        positions=positions,
        rotations=rotations,
        scales=scales,
        opacities=opacities,
        colors=colors,
    )


def write_ply_gaussians(path: str | Path, data: GaussianData) -> None:
    """
    Write Gaussian splat data to PLY file.

    Converts from napari conventions to PLY format:
    - ZYX -> XYZ coordinate ordering
    - XYZW -> WXYZ quaternion format

    Parameters
    ----------
    path : str or Path
        Output PLY file path
    data : GaussianData
        Gaussian splat data in napari conventions
    """
    from plyfile import PlyData, PlyElement

    n = len(data.positions)

    # Convert positions ZYX -> XYZ
    z, y, x = data.positions[:, 0], data.positions[:, 1], data.positions[:, 2]

    # Transform rotations from ZYX space back to XYZ space
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
    R_matrices_zyx = Rotation.from_quat(data.rotations).as_matrix()
    R_matrices_xyz = P.T @ R_matrices_zyx @ P
    rotations_xyzw = Rotation.from_matrix(R_matrices_xyz).as_quat()

    # Convert XYZW -> WXYZ for PLY
    rot_1, rot_2, rot_3, rot_0 = (
        rotations_xyzw[:, 0],
        rotations_xyzw[:, 1],
        rotations_xyzw[:, 2],
        rotations_xyzw[:, 3],
    )

    # Convert scales ZYX -> XYZ and apply logarithm
    scales_xyz = data.scales[:, ::-1]  # ZYX -> XYZ
    scale_0, scale_1, scale_2 = (
        np.log(scales_xyz[:, 0]),
        np.log(scales_xyz[:, 1]),
        np.log(scales_xyz[:, 2]),
    )

    # Convert opacity to logit
    opacity = np.clip(data.opacities, 1e-7, 1 - 1e-7)
    opacity_logit = np.log(opacity / (1 - opacity))

    # Prepare vertex data
    vertex_data = [
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'),
        ('scale_1', 'f4'),
        ('scale_2', 'f4'),
        ('rot_0', 'f4'),
        ('rot_1', 'f4'),
        ('rot_2', 'f4'),
        ('rot_3', 'f4'),
    ]

    vertex_values = [
        x, y, z,
        opacity_logit,
        scale_0, scale_1, scale_2,
        rot_0, rot_1, rot_2, rot_3,
    ]

    # Handle colors/SH coefficients
    if data.colors is not None:
        if data.colors.ndim == 2:
            # Simple RGB colors (SH degree 0)
            vertex_data.extend([
                ('f_dc_0', 'f4'),
                ('f_dc_1', 'f4'),
                ('f_dc_2', 'f4'),
            ])
            vertex_values.extend([
                data.colors[:, 0],
                data.colors[:, 1],
                data.colors[:, 2],
            ])
        else:
            # Spherical harmonics coefficients
            # DC component
            vertex_data.extend([
                ('f_dc_0', 'f4'),
                ('f_dc_1', 'f4'),
                ('f_dc_2', 'f4'),
            ])
            vertex_values.extend([
                data.colors[:, 0, 0],
                data.colors[:, 0, 1],
                data.colors[:, 0, 2],
            ])

            # Rest of SH coefficients
            num_sh_rest = data.colors.shape[1] - 1  # Exclude DC
            for i in range(num_sh_rest):
                for j in range(3):  # RGB
                    vertex_data.append((f'f_rest_{i*3+j}', 'f4'))
                    vertex_values.append(data.colors[:, i + 1, j])

    # Create structured array
    vertex_array = np.empty(n, dtype=vertex_data)
    for (name, _), values in zip(vertex_data, vertex_values):
        vertex_array[name] = values

    # Write PLY file
    el = PlyElement.describe(vertex_array, 'vertex')
    PlyData([el]).write(str(path))


def compute_covariance_matrices(
    rotations: NDArray[np.float32],
    scales: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Compute 3D covariance matrices from rotation quaternions and scales.

    Σ = R @ S @ S^T @ R^T

    Parameters
    ----------
    rotations : NDArray
        (N, 4) array of quaternions in XYZW format
    scales : NDArray
        (N, 3) array of scale parameters

    Returns
    -------
    NDArray
        (N, 3, 3) array of covariance matrices
    """
    n = len(rotations)
    R = Rotation.from_quat(rotations).as_matrix()  # (N, 3, 3)
    S = np.zeros((n, 3, 3), dtype=np.float32)
    S[:, 0, 0] = scales[:, 0]
    S[:, 1, 1] = scales[:, 1]
    S[:, 2, 2] = scales[:, 2]

    # Σ = R @ S @ S^T @ R^T = R @ diag(s^2) @ R^T
    S_squared = S @ S.swapaxes(-2, -1)
    covariances = R @ S_squared @ R.swapaxes(-2, -1)

    return covariances.astype(np.float32)


def compute_mahalanobis_distance(
    points: NDArray[np.float32],
    center: NDArray[np.float32],
    covariance: NDArray[np.float32],
) -> NDArray[np.float32]:
    """
    Compute Mahalanobis distance from points to a Gaussian.

    d^2 = (x - μ)^T @ Σ^(-1) @ (x - μ)

    Parameters
    ----------
    points : NDArray
        (M, 3) array of points
    center : NDArray
        (3,) Gaussian center
    covariance : NDArray
        (3, 3) covariance matrix

    Returns
    -------
    NDArray
        (M,) array of squared Mahalanobis distances
    """
    diff = points - center
    cov_inv = np.linalg.pinv(covariance)  # Use pseudo-inverse for stability
    distances_sq = np.sum(diff @ cov_inv * diff, axis=1)
    return distances_sq.astype(np.float32)
