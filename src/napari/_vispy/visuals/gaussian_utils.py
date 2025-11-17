"""Utilities for Gaussian Splatting visual rendering."""

from __future__ import annotations

import numpy as np


def compute_depth_order(
    positions: np.ndarray,
    view_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute depth-based ordering for Gaussians.

    For correct alpha blending, Gaussians should be rendered back-to-front.
    This function computes the depth of each Gaussian in view space and
    returns indices sorted by depth (farthest first).

    Parameters
    ----------
    positions : ndarray (N, 3)
        Gaussian positions in world space (XYZ)
    view_matrix : ndarray (4, 4)
        View matrix transforming world space to camera space

    Returns
    -------
    sort_indices : ndarray (N,)
        Indices that sort Gaussians from back to front
    """
    if len(positions) == 0:
        return np.array([], dtype=int)

    # Transform positions to view space
    positions_homogeneous = np.column_stack([
        positions,
        np.ones(len(positions))
    ])
    positions_view = positions_homogeneous @ view_matrix.T

    # Get Z depth (in view space, -Z is forward)
    depths = positions_view[:, 2]

    # Sort by depth (farthest first, i.e., smallest Z first for back-to-front)
    sort_indices = np.argsort(depths)

    return sort_indices


def compute_gaussian_sizes(
    scales: np.ndarray,
    point_size_multiplier: float = 1.0,
    size_scale: float = 10.0,
) -> np.ndarray:
    """
    Compute visual sizes for Gaussian markers.

    Parameters
    ----------
    scales : ndarray (N, 3)
        Gaussian scale parameters
    point_size_multiplier : float
        Global size multiplier from layer settings
    size_scale : float
        Base scaling factor for visibility

    Returns
    -------
    sizes : ndarray (N,)
        Visual size for each Gaussian
    """
    # Use maximum scale dimension (most visible extent)
    # Multiply by 2 for diameter, by 3 for 3-sigma, and by scale factor
    sizes = np.max(scales, axis=1) * 2.0 * 3.0 * size_scale * point_size_multiplier
    return sizes


def apply_opacity_to_colors(
    colors: np.ndarray,
    opacities: np.ndarray,
) -> np.ndarray:
    """
    Apply opacity values to RGB colors for rendering.

    Parameters
    ----------
    colors : ndarray (N, 3)
        RGB colors [0, 1]
    opacities : ndarray (N,)
        Opacity values [0, 1]

    Returns
    -------
    rgba_colors : ndarray (N, 4)
        RGBA colors with opacity applied
    """
    rgba_colors = np.column_stack([colors, opacities])
    return rgba_colors


def filter_by_opacity_threshold(
    positions: np.ndarray,
    rotations: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    colors: np.ndarray,
    threshold: float = 0.01,
) -> tuple[np.ndarray, ...]:
    """
    Filter out Gaussians below opacity threshold for performance.

    Very transparent Gaussians contribute little to the final image
    and can be culled for performance.

    Parameters
    ----------
    positions : ndarray (N, 3)
    rotations : ndarray (N, 4)
    scales : ndarray (N, 3)
    opacities : ndarray (N,)
    colors : ndarray (N, 3)
    threshold : float
        Minimum opacity to render (default 0.01 = 1%)

    Returns
    -------
    filtered_data : tuple
        (positions, rotations, scales, opacities, colors) with low-opacity removed
    """
    if len(opacities) == 0:
        return positions, rotations, scales, opacities, colors

    mask = opacities >= threshold

    return (
        positions[mask],
        rotations[mask],
        scales[mask],
        opacities[mask],
        colors[mask],
    )
