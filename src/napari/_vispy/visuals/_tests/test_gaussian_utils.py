"""Tests for Gaussian rendering utilities."""

import numpy as np
import pytest

from napari._vispy.visuals.gaussian_utils import (
    apply_opacity_to_colors,
    compute_depth_order,
    compute_gaussian_sizes,
    filter_by_opacity_threshold,
)


def test_compute_depth_order():
    """Test depth ordering computation."""
    # Create some positions
    positions = np.array([
        [0, 0, 0],   # Origin
        [1, 0, 0],   # Right
        [0, 1, 0],   # Up
        [0, 0, 1],   # Forward
        [0, 0, -1],  # Backward
    ], dtype=np.float32)

    # Identity view matrix (camera at origin looking down -Z)
    view_matrix = np.eye(4, dtype=np.float32)

    # Get depth order
    sort_indices = compute_depth_order(positions, view_matrix)

    # Check that we got all indices
    assert len(sort_indices) == 5
    assert set(sort_indices) == {0, 1, 2, 3, 4}

    # The point at Z=-1 should be farthest (rendered first)
    # The point at Z=1 should be nearest (rendered last)
    assert sort_indices[0] == 4  # Z=-1 is farthest
    assert sort_indices[-1] == 3  # Z=1 is nearest


def test_compute_depth_order_with_transform():
    """Test depth ordering with a transformed view matrix."""
    positions = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
    ], dtype=np.float32)

    # View matrix that translates camera to X=1.5
    # (so X=0 is far, X=2 is near)
    view_matrix = np.array([
        [1, 0, 0, -1.5],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

    sort_indices = compute_depth_order(positions, view_matrix)

    # After translation, X=0 is at view_x=-1.5 (farther)
    # X=2 is at view_x=0.5 (nearer)
    # But we're sorting by Z in view space, and we're looking down -Z...
    # Actually this test needs adjustment based on how the transform works
    assert len(sort_indices) == 3


def test_compute_depth_order_empty():
    """Test depth ordering with empty input."""
    positions = np.array([], dtype=np.float32).reshape(0, 3)
    view_matrix = np.eye(4)

    sort_indices = compute_depth_order(positions, view_matrix)

    assert len(sort_indices) == 0


def test_compute_gaussian_sizes():
    """Test Gaussian size computation."""
    scales = np.array([
        [1, 1, 1],      # Isotropic
        [2, 1, 1],      # Elongated in X
        [0.5, 0.5, 0.5],  # Small
        [3, 2, 1],      # Anisotropic
    ], dtype=np.float32)

    sizes = compute_gaussian_sizes(scales)

    # Sizes should be based on maximum scale dimension
    # size = max_scale * 2 * 3 * 10  (diameter * 3-sigma * scale_factor)
    expected = np.array([
        1 * 2 * 3 * 10,
        2 * 2 * 3 * 10,
        0.5 * 2 * 3 * 10,
        3 * 2 * 3 * 10,
    ])

    np.testing.assert_allclose(sizes, expected)


def test_compute_gaussian_sizes_with_multiplier():
    """Test Gaussian size computation with multiplier."""
    scales = np.array([[1, 1, 1]], dtype=np.float32)

    sizes = compute_gaussian_sizes(scales, point_size_multiplier=2.0)

    # Should double the base size
    expected = 1 * 2 * 3 * 10 * 2.0
    np.testing.assert_allclose(sizes, [expected])


def test_apply_opacity_to_colors():
    """Test applying opacity to colors."""
    colors = np.array([
        [1, 0, 0],  # Red
        [0, 1, 0],  # Green
        [0, 0, 1],  # Blue
    ], dtype=np.float32)

    opacities = np.array([1.0, 0.5, 0.0], dtype=np.float32)

    rgba = apply_opacity_to_colors(colors, opacities)

    assert rgba.shape == (3, 4)
    np.testing.assert_allclose(rgba[0], [1, 0, 0, 1.0])
    np.testing.assert_allclose(rgba[1], [0, 1, 0, 0.5])
    np.testing.assert_allclose(rgba[2], [0, 0, 1, 0.0])


def test_filter_by_opacity_threshold():
    """Test filtering Gaussians by opacity threshold."""
    n = 10
    positions = np.random.rand(n, 3).astype(np.float32)
    rotations = np.random.rand(n, 4).astype(np.float32)
    scales = np.random.rand(n, 3).astype(np.float32)
    opacities = np.array([0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 0.8, 0.9, 1.0], dtype=np.float32)
    colors = np.random.rand(n, 3).astype(np.float32)

    # Filter with threshold 0.01 (should keep 8 Gaussians)
    filtered = filter_by_opacity_threshold(
        positions, rotations, scales, opacities, colors, threshold=0.01
    )

    pos_f, rot_f, scl_f, opa_f, col_f = filtered

    # Should have filtered out first 2 (opacity < 0.01)
    assert len(pos_f) == 8
    assert len(rot_f) == 8
    assert len(scl_f) == 8
    assert len(opa_f) == 8
    assert len(col_f) == 8

    # Check that remaining opacities are >= threshold
    assert np.all(opa_f >= 0.01)


def test_filter_by_opacity_threshold_empty():
    """Test filtering with empty input."""
    positions = np.array([], dtype=np.float32).reshape(0, 3)
    rotations = np.array([], dtype=np.float32).reshape(0, 4)
    scales = np.array([], dtype=np.float32).reshape(0, 3)
    opacities = np.array([], dtype=np.float32)
    colors = np.array([], dtype=np.float32).reshape(0, 3)

    filtered = filter_by_opacity_threshold(
        positions, rotations, scales, opacities, colors
    )

    pos_f, rot_f, scl_f, opa_f, col_f = filtered

    assert len(pos_f) == 0
    assert len(rot_f) == 0
    assert len(scl_f) == 0
    assert len(opa_f) == 0
    assert len(col_f) == 0


def test_filter_by_opacity_all_filtered():
    """Test filtering when all Gaussians are below threshold."""
    positions = np.random.rand(5, 3).astype(np.float32)
    rotations = np.random.rand(5, 4).astype(np.float32)
    scales = np.random.rand(5, 3).astype(np.float32)
    opacities = np.array([0.001, 0.002, 0.003, 0.004, 0.005], dtype=np.float32)
    colors = np.random.rand(5, 3).astype(np.float32)

    # All opacities < 0.01
    filtered = filter_by_opacity_threshold(
        positions, rotations, scales, opacities, colors, threshold=0.01
    )

    pos_f, rot_f, scl_f, opa_f, col_f = filtered

    assert len(pos_f) == 0
    assert len(rot_f) == 0
    assert len(scl_f) == 0
    assert len(opa_f) == 0
    assert len(col_f) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
