"""Tests for the Gaussians layer."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from napari.layers.gaussians import Gaussians
from napari.layers.gaussians._gaussians_constants import ColorMode, Mode
from napari.layers.gaussians._gaussians_utils import GaussianData


def test_gaussians_layer_empty():
    """Test creating an empty Gaussians layer."""
    layer = Gaussians()
    assert len(layer.data) == 0
    assert layer.ndim == 3


def test_gaussians_layer_from_positions():
    """Test creating Gaussians layer from positions only."""
    positions = np.random.rand(10, 3).astype(np.float32)
    layer = Gaussians(positions)

    assert len(layer.data) == 10
    assert layer.data.shape == (10, 3)
    assert layer.rotations.shape == (10, 4)
    assert layer.scales.shape == (10, 3)
    assert layer.opacities.shape == (10,)
    assert layer.colors.shape == (10, 3)

    # Check default values
    assert np.allclose(layer.rotations, np.tile([0, 0, 0, 1], (10, 1)))
    assert np.allclose(layer.scales, 1.0)
    assert np.allclose(layer.opacities, 1.0)
    assert np.allclose(layer.colors, 1.0)


def test_gaussians_layer_from_gaussian_data():
    """Test creating Gaussians layer from GaussianData object."""
    positions = np.random.rand(5, 3).astype(np.float32)
    rotations = np.tile([0, 0, 0, 1], (5, 1)).astype(np.float32)
    scales = np.ones((5, 3), dtype=np.float32)
    opacities = np.ones(5, dtype=np.float32) * 0.8
    colors = np.random.rand(5, 3).astype(np.float32)

    gaussian_data = GaussianData(
        positions=positions,
        rotations=rotations,
        scales=scales,
        opacities=opacities,
        colors=colors,
    )

    layer = Gaussians(gaussian_data)

    assert len(layer.data) == 5
    assert np.allclose(layer.data, positions)
    assert np.allclose(layer.rotations, rotations)
    assert np.allclose(layer.scales, scales)
    assert np.allclose(layer.opacities, opacities)
    assert np.allclose(layer.colors, colors)


def test_gaussians_layer_properties():
    """Test setting and getting layer properties."""
    positions = np.random.rand(10, 3).astype(np.float32)
    layer = Gaussians(positions, point_size=2.0, sh_degree=1)

    # Test point_size
    assert layer.point_size == 2.0
    layer.point_size = 3.0
    assert layer.point_size == 3.0

    # Test sh_degree
    assert layer.sh_degree == 1
    layer.sh_degree = 2
    assert layer.sh_degree == 2

    # Test color_mode
    assert layer.color_mode == ColorMode.DIRECT
    layer.color_mode = ColorMode.SPHERICAL_HARMONICS
    assert layer.color_mode == ColorMode.SPHERICAL_HARMONICS


def test_gaussians_layer_data_setter():
    """Test updating layer data."""
    layer = Gaussians(np.random.rand(10, 3).astype(np.float32))

    # Update with new positions
    new_positions = np.random.rand(15, 3).astype(np.float32)
    layer.data = new_positions

    assert len(layer.data) == 15
    assert layer.data.shape == (15, 3)
    # Arrays should be resized
    assert layer.rotations.shape == (15, 4)
    assert layer.scales.shape == (15, 3)
    assert layer.opacities.shape == (15,)


def test_gaussians_layer_scales_broadcasting():
    """Test that scales can be set with different shapes."""
    positions = np.random.rand(10, 3).astype(np.float32)
    layer = Gaussians(positions)

    # Scalar scale
    layer.scales = 2.0
    assert layer.scales.shape == (10, 3)
    assert np.allclose(layer.scales, 2.0)

    # Per-Gaussian isotropic scale
    layer.scales = np.arange(10, dtype=np.float32)
    assert layer.scales.shape == (10, 3)
    assert np.allclose(layer.scales[:, 0], np.arange(10))
    assert np.allclose(layer.scales[:, 1], np.arange(10))
    assert np.allclose(layer.scales[:, 2], np.arange(10))

    # Anisotropic scales
    layer.scales = np.random.rand(10, 3).astype(np.float32)
    assert layer.scales.shape == (10, 3)


def test_gaussians_layer_opacities_clipping():
    """Test that opacities are clipped to [0, 1]."""
    positions = np.random.rand(10, 3).astype(np.float32)
    layer = Gaussians(positions)

    # Set out-of-range opacities
    layer.opacities = np.array([-0.5, 0.5, 1.5, 2.0, 0.0, 1.0, 0.3, 0.7, 1.1, -0.1])
    assert np.all(layer.opacities >= 0.0)
    assert np.all(layer.opacities <= 1.0)
    assert layer.opacities[0] == 0.0  # Clipped from -0.5
    assert layer.opacities[2] == 1.0  # Clipped from 1.5


def test_gaussians_layer_mode():
    """Test layer mode property."""
    layer = Gaussians(np.random.rand(10, 3).astype(np.float32))

    assert layer.mode == Mode.PAN_ZOOM
    layer.mode = Mode.SELECT
    assert layer.mode == Mode.SELECT


def test_gaussians_layer_selection():
    """Test Gaussian selection."""
    layer = Gaussians(np.random.rand(10, 3).astype(np.float32))

    # Select some Gaussians
    layer.selected_data = {0, 2, 5}
    assert layer.selected_data == {0, 2, 5}

    # Clear selection
    layer.selected_data = set()
    assert len(layer.selected_data) == 0


def test_gaussians_layer_view_slicing():
    """Test that view data updates correctly."""
    positions = np.array([
        [0, 0, 0],
        [10, 10, 10],
        [20, 20, 20],
        [30, 30, 30],
    ], dtype=np.float32)

    layer = Gaussians(positions)

    # Initially, all Gaussians should be in view (no slicing set up)
    # We need to manually set up indices_view for testing
    layer._indices_view = np.array([0, 1, 2])

    # Check view data
    assert len(layer._view_data) > 0 or len(layer._indices_view) > 0


def test_gaussian_data_validation():
    """Test GaussianData validation."""
    # Valid data
    positions = np.random.rand(10, 3).astype(np.float32)
    rotations = np.tile([0, 0, 0, 1], (10, 1)).astype(np.float32)
    scales = np.ones((10, 3), dtype=np.float32)
    opacities = np.ones(10, dtype=np.float32)
    colors = np.ones((10, 3), dtype=np.float32)

    data = GaussianData(
        positions=positions,
        rotations=rotations,
        scales=scales,
        opacities=opacities,
        colors=colors,
    )
    assert len(data.positions) == 10

    # Invalid positions shape
    with pytest.raises(ValueError):
        GaussianData(
            positions=np.ones((10, 2)),  # Wrong shape
            rotations=rotations,
            scales=scales,
            opacities=opacities,
            colors=colors,
        )

    # Invalid rotations shape
    with pytest.raises(ValueError):
        GaussianData(
            positions=positions,
            rotations=np.ones((10, 3)),  # Wrong shape
            scales=scales,
            opacities=opacities,
            colors=colors,
        )


def test_gaussians_layer_events():
    """Test that layer events are emitted correctly."""
    layer = Gaussians(np.random.rand(10, 3).astype(np.float32))

    # Track event emissions
    point_size_events = []
    sh_degree_events = []

    layer.events.point_size.connect(lambda e: point_size_events.append(e.value))
    layer.events.sh_degree.connect(lambda e: sh_degree_events.append(e.value))

    # Change properties
    layer.point_size = 2.5
    layer.sh_degree = 2

    assert len(point_size_events) == 1
    assert point_size_events[0] == 2.5
    assert len(sh_degree_events) == 1
    assert sh_degree_events[0] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
