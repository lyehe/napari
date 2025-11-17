"""Gaussians layer for 3D Gaussian Splatting visualization."""

from __future__ import annotations

import typing
import warnings
from collections.abc import Callable, Set as AbstractSet
from copy import copy
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Optional,
)

import numpy as np
import numpy.typing as npt
from psygnal.containers import Selection

from napari.layers.base import Layer, no_op
from napari.layers.base._base_constants import ActionType
from napari.layers.base._base_mouse_bindings import (
    highlight_box_handles,
    transform_with_box,
)
from napari.layers.gaussians._gaussians_constants import (
    ColorMode,
    GaussiansProjectionMode,
    Mode,
)
from napari.layers.gaussians._gaussians_utils import (
    GaussianData,
    read_ply_gaussians,
    write_ply_gaussians,
)
from napari.layers.gaussians._slice import _GaussianSliceRequest, _GaussianSliceResponse
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice
from napari.layers.utils.color_manager import ColorManager
from napari.layers.utils.color_transformations import ColorType
from napari.layers.utils.layer_utils import (
    _features_to_properties,
    _FeatureTable,
)
from napari.utils.colormaps import Colormap, ValidColormapArg
from napari.utils.events import Event
from napari.utils.events.custom_types import Array
from napari.utils.translations import trans

if TYPE_CHECKING:
    from napari.components.dims import Dims

DEFAULT_COLOR_CYCLE = np.array([[1, 0, 1, 1], [0, 1, 0, 1]])


class Gaussians(Layer):
    """
    Gaussians layer for 3D Gaussian Splatting.

    Parameters
    ----------
    data : array (N, 3), GaussianData, or str/Path
        Gaussian splat data. Can be:
        - (N, 3) array of positions (ZYX order)
        - GaussianData object with full splat data
        - Path to PLY file to load
    rotations : array (N, 4), optional
        Rotation quaternions in XYZW format (scipy convention).
        Defaults to identity rotations.
    scales : array (N, 3) or float, optional
        Scale parameters for each Gaussian. Can be scalar for isotropic.
        Defaults to 1.0.
    opacities : array (N,) or float, optional
        Opacity values [0, 1] for each Gaussian. Defaults to 1.0.
    colors : array (N, 3) or (N, K, 3), optional
        RGB colors [0, 1] or spherical harmonics coefficients.
        For SH, K = (degree+1)^2. Defaults to white.
    features : dict or DataFrame, optional
        Feature table where each row corresponds to a Gaussian.
    point_size : float, optional
        Base size multiplier for rendering. Defaults to 1.0.
    sh_degree : int, optional
        Spherical harmonics degree to use (0-3). Defaults to 0 (DC only).
    color_mode : str or ColorMode, optional
        Color mode: 'direct', 'cycle', 'colormap', or 'spherical_harmonics'.
        Defaults to 'direct'.
    ndim : int, optional
        Number of dimensions. Defaults to 3 for Gaussians.
    name : str, optional
        Layer name.
    metadata : dict, optional
        Layer metadata.
    scale : tuple of float, optional
        Scale factors for the layer.
    translate : tuple of float, optional
        Translation values for the layer.
    rotate : float, 3-tuple, or ndarray, optional
        Rotation for the layer.
    shear : array, optional
        Shear matrix for the layer.
    affine : array or Affine, optional
        Affine transformation matrix.
    opacity : float, optional
        Layer opacity [0, 1]. Defaults to 1.0.
    blending : str, optional
        Blending mode. Defaults to 'translucent'.
    visible : bool, optional
        Whether the layer is visible. Defaults to True.
    cache : bool, optional
        Whether to cache data. Defaults to True.
    projection_mode : str, optional
        Projection mode for thick slices: 'none' or 'all'. Defaults to 'all'.
    out_of_slice_display : bool, optional
        Whether to display Gaussians extending into the slice. Defaults to True.

    Attributes
    ----------
    data : array (N, 3)
        Gaussian positions in ZYX coordinates.
    rotations : array (N, 4)
        Rotation quaternions in XYZW format.
    scales : array (N, 3)
        Scale parameters for each Gaussian.
    opacities : array (N,)
        Opacity values [0, 1].
    colors : array
        Color or SH coefficient data.
    point_size : float
        Base size multiplier for rendering.
    sh_degree : int
        Active spherical harmonics degree.
    color_mode : ColorMode
        Active color mode.
    selected_data : Selection
        Indices of selected Gaussians.
    mode : Mode
        Current interaction mode.
    """

    _modeclass = Mode
    _projectionclass = GaussiansProjectionMode

    _drag_modes: ClassVar[dict[Mode, Callable[['Gaussians', Event], Any]]] = {
        Mode.PAN_ZOOM: no_op,
        Mode.TRANSFORM: transform_with_box,
        Mode.SELECT: no_op,  # TODO: implement selection
    }

    _move_modes: ClassVar[dict[Mode, Callable[['Gaussians', Event], Any]]] = {
        Mode.PAN_ZOOM: no_op,
        Mode.TRANSFORM: highlight_box_handles,
        Mode.SELECT: no_op,  # TODO: implement highlight
    }

    _cursor_modes: ClassVar[dict[Mode, str]] = {
        Mode.PAN_ZOOM: 'standard',
        Mode.TRANSFORM: 'standard',
        Mode.SELECT: 'crosshair',
    }

    def __init__(
        self,
        data=None,
        *,
        rotations=None,
        scales=None,
        opacities=None,
        colors=None,
        features=None,
        feature_defaults=None,
        properties=None,
        property_choices=None,
        point_size=1.0,
        sh_degree=0,
        color_mode='direct',
        colormap='viridis',
        contrast_limits=None,
        color_cycle=None,
        ndim=3,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=1.0,
        blending='translucent',
        visible=True,
        cache=True,
        projection_mode='all',
        out_of_slice_display=True,
        experimental_clipping_planes=None,
    ) -> None:
        # Handle different input types
        gaussian_data = self._parse_data_input(data, rotations, scales, opacities, colors)

        # Initialize selection state
        self._selected_data_stored = set()
        self._selected_data_history = set()
        self._selected_data: Selection[int] = Selection()
        self._selected_view = []
        self._value = None
        self._value_stored = None
        self._highlight_index = []
        self._highlight_box = None
        self._mode = Mode.PAN_ZOOM
        self._status = self.mode

        # Initialize view data
        self.__indices_view = np.empty(0, int)
        self._view_size_scale = []

        # Drag and interaction state
        self._drag_start: Optional[np.ndarray] = None
        self._drag_box: Optional[np.ndarray] = None
        self._drag_box_stored: Optional[np.ndarray] = None
        self._is_selecting = False
        self._clipboard = {}

        # Initialize base layer
        super().__init__(
            gaussian_data.positions,
            ndim,
            affine=affine,
            blending=blending,
            cache=cache,
            experimental_clipping_planes=experimental_clipping_planes,
            metadata=metadata,
            name=name,
            opacity=opacity,
            projection_mode=projection_mode,
            rotate=rotate,
            scale=scale,
            shear=shear,
            translate=translate,
            visible=visible,
        )

        # Add custom events
        self.events.add(
            point_size=Event,
            sh_degree=Event,
            color_mode=Event,
            rotations=Event,
            scales=Event,
            opacities=Event,
            colors=Event,
            out_of_slice_display=Event,
            features=Event,
            feature_defaults=Event,
            properties=Event,
        )

        # Store Gaussian-specific data
        self._data = gaussian_data.positions
        self._rotations = gaussian_data.rotations
        self._scales = gaussian_data.scales
        self._opacities = gaussian_data.opacities
        self._colors = gaussian_data.colors if gaussian_data.colors is not None else np.ones((len(gaussian_data.positions), 3), dtype=np.float32)

        # Feature table
        self._feature_table = _FeatureTable.from_layer(
            features=features,
            feature_defaults=feature_defaults,
            properties=properties,
            property_choices=property_choices,
            num_data=len(self.data),
        )

        # Color manager
        color_properties = (
            self._feature_table.properties()
            if self._data.size > 0
            else self._feature_table.currents()
        )
        self._color_manager = ColorManager._from_layer_kwargs(
            n_colors=len(self._data),
            colors=colors if colors is not None else 'white',
            continuous_colormap=colormap,
            contrast_limits=contrast_limits,
            categorical_colormap=color_cycle,
            properties=color_properties,
        )

        # Rendering parameters
        self._point_size = point_size
        self._sh_degree = sh_degree
        self._color_mode = ColorMode(color_mode)
        self._out_of_slice_display = out_of_slice_display

        # Set rendering quality (affects shader level of detail)
        self._rendering_quality = 1.0

        # Initialize mouse callbacks
        self.mouse_pan = None
        self.mouse_zoom = None

        # Trigger refresh
        self.refresh()

    def _parse_data_input(
        self,
        data: Any,
        rotations: Any,
        scales: Any,
        opacities: Any,
        colors: Any,
    ) -> GaussianData:
        """Parse different input formats into GaussianData."""
        # Case 1: data is a file path
        if isinstance(data, (str, Path)):
            return read_ply_gaussians(data)

        # Case 2: data is already GaussianData
        if isinstance(data, GaussianData):
            return data

        # Case 3: data is positions array
        if data is None or (isinstance(data, np.ndarray) and len(data) == 0):
            # Empty layer
            positions = np.empty((0, 3), dtype=np.float32)
            rotations = np.empty((0, 4), dtype=np.float32)
            scales = np.empty((0, 3), dtype=np.float32)
            opacities = np.empty((0,), dtype=np.float32)
            colors = np.empty((0, 3), dtype=np.float32)
        else:
            positions = np.asarray(data, dtype=np.float32)
            if positions.ndim != 2 or positions.shape[1] != 3:
                raise ValueError(f'Data must be (N, 3) array, got shape {positions.shape}')

            n = len(positions)

            # Handle rotations
            if rotations is None:
                # Identity quaternions (XYZW format: [0, 0, 0, 1])
                rotations = np.zeros((n, 4), dtype=np.float32)
                rotations[:, 3] = 1.0
            else:
                rotations = np.asarray(rotations, dtype=np.float32)
                if rotations.shape != (n, 4):
                    raise ValueError(f'Rotations must be (N, 4), got {rotations.shape}')

            # Handle scales
            if scales is None:
                scales = np.ones((n, 3), dtype=np.float32)
            else:
                scales = np.asarray(scales, dtype=np.float32)
                if scales.ndim == 0:  # scalar
                    scales = np.full((n, 3), scales, dtype=np.float32)
                elif scales.shape == (n,):  # isotropic per-Gaussian
                    scales = np.tile(scales[:, None], (1, 3))
                elif scales.shape != (n, 3):
                    raise ValueError(f'Scales must be scalar, (N,), or (N, 3), got {scales.shape}')

            # Handle opacities
            if opacities is None:
                opacities = np.ones(n, dtype=np.float32)
            else:
                opacities = np.asarray(opacities, dtype=np.float32)
                if opacities.shape != (n,):
                    if opacities.ndim == 0:  # scalar
                        opacities = np.full(n, opacities, dtype=np.float32)
                    else:
                        raise ValueError(f'Opacities must be scalar or (N,), got {opacities.shape}')

            # Handle colors
            if colors is None:
                colors = np.ones((n, 3), dtype=np.float32)  # White
            else:
                colors = np.asarray(colors, dtype=np.float32)
                # Can be (N, 3) for RGB or (N, K, 3) for SH
                if colors.shape not in [(n, 3), (n, 1, 3)]:
                    if colors.ndim == 3 and colors.shape[0] == n and colors.shape[2] == 3:
                        pass  # Valid SH format
                    else:
                        raise ValueError(f'Colors must be (N, 3) or (N, K, 3), got {colors.shape}')

        return GaussianData(
            positions=positions,
            rotations=rotations,
            scales=scales,
            opacities=opacities,
            colors=colors,
        )

    @property
    def data(self) -> np.ndarray:
        """(N, 3) array of Gaussian positions in ZYX coordinates."""
        return self._data

    @data.setter
    def data(self, data: np.ndarray) -> None:
        """Set Gaussian positions."""
        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(f'Data must be (N, 3) array, got shape {data.shape}')

        prev_n = len(self._data)
        new_n = len(data)

        self._data = data

        # Resize other arrays if needed
        if new_n != prev_n:
            self._resize_arrays(prev_n, new_n)

        self.events.data(value=self.data)
        self.refresh()

    def _resize_arrays(self, prev_n: int, new_n: int) -> None:
        """Resize rotation, scale, opacity, and color arrays when data size changes."""
        if new_n > prev_n:
            # Pad with defaults
            pad = new_n - prev_n
            self._rotations = np.vstack([
                self._rotations,
                np.tile([0, 0, 0, 1], (pad, 1))
            ]).astype(np.float32)
            self._scales = np.vstack([
                self._scales,
                np.ones((pad, 3))
            ]).astype(np.float32)
            self._opacities = np.concatenate([
                self._opacities,
                np.ones(pad)
            ]).astype(np.float32)

            # Handle colors - need to preserve SH dimension if present
            if self._colors.ndim == 2:
                # Simple RGB colors
                self._colors = np.vstack([
                    self._colors,
                    np.ones((pad, 3))
                ]).astype(np.float32)
            else:
                # SH coefficients (N, K, 3)
                K = self._colors.shape[1]
                pad_colors = np.zeros((pad, K, 3), dtype=np.float32)
                pad_colors[:, 0, :] = 1.0  # DC component = white
                # Higher order SH coefficients default to 0
                self._colors = np.concatenate([self._colors, pad_colors], axis=0)
        else:
            # Truncate
            self._rotations = self._rotations[:new_n]
            self._scales = self._scales[:new_n]
            self._opacities = self._opacities[:new_n]
            if self._colors.ndim == 2:
                self._colors = self._colors[:new_n]
            else:
                self._colors = self._colors[:new_n, :, :]

    @property
    def rotations(self) -> np.ndarray:
        """(N, 4) array of rotation quaternions in XYZW format."""
        return self._rotations

    @rotations.setter
    def rotations(self, value: np.ndarray) -> None:
        """Set rotation quaternions."""
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (len(self._data), 4):
            raise ValueError(f'Rotations must be (N, 4), got {value.shape}')
        self._rotations = value
        self.events.rotations(value=self._rotations)
        self.refresh()

    @property
    def scales(self) -> np.ndarray:
        """(N, 3) array of scale parameters."""
        return self._scales

    @scales.setter
    def scales(self, value: np.ndarray | float) -> None:
        """Set scale parameters."""
        value = np.asarray(value, dtype=np.float32)
        n = len(self._data)

        if value.ndim == 0:  # scalar
            value = np.full((n, 3), value, dtype=np.float32)
        elif value.shape == (n,):  # isotropic
            value = np.tile(value[:, None], (1, 3))
        elif value.shape != (n, 3):
            raise ValueError(f'Scales must be scalar, (N,), or (N, 3), got {value.shape}')

        self._scales = value
        self.events.scales(value=self._scales)
        self.refresh()

    @property
    def opacities(self) -> np.ndarray:
        """(N,) array of opacity values [0, 1]."""
        return self._opacities

    @opacities.setter
    def opacities(self, value: np.ndarray | float) -> None:
        """Set opacity values."""
        value = np.asarray(value, dtype=np.float32)
        n = len(self._data)

        if value.ndim == 0:  # scalar
            value = np.full(n, value, dtype=np.float32)
        elif value.shape != (n,):
            raise ValueError(f'Opacities must be scalar or (N,), got {value.shape}')

        self._opacities = np.clip(value, 0, 1)
        self.events.opacities(value=self._opacities)
        self.refresh()

    @property
    def colors(self) -> np.ndarray:
        """Color or spherical harmonics coefficient data."""
        return self._colors

    @colors.setter
    def colors(self, value: np.ndarray) -> None:
        """Set colors or SH coefficients."""
        value = np.asarray(value, dtype=np.float32)
        n = len(self._data)

        # Validate shape
        if value.shape not in [(n, 3), (n, 1, 3)]:
            if value.ndim == 3 and value.shape[0] == n and value.shape[2] == 3:
                pass  # Valid SH format
            else:
                raise ValueError(f'Colors must be (N, 3) or (N, K, 3), got {value.shape}')

        self._colors = value
        self.events.colors(value=self._colors)
        self.refresh()

    @property
    def point_size(self) -> float:
        """Base size multiplier for rendering."""
        return self._point_size

    @point_size.setter
    def point_size(self, value: float) -> None:
        """Set point size multiplier."""
        self._point_size = float(value)
        self.events.point_size(value=self._point_size)

    @property
    def sh_degree(self) -> int:
        """Active spherical harmonics degree (0-3)."""
        return self._sh_degree

    @sh_degree.setter
    def sh_degree(self, value: int) -> None:
        """Set SH degree."""
        value = int(value)
        if value < 0 or value > 3:
            raise ValueError(f'SH degree must be 0-3, got {value}')
        self._sh_degree = value
        self.events.sh_degree(value=self._sh_degree)
        self.refresh()

    @property
    def color_mode(self) -> ColorMode:
        """Color mode for rendering."""
        return self._color_mode

    @color_mode.setter
    def color_mode(self, value: str | ColorMode) -> None:
        """Set color mode."""
        self._color_mode = ColorMode(value)
        self.events.color_mode(value=self._color_mode)
        self.refresh()

    @property
    def out_of_slice_display(self) -> bool:
        """Whether to display Gaussians extending into the slice."""
        return self._out_of_slice_display

    @out_of_slice_display.setter
    def out_of_slice_display(self, value: bool) -> None:
        """Set out of slice display mode."""
        self._out_of_slice_display = bool(value)
        self.events.out_of_slice_display(value=self._out_of_slice_display)
        self.refresh()

    @property
    def _indices_view(self) -> np.ndarray:
        """Indices of Gaussians in current view."""
        return self.__indices_view

    @_indices_view.setter
    def _indices_view(self, value: np.ndarray) -> None:
        """Set view indices."""
        self.__indices_view = value

    @property
    def _view_data(self) -> np.ndarray:
        """Gaussian positions in current view."""
        if len(self._indices_view) > 0:
            data = self.data[np.ix_(self._indices_view, self._slice_input.displayed)]
        else:
            data = np.zeros((0, self._slice_input.ndisplay))
        return data

    @property
    def _view_rotations(self) -> np.ndarray:
        """Rotations in current view."""
        if len(self._indices_view) > 0:
            return self.rotations[self._indices_view]
        return np.empty((0, 4), dtype=np.float32)

    @property
    def _view_scales(self) -> np.ndarray:
        """Scales in current view."""
        if len(self._indices_view) > 0:
            return self.scales[self._indices_view]
        return np.empty((0, 3), dtype=np.float32)

    @property
    def _view_opacities(self) -> np.ndarray:
        """Opacities in current view."""
        if len(self._indices_view) > 0:
            return self.opacities[self._indices_view]
        return np.empty(0, dtype=np.float32)

    @property
    def _view_colors(self) -> np.ndarray:
        """Colors in current view."""
        if len(self._indices_view) > 0:
            if self._colors.ndim == 2:
                return self._colors[self._indices_view]
            else:
                return self._colors[self._indices_view, :, :]
        return np.empty((0, 3), dtype=np.float32)

    def _make_slice_request(self, dims: Dims) -> _GaussianSliceRequest:
        """Create a slice request for the current view."""
        slice_input = _SliceInput(
            ndisplay=dims.ndisplay,
            point=dims.point,
            order=dims.order,
            world_slice=self._world_to_data_ray(dims.current_step),
        )

        return _GaussianSliceRequest(
            slice_input=slice_input,
            data=self.data,
            rotations=self.rotations,
            scales=self.scales,
            data_slice=self._data_slice,
            projection_mode=self.projection_mode,
            out_of_slice_display=self.out_of_slice_display,
        )

    def _update_slice_response(self, response: _GaussianSliceResponse) -> None:
        """Update the layer with slice response data."""
        self._slice_input = response.slice_input
        self._indices_view = response.indices
        self._view_size_scale = response.scale

    def _get_value(self, position: np.ndarray) -> Optional[int]:
        """Get Gaussian index at position (for selection)."""
        if len(self._view_data) == 0:
            return None

        # TODO: Implement proper ray-ellipsoid intersection
        # For now, use simple distance check
        distances = np.linalg.norm(self._view_data - position[self._slice_input.displayed], axis=1)
        min_idx = np.argmin(distances)

        # Check if within reasonable distance
        if distances[min_idx] < self.point_size * np.mean(self._view_scales[min_idx]):
            return self._indices_view[min_idx]

        return None

    def save(self, path: str | Path, *, plugin: Optional[str] = None) -> list[str]:
        """
        Save Gaussians to PLY file.

        Parameters
        ----------
        path : str or Path
            Output file path (should end in .ply)
        plugin : str, optional
            Plugin to use for saving (not used for PLY format)

        Returns
        -------
        list of str
            List containing the saved file path
        """
        path = Path(path)
        if path.suffix.lower() != '.ply':
            warnings.warn('Gaussians are typically saved as .ply files', UserWarning, stacklevel=2)

        gaussian_data = GaussianData(
            positions=self.data,
            rotations=self.rotations,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.colors,
        )

        write_ply_gaussians(path, gaussian_data)
        return [str(path)]

    @property
    def features(self):
        """Feature table (DataFrame-like) with one row per Gaussian."""
        return self._feature_table.values

    @features.setter
    def features(self, features) -> None:
        """Set features table."""
        self._feature_table.set_values(features, num_data=len(self.data))
        self.events.features(value=self.features)

    @property
    def feature_defaults(self):
        """Default values for features."""
        return self._feature_table.defaults

    @feature_defaults.setter
    def feature_defaults(self, defaults) -> None:
        """Set feature defaults."""
        self._feature_table.set_defaults(defaults)
        self.events.feature_defaults(value=self.feature_defaults)

    @property
    def properties(self) -> dict[str, np.ndarray]:
        """Properties dictionary (same as features but as dict)."""
        return self._feature_table.properties()

    @properties.setter
    def properties(self, properties: dict) -> None:
        """Set properties."""
        self.features = properties
        self.events.properties(value=self.properties)

    @property
    def selected_data(self) -> Selection:
        """Indices of selected Gaussians."""
        return self._selected_data

    @selected_data.setter
    def selected_data(self, selected_data: AbstractSet[int]) -> None:
        """Set selected Gaussians."""
        self._selected_data.clear()
        self._selected_data.update(selected_data)
        self._set_highlight()

    def _set_highlight(self) -> None:
        """Update highlight state."""
        # Update selected indices in view
        if len(self._selected_data) > 0:
            selected_indices = list(self._selected_data)
            self._selected_view = [
                i for i, idx in enumerate(self._indices_view)
                if idx in selected_indices
            ]
        else:
            self._selected_view = []

        self.events.highlight()
