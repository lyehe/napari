from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

from napari.layers.base._slice import _next_request_id
from napari.layers.gaussians._gaussians_constants import GaussiansProjectionMode
from napari.layers.utils._slice_input import _SliceInput, _ThickNDSlice


@dataclass(frozen=True)
class _GaussianSliceResponse:
    """Contains all the output data of slicing a Gaussians layer.

    Attributes
    ----------
    indices : array like
        Indices of the sliced Gaussian data.
    scale : array like or float
        Used to scale the sliced Gaussians for visualization.
        Should be broadcastable to indices.
    slice_input : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    request_id : int
        The identifier of the request from which this was generated.
    """

    indices: np.ndarray = field(repr=False)
    scale: Any = field(repr=False)
    slice_input: _SliceInput
    request_id: int


@dataclass(frozen=True)
class _GaussianSliceRequest:
    """A callable that stores all the input data needed to slice a Gaussians layer.

    This should be treated as deeply immutable structure, even though some
    fields can be modified in place. It is like a function that has captured
    all its inputs already.

    In general, calling an instance of this may take a long time, so you may
    want to run it off the main thread.

    Attributes
    ----------
    slice_input : _SliceInput
        Describes the slicing plane or bounding box in the layer's dimensions.
    data : Any
        The layer's positions data field (N, D).
    rotations : array like
        (N, 4) rotation quaternions in XYZW format.
    scales : array like
        (N, 3) scale parameters.
    data_slice : _ThickNDSlice
        The slicing coordinates and margins in data space.
    projection_mode : GaussiansProjectionMode
        How to handle thick slices.
    out_of_slice_display : bool
        Whether to show Gaussians that extend into the slice.
    """

    slice_input: _SliceInput
    data: Any = field(repr=False)
    rotations: Any = field(repr=False)
    scales: Any = field(repr=False)
    data_slice: _ThickNDSlice = field(repr=False)
    projection_mode: GaussiansProjectionMode
    out_of_slice_display: bool = field(repr=False)
    id: int = field(default_factory=_next_request_id)

    def __call__(self) -> _GaussianSliceResponse:
        # Return early if no data
        if len(self.data) == 0:
            return _GaussianSliceResponse(
                indices=np.array([], dtype=int),
                scale=np.empty(0),
                slice_input=self.slice_input,
                request_id=self.id,
            )

        not_disp = list(self.slice_input.not_displayed)
        if not not_disp:
            # If we want to display everything, then use all indices.
            # scale is only impacted by not displayed data, therefore 1
            return _GaussianSliceResponse(
                indices=np.arange(len(self.data), dtype=int),
                scale=1,
                slice_input=self.slice_input,
                request_id=self.id,
            )

        slice_indices, scale = self._get_slice_data(not_disp)

        return _GaussianSliceResponse(
            indices=slice_indices,
            scale=scale,
            slice_input=self.slice_input,
            request_id=self.id,
        )

    def _get_slice_data(self, not_disp: list[int]) -> tuple[npt.NDArray, int | npt.NDArray]:
        """
        Get indices of Gaussians visible in the slice.

        For Gaussians, we need to account for their oriented extent,
        not just their center position. A Gaussian is visible if its
        3-sigma extent intersects the slice plane.
        """
        data = self.data[:, not_disp]
        scale = 1

        point, m_left, m_right = self.data_slice[not_disp].as_array()

        if self.projection_mode == 'none':
            low = point.copy()
            high = point.copy()
        else:
            low = point - m_left
            high = point + m_right

        # assume slice thickness of 1 in data pixels
        # (same as before thick slices were implemented)
        too_thin_slice = np.isclose(high, low)
        low[too_thin_slice] -= 0.5
        high[too_thin_slice] += 0.5

        # Check if Gaussian centers are inside the slice
        inside_slice = np.all((data >= low) & (data <= high), axis=1)
        slice_indices = np.where(inside_slice)[0].astype(int)

        # For Gaussians, also check if their oriented extent intersects the slice
        if self.out_of_slice_display and self.slice_input.ndim > 2:
            # Compute oriented extent for each Gaussian
            # Use 3-sigma as the extent threshold (99.7% of Gaussian mass)
            oriented_extents = self._compute_oriented_extents(not_disp)

            # Check distance from slice boundaries
            dist_from_low = np.abs(data - low)
            dist_from_high = np.abs(data - high)
            distances = np.minimum(dist_from_low, dist_from_high)
            distances[inside_slice] = 0  # Centers inside are at distance 0

            # Display Gaussians that extend into the slice
            matches = np.all(distances <= oriented_extents, axis=1)
            if not np.any(matches):
                return np.empty(0, dtype=int), 1

            # Rescale based on how much the Gaussian extends into slice
            extent_match = oriented_extents[matches]
            scale_per_dim = np.clip(
                (extent_match - distances[matches]) / (extent_match + 1e-8),
                0, 1
            )
            scale = np.prod(scale_per_dim, axis=1)
            slice_indices = np.where(matches)[0].astype(int)

        return slice_indices, scale

    def _compute_oriented_extents(self, not_disp: list[int]) -> npt.NDArray:
        """
        Compute the oriented extent of each Gaussian in non-displayed dimensions.

        For each Gaussian, we compute how far it extends in each dimension
        when accounting for rotation. We use 3-sigma as the extent.

        Parameters
        ----------
        not_disp : list of int
            Indices of non-displayed dimensions

        Returns
        -------
        extents : ndarray (N, len(not_disp))
            Extent of each Gaussian in each non-displayed dimension
        """
        n = len(self.data)
        num_not_disp = len(not_disp)

        # Get rotation matrices and scales
        if len(self.rotations) > 0:
            R = Rotation.from_quat(self.rotations).as_matrix()  # (N, 3, 3)
            scales_3d = self.scales * 3.0  # 3-sigma threshold

            # For each Gaussian, compute extent in each dimension
            # The extent in dimension i is the maximum projection of the
            # rotated ellipsoid axes onto dimension i
            extents = np.zeros((n, num_not_disp), dtype=np.float32)

            for i, dim_idx in enumerate(not_disp):
                if dim_idx < 3:  # Only for spatial dimensions
                    # Project scaled axes onto this dimension
                    # Extent = max |R[:, j] * scale[j]| for dimension dim_idx
                    for j in range(3):
                        extents[:, i] += (R[:, dim_idx, j] * scales_3d[:, j]) ** 2
                    extents[:, i] = np.sqrt(extents[:, i])
        else:
            # Fallback: use isotropic extent
            if len(self.scales) > 0:
                max_scale = np.max(self.scales, axis=1, keepdims=True) * 3.0
                extents = np.tile(max_scale, (1, num_not_disp))
            else:
                extents = np.ones((n, num_not_disp), dtype=np.float32)

        return extents
