"""Vispy layer for Gaussians."""

from __future__ import annotations

import numpy as np

from napari._vispy.layers.base import VispyBaseLayer
from napari._vispy.visuals.gaussian_splatting import GaussianSplattingVisual


class VispyGaussiansLayer(VispyBaseLayer):
    """
    Vispy view for the Gaussians layer.

    This class handles the rendering of Gaussian splats using vispy.
    It bridges the napari Gaussians layer to the GaussianSplattingVisual.
    """

    def __init__(self, layer) -> None:
        # Create the visual node
        node = GaussianSplattingVisual()

        # Initialize base layer
        super().__init__(layer, node)

        # Connect layer events to update methods
        self.layer.events.point_size.connect(self._on_point_size_change)
        self.layer.events.sh_degree.connect(self._on_data_change)
        self.layer.events.color_mode.connect(self._on_data_change)
        self.layer.events.rotations.connect(self._on_data_change)
        self.layer.events.scales.connect(self._on_data_change)
        self.layer.events.opacities.connect(self._on_data_change)
        self.layer.events.colors.connect(self._on_data_change)
        self.layer.events.highlight.connect(self._on_highlight_change)

        # Initial update
        self.reset()
        self._on_data_change()

    def _on_data_change(self):
        """Update the visual when layer data changes."""
        # Get view data (already sliced by the layer)
        positions = None
        rotations = None
        scales = None
        opacities = None
        colors = None

        if len(self.layer._view_data) > 0:
            # Convert positions from ZYX (napari) to XYZ (vispy)
            # Note: layer._view_data is already in display dimensions
            # We need to get full 3D positions for proper rendering
            indices = self.layer._indices_view
            if len(indices) > 0:
                # Get full 3D positions and reverse to XYZ
                positions_zyx = self.layer.data[indices]
                positions = np.asarray(positions_zyx[:, ::-1], dtype=np.float32)

                # Get rotations and transform from ZYX to XYZ space
                rotations_zyx = self.layer.rotations[indices]
                rotations = self._transform_rotations_to_vispy(rotations_zyx)

                # Get scales and reverse to XYZ order
                scales_zyx = self.layer.scales[indices]
                scales = np.asarray(scales_zyx[:, ::-1], dtype=np.float32)

                # Get opacities (no transformation needed)
                opacities = self.layer.opacities[indices].astype(np.float32)

                # Get colors based on color mode
                colors = self._get_colors(indices)

                # Apply view size scaling if present
                if hasattr(self.layer, '_view_size_scale') and len(self.layer._view_size_scale) > 0:
                    scale_factors = np.asarray(self.layer._view_size_scale)
                    if scale_factors.ndim == 0:
                        scales = scales * scale_factors
                    else:
                        scales = scales * scale_factors[:, None]

        # Get view matrix for depth sorting
        view_matrix = None
        if hasattr(self.node, 'get_transform'):
            try:
                # Get the view transform from the parent
                view_tr = self.node.get_transform('visual', 'canvas')
                if hasattr(view_tr, 'matrix'):
                    view_matrix = view_tr.matrix
            except Exception:
                pass  # Continue without view matrix

        # Get point size multiplier from layer
        point_size_multiplier = self.layer.point_size if hasattr(self.layer, 'point_size') else 1.0

        # Update the visual with all parameters
        # The GaussianSplattingVisual.set_data() now accepts view_matrix and point_size_multiplier
        self.node.set_data(
            positions=positions,
            rotations=rotations,
            scales=scales,
            opacities=opacities,
            colors=colors,
            view_matrix=view_matrix,
            point_size_multiplier=point_size_multiplier,
        )

        # Update selection highlights
        self._on_highlight_change()

        # Trigger visual update
        self.node.update()

        # Update transform
        self._on_matrix_change()

    def _transform_rotations_to_vispy(self, rotations_zyx: np.ndarray) -> np.ndarray:
        """
        Transform rotation quaternions from ZYX (napari) to XYZ (vispy) space.

        This uses the rotation matrix method to properly transform the rotations.

        Parameters
        ----------
        rotations_zyx : array (N, 4)
            Quaternions in XYZW format, representing rotations in ZYX space

        Returns
        -------
        rotations_xyz : array (N, 4)
            Quaternions in XYZW format, representing rotations in XYZ space
        """
        from scipy.spatial.transform import Rotation

        # Permutation matrix: ZYX -> XYZ
        P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32)

        # Convert quaternions to rotation matrices
        R_matrices_zyx = Rotation.from_quat(rotations_zyx).as_matrix()

        # Transform to XYZ space: R_xyz = P^T @ R_zyx @ P
        R_matrices_xyz = P.T @ R_matrices_zyx @ P

        # Convert back to quaternions
        rotations_xyz = Rotation.from_matrix(R_matrices_xyz).as_quat()

        return rotations_xyz.astype(np.float32)

    def _get_colors(self, indices: np.ndarray) -> np.ndarray:
        """
        Get colors for the given indices based on color mode.

        Parameters
        ----------
        indices : array
            Indices of Gaussians to get colors for

        Returns
        -------
        colors : array (N, 3)
            RGB colors [0, 1]
        """
        from napari.layers.gaussians._gaussians_constants import ColorMode

        if self.layer.color_mode == ColorMode.SPHERICAL_HARMONICS:
            # Use spherical harmonics (for now, just DC component)
            if self.layer._colors.ndim == 3:
                # SH coefficients - use DC component (index 0)
                colors = self.layer._colors[indices, 0, :]
            else:
                # Already RGB
                colors = self.layer._colors[indices]
        elif self.layer.color_mode == ColorMode.DIRECT:
            # Direct color from colors array
            if self.layer._colors.ndim == 3:
                colors = self.layer._colors[indices, 0, :]
            else:
                colors = self.layer._colors[indices]
        elif self.layer.color_mode == ColorMode.CYCLE:
            # Use color manager with cycle
            if hasattr(self.layer, '_color_manager'):
                colors = self.layer._color_manager.face_color[indices, :3]
            else:
                colors = self.layer._colors[indices] if self.layer._colors.ndim == 2 else self.layer._colors[indices, 0, :]
        elif self.layer.color_mode == ColorMode.COLORMAP:
            # Use color manager with colormap
            if hasattr(self.layer, '_color_manager'):
                colors = self.layer._color_manager.face_color[indices, :3]
            else:
                colors = self.layer._colors[indices] if self.layer._colors.ndim == 2 else self.layer._colors[indices, 0, :]
        else:
            # Fallback
            if self.layer._colors.ndim == 3:
                colors = self.layer._colors[indices, 0, :]
            else:
                colors = self.layer._colors[indices]

        # Ensure valid range [0, 1]
        colors = np.clip(colors, 0, 1).astype(np.float32)

        return colors

    def _on_point_size_change(self):
        """Update point size multiplier."""
        self.node.point_size = self.layer.point_size
        self.node.update()

    def _on_highlight_change(self):
        """Update selection highlights."""
        if len(self.layer.selected_data) > 0 and len(self.layer._view_data) > 0:
            # Get selected positions in view
            selected_indices = list(self.layer.selected_data)
            view_indices = self.layer._indices_view

            # Find which selected Gaussians are in view
            selected_in_view = [idx for idx in selected_indices if idx in view_indices]

            if selected_in_view:
                # Get positions in XYZ (vispy) coordinates
                positions_zyx = self.layer.data[selected_in_view]
                positions = np.asarray(positions_zyx[:, ::-1], dtype=np.float32)

                self.node.set_selection_data(positions=positions)
            else:
                self.node.set_selection_data(positions=None)
        else:
            self.node.set_selection_data(positions=None)

    def reset(self, event=None):
        """Reset the layer view."""
        super().reset()
        self._on_point_size_change()

    def _on_matrix_change(self):
        """Update transformation matrix when layer transform changes."""
        # Get displayed dimensions
        dims_displayed = self.layer._slice_input.displayed
        transform = self.layer._transforms.simplified.set_slice(dims_displayed)

        # Convert NumPy axis ordering (ZYX) to VisPy axis ordering (XYZ)
        translate = transform.translate[::-1]
        matrix = transform.linear_matrix[::-1, ::-1].T

        # Update visual transform
        self.node.transform.matrix = matrix
        self.node.transform.translate = translate
