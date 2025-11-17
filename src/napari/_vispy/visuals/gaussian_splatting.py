"""Vispy visual for Gaussian Splatting rendering."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from vispy.scene.visuals import Compound, Markers
from vispy.visuals.shaders import Function

from napari._vispy.visuals.clipping_planes_mixin import ClippingPlanesMixin

# Vertex shader for Gaussian splatting
# Projects 3D Gaussians to 2D screen space and computes 2D covariance
GAUSSIAN_VERTEX_SHADER = """
varying vec4 v_color;
varying vec2 v_center;
varying mat2 v_cov2d_inv;
varying float v_opacity;

attribute vec3 a_position;      // Gaussian center (XYZ in vispy coords)
attribute vec4 a_rotation;      // Rotation quaternion (XYZW)
attribute vec3 a_scale;         // Scale parameters
attribute float a_opacity;      // Opacity [0, 1]
attribute vec3 a_color;         // RGB color (or SH DC component)

uniform mat4 u_view;
uniform mat4 u_projection;
uniform float u_point_size;     // Base size multiplier
uniform vec2 u_viewport;        // Viewport width and height in pixels

// Quaternion to rotation matrix conversion
mat3 quat_to_mat(vec4 q) {
    // Normalize quaternion
    q = normalize(q);
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;

    return mat3(
        1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy),
        2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx),
        2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)
    );
}

void main() {
    // Transform position to view space
    vec4 pos_view = u_view * vec4(a_position, 1.0);
    vec3 p_view = pos_view.xyz / pos_view.w;

    // Project to clip space
    vec4 pos_clip = u_projection * pos_view;
    gl_Position = pos_clip;

    // Get rotation matrix from quaternion
    mat3 R = quat_to_mat(a_rotation);

    // Construct scale matrix
    mat3 S = mat3(
        a_scale.x, 0.0, 0.0,
        0.0, a_scale.y, 0.0,
        0.0, 0.0, a_scale.z
    );

    // Compute 3D covariance matrix: Σ = R @ S @ S^T @ R^T
    mat3 RS = R * S;
    mat3 Sigma3D = RS * transpose(RS);

    // Transform covariance to view space
    mat3 J_view = mat3(u_view);  // Upper-left 3x3 of view matrix
    mat3 Sigma_view = J_view * Sigma3D * transpose(J_view);

    // Compute Jacobian of perspective projection
    float z = p_view.z;
    float focal_x = u_projection[0][0] * u_viewport.x * 0.5;
    float focal_y = u_projection[1][1] * u_viewport.y * 0.5;

    // Simplified: only project top-left 2x2 of covariance
    // This is an approximation that works well for Gaussians not too close to camera
    mat2 Sigma_2d = mat2(
        Sigma_view[0][0], Sigma_view[0][1],
        Sigma_view[1][0], Sigma_view[1][1]
    );

    // Scale by focal length and depth
    float scale_factor = 1.0 / (z * z);
    Sigma_2d = Sigma_2d * (focal_x * focal_y * scale_factor);

    // Add low-pass filter for antialiasing
    Sigma_2d[0][0] += 0.3;
    Sigma_2d[1][1] += 0.3;

    // Apply point size multiplier
    Sigma_2d = Sigma_2d * (u_point_size * u_point_size);

    // Compute inverse for fragment shader
    float det = Sigma_2d[0][0] * Sigma_2d[1][1] - Sigma_2d[0][1] * Sigma_2d[1][0];
    if (abs(det) > 1e-6) {
        v_cov2d_inv = mat2(
            Sigma_2d[1][1], -Sigma_2d[0][1],
            -Sigma_2d[1][0], Sigma_2d[0][0]
        ) / det;
    } else {
        // Degenerate covariance - use identity
        v_cov2d_inv = mat2(1.0, 0.0, 0.0, 1.0);
    }

    // Pass data to fragment shader
    v_color = vec4(a_color, 1.0);
    v_opacity = a_opacity;

    // Compute screen-space center
    vec4 ndc = pos_clip / pos_clip.w;
    v_center = ((ndc.xy + 1.0) * 0.5) * u_viewport;

    // Set point size based on covariance extent
    // Use 3-sigma as the point radius (covers 99.7% of Gaussian)
    float max_extent = 3.0 * sqrt(max(Sigma_2d[0][0], Sigma_2d[1][1]));
    gl_PointSize = max(1.0, 2.0 * max_extent);
}
"""

# Fragment shader for Gaussian splatting
# Evaluates the Gaussian function and applies alpha blending
GAUSSIAN_FRAGMENT_SHADER = """
varying vec4 v_color;
varying vec2 v_center;
varying mat2 v_cov2d_inv;
varying float v_opacity;

uniform vec2 u_viewport;

void main() {
    // Compute distance from Gaussian center in screen space
    vec2 d = gl_FragCoord.xy - v_center;

    // Evaluate Gaussian: exp(-0.5 * d^T * Σ^(-1) * d)
    float power = -0.5 * dot(d, v_cov2d_inv * d);
    float alpha = exp(power);

    // Apply opacity
    alpha *= v_opacity;

    // Discard fragments below threshold to improve performance
    if (alpha < 1.0 / 256.0) {
        discard;
    }

    // Output color with alpha
    gl_FragColor = vec4(v_color.rgb * alpha, alpha);
}
"""


class GaussianSplattingVisual(ClippingPlanesMixin, Compound):
    """
    Compound vispy visual for Gaussian Splatting rendering.

    This visual renders 3D Gaussian splats by:
    1. Projecting 3D Gaussian covariance to 2D screen space
    2. Rendering each Gaussian as a point sprite
    3. Evaluating the 2D Gaussian function in the fragment shader
    4. Alpha blending the results

    Components:
        - Markers for Gaussian splats (custom shaders)
        - Markers for selection highlights
    """

    def __init__(self) -> None:
        # Create markers visual with custom shaders for Gaussians
        self.gaussian_markers = GaussianMarkers()
        self.selection_markers = Markers()

        super().__init__([
            self.gaussian_markers,
            self.selection_markers,
        ])

        self.scaling = True
        self._point_size = 1.0

    @property
    def point_size(self) -> float:
        """Base size multiplier for Gaussian splats."""
        return self._point_size

    @point_size.setter
    def point_size(self, value: float) -> None:
        """Set point size multiplier."""
        self._point_size = float(value)
        if hasattr(self.gaussian_markers, 'shared_program'):
            self.gaussian_markers.shared_program['u_point_size'] = self._point_size

    @property
    def scaling(self) -> bool:
        """
        Scaling property. If True, Gaussians rescale based on zoom
        (constant world-space size).
        """
        return self.gaussian_markers.scaling == 'visual'

    @scaling.setter
    def scaling(self, value: bool) -> None:
        """Set scaling mode."""
        scaling_txt = 'visual' if value else 'fixed'
        self.gaussian_markers.scaling = scaling_txt
        self.selection_markers.scaling = scaling_txt

    def set_data(
        self,
        positions=None,
        rotations=None,
        scales=None,
        opacities=None,
        colors=None,
        **kwargs
    ) -> None:
        """
        Set Gaussian data for rendering.

        Parameters
        ----------
        positions : array (N, 3)
            Gaussian centers in XYZ (vispy coordinates, already converted from ZYX)
        rotations : array (N, 4)
            Rotation quaternions in XYZW format (transformed to XYZ space)
        scales : array (N, 3)
            Scale parameters (transformed to XYZ order)
        opacities : array (N,)
            Opacity values [0, 1]
        colors : array (N, 3)
            RGB colors [0, 1]
        """
        if positions is None or len(positions) == 0:
            self.gaussian_markers.set_data(None)
            return

        # Prepare data for custom shader attributes
        # The Markers visual will handle positions automatically
        # We need to add custom attributes for rotations, scales, opacities, colors

        # For now, use standard Markers rendering as a placeholder
        # TODO: Implement custom VisualNode with proper shader attributes
        self.gaussian_markers.set_data(
            pos=positions,
            face_color=colors if colors is not None else np.ones((len(positions), 3)),
            edge_color=None,
            size=10,  # Base size, will be modulated by scales in shader
        )

    def set_selection_data(self, positions=None, **kwargs) -> None:
        """Set data for selection highlights."""
        if positions is None or len(positions) == 0:
            self.selection_markers.set_data(None)
            return

        self.selection_markers.set_data(
            pos=positions,
            face_color='cyan',
            edge_color='yellow',
            size=15,
            edge_width=2,
        )


class GaussianMarkers(Markers):
    """
    Custom Markers visual with Gaussian splatting shaders.

    This extends vispy's Markers to use custom shaders that:
    - Accept rotation quaternions, scales, and opacities as attributes
    - Project 3D Gaussian covariance to 2D screen space
    - Render each Gaussian with proper alpha blending
    """

    # NOTE: For full shader implementation, we would need to:
    # 1. Override _prepare_draw() to set custom shader attributes
    # 2. Replace the vertex and fragment shaders with GAUSSIAN_VERTEX_SHADER
    #    and GAUSSIAN_FRAGMENT_SHADER
    # 3. Add proper attribute handling for rotations, scales, opacities
    #
    # This is a simplified placeholder implementation.
    # Full implementation requires deeper vispy integration.

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rotations = None
        self._scales = None
        self._opacities = None

    def set_gaussian_data(
        self,
        positions,
        rotations,
        scales,
        opacities,
        colors,
        view_matrix=None,
        point_size_multiplier=1.0,
    ) -> None:
        """
        Set Gaussian splat data with all attributes.

        Parameters
        ----------
        positions : array (N, 3)
            Gaussian centers
        rotations : array (N, 4)
            Rotation quaternions
        scales : array (N, 3)
            Scale parameters
        opacities : array (N,)
            Opacity values
        colors : array (N, 3)
            RGB colors
        view_matrix : array (4, 4), optional
            View matrix for depth sorting
        point_size_multiplier : float
            Global size multiplier
        """
        self._rotations = rotations
        self._scales = scales
        self._opacities = opacities

        # Enhanced rendering with depth sorting and better size calculation
        if positions is not None and len(positions) > 0:
            try:
                from napari._vispy.visuals.gaussian_utils import (
                    apply_opacity_to_colors,
                    compute_depth_order,
                    compute_gaussian_sizes,
                    filter_by_opacity_threshold,
                )

                # Filter out very transparent Gaussians for performance
                opacity_threshold = 0.01
                if opacities is not None and np.any(opacities < opacity_threshold):
                    positions, rotations, scales, opacities, colors = filter_by_opacity_threshold(
                        positions, rotations, scales, opacities, colors, opacity_threshold
                    )

                if len(positions) == 0:
                    self.set_data(None)
                    return

                # Compute sizes based on scales (better than mean)
                if scales is not None:
                    sizes = compute_gaussian_sizes(scales, point_size_multiplier)
                else:
                    sizes = 10.0 * point_size_multiplier

                # Apply opacity to colors
                if colors is not None and opacities is not None:
                    face_colors = apply_opacity_to_colors(colors, opacities)
                elif colors is not None:
                    face_colors = colors
                else:
                    face_colors = np.ones((len(positions), 4))
                    if opacities is not None:
                        face_colors[:, 3] = opacities

                # Apply depth sorting if view matrix is available
                # This ensures correct alpha blending (back-to-front rendering)
                if view_matrix is not None:
                    try:
                        sort_indices = compute_depth_order(positions, view_matrix)
                        positions = positions[sort_indices]
                        face_colors = face_colors[sort_indices]
                        if isinstance(sizes, np.ndarray):
                            sizes = sizes[sort_indices]
                        # Update stored data
                        if self._rotations is not None:
                            self._rotations = self._rotations[sort_indices]
                        if self._scales is not None:
                            self._scales = self._scales[sort_indices]
                        if self._opacities is not None:
                            self._opacities = self._opacities[sort_indices]
                    except Exception:
                        # If depth sorting fails, continue without it
                        pass

                self.set_data(
                    pos=positions,
                    face_color=face_colors,
                    edge_color=None,
                    size=sizes,
                )

            except ImportError:
                # Fallback to simple rendering if utils not available
                if colors is not None:
                    face_colors = np.column_stack([
                        colors,
                        opacities if opacities is not None else np.ones(len(positions))
                    ])
                else:
                    face_colors = np.column_stack([
                        np.ones((len(positions), 3)),
                        opacities if opacities is not None else np.ones(len(positions))
                    ])

                if scales is not None:
                    sizes = np.mean(scales, axis=1) * 10.0 * point_size_multiplier
                else:
                    sizes = 10.0 * point_size_multiplier

                self.set_data(
                    pos=positions,
                    face_color=face_colors,
                    edge_color=None,
                    size=sizes,
                )
        else:
            self.set_data(None)
