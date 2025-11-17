"""Vispy visual for Gaussian Splatting rendering."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from vispy import gloo
from vispy.scene import visuals
from vispy.scene.visuals import Compound, Markers, create_visual_node
from vispy.visuals import Visual
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

    This visual renders 3D Gaussian splats using custom shaders that:
    1. Project 3D Gaussian covariance to 2D screen space
    2. Render each Gaussian as a point sprite
    3. Evaluate the 2D Gaussian function in the fragment shader
    4. Alpha blend the results

    Components:
        - GaussianMarkersNode: Custom visual with full shader implementation
        - Markers: Selection highlights
    """

    def __init__(self) -> None:
        # Create custom Gaussian visual with full shader implementation
        # Note: We create the node directly (scene-graph version)
        self.gaussian_markers = GaussianMarkersNode()
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
        # The point size is passed through set_gaussian_data, not set directly

    @property
    def scaling(self) -> bool:
        """
        Scaling property. If True, Gaussians rescale based on zoom
        (constant world-space size).

        Note: For custom shader implementation, scaling is always 'visual'
        since size calculation happens in the shader.
        """
        return True  # Always true for shader-based rendering

    @scaling.setter
    def scaling(self, value: bool) -> None:
        """Set scaling mode for selection markers only."""
        # Gaussian rendering uses shaders, so only set for selection markers
        scaling_txt = 'visual' if value else 'fixed'
        self.selection_markers.scaling = scaling_txt

    def set_data(
        self,
        positions=None,
        rotations=None,
        scales=None,
        opacities=None,
        colors=None,
        view_matrix=None,
        point_size_multiplier=1.0,
        **kwargs
    ) -> None:
        """
        Set Gaussian data for rendering using custom shaders.

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
        view_matrix : array (4, 4), optional
            View matrix for depth sorting
        point_size_multiplier : float
            Global size multiplier
        """
        # Use the new custom shader implementation
        self.gaussian_markers.set_gaussian_data(
            positions=positions,
            rotations=rotations,
            scales=scales,
            opacities=opacities,
            colors=colors,
            view_matrix=view_matrix,
            point_size_multiplier=point_size_multiplier,
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


class GaussianMarkers(Visual):
    """
    Custom Visual with full Gaussian splatting shader implementation.

    This visual implements proper 3D Gaussian rendering using custom shaders that:
    - Accept rotation quaternions, scales, and opacities as vertex attributes
    - Project 3D Gaussian covariance to 2D screen space
    - Render each Gaussian with proper alpha blending using point sprites
    """

    def __init__(self) -> None:
        super().__init__(vcode=GAUSSIAN_VERTEX_SHADER, fcode=GAUSSIAN_FRAGMENT_SHADER)

        # Set up GL state for proper blending and point sprites
        self.set_gl_state('translucent', depth_test=True, cull_face=False, blend=True,
                         blend_func=('src_alpha', 'one_minus_src_alpha'))

        # Enable point sprites - this allows gl_PointSize in vertex shader
        self._draw_mode = 'points'

        # Try to enable program point size (may not be needed in modern GL)
        try:
            from vispy import gloo
            # This will be set during draw
            self._enable_program_point_size = True
        except Exception:
            self._enable_program_point_size = False

        # Initialize data attributes
        self._n_gaussians = 0
        self._positions = None
        self._rotations = None
        self._scales = None
        self._opacities = None
        self._colors = None

        # Vertex buffer objects
        self._vbo_pos = gloo.VertexBuffer()
        self._vbo_rot = gloo.VertexBuffer()
        self._vbo_scale = gloo.VertexBuffer()
        self._vbo_opacity = gloo.VertexBuffer()
        self._vbo_color = gloo.VertexBuffer()

        # Shader program
        self.shared_program['a_position'] = self._vbo_pos
        self.shared_program['a_rotation'] = self._vbo_rot
        self.shared_program['a_scale'] = self._vbo_scale
        self.shared_program['a_opacity'] = self._vbo_opacity
        self.shared_program['a_color'] = self._vbo_color

        # Initialize uniforms with defaults
        self.shared_program['u_point_size'] = 1.0
        self.shared_program['u_viewport'] = (800.0, 600.0)

        # Flag to track if we need to update
        self._data_changed = False

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
        positions : array (N, 3) or None
            Gaussian centers in XYZ coordinates
        rotations : array (N, 4)
            Rotation quaternions in XYZW format
        scales : array (N, 3)
            Scale parameters
        opacities : array (N,)
            Opacity values [0, 1]
        colors : array (N, 3)
            RGB colors [0, 1]
        view_matrix : array (4, 4), optional
            View matrix for depth sorting
        point_size_multiplier : float
            Global size multiplier
        """
        # Handle empty data
        if positions is None or len(positions) == 0:
            self._n_gaussians = 0
            self._positions = None
            self._data_changed = True
            self.update()
            return

        # Apply preprocessing: filtering and depth sorting
        try:
            from napari._vispy.visuals.gaussian_utils import (
                compute_depth_order,
                filter_by_opacity_threshold,
            )

            # Filter out very transparent Gaussians for performance
            opacity_threshold = 0.01
            if opacities is not None and np.any(opacities < opacity_threshold):
                positions, rotations, scales, opacities, colors = filter_by_opacity_threshold(
                    positions, rotations, scales, opacities, colors, opacity_threshold
                )

            if len(positions) == 0:
                self._n_gaussians = 0
                self._positions = None
                self._data_changed = True
                self.update()
                return

            # Apply depth sorting if view matrix is available
            # This ensures correct alpha blending (back-to-front rendering)
            if view_matrix is not None:
                try:
                    sort_indices = compute_depth_order(positions, view_matrix)
                    positions = positions[sort_indices]
                    rotations = rotations[sort_indices]
                    scales = scales[sort_indices]
                    opacities = opacities[sort_indices]
                    colors = colors[sort_indices]
                except Exception:
                    # If depth sorting fails, continue without it
                    pass

        except ImportError:
            # If utils not available, proceed without filtering/sorting
            pass

        # Store data
        self._n_gaussians = len(positions)
        self._positions = np.ascontiguousarray(positions, dtype=np.float32)
        self._rotations = np.ascontiguousarray(rotations, dtype=np.float32)
        self._scales = np.ascontiguousarray(scales, dtype=np.float32)
        self._opacities = np.ascontiguousarray(opacities, dtype=np.float32)
        self._colors = np.ascontiguousarray(colors, dtype=np.float32)

        # Update point size uniform
        self.shared_program['u_point_size'] = float(point_size_multiplier)

        # Mark data as changed
        self._data_changed = True
        self.update()

    def _prepare_transforms(self, view):
        """Prepare transforms and update uniforms."""
        # Update viewport size
        if view.size is not None:
            self.shared_program['u_viewport'] = tuple(view.size)

        # Get visual to canvas transform and extract view/projection
        # In vispy's scene graph, we need to get the full transform
        if hasattr(view, 'camera'):
            # Get view matrix (camera transform)
            view_mat = view.camera.view_matrix
            proj_mat = view.camera.projection_matrix

            self.shared_program['u_view'] = view_mat
            self.shared_program['u_projection'] = proj_mat
        else:
            # Fallback to identity matrices
            self.shared_program['u_view'] = np.eye(4, dtype=np.float32)
            self.shared_program['u_projection'] = np.eye(4, dtype=np.float32)

    def _prepare_draw(self, view):
        """Prepare for drawing."""
        # Upload data to GPU if changed
        if self._data_changed and self._positions is not None:
            self._vbo_pos.set_data(self._positions)
            self._vbo_rot.set_data(self._rotations)
            self._vbo_scale.set_data(self._scales)
            self._vbo_opacity.set_data(self._opacities)
            self._vbo_color.set_data(self._colors)
            self._data_changed = False

        # Prepare transforms
        self._prepare_transforms(view)

    def _compute_bounds(self, axis, view):
        """Compute bounds for camera auto-range."""
        if self._positions is None or len(self._positions) == 0:
            return None

        # Use positions extended by maximum scale
        if self._scales is not None:
            max_scale = np.max(self._scales)
            pos_min = np.min(self._positions, axis=0)
            pos_max = np.max(self._positions, axis=0)
            return (pos_min[axis] - max_scale, pos_max[axis] + max_scale)
        else:
            return (np.min(self._positions[:, axis]), np.max(self._positions[:, axis]))

    def draw(self, transforms):
        """Draw the visual."""
        if self._n_gaussians == 0 or self._positions is None:
            return

        # Draw all Gaussians as points
        Visual.draw(self, transforms)


# Create scene-graph compatible wrapper
GaussianMarkersNode = create_visual_node(GaussianMarkers)
