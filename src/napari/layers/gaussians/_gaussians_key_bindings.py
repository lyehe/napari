"""Key bindings for the Gaussians layer."""

import numpy as np

from napari.layers.gaussians.gaussians import Gaussians


@Gaussians.bind_key('Space')
def hold_to_pan_zoom(layer: Gaussians) -> None:
    """Hold to pan/zoom."""
    if layer._mode != layer._modeclass.PAN_ZOOM:
        layer._mode_history = layer._mode
    layer.mode = layer._modeclass.PAN_ZOOM


@Gaussians.bind_key('Space', overwrite=True)
def release_pan_zoom(layer: Gaussians) -> None:
    """Release pan/zoom."""
    if hasattr(layer, '_mode_history'):
        layer.mode = layer._mode_history
        del layer._mode_history


@Gaussians.bind_key('s')
def activate_select_mode(layer: Gaussians) -> None:
    """Activate select mode."""
    layer.mode = layer._modeclass.SELECT


@Gaussians.bind_key('z')
def activate_pan_zoom_mode(layer: Gaussians) -> None:
    """Activate pan/zoom mode."""
    layer.mode = layer._modeclass.PAN_ZOOM


@Gaussians.bind_key('Control-C')
def copy_gaussians(layer: Gaussians) -> None:
    """Copy selected Gaussians to clipboard."""
    if len(layer.selected_data) > 0:
        indices = list(layer.selected_data)
        layer._clipboard = {
            'positions': layer.data[indices].copy(),
            'rotations': layer.rotations[indices].copy(),
            'scales': layer.scales[indices].copy(),
            'opacities': layer.opacities[indices].copy(),
            'colors': layer.colors[indices].copy() if layer.colors.ndim == 2 else layer.colors[indices, :, :].copy(),
        }


@Gaussians.bind_key('Control-V')
def paste_gaussians(layer: Gaussians) -> None:
    """Paste Gaussians from clipboard."""
    if layer._clipboard:
        # TODO: Implement paste functionality
        pass


@Gaussians.bind_key('Backspace')
@Gaussians.bind_key('Delete')
def delete_selected_gaussians(layer: Gaussians) -> None:
    """Delete selected Gaussians."""
    if len(layer.selected_data) > 0:
        indices = list(layer.selected_data)
        mask = np.ones(len(layer.data), dtype=bool)
        mask[indices] = False

        # Update all arrays
        layer._data = layer.data[mask]
        layer._rotations = layer.rotations[mask]
        layer._scales = layer.scales[mask]
        layer._opacities = layer.opacities[mask]
        if layer._colors.ndim == 2:
            layer._colors = layer.colors[mask]
        else:
            layer._colors = layer.colors[mask, :, :]

        layer.selected_data = set()
        layer.events.data()
        layer.refresh()
