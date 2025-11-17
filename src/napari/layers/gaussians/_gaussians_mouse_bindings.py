"""Mouse bindings for the Gaussians layer."""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from napari.layers.gaussians import Gaussians


def select(layer: 'Gaussians', event):
    """Select Gaussians."""
    # Get Gaussian under cursor
    value = layer._get_value(event.position)

    if value is not None:
        # Toggle selection
        if event.modifiers.get('shift', False):
            # Add to selection
            if value in layer.selected_data:
                layer.selected_data.remove(value)
            else:
                layer.selected_data.add(value)
        else:
            # Replace selection
            layer.selected_data = {value}
    else:
        # Click on empty space - clear selection if not shift
        if not event.modifiers.get('shift', False):
            layer.selected_data = set()

    layer.refresh()


def highlight(layer: 'Gaussians', event):
    """Highlight Gaussian under cursor."""
    value = layer._get_value(event.position)

    if value is not None:
        layer._highlight_index = [value]
    else:
        layer._highlight_index = []

    layer.events.highlight()
