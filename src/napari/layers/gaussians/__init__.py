from napari.layers.gaussians import _gaussians_key_bindings
from napari.layers.gaussians.gaussians import Gaussians

# Note that importing _gaussians_key_bindings is needed as the Gaussians layer gets
# decorated with keybindings during that process, but it is not directly needed
# by our users and so is deleted below
del _gaussians_key_bindings

__all__ = ['Gaussians']
