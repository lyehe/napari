from collections import OrderedDict
from enum import auto

from napari.utils.misc import StringEnum
from napari.utils.translations import trans


class ColorMode(StringEnum):
    """
    ColorMode: Color setting mode for Gaussians.

    DIRECT (default mode) allows each Gaussian to be set arbitrarily

    CYCLE allows the color to be set via a color cycle over an attribute

    COLORMAP allows color to be set via a color map over an attribute

    SPHERICAL_HARMONICS uses view-dependent spherical harmonics coefficients
    """

    DIRECT = auto()
    CYCLE = auto()
    COLORMAP = auto()
    SPHERICAL_HARMONICS = auto()


class Mode(StringEnum):
    """
    Mode: Interactive mode. The normal, default mode is PAN_ZOOM, which
    allows for normal interactivity with the canvas.

    SELECT allows the user to select Gaussians by clicking on them
    """

    PAN_ZOOM = auto()
    TRANSFORM = auto()
    SELECT = auto()


class GaussiansProjectionMode(StringEnum):
    """
    Projection mode for aggregating a thick nD slice onto displayed dimensions.

        * NONE: ignore slice thickness, only using the dims point
        * ALL: project all Gaussians in the slice onto displayed dimensions
    """

    NONE = auto()
    ALL = auto()


GAUSSIANS_PROJECTION_MODE_TRANSLATIONS = {
    trans._('none'): GaussiansProjectionMode.NONE,
    trans._('all'): GaussiansProjectionMode.ALL,
}
