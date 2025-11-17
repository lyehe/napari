"""Qt controls for the Gaussians layer."""

from typing import TYPE_CHECKING

from napari._qt.layer_controls.qt_layer_controls_base import QtLayerControls
from napari._qt.layer_controls.widgets import (
    QtOutSliceCheckBoxControl,
    QtProjectionModeControl,
)
from napari._qt.widgets.qt_mode_buttons import QtModePushButton
from napari.layers.gaussians._gaussians_constants import Mode
from napari.utils.action_manager import action_manager
from napari.utils.translations import trans

if TYPE_CHECKING:
    import napari.layers


class QtGaussiansControls(QtLayerControls):
    """
    Qt view and controls for the napari Gaussians layer.

    Parameters
    ----------
    layer : napari.layers.Gaussians
        An instance of a napari Gaussians layer.

    Attributes
    ----------
    _projection_mode_control : QtProjectionModeControl
        Widget that wraps dropdown menu to select the projection mode.
    _out_slice_checkbox_control : QtOutSliceCheckBoxControl
        Widget that wraps a checkbox to indicate whether to render out of slice.
    select_button : QtModeRadioButton
        Button to select Gaussians from layer.
    delete_button : QtModePushButton
        Button to delete Gaussians from layer.
    """

    layer: 'napari.layers.Gaussians'
    MODE = Mode
    PAN_ZOOM_ACTION_NAME = 'activate_gaussians_pan_zoom_mode'
    TRANSFORM_ACTION_NAME = 'activate_gaussians_transform_mode'

    def __init__(self, layer) -> None:
        super().__init__(layer)

        # Setup buttons
        self.select_button = self._radio_button(
            layer,
            'select_gaussians',
            Mode.SELECT,
            True,
            'activate_gaussians_select_mode',
        )
        self.delete_button = QtModePushButton(
            layer,
            'delete_shape',
        )
        action_manager.bind_button(
            'napari:delete_selected_gaussians', self.delete_button
        )
        self._EDIT_BUTTONS += (self.delete_button,)
        self._on_editable_or_visible_change()

        self.button_grid.addWidget(self.delete_button, 0, 3)
        self.button_grid.addWidget(self.select_button, 0, 4)

        # Setup widget controls
        self._projection_mode_control = QtProjectionModeControl(self, layer)
        self._add_widget_controls(self._projection_mode_control)

        # Point size slider (using simple Qt slider for now)
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget

        point_size_widget = QWidget()
        point_size_layout = QHBoxLayout()
        point_size_layout.setContentsMargins(0, 0, 0, 0)
        point_size_label = QLabel(trans._('point size:'))
        self.point_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.point_size_slider.setMinimum(10)  # 0.1 * 100
        self.point_size_slider.setMaximum(500)  # 5.0 * 100
        self.point_size_slider.setValue(int(layer.point_size * 100))
        self.point_size_slider.setSingleStep(5)
        self.point_size_slider.valueChanged.connect(self._on_point_size_change)
        point_size_layout.addWidget(point_size_label)
        point_size_layout.addWidget(self.point_size_slider)
        point_size_widget.setLayout(point_size_layout)
        self.layout().addWidget(point_size_widget)

        # SH degree slider
        sh_degree_widget = QWidget()
        sh_degree_layout = QHBoxLayout()
        sh_degree_layout.setContentsMargins(0, 0, 0, 0)
        sh_degree_label = QLabel(trans._('SH degree:'))
        self.sh_degree_slider = QSlider(Qt.Orientation.Horizontal)
        self.sh_degree_slider.setMinimum(0)
        self.sh_degree_slider.setMaximum(3)
        self.sh_degree_slider.setValue(layer.sh_degree)
        self.sh_degree_slider.setSingleStep(1)
        self.sh_degree_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.sh_degree_slider.setTickInterval(1)
        self.sh_degree_slider.valueChanged.connect(self._on_sh_degree_change)
        sh_degree_layout.addWidget(sh_degree_label)
        sh_degree_layout.addWidget(self.sh_degree_slider)
        sh_degree_widget.setLayout(sh_degree_layout)
        self.layout().addWidget(sh_degree_widget)

        # Color mode combo box
        from qtpy.QtWidgets import QComboBox

        color_mode_widget = QWidget()
        color_mode_layout = QHBoxLayout()
        color_mode_layout.setContentsMargins(0, 0, 0, 0)
        color_mode_label = QLabel(trans._('color mode:'))
        self.color_mode_combo = QComboBox()
        self.color_mode_combo.addItems(['direct', 'cycle', 'colormap', 'spherical_harmonics'])
        self.color_mode_combo.setCurrentText(str(layer.color_mode))
        self.color_mode_combo.currentTextChanged.connect(self._on_color_mode_change)
        color_mode_layout.addWidget(color_mode_label)
        color_mode_layout.addWidget(self.color_mode_combo)
        color_mode_widget.setLayout(color_mode_layout)
        self.layout().addWidget(color_mode_widget)

        self._out_slice_checkbox_control = QtOutSliceCheckBoxControl(
            self, layer
        )
        self._add_widget_controls(self._out_slice_checkbox_control)

        # Connect layer events
        layer.events.point_size.connect(self._on_layer_point_size_change)
        layer.events.sh_degree.connect(self._on_layer_sh_degree_change)
        layer.events.color_mode.connect(self._on_layer_color_mode_change)

    def _on_point_size_change(self, value):
        """Update layer point size from slider."""
        with self.layer.events.point_size.blocker():
            self.layer.point_size = value / 100.0

    def _on_sh_degree_change(self, value):
        """Update layer SH degree from slider."""
        with self.layer.events.sh_degree.blocker():
            self.layer.sh_degree = value

    def _on_color_mode_change(self, text):
        """Update layer color mode from combo box."""
        with self.layer.events.color_mode.blocker():
            self.layer.color_mode = text

    def _on_layer_point_size_change(self):
        """Update slider when layer point size changes."""
        with self.layer.events.point_size.blocker():
            self.point_size_slider.setValue(int(self.layer.point_size * 100))

    def _on_layer_sh_degree_change(self):
        """Update slider when layer SH degree changes."""
        with self.layer.events.sh_degree.blocker():
            self.sh_degree_slider.setValue(self.layer.sh_degree)

    def _on_layer_color_mode_change(self):
        """Update combo box when layer color mode changes."""
        with self.layer.events.color_mode.blocker():
            self.color_mode_combo.setCurrentText(str(self.layer.color_mode))

    def _on_mode_change(self, event):
        """
        Update controls when layer mode changes.

        Available modes:
        * SELECT
        * PAN_ZOOM
        * TRANSFORM
        """
        super()._on_mode_change(event)

    def _on_ndisplay_changed(self):
        """Update when display dimensionality changes."""
        # Gaussians layer is primarily for 3D viewing
        self.layer.editable = self.ndisplay == 3
        super()._on_ndisplay_changed()

    def close(self):
        """Disconnect events when closing."""
        self.layer.events.point_size.disconnect(self._on_layer_point_size_change)
        self.layer.events.sh_degree.disconnect(self._on_layer_sh_degree_change)
        self.layer.events.color_mode.disconnect(self._on_layer_color_mode_change)
        super().close()
