"""
Shape plotter module for visualizing detected shapes on images.

Provides modular shape drawing capabilities using Plotly.
Includes automatic coordinate scaling for different image dimensions.
"""

from nanorange.image_analyzer.plotter.shape_plotter.base_drawer import BaseShapeDrawer, DrawStyle
from nanorange.image_analyzer.plotter.shape_plotter.circle_drawer import CircleDrawer
from nanorange.image_analyzer.plotter.shape_plotter.ellipse_drawer import EllipseDrawer
from nanorange.image_analyzer.plotter.shape_plotter.shape_plotter import (
    ShapePlotter,
    PlotterConfig,
    ScaleFactors,
    calculate_tile_offsets,
    plot_shapes
)

__all__ = [
    "BaseShapeDrawer",
    "DrawStyle",
    "CircleDrawer",
    "EllipseDrawer",
    "ShapePlotter",
    "PlotterConfig",
    "ScaleFactors",
    "calculate_tile_offsets",
    "plot_shapes",
]
