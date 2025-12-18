"""Shape fitting algorithms for geometric feature extraction."""

from nanorange.image_analyzer.geometry_fitter.fit_shapes.base_shape import (
    BaseShapeFitter,
    FitResult,
    ShapeParams
)
from nanorange.image_analyzer.geometry_fitter.fit_shapes.fit_circle import (
    CircleFitter,
    CircleParams,
    fit_circle,
    circle_fit_error,
    merge_circles
)
from nanorange.image_analyzer.geometry_fitter.fit_shapes.fit_ellipse import (
    EllipseFitter,
    EllipseParams,
    fit_ellipse,
    ellipse_fit_error,
    merge_ellipses
)

__all__ = [
    # Base
    "BaseShapeFitter",
    "FitResult",
    "ShapeParams",
    # Circle
    "CircleFitter",
    "CircleParams",
    "fit_circle",
    "circle_fit_error",
    "merge_circles",
    # Ellipse
    "EllipseFitter",
    "EllipseParams",
    "fit_ellipse",
    "ellipse_fit_error",
    "merge_ellipses",
]
