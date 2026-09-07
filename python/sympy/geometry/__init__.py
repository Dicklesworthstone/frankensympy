"""Symbolic 2D and 3D geometry submodule for FrankenSymPy."""

from __future__ import annotations

from .point import Point, Point2D, Point3D
from .line import (
    Line,
    Line2D,
    Line3D,
    LinearEntity,
    Ray,
    Ray2D,
    Ray3D,
    Segment,
    Segment2D,
    Segment3D,
)
from .ellipse import Circle, Ellipse, Sphere
from .polygon import Polygon, Triangle
from .plane import Plane
from .util import are_collinear, are_coplanar, centroid, intersection
from . import util

__all__ = [
    "Circle",
    "Ellipse",
    "Line",
    "Line2D",
    "Line3D",
    "LinearEntity",
    "Plane",
    "Point",
    "Point2D",
    "Point3D",
    "Polygon",
    "Ray",
    "Ray2D",
    "Ray3D",
    "Segment",
    "Segment2D",
    "Segment3D",
    "Sphere",
    "Triangle",
    "are_collinear",
    "are_coplanar",
    "centroid",
    "intersection",
    "util",
]
