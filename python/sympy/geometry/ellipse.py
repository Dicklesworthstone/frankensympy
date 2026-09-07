"""Ellipses and circles for FrankenSymPy geometry."""

from __future__ import annotations

from typing import Any

from ..core import Basic, Expr, _native, _native_expr, _wrap, simplify
from .point import Point, Point2D


class Ellipse(Basic):
    """Base class for ellipses and circles."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class Circle(Ellipse):
    """A 2D geometric circle."""

    __slots__ = ("_native_circle", "_center", "_radius")

    def __new__(cls, center: Any, radius: Any):
        c_pt = Point(center) if not isinstance(center, Point) else center
        if not isinstance(c_pt, Point2D):
            raise TypeError("Circle center must be a 2D Point")
        obj = object.__new__(cls)
        obj._center = c_pt
        obj._radius = _wrap(_native_expr(radius))
        obj._native_circle = _native.Circle(c_pt._native_pt, _native_expr(radius))
        return obj

    @property
    def center(self) -> Point2D:
        return self._center

    @property
    def radius(self) -> Expr:
        return self._radius

    @property
    def hradius(self) -> Expr:
        return self._radius

    @property
    def vradius(self) -> Expr:
        return self._radius

    @property
    def area(self) -> Expr:
        return _wrap(self._native_circle.area())

    @property
    def circumference(self) -> Expr:
        return _wrap(self._native_circle.circumference())

    def contains(self, other: Any) -> bool:
        if isinstance(other, Point):
            d2 = self.center.distance(other) ** 2
            r2 = self.radius ** 2
            return simplify(d2 - r2) == 0
        return False

    def __repr__(self) -> str:
        return f"Circle({self.center}, {self.radius})"

    def __str__(self) -> str:
        return f"Circle({self.center}, {self.radius})"


class Sphere(Basic):
    """A 3D geometric sphere."""

    __slots__ = ("_center", "_native_sphere", "_radius")

    def __new__(cls, center: Any, radius: Any):
        from .point import Point3D
        c_pt = Point(center) if not isinstance(center, Point) else center
        if not isinstance(c_pt, Point3D):
            raise TypeError("Sphere center must be a 3D Point")
        obj = object.__new__(cls)
        obj._center = c_pt
        obj._radius = _wrap(_native_expr(radius))
        obj._native_sphere = _native.Sphere(c_pt._native_pt, _native_expr(radius))
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def center(self) -> Any:
        return self._center

    @property
    def radius(self) -> Expr:
        return self._radius

    @property
    def volume(self) -> Expr:
        return _wrap(self._native_sphere.volume())

    @property
    def surface_area(self) -> Expr:
        return _wrap(self._native_sphere.surface_area())

    @property
    def area(self) -> Expr:
        return self.surface_area

    def contains(self, other: Any) -> bool:
        if isinstance(other, Point):
            d2 = self.center.distance(other) ** 2
            r2 = self.radius ** 2
            return simplify(d2 - r2) == 0
        return False

    def __repr__(self) -> str:
        return f"Sphere({self.center}, {self.radius})"

    def __str__(self) -> str:
        return f"Sphere({self.center}, {self.radius})"


__all__ = [
    "Circle",
    "Ellipse",
    "Sphere",
]
