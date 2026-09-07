"""Polygons and triangles for FrankenSymPy geometry."""

from __future__ import annotations

from typing import Any

from ..core import Basic, Expr, Rational, _native, _wrap, simplify
from .point import Point, Point2D


class Polygon(Basic):
    """A 2D polygon defined by vertices in order."""

    __slots__ = ("_vertices", "_native_poly")

    def __new__(cls, *args: Any):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = tuple(args[0])
        if len(args) == 3 and cls is not Polygon:
            return Triangle(*args)
        points: list[Point2D] = []
        for a in args:
            pt = Point(a) if not isinstance(a, Point) else a
            if not isinstance(pt, Point2D):
                raise TypeError("Polygon vertices must be 2D points")
            points.append(pt)

        obj = object.__new__(cls)
        obj._vertices = tuple(points)
        obj._native_poly = _native.Polygon2D([p._native_pt for p in points])
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def vertices(self) -> tuple[Point2D, ...]:
        return self._vertices

    @property
    def area(self) -> Expr:
        d_area = _wrap(self._native_poly.double_signed_area())
        return simplify(abs(d_area) * Rational(1, 2))

    @property
    def centroid(self) -> Point2D:
        c = self._native_poly.centroid()
        return Point2D(_wrap(c.x), _wrap(c.y))

    def is_convex(self) -> bool | None:
        return self._native_poly.is_convex()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(repr(v) for v in self.vertices)})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(str(v) for v in self.vertices)})"


class Triangle(Polygon):
    """A 2D triangle."""

    __slots__ = ("_native_tri",)

    def __new__(cls, *args: Any):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = tuple(args[0])
        if len(args) != 3:
            raise ValueError(f"Triangle requires 3 vertices, got {len(args)}")
        points: list[Point2D] = []
        for a in args:
            pt = Point(a) if not isinstance(a, Point) else a
            if not isinstance(pt, Point2D):
                raise TypeError("Triangle vertices must be 2D points")
            points.append(pt)

        obj = object.__new__(cls)
        obj._vertices = tuple(points)
        obj._native_tri = _native.Triangle2D(
            points[0]._native_pt, points[1]._native_pt, points[2]._native_pt
        )
        obj._native_poly = _native.Polygon2D([p._native_pt for p in points])
        return obj

    @property
    def centroid(self) -> Point2D:
        c = self._native_tri.centroid()
        return Point2D(_wrap(c.x), _wrap(c.y))

    @property
    def area(self) -> Expr:
        d_area = _wrap(self._native_tri.double_signed_area())
        return simplify(abs(d_area) * Rational(1, 2))

    def is_right(self) -> bool | None:
        return self._native_tri.is_right()

    def is_isosceles(self) -> bool | None:
        return self._native_tri.is_isosceles()

    def is_equilateral(self) -> bool | None:
        return self._native_tri.is_equilateral()

    def is_collinear(self) -> bool | None:
        return self._native_tri.is_collinear()


__all__ = [
    "Polygon",
    "Triangle",
]
