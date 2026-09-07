"""Lines, segments, and rays for FrankenSymPy geometry."""

from __future__ import annotations

from typing import Any

from ..core import Basic, Expr, _native, _wrap
from .point import Point, Point2D


class LinearEntity(Basic):
    """Base class for lines, rays, segments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class Segment(LinearEntity):
    """A directed line segment between two points."""

    __slots__ = ("_p1", "_p2")

    def __new__(cls, p1: Any, p2: Any):
        pt1 = Point(p1) if not isinstance(p1, Point) else p1
        pt2 = Point(p2) if not isinstance(p2, Point) else p2
        if isinstance(pt1, Point2D) and isinstance(pt2, Point2D):
            return Segment2D(pt1, pt2)
        obj = object.__new__(cls)
        obj._p1 = pt1
        obj._p2 = pt2
        return obj

    @property
    def p1(self) -> Point:
        return self._p1

    @property
    def p2(self) -> Point:
        return self._p2

    @property
    def points(self) -> tuple[Point, Point]:
        return (self.p1, self.p2)

    @property
    def length(self) -> Expr:
        return self.p1.distance(self.p2)

    @property
    def midpoint(self) -> Point:
        return self.p1.midpoint(self.p2)

    def __repr__(self) -> str:
        return f"Segment({self.p1}, {self.p2})"

    def __str__(self) -> str:
        return f"Segment({self.p1}, {self.p2})"


class Segment2D(Segment):
    """A 2D segment."""

    __slots__ = ("_native_seg",)

    def __new__(cls, p1: Any, p2: Any):
        pt1 = Point(p1) if not isinstance(p1, Point) else p1
        pt2 = Point(p2) if not isinstance(p2, Point) else p2
        if not (isinstance(pt1, Point2D) and isinstance(pt2, Point2D)):
            raise TypeError("Segment2D requires 2D points")
        obj = object.__new__(cls)
        obj._p1 = pt1
        obj._p2 = pt2
        obj._native_seg = _native.Segment2D(pt1._native_pt, pt2._native_pt)
        return obj

    @property
    def midpoint(self) -> Point2D:
        mid = self._native_seg.midpoint()
        return Point2D(_wrap(mid.x), _wrap(mid.y))

    @property
    def length(self) -> Expr:
        return self.p1.distance(self.p2)

    def __repr__(self) -> str:
        return f"Segment2D({self.p1}, {self.p2})"

    def __str__(self) -> str:
        return f"Segment2D({self.p1}, {self.p2})"


class Line(LinearEntity):
    """An infinite line passing through two points."""

    __slots__ = ("_p1", "_p2")

    def __new__(cls, p1: Any, p2: Any):
        pt1 = Point(p1) if not isinstance(p1, Point) else p1
        pt2 = Point(p2) if not isinstance(p2, Point) else p2
        if isinstance(pt1, Point2D) and isinstance(pt2, Point2D):
            return Line2D(pt1, pt2)
        obj = object.__new__(cls)
        obj._p1 = pt1
        obj._p2 = pt2
        return obj

    @property
    def p1(self) -> Point:
        return self._p1

    @property
    def p2(self) -> Point:
        return self._p2

    @property
    def points(self) -> tuple[Point, Point]:
        return (self.p1, self.p2)

    def __repr__(self) -> str:
        return f"Line({self.p1}, {self.p2})"

    def __str__(self) -> str:
        return f"Line({self.p1}, {self.p2})"


class Line2D(Line):
    """A 2D line."""

    __slots__ = ("_native_line",)

    def __new__(cls, p1: Any, p2: Any):
        pt1 = Point(p1) if not isinstance(p1, Point) else p1
        pt2 = Point(p2) if not isinstance(p2, Point) else p2
        if not (isinstance(pt1, Point2D) and isinstance(pt2, Point2D)):
            raise TypeError("Line2D requires 2D points")
        obj = object.__new__(cls)
        obj._p1 = pt1
        obj._p2 = pt2
        obj._native_line = _native.Line2D(pt1._native_pt, pt2._native_pt)
        return obj

    def intersection(self, other: Any) -> list[Point2D]:
        if isinstance(other, Line2D):
            pt = self._native_line.intersection(other._native_line)
            return [Point2D(_wrap(pt.x), _wrap(pt.y))]
        raise TypeError("intersection with non-Line2D is not implemented")

    def __repr__(self) -> str:
        return f"Line2D({self.p1}, {self.p2})"

    def __str__(self) -> str:
        return f"Line2D({self.p1}, {self.p2})"


class Ray(LinearEntity):
    """A ray starting at p1 and passing through p2."""

    __slots__ = ("_p1", "_p2")

    def __new__(cls, p1: Any, p2: Any):
        pt1 = Point(p1) if not isinstance(p1, Point) else p1
        pt2 = Point(p2) if not isinstance(p2, Point) else p2
        obj = object.__new__(cls)
        obj._p1 = pt1
        obj._p2 = pt2
        return obj

    @property
    def source(self) -> Point:
        return self._p1

    @property
    def p1(self) -> Point:
        return self._p1

    @property
    def p2(self) -> Point:
        return self._p2

    @property
    def points(self) -> tuple[Point, Point]:
        return (self.p1, self.p2)

    def __repr__(self) -> str:
        return f"Ray({self.p1}, {self.p2})"

    def __str__(self) -> str:
        return f"Ray({self.p1}, {self.p2})"


Ray2D = Ray


__all__ = [
    "Line",
    "Line2D",
    "LinearEntity",
    "Ray",
    "Ray2D",
    "Segment",
    "Segment2D",
]
