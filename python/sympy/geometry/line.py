"""Lines, segments, and rays for FrankenSymPy geometry."""

from __future__ import annotations

from typing import Any

from ..core import Basic, Expr, _native, _wrap, simplify
from .point import Point, Point2D


class LinearEntity(Basic):
    """Base class for lines, rays, segments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

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
    def args(self) -> tuple[Point, Point]:
        return (self.p1, self.p2)

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return False
        return self.args == other.args

    def __hash__(self) -> int:
        return hash((type(self), self.args))

    def is_parallel(self, other: Any) -> bool:
        if not isinstance(other, LinearEntity):
            raise TypeError("is_parallel requires a LinearEntity")
        v1 = self.p2 - self.p1
        v2 = other.p2 - other.p1
        cross_x = simplify(v1.y * getattr(v2, "z", 0) - getattr(v1, "z", 0) * v2.y)
        cross_y = simplify(getattr(v1, "z", 0) * v2.x - v1.x * getattr(v2, "z", 0))
        cross_z = simplify(v1.x * v2.y - v1.y * v2.x)
        return cross_x == 0 and cross_y == 0 and cross_z == 0

    def is_perpendicular(self, other: Any) -> bool:
        if not isinstance(other, LinearEntity):
            raise TypeError("is_perpendicular requires a LinearEntity")
        v1 = self.p2 - self.p1
        v2 = other.p2 - other.p1
        return simplify(v1.dot(v2)) == 0

    def is_collinear(self, other: Any) -> bool:
        from .util import are_collinear
        if isinstance(other, LinearEntity):
            return are_collinear(self.p1, self.p2, other.p1, other.p2)
        if isinstance(other, Point):
            return are_collinear(self.p1, self.p2, other)
        return False

    def intersection(self, other: Any) -> list[Any]:
        if not isinstance(other, LinearEntity):
            if hasattr(other, "intersection"):
                return other.intersection(self)
            raise TypeError(f"intersection with {type(other).__name__} is not implemented")

        if len(self.p1) != len(other.p1):
            raise ValueError("Entities must have the same dimension")

        from .util import are_collinear

        if len(self.p1) == 2:
            try:
                line_self = _native.Line2D(self.p1._native_pt, self.p2._native_pt)
                line_other = _native.Line2D(other.p1._native_pt, other.p2._native_pt)
                pt_native = line_self.intersection(line_other)
                pt = Point2D(_wrap(pt_native.x), _wrap(pt_native.y))
                if self.contains(pt) and other.contains(pt):
                    return [pt]
                return []
            except Exception:
                pass

            if not are_collinear(self.p1, self.p2, other.p1, other.p2):
                return []

            if isinstance(self, Line) and isinstance(other, Line):
                return [self]
            if isinstance(self, Line):
                return [other]
            if isinstance(other, Line):
                return [self]

            candidates = [
                p for p in (self.p1, self.p2, getattr(other, "p1", None), getattr(other, "p2", None))
                if p is not None and self.contains(p) and other.contains(p)
            ]
            unique: list[Point] = []
            for p in candidates:
                if not any(p == u for u in unique):
                    unique.append(p)
            if len(unique) >= 2:
                return [Segment(unique[0], unique[1])]
            elif len(unique) == 1:
                return [unique[0]]
            return []

        from .util import are_coplanar
        if not are_coplanar(self.p1, self.p2, other.p1, other.p2):
            return []

        if are_collinear(self.p1, self.p2, other.p1, other.p2):
            if isinstance(self, Line) and isinstance(other, Line):
                return [self]
            if isinstance(self, Line):
                return [other]
            if isinstance(other, Line):
                return [self]
            candidates = [
                p for p in (self.p1, self.p2, getattr(other, "p1", None), getattr(other, "p2", None))
                if p is not None and self.contains(p) and other.contains(p)
            ]
            unique = []
            for p in candidates:
                if not any(p == u for u in unique):
                    unique.append(p)
            if len(unique) >= 2:
                return [Segment(unique[0], unique[1])]
            elif len(unique) == 1:
                return [unique[0]]
            return []

        v1 = self.p2 - self.p1
        v2 = other.p2 - other.p1
        cross_x = simplify(v1.y * v2.z - v1.z * v2.y)
        cross_y = simplify(v1.z * v2.x - v1.x * v2.z)
        cross_z = simplify(v1.x * v2.y - v1.y * v2.x)
        if cross_x == 0 and cross_y == 0 and cross_z == 0:
            return []

        diff = other.p1 - self.p1
        if cross_z != 0:
            t_num = diff.x * (-v2.y) - diff.y * (-v2.x)
            t = simplify(-t_num / cross_z)
        elif cross_y != 0:
            t_num = diff.x * (-v2.z) - diff.z * (-v2.x)
            t = simplify(t_num / cross_y)
        else:
            t_num = diff.y * (-v2.z) - diff.z * (-v2.y)
            t = simplify(-t_num / cross_x)

        pt = self.p1 + v1 * t
        if self.contains(pt) and other.contains(pt):
            return [pt]
        return []


class Segment(LinearEntity):
    """A directed line segment between two points."""

    __slots__ = ("_p1", "_p2")

    def __new__(cls, p1: Any, p2: Any):
        pt1 = Point(p1) if not isinstance(p1, Point) else p1
        pt2 = Point(p2) if not isinstance(p2, Point) else p2
        if isinstance(pt1, Point2D) and isinstance(pt2, Point2D):
            return Segment2D(pt1, pt2)
        from .point import Point3D
        if isinstance(pt1, Point3D) and isinstance(pt2, Point3D):
            return Segment3D(pt1, pt2)
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

    def contains(self, other: Any) -> bool:
        from .util import are_collinear
        if isinstance(other, Point):
            if not are_collinear(self.p1, self.p2, other):
                return False
            dot = (other - self.p1).dot(self.p2 - other)
            return dot >= 0
        if isinstance(other, Segment):
            return self.contains(other.p1) and self.contains(other.p2)
        return False

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


class Segment3D(Segment):
    """A 3D segment."""

    __slots__ = ("_native_seg",)

    def __new__(cls, p1: Any, p2: Any):
        from .point import Point3D
        pt1 = Point(p1) if not isinstance(p1, Point) else p1
        pt2 = Point(p2) if not isinstance(p2, Point) else p2
        if not (isinstance(pt1, Point3D) and isinstance(pt2, Point3D)):
            raise TypeError("Segment3D requires 3D points")
        obj = object.__new__(cls)
        obj._p1 = pt1
        obj._p2 = pt2
        obj._native_seg = _native.Segment3D(pt1._native_pt, pt2._native_pt)
        return obj

    @property
    def midpoint(self) -> Any:
        from .point import Point3D
        mid = self._native_seg.midpoint()
        return Point3D(_wrap(mid.x), _wrap(mid.y), _wrap(mid.z))

    @property
    def length(self) -> Expr:
        return self.p1.distance(self.p2)

    def __repr__(self) -> str:
        return f"Segment3D({self.p1}, {self.p2})"

    def __str__(self) -> str:
        return f"Segment3D({self.p1}, {self.p2})"


class Line(LinearEntity):
    """An infinite line passing through two points."""

    __slots__ = ("_p1", "_p2")

    def __new__(cls, p1: Any, p2: Any):
        pt1 = Point(p1) if not isinstance(p1, Point) else p1
        pt2 = Point(p2) if not isinstance(p2, Point) else p2
        if isinstance(pt1, Point2D) and isinstance(pt2, Point2D):
            return Line2D(pt1, pt2)
        from .point import Point3D
        if isinstance(pt1, Point3D) and isinstance(pt2, Point3D):
            return Line3D(pt1, pt2)
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

    def contains(self, other: Any) -> bool:
        from .util import are_collinear
        if isinstance(other, Point):
            return are_collinear(self.p1, self.p2, other)
        if isinstance(other, LinearEntity):
            return are_collinear(self.p1, self.p2, other.p1, other.p2)
        return False

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

    def intersection(self, other: Any) -> list[Any]:
        return super().intersection(other)

    def __repr__(self) -> str:
        return f"Line2D({self.p1}, {self.p2})"

    def __str__(self) -> str:
        return f"Line2D({self.p1}, {self.p2})"


class Line3D(Line):
    """A 3D line."""

    __slots__ = ("_native_line",)

    def __new__(cls, p1: Any, p2: Any):
        from .point import Point3D
        pt1 = Point(p1) if not isinstance(p1, Point) else p1
        pt2 = Point(p2) if not isinstance(p2, Point) else p2
        if not (isinstance(pt1, Point3D) and isinstance(pt2, Point3D)):
            raise TypeError("Line3D requires 3D points")
        obj = object.__new__(cls)
        obj._p1 = pt1
        obj._p2 = pt2
        obj._native_line = _native.Line3D(pt1._native_pt, pt2._native_pt)
        return obj

    @property
    def direction(self) -> Any:
        from .point import Point3D
        d = self._native_line.direction()
        return Point3D(_wrap(d.x), _wrap(d.y), _wrap(d.z))

    def __repr__(self) -> str:
        return f"Line3D({self.p1}, {self.p2})"

    def __str__(self) -> str:
        return f"Line3D({self.p1}, {self.p2})"


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

    def contains(self, other: Any) -> bool:
        from .util import are_collinear
        if isinstance(other, Point):
            if not are_collinear(self.p1, self.p2, other):
                return False
            dot = (other - self.p1).dot(self.p2 - self.p1)
            return dot >= 0
        if isinstance(other, Ray):
            return self.contains(other.p1) and (other.p2 - other.p1).dot(self.p2 - self.p1) > 0
        if isinstance(other, Segment):
            return self.contains(other.p1) and self.contains(other.p2)
        return False

    def __repr__(self) -> str:
        return f"Ray({self.p1}, {self.p2})"

    def __str__(self) -> str:
        return f"Ray({self.p1}, {self.p2})"


Ray2D = Ray
Ray3D = Ray


__all__ = [
    "Line",
    "Line2D",
    "Line3D",
    "LinearEntity",
    "Ray",
    "Ray2D",
    "Ray3D",
    "Segment",
    "Segment2D",
    "Segment3D",
]
