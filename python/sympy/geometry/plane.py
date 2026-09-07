"""Planes in 3D Euclidean space for FrankenSymPy geometry."""

from __future__ import annotations

from typing import Any

from ..core import Basic, Expr, _native, _wrap
from .line import Line3D
from .point import Point, Point3D


class Plane(Basic):
    """A plane in 3D Euclidean space."""

    __slots__ = ("_native_plane", "_normal", "_point")

    def __new__(cls, *args: Any):
        if len(args) == 2:
            p = Point(args[0]) if not isinstance(args[0], Point) else args[0]
            n = Point(args[1]) if not isinstance(args[1], Point) else args[1]
            if not (isinstance(p, Point3D) and isinstance(n, Point3D)):
                raise TypeError("Plane requires 3D Point and normal")
            obj = object.__new__(cls)
            obj._point = p
            obj._normal = n
            obj._native_plane = _native.Plane3D(p._native_pt, n._native_pt)
            return obj
        if len(args) == 3:
            p1 = Point(args[0]) if not isinstance(args[0], Point) else args[0]
            p2 = Point(args[1]) if not isinstance(args[1], Point) else args[1]
            p3 = Point(args[2]) if not isinstance(args[2], Point) else args[2]
            if not (isinstance(p1, Point3D) and isinstance(p2, Point3D) and isinstance(p3, Point3D)):
                raise TypeError("Plane from three points requires 3D Points")
            obj = object.__new__(cls)
            obj._point = p1
            obj._native_plane = _native.Plane3D.from_three_points(p1._native_pt, p2._native_pt, p3._native_pt)
            n = obj._native_plane.normal
            obj._normal = Point3D(_wrap(n.x), _wrap(n.y), _wrap(n.z))
            return obj
        raise ValueError(f"Plane requires 2 or 3 arguments, got {len(args)}")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def p1(self) -> Point3D:
        return self._point

    @property
    def point(self) -> Point3D:
        return self._point

    @property
    def normal_vector(self) -> Point3D:
        return self._normal

    @property
    def args(self) -> tuple[Point3D, Point3D]:
        return (self.p1, self.normal_vector)

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return False
        return self.args == other.args

    def __hash__(self) -> int:
        return hash((type(self), self.args))

    def eval_at_point(self, pt: Any) -> Expr:
        p = Point(pt) if not isinstance(pt, Point) else pt
        if not isinstance(p, Point3D):
            raise TypeError("eval_at_point requires a 3D Point")
        return _wrap(self._native_plane.eval_at_point(p._native_pt))

    def distance(self, pt: Any) -> Expr:
        from ..core import sqrt
        p = Point(pt) if not isinstance(pt, Point) else pt
        if not isinstance(p, Point3D):
            raise TypeError("distance requires a 3D Point")
        d2 = _wrap(self._native_plane.distance_squared(p._native_pt))
        return sqrt(d2)

    def is_parallel(self, other: Any) -> bool:
        if isinstance(other, Plane):
            return self._native_plane.is_parallel(other._native_plane)
        raise TypeError("is_parallel requires a Plane")

    def is_perpendicular(self, other: Any) -> bool:
        if isinstance(other, Plane):
            return self._native_plane.is_perpendicular(other._native_plane)
        raise TypeError("is_perpendicular requires a Plane")

    def contains(self, other: Any) -> bool:
        if isinstance(other, Point):
            return self.eval_at_point(other) == 0
        from .line import LinearEntity
        if isinstance(other, LinearEntity):
            return self.contains(other.p1) and self.contains(other.p2)
        if isinstance(other, Plane):
            return self == other
        return False

    def intersection(self, other: Any) -> list[Any]:
        if isinstance(other, Plane):
            l = self._native_plane.intersection_plane(other._native_plane)
            p1 = Point3D(_wrap(l.p1.x), _wrap(l.p1.y), _wrap(l.p1.z))
            p2 = Point3D(_wrap(l.p2.x), _wrap(l.p2.y), _wrap(l.p2.z))
            return [Line3D(p1, p2)]
        from .line import LinearEntity
        if isinstance(other, LinearEntity):
            line3d = Line3D(other.p1, other.p2)
            try:
                pt = self._native_plane.intersection_line(line3d._native_line)
                p = Point3D(_wrap(pt.x), _wrap(pt.y), _wrap(pt.z))
                if other.contains(p):
                    return [p]
                return []
            except Exception:
                if self.contains(other):
                    return [other]
                return []
        raise TypeError(f"intersection with {type(other).__name__} is not implemented")

    def __repr__(self) -> str:
        return f"Plane({self.point}, {self.normal_vector})"

    def __str__(self) -> str:
        return f"Plane({self.point}, {self.normal_vector})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Plane):
            return self._native_plane == other._native_plane
        return False


__all__ = [
    "Plane",
]
