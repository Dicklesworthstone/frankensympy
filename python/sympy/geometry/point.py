"""Geometric points for FrankenSymPy."""

from __future__ import annotations

from typing import Any

from ..core import Basic, Expr, _native, _native_expr, _wrap, sqrt


class Point(Basic):
    """A geometric point in 2D or 3D Euclidean space."""

    def __new__(cls, *args: Any):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = tuple(args[0])
        if len(args) == 2:
            return Point2D(*args)
        if len(args) == 3:
            return Point3D(*args)
        raise ValueError(f"Point requires 2 or 3 coordinates, got {len(args)}")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class Point2D(Point):
    """A 2D point."""

    __slots__ = ("_native_pt",)

    def __new__(cls, x: Any, y: Any):
        obj = object.__new__(cls)
        obj._native_pt = _native.Point2D(_native_expr(x), _native_expr(y))
        return obj

    @property
    def x(self) -> Expr:
        return _wrap(self._native_pt.x)

    @property
    def y(self) -> Expr:
        return _wrap(self._native_pt.y)

    @property
    def args(self) -> tuple[Expr, Expr]:
        return (self.x, self.y)

    def distance(self, other: Any) -> Expr:
        other_pt = Point(other) if not isinstance(other, Point) else other
        if not isinstance(other_pt, Point2D):
            raise TypeError("distance requires a 2D Point")
        d2 = _wrap(self._native_pt.distance_squared(other_pt._native_pt))
        return sqrt(d2)

    def midpoint(self, other: Any) -> Point2D:
        other_pt = Point(other) if not isinstance(other, Point) else other
        if not isinstance(other_pt, Point2D):
            raise TypeError("midpoint requires a 2D Point")
        mid = self._native_pt.midpoint(other_pt._native_pt)
        return Point2D(_wrap(mid.x), _wrap(mid.y))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Point2D):
            return self._native_pt == other._native_pt
        if isinstance(other, (list, tuple)) and len(other) == 2:
            return self.args == (other[0], other[1])
        return False

    def __hash__(self) -> int:
        return hash((Point2D, self.x, self.y))

    def __getitem__(self, idx: int) -> Expr:
        return self.args[idx]

    def __len__(self) -> int:
        return 2

    def __repr__(self) -> str:
        return f"Point2D({self.x}, {self.y})"

    def __str__(self) -> str:
        return f"Point2D({self.x}, {self.y})"


class Point3D(Point):
    """A 3D point."""

    __slots__ = ("_native_pt",)

    def __new__(cls, x: Any, y: Any, z: Any):
        obj = object.__new__(cls)
        obj._native_pt = _native.Point3D(_native_expr(x), _native_expr(y), _native_expr(z))
        return obj

    @property
    def x(self) -> Expr:
        return _wrap(self._native_pt.x)

    @property
    def y(self) -> Expr:
        return _wrap(self._native_pt.y)

    @property
    def z(self) -> Expr:
        return _wrap(self._native_pt.z)

    @property
    def args(self) -> tuple[Expr, Expr, Expr]:
        return (self.x, self.y, self.z)

    def distance(self, other: Any) -> Expr:
        other_pt = Point(other) if not isinstance(other, Point) else other
        if not isinstance(other_pt, Point3D):
            raise TypeError("distance requires a 3D Point")
        d2 = _wrap(self._native_pt.distance_squared(other_pt._native_pt))
        return sqrt(d2)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Point3D):
            return self._native_pt == other._native_pt
        if isinstance(other, (list, tuple)) and len(other) == 3:
            return self.args == (other[0], other[1], other[2])
        return False

    def __hash__(self) -> int:
        return hash((Point3D, self.x, self.y, self.z))

    def __getitem__(self, idx: int) -> Expr:
        return self.args[idx]

    def __len__(self) -> int:
        return 3

    def __repr__(self) -> str:
        return f"Point3D({self.x}, {self.y}, {self.z})"

    def __str__(self) -> str:
        return f"Point3D({self.x}, {self.y}, {self.z})"


__all__ = [
    "Point",
    "Point2D",
    "Point3D",
]
