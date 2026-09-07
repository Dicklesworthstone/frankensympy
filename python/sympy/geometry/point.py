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

    @property
    def coordinates(self) -> tuple[Expr, ...]:
        return self.args

    @property
    def dimension(self) -> int:
        return len(self)

    def is_collinear(self, *points: Any) -> bool:
        from .util import are_collinear
        return are_collinear(self, *points)

    @staticmethod
    def are_collinear(*points: Any) -> bool:
        from .util import are_collinear
        return are_collinear(*points)

    def is_coplanar(self, *points: Any) -> bool:
        from .util import are_coplanar
        return are_coplanar(self, *points)

    @staticmethod
    def are_coplanar(*points: Any) -> bool:
        from .util import are_coplanar
        return are_coplanar(*points)

    def __add__(self, other: Any) -> Point:
        if isinstance(other, (list, tuple)):
            other = Point(*other)
        elif not isinstance(other, Point):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("Points must have the same dimension")
        return Point(*(self[i] + other[i] for i in range(len(self))))

    def __radd__(self, other: Any) -> Point:
        return self.__add__(other)

    def __sub__(self, other: Any) -> Point:
        if isinstance(other, (list, tuple)):
            other = Point(*other)
        elif not isinstance(other, Point):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("Points must have the same dimension")
        return Point(*(self[i] - other[i] for i in range(len(self))))

    def __rsub__(self, other: Any) -> Point:
        if isinstance(other, (list, tuple)):
            other = Point(*other)
        elif not isinstance(other, Point):
            return NotImplemented
        if len(self) != len(other):
            raise ValueError("Points must have the same dimension")
        return Point(*(other[i] - self[i] for i in range(len(self))))

    def __neg__(self) -> Point:
        return Point(*(-self[i] for i in range(len(self))))

    def __mul__(self, factor: Any) -> Point:
        return Point(*(self[i] * factor for i in range(len(self))))

    def __rmul__(self, factor: Any) -> Point:
        return self.__mul__(factor)

    def __truediv__(self, divisor: Any) -> Point:
        return Point(*(self[i] / divisor for i in range(len(self))))

    def taxicab_distance(self, other: Any) -> Expr:
        from ..core import Abs, Rational
        other_pt = Point(other) if not isinstance(other, Point) else other
        if len(self) != len(other_pt):
            raise ValueError("taxicab_distance requires points of the same dimension")
        return sum((Abs(self[i] - other_pt[i]) for i in range(len(self))), start=Rational(0))


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

    @property
    def origin(self) -> Point2D:
        return Point2D(0, 0)

    def dot(self, other: Any) -> Expr:
        other_pt = Point(other) if not isinstance(other, Point) else other
        if not isinstance(other_pt, Point2D):
            raise TypeError("dot requires a 2D Point")
        return _wrap(self._native_pt.dot(other_pt._native_pt))

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

    @property
    def origin(self) -> Point3D:
        return Point3D(0, 0, 0)

    def dot(self, other: Any) -> Expr:
        other_pt = Point(other) if not isinstance(other, Point) else other
        if not isinstance(other_pt, Point3D):
            raise TypeError("dot requires a 3D Point")
        return _wrap(self._native_pt.dot(other_pt._native_pt))

    def midpoint(self, other: Any) -> Point3D:
        other_pt = Point(other) if not isinstance(other, Point) else other
        if not isinstance(other_pt, Point3D):
            raise TypeError("midpoint requires a 3D Point")
        mid = self._native_pt.midpoint(other_pt._native_pt)
        return Point3D(_wrap(mid.x), _wrap(mid.y), _wrap(mid.z))

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
