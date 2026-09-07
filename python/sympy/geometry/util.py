"""Geometry utility functions for FrankenSymPy."""

from typing import Any, List
from .point import Point, Point2D, Point3D
from .polygon import Triangle, Polygon
from ..core import Rational, simplify


def intersection(*entities: Any) -> List[Any]:
    """Find the intersection of geometric entities."""
    if len(entities) < 2:
        return []
    first = entities[0]
    res = first.intersection(entities[1])
    for entity in entities[2:]:
        new_res = []
        for item in res:
            if hasattr(entity, "contains") and entity.contains(item):
                new_res.append(item)
        res = new_res
    return res


def are_collinear(*points: Any) -> bool:
    """Return True if all points are collinear."""
    pts = [Point(p) if not isinstance(p, Point) else p for p in points]
    if len(pts) <= 2:
        return True
    p0, p1 = pts[0], pts[1]
    for pk in pts[2:]:
        if isinstance(p0, Point2D) and isinstance(p1, Point2D) and isinstance(pk, Point2D):
            tri = Triangle(p0, p1, pk)
            coll = tri.is_collinear()
            if coll is False:
                return False
        else:
            v1 = (p1.x - p0.x, p1.y - p0.y, getattr(p1, "z", 0) - getattr(p0, "z", 0))
            vk = (pk.x - p0.x, pk.y - p0.y, getattr(pk, "z", 0) - getattr(p0, "z", 0))
            cross_x = simplify(v1[1] * vk[2] - v1[2] * vk[1])
            cross_y = simplify(v1[2] * vk[0] - v1[0] * vk[2])
            cross_z = simplify(v1[0] * vk[1] - v1[1] * vk[0])
            if cross_x != 0 or cross_y != 0 or cross_z != 0:
                return False
    return True


def centroid(*entities: Any) -> Any:
    """Compute the centroid of geometric entities or points."""
    if len(entities) == 1:
        ent = entities[0]
        if hasattr(ent, "centroid"):
            return ent.centroid
        if isinstance(ent, (list, tuple)):
            entities = tuple(ent)
    pts = [Point(p) if not isinstance(p, Point) else p for p in entities]
    if not pts:
        raise ValueError("centroid requires at least one point")
    n = len(pts)
    sum_x = sum((p.x for p in pts), start=Rational(0))
    sum_y = sum((p.y for p in pts), start=Rational(0))
    if any(isinstance(p, Point3D) for p in pts):
        sum_z = sum((getattr(p, "z", 0) for p in pts), start=Rational(0))
        return Point3D(simplify(sum_x / n), simplify(sum_y / n), simplify(sum_z / n))
    return Point2D(simplify(sum_x / n), simplify(sum_y / n))


__all__ = [
    "are_collinear",
    "centroid",
    "intersection",
]
