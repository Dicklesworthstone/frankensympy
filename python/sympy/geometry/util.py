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
    if len(points) == 1 and isinstance(points[0], (list, tuple, set)):
        points = tuple(points[0])
    pts = [Point(p) if not isinstance(p, Point) else p for p in points]
    if len(pts) <= 2:
        return True
    # Find the first pair of distinct points
    p0 = pts[0]
    p1 = None
    for p in pts[1:]:
        if p != p0:
            p1 = p
            break
    if p1 is None:
        # All points are coincident
        return True

    v1_x = p1.x - p0.x
    v1_y = p1.y - p0.y
    v1_z = getattr(p1, "z", Rational(0)) - getattr(p0, "z", Rational(0))

    for pk in pts:
        if pk == p0 or pk == p1:
            continue
        vk_x = pk.x - p0.x
        vk_y = pk.y - p0.y
        vk_z = getattr(pk, "z", Rational(0)) - getattr(p0, "z", Rational(0))

        cross_x = simplify(v1_y * vk_z - v1_z * vk_y)
        cross_y = simplify(v1_z * vk_x - v1_x * vk_z)
        cross_z = simplify(v1_x * vk_y - v1_y * vk_x)
        if cross_x != 0 or cross_y != 0 or cross_z != 0:
            return False
    return True


def are_coplanar(*points: Any) -> bool:
    """Return True if all points are coplanar."""
    if len(points) == 1 and isinstance(points[0], (list, tuple, set)):
        points = tuple(points[0])
    pts = [Point(p) if not isinstance(p, Point) else p for p in points]
    if len(pts) <= 3:
        return True
    if not any(isinstance(p, Point3D) for p in pts):
        # All 2D points lie in the z = 0 plane.
        return True

    # Find the first pair of distinct points
    p0 = pts[0]
    p1 = None
    for p in pts[1:]:
        if p != p0:
            p1 = p
            break
    if p1 is None:
        # All points are coincident
        return True

    v1_x = p1.x - p0.x
    v1_y = p1.y - p0.y
    v1_z = getattr(p1, "z", Rational(0)) - getattr(p0, "z", Rational(0))

    # Find a third point not collinear with p0 and p1
    p2 = None
    nx = ny = nz = Rational(0)
    for p in pts:
        if p == p0 or p == p1:
            continue
        vk_x = p.x - p0.x
        vk_y = p.y - p0.y
        vk_z = getattr(p, "z", Rational(0)) - getattr(p0, "z", Rational(0))

        cx = simplify(v1_y * vk_z - v1_z * vk_y)
        cy = simplify(v1_z * vk_x - v1_x * vk_z)
        cz = simplify(v1_x * vk_y - v1_y * vk_x)
        if cx != 0 or cy != 0 or cz != 0:
            p2 = p
            nx, ny, nz = cx, cy, cz
            break

    if p2 is None:
        # All points are collinear, hence coplanar
        return True

    # For every other point, test scalar triple product: (pk - p0) . normal == 0
    for pk in pts:
        if pk == p0 or pk == p1 or pk == p2:
            continue
        vk_x = pk.x - p0.x
        vk_y = pk.y - p0.y
        vk_z = getattr(pk, "z", Rational(0)) - getattr(p0, "z", Rational(0))

        dot = simplify(vk_x * nx + vk_y * ny + vk_z * nz)
        if dot != 0:
            return False
    return True


def centroid(*entities: Any) -> Any:
    """Compute the centroid of geometric entities or points."""
    if len(entities) == 1:
        ent = entities[0]
        if hasattr(ent, "centroid"):
            return ent.centroid
        if isinstance(ent, (list, tuple, set)):
            entities = tuple(ent)
            if len(entities) == 1 and hasattr(entities[0], "centroid"):
                return entities[0].centroid
    pts = []
    for ent in entities:
        if hasattr(ent, "centroid"):
            pts.append(ent.centroid)
        elif isinstance(ent, Point):
            pts.append(ent)
        else:
            pts.append(Point(ent))
    if not pts:
        raise ValueError("centroid requires at least one entity or point")
    n = len(pts)
    sum_x = sum((p.x for p in pts), start=Rational(0))
    sum_y = sum((p.y for p in pts), start=Rational(0))
    if any(isinstance(p, Point3D) for p in pts):
        sum_z = sum((getattr(p, "z", Rational(0)) for p in pts), start=Rational(0))
        return Point3D(simplify(sum_x / n), simplify(sum_y / n), simplify(sum_z / n))
    return Point2D(simplify(sum_x / n), simplify(sum_y / n))


__all__ = [
    "are_collinear",
    "are_coplanar",
    "centroid",
    "intersection",
]
