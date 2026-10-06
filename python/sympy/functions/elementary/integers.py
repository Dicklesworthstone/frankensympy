"""Integer-valued elementary functions for FrankenSymPy."""

from ...core import _NativeFunction


class floor(_NativeFunction):
    __slots__ = ()


class ceiling(_NativeFunction):
    __slots__ = ()


__all__ = [
    "ceiling",
    "floor",
]
