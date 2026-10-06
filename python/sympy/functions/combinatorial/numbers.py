"""Combinatorial number functions for FrankenSymPy."""

from ...core import _NativeFunction


class fibonacci(_NativeFunction):
    __slots__ = ()


class lucas(_NativeFunction):
    __slots__ = ()


class bernoulli(_NativeFunction):
    __slots__ = ()


class bell(_NativeFunction):
    __slots__ = ()


class harmonic(_NativeFunction):
    __slots__ = ()


class catalan(_NativeFunction):
    __slots__ = ()


__all__ = [
    "bell",
    "bernoulli",
    "catalan",
    "fibonacci",
    "harmonic",
    "lucas",
]
