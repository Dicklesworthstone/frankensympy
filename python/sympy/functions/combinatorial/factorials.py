"""Combinatorial factorial functions for FrankenSymPy."""

from ...core import _NativeFunction


class factorial(_NativeFunction):
    __slots__ = ()


class subfactorial(_NativeFunction):
    __slots__ = ()


class binomial(_NativeFunction):
    __slots__ = ()


__all__ = [
    "binomial",
    "factorial",
    "subfactorial",
]
