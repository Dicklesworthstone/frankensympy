"""Mod (upstream sympy.core.mod)."""

from __future__ import annotations

from . import _NativeFunction


class Mod(_NativeFunction):
    """``Mod(p, q)``: ``p - q*floor(p/q)``; exact numbers evaluate natively
    (also after substitution), ``Mod(Mod(x, q), q)`` collapses."""

    __slots__ = ()

    @classmethod
    def eval(cls, p, q):
        if type(p) is Mod and p.args[1] == q:
            return p
        return super().eval(p, q)


__all__ = ["Mod"]
