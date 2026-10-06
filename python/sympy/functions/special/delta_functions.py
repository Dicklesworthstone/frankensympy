"""Heaviside step and Dirac delta functions (native evaluation rules)."""

from __future__ import annotations

from ...core import _NativeFunction


class Heaviside(_NativeFunction):
    """Heaviside step: 0 for negative, 1 for positive, 1/2 at 0 (upstream H0)."""

    __slots__ = ()

    @property
    def pargs(self) -> tuple:
        return self.args[:1]


class DiracDelta(_NativeFunction):
    """Dirac delta: 0 away from the origin; held at symbolic arguments."""

    __slots__ = ()


__all__ = ["DiracDelta", "Heaviside"]
