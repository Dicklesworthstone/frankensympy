"""Heaviside step and Dirac delta functions (native evaluation rules)."""

from __future__ import annotations

from ...core import _NativeFunction


class Heaviside(_NativeFunction):
    """Heaviside step: 0 for negative, 1 for positive, 1/2 at 0 (upstream H0)."""

    __slots__ = ()

    @classmethod
    def _eval_known_sign(cls, arg, H0=None):
        """Use definite shell assumptions after closed native admission.

        The inherited eval dispatch calls this protected built-in helper;
        opaque arguments keep the ordinary retained Function path.
        """
        from ...core import S

        if arg.is_positive is True:
            return S.One
        if arg.is_negative is True:
            return S.Zero
        if arg.is_zero is True:
            return S.Half if H0 is None else H0
        return None

    @property
    def pargs(self) -> tuple:
        return self.args[:1]


class DiracDelta(_NativeFunction):
    """Dirac delta: 0 away from the origin; held at symbolic arguments."""

    __slots__ = ()


__all__ = ["DiracDelta", "Heaviside"]
