from __future__ import annotations

from typing import Any

from ...core import Abs, Function, _NativeFunction, _native, _native_expr, _wrap


class sign(_NativeFunction):
    """Complex sign; exact values and factor extraction are native,
    assumption-known signs fold."""

    __slots__ = ()

    @classmethod
    def eval(cls, arg):
        from ...core import Integer, Rational

        if not isinstance(arg, Rational):
            if arg.is_positive:
                return Integer(1)
            if arg.is_negative:
                return Integer(-1)
            if arg.is_zero:
                return Integer(0)
        return super().eval(arg)


def _ri(e: Any) -> Any:
    """(re, im) of ``e`` built from real parts, ``I``, sums, products,
    integer powers and ``exp`` of a complex argument; None when some piece
    is undecided."""
    from ...core import Add, I, Integer, Mul, Pow

    if getattr(e, "is_real", None) is True:
        return e, Integer(0)
    if e == I:
        return Integer(0), Integer(1)
    if isinstance(e, Add):
        r, i = Integer(0), Integer(0)
        for t in e.args:
            part = _ri(t)
            if part is None:
                return None
            r, i = r + part[0], i + part[1]
        return r, i
    if isinstance(e, Mul):
        r, i = Integer(1), Integer(0)
        for f in e.args:
            part = _ri(f)
            if part is None:
                return None
            a, b = part
            r, i = r * a - i * b, r * b + i * a
        return r, i
    if isinstance(e, Pow) and isinstance(e.args[1], Integer):
        part = _ri(e.args[0])
        if part is None:
            return None
        k = int(e.args[1])
        a, b = part
        if k < 0:
            d = a**2 + b**2
            a, b, k = a / d, -b / d, -k
        if k > 12:
            return None
        r, i = Integer(1), Integer(0)
        for _ in range(k):
            r, i = r * a - i * b, r * b + i * a
        return r, i
    if type(e).__name__ == "exp":
        part = _ri(e.args[0])
        if part is None:
            return None
        from .. import cos, exp, sin

        a, b = part
        if b == 0:
            return e, Integer(0)
        return exp(a) * cos(b), exp(a) * sin(b)
    return None


def _split_real_imag(arg: Any):
    """(re, im) of an expression decomposable by ``_ri``, expanded."""
    from ...core import expand

    part = _ri(expand(arg))
    if part is None:
        return None
    return expand(part[0]), expand(part[1])


class re(Function):
    """Real part of a complex expression."""

    @classmethod
    def eval(cls, arg: Any) -> Any:
        if arg is None:
            return None
        from ...core import S
        if getattr(arg, "is_real", None) is True:
            return arg
        split = _split_real_imag(arg)
        if split is not None:
            return split[0]
        if getattr(arg, "is_imaginary", None) is True:
            return S.Zero
        if hasattr(arg, "as_real_imag"):
            try:
                r, _ = arg.as_real_imag()
                if not isinstance(r, re):
                    return r
            except Exception:
                pass
        return None


class im(Function):
    """Imaginary part of a complex expression."""

    @classmethod
    def eval(cls, arg: Any) -> Any:
        if arg is None:
            return None
        from ...core import S
        if getattr(arg, "is_real", None) is True:
            return S.Zero
        split = _split_real_imag(arg)
        if split is not None:
            return split[1]
        if hasattr(arg, "as_real_imag"):
            try:
                _, i = arg.as_real_imag()
                if not isinstance(i, im):
                    return i
            except Exception:
                pass
        return None


class conjugate(Function):
    """Complex conjugate of an expression."""

    @classmethod
    def eval(cls, arg: Any) -> Any:
        if arg is None:
            return None
        if getattr(arg, "is_real", None) is True:
            return arg
        from ...core import I
        if arg == I:
            return -I
        if arg == -I:
            return I
        if hasattr(arg, "as_real_imag"):
            try:
                r, i = arg.as_real_imag()
                if not isinstance(r, re) and not isinstance(i, im):
                    return r - I * i
            except Exception:
                pass
        return None


class arg(Function):
    """Argument (phase angle in [-pi, pi]) of a complex expression."""

    @classmethod
    def eval(cls, arg: Any) -> Any:
        if arg is None:
            return None
        from ...core import I, Rational, S, pi
        if getattr(arg, "is_positive", None) is True:
            return S.Zero
        if getattr(arg, "is_negative", None) is True:
            return pi
        if arg == I:
            return pi / 2
        if arg == -I:
            return -pi / 2
        if hasattr(arg, "as_real_imag"):
            try:
                r, i = arg.as_real_imag()
                if not isinstance(r, re) and not isinstance(i, im):
                    if r == 0 and i > 0:
                        return pi / 2
                    if r == 0 and i < 0:
                        return -pi / 2
                    if r > 0 and i == 0:
                        return S.Zero
                    if r < 0 and i == 0:
                        return pi
                    if r > 0 and r == i:
                        return pi / 4
                    if r < 0 and -r == i:
                        return Rational(3, 4) * pi
                    if r < 0 and r == i:
                        return Rational(-3, 4) * pi
                    if r > 0 and -r == i:
                        return -pi / 4
            except Exception:
                pass
        return None


__all__ = [
    "Abs",
    "arg",
    "conjugate",
    "im",
    "re",
    "sign",
]
