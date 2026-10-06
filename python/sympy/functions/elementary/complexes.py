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


def _split_real_imag(arg: Any):
    """(re, im) of an expression whose expanded terms are each real or a
    real multiple of I; None when some term is undecided."""
    from ...core import Add, I, Integer, Mul, expand

    e = expand(arg)
    terms = e.args if isinstance(e, Add) else (e,)
    re_part = Integer(0)
    im_part = Integer(0)
    for t in terms:
        if getattr(t, "is_real", None) is True:
            re_part = re_part + t
            continue
        if t == I:
            im_part = im_part + 1
            continue
        if isinstance(t, Mul) and I in t.args:
            rest = Integer(1)
            for f in t.args:
                if f != I:
                    rest = rest * f
            if getattr(rest, "is_real", None) is True:
                im_part = im_part + rest
                continue
        return None
    return re_part, im_part


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
