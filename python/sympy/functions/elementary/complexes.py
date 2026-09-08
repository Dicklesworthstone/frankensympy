from typing import Any

from ...core import Abs, Function, _native, _native_expr, _wrap


def sign(expression):
    return _wrap(_native.py_sign(_native_expr(expression)))


class re(Function):
    """Real part of a complex expression."""

    @classmethod
    def eval(cls, arg: Any) -> Any:
        if arg is None:
            return None
        from ...core import S
        if getattr(arg, "is_real", None) is True:
            return arg
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
