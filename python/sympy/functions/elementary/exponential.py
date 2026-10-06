"""Exponential and logarithmic functions for FrankenSymPy."""

from ...core import Function, _NativeFunction


class exp(_NativeFunction):
    __slots__ = ()


class log(_NativeFunction):
    """Natural logarithm; ``log(x, b)`` is ``log(x)/log(b)`` as upstream."""

    __slots__ = ()

    def __new__(cls, arg, base=None, **options):
        if base is not None:
            from ...core import sympify

            base = sympify(base)
            numerator = cls(arg, **options)
            if base == sympify(1):
                raise ValueError("log base must not be 1")
            return numerator / cls(base, **options)
        return super().__new__(cls, arg, **options)


ln = log


class LambertW(_NativeFunction):
    """Lambert W function (principal branch): W(0) = 0, W(E) = 1."""

    __slots__ = ()


__all__ = [
    "LambertW",
    "exp",
    "ln",
    "log",
]
