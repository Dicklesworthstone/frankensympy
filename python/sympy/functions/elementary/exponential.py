"""Exponential and logarithmic functions for FrankenSymPy."""

from ...core import Function, _NativeFunction


class exp(_NativeFunction):
    __slots__ = ()


class log(_NativeFunction):
    """Natural logarithm; ``log(x, b)`` is ``log(x)/log(b)`` as upstream."""

    __slots__ = ()

    @classmethod
    def eval(cls, arg):
        # log(exp(a)) = a for real a (principal branch).
        if type(arg).__name__ == "exp" and getattr(arg.args[0], "is_real", None):
            return arg.args[0]
        return super().eval(arg)

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
    """Lambert W with an optional integer branch index.

    ``LambertW(z)`` is the principal branch. ``LambertW(z, -1)`` represents
    the lower real branch on ``[-1/E, 0)``. Exact branch-point values and
    derivatives retain the selected branch.
    """

    __slots__ = ()


__all__ = [
    "LambertW",
    "exp",
    "ln",
    "log",
]
