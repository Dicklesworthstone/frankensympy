"""Trigonometric functions for FrankenSymPy.

Each function is a real Function subclass whose automatic evaluation
(special values at rational multiples of pi, parity, period reduction,
inverse compositions) is performed by the native kernel.
"""

from ...core import _NativeFunction


class sin(_NativeFunction):
    __slots__ = ()


class cos(_NativeFunction):
    __slots__ = ()


class tan(_NativeFunction):
    __slots__ = ()


class cot(_NativeFunction):
    __slots__ = ()


class sec(_NativeFunction):
    __slots__ = ()


class csc(_NativeFunction):
    __slots__ = ()


class asin(_NativeFunction):
    __slots__ = ()


class acos(_NativeFunction):
    __slots__ = ()


class atan(_NativeFunction):
    __slots__ = ()


class acot(_NativeFunction):
    __slots__ = ()


class asec(_NativeFunction):
    __slots__ = ()


class acsc(_NativeFunction):
    __slots__ = ()


class sinc(_NativeFunction):
    __slots__ = ()


class atan2(_NativeFunction):
    """Two-argument arctangent: the principal argument of ``x + I*y``."""

    __slots__ = ()
