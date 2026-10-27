"""Certified proper real integration with exact rational error bounds.

The native generator proposes a bounded adaptive partition. An independent
lower-layer checker evaluates the original source on every closed cell,
including all original domain restrictions, and verifies the requested
absolute enclosure width. No symbolic primitive or sampled error estimate
is required.
"""

from fractions import Fraction

from ..core import Dummy, Symbol, _native, _wrap
from ..utilities._certified_numeric import (
    _check_assumptions, _native_number, _rational, _sequence, bound_assumptions,
)
from ..utilities.lambdify import _functions, _numeric_source

__all__ = ["certified_integral", "IntegralEnclosureReceipt"]


def _integral_binding(variable):
    """Capture the actual native atom before inspecting mutable shell facts.

    The generic shell binding helper canonicalizes arbitrary fact values.
    A certified API instead checks exact Boolean snapshots against the
    existing immutable payload, without invoking truth/string hooks.
    """
    if type(variable) not in (Symbol, Dummy):
        raise TypeError("integration variable must be an exact Symbol or Dummy")
    payload = variable._value
    if type(payload) is not _native.Expr:
        raise TypeError("integration variable requires an exact native symbol payload")
    binding = payload.symbol_binding
    if binding is None:
        raise TypeError("integration variable payload is not a native symbol")
    return binding


class IntegralEnclosureReceipt:
    """Immutable source-bound integral enclosure with an exact error bound.

    ``bounds`` enclose the proper real integral over ``interval``;
    ``midpoint`` differs from that integral by at most ``error_bound``.
    The complete width is at most ``absolute_tolerance``. ``variable`` is
    an immutable native SymbolBinding, retaining its declared identity.
    """

    __slots__ = ("_receipt",)

    def __init__(self, receipt):
        if hasattr(self, "_receipt"):
            raise AttributeError("accepted integral receipts are immutable")
        if type(receipt) is not _native.IntegralEnclosureReceipt:
            raise TypeError("an accepted native integral enclosure is required")
        object.__setattr__(self, "_receipt", receipt)

    def __setattr__(self, name, value):
        raise AttributeError("accepted integral receipts are immutable")

    @property
    def integrand(self):
        return _wrap(self._receipt.integrand)

    @property
    def variable(self):
        return self._receipt.variable

    @property
    def interval(self):
        return tuple(_wrap(value) for value in self._receipt.interval)

    @property
    def bounds(self):
        return tuple(_wrap(value) for value in self._receipt.bounds)

    @property
    def midpoint(self):
        return _wrap(self._receipt.midpoint)

    @property
    def error_bound(self):
        return _wrap(self._receipt.error_bound)

    @property
    def absolute_tolerance(self):
        return _wrap(self._receipt.absolute_tolerance)

    @property
    def precision_digits(self):
        return self._receipt.precision_digits

    @property
    def partition(self):
        return tuple(_wrap(value) for value in self._receipt.partition)

    @property
    def interval_count(self):
        return self._receipt.interval_count

    @property
    def claim(self):
        return "proper real integral enclosed within the requested absolute width"

    def __repr__(self):
        return (f"IntegralEnclosureReceipt(bounds={self.bounds!r}, "
                f"interval={self.interval!r}, error_bound={self.error_bound!r}, "
                f"interval_count={self.interval_count!r})")


def certified_integral(integrand, limits, *, tolerance=Fraction(1, 10000),
                       precision=32, max_intervals=512):
    """Enclose a proper real integral without finding an antiderivative.

    ``limits`` is an exact list/tuple ``(x, a, b)`` with a Symbol or Dummy
    and finite exact rational endpoints. Bounds and tolerance admit exact
    int, Rational and fractions.Fraction values, never floating inputs or
    conversion hooks. ``tolerance`` bounds the complete enclosure width,
    so the midpoint's absolute error is at most half that tolerance.

    The supported exact source consists of the declared real variable,
    rational constants, pi/E, arithmetic, real powers on a verified smooth
    domain and the native C1 elementary functions. Unknown parameters,
    original poles, nonsmooth endpoints, nonreal branches and custom
    symbolic objects refuse. Reversed limits negate the enclosure; equal
    limits return zero only after checking the original source at the point.

    Refinement is deterministic and bounded at 1..1024 intervals, with
    precision 8..128 decimal digits. Exhaustion raises ValueError and
    publishes no partial receipt. This API is separate from symbolic
    ``integrate`` and does not assert an improper integral or principal value.
    """
    variable, lower, upper = _sequence(limits, 3, "integration limits")
    binding = _integral_binding(variable)
    if type(precision) is not int or not 8 <= precision <= 128:
        raise ValueError("certified precision must be an integer from 8 through 128")
    if type(max_intervals) is not int or not 1 <= max_intervals <= 1024:
        raise ValueError("max_intervals must be an integer from 1 through 1024")
    lower, upper, tolerance = _rational(lower), _rational(upper), _rational(tolerance)
    if tolerance <= 0:
        raise ValueError("certified integral tolerance must be positive")
    assumptions = bound_assumptions([variable], [binding])
    _check_assumptions([(min(lower, upper), max(lower, upper))], assumptions)
    if type(integrand) is bool:
        raise TypeError("certified integral source requires an exact mathematical expression")
    inexact = [False]
    source = _numeric_source(integrand, _functions(), [0], inexact=inexact)
    if inexact[0]:
        raise ValueError("certified integral source requires exact literals; replace Float with explicit Rational")
    native = _native.certified_integral_native(
        source, binding, _native_number(lower), _native_number(upper),
        _native_number(tolerance), precision, max_intervals,
    )
    return IntegralEnclosureReceipt(native)
