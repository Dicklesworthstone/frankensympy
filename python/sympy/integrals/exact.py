"""Exact real rational integrals with original-source-bound receipts.

This FrankenSymPy extension exposes checked ordinary and convergent improper
integrals. It does not request a Cauchy principal value or permit an upstream
execution fallback.
"""

from fractions import Fraction

from ..core import (
    Add, Dummy, Expr, Half, Integer, Mul, NegativeOne, One, Pow, Rational,
    Symbol, Zero, _lift_native_result, _native, _native_expr, _symbol_binding,
)

_ATOMS = (Integer, Rational, Zero, One, NegativeOne, Half, Symbol, Dummy)


def _rational_literal(value):
    if type(value) is int:
        if value.bit_length() > 4096:
            raise ValueError("exact integral coefficient exceeds 4096 bits")
        return Integer(value)
    if type(value) is Fraction:
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > 4096:
            raise ValueError("exact integral coefficient exceeds 4096 bits")
        return Rational(value.numerator, value.denominator)
    return value


def _exact_source(expression):
    """Check original surface owners before the exact native recurrence."""
    expression = _rational_literal(expression)
    if type(expression) not in _ATOMS + (Add, Mul, Pow):
        raise TypeError("exact rational integration requires exact rational expressions")
    # Borrow the entire native payload first, before shell .args could copy
    # children. This shared preflight checks syntax, size and coefficient bits.
    _native.ideal_source_preflight_native(_native_expr(expression))
    pending = [(expression, 0)]
    visited = 0
    while pending:
        node, depth = pending.pop()
        visited += 1
        if visited > 4096 or depth > 64:
            raise ValueError("exact integral source exceeds its bounded envelope")
        if type(node) not in _ATOMS + (Add, Mul, Pow):
            raise TypeError("exact rational integration refuses custom or inexact source nodes")
        _native_expr(node)
        children = getattr(node, "_struct_args", None)
        if children is None:
            children = getattr(node, "_args", ())
        if type(children) is not tuple:
            raise TypeError("exact rational integration requires original argument tuples")
        if len(children) > 4096 - visited - len(pending):
            raise ValueError("exact integral source exceeds its bounded envelope")
        pending.extend((child, depth + 1) for child in children)
    return expression


def _endpoint(value):
    value = _rational_literal(value)
    if type(value) not in (Integer, Rational, Zero, One, NegativeOne, Half, Expr):
        raise TypeError("exact integral endpoints require rational literals or real infinities")
    return value


class ExactRationalIntegralReceipt:
    """Immutable checked exact value of an ordinary real definite integral.

    The receipt retains the original integrand, typed variable and endpoints.
    Every original reciprocal guard is checked on the integration interval.
    Endpoint singularities are accepted only when the separate endpoint limits
    converge; interior holes require explicit splitting, even if removable.

    This is an in-process mathematical receipt. It is not a portable proof
    artifact or a certification of the complete SymPy compatibility profile.
    """

    __slots__ = ("_receipt", "_owners")

    def __init__(self, receipt, owners=()):
        if hasattr(self, "_receipt"):
            raise AttributeError("accepted exact integral receipts are immutable")
        if type(receipt) is not _native.ExactRationalIntegralReceipt:
            raise TypeError("an accepted native exact integral receipt is required")
        if type(owners) is not tuple:
            raise TypeError("exact integral owners must be an immutable tuple")
        object.__setattr__(self, "_receipt", receipt)
        object.__setattr__(self, "_owners", owners)

    def __setattr__(self, name, value):
        raise AttributeError("accepted exact integral receipts are immutable")

    def __delattr__(self, name):
        raise AttributeError("accepted exact integral receipts are immutable")

    @property
    def integrand(self):
        return _lift_native_result(self._receipt.integrand, *self._owners)

    @property
    def variable(self):
        return _lift_native_result(self._receipt.variable, *self._owners)

    @property
    def interval(self):
        return tuple(_lift_native_result(value, *self._owners)
                     for value in self._receipt.interval)

    @property
    def value(self):
        return _lift_native_result(self._receipt.value, *self._owners)

    @property
    def original_guard_count(self):
        return self._receipt.original_guard_count

    @property
    def claim(self):
        return "exact ordinary or convergent improper real integral of the original source"

    def __repr__(self):
        return (f"ExactRationalIntegralReceipt(value={self.value!r}, "
                f"interval={self.interval!r})")


def exact_rational_integral(integrand, limits):
    """Return a checked exact real rational integral and its source receipt.

    limits is an exact tuple/list (x, a, b). The variable must be an exact
    Symbol or Dummy; endpoints may be exact rational literals or either real
    infinity. Rational input uses only the declared variable, rational
    constants, addition, multiplication and literal integer powers.

    Hermite reduction handles repeated denominators. The remaining primitive
    uses rational logarithmic derivatives and real quadratic atan/log terms.
    Coupled logarithmic limits are checked together at infinity. Unsupported
    primitives, divergence, original interior poles and exhausted work bounds
    raise explicitly and publish no partial receipt. Each phase has a separate
    cumulative work budget, including source admission and domain checks.
    """
    if type(limits) not in (tuple, list) or len(limits) != 3:
        raise TypeError("exact rational integral limits must be (variable, lower, upper)")
    variable, lower, upper = limits
    if type(variable) not in (Symbol, Dummy):
        raise TypeError("exact rational integration requires an exact Symbol or Dummy")
    source = _exact_source(integrand)
    lower, upper = _endpoint(lower), _endpoint(upper)
    native = _native.exact_rational_integral_native(
        _native_expr(source), _symbol_binding(variable),
        _native_expr(lower), _native_expr(upper),
    )
    return ExactRationalIntegralReceipt(native, (source, variable, lower, upper))


__all__ = ["ExactRationalIntegralReceipt", "exact_rational_integral"]
