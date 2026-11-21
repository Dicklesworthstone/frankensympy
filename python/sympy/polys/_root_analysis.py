"""Exact real-root queries for the built-in QQ/ZZ polynomial shell.

The native reference checker establishes completeness in the requested closed
real interval, distinct root count, original multiplicity and any required
rational width. Returned isolators use bounded bisection; their endpoints can
differ from upstream's continued-fraction choices. Complex rectangles,
extension coefficient fields and multi-polynomial isolation are not admitted.
"""

from ..core import (
    Basic, Dummy, Half, Integer, NegativeOne, One, Rational, S, Symbol, Zero,
    _active_callback_session,
    _charge_nested_if_active, _exact_ratio, _native, _native_expr, _wrap,
)
from . import _finitefield as retained
from ._evaluation_owners import require_original_owners
from .polyerrors import DomainError, PolynomialError


_RATIONAL_TYPES = (int, Integer, Rational, Zero, One, NegativeOne, Half)
_INTEGER_TYPES = (int, Integer, Zero, One, NegativeOne)


def _rational_control(value, label, infinity=None):
    if value is None or (infinity is not None and value is infinity):
        return None
    if type(value) is float:
        # A Python float is an immutable input tolerance/bound. Preserve its
        # exact binary value as a rational; no root is computed in binary64.
        try:
            numerator, denominator = value.as_integer_ratio()
        except (ValueError, OverflowError) as error:
            raise ValueError("%s must be a finite real bound" % label) from error
        return Rational(numerator, denominator)
    if not retained._exact(value, _RATIONAL_TYPES):
        raise DomainError("%s requires a rational real value" % label)
    return value


def _validate_atoms(snapshot):
    """Literal owners cannot conceal another retained expression graph."""
    from .polytools import _IDEAL_ABSENT

    atoms = (Integer, Rational, Zero, One, NegativeOne, Half, Symbol, Dummy)
    fixed = {
        Zero: _native.py_integer(0),
        One: _native.py_integer(1),
        NegativeOne: _native.py_integer(-1),
        Half: _native.py_rational(1, 2),
    }
    _charge_nested_if_active(len(snapshot.slots) + 1)
    for node, owner, name, _, value in snapshot.slots:
        kind = type(node)
        if not any(kind is atom for atom in atoms):
            continue
        if owner is Basic and name in ("_struct_args", "_surface_args"):
            if (value is not _IDEAL_ABSENT and value is not None
                    and (type(value) is not tuple or value)):
                raise ValueError("root-query atom has inconsistent retained arguments")
        if owner is not Basic or name != "_value":
            continue
        if type(value) is not _native.Expr:
            raise ValueError("root-query atom has invalid native storage")
        if kind is Symbol or kind is Dummy:
            valid = value.is_symbol
        elif kind in (Integer, Zero, One, NegativeOne):
            valid = value.is_integer
        else:
            valid = value.is_integer or value.is_rational
        if not valid or (kind in fixed and value != fixed[kind]):
            raise ValueError("root-query atom does not match its native owner")


def analyze(polynomial, operation, lower=None, upper=None, eps=None, steps=None,
            *, all=False, fast=False, sqf=False, check_sqf=False):
    for name, value in (("all", all), ("fast", fast), ("sqf", sqf),
                        ("check_sqf", check_sqf)):
        if type(value) is not bool:
            raise TypeError("%s must be a bool" % name)
    if all:
        raise NotImplementedError("complex root rectangles are not implemented")
    require_original_owners()
    session = _active_callback_session()
    _charge_nested_if_active(1)
    state = retained._PolyState(polynomial, require_field=False)
    if retained.is_finite(polynomial):
        raise DomainError("real-root analysis requires a characteristic-zero polynomial")
    namespace = state.namespace
    if len(state.gens) != 1:
        raise PolynomialError("real-root analysis requires a univariate polynomial")
    if (type(namespace.get("_domain")) is not str
            or namespace["_domain"] not in ("ZZ", "QQ")
            or type(namespace.get("_domain_given")) is not bool):
        raise DomainError("real-root analysis supports only ZZ and QQ")
    if type(namespace.get("_extra_syms")) is not frozenset or namespace["_extra_syms"]:
        raise DomainError("real-root analysis requires rational coefficients")
    lower = _rational_control(lower, "inf", S.NegativeInfinity)
    upper = _rational_control(upper, "sup", S.Infinity)
    eps = _rational_control(eps, "eps")
    if operation == "intervals" and eps is None:
        eps = Integer(1)
    if steps is not None and not retained._exact(steps, _INTEGER_TYPES):
        raise TypeError("steps must be an integer")
    controls = [value for value in (lower, upper, eps, steps) if value is not None]
    captured = retained._Retained([state.expression, *controls], state.gens, (state,))
    _validate_atoms(captured.snapshot)
    captured.unchanged()
    request = retained._Request([state.expression, *controls], state.gens, (state,))
    _validate_atoms(request.snapshot)

    def unchanged():
        require_original_owners()
        captured.unchanged()
        request.unchanged()
        if _active_callback_session() is not session:
            raise RuntimeError("root-query evaluation session changed")

    unchanged()
    _charge_nested_if_active(len(request.snapshot.slots) + len(controls) + 1)
    if steps is not None:
        steps = max(0, retained._integer(steps, "steps"))
        if steps > 32768:
            raise ValueError("root refinement supports at most 32768 bisections")
    if operation == "refine":
        if lower is None or upper is None:
            raise ValueError("root refinement requires two finite rational endpoints")
        # The retained exact-class inputs have been admitted before arithmetic.
        # Multiplication is bounded by the 4096-bit control limit.
        left = _exact_ratio(lower)
        right = _exact_ratio(upper)
        if left is None or right is None:
            raise DomainError("root refinement requires rational endpoints")
        size = max(abs(left[0]).bit_length(), left[1].bit_length(),
                   abs(right[0]).bit_length(), right[1].bit_length())
        _charge_nested_if_active(((2 * size + 31) // 32) ** 2 + 1)
        if left[0] * right[1] > right[0] * left[1]:
            lower, upper = upper, lower
    unchanged()
    raw = _native.real_root_analysis_native(
        request.raw[0], request.bindings[0], operation,
        None if lower is None else _native_expr(lower),
        None if upper is None else _native_expr(upper),
        None if eps is None else _native_expr(eps),
        steps, sqf or check_sqf,
        namespace["_domain_given"] and namespace["_domain"] == "ZZ",
        session,
    )
    unchanged()
    _charge_nested_if_active(len(raw) * 3 + 1)
    if operation == "count":
        result = Integer(len(raw))
    elif operation == "refine":
        if len(raw) != 1:
            raise RuntimeError("root refinement returned no unique checked interval")
        result = (_wrap(raw[0][0]), _wrap(raw[0][1]))
    elif sqf:
        result = [(_wrap(lo), _wrap(hi)) for lo, hi, _ in raw]
    else:
        result = [((_wrap(lo), _wrap(hi)), multiplicity)
                  for lo, hi, multiplicity in raw]
    unchanged()
    return result


def polynomial_input(expression):
    from .polytools import Poly, PurePoly
    if retained._exact(expression, (Poly, PurePoly)):
        return expression
    if retained._exact(expression, (list, tuple)):
        raise NotImplementedError("joint isolation of multiple polynomials is not implemented")
    return Poly(expression)
