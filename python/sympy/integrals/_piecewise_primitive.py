"""Continuous, checked polynomial primitives of rational-cut Piecewise inputs.

The first-match guard solver supplies a complete partition of the real line.
On every open component the selected original source must be a polynomial
over QQ. Its primitive is independently differentiated and the original
source-bound rational equation checker proves equality. Separate exact
evaluation of the constructed expressions checks continuity at every cut.
This is an ordinary continuous primitive, with no differentiability claim at
a jump or an isolated exceptional value of the integrand.

The leftmost polynomial primitive has constant coefficient zero, following
the pinned public convention. Later constants enforce continuity; the final
Piecewise value at zero need not be zero. Isolated values and singleton gaps
do not affect this convention. An uncovered open interval, an unsupported
guard, or a nonpolynomial selected branch leaves the request unevaluated.

Input degree is at most 64 and output degree at most 65, subject to the
underlying checked operations' own limits. A conservative coefficient-growth
bound precedes the otherwise unmetered polynomial coefficient conversion.
This composition issues no new portable certificate or general integration
claim. Admitted operational and verification failures are terminal.
"""

from fractions import Fraction

from ..core import (
    Add, Integer, Le, Lt, Mul, _lift_native_result, _native, _native_expr,
    _require_closed_function_class, _surface_binding_key, _symbol_binding,
    _wrap, oo,
)
from ..functions.elementary.piecewise import Piecewise
from ..sets import Interval
from ..solvers._piecewise_real import (
    _Capture, _checked_domain, _guard, _restrict,
)
from ._piecewise import _rational_components, _scalar


_MAX_DEGREE = 64
_MAX_BITS = 4096
_MAX_COMPONENTS = 64
_MAX_WORK = 250000


class _Work:
    """Bound exact Python arithmetic independently of native work meters."""

    def __init__(self):
        self.operations = 0

    def charge(self):
        self.operations += 1
        if self.operations > _MAX_WORK:
            raise ValueError("Piecewise primitive exceeds its rational work limit")

    def value(self, value):
        self.charge()
        if (type(value) is not Fraction
                or max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > _MAX_BITS):
            raise ValueError("Piecewise primitive exceeds its rational coefficient limit")
        return value

    def add(self, left, right):
        return self.value(left + right)

    def mul(self, left, right):
        return self.value(left * right)

    def divide(self, left, right):
        return self.value(left / right)

    def power(self, base, exponent):
        value = Fraction(1)
        while exponent:
            if exponent & 1:
                value = self.mul(value, base)
            exponent >>= 1
            if exponent:
                base = self.mul(base, base)
        return value


def _native_fraction(value, work):
    if type(value) is not _native.Expr or not (value.is_integer or value.is_rational):
        raise ValueError("Piecewise primitive requires an exact rational coefficient")
    return work.value(Fraction(value.exact_numerator(), value.exact_denominator()))


def _polynomial_bound(source, variable, maximum_degree=_MAX_DEGREE):
    """Bound all coefficient-converter intermediates before conversion.

    For a polynomial P/D, retain an upper degree, a strict binary bound on
    the integer coefficient L1 norm of P, and a non-strict binary bound on
    the positive common denominator D. Addition uses cross-multiplication;
    multiplication uses the submultiplicativity of the L1 norm. Bounds are
    checked at every node, including held children of zero products and
    zeroth powers. Cancellation never permits skipping the original tree.
    """
    _native.ideal_source_preflight_native(source)
    visited = 0

    def bounded(envelope):
        degree, numerator_bits, denominator_bits = envelope
        if degree > maximum_degree:
            raise ValueError("Piecewise primitive exceeds its polynomial degree limit")
        # The denominator bound is non-strict: D <= 2**b has at most b+1
        # bits. The numerator bound is strict and needs no extra bit.
        if numerator_bits > _MAX_BITS or denominator_bits >= _MAX_BITS:
            raise ValueError("Piecewise primitive exceeds its coefficient-growth limit")
        return envelope

    def visit(node, depth):
        nonlocal visited
        visited += 1
        if visited > 4096 or depth > 64:
            raise ValueError("Piecewise primitive exceeds its polynomial source limit")
        if node.is_integer or node.is_rational:
            numerator, denominator = node.exact_numerator(), node.exact_denominator()
            return bounded((0, abs(numerator).bit_length(), (denominator - 1).bit_length()))
        if node.is_symbol:
            return bounded((1, 1, 0)) if node == variable else None
        children = node.args
        if len(children) > 256:
            raise ValueError("Piecewise primitive exceeds its polynomial source limit")
        if node.is_pow:
            if len(children) != 2:
                raise ValueError("invalid polynomial primitive power")
            base = visit(children[0], depth + 1)
            exponent = children[1]
            if not exponent.is_integer:
                return None
            power = exponent.exact_numerator()
            if power < 0 or base is None:
                return None
            if power > maximum_degree:
                raise ValueError("Piecewise primitive exceeds its polynomial exponent limit")
            return ((0, 1, 0) if power == 0 else
                    bounded(tuple(value * power for value in base)))
        if not (node.is_add or node.is_mul):
            return None
        parts = [visit(child, depth + 1) for child in children]
        if any(part is None for part in parts):
            return None
        result = (0, 0, 0) if node.is_add else (0, 1, 0)
        for degree, numerator_bits, denominator_bits in parts:
            old_degree, old_numerator, old_denominator = result
            if node.is_add:
                result = (max(old_degree, degree),
                          max(old_numerator + denominator_bits,
                              numerator_bits + old_denominator) + 1,
                          old_denominator + denominator_bits)
            else:
                result = (old_degree + degree, old_numerator + numerator_bits,
                          old_denominator + denominator_bits)
            bounded(result)
        return result

    return visit(source, 0)


def _coefficients(source, binding, degree_bound, work):
    coefficients = _native.poly_coeffs_native(source, binding)
    if type(coefficients) is not list or not 1 <= len(coefficients) <= degree_bound + 1:
        raise ValueError("invalid bounded Piecewise polynomial coefficients")
    values = tuple(_native_fraction(value, work) for value in reversed(coefficients))
    if len(values) > 1 and not values[-1]:
        raise ValueError("noncanonical Piecewise polynomial coefficients")
    return values


def _primitive_coefficients(coefficients, work):
    values = [Fraction(0)]
    values.extend(work.divide(value, Fraction(degree + 1))
                  for degree, value in enumerate(coefficients))
    while len(values) > 1 and not values[-1]:
        values.pop()
    return tuple(values)


def _evaluate_coefficients(coefficients, point, work):
    value = Fraction(0)
    for coefficient in reversed(coefficients):
        value = work.add(work.mul(value, point), coefficient)
    return value


def _native_polynomial(coefficients, variable):
    terms = []
    for degree, coefficient in enumerate(coefficients):
        if not coefficient:
            continue
        scalar = _native.py_rational(coefficient.numerator, coefficient.denominator)
        if not degree:
            terms.append(scalar)
            continue
        power = variable if degree == 1 else _native.py_pow(variable, _native.py_integer(degree))
        terms.append(power if coefficient == 1 else scalar * power)
    if not terms:
        return _native.py_integer(0)
    # The checked derivative of a sum evaluates each monomial subderivative.
    # Keep that shape for one term too: an unsimplified top-level product rule
    # could otherwise retain 0*x**65, outside the identity solver's exponent
    # bound even though the actual derivative has degree at most 64. Lifting
    # through the ordinary Add constructor removes this unary sum afterward.
    return _native.py_add(*terms)


def _evaluate_native(source, variable, point, work):
    """Check the actual candidate tree, independently of coefficient Horner."""
    visited = 0

    def visit(node, depth):
        nonlocal visited
        visited += 1
        work.charge()
        if visited > 4096 or depth > 64:
            raise ValueError("Piecewise primitive verifier exceeds its expression limit")
        if node.is_integer or node.is_rational:
            return _native_fraction(node, work)
        if node.is_symbol:
            if node != variable:
                raise ValueError("Piecewise primitive verifier found a different typed variable")
            return work.value(point)
        children = node.args
        if len(children) > 256:
            raise ValueError("Piecewise primitive verifier exceeds its expression limit")
        if node.is_pow:
            if len(children) != 2 or not children[1].is_integer:
                raise ValueError("invalid polynomial primitive verifier power")
            exponent = children[1].exact_numerator()
            if not 0 <= exponent <= _MAX_DEGREE + 1:
                raise ValueError("Piecewise primitive verifier exceeds its power limit")
            return work.power(visit(children[0], depth + 1), exponent)
        if node.is_add:
            result = Fraction(0)
            for child in children:
                result = work.add(result, visit(child, depth + 1))
            return result
        if node.is_mul:
            result = Fraction(1)
            for child in children:
                result = work.mul(result, visit(child, depth + 1))
            return result
        raise ValueError("Piecewise primitive verifier requires a polynomial candidate")

    return visit(source, 0)


def _verify_derivative(candidate, original, binding):
    derivative, _receipt = candidate.diff_with_receipt(binding)
    # Retain both operands, rather than simplifying away the original source
    # before the independent equation check can bind and inspect it.
    residual = Add(_wrap(derivative), Mul(Integer(-1), original, evaluate=False), evaluate=False)
    checked = _native.solve_real_rational_native(_native_expr(residual), binding)
    if (type(checked) is not tuple or len(checked) != 3
            or checked[0] is not True or type(checked[2]) is not int or checked[2] != 0):
        raise ValueError("Piecewise primitive derivative identity was not verified")


def integrate_piecewise_polynomial(expression, variable):
    """Return a continuous checked primitive, or None outside this lane."""
    if type(expression) is not Piecewise:
        return None
    _require_closed_function_class(Piecewise)
    capture = _Capture()
    capture.basic(expression)
    branches = capture.retain(expression, Piecewise, "_piecewise_branches")
    if type(branches) is not tuple or not 1 <= len(branches) <= 16:
        raise ValueError("polynomial Piecewise integration requires one to sixteen retained branches")
    capture.scalar(variable, 0)
    prepared = []
    for pair in branches:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("invalid retained Piecewise integration branch")
        prepared.append((capture.scalar(pair[0], 0), capture.condition(pair[1])))
    native_variable = _native_expr(variable)
    if not native_variable.is_symbol:
        raise TypeError("polynomial Piecewise integration requires a Symbol")
    binding = _symbol_binding(variable)
    if _surface_binding_key(native_variable.symbol_binding) != _surface_binding_key(binding):
        raise ValueError("Piecewise primitive variable has inconsistent typed assumptions")
    variable, sources, eligible = capture.freeze(variable)
    capture.unchanged()
    if not eligible:
        return None

    lower, upper = (-1, Fraction(0)), (1, Fraction(0))
    remaining = Interval(-oo, oo)
    snapshots, components = [], []

    def finish(value):
        capture.unchanged()
        for snapshot in snapshots:
            snapshot.unchanged()
        return value

    for branch, condition in prepared:
        rest = _checked_domain(remaining)
        snapshots.append(rest)
        intervals = _rational_components(rest.tree, lower, upper)
        if intervals is None:
            return finish(None)
        if not intervals:
            break
        selected = _guard(condition, sources, variable, remaining)
        if selected is None:
            return finish(None)
        chosen = _checked_domain(selected)
        snapshots.append(chosen)
        intervals = _rational_components(chosen.tree, lower, upper)
        if intervals is None:
            return finish(None)
        if len(components) + len(intervals) > _MAX_COMPONENTS:
            raise ValueError("Piecewise primitive exceeds sixty-four interval components")
        components.extend((a, b, branch, chosen) for a, b in intervals)
        remaining = _restrict(remaining, selected, True)

    rest = _checked_domain(remaining)
    snapshots.append(rest)
    intervals = _rational_components(rest.tree, lower, upper)
    if intervals is None or intervals:
        return finish(None)
    components.sort(key=lambda component: component[:2])
    position = lower
    for start, end, _, _ in components:
        if start != position or end <= start:
            raise ValueError("checked Piecewise primitive components do not form a complete partition")
        position = end
    if position != upper:
        raise ValueError("checked Piecewise primitive components leave an uncovered interval")

    # Every selected source is admitted before the first coefficient conversion
    # or candidate construction. A singleton-only branch needs no primitive.
    bounds = {}
    for _, _, branch, _ in components:
        if branch not in bounds:
            bounds[branch] = _polynomial_bound(_native_expr(sources[branch]), native_variable)
    if any(bound is None for bound in bounds.values()):
        return finish(None)
    finish(None)

    work = _Work()
    primitives = {}
    for branch, bound in bounds.items():
        coefficients = _coefficients(_native_expr(sources[branch]), binding, bound[0], work)
        primitives[branch] = _primitive_coefficients(coefficients, work)
        finish(None)

    checked_pieces = []
    previous_coefficients = previous_native = None
    for start, end, branch, domain in components:
        coefficients = primitives[branch]
        if previous_coefficients is not None:
            point = work.value(start[1])
            shift = work.add(_evaluate_coefficients(previous_coefficients, point, work),
                             -_evaluate_coefficients(coefficients, point, work))
            coefficients = (work.add(coefficients[0], shift), *coefficients[1:])
        candidate = _native_polynomial(coefficients, native_variable)
        if _polynomial_bound(candidate, native_variable, _MAX_DEGREE + 1) is None:
            raise ValueError("Piecewise primitive construction produced a nonpolynomial candidate")
        _verify_derivative(candidate, sources[branch], binding)
        if previous_native is None:
            if _evaluate_native(candidate, native_variable, Fraction(0), work) != 0:
                raise ValueError("Piecewise primitive initial constant was not verified")
        elif (_evaluate_native(previous_native, native_variable, start[1], work)
              != _evaluate_native(candidate, native_variable, start[1], work)):
            raise ValueError("Piecewise primitive continuity was not verified")
        finish(None)
        if checked_pieces and checked_pieces[-1][0] == coefficients:
            checked_pieces[-1] = (coefficients, candidate, end, domain)
        else:
            checked_pieces.append((coefficients, candidate, end, domain))
        previous_coefficients, previous_native = coefficients, candidate

    pairs = []
    for _, candidate, end, domain in checked_pieces:
        value = _lift_native_result(candidate, *capture.symbols, variable)
        if end == upper:
            condition = True
        else:
            closed = domain.contains(("rational", end[1]))
            if closed is not True and closed is not False:
                raise ValueError("Piecewise primitive endpoint ownership was not verified")
            condition = (Le if closed else Lt)(variable, _scalar(end))
        pairs.append((value, condition))
        finish(None)
    return finish(Piecewise(*pairs))
