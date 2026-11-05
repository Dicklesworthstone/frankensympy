"""Exact ordinary and convergent improper rational Piecewise integrals.

Checked rational sign charts determine the original first-match domains. The
set trees then give complete open interval components, without numerical
sampling. Each component uses the existing source-bound rational integral
checker, including its original-pole and separate endpoint obligations.
Finite bounds and partition points are rational; exact real infinities may
bound the request. Every selected tail must converge independently, so this
does not compute a Cauchy principal value by canceling divergent tails.

Finite isolated values and missing singleton points have zero integral, as in
the pinned public convention. An uncovered open interval is undefined and
returns nan. Unknown guards and nonrational partition endpoints remain
unevaluated; admitted operational failures propagate without another lane.
"""

from fractions import Fraction

from ..core import (
    Basic, Expr, Integer, Rational, _native, _native_expr,
    _require_closed_function_class, nan, oo,
)
from ..functions.elementary.piecewise import Piecewise
from ..sets import Interval
from ..solvers._piecewise_real import (
    _ABSENT, _Capture, _checked_domain, _guard, _restrict,
)
from ..solvers._real_domain import _ATOMS, _NEG_INF, _POS_INF, _is_exact
from .exact import _rational_literal, exact_rational_integral


_MAX_COMPONENTS = 64
_MAX_CUTS = 256


def _rational_components(tree, lower, upper):
    """Complete positive-length components of a checked rational set tree.

    Bounds are exact ordered pairs: (-1, 0) and (1, 0) denote the real
    infinities, and (0, Fraction(...)) denotes a finite rational. Sorting and
    comparison never involve binary floats or symbolic truth coercion.

    Every atomic interval boundary and finite point becomes a cut. Membership
    of an entire open cell follows from endpoint comparisons in the Boolean
    set tree, rather than a representative point. Adjacent cells join only
    when their common point also belongs to the set: an explicit puncture
    must remain a separate improper endpoint for the integral checker.

    None means an endpoint lies outside this bounded rational-endpoint lane.
    """
    cuts = {lower, upper}
    pending = [tree]
    while pending:
        node = pending.pop()
        kind = node[0]
        if kind in ("EmptySet", "UniversalSet"):
            continue
        if kind == "Interval":
            endpoints = node[1:3]
        elif kind == "FiniteSet":
            endpoints = node[1]
        elif kind in ("Union", "Intersection", "Complement"):
            pending.extend(node[1])
            continue
        else:
            raise ValueError("invalid checked Piecewise integration domain")
        for endpoint in endpoints:
            if endpoint is None or endpoint[0] not in ("rational", "infinity"):
                return None
            key = ((0, endpoint[1]) if endpoint[0] == "rational"
                   else (endpoint[1], Fraction(0)))
            if lower < key < upper:
                cuts.add(key)
                if len(cuts) > _MAX_CUTS:
                    raise ValueError("Piecewise integration has too many rational partition cuts")

    def compare(bound, value):
        key = ((0, bound[1]) if bound[0] == "rational"
               else (bound[1], Fraction(0)))
        return (key > value) - (key < value)

    def contains(node, left, right):
        kind = node[0]
        if kind == "EmptySet":
            return False
        if kind == "UniversalSet":
            return True
        if kind == "Interval":
            lo, hi = compare(node[1], left), compare(node[2], right)
            if left == right:
                return ((lo < 0 or lo == 0 and not node[3])
                        and (hi > 0 or hi == 0 and not node[4]))
            if compare(node[1], right) >= 0 or compare(node[2], left) <= 0:
                return False
            if lo <= 0 and hi >= 0:
                return True
            raise ValueError("checked Piecewise integration cell crosses an unretained boundary")
        if kind == "FiniteSet":
            return left == right and any(compare(point, left) == 0 for point in node[1])
        if kind == "Complement":
            if len(node[1]) != 1:
                raise ValueError("invalid checked Piecewise integration complement")
            return not contains(node[1][0], left, right)
        values = (contains(child, left, right) for child in node[1])
        return any(values) if kind == "Union" else all(values)

    ordered = sorted(cuts)
    intervals = []
    for left, right in zip(ordered, ordered[1:]):
        if not contains(tree, left, right):
            continue
        if (intervals and intervals[-1][1] == left
                and contains(tree, left, left)):
            intervals[-1] = (intervals[-1][0], right)
        else:
            intervals.append((left, right))
    return intervals


def _fraction(source):
    native = _native_expr(source)
    if not (native.is_integer or native.is_rational):
        raise ValueError("Piecewise integration requires retained rational endpoint values")
    numerator, denominator = native.exact_numerator(), native.exact_denominator()
    if max(abs(numerator).bit_length(), denominator.bit_length()) > 4096:
        raise ValueError("Piecewise integration endpoint exceeds 4096 bits")
    return Fraction(numerator, denominator)


def _scalar(value):
    if value[0] < 0:
        return -oo
    if value[0] > 0:
        return oo
    return Rational(value[1].numerator, value[1].denominator)


def _capture_endpoint(value, capture):
    """Retain a rational source or an exact literal real infinity.

    Infinities are captured separately from rational scalar leaves: they are
    interval bounds, never polynomial coefficients or integrand values.
    """
    if _is_exact(value, _ATOMS):
        return ("finite", capture.scalar(value, 0))
    if type(value) is not Expr:
        return None
    native = capture.retain(value, Basic, "_value")
    for name in ("_struct_args", "_surface_args"):
        children = capture.retain(value, Basic, name)
        if (children is not _ABSENT and children is not None
                and (type(children) is not tuple or len(children) != 0)):
            return None
    if type(native) is _native.Expr:
        if native == _NEG_INF:
            return ("infinity", -1)
        if native == _POS_INF:
            return ("infinity", 1)
    return None


def integrate_piecewise_rational(expression, variable, lower, upper):
    """Return a complete ordinary/improper value, or None for unsupported input.

    This composes existing checked operations; it does not issue a new
    portable proof or claim a general symbolic Piecewise integration method.
    One request has at most sixteen branches and sixty-four integral
    components, each retaining the rational checker's bounded work phases.
    """
    if type(expression) is not Piecewise:
        return None
    lower, upper = _rational_literal(lower), _rational_literal(upper)
    _require_closed_function_class(Piecewise)
    capture = _Capture()
    bounds = (_capture_endpoint(lower, capture), _capture_endpoint(upper, capture))
    if any(bound is None for bound in bounds):
        return None
    capture.basic(expression)
    branches = capture.retain(expression, Piecewise, "_piecewise_branches")
    if type(branches) is not tuple or not 1 <= len(branches) <= 16:
        raise ValueError("rational Piecewise integration requires one to sixteen retained branches")
    capture.scalar(variable, 0)
    prepared = []
    for pair in branches:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("invalid retained Piecewise integration branch")
        prepared.append((capture.scalar(pair[0], 0), capture.condition(pair[1])))
    variable, sources, eligible = capture.freeze(variable)
    capture.unchanged()
    if not eligible:
        return None

    left, right = ((0, _fraction(sources[value])) if kind == "finite"
                   else (value, Fraction(0)) for kind, value in bounds)
    if left == right:
        if left[0] != 0:
            return None  # Equal infinite bounds do not define a finite point.
        # A zero-width interval has no active neighborhood. The bounded
        # original source and guard grammar still precedes this convention.
        capture.unchanged()
        return Integer(0)
    direction = 1 if left < right else -1
    left, right = min(left, right), max(left, right)
    remaining = Interval(_scalar(left), _scalar(right))
    snapshots = []
    components = []

    def finish(value):
        capture.unchanged()
        for snapshot in snapshots:
            snapshot.unchanged()
        return value

    for branch, condition in prepared:
        rest = _checked_domain(remaining)
        snapshots.append(rest)
        open_rest = _rational_components(rest.tree, left, right)
        if open_rest is None:
            return finish(None)
        if not open_rest:
            break  # Only finitely many points remain, with zero integral.
        selected = _guard(condition, sources, variable, remaining)
        if selected is None:
            return finish(None)
        chosen = _checked_domain(selected)
        snapshots.append(chosen)
        intervals = _rational_components(chosen.tree, left, right)
        if intervals is None:
            return finish(None)
        if len(components) + len(intervals) > _MAX_COMPONENTS:
            raise ValueError("Piecewise integration exceeds sixty-four interval components")
        components.extend((sources[branch], a, b) for a, b in intervals)
        remaining = _restrict(remaining, selected, True)

    rest = _checked_domain(remaining)
    snapshots.append(rest)
    open_rest = _rational_components(rest.tree, left, right)
    if open_rest is None:
        return finish(None)
    if open_rest:
        # The branch partition is exact: a positive-length interval has no
        # selected value. This is the pinned undefined-integral convention.
        return finish(nan)

    # Independently check coverage and disjointness of the planned open
    # components before any integral is evaluated or any sum can be returned.
    position = left
    for start, end in sorted((a, b) for _, a, b in components):
        if start != position or end <= start:
            raise ValueError("checked Piecewise integral components do not form a complete partition")
        position = end
    if position != right:
        raise ValueError("checked Piecewise integral components leave an uncovered interval")
    finish(None)

    total = Integer(0)
    for source, start, end in components:
        receipt = exact_rational_integral(source, (variable, _scalar(start), _scalar(end)))
        total += receipt.value
        finish(None)
    return finish(total if direction == 1 else -total)
