"""Bounded exact real domains, images and extrema of rational functions.

The original source's complete checked domain chart is authoritative even
when algebraic cancellation would remove a pole. Complete checked derivative
roots then reduce each continuous component to endpoint limits and attained
critical values. Their order uses exact rational/algebraic arithmetic only.
This public compatibility helper does not issue a new native certificate.
"""

from fractions import Fraction
from functools import cmp_to_key

from ..core import (
    Add, Dummy, Half, Integer, Mul, NegativeOne, One, Pow, Rational, Symbol,
    Zero, _native, _native_expr, _require_closed_function_class, oo,
)
from ..sets.sets import (
    EmptySet, Interval, Reals, Set, _EMPTY_SET, _REALS, _wrap_set,
)
from ..core._function_capabilities import _implementation, _owner_matches, _owner_record
from .singularities import (
    _get_symbol_and_interval, _monotonicity_chart, _monotonicity_interval,
)


_ATOMS = (Integer, Rational, Zero, One, NegativeOne, Half)
_MISSING = object()
_EXACT_VALUE_NAMES = (
    "p", "q", "__int__", "__neg__", "__truediv__", "__rtruediv__",
    "__add__", "__radd__", "__mul__", "__rmul__", "__pow__", "__str__",
    "_assumptions", "_dummy_name", "_dummy_number", "name", "dummy_index",
    "_native_set", "start", "end", "left_open", "right_open",
)


def _exact_value_record(owner):
    # Literal access, arithmetic and printing are not all FunctionCapabilities
    # semantic names, but they bind this lane's values to the native source.
    namespace = type.__dict__["__dict__"].__get__(owner)
    return tuple(_implementation(namespace.get(name, _MISSING))
                 for name in _EXACT_VALUE_NAMES)


def _surface_matches(owner, surface, exact):
    if not _owner_matches(owner, surface):
        return False
    try:
        current = _exact_value_record(owner)
    except ValueError:
        return False
    return all(
        len(tokens) == len(expected)
        and all(value is recorded for value, recorded in zip(tokens, expected))
        for tokens, expected in zip(current, exact)
    )


# util imports this module during package initialization, before user code can
# replace built-in argument/coefficient accessors or symbol identity methods.
# Match implementations without executing descriptors before any source read.
_SURFACE_OWNERS = tuple(
    (owner, _owner_record(owner), _exact_value_record(owner))
    for owner in dict.fromkeys(
        owner for cls in (Add, Mul, Pow, Symbol, Dummy, *_ATOMS, Interval, Reals, EmptySet)
        for owner in (*type.__dict__["__mro__"].__get__(cls),
                      *type.__dict__["__mro__"].__get__(type(cls)))
    )
)

_NEGATIVE_INFINITY = (-1, None)
_POSITIVE_INFINITY = (1, None)
STATIONARY_UNSUPPORTED = object()
_STATIONARY_DOMAIN_SLOT = Set.__dict__["_native_set"]


def _require_original_surface():
    if any(not _surface_matches(owner, surface, exact)
           for owner, surface, exact in _SURFACE_OWNERS):
        raise NotImplementedError(
            "exact rational range requires the original built-in arithmetic implementations"
        )


def _finite(value):
    return 0, value


def _expression(bound):
    return -oo if bound[0] < 0 else oo if bound[0] > 0 else bound[1]


class _Work:
    """Intrinsic bounds shared by every Python-side stage of one request.

    Native chart, root and algebraic checks retain their separate bounded
    generator/verifier meters. No caller-owned runtime budget is implied.
    """

    def __init__(self):
        self.operations = 0
        self.decisions = 0

    def step(self, amount=1):
        self.operations += amount
        if self.operations > 4_000_000:
            raise NotImplementedError("exact rational range arithmetic limit exceeded")

    def scalar(self, value):
        if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 32_768:
            raise NotImplementedError("exact rational range coefficient limit exceeded")
        return value

    def binary(self, left, right, multiply=False):
        # Bound unreduced numerator/denominator growth before constructing it.
        bits = (max(abs(left.numerator).bit_length(), left.denominator.bit_length())
                + max(abs(right.numerator).bit_length(), right.denominator.bit_length()) + 1)
        if bits > 32_768:
            raise NotImplementedError("exact rational range coefficient limit exceeded")
        self.step(max(1, (bits + 31)//32)**2)
        return self.scalar(left*right if multiply else left + right)

    def compare(self, left, right):
        self.decisions += 1
        if self.decisions > 16_384:
            raise NotImplementedError("exact rational range comparison limit exceeded")
        if left[0] != right[0]:
            return (left[0] > right[0]) - (left[0] < right[0])
        if left[0]:
            return 0
        a, b = left[1], right[1]
        if type(a) in _ATOMS and type(b) in _ATOMS:
            qa, qb = self.scalar(Fraction(a.p, a.q)), self.scalar(Fraction(b.p, b.q))
            difference = self.binary(qa, -qb)
            return (difference > 0) - (difference < 0)
        from ..polys.rootoftools import _algebraic_comparison

        return _algebraic_comparison(a, b)


def _trim(coefficients):
    while len(coefficients) > 1 and coefficients[-1] == 0:
        coefficients.pop()
    return tuple(coefficients)


def _poly_add(left, right, work):
    work.step(max(len(left), len(right)))
    return _trim([
        work.binary(left[i] if i < len(left) else Fraction(0),
                    right[i] if i < len(right) else Fraction(0))
        for i in range(max(len(left), len(right)))
    ])


def _poly_mul(left, right, work):
    if left == (0,) or right == (0,):
        return (Fraction(0),)
    length = len(left) + len(right) - 1
    if length > 129:
        raise NotImplementedError("exact rational range intermediate degree limit exceeded")
    work.step(length)
    result = [Fraction(0)] * length
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] = work.binary(result[i + j], work.binary(a, b, True))
    return _trim(result)


def _poly_power(coefficients, exponent, work):
    result = (Fraction(1),)
    while exponent:
        if exponent & 1:
            result = _poly_mul(result, coefficients, work)
        exponent //= 2
        if exponent:
            coefficients = _poly_mul(coefficients, coefficients, work)
    return result


def _fraction(expression, symbol, work):
    """Read only the already admitted original QQ(x) tree, without strings.

    Domain restrictions were checked separately and are never inferred from
    this numerator/denominator representation. In particular, a zero product
    or zeroth power cannot grant a previously excluded point.
    """
    kind = type(expression)
    work.step()
    if kind in _ATOMS:
        value = work.scalar(Fraction(expression.p, expression.q))
        return (value,), (Fraction(1),)
    if kind in (Symbol, Dummy):
        if expression != symbol:
            raise NotImplementedError("exact rational range requires rational coefficients")
        return (Fraction(0), Fraction(1)), (Fraction(1),)
    children = expression.args
    if kind is Pow:
        numerator, denominator = _fraction(children[0], symbol, work)
        exponent = int(children[1])
        if abs(exponent) > 128:
            raise NotImplementedError("exact rational range exponent limit exceeded")
        if exponent < 0:
            numerator, denominator = denominator, numerator
            exponent = -exponent
        return _poly_power(numerator, exponent, work), _poly_power(denominator, exponent, work)
    if kind not in (Add, Mul):
        raise NotImplementedError("exact rational range requires a rational function")
    numerator = (Fraction(0 if kind is Add else 1),)
    denominator = (Fraction(1),)
    for child in children:
        a, b = _fraction(child, symbol, work)
        if kind is Add:
            numerator = _poly_add(_poly_mul(numerator, b, work),
                                  _poly_mul(a, denominator, work), work)
        else:
            numerator = _poly_mul(numerator, a, work)
        denominator = _poly_mul(denominator, b, work)
    return numerator, denominator


def _rational(value):
    return Rational(value.numerator, value.denominator)


def _algebraic_value(expression, work):
    """Retain an exact value; reduce it to a rational only after checking."""
    work.step()
    if type(expression) in _ATOMS:
        return expression
    from ..core import _wrap
    from ..polys.rootoftools import _algebraic_expression_value

    checked = _algebraic_expression_value(expression, 1)
    lower, upper = _wrap(checked.lower), _wrap(checked.upper)
    if type(lower) not in _ATOMS or type(upper) not in _ATOMS:
        raise ValueError("algebraic range evaluation returned nonrational bounds")
    if work.compare(_finite(lower), _finite(upper)) == 0:
        return lower
    return expression


def _quotient_value(numerator, denominator, work):
    if type(numerator) in _ATOMS and type(denominator) in _ATOMS:
        a = work.scalar(Fraction(numerator.p, numerator.q))
        b = work.scalar(Fraction(denominator.p, denominator.q))
        if b == 0:
            raise ValueError("zero rational endpoint denominator")
        return _rational(work.binary(a, 1/b, True))
    quotient = Mul(numerator, Pow(denominator, Integer(-1), evaluate=False), evaluate=False)
    return _algebraic_value(quotient, work)


def _polynomial_value(coefficients, point, work):
    if type(point) in _ATOMS:
        x = work.scalar(Fraction(point.p, point.q))
        value = Fraction(0)
        for coefficient in reversed(coefficients):
            value = work.binary(work.binary(value, x, True), coefficient)
        return _rational(value)
    # A shallow coefficient sum avoids a deep Horner graph at algebraic
    # points. These are generated exact coefficients, not reconstructed text.
    terms = []
    for exponent, coefficient in enumerate(coefficients):
        work.step()
        if coefficient == 0:
            continue
        value = _rational(coefficient)
        if exponent:
            value = Mul(value, Pow(point, Integer(exponent), evaluate=False), evaluate=False)
        terms.append(value)
    if not terms:
        return Integer(0)
    value = terms[0] if len(terms) == 1 else Add(*terms, evaluate=False)
    return _algebraic_value(value, work)


def _leading_at(coefficients, point, work):
    """Exact order of vanishing and the first nonzero derivative at point."""
    order = 0
    while coefficients != (0,):
        value = _polynomial_value(coefficients, point, work)
        sign = work.compare(_finite(value), _finite(Integer(0)))
        if sign:
            return order, value, sign
        work.step(len(coefficients))
        coefficients = tuple(work.binary(Fraction(i), c, True)
                             for i, c in enumerate(coefficients) if i) or (Fraction(0),)
        order += 1
    return None


def _limit(numerator, denominator, bound, direction, work):
    """One-sided rational limit, including retained removable holes."""
    if denominator == (0,):
        raise ValueError("a nowhere-defined rational function has no range component")
    if numerator == (0,):
        return _finite(Integer(0))
    if bound[0]:
        degree = len(numerator) - len(denominator)
        if degree < 0:
            return _finite(Integer(0))
        if degree == 0:
            inverse = work.scalar(1/denominator[-1])
            return _finite(_rational(work.binary(numerator[-1], inverse, True)))
        sign = 1 if (numerator[-1] > 0) == (denominator[-1] > 0) else -1
        if bound[0] < 0 and degree % 2:
            sign = -sign
        return sign, None
    p = _leading_at(numerator, bound[1], work)
    q = _leading_at(denominator, bound[1], work)
    if p is None or q is None:
        raise ValueError("invalid rational endpoint valuation")
    if p[0] > q[0]:
        return _finite(Integer(0))
    if p[0] == q[0]:
        return _finite(_quotient_value(p[1], q[1], work))
    sign = p[2]*q[2]
    if direction < 0 and (q[0] - p[0]) % 2:
        sign = -sign
    return sign, None


def _chart_pieces(chart):
    roots, cells = chart
    pieces = []
    position = 0
    while position < len(cells):
        if not cells[position]:
            position += 1
            continue
        first = position
        while position + 1 < len(cells) and cells[position + 1]:
            position += 1
        last = position
        lower = _NEGATIVE_INFINITY if first == 0 else roots[(first - 1)//2]
        upper = _POSITIVE_INFINITY if last == 2*len(roots) else roots[last//2]
        pieces.append((lower, upper, first % 2 == 0, last % 2 == 0))
        position += 1
    return pieces


def _intersection(left, right, work):
    order = work.compare(left[0], right[0])
    lower = left[0] if order >= 0 else right[0]
    left_open = left[2] if order > 0 else right[2] if order < 0 else left[2] or right[2]
    order = work.compare(left[1], right[1])
    upper = left[1] if order <= 0 else right[1]
    right_open = left[3] if order < 0 else right[3] if order > 0 else left[3] or right[3]
    order = work.compare(lower, upper)
    if order > 0 or order == 0 and (lower[0] or left_open or right_open):
        return None
    return lower, upper, left_open, right_open


def _admit(expression, symbol, domain, work, guard=None):
    # Admission precedes both source args access and interval lifting. Even an
    # already-constructed Interval must not invoke a changed CRootOf.is_real,
    # poly or index implementation while recovering its native endpoints.
    from ..polys.rootoftools import CRootOf

    _require_closed_function_class(CRootOf)
    _require_original_surface()
    expression, domain, symbol = _get_symbol_and_interval(expression, domain, symbol)
    interval = _monotonicity_interval(domain)
    if interval is None:
        return expression, symbol, []
    chart = _monotonicity_chart(Mul(0, expression, evaluate=False), symbol, ">=")
    if guard is not None:
        guard()
    pieces = []
    for piece in _chart_pieces(chart):
        clipped = _intersection(piece, interval, work)
        if clipped is not None:
            pieces.append(clipped)
    return expression, symbol, pieces


def _merge(pieces, work):
    def compare(left, right):
        result = work.compare(left[0], right[0])
        return result or ((left[2] > right[2]) - (left[2] < right[2]))

    pieces = sorted(pieces, key=cmp_to_key(compare))
    merged = []
    for piece in pieces:
        if not merged:
            merged.append(piece)
            continue
        previous = merged[-1]
        order = work.compare(piece[0], previous[1])
        if order > 0 or order == 0 and piece[2] and previous[3]:
            merged.append(piece)
            continue
        end = work.compare(piece[1], previous[1])
        upper = piece[1] if end > 0 else previous[1]
        right_open = piece[3] if end > 0 else previous[3] if end < 0 else piece[3] and previous[3]
        merged[-1] = previous[0], upper, previous[2], right_open
    return merged


def _as_set(pieces, work):
    # Bounds have already been ordered exactly. Lower their typed values
    # directly: the public Interval constructor consults str() for infinity
    # compatibility, which must not reinterpret a finite checked value here.
    parts = []
    for lower, upper, left_open, right_open in pieces:
        if work.compare(lower, upper) == 0:
            parts.append(_native.SymSet.finite([_native_expr(lower[1])]))
        else:
            parts.append(_native.SymSet.interval(
                _native_expr(_expression(lower)), _native_expr(_expression(upper)),
                left_open, right_open,
            ))
    return _wrap_set(_native.SymSet.union_many(parts))


def _range_pieces(expression, symbol, domain, work):
    from ..core import diff_with_receipt

    expression, symbol, components = _admit(expression, symbol, domain, work)
    if not components:
        return []
    numerator, denominator = _fraction(expression, symbol, work)
    derivative, _receipt = diff_with_receipt(expression, symbol)
    critical, _cells = _monotonicity_chart(derivative, symbol, ">=")
    pieces = []
    for lower, upper, left_open, right_open in components:
        if work.compare(lower, upper) == 0:
            value = _limit(numerator, denominator, lower, 1, work)
            if value[0]:
                raise ValueError("a defined rational point has an infinite value")
            pieces.append((value, value, False, False))
            continue
        candidates = [(_limit(numerator, denominator, lower, 1, work), not left_open),
                      (_limit(numerator, denominator, upper, -1, work), not right_open)]
        for point in critical:
            if work.compare(lower, point) < 0 and work.compare(point, upper) < 0:
                value = _limit(numerator, denominator, point, 1, work)
                if value[0]:
                    raise ValueError("a continuous rational component contains a pole")
                candidates.append((value, True))
        lo, lo_attained = candidates[0]
        hi, hi_attained = candidates[0]
        for value, attained in candidates[1:]:
            low_order, high_order = work.compare(value, lo), work.compare(value, hi)
            if low_order < 0:
                lo, lo_attained = value, attained
            elif low_order == 0:
                lo_attained = lo_attained or attained
            if high_order > 0:
                hi, hi_attained = value, attained
            elif high_order == 0:
                hi_attained = hi_attained or attained
        if work.compare(lo, hi) == 0:
            # A nonempty connected domain with this singleton image attains
            # its value even when both source endpoints are excluded.
            pieces.append((lo, hi, False, False))
        else:
            pieces.append((lo, hi, bool(lo[0]) or not lo_attained,
                           bool(hi[0]) or not hi_attained))
    return _merge(pieces, work)


def rational_stationary_points(expression, symbol, domain):
    """Complete stationary points of an admitted original rational source.

    Nonrational syntax and non-interval domains retain the existing public
    solver route, selected before any native mathematical check. Once this
    lane admits a source, failed checks and exhausted bounds are terminal.
    The derivative's two complete sign charts establish its exact zero set;
    the original domain is intersected independently before publication.
    """
    from ..core import Expr, _wrap, diff_with_receipt
    from ..polys.rootoftools import CRootOf

    if (domain is not None and domain is not Reals and domain is not EmptySet
            and type(domain) not in (Interval, Reals, EmptySet)):
        return STATIONARY_UNSUPPORTED
    if type(expression) in (int, str, Expr, _native.Expr):
        _require_original_surface()
        expression = _wrap(_native_expr(expression))
    kinds = (Add, Mul, Pow, Symbol, Dummy, *_ATOMS)
    if type(expression) not in kinds:
        return STATIONARY_UNSUPPORTED
    _require_original_surface()
    _require_closed_function_class(CRootOf)
    domain = _REALS if domain is None or domain is Reals else _EMPTY_SET if domain is EmptySet else domain
    domain_native = _STATIONARY_DOMAIN_SLOT.__get__(domain, Set)
    if type(domain_native) is not _native.SymSet:
        raise NotImplementedError("stationary points require valid original interval storage")
    pending = [(expression, 0)]
    records = []
    symbols = set()

    def snapshot(node):
        # Canonical nodes may lift fresh child wrappers on every args read.
        # Retain their actual slots, including absent held/structural slots,
        # rather than requiring those temporary wrappers to have one identity.
        storage = []
        for name in ("_args", "_surface_args", "_struct_args", "_assumptions",
                     "_dummy_name", "_dummy_number"):
            try:
                value = object.__getattribute__(node, name)
            except AttributeError:
                value = _MISSING
            storage.append((name, value))
        facts = None
        if type(node) in (Symbol, Dummy):
            assumptions = object.__getattribute__(node, "_assumptions")
            if (type(assumptions) is not dict or len(assumptions) > 64
                    or any(type(key) is not str or type(value) is not bool
                           for key, value in assumptions.items())):
                raise NotImplementedError("stationary points require literal symbol assumptions")
            facts = tuple(sorted(assumptions.items()))
        return node, _native_expr(node), storage, facts

    while pending:
        node, depth = pending.pop()
        if len(records) >= 2048 or depth > 64:
            raise NotImplementedError("exact stationary-point source limit exceeded")
        kind = type(node)
        if kind not in kinds:
            return STATIONARY_UNSUPPORTED
        children = node.args
        if (type(children) is not tuple or (kind is Pow and len(children) != 2)
                or (kind in (Symbol, Dummy, *_ATOMS) and children)):
            raise NotImplementedError("stationary points require valid original source arguments")
        if kind is Pow and type(children[1]) not in (Integer, Zero, One, NegativeOne):
            return STATIONARY_UNSUPPORTED
        if len(children) > 256 or len(children) > 2048 - len(records) - len(pending) - 1:
            raise NotImplementedError("exact stationary-point source limit exceeded")
        if kind in (Symbol, Dummy):
            symbols.add(node)
        records.append(snapshot(node))
        pending.extend((child, depth + 1) for child in children)
    if ((symbol is None and len(symbols) > 1)
            or (symbol is not None and type(symbol) not in (Symbol, Dummy))
            or (symbol is not None and symbols - {symbol})):
        return STATIONARY_UNSUPPORTED
    if symbol is not None:
        records.append(snapshot(symbol))

    def unchanged():
        _require_original_surface()
        _require_closed_function_class(CRootOf)
        if _STATIONARY_DOMAIN_SLOT.__get__(domain, Set) is not domain_native:
            raise NotImplementedError("exact stationary-point domain changed during checking")
        for node, native, storage, facts in records:
            if _native_expr(node) is not native:
                raise NotImplementedError("exact stationary-point source changed during checking")
            for name, original in storage:
                try:
                    current = object.__getattribute__(node, name)
                except AttributeError:
                    current = _MISSING
                if current is not original:
                    raise NotImplementedError("exact stationary-point source changed during checking")
            if facts is not None:
                assumptions = object.__getattribute__(node, "_assumptions")
                if (len(assumptions) != len(facts)
                        or any(dict.get(assumptions, key, _MISSING) is not value
                               for key, value in facts)):
                    raise NotImplementedError("stationary-point symbol assumptions changed during checking")

    work = _Work()
    try:
        expression, symbol, components = _admit(expression, symbol, domain, work, unchanged)
        unchanged()
        if not components:
            result = _as_set([], work)
            unchanged()
            return result
        derivative, _receipt = diff_with_receipt(expression, symbol)
        unchanged()
        roots, nonnegative = _monotonicity_chart(derivative, symbol, ">=")
        unchanged()
        other_roots, nonpositive = _monotonicity_chart(derivative, symbol, "<=")
        unchanged()
        if (len(roots) != len(other_roots) or len(nonnegative) != len(nonpositive)
                or any(work.compare(left, right) for left, right in zip(roots, other_roots))):
            raise ValueError("inconsistent stationary-point critical-root charts")
        zeros = _chart_pieces((roots, [left and right for left, right in
                                      zip(nonnegative, nonpositive)]))
        pieces = []
        for component in components:
            for zero in zeros:
                work.step()
                clipped = _intersection(component, zero, work)
                if clipped is not None:
                    pieces.append(clipped)
        result = _as_set(_merge(pieces, work), work)
        unchanged()
        return result
    except (OverflowError, ValueError) as exc:
        raise NotImplementedError(
            "exact rational stationary points could not be determined: %s" % exc
        ) from None


def rational_analysis(expression, symbol, domain, operation):
    """One bounded public request; exhausted/unsupported decisions refuse."""
    work = _Work()
    try:
        if operation == "domain":
            _, _, pieces = _admit(expression, symbol, domain, work)
            return _as_set(pieces, work)
        pieces = _range_pieces(expression, symbol, domain, work)
        if operation == "range":
            return _as_set(pieces, work)
        if pieces:
            bound = pieces[0][0] if operation == "minimum" else pieces[-1][1]
            return _expression(bound)
    except (OverflowError, ValueError) as exc:
        raise NotImplementedError("exact rational %s could not be determined: %s" % (operation, exc)) from None
    raise ValueError("%s value not defined for empty domain." % operation.capitalize())
