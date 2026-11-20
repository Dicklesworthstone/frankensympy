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
    Zero, _charge_nested_if_active, _native, _native_expr,
    _require_closed_function_class, _surface_binding_key, _symbol_binding, oo,
)
from ..sets.sets import (
    Complement, EmptySet, FiniteSet, Intersection, Interval, Reals, Set, Union,
    _EMPTY_SET, _REALS, _wrap_set,
)
from ..core._function_capabilities import _implementation, _owner_matches, _owner_record
from .singularities import _monotonicity_bound, _monotonicity_chart


_ATOMS = (Integer, Rational, Zero, One, NegativeOne, Half)
_DOMAINS = (Interval, Reals, EmptySet, FiniteSet, Union, Intersection, Complement)
_MISSING = object()
_EXACT_VALUE_NAMES = (
    "p", "q", "__int__", "__neg__", "__truediv__", "__rtruediv__",
    "__add__", "__radd__", "__mul__", "__rmul__", "__pow__", "__str__",
    "_assumptions", "_dummy_name", "_dummy_number", "name", "dummy_index",
    "_native_set", "_py_args", "_surface_elements",
    "start", "end", "left_open", "right_open",
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
        owner for cls in (Add, Mul, Pow, Symbol, Dummy, *_ATOMS, *_DOMAINS)
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


def _check_atom(kind, native):
    if not (native.is_integer if kind not in (Rational, Half)
            else native.is_integer or native.is_rational):
        raise NotImplementedError("exact rational number does not match its native value")
    fixed = {Zero: (0, 1), One: (1, 1), NegativeOne: (-1, 1), Half: (1, 2)}.get(kind)
    if fixed is not None and (native.exact_numerator(), native.exact_denominator()) != fixed:
        raise NotImplementedError("exact rational constant does not match its native value")


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
        _charge_nested_if_active(amount)

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
        self.step()
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


def _domain_pieces(domain, work, guard):
    """Partition one exact real set expression without sampling its gaps.

    The existing domain reader bounds and snapshots the complete native tree,
    including every complement child, before any decision. An ordered list of
    its exact boundaries then makes membership constant on each point/open
    gap cell. Set operations become bounded Boolean operations on that common
    partition, avoiding recursive solver calls or expanding intersections into
    a Cartesian product of unions.
    """
    from ..core import _wrap
    from ..polys.rootoftools import CRootOf
    from ..solvers._real_domain import _Domain

    _require_closed_function_class(CRootOf)
    _require_original_surface()
    work.step()
    snapshot = _Domain(domain)
    guard()
    if not snapshot.real:
        return STATIONARY_UNSUPPORTED

    endpoints = {}
    for native, descriptor in snapshot.domain_values:
        work.step()
        if descriptor is None:
            return STATIONARY_UNSUPPORTED
        value = _wrap(native)
        if type(value) not in (*_ATOMS, CRootOf):
            # Range evaluation currently accepts rational arithmetic in
            # genuine indexed roots. Other exact/symbolic endpoint grammars
            # remain outside this lane even if a solver can compare them.
            if descriptor[0] != "infinity":
                return STATIONARY_UNSUPPORTED
        bound = _monotonicity_bound(value)
        if descriptor[0] != "infinity":
            endpoints.setdefault(descriptor, bound)

    def compare(left, right):
        work.step()
        order = snapshot.compare(left, right)
        if order is None:
            raise NotImplementedError("exact rational domain boundary order is undecided")
        return order

    ordered = sorted(endpoints, key=cmp_to_key(compare))
    indices, boundaries = {}, []
    previous = None
    for descriptor in ordered:
        work.step()
        if previous is None or compare(previous, descriptor):
            boundaries.append(endpoints[descriptor])
            previous = descriptor
        # Distinct polynomials or bound names may denote the same real root.
        # Every alias receives the same point cell, by checked exact order.
        indices[descriptor] = len(boundaries) - 1
    width = 2*len(boundaries) + 1

    def partition(tree):
        work.step(width)
        kind = tree[0]
        if kind == "EmptySet":
            return [False]*width
        if kind == "UniversalSet":
            # This is an internal real-line cell universe. The complete
            # outer domain was separately required to be a subset of Reals.
            return [True]*width
        if kind == "Interval":
            lower, upper = tree[1:3]
            first = ((0 if lower[1] < 0 else width) if lower[0] == "infinity"
                     else 2*indices[lower] + (2 if tree[3] else 1))
            stop = ((0 if upper[1] < 0 else width) if upper[0] == "infinity"
                    else 2*indices[upper] + (1 if tree[4] else 2))
            return [first <= position < stop for position in range(width)]
        if kind == "FiniteSet":
            cells = [False]*width
            for descriptor in tree[1]:
                work.step()
                cells[2*indices[descriptor] + 1] = True
            return cells
        if kind == "Complement":
            return [not cell for cell in partition(tree[1][0])]
        cells = [kind == "Intersection"]*width
        for child in tree[1]:
            other = partition(child)
            work.step(width)
            if kind == "Union":
                cells = [left or right for left, right in zip(cells, other)]
            else:
                cells = [left and right for left, right in zip(cells, other)]
        return cells

    cells = partition(snapshot.tree)
    snapshot.unchanged()
    guard()
    work.step(width)
    return _chart_pieces((boundaries, cells))


def _intersect_pieces(left, right, work):
    """Intersect sorted disjoint interval/point lists in one bounded sweep."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        work.step()
        clipped = _intersection(left[i], right[j], work)
        if clipped is not None:
            result.append(clipped)
        order = work.compare(left[i][1], right[j][1])
        if order <= 0:
            i += 1
        if order >= 0:
            j += 1
    return result


def _admit(expression, symbol, components, work, guard):
    if not components:
        return []
    chart = _monotonicity_chart(Mul(0, expression, evaluate=False), symbol, ">=")
    guard()
    return _intersect_pieces(_chart_pieces(chart), components, work)


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


def _range_pieces(expression, symbol, components, work, guard):
    from ..core import diff_with_receipt

    components = _admit(expression, symbol, components, work, guard)
    if not components:
        return []
    numerator, denominator = _fraction(expression, symbol, work)
    derivative, _receipt = diff_with_receipt(expression, symbol)
    if guard is not None:
        guard()
    critical, _cells = _monotonicity_chart(derivative, symbol, ">=")
    if guard is not None:
        guard()
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


def _rational_source_request(expression, symbol, domain, work):
    """Select a rational lane and retain its original source until return.

    Unsupported grammar is decided before mathematical work. A selected
    request never converts an operational or verification failure into an
    unsupported result. Stationary points, extrema and images share these
    exact-class, typed-symbol and mutable-source checks.
    """
    from ..core import Expr, _wrap
    from ..polys.rootoftools import CRootOf

    if (domain is not None and domain is not Reals and domain is not EmptySet
            and not any(type(domain) is owner for owner in _DOMAINS)):
        return STATIONARY_UNSUPPORTED
    if any(type(expression) is owner for owner in (int, str, Expr, _native.Expr)):
        _require_original_surface()
        expression = _wrap(_native_expr(expression))
    kinds = (Add, Mul, Pow, Symbol, Dummy, *_ATOMS)
    if not any(type(expression) is owner for owner in kinds):
        return STATIONARY_UNSUPPORTED
    _require_original_surface()
    _require_closed_function_class(CRootOf)
    domain = _REALS if domain is None or domain is Reals else _EMPTY_SET if domain is EmptySet else domain
    domain_native = _STATIONARY_DOMAIN_SLOT.__get__(domain, Set)
    if domain_native is None and type(domain) in (Union, Intersection, Complement):
        return STATIONARY_UNSUPPORTED
    if type(domain_native) is not _native.SymSet:
        raise NotImplementedError("exact rational analysis requires valid original domain storage")
    expected_kind = next(name for owner, name in (
        (Interval, "Interval"), (Reals, "Interval"), (EmptySet, "EmptySet"),
        (FiniteSet, "FiniteSet"), (Union, "Union"),
        (Intersection, "Intersection"), (Complement, "Complement"),
    ) if type(domain) is owner)
    if domain_native.kind != expected_kind:
        raise NotImplementedError("exact rational domain class does not match its native storage")
    domain_native.real_domain_preflight()
    if (type(domain) is Reals and
            (domain_native.start != _native_expr(-oo) or domain_native.end != _native_expr(oo)
             or not domain_native.left_open or not domain_native.right_open)):
        raise NotImplementedError("exact rational Reals domain does not match its native storage")
    try:
        domain_arguments = object.__getattribute__(domain, "_py_args")
    except AttributeError:
        domain_arguments = _MISSING
    if domain_arguments is not _MISSING and domain_arguments is not None:
        raise NotImplementedError("exact rational domain does not match its native storage")
    try:
        domain_elements = object.__getattribute__(domain, "_surface_elements")
    except AttributeError:
        domain_elements = _MISSING
    pending = [(expression, 0)]
    records = []
    domain_records = []
    arithmetic = []
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
                raise NotImplementedError("exact rational analysis requires literal symbol assumptions")
            facts = tuple(sorted(assumptions.items()))
        return node, _native_expr(node), storage, facts

    if domain_elements is not _MISSING and domain_elements is not None:
        if (type(domain) is not FiniteSet or type(domain_elements) is not tuple
                or len(domain_elements) > 128):
            raise NotImplementedError("exact rational domain requires bounded finite points in original storage")
        # Preflight the borrowed graph before cloning members from native
        # storage. FiniteSet keeps separate Python objects for their visible
        # identity, so stale or forged tuples cannot silently use old members.
        if domain_native.kind != "FiniteSet":
            raise NotImplementedError("finite domain does not match its native storage")
        members = domain_native.elements
        if len(members) != len(domain_elements):
            raise NotImplementedError("finite domain elements do not match their native storage")
        for member, native in zip(domain_elements, members):
            work.step()
            if not any(type(member) is owner for owner in (*_ATOMS, CRootOf)):
                return STATIONARY_UNSUPPORTED
            if _native_expr(member) != native:
                raise NotImplementedError("finite domain elements do not match their native storage")
        member_nodes = [(member, 0) for member in domain_elements]
        while member_nodes:
            work.step()
            node, depth = member_nodes.pop()
            if len(domain_records) >= 16384 or depth > 64:
                raise NotImplementedError("finite domain scalar source limit exceeded")
            kind = type(node)
            if not any(kind is owner for owner in (*_ATOMS, CRootOf, Add, Mul, Pow, Symbol, Dummy)):
                raise NotImplementedError("finite domain elements require original exact scalar owners")
            children = node.args
            if type(children) is not tuple or len(children) > 256:
                raise NotImplementedError("finite domain elements require bounded original arguments")
            record = snapshot(node)
            native = record[1]
            if kind in (Symbol, Dummy):
                if (children or not native.is_symbol or _surface_binding_key(native.symbol_binding)
                        != _surface_binding_key(_symbol_binding(node))):
                    raise NotImplementedError("finite domain symbol has inconsistent typed assumptions")
            elif kind in _ATOMS:
                if children:
                    raise NotImplementedError("finite domain number does not match its native value")
                _check_atom(kind, native)
            elif kind is CRootOf:
                native_children = native.args
                if (native.func_name != "CRootOf" or len(children) != 2
                        or len(native_children) != len(children)
                        or any(_native_expr(child) != stored for child, stored
                               in zip(children, native_children))):
                    raise NotImplementedError("finite domain root does not match its original arguments")
            else:
                operation = "add" if kind is Add else "mul" if kind is Mul else "pow"
                _native.ideal_source_arguments_native(
                    native, operation, [_native_expr(child) for child in children],
                )
            domain_records.append(record)
            if len(children) > 16384-len(domain_records)-len(member_nodes):
                raise NotImplementedError("finite domain scalar source limit exceeded")
            member_nodes.extend((child, depth + 1) for child in children)

    while pending:
        work.step()
        node, depth = pending.pop()
        if len(records) >= 2048 or depth > 64:
            raise NotImplementedError("exact rational source limit exceeded")
        kind = type(node)
        if not any(kind is owner for owner in kinds):
            return STATIONARY_UNSUPPORTED
        children = node.args
        if (type(children) is not tuple or (kind is Pow and len(children) != 2)
                or (kind in (Symbol, Dummy, *_ATOMS) and children)):
            raise NotImplementedError("exact rational analysis requires valid original source arguments")
        if kind is Pow and not any(type(children[1]) is owner
                                   for owner in (Integer, Zero, One, NegativeOne)):
            return STATIONARY_UNSUPPORTED
        if len(children) > 256 or len(children) > 2048 - len(records) - len(pending) - 1:
            raise NotImplementedError("exact rational source limit exceeded")
        if kind in (Symbol, Dummy):
            symbols.add(node)
        record = snapshot(node)
        native = record[1]
        if kind in (Symbol, Dummy):
            if (not native.is_symbol or _surface_binding_key(native.symbol_binding)
                    != _surface_binding_key(_symbol_binding(node))):
                raise NotImplementedError("exact rational symbol has inconsistent typed assumptions")
        elif kind in (Add, Mul, Pow):
            operation = "add" if kind is Add else "mul" if kind is Mul else "pow"
            arithmetic.append((native, operation, children))
        else:
            _check_atom(kind, native)
        records.append(record)
        pending.extend((child, depth + 1) for child in children)
    if ((symbol is None and len(symbols) > 1)
            or (symbol is not None and type(symbol) is not Symbol and type(symbol) is not Dummy)
            or (symbol is not None and symbols - {symbol})):
        return STATIONARY_UNSUPPORTED
    # Select the complete grammar before running a structural check whose
    # contract deliberately refuses Float/function atoms. Such unsupported
    # syntax must still remain a formal imageset, rather than an attempted
    # mathematical computation with a swallowed failure.
    for native, operation, children in arithmetic:
        work.step(len(children) + 1)
        _native.ideal_source_arguments_native(
            native, operation, [_native_expr(child) for child in children],
        )
    if symbol is None:
        symbol = next(iter(symbols)) if symbols else Symbol("x")
    coordinate = snapshot(symbol)
    if (not coordinate[1].is_symbol or _surface_binding_key(coordinate[1].symbol_binding)
            != _surface_binding_key(_symbol_binding(symbol))):
        raise NotImplementedError("exact rational coordinate has inconsistent typed assumptions")
    records.append(coordinate)
    records.extend(domain_records)

    def unchanged():
        _require_original_surface()
        _require_closed_function_class(CRootOf)
        if _STATIONARY_DOMAIN_SLOT.__get__(domain, Set) is not domain_native:
            raise NotImplementedError("exact rational domain changed during checking")
        try:
            current_arguments = object.__getattribute__(domain, "_py_args")
        except AttributeError:
            current_arguments = _MISSING
        if current_arguments is not domain_arguments:
            raise NotImplementedError("exact rational domain changed during checking")
        try:
            current_elements = object.__getattribute__(domain, "_surface_elements")
        except AttributeError:
            current_elements = _MISSING
        if current_elements is not domain_elements:
            raise NotImplementedError("exact rational finite domain elements changed during checking")
        for node, native, storage, facts in records:
            if _native_expr(node) is not native:
                raise NotImplementedError("exact rational source changed during checking")
            for name, original in storage:
                try:
                    current = object.__getattribute__(node, name)
                except AttributeError:
                    current = _MISSING
                if current is not original:
                    raise NotImplementedError("exact rational source changed during checking")
            if facts is not None:
                assumptions = object.__getattribute__(node, "_assumptions")
                if (len(assumptions) != len(facts)
                        or any(dict.get(assumptions, key, _MISSING) is not value
                               for key, value in facts)):
                    raise NotImplementedError("exact rational symbol assumptions changed during checking")

    return expression, symbol, domain, unchanged


def rational_image(expression, symbol, domain, guard=None):
    """Return an exact real set image, or decline unsupported input syntax.

    This selector is used by imageset, where unsupported syntax remains a
    formal ImageSet. Bounds and source grammar are selected before running
    the checked range engine; all failures after selection stay terminal.
    """
    work = _Work()
    try:
        request = _rational_source_request(expression, symbol, domain, work)
        if request is STATIONARY_UNSUPPORTED:
            return STATIONARY_UNSUPPORTED
        expression, symbol, domain, source_unchanged = request

        def unchanged():
            source_unchanged()
            if guard is not None:
                guard()

        unchanged()
        components = _domain_pieces(domain, work, unchanged)
        unchanged()
        if components is STATIONARY_UNSUPPORTED:
            return STATIONARY_UNSUPPORTED
        pieces = _range_pieces(expression, symbol, components, work, unchanged)
        unchanged()
        result = _as_set(pieces, work)
        unchanged()
        return result
    except (OverflowError, ValueError) as exc:
        raise NotImplementedError("exact rational image could not be determined: %s" % exc) from None


def rational_stationary_points(expression, symbol, domain):
    """Complete stationary points of an admitted original rational source.

    Nonrational syntax and unsupported set domains retain the existing public
    solver route, selected before any native mathematical check. Once this
    lane admits a source, failed checks and exhausted bounds are terminal.
    The derivative's two complete sign charts establish its exact zero set;
    the original domain is intersected independently before publication.
    """
    from ..core import diff_with_receipt

    work = _Work()
    try:
        request = _rational_source_request(expression, symbol, domain, work)
        if request is STATIONARY_UNSUPPORTED:
            return STATIONARY_UNSUPPORTED
        expression, symbol, domain, unchanged = request
        unchanged()
        components = _domain_pieces(domain, work, unchanged)
        unchanged()
        if components is STATIONARY_UNSUPPORTED:
            return STATIONARY_UNSUPPORTED
        components = _admit(expression, symbol, components, work, unchanged)
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
        pieces = _intersect_pieces(components, zeros, work)
        result = _as_set(pieces, work)
        unchanged()
        return result
    except (OverflowError, ValueError) as exc:
        raise NotImplementedError(
            "exact rational stationary points could not be determined: %s" % exc
        ) from None


def _rational_analysis_request(request, domain, operation, work):
    """Complete one retained request without resetting its work ledger."""
    try:
        if request is STATIONARY_UNSUPPORTED:
            if (domain is not None and domain is not Reals and domain is not EmptySet
                    and not isinstance(domain, Set)):
                raise TypeError("exact rational analysis domain must be a real set")
            raise NotImplementedError("exact rational analysis requires a supported rational source and real domain")
        expression, symbol, domain, unchanged = request
        unchanged()
        components = _domain_pieces(domain, work, unchanged)
        unchanged()
        if components is STATIONARY_UNSUPPORTED:
            raise NotImplementedError("exact rational analysis requires rational or real CRootOf domain boundaries")
        if operation == "domain":
            pieces = _admit(expression, symbol, components, work, unchanged)
            result = _as_set(pieces, work)
            unchanged()
            return result
        pieces = _range_pieces(expression, symbol, components, work, unchanged)
        unchanged()
        if operation == "range":
            result = _as_set(pieces, work)
            unchanged()
            return result
        if pieces:
            bound = pieces[0][0] if operation == "minimum" else pieces[-1][1]
            result = _expression(bound)
            unchanged()
            return result
    except (OverflowError, ValueError) as exc:
        raise NotImplementedError("exact rational %s could not be determined: %s" % (operation, exc)) from None
    raise ValueError("%s value not defined for empty domain." % operation.capitalize())


def rational_analysis(expression, symbol, domain, operation):
    """One bounded public request; exhausted/unsupported decisions refuse."""
    work = _Work()
    try:
        request = _rational_source_request(expression, symbol, domain, work)
    except (OverflowError, ValueError) as exc:
        raise NotImplementedError("exact rational %s could not be determined: %s" % (operation, exc)) from None
    return _rational_analysis_request(request, domain, operation, work)
