"""Bounded solution checking against the original scalar source.

Every original argument is admitted before substitution or subtraction can
erase its domain. In particular, the reciprocal in a held zero product,
zeroth power, or identical equation sides still has to be defined at the
candidate. An unresolved source domain keeps an otherwise zero residual
inconclusive. This is a diagnostic checker, not a portable proof certificate.
"""

from ..core import (
    Add, Atom, AtomicExpr, Basic, ComplexInfinity, Dummy, Eq, Expr, Float,
    Half, Integer, Mul, NegativeOne, Number, One, Pow, Rational, S, Symbol,
    Zero, _ExactDecimalFloat, _charge_nested_if_active, _closed_function_classes,
    _is_closed_function_class, _native, _native_expr, _wrap, expand,
)
from ..core._function_capabilities import _implementation, _owner_matches, _owner_record
from ..core._opaque_arithmetic import _children, stored_args
from ..simplify import simplify

_MAX_WORK = 65_536
_MAX_NODES = 8192
_MAX_DEPTH = 128
_MAX_RULES = 64
_MAX_INPUT_BYTES = 8192
_SCALARS = (Basic, Atom, AtomicExpr, Expr, Number, Integer, Rational, Zero, One,
            NegativeOne, Half, Float, _ExactDecimalFloat, ComplexInfinity,
            Symbol, Dummy, Add, Mul, Pow, Eq)
_OWNERS = tuple((owner, _owner_record(owner)) for owner in _SCALARS)
_MRO = type.__dict__["__mro__"]
_DICT = type.__dict__["__dict__"]
_OWNER_MROS = tuple((owner, _MRO.__get__(owner)) for owner in _SCALARS)
_READS = ("subs", "atoms", "free_symbols", "name", "p", "q", "__sub__", "__rsub__", "__neg__")
# The shared function capabilities cover symbolic hooks, but these public
# methods are also used by this checker. Include every original owner so an
# override on a built-in function or its base cannot manufacture a verdict.
_METHOD_OWNERS = {}
for _class in (*_SCALARS, *_closed_function_classes):
    for _owner in _MRO.__get__(_class):
        _METHOD_OWNERS[id(_owner)] = _owner
_METHODS = tuple((owner, name, _implementation(_DICT.__get__(owner).get(name)))
                 for owner in _METHOD_OWNERS.values() for name in _READS)
_ENTIRE_NAMES = frozenset({
    "sin", "cos", "exp", "sinh", "cosh", "erf", "erfc", "Abs", "abs",
    "conjugate", "re", "im",
})
_NAME = type.__dict__["__name__"]
_ILLEGAL = frozenset((S.NaN, S.ComplexInfinity, S.Infinity, S.NegativeInfinity))


class _Unsupported(Exception):
    pass


class _Work:
    def __init__(self):
        self.remaining = _MAX_WORK
        self.sizes = {}
        self.exhausted = False
        self.failure = None

    def charge(self, count=1):
        if count < 0 or count > self.remaining:
            self.exhausted = True
            raise _Unsupported
        self.remaining -= count
        try:
            _charge_nested_if_active(count)
        except Exception as error:
            self.failure = error
            raise


def _require_original():
    if any(not _owner_matches(owner, record) for owner, record in _OWNERS):
        raise _Unsupported
    for owner, expected in _OWNER_MROS:
        current = _MRO.__get__(owner)
        if len(current) != len(expected) or any(a is not b for a, b in zip(current, expected)):
            raise _Unsupported
    for owner, name, expected in _METHODS:
        current = _implementation(_DICT.__get__(owner).get(name))
        if len(current) != len(expected) or any(a is not b for a, b in zip(current, expected)):
            raise _Unsupported


def _scalar(value, work):
    if any(owner is Basic for owner in _MRO.__get__(type(value))):
        return value
    kind = type(value)
    if kind is int:
        bits = value.bit_length()
        if bits > 4096:
            raise _Unsupported
        work.charge(bits + 1)
    elif kind is str:
        if len(value) > _MAX_INPUT_BYTES:
            raise _Unsupported
        byte_count = len(value.encode("utf-8"))
        if byte_count > _MAX_INPUT_BYTES:
            raise _Unsupported
        work.charge(byte_count + 1)
        token = value.strip()
        if not token.isidentifier() and token not in ("-oo", "+oo"):
            digits = token[1:] if token[:1] in ("-", "+") else token
            if (not digits or len(digits) > 1234
                    or not all("0" <= digit <= "9" for digit in digits)
                    or int(token).bit_length() > 4096):
                # The native parser may evaluate arithmetic while parsing.
                # Admit names and bounded integers only; callers can supply
                # structured symbolic objects for other expressions.
                raise _Unsupported
    elif kind is not float and kind is not _native.Expr:
        raise _Unsupported
    return _wrap(_native_expr(value))


def _source_nodes(sources, work):
    """Read retained owners before any user-visible transformation."""
    pending = [(source, 0, None) for source in sources]
    seen, active = set(), set()
    records = []
    while pending:
        node, depth, children = pending.pop()
        work.charge()
        if children is not None:
            active.remove(id(node))
            seen.add(id(node))
            records.append((node, children))
            work.sizes[id(node)] = min(
                _MAX_WORK + 1, 1 + sum(work.sizes[id(child)] for child in children)
            )
            continue
        if depth > _MAX_DEPTH or len(records) >= _MAX_NODES:
            raise _Unsupported
        if id(node) in active:
            raise _Unsupported
        if id(node) in seen:
            continue
        cls = type(node)
        if not any(cls is scalar for scalar in _SCALARS) and not _is_closed_function_class(cls):
            raise _Unsupported
        children = stored_args(node)
        if children is None:
            children = _children(node)
        if type(children) is not tuple or len(children) > _MAX_NODES - len(records):
            raise _Unsupported
        if cls is Pow and len(children) != 2 or cls is Eq and len(children) != 2:
            raise _Unsupported
        if cls in (Integer, Rational, Zero, One, NegativeOne, Half):
            if max(abs(node.p).bit_length(), node.q.bit_length()) > 4096:
                raise _Unsupported
        if cls is Pow and type(children[1]) in (Integer, Zero, One, NegativeOne):
            if abs(children[1].p) > 4096:
                raise _Unsupported
        active.add(id(node))
        pending.append((node, depth, children))
        pending.extend((child, depth + 1, None) for child in children)
    return records


def _substitute(source, items, work):
    value = source
    size = work.sizes.get(id(source), _MAX_NODES)
    for old, new in items:
        # Substitution traverses both the source and replacement, and can
        # duplicate a replacement at every source node. Charge that upper
        # bound before the native transformation allocates the new tree.
        size = min(_MAX_WORK + 1, size * work.sizes.get(id(new), _MAX_NODES))
        work.charge(size + work.sizes.get(id(old), _MAX_NODES))
        value = value.subs(old, new)
    return value


def _power_parts(children, items, work):
    # Substitution is sequential. A later rule can replace a huge exponent
    # with zero, but the intervening exact power would already have been
    # materialized. Admit every prefix before substituting the parent power.
    for count in range(len(items) + 1):
        prefix = items[:count]
        base = _substitute(children[0], prefix, work)
        exponent = _substitute(children[1], prefix, work)
        if isinstance(exponent, Rational):
            if max(abs(exponent.p).bit_length(), exponent.q.bit_length()) > 4096:
                raise _Unsupported
            if abs(exponent.p) > 4096:
                raise _Unsupported
            if isinstance(base, Rational):
                if max(abs(base.p).bit_length(), base.q.bit_length()) * abs(exponent.p) > 65_536:
                    raise _Unsupported
        elif not exponent.free_symbols:
            # A nested, still unevaluated numeric exponent is not an
            # admitted bound on the cost of exact exponentiation.
            raise _Unsupported
    return base, exponent


def _domain_status(records, items, work):
    """False is an actual undefined value; None is an unresolved obligation."""
    unknown = False
    for node, children in records:
        work.charge()
        if type(node) is Eq:
            continue  # its original sides are separate source records
        if type(node) is Pow:
            base, exponent = _power_parts(children, items, work)
            # Nonnegative integer powers and positive real powers are defined
            # at zero. All other admitted powers need a nonzero base (or an
            # explicit determination at zero), including symbolic exponents.
            nonnegative_integer = exponent.is_integer is True and exponent.is_nonnegative is True
            if not nonnegative_integer and exponent.is_positive is not True:
                zero = base.is_zero
                if zero is None and base.args:
                    zero = simplify(base).is_zero
                if zero is True and exponent.is_negative is True:
                    return False
                unknown = unknown or zero is not False
        value = _substitute(node, items, work)
        if value.atoms() & _ILLEGAL:
            return False
        if _is_closed_function_class(type(node)):
            name = _NAME.__get__(type(node))
            if name in ("log", "ln") and len(children) == 1:
                if _substitute(children[0], items, work).is_zero is True:
                    return False
            # Entire built-ins introduce no new finite-input exclusion.
            # A remaining logarithm, meromorphic/special function or branch
            # operation needs its own established finite-domain fact.
            if name not in _ENTIRE_NAMES and value.is_finite is not True:
                unknown = True
    return None if unknown else True


def _residual_check(source, items, work):
    if type(source) is Eq:
        left, right = stored_args(source) or _children(source)
        subbed = _substitute(left, items, work) - _substitute(right, items, work)
    else:
        subbed = _substitute(source, items, work)
    if subbed.atoms() & _ILLEGAL:
        return False
    # Exact arithmetic already decides a literal residual; it needs no
    # heuristic rewrite or expansion and cannot acquire a different verdict
    # from one. Symbolic residuals retain the existing simplifier lane below.
    if isinstance(subbed, Rational):
        return subbed.p == 0
    simplified = simplify(subbed)
    if simplified.atoms() & _ILLEGAL:
        return False
    if simplified == 0 or simplified.is_zero is True:
        return True
    expanded = expand(subbed)
    if expanded == 0:
        return True
    zero_status = expanded.is_zero
    if zero_status is not None:
        return zero_status
    # Retain the existing rational-expression negative lane. An unresolved
    # function or radical is not disproved by failure to simplify it to zero.
    from ..core import Function

    for node, children in _source_nodes((expanded,), work):
        if isinstance(node, Function):
            return None
        if type(node) is Pow and not isinstance(children[1], Integer):
            return None
    return False


def _check_scalar(expression, symbol, val, work, *, domain_only=False):
    if type(symbol) is dict:
        if len(symbol) > _MAX_RULES:
            return None
        pairs = tuple(dict.items(symbol))
    elif val is not None:
        pairs = ((symbol, val),)
    else:
        raise ValueError("checksol requires either a mapping or (symbol, val)")
    if len(pairs) > _MAX_RULES:
        return None
    try:
        work.sizes.clear()
        _require_original()
        source = _scalar(expression, work)
        records = _source_nodes((source,), work)
        items = ()
        candidate_unknown = False
        # Numeric equations do not consume candidate bindings. Preserve the
        # established checksol(0, {x: oo}) == True contract.
        if source.free_symbols:
            items = tuple((_scalar(old, work), _scalar(new, work)) for old, new in pairs)
            _source_nodes(tuple(value for pair in items for value in pair), work)
            if domain_only:
                # Solver rows can contain free parameters with the identity
                # of another input coordinate. Sequential substitution would
                # capture that parameter and could manufacture a pole. These
                # original/candidate nodes have passed exact owner admission;
                # keep the ambiguous mapping inconclusive instead of running
                # any further substitution against a captured coordinate.
                for index, (_, candidate) in enumerate(items):
                    work.charge(work.sizes.get(id(candidate), _MAX_NODES) + len(items))
                    free = candidate.free_symbols
                    if any(other != index and old in free
                           for other, (old, _) in enumerate(items)):
                        return None
            for index, (_, candidate) in enumerate(items):
                candidate_nodes = _source_nodes((candidate,), work)
                if any(node.atoms() & _ILLEGAL for node, _ in candidate_nodes):
                    return False
                status = _domain_status(candidate_nodes, items[index + 1:], work)
                if status is False:
                    return False
                candidate_unknown = candidate_unknown or status is None
        domain = _domain_status(records, items, work)
        if domain is False:
            return False
        if domain_only:
            # solve's generators own their residual checks. This admission
            # only prevents original undefined subexpressions from vanishing
            # through normalization, including inexact sources before nfloat.
            _require_original()
            return None if domain is None or candidate_unknown else True
        result = _residual_check(source, items, work)
        _require_original()
        return None if result is True and (domain is None or candidate_unknown) else result
    except (_Unsupported, NotImplementedError, ValueError, TypeError, OverflowError, RecursionError):
        return None


def _check_conjunction(expression, symbol, val, work, *, domain_only=False):
    """Check a conjunction using the caller's cumulative work allowance."""
    pending = [(expression, 0, False)]
    active = set()
    visited = 0
    inconclusive = False
    while pending:
        equation, depth, exiting = pending.pop()
        if exiting:
            active.remove(id(equation))
            continue
        try:
            work.charge()
        except _Unsupported:
            return None
        visited += 1
        if visited > _MAX_NODES:
            return None
        if depth > _MAX_DEPTH:
            inconclusive = True
            continue
        if type(equation) in (list, tuple, set):
            if not equation:
                raise ValueError("no functions to check")
            if len(equation) > _MAX_RULES or id(equation) in active:
                inconclusive = True
                continue
            active.add(id(equation))
            pending.append((equation, depth, True))
            pending.extend((child, depth + 1, False) for child in reversed(tuple(equation)))
            continue
        result = _check_scalar(equation, symbol, val, work, domain_only=domain_only)
        # Public checksol keeps its fuzzy diagnostic behavior. A solver's
        # domain admission must not publish a row after an enclosing callback
        # budget failed inside the scalar check and was converted to unknown.
        if domain_only and work.failure is not None:
            raise work.failure
        if result is False:
            return False
        inconclusive = inconclusive or result is None
    return None if inconclusive else True


def _check_with_work(expression, symbol, val, work, *, domain_only=False):
    """Finish domain admission outside the scalar checker's fuzzy catches."""
    verdict = _check_conjunction(
        expression, symbol, val, work, domain_only=domain_only,
    )
    if domain_only:
        if work.failure is not None:
            raise work.failure
        # Core substitution can charge an enclosing callback session directly,
        # outside _Work. An inner failure may have become an unknown verdict;
        # the session's consumed allowance remains exhausted. Recheck it here
        # before a caller can admit even an inconclusive candidate. Keep the
        # local exhaustion flag/unknown contract for callers to translate.
        try:
            work.charge()
        except _Unsupported:
            return None
        if work.exhausted:
            return None
    return verdict


def checksol(expression, symbol, val=None):
    """Check original equations under one bounded, fuzzy conjunction."""
    return _check_with_work(expression, symbol, val, _Work())


class _SolveSourceCheck:
    """Retain original equations and share bounded domain checking across rows."""

    def __init__(self, expression):
        if type(expression) in (list, tuple):
            if len(expression) > _MAX_NODES:
                raise ValueError("solve source exceeds its equation limit")
            # The generator may receive the caller's mutable list. Keep the
            # original equation owners even if that list changes meanwhile.
            expression = tuple(expression)
        self.source = expression
        self.work = _Work()

    def filter(self, solutions, variables):
        if len(solutions) > _MAX_NODES:
            raise ValueError("solve source checking exceeds its candidate limit")
        admitted = []
        for solution in solutions:
            if len(solution) != len(variables):
                raise ValueError("solve result does not match its requested coordinates")
            try:
                self.work.charge()
            except _Unsupported:
                raise ValueError("solve source checking exceeds its work limit") from None
            # Keep the checker's existing finite substitution envelope. A
            # larger mapping has no verdict, rather than a partial check
            # which silently ignores one of its coordinate assignments.
            if len(variables) > _MAX_RULES:
                admitted.append(solution)
                continue
            replacements = {
                variable: root for variable, root in zip(variables, solution)
                if root is not variable
            }
            verdict = _check_with_work(
                self.source, replacements, None, self.work, domain_only=True,
            )
            if self.work.exhausted:
                raise ValueError("solve source checking exceeds its work limit")
            if verdict is not False:
                admitted.append(solution)
        return admitted
