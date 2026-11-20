"""Exact real images of elementary functions of one rational expression.

The rational leaf supplies a complete checked image, including its original
held poles. Intersecting that image with the outer function's real branch and
applying a strictly monotone map is an exact set image, not interval arithmetic
on independent operands. Absolute value and cosh first fold the algebraic
image at zero. All ordering and merging precedes transcendental evaluation.

This shell composition does not issue a new portable mathematical certificate.
Its intrinsic bounds and the native checks' meters remain separate.
"""

from fractions import Fraction

from ..core import (
    Dummy, Expr, Integer, Rational, Symbol,
    _native, _native_expr, _wrap,
)
from ..sets.sets import _wrap_set
from . import _rational_range as _rational
from ._elementary_domain import _Source
from ._rational_range import (
    STATIONARY_UNSUPPORTED, _NEGATIVE_INFINITY, _POSITIVE_INFINITY,
    _domain_pieces, _expression, _finite, _intersection, _merge, _range_pieces,
    _rational_analysis_request, _rational_source_request, _require_original_surface,
)


_FUNCTIONS = frozenset({
    "exp", "log", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    "asin", "acos", "atan", "asech", "Abs",
})


def _unsupported():
    raise NotImplementedError(
        "exact elementary range requires a supported function of one rational "
        "expression, optionally scaled and translated by rational constants"
    )


def _scalar(node, work):
    if node.tag != "number":
        _unsupported()
    return work.scalar(Fraction(node.native.exact_numerator(), node.native.exact_denominator()))


def _select(node, work):
    """Select the whole grammar before running mathematical algorithms."""
    work.step()
    if node.rational:
        return node, "identity", None, Fraction(1), Fraction(0)
    if node.tag in _FUNCTIONS and node.children[0].rational:
        return node.children[0], node.tag, None, Fraction(1), Fraction(0)
    if node.tag == "Pow":
        base = node.children[0]
        if base.rational:
            return base, "Pow", node.power, Fraction(1), Fraction(0)
        if node.power in (0, 1):
            leaf, tag, power, scale, offset = _select(base, work)
            if node.power == 0:
                scale, offset = Fraction(0), Fraction(1)
            return leaf, tag, power, scale, offset
        _unsupported()
    if node.tag not in ("Add", "Mul"):
        _unsupported()
    varying = [child for child in node.children if not child.rational]
    if len(varying) != 1:
        _unsupported()
    leaf, tag, power, scale, offset = _select(varying[0], work)
    constant = Fraction(0 if node.tag == "Add" else 1)
    for child in node.children:
        if child is varying[0]:
            continue
        constant = work.binary(constant, _scalar(child, work), node.tag == "Mul")
    if node.tag == "Add":
        offset = work.binary(offset, constant)
    else:
        scale, offset = work.binary(scale, constant, True), work.binary(offset, constant, True)
    return leaf, tag, power, scale, offset


def _negate(bound, work):
    if bound[0]:
        return -bound[0], None
    if type(bound[1]) in _rational._ATOMS:
        value = work.scalar(Fraction(bound[1].p, bound[1].q))
        return _finite(_rational._rational(work.binary(Fraction(-1), value, True)))
    return _finite(_wrap(_native.py_mul(_native.py_integer(-1), _native_expr(bound[1]))))


def _clip(pieces, interval, work):
    result = []
    for piece in pieces:
        work.step()
        clipped = _intersection(piece, interval, work)
        if clipped is not None:
            result.append(clipped)
    return result


def _fold_absolute(pieces, work):
    zero = _finite(Integer(0))
    positive = _clip(pieces, (zero, _POSITIVE_INFINITY, False, True), work)
    for lower, upper, left_open, right_open in _clip(
            pieces, (_NEGATIVE_INFINITY, zero, True, False), work):
        positive.append((_negate(upper, work), _negate(lower, work), right_open, left_open))
    return _merge(positive, work)


def _branch_image(pieces, tag, power, work):
    zero, one, minus_one = (_finite(Integer(value)) for value in (0, 1, -1))
    if tag in ("Abs", "cosh"):
        return _fold_absolute(pieces, work), 1
    if tag == "log":
        interval = zero, _POSITIVE_INFINITY, True, True
    elif tag in ("asin", "acos", "atanh"):
        interval = minus_one, one, tag == "atanh", tag == "atanh"
    elif tag == "acosh":
        interval = one, _POSITIVE_INFINITY, False, True
    elif tag == "asech":
        interval = zero, one, True, False
    elif tag == "Pow":
        # SymPy's continuity convention for odd denominator powers does not
        # make a negative principal power real. Do not silently discard those
        # defined nonreal values and call the remaining set the real image.
        if power.denominator % 2 and any(work.compare(piece[0], zero) < 0 for piece in pieces):
            raise NotImplementedError("a principal fractional power is nonreal on negative inputs")
        interval = zero, _POSITIVE_INFINITY, power < 0, True
    else:
        return pieces, 1
    direction = -1 if tag in ("acos", "asech") or tag == "Pow" and power < 0 else 1
    return _clip(pieces, interval, work), direction


def _evaluate(tag, value):
    native = _native_expr(value)
    evaluated = _native.py_eval_function(tag, [native])
    if evaluated is None:
        evaluated = _native.py_function(tag, native)
    return _finite(_wrap(evaluated))


def _endpoint(bound, tag, power, work):
    """Exact values and one-sided real limits at the branch endpoints."""
    work.step()
    if tag == "Abs":
        return bound  # The algebraic input image has already been folded.
    if bound[0]:
        sign = bound[0]
        if tag in ("sinh", "asinh"):
            return bound
        if tag in ("cosh", "acosh"):
            return _POSITIVE_INFINITY
        if tag == "exp":
            return _finite(Integer(0)) if sign < 0 else _POSITIVE_INFINITY
        if tag == "log":
            return _POSITIVE_INFINITY
        if tag == "tanh":
            return _finite(Integer(sign))
        if tag == "atan":
            return _finite(_wrap(_native.py_mul(
                _native.py_rational(sign, 2), _native_expr(Expr("pi")),
            )))
        if tag == "Pow":
            return _finite(Integer(0)) if power < 0 else _POSITIVE_INFINITY
        raise ValueError("elementary image escaped its finite real branch")
    value = bound[1]
    zero = _finite(Integer(0))
    if tag in ("log", "asech", "Pow") and work.compare(bound, zero) == 0:
        if tag == "log":
            return _NEGATIVE_INFINITY
        if tag == "asech" or power < 0:
            return _POSITIVE_INFINITY
        return zero
    if tag == "atanh":
        for sign in (-1, 1):
            if work.compare(bound, _finite(Integer(sign))) == 0:
                return sign, None
    if tag == "Pow":
        return _finite(_wrap(_native.py_pow(
            _native_expr(value), _native.py_rational(power.numerator, power.denominator),
        )))
    return _evaluate(tag, value)


def _affine(bound, scale, offset, work):
    if bound[0]:
        return (bound[0] if scale > 0 else -bound[0]), None
    if scale == 1 and offset == 0:
        return bound
    if type(bound[1]) in _rational._ATOMS:
        value = work.scalar(Fraction(bound[1].p, bound[1].q))
        value = work.binary(work.binary(scale, value, True), offset)
        return _finite(_rational._rational(value))
    native = _native_expr(bound[1])
    if scale != 1:
        native = _native.py_mul(_native_expr(_rational._rational(scale)), native)
    if offset:
        native = _native.py_add(native, _native_expr(_rational._rational(offset)))
    return _finite(_wrap(native))


def _render(pieces, operation):
    if operation != "range":
        if not pieces:
            raise ValueError("%s value not defined for empty domain." % operation.capitalize())
        return _expression(pieces[0][0] if operation == "minimum" else pieces[-1][1])
    parts = []
    for lower, upper, left_open, right_open, singleton in pieces:
        if singleton:
            parts.append(_native.SymSet.finite([_native_expr(lower[1])]))
        else:
            parts.append(_native.SymSet.interval(
                _native_expr(_expression(lower)), _native_expr(_expression(upper)),
                left_open, right_open,
            ))
    return _wrap_set(_native.SymSet.union_many(parts))


def elementary_analysis(expression, symbol, domain, operation):
    """Select an exact rational or elementary image; failures stay terminal."""
    # Both lanes retain one request-wide ledger through source selection,
    # domain partitioning, rational images and elementary endpoint mapping.
    work = _rational._Work()
    try:
        request = _rational_source_request(expression, symbol, domain, work)
    except (OverflowError, ValueError) as exc:
        raise NotImplementedError(
            "exact rational %s could not be determined: %s" % (operation, exc)
        ) from None
    if request is not STATIONARY_UNSUPPORTED:
        return _rational_analysis_request(request, domain, operation, work)
    _require_original_surface()
    if type(expression) is str:
        if len(expression) > 4096:
            raise ValueError("elementary range source text exceeds 4096 characters")
        expression = _wrap(_native_expr(expression))
    capture = _Source()
    tree = capture.read(expression)
    if tree.rational:
        # Keep existing unsupported-domain behavior for the rational API.
        return _rational_analysis_request(request, domain, operation, work)
    leaf, tag, power, scale, offset = _select(tree, work)
    source_symbols = tuple(capture.capture.symbols)
    if symbol is None:
        if len(source_symbols) > 1:
            _unsupported()
        symbol = source_symbols[0] if source_symbols else Dummy("_fsym_range_coordinate")
    if type(symbol) is not Symbol and type(symbol) is not Dummy:
        raise TypeError("elementary range variable must be an exact Symbol or Dummy")
    capture.symbol(symbol)
    if any(coordinate != symbol for coordinate in source_symbols):
        _unsupported()
    try:
        request = _rational_source_request(_wrap(leaf.native), symbol, domain, work)
    except (OverflowError, ValueError) as exc:
        raise NotImplementedError(
            "exact elementary %s could not be determined: %s" % (operation, exc)
        ) from None
    if request is STATIONARY_UNSUPPORTED:
        _unsupported()
    base, symbol, domain, base_unchanged = request

    def unchanged():
        capture.unchanged()
        base_unchanged()

    try:
        unchanged()
        components = _domain_pieces(domain, work, unchanged)
        unchanged()
        if components is STATIONARY_UNSUPPORTED:
            _unsupported()
        pieces = _range_pieces(base, symbol, components, work, unchanged)
        unchanged()
        pieces, direction = _branch_image(pieces, tag, power, work)
        mapped = []
        for lower, upper, left_open, right_open in pieces:
            singleton = work.compare(lower, upper) == 0
            lo, hi = _endpoint(lower, tag, power, work), _endpoint(upper, tag, power, work)
            if direction < 0:
                lo, hi, left_open, right_open = hi, lo, right_open, left_open
            mapped.append((lo, hi, bool(lo[0]) or left_open, bool(hi[0]) or right_open, singleton))
        if direction < 0:
            mapped.reverse()
        if scale == 0 and mapped:
            value = _finite(Rational(offset.numerator, offset.denominator))
            mapped = [(value, value, False, False, True)]
        elif scale != 0:
            translated = []
            for lower, upper, left_open, right_open, singleton in mapped:
                work.step()
                lo, hi = _affine(lower, scale, offset, work), _affine(upper, scale, offset, work)
                if scale < 0:
                    lo, hi, left_open, right_open = hi, lo, right_open, left_open
                translated.append((lo, hi, left_open, right_open, singleton))
            mapped = translated if scale > 0 else list(reversed(translated))
        unchanged()
    except (OverflowError, ValueError) as exc:
        raise NotImplementedError(
            "exact elementary %s could not be determined: %s" % (operation, exc)
        ) from None
    result = _render(mapped, operation)
    unchanged()
    return result
