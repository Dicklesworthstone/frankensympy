"""Exact rational inequalities with complete real algebraic boundaries.

The native checker determines the complete ordered sign chart, retaining
every original denominator restriction. This module only lifts its checked
critical roots and joins adjacent admitted cells. Comparisons of floating
approximations play no role in root identity, order, or interval inclusion.
"""

from ..core import (
    Add, Dummy, Ge, Gt, Half, Integer, Le, Lt, Mul, NegativeOne, One,
    Pow, Rational, Symbol, Zero, _lift_native_result, _native, _native_expr,
    _require_symbol, _symbol_binding, oo,
)


def _rational_surface(expression, symbol):
    """Admit the exact closed grammar before lowering any custom surface."""
    pending = [(expression, 0)]
    nodes = 0
    while pending:
        node, depth = pending.pop()
        nodes += 1
        if nodes > 2048 or depth > 64:
            raise NotImplementedError("exact rational inequality source limit exceeded")
        kind = type(node)
        if kind in (Integer, Rational, Zero, One, NegativeOne, Half):
            continue
        if kind in (Symbol, Dummy):
            if node != symbol:
                return False
            continue
        if kind not in (Add, Mul, Pow):
            return False
        children = node.args
        if kind is Pow and type(children[1]) not in (Integer, Zero, One, NegativeOne):
            return False
        if len(children) > 256 or len(pending) + len(children) > 2048:
            raise NotImplementedError("exact rational inequality source limit exceeded")
        pending.extend((child, depth + 1) for child in children)
    return True


def _lift_sign_chart(roots, gaps, points):
    """Join checked adjacent cells without asking approximate endpoint order."""
    from ..sets import FiniteSet, Interval, Reals, Union

    count = len(roots)
    if len(gaps) != count + 1 or len(points) != count:
        raise ValueError("incomplete rational inequality sign chart")
    if not count:
        return Reals() if gaps[0] else FiniteSet()
    cells = [False] * (2 * count + 1)
    cells[::2] = gaps
    cells[1::2] = points
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
        if first == last and first % 2:
            pieces.append(FiniteSet(roots[first // 2]))
        else:
            lower = (-oo if first == 0 else roots[(first - 1) // 2])
            upper = (oo if last == 2 * count else roots[last // 2])
            pieces.append(Interval(lower, upper, first % 2 == 0, last % 2 == 0))
        position += 1
    return Union(*pieces)


def _solve_rational_inequality(relation, symbol):
    """Return a complete exact set, or None for a different surface grammar.

    A recognized resource refusal is explicit: it must not fall through to
    another solver after losing an original denominator or held child.
    """
    if type(relation) not in (Lt, Le, Gt, Ge):
        return None
    symbol = _require_symbol(symbol)
    # Check both original sides before constructing their held difference.
    if not _rational_surface(relation.lhs, symbol) or not _rational_surface(relation.rhs, symbol):
        return None
    source = Add(relation.lhs, Mul(-1, relation.rhs, evaluate=False), evaluate=False)
    if not _rational_surface(source, symbol):
        return None
    try:
        checked = _native.solve_real_rational_inequality_native(
            _native_expr(source), _symbol_binding(symbol), relation.rel_op,
        )
        if checked is None:
            raise NotImplementedError("exact rational inequality has an unsupported typed source")
        polynomial, gaps, points = checked
        count = len(points)
        if count > 64 or len(gaps) != count + 1:
            raise ValueError("invalid bounded rational inequality chart")
        if count:
            from ..polys.polytools import Poly
            from ..polys.rootoftools import real_roots

            expression = _lift_native_result(polynomial, source, symbol)
            associated = real_roots(Poly(expression, symbol), multiple=False)
            if len(associated) != count or any(m != 1 for _, m in associated):
                raise ValueError("rational inequality root association is incomplete")
            roots = [root for root, _ in associated]
            if len(set(roots)) != count:
                raise ValueError("distinct inequality endpoints were lost during lifting")
        else:
            roots = []
        return _lift_sign_chart(roots, gaps, points)
    except (ValueError, OverflowError) as exc:
        raise NotImplementedError(
            "exact rational inequality could not be completed: %s" % exc
        ) from None


def solve_univariate_inequality(expr, gen, relational=True, domain=None, continuous=False):
    """Solve an admitted univariate real inequality as a relation or a set.

    Exact rational functions use complete checked sign decomposition; the
    existing elementary lane handles its supported nonrational expressions.
    ``continuous`` does not suppress original domain checks.
    """
    from .. import _set_as_relational, _solve_inequality_set
    from ..core import S
    from ..sets import EmptySet, Intersection, Reals, Set

    symbol = _require_symbol(gen)
    if domain is None:
        domain = Reals()
    if not isinstance(domain, Set):
        raise TypeError("inequality domain must be a set")
    if expr is True or expr is S.true:
        result = Reals()
    elif expr is False or expr is S.false:
        result = EmptySet()
    else:
        result = _solve_inequality_set(expr, symbol)
    if domain != Reals():
        result = Intersection(result, domain)
    return _set_as_relational(result, symbol) if relational else result


__all__ = ["solve_univariate_inequality"]
