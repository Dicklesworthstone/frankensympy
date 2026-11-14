"""Exact rational inequalities with complete real algebraic boundaries.

The native checker determines the complete ordered sign chart, retaining
every original denominator restriction. This module only lifts its checked
critical roots and joins adjacent admitted cells. Comparisons of floating
approximations play no role in root identity, order, or interval inclusion.
"""

from ..core import (
    Add, Basic, Dummy, Eq, Expr, Ge, Gt, Half, Integer, Le, Lt, Mul, Ne, NegativeOne, One,
    Pow, Rational, Relational, Symbol, Zero, _lift_native_result, _native, _native_expr,
    _require_symbol, _surface_binding_key, _symbol_binding, _wrap, oo,
)
from ..core._function_capabilities import _owner_matches, _owner_record
from ..logic.boolalg import And, Or
from ..sets import EmptySet, FiniteSet, Interval, Reals, Set, Union


_CONSTRAINT_MRO = type.__dict__["__mro__"]


def _constraint_owner_records():
    records = []
    for cls in (Basic, Expr, Integer, Rational, Zero, One, NegativeOne, Half,
                Symbol, Dummy, Add, Mul, Pow, Relational, Eq, Ne, Lt, Le, Gt,
                Ge, Set, EmptySet, FiniteSet, Interval, Reals, Union, And, Or):
        for owner in _CONSTRAINT_MRO.__get__(cls):
            if owner is object or any(owner is previous for previous, _, _ in records):
                continue
            records.append((owner, _owner_record(owner), _CONSTRAINT_MRO.__get__(owner)))
    return tuple(records)


_CONSTRAINT_OWNERS = _constraint_owner_records()


def _constraint_owners_unchanged():
    for owner, record, mro in _CONSTRAINT_OWNERS:
        current = _CONSTRAINT_MRO.__get__(owner)
        if (not _owner_matches(owner, record) or len(current) != len(mro)
                or any(a is not b for a, b in zip(current, mro))):
            raise NotImplementedError("real constraints require original built-in source and result classes")


def _lift_constraint_chart(checked, variable):
    """Associate every common boundary exactly before joining admitted cells."""
    from ..polys.polytools import Poly
    from ..polys.rootoftools import CRootOf, _real_radical, real_roots
    from ..sets import Reals
    from ._piecewise_real import _checked_domain
    from ._real_domain import _ATOMS, _is_exact

    polynomial, gaps, points = checked
    count = len(points)
    if (count > 64 or len(gaps) != count + 1
            or any(type(value) is not bool for value in (*gaps, *points))):
        raise ValueError("invalid complete real constraint chart")
    roots = []
    if count:
        associated = real_roots(Poly(_wrap(polynomial), variable), multiple=False, radicals=False)
        if (len(associated) != count or any(m != 1 for _, m in associated)
                or len({root for root, _ in associated}) != count):
            raise ValueError("real constraint root association is incomplete")
        coefficients = tuple(_native.real_root_coefficients_native(polynomial, _symbol_binding(variable)))
        order = _checked_domain(Reals())
        for index, (root, _) in enumerate(associated):
            if not _is_exact(root, _ATOMS) and type(root) is not CRootOf:
                raise ValueError("real constraint returned an unsupported root")
            descriptor = order.scalar(_native_expr(root))
            if order.compare(descriptor, ("root", coefficients, index, count)) != 0:
                raise ValueError("real constraint root association changed an ordered root")
            explicit = _real_radical(root.poly, root.index) if type(root) is CRootOf else None
            roots.append(root if explicit is None else explicit)
    return _lift_sign_chart(roots, gaps, points)


def _solve_rational_constraints(relations, symbols, conjunction_false=False):
    """Return a complete relational result, or None for another grammar.

    All original sides and typed symbol owners are captured before inference,
    subtraction, or Boolean identities. The shared native partition decides
    intersections without asking the generic set normalizer to order CRootOfs.
    """
    from ._piecewise_real import _Capture

    _constraint_owners_unchanged()

    class Capture(_Capture):
        def touch(self, depth):
            self.nodes += 1
            if self.nodes > 65536 or depth > 64:
                raise ValueError("real constraint resource limit: source nodes/depth")

        def basic(self, value):
            retained = super().basic(value)
            kind = type(value)
            if kind in (Integer, Rational, Zero, One, NegativeOne, Half, Add, Mul, Pow):
                native = _native_expr(value)
                valid = ((kind in (Integer, Zero, One, NegativeOne) and native.is_integer)
                         or (kind in (Rational, Half) and (native.is_integer or native.is_rational))
                         or kind is Add and native.is_add or kind is Mul and native.is_mul
                         or kind is Pow and native.is_pow)
                if not valid:
                    raise ValueError("real constraint scalar owner does not match its native source")
                fixed = {Zero: (0, 1), One: (1, 1), NegativeOne: (-1, 1), Half: (1, 2)}.get(kind)
                if fixed is not None and (native.exact_numerator(), native.exact_denominator()) != fixed:
                    raise ValueError("real constraint constant owner does not match its native value")
            return retained

        def symbol(self, value):
            super().symbol(value)
            native = _native_expr(value)
            if (not native.is_symbol or _surface_binding_key(native.symbol_binding)
                    != _surface_binding_key(_symbol_binding(value))):
                raise ValueError("real constraint symbol has inconsistent typed assumptions")

    if len(relations) > 32:
        raise NotImplementedError("real constraint resource limit: at most 32 relations")
    capture = Capture()
    operations = []
    owners = ((Eq, "=="), (Ne, "!="), (Lt, "<"), (Le, "<="), (Gt, ">"), (Ge, ">="))
    variable = symbols
    if type(symbols) in (list, tuple, set):
        if len(symbols) != 1:
            raise NotImplementedError("only univariate inequalities are supported")
        variable = next(iter(symbols))
    if variable is not None:
        if type(variable) not in (Symbol, Dummy):
            raise TypeError("real constraints require an exact symbol")
        _native.real_constraint_source_preflight_native(_native_expr(variable))
        capture.symbol(variable)
    try:
        sides = []
        for relation in relations:
            operation = next((op for owner, op in owners if type(relation) is owner), None)
            if operation is None:
                raise NotImplementedError("real constraints require exact relational owners")
            left, right = capture.sides(relation)
            operations.append(operation)
            sides.extend((left, right))
        # Borrow every original payload before Python accessors can clone its
        # children; unsupported first sources cannot hide later size failures.
        grammar = True
        for side in sides:
            grammar &= _native.real_constraint_source_preflight_native(_native_expr(side))
        for side in sides:
            capture.scalar(side, 0)
        capture.unchanged()
        inferred_constant = symbols is None and not capture.symbols
        if not grammar:
            # Unsupported closed built-ins keep their existing scalar route.
            # In particular, internal Piecewise guard tags must not become
            # extra variables during rational-system variable inference.
            for value, slot, children in capture.stamps:
                if slot.__name__ not in ("_args", "_struct_args", "_surface_args") or type(children) is not tuple:
                    continue
                operator = next((op for owner, op in ((Add, "add"), (Mul, "mul"), (Pow, "pow"))
                                if type(value) is owner), None)
                if operator is not None:
                    native = _native_expr(value)
                    if _native.real_constraint_source_preflight_native(native):
                        _native.ideal_source_arguments_native(native, operator,
                                                             [_native_expr(child) for child in children])
            capture.unchanged()
            return None
        variable, frozen, supported = capture.freeze(variable)
        if not supported:
            return None
        _constraint_owners_unchanged()
        sources = [Add(frozen[index], Mul(-1, frozen[index + 1], evaluate=False), evaluate=False)
                   for index in range(0, len(frozen), 2)]
        capture.unchanged()
        checked = _native.solve_real_rational_constraints_native(
            [_native_expr(source) for source in sources], _symbol_binding(variable),
            operations, conjunction_false,
        )
        capture.unchanged()
        _constraint_owners_unchanged()
        if checked is None:
            raise NotImplementedError("admitted real constraints could not be completed")
        result = _lift_constraint_chart(checked, variable)
        capture.unchanged()
        _constraint_owners_unchanged()
        if inferred_constant:
            from ..logic.boolalg import false, true
            if type(result) not in (EmptySet, Reals):
                raise ValueError("constant real constraints produced variable boundaries")
            return false if type(result) is EmptySet else true
        from .. import _set_as_relational
        relational = _set_as_relational(result, variable)
        capture.unchanged()
        _constraint_owners_unchanged()
        return relational
    except (ValueError, OverflowError) as exc:
        raise NotImplementedError("exact rational constraints could not be completed: %s" % exc) from None


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
