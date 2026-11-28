"""Rational points of a finite rational ideal from its checked lex basis."""

from fractions import Fraction

from ..core import _active_callback_session, _exact_ratio, _native, _native_expr
from ..polys import _coefficient_queries as queries
from ..polys import _finitefield as retained
from ..polys._ground_roots import rational_ground_roots
from ..polys._root_analysis import _validate_atoms
from ..polys.polyerrors import DomainError, GeneratorsNeeded, PolynomialError
from ..polys.polytools import Poly, _rational_ideal_source
from .polysys import _polynomial_system_inputs


def _triangular_blocks(basis, bindings):
    """Find a monic pivot whose degree survives every suffix specialization."""
    blocks = [[] for _ in bindings]
    pivots = [None] * len(bindings)
    for equation in basis:
        rows = queries._project(equation, bindings, "QQ", "lex", None)
        if not rows:
            raise ValueError("checked reduced basis contains a zero element")
        powers = rows[0][0]
        support = [i for i, exponent in enumerate(powers) if exponent]
        if not support:
            return None  # A nonzero constant in the basis proves inconsistency.
        axis = support[0]
        blocks[axis].append(equation)
        if len(support) == 1:
            pivots[axis] = equation
    # A zero-dimensional leading ideal contains a pure power of every
    # variable. In lex order no trailing variable can occur alongside the
    # highest power of that axis in its pure-power pivot.
    if any(pivot is None for pivot in pivots):
        raise PolynomialError("rational triangular solving requires a zero-dimensional ideal")
    return pivots, blocks


def _is_zero(value):
    if not (value.is_integer or value.is_rational):
        raise ValueError("triangular specialization did not produce an exact scalar")
    return value.exact_numerator() == 0


def solve_rational_triangular(polys, gens, options):
    unknown = options.keys() - {"domain", "extension"}
    if unknown:
        raise NotImplementedError("unsupported triangular solver option %r" % min(unknown))
    if options.get("extension", False) is not False:
        raise NotImplementedError("triangular algebraic-extension solving is not implemented")
    domain = options.get("domain", "QQ")
    if type(domain) is not str or domain not in ("ZZ", "QQ"):
        raise DomainError("triangular ground solving supports rational coefficients")
    if _active_callback_session() is not None:
        raise NotImplementedError("triangular solving requires standalone Groebner evaluation")

    states = []
    expressions = []
    for item in polys:
        if len(expressions) == 64:
            raise ValueError("triangular solving supports at most 64 original equations")
        if isinstance(item, Poly):
            states.append(retained._PolyState(item, require_field=False))
            item = _rational_ideal_source(item)
        expressions.append(item)
    equations, axes = _polynomial_system_inputs(expressions, gens)
    if not axes or (not gens and len(axes) != 1):
        raise GeneratorsNeeded("specify the generators of the triangular system")
    request = retained._Request(equations, axes, states)
    queries._check_dummies(request)
    _validate_atoms(request.snapshot)
    bindings = request.bindings
    if domain == "ZZ":
        for equation in request.raw:
            queries._project(equation, bindings, domain, "lex", None)
    basis = _native.groebner_basis_native(request.raw, bindings, "lex")
    triangular = _triangular_blocks(basis, bindings)
    request.unchanged()
    if triangular is None:
        return []
    pivots, blocks = triangular
    evaluations = 0

    def evaluate(equation, assignments):
        # Bound total fiber and original-equation work, as well as the
        # already bounded native cost of each independently checked evaluation.
        nonlocal evaluations
        evaluations += 1
        if evaluations > 8192:
            raise ValueError("triangular system exceeds its evaluation budget")
        value, _ = _native.polynomial_evaluate_native(
            equation, bindings, assignments, "QQ", True)
        return value

    suffixes = [()]
    for axis in reversed(range(len(axes))):
        next_suffixes = []
        for suffix in suffixes:
            tail = [_native_expr(value) for value in suffix]
            pivot = pivots[axis]
            specialized = evaluate(pivot, [None] * (axis + 1) + tail)
            roots = rational_ground_roots(specialized, bindings[axis], "QQ", None)
            for root in roots:
                assignment = [None] * axis + [_native_expr(root), *tail]
                if all(_is_zero(evaluate(other, assignment))
                       for other in blocks[axis] if other is not pivot):
                    if len(next_suffixes) == 256:
                        raise ValueError("triangular system exceeds its 256-point branch limit")
                    next_suffixes.append((root, *suffix))
        suffixes = next_suffixes
        if not suffixes:
            break

    for solution in suffixes:
        assignment = [_native_expr(value) for value in solution]
        if not all(_is_zero(evaluate(source, assignment)) for source in request.raw):
            raise ValueError("a triangular solution fails an original equation")
    suffixes.sort(key=lambda solution: tuple(Fraction(*_exact_ratio(v)) for v in solution))
    request.unchanged()
    return suffixes
