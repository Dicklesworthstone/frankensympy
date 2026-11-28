"""A complete rational fundamental system for regular integer-lattice recurrences."""

from fractions import Fraction

from ..core import Add, Dummy, Eq, Integer, Mul, Pow, Rational, Symbol
from ..polys import _coefficient_queries as queries
from ..polys._finitefield import _Request
from ._polynomial_recurrence import (
    _MAX_DEGREE, _Work, _bounded, _evaluate, _expression, _solve_recurrence, _source_rows,
)
from ._rational_recurrence import _Ring
from .ode import _checkodesol_children


def _sequence_atoms(sides, cls, n):
    # A closed polynomial source may be rewritten with fresh coordinates
    # for sequence evaluations. Other functions, parameter branches and
    # original divisions continue through rsolve's existing source lanes.
    pending = list(sides)
    atoms = {}
    variable_outside = False
    while pending:
        node = pending.pop()
        if type(node) is cls:
            if len(node.args) != 1:
                return None
            if node not in atoms:
                atoms[node] = Dummy("rsolve_value")
        elif type(node) in (Symbol, Dummy):
            if node != n:
                return None
            variable_outside = True
        elif type(node) in (Add, Mul, Pow):
            children = _checkodesol_children(node)
            if type(node) is Pow and (not isinstance(children[1], Integer)
                                      or not 0 <= children[1].p <= _MAX_DEGREE):
                return None
            pending.extend(children)
        elif not isinstance(node, Rational):
            return None
    # The checked sparse ring owns at most eight coordinates, including n.
    return atoms if variable_outside and 2 <= len(atoms) <= 7 else None


def _shifted_rows(sides, atoms, n, work):
    indices = _Request([fn.args[0] for fn in atoms], (n,))
    queries._check_dummies(indices)
    offsets = []
    for args in _source_rows(indices, work):
        if args.get(1) != 1 or args.keys() - {0, 1}:
            return None
        k = args.get(0, Fraction(0))
        if k.denominator != 1 or abs(k) > 1024:
            return None
        offsets.append(int(k))
    lo, hi = min(offsets), max(offsets)
    if lo == hi or hi - lo > 32:
        return None
    coefficients = [{} for _ in range(hi-lo+1)]
    forcing = {}
    # Project both original equality sides separately: an inadmissible
    # product cannot turn into a linear term by moving it across '='.
    request = _Request([side.xreplace(atoms) for side in sides], (n, *atoms.values()))
    queries._check_dummies(request)
    for side, raw in enumerate(request.raw):
        terms = queries._project(raw, request.bindings, "QQ", "lex", work.session)
        work.charge(len(terms)+1)
        for (k, *powers), scalar in terms:
            if k > _MAX_DEGREE:
                raise NotImplementedError("rsolve rational fundamental system exceeds degree 64")
            count = sum(powers)
            if count > 1:
                raise NotImplementedError("rsolve: nonlinear recurrence")
            a = _bounded(Fraction(scalar.exact_numerator(), scalar.exact_denominator()))
            row = coefficients[offsets[powers.index(1)]-lo] if count else forcing
            if side:
                a = -a
            delta = a if count else -a
            v = _bounded(row.get(k, 0) + delta)
            if v:
                row[k] = v
            else:
                row.pop(k, None)
    indices.unchanged()
    request.unchanged()
    if not coefficients[0] or not coefficients[-1]:
        return None
    if not any(max(row, default=0) > 0 for row in coefficients):
        return None
    return lo, coefficients, forcing


def _regular(row, ring):
    if max(row) == 0:
        return True
    middle, radius = ring.root_disk(ring.source(row))
    return all(_evaluate(row, k, ring.work) != 0
               for k in range(middle-radius+1, middle+radius))


def _verify_basis(solution, constants, n, work):
    # At a regular lattice point a basis is independent if its values on
    # r consecutive indices have full rank (the discrete Casoratian).
    size = len(constants)
    work.charge(size**3)
    zeros = dict.fromkeys(constants, Integer(0))
    probes = [zeros.copy() for _ in constants]
    for probe, c in zip(probes, constants):
        probe[c] = Integer(1)
    matrix = []
    for k in range(size):
        sample = solution.subs(n, k)
        base = sample.xreplace(zeros)
        row = []
        for probe in probes:
            value = sample.xreplace(probe) - base
            if not isinstance(value, Rational):
                raise RuntimeError("rsolve rational basis has a nonrational lattice value")
            row.append(_bounded(Fraction(value.p, value.q)))
        matrix.append(row)
    for col in range(size):
        pivot = next((j for j in range(col, size) if matrix[j][col]), None)
        if pivot is None:
            raise RuntimeError("rsolve rational fundamental directions are dependent")
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        for j in range(col+1, size):
            ratio = _bounded(matrix[j][col] / matrix[col][col])
            if ratio:
                for i in range(col+1, size):
                    matrix[j][i] = _bounded(matrix[j][i] - ratio*matrix[col][i])


def complete_rational_sequence(original, func, n, initial):
    """Return a family only if it spans the entire sequence solution space."""
    from ..polys import cancel
    from .solvers import linsolve

    if any(not isinstance(value, Rational) for _, value in initial):
        return None
    sides = (original.lhs, original.rhs) if type(original) is Eq else (original,)
    atoms = _sequence_atoms(sides, func, n)
    if atoms is None:
        return None
    work = _Work()
    data = _shifted_rows(sides, atoms, n, work)
    if data is None:
        return None
    lo, coeffs, f = data
    # Preserve the existing first-order product/factorial route whenever
    # it already handles a constant leading coefficient.
    order = len(coeffs)-1
    if order == 1 and max(coeffs[-1]) == 0:
        return None
    ring = _Ring(n, work)
    if not all(_regular(row, ring) for row in (coeffs[0], coeffs[-1])):
        raise NotImplementedError("rsolve: singular endpoint on the integer lattice")
    rows = [_expression(a, n) for a in coeffs]
    forcing = _expression(f, n)
    if lo:
        rows = [ring.source(a).shift(-lo).as_expr() for a in coeffs]
        forcing = ring.source(f).shift(-lo).as_expr()
    family = _solve_recurrence(rows, forcing, n, 0, {"symbols": True}, rational=True)
    # Both endpoint coefficients are nonzero on Z. Every r consecutive
    # values therefore extend uniquely in both directions. Independent
    # rational directions are independent as sequences (a nonzero numerator
    # cannot vanish on Z), and a finite pole chain would require a zero
    # endpoint coefficient. Exactly r directions prove this family exhaustive.
    if family is None or len(family[1]) != order:
        raise NotImplementedError("rsolve: rational directions do not span this variable-coefficient recurrence")
    solution, constants = family
    _verify_basis(solution, constants, n, work)
    if initial:
        equations = [solution.subs(n, k)-v for k, v in initial]
        choices = linsolve(equations, *constants)
        if not choices:
            raise ValueError("rsolve: inconsistent initial conditions")
        assignment = dict(zip(constants, next(iter(choices))))
        solution = solution.xreplace(assignment)
        for k, v in initial:
            if cancel(solution.subs(n, k)-v) != 0:
                raise RuntimeError("rsolve rational solution failed an original initial condition")
    # Re-evaluate the published expression in the original, unshifted
    # equation; this check is independent of shift extraction and IC fitting.
    substituted = {atom: solution.xreplace({n: atom.args[0]}) for atom in atoms}
    residual = sides[0].xreplace(substituted)
    if len(sides) == 2:
        residual -= sides[1].xreplace(substituted)
    if cancel(residual) != 0:
        raise RuntimeError("rsolve rational solution failed the original unshifted recurrence")
    queries._current(work.session)
    return solution
