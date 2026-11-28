"""Complete algebraic points for finite ideals with totally real projections."""

from ..core import Symbol, _native, _native_expr, _surface_binding_key, _wrap
from ..polys.polytools import Poly
from ..polys.rootoftools import CRootOf, _native_coefficients, real_roots
from ..utilities.iterables import default_sort_key


def _coordinate_roots(request, axis, pivot):
    binding = request.bindings[axis]
    if len(pivot.free_symbol_bindings) == 1:
        # This univariate member of the verified basis already annihilates
        # the coordinate at every point. Extraneous combinations are checked
        # against the other triangular equations and every original source.
        polynomial = pivot
    else:
        others = request.bindings[:axis] + request.bindings[axis+1:]
        projection = _native.ideal_eliminate_native(request.raw, request.bindings, others)
        if len(projection) != 1:
            raise ValueError("finite coordinate projection lacks a unique eliminant")
        polynomial = projection[0]
    coefficients = _native.real_root_coefficients_native(polynomial, binding)
    # The compatibility result uses x as the canonical bound generator in
    # indexed roots, independently of the system's user-supplied symbols.
    p = Poly.from_list([_wrap(c) for c in reversed(coefficients)], gens=Symbol("x"))
    roots = real_roots(p, multiple=False, radicals=False)
    if sum(multiplicity for _, multiplicity in roots) != len(coefficients) - 1:
        raise NotImplementedError("triangular algebraic solving requires totally real coordinate projections")
    result = []
    for root, _ in roots:
        if type(root) is CRootOf:
            values = _native_coefficients(root.poly)
            key = (tuple((c.exact_numerator(), c.exact_denominator()) for c in values),
                   root.index)
            result.append((root, values, key))
        else:
            result.append((root, None, None))
    return result


def algebraic_points(request, pivots, blocks):
    """Intersect exact coordinate projections without choosing numerical pairs."""
    coordinates = [_coordinate_roots(request, i, pivot) for i, pivot in enumerate(pivots)]
    variables = [_native_expr(axis) for axis in request.gens]
    keys = [_surface_binding_key(binding) for binding in request.bindings]
    evaluations = 0

    def vanishes(equation, point):
        nonlocal evaluations
        evaluations += 1
        if evaluations > 8192:
            raise ValueError("algebraic triangular system exceeds its evaluation budget")
        start = len(variables) - len(point)
        rationals = [None] * start + [
            _native_expr(root) if values is None else None for root, values, _ in point]
        # This original-source, typed substitution has its own coefficient
        # identity check. It normalizes constant/zero inputs and removes all
        # rational coordinates before constructing an algebraic extension.
        source, _ = _native.polynomial_evaluate_native(
            equation, request.bindings, rationals, "QQ", True)
        active = {_surface_binding_key(b) for b in source.free_symbol_bindings}
        if not active:
            return source.exact_numerator() == 0
        bindings, polynomials, indices = [], [], []
        representatives = {}
        for axis, (_, values, key) in enumerate(point, start):
            if keys[axis] not in active:
                continue
            if key in representatives:
                # Identical defining polynomial and root index denote the
                # same embedding even when two coordinates use different axes.
                source = source.subs(variables[axis], representatives[key])
            else:
                representatives[key] = variables[axis]
                bindings.append(request.bindings[axis])
                polynomials.append(values)
                indices.append(key[1])
        if len(bindings) == 1:
            value = _native.evaluate_real_algebraic_native(
                source, bindings[0], polynomials[0], indices[0], 1)
        else:
            value = _native.evaluate_real_algebraic_expression_native(
                source, bindings, polynomials, indices, 1)
        if value is None:
            raise ValueError("algebraic triangular residual could not be decided")
        lower, upper = value.lower.exact_numerator(), value.upper.exact_numerator()
        if lower == 0 == upper:
            return True
        if lower > 0 or upper < 0:
            return False
        raise ValueError("algebraic triangular residual has indeterminate sign")

    suffixes = [()]
    for axis in reversed(range(len(variables))):
        next_suffixes = []
        for suffix in suffixes:
            for coordinate in coordinates[axis]:
                point = (coordinate, *suffix)
                if all(vanishes(equation, point) for equation in blocks[axis]):
                    if len(next_suffixes) == 256:
                        raise ValueError("algebraic triangular system exceeds its 256-point branch limit")
                    next_suffixes.append(point)
        suffixes = next_suffixes
    result = []
    for point in suffixes:
        if not all(vanishes(equation, point) for equation in request.raw):
            raise ValueError("an algebraic triangular solution fails an original equation")
        result.append(tuple(root for root, _, _ in point))
    return sorted(result, key=default_sort_key)
