"""Algebraic points of separated ideals and totally real finite systems."""

from itertools import product
from math import gcd, lcm

from ..core import Integer, Symbol, _native, _native_expr, _surface_binding_key, _wrap
from ..polys.polytools import Poly
from ..polys.rootoftools import CRootOf, _factor_data, _make_root, _native_coefficients, real_roots


def _coordinate_polynomial(request, axis, pivot):
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
    return Poly.from_list([_wrap(c) for c in reversed(coefficients)], gens=Symbol("x"))


def _root_scaling(polynomial):
    """Return the exact positive integer change x = scale*u used by SymPy."""
    coefficients = _native_coefficients(polynomial)
    denominator = lcm(*(c.exact_denominator() for c in coefficients))
    integers = [c.exact_numerator() * (denominator // c.exact_denominator())
                for c in coefficients]
    content = gcd(*integers)
    integers = [n // content for n in integers]
    degree = len(integers) - 1
    lower = [(degree-i, abs(c)) for i, c in enumerate(integers[:-1]) if c]
    if not lower or abs(integers[-1]) >= lower[0][1]:
        return 1, polynomial
    if len(lower) == 1:
        # For a binomial the pinned root representation extracts a scale only
        # when the entire trailing coefficient is a perfect power.
        exponent, coefficient = lower[0]
        low, high = 1, 1 << ((coefficient.bit_length() + exponent - 1) // exponent)
        while low < high:
            middle = (low + high + 1) // 2
            if middle**exponent <= coefficient:
                low = middle
            else:
                high = middle - 1
        scale = low if low**exponent == coefficient else 1
    else:
        common = gcd(*(c for _, c in lower))
        if common.bit_length() > 64:
            raise ValueError("algebraic root coefficient scaling exceeds the integer-factor bound")
        scale = 1
        if common > 1:
            for prime in _native.factorize(common):
                powers = []
                for exponent, coefficient in lower:
                    valuation = 0
                    while coefficient % prime == 0:
                        coefficient //= prime
                        valuation += 1
                    powers.append(valuation // exponent)
                scale *= prime**min(powers)
    if scale == 1:
        return 1, polynomial
    transformed = []
    for i, coefficient in enumerate(integers):
        value, remainder = divmod(coefficient, scale**(degree-i))
        if remainder:
            raise ValueError("algebraic root scaling failed its exact coefficient identity")
        transformed.append(Integer(value))
    return scale, Poly.from_list(transformed[::-1], gens=polynomial.gen)


def _coordinate_roots(request, axis, pivot):
    p = _coordinate_polynomial(request, axis, pivot)
    roots = real_roots(p, multiple=False, radicals=False)
    if sum(multiplicity for _, multiplicity in roots) != p.degree():
        raise NotImplementedError("triangular algebraic solving requires totally real coordinate projections")
    scale, normalized = _root_scaling(p)
    visible = roots if scale == 1 else real_roots(normalized, multiple=False, radicals=False)
    if [m for _, m in roots] != [m for _, m in visible]:
        raise ValueError("positive root scaling changed ordered real-root multiplicities")
    result = []
    for (root, _), (scaled, _) in zip(roots, visible):
        output = scaled if scale == 1 else scale * scaled
        if type(root) is CRootOf:
            values = _native_coefficients(root.poly)
            key = (tuple((c.exact_numerator(), c.exact_denominator()) for c in values),
                   root.index)
            result.append((output, values, key))
        else:
            result.append((output, None, None))
    return result


def _separated_points(request, pivots):
    """Enumerate complete local index sets with no complex embedding choices."""
    for source in request.raw:
        # This coefficient identity holds for every simultaneous choice of
        # factor roots, including nonreal ones. There is no floating residual.
        _, remainder = _native.groebner_reduce_native(
            pivots, request.bindings, "lex", source, False)
        if not remainder.is_integer or remainder.exact_numerator() != 0:
            raise ValueError("separated coordinate equations do not imply an original equation")
    axes = []
    size = 1
    for axis, pivot in enumerate(pivots):
        p = _coordinate_polynomial(request, axis, pivot)
        scale, normalized = _root_scaling(p)
        factors, _ = _factor_data(normalized)
        roots = []
        for factor, _ in factors:
            # Native factorization checks coprimality and the original
            # product. Indexed-root admission checks square-freeness. Each
            # degree-d factor therefore contributes exactly its local indices
            # 0..d-1, regardless of real/nonreal order or multiplicity.
            for i in range(factor.degree()):
                root = _make_root(CRootOf, factor, normalized.gen, i, False, False)
                roots.append(root if scale == 1 else scale * root)
        size *= len(roots)
        if size > 256:
            raise ValueError("separated algebraic system exceeds its 256-point branch limit")
        axes.append(roots)
    return list(product(*axes))


def algebraic_points(request, pivots, blocks):
    """Intersect exact coordinate projections without choosing numerical pairs."""
    if all(len(block) == 1 and len(pivot.free_symbol_bindings) == 1
           for pivot, block in zip(pivots, blocks)):
        return _separated_points(request, pivots)
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
    return result
