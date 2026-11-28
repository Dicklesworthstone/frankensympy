"""Ground-field roots from exact rational isolators or prime-field factors."""

from fractions import Fraction
from math import ceil, floor, lcm

from ..core import Rational, _active_callback_session, _native, _native_expr
from . import _coefficient_queries as queries
from . import _finitefield as retained
from ._root_analysis import _validate_atoms
from .polyerrors import DomainError, GeneratorsNeeded, MultivariatePolynomialError


def _ratio(value):
    return Fraction(value.exact_numerator(), value.exact_denominator())


def _rational_ground_roots(request, domain, session):
    """Check the primitive leading-coefficient grid against every real root."""
    raw = request.raw[0]
    rows = queries._project(raw, request.bindings, domain, "lex", session)
    if not rows or rows[0][0][0] == 0:
        return {}
    degree = rows[0][0][0]
    if degree > 64:
        raise ValueError("rational ground roots support degree at most 64")
    coefficients = [Fraction(0)] * (degree + 1)
    for (exponent,), coefficient in rows:
        coefficients[exponent] = _ratio(coefficient)
    lead = coefficients[-1]
    if degree == 1:
        root = -coefficients[0] / lead
        return {Rational(root.numerator, root.denominator): 1}

    # Clearing the denominators of the monic polynomial produces a
    # primitive integer polynomial with positive leading coefficient L.
    # The rational-root theorem puts every rational zero on (1/L) * ZZ.
    grid = 1
    for coefficient in coefficients:
        grid = lcm(grid, (coefficient / lead).denominator)
        if grid.bit_length() >= 4096:
            raise ValueError("rational root denominator grid exceeds its bit limit")
    width = _native_expr(Rational(1, 2 * grid))
    intervals = _native.real_root_analysis_native(
        raw, request.bindings[0], "intervals", None, None, width,
        None, False, domain == "ZZ", session,
    )
    roots = {}
    for lower, upper, multiplicity in intervals:
        queries._charge(session, degree + 1)
        first = ceil(_ratio(lower) * grid)
        last = floor(_ratio(upper) * grid)
        if first > last:
            continue
        if first != last:
            raise ValueError("a rational-root isolator spans multiple grid points")
        candidate = Fraction(first, grid)
        residual = lead
        for coefficient in reversed(coefficients[:-1]):
            residual = residual * candidate + coefficient
        if not residual:
            root = Rational(candidate.numerator, candidate.denominator)
            roots[root] = multiplicity
    return roots


def ground_roots_entry(source, gens, options):
    from .polytools import Poly, _ideal_polynomial_source

    unknown = options.keys() - {"domain", "gens", "modulus", "symmetric"}
    if unknown:
        raise NotImplementedError("unsupported ground-root option %r" % min(unknown))
    axes = retained._generator_tuple(gens, options)
    state = (retained._PolyState(source, require_field=False)
             if isinstance(source, Poly) else None)
    expression = state.expression if state is not None else source
    if state is not None:
        if axes and axes != state.gens:
            raise ValueError("ground-root generators differ from the polynomial's ring")
        axes = state.gens
    if retained.requested(source, options):
        if _active_callback_session() is not None:
            raise NotImplementedError("prime-field ground roots require standalone factorization")
        if state is not None and retained.is_finite(source):
            prime = options.get("modulus")
            domain = options.get("domain")
            if ((prime is not None and retained._prime(prime) != state.prime)
                    or (domain is not None and retained._domain_prime(domain) != state.prime)):
                raise DomainError("ground roots must preserve the polynomial's characteristic")
        polynomial = (source if retained.is_finite(source) and not options
                      else Poly(source, *gens, **options))
        field_state = retained._PolyState(polynomial)
        if len(field_state.gens) != 1:
            raise MultivariatePolynomialError("Cannot compute ground roots of a multivariate polynomial")
        roots = {}
        for factor, multiplicity in retained.factors(polynomial)[1]:
            coefficients = retained.coefficients(factor)
            if len(coefficients) == 2:
                # The monic factor x + b has root -b, with the caller's
                # selected symmetric or unsigned lift of b (upstream policy).
                roots[-coefficients[1]] = multiplicity
        field_state.unchanged()
        if state is not None:
            state.unchanged()
        return roots
    if "symmetric" in options:
        raise DomainError("symmetric lifting requires a prime field")
    domain = options.get("domain")
    if state is not None:
        original_domain = state.namespace.get("_domain")
        if type(original_domain) is not str or original_domain not in ("QQ", "ZZ"):
            raise DomainError("ground roots require a rational or prime coefficient field")
        if state.namespace.get("_extra_syms"):
            raise DomainError("coefficient parameters are not supported by rational ground roots")
        if domain is None:
            domain = original_domain
    if domain is None:
        domain = "QQ"
    if type(domain) is not str or domain not in ("QQ", "ZZ"):
        raise DomainError("ground roots require a rational or prime coefficient field")
    expression = _ideal_polynomial_source(expression)
    if not axes:
        axes = tuple(expression.free_symbols)
        if not axes:
            raise GeneratorsNeeded("constant ground roots require an explicit generator")
    if len(axes) != 1:
        raise MultivariatePolynomialError("Cannot compute ground roots of a multivariate polynomial")
    session = _active_callback_session()
    request = retained._Request([expression], axes, (state,) if state else ())
    _validate_atoms(request.snapshot)
    queries._check_dummies(request)
    result = _rational_ground_roots(request, domain, session)
    queries._interfaces()
    queries._current(session)
    request.unchanged()
    return result
