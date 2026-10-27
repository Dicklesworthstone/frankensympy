"""Actual-shell coverage for the checked rational definite-integration lane."""

import sympy as s


def test_irreducible_repeated_denominators_and_reversed_limits():
    x = s.Symbol("x")
    for multiplicity in (1, 2, 3):
        integrand = (1 - (2 * multiplicity - 1) * x**2) / (1 + x**2) ** (
            multiplicity + 1
        )
        expected = s.Rational(1, 2**multiplicity)
        assert s.integrate(integrand, (x, 0, 1)) == expected
        assert s.integrate(integrand, (x, 1, 0)) == -expected


def test_improper_integrals_use_separate_finite_endpoint_limits():
    x = s.Symbol("x")
    integrand = (1 - x**2) / (1 + x**2) ** 2
    assert s.integrate(integrand, (x, -s.oo, s.oo)) == 0
    assert s.integrate(integrand, (x, -s.oo, 1)) == s.Rational(1, 2)
    assert s.integrate(1 / x**2, (x, 1, s.oo)) == 1
    assert s.integrate(1 / x**2, (x, 0, 1)) == s.oo
    assert s.integrate(1 / x**2, (x, 1, 0)) == -s.oo
    assert isinstance(s.integrate(x, (x, -s.oo, s.oo)), s.Integral)


def test_held_original_holes_are_preserved_inside_the_interval():
    x = s.Symbol("x")
    reciprocal = s.Pow(x, -1, evaluate=False)
    cancelled = s.Mul(x, reciprocal, evaluate=False)
    nested = s.Pow(reciprocal, -1, evaluate=False)
    for integrand in (cancelled, nested):
        result = s.integrate(integrand, (x, -1, 1))
        assert isinstance(result, s.Integral)
    # At an endpoint the same hole is a convergent improper integral.
    assert s.integrate(cancelled, (x, 0, 2)) == 2
    assert s.integrate(nested, (x, 0, 2)) == 2


def test_exact_large_endpoint_difference_and_irrational_interior_pole():
    x = s.Symbol("x")
    start = 10**40
    assert s.integrate(1 / x**2, (x, start, start + 1)) == s.Rational(
        1, start * (start + 1)
    )
    singular = -2 * x / (x**2 - 2) ** 2
    assert isinstance(s.integrate(singular, (x, 1, 2)), s.Integral)
