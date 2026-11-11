"""General rational quartics through the actual public polynomial consumers."""

import math

import pytest
import sympy as s


def polynomial(coefficients, variable):
    return sum(s.Integer(c)*variable**i for i, c in enumerate(coefficients))


@pytest.mark.parametrize("coefficients,real_count", [
    ((1, 1, 0, 0, 1), 0),
    ((-1, 1, 0, 0, 1), 2),
    ((1, 1, -5, 0, 1), 4),
    ((1, 2, 3, 4, 5), 0),
    ((3, 4, 6, 4, 1), 0),
    ((-3, 1, -6, 0, 1), 2),
    ((-3, 1, 6, 0, 1), 2),
])
def test_public_roots_cover_real_complex_nonmonic_and_resolvent_boundary_cases(
    coefficients, real_count, monkeypatch,
):
    def forbidden(*args, **kwargs):
        raise AssertionError("radical roots must not come from numerical root proposals")

    monkeypatch.setattr(s, "_numeric_poly_roots", forbidden)
    x = s.Symbol("quartic_x")
    source = polynomial(coefficients, x)
    roots = s.Poly(source, x).roots()
    assert len(roots) == 4 and list(roots.values()) == [1]*4
    assert all(root.free_symbols == set() for root in roots)
    # Independent approximate observations check the returned surface values;
    # production admission uses the exact coefficient-reconstruction guard.
    values = [complex(root) for root in roots]
    assert all(math.isfinite(root.real) and math.isfinite(root.imag) for root in values)
    assert sum(abs(root.imag) < 1e-7 for root in values) == real_count
    for root in values:
        residual, scale = 0j, 0.0
        for coefficient in reversed(coefficients):
            residual = residual*root + coefficient
            scale = scale*abs(root) + abs(coefficient)
        assert abs(residual) < 1e-8*max(scale, 1.0)


def test_public_multiplicities_and_exact_rational_scaling_are_preserved():
    x = s.Symbol("quartic_multiplicity_x")
    source = x**4 + x + 1
    roots = s.roots(source, x)
    assert s.roots(-s.Rational(7, 11)*source, x) == roots
    repeated = s.roots(source**2*(x - 2)**3, x)
    assert repeated == {**{root: 2 for root in roots}, s.Integer(2): 3}
    assert sum(repeated.values()) == 11


def test_equation_solving_preserves_original_quartic_denominator_holes():
    x = s.Symbol("quartic_equation_x")
    source = x**4 + x + 1
    assert set(s.solve(source, x)) == set(s.roots(source, x))
    reciprocal = s.Pow(source, -1, evaluate=False)
    assert s.solve(s.Mul((x - 3)*source, reciprocal, evaluate=False), x) == [3]
    assert s.solve(s.Mul(source**2, reciprocal, evaluate=False), x) == []


def test_companion_matrix_exposes_the_complete_complex_quartic_spectrum():
    x = s.Symbol("quartic_spectrum_x")
    expected = s.roots(x**4 + x + 1, x)
    companion = s.Matrix([
        [0, 0, 0, -1],
        [1, 0, 0, -1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ])
    assert companion.eigenvals() == expected
    spectrum = companion.eigenvalues()
    assert set(spectrum) == set(expected)
    assert len(spectrum) == 4
    assert {root: spectrum.count(root) for root in spectrum} == expected


@pytest.mark.parametrize("exponent", [
    s.Rational(1, 2),
    s.Rational(1, 3),
    s.Rational(-1, 2),
])
def test_held_negative_powers_use_principal_complex_numeric_evaluation(exponent):
    expression = s.Pow(-2, exponent, evaluate=False)
    expected = complex(-2, 0)**(exponent.p/exponent.q)

    observed = complex(expression)
    assert math.isfinite(observed.real) and math.isfinite(observed.imag)
    assert observed == pytest.approx(expected, rel=1e-10, abs=1e-10)

    evaluated = s.N(expression, 12)
    real, imaginary = evaluated.as_real_imag()
    observed_evalf = complex(float(real), float(imaginary))
    assert math.isfinite(observed_evalf.real) and math.isfinite(observed_evalf.imag)
    assert observed_evalf == pytest.approx(expected, rel=1e-10, abs=1e-10)
