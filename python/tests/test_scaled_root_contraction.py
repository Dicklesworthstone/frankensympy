"""Original-source Krawczyk proofs on rectangular boxes with different axis scales."""

from fractions import Fraction

import pytest
import sympy as s


def test_coupled_root_proves_scaled_contraction_and_replays_its_metric():
    x, y = s.symbols("scaled_root_x scaled_root_y")
    model = s.compile_residual_system([x - 4*y**2, y - x**2/128], (x, y))
    domain = [(-2, 2), (Fraction(-1, 4), Fraction(1, 4))]
    proof = model.certify_root_box(domain)
    assert proof.contraction_norm == "box_radius_scaled_infinity"
    assert proof.contraction_bound == s.Rational(1, 4)
    replayed = model.replay_root_certificate(proof.to_bytes())
    assert replayed.contraction_norm == proof.contraction_norm
    assert replayed.contraction_bound == proof.contraction_bound
    assert replayed.root_enclosure == proof.root_enclosure
    with pytest.raises(AttributeError):
        proof.contraction_norm = "infinity"
    ordinary = s.compile_residual_system([x, y], (x, y)).certify_root_box(domain)
    assert ordinary.contraction_norm == "infinity"
    assert ordinary.contraction_bound == 0


def test_uniform_scaled_root_family_exposes_the_physical_derivative_bounds():
    x, y, p = s.symbols("scaled_family_x scaled_family_y scaled_family_p")
    model = s.compile_residual_system([x - 4*y**2, y - x**2/128 - p], (x, y, p))
    domain = [(-2, 2), (Fraction(-1, 4), Fraction(1, 4))]
    proof = model.certify_parametric_root_box(domain, [(Fraction(-1, 8), Fraction(1, 8))])
    assert proof.contraction_norm == "box_radius_scaled_infinity"
    assert proof.contraction_bound == s.Rational(1, 4)
    derivative = proof.enclose_sensitivity()
    assert derivative.solution_jacobian == (
        ((s.Rational(-8, 3), s.Rational(8, 3)),),
        ((s.Rational(2, 3), s.Rational(4, 3)),),
    )
