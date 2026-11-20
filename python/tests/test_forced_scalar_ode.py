"""Public typed scalar ODE coverage for polynomial and resonant forcing."""

import pytest
import sympy as s


def _assert_scalar_solution(solution, coefficients, forcing, x):
    """Check the original scalar equation and both initial coordinates."""
    a, b, c = coefficients
    first = s.diff(solution, x)
    residual = a * s.diff(first, x) + b * first + c * solution - forcing
    assert s.simplify(s.expand(residual)) == 0

    c1, c2 = s.symbols("C1 C2")
    assert s.simplify(solution.subs(x, 0) - c1) == 0
    assert s.simplify(first.subs(x, 0) - c2) == 0
    assert solution.free_symbols <= {x, c1, c2}


@pytest.mark.parametrize(
    "coefficients, build_forcing",
    [
        ((1, 0, 0), lambda x: x),
        ((1, -2, 1), lambda x: x * s.exp(x)),
        ((1, 0, 1), lambda x: x * s.cos(x)),
        ((-2, 2, 0), lambda x: x**2),
    ],
    ids=[
        "polynomial-double-integrator",
        "repeated-exponential-resonance",
        "oscillatory-resonance",
        "negative-leading-coefficient",
    ],
)
def test_public_scalar_ode_polynomial_elementary_forcing(coefficients, build_forcing):
    x = s.Symbol("x")
    forcing = build_forcing(x)
    solution = s.dsolve_const_coeff_second_order_nonhomogeneous(
        *coefficients, forcing, x
    )
    _assert_scalar_solution(solution, coefficients, forcing, x)


def test_public_scalar_ode_preserves_dummy_axis_without_text_bridge(monkeypatch):
    from sympy.solvers import ode

    def text_bridge_forbidden(*args, **kwargs):
        raise AssertionError("the public scalar solver must retain typed inputs")

    monkeypatch.setattr(
        ode._native,
        "dsolve_const_coeff_second_order_nonhomogeneous_expr",
        text_bridge_forbidden,
    )
    x = s.Dummy("x")
    solution = s.dsolve_const_coeff_second_order_nonhomogeneous(1, 0, 0, x**2, x)
    _assert_scalar_solution(solution, (1, 0, 0), x**2, x)
    assert x in solution.free_symbols
    assert any(atom is x for atom in solution.free_symbols)
    assert s.Symbol("x") not in solution.free_symbols


def test_public_scalar_ode_rejects_same_name_foreign_axis():
    x = s.Symbol("x", real=True)
    other_x = s.Symbol("x")
    assert x != other_x
    with pytest.raises(ValueError):
        s.dsolve_const_coeff_second_order_nonhomogeneous(1, 0, 0, x + other_x, x)


def test_public_scalar_ode_rejects_hidden_source_pole():
    x = s.Symbol("x")
    reciprocal = s.Pow(x, -1, evaluate=False)
    held_zero_over_x = s.Mul(0, reciprocal, evaluate=False)
    with pytest.raises(ValueError):
        s.dsolve_const_coeff_second_order_nonhomogeneous(1, 0, 0, held_zero_over_x, x)
