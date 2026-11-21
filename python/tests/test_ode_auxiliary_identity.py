"""ODE auxiliary unknowns must never capture original coordinates or parameters."""

import pytest
import sympy as s


def _residual_zero(expression):
    assert s.simplify(s.expand(expression)) == 0


@pytest.mark.parametrize("parameter", [
    s.Symbol("_fsym_r"),
    s.Symbol("_fsym_r", positive=True),
    s.Dummy("_fsym_r"),
])
def test_characteristic_coordinate_preserves_user_parameter(parameter):
    x = s.Symbol("x")
    y = s.Function("f")(x)
    solution = s.dsolve(s.diff(y, x) - parameter*y, y)

    assert solution.lhs == y
    assert parameter in solution.rhs.free_symbols
    assert solution.rhs.free_symbols <= {x, parameter, s.Symbol("C1")}
    _residual_zero(s.diff(solution.rhs, x) - parameter*solution.rhs)
    _residual_zero(solution.rhs.subs(s.Symbol("C1"), 1) - s.exp(parameter*x))


@pytest.mark.parametrize("parameter", [
    s.Symbol("_fsym_r"),
    s.Dummy("_fsym_r"),
])
def test_euler_indicial_coordinate_preserves_user_parameter(parameter):
    x = s.Symbol("x", positive=True)
    y = s.Function("f")(x)
    solution = s.dsolve(x*s.diff(y, x) - parameter*y, y)

    assert parameter in solution.rhs.free_symbols
    assert solution.rhs.free_symbols <= {x, parameter, s.Symbol("C1")}
    _residual_zero(x*s.diff(solution.rhs, x) - parameter*solution.rhs)
    _residual_zero(solution.rhs.subs(s.Symbol("C1"), 1) - x**parameter)


def test_forced_ode_keeps_every_colliding_trial_parameter():
    x = s.Symbol("x")
    parameters = s.symbols("_fsym_a0 _fsym_a1 _fsym_a2")
    forcing = sum(parameter*x**index for index, parameter in enumerate(parameters))
    y = s.Function("f")(x)
    solution = s.dsolve(s.diff(y, x, 2) - forcing, y)

    assert set(parameters) <= solution.rhs.free_symbols
    assert solution.rhs.free_symbols <= {x, *parameters, *s.symbols("C1 C2")}
    _residual_zero(s.diff(solution.rhs, x, 2) - forcing)
    particular = solution.rhs.subs({s.Symbol("C1"): 0, s.Symbol("C2"): 0})
    expected = sum(parameter*x**(index + 2)/((index + 1)*(index + 2))
                   for index, parameter in enumerate(parameters))
    _residual_zero(particular - expected)


def test_forced_ode_does_not_fit_its_independent_variable():
    x = s.Symbol("_fsym_a0")
    y = s.Function("f")(x)
    solution = s.dsolve(s.diff(y, x, 2) - 1, y)

    _residual_zero(s.diff(solution.rhs, x, 2) - 1)
    assert solution.rhs.free_symbols <= {x, *s.symbols("C1 C2")}
    particular = solution.rhs.subs({s.Symbol("C1"): 0, s.Symbol("C2"): 0})
    _residual_zero(particular - x**2/2)


@pytest.mark.parametrize("parameter", [
    s.Symbol("_fsym_r"),
    s.Symbol("_fsym_r", positive=True),
])
def test_characteristic_basis_helper_keeps_same_named_coefficient(parameter):
    from sympy.solvers.ode import _assemble, _homogeneous_basis

    x = s.Symbol("x")
    groups = _homogeneous_basis([-parameter, s.Integer(1)], x)
    solution, basis = _assemble(groups, x)
    assert len(basis) == 1
    _residual_zero(s.diff(solution, x) - parameter*solution)
    _residual_zero(basis[0] - s.exp(parameter*x))
