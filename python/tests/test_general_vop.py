"""Checked scalar variation of parameters beyond second order.

These tests require the actual FrankenSymPy shell. Positive examples use
closed-form exact residuals, including repeated characteristic roots, Euler
singularities and typed axes; candidates and hidden source holes are separate
negative obligations.
"""

import pytest
import sympy as s

from sympy.solvers import ode as ode_module
from sympy.solvers.ode import _particular_vop


def assert_zero(expression):
    assert s.simplify(s.expand(expression)) == 0


def pure_derivative_problem(order, forcing, variable, leading=1):
    return _particular_vop(
        [variable**degree for degree in range(order)], forcing, leading, variable,
        coefficients=[0]*order + [leading],
    )


@pytest.mark.parametrize("order,leading,forcing_power", [(3, 1, -1), (4, 3, -2)])
def test_public_higher_order_rational_forcing(order, leading, forcing_power):
    x = s.Symbol("vop_rational_x", positive=True)
    y = s.Function("vop_rational_y")(x)
    forcing = x**forcing_power
    equation = leading*s.diff(y, x, order) - forcing
    solution = s.dsolve(equation, y)
    assert solution.lhs == y
    assert solution.rhs.free_symbols == {x, *(s.Symbol(f"C{i}") for i in range(1, order+1))}
    assert_zero(leading*s.diff(solution.rhs, x, order) - forcing)
    assert s.checkodesol(equation, solution, y) == (True, 0)


def test_explicit_hint_handles_resonance_with_a_repeated_characteristic_root():
    x = s.Symbol("vop_resonant_x", real=True)
    y = s.Function("vop_resonant_y")(x)
    equation = s.diff(y, x, 3) - 3*s.diff(y, x, 2) + 3*s.diff(y, x) - y - s.exp(x)
    solution = s.dsolve(equation, y, hint="nth_linear_constant_coeff_variation_of_parameters")
    assert_zero(s.diff(solution.rhs, x, 3) - 3*s.diff(solution.rhs, x, 2)
                + 3*s.diff(solution.rhs, x) - solution.rhs - s.exp(x))
    assert solution.rhs.free_symbols == {x, s.Symbol("C1"), s.Symbol("C2"), s.Symbol("C3")}


def test_default_route_combines_rational_forcing_with_a_repeated_exponential_basis():
    x = s.Symbol("vop_exp_rational_x", positive=True)
    y = s.Function("vop_exp_rational_y")(x)
    equation = (s.diff(y, x, 3) - 3*s.diff(y, x, 2) + 3*s.diff(y, x)
                - y - s.exp(x)/x)
    solution = s.dsolve(equation, y)
    assert_zero(s.diff(solution.rhs, x, 3) - 3*s.diff(solution.rhs, x, 2)
                + 3*s.diff(solution.rhs, x) - solution.rhs - s.exp(x)/x)


@pytest.mark.parametrize("hint", ["default", "nth_linear_euler_eq_nonhomogeneous_variation_of_parameters"])
def test_euler_equation_uses_the_original_variable_leading_coefficient(hint):
    x = s.Symbol("vop_euler_x", positive=True)
    y = s.Function("vop_euler_y")(x)
    equation = x**3*s.diff(y, x, 3) - 1
    solution = s.dsolve(equation, y, hint=hint)
    assert_zero(x**3*s.diff(solution.rhs, x, 3) - 1)
    assert_zero(s.diff(solution.rhs - s.log(x)/2, x, 3))


def test_higher_order_initial_conditions_are_checked_against_the_full_solution():
    x = s.Symbol("vop_initial_x", positive=True)
    f = s.Function("vop_initial_y")
    y = f(x)
    equation = s.diff(y, x, 3) - 1/x
    solution = s.dsolve(equation, y, ics={
        f(1): 0,
        s.diff(y, x).subs(x, 1): 0,
        s.diff(y, x, 2).subs(x, 1): 0,
    })
    assert solution.rhs.free_symbols == {x}
    assert_zero(s.diff(solution.rhs, x, 3) - 1/x)
    for order in range(3):
        assert_zero(s.diff(solution.rhs, x, order).subs(x, 1))


def test_exact_dummy_coordinate_and_same_print_parameter_stay_distinct():
    for x in (s.Dummy("vop_axis", positive=True), s.Symbol("vop_axis", positive=True)):
        parameter = s.Symbol("vop_axis", negative=True)
        result = pure_derivative_problem(3, parameter/x, x)
        assert result.free_symbols == {x, parameter}
        assert any(symbol is x for symbol in result.free_symbols)
        assert any(symbol is parameter for symbol in result.free_symbols)
        assert_zero(s.diff(result, x, 3) - parameter/x)


def test_eighth_order_boundary_remains_exact():
    x = s.Symbol("vop_eighth_x", positive=True)
    result = pure_derivative_problem(8, 1/x, x)
    assert_zero(s.diff(result, x, 8) - 1/x)


@pytest.mark.parametrize("leading", [0, s.Symbol("vop_unknown_leading")])
def test_zero_or_undecided_leading_coefficient_refuses(leading):
    x = s.Symbol("vop_lead_x", positive=True)
    with pytest.raises(NotImplementedError, match="leading coefficient"):
        pure_derivative_problem(3, 1/x, x, leading)


def test_dependent_basis_cannot_publish_a_particular_solution():
    x = s.Symbol("vop_dependent_x", positive=True)
    with pytest.raises(NotImplementedError, match="Wronskian"):
        _particular_vop([1, x, x], 1/x, 1, x, coefficients=[0, 0, 0, 1])


def test_incorrect_homogeneous_basis_is_rejected_before_integrating(monkeypatch):
    x = s.Symbol("vop_wrong_basis_x", positive=True)
    monkeypatch.setattr(s, "integrate", lambda *args: pytest.fail("invalid basis reached integration"))
    with pytest.raises(ValueError, match="homogeneous basis"):
        _particular_vop([1, x, s.exp(x)], 1/x, 1, x, coefficients=[0, 0, 0, 1])


def test_wrong_adjugate_candidate_fails_an_independent_wronskian_residual(monkeypatch):
    x = s.Symbol("vop_cofactor_x", positive=True)
    original = ode_module._vop_cofactors

    def incorrect(rows, work):
        values = original(rows, work)
        values[0] *= 2
        return values

    monkeypatch.setattr(ode_module, "_vop_cofactors", incorrect)
    with pytest.raises(ValueError, match="linear solve failed"):
        pure_derivative_problem(3, 1/x, x)


@pytest.mark.parametrize("perturbation", [s.Integer(1), s.Rational(1, 10**60)])
def test_wrong_antiderivative_is_rejected_exactly(monkeypatch, perturbation):
    x = s.Symbol("vop_bad_primitive_x", positive=True)
    original = s.integrate
    monkeypatch.setattr(s, "integrate", lambda value, axis: original(value, axis) + perturbation*axis)
    with pytest.raises(ValueError, match="primitive failed"):
        pure_derivative_problem(3, 1/x, x)


def test_antiderivative_cannot_introduce_an_unowned_parameter(monkeypatch):
    x = s.Symbol("vop_unowned_x", positive=True)
    foreign = s.Symbol("vop_unowned_constant")
    original = s.integrate
    monkeypatch.setattr(s, "integrate", lambda value, axis: original(value, axis) + foreign)
    with pytest.raises(ValueError, match="unrelated symbol"):
        pure_derivative_problem(3, 1/x, x)


def test_unevaluated_primitive_is_a_terminal_refusal(monkeypatch):
    x = s.Symbol("vop_formal_x", positive=True)
    monkeypatch.setattr(s, "integrate", lambda value, axis: s.Integral(value, axis))
    with pytest.raises(NotImplementedError, match="particular integral not found"):
        pure_derivative_problem(3, 1/x, x)


def test_integrator_operational_failure_propagates_without_retry(monkeypatch):
    x = s.Symbol("vop_failure_x", positive=True)
    failure = InterruptedError("cancelled bounded VOP integration")
    calls = []

    def interrupted(*args):
        calls.append(args)
        raise failure

    monkeypatch.setattr(s, "integrate", interrupted)
    with pytest.raises(InterruptedError) as caught:
        pure_derivative_problem(3, 1/x, x)
    assert caught.value is failure
    assert len(calls) == 1


@pytest.mark.parametrize("base", [s.Integer(0), s.Symbol("vop_hidden_pole")])
def test_original_hidden_inverse_domains_are_not_cancelled(base):
    x = s.Symbol("vop_domain_x", positive=True)
    hole = s.Mul(0, s.Pow(base, -1, evaluate=False), evaluate=False)
    source = s.Add(1/x, hole, evaluate=False)
    with pytest.raises((ValueError, NotImplementedError)):
        _particular_vop([1, x, x*x], 1/x, 1, x, coefficients=[0, 0, 0, 1], source=source)


def test_original_undefined_logarithm_cannot_hide_in_a_zero_product():
    x = s.Symbol("vop_log_domain_x", positive=True)
    hole = s.Mul(0, s.log(0, evaluate=False), evaluate=False)
    source = s.Add(1/x, hole, evaluate=False)
    with pytest.raises(ValueError, match="undefined logarithm"):
        _particular_vop([1, x, x*x], 1/x, 1, x, coefficients=[0, 0, 0, 1], source=source)


def test_symbol_mutation_during_integration_prevents_publication(monkeypatch):
    x = s.Symbol("vop_mutation_x", positive=True)
    original = s.integrate
    facts = x._assumptions

    def changed(value, axis):
        result = original(value, axis)
        object.__setattr__(x, "_assumptions", {"positive": False})
        return result

    monkeypatch.setattr(s, "integrate", changed)
    try:
        with pytest.raises(ValueError, match="symbol changed"):
            pure_derivative_problem(3, 1/x, x)
    finally:
        object.__setattr__(x, "_assumptions", facts)


@pytest.mark.parametrize("grow", [False, True])
def test_in_place_assumption_mutation_refuses_without_comparing_foreign_objects(monkeypatch, grow):
    x = s.Symbol("vop_inplace_x", positive=True)
    original = s.integrate
    facts, entries = x._assumptions, dict(x._assumptions)

    class ForeignFact:
        def __eq__(self, other):
            pytest.fail("source recheck invoked a foreign assumption comparison")

    def changed(value, axis):
        result = original(value, axis)
        if grow:
            facts.update((f"extra_{index}", ForeignFact()) for index in range(100))
        else:
            facts["positive"] = ForeignFact()
        return result

    monkeypatch.setattr(s, "integrate", changed)
    try:
        with pytest.raises(ValueError, match="symbol changed"):
            pure_derivative_problem(3, 1/x, x)
    finally:
        facts.clear()
        facts.update(entries)


def test_order_and_expansion_bounds_precede_candidate_generation(monkeypatch):
    x = s.Symbol("vop_bound_x", positive=True)
    monkeypatch.setattr(ode_module, "_vop_cofactors", lambda *args: pytest.fail("oversized source reached generation"))
    with pytest.raises(NotImplementedError, match="three through eight"):
        pure_derivative_problem(9, 1/x, x)
    for source in (s.Pow(x + 1, 10**6, evaluate=False), s.Pow(x + 1, 64, evaluate=False)):
        with pytest.raises(ValueError, match="bound|4096 terms"):
            _particular_vop([1, x, x*x], 1/x, 1, x, coefficients=[0, 0, 0, 1], source=source)


def test_approximate_sources_do_not_enter_exact_vop():
    x = s.Symbol("vop_float_x", positive=True)
    with pytest.raises(NotImplementedError, match="exact sources"):
        pure_derivative_problem(3, s.Float(0.5)/x, x)


def test_mismatched_explicit_linear_hint_is_not_silently_ignored():
    x = s.Symbol("vop_hint_x", positive=True)
    y = s.Function("vop_hint_y")(x)
    with pytest.raises(ValueError, match="does not match"):
        s.dsolve(x**3*s.diff(y, x, 3) - 1, y,
                 hint="nth_linear_constant_coeff_variation_of_parameters")
    with pytest.raises(ValueError, match="does not match"):
        s.dsolve(s.diff(y, x, 3) - 1/x, y,
                 hint="nth_linear_euler_eq_nonhomogeneous_variation_of_parameters")


def test_first_and_second_order_particular_solutions_keep_their_existing_path():
    x = s.Symbol("vop_legacy_x", positive=True)
    first = _particular_vop([s.Integer(1)], x, 1, x)
    second = _particular_vop([s.Integer(1), x], 1/x, 1, x)
    assert_zero(s.diff(first, x) - x)
    assert_zero(s.diff(second, x, 2) - 1/x)
