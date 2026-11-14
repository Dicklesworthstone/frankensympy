"""Bounded nonlinear ODE families and the original-source trust boundary."""

from unittest.mock import patch

import pytest

from sympy import (
    Add, Derivative, Dummy, Eq, Function, Integer, Mul, Pow, Rational,
    Symbol, dsolve, exp, sqrt,
)
from sympy.core import _native, _native_expr, _symbol_binding


def equation(n=2, p=2, forcing=None, variable=None):
    x = variable if variable is not None else Symbol("bernoulli_x")
    y = Function("bernoulli_y")(x)
    q = x if forcing is None else forcing(x)
    return x, y, Derivative(y, x) + p*y - q*y**n


@pytest.mark.parametrize("hint", ["default", "Bernoulli"])
def test_quadratic_nonlinearity_reaches_checked_native_family(hint):
    x, y, source = equation()
    result = dsolve(source, y, hint=hint)
    assert type(result) is Eq
    assert result.lhs is y
    assert result.rhs == 1/(Symbol("C1")*exp(2*x) + x/2 + Rational(1, 4))


def test_scaled_original_equation_and_both_equality_sides_agree():
    x, y, source = equation()
    expected = dsolve(source, y)
    assert dsolve(Rational(-7, 3)*source, y) == expected
    assert dsolve(Eq(Derivative(y, x) + 2*y, x*y**2, evaluate=False), y) == expected


@pytest.mark.parametrize("n, offset, rate", [(3, Rational(1, 8), 4), (-1, Rational(-1, 8), -4)])
def test_both_quadratic_algebraic_branches_are_returned(n, offset, rate):
    x, y, source = equation(n=n)
    result = dsolve(source, y, hint="Bernoulli")
    u = Symbol("C1")*exp(rate*x) + x/2 + offset
    root = sqrt(1/u if n == 3 else u)
    assert result == [Eq(y, -root, evaluate=False), Eq(y, root, evaluate=False)]


def test_zero_linear_coefficient_integrates_polynomial_forcing():
    x, y, source = equation(n=2, p=0, forcing=lambda x: 3*x*x + 2)
    assert dsolve(source, y, hint="Bernoulli") == Eq(y, 1/(Symbol("C1") - x**3 - 2*x))


@pytest.mark.parametrize("n", [-16, -2, 4, 16])
def test_higher_integer_powers_remain_branch_neutral_relations(n):
    x, y, source = equation(n=n, p=0)
    result = dsolve(source, y, hint="Bernoulli")
    assert result.lhs == y**(1-n)
    assert result.rhs == Symbol("C1") + Rational(1-n, 2)*x*x


def test_simplify_false_preserves_implicit_nonzero_relation():
    x, y, source = equation()
    result = dsolve(source, y, hint="Bernoulli", simplify=False)
    assert result.lhs == 1/y
    assert result.rhs == Symbol("C1")*exp(2*x) + x/2 + Rational(1, 4)


@pytest.mark.parametrize("x", [Dummy("bernoulli_x"), Symbol("bernoulli_x", positive=True), Symbol("C1")])
def test_typed_coordinates_and_fresh_constants_are_retained(x):
    _, y, source = equation(variable=x)
    result = dsolve(source, y)
    assert result.lhs is y
    assert any(symbol is x for symbol in result.rhs.free_symbols)
    constant = Symbol("C2") if x.name == "C1" else Symbol("C1")
    assert result.rhs.free_symbols == {x, constant}


def test_degree_sixteen_polynomial_is_supported_but_larger_forcing_refuses():
    x, y, source = equation(n=2, p=0, forcing=lambda x: 17*x**16)
    result = dsolve(source, y, hint="Bernoulli")
    assert result.rhs == 1/(Symbol("C1") - x**17)
    x, y, source = equation(n=2, forcing=lambda x: x**17)
    with pytest.raises(ValueError, match="Bernoulli"):
        dsolve(source, y, hint="Bernoulli")


def test_nonmatching_mixed_powers_and_variable_linear_coefficient_refuse():
    x, y, source = equation()
    for unsupported in (source-y**3, Derivative(y, x)+x*y-x*y**2,
                        Derivative(y, x)+2*y-x*y**17):
        with pytest.raises(ValueError, match="Bernoulli"):
            dsolve(unsupported, y, hint="Bernoulli")


def test_original_undefined_children_cannot_disappear_under_zero_or_zero_power():
    x, y, source = equation()
    bad = Pow(Integer(0), Integer(-1), evaluate=False)
    for hole in (Mul(0, bad, evaluate=False), Pow(bad, 0, evaluate=False),
                 Mul(0, Pow(0, 0, evaluate=False), evaluate=False)):
        held = Add(source, hole, evaluate=False)
        with pytest.raises(ValueError):
            dsolve(held, y, hint="Bernoulli")


def test_original_reciprocal_domains_cannot_be_lost_in_a_legacy_fallback():
    x, y, _ = equation()
    inverse = Pow(y, -1, evaluate=False)
    for erased in (Mul(0, inverse, evaluate=False),
                   Add(inverse, Mul(-1, inverse, evaluate=False), evaluate=False)):
        source = Add(Derivative(y, x), y, erased, evaluate=False)
        with patch.object(_native, "bernoulli_native", side_effect=AssertionError("wrong classifier")):
            with pytest.raises(ValueError, match="reciprocal domains"):
                dsolve(source, y, ics={y.func(0): 0})


def test_unknown_or_same_print_coordinates_do_not_enter_the_rational_lane():
    x = Symbol("bernoulli_x", positive=True)
    other = Symbol("bernoulli_x", negative=True)
    _, y, source = equation(variable=x)
    with pytest.raises(ValueError, match="Bernoulli"):
        dsolve(source + other*y**2, y, hint="Bernoulli")


def test_native_failures_are_terminal_and_not_retried_as_separable():
    _, y, source = equation()
    with patch.object(_native, "bernoulli_native", side_effect=ValueError("native budget exhausted")) as call:
        with pytest.raises(ValueError, match="native budget exhausted"):
            dsolve(source, y)
        assert call.call_count == 1


def test_original_source_mutation_during_native_execution_is_rejected():
    _, y, source = equation()
    original = _native.bernoulli_native
    retained = source._value

    def changed(*args):
        result = original(*args)
        source._value = _native_expr(42)
        return result

    try:
        with patch.object(_native, "bernoulli_native", side_effect=changed):
            with pytest.raises(ValueError, match="changed"):
                dsolve(source, y)
    finally:
        source._value = retained


def test_native_boundary_rejects_nonrational_sources_and_constant_aliases():
    x, c = Symbol("x"), Symbol("C1")
    for p, q, n, constant in ((x, [1], 2, c), (1, [], 2, c),
                              (1, [1, 0], 2, c), (1, [1], 1, c), (1, [1], 2, x)):
        with pytest.raises(ValueError):
            _native.bernoulli_native(_native_expr(p), [_native_expr(v) for v in q], n,
                                     _symbol_binding(x), _symbol_binding(constant))


def test_initial_conditions_are_explicitly_outside_the_current_lane():
    x, y, source = equation()
    with pytest.raises(NotImplementedError, match="initial conditions"):
        dsolve(source, y, hint="Bernoulli", ics={y.func(0): 1})


def test_existing_linear_and_separable_routes_do_not_need_the_new_native_api():
    x, y, _ = equation()
    with patch.object(_native, "bernoulli_native", side_effect=AssertionError("wrong classifier")):
        # This tests classification only; legacy solving uses the ordinary
        # production expansion/simplification helpers.
        from sympy.solvers._bernoulli import solve_bernoulli
        assert solve_bernoulli(Derivative(y, x)+y-x, y, "default", None, True) is None
        assert solve_bernoulli(Derivative(y, x)-y**2, y, "default", {y.func(0): 1}, True) is None


def test_axis_assumption_mutation_cannot_change_the_bound_equation():
    x, y, source = equation(variable=Symbol("bernoulli_axis_mutation", real=True))
    original = _native.bernoulli_native
    retained = dict(x._assumptions)

    def changed(*args):
        result = original(*args)
        x._assumptions["positive"] = True
        return result

    try:
        with patch.object(_native, "bernoulli_native", side_effect=changed):
            with pytest.raises(ValueError, match="assumptions"):
                dsolve(source, y)
    finally:
        x._assumptions.clear()
        x._assumptions.update(retained)


def test_stale_coordinate_assumptions_are_rejected_before_native_work():
    x, y, source = equation(variable=Symbol("bernoulli_stale_axis", real=True))
    retained = dict(x._assumptions)
    try:
        x._assumptions["positive"] = True
        with patch.object(_native, "bernoulli_native", side_effect=AssertionError("stale source admitted")):
            with pytest.raises(ValueError, match="native identity"):
                dsolve(source, y)
    finally:
        x._assumptions.clear()
        x._assumptions.update(retained)


def test_generated_constant_mutation_is_rejected_before_result_lifting():
    _, y, source = equation()
    constant = Symbol("C1")
    retained = dict(constant._assumptions)
    original = _native.bernoulli_native

    def changed(*args):
        result = original(*args)
        constant._assumptions["positive"] = True
        return result

    try:
        with patch.object(_native, "bernoulli_native", side_effect=changed):
            with pytest.raises(ValueError, match="assumptions"):
                dsolve(source, y)
    finally:
        constant._assumptions.clear()
        constant._assumptions.update(retained)


def test_undefined_function_mutation_is_rejected_before_result_publication():
    x = Symbol("bernoulli_function_mutation")
    function = Function("bernoulli_mutable_function")
    y = function(x)
    source = Derivative(y, x) + 2*y - x*y**2
    original = _native.bernoulli_native

    def changed(*args):
        result = original(*args)
        function.eval = classmethod(lambda cls, *args: Integer(0))
        return result

    try:
        with patch.object(_native, "bernoulli_native", side_effect=changed):
            with pytest.raises(NotImplementedError, match="undefined function"):
                dsolve(source, y)
    finally:
        del function.eval
