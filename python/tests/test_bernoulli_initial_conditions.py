"""Exact initial values, local branches and retained Bernoulli source domains."""

from unittest.mock import patch

import pytest

from sympy import (
    Add, Derivative, Dummy, Eq, Float, Function, Integer, Mul, Pow, Rational,
    Symbol, checkodesol, diff, dsolve, exp, simplify, sqrt,
)
from sympy.core import _native, _native_expr


def family(n=2, p=2, variable=None):
    x = Symbol("bernoulli_initial_x") if variable is None else variable
    y = Function("bernoulli_initial_y")(x)
    return x, y, Derivative(y, x) + p*y - x*y**n


@pytest.mark.parametrize("hint", ["default", "Bernoulli"])
def test_rational_point_binds_the_checked_nonseparable_family(hint):
    x, y, source = family()
    result = dsolve(source, y, hint=hint, ics={y.func(1): 2})
    expected = 1/(x/2 + Rational(1, 4) - exp(2*(x-1))/4)
    assert result == Eq(y, expected)
    assert result.rhs.subs(x, 1) == 2
    assert result.rhs.free_symbols == {x}
    # This evaluates the original nonlinear ODE, not its IC-fitting formula.
    assert checkodesol(source, result, y) == (True, Integer(0))


@pytest.mark.parametrize("n", [3, -1])
@pytest.mark.parametrize("value", [Rational(-2), Rational(2)])
def test_quadratic_graph_selects_the_original_value_branch(n, value):
    x, y, source = family(n=n)
    point = Rational(1, 2)
    result = dsolve(source, y, hint="Bernoulli", ics={y.func(point): value})
    particular = x/2 + (Rational(1, 8) if n == 3 else Rational(-1, 8))
    u = particular + (value**(1-n) - particular.subs(x, point))*exp((n-1)*2*(x-point))
    root = sqrt(1/u if n == 3 else u)
    assert result == Eq(y, root if value > 0 else -root)
    assert result.rhs.subs(x, point) == value
    # An exact transformed differential identity is a separate check from
    # the root sign and original initial value.
    assert simplify(diff(u, x) - (n-1)*2*u - (1-n)*x) == 0


@pytest.mark.parametrize("x", [Dummy("bernoulli_initial_x"), Symbol("C1")])
def test_initial_value_keeps_the_original_typed_coordinate(x):
    _, y, source = family(variable=x)
    result = dsolve(source, y, ics={y.func(0): 1})
    assert result.lhs is y
    assert result.rhs.free_symbols == {x}
    assert any(symbol is x for symbol in result.rhs.free_symbols)


def test_zero_linear_coefficient_and_negative_initial_value():
    x, y, _ = family()
    source = Derivative(y, x) - (3*x*x+2)*y*y
    result = dsolve(source, y, hint="Bernoulli", ics={y.func(1): -1})
    assert result == Eq(y, 1/(2-x**3-2*x))
    assert result.rhs.subs(x, 1) == -1
    assert checkodesol(source, result, y) == (True, Integer(0))


@pytest.mark.parametrize("n", [-16, -3, -2, 4, 16])
def test_implicit_power_relation_cannot_replace_a_selected_initial_branch(n):
    _, y, source = family(n=n, p=0)
    with patch.object(_native, "bernoulli_native", side_effect=AssertionError("ambiguous initial branch")):
        with pytest.raises(NotImplementedError, match="unambiguous explicit branch"):
            dsolve(source, y, hint="Bernoulli", ics={y.func(0): -2})


@pytest.mark.parametrize("n", [3, -1])
def test_simplify_false_cannot_erase_a_quadratic_initial_branch(n):
    _, y, source = family(n=n)
    with patch.object(_native, "bernoulli_native", side_effect=AssertionError("ambiguous initial branch")):
        with pytest.raises(NotImplementedError, match="unambiguous explicit branch"):
            dsolve(source, y, hint="Bernoulli", simplify=False, ics={y.func(0): -2})


def test_simplify_false_retains_the_bound_implicit_relation():
    x, y, source = family()
    result = dsolve(source, y, hint="Bernoulli", simplify=False, ics={y.func(0): 1})
    assert result == Eq(1/y, Rational(3, 4)*exp(2*x) + x/2 + Rational(1, 4))


@pytest.mark.parametrize("n", [2, 3, 4, 16])
@pytest.mark.parametrize("simplify_result", [True, False])
def test_zero_initial_values_keep_the_separate_zero_solution(n, simplify_result):
    x, y, source = family(n=n)
    result = dsolve(source, y, hint="Bernoulli", simplify=simplify_result,
                    ics={y.func(0): 0, y.func(1): Integer(0)})
    assert result == Eq(y, 0)
    assert checkodesol(source, result, y) == (True, Integer(0))


@pytest.mark.parametrize("n", [-1, -2, -16])
def test_zero_initial_value_is_not_admitted_through_a_source_pole(n):
    _, y, source = family(n=n)
    with pytest.raises(ValueError, match="original Bernoulli domain"):
        dsolve(source, y, hint="Bernoulli", ics={y.func(0): 0})


def test_cancelled_reciprocal_still_excludes_the_zero_solution():
    _, y, source = family()
    hole = Mul(0, Pow(y, -1, evaluate=False), evaluate=False)
    retained = Add(source, hole, evaluate=False)
    with pytest.raises(ValueError, match="original Bernoulli domain"):
        dsolve(retained, y, hint="Bernoulli", ics={y.func(0): 0})


def test_conflicting_zero_and_nonzero_initial_values_are_rejected():
    _, y, source = family()
    with pytest.raises(ValueError, match="initial conditions"):
        dsolve(source, y, hint="Bernoulli", ics={y.func(0): 0, y.func(1): 1})


def test_consistent_overdetermined_values_select_a_particular_solution():
    x, y, source = family()
    result = dsolve(source, y, hint="Bernoulli",
                    ics={y.func(0): 4, y.func(1): Rational(4, 3)})
    assert result == Eq(y, 1/(x/2+Rational(1, 4)))
    assert checkodesol(source, result, y) == (True, Integer(0))


def test_squared_conditions_cannot_erase_a_conflicting_original_sign():
    _, y, source = family(n=3)
    # Both transformed observations lie on u=x/2+1/8. The same explicit
    # nonzero square-root branch cannot have opposite signs at these points.
    with pytest.raises(ValueError, match="initial conditions"):
        dsolve(source, y, hint="Bernoulli",
               ics={y.func(Rational(7, 4)): 1, y.func(Rational(31, 4)): Rational(-1, 2)})


def test_unknown_transcendental_condition_is_not_treated_as_equality():
    _, y, source = family()
    with pytest.raises(NotImplementedError, match="verify Bernoulli initial conditions"):
        dsolve(source, y, hint="Bernoulli", ics={y.func(0): 1, y.func(1): 1})


def test_high_power_multi_point_branches_remain_explicitly_unsupported():
    _, y, source = family(n=4, p=0)
    with pytest.raises(NotImplementedError, match="explicit branch"):
        dsolve(source, y, hint="Bernoulli", ics={y.func(0): 1, y.func(1): 2})


@pytest.mark.parametrize("bad_value", [Float(1), Symbol("C1"), True])
def test_inexact_symbolic_or_boolean_values_refuse_before_native_work(bad_value):
    _, y, source = family()
    with patch.object(_native, "bernoulli_native", side_effect=AssertionError("unadmitted data")):
        with pytest.raises(NotImplementedError, match="exact rational"):
            dsolve(source, y, hint="Bernoulli", ics={y.func(0): bad_value})


def test_symbolic_points_and_foreign_functions_are_not_rebound():
    _, y, source = family()
    for conditions in ({y.func(Symbol("a")): 1}, {Function("another_y")(0): 1}):
        with patch.object(_native, "bernoulli_native", side_effect=AssertionError("unadmitted data")):
            with pytest.raises(NotImplementedError):
                dsolve(source, y, hint="Bernoulli", ics=conditions)


def test_derivative_initial_data_remains_explicitly_unsupported():
    x, y, source = family()
    with pytest.raises(NotImplementedError, match="value initial conditions"):
        dsolve(source, y, hint="Bernoulli", ics={Derivative(y, x).subs(x, 0): 1})


def test_custom_mappings_and_value_conversion_hooks_are_not_executed():
    _, y, source = family()

    class Mapping(dict):
        def __bool__(self):
            raise AssertionError("custom mapping hook")

    class Value:
        def _sympy_(self):
            raise AssertionError("custom numeric hook")

    with pytest.raises(TypeError, match="built-in dictionary"):
        dsolve(source, y, hint="Bernoulli", ics=Mapping({y.func(0): 1}))
    with pytest.raises(NotImplementedError, match="exact rational"):
        dsolve(source, y, hint="Bernoulli", ics={y.func(0): Value()})


def test_initial_data_count_and_growth_are_preflighted():
    _, y, source = family(n=3)
    with patch.object(_native, "bernoulli_native", side_effect=AssertionError("unbounded data")):
        with pytest.raises(ValueError, match="16 observations"):
            dsolve(source, y, hint="Bernoulli", ics={y.func(i): 1 for i in range(17)})
        with pytest.raises(ValueError, match="1024 bits"):
            dsolve(source, y, hint="Bernoulli", ics={y.func(0): 1 << 1024})
    with pytest.raises(ValueError, match="transformed initial value"):
        dsolve(source, y, hint="Bernoulli", ics={y.func(0): 1 << 600})


def test_initial_mapping_mutation_during_native_generation_is_rejected():
    _, y, source = family()
    key = y.func(0)
    conditions = {key: 1}
    native = _native.bernoulli_native

    def changed(*args):
        result = native(*args)
        conditions[key] = 2
        return result

    with patch.object(_native, "bernoulli_native", side_effect=changed):
        with pytest.raises(ValueError, match="initial conditions changed"):
            dsolve(source, y, hint="Bernoulli", ics=conditions)


def test_initial_value_mutation_during_native_generation_is_rejected():
    _, y, source = family()
    value = Rational(7, 11)
    saved = value._value
    native = _native.bernoulli_native

    def changed(*args):
        result = native(*args)
        value._value = _native_expr(2)
        return result

    try:
        with patch.object(_native, "bernoulli_native", side_effect=changed):
            with pytest.raises(ValueError, match="source changed"):
                dsolve(source, y, hint="Bernoulli", ics={y.func(0): value})
    finally:
        value._value = saved
