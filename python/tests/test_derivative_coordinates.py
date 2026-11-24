"""Coordinate selection and formal evaluation on the public scalar derivative APIs."""

import pytest
import sympy as s


def test_inference_and_held_forms_use_the_actual_coordinate():
    x = s.Dummy("t", real=True)
    source = x**3
    assert s.diff(source) == 3*x**2
    assert source.diff() == 3*x**2
    for held in (s.diff(source, evaluate=False), source.diff(evaluate=False),
                 s.Derivative(source)):
        assert held.expr == source and held.variables == (x,)
        assert held.doit() == 3*x**2
    assert s.diff(7) == s.S.Zero
    assert s.Derivative(s.Rational(2, 3)) == s.S.Zero


def test_no_guess_between_distinct_same_named_coordinates():
    x = s.Symbol("coordinate")
    y = s.Symbol("coordinate", positive=True)
    for operation in (s.diff, s.Derivative):
        with pytest.raises(ValueError):
            operation(x + y)
        with pytest.raises(ValueError):
            operation(s.Function("unknown_value")(0))


@pytest.mark.parametrize("evaluate", [False, True])
def test_zero_and_one_singleton_orders(evaluate):
    x = s.Symbol("coordinate")
    expr = x**3
    assert s.diff(expr, (x, s.S.Zero), evaluate=evaluate) == expr
    first = s.diff(expr, x, s.S.One, evaluate=evaluate)
    assert first == (3*x**2 if evaluate else s.Derivative(expr, x))
    with pytest.raises(ValueError):
        s.diff(expr, x, s.S.NegativeOne, evaluate=evaluate)


@pytest.mark.parametrize("evaluate", [False, True])
def test_only_independent_targets_can_be_coordinates(evaluate):
    x, y = s.symbols("x y")
    f = s.Function("coordinate_field")
    for invalid in (x*y, s.sin(x), s.Derivative(x**2, x), s.Derivative(f(x), y),
                    s.Derivative(f(x+x**2), x), s.Derivative(f(x, x), x)):
        with pytest.raises(ValueError):
            s.diff(x, invalid, evaluate=evaluate)
    field = f(x, y)
    jet = s.Derivative(field, x, y)
    source = field**2 + jet**2
    if evaluate:
        assert s.diff(source, field) == 2*field
        assert s.diff(source, jet) == 2*jet
    else:
        assert s.diff(source, jet, evaluate=False).variables == (jet,)


def test_holding_or_rejecting_an_order_preserves_callback_effects():
    x = s.Symbol("hook_coordinate")
    effects = []

    class Law(s.Function):
        def _eval_derivative(self, var):
            effects.append(var)
            return 5*x

    source = Law(x)
    held = source.diff(evaluate=False)
    assert held.expr is source and held.variables == (x,)
    assert effects == []
    with pytest.raises(ValueError):
        s.diff(source, x, x + 1)
    assert effects == []
    assert held.doit() == 5*x
    assert effects == [x]


def test_order_limit_is_checked_before_evaluating_a_prefix():
    from sympy.core import _MAX_DIFF_ORDER

    calls = []
    x, y = s.symbols("budget_x budget_y")

    class Tracked(s.Function):
        def _eval_derivative(self, var):
            calls.append(var)
            return self

    for constructor in (s.diff, s.Derivative):
        with pytest.raises(ValueError, match="order limit"):
            constructor(Tracked(x, y), x, (y, _MAX_DIFF_ORDER))
    assert calls == []


def test_explicit_independent_coordinate_declaration():
    class State(s.Function):
        _diff_wrt = True

    x = s.Symbol("state_coordinate")
    state = State(x)
    assert s.diff(state**2, state) == 2*state
