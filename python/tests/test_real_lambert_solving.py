"""Complete real Lambert branches, original sources, and terminal requests.

This bounded solveset extension returns explicit finite sets where the pinned
upstream solveset commonly retains ConditionSet. Native coefficient/branch
certificates establish completeness; these are not upstream form-parity tests.
"""

import importlib

import pytest
import sympy as s


CANDIDATE = hasattr(s.Matrix([[1]]), "_native_storage")
pytestmark = pytest.mark.skipif(not CANDIDATE, reason="complete real Lambert solver extension")
STEPS = 20_000_000


def no_fallback(*args, **kwargs):
    raise AssertionError("an admitted real Lambert equation used another solver")


@pytest.mark.parametrize("q", [s.Rational(1, 7), s.Integer(1), s.Integer(5)])
def test_positive_argument_has_exactly_one_real_branch(q, monkeypatch):
    from sympy.core import _native

    x = s.Symbol("lambert_positive")
    monkeypatch.setattr(_native, "solve_native", no_fallback)
    expected = s.FiniteSet(s.LambertW(q))
    assert s.solveset(x*s.exp(x)-q, x, s.S.Reals) == expected
    assert s.solveset(s.Eq(x*s.exp(x), q, evaluate=False), x, s.S.Reals) == expected


@pytest.mark.parametrize("q", [-s.Rational(1, 4), -s.Rational(1, 3), -s.Rational(1, 10**20)])
def test_negative_arguments_above_branch_point_keep_both_real_branches(q, monkeypatch):
    from sympy.core import _native

    x = s.Dummy("lambert_negative")
    monkeypatch.setattr(_native, "solve_native", no_fallback)
    result = s.solveset(x*s.exp(x)-q, x, s.S.Reals)
    assert result == s.FiniteSet(s.LambertW(q), s.LambertW(q, -1))
    assert len(result) == 2


@pytest.mark.parametrize("q", [-s.Integer(1), -s.Rational(1, 2), -s.Integer(100)])
def test_negative_arguments_below_branch_point_have_no_real_root(q, monkeypatch):
    from sympy.core import _native

    x = s.Symbol("lambert_empty")
    monkeypatch.setattr(_native, "solve_native", no_fallback)
    assert s.solveset(x*s.exp(x)-q, x, s.S.Reals) == s.S.EmptySet


@pytest.mark.parametrize("scale,shift", [(1, 0), (2, 3), (-3, 2), (s.Rational(2, 7), -1)])
def test_exponential_equals_affine_has_both_roots_under_invertible_coordinate_changes(scale, shift):
    x = s.Symbol("lambert_affine")
    y = scale*x + shift
    result = s.solveset(s.Eq(s.exp(y), 3*y, evaluate=False), x, s.S.Reals)
    assert result == s.FiniteSet(
        (-s.LambertW(-s.Rational(1, 3))-shift)/scale,
        (-s.LambertW(-s.Rational(1, 3), -1)-shift)/scale,
    )


@pytest.mark.parametrize("scale,shift", [(1, 0), (2, 3), (-3, 2)])
def test_exact_branch_point_is_one_distinct_double_root(scale, shift):
    x = s.Symbol("lambert_double")
    y = scale*x + shift
    assert s.solveset(s.exp(y)-y-1, x, s.S.Reals) == s.FiniteSet(-s.Rational(shift, scale))


def test_affine_multiplier_and_exponent_shifts_are_bound_independently():
    x = s.Symbol("lambert_shifted")
    source = (2*x+3)*s.exp(4*x-5)-7
    assert s.solveset(source, x, s.S.Reals) == s.FiniteSet(
        s.LambertW(14*s.exp(11))/4-s.Rational(3, 2))


@pytest.mark.parametrize("offset", [0, 1, -2, s.Rational(3, 7)])
def test_zero_target_keeps_the_single_finite_root(offset):
    x = s.Symbol("lambert_zero")
    source = (x-offset)*s.exp(2*x+3)
    assert s.solveset(source, x, s.S.Reals) == s.FiniteSet(offset)


def test_variable_inference_and_cancellation_preserve_the_original_equation():
    x = s.Symbol("lambert_inferred")
    source = s.Add(x*s.exp(x), x*x, -x*x, -1, evaluate=False)
    assert s.solveset(source, domain=s.S.Reals) == s.FiniteSet(s.LambertW(1))
    equation = s.Eq(source, s.Integer(0), evaluate=False)
    assert s.solveset(equation, x, s.S.Reals) == s.FiniteSet(s.LambertW(1))


@pytest.mark.parametrize("domain,expected", [
    (s.Interval(-1, 1), s.FiniteSet(0)),
    (s.Interval.open(0, 1), s.S.EmptySet),
    (s.FiniteSet(-2, 0, 3), s.FiniteSet(0)),
])
def test_rational_branch_point_roots_are_filtered_by_exact_real_domain(domain, expected):
    x = s.Symbol("lambert_restricted")
    assert s.solveset(s.exp(x)-x-1, x, domain) == expected


def test_unknown_order_of_lambert_values_retains_an_explicit_intersection():
    x = s.Symbol("lambert_interval")
    domain = s.Interval(0, 1)
    result = s.solveset(x*s.exp(x)-1, x, domain)
    assert result == s.Intersection(domain, s.FiniteSet(s.LambertW(1)))


@pytest.mark.parametrize("wrapper", [
    lambda pole: s.Mul(0, pole, evaluate=False),
    lambda pole: s.Pow(pole, 0, evaluate=False),
])
def test_original_undefined_constant_subexpressions_cannot_disappear(wrapper):
    x = s.Symbol("lambert_hidden_pole")
    zero = s.Add(1, -1, evaluate=False)
    pole = s.Pow(zero, -1, evaluate=False)
    source = s.Add(x*s.exp(x)-1, wrapper(pole), evaluate=False)
    with pytest.raises(ValueError, match="original|nonpositive|zero"):
        s.solveset(source, x, s.S.Reals)


def test_variable_poles_are_not_normalized_into_a_globally_defined_certificate():
    from sympy.core import _native, _native_expr, _symbol_binding

    x = s.Symbol("lambert_variable_pole")
    pole = s.Pow(x, -1, evaluate=False)
    source = s.Add(x*s.exp(x)-1, s.Mul(0, pole, evaluate=False), evaluate=False)
    assert _native.solve_real_lambert_native(_native_expr(source), _symbol_binding(x)) is None


def test_distinct_typed_coordinates_and_unmatched_exponentials_do_not_enter_the_lane():
    from sympy.core import _native, _native_expr, _symbol_binding

    x, other = s.Dummy("same"), s.Dummy("same")
    for source in (x*s.exp(other)-1, x*s.exp(x)+s.exp(2*x)-1, x+s.exp(x)*x):
        assert _native.solve_real_lambert_native(_native_expr(source), _symbol_binding(x)) is None


def test_retained_source_mutation_cannot_publish_stale_native_roots(monkeypatch):
    importlib.import_module("sympy.solvers._lambert_real")
    coefficient = s.Rational(37, 19)
    x = s.Symbol("lambert_mutation")
    source = s.Add(s.Mul(coefficient, x, s.exp(x), evaluate=False), -1, evaluate=False)
    monkeypatch.setattr(coefficient, "_value", s.Integer(2)._value)
    with pytest.raises((ValueError, RuntimeError), match="retained|source|match"):
        s.solveset(source, x, s.S.Reals)


def test_replaced_native_solver_is_refused_before_invocation(monkeypatch):
    from sympy.core import _native

    importlib.import_module("sympy.solvers._lambert_real")
    calls = []

    def forged(*args, **kwargs):
        calls.append(True)
        return []

    monkeypatch.setattr(_native, "solve_real_lambert_native", forged)
    x = s.Symbol("lambert_interface")
    with pytest.raises(NotImplementedError, match="original native interfaces"):
        s.solveset(x*s.exp(x)-1, x, s.S.Reals)
    assert calls == []


@pytest.mark.parametrize("name", ["Add", "Mul", "py_integer"])
def test_replaced_held_source_builders_cannot_change_the_verified_equation(name, monkeypatch):
    from sympy.core import _native

    helper = importlib.import_module("sympy.solvers._lambert_real")
    x = s.Symbol("lambert_source_builder")
    source = x*s.exp(x)-1
    calls = []

    def changed_source(*args, **kwargs):
        calls.append(True)
        raise AssertionError("replaced source builder was invoked")

    monkeypatch.setattr(_native, name, changed_source)
    with pytest.raises(NotImplementedError, match="original native interfaces"):
        helper.solve_lambert_real(source, x, s.S.Reals)
    assert calls == []


def test_native_generator_and_verifier_share_cumulative_callback_accounting():
    from sympy.core import SupervisedCallbackSession, _native, _native_expr, _symbol_binding

    x = s.Symbol("lambert_meter")
    source, binding = _native_expr(x*s.exp(x)+s.Rational(1, 4)), _symbol_binding(x)

    def run(session):
        return _native.solve_real_lambert_native(source, binding, session=session)

    measured = SupervisedCallbackSession(max_nested_steps=STEPS)
    assert len(run(measured)) == 2
    cost = measured.consumed_nested_steps
    assert cost > 0
    limited = SupervisedCallbackSession(max_nested_steps=2*cost-1)
    assert len(run(limited)) == 2
    with pytest.raises(ValueError, match="budget exhausted"):
        run(limited)
    spent = limited.consumed_nested_steps
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        run(limited)
    assert limited.consumed_nested_steps == spent


def test_cancelled_public_request_cannot_publish_roots():
    from sympy.core import SupervisedCallbackSession, supervised_callback_session

    x = s.Symbol("lambert_cancel")
    source = x*s.exp(x)-1
    session = SupervisedCallbackSession(max_nested_steps=STEPS)
    session.cancel()
    with supervised_callback_session(session):
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            s.solveset(source, x, s.S.Reals)


def test_oversized_exact_coefficient_is_a_terminal_resource_failure():
    x = s.Symbol("lambert_bits")
    with pytest.raises(ValueError, match="limit|bound|bits"):
        s.solveset(x*s.exp(x)-s.Integer(1 << 4097), x, s.S.Reals)
