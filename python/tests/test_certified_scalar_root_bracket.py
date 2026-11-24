"""Public exact scalar root existence and operational refusal contracts."""

from fractions import Fraction

import pytest

from sympy import (
    Abs, EvaluationBudget, EvaluationBudgetExceeded, EvaluationCancelled,
    Float, Mul, Piecewise, Pow, Rational, RootBoxReceipt,
    ScalarRootBracketReceipt, Symbol, certified_root_bracket,
)
from sympy.core import _native

x = Symbol("bracket_x", real=True)


def assert_contains(receipt, root, tolerance):
    lower, upper = receipt.bounds
    assert lower <= root <= upper
    assert upper - lower <= tolerance
    assert abs(receipt.midpoint - root) <= receipt.error_bound
    assert receipt.error_bound <= tolerance / 2


@pytest.mark.parametrize("expression", [
    (3*x - 1)**3,
    Abs(3*x - 1) + 2*(3*x - 1),
])
@pytest.mark.parametrize("sign", [1, -1])
def test_repeated_and_nonsmooth_roots_have_independent_existence_evidence(expression, sign):
    tolerance = Rational(1, 10**6)
    receipt = certified_root_bracket(sign*expression, (x, -1, 1), tolerance=tolerance)
    assert type(receipt) is ScalarRootBracketReceipt
    assert receipt.expression == sign*expression
    assert receipt.interval == (-1, 1)
    assert receipt.absolute_tolerance == tolerance
    actual_binding, expected_binding = receipt.variable, x._value.symbol_binding
    assert actual_binding.name == expected_binding.name
    assert actual_binding.plain == expected_binding.plain
    assert actual_binding.identity_hex == expected_binding.identity_hex
    assert_contains(receipt, Rational(1, 3), tolerance)
    left, right = receipt.endpoint_values
    assert (left[1] < 0 and right[0] > 0) or (left[0] > 0 and right[1] < 0)


@pytest.mark.parametrize("expression,interval,expected", [
    ((x-1)**3, (0, 2), 1),
    ((x-1)**2, (1, 2), 1),
    ((x-1)**2, (0, 1), 1),
    (x**2, (0, 0), 0),
    (0, (-2, 3), -2),
])
def test_exact_endpoints_midpoints_and_singleton_requests(expression, interval, expected):
    receipt = certified_root_bracket(expression, (x, *interval))
    assert receipt.bounds == (expected, expected)
    assert receipt.midpoint == expected and receipt.error_bound == 0
    assert receipt.endpoint_values == ((0, 0), (0, 0))


def test_multiple_roots_in_an_accepted_bracket_do_not_claim_uniqueness():
    receipt = certified_root_bracket(
        x*(x-1)*(x+1), (x, -2, 2), tolerance=4, max_bisections=0)
    assert receipt.bounds == (-2, 2)
    assert receipt.claim == "at least one real root in the certified interval"
    assert not isinstance(receipt, RootBoxReceipt)
    assert not hasattr(receipt, "contraction_bound")
    with pytest.raises(TypeError, match="native"):
        RootBoxReceipt(receipt._receipt)


def test_an_already_small_bracket_and_endpoint_need_no_search_iterations():
    receipt = certified_root_bracket(3*x-1, (x, 0, 1), tolerance=1, max_bisections=0)
    assert receipt.bounds == (0, 1)
    assert certified_root_bracket(x, (x, 0, 1), max_bisections=0).bounds == (0, 0)
    with pytest.raises(ValueError, match="width|accuracy"):
        certified_root_bracket(3*x-1, (x, 0, 1), tolerance=Rational(1, 100), max_bisections=0)


@pytest.mark.parametrize("expression", [x**2 + 1, (x-1)**2])
def test_same_sign_endpoints_do_not_assert_root_existence(expression):
    with pytest.raises(ValueError, match="opposite signs|zero value"):
        certified_root_bracket(expression, (x, -1, 2))


@pytest.mark.parametrize("wrapper", [
    lambda pole: pole,
    lambda pole: Mul(0, pole, evaluate=False),
    lambda pole: Pow(pole, 0, evaluate=False),
])
def test_original_poles_cannot_be_hidden_by_zero_or_zeroth_powers(wrapper):
    source = wrapper(Pow(x, -1, evaluate=False))
    with pytest.raises(ValueError):
        certified_root_bracket(source, (x, -1, 1))


def test_discontinuous_piecewise_sign_changes_are_refused():
    source = Piecewise((-1, x < 0), (1, True))
    with pytest.raises(ValueError):
        certified_root_bracket(source, (x, -1, 1))


def test_narrow_candidates_cannot_skip_a_pole_elsewhere_in_the_requested_interval():
    source = Mul(x-1, Pow(x, -1, evaluate=False), evaluate=False)
    with pytest.raises(ValueError):
        certified_root_bracket(source, (x, -1, 2))


@pytest.mark.parametrize("limits", [
    (x, Float("-1.0"), 1), (x, -1, 1.0), (x, True, 1), (x, 2, 1),
    (Symbol("unknown_assumption", positive=True), -1, 1),
])
def test_exact_bounds_and_symbol_assumptions_are_required(limits):
    with pytest.raises((TypeError, ValueError)):
        certified_root_bracket(limits[0], limits)


@pytest.mark.parametrize("options", [
    {"tolerance": 0}, {"tolerance": -1}, {"tolerance": 0.01},
    {"precision": 7}, {"precision": 129}, {"precision": True},
    {"max_bisections": -1}, {"max_bisections": 1025}, {"max_bisections": True},
])
def test_invalid_accuracy_and_resource_parameters_refuse(options):
    with pytest.raises((TypeError, ValueError)):
        certified_root_bracket(x, (x, -1, 1), **options)


def test_exact_fraction_inputs_and_undeclared_or_inexact_sources():
    receipt = certified_root_bracket(
        3*x-1, (x, Fraction(-1, 2), Fraction(1, 2)),
        tolerance=Fraction(1, 100))
    assert_contains(receipt, Rational(1, 3), Rational(1, 100))
    for source in (x + Symbol("unbound"), x + Float("0.1"), True):
        with pytest.raises((TypeError, ValueError)):
            certified_root_bracket(source, (x, -1, 1))


def test_receipts_have_no_public_mutation_or_unchecked_constructor():
    receipt = certified_root_bracket(x, (x, -1, 1))
    with pytest.raises(TypeError):
        ScalarRootBracketReceipt(object())
    with pytest.raises(TypeError):
        _native.ScalarRootBracketReceipt()
    with pytest.raises(AttributeError):
        receipt._receipt = None
    with pytest.raises(AttributeError):
        del receipt._receipt
    with pytest.raises(AttributeError):
        receipt.__init__(receipt._receipt)
    assert "ScalarRootBracketReceipt" in repr(receipt)


@pytest.mark.parametrize("phase", ["budget", "verification_budget"])
def test_each_phase_has_an_independent_terminal_budget(phase):
    budget = EvaluationBudget(max_steps=0)
    with pytest.raises(EvaluationBudgetExceeded):
        certified_root_bracket(x, (x, -1, 1), **{phase: budget})
    assert budget.remaining_steps == 0


@pytest.mark.parametrize("phase", ["budget", "verification_budget"])
def test_precancelled_phases_stop_before_source_capture(phase, monkeypatch):
    from sympy.solvers import certified

    budget = EvaluationBudget()
    budget.cancel()

    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled request copied its source")

    monkeypatch.setattr(certified, "_numeric_source", forbidden)
    with pytest.raises(EvaluationCancelled):
        certified_root_bracket(x, (x, -1, 1), **{phase: budget})


def test_one_explicit_ledger_can_cover_both_phases_and_repeated_calls():
    maximum = 10_000_000
    measured = EvaluationBudget(max_steps=maximum, max_memory_bytes=512*1024*1024)
    certified_root_bracket(x, (x, -1, 1), budget=measured, verification_budget=measured)
    cost = maximum - measured.remaining_steps
    assert cost > 0
    limited = EvaluationBudget(max_steps=2*cost-1, max_memory_bytes=512*1024*1024)
    certified_root_bracket(x, (x, -1, 1), budget=limited, verification_budget=limited)
    with pytest.raises(EvaluationBudgetExceeded):
        certified_root_bracket(x, (x, -1, 1), budget=limited, verification_budget=limited)
    assert limited.remaining_steps < cost


@pytest.mark.parametrize("phase", ["budget", "verification_budget"])
def test_cancellation_at_native_return_still_prevents_publication(phase, monkeypatch):
    original = _native.certified_scalar_root_bracket_native
    budget = EvaluationBudget()

    def cancelled_result(*args, **kwargs):
        result = original(*args, **kwargs)
        budget.cancel()
        return result

    monkeypatch.setattr(_native, "certified_scalar_root_bracket_native", cancelled_result)
    with pytest.raises(EvaluationCancelled):
        certified_root_bracket(x, (x, -1, 1), **{phase: budget})


def test_native_bridge_refuses_nonliteral_bounds_and_preserves_resource_errors():
    binding = x._value.symbol_binding
    with pytest.raises(TypeError, match="native rational"):
        _native.certified_scalar_root_bracket_native(
            x._value, binding, -1, Rational(1)._value, Rational(1, 100)._value)
    with pytest.raises(EvaluationBudgetExceeded, match="bisection"):
        _native.certified_scalar_root_bracket_native(
            x._value, binding, Rational(-1)._value, Rational(1)._value,
            Rational(1, 100)._value, max_bisections=1025)
