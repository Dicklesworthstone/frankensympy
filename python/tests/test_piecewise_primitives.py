"""Continuous Piecewise polynomial primitives through the checked public lane.

The expected constants and endpoint conditions are pinned to SymPy 1.14.0.
Derivative assertions concern individual polynomial branches; they make no
claim that the Piecewise primitive is differentiable at an integrand jump.
"""

from fractions import Fraction

import pytest
import sympy as s
from sympy.core import _native, _native_expr, _symbol_binding
from sympy.integrals import _piecewise_primitive as primitive


def _forbid_fallback(*args, **kwargs):
    raise AssertionError("Piecewise primitive must not use a fallback integral lane")


def _equal(actual, expected):
    assert s.simplify(s.expand(actual - expected)) == 0


def _same_primitive(actual, expected):
    if type(expected) is not s.Piecewise:
        assert type(actual) is not s.Piecewise
        _equal(actual, expected)
        return
    assert type(actual) is s.Piecewise
    assert len(actual.args) == len(expected.args)
    for left, right in zip(actual.args, expected.args):
        assert left.cond == right.cond
        _equal(left.expr, right.expr)


@pytest.mark.parametrize("closed", [False, True])
def test_two_polynomial_branches_are_joined_continuously(closed, monkeypatch):
    x = s.Symbol("piecewise_primitive_polynomial", real=True)
    condition = x <= 1 if closed else x < 1
    source = s.Piecewise((x, condition), (x*x, True), evaluate=False)
    expected = s.Piecewise((x*x/2, condition), (x**3/3 + s.Rational(1, 6), True))
    monkeypatch.setattr(_native, "integrate_native", _forbid_fallback)
    actual = s.integrate(source, x)
    _same_primitive(actual, expected)
    _same_primitive(s.Integral(source, x).doit(), expected)
    _equal(actual.args[0].expr.subs(x, 1), actual.args[1].expr.subs(x, 1))
    _equal(s.diff(actual.args[0].expr, x), x)
    _equal(s.diff(actual.args[1].expr, x), x*x)


def test_leftmost_zero_constant_does_not_force_the_global_value_at_zero():
    x = s.Symbol("piecewise_primitive_anchor")
    source = s.Piecewise((1, x < -1), (2, True), evaluate=False)
    expected = s.Piecewise((x, x < -1), (2*x + 1, True))
    actual = s.integrate(source, x)
    _same_primitive(actual, expected)
    assert actual.subs(x, 0) == 1
    assert actual.args[0].expr.subs(x, 0) == 0


@pytest.mark.parametrize("closed", [False, True])
def test_reverse_branch_order_preserves_first_match_endpoint_ownership(closed):
    x = s.Symbol("piecewise_primitive_reverse")
    condition = x >= 1 if closed else x > 1
    source = s.Piecewise((x, condition), (2, True), evaluate=False)
    left_condition = x < 1 if closed else x <= 1
    expected = s.Piecewise((2*x, left_condition), (x*x/2 + s.Rational(3, 2), True))
    _same_primitive(s.integrate(source, x), expected)


def test_overlapping_conditions_propagate_constants_in_spatial_order():
    x = s.Symbol("piecewise_primitive_first_match")
    source = s.Piecewise((1, x < 0), (2, x < 1), (3, True), evaluate=False)
    original = source.args
    expected = s.Piecewise((x, x < 0), (2*x, x < 1), (3*x - 1, True))
    _same_primitive(s.integrate(source, x), expected)
    assert source.args == original
    duplicate = s.Piecewise((1, x >= 0), (99, x >= 0), (2, True), evaluate=False)
    _same_primitive(s.integrate(duplicate, x), s.Piecewise((2*x, x < 0), (x, True)))


def test_equal_neighboring_primitives_merge_after_each_original_branch_is_checked(monkeypatch):
    x = s.Symbol("piecewise_primitive_merge")
    source = s.Piecewise((1, x < 0), (2, x < 1), (2, x < 2), (3, True), evaluate=False)
    expected = s.Piecewise((x, x < 0), (2*x, x < 2), (3*x - 2, True))
    original = primitive._verify_derivative
    checked = []

    def record(candidate, selected, binding):
        checked.append(selected)
        return original(candidate, selected, binding)

    monkeypatch.setattr(primitive, "_verify_derivative", record)
    _same_primitive(s.integrate(source, x), expected)
    assert len(checked) == 4


def test_disconnected_nonlinear_guard_uses_complete_rational_partition(monkeypatch):
    x = s.Symbol("piecewise_primitive_nonlinear")
    source = s.Piecewise((x*x, x*x < 1), (1, True), evaluate=False)
    expected = s.Piecewise((x, x <= -1),
                           (x**3/3 - s.Rational(2, 3), x < 1),
                           (x - s.Rational(4, 3), True))
    monkeypatch.setattr(s, "_numeric_poly_roots", _forbid_fallback)
    monkeypatch.setattr(_native, "integrate_native", _forbid_fallback)
    _same_primitive(s.integrate(source, x), expected)


def test_compact_support_and_disconnected_first_branch_have_continuous_tails():
    x = s.Symbol("piecewise_primitive_boolean")
    source = s.Piecewise((1, s.And(x > -1, x < 1)), (0, True), evaluate=False)
    expected = s.Piecewise((0, x <= -1), (x + 1, x < 1), (2, True))
    _same_primitive(s.integrate(source, x), expected)
    outside = s.Piecewise((2, s.Or(x < -1, x > 1)), (1, True), evaluate=False)
    expected = s.Piecewise((2*x, x < -1), (x - 1, x <= 1), (2*x - 2, True))
    _same_primitive(s.integrate(outside, x), expected)


def test_rational_cuts_and_coefficients_keep_exact_continuity_constants():
    x = s.Symbol("piecewise_primitive_rational")
    q = s.Rational(1, 2)
    source = s.Piecewise((x/3, x < q), (x*x/5, True), evaluate=False)
    expected = s.Piecewise((x*x/6, x < q), (x**3/15 + s.Rational(1, 30), True))
    actual = s.integrate(source, x)
    _same_primitive(actual, expected)
    _equal(actual.args[0].expr.subs(x, q), actual.args[1].expr.subs(x, q))


@pytest.mark.parametrize("kind", ["value", "pole", "undefined", "gap"])
def test_singleton_values_and_gaps_do_not_change_the_continuous_primitive(kind):
    x = s.Symbol("piecewise_primitive_singleton_" + kind)
    if kind == "gap":
        source = s.Piecewise((1, x < 0), (1, x > 0), evaluate=False)
    else:
        value = {"value": lambda: s.Integer(99),
                 "pole": lambda: s.Pow(x, -1, evaluate=False),
                 "undefined": lambda: s.Pow(0, -1, evaluate=False)}[kind]()
        source = s.Piecewise((value, s.Eq(x, 0, evaluate=False)), (1, True), evaluate=False)
    _same_primitive(s.integrate(source, x), x)


def test_singleton_gap_between_different_slopes_gets_the_continuous_extension():
    x = s.Symbol("piecewise_primitive_point_gap")
    source = s.Piecewise((1, x < 0), (2, x > 0), evaluate=False)
    _same_primitive(s.integrate(source, x), s.Piecewise((x, x < 0), (2*x, True)))


def test_rational_guard_with_previously_selected_pole_is_supported():
    # The checked guard partition extends the pinned oracle, which leaves
    # this rational-guard request unevaluated. The expected result follows
    # exactly from the sign of 1/x on the two open half-lines.
    x = s.Symbol("piecewise_primitive_guard_pole")
    guard = s.Gt(s.Pow(x, -1, evaluate=False), 0, evaluate=False)
    source = s.Piecewise((99, s.Eq(x, 0, evaluate=False)), (1, guard), (2, True), evaluate=False)
    _same_primitive(s.integrate(source, x), s.Piecewise((2*x, x < 0), (x, True)))


@pytest.mark.parametrize("kind", ["open_gap", "missing_tail", "point_only", "rational",
                                  "function", "algebraic_cut", "unknown", "guard_pole"])
def test_unsupported_requests_preserve_the_entire_unevaluated_integral(kind, monkeypatch):
    x = s.Symbol("piecewise_primitive_unsupported_" + kind)
    source = {
        "open_gap": lambda: s.Piecewise((x, x < 0), (x*x, x > 1), evaluate=False),
        "missing_tail": lambda: s.Piecewise((x, x < 0), evaluate=False),
        "point_only": lambda: s.Piecewise((1, s.Eq(x, 0, evaluate=False)), evaluate=False),
        "rational": lambda: s.Piecewise((x, x < 0), (1/(x + 1), True), evaluate=False),
        "function": lambda: s.Piecewise((x, x < 0), (s.sin(x), True), evaluate=False),
        "algebraic_cut": lambda: s.Piecewise((1, x*x < 2), (2, True), evaluate=False),
        "unknown": lambda: s.Piecewise((1, s.Symbol("True")), (2, True), evaluate=False),
        "guard_pole": lambda: s.Piecewise((1, s.Gt(1/x, 0, evaluate=False)), (2, True), evaluate=False),
    }[kind]()
    monkeypatch.setattr(_native, "poly_coeffs_native", _forbid_fallback)
    monkeypatch.setattr(_native, "integrate_native", _forbid_fallback)
    actual = s.integrate(source, x)
    assert type(actual) is s.Integral
    assert actual.function is source
    assert actual.limits == ((x,),)


@pytest.mark.parametrize("dummy", [False, True])
def test_original_typed_symbol_owner_survives_each_native_boundary(dummy, monkeypatch):
    x = (s.Dummy("piecewise_primitive_owner", real=True) if dummy else
         s.Symbol("piecewise_primitive_owner", real=True))
    source = s.Piecewise((x, x < 0), (x*x, True), evaluate=False)
    original = _native.poly_coeffs_native
    seen = []

    def record(expression, binding):
        seen.append((binding.name, binding.identity_hex))
        return original(expression, binding)

    monkeypatch.setattr(_native, "poly_coeffs_native", record)
    result = s.integrate(source, x)
    assert type(result) is s.Piecewise
    assert result.free_symbols == {x}
    assert all(symbol is x for symbol in result.free_symbols)
    binding = _symbol_binding(x)
    assert seen == [(binding.name, binding.identity_hex)] * 2


def test_same_name_different_assumptions_are_not_an_admitted_univariate_source():
    x = s.Symbol("piecewise_primitive_collision", real=True)
    other = s.Symbol("piecewise_primitive_collision", positive=True)
    source = s.Piecewise((other, x < 0), (x, True), evaluate=False)
    result = s.integrate(source, x)
    assert type(result) is s.Integral
    assert result.function is source


def test_preexisting_typed_assumptions_mismatch_refuses_before_native_work(monkeypatch):
    x = s.Symbol("piecewise_primitive_stale_binding", real=True)
    source = s.Piecewise((x, x < 0), (x*x, True), evaluate=False)
    slot = s.Symbol.__dict__["_assumptions"]
    assumptions = slot.__get__(x, s.Symbol)
    previous = dict(assumptions)
    assumptions["positive"] = True
    monkeypatch.setattr(_native, "poly_coeffs_native", _forbid_fallback)
    monkeypatch.setattr(_native, "solve_real_rational_native", _forbid_fallback)
    try:
        with pytest.raises(ValueError, match="inconsistent typed assumptions"):
            s.integrate(source, x)
    finally:
        assumptions.clear()
        assumptions.update(previous)


def test_degree_sixty_four_primitive_is_checked_without_spurious_power_sixty_five_refusal():
    x = s.Symbol("piecewise_primitive_degree_limit")
    source = s.Piecewise((x**64, x < 0), (0, True), evaluate=False)
    expected = s.Piecewise((x**65/65, x < 0), (0, True))
    _same_primitive(s.integrate(source, x), expected)


@pytest.mark.parametrize("kind", ["degree", "hidden_zero", "hidden_zero_power", "coefficient"])
def test_expansion_limits_precede_the_unmetered_coefficient_converter(kind, monkeypatch):
    x = s.Symbol("piecewise_primitive_bound_" + kind)
    too_large = s.Pow(x, 65, evaluate=False)
    branch = {
        "degree": lambda: too_large,
        "hidden_zero": lambda: s.Mul(0, too_large, evaluate=False),
        "hidden_zero_power": lambda: s.Pow(too_large, 0, evaluate=False),
        "coefficient": lambda: s.Mul(s.Integer(1 << 3000), s.Integer(1 << 3000), evaluate=False),
    }[kind]()
    source = s.Piecewise((branch, x < 0), (0, True), evaluate=False)
    monkeypatch.setattr(_native, "poly_coeffs_native", _forbid_fallback)
    monkeypatch.setattr(_native, "integrate_native", _forbid_fallback)
    with pytest.raises(ValueError, match="limit"):
        s.integrate(source, x)


def test_python_rational_budget_failure_is_terminal(monkeypatch):
    x = s.Symbol("piecewise_primitive_work_limit")
    source = s.Piecewise((x, x < 0), (x*x, True), evaluate=False)
    monkeypatch.setattr(primitive, "_MAX_WORK", 1)
    monkeypatch.setattr(_native, "integrate_native", _forbid_fallback)
    with pytest.raises(ValueError, match="rational work limit"):
        s.integrate(source, x)


@pytest.mark.parametrize("failure", [ValueError("native budget exhausted"),
                                      NotImplementedError("native refusal"),
                                      MemoryError("native allocation refused")])
def test_later_coefficient_failure_propagates_without_partial_result(failure, monkeypatch):
    x = s.Symbol("piecewise_primitive_later_failure")
    source = s.Piecewise((x, x < 0), (x*x, True), evaluate=False)
    original = _native.poly_coeffs_native
    calls = []

    def fail_second(*args):
        calls.append(args)
        if len(calls) == 2:
            raise failure
        return original(*args)

    monkeypatch.setattr(_native, "poly_coeffs_native", fail_second)
    monkeypatch.setattr(_native, "integrate_native", _forbid_fallback)
    with pytest.raises(type(failure)) as error:
        s.integrate(source, x)
    assert error.value is failure
    assert len(calls) == 2


def test_incorrect_candidate_derivative_is_rejected_independently(monkeypatch):
    x = s.Symbol("piecewise_primitive_bad_derivative")
    source = s.Piecewise((1, x < 0), (2, True), evaluate=False)
    original = primitive._native_polynomial

    def corrupt(coefficients, variable):
        return original(coefficients, variable) + variable

    monkeypatch.setattr(primitive, "_native_polynomial", corrupt)
    monkeypatch.setattr(_native, "integrate_native", _forbid_fallback)
    with pytest.raises(ValueError, match="derivative identity was not verified"):
        s.integrate(source, x)


@pytest.mark.parametrize("kind", ["unsupported", "finite", "excluded_root"])
def test_derivative_checker_requires_the_complete_real_line(kind, monkeypatch):
    x = s.Symbol("piecewise_primitive_incomplete_proof_" + kind)
    source = s.Piecewise((1, x < 0), (2, True), evaluate=False)
    original = _native.poly_coeffs_native

    def incomplete(*args):
        if kind == "unsupported":
            return None
        return (kind == "excluded_root", _native.py_integer(1), int(kind == "excluded_root"))

    def after_partition(*args):
        result = original(*args)
        monkeypatch.setattr(_native, "solve_real_rational_native", incomplete)
        return result

    monkeypatch.setattr(_native, "poly_coeffs_native", after_partition)
    monkeypatch.setattr(_native, "integrate_native", _forbid_fallback)
    with pytest.raises(ValueError, match="derivative identity was not verified"):
        s.integrate(source, x)


@pytest.mark.parametrize("position, message", [(1, "initial constant"), (2, "continuity")])
def test_correct_derivatives_do_not_hide_bad_constants(position, message, monkeypatch):
    x = s.Symbol("piecewise_primitive_bad_constant")
    source = s.Piecewise((1, x < 0), (2, True), evaluate=False)
    original = primitive._native_polynomial
    calls = []

    def corrupt(coefficients, variable):
        calls.append(coefficients)
        candidate = original(coefficients, variable)
        return candidate + _native.py_integer(1) if len(calls) == position else candidate

    monkeypatch.setattr(primitive, "_native_polynomial", corrupt)
    monkeypatch.setattr(_native, "integrate_native", _forbid_fallback)
    with pytest.raises(ValueError, match=message + " was not verified"):
        s.integrate(source, x)


def test_source_mutation_during_generation_prevents_publication(monkeypatch):
    x = s.Symbol("piecewise_primitive_mutation")
    source = s.Piecewise((x, x < 0), (x*x, True), evaluate=False)
    original = _native.poly_coeffs_native
    slot = s.Piecewise.__dict__["_piecewise_branches"]

    def mutate(*args):
        result = original(*args)
        slot.__set__(source, ((s.Integer(99), s.true),))
        return result

    monkeypatch.setattr(_native, "poly_coeffs_native", mutate)
    with pytest.raises(RuntimeError, match="source mutated"):
        s.integrate(source, x)


def test_preexisting_held_argument_mismatch_refuses_before_partitioning(monkeypatch):
    x = s.Symbol("piecewise_primitive_stale_held")
    held = s.Add(x, 1, evaluate=False)
    source = s.Piecewise((held, x < 0), (0, True), evaluate=False)
    s.Add.__dict__["_args"].__set__(held, (x, s.Integer(2)))
    for name in ("solve_real_rational_native", "solve_real_rational_inequality_native",
                 "poly_coeffs_native", "integrate_native"):
        monkeypatch.setattr(_native, name, _forbid_fallback)
    with pytest.raises(ValueError, match="retained surface arguments"):
        s.integrate(source, x)


def test_rational_arithmetic_rejects_height_growth_without_float_conversion():
    work = primitive._Work()
    huge = work.value(Fraction(1 << 3000))
    with pytest.raises(ValueError, match="rational coefficient limit"):
        work.mul(huge, huge)


def test_native_endpoint_evaluator_checks_the_constructed_expression():
    x = s.Symbol("piecewise_primitive_endpoint_verifier", real=True)
    candidate = _native_expr((x*x + 1)*(x - s.Rational(1, 3)))
    value = primitive._evaluate_native(candidate, _native_expr(x), Fraction(1, 2), primitive._Work())
    assert value == Fraction(5, 24)
    with pytest.raises(ValueError, match="different typed variable"):
        primitive._evaluate_native(candidate, _native_expr(s.Symbol(str(x), positive=True)),
                                   Fraction(1, 2), primitive._Work())
