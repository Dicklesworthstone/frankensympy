"""Reciprocal inverse functions reach checked public numerical workflows."""

import math
from decimal import Decimal, localcontext

import pytest
import sympy as s

from sympy.core import _native_expr


# Independent rational derivative anchors. Circular values share atan(3/4),
# hyperbolic values share log(2); neither oracle evaluates the new functions.
CASES = {
    "acot": (4/3, math.atan(3/4), -9/25, 216/625, 2.0),
    "acoth": (5/3, math.log(2), -9/16, 135/128, 2.0),
    "asec": (5/4, math.atan(3/4), 16/15, -2176/675, 2.0),
    "acsc": (5/3, math.atan(3/4), -9/20, 1107/1600, 2.0),
    "asech": (4/5, math.log(2), -25/12, -875/432, 0.5),
    "acsch": (4/3, math.log(2), -9/20, 1107/2000, 1.0),
}


@pytest.mark.parametrize("name,numerator,denominator,expected", [
    ("acot", 4, 3, "0.64350110879328438680280922871732263804"),
    ("acoth", 5, 3, "0.69314718055994530941723212145817656808"),
    ("asec", 5, 4, "0.64350110879328438680280922871732263804"),
    ("acsc", 5, 3, "0.64350110879328438680280922871732263804"),
    ("asech", 4, 5, "0.69314718055994530941723212145817656808"),
    ("acsch", 4, 3, "0.69314718055994530941723212145817656808"),
])
def test_public_evalf_retains_thirty_digits_from_exact_real_arguments(
    name, numerator, denominator, expected,
):
    source = getattr(s, name)(s.Rational(numerator, denominator), evaluate=False)
    with localcontext() as context:
        context.prec = 50
        for value in (s.N(source, 30), source.evalf(30)):
            assert abs(Decimal(str(value)) - Decimal(expected)) < Decimal("1e-29")


# These references come from 110-digit fixed-point series on each exact
# binary64 input: log via an atanh series, atan(sqrt(x*x-1)), and Machin pi.
# They do not reuse the production reciprocal/atan2 value formulas.
@pytest.mark.parametrize("point,reference_values", [
    (1.00000001, (
        "9.55691396779489086403189279587969017858",
        "0.00014142135521831176881057371181605432",
        "1.57065490543967830746251111792793538778",
    )),
    (1.000000000001, (
        "14.16203969992903756665553006488733404574",
        "0.00000141427642318007092633958070705303",
        "1.57079491251847343916039535205904438907",
    )),
    (1.0001, (
        "4.95176877564313991960551952262753552300",
        "0.01414154640608337903549021978231684299",
        "1.55665478038881324019583147185743459911",
    )),
])
@pytest.mark.parametrize("name,reference_index", [("acoth", 0), ("asec", 1), ("acsc", 2)])
@pytest.mark.parametrize("sign", [-1, 1])
def test_values_near_real_branch_endpoints_preserve_binary64_accuracy(
    point, reference_values, name, reference_index, sign,
):
    with localcontext() as context:
        context.prec = 50
        reference = Decimal(reference_values[reference_index])
        if sign < 0:
            reference = (
                Decimal("3.14159265358979323846264338327950288419716939937510") - reference
                if name == "asec" else -reference
            )
        expected = float(reference)
    x = s.Symbol("reciprocal_near_endpoint")
    source = getattr(s, name)(x)
    for program in (s.lambdify(x, source, modules="math"), s.compile_numeric(x, source)):
        assert program(sign*point) == pytest.approx(expected, rel=2e-15, abs=0)


@pytest.mark.parametrize("name", CASES)
def test_values_jacobians_and_hessians_retain_both_chain_terms(name):
    anchor, expected, slope, curvature, _ = CASES[name]
    x, y = s.symbols("reciprocal_chain_x reciprocal_chain_y")
    source = getattr(s, name)(x*x + y)
    point = (0.5, anchor - 0.25)
    assert s.compile_numeric((x, y), source)(*point) == pytest.approx(expected)
    assert s.lambdify((x, y), source, modules="math")(*point) == pytest.approx(expected)
    assert s.lambdify((x, y), source, modules="frankensympy")(*point) == pytest.approx(expected)

    residual = s.compile_residual_system(source, (x, y))
    values, jacobian = residual.evaluate(*point)
    assert values == pytest.approx([expected])
    assert jacobian[0] == pytest.approx([slope, slope])
    assert residual.jacobian_vector_product([2, 3], *point) == pytest.approx([5*slope])
    assert residual.jacobian_transpose_vector_product([2], *point) == pytest.approx(
        [2*slope, 2*slope]
    )

    objective = s.compile_objective(source, (x, y))
    value, gradient, hessian = objective.evaluate(*point)
    assert value == pytest.approx(expected)
    assert gradient == pytest.approx([slope, slope])
    assert hessian[0] == pytest.approx([curvature + 2*slope, curvature])
    assert hessian[1] == pytest.approx([curvature, curvature])
    assert objective.hessian_vector_product([2, 3], *point) == pytest.approx(
        [5*curvature + 4*slope, 5*curvature]
    )


@pytest.mark.parametrize("name", [name for name in CASES if name != "asech"])
def test_negative_branches_keep_the_principal_value_and_derivative_signs(name):
    anchor, expected, slope, curvature, _ = CASES[name]
    x = s.Symbol("reciprocal_negative")
    source = getattr(s, name)(x)
    expected = math.pi - expected if name == "asec" else -expected
    assert s.lambdify(x, source, modules="math")(-anchor) == pytest.approx(expected)
    value, gradient, hessian = s.compile_objective(source, x).evaluate(-anchor)
    assert value == pytest.approx(expected)
    assert gradient == pytest.approx([slope])
    assert hessian[0] == pytest.approx([-curvature])


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("budgeted", [False, True])
def test_scalar_nsolve_uses_the_registered_real_functions_and_checked_jets(name, budgeted):
    anchor, expected, _, _, initial = CASES[name]
    x = s.Symbol("reciprocal_root")
    # A numeric target deliberately avoids an unrelated atan/log instruction
    # selecting the checked derivative lane on behalf of the new function.
    source = getattr(s, name)(x) - s.Float(expected)
    budget = s.EvaluationBudget() if budgeted else None
    before = budget.remaining_steps if budget is not None else None
    root = s.nsolve(source, x, initial, budget=budget)
    assert float(root) == pytest.approx(anchor, rel=2e-11, abs=2e-12)
    assert abs(s.compile_numeric(x, source)(root)) <= 2e-12
    if budget is not None:
        assert budget.remaining_steps < before


@pytest.mark.parametrize("budgeted", [False, True])
def test_scalar_nsolve_refuses_a_derivative_at_the_acot_branch_jump(budgeted):
    x = s.Symbol("reciprocal_discontinuous_root")
    source = s.acot(x) - s.Float(0.5)
    budget = s.EvaluationBudget() if budgeted else None
    with pytest.raises(ValueError):
        s.nsolve(source, x, 0, budget=budget)


@pytest.mark.parametrize("name,point,expected,decimal_value", [
    ("acot", -0.0, math.pi/2, "1.57079632679489661923132169163975144210"),
    ("acot", 0.0, math.pi/2, "1.57079632679489661923132169163975144210"),
    ("asec", -1, math.pi, "3.14159265358979323846264338327950288420"),
    ("asec", 1, 0, "0"),
    ("acsc", -1, -math.pi/2, "-1.57079632679489661923132169163975144210"),
    ("acsc", 1, math.pi/2, "1.57079632679489661923132169163975144210"),
    ("asech", 1, 0, "0"),
])
def test_closed_value_endpoints_do_not_claim_smooth_derivatives(
    name, point, expected, decimal_value,
):
    x = s.Symbol("reciprocal_endpoint")
    source = getattr(s, name)(x)
    assert s.compile_numeric(x, source)(point) == pytest.approx(expected)
    assert s.lambdify(x, source, modules="math")(point) == pytest.approx(expected)
    held = getattr(s, name)(s.Integer(int(point)), evaluate=False)
    with localcontext() as context:
        context.prec = 50
        assert abs(Decimal(str(s.N(held, 30))) - Decimal(decimal_value)) < Decimal("1e-29")
    with pytest.raises(ValueError):
        s.compile_residual_system(source, x).evaluate(point)
    with pytest.raises(ValueError):
        s.compile_objective(source, x).hessian_vector_product([0], point)


@pytest.mark.parametrize("name,points", [
    ("acoth", (-1, 0, 1)),
    ("asec", (-0.5, 0, 0.5)),
    ("acsc", (-0.5, 0, 0.5)),
    ("asech", (-0.5, 0, 2)),
    ("acsch", (0,)),
])
def test_held_zero_and_zero_directions_retain_original_value_domains(name, points):
    x = s.Symbol("reciprocal_original_domain")
    source = s.Mul(0, getattr(s, name)(x), evaluate=False)
    native = s.compile_numeric(x, source)
    ordinary = s.lambdify(x, source, modules="math")
    residual = s.compile_residual_system(source, x)
    objective = s.compile_objective(source, x)
    for point in points:
        with pytest.raises(ValueError):
            native(point)
        with pytest.raises(ValueError, match="math domain error"):
            ordinary(point)
        with pytest.raises(ValueError):
            residual.jacobian_vector_product([0], point)
        with pytest.raises(ValueError):
            objective.hessian_vector_product([0], point)


@pytest.mark.parametrize("name,sign", [
    ("acot", 1), ("acot", -1), ("asech", 1), ("acsch", 1), ("acsch", -1),
])
def test_finite_tiny_argument_values_do_not_overflow_an_intermediate_reciprocal(name, sign):
    tiny = math.ulp(0.0)
    expected = math.pi/2 if name == "acot" else math.log(2) - math.log(tiny)
    x = s.Symbol("reciprocal_tiny_value")
    source = getattr(s, name)(x)
    for program in (s.compile_numeric(x, source), s.lambdify(x, source, modules="math")):
        value = program(sign*tiny)
        assert math.isfinite(value)
        assert value == pytest.approx(sign*expected, rel=3e-15)


@pytest.mark.parametrize("name,point,slope,curvature", [
    ("acot", 1e100, -1e-200, 2e-300),
    ("acoth", 1e100, -1e-200, 2e-300),
    ("asec", 1e100, 1e-200, -2e-300),
    ("acsc", 1e100, -1e-200, 2e-300),
    ("asech", 1e-100, -1e100, 1e200),
    ("acsch", 1e100, -1e-200, 2e-300),
])
def test_checked_jets_keep_representable_extreme_derivatives(name, point, slope, curvature):
    x = s.Symbol("reciprocal_extreme_jet")
    value, gradient, hessian = s.compile_objective(getattr(s, name)(x), x).evaluate(point)
    assert math.isfinite(value)
    assert gradient[0] != 0 and hessian[0][0] != 0
    assert gradient == pytest.approx([slope], rel=5e-13, abs=0)
    assert hessian[0] == pytest.approx([curvature], rel=5e-13, abs=0)


@pytest.mark.parametrize("name", CASES)
def test_verified_artifacts_replay_new_jets_and_refuse_opcode_substitution(name):
    anchor, _, _, _, _ = CASES[name]
    x = s.Symbol("reciprocal_transport", real=True)
    source = getattr(s, name)(x)
    symbolic = s.diff(source, x)
    residual = s.compile_residual_system(source, x)
    for restored in (
        s.load_residual_system(residual.to_bytes(), residuals=source, args=x),
        s.load_portable_residual_system(residual.to_portable_bytes(), residuals=source, args=x),
    ):
        assert restored.evaluate(anchor) == residual.evaluate(anchor)
    first = ('"' + name.capitalize() + 'Derivative"').encode()
    second = ('"' + name.capitalize() + 'SecondDerivative"').encode()
    encoded = residual.to_bytes()
    assert first in encoded
    with pytest.raises(ValueError):
        s.load_residual_system(encoded.replace(first, second), residuals=source, args=x)
    with pytest.raises(ValueError):
        s.load_residual_system(encoded, residuals=getattr(s, name)(2*x), args=x)

    objective = s.compile_objective(source, x)
    for restored in (
        s.load_objective(objective.to_bytes(), objective=source, args=x),
        s.load_portable_objective(objective.to_portable_bytes(), objective=source, args=x),
    ):
        assert restored.evaluate(anchor) == objective.evaluate(anchor)
    encoded = objective.to_bytes()
    assert second in encoded
    with pytest.raises(ValueError):
        s.load_objective(encoded.replace(second, first), objective=source, args=x)
    assert s.diff(source, x) == symbolic
    assert "_fsym_" not in str(symbolic)


@pytest.mark.parametrize("name", CASES)
def test_new_operations_keep_cumulative_budgets_and_terminal_cancellation(name):
    anchor, _, _, _, _ = CASES[name]
    x = s.Symbol("reciprocal_budget")
    source = getattr(s, name)(x)
    programs = (
        s.compile_numeric(x, source),
        s.compile_residual_system(source, x).evaluate,
        s.compile_objective(source, x).evaluate,
    )
    shared = s.EvaluationBudget()
    for evaluate in programs:
        before = shared.remaining_steps
        evaluate(anchor, budget=shared)
        assert shared.remaining_steps < before
        with pytest.raises(s.EvaluationBudgetExceeded):
            evaluate(anchor, budget=s.EvaluationBudget(max_steps=1))
    shared.cancel()
    for evaluate in programs:
        with pytest.raises(s.EvaluationCancelled):
            evaluate(anchor, budget=shared)


@pytest.mark.parametrize("name", CASES)
def test_hybrid_programs_admit_registered_intrinsics_in_native_storage(name):
    anchor, expected, _, _, _ = CASES[name]
    x = s.Symbol("reciprocal_native_storage")
    source = _native_expr(getattr(s, name)(x))
    program = s.compile_numeric_hybrid(x, source, callbacks={})
    assert program(anchor) == pytest.approx(expected)


def test_new_intrinsics_compose_with_one_supervised_callback_invocation():
    x = s.Symbol("reciprocal_callback")
    calls = []

    class Law(s.Function):
        nargs = 1

    source = s.Add(s.asech(x), Law(x, evaluate=False), evaluate=False)
    program = s.compile_numeric_hybrid(
        x, source, callbacks={Law: lambda value: calls.append(value) or 2*value},
    )
    assert program(0.8) == pytest.approx(math.log(2) + 1.6)
    assert calls == [0.8]
    assert program.evidence == "approximate_with_user_callbacks"


@pytest.mark.parametrize("name", CASES)
def test_same_name_opaque_functions_cannot_acquire_intrinsic_authority(name):
    x = s.Symbol("reciprocal_opaque")
    for spelling in (name, "_fsym_" + name + "_d1", "_fsym_" + name + "_d2"):
        source = s.Function(spelling)(x)
        for compile_program in (
            lambda: s.compile_numeric(x, source),
            lambda: s.lambdify(x, source, modules="math"),
            lambda: s.compile_residual_system(source, x),
            lambda: s.compile_objective(source, x),
        ):
            with pytest.raises((NotImplementedError, ValueError, TypeError)):
                compile_program()
