"""Checked inverse-function programs retain representable extreme derivatives."""

import math
import sys

import pytest
import sympy as s


@pytest.mark.parametrize("function, point, slope, curvature", [
    (s.atan, 1e100, 1e-200, -2e-300),
    (s.atan, -1e100, 1e-200, 2e-300),
    (s.asinh, 1e120, 1e-120, -1e-240),
    (s.asinh, -1e120, 1e-120, 1e-240),
])
def test_extreme_inverse_curvature_is_not_a_false_zero(function, point, slope, curvature):
    x = s.Symbol("inverse_tail")
    source = function(x)
    objective = s.compile_objective(source, x)
    residual = s.compile_residual_system(source, x)
    _, gradient, hessian = objective.evaluate(point)
    assert gradient[0] != 0 and hessian[0][0] != 0
    assert gradient == pytest.approx([slope], rel=4e-13, abs=0)
    assert hessian[0] == pytest.approx([curvature], rel=4e-13, abs=0)
    assert objective.hessian_vector_product([3], point) == pytest.approx(
        [3*curvature], rel=4e-13, abs=0,
    )
    assert residual.evaluate(point)[1][0] == pytest.approx([slope], rel=4e-13, abs=0)
    assert residual.jacobian_vector_product([3], point) == pytest.approx(
        [3*slope], rel=4e-13, abs=0,
    )
    assert residual.jacobian_transpose_vector_product([2], point) == pytest.approx(
        [2*slope], rel=4e-13, abs=0,
    )


@pytest.mark.parametrize("function, point, slope", [
    (s.atan, 1e155, 1e-310),
    (s.atan, -1e155, 1e-310),
    (s.asinh, sys.float_info.max, 1/sys.float_info.max),
    (s.asinh, -sys.float_info.max, 1/sys.float_info.max),
])
def test_overflowing_argument_square_cannot_erase_a_representable_slope(function, point, slope):
    x = s.Symbol("inverse_largest")
    program = s.compile_objective(function(x), x)
    value, gradient, _ = program.evaluate(point)
    assert math.isfinite(value)
    assert gradient[0] > 0
    assert gradient == pytest.approx([slope], rel=4e-13, abs=2*math.ulp(0.0))


@pytest.mark.parametrize("function, factor", [(s.atan, 2), (s.asinh, 1)])
def test_zero_and_subnormal_inverse_curvatures_keep_their_sign(function, factor):
    x = s.Symbol("inverse_smallest")
    program = s.compile_objective(function(x), x)
    for point in (-math.ulp(0.0), -0.0, 0.0, math.ulp(0.0)):
        _, gradient, hessian = program.evaluate(point)
        assert gradient == [1.0]
        assert hessian[0] == [-factor*point]


@pytest.mark.parametrize("function, magnitude, slope, curvature", [
    (s.atan, 1e100, 1e-200, -2e-300),
    (s.asinh, 1e120, 1e-120, -1e-240),
])
def test_nonlinear_inverse_arguments_keep_both_second_order_chain_terms(
    function, magnitude, slope, curvature,
):
    x, y = s.symbols("inverse_chain_x inverse_chain_y")
    program = s.compile_objective(function(x*x + y), (x, y))
    _, gradient, hessian = program.evaluate(3, magnitude)
    assert gradient == pytest.approx([6*slope, slope], rel=4e-13, abs=0)
    assert hessian[0] == pytest.approx(
        [36*curvature + 2*slope, 6*curvature], rel=4e-13, abs=0,
    )
    assert hessian[1] == pytest.approx([6*curvature, curvature], rel=4e-13, abs=0)
    assert program.hessian_vector_product([2, 3], 3, magnitude) == pytest.approx(
        [90*curvature + 4*slope, 15*curvature], rel=4e-13, abs=0,
    )


@pytest.mark.parametrize("function, prefix", [(s.atan, b"Atan"), (s.asinh, b"Asinh")])
def test_inverse_jet_artifacts_replay_and_refuse_derivative_substitution(function, prefix):
    x = s.Symbol("inverse_replay", real=True)
    source = function(x)
    residual = s.compile_residual_system(source, x)
    for restored in (
        s.load_residual_system(residual.to_bytes(), residuals=source, args=x),
        s.load_portable_residual_system(residual.to_portable_bytes(), residuals=source, args=x),
    ):
        assert restored.evaluate(1e120) == residual.evaluate(1e120)
    encoded = residual.to_bytes()
    first = b'"' + prefix + b'Derivative"'
    second = b'"' + prefix + b'SecondDerivative"'
    assert first in encoded
    with pytest.raises(ValueError):
        s.load_residual_system(encoded.replace(first, second), residuals=source, args=x)
    objective = s.compile_objective(source, x)
    encoded = objective.to_bytes()
    restored = s.load_objective(encoded, objective=source, args=x)
    assert restored.evaluate(1e120) == objective.evaluate(1e120)
    assert second in encoded
    with pytest.raises(ValueError):
        s.load_objective(encoded.replace(second, first), objective=source, args=x)


@pytest.mark.parametrize("function", [s.atan, s.asinh])
def test_stable_numeric_jets_preserve_public_symbolic_forms_and_original_domains(function):
    x = s.Symbol("inverse_contract")
    source = function(x)
    before = s.diff(source, x)
    program = s.compile_objective(source, x)
    s.compile_residual_system(source, x)
    assert s.diff(source, x) == before
    assert "_fsym_" not in str(before)
    cancelled = s.EvaluationBudget()
    cancelled.cancel()
    with pytest.raises(s.EvaluationCancelled):
        program.evaluate(1e100, budget=cancelled)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.evaluate(1e100, budget=s.EvaluationBudget(max_steps=1))
    held = s.Mul(0, function(s.log(x)), evaluate=False)
    guarded = s.compile_objective(held, x)
    for point in (0, -1):
        with pytest.raises(ValueError):
            guarded.hessian_vector_product([0], point)


@pytest.mark.parametrize("name", [
    "_fsym_atan_d1", "_fsym_atan_d2", "_fsym_asinh_d1", "_fsym_asinh_d2",
])
def test_opaque_python_functions_do_not_acquire_private_jet_semantics(name):
    x = s.Symbol("inverse_opaque")
    source = s.Function(name)(x)
    for operation in (
        lambda: s.compile_numeric(x, source),
        lambda: s.compile_residual_system(source, x),
        lambda: s.compile_objective(source, x),
    ):
        with pytest.raises((ValueError, TypeError, NotImplementedError)):
            operation()
