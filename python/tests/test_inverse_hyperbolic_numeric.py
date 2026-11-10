"""Inverse hyperbolic functions through the actual native and certified APIs."""

import math

import pytest
import sympy as s

from sympy.core import _native, _native_expr
from sympy.solvers.numerical import nsolve_certified


CASES = (
    (s.asinh, math.asinh, s.Rational(3, 4), s.Rational(4, 5), s.Rational(-48, 125)),
    (s.acosh, math.acosh, s.Rational(5, 4), s.Rational(4, 3), s.Rational(-80, 27)),
    (s.atanh, math.atanh, s.Rational(3, 5), s.Rational(25, 16), s.Rational(375, 128)),
)


def tight(interval, exact):
    assert interval[0] <= exact <= interval[1]
    assert interval[1] - interval[0] <= s.Rational(1, 10**24)


@pytest.mark.parametrize("function,reference,anchor,slope,curvature", CASES)
def test_native_values_and_verified_derivatives_share_typed_source(
    function, reference, anchor, slope, curvature,
):
    x, y = s.symbols("inverse_hyperbolic_x inverse_hyperbolic_y")
    expression = function(anchor + 2*x + y)
    assert s.compile_numeric((x, y), expression)(0, 0) == pytest.approx(math.log(2))
    assert s.lambdify(x, function(x), modules="frankensympy")(anchor) == pytest.approx(reference(float(anchor)))
    residual = s.compile_residual_system([expression], (x, y))
    values, jacobian = residual.evaluate(0, 0)
    assert values == pytest.approx([math.log(2)])
    assert jacobian[0] == pytest.approx([float(2*slope), float(slope)])
    assert residual.jacobian_vector_product([2, 3], 0, 0) == pytest.approx([float(7*slope)])
    objective = s.compile_objective(expression, (x, y))
    value, gradient, hessian = objective.evaluate(0, 0)
    assert value == pytest.approx(math.log(2))
    assert gradient == pytest.approx([float(2*slope), float(slope)])
    assert hessian[0] == pytest.approx([float(4*curvature), float(2*curvature)])
    assert hessian[1] == pytest.approx([float(2*curvature), float(curvature)])
    assert objective.hessian_vector_product([2, 3], 0, 0) == pytest.approx(
        [float(14*curvature), float(7*curvature)]
    )
    restored = s.load_residual_system(residual.to_bytes(), residuals=[expression], args=(x, y))
    assert restored.evaluate(0, 0) == residual.evaluate(0, 0)


@pytest.mark.parametrize("function,reference,anchor,slope,curvature", CASES)
def test_original_source_certification_includes_both_second_order_chain_terms(
    function, reference, anchor, slope, curvature, monkeypatch,
):
    x = s.Symbol("inverse_hyperbolic_source")
    source = function(anchor + 2*x + 3*x*x)

    def forbidden(*args, **kwargs):
        raise AssertionError("source checking must not depend on symbolic derivative compilation")

    monkeypatch.setattr(_native, "VerifiedResidualSystem", forbidden)
    monkeypatch.setattr(_native, "VerifiedObjective", forbidden)
    residual = s.enclose_residual_system(source, x, [(0, 0)])
    objective = s.enclose_objective(source, x, [(0, 0)])
    tight(residual.jacobian[0][0], 2*slope)
    tight(objective.gradient[0], 2*slope)
    tight(objective.hessian[0][0], 4*curvature + 6*slope)
    assert residual.residual_values[0] == objective.value
    assert residual.residuals == (source,) and objective.objective == source


@pytest.mark.parametrize("function,reference,anchor,slope,curvature", CASES)
def test_root_search_reaches_independent_certification(function, reference, anchor, slope, curvature):
    x = s.Symbol("inverse_hyperbolic_root")
    source = function(x) - s.log(2)
    receipt = nsolve_certified(source, x, float(anchor) + 0.01)
    assert receipt.root_enclosure[0][0] <= anchor <= receipt.root_enclosure[0][1]
    assert receipt.residuals == (source,) and receipt.contraction_bound < 1
    box = [(anchor - s.Rational(1, 100), anchor + s.Rational(1, 100))]
    direct = s.enclose_objective(source, x, box)
    compiled = s.compile_objective(source, x).enclose(box)
    assert (direct.value, direct.gradient, direct.hessian) == (compiled.value, compiled.gradient, compiled.hessian)


@pytest.mark.parametrize("function,points", [
    (s.acosh, (-2, 0, s.Rational(99, 100))),
    (s.atanh, (-2, -1, 1, 2)),
])
def test_invalid_original_values_survive_held_zero_arithmetic(function, points):
    x = s.Symbol("inverse_hyperbolic_domain")
    source = s.Mul(0, function(x), evaluate=False)
    program = s.compile_numeric(x, source)
    residual = s.compile_residual_system([source], x)
    for point in points:
        with pytest.raises(ValueError):
            program(point)
        with pytest.raises(ValueError):
            residual.jacobian_vector_product([0], point)
        for operation in (s.enclose_residual_system, s.enclose_objective):
            with pytest.raises(ValueError):
                operation(source, x, [(point, point)])


def test_acosh_closed_value_endpoint_does_not_claim_a_derivative():
    x = s.Symbol("inverse_hyperbolic_endpoint")
    source = s.acosh(x)
    assert s.compile_numeric(x, source)(1) == 0
    with pytest.raises(ValueError):
        s.compile_objective(source, x).evaluate(1)
    for operation in (s.enclose_residual_system, s.enclose_objective):
        with pytest.raises(ValueError):
            operation(source, x, [(1, 1)])
        with pytest.raises(ValueError):
            operation(source, x, [(1 - s.Rational(1, 10**80), 1)])


@pytest.mark.parametrize("name", ["asinh", "acosh", "atanh"])
def test_undefined_same_name_functions_cannot_acquire_native_authority(name):
    x = s.Symbol("inverse_hyperbolic_opaque")
    source = s.Function(name)(x)
    with pytest.raises(NotImplementedError):
        s.compile_numeric(x, source)
    with pytest.raises(NotImplementedError):
        s.enclose_objective(source, x, [(2, 2)])


def test_new_intrinsics_compose_with_supervised_hybrid_callbacks():
    x = s.Symbol("inverse_hyperbolic_hybrid")
    calls = []

    class Law(s.Function):
        nargs = 1

    source = s.Add(s.asinh(x), Law(x, evaluate=False), evaluate=False)
    program = s.compile_numeric_hybrid(x, source, callbacks={Law: lambda value: calls.append(value) or value})
    assert program(0.75) == pytest.approx(math.log(2) + 0.75)
    assert calls == [0.75]
    assert program.evidence == "approximate_with_user_callbacks"


@pytest.mark.parametrize("function,reference,anchor,slope,curvature", CASES)
def test_hybrid_native_storage_admits_the_registered_intrinsics(
    function, reference, anchor, slope, curvature,
):
    x = s.Symbol("inverse_hyperbolic_native_storage")
    source = _native_expr(function(x))
    program = s.compile_numeric_hybrid(x, source, callbacks={})
    assert program(float(anchor)) == pytest.approx(math.log(2))


def test_inverse_hyperbolic_opcode_forgery_is_rejected():
    x = s.Symbol("inverse_hyperbolic_forgery")
    source = x*s.asinh(x)
    encoded = s.compile_residual_system(source, x).to_bytes()
    assert b'"Asinh"' in encoded
    forged = encoded.replace(b'"Asinh"', b'"Atanh"')
    with pytest.raises(ValueError):
        s.load_residual_system(forged, residuals=source, args=x)
