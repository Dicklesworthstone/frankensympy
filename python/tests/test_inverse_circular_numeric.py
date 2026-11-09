"""Principal inverse circular functions through actual native programs."""

import math

import pytest

from sympy import (
    Function, Mul, Symbol, acos, asin, atan, compile_numeric, compile_objective,
    compile_residual_system, load_residual_system,
)


@pytest.mark.parametrize("function, reference", [
    (asin, math.asin), (acos, math.acos), (atan, math.atan),
])
@pytest.mark.parametrize("point", [-1.0, -0.6, 0.0, 0.6, 1.0])
def test_native_principal_values_include_closed_value_endpoints(function, reference, point):
    x = Symbol("inverse_value_x")
    program = compile_numeric(x, function(x))
    assert program(point) == pytest.approx(reference(point), rel=2e-14, abs=2e-14)


@pytest.mark.parametrize("function, reference, slope, curvature", [
    (asin, math.asin, 5 / 4, 75 / 64),
    (acos, math.acos, -5 / 4, -75 / 64),
    (atan, math.atan, 25 / 34, -375 / 578),
])
def test_checked_derivatives_and_products_preserve_composed_chain_factors(
    function, reference, slope, curvature,
):
    x, y = Symbol("inverse_x"), Symbol("inverse_y")
    expression = function(2*x + y)
    system = compile_residual_system([expression], (x, y))
    values, jacobian = system.evaluate(0.2, 0.2)
    assert values == pytest.approx([reference(0.6)])
    assert jacobian[0] == pytest.approx([2*slope, slope])
    assert system.jacobian_vector_product([2, 3], 0.2, 0.2) == pytest.approx([7*slope])
    assert system.jacobian_transpose_vector_product([5], 0.2, 0.2) == pytest.approx(
        [10*slope, 5*slope]
    )
    objective = compile_objective(expression, (x, y))
    value, gradient, hessian = objective.evaluate(0.2, 0.2)
    assert value == pytest.approx(reference(0.6))
    assert gradient == pytest.approx([2*slope, slope])
    assert hessian[0] == pytest.approx([4*curvature, 2*curvature])
    assert hessian[1] == pytest.approx([2*curvature, curvature])
    assert objective.hessian_vector_product([2, 3], 0.2, 0.2) == pytest.approx(
        [14*curvature, 7*curvature]
    )


@pytest.mark.parametrize("function", [asin, acos])
@pytest.mark.parametrize("point", [-2.0, 2.0])
def test_held_zero_cannot_erase_an_inverse_function_domain(function, point):
    x = Symbol("inverse_domain_x")
    expression = Mul(0, function(x), evaluate=False)
    with pytest.raises(ValueError):
        compile_numeric(x, expression)(point)
    system = compile_residual_system([expression], x)
    with pytest.raises(ValueError):
        system.evaluate_sparse(point)
    with pytest.raises(ValueError):
        system.jacobian_vector_product([0], point)


@pytest.mark.parametrize("function", [asin, acos])
@pytest.mark.parametrize("point", [-1.0, 1.0])
def test_finite_endpoint_value_does_not_admit_singular_derivatives(function, point):
    x = Symbol("inverse_endpoint_x")
    assert math.isfinite(compile_numeric(x, function(x))(point))
    system = compile_residual_system([function(x)], x)
    with pytest.raises(ValueError):
        system.evaluate(point)
    objective = compile_objective(function(x), x)
    with pytest.raises(ValueError):
        objective.hessian_vector_product([0], point)


@pytest.mark.parametrize("function", [asin, acos, atan])
def test_inverse_function_artifacts_replay_with_the_original_typed_source(function):
    x = Symbol("inverse_transport_x")
    residuals = [function(x)]
    program = compile_residual_system(residuals, x)
    data = program.to_bytes()
    restored = load_residual_system(data, residuals=residuals, args=x)
    assert restored.evaluate(0.25) == program.evaluate(0.25)
    with pytest.raises(ValueError):
        load_residual_system(data, residuals=[function(2*x)], args=x)


def test_serialized_inverse_opcode_substitution_is_rejected():
    x = Symbol("inverse_forgery_x")
    # The transport stores derivative bytecode. This derivative retains asin.
    residuals = [x*asin(x)]
    program = compile_residual_system(residuals, x)
    encoded = program.to_bytes()
    assert b'"Asin"' in encoded
    forged = encoded.replace(b'"Asin"', b'"Acos"')
    with pytest.raises(ValueError):
        load_residual_system(forged, residuals=residuals, args=x)


@pytest.mark.parametrize("name", ["asin", "acos", "atan"])
def test_new_native_opcodes_do_not_admit_undefined_functions_by_name(name):
    x = Symbol("inverse_opaque_x")
    with pytest.raises(NotImplementedError):
        compile_numeric(x, Function(name)(x))
