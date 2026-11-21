"""The analytic sinc extension composes through numerical and certified APIs."""

import math

import pytest
import sympy as s

from sympy.core import _native, _native_expr


@pytest.mark.parametrize("backend", ("math", "frankensympy"))
@pytest.mark.parametrize("raw", (False, True))
def test_registered_sinc_reaches_scalar_callables_without_a_zero_pole(backend, raw):
    x = s.Symbol("sinc_scalar")
    source = s.sinc(x)
    if raw:
        source = _native_expr(source)
    function = s.lambdify(x, source, modules=backend)
    for point in (-2.0, -1.0, -1e-12, -0.0, 0.0, 1e-12, 0.5, 1.0, 2.0, 1e200):
        expected = 1.0 if point == 0 else math.sin(point) / point
        assert function(point) == pytest.approx(expected, rel=3e-14, abs=0)
        if backend == "math" and point == 0:
            assert type(function(point)) is int


@pytest.mark.parametrize("point", (-0.0, 0.0, -1e-12, 1e-12, -1e-100, 1e-100))
def test_verified_derivatives_and_directional_products_preserve_the_origin_jet(point):
    x = s.Symbol("sinc_origin")
    residual = s.compile_residual_system(s.sinc(x), x)
    objective = s.compile_objective(s.sinc(x), x)
    value, gradient, hessian = objective.evaluate(point)
    assert value == 1.0
    assert gradient == pytest.approx([-point/3], rel=3e-14, abs=0)
    assert hessian[0] == pytest.approx([-1/3], rel=3e-14)
    values, jacobian = residual.evaluate(point)
    assert values == [value]
    assert jacobian[0] == gradient
    assert residual.jacobian_vector_product([6], point) == pytest.approx([-2*point], rel=3e-14, abs=0)
    assert objective.hessian_vector_product([6], point) == pytest.approx([-2.0])


def test_nonlinear_argument_and_nested_sinc_preserve_second_order_chain_terms():
    x, y = s.symbols("sinc_chain_x sinc_chain_y")
    source = s.sinc(2*x + y + 3*x*x)
    objective = s.compile_objective(source, (x, y))
    value, gradient, hessian = objective.evaluate(0, 0)
    assert value == 1
    assert gradient == [0, 0]
    assert hessian[0] == pytest.approx([-4/3, -2/3])
    assert hessian[1] == pytest.approx([-2/3, -1/3])
    assert objective.hessian_vector_product([2, 3], 0, 0) == pytest.approx([-14/3, -7/3])
    nested = s.compile_objective(s.sinc(s.sinc(x)), x)
    value, gradient, hessian = nested.evaluate(0)
    assert value == pytest.approx(math.sin(1))
    assert gradient == [0]
    assert hessian[0] == pytest.approx([(math.sin(1)-math.cos(1))/3])


def test_residual_and_objective_artifacts_replay_origin_derivatives_and_reject_forgery():
    x = s.Symbol("sinc_transport", real=True)
    source = s.sinc(x)
    residual = s.compile_residual_system(source, x)
    for restored in (
        s.load_residual_system(residual.to_bytes(), residuals=source, args=x),
        s.load_portable_residual_system(residual.to_portable_bytes(), residuals=source, args=x),
    ):
        for point in (0, 1e-12, 0.5):
            assert restored.evaluate(point) == residual.evaluate(point)
    encoded = residual.to_bytes()
    assert b'"Sinc"' in encoded and b'"SincDerivative"' in encoded
    for original, replacement in ((b'"Sinc"', b'"Sin"'),
                                  (b'"SincDerivative"', b'"SincSecondDerivative"')):
        with pytest.raises(ValueError):
            s.load_residual_system(encoded.replace(original, replacement), residuals=source, args=x)
    objective = s.compile_objective(source, x)
    restored = s.load_objective(objective.to_bytes(), objective=source, args=x)
    assert restored.evaluate(0) == objective.evaluate(0)


def test_direct_source_enclosures_do_not_depend_on_numeric_derivative_generation(monkeypatch):
    x = s.Symbol("sinc_certificate")
    source = s.sinc(2*x)

    def forbidden(*args, **kwargs):
        raise AssertionError("source certificates must not generate derivative programs")

    monkeypatch.setattr(_native, "VerifiedResidualSystem", forbidden)
    monkeypatch.setattr(_native, "VerifiedObjective", forbidden)
    residual = s.enclose_residual_system(source, x, [(0, 0)])
    objective = s.enclose_objective(source, x, [(0, 0)])
    assert residual.residual_values == ((1, 1),)
    assert residual.jacobian == (((0, 0),),)
    assert objective.value == (1, 1)
    assert objective.gradient == ((0, 0),)
    assert objective.hessian == (((s.Rational(-4, 3), s.Rational(-4, 3)),),)


def test_compiled_enclosures_and_stationary_certification_cross_the_removable_singularity():
    x = s.Symbol("sinc_stationary")
    box = [(s.Rational(-1, 4), s.Rational(1, 4))]
    program = s.compile_objective(s.sinc(x), x)
    direct = s.enclose_objective(s.sinc(x), x, box)
    compiled = program.enclose(box)
    assert (direct.value, direct.gradient, direct.hessian) == (
        compiled.value, compiled.gradient, compiled.hessian,
    )
    assert compiled.value[0] <= 1 <= compiled.value[1]
    assert compiled.hessian[0][0][1] < 0
    receipt = program.certify_stationary_point(box)
    assert receipt.kind == "strict_maximum"
    assert receipt.stationary_enclosure[0][0] <= 0 <= receipt.stationary_enclosure[0][1]
    assert receipt.contraction_bound < 1
    minimum = s.compile_objective(-s.sinc(x), x).certify_stationary_point(box)
    assert minimum.kind == "strict_minimum"


@pytest.mark.parametrize("held", ("reciprocal", "zero_product", "log"))
def test_sinc_does_not_erase_original_argument_poles(held):
    x = s.Symbol("sinc_original_domain")
    argument = {"reciprocal": 1/x,
                "zero_product": s.Mul(0, 1/x, evaluate=False),
                "log": s.log(x)}[held]
    source = s.sinc(argument, evaluate=False)
    for construct in (s.compile_residual_system, s.compile_objective):
        program = construct(source, x)
        with pytest.raises(ValueError):
            program.evaluate(0)
    for enclose in (s.enclose_residual_system, s.enclose_objective):
        with pytest.raises(ValueError):
            enclose(source, x, [(0, 0)])


@pytest.mark.parametrize("name", ("sinc", "_fsym_sinc_d1", "_fsym_sinc_d2"))
def test_opaque_function_names_do_not_acquire_builtin_numerical_semantics(name):
    x = s.Symbol("sinc_opaque")
    opaque = s.Function(name)(x)
    for operation in (lambda: s.compile_numeric(x, opaque),
                      lambda: s.compile_residual_system(opaque, x),
                      lambda: s.compile_objective(opaque, x)):
        with pytest.raises((ValueError, TypeError, NotImplementedError)):
            operation()


def test_sinc_numerical_calls_keep_cumulative_budget_and_cancellation():
    x = s.Symbol("sinc_budget")
    program = s.compile_objective(s.sinc(x), x)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.evaluate(0.5, budget=s.EvaluationBudget(max_steps=1))
    cancelled = s.EvaluationBudget()
    cancelled.cancel()
    with pytest.raises(s.EvaluationCancelled):
        program.evaluate(0.5, budget=cancelled)


def test_public_symbolic_derivative_retains_its_existing_form():
    x = s.Symbol("sinc_symbolic")
    original = s.diff(s.sinc(x), x)
    s.compile_residual_system(s.sinc(x), x)
    s.compile_objective(s.sinc(x), x)
    assert s.diff(s.sinc(x), x) == original
    assert "_fsym_sinc_" not in str(original)
