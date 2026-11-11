"""Gaussian error functions through native programs and source certificates."""

import math

import pytest
import sympy as s

from sympy.core import _native, _native_expr, _wrap
from sympy.solvers.numerical import nsolve_certified


CASES = ((s.erf, math.erf, 1), (s.erfc, math.erfc, -1))


def tight(interval, reference):
    lower, upper = interval
    assert lower <= upper
    assert upper - lower < s.Rational(1, 10**24)
    assert float(lower) == pytest.approx(reference, rel=3e-14, abs=1e-15)
    assert float(upper) == pytest.approx(reference, rel=3e-14, abs=1e-15)


@pytest.mark.parametrize("function,reference,sign", CASES)
def test_native_values_and_checked_nonlinear_derivatives(function, reference, sign):
    x, y = s.symbols("gaussian_x gaussian_y")
    expression = function(x*x + y)
    compiled = s.compile_numeric((x, y), expression)
    assert compiled(0.5, 0.25) == pytest.approx(reference(0.5))
    scalar = s.lambdify(x, function(x), modules="frankensympy")
    for point in (-26.0, -4.0, -1.18, -0.5, 0.0, 0.5, 1.18, 4.0, 20.0, 26.0):
        assert scalar(point) == pytest.approx(reference(point), rel=4e-14, abs=1e-320)
    slope = sign*2*math.exp(-0.25)/math.sqrt(math.pi)
    residual = s.compile_residual_system([expression], (x, y))
    values, jacobian = residual.evaluate(0.5, 0.25)
    assert values == pytest.approx([reference(0.5)])
    assert jacobian[0] == pytest.approx([slope, slope])
    assert residual.jacobian_vector_product([2, 3], 0.5, 0.25) == pytest.approx([5*slope])
    objective = s.compile_objective(expression, (x, y))
    value, gradient, hessian = objective.evaluate(0.5, 0.25)
    assert value == pytest.approx(reference(0.5))
    assert gradient == pytest.approx([slope, slope])
    assert hessian[0] == pytest.approx([slope, -slope])
    assert hessian[1] == pytest.approx([-slope, -slope])
    assert objective.hessian_vector_product([2, 3], 0.5, 0.25) == pytest.approx([-slope, -5*slope])
    restored = s.load_residual_system(residual.to_bytes(), residuals=[expression], args=(x, y))
    assert restored.evaluate(0.5, 0.25) == residual.evaluate(0.5, 0.25)


@pytest.mark.parametrize("function,reference,sign", CASES)
def test_original_source_certification_is_independent_of_symbolic_derivative_programs(
    function, reference, sign, monkeypatch,
):
    x = s.Symbol("gaussian_source")
    source = function(s.Rational(1, 2) + 2*x + 3*x*x)

    def forbidden(*args, **kwargs):
        raise AssertionError("source bounds must not compile symbolic derivative programs")

    monkeypatch.setattr(_native, "VerifiedResidualSystem", forbidden)
    monkeypatch.setattr(_native, "VerifiedObjective", forbidden)
    residual = s.enclose_residual_system(source, x, [(0, 0)])
    objective = s.enclose_objective(source, x, [(0, 0)])
    slope = sign*2*math.exp(-0.25)/math.sqrt(math.pi)
    tight(objective.value, reference(0.5))
    tight(residual.jacobian[0][0], 2*slope)
    tight(objective.gradient[0], 2*slope)
    # f''(1/2)=-f'(1/2): both chain terms give -4*slope+6*slope.
    tight(objective.hessian[0][0], 2*slope)
    assert residual.residual_values[0] == objective.value
    assert residual.residuals == (source,) and objective.objective == source


def test_certified_complementary_tail_survives_below_absolute_precision():
    x = s.Symbol("gaussian_tail")
    source = s.erfc(x)
    result = s.enclose_objective(source, x, [(20, 20)], precision=24)
    lower, upper = result.value
    assert 0 < lower < upper < s.Rational(1, 10**100)
    assert upper - lower <= lower*s.Rational(1, 10**26)
    assert float(lower) == pytest.approx(math.erfc(20), rel=4e-14, abs=0)
    assert result.gradient[0][1] < 0
    assert result.hessian[0][0][0] > 0
    direct = s.enclose_residual_system(source, x, [(20, 20)], precision=24)
    assert direct.residual_values[0] == result.value
    assert direct.jacobian[0][0] == result.gradient[0]


@pytest.mark.parametrize("function,reference,sign", CASES)
def test_gaussian_root_search_reaches_independent_source_certification(function, reference, sign):
    x = s.Symbol("gaussian_root")
    source = function(x) - (0 if sign == 1 else 1)
    receipt = nsolve_certified(source, x, 0.1)
    assert receipt.root_enclosure[0][0] <= 0 <= receipt.root_enclosure[0][1]
    assert receipt.contraction_bound < 1 and receipt.residuals == (source,)
    box = [(s.Rational(-1, 10), s.Rational(1, 10))]
    direct = s.enclose_objective(source, x, box)
    compiled = s.compile_objective(source, x).enclose(box)
    assert (direct.value, direct.gradient, direct.hessian) == (compiled.value, compiled.gradient, compiled.hessian)


def test_error_function_objective_has_a_certified_local_minimum():
    x = s.Symbol("gaussian_minimum")
    source = x*s.erf(x)
    result = s.compile_objective(source, x).certify_stationary_point(
        [(s.Rational(-1, 10), s.Rational(1, 10))],
    )
    assert result.kind == "strict_minimum"
    assert result.stationary_enclosure[0][0] <= 0 <= result.stationary_enclosure[0][1]
    assert result.hessian_enclosure[0][0][0] > 0
    assert result.objective == source and result.contraction_bound < 1


@pytest.mark.parametrize("function,reference,sign", CASES)
def test_nonfinite_inputs_and_original_reciprocals_do_not_disappear(function, reference, sign):
    x = s.Symbol("gaussian_domain")
    program = s.compile_numeric(x, function(x))
    for point in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            program(point)
    hidden = s.Mul(0, function(s.Pow(x, -1, evaluate=False)), evaluate=False)
    residual = s.compile_residual_system([hidden], x)
    with pytest.raises(ValueError):
        residual.jacobian_vector_product([0], 0)
    for operation in (s.enclose_residual_system, s.enclose_objective):
        with pytest.raises(ValueError):
            operation(hidden, x, [(0, 0)])


@pytest.mark.parametrize("name", ("erf", "erfc"))
def test_undefined_names_cannot_acquire_builtin_function_authority(name):
    x = s.Symbol("gaussian_opaque")
    source = s.Function(name)(x)
    with pytest.raises(NotImplementedError):
        s.compile_numeric(x, source)
    with pytest.raises(NotImplementedError):
        s.enclose_objective(source, x, [(0, 0)])


@pytest.mark.parametrize("function,reference,sign", CASES)
def test_undefined_names_cannot_capture_native_gaussian_results(function, reference, sign):
    x = s.Symbol("gaussian_native_identity")
    unknown = s.Function(function.__name__)
    opaque = unknown(x)
    raw = _native.py_function(function.__name__, _native_expr(x))
    recovered = _wrap(raw)
    assert type(recovered) is function and recovered.args[0] is x
    assert type(function(x)) is function
    assert type(opaque) is unknown
    assert s.compile_numeric(x, function(x) + x)(0.5) == pytest.approx(reference(0.5) + 0.5)
    assert s.compile_numeric_hybrid(x, recovered, callbacks={})(0.5) == pytest.approx(reference(0.5))
    with pytest.raises(NotImplementedError):
        s.compile_numeric(x, opaque)


@pytest.mark.parametrize("function,reference,sign", CASES)
def test_gaussian_native_identity_preserves_changed_implementation_refusals(
    function, reference, sign, monkeypatch,
):
    x = s.Symbol("gaussian_changed_implementation")
    source = function(x, evaluate=False)
    raw = _native_expr(source)
    unknown = s.Function(function.__name__)
    assert unknown is not function
    calls = []

    def forbidden(*args):
        calls.append(args)
        raise AssertionError("closed Gaussian admission must not execute changed hooks")

    monkeypatch.setattr(function, "_eval_derivative", forbidden, raising=False)
    assert type(_wrap(raw)) is function
    for candidate in (source, raw):
        for operation in (
            lambda: _native_expr(candidate),
            lambda: s.compile_numeric(x, candidate),
            lambda: s.compile_numeric_hybrid(x, candidate, callbacks={}),
            lambda: s.enclose_objective(candidate, x, [(0, 0)]),
        ):
            with pytest.raises(NotImplementedError):
                operation()
    assert calls == []


@pytest.mark.parametrize("function,reference,sign", CASES)
def test_gaussian_subclasses_do_not_inherit_closed_authority(function, reference, sign):
    class CustomGaussian(function):
        pass

    x = s.Symbol("gaussian_subclass")
    source = CustomGaussian(x, evaluate=False)
    assert type(source) is CustomGaussian
    for operation in (
        lambda: s.compile_numeric(x, source),
        lambda: s.compile_numeric_hybrid(x, source, callbacks={}),
        lambda: s.enclose_objective(source, x, [(0, 0)]),
    ):
        with pytest.raises(NotImplementedError):
            operation()


@pytest.mark.parametrize("function,reference,sign", CASES)
def test_native_gaussian_operations_compose_with_supervised_callbacks(function, reference, sign):
    x = s.Symbol("gaussian_hybrid")
    calls = []

    class Law(s.Function):
        nargs = 1

    source = s.Add(function(x), Law(x, evaluate=False), evaluate=False)
    program = s.compile_numeric_hybrid(
        x, source, callbacks={Law: lambda value: calls.append(value) or value},
    )
    assert program(0.5) == pytest.approx(reference(0.5) + 0.5)
    assert calls == [0.5] and program.evidence == "approximate_with_user_callbacks"
    native = s.compile_numeric_hybrid(x, _native_expr(function(x)), callbacks={})
    assert native(0.5) == pytest.approx(reference(0.5))


def test_error_function_opcode_substitution_is_rejected():
    x = s.Symbol("gaussian_forgery")
    source = x*s.erf(x)
    encoded = s.compile_residual_system(source, x).to_bytes()
    assert b'"Erf"' in encoded
    forged = encoded.replace(b'"Erf"', b'"Erfc"')
    with pytest.raises(ValueError):
        s.load_residual_system(forged, residuals=source, args=x)
