"""Reciprocal elementary functions through original-source numerical APIs."""

import math

import pytest
import sympy as s

from sympy.core import _native, _native_expr


FUNCTIONS = (s.cot, s.sec, s.csc, s.coth, s.sech, s.csch)
SINGULAR_AT_ZERO = (s.cot, s.csc, s.coth, s.csch)


def reference(function, x):
    """Independent scalar value and first two derivatives at ordinary points."""
    if function is s.cot:
        a, b = math.sin(x), math.cos(x)
        return b/a, -1/(a*a), 2*b/(a*a*a)
    if function is s.sec:
        a, b = math.sin(x), math.cos(x)
        return 1/b, a/(b*b), (1+a*a)/(b*b*b)
    if function is s.csc:
        a, b = math.sin(x), math.cos(x)
        return 1/a, -b/(a*a), (1+b*b)/(a*a*a)
    if function is s.coth:
        a, b = math.sinh(x), math.cosh(x)
        return b/a, -1/(a*a), 2*b/(a*a*a)
    if function is s.sech:
        a, b = math.sinh(x), math.cosh(x)
        return 1/b, -a/(b*b), (a*a-1)/(b*b*b)
    a, b = math.sinh(x), math.cosh(x)
    return 1/a, -b/(a*a), (1+b*b)/(a*a*a)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("backend", ("math", "frankensympy"))
@pytest.mark.parametrize("raw", (False, True))
def test_exact_registered_functions_and_native_storage_reach_scalar_callables(function, backend, raw):
    x = s.Symbol("reciprocal_scalar")
    source = function(x)
    if raw:
        source = _native_expr(source)
    program = s.lambdify(x, source, modules=backend)
    for point in (-2.5, -0.75, 0.25, 0.75, 2.5):
        assert program(point) == pytest.approx(reference(function, point)[0], rel=3e-14)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_verified_jacobian_hessian_and_products_include_both_chain_terms(function):
    x, y = s.symbols("reciprocal_chain_x reciprocal_chain_y")
    anchor = s.Rational(3, 4)
    source = function(anchor + 2*x + y + 3*x*x)
    value, slope, curvature = reference(function, 0.75)
    residual = s.compile_residual_system(source, (x, y))
    values, jacobian = residual.evaluate(0, 0)
    assert values == pytest.approx([value])
    assert jacobian[0] == pytest.approx([2*slope, slope])
    assert residual.jacobian_vector_product([2, 3], 0, 0) == pytest.approx([7*slope])
    assert residual.jacobian_transpose_vector_product([5], 0, 0) == pytest.approx([10*slope, 5*slope])
    objective = s.compile_objective(source, (x, y))
    actual, gradient, hessian = objective.evaluate(0, 0)
    assert actual == pytest.approx(value)
    assert gradient == pytest.approx([2*slope, slope])
    assert hessian[0] == pytest.approx([4*curvature+6*slope, 2*curvature])
    assert hessian[1] == pytest.approx([2*curvature, curvature])
    assert objective.hessian_vector_product([2, 3], 0, 0) == pytest.approx(
        [14*curvature+12*slope, 7*curvature]
    )


@pytest.mark.parametrize("function", FUNCTIONS)
def test_ordinary_and_portable_residual_artifacts_keep_reciprocal_sources(function):
    x = s.Symbol("reciprocal_transport", positive=True)
    source = x*function(x)
    program = s.compile_residual_system(source, x)
    legacy = s.load_residual_system(program.to_bytes(), residuals=source, args=x)
    portable = s.load_portable_residual_system(program.to_portable_bytes(), residuals=source, args=x)
    assert legacy.evaluate(0.75) == program.evaluate(0.75)
    assert portable.evaluate(0.75) == program.evaluate(0.75)
    with pytest.raises(ValueError):
        s.load_residual_system(program.to_bytes(), residuals=source + 1, args=x)


@pytest.mark.parametrize("function,opcode,other", (
    (s.cot, b'"Cot"', b'"Coth"'),
    (s.sec, b'"Sec"', b'"Sech"'),
    (s.csc, b'"Csc"', b'"Csch"'),
    (s.coth, b'"Coth"', b'"Cot"'),
    (s.sech, b'"Sech"', b'"Sec"'),
    (s.csch, b'"Csch"', b'"Csc"'),
))
def test_reciprocal_opcode_substitution_cannot_gain_independent_admission(function, opcode, other):
    x = s.Symbol("reciprocal_forgery")
    source = x*function(x)
    encoded = s.compile_residual_system(source, x).to_bytes()
    assert opcode in encoded
    with pytest.raises(ValueError):
        s.load_residual_system(encoded.replace(opcode, other), residuals=source, args=x)


EXACT_ANCHORS = (
    (s.cot, s.atan(s.Rational(3, 4)), s.Rational(4, 3), s.Rational(-25, 9), s.Rational(200, 27)),
    (s.sec, s.atan(s.Rational(3, 4)), s.Rational(5, 4), s.Rational(15, 16), s.Rational(85, 32)),
    (s.csc, s.atan(s.Rational(3, 4)), s.Rational(5, 3), s.Rational(-20, 9), s.Rational(205, 27)),
    (s.coth, s.log(2), s.Rational(5, 3), s.Rational(-16, 9), s.Rational(160, 27)),
    (s.sech, s.log(2), s.Rational(4, 5), s.Rational(-12, 25), s.Rational(-28, 125)),
    (s.csch, s.log(2), s.Rational(4, 3), s.Rational(-20, 9), s.Rational(164, 27)),
)


def tight(interval, expected):
    assert interval[0] <= expected <= interval[1]
    assert interval[1] - interval[0] <= s.Rational(1, 10**22)


@pytest.mark.parametrize("function,anchor,value,slope,curvature", EXACT_ANCHORS)
def test_independent_original_source_enclosures_have_exact_rational_anchors(
    function, anchor, value, slope, curvature, monkeypatch,
):
    x = s.Symbol("reciprocal_interval")
    source = function(anchor + 2*x + 3*x*x)

    def forbidden(*args, **kwargs):
        raise AssertionError("direct source certificates cannot use derivative program generation")

    monkeypatch.setattr(_native, "VerifiedResidualSystem", forbidden)
    monkeypatch.setattr(_native, "VerifiedObjective", forbidden)
    residual = s.enclose_residual_system(source, x, [(0, 0)])
    objective = s.enclose_objective(source, x, [(0, 0)])
    assert residual.residuals == (source,) and objective.objective == source
    tight(residual.residual_values[0], value)
    tight(residual.jacobian[0][0], 2*slope)
    tight(objective.value, value)
    tight(objective.gradient[0], 2*slope)
    tight(objective.hessian[0][0], 4*curvature + 6*slope)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_compiled_and_direct_enclosures_bind_the_same_nonzero_box(function):
    x = s.Symbol("reciprocal_box")
    source = function(x)
    box = [(s.Rational(1, 2), s.Rational(3, 4))]
    direct = s.enclose_objective(source, x, box)
    compiled = s.compile_objective(source, x).enclose(box)
    assert (direct.value, direct.gradient, direct.hessian) == (
        compiled.value, compiled.gradient, compiled.hessian,
    )
    assert direct.objective == source and direct.domain == tuple(box)
    for point in (0.5, 0.625, 0.75):
        value, slope, curvature = reference(function, point)
        for interval, expected in ((direct.value, value), (direct.gradient[0], slope),
                                   (direct.hessian[0][0], curvature)):
            assert float(interval[0]) <= expected + 1e-13
            assert float(interval[1]) >= expected - 1e-13


@pytest.mark.parametrize("function", SINGULAR_AT_ZERO)
@pytest.mark.parametrize("held", ("zero_product", "zero_power"))
def test_original_poles_survive_zero_arithmetic_and_zero_directions(function, held):
    x = s.Symbol("reciprocal_pole")
    source = (s.Mul(0, function(x), evaluate=False) if held == "zero_product"
              else s.Pow(function(x), 0, evaluate=False))
    scalar = s.compile_numeric(x, source)
    residual = s.compile_residual_system(source, x)
    objective = s.compile_objective(source, x)
    for point in (0.0, -0.0):
        with pytest.raises(ValueError):
            scalar(point)
        with pytest.raises(ValueError):
            residual.jacobian_vector_product([0], point)
        with pytest.raises(ValueError):
            objective.hessian_vector_product([0], point)
        for enclose in (s.enclose_residual_system, s.enclose_objective):
            with pytest.raises(ValueError):
                enclose(source, x, [(0, 0)])


@pytest.mark.parametrize("function,box", (
    (s.cot, [(-1, 1)]), (s.csc, [(-1, 1)]), (s.sec, [(1, 2)]),
    (s.coth, [(-1, 1)]), (s.csch, [(-1, 1)]),
))
def test_certificates_refuse_boxes_crossing_original_reciprocal_poles(function, box):
    x = s.Symbol("reciprocal_pole_box")
    for operation in (s.enclose_residual_system, s.enclose_objective):
        with pytest.raises(ValueError):
            operation(function(x), x, box)


@pytest.mark.parametrize("function", (s.coth, s.sech, s.csch))
def test_native_hyperbolic_reciprocals_do_not_overflow_a_finite_final_value(function):
    x = s.Symbol("reciprocal_extreme")
    native = s.compile_numeric(x, function(x))
    for magnitude in (500.0, 710.0, 1000.0, 1e308):
        for sign in (-1.0, 1.0):
            value = native(sign*magnitude)
            assert math.isfinite(value)
            if function is s.coth:
                assert value == sign
            else:
                expected = 2*math.exp(-magnitude)
                if function is s.csch:
                    expected = math.copysign(expected, sign)
                assert value == pytest.approx(expected, rel=2e-14, abs=1e-323)
                if value == 0 and function is s.csch:
                    assert math.copysign(1, value) == sign
    # The ordinary math backend keeps the pinned printer's exp arithmetic.
    math_program = s.lambdify(x, function(x), modules="math")
    for point in (-1000.0, 1000.0):
        with pytest.raises(OverflowError):
            math_program(point)


@pytest.mark.parametrize("function", (s.coth, s.csch))
def test_native_small_reciprocals_avoid_exponential_subtraction_cancellation(function):
    x = s.Symbol("reciprocal_tiny")
    native = s.compile_numeric(x, function(x))
    for point in (-1e-300, -1e-12, 1e-12, 1e-300):
        assert native(point) == pytest.approx(1/point, rel=3e-14)


def test_verified_coth_derivatives_do_not_round_to_a_false_stationary_point():
    x = s.Symbol("reciprocal_coth_derivative_tail")
    source = s.coth(x)
    residual = s.compile_residual_system(source, x)
    objective = s.compile_objective(source, x)
    restored = s.load_residual_system(residual.to_bytes(), residuals=source, args=x)
    for point in (-100.0, -20.0, 20.0, 100.0):
        denominator = math.sinh(point)
        slope = -1/(denominator*denominator)
        curvature = 2*math.cosh(point)/(denominator*denominator*denominator)
        _, gradient, hessian = objective.evaluate(point)
        assert gradient[0] != 0
        assert gradient[0] == pytest.approx(slope, rel=5e-14, abs=0)
        assert hessian[0][0] == pytest.approx(curvature, rel=5e-14, abs=0)
        assert residual.jacobian_vector_product([1], point) == pytest.approx([slope], rel=5e-14, abs=0)
        assert restored.evaluate(point) == residual.evaluate(point)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_native_nonfinite_inputs_and_batch_failures_are_atomic(function):
    x = s.Symbol("reciprocal_batch")
    program = s.compile_numeric(x, function(x))
    for value in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            program(value)
        marker = object()
        output = marker
        with pytest.raises(ValueError):
            output = program.evaluate_batch([(0.75,), (value,), (1.0,)])
        assert output is marker
    budget = s.EvaluationBudget(max_steps=10000, max_memory_bytes=1000000, max_allocations=10000)
    before = budget.remaining_steps
    program(0.75, budget=budget)
    assert budget.remaining_steps < before
    budget.cancel()
    with pytest.raises(s.EvaluationCancelled):
        program(0.75, budget=budget)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_same_spelling_opaque_functions_do_not_gain_registered_authority(function):
    x = s.Symbol("reciprocal_opaque")
    source = s.Function(function.__name__)(x)
    for operation in (s.compile_numeric, lambda a, b: s.lambdify(a, b, modules="math")):
        with pytest.raises(NotImplementedError):
            operation(x, source)
    with pytest.raises(NotImplementedError):
        s.enclose_objective(source, x, [(1, 1)])


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("backend", ("math", "frankensympy"))
@pytest.mark.parametrize("arity", (0, 2))
def test_malformed_native_unary_arity_is_rejected_before_a_callable_is_published(
    function, backend, arity,
):
    x = s.Symbol("reciprocal_native_arity")
    source = _native.py_function(function.__name__, *([_native_expr(x)] * arity))
    with pytest.raises(ValueError):
        s.lambdify(x, source, modules=backend)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("backend", ("math", "frankensympy"))
@pytest.mark.parametrize("arity", (0, 2))
def test_malformed_retained_unary_arguments_refuse_during_source_admission(function, backend, arity):
    x = s.Symbol("reciprocal_retained_arity")
    source = function(x, evaluate=False)
    old_args = source._args
    try:
        source._args = tuple([x] * arity)
        with pytest.raises(ValueError):
            s.lambdify(x, source, modules=backend)
    finally:
        source._args = old_args


@pytest.mark.parametrize("function", FUNCTIONS)
def test_registered_intrinsics_compose_with_supervised_callbacks(function):
    x = s.Symbol("reciprocal_hybrid")
    calls = []

    class Law(s.Function):
        nargs = 1

    source = s.Add(function(x), Law(x, evaluate=False), evaluate=False)
    program = s.compile_numeric_hybrid(
        x, source, callbacks={Law: lambda value: calls.append(value) or 2*value},
    )
    assert program(0.75) == pytest.approx(reference(function, 0.75)[0] + 1.5)
    assert calls == [0.75]


def test_same_named_typed_variables_remain_distinct_in_reciprocal_programs():
    x = s.Symbol("reciprocal_same_name", positive=True)
    y = s.Symbol("reciprocal_same_name", negative=True)
    source = s.sec(x) + s.csch(y)
    program = s.compile_residual_system(source, (x, y))
    values, jacobian = program.evaluate(0.75, -0.5)
    a, da, _ = reference(s.sec, 0.75)
    b, db, _ = reference(s.csch, -0.5)
    assert values == pytest.approx([a+b])
    assert jacobian[0] == pytest.approx([da, db])
