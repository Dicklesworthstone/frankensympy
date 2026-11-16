"""Principal atan2 through typed numeric programs and exact source certificates."""

import math

import pytest
import sympy as s

from sympy.core import _native, _native_expr


ENCLOSERS = (s.enclose_residual_system, s.enclose_objective)


def tight_rational(interval, reference):
    lower, upper = interval
    assert lower <= reference <= upper
    assert upper - lower <= s.Rational(1, 10**24)


def tight_value(interval, reference):
    lower, upper = interval
    assert lower <= upper
    assert upper - lower <= s.Rational(1, 10**24)
    assert float(lower) == pytest.approx(reference, rel=3e-14, abs=1e-15)
    assert float(upper) == pytest.approx(reference, rel=3e-14, abs=1e-15)


def ample_budget(**limits):
    options = dict(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    options.update(limits)
    return s.EvaluationBudget(**options)


@pytest.mark.parametrize("backend", ("math", "frankensympy"))
@pytest.mark.parametrize("raw", (False, True))
def test_scalar_lowering_keeps_y_then_x_and_the_documented_backend_domains(backend, raw):
    y, x = s.symbols("angle_y angle_x")
    source = s.atan2(y, x, evaluate=False)
    if raw:
        source = _native_expr(source)
    program = s.lambdify((y, x), source, modules=backend)
    for point in ((4, 3), (1, 1), (1, -1), (-1, -1), (-1, 1), (1, 0), (-1, 0)):
        assert program(*point) == pytest.approx(math.atan2(*point))
    assert program(0, 1) == 0
    # The math backend preserves Python's math semantics. The finite-real
    # native backend refuses the origin and gives exact zero no branch side.
    if backend == "math":
        assert program(0.0, 0.0) == math.atan2(0.0, 0.0)
        assert program(-0.0, -1.0) == -math.pi
    else:
        for zero in (0.0, -0.0):
            with pytest.raises(ValueError):
                program(zero, 0.0)
            assert program(zero, -1.0) == math.pi
        for nonfinite in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError):
                program(nonfinite, 1)
            with pytest.raises(ValueError):
                program(1, nonfinite)


@pytest.mark.parametrize("backend", ("math", "frankensympy"))
@pytest.mark.parametrize("arity", (0, 1, 3))
def test_malformed_native_binary_arity_is_refused_before_execution(backend, arity):
    y, x = s.symbols("angle_arity_y angle_arity_x")
    source = _native.py_function("atan2", *([_native_expr(y)] * arity))
    with pytest.raises(ValueError):
        s.lambdify((y, x), source, modules=backend)


def test_native_batch_uses_the_same_domain_and_returns_no_partial_rows():
    y, x = s.symbols("angle_batch_y angle_batch_x")
    program = s.compile_numeric((y, x), s.atan2(y, x))
    points = [(4, 3), (-1, 1), (1, -1)]
    assert program.evaluate_batch(points) == pytest.approx([math.atan2(*p) for p in points])
    sentinel = object()
    result = sentinel
    with pytest.raises(ValueError):
        result = program.evaluate_batch([(4, 3), (0, 0), (1, 1)])
    assert result is sentinel
    assert program(4, 3) == pytest.approx(math.atan2(4, 3))
    budget = ample_budget()
    before = budget.remaining_steps
    program.evaluate_batch(points, budget=budget)
    assert budget.remaining_steps < before
    budget.cancel()
    with pytest.raises(s.EvaluationCancelled):
        program(4, 3, budget=budget)


def test_checked_numeric_jacobian_and_hessian_have_independent_rational_anchors():
    y, x = s.symbols("angle_derivative_y angle_derivative_x")
    source = s.atan2(y, x)
    residual = s.compile_residual_system(source, (y, x))
    values, jacobian = residual.evaluate(4, 3)
    assert values == pytest.approx([math.atan2(4, 3)])
    assert jacobian[0] == pytest.approx([3/25, -4/25])
    assert residual.jacobian_vector_product([2, 3], 4, 3) == pytest.approx([-6/25])
    objective = s.compile_objective(source, (y, x))
    value, gradient, hessian = objective.evaluate(4, 3)
    assert value == pytest.approx(math.atan2(4, 3))
    assert gradient == pytest.approx([3/25, -4/25])
    assert hessian[0] == pytest.approx([-24/625, 7/625])
    assert hessian[1] == pytest.approx([7/625, 24/625])
    assert objective.hessian_vector_product([2, 3], 4, 3) == pytest.approx([-27/625, 86/625])
    restored = s.load_residual_system(residual.to_bytes(), residuals=source, args=(y, x))
    assert restored.evaluate(4, 3) == residual.evaluate(4, 3)
    for program in (residual, restored, objective):
        with pytest.raises(ValueError):
            program.evaluate(0, -1)
        with pytest.raises(ValueError):
            program.evaluate(0, 0)
    with pytest.raises(ValueError):
        s.load_residual_system(residual.to_bytes(), residuals=s.atan2(x, y), args=(y, x))


def test_both_nonlinear_arguments_contribute_to_checked_and_certified_second_derivatives():
    u, v = s.symbols("angle_chain_u angle_chain_v")
    source = s.atan2(u*u + v, u - v*v)
    numeric = s.compile_objective(source, (u, v))
    value, gradient, hessian = numeric.evaluate(2, 1)
    assert value == pytest.approx(math.atan2(5, 1))
    assert gradient == pytest.approx([-1/26, 11/26])
    assert hessian[0] == pytest.approx([47/338, -57/169])
    assert hessian[1] == pytest.approx([-57/169, 97/338])
    assert numeric.hessian_vector_product([2, -1], 2, 1) == pytest.approx([104/169, -325/338])
    certified = s.enclose_objective(source, (u, v), [(2, 2), (1, 1)])
    tight_value(certified.value, math.atan2(5, 1))
    for actual, expected in zip(certified.gradient, (s.Rational(-1, 26), s.Rational(11, 26))):
        tight_rational(actual, expected)
    expected = ((s.Rational(47, 338), s.Rational(-57, 169)),
                (s.Rational(-57, 169), s.Rational(97, 338)))
    for row, references in zip(certified.hessian, expected):
        for actual, reference in zip(row, references):
            tight_rational(actual, reference)


def test_direct_angle_certification_does_not_call_symbolic_derivative_programs(monkeypatch):
    y, x = s.symbols("angle_source_y angle_source_x")
    source = s.atan2(y, x)

    def forbidden(*args, **kwargs):
        raise AssertionError("source certification must not compile symbolic derivatives")

    monkeypatch.setattr(_native, "VerifiedResidualSystem", forbidden)
    monkeypatch.setattr(_native, "VerifiedObjective", forbidden)
    residual = s.enclose_residual_system(source, (y, x), [(4, 4), (3, 3)])
    objective = s.enclose_objective(source, (y, x), [(4, 4), (3, 3)])
    assert residual.residuals == (source,) and objective.objective == source
    assert residual.residual_values[0] == objective.value
    tight_value(objective.value, math.atan2(4, 3))
    for actual, reference in zip(objective.gradient, (s.Rational(3, 25), s.Rational(-4, 25))):
        tight_rational(actual, reference)
    for actual, reference in zip(residual.jacobian[0], (s.Rational(3, 25), s.Rational(-4, 25))):
        tight_rational(actual, reference)
    for row, references in zip(objective.hessian, ((-24, 7), (7, 24))):
        for actual, numerator in zip(row, references):
            tight_rational(actual, s.Rational(numerator, 625))


def test_nonzero_box_enclosures_bind_the_original_angle_and_contain_closed_form_derivatives():
    y, x = s.symbols("angle_box_y angle_box_x")
    source = s.atan2(y, x)
    box = [(3, 5), (2, 4)]
    residual = s.enclose_residual_system(source, (y, x), box)
    objective = s.enclose_objective(source, (y, x), box)
    compiled_residual = s.compile_residual_system(source, (y, x)).enclose(box)
    compiled_objective = s.compile_objective(source, (y, x)).enclose(box)
    assert residual.residual_values == compiled_residual.residual_values
    assert residual.jacobian == compiled_residual.jacobian
    assert (objective.value, objective.gradient, objective.hessian) == (
        compiled_objective.value, compiled_objective.gradient, compiled_objective.hessian,
    )
    assert objective.domain == tuple(box) and objective.objective == source
    assert float(objective.value[0]) <= math.atan2(3, 4) + 1e-14
    assert float(objective.value[1]) >= math.atan2(5, 2) - 1e-14
    for yy, xx in ((3, 2), (3, 4), (4, 3), (5, 2), (5, 4)):
        radius = xx*xx + yy*yy
        gradient = (s.Rational(xx, radius), s.Rational(-yy, radius))
        hessian = ((-2*xx*yy, yy*yy - xx*xx), (yy*yy - xx*xx, 2*xx*yy))
        for actual, reference in zip(objective.gradient, gradient):
            assert actual[0] <= reference <= actual[1]
        for actual, reference in zip(residual.jacobian[0], gradient):
            assert actual[0] <= reference <= actual[1]
        for row, numerators in zip(objective.hessian, hessian):
            for actual, numerator in zip(row, numerators):
                assert actual[0] <= s.Rational(numerator, radius*radius) <= actual[1]


@pytest.mark.parametrize("operation", ENCLOSERS)
@pytest.mark.parametrize("box", (
    [(0, 0), (0, 0)],
    [(-1, 1), (-1, 1)],
    [(0, 0), (-2, -1)],
    [(-1, 1), (-2, -1)],
    [(0, 1), (-2, -1)],
    [(-1, 0), (-2, -1)],
))
def test_smooth_source_certificates_refuse_origin_and_every_touched_negative_axis(operation, box):
    y, x = s.symbols("angle_cut_y angle_cut_x")
    with pytest.raises(ValueError):
        operation(s.atan2(y, x), (y, x), box)


@pytest.mark.parametrize("operation", ENCLOSERS)
def test_smooth_boxes_cross_positive_and_vertical_axes_and_keep_exact_near_cut_sides(operation):
    y, x = s.symbols("angle_axes_y angle_axes_x")
    source = s.atan2(y, x)
    for box in (
        [(-1, 1), (1, 2)],
        [(1, 2), (-1, 1)],
        [(-2, -1), (-1, 1)],
    ):
        result = operation(source, (y, x), box)
        assert result.domain == tuple(box)
    tiny = s.Rational(1, 10**80)
    for sign in (-1, 1):
        result = operation(source, (y, x), [(sign*tiny, sign*tiny), (-1, -1)])
        value = result.residual_values[0] if operation is s.enclose_residual_system else result.value
        assert value[1] < 0 if sign < 0 else value[0] > 0


@pytest.mark.parametrize("operation", ENCLOSERS)
@pytest.mark.parametrize("held", ("zero_product", "zero_power"))
def test_held_angle_operations_preserve_original_domains_and_smoothness(operation, held):
    y, x = s.symbols("angle_held_y angle_held_x")
    angle = s.atan2(y, x)
    source = (s.Mul(0, angle, evaluate=False) if held == "zero_product"
              else s.Pow(angle, 0, evaluate=False))
    for point in ((0, 0), (0, -1)):
        with pytest.raises(ValueError):
            operation(source, (y, x), [(point[0], point[0]), (point[1], point[1])])
    result = operation(source, (y, x), [(1, 1), (1, 1)])
    value = result.residual_values[0] if operation is s.enclose_residual_system else result.value
    expected = 0 if held == "zero_product" else 1
    assert value == (expected, expected)
    if held == "zero_power":
        with pytest.raises(ValueError):
            operation(source, (y, x), [(0, 0), (1, 1)])


def test_zero_numeric_products_and_directions_cannot_erase_angle_domains():
    y, x = s.symbols("angle_zero_y angle_zero_x")
    source = s.Mul(0, s.atan2(y, x), evaluate=False)
    residual = s.compile_residual_system(source, (y, x))
    objective = s.compile_objective(source, (y, x))
    for point in ((0, 0), (0, -1)):
        with pytest.raises(ValueError):
            residual.jacobian_vector_product([0, 0], *point)
        with pytest.raises(ValueError):
            objective.hessian_vector_product([0, 0], *point)
    assert residual.evaluate(1, 1)[0] == [0.0]
    assert objective.evaluate(1, 1)[0] == 0.0


@pytest.mark.parametrize("operation", ENCLOSERS)
@pytest.mark.parametrize("position", (0, 1))
def test_certified_ingress_traverses_both_binary_arguments(operation, position):
    y, x, unbound = s.symbols("angle_admit_y angle_admit_x angle_admit_unbound")
    arguments = [y, x]
    arguments[position] = unbound
    with pytest.raises(ValueError):
        operation(s.atan2(*arguments), (y, x), [(1, 1), (1, 1)])
    arguments[position] = s.gamma(y)
    with pytest.raises(ValueError):
        operation(s.atan2(*arguments), (y, x), [(1, 1), (1, 1)])


def test_opaque_atan2_spelling_and_subclasses_cannot_acquire_closed_function_authority():
    y, x = s.symbols("angle_opaque_y angle_opaque_x")
    calls = []

    class CustomAngle(s.atan2):
        def _imp_(self, *args):
            calls.append(args)
            return 0

    for source in (s.Function("atan2")(y, x), CustomAngle(y, x, evaluate=False)):
        for backend in ("math", "frankensympy"):
            with pytest.raises(NotImplementedError):
                s.lambdify((y, x), source, modules=backend)
        for operation in ENCLOSERS:
            with pytest.raises(NotImplementedError):
                operation(source, (y, x), [(1, 1), (1, 1)])
    assert calls == []


@pytest.mark.parametrize("operation", ENCLOSERS)
@pytest.mark.parametrize("limit", ("max_steps", "max_memory_bytes", "max_allocations"))
def test_angle_certificates_refuse_exhaustion_and_cancellation_before_publication(operation, limit):
    y, x = s.symbols("angle_budget_y angle_budget_x")
    source = s.atan2(y, x)
    box = [(3, 4), (2, 3)]
    with pytest.raises(s.EvaluationBudgetExceeded):
        operation(source, (y, x), box, budget=ample_budget(**{limit: 0}))
    cancelled = ample_budget()
    cancelled.cancel()
    with pytest.raises(s.EvaluationCancelled):
        operation(source, (y, x), box, budget=cancelled)
