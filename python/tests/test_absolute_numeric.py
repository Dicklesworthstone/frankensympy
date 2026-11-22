"""Absolute-value branches compose through checked numerical public APIs."""

import math

import pytest
import sympy as s


@pytest.mark.parametrize("sign", (-1, 1))
def test_nonzero_absolute_jets_cover_the_binary64_range_without_division(sign):
    x = s.Symbol("absolute_range")
    source = s.Abs(x)
    scalar = s.compile_numeric(x, source)
    residual = s.compile_residual_system(source, x)
    objective = s.compile_objective(source, x)
    for magnitude in (math.nextafter(0.0, 1.0), 1e-300, 1.0, 1e200,
                      float.fromhex("0x1.fffffffffffffp+1023")):
        point = sign*magnitude
        assert scalar(point) == magnitude
        assert residual.evaluate(point) == ([magnitude], [[float(sign)]])
        assert objective.evaluate(point) == (magnitude, [float(sign)], [[0.0]])
    assert scalar(0.0) == scalar(-0.0) == 0.0


@pytest.mark.parametrize("point, value, gradient, curvature", (
    ((2, 1), 3, [4, 2], 2),
    ((1, -1), 4, [-2, -2], -2),
))
def test_nonlinear_absolute_chain_keeps_gradient_hessian_and_directional_products(
        point, value, gradient, curvature):
    x, y = s.symbols("absolute_chain_x absolute_chain_y")
    source = s.Abs(x*x + 2*y - 3)
    residual = s.compile_residual_system(source, (x, y))
    objective = s.compile_objective(source, (x, y))
    assert residual.evaluate(*point) == ([value], [gradient])
    assert objective.evaluate(*point) == (value, gradient, [[curvature, 0], [0, 0]])
    assert residual.jacobian_vector_product([3, -2], *point) == [3*gradient[0]-2*gradient[1]]
    assert residual.jacobian_transpose_vector_product([5], *point) == [5*v for v in gradient]
    assert objective.hessian_vector_product([3, -2], *point) == [3*curvature, 0]


def test_nested_absolute_values_keep_inner_and_outer_cusp_domains():
    x = s.Symbol("absolute_nested")
    source = s.Abs(s.Abs(x)-2)
    objective = s.compile_objective(source, x)
    for point, gradient in ((-3, -1), (-1, 1), (1, -1), (3, 1)):
        assert objective.evaluate(point) == (1, [gradient], [[0]])
    for point in (-2, 0, 2):
        with pytest.raises(ValueError):
            objective.evaluate(point)


@pytest.mark.parametrize("held", ("direct", "zero_product", "zero_power", "square"))
def test_cusps_survive_held_arithmetic_and_zero_directional_products(held):
    x = s.Symbol("absolute_cusp")
    primitive = s.Abs(x, evaluate=False)
    source = {
        "direct": primitive,
        "zero_product": s.Mul(0, primitive, evaluate=False),
        "zero_power": s.Pow(primitive, 0, evaluate=False),
        "square": s.Abs(x*x, evaluate=False),
    }[held]
    residual = s.compile_residual_system(source, x)
    objective = s.compile_objective(source, x)
    for point in (0.0, -0.0):
        for operation in (
            lambda: residual.evaluate(point),
            lambda: residual.jacobian_vector_product([0], point),
            lambda: residual.jacobian_transpose_vector_product([0], point),
            lambda: objective.evaluate(point),
            lambda: objective.hessian_vector_product([0], point),
        ):
            with pytest.raises(ValueError):
                operation()


@pytest.mark.parametrize("held", ("reciprocal", "zero_product", "log"))
def test_absolute_derivatives_retain_original_child_poles(held):
    x = s.Symbol("absolute_original_domain")
    argument = {
        "reciprocal": 1/x,
        "zero_product": s.Add(1, s.Mul(0, 1/x, evaluate=False), evaluate=False),
        "log": s.log(x),
    }[held]
    source = s.Abs(argument, evaluate=False)
    for constructor in (s.compile_residual_system, s.compile_objective):
        program = constructor(source, x)
        program.evaluate(2)
        with pytest.raises(ValueError):
            program.evaluate(0)


@pytest.mark.parametrize("sign", (-1, 1))
def test_smooth_branch_root_and_minimum_proposals_reach_independent_certification(sign):
    x = s.Symbol("absolute_solve")
    residual = s.Abs(x)-2
    assert float(s.nsolve(residual, x, sign*3, backend="sparse")) == pytest.approx(sign*2)
    root = s.nsolve_certified(residual, x, sign*3)
    assert root.residuals == (residual,)
    assert root.root_enclosure == ((sign*2, sign*2),)
    assert root.contraction_bound == 0
    program = s.compile_objective(residual**2, x)
    candidate = program.minimize(sign*3)
    assert candidate.point == pytest.approx((sign*2,))
    assert candidate.evidence == "approximate"
    minimum = program.minimize_certified(sign*3)
    assert minimum.objective == residual**2
    assert minimum.kind == "strict_minimum"
    assert minimum.stationary_enclosure == ((sign*2, sign*2),)
    assert minimum.hessian_enclosure == (((2, 2),),)


def test_exact_source_enclosures_agree_with_compiled_smooth_branch_programs():
    x = s.Symbol("absolute_enclosure")
    source = s.Abs(x)-2
    residual = s.compile_residual_system(source, x)
    objective = s.compile_objective(source**2, x)
    for box in ([(-3, -1)], [(1, 3)]):
        direct = s.enclose_residual_system(source, x, box)
        compiled = residual.enclose(box)
        assert (compiled.residual_values, compiled.jacobian) == (
            direct.residual_values, direct.jacobian,
        )
        direct = s.enclose_objective(source**2, x, box)
        compiled = objective.enclose(box)
        assert (compiled.value, compiled.gradient, compiled.hessian) == (
            direct.value, direct.gradient, direct.hessian,
        )
    for program in (residual, objective):
        with pytest.raises(ValueError):
            program.enclose([(-1, 1)])


def test_absolute_program_transport_replays_both_branches_and_rejects_forged_opcodes():
    x = s.Symbol("absolute_transport", real=True)
    source = s.Abs(x)
    for constructor, loader, portable_loader in (
        (s.compile_residual_system, s.load_residual_system, s.load_portable_residual_system),
        (s.compile_objective, s.load_objective, s.load_portable_objective),
    ):
        program = constructor(source, x)
        sources = {"residuals" if constructor is s.compile_residual_system else "objective": source,
                   "args": x}
        for restored in (loader(program.to_bytes(), **sources),
                         portable_loader(program.to_portable_bytes(), **sources)):
            for point in (-3, 3):
                assert restored.evaluate(point) == program.evaluate(point)
            with pytest.raises(ValueError):
                restored.evaluate(0)
        encoded = program.to_bytes()
        assert b'"AbsDerivative"' in encoded
        with pytest.raises(ValueError):
            loader(encoded.replace(b'"AbsDerivative"', b'"AbsSecondDerivative"'), **sources)
        if constructor is s.compile_objective:
            assert b'"Abs"' in encoded
            assert b'"AbsSecondDerivative"' in encoded
            with pytest.raises(ValueError):
                loader(encoded.replace(b'"Abs"', b'"AbsDerivative"'), **sources)
            with pytest.raises(ValueError):
                loader(encoded.replace(b'"AbsSecondDerivative"', b'"AbsDerivative"'), **sources)


def test_same_name_symbols_and_dummies_remain_distinct_absolute_coordinates():
    plain = s.Symbol("absolute_typed")
    positive = s.Symbol("absolute_typed", positive=True)
    left, right = s.Dummy("absolute_typed"), s.Dummy("absolute_typed")
    variables = (plain, positive, left, right)
    source = s.Abs(plain-positive) + s.Abs(left-right)
    program = s.compile_objective(source, ((plain, positive), (left, right)))
    assert all(actual is expected for actual, expected in zip(program.variables, variables))
    assert program.evaluate((1, 4), (6, 2)) == (
        7, [-1, 1, 1, -1], [[0]*4 for _ in range(4)],
    )
    restored = s.load_objective(program.to_bytes(), objective=source,
                               args=((plain, positive), (left, right)))
    assert restored.evaluate((1, 4), (6, 2)) == program.evaluate((1, 4), (6, 2))


@pytest.mark.parametrize("name", ("abs", "Abs", "_fsym_abs_d1", "_fsym_abs_d2"))
def test_opaque_names_do_not_acquire_absolute_jet_capabilities(name):
    x = s.Symbol("absolute_opaque")
    opaque = s.Function(name)(x)
    for operation in (lambda: s.compile_numeric(x, opaque),
                      lambda: s.compile_residual_system(opaque, x),
                      lambda: s.compile_objective(opaque, x)):
        with pytest.raises((ValueError, TypeError, NotImplementedError)):
            operation()


def test_absolute_compilation_keeps_budget_cancellation_and_public_symbolic_form():
    x = s.Symbol("absolute_budget")
    source = s.Abs(x)
    symbolic = s.diff(source, x)
    for constructor in (s.compile_residual_system, s.compile_objective):
        with pytest.raises(s.EvaluationBudgetExceeded):
            constructor(source, x, budget=s.EvaluationBudget(max_steps=0))
        program = constructor(source, x)
        with pytest.raises(s.EvaluationBudgetExceeded):
            program.evaluate(2, budget=s.EvaluationBudget(max_steps=0))
        cancelled = s.EvaluationBudget()
        cancelled.cancel()
        with pytest.raises(s.EvaluationCancelled):
            program.evaluate(2, budget=cancelled)
    assert s.diff(source, x) == symbolic
    assert "_fsym_abs_" not in str(symbolic)
