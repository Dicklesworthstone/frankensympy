"""Sparse root candidates require original residuals, not stationarity alone."""

import math

import pytest
import sympy as s


BUDGET_FIELDS = (
    "remaining_steps", "remaining_memory_bytes", "remaining_allocations",
)


@pytest.fixture(scope="module")
def large_root_system():
    variables = (
        s.Symbol("sparse_root_same"),
        s.Symbol("sparse_root_same", positive=True),
        s.Dummy("sparse_root_same"),
        s.Dummy("sparse_root_same"),
        *s.symbols("sparse_root_axis_0:252"),
    )
    targets = tuple(index % 7 + 1 for index in range(len(variables)))
    residuals = [variables[0] - targets[0]] + [
        2*variables[index] - variables[index - 1] - (2*targets[index] - targets[index - 1])
        for index in range(1, len(variables))
    ]
    source = s.SparseMatrix(residuals)
    captured = source.as_immutable()
    arguments = list(variables)
    program = s.compile_residual_system(source, arguments)
    source[0, 0] = 999
    arguments[0] = s.Symbol("replaced_sparse_root_axis")
    return variables, targets, captured, program


def test_256_variable_banded_system_uses_the_explicit_sparse_root_backend(large_root_system):
    variables, targets, _, program = large_root_system
    for target in (program, program.native_system):
        with pytest.raises(ValueError, match="square"):
            target.solve_candidate([1] * 256)
    candidate = program.solve_candidate([1] * 256, backend="sparse")
    assert type(candidate) is s.NumericRootCandidate
    assert candidate.root == pytest.approx(targets, abs=1e-9)
    assert all(actual is expected for actual, expected in zip(candidate.variables, variables))
    residuals = program.native_system.residuals(candidate.root)
    assert candidate.residual_norm == max(map(abs, residuals))
    assert candidate.residual_norm <= 1e-12 and candidate.iterations > 0
    assert candidate.backend == "sparse" and candidate.evidence == "approximate"
    assert not hasattr(candidate, "certificate") and not hasattr(candidate, "root_enclosure")
    native = program.native_system.solve_candidate([1] * 256, backend="sparse")
    assert native[0] == pytest.approx(targets, abs=1e-9)
    assert native[1] <= 1e-12 and native[2] > 0
    for field in ("root", "variables", "residual_norm", "iterations", "backend", "_backend"):
        with pytest.raises(AttributeError):
            setattr(candidate, field, None)
        with pytest.raises(AttributeError):
            delattr(candidate, field)


def test_large_transport_and_nested_arguments_preserve_typed_root_coordinates(large_root_system):
    variables, targets, source, program = large_root_system
    encoded = program.to_bytes()
    restored = s.load_residual_system(encoded, residuals=source, args=[variables])
    assert restored.to_bytes() == encoded
    candidate = restored.solve_candidate([1] * 256, backend="sparse")
    assert candidate.root == pytest.approx(targets, abs=1e-9)
    assert all(actual is expected for actual, expected in zip(candidate.variables, variables))
    assert max(map(abs, restored.residuals(candidate.root))) <= 1e-12
    assert restored.solve_candidate(targets, backend="sparse", maxiter=0, tol=0).iterations == 0
    wrong_order = (variables[1], variables[0], *variables[2:])
    with pytest.raises(ValueError, match="expected typed order"):
        s.load_residual_system(encoded, residuals=source, args=wrong_order)


def test_nsolve_sparse_returns_the_existing_matrix_and_dictionary_shapes(large_root_system):
    variables, targets, source, _ = large_root_system
    result = s.nsolve(source, variables, [1] * 256, backend="sparse")
    assert isinstance(result, s.Matrix) and result.shape == (256, 1)
    assert [float(value) for value in result] == pytest.approx(targets, abs=1e-9)
    mappings = s.nsolve(
        source, variables, targets, backend="sparse", maxiter=0, tol=0, dict=True,
    )
    assert type(mappings) is list and len(mappings) == 1
    assert list(mappings[0]) == list(variables)
    assert all(actual is expected for actual, expected in zip(mappings[0], variables))
    assert [float(mappings[0][axis]) for axis in variables] == list(targets)


def test_scalar_equations_and_existing_defaults_keep_their_public_result_forms():
    x = s.Symbol("sparse_scalar_root")
    equation = s.Eq(x*x, 2, evaluate=False)
    root = s.nsolve(equation, x, 1, backend="sparse")
    assert type(root) is s.Float and float(root) == pytest.approx(math.sqrt(2), abs=1e-12)
    assert type(s.nsolve(equation, x, 1, backend="sparse", dict=True)[0][x]) is s.Float
    assert s.nsolve(x-1, x, 0) == s.nsolve(x-1, x, 0, backend="python")
    program = s.compile_residual_system([x-1], x)
    assert program.solve_candidate(0).backend == "frankennumpy"
    assert program.solve_candidate(0, backend="frankennumpy").root == (1.0,)
    assert s.nsolve([x-1, 2*x-2], [x], [0]).shape == (1, 1)
    with pytest.raises(ValueError, match="square"):
        s.nsolve([x-1, 2*x-2], [x], [0], backend="sparse")


def test_nonlinear_and_rank_deficient_roots_retain_the_original_residual_test():
    x, y = s.symbols("sparse_nonlinear_root_x sparse_nonlinear_root_y")
    nonlinear = s.compile_residual_system([x*x+y-5, x-y-1], (x, y))
    result = nonlinear.solve_candidate([1, 1], backend="sparse")
    assert result.root == pytest.approx((2, 1), abs=1e-10)
    assert result.residual_norm <= 1e-12
    singular = s.compile_residual_system([x-1, 2*x-2], (x, y))
    result = singular.solve_candidate([0, 7], backend="sparse")
    assert result.root == pytest.approx((1, 7), abs=1e-12)
    assert result.residual_norm <= 1e-12 and result.evidence == "approximate"
    initial_root = s.compile_residual_system([x*x, y], (x, y))
    result = initial_root.solve_candidate([0, 0], backend="sparse", tol=0, maxiter=0)
    assert result.root == (0.0, 0.0) and result.iterations == 0


@pytest.mark.parametrize("kind", ["constant", "stationary", "inconsistent"])
def test_stationary_nonroots_never_become_root_candidates(kind):
    x, y = s.symbols("sparse_false_root_x sparse_false_root_y")
    if kind == "inconsistent":
        program = s.compile_residual_system([x-1, x-3], (x, y))
        point = [0, 7]
    else:
        program = s.compile_residual_system([1 if kind == "constant" else x*x+1], x)
        point = [0]
    for target in (program, program.native_system):
        with pytest.raises(ValueError, match="convergence"):
            target.solve_candidate(point, backend="sparse", maxiter=4)
    with pytest.raises(ValueError):
        s.nsolve(x*x+1, x, 0, backend="sparse", maxiter=4)


@pytest.mark.parametrize("scale", [s.Rational(1, 10**200), s.Integer(10**200)])
def test_extreme_row_scales_cannot_replace_original_residual_convergence(scale):
    x = s.Symbol("sparse_scaled_root")
    program = s.compile_residual_system([scale * (x-1)], x)
    result = program.solve_candidate(0, backend="sparse", tol=0)
    assert result.root == (1.0,) and result.residual_norm == 0
    assert result.iterations > 0
    native = program.native_system.solve_candidate([0.0], backend="sparse", tol=0)
    assert native[0] == [1.0] and native[1] == 0 and native[2] > 0


def test_weak_jacobian_does_not_mistake_a_small_gradient_for_a_root():
    x = s.Symbol("sparse_weak_root")
    program = s.compile_residual_system([x / 10**200 - 1], x)
    result = program.solve_candidate(0, backend="sparse", max_step_norm=1e210)
    assert result.root == pytest.approx((1e200,), rel=1e-12)
    assert result.residual_norm <= 1e-12 and result.iterations > 0


@pytest.mark.parametrize("kind,initial,root", [
    ("log", 10, 1), ("sqrt", 4, 1), ("exp", -7, 0),
])
def test_invalid_trial_domains_backtrack_before_root_publication(kind, initial, root):
    x = s.Symbol("sparse_root_trial")
    source = {"log": s.log(x), "sqrt": s.sqrt(x)-1, "exp": s.exp(x)-1}[kind]
    program = s.compile_residual_system([source], x)
    result = program.solve_candidate(initial, backend="sparse")
    assert result.root == pytest.approx((root,), abs=1e-10)
    assert result.residual_norm <= 1e-12 and result.iterations > 0


@pytest.mark.parametrize("kind,initial", [("log", -1), ("sqrt", 0), ("exp", 1000)])
def test_invalid_initial_source_or_derivative_is_terminal_and_keeps_work(kind, initial):
    x = s.Symbol("sparse_root_initial")
    source = {"log": s.log(x), "sqrt": s.sqrt(x)-1, "exp": s.exp(x)-1}[kind]
    program = s.compile_residual_system([source], x)
    for target in (program, program.native_system):
        budget = s.EvaluationBudget()
        before = budget.remaining_steps
        with pytest.raises(ValueError):
            target.solve_candidate([initial], backend="sparse", budget=budget)
        assert budget.remaining_steps < before


@pytest.mark.parametrize("hidden_kind", ["zero_product", "zero_power"])
def test_initial_zero_residuals_cannot_hide_original_source_holes(hidden_kind):
    x = s.Symbol("sparse_root_held")
    source = (s.Mul(0, s.log(x), evaluate=False) if hidden_kind == "zero_product"
              else s.Add(s.Pow(s.log(x), 0, evaluate=False), -1, evaluate=False))
    program = s.compile_residual_system([source], x)
    for target in (program, program.native_system):
        with pytest.raises(ValueError):
            target.solve_candidate([0], backend="sparse", maxiter=0)


def test_step_clipping_and_iteration_limits_do_not_authorize_stagnation():
    x = s.Symbol("sparse_root_steps")
    program = s.compile_residual_system([x-2], x)
    result = program.solve_candidate(0, backend="sparse", max_step_norm=0.25, maxiter=8, tol=0)
    assert result.root == (2.0,) and result.iterations == 8
    with pytest.raises(ValueError, match="convergence"):
        program.solve_candidate(0, backend="sparse", max_step_norm=0.25, maxiter=7, tol=0)
    with pytest.raises(ValueError, match="convergence"):
        program.solve_candidate(0, backend="sparse", maxiter=0)
    with pytest.raises(ValueError, match="convergence"):
        program.solve_candidate(
            1, backend="sparse", max_step_norm=float.fromhex("0x0.0000000000001p-1022"),
        )
    assert program.solve_candidate(2, backend="sparse", maxiter=0, tol=0).iterations == 0
    assert float(s.nsolve(x-2, x, 2, backend="sparse", maxiter=0, tol=0)) == 2


@pytest.mark.parametrize("options", [
    {"backend": "auto"}, {"backend": "SPARSE"},
    {"backend": "sparse", "tol": -1}, {"backend": "sparse", "tol": math.nan},
    {"backend": "sparse", "maxiter": -1}, {"backend": "sparse", "maxiter": 1025},
    {"backend": "sparse", "max_step_norm": 0},
    {"backend": "sparse", "max_step_norm": math.inf},
])
def test_invalid_controls_refuse_before_initial_conversion_or_budget_use(options):
    x = s.Symbol("sparse_root_controls")
    program = s.compile_residual_system([x-1], x)
    effects = []

    class Initial:
        def __float__(self):
            effects.append("float")
            raise AssertionError("invalid controls reached initial conversion")

    for target in (program, program.native_system):
        budget = s.EvaluationBudget(max_steps=0, max_memory_bytes=0, max_allocations=0)
        with pytest.raises(ValueError):
            target.solve_candidate([Initial()], budget=budget, **options)
        assert tuple(getattr(budget, field) for field in BUDGET_FIELDS) == (0, 0, 0)
    with pytest.raises(ValueError):
        s.nsolve(x-1, x, Initial(), **options)
    assert effects == []


@pytest.mark.parametrize("backend", [None, 1, b"sparse"])
def test_backend_selectors_require_strings_at_every_entrypoint(backend):
    x = s.Symbol("sparse_root_backend_type")
    program = s.compile_residual_system([x-1], x)
    for target in (program, program.native_system):
        with pytest.raises(TypeError):
            target.solve_candidate([0], backend=backend)
    with pytest.raises(TypeError):
        s.nsolve(x-1, x, 0, backend=backend)


def test_non_square_and_oversized_requests_refuse_before_input_hooks_or_work():
    x, y = s.symbols("sparse_root_shape_x sparse_root_shape_y")
    programs = (
        s.compile_residual_system([], ()),
        s.compile_residual_system([x, x+1], x),
        s.compile_residual_system([x+y], (x, y)),
    )
    for program in programs:
        for target in (program, program.native_system):
            budget = s.EvaluationBudget(max_steps=0, max_memory_bytes=0, max_allocations=0)
            with pytest.raises(ValueError, match="square"):
                target.solve_candidate([object()], backend="sparse", budget=budget)
            assert tuple(getattr(budget, field) for field in BUDGET_FIELDS) == (0, 0, 0)
    variables = s.symbols("sparse_root_too_wide_0:257")
    with pytest.raises(ValueError):
        s.nsolve(variables, variables, [object()] * 257, backend="sparse")


@pytest.mark.parametrize("limit,remaining", [
    ("max_steps", "remaining_steps"),
    ("max_memory_bytes", "remaining_memory_bytes"),
    ("max_allocations", "remaining_allocations"),
])
def test_sparse_root_iterations_and_rejected_trials_share_one_budget(limit, remaining):
    x = s.Symbol("sparse_root_budget")
    program = s.compile_residual_system([s.log(x)], x)
    measured = s.EvaluationBudget()
    before = getattr(measured, remaining)
    root = program.solve_candidate(10, backend="sparse", budget=measured).root
    cost = before - getattr(measured, remaining)
    assert cost > 0
    exact = s.EvaluationBudget(**{limit: cost})
    assert program.solve_candidate(10, backend="sparse", budget=exact).root == root
    assert getattr(exact, remaining) == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_candidate(10, backend="sparse", budget=exact)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.solve_candidate(10, backend="sparse", budget=s.EvaluationBudget(**{limit: cost-1}))


def test_cancellation_remains_typed_through_compiled_native_and_nsolve_paths():
    x = s.Symbol("sparse_root_cancel")
    program = s.compile_residual_system([x-1], x)
    budget = s.EvaluationBudget()
    budget.cancel()
    for call in (
        lambda: program.solve_candidate(0, backend="sparse", budget=budget),
        lambda: program.native_system.solve_candidate([0.0], backend="sparse", budget=budget),
        lambda: s.nsolve(x-1, x, 0, backend="sparse", budget=budget),
    ):
        with pytest.raises(s.EvaluationCancelled):
            call()
    with pytest.raises(s.EvaluationCancelled):
        program.solve_candidate(0, tol=-1, budget=budget)
    with pytest.raises(ValueError):
        program.solve_candidate(0, backend="sparse", tol=-1, budget=budget)
    assert tuple(getattr(budget, field) for field in BUDGET_FIELDS) == (
        10_000_000, 256 * 1024 * 1024, 100_000,
    )


@pytest.mark.parametrize("entrypoint", ["compiled", "nsolve"])
def test_sparse_proposals_can_acquire_root_evidence_only_from_the_existing_checker(entrypoint):
    x = s.Symbol("sparse_certified_root")
    source = x*x-2
    budget = s.EvaluationBudget(max_steps=1 << 45, max_memory_bytes=1 << 40, max_allocations=1 << 30)
    if entrypoint == "compiled":
        receipt = s.compile_residual_system([source], x).solve_certified(1, backend="sparse", budget=budget)
    else:
        receipt = s.nsolve_certified(source, x, 1, backend="sparse", budget=budget)
    assert type(receipt) is s.RootBoxReceipt and receipt.residuals == (source,)
    lo, hi = receipt.root_enclosure[0]
    assert 0 < lo <= hi and lo*lo <= 2 <= hi*hi
    assert receipt.contraction_bound < 1


def test_sparse_selection_cannot_bypass_certification_shape_or_source_preflight(monkeypatch):
    variables = s.symbols("sparse_certified_too_wide_0:17")
    wide = s.compile_residual_system([axis-1 for axis in variables], variables)
    x = s.Symbol("sparse_certified_inexact")
    inexact = s.compile_residual_system([s.Add(x, s.Float(-1.25), evaluate=False)], x)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("candidate")
        raise AssertionError("certification preflight reached candidate generation")

    class Initial:
        def __float__(self):
            calls.append("initial")
            return 0.0

    monkeypatch.setattr(type(wide), "solve_candidate", forbidden)
    budget = s.EvaluationBudget(max_steps=0)
    for program in (wide, inexact):
        with pytest.raises((ValueError, NotImplementedError)):
            program.solve_certified(Initial(), backend="sparse", budget=budget)
    with pytest.raises(ValueError):
        s.nsolve_certified(variables, variables, Initial(), backend="sparse", budget=budget)
    assert calls == [] and budget.remaining_steps == 0


def test_undefined_intrinsic_spelling_does_not_gain_sparse_root_capability():
    x = s.Symbol("sparse_root_opaque")
    with pytest.raises(NotImplementedError):
        s.nsolve(s.Function("sin")(x), x, 0, backend="sparse")
