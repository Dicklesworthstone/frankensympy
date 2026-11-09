"""Large row-local residual programs retain sparse storage and checked domains."""

import json

import pytest
import sympy as s


VARIABLES = 256
RESIDUALS = 4096
BUDGET_FIELDS = (
    "remaining_steps", "remaining_memory_bytes", "remaining_allocations",
)


@pytest.fixture(scope="module")
def large_system():
    # Equal spellings must not merge assumption or Dummy identities when
    # locating the variables that actually occur in an individual row.
    same_name = (
        s.Symbol("large_sparse_coordinate"),
        s.Symbol("large_sparse_coordinate", positive=True),
        s.Dummy("large_sparse_coordinate"),
        s.Dummy("large_sparse_coordinate"),
    )
    variables = same_name + s.symbols("large_sparse_axis_0:252")
    targets = tuple(index % 7 + 1 for index in range(VARIABLES))
    residuals = [
        variables[row % VARIABLES] - targets[row % VARIABLES]
        for row in range(RESIDUALS)
    ]
    source = s.SparseMatrix(residuals)
    expected_source = source.as_immutable()
    arguments = list(variables)
    program = s.compile_residual_system(source, arguments)
    source[0, 0] = 999
    arguments[0] = s.Symbol("replaced_large_sparse_argument")
    return variables, targets, expected_source, program


def test_full_public_axes_compile_and_evaluate_without_dense_derivative_storage(large_system):
    variables, targets, _, program = large_system
    point = [1] * VARIABLES
    expected = [1 - targets[row % VARIABLES] for row in range(RESIDUALS)]
    assert program.native_system.variable_count == VARIABLES
    assert program.native_system.residual_count == RESIDUALS
    assert program.jacobian_pattern == tuple(
        (row, row % VARIABLES) for row in range(RESIDUALS)
    )
    assert program.native_system.nonzero_derivatives == RESIDUALS
    assert program.native_system.proofs_verified == RESIDUALS
    assert all(actual is declared for actual, declared in zip(program.variables, variables))
    assert program.residuals(*point) == expected
    assert program.native_system.residuals(point) == expected
    assert program.evaluate_sparse(*point) == (expected, [1] * RESIDUALS)
    assert program.native_system.evaluate_sparse(point) == (expected, [1] * RESIDUALS)


def test_full_public_axes_apply_both_products_in_typed_column_order(large_system):
    _, _, _, program = large_system
    point = [1] * VARIABLES
    direction = [index % 5 - 2 for index in range(VARIABLES)]
    weights = [row % 3 + 1 for row in range(RESIDUALS)]
    forward = [direction[row % VARIABLES] for row in range(RESIDUALS)]
    transpose = [
        sum(weights[row] for row in range(column, RESIDUALS, VARIABLES))
        for column in range(VARIABLES)
    ]
    assert program.jacobian_vector_product(direction, *point) == forward
    assert program.native_system.jacobian_vector_product(point, direction) == forward
    assert program.jacobian_transpose_vector_product(weights, *point) == transpose
    assert program.native_system.jacobian_transpose_vector_product(point, weights) == transpose


def test_full_public_axes_fit_with_the_explicit_sparse_solver(large_system):
    variables, targets, _, program = large_system
    result = program.solve_least_squares([1] * VARIABLES, backend="sparse")
    assert result.point == pytest.approx(targets, abs=1e-10)
    assert len(result.residuals) == RESIDUALS
    assert max(map(abs, result.residuals)) <= 1e-10
    assert result.gradient_norm <= 1e-10
    assert result.cost < 1e-16
    assert result.backend == "sparse_lsqr" and result.evidence == "approximate"
    assert all(actual is declared for actual, declared in zip(result.variables, variables))
    native = program.native_system.solve_least_squares([1] * VARIABLES, backend="sparse")
    assert native[0] == pytest.approx(targets, abs=1e-10)
    assert native[3] <= 1e-10 and native[-1] == "gradient_tolerance"


def test_large_transport_replays_schema_one_and_preserves_nested_typed_bindings(large_system):
    variables, targets, source, program = large_system
    encoded = program.to_bytes()
    assert len(encoded) <= 16 * 1024 * 1024
    document = json.loads(encoded)
    assert document["schema_version"] == 1
    assert set(document) == {"schema_version", "jacobian"}
    assert len(document["jacobian"]["entries"]) == RESIDUALS
    restored = s.load_residual_system(encoded, residuals=source, args=[variables])
    assert restored.to_bytes() == encoded
    assert all(actual is declared for actual, declared in zip(restored.variables, variables))
    assert restored.jacobian_pattern == program.jacobian_pattern
    assert restored.evaluate_sparse(targets) == ([0] * RESIDUALS, [1] * RESIDUALS)
    assert restored.jacobian_vector_product([1] * VARIABLES, targets) == [1] * RESIDUALS
    assert restored.jacobian_transpose_vector_product([1] * RESIDUALS, targets) == [16] * VARIABLES
    assert restored.solve_least_squares(targets, backend="sparse", maxiter=0).point == targets

    reordered = (variables[1], variables[0], *variables[2:])
    with pytest.raises(ValueError, match="expected typed order"):
        s.load_residual_system(encoded, residuals=source, args=reordered)
    with pytest.raises(ValueError, match="expected exact source"):
        s.load_residual_system(encoded, residuals=[0] * RESIDUALS, args=variables)

    # Keep sparsity metadata internally consistent while omitting one nonzero.
    # The independent checker must still discover that row's real dependency.
    jacobian = document["jacobian"]
    jacobian["entries"].pop(0)
    jacobian["pattern"]["nonzeros"].pop(0)
    jacobian["pattern"]["nnz"] -= 1
    forged = json.dumps(document, separators=(",", ":")).encode()
    with pytest.raises(ValueError, match=r"Omitted entry \(0, 0\)"):
        s.load_residual_system(forged, residuals=source, args=variables)


@pytest.mark.parametrize("operation", ["evaluate", "jacobian", "evaluate_batch"])
@pytest.mark.parametrize("native", [False, True])
def test_oversized_dense_results_refuse_before_input_conversion_and_budget_use(
    large_system, operation, native,
):
    _, _, _, program = large_system
    effects = []

    class ForbiddenInput:
        def __float__(self):
            effects.append("float")
            raise AssertionError("dense shape refusal must precede input conversion")

    point = [ForbiddenInput()] * VARIABLES
    target = program.native_system if native else program
    cancelled = s.EvaluationBudget()
    cancelled.cancel()
    for budget in (
        s.EvaluationBudget(),
        s.EvaluationBudget(max_steps=0, max_memory_bytes=0, max_allocations=0),
        cancelled,
    ):
        before = tuple(getattr(budget, field) for field in BUDGET_FIELDS)
        with pytest.raises(ValueError, match="[Dd]ense.*65536"):
            if operation == "evaluate_batch":
                target.evaluate_batch([point], budget=budget)
            elif native:
                getattr(target, operation)(point, budget=budget)
            else:
                getattr(target, operation)(*point, budget=budget)
        assert tuple(getattr(budget, field) for field in BUDGET_FIELDS) == before
        assert effects == []


def test_dense_boundary_and_separate_batch_value_limit_remain_inclusive():
    variables = s.symbols("dense_sparse_boundary_0:256")
    program = s.compile_residual_system(variables, variables)
    point = tuple(range(VARIABLES))
    identity = [
        [int(row == column) for column in range(VARIABLES)]
        for row in range(VARIABLES)
    ]
    assert program.evaluate(*point) == (list(point), identity)
    assert program.native_system.jacobian(point) == identity
    assert program.evaluate_batch([point] * 3) == [(list(point), identity)] * 3
    # Four rows would hold 263168 values, above the unchanged 262144 cap.
    for target in (program, program.native_system):
        budget = s.EvaluationBudget(max_steps=0, max_memory_bytes=0, max_allocations=0)
        with pytest.raises(ValueError, match="total input/output value"):
            target.evaluate_batch([[object()] * VARIABLES] * 4, budget=budget)
        assert tuple(getattr(budget, field) for field in BUDGET_FIELDS) == (0, 0, 0)


@pytest.fixture(scope="module")
def large_constants():
    variables = s.symbols("large_unused_axis_0:256")
    valid = s.compile_residual_system([7] * RESIDUALS, variables)
    source = [s.Integer(7)] * (RESIDUALS - 1) + [s.log(-1, evaluate=False)]
    invalid = s.compile_residual_system(source, variables)
    return valid, invalid


def test_full_unused_axes_keep_empty_patterns_and_stationary_nonroot_semantics(large_constants):
    program, _ = large_constants
    point = [3] * VARIABLES
    assert program.jacobian_pattern == ()
    assert program.evaluate_sparse(*point) == ([7] * RESIDUALS, [])
    assert program.native_system.evaluate_sparse(point) == ([7] * RESIDUALS, [])
    assert program.jacobian_vector_product([0] * VARIABLES, *point) == [0] * RESIDUALS
    assert program.jacobian_transpose_vector_product([1] * RESIDUALS, *point) == [0] * VARIABLES
    candidate = program.solve_least_squares(point, backend="sparse", maxiter=0)
    assert candidate.point == tuple(point)
    assert candidate.cost == 49 * RESIDUALS / 2 and candidate.gradient_norm == 0
    assert candidate.iterations == 0 and candidate.evaluations == 2
    assert candidate.evidence == "approximate" and not hasattr(candidate, "certificate")


def sparse_call(program, operation, point, *, native, budget):
    target = program.native_system if native else program
    arguments = (point,) if native else tuple(point)
    if operation == "residuals":
        return target.residuals(*arguments, budget=budget)
    if operation == "sparse":
        return target.evaluate_sparse(*arguments, budget=budget)
    if operation == "forward":
        direction = [0] * VARIABLES
        if native:
            return target.jacobian_vector_product(point, direction, budget=budget)
        return target.jacobian_vector_product(direction, *point, budget=budget)
    if operation == "transpose":
        weights = [0] * RESIDUALS
        if native:
            return target.jacobian_transpose_vector_product(point, weights, budget=budget)
        return target.jacobian_transpose_vector_product(weights, *point, budget=budget)
    return target.solve_least_squares(point, backend="sparse", maxiter=0, budget=budget)


@pytest.mark.parametrize("operation", ["residuals", "sparse", "forward", "transpose", "solve"])
@pytest.mark.parametrize("native", [False, True])
def test_variable_independent_last_row_domains_cannot_disappear_from_sparse_work(
    large_constants, operation, native,
):
    _, program = large_constants
    assert program.jacobian_pattern == ()
    budget = s.EvaluationBudget()
    before = budget.remaining_steps
    with pytest.raises(ValueError):
        sparse_call(program, operation, [0] * VARIABLES, native=native, budget=budget)
    assert budget.remaining_steps < before


@pytest.mark.parametrize("operation", ["residuals", "sparse", "forward", "transpose", "solve"])
@pytest.mark.parametrize("native", [False, True])
def test_large_sparse_calls_remain_cancellable_before_publication(large_system, operation, native):
    _, _, _, program = large_system
    budget = s.EvaluationBudget()
    before = tuple(getattr(budget, field) for field in BUDGET_FIELDS)
    budget.cancel()
    with pytest.raises(s.EvaluationCancelled):
        sparse_call(program, operation, [1] * VARIABLES, native=native, budget=budget)
    assert tuple(getattr(budget, field) for field in BUDGET_FIELDS) == before


@pytest.mark.parametrize("limit,remaining", [
    ("max_steps", "remaining_steps"),
    ("max_memory_bytes", "remaining_memory_bytes"),
    ("max_allocations", "remaining_allocations"),
])
def test_large_sparse_work_uses_the_same_cumulative_budget(large_system, limit, remaining):
    _, _, _, program = large_system
    point = [1] * VARIABLES
    measured = s.EvaluationBudget()
    before = getattr(measured, remaining)
    expected = program.evaluate_sparse(*point, budget=measured)
    cost = before - getattr(measured, remaining)
    assert cost > 0
    exact = s.EvaluationBudget(**{limit: cost})
    assert program.evaluate_sparse(*point, budget=exact) == expected
    assert getattr(exact, remaining) == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.evaluate_sparse(*point, budget=exact)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.evaluate_sparse(*point, budget=s.EvaluationBudget(**{limit: cost - 1}))


def test_source_node_and_row_dependency_work_limits_are_not_relaxed():
    x = s.Symbol("large_source_node_bound")
    row = s.Add(*([x] * 16), evaluate=False)
    with pytest.raises(ValueError, match="node"):
        s.compile_residual_system([row] * RESIDUALS, x)
    variables = s.symbols("large_dependency_bound_0:256")
    # 257 nodes times 256 declared dependencies exceeds 65536, despite one row.
    with pytest.raises(ValueError, match="differentiation work"):
        s.compile_residual_system([s.Add(*variables, evaluate=False)], variables)


def test_undeclared_same_named_symbol_is_not_a_variable_independent_constant(large_system):
    variables, _, _, _ = large_system
    foreign = s.Dummy("large_sparse_coordinate")
    with pytest.raises(ValueError):
        s.compile_residual_system([foreign] * RESIDUALS, variables)
