"""Checked sparse residual derivatives through the actual public/native boundary."""

import math

import pytest
import sympy as s


def _rectangular_system():
    x, y, z, unused = s.symbols("sparse_product_x sparse_product_y sparse_product_z sparse_product_unused")
    program = s.compile_residual_system([x*x+y, 7, y*z-x], (x, y, z, unused))
    return program, (2, 3, 5, 11)


def _operation(program, name, point, *, budget=None):
    if name == "sparse":
        return program.evaluate_sparse(*point, budget=budget)
    if name == "forward":
        return program.jacobian_vector_product([2, -3, 7, 99], *point, budget=budget)
    return program.jacobian_transpose_vector_product([2, 11, -3], *point, budget=budget)


def test_rectangular_sparse_values_and_both_products_have_independent_expected_values():
    program, point = _rectangular_system()
    assert program.jacobian_pattern == ((0, 0), (0, 1), (2, 0), (2, 1), (2, 2))
    assert program.evaluate_sparse(*point) == ([7, 7, 13], [4, 1, -1, 5, 3])
    assert program.jacobian_vector_product([2, -3, 7, 99], *point) == [5, 0, 4]
    assert program.jacobian_transpose_vector_product([2, 11, -3], *point) == [11, -13, -9, 0]
    assert program.native_system.nonzero_derivatives == 5
    assert program.native_system.proofs_verified == 5


def test_structural_pattern_keeps_derivatives_that_vanish_only_at_the_evaluation_point():
    x, y = s.symbols("sparse_zero_x sparse_zero_y")
    program = s.compile_residual_system([x*y, x*x, y], (x, y))
    pattern = ((0, 0), (0, 1), (1, 0), (2, 1))
    assert program.jacobian_pattern == pattern
    assert program.evaluate_sparse(0, 3) == ([0, 0, 3], [3, 0, 0, 1])
    assert program.jacobian_vector_product([2, 5], 0, 3) == [6, 0, 5]
    assert program.jacobian_transpose_vector_product([7, 11, 13], 0, 3) == [21, 13]
    assert program.evaluate_sparse(2, 3) == ([6, 4, 3], [3, 2, 4, 1])
    assert program.jacobian_pattern == pattern


def test_transcendental_entries_and_products_match_closed_form_derivatives():
    x, y = s.symbols("sparse_transcendental_x sparse_transcendental_y")
    program = s.compile_residual_system([s.exp(x)*y, s.sin(x)+s.log(y)], (x, y))
    residuals, entries = program.evaluate_sparse(0, 2)
    assert residuals == pytest.approx([2, math.log(2)])
    assert entries == pytest.approx([2, 1, 1, 0.5])
    assert program.jacobian_vector_product([3, 5], 0, 2) == pytest.approx([11, 5.5])
    assert program.jacobian_transpose_vector_product([7, 11], 0, 2) == pytest.approx([25, 12.5])


def test_typed_same_named_symbols_dummies_and_nested_arguments_keep_coordinate_order():
    plain = s.Symbol("sparse_typed")
    positive = s.Symbol("sparse_typed", positive=True)
    left, right = s.Dummy("sparse_typed"), s.Dummy("sparse_typed")
    source = s.SparseMatrix([plain**2+2*positive, left*right, positive-plain])
    program = s.compile_residual_system(source, ((right, plain), (left, positive)))
    assert all(actual is expected for actual, expected in zip(
        program.variables, (right, plain, left, positive),
    ))
    assert program.jacobian_pattern == ((0, 1), (0, 3), (1, 0), (1, 2), (2, 1), (2, 3))
    assert program.evaluate_sparse((7, 2), (5, 3)) == ([10, 35, 1], [4, 2, 5, 7, -1, 1])
    assert program.jacobian_vector_product([11, 13, 17, 19], (7, 2), (5, 3)) == [90, 174, 6]
    assert program.jacobian_transpose_vector_product([2, 3, 5], (7, 2), (5, 3)) == [15, 3, 21, 9]
    with pytest.raises(TypeError):
        program.evaluate_sparse(7, 2, 5, 3)
    with pytest.raises(TypeError):
        program.jacobian_vector_product([[11, 13], [17, 19]], (7, 2), (5, 3))


def test_sparse_source_layout_and_transport_are_immutable_snapshots():
    x, y, z = s.symbols("sparse_snapshot_x sparse_snapshot_y sparse_snapshot_z")
    source = s.SparseMatrix([x*y, x+y])
    expected_source = source.as_immutable()
    arguments = [x, y]
    program = s.compile_residual_system(source, arguments)
    encoded = program.to_bytes()
    source[0, 0] = 999
    arguments[1] = z
    assert program.variables == (x, y)
    assert program.jacobian_pattern == ((0, 0), (0, 1), (1, 0), (1, 1))
    assert program.evaluate_sparse(2, 3) == ([6, 5], [3, 2, 1, 1])
    restored = s.load_residual_system(encoded, residuals=expected_source, args=[(x, y)])
    assert restored.to_bytes() == encoded
    assert restored.jacobian_pattern == program.jacobian_pattern
    assert restored.evaluate_sparse((2, 3)) == ([6, 5], [3, 2, 1, 1])
    assert restored.jacobian_vector_product([5, 7], (2, 3)) == [29, 12]
    assert restored.jacobian_transpose_vector_product([5, 7], (2, 3)) == [22, 17]
    with pytest.raises(ValueError):
        s.load_residual_system(encoded, residuals=source, args=(x, y))


def test_restored_products_require_full_typed_source_binding():
    plain = s.Symbol("sparse_transport_same_name")
    positive = s.Symbol("sparse_transport_same_name", positive=True)
    source = [plain**2, 3*positive]
    program = s.compile_residual_system(source, (plain, positive))
    encoded = program.to_bytes()
    restored = s.load_residual_system(encoded, residuals=source, args=(plain, positive))
    assert restored.jacobian_vector_product([5, 7], 2, 3) == [20, 21]
    assert restored.jacobian_transpose_vector_product([5, 7], 2, 3) == [20, 21]
    with pytest.raises(ValueError):
        s.load_residual_system(encoded, residuals=source, args=(positive, plain))
    with pytest.raises(ValueError):
        s.load_residual_system(encoded, residuals=[positive**2, 3*plain], args=(plain, positive))


def test_empty_residual_and_variable_axes_preserve_product_dimensions():
    x, y = s.symbols("sparse_empty_x sparse_empty_y")
    no_rows = s.compile_residual_system([], (x, y))
    assert no_rows.jacobian_pattern == ()
    assert no_rows.evaluate_sparse(2, 3) == ([], [])
    assert no_rows.jacobian_vector_product([5, 7], 2, 3) == []
    assert no_rows.jacobian_transpose_vector_product([], 2, 3) == [0, 0]
    no_columns = s.compile_residual_system([2, 3], ())
    assert no_columns.evaluate_sparse() == ([2, 3], [])
    assert no_columns.jacobian_vector_product([]) == [0, 0]
    assert no_columns.jacobian_transpose_vector_product([5, 7]) == []
    empty = s.compile_residual_system([], ())
    assert empty.evaluate_sparse() == ([], [])
    assert empty.jacobian_vector_product([]) == []
    assert empty.jacobian_transpose_vector_product([]) == []


def test_transpose_weights_follow_residual_count_beyond_the_variable_limit():
    x = s.Symbol("sparse_many_rows")
    row_count = 300
    program = s.compile_residual_system([x]*row_count, x)
    assert program.jacobian_pattern == tuple((row, 0) for row in range(row_count))
    assert program.evaluate_sparse(2) == ([2]*row_count, [1]*row_count)
    assert program.jacobian_vector_product([7], 2) == [7]*row_count
    assert program.jacobian_transpose_vector_product(list(range(1, row_count+1)), 2) == [45150]
    with pytest.raises(TypeError):
        program.jacobian_transpose_vector_product([1]*(row_count-1), 2)


@pytest.mark.parametrize("operation", ["sparse", "forward", "transpose"])
def test_original_domains_are_checked_even_with_zero_derivatives_and_zero_weights(operation):
    x = s.Symbol("sparse_held_domain")
    source = s.Mul(0, s.log(x), evaluate=False)
    program = s.compile_residual_system([source], x)
    assert program.jacobian_pattern == ()
    calls = {
        "sparse": lambda value: program.evaluate_sparse(value),
        "forward": lambda value: program.jacobian_vector_product([0], value),
        "transpose": lambda value: program.jacobian_transpose_vector_product([0], value),
    }
    for point in (0, -1):
        with pytest.raises(ValueError):
            calls[operation](point)
    assert calls[operation](1) == (([0], []) if operation == "sparse" else [0])


@pytest.mark.parametrize("operation", ["sparse", "forward", "transpose"])
def test_zero_weights_do_not_hide_a_singular_derivative_of_a_valid_residual(operation):
    x, y = s.symbols("sparse_singular_x sparse_singular_y")
    program = s.compile_residual_system([x, s.sqrt(y)], (x, y))
    assert program.residuals(2, 0) == [2, 0]
    calls = {
        "sparse": lambda: program.evaluate_sparse(2, 0),
        "forward": lambda: program.jacobian_vector_product([1, 0], 2, 0),
        "transpose": lambda: program.jacobian_transpose_vector_product([1, 0], 2, 0),
    }
    with pytest.raises(ValueError):
        calls[operation]()
    assert program.evaluate_sparse(2, 4) == ([2, 2], [1, 0.25])


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("overflow", ["multiply", "accumulate"])
def test_products_reject_nonfinite_multiplication_and_accumulation(transpose, overflow):
    x, y = s.symbols("sparse_overflow_x sparse_overflow_y")
    source = [2*x, 2*y] if overflow == "multiply" else [x+y, x+y]
    program = s.compile_residual_system(source, (x, y))
    method = (program.jacobian_transpose_vector_product if transpose
              else program.jacobian_vector_product)
    with pytest.raises(ValueError):
        method([1e308, 1e308], 0, 0)
    assert method([1, 1], 0, 0) == [2, 2]


@pytest.mark.parametrize("nonfinite", [math.inf, -math.inf, math.nan])
def test_empty_sparsity_does_not_bypass_finite_input_and_weight_admission(nonfinite):
    x = s.Symbol("sparse_constant_input")
    program = s.compile_residual_system([7], x)
    with pytest.raises(ValueError):
        program.evaluate_sparse(nonfinite)
    with pytest.raises(ValueError):
        program.jacobian_vector_product([nonfinite], 2)
    with pytest.raises(ValueError):
        program.jacobian_transpose_vector_product([nonfinite], 2)


def test_product_vectors_refuse_bad_shapes_and_conversion_hooks_before_evaluation():
    program, point = _rectangular_system()
    effects = []

    class Hostile:
        def __float__(self):
            effects.append("float")
            raise AssertionError("unadmitted real-number conversion")

    class CustomList(list):
        def __iter__(self):
            effects.append("iterate")
            raise AssertionError("unadmitted vector container")

    for method, size in ((program.jacobian_vector_product, 4),
                         (program.jacobian_transpose_vector_product, 3)):
        bad_vectors = ([], [1]*(size-1), [1]*(size+1), "1", iter([1]*size),
                       CustomList([1]*size), [Hostile()]*size, ["1"]*size, [1j]*size)
        for vector in bad_vectors:
            with pytest.raises(TypeError):
                method(vector, *point)
    with pytest.raises(TypeError):
        program.evaluate_sparse(Hostile(), 3, 5, 11)
    assert effects == []


@pytest.mark.parametrize("operation", ["sparse", "forward", "transpose"])
def test_shared_budgets_charge_completed_work_and_refuse_late_exhaustion(operation):
    program, point = _rectangular_system()
    probe = s.EvaluationBudget(max_steps=1_000_000)
    expected = _operation(program, operation, point, budget=probe)
    cost = 1_000_000-probe.remaining_steps
    assert cost > 1
    exact = s.EvaluationBudget(max_steps=cost)
    assert _operation(program, operation, point, budget=exact) == expected
    assert exact.remaining_steps == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        _operation(program, operation, point, budget=exact)
    assert exact.remaining_steps == 0
    short = s.EvaluationBudget(max_steps=cost-1)
    with pytest.raises(s.EvaluationBudgetExceeded):
        _operation(program, operation, point, budget=short)
    assert short.remaining_steps < cost-1
    assert _operation(program, operation, point) == expected


def test_sparse_and_both_product_methods_share_one_ledger_without_reset():
    program, point = _rectangular_system()
    costs = []
    for name in ("sparse", "forward", "transpose"):
        probe = s.EvaluationBudget(max_steps=1_000_000)
        _operation(program, name, point, budget=probe)
        costs.append(1_000_000-probe.remaining_steps)
    shared = s.EvaluationBudget(max_steps=sum(costs))
    for name, cost in zip(("sparse", "forward", "transpose"), costs):
        before = shared.remaining_steps
        _operation(program, name, point, budget=shared)
        assert shared.remaining_steps == before-cost
    assert shared.remaining_steps == 0
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.evaluate_sparse(*point, budget=shared)


@pytest.mark.parametrize("operation", ["sparse", "forward", "transpose"])
@pytest.mark.parametrize("limit", ["max_memory_bytes", "max_allocations"])
def test_input_storage_refusal_is_atomic_and_precedes_domain_work(operation, limit):
    program, _ = _rectangular_system()
    budget = s.EvaluationBudget(max_steps=1000, **{limit: 0})
    before = (budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations)
    with pytest.raises(s.EvaluationBudgetExceeded):
        _operation(program, operation, (1e308, 3, 5, 11), budget=budget)
    assert (budget.remaining_steps, budget.remaining_memory_bytes, budget.remaining_allocations) == before


def test_derivative_domain_failure_retains_spent_work_and_allows_later_valid_use():
    x = s.Symbol("sparse_failed_work")
    program = s.compile_residual_system([s.sqrt(x)], x)
    budget = s.EvaluationBudget(max_steps=100_000)
    with pytest.raises(ValueError):
        program.jacobian_vector_product([0], 0, budget=budget)
    after_failure = budget.remaining_steps
    assert 0 < after_failure < 100_000
    assert program.jacobian_transpose_vector_product([2], 4, budget=budget) == [0.5]
    assert budget.remaining_steps < after_failure


def test_cancelled_budgets_cover_zero_pattern_and_empty_outputs():
    x = s.Symbol("sparse_cancelled")
    constant = s.compile_residual_system([7], x)
    empty = s.compile_residual_system([], ())
    budget = s.EvaluationBudget()
    budget.cancel()
    for call in (
        lambda: constant.evaluate_sparse(2, budget=budget),
        lambda: constant.jacobian_vector_product([0], 2, budget=budget),
        lambda: constant.jacobian_transpose_vector_product([0], 2, budget=budget),
        lambda: empty.evaluate_sparse(budget=budget),
        lambda: empty.jacobian_vector_product([], budget=budget),
        lambda: empty.jacobian_transpose_vector_product([], budget=budget),
    ):
        with pytest.raises(s.EvaluationCancelled):
            call()
    assert budget.cancelled


def test_public_results_and_pattern_cannot_modify_the_admitted_system():
    program, point = _rectangular_system()
    original = program.jacobian_pattern
    assert type(original) is tuple and all(type(coordinate) is tuple for coordinate in original)
    with pytest.raises(AttributeError):
        program.jacobian_pattern = ()
    residuals, sparse = program.evaluate_sparse(*point)
    residuals[0] = sparse[0] = 999
    forward = program.jacobian_vector_product([2, -3, 7, 99], *point)
    transpose = program.jacobian_transpose_vector_product([2, 11, -3], *point)
    forward[0] = transpose[0] = 999
    assert program.jacobian_pattern == original
    assert program.evaluate_sparse(*point) == ([7, 7, 13], [4, 1, -1, 5, 3])
    assert program.jacobian_vector_product([2, -3, 7, 99], *point) == [5, 0, 4]
    assert program.jacobian_transpose_vector_product([2, 11, -3], *point) == [11, -13, -9, 0]


def test_sparse_evaluation_avoids_the_dense_output_allocation_budget():
    # This checks logical allocation accounting, not execution-time performance.
    variables = tuple(s.Symbol("sparse_diagonal_"+str(index)) for index in range(32))
    program = s.compile_residual_system([variable**2 for variable in variables], variables)
    point = (2,)*len(variables)
    memory = 16*1024*1024
    sparse_budget = s.EvaluationBudget(max_memory_bytes=memory)
    assert program.evaluate_sparse(*point, budget=sparse_budget) == ([4]*32, [4]*32)
    sparse_bytes = memory-sparse_budget.remaining_memory_bytes
    dense_budget = s.EvaluationBudget(max_memory_bytes=memory)
    program.evaluate(*point, budget=dense_budget)
    dense_bytes = memory-dense_budget.remaining_memory_bytes
    assert sparse_bytes < dense_bytes
    admitted = s.EvaluationBudget(max_memory_bytes=sparse_bytes)
    assert program.evaluate_sparse(*point, budget=admitted) == ([4]*32, [4]*32)
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.evaluate(*point, budget=s.EvaluationBudget(max_memory_bytes=sparse_bytes))
