"""Sparse user-supplied Jacobians through real supervised callback requests."""

import math
import pickle

import pytest
import sympy as s

from sympy.core import SupervisedCallbackSession, _native_expr
from sympy.utilities.hybrid_system import compile_hybrid_residual_system


def _budget(**changes):
    options = dict(max_steps=1 << 30, max_memory_bytes=1 << 30, max_allocations=1 << 20)
    options.update(changes)
    return s.EvaluationBudget(**options)


def _session(**changes):
    options = dict(max_invocations=4096, max_nested_steps=20_000_000)
    options.update(changes)
    return SupervisedCallbackSession(**options)


def _call(program, operation, point, *, vector=None, budget=None, session=None):
    if operation == "sparse":
        return program.evaluate_sparse(*point, budget=budget, session=session)
    if vector is None:
        count = program.variable_count if operation == "forward" else program.residual_count
        vector = [0] * count
    method = (program.jacobian_vector_product if operation == "forward"
              else program.jacobian_transpose_vector_product)
    return method(vector, *point, budget=budget, session=session)


def _rectangular():
    x, y = s.symbols("hybrid_sparse_x hybrid_sparse_y")
    events = []

    class Law(s.Function):
        def _eval_derivative(self, variable):
            raise AssertionError("an asserted Jacobian must not invoke derivative hooks")

    class Slope(s.Function):
        pass

    def law(value):
        events.append(("law", value))
        return value * value

    def slope(value):
        events.append(("slope", value))
        return 2 * value

    residuals = s.SparseMatrix([
        s.Add(Law(x, evaluate=False), y, -5, evaluate=False), 3*y-3, 7,
    ])
    jacobian = s.SparseMatrix([[Slope(x, evaluate=False), 1], [0, 3], [0, 0]])
    program = compile_hybrid_residual_system(
        residuals, (x, y), jacobian=jacobian, callbacks={Law: law, Slope: slope},
    )
    return program, events


def test_rectangular_sparse_products_and_legacy_dense_calls_share_the_captured_graph():
    program, events = _rectangular()
    active = _session()
    assert program.jacobian_pattern == ((0, 0), (0, 1), (1, 1))
    assert program.evaluate_sparse(2, 1, session=active) == ([0, 0, 7], [4, 1, 3])
    assert program.jacobian_vector_product([5, 7], 2, 1, session=active) == [27, 21, 0]
    assert program.jacobian_transpose_vector_product([2, 3, 11], 2, 1, session=active) == [8, 11]
    assert program(2, 1, session=active) == ([0, 0, 7], [[4, 1], [0, 3], [0, 0]])
    assert program.jacobian(2, 1, session=active) == [[4, 1], [0, 3], [0, 0]]
    assert events == [("law", 2.0), ("slope", 2.0)] * 5
    assert program.residuals(3, 2, session=active) == [6, 3, 7]
    assert events[-1] == ("law", 3.0)
    assert program.evaluate_batch([(2, 1), (3, 2)], session=active) == [
        ([0, 0, 7], [[4, 1], [0, 3], [0, 0]]),
        ([6, 3, 7], [[6, 1], [0, 3], [0, 0]]),
    ]
    assert active.invocations == 15 and active.cached_effect_count() == 0
    assert program.evaluate_batch([], session=active) == []
    assert program.variable_count == 2 and program.residual_count == 3
    assert program.evidence == "approximate_with_user_callbacks"
    assert program.jacobian_evidence == "user_supplied"
    for name in ("native_system", "proofs_verified", "enclose", "certify_root_box",
                 "solve_certified", "to_bytes"):
        assert not hasattr(program, name)
    with pytest.raises(TypeError):
        pickle.dumps(program)
    with pytest.raises(AttributeError):
        program.jacobian_pattern = ()
    residuals, entries = program.evaluate_sparse(2, 1)
    residuals[0], entries[0] = 999, 999
    assert program.evaluate_sparse(2, 1) == ([0, 0, 7], [4, 1, 3])


def test_sparse_sources_layout_and_callback_mapping_are_captured_before_later_mutation():
    x, y, z = s.symbols("hybrid_sparse_snapshot_x hybrid_sparse_snapshot_y hybrid_sparse_snapshot_z")
    events = []

    class Law(s.Function):
        pass

    class Slope(s.Function):
        pass

    residuals = s.SparseMatrix([Law(x, evaluate=False), y])
    jacobian = s.SparseMatrix([[Slope(x, evaluate=False), 0], [0, 1]])
    layout = [(x, y)]
    callbacks = {
        Law: lambda value: events.append("law") or value*value,
        Slope: lambda value: events.append("slope") or 2*value,
    }
    program = compile_hybrid_residual_system(residuals, layout, jacobian=jacobian, callbacks=callbacks)
    residuals[0, 0], jacobian[0, 0] = 999, 999
    layout[0] = (z, x)
    callbacks[Law] = callbacks[Slope] = lambda value: 999
    assert program.variables[0] is x and program.variables[1] is y
    assert program.jacobian_pattern == ((0, 0), (1, 1))
    assert program.evaluate_sparse((2, 3)) == ([4, 3], [4, 1])
    assert program.jacobian_vector_product([5, 7], (2, 3)) == [20, 7]
    assert program.jacobian_transpose_vector_product([5, 7], (2, 3)) == [20, 7]
    assert events == ["law", "slope"] * 3


@pytest.mark.parametrize("operation", ["sparse", "forward", "transpose"])
def test_same_named_callback_classes_remain_distinct_in_residual_and_sparse_derivative_slots(operation):
    x, y = s.symbols("hybrid_sparse_same_x hybrid_sparse_same_y")
    first = type("HybridSparseSame", (s.Function,), {})
    second = type("HybridSparseSame", (s.Function,), {})
    events = []

    def left(value):
        events.append(("first", value))
        return 2 * value

    def right(value):
        events.append(("second", value))
        return value + 10

    program = compile_hybrid_residual_system(
        [first(x, evaluate=False), second(x, evaluate=False)], (x, y),
        jacobian=[[first(x, evaluate=False), 0], [0, second(y, evaluate=False)]],
        callbacks={first: left, second: right},
    )
    assert first is not second and first.__name__ == second.__name__
    expected = {"sparse": ([4, 12], [4, 13]), "forward": [20, 91], "transpose": [20, 91]}
    assert _call(program, operation, (2, 3), vector=[5, 7]) == expected[operation]
    assert events == [("first", 2.0), ("second", 2.0), ("first", 2.0), ("second", 3.0)]


def test_repeated_occurrences_stay_fresh_and_ordered_across_sparse_and_product_calls():
    x, y = s.symbols("hybrid_sparse_occurrence_x hybrid_sparse_occurrence_y")
    seen = []

    class Law(s.Function):
        pass

    def callback(value):
        seen.append(value)
        return len(seen)

    node = Law(x, evaluate=False)
    program = compile_hybrid_residual_system(
        [node, node], (x, y), jacobian=[[node, 0], [node, 0]], callbacks={Law: callback},
    )
    active = _session()
    assert program.evaluate_sparse(2, 3, session=active) == ([1, 2], [3, 4])
    assert program.jacobian_vector_product([2, 0], 2, 3, session=active) == [14, 16]
    assert program.jacobian_transpose_vector_product([3, 5], 2, 3, session=active) == [93, 0]
    assert seen == [2.0] * 12
    assert active.invocations == 12 and active.cached_effect_count() == 0
    assert len({entry[2] for entry in active.trace() if entry[3] == "ok"}) == 12


def test_only_original_exact_integer_zero_atoms_are_omitted():
    variables = s.symbols("hz0 hz1 hz2 hz3 hz4 hz5 hz6")
    seen = []

    class Law(s.Function):
        pass

    entries = [0, s.S.Zero, _native_expr(s.Integer(0)), False,
               0.0, s.Float(0), _native_expr(s.Float(0))]
    program = compile_hybrid_residual_system(
        [Law(variables[0], evaluate=False)], variables, jacobian=[entries],
        callbacks={Law: lambda value: seen.append(value) or value},
    )
    assert program.jacobian_pattern == ((0, 4), (0, 5), (0, 6))
    assert program.evaluate_sparse(*range(7)) == ([0], [0, 0, 0])
    assert program.jacobian_vector_product([0]*7, *range(7)) == [0]
    assert program.jacobian_transpose_vector_product([0], *range(7)) == [0]*7
    assert seen == [0.0] * 3


@pytest.mark.parametrize("operation", ["sparse", "forward", "transpose"])
def test_held_and_callback_zeros_preserve_pattern_and_every_effect_at_zero_weights(operation):
    x, y, z = s.symbols("hybrid_sparse_zero_x hybrid_sparse_zero_y hybrid_sparse_zero_z")
    seen = []

    class Law(s.Function):
        pass

    node = Law(x, evaluate=False)
    program = compile_hybrid_residual_system(
        [node, z], (x, y, z),
        jacobian=[[0, s.S.Zero, s.Float(0)],
                  [s.Mul(0, node, evaluate=False), Law(y, evaluate=False), 0.0]],
        callbacks={Law: lambda value: seen.append(value) or 0},
    )
    assert program.jacobian_pattern == ((0, 2), (1, 0), (1, 1), (1, 2))
    expected = {"sparse": ([0, 5], [0, 0, 0, 0]), "forward": [0, 0], "transpose": [0, 0, 0]}
    assert _call(program, operation, (2, 3, 5)) == expected[operation]
    assert seen == [2.0, 2.0, 3.0]


def test_zero_classification_does_not_run_opaque_properties_equality_or_printers():
    x = s.Symbol("hybrid_sparse_guard")
    effects = []

    class Guard(s.Function):
        @property
        def args(self):
            raise AssertionError("read the owning stored argument slot")

        def __str__(self):
            raise AssertionError("printers do not classify sparse zeros")

        def __eq__(self, other):
            raise AssertionError("opaque equality does not classify sparse zeros")

        def __hash__(self):
            raise AssertionError("opaque hashes do not classify sparse zeros")

        def __float__(self):
            raise AssertionError("opaque objects require their captured callback")

    node = Guard(x, evaluate=False)
    program = compile_hybrid_residual_system(
        [node], x, jacobian=[[node]],
        callbacks={Guard: lambda value: effects.append(value) or 0},
    )
    assert program.jacobian_pattern == ((0, 0),)
    assert program.evaluate_sparse(2) == ([0], [0])
    assert effects == [2.0, 2.0]


def test_typed_same_named_coordinates_and_nested_arguments_keep_flat_product_order():
    plain = s.Symbol("hybrid_sparse_typed")
    positive = s.Symbol("hybrid_sparse_typed", positive=True)
    left, right = s.Dummy("hybrid_sparse_typed"), s.Dummy("hybrid_sparse_typed")

    class Law(s.Function):
        pass

    class Slope(s.Function):
        pass

    program = compile_hybrid_residual_system(
        [s.Add(Law(plain, evaluate=False), 2*positive, evaluate=False), left*right],
        ((right, plain), (left, positive)),
        jacobian=[[0, Slope(plain, evaluate=False), 0, 2], [left, 0, right, 0]],
        callbacks={Law: lambda value: value*value, Slope: lambda value: 2*value},
    )
    assert all(actual is expected for actual, expected in zip(
        program.variables, (right, plain, left, positive),
    ))
    assert program.jacobian_pattern == ((0, 1), (0, 3), (1, 0), (1, 2))
    assert program.evaluate_sparse((7, 2), (5, 3)) == ([10, 35], [4, 2, 5, 7])
    assert program.jacobian_vector_product([11, 13, 17, 19], (7, 2), (5, 3)) == [90, 174]
    assert program.jacobian_transpose_vector_product([2, 3], (7, 2), (5, 3)) == [15, 8, 21, 4]
    with pytest.raises(TypeError):
        program.evaluate_sparse(7, 2, 5, 3)
    with pytest.raises(TypeError):
        program.jacobian_vector_product([[11, 13], [17, 19]], (7, 2), (5, 3))


@pytest.mark.parametrize("operation", ["sparse", "forward", "transpose"])
@pytest.mark.parametrize("location", ["residual", "derivative"])
def test_held_native_domains_are_checked_even_with_zero_products(operation, location):
    x = s.Symbol("hybrid_sparse_held_domain")
    held = s.Mul(0, s.log(x), evaluate=False)
    residuals, jacobian = ([held], [[0]]) if location == "residual" else ([x], [[held]])
    program = compile_hybrid_residual_system(residuals, x, jacobian=jacobian, callbacks={})
    assert program.jacobian_pattern == (() if location == "residual" else ((0, 0),))
    for point in (0, -1):
        with pytest.raises(ValueError):
            _call(program, operation, (point,))
    sparse = ([0], []) if location == "residual" else ([1], [0])
    assert _call(program, operation, (1,)) == (sparse if operation == "sparse" else [0])


@pytest.mark.parametrize("operation", ["sparse", "forward", "transpose"])
def test_valid_residual_effect_precedes_a_singular_derivative_even_at_zero_weights(operation):
    x = s.Symbol("hybrid_sparse_singular")
    seen = []

    class Law(s.Function):
        pass

    program = compile_hybrid_residual_system(
        [Law(x, evaluate=False)], x, jacobian=[[1/(2*s.sqrt(x))]],
        callbacks={Law: lambda value: seen.append(value) or math.sqrt(value)},
    )
    active = _session()
    with pytest.raises(ValueError):
        _call(program, operation, (0,), session=active)
    assert seen == [0.0] and active.invocations == 1
    assert not active.unknown_effect_tainted
    assert program.evaluate_sparse(4, session=active) == ([2], [0.25])
    assert seen == [0.0, 4.0]


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("overflow", ["multiply", "accumulate"])
def test_all_callbacks_finish_once_before_product_overflow_refuses_publication(transpose, overflow):
    x, y = s.symbols("hybrid_sparse_overflow_x hybrid_sparse_overflow_y")
    seen = []

    class Law(s.Function):
        pass

    class Slope(s.Function):
        pass

    program = compile_hybrid_residual_system(
        [Law(x, evaluate=False), Law(y, evaluate=False)], (x, y),
        jacobian=[[Slope(x, evaluate=False), Slope(y, evaluate=False)]] * 2,
        callbacks={
            Law: lambda value: seen.append(("r", value)) or 0,
            Slope: lambda value: seen.append(("j", value)) or 1e308,
        },
    )
    vector = [2, 0] if overflow == "multiply" else [1, 1]
    method = program.jacobian_transpose_vector_product if transpose else program.jacobian_vector_product
    active = _session()
    with pytest.raises(ValueError, match="finite"):
        method(vector, 2, 3, session=active)
    ordered = [("r", 2.0), ("r", 3.0), ("j", 2.0), ("j", 3.0), ("j", 2.0), ("j", 3.0)]
    assert seen == ordered and active.invocations == 6
    assert not active.unknown_effect_tainted and active.cached_effect_count() == 0
    assert program.evaluate_sparse(2, 3, session=active) == ([0, 0], [1e308]*4)
    assert seen == ordered * 2


@pytest.mark.parametrize("operation", ["forward", "transpose"])
def test_vectors_are_fully_admitted_before_any_callback(operation):
    program, events = _rectangular()
    size = program.variable_count if operation == "forward" else program.residual_count

    class PretendReal:
        def __float__(self):
            raise AssertionError("user float conversion must not run")

    class PretendList(list):
        def __len__(self):
            raise AssertionError("container subclasses must not be inspected")

    invalid = [[], [1]*(size+1), (value for value in range(size)),
               PretendList([1]*size), [PretendReal()]*size,
               ["1"]*size, [1j]*size, [[1]]*size]
    for vector in invalid:
        with pytest.raises(TypeError):
            _call(program, operation, (2, 1), vector=vector)
    for value in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError):
            _call(program, operation, (2, 1), vector=[value]*size)
    assert events == []


@pytest.mark.parametrize("operation", ["sparse", "forward", "transpose"])
def test_nested_sparse_requests_inherit_the_same_session_and_cumulative_ledger(operation):
    x = s.Symbol("hybrid_sparse_nested")
    events = []

    class Inner(s.Function):
        pass

    class Outer(s.Function):
        pass

    inner = compile_hybrid_residual_system(
        [Inner(x, evaluate=False)], x, jacobian=[[2*x]],
        callbacks={Inner: lambda value: events.append("inner") or value*value},
    )

    def inner_value(value):
        result = _call(inner, operation, (value,), vector=[1])
        return result[1][0] if operation == "sparse" else result[0]

    outer = s.compile_numeric_hybrid(
        x, Outer(x, evaluate=False), callbacks={Outer: lambda value: inner_value(value)+value},
    )
    measuring = _budget()
    _call(inner, operation, (3,), vector=[1], budget=measuring, session=_session())
    inner_cost = (1 << 30) - measuring.remaining_steps
    events.clear()
    active, shared = _session(), _budget()
    assert outer(3, budget=shared, session=active) == 9
    assert events == ["inner"] and active.invocations == 2
    assert active.cached_effect_count() == 0
    assert (1 << 30) - shared.remaining_steps > inner_cost


@pytest.mark.parametrize("replace", ["budget", "session"])
def test_nested_product_requests_cannot_replace_the_active_owner(replace):
    x = s.Symbol("hybrid_sparse_replacement")
    events = []

    class Inner(s.Function):
        pass

    class Outer(s.Function):
        pass

    inner = compile_hybrid_residual_system(
        [Inner(x, evaluate=False)], x, jacobian=[[1]],
        callbacks={Inner: lambda value: events.append(value) or value},
    )
    replacement = _budget() if replace == "budget" else _session()

    def callback(value):
        return inner.jacobian_vector_product([1], value, **{replace: replacement})[0]

    outer = s.compile_numeric_hybrid(x, Outer(x, evaluate=False), callbacks={Outer: callback})
    active = _session()
    match = "active EvaluationBudget" if replace == "budget" else "conflicts"
    with pytest.raises(ValueError, match=match):
        outer(3, budget=_budget(), session=active)
    assert events == [] and active.unknown_effect_tainted
    if replace == "budget":
        assert replacement.remaining_steps == 1 << 30
    else:
        assert replacement.invocations == 0


@pytest.mark.parametrize("operation", ["sparse", "forward", "transpose"])
def test_callback_exceptions_keep_the_original_object_and_taint_the_owning_request(operation):
    x = s.Symbol("hybrid_sparse_exception")
    seen = []

    class Law(s.Function):
        pass

    class Failure(ValueError):
        def __str__(self):
            raise AssertionError("do not format a callback exception")

    marker = Failure()

    def callback(value):
        seen.append(value)
        raise marker

    program = compile_hybrid_residual_system(
        [Law(x, evaluate=False)], x, jacobian=[[0]], callbacks={Law: callback},
    )
    active = _session()
    with pytest.raises(Failure) as caught:
        _call(program, operation, (2,), session=active)
    assert caught.value is marker and seen == [2.0]
    assert active.unknown_effect_tainted and active.invocations == 1
    with pytest.raises(RuntimeError, match="unknown effect"):
        _call(program, operation, (2,), session=active)
    assert seen == [2.0]


@pytest.mark.parametrize("operation", ["sparse", "forward", "transpose"])
def test_cancellation_inside_a_derivative_callback_never_publishes_a_sparse_result(operation):
    x = s.Symbol("hybrid_sparse_cancel")
    seen, ledger = [], _budget()

    class Law(s.Function):
        pass

    class Slope(s.Function):
        pass

    def cancel(value):
        seen.append("slope")
        ledger.cancel()
        return 1

    program = compile_hybrid_residual_system(
        [Law(x, evaluate=False)], x, jacobian=[[Slope(x, evaluate=False)]],
        callbacks={Law: lambda value: seen.append("law") or value, Slope: cancel},
    )
    active = _session()
    with pytest.raises(s.EvaluationCancelled):
        _call(program, operation, (2,), budget=ledger, session=active)
    assert seen == ["law", "slope"]
    assert active.invocations == 2 and active.unknown_effect_tainted


@pytest.mark.parametrize("operation", ["sparse", "forward", "transpose"])
@pytest.mark.parametrize("which", ["budget", "session"])
def test_pre_cancelled_sparse_requests_run_no_effects(operation, which):
    program, events = _rectangular()
    active, ledger = _session(), _budget()
    (ledger if which == "budget" else active).cancel()
    error = s.EvaluationCancelled if which == "budget" else RuntimeError
    with pytest.raises(error):
        _call(program, operation, (2, 1), budget=ledger, session=active)
    assert events == [] and active.invocations == 0


def test_one_budget_spans_sparse_evaluation_and_both_products_without_resetting():
    program, events = _rectangular()
    operations = ("sparse", "forward", "transpose")
    costs = []
    for operation in operations:
        measuring = _budget()
        _call(program, operation, (2, 1), budget=measuring, session=_session())
        costs.append((1 << 30) - measuring.remaining_steps)
    assert all(cost > 0 for cost in costs)
    events.clear()
    shared, active = _budget(max_steps=sum(costs)), _session()
    for operation in operations:
        _call(program, operation, (2, 1), budget=shared, session=active)
    assert shared.remaining_steps == 0
    assert events == [("law", 2.0), ("slope", 2.0)] * 3
    with pytest.raises(s.EvaluationBudgetExceeded):
        program.evaluate_sparse(2, 1, budget=shared, session=active)
    assert active.invocations == 6


@pytest.mark.parametrize("operation", ["forward", "transpose"])
def test_late_product_budget_failure_preserves_completed_effects_without_partial_values(operation):
    program, events = _rectangular()
    measuring = _budget()
    _call(program, operation, (2, 1), budget=measuring, session=_session())
    cost = (1 << 30) - measuring.remaining_steps
    events.clear()
    active = _session()
    with pytest.raises(s.EvaluationBudgetExceeded):
        _call(program, operation, (2, 1), budget=_budget(max_steps=cost-1), session=active)
    assert events == [("law", 2.0), ("slope", 2.0)]
    assert active.invocations == 2 and not active.unknown_effect_tainted


@pytest.mark.parametrize("operation", ["sparse", "forward", "transpose"])
def test_sparse_requests_keep_cumulative_callback_invocation_limits(operation):
    program, events = _rectangular()
    active = _session(max_invocations=1)
    with pytest.raises(ValueError, match="invocation count"):
        _call(program, operation, (2, 1), session=active)
    assert events == [("law", 2.0)]
    assert active.invocations == 1 and active.unknown_effect_tainted


def test_dense_solver_still_uses_the_explicit_compact_jacobian_without_symbolic_hooks():
    x, y = s.symbols("hybrid_sparse_solve_x hybrid_sparse_solve_y")
    events = []

    class Law(s.Function):
        def _eval_derivative(self, variable):
            raise AssertionError("solving does not prove the asserted Jacobian")

    program = compile_hybrid_residual_system(
        s.SparseMatrix([s.Add(Law(x, evaluate=False), -4, evaluate=False), y-3]),
        (x, y), jacobian=s.SparseMatrix([[2*x, 0], [0, 1]]),
        callbacks={Law: lambda value: events.append(value) or value*value},
    )
    assert program.jacobian_pattern == ((0, 0), (1, 1))
    assert program.evaluate_sparse(2, 3) == ([0, 0], [4, 1])
    assert program.jacobian_vector_product([5, 7], 2, 3) == [20, 7]
    active, ledger = _session(), _budget()
    root = program.solve([1.5, 0], session=active, budget=ledger)
    assert type(root) is s.Matrix and root.shape == (2, 1)
    assert all(type(root[row, 0]) is s.Float for row in range(2))
    assert [float(root[row, 0]) for row in range(2)] == pytest.approx([2, 3], abs=1e-12)
    assert active.invocations > 0 and ledger.remaining_steps < 1 << 30
    assert program.evidence == "approximate_with_user_callbacks"
    assert program.jacobian_evidence == "user_supplied"


def test_legacy_dense_batch_caps_and_all_row_finiteness_precede_every_callback():
    from sympy.utilities.lambdify import _MAX_BATCH_ROWS, _MAX_BATCH_VALUES

    variables = s.symbols("hb0 hb1 hb2 hb3 hb4 hb5 hb6 hb7")
    events = []

    class Law(s.Function):
        pass

    rows, columns = 16, len(variables)
    program = compile_hybrid_residual_system(
        [Law(variables[0], evaluate=False)] + [0]*(rows-1), variables,
        jacobian=[[0]*columns for _ in range(rows)],
        callbacks={Law: lambda value: events.append(value) or value},
    )
    assert program.jacobian_pattern == ()
    dense_outputs = rows * (columns + 1)
    batch_count = _MAX_BATCH_VALUES // dense_outputs + 1
    compact_outputs = rows + len(program.jacobian_pattern)
    assert batch_count <= _MAX_BATCH_ROWS
    assert batch_count * max(columns, compact_outputs) <= _MAX_BATCH_VALUES
    point = tuple(range(columns))
    active = _session()
    with pytest.raises(ValueError, match="total input/output value limit"):
        program.evaluate_batch([point]*batch_count, session=active)
    for nonfinite in (math.nan, math.inf, -math.inf):
        invalid_last_row = point[:-1] + (nonfinite,)
        with pytest.raises(ValueError, match="finite"):
            program.evaluate_batch([point, invalid_last_row], session=active)
    assert events == [] and active.invocations == 0
    assert not active.unknown_effect_tainted


@pytest.mark.parametrize("origin,stored,coefficient", [
    ("int_zero", False, 0),
    ("bool_false", False, 0),
    ("canonical_zero", False, 0),
    ("native_integer_zero", False, 0),
    ("native_rational_zero", True, 0),
    ("float_zero", True, 0),
    ("surface_float_zero", True, 0),
    ("native_float_zero", True, 0),
    ("bool_true", True, 1),
    ("int_nonzero", True, 2),
])
def test_exact_atom_provenance_survives_native_numeric_lowering(origin, stored, coefficient):
    from sympy.core import _native

    x = s.Symbol("hybrid_sparse_zero_origin")
    effects = []

    class Law(s.Function):
        pass

    atoms = {
        "int_zero": lambda: 0,
        "bool_false": lambda: False,
        "canonical_zero": lambda: s.S.Zero,
        "native_integer_zero": lambda: _native_expr(s.Integer(0)),
        "native_rational_zero": lambda: _native.py_rational(0, 1),
        "float_zero": lambda: 0.0,
        "surface_float_zero": lambda: s.Float(0),
        "native_float_zero": lambda: _native_expr(s.Float(0)),
        "bool_true": lambda: True,
        "int_nonzero": lambda: 2,
    }
    program = compile_hybrid_residual_system(
        [Law(x, evaluate=False)], x, jacobian=[[atoms[origin]()]],
        callbacks={Law: lambda value: effects.append(value) or value},
    )
    assert program.jacobian_pattern == (((0, 0),) if stored else ())
    assert program.evaluate_sparse(2) == ([2], [coefficient] if stored else [])
    assert program.jacobian_vector_product([3], 2) == [3*coefficient]
    assert program.jacobian_transpose_vector_product([5], 2) == [5*coefficient]
    assert effects == [2.0, 2.0, 2.0]
