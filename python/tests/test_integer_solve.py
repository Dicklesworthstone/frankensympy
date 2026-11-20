"""Complete integer affine families and checked integer inconsistency.

This API is an explicitly native extension. Tests derive exact families and
primitive lattice membership; no production oracle or numerical rank test
can establish its integer completeness contract.
"""

from functools import reduce
from math import gcd
from unittest.mock import patch

import pytest
import sympy as s
from sympy.core import SupervisedCallbackSession, _native, supervised_callback_session
from sympy.matrices import normalforms
from sympy.matrices._integer_solve import solve_integer


CLASSES = (s.Matrix, s.ImmutableMatrix, s.SparseMatrix, s.ImmutableSparseMatrix)
AMPLE_STEPS = 10_000_000


def _zero(cls, rows, columns):
    entries = {} if cls in (s.SparseMatrix, s.ImmutableSparseMatrix) else [0] * (rows * columns)
    return cls(rows, columns, entries)


def _integer_entries(matrix):
    assert all(entry.is_integer is True for entry in matrix)


@pytest.mark.parametrize("coefficient_class", CLASSES)
@pytest.mark.parametrize("rhs_class", CLASSES)
def test_mixed_classes_preserve_coefficient_owner_and_complete_multiple_rhs_family(
        coefficient_class, rhs_class):
    source = coefficient_class([[2, 3, 5], [0, 2, 4]])
    rhs = rhs_class([[1, 2], [2, 4]])
    particular, kernel = solve_integer(source, rhs)
    assert type(particular) is type(kernel) is coefficient_class
    assert particular.shape == (3, 2)
    assert kernel.shape == (3, 1)
    _integer_entries(particular)
    _integer_entries(kernel)
    assert source * particular == rhs
    assert source * kernel == s.zeros(2, 1)
    family = particular + kernel * s.Matrix([[-3, 7]])
    assert source * family == rhs
    assert source == coefficient_class([[2, 3, 5], [0, 2, 4]])
    assert rhs == rhs_class([[1, 2], [2, 4]])


def test_matrix_method_exposes_the_checked_native_extension():
    source = s.ImmutableSparseMatrix([[2, 3]])
    rhs = s.ImmutableMatrix([[1]])
    assert source.solve_integer(rhs) == solve_integer(source, rhs)


@pytest.mark.parametrize("cls", CLASSES)
def test_rationally_solvable_parity_obstruction_returns_no_integer_solution(cls):
    source = cls([[2]])
    rhs = cls([[1]])
    assert source.inv() * rhs == s.Matrix([[s.Rational(1, 2)]])
    assert solve_integer(source, rhs) is None


@pytest.mark.parametrize("cls", CLASSES)
def test_one_inconsistent_rhs_column_refuses_the_entire_integer_system(cls):
    assert solve_integer(cls([[2, 4]]), cls([[2, 1]])) is None
    assert solve_integer(cls([[1, 0], [0, 0]]), cls([[0], [1]])) is None


def test_overdetermined_unique_integer_solution_has_an_empty_kernel():
    source = s.Matrix([[1, 0], [0, 1], [1, 1]])
    rhs = s.Matrix([[2], [3], [5]])
    particular, kernel = solve_integer(source, rhs)
    assert particular == s.Matrix([[2], [3]])
    assert kernel.shape == (2, 0)
    assert source * particular == rhs


@pytest.mark.parametrize("cls", CLASSES)
def test_primitive_kernel_contains_vectors_missed_by_scaled_rational_nullspace(cls):
    source = cls([[2, 3, 5]])
    particular, kernel = solve_integer(source, cls([[1]]))
    assert source * particular == s.Matrix([[1]])
    assert kernel.shape == (3, 2)
    assert source * kernel == s.zeros(1, 2)
    # A tempting scaled rational basis (-3,2,0), (-5,0,2) misses this
    # lattice vector because its last two coordinates are odd.
    target = s.Matrix([[-1], [-1], [1]])
    coordinates, parameters = s.Matrix(kernel).gauss_jordan_solve(target)
    assert parameters.rows == 0
    _integer_entries(coordinates)
    assert kernel * coordinates == target
    # Full column-rank integer generators form a primitive sublattice iff
    # their maximal minors have gcd one. This also excludes a finite-index
    # sublattice that merely satisfies A*K=0.
    minors = [int(kernel[i, 0] * kernel[j, 1] - kernel[i, 1] * kernel[j, 0])
              for i in range(3) for j in range(i + 1, 3)]
    assert reduce(gcd, map(abs, minors)) == 1


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("shape", [(2, 3, 0), (0, 3, 2), (0, 0, 0),
                                    (2, 0, 0), (2, 0, 3), (0, 0, 4)])
def test_empty_shapes_and_zero_rhs_columns_retain_the_complete_kernel(cls, shape):
    rows, columns, rhs_columns = shape
    source = _zero(cls, rows, columns)
    rhs = _zero(cls, rows, rhs_columns)
    particular, kernel = solve_integer(source, rhs)
    assert type(particular) is type(kernel) is cls
    assert particular.shape == (columns, rhs_columns)
    assert kernel.shape == (columns, columns)
    assert source * particular == rhs
    if columns:
        assert abs(kernel.det()) == 1


@pytest.mark.parametrize("cls", CLASSES)
def test_no_unknowns_cannot_satisfy_a_nonzero_rhs(cls):
    assert solve_integer(_zero(cls, 2, 0), cls([[0], [1]])) is None


def test_exact_large_coefficients_are_not_rounded_or_narrowed():
    large = 1 << 80
    source = s.Matrix([[large, large + 1]])
    particular, kernel = solve_integer(source, s.Matrix([[1]]))
    assert source * particular == s.Matrix([[1]])
    assert source * kernel == s.zeros(1, 1)
    _integer_entries(particular)
    assert gcd(abs(int(kernel[0, 0])), abs(int(kernel[1, 0]))) == 1


def test_native_only_mixed_storage_inputs_retain_checked_integer_sources():
    source = s.ImmutableMatrix(_native.Matrix(1, 2, [
        _native.py_integer(2), _native.py_integer(3),
    ]))
    rhs = s.SparseMatrix(_native.SparseMatrix(1, 1, {(0, 0): _native.py_integer(1)}))
    assert source._surface is None and rhs._surface is None
    particular, kernel = solve_integer(source, rhs)
    assert type(particular) is type(kernel) is s.ImmutableMatrix
    assert source * particular == rhs
    assert source * kernel == s.zeros(1, 1)


@pytest.mark.parametrize("operand", ["coefficients", "rhs"])
@pytest.mark.parametrize("entry", [
    s.Rational(1, 2), s.Float(0), s.Symbol("integer_rhs_unknown"),
    s.Add(1, -1, evaluate=False),
    s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False),
])
def test_both_original_sources_require_literal_integer_entries(operand, entry):
    source = s.Matrix([[entry if operand == "coefficients" else 2]])
    rhs = s.ImmutableSparseMatrix([[entry if operand == "rhs" else 2]])
    with pytest.raises((NotImplementedError, ValueError)):
        solve_integer(source, rhs)


def test_zero_rhs_columns_do_not_erase_an_invalid_original_coefficient():
    hidden = s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False)
    with pytest.raises((NotImplementedError, ValueError)):
        solve_integer(s.Matrix([[hidden]]), _zero(s.SparseMatrix, 1, 0))


def test_dimension_mismatch_is_a_refusal_and_not_integer_inconsistency():
    with pytest.raises(ValueError, match="same number of rows"):
        solve_integer(s.eye(2), s.Matrix([[1]]))


@pytest.mark.parametrize("operand", ["coefficients", "rhs"])
def test_custom_matrix_protocols_are_not_called(operand):
    observed = []

    class CustomMatrix(s.Matrix):
        def __getitem__(self, index):
            observed.append(index)
            raise AssertionError("custom matrix indexing executed")

    source = CustomMatrix([[2]]) if operand == "coefficients" else s.Matrix([[2]])
    rhs = CustomMatrix([[2]]) if operand == "rhs" else s.Matrix([[2]])
    with pytest.raises(NotImplementedError, match="built-in"):
        solve_integer(source, rhs)
    assert observed == []


@pytest.mark.parametrize("operand", ["coefficients", "rhs"])
def test_stale_retained_cells_cannot_rebind_either_source(operand):
    source, rhs = s.Matrix([[2]]), s.SparseMatrix([[2]])
    if operand == "coefficients":
        source._surface = (s.Integer(3),)
    else:
        rhs._surface = {(0, 0): s.Integer(3)}
    with pytest.raises(RuntimeError, match="retained cells differ"):
        solve_integer(source, rhs)


@pytest.mark.parametrize("operand", ["coefficients", "rhs"])
@pytest.mark.parametrize("consistent", [False, True])
def test_late_mutation_blocks_families_and_inconsistency_publication(operand, consistent):
    source, rhs = s.Matrix([[2]]), s.Matrix([[2 if consistent else 1]])
    selected = source if operand == "coefficients" else rhs
    original = normalforms._check_rational_source
    checks = []

    def changed(snapshot):
        if snapshot[0] is selected:
            checks.append(1)
            if len(checks) == 3:
                selected[0, 0] = selected[0, 0] + 1
        original(snapshot)

    with patch.object(normalforms, "_check_rational_source", changed):
        with pytest.raises(RuntimeError, match="mutated"):
            solve_integer(source, rhs)
    assert len(checks) == 3


def test_coefficient_mutation_during_rhs_capture_cannot_rebind_a_solve():
    source, rhs = s.Matrix([[2]]), s.Matrix([[2]])
    original = normalforms._rational_source_snapshot

    def changed(matrix, **kwargs):
        captured = original(matrix, **kwargs)
        if matrix is rhs:
            source[0, 0] = 3
        return captured

    with patch.object(normalforms, "_rational_source_snapshot", changed):
        with pytest.raises(RuntimeError, match="mutated"):
            solve_integer(source, rhs)


@pytest.mark.parametrize("cancelled", [False, True])
def test_terminal_active_requests_stop_before_any_source_snapshot(cancelled):
    source, rhs = s.Matrix([[2]]), s.SparseMatrix([[2]])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS if cancelled else 0)
    if cancelled:
        session.cancel()
    expected = RuntimeError if cancelled else ValueError
    with patch.object(normalforms, "_rational_source_snapshot",
                      side_effect=AssertionError("terminal solve copied a source")):
        with supervised_callback_session(session):
            with pytest.raises(expected, match="cancelled|budget exhausted"):
                solve_integer(source, rhs)


def _native_cost(source, rhs):
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    source._native_storage.solve_integer(rhs._native_storage, session=session)
    assert session.consumed_nested_steps > 0
    return session.consumed_nested_steps


@pytest.mark.parametrize("cls", (s.Matrix, s.SparseMatrix))
def test_generation_verification_and_native_lifting_share_one_active_allowance(cls):
    source, rhs = cls([[2, 3, 5]]), cls([[1]])
    cost = _native_cost(source, rhs)
    session = SupervisedCallbackSession(max_nested_steps=2 * cost - 1)
    source._native_storage.solve_integer(rhs._native_storage, session=session)
    assert session.consumed_nested_steps == cost
    with pytest.raises(ValueError, match="nested native request budget exhausted"):
        source._native_storage.solve_integer(rhs._native_storage, session=session)
    consumed = session.consumed_nested_steps
    assert cost <= consumed <= session.max_nested_steps
    assert session.cancelled is True
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        source._native_storage.solve_integer(rhs._native_storage, session=session)
    assert session.consumed_nested_steps == consumed


@pytest.mark.parametrize("consistent", [False, True])
def test_public_sources_and_output_share_budget_and_late_cancellation_is_terminal(consistent):
    source, rhs = s.Matrix([[2]]), s.SparseMatrix([[2 if consistent else 1]])
    cost = _native_cost(source, rhs)
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as measured:
        solve_integer(source, rhs)
    assert measured.consumed_nested_steps > cost
    original = normalforms._check_rational_source
    checks = []
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as late:
        def cancelled(snapshot):
            original(snapshot)
            if snapshot[0] is rhs:
                checks.append(1)
                if len(checks) == 4:
                    late.cancel()

        with patch.object(normalforms, "_check_rational_source", cancelled):
            with pytest.raises(RuntimeError, match="cancelled|tainted"):
                solve_integer(source, rhs)
    assert len(checks) == 4
