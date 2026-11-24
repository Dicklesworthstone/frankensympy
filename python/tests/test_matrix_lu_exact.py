"""Rectangular exact LU factors, public swap lists and retained sources.

Ordinary fixtures run unchanged against isolated SymPy 1.14.0. Native-only
fixtures cover the bounded checked extension and its publication boundary.
"""

import pytest

from sympy import (
    Add, I, ImmutableMatrix, ImmutableSparseMatrix, Integer, Matrix, Mul, Pow,
    Rational, SparseMatrix, eye,
)


CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)
CANDIDATE = hasattr(Matrix([[1]]), "_native_storage")
native_only = pytest.mark.skipif(not CANDIDATE, reason="native source and request contract")
AMPLE_STEPS = 20_000_000


def assert_exact(left, right):
    assert left.shape == right.shape
    assert all((a - b).expand() == 0 for a, b in zip(left, right))


def apply_swaps(matrix, swaps):
    """Apply row swaps without depending on a second public permutation API."""
    order = list(range(matrix.rows))
    for pair in swaps:
        assert type(pair) is list and len(pair) == 2
        first, second = pair
        assert type(first) is type(second) is int
        assert 0 <= first < matrix.rows and 0 <= second < matrix.rows
        order[first], order[second] = order[second], order[first]
    return Matrix(matrix.rows, matrix.cols, [
        matrix[order[row], column]
        for row in range(matrix.rows) for column in range(matrix.cols)
    ])


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entries, expected_swaps", [
    ([[2, 3], [4, 7]], []),
    ([[0, 2], [3, 4]], [[0, 1]]),
    ([[0, 0, 1], [1, 0, 2], [0, 1, 3]], [[0, 1], [1, 2]]),
    ([[I, 1], [2*I, 3]], []),
    ([[0, I], [1, 0], [2, 3]], [[0, 1]]),
    ([[0, I, 1, 2], [I, 0, 2, 3]], [[0, 1]]),
    ([[0, 1, 2], [0, 2, 4], [0, 3, 6]], []),
    ([[0, 0, I], [0, 0, 2*I], [0, 0, 0]], []),
    ([[0, 0], [0, 0]], []),
    ([[Rational(1, 3), Rational(2, 5)], [Rational(2, 3), Rational(7, 5)]], []),
])
def test_lu_preserves_rectangular_shapes_classes_and_exact_row_equivalence(
    cls, entries, expected_swaps,
):
    source = cls(entries)
    before = source.copy()
    lower, upper, swaps = source.LUdecomposition()
    assert type(lower) is type(upper) is cls
    assert type(swaps) is list and swaps == expected_swaps
    assert lower.shape == (source.rows, source.rows)
    assert upper.shape == source.shape
    for row in range(lower.rows):
        assert lower[row, row] == 1
        assert all(lower[row, column] == 0 for column in range(row + 1, lower.cols))
    for row in range(upper.rows):
        assert all(upper[row, column] == 0 for column in range(min(row, upper.cols)))
    assert_exact(lower * upper, apply_swaps(source, swaps))
    assert_exact(source, before)


@pytest.mark.parametrize("cls", CLASSES)
def test_skipped_columns_use_pivot_row_positions_in_lower_factor(cls):
    source = cls([[0, 1, 2], [0, 2, 4], [0, 3, 6]])
    lower, upper, swaps = source.LUdecomposition()
    assert swaps == []
    assert_exact(lower, Matrix([[1, 0, 0], [2, 1, 0], [3, 0, 1]]))
    assert_exact(upper, Matrix([[0, 1, 2], [0, 0, 0], [0, 0, 0]]))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rows, columns", [(0, 0), (0, 3), (3, 0)])
@pytest.mark.parametrize("rankcheck", [False, True])
def test_zero_area_lu_retains_shapes_and_has_full_possible_rank(
    cls, rows, columns, rankcheck,
):
    source = cls(rows, columns, [])
    lower, upper, swaps = source.LUdecomposition(rankcheck=rankcheck)
    assert type(lower) is type(upper) is cls
    assert lower.shape == (rows, rows) and upper.shape == (rows, columns)
    assert swaps == []
    assert_exact(lower, eye(rows))
    assert_exact(lower * upper, source)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entries", [
    [[1, 2], [2, 4]],
    [[I, 1], [2*I, 2], [0, 0]],
    [[0, 0, I], [0, 0, 2*I]],
    [[0, 0], [0, 0]],
])
def test_rankcheck_refuses_deficient_square_tall_and_wide_sources(cls, entries):
    source = cls(entries)
    with pytest.raises(ValueError, match="[Rr]ank|rank"):
        source.LUdecomposition(rankcheck=True)
    lower, upper, swaps = source.LUdecomposition(rankcheck=False)
    assert_exact(lower * upper, apply_swaps(source, swaps))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entries", [
    [[I, 1], [0, 2]],
    [[0, I], [1, 0], [2, 3]],
    [[0, I, 1], [I, 0, 2]],
])
def test_rankcheck_accepts_full_minimum_axis_rank(cls, entries):
    source = cls(entries)
    lower, upper, swaps = source.LUdecomposition(rankcheck=True)
    assert_exact(lower * upper, apply_swaps(source, swaps))


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_native_lu_alias_preserves_its_permutation_matrix_contract(cls):
    source = cls([[0, 0, I], [1, 0, 2], [0, 1, 3]])
    permutation, lower, upper = source.lu()
    assert type(permutation) is type(lower) is type(upper) is cls
    public_lower, public_upper, swaps = source.LUdecomposition()
    assert swaps == [[0, 1], [1, 2]]
    assert_exact(permutation * source, lower * upper)
    assert_exact(permutation, apply_swaps(eye(source.rows), swaps))
    assert_exact(lower, public_lower)
    assert_exact(upper, public_upper)


@native_only
def test_tiny_exact_complex_pivot_is_not_replaced_by_a_larger_float_pivot():
    tiny = Rational(1, 10**400)
    source = Matrix([[I*tiny, 1], [I, 2]])
    lower, upper, swaps = source.LUdecomposition(rankcheck=True)
    assert swaps == []
    assert lower[1, 0] == 10**400
    assert_exact(lower * upper, source)


@native_only
@pytest.mark.parametrize("operation", ["LUdecomposition", "lu"])
def test_defined_held_lu_source_remains_owned_by_the_original_matrix(operation):
    entry = Pow(Add(1, I, evaluate=False), -1, evaluate=False)
    source = Matrix([[entry, 1], [0, I]])
    retained = source._surface
    assert getattr(source, operation)() is not None
    assert source._surface is retained and source._surface[0] is entry


@native_only
@pytest.mark.parametrize("operation", ["LUdecomposition", "lu"])
@pytest.mark.parametrize("wrapper", [
    lambda pole: Mul(0, pole, evaluate=False),
    lambda pole: Pow(pole, 0, evaluate=False),
])
def test_original_poles_cannot_disappear_behind_zero_rank_lu(operation, wrapper):
    zero = Add(I, -I, evaluate=False)
    source = Matrix([[wrapper(Pow(zero, -1, evaluate=False))]])
    with pytest.raises(ValueError, match="division by zero"):
        getattr(source, operation)()


@native_only
@pytest.mark.parametrize("operation", ["LUdecomposition", "lu"])
def test_changed_retained_lu_scalar_cannot_publish_a_stale_factorization(operation, monkeypatch):
    coefficient = Rational(31, 19)
    entry = Mul(coefficient, I, evaluate=False)
    source = Matrix([[entry, 1], [0, I]])
    monkeypatch.setattr(coefficient, "_value", Integer(3)._value)
    with pytest.raises(RuntimeError, match="retained.*differ"):
        getattr(source, operation)()


@native_only
def test_replaced_native_lu_descriptor_is_refused_before_invocation(monkeypatch):
    source = Matrix([[I]])
    called = []

    def forbidden(*args, **kwargs):
        called.append(True)
        raise AssertionError("replaced native LU method executed")

    monkeypatch.setattr(type(source._native_storage), "gaussian_lu", forbidden)
    with pytest.raises(NotImplementedError, match="original native interfaces"):
        source.LUdecomposition()
    assert called == []


@native_only
def test_repeated_native_lu_calls_keep_one_cumulative_request():
    from sympy.core import SupervisedCallbackSession

    source = Matrix([[I, 1], [0, I]])._native_storage
    measured = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    source.gaussian_lu(session=measured)
    cost = measured.consumed_nested_steps
    assert cost > 0
    limited = SupervisedCallbackSession(max_nested_steps=2*cost - 1)
    source.gaussian_lu(session=limited)
    with pytest.raises(ValueError, match="budget exhausted"):
        source.gaussian_lu(session=limited)
    spent = limited.consumed_nested_steps
    assert cost <= spent <= limited.max_nested_steps
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        source.gaussian_lu(session=limited)
    assert limited.consumed_nested_steps == spent


@native_only
@pytest.mark.parametrize("operation", ["LUdecomposition", "lu"])
def test_cancelled_lu_request_stops_before_original_scalar_binding(operation, monkeypatch):
    from sympy.core import SupervisedCallbackSession, supervised_callback_session
    from sympy.matrices import _definiteness

    source = Matrix([[I]])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    session.cancel()

    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled LU request copied original scalars")

    monkeypatch.setattr(_definiteness, "_bind_gaussian", forbidden)
    with supervised_callback_session(session):
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            getattr(source, operation)()


@native_only
def test_checked_lu_rejects_custom_callbacks_and_nonboolean_rankcheck():
    called = []

    def callback(value):
        called.append(value)
        return value

    source = Matrix([[I]])
    for keyword in ("iszerofunc", "simpfunc"):
        with pytest.raises(NotImplementedError, match="callback|override"):
            source.LUdecomposition(**{keyword: callback})
    assert called == []
    with pytest.raises(TypeError, match="rankcheck|bool"):
        source.LUdecomposition(rankcheck=1)


@native_only
def test_large_standalone_literal_rational_square_keeps_existing_lu_support():
    source = eye(33)
    lower, upper, swaps = source.LUdecomposition()
    assert swaps == []
    assert_exact(lower, source)
    assert_exact(upper, source)
    permutation, alias_lower, alias_upper = source.lu()
    assert_exact(permutation, source)
    assert_exact(alias_lower, lower)
    assert_exact(alias_upper, upper)
