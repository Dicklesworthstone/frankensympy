"""Exact complex row spaces, kernels and complete original-source solutions.

Ordinary mathematical fixtures can be compared in isolated SymPy 1.14.0.
The native-only cases check ownership, exact arithmetic and terminal requests.
"""

import pytest

from sympy import (
    Add, Float, I, ImmutableMatrix, ImmutableSparseMatrix, Integer, Matrix, Mul, Pow,
    Rational, SparseMatrix, Symbol, eye, zeros,
)


CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)
CANDIDATE = hasattr(Matrix([[1]]), "_native_storage")
native_only = pytest.mark.skipif(not CANDIDATE, reason="native source and request contract")
AMPLE_STEPS = 20_000_000


def assert_exact(left, right):
    assert left.shape == right.shape
    assert all((a - b).expand() == 0 for a, b in zip(left, right))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entries,reduced,pivots", [
    ([[I]], [[1]], (0,)),
    ([[0, I, 1], [0, 0, 0]], [[0, 1, -I], [0, 0, 0]], (1,)),
    ([[1, I], [I, -1]], [[1, I], [0, 0]], (0,)),
    ([[1, I], [I, 1]], [[1, 0], [0, 1]], (0, 1)),
    ([[0, I, 1, 0], [I, 0, 1, 0], [0, 2*I, 2, 0]],
     [[1, 0, -I, 0], [0, 1, -I, 0], [0, 0, 0, 0]], (0, 1)),
])
def test_complex_rref_rank_and_pivot_columns(cls, entries, reduced, pivots):
    source = cls(entries)
    before = source.copy()
    actual, actual_pivots = source.rref()
    assert type(actual) is cls
    assert actual_pivots == pivots
    assert_exact(actual, Matrix(reduced))
    assert source.rank() == len(pivots)
    assert_exact(source.rref(pivots=False), actual)
    assert source.rref(simplify=True, normalize_last=False) == (actual, pivots)
    assert_exact(source, before)


@pytest.mark.parametrize("cls", CLASSES)
def test_complex_nullspace_is_complete_in_increasing_free_column_order(cls):
    source = cls([[0, I, 1, 0], [I, 0, 1, 0], [0, 2*I, 2, 0]])
    basis = source.nullspace()
    assert len(basis) == 2
    assert all(type(vector) is cls for vector in basis)
    assert_exact(basis[0], Matrix([I, I, 1, 0]))
    assert_exact(basis[1], Matrix([0, 0, 0, 1]))
    for vector in basis:
        assert_exact(source*vector, zeros(3, 1))
    assert source.rank() + len(basis) == source.cols
    assert_exact(source.columnspace()[0], source.col(0))
    assert_exact(source.columnspace()[1], source.col(1))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rhs_cls", CLASSES)
def test_complex_affine_family_keeps_all_rhs_and_free_parameter_positions(cls, rhs_cls):
    source = cls([[I, 1, 0], [2*I, 2, 0]])
    rhs = rhs_cls([[1, I], [2, 2*I]])
    solution, parameters, free = source.gauss_jordan_solve(rhs, freevar=True)
    assert type(solution) is type(parameters) is cls
    assert free == [1, 2]
    assert parameters.shape == (2, 2)
    t0, t1, t2, t3 = list(parameters)
    assert len({t0, t1, t2, t3}) == 4
    assert_exact(solution, Matrix([
        [-I + I*t0, 1 + I*t1], [t0, t1], [t2, t3],
    ]))
    assert_exact(source*solution, rhs)
    for values in ((0, 1, -2, 3), (I, -I, 2+I, Rational(1, 3))):
        assert_exact(source*solution.subs(dict(zip((t0, t1, t2, t3), values))), rhs)
    assert len(source.gauss_jordan_solve(rhs)) == 2


@pytest.mark.parametrize("cls", CLASSES)
def test_unique_and_inconsistent_complex_rhs_are_distinct(cls):
    source = cls([[I, 1], [0, I], [I, 1+I]])
    expected = Matrix([[2-I, 3], [1+I, -I]])
    rhs = cls(source*expected)
    solution, parameters, free = source.gauss_jordan_solve(rhs, freevar=True)
    assert_exact(solution, expected)
    assert parameters.shape == (0, 2) and free == []
    bad = Matrix(rhs)
    bad[2, 1] = bad[2, 1] + I
    with pytest.raises(ValueError, match="no solution"):
        source.gauss_jordan_solve(cls(bad))


@pytest.mark.parametrize("cls", CLASSES)
def test_rational_coefficients_with_complex_rhs_select_the_complex_family(cls):
    source = cls([[1, 2], [2, 4]])
    solution, parameters, free = source.gauss_jordan_solve(cls([[I], [2*I]]), freevar=True)
    assert free == [1]
    assert_exact(solution, Matrix([I-2*parameters[0, 0], parameters[0, 0]]))


@native_only
def test_tiny_nonzero_complex_pivots_do_not_underflow():
    tiny = Rational(1, 10**400)
    source = Matrix([[I*tiny, 1], [0, I*tiny]])
    assert source.rank() == 2
    assert source.nullspace() == []
    assert_exact(source.rref()[0], eye(2))


@native_only
@pytest.mark.parametrize("operation", ["rref", "rank", "nullspace", "gauss_jordan_solve"])
def test_defined_held_sources_remain_intact(operation):
    source = Pow(Add(1, I, evaluate=False), -1, evaluate=False)
    matrix = Matrix([[source, 1]])
    retained = matrix._surface
    result = (matrix.gauss_jordan_solve(Matrix([1])) if operation == "gauss_jordan_solve"
              else getattr(matrix, operation)())
    assert result is not None
    assert matrix._surface is retained and matrix._surface[0] is source


@native_only
@pytest.mark.parametrize("operation", ["rref", "rank", "nullspace", "gauss_jordan_solve"])
@pytest.mark.parametrize("wrapper", [
    lambda pole: Mul(0, pole, evaluate=False),
    lambda pole: Pow(pole, 0, evaluate=False),
])
def test_undefined_original_cells_cannot_be_erased_by_reduction(operation, wrapper):
    zero = Add(I, -I, evaluate=False)
    matrix = Matrix([[wrapper(Pow(zero, -1, evaluate=False))]])
    with pytest.raises(ValueError, match="division by zero"):
        if operation == "gauss_jordan_solve":
            matrix.gauss_jordan_solve(Matrix([0]))
        else:
            getattr(matrix, operation)()


@native_only
@pytest.mark.parametrize("operation", ["rref", "nullspace", "gauss_jordan_solve"])
def test_changed_retained_complex_children_cannot_publish_stale_results(operation, monkeypatch):
    coefficient = Rational(29, 17)
    source = Mul(coefficient, I, evaluate=False)
    matrix = Matrix([[source, 1]])
    monkeypatch.setattr(coefficient, "_value", Integer(3)._value)
    with pytest.raises(RuntimeError, match="retained.*differ"):
        if operation == "gauss_jordan_solve":
            matrix.gauss_jordan_solve(Matrix([0]))
        else:
            getattr(matrix, operation)()


@native_only
@pytest.mark.parametrize("method", ["gaussian_rref", "gaussian_nullspace", "gaussian_gauss_jordan_solve"])
def test_replaced_native_methods_are_refused_without_invocation(method, monkeypatch):
    matrix = Matrix([[I]])
    called = []

    def forbidden(*args, **kwargs):
        called.append(True)
        raise AssertionError("replaced native method executed")

    monkeypatch.setattr(type(matrix._native_storage), method, forbidden)
    with pytest.raises(NotImplementedError, match="original native interfaces"):
        if method == "gaussian_gauss_jordan_solve":
            matrix.gauss_jordan_solve(Matrix([1]))
        elif method == "gaussian_nullspace":
            matrix.nullspace()
        else:
            matrix.rref()
    assert called == []


@native_only
@pytest.mark.parametrize("method", ["gaussian_rref", "gaussian_nullspace", "gaussian_gauss_jordan_solve"])
def test_repeated_native_requests_keep_cumulative_accounting(method):
    from sympy.core import SupervisedCallbackSession

    source = Matrix([[I, 1], [0, 0]])._native_storage
    rhs = Matrix([[1], [0]])._native_storage

    def run(session):
        operation = getattr(source, method)
        return operation(rhs, session=session) if method.endswith("solve") else operation(session=session)

    measured = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    run(measured)
    cost = measured.consumed_nested_steps
    assert cost > 0
    limited = SupervisedCallbackSession(max_nested_steps=2*cost - 1)
    run(limited)
    with pytest.raises(ValueError, match="budget exhausted"):
        run(limited)
    spent = limited.consumed_nested_steps
    assert cost <= spent <= limited.max_nested_steps
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        run(limited)
    assert limited.consumed_nested_steps == spent


@native_only
@pytest.mark.parametrize("operation", ["rref", "rank", "nullspace", "gauss_jordan_solve"])
def test_cancelled_requests_stop_before_source_copying(operation, monkeypatch):
    from sympy.core import SupervisedCallbackSession, supervised_callback_session
    from sympy.matrices import _definiteness

    matrix = Matrix([[I]])
    rhs = Matrix([1])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    session.cancel()

    def forbidden(*args):
        raise AssertionError("cancelled request copied original cells")

    monkeypatch.setattr(_definiteness, "_bind_gaussian", forbidden)
    with supervised_callback_session(session):
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            if operation == "gauss_jordan_solve":
                matrix.gauss_jordan_solve(rhs)
            else:
                getattr(matrix, operation)()


@native_only
def test_custom_zero_callbacks_and_complex_subclasses_are_not_silently_bypassed():
    called = []
    with pytest.raises(NotImplementedError, match="custom callbacks"):
        Matrix([[I]]).rref(iszerofunc=lambda value: called.append(value))
    assert called == []

    class CustomMatrix(Matrix):
        pass

    with pytest.raises(NotImplementedError, match="custom matrix subclasses"):
        CustomMatrix([[I]]).rref()


@native_only
def test_legacy_literal_rational_subclasses_keep_their_result_constructor():
    class CustomMatrix(Matrix):
        pass

    matrix = CustomMatrix([[1, 2], [0, 0]])
    reduced, pivots = matrix.rref()
    assert type(reduced) is CustomMatrix and pivots == (0,)
    assert matrix.rank() == 1
    assert type(matrix.nullspace()[0]) is CustomMatrix


@native_only
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entry", [Symbol("legacy_x"), Float("0.5")])
def test_symbolic_and_float_sources_keep_the_existing_reduction_lane(cls, entry):
    source = cls([[1, entry], [0, 1]])
    reduced, pivots = source.rref()
    assert type(reduced) is cls
    assert_exact(reduced, eye(2))
    assert pivots == (0, 1) and source.rank() == 2
    assert source.nullspace() == []


@native_only
@pytest.mark.parametrize("cls", [SparseMatrix, ImmutableSparseMatrix])
def test_active_rational_sparse_rref_keeps_the_pinned_dense_result_class(cls):
    from sympy.core import SupervisedCallbackSession, supervised_callback_session

    source = cls([[1, 2], [0, 0]])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    with supervised_callback_session(session):
        reduced, pivots = source.rref()
        assert type(reduced) is Matrix and pivots == (0,)
        basis = source.nullspace()
        assert len(basis) == 1 and type(basis[0]) is cls
    assert session.consumed_nested_steps > 0


@native_only
def test_plain_symbolic_dense_subclass_retains_its_legacy_reduction():
    class CustomMatrix(Matrix):
        pass

    source = CustomMatrix([[1, Symbol("subclass_x")], [0, 1]])
    reduced, pivots = source.rref()
    assert type(reduced) is CustomMatrix and pivots == (0, 1)
    assert_exact(reduced, eye(2))
