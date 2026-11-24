"""Exact public LDL/Cholesky solves for square and inconsistent tall systems.

Ordinary values, shapes and result classes are checked against isolated SymPy
1.14.0. Native-only cases exercise original-source, uniqueness and cumulative
request obligations; deficient systems refuse rather than publishing nan/zoo.
"""

import pytest

from sympy import (
    Add, I, ImmutableMatrix, ImmutableSparseMatrix, Integer, Matrix, Mul, Pow,
    Rational, SparseMatrix, Symbol, zeros,
)


CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)
METHODS = ("LDLsolve", "cholesky_solve")
CANDIDATE = hasattr(Matrix([[1]]), "_native_storage")
native_only = pytest.mark.skipif(not CANDIDATE, reason="native source and request contract")
AMPLE_STEPS = 20_000_000


def assert_exact(left, right):
    assert left.shape == right.shape
    assert all((a - b).expand() == 0 for a, b in zip(left, right))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rhs_cls", CLASSES)
@pytest.mark.parametrize("complex_data", [False, True])
def test_inconsistent_tall_systems_keep_coefficient_class(method, cls, rhs_cls, complex_data):
    value = 2*I if complex_data else 2
    coefficients = cls([[1], [value]])
    rhs = rhs_cls([[1, 0], [0, 1]])
    result = getattr(coefficients, method)(rhs)
    assert type(result) is cls
    expected = -2*I/5 if complex_data else Rational(2, 5)
    assert_exact(result, Matrix([[Rational(1, 5), expected]]))
    assert_exact(coefficients.H*(coefficients*result - rhs), zeros(1, 2))
    assert_exact(coefficients, cls([[1], [value]]))
    assert_exact(rhs, rhs_cls([[1, 0], [0, 1]]))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entries,values,expected", [
    ([[0, 1], [1, 0]], [[1, 2], [3, 4]], [[3, 4], [1, 2]]),
    ([[1, 2], [0, 1]], [[3, 0], [1, 2]], [[1, -4], [1, 2]]),
    ([[I, 1], [0, 1]], [[1, 0], [0, 1]], [[-I, I], [0, 1]]),
])
def test_square_coefficients_need_not_be_positive_or_hermitian(
    method, cls, entries, values, expected,
):
    coefficients, rhs = cls(entries), ImmutableSparseMatrix(values)
    result = getattr(coefficients, method)(rhs)
    assert type(result) is cls
    assert_exact(result, Matrix(expected))
    assert_exact(coefficients*result, rhs)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cls", CLASSES)
def test_coupled_complex_columns_check_every_residual(method, cls):
    coefficients = cls([[1, I], [0, 0], [0, 1]])
    rhs = Matrix([[1, 0], [3, I], [0, 1]])
    result = getattr(coefficients, method)(rhs)
    assert_exact(result, Matrix([[1, -I], [0, 1]]))
    assert_exact(coefficients.H*(coefficients*result - rhs), zeros(2, 2))


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rows,rhs_columns", [(0, 0), (0, 2), (3, 0), (3, 2)])
def test_zero_unknowns_preserve_full_rhs_shape(method, cls, rows, rhs_columns):
    coefficients = cls(rows, 0, [])
    rhs = SparseMatrix(rows, rhs_columns, [I]*(rows*rhs_columns))
    result = getattr(coefficients, method)(rhs)
    assert type(result) is cls and result.shape == (0, rhs_columns)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rows", [0, 1])
def test_underdetermined_sources_refuse_even_for_empty_rhs(method, cls, rows):
    coefficients = cls(rows, 2, [1]*(2*rows))
    with pytest.raises(NotImplementedError, match="Under-determined"):
        getattr(coefficients, method)(Matrix(rows, 0, []))


@native_only
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("complex_data", [False, True])
def test_empty_rhs_does_not_remove_the_full_rank_obligation(method, cls, complex_data):
    value = I if complex_data else 2
    full_rank = cls([[1], [value]])
    result = getattr(full_rank, method)(ImmutableSparseMatrix(2, 0, {}))
    assert type(result) is cls and result.shape == (1, 0)
    deficient = cls([[1, 2], [value, 2*value], [0, 0]])
    with pytest.raises(ValueError, match="singular|rank|unique"):
        getattr(deficient, method)(SparseMatrix(3, 0, {}))


@native_only
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cls", [SparseMatrix, ImmutableSparseMatrix])
def test_large_rational_tall_systems_retain_the_sparse_kernel(method, cls, monkeypatch):
    from sympy.matrices import _pseudoinverse

    size = 64
    entries = {(row, row): 1 for row in range(size)}
    entries.update({(size + row, row): 2 for row in range(size)})
    coefficients = cls(2*size, size, entries)
    rhs = Matrix(2*size, 2, [1, 0]*size + [0, 1]*size)

    def forbidden(*args, **kwargs):
        raise AssertionError("literal rational factor solve entered a dense source lane")

    monkeypatch.setattr(_pseudoinverse, "_capture", forbidden)
    monkeypatch.setattr(SparseMatrix, "to_dense", forbidden)
    result = getattr(coefficients, method)(rhs)
    assert type(result) is cls
    assert_exact(result, cls(size, 2, [Rational(1, 5), Rational(2, 5)]*size))


@native_only
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cls", [Matrix, SparseMatrix])
def test_defined_held_operands_and_native_only_results_remain_reusable(method, cls):
    value = Pow(Add(1, I, evaluate=False), -1, evaluate=False)
    right = Pow(Add(1, -I, evaluate=False), -1, evaluate=False)
    coefficients, rhs = cls([[value]]), cls([[right]])
    old_left, old_right = coefficients._surface, rhs._surface
    result = getattr(coefficients, method)(rhs)
    assert_exact(result, Matrix([[I]]))
    assert coefficients._surface is old_left and rhs._surface is old_right
    assert result._surface is None
    assert_exact(getattr(result, method)(cls([[1]])), Matrix([[-I]]))


@native_only
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cls", [Matrix, SparseMatrix])
@pytest.mark.parametrize("side", ["coefficients", "rhs"])
def test_original_hidden_poles_cannot_disappear_in_normal_equations(method, cls, side):
    zero = Add(I, -I, evaluate=False)
    hidden = Mul(0, Pow(zero, -1, evaluate=False), evaluate=False)
    coefficients = cls([[hidden if side == "coefficients" else I]])
    rhs = cls([[hidden if side == "rhs" else 1]])
    with pytest.raises(ValueError, match="division by zero"):
        getattr(coefficients, method)(rhs)


@native_only
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cls", [Matrix, SparseMatrix])
@pytest.mark.parametrize("side", ["coefficients", "rhs"])
def test_changed_scalar_owners_cannot_publish_stale_factor_solutions(method, cls, side, monkeypatch):
    coefficient = Rational(29, 17)
    value = Mul(coefficient, I, evaluate=False)
    coefficients = cls([[value if side == "coefficients" else I]])
    rhs = cls([[value if side == "rhs" else 1]])
    monkeypatch.setattr(coefficient, "_value", Integer(3)._value)
    with pytest.raises(RuntimeError, match="retained.*differ"):
        getattr(coefficients, method)(rhs)


@native_only
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("side", ["coefficients", "rhs"])
def test_source_mutation_after_gaussian_binding_prevents_publication(method, side, monkeypatch):
    from sympy.matrices import _definiteness

    coefficients, rhs = SparseMatrix([[I]]), Matrix([[I]])
    changed = coefficients if side == "coefficients" else rhs
    native = changed._native_storage
    original = _definiteness._bind_gaussian

    def bind_then_change(*args):
        original(*args)
        if args[0] is native:
            changed[0, 0] = 3

    monkeypatch.setattr(_definiteness, "_bind_gaussian", bind_then_change)
    with pytest.raises(RuntimeError, match="mutated"):
        getattr(coefficients, method)(rhs)


@native_only
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("complex_data", [False, True])
def test_replaced_native_solver_is_refused_without_a_fallback(method, complex_data, monkeypatch):
    value = I if complex_data else 2
    coefficients, rhs = SparseMatrix([[value]]), SparseMatrix([[1]])
    native_type = type(Matrix([[1]])._native_storage if complex_data else coefficients._native_storage)
    calls = []

    def changed(*args, **kwargs):
        calls.append(True)
        raise AssertionError("replaced native least-squares solver executed")

    monkeypatch.setattr(native_type, "solve_least_squares", changed)
    with pytest.raises(NotImplementedError, match="original.*native interfaces"):
        getattr(coefficients, method)(rhs)
    assert calls == []


@native_only
@pytest.mark.parametrize("method", METHODS)
def test_complex_rhs_bounds_original_sparse_coefficients_before_copy(method, monkeypatch):
    from sympy.matrices import _definiteness

    coefficients = SparseMatrix(33, 1, {(0, 0): 1})
    rhs = SparseMatrix(33, 1, {(0, 0): I})

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized Gaussian request copied rational source cells")

    monkeypatch.setattr(_definiteness, "_rational_source_snapshot", forbidden)
    with pytest.raises(ValueError, match="dimension|32"):
        getattr(coefficients, method)(rhs)


@native_only
@pytest.mark.parametrize("method", METHODS)
def test_dense_rational_subclasses_keep_legacy_result_hooks(method):
    constructed = []

    class CustomMatrix(Matrix):
        def _new(self, *args, **kwargs):
            constructed.append(True)
            return type(self)(*args, **kwargs)

    coefficients = CustomMatrix([[1], [2]])
    result = getattr(coefficients, method)(Matrix([[1], [0]]))
    assert type(result) is CustomMatrix
    assert_exact(result, Matrix([[Rational(1, 5)]]))
    assert constructed == [True]


@native_only
@pytest.mark.parametrize("method", METHODS)
def test_legacy_constructor_effects_cannot_rebase_original_sources(method):
    class CustomMatrix(Matrix):
        def _new(self, *args, **kwargs):
            self[0, 0] = 7
            return type(self)(*args, **kwargs)

    with pytest.raises(RuntimeError, match="mutated"):
        getattr(CustomMatrix([[1], [2]]), method)(Matrix([[1], [0]]))


@native_only
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cls", [Matrix, SparseMatrix])
def test_unsupported_symbolic_sources_refuse_without_publishing(method, cls):
    coefficients = cls([[Symbol("factor_solve_unknown")], [1]])
    with pytest.raises((ValueError, NotImplementedError)):
        getattr(coefficients, method)(cls([[1], [0]]))


@native_only
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cancelled", [False, True])
def test_terminal_requests_stop_before_source_copying(method, cancelled, monkeypatch):
    from sympy.core import SupervisedCallbackSession, supervised_callback_session
    from sympy.matrices import _definiteness

    coefficients, rhs = SparseMatrix([[1], [2]]), Matrix([[1], [0]])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS if cancelled else 0)
    if cancelled:
        session.cancel()

    def forbidden(*args, **kwargs):
        raise AssertionError("terminal factor-solve request copied source cells")

    monkeypatch.setattr(_definiteness, "_rational_source_snapshot", forbidden)
    with supervised_callback_session(session):
        with pytest.raises(RuntimeError if cancelled else ValueError, match="cancelled|budget exhausted"):
            getattr(coefficients, method)(rhs)


@native_only
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("complex_data", [False, True])
def test_sparse_factor_solves_share_one_cumulative_request(method, complex_data):
    from sympy.core import SupervisedCallbackSession, supervised_callback_session

    value = 2*I if complex_data else 2
    coefficients, rhs = SparseMatrix([[1], [value]]), Matrix([[1], [0]])
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as measured:
        getattr(coefficients, method)(rhs)
    cost = measured.consumed_nested_steps
    assert cost > 0
    limited = SupervisedCallbackSession(max_nested_steps=2*cost - 1)
    with supervised_callback_session(limited):
        getattr(coefficients, method)(rhs)
        with pytest.raises(ValueError, match="budget exhausted"):
            getattr(coefficients, method)(rhs)
    assert cost <= limited.consumed_nested_steps <= limited.max_nested_steps


@native_only
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cls", [Matrix, SparseMatrix])
def test_cancellation_at_publication_is_terminal(method, cls, monkeypatch):
    from sympy.core import supervised_callback_session
    from sympy.matrices import _row_reduction

    coefficients, rhs = cls([[I]]), cls([[1]])
    original = _row_reduction._publish
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as session:
        def cancel_before_publish(*args):
            session.cancel()
            return original(*args)

        monkeypatch.setattr(_row_reduction, "_publish", cancel_before_publish)
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            getattr(coefficients, method)(rhs)
