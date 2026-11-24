"""Sparse square/tall Q(i) unique solves and source-bound exact inverses."""

import pytest

from sympy import (
    Add, I, ImmutableMatrix, ImmutableSparseMatrix, Integer, Matrix, Mul, Pow,
    Rational, SparseMatrix, Symbol, eye,
)


CLASSES = (SparseMatrix, ImmutableSparseMatrix)
RHS_CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)
CANDIDATE = hasattr(Matrix([[1]]), "_native_storage")
native_only = pytest.mark.skipif(not CANDIDATE, reason="native source and request contract")
AMPLE_STEPS = 20_000_000


def assert_exact(left, right):
    assert left.shape == right.shape
    assert all((a - b).expand() == 0 for a, b in zip(left, right))


def system(cls, rhs_cls, tall):
    entries = [[I, 1], [1, I]]
    values = [[4*I, 3+2*I], [-2*I, 4+5*I]]
    if tall:
        entries.append([1+I, 1+I])
        values.append([2*I, 7+7*I])
    return cls(entries), rhs_cls(values)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rhs_cls", RHS_CLASSES)
@pytest.mark.parametrize("tall", [False, True])
def test_sparse_complex_lusolve_checks_original_residual_and_keeps_rhs_class(cls, rhs_cls, tall):
    coefficients, rhs = system(cls, rhs_cls, tall)
    result = coefficients.LUsolve(rhs)
    assert type(result) is rhs_cls
    assert_exact(result, Matrix([[2-I, 3+I], [-1+2*I, 4-I]]))
    assert_exact(coefficients*result, rhs)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rhs_cls", RHS_CLASSES)
def test_square_sparse_solve_observes_mutability_class_promotion(cls, rhs_cls):
    coefficients, rhs = system(cls, rhs_cls, False)
    result = coefficients.solve(rhs)
    assert type(result) is (ImmutableSparseMatrix if cls is ImmutableSparseMatrix else rhs_cls)
    assert_exact(result, Matrix([[2-I, 3+I], [-1+2*I, 4-I]]))
    assert_exact(coefficients*result, rhs)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rhs_cls", [Matrix, ImmutableSparseMatrix])
def test_complex_rhs_selects_the_same_checked_lane_for_rational_sparse_coefficients(cls, rhs_cls):
    coefficients = cls([[0, 2], [3, 1], [3, 3]])
    rhs = rhs_cls([[-2+4*I, 8-2*I], [5-I, 13+2*I], [3+3*I, 21]])
    result = coefficients.LUsolve(rhs)
    assert type(result) is rhs_cls
    assert_exact(result, Matrix([[2-I, 3+I], [-1+2*I, 4-I]]))
    assert_exact(coefficients*result, rhs)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entries,expected", [
    ([[I]], [[-I]]),
    ([[1+I]], [[Rational(1, 2)-I/2]]),
    ([[1, I], [I, 1]], [[Rational(1, 2), -I/2], [-I/2, Rational(1, 2)]]),
])
def test_sparse_inverse_has_both_exact_inverse_identities(cls, entries, expected):
    source = cls(entries)
    result = source.inv()
    assert type(result) is cls
    assert_exact(result, Matrix(expected))
    assert_exact(source*result, eye(source.rows))
    assert_exact(result*source, eye(source.rows))


@pytest.mark.parametrize("cls", CLASSES)
def test_symbolic_sparse_inverse_keeps_its_preselected_symbolic_lane(cls):
    x = Symbol("sparse_inverse_parameter")
    source = cls([[x, 1], [1, 0]])
    assert_exact(source.inv(), Matrix([[0, 1], [1, -x]]))


def test_inconsistent_tall_sparse_system_does_not_publish_a_least_squares_solution():
    source, rhs = SparseMatrix([[1], [2*I]]), Matrix([[1], [0]])
    with pytest.raises(ValueError):
        source.LUsolve(rhs)


def test_rectangular_sparse_solve_still_refuses_and_lusolve_requires_uniqueness():
    with pytest.raises(ValueError, match="determined"):
        SparseMatrix([[1], [I]]).solve(Matrix([[1], [I]]))
    with pytest.raises(ValueError, match="determined"):
        SparseMatrix([[1, I]]).solve(Matrix([[1]]))
    with pytest.raises(NotImplementedError, match="Underdetermined"):
        SparseMatrix([[1, I]]).LUsolve(Matrix([[1]]))
    with pytest.raises(ValueError):
        SparseMatrix([[1, I], [2, 2*I], [3, 3*I]]).LUsolve(Matrix([[1], [2], [3]]))


@native_only
@pytest.mark.parametrize("operation", ["solve", "LUsolve", "inv"])
def test_original_sparse_sources_do_not_escape_through_public_dense_conversion(operation, monkeypatch):
    source, rhs = SparseMatrix([[I]]), ImmutableSparseMatrix([[1]])

    def forbidden(*args):
        raise AssertionError("exact sparse source escaped through public conversion")

    monkeypatch.setattr(SparseMatrix, "to_dense", forbidden)
    result = source.inv() if operation == "inv" else getattr(source, operation)(rhs)
    assert_exact(result, Matrix([[-I]]))


@native_only
@pytest.mark.parametrize("operation", ["solve", "LUsolve", "inv"])
def test_sparse_held_sources_remain_bound_through_exact_publication(operation):
    value = Pow(Add(1, I, evaluate=False), -1, evaluate=False)
    right = Pow(Add(1, -I, evaluate=False), -1, evaluate=False)
    source, rhs = SparseMatrix([[value]]), SparseMatrix([[right]])
    left_surface, right_surface = source._surface, rhs._surface
    result = source.inv() if operation == "inv" else getattr(source, operation)(rhs)
    assert_exact(result, Matrix([[1+I if operation == "inv" else I]]))
    assert source._surface is left_surface and left_surface[0, 0] is value
    assert rhs._surface is right_surface and right_surface[0, 0] is right


@native_only
@pytest.mark.parametrize("side", ["source", "rhs"])
def test_original_hidden_poles_cannot_be_removed_before_sparse_solving(side):
    zero = Add(I, -I, evaluate=False)
    hidden = Mul(0, Pow(zero, -1, evaluate=False), evaluate=False)
    source = SparseMatrix([[hidden if side == "source" else I]])
    rhs = SparseMatrix([[hidden if side == "rhs" else 1]])
    with pytest.raises(ValueError, match="division by zero"):
        source.LUsolve(rhs)
    with pytest.raises(ValueError, match="division by zero"):
        source.solve(rhs)
    if side == "source":
        with pytest.raises(ValueError, match="division by zero"):
            source.inv()


@native_only
@pytest.mark.parametrize("side", ["source", "rhs"])
def test_retained_scalar_mutations_cannot_publish_stale_sparse_solutions(side, monkeypatch):
    coefficient = Rational(19, 13)
    value = Mul(coefficient, I, evaluate=False)
    source = SparseMatrix([[value if side == "source" else I]])
    rhs = SparseMatrix([[value if side == "rhs" else 1]])
    monkeypatch.setattr(coefficient, "_value", Integer(3)._value)
    with pytest.raises(RuntimeError, match="retained.*differ"):
        source.LUsolve(rhs)
    if side == "source":
        with pytest.raises(RuntimeError, match="retained.*differ"):
            source.inv()


@native_only
@pytest.mark.parametrize("operation", ["solve", "LUsolve", "inv"])
def test_changed_native_sparse_publication_is_terminal(operation, monkeypatch):
    source, rhs = SparseMatrix([[I]]), SparseMatrix([[1]])
    called = []

    def changed(*args):
        called.append(True)
        raise AssertionError("replaced native conversion executed")

    monkeypatch.setattr(type(Matrix([[1]])._native_storage), "to_sparse", changed)
    with pytest.raises(NotImplementedError, match="original.*native interfaces"):
        source.inv() if operation == "inv" else getattr(source, operation)(rhs)
    assert called == []


@native_only
def test_complex_request_bounds_rational_sparse_source_before_copy(monkeypatch):
    from sympy.matrices import _definiteness

    source = SparseMatrix(33, 1, {(0, 0): 1})
    rhs = SparseMatrix(33, 1, {(0, 0): I})

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized rational source was copied")

    monkeypatch.setattr(_definiteness, "_rational_source_snapshot", forbidden)
    with pytest.raises(ValueError, match="dimension|32"):
        source.LUsolve(rhs)


@native_only
def test_large_literal_rational_sparse_solves_keep_their_existing_sparse_lane(monkeypatch):
    from sympy.matrices import _pseudoinverse

    source = SparseMatrix(33, 33, {(i, i): 2 for i in range(33)})
    rhs = SparseMatrix(33, 1, {(i, 0): 2*i for i in range(33)})

    def forbidden(*args, **kwargs):
        raise AssertionError("literal rational source entered bounded dense solve")

    monkeypatch.setattr(_pseudoinverse, "_capture", forbidden)
    for result in (source.solve(rhs), source.LUsolve(rhs)):
        assert_exact(result, SparseMatrix(33, 1, {(i, 0): i for i in range(33)}))
    assert_exact(source*source.inv(), eye(33))


@native_only
def test_empty_rhs_requires_full_rank_and_native_only_sparse_inverse_is_reusable():
    full_rank = SparseMatrix([[1], [I]])
    result = full_rank.LUsolve(ImmutableSparseMatrix(2, 0, {}))
    assert type(result) is ImmutableSparseMatrix and result.shape == (1, 0)
    deficient = SparseMatrix([[1, I], [2, 2*I], [3, 3*I]])
    with pytest.raises(ValueError, match="singular"):
        deficient.LUsolve(SparseMatrix(3, 0, {}))
    source = SparseMatrix([[I, 1], [0, 1]])
    object.__setattr__(source, "_surface", None)
    inverse = source.inv()
    assert inverse._surface is None
    assert_exact(inverse.inv(), source)


@native_only
@pytest.mark.parametrize("operation", ["solve", "LUsolve", "inv"])
def test_terminal_sparse_request_refuses_before_copying(operation, monkeypatch):
    from sympy.core import SupervisedCallbackSession, supervised_callback_session
    from sympy.matrices import _definiteness

    source, rhs = SparseMatrix([[I]]), SparseMatrix([[1]])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    session.cancel()

    def forbidden(*args):
        raise AssertionError("cancelled sparse request copied cells")

    monkeypatch.setattr(_definiteness, "_bind_gaussian", forbidden)
    with supervised_callback_session(session):
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            source.inv() if operation == "inv" else getattr(source, operation)(rhs)


@native_only
def test_sparse_complex_lusolve_uses_one_cumulative_public_budget():
    from sympy.core import SupervisedCallbackSession, supervised_callback_session

    source, rhs = system(SparseMatrix, SparseMatrix, True)
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as measured:
        source.LUsolve(rhs)
    cost = measured.consumed_nested_steps
    assert cost > 0
    limited = SupervisedCallbackSession(max_nested_steps=2*cost - 1)
    with supervised_callback_session(limited):
        source.LUsolve(rhs)
        with pytest.raises(ValueError, match="budget exhausted"):
            source.LUsolve(rhs)
    assert cost <= limited.consumed_nested_steps <= limited.max_nested_steps
