"""Exact Hermitian least squares and retained two-source publication.

Positive fixtures and result classes were independently checked with isolated
SymPy 1.14.0. Native-only cases exercise the bounded source and request contract;
the PINV family additionally supports singular Gaussian-rational systems.
"""

import pytest

from sympy import (
    Add, I, ImmutableMatrix, ImmutableSparseMatrix, Integer, Matrix, Mul, Pow,
    Rational, SparseMatrix, Symbol, zeros,
)


CLASSES = (Matrix, ImmutableMatrix)
RHS_CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)
CANDIDATE = hasattr(Matrix([[1]]), "_native_storage")
native_only = pytest.mark.skipif(not CANDIDATE, reason="native source and request contract")
AMPLE_STEPS = 20_000_000


def assert_exact(left, right):
    assert left.shape == right.shape
    assert all((a - b).expand() == 0 for a, b in zip(left, right))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entries,rhs,expected", [
    # Transpose normal equations incorrectly return -1/3 here.
    ([[1], [2*I]], [[1], [0]], [[Rational(1, 5)]]),
    # A.T*A is zero even though this nonzero column has Hermitian norm 2.
    ([[1], [I]], [[1], [0]], [[Rational(1, 2)]]),
    ([[1], [2*I]], [[1, 0], [0, 1]], [[Rational(1, 5), -2*I/5]]),
    # Both components of B matter even when A itself is real.
    ([[1], [2]], [[I, 1], [0, I]], [[I/5, Rational(1, 5) + 2*I/5]]),
    ([[1, I], [0, 0], [0, 1]], [[1, 0], [3, I], [0, 1]], [[1, -I], [0, 1]]),
])
def test_default_complex_least_squares_matches_exact_hermitian_fixtures(cls, entries, rhs, expected):
    matrix = cls(entries)
    right = Matrix(rhs)
    result = matrix.solve_least_squares(right)
    assert type(result) is cls
    assert_exact(result, Matrix(expected))
    assert_exact(matrix.H * (matrix * result - right), zeros(matrix.cols, right.cols))
    assert_exact(matrix, cls(entries))
    assert_exact(right, Matrix(rhs))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rhs_cls", RHS_CLASSES)
def test_explicit_ch_preserves_coefficient_class_for_dense_and_sparse_right_sides(cls, rhs_cls):
    matrix = cls([[1], [2*I]])
    right = rhs_cls([[1, 0], [0, 1]])
    result = matrix.solve_least_squares(right, method="CH")
    assert type(result) is cls
    assert_exact(result, Matrix([[Rational(1, 5), -2*I/5]]))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rows,rhs_columns", [(0, 0), (0, 2), (3, 0), (3, 2)])
def test_zero_unknowns_preserve_the_complete_rhs_shape(cls, rows, rhs_columns):
    matrix = cls(rows, 0, [])
    right = cls(rows, rhs_columns, [I] * (rows * rhs_columns))
    result = matrix.solve_least_squares(right)
    assert type(result) is cls and result.shape == (0, rhs_columns)


@native_only
@pytest.mark.parametrize("entries", [
    [[1, 2], [I, 2*I]],
    [[1, I]],
    [[0, 0], [0, 0]],
])
def test_unique_ch_lane_refuses_rank_deficiency(entries):
    matrix = Matrix(entries)
    # Preserve the existing native SingularMatrix -> ValueError contract.
    # Upstream uses TypeError; no exception-family compatibility is claimed.
    with pytest.raises(ValueError, match="singular|rank|unique"):
        matrix.solve_least_squares(Matrix(matrix.rows, 1, [1] * matrix.rows))


@native_only
def test_complex_pinv_family_returns_the_complete_singular_family():
    matrix = Matrix([[1, 2], [I, 2*I]])
    rhs = Matrix([[1], [0]])
    family = matrix.solve_least_squares(rhs, method="PINV")
    assert len(family.free_symbols) == 2
    assert_exact(matrix.H*(matrix*family-rhs), zeros(2, 1))
    assert_exact(matrix.pinv_solve(rhs, zeros(2, 1)),
                 Matrix([[Rational(1, 10)], [Rational(1, 5)]]))


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_held_defined_sources_in_both_operands_remain_bound(cls):
    left = Pow(Add(1, I, evaluate=False), -1, evaluate=False)
    right = Pow(Add(1, -I, evaluate=False), -1, evaluate=False)
    matrix, rhs = cls([[left]]), Matrix([[right]])
    old_left, old_right = matrix._surface, rhs._surface
    result = matrix.solve_least_squares(rhs)
    assert_exact(result, Matrix([[I]]))
    assert matrix._surface is old_left and matrix._surface[0] is left
    assert rhs._surface is old_right and rhs._surface[0] is right


@native_only
@pytest.mark.parametrize("side", ["matrix", "rhs"])
@pytest.mark.parametrize("wrapper", [
    lambda pole: Mul(0, pole, evaluate=False),
    lambda pole: Pow(pole, 0, evaluate=False),
])
def test_original_undefined_children_cannot_disappear_in_normal_equations(side, wrapper):
    zero = Add(I, -I, evaluate=False)
    hidden = wrapper(Pow(zero, -1, evaluate=False))
    matrix, rhs = Matrix([[1]]), Matrix([[0]])
    if side == "matrix":
        matrix = Matrix([[hidden]])
    else:
        # Even a rank-zero coefficient matrix must admit every original RHS.
        matrix, rhs = Matrix([[0]]), Matrix([[hidden]])
    with pytest.raises(ValueError, match="division by zero"):
        matrix.solve_least_squares(rhs)


@native_only
@pytest.mark.parametrize("side", ["matrix", "rhs"])
@pytest.mark.parametrize("mutation", ["root", "child", "arguments", "surface", "structural"])
def test_retained_scalar_changes_in_either_operand_are_terminal(side, mutation, monkeypatch):
    coefficient = Rational(17, 11)
    child = Mul(coefficient, I, evaluate=False)
    source = Add(1, child, evaluate=False)
    matrix, rhs = Matrix([[I]]), Matrix([[I]])
    if side == "matrix":
        matrix = Matrix([[source]])
    else:
        rhs = Matrix([[source]])
    if mutation == "root":
        monkeypatch.setattr(source, "_value", Integer(3)._value)
    elif mutation == "child":
        monkeypatch.setattr(coefficient, "_value", Integer(3)._value)
    elif mutation == "arguments":
        monkeypatch.setattr(child, "_args", (Integer(3), I))
    elif mutation == "surface":
        monkeypatch.setattr(child, "_surface_args", (Integer(3), I))
    else:
        monkeypatch.setattr(child, "_struct_args", (Integer(3), I))
    with pytest.raises(RuntimeError, match="retained.*differ"):
        matrix.solve_least_squares(rhs)


@native_only
@pytest.mark.parametrize("side", ["matrix", "rhs"])
def test_matrix_changes_after_source_binding_prevent_publication(side, monkeypatch):
    from sympy.matrices import _definiteness

    matrix, rhs = Matrix([[1], [2*I]]), Matrix([[1 + I], [2*I]])
    changed = matrix if side == "matrix" else rhs
    native = changed._native_storage
    original = _definiteness._bind_gaussian

    def bind_then_change(*args):
        original(*args)
        if args[0] is native:
            changed[0, 0] = 3

    monkeypatch.setattr(_definiteness, "_bind_gaussian", bind_then_change)
    with pytest.raises(RuntimeError, match="mutated"):
        matrix.solve_least_squares(rhs)


@native_only
def test_native_only_operands_reuse_checked_storage_without_public_cells():
    matrix, rhs = Matrix([[1], [I]]), Matrix([[1], [0]])
    object.__setattr__(matrix, "_surface", None)
    object.__setattr__(rhs, "_surface", None)
    result = matrix.solve_least_squares(rhs)
    assert_exact(result, Matrix([[Rational(1, 2)]]))
    assert result._surface is None


@native_only
@pytest.mark.parametrize("rhs_cls", [SparseMatrix, ImmutableSparseMatrix])
def test_changed_sparse_native_conversion_is_refused_before_it_executes(rhs_cls, monkeypatch):
    matrix, rhs = Matrix([[1], [I]]), rhs_cls([[I], [1]])
    called = []

    def changed(self):
        called.append(True)
        raise AssertionError("replaced native conversion must not execute")

    monkeypatch.setattr(type(rhs._native_storage), "to_dense", changed)
    with pytest.raises(NotImplementedError, match="original.*native interfaces"):
        matrix.solve_least_squares(rhs)
    assert called == []


@native_only
@pytest.mark.parametrize("rhs_cls", [SparseMatrix, ImmutableSparseMatrix])
@pytest.mark.parametrize("adversary", ["hidden_pole", "retained_mutation"])
def test_sparse_rhs_original_children_survive_conversion(rhs_cls, adversary, monkeypatch):
    matrix = Matrix([[I]])
    if adversary == "hidden_pole":
        zero = Add(I, -I, evaluate=False)
        pole = Pow(zero, -1, evaluate=False)
        rhs = rhs_cls([[Mul(0, pole, evaluate=False)]])
        error, message = ValueError, "division by zero"
    else:
        coefficient = Rational(19, 13)
        rhs = rhs_cls([[Mul(coefficient, I, evaluate=False)]])
        monkeypatch.setattr(coefficient, "_value", Integer(3)._value)
        error, message = RuntimeError, "retained.*differ"
    with pytest.raises(error, match=message):
        matrix.solve_least_squares(rhs)


@native_only
@pytest.mark.parametrize("oversized", ["literal_sparse_rhs", "literal_dense_matrix"])
def test_joint_complex_lane_bounds_literal_sources_before_snapshot_or_conversion(oversized, monkeypatch):
    from sympy.matrices import _definiteness

    if oversized == "literal_sparse_rhs":
        matrix, rhs = Matrix([[I]]), SparseMatrix(1, 33, {})
    else:
        matrix, rhs = Matrix(1, 33, [1] * 33), Matrix([[I]])
    copied = []

    def forbidden(*args, **kwargs):
        copied.append(True)
        raise AssertionError("oversized mixed request copied a literal source")

    # Leave every native descriptor intact: this must reach borrowed Gaussian
    # preflight, not an earlier interface-tampering refusal. Sparse conversion
    # follows this snapshot, so refusing here also detects premature densifying.
    monkeypatch.setattr(_definiteness, "_rational_source_snapshot", forbidden)
    with pytest.raises(ValueError, match="dimension|32"):
        matrix.solve_least_squares(rhs)
    assert copied == []


@native_only
@pytest.mark.parametrize("side", ["matrix", "rhs"])
def test_symbolic_and_oversized_original_sources_are_explicit_refusals(side):
    matrix, rhs = Matrix([[I]]), Matrix([[I]])
    unsupported = Matrix([[Symbol("least_squares_unknown")]])
    if side == "matrix":
        matrix = unsupported
    else:
        rhs = unsupported
    with pytest.raises(ValueError, match="exact rational|Gaussian|domain"):
        matrix.solve_least_squares(rhs)
    if side == "matrix":
        matrix, rhs = Matrix(33, 1, [I] * 33), Matrix(33, 1, [1] * 33)
    else:
        matrix, rhs = Matrix([[I]]), Matrix(1, 33, [I] * 33)
    with pytest.raises(ValueError, match="dimension|32"):
        matrix.solve_least_squares(rhs)


@native_only
@pytest.mark.parametrize("cancelled", [False, True])
def test_terminal_requests_stop_before_copying_either_complex_source(cancelled, monkeypatch):
    from sympy.core import SupervisedCallbackSession, supervised_callback_session
    from sympy.matrices import _definiteness

    matrix, rhs = Matrix([[1], [I]]), Matrix([[I], [0]])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS if cancelled else 0)
    if cancelled:
        session.cancel()

    def forbidden(*args):
        raise AssertionError("terminal request copied an original matrix")

    monkeypatch.setattr(_definiteness, "_bind_gaussian", forbidden)
    with supervised_callback_session(session):
        with pytest.raises(RuntimeError if cancelled else ValueError, match="cancelled|budget exhausted"):
            matrix.solve_least_squares(rhs)


@native_only
def test_repeated_native_solves_share_the_same_request_allowance():
    from sympy.core import SupervisedCallbackSession

    matrix, rhs = Matrix([[1], [I]])._native_storage, Matrix([[1], [0]])._native_storage
    measured = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    matrix.solve_least_squares(rhs, session=measured)
    cost = measured.consumed_nested_steps
    assert cost > 0
    limited = SupervisedCallbackSession(max_nested_steps=2*cost - 1)
    matrix.solve_least_squares(rhs, session=limited)
    assert limited.consumed_nested_steps == cost
    with pytest.raises(ValueError, match="budget exhausted"):
        matrix.solve_least_squares(rhs, session=limited)
    spent = limited.consumed_nested_steps
    assert cost <= spent <= limited.max_nested_steps and limited.cancelled
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        matrix.solve_least_squares(rhs, session=limited)
    assert limited.consumed_nested_steps == spent


@native_only
def test_late_cancellation_prevents_publication_after_native_verification(monkeypatch):
    from sympy.core import supervised_callback_session
    from sympy.matrices import _definiteness

    matrix, rhs = Matrix([[1], [I]]), Matrix([[I], [0]])
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as measured:
        matrix.solve_least_squares(rhs)
    cost = measured.consumed_nested_steps
    original = _definiteness._unchanged
    cancelled = []
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as session:
        def cancel_at_publication(*args):
            original(*args)
            if session.consumed_nested_steps == cost - 1:
                cancelled.append(True)
                session.cancel()

        monkeypatch.setattr(_definiteness, "_unchanged", cancel_at_publication)
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            matrix.solve_least_squares(rhs)
    assert cancelled


@native_only
@pytest.mark.parametrize("side", ["matrix", "rhs"])
def test_custom_dense_subclasses_require_explicit_override_lanes(side):
    class Custom(Matrix):
        pass

    matrix, rhs = Matrix([[I]]), Matrix([[1]])
    if side == "matrix":
        matrix = Custom([[I]])
    else:
        rhs = Custom([[1]])
    with pytest.raises(NotImplementedError, match="built-in dense Matrix"):
        matrix.solve_least_squares(rhs)
