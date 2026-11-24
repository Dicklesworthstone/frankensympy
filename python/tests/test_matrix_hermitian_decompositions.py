"""Checked Hermitian LDL/Cholesky through every built-in matrix class.

Ordinary exact positive-definite fixtures can also run against SymPy 1.14.
Native-only tests cover the bounded checker, original source and request.
"""

import pytest

from sympy import (
    Add, I, ImmutableMatrix, ImmutableSparseMatrix, Integer, Matrix, Mul, Pow,
    Rational, SparseMatrix, Symbol, eye, simplify, sqrt,
)


CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)
CANDIDATE = hasattr(Matrix([[1]]), "_native_storage")
native_only = pytest.mark.skipif(not CANDIDATE, reason="native source and request contract")
AMPLE_STEPS = 20_000_000


def assert_exact(left, right):
    assert left.shape == right.shape
    assert all(simplify((a - b).expand()) == 0 for a, b in zip(left, right))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entries, lower, diagonal", [
    ([[4, 12, -16], [12, 37, -43], [-16, -43, 98]],
     [[1, 0, 0], [3, 1, 0], [-4, 5, 1]], [4, 1, 9]),
    ([[2, 2 - 2*I], [2 + 2*I, 7]], [[1, 0], [1 + I, 1]], [2, 3]),
    ([[Rational(1, 2), -I/3], [I/3, 1]], [[1, 0], [2*I/3, 1]],
     [Rational(1, 2), Rational(7, 9)]),
])
def test_exact_hermitian_ldl_matches_hand_computed_factors(cls, entries, lower, diagonal):
    source = cls(entries)
    l, d = source.LDLdecomposition()
    assert type(l) is cls and type(d) is cls
    assert_exact(l, Matrix(lower))
    assert_exact(d, Matrix.diag(*diagonal))
    assert_exact(l*d*l.H, source)
    if CANDIDATE:
        alias_l, alias_d = source.ldl()
        assert_exact(alias_l, l)
        assert_exact(alias_d, d)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entries, expected", [
    ([[4, 12, -16], [12, 37, -43], [-16, -43, 98]],
     [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]),
    ([[2, 2 - 2*I], [2 + 2*I, 7]], [[sqrt(2), 0], [(1 + I)*sqrt(2), sqrt(3)]]),
    ([[Rational(1, 2), -I/3], [I/3, 1]],
     [[sqrt(2)/2, 0], [I*sqrt(2)/3, sqrt(7)/3]]),
])
def test_cholesky_uses_the_positive_hermitian_branch(cls, entries, expected):
    source = cls(entries)
    factor = source.cholesky()
    assert type(factor) is cls
    assert_exact(factor, Matrix(expected))
    assert_exact(factor*factor.H, source)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("size", [0, 1, 5])
def test_empty_scalar_and_identity_factors_preserve_shape_and_class(cls, size):
    source = cls(eye(size))
    l, d = source.LDLdecomposition()
    cholesky = source.cholesky()
    assert type(l) is type(d) is type(cholesky) is cls
    assert_exact(l, source)
    assert_exact(d, source)
    assert_exact(cholesky, source)


@pytest.mark.parametrize("cls", [Matrix, ImmutableMatrix])
def test_dense_transpose_options_remain_available_for_indefinite_rational_forms(cls):
    source = cls([[1, 2], [2, 1]])
    l, d = source.LDLdecomposition(hermitian=False)
    assert_exact(d, Matrix.diag(1, -3))
    assert_exact(l*d*l.T, source)
    factor = source.cholesky(hermitian=False)
    assert_exact(factor*factor.T, source)


@pytest.mark.parametrize("cls", [Matrix, ImmutableMatrix])
def test_standalone_symbolic_cholesky_keeps_its_established_lane(cls):
    positive = Symbol("hermitian_positive", positive=True)
    source = cls([[positive]])
    factor = source.cholesky()
    assert type(factor) is cls
    assert_exact(factor, Matrix([[sqrt(positive)]]))


@native_only
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("operation", ["LDLdecomposition", "cholesky"])
@pytest.mark.parametrize("entries", [
    [[0]],
    [[-1]],
    [[1, 1], [1, 1]],
    [[1, 2], [2, 1]],
    [[0, 0], [0, 1]],
])
def test_nonpositive_pivots_never_publish_nan_or_zero_divisor_factors(cls, operation, entries):
    with pytest.raises(ValueError, match="positive"):
        getattr(cls(entries), operation)()


@native_only
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("operation", ["LDLdecomposition", "cholesky"])
def test_nonhermitian_complex_forms_are_rejected(cls, operation):
    with pytest.raises(ValueError, match="Hermitian"):
        getattr(cls([[1, I], [I, 2]]), operation)()


@native_only
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("operation", ["LDLdecomposition", "cholesky"])
def test_exact_tiny_positive_pivots_do_not_use_float_signs(cls, operation):
    tiny = Rational(1, 10**40)
    source = cls([[1, 1], [1, 1 + tiny]])
    result = getattr(source, operation)()
    if operation == "LDLdecomposition":
        l, d = result
        assert d[1, 1] == tiny
        assert_exact(l*d*l.H, source)
    else:
        assert result[1, 1] == Rational(1, 10**20)
        assert_exact(result*result.H, source)


@native_only
@pytest.mark.parametrize("operation", ["LDLdecomposition", "cholesky"])
@pytest.mark.parametrize("wrapper", [
    lambda pole: Mul(0, pole, evaluate=False),
    lambda pole: Pow(pole, 0, evaluate=False),
])
def test_original_held_poles_cannot_disappear_in_hermitian_admission(operation, wrapper):
    zero = Add(I, -I, evaluate=False)
    pole = Pow(zero, -1, evaluate=False)
    source = Matrix([[Add(1, wrapper(pole), evaluate=False)]])
    with pytest.raises(ValueError, match="division by zero"):
        getattr(source, operation)()


@native_only
@pytest.mark.parametrize("operation", ["LDLdecomposition", "cholesky"])
def test_defined_held_cells_remain_owned_by_the_original_matrix(operation):
    entry = Pow(Add(1, 1, evaluate=False), -1, evaluate=False)
    source = Matrix([[entry, -I/4], [I/4, 1]])
    retained = source._surface
    getattr(source, operation)()
    assert source._surface is retained and source._surface[0] is entry


@native_only
@pytest.mark.parametrize("operation", ["LDLdecomposition", "cholesky"])
def test_mutated_retained_scalar_cannot_publish_stale_factors(operation, monkeypatch):
    coefficient = Rational(31, 19)
    entry = Add(coefficient, Mul(0, I, evaluate=False), evaluate=False)
    source = Matrix([[entry]])
    monkeypatch.setattr(coefficient, "_value", Integer(3)._value)
    with pytest.raises(RuntimeError, match="retained.*differ"):
        getattr(source, operation)()


@native_only
@pytest.mark.parametrize("operation, native_name", [
    ("LDLdecomposition", "gaussian_ldl"),
    ("cholesky", "gaussian_cholesky"),
])
def test_replaced_native_factorization_descriptor_is_not_invoked(operation, native_name, monkeypatch):
    source = Matrix([[2, -I], [I, 2]])
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("replacement native factorization ran")

    monkeypatch.setattr(type(source._native_storage), native_name, forbidden)
    with pytest.raises(NotImplementedError, match="original native interfaces"):
        getattr(source, operation)()
    assert calls == []


@native_only
@pytest.mark.parametrize("operation", ["LDLdecomposition", "cholesky"])
def test_source_mutation_during_factor_publication_is_detected(operation, monkeypatch):
    from sympy.matrices import _hermitian

    source = Matrix([[2, -I], [I, 2]])
    original = _hermitian._publish

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        source[0, 0] = 3
        return result

    monkeypatch.setattr(_hermitian, "_publish", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        getattr(source, operation)()


@native_only
@pytest.mark.parametrize("native_name", ["gaussian_ldl", "gaussian_cholesky"])
def test_repeated_native_calls_consume_one_cumulative_request(native_name):
    from sympy.core import SupervisedCallbackSession

    source = Matrix([[2, -I], [I, 2]])._native_storage
    measured = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    getattr(source, native_name)(session=measured)
    cost = measured.consumed_nested_steps
    assert cost > 0
    limited = SupervisedCallbackSession(max_nested_steps=2*cost - 1)
    getattr(source, native_name)(session=limited)
    with pytest.raises(ValueError, match="budget exhausted"):
        getattr(source, native_name)(session=limited)
    spent = limited.consumed_nested_steps
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        getattr(source, native_name)(session=limited)
    assert limited.consumed_nested_steps == spent


@native_only
@pytest.mark.parametrize("operation", ["LDLdecomposition", "cholesky"])
def test_cancelled_request_stops_before_original_scalar_copy(operation, monkeypatch):
    from sympy.core import SupervisedCallbackSession, supervised_callback_session
    from sympy.matrices import _definiteness

    source = Matrix([[2, -I], [I, 2]])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    session.cancel()

    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled request copied retained scalars")

    monkeypatch.setattr(_definiteness, "_bind_gaussian", forbidden)
    with supervised_callback_session(session):
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            getattr(source, operation)()


@native_only
@pytest.mark.parametrize("operation", ["LDLdecomposition", "cholesky"])
def test_large_standalone_literal_rational_dense_lane_remains_available(operation):
    source = eye(33)
    result = getattr(source, operation)()
    for factor in result if operation == "LDLdecomposition" else (result,):
        assert_exact(factor, source)
    negative = eye(33)
    negative[32, 32] = -1
    with pytest.raises(ValueError, match="positive"):
        getattr(negative, operation)()


@native_only
def test_native_true_lane_does_not_claim_sparse_transpose_or_callback_overrides():
    source = SparseMatrix([[1, 2], [2, 1]])
    with pytest.raises(NotImplementedError, match="hermitian=True"):
        source.LDLdecomposition(hermitian=False)
    with pytest.raises(NotImplementedError, match="hermitian=True"):
        source.cholesky(hermitian=False)
    for operation in ("LDLdecomposition", "cholesky"):
        with pytest.raises(TypeError, match="bool"):
            getattr(Matrix([[1]]), operation)(hermitian=1)


@native_only
@pytest.mark.parametrize("operation", ["LDLdecomposition", "cholesky"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_active_transpose_request_cannot_enter_an_unmetered_legacy_lane(operation, cancelled):
    from sympy.core import SupervisedCallbackSession, supervised_callback_session

    source = Matrix([[1, 2], [2, 1]])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    if cancelled:
        session.cancel()
    with supervised_callback_session(session):
        if cancelled:
            with pytest.raises(RuntimeError, match="cancelled|tainted"):
                getattr(source, operation)(hermitian=False)
        else:
            with pytest.raises(NotImplementedError, match="supervised lane"):
                getattr(source, operation)(hermitian=False)
