"""Exact complex determinants, original domains and terminal publication.

Ordinary fixtures run unchanged in isolated SymPy1.14.0. Native-only cases
exercise retained owners, the bounded checked lane and cumulative requests.
"""

import pytest

from sympy import (
    Add, Float, I, ImmutableMatrix, ImmutableSparseMatrix, Integer, Matrix,
    Mul, Pow, Rational, SparseMatrix, Symbol, eye,
)


CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)
CANDIDATE = hasattr(Matrix([[1]]), "_native_storage")
native_only = pytest.mark.skipif(not CANDIDATE, reason="native source and request contract")
AMPLE_STEPS = 20_000_000


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("size, expected", [(9, 16 + 16*I), (16, 256), (32, 65536)])
def test_triangular_complex_determinants_reach_thirty_two_axes(cls, size, expected):
    source = cls(size, size, [
        1 + I if row == column else Integer(row < column)
        for row in range(size) for column in range(size)
    ])
    before = source.copy()
    assert (source.det() - expected).expand() == 0
    assert source == before


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entries, expected", [
    ([[0, I], [1, 0]], -I),
    ([[0, 0, I], [1, 0, 0], [0, 2, 0]], 2*I),
    ([[1, I], [I, -1]], 0),
    ([[I, 1], [1, I]], -2),
    ([[I, 1], [2*I, 2]], 0),
    ([[1 + I, 2], [3, 4 - I]], -1 + 3*I),
])
def test_permutation_signs_and_complete_singular_results(cls, entries, expected):
    assert (cls(entries).det() - expected).expand() == 0


@pytest.mark.parametrize("cls", CLASSES)
def test_rational_complex_entries_and_tiny_nonzero_determinants_stay_exact(cls):
    source = cls([[I/7, Rational(1, 7)], [Rational(1, 7), I/7]])
    assert source.det() == Rational(-2, 49)
    tiny = Rational(1, 10**400)
    source = cls([[I*tiny, 1], [0, I*tiny]])
    assert source.det() == -tiny**2


@pytest.mark.parametrize("cls", CLASSES)
def test_empty_determinant_retains_the_multiplicative_identity(cls):
    assert cls(0, 0, []).det() == 1


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entry", [
    Add(1, 1, evaluate=False),
    Mul(0, I, evaluate=False),
    Pow(I, -1, evaluate=False),
])
def test_singleton_determinant_preserves_the_original_held_object(cls, entry):
    source = cls([[entry]])
    assert source.det() is entry
    assert source[0, 0] is entry


@pytest.mark.parametrize("cls", CLASSES)
def test_literal_rational_and_symbolic_sources_keep_existing_domains(cls):
    assert cls(eye(33)).det() == 1
    x = Symbol("legacy_determinant_x")
    assert cls([[x, I], [0, 1]]).det() == x


@pytest.mark.parametrize("entry", [Symbol("subclass_determinant_x"), Float("0.5")])
def test_plain_dense_symbolic_and_float_subclasses_keep_their_legacy_determinant(entry):
    class CustomMatrix(Matrix):
        pass

    assert CustomMatrix([[2, entry], [0, 3]]).det() == 6


@native_only
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("wrapper", [
    lambda pole: Mul(0, pole, evaluate=False),
    lambda pole: Pow(pole, 0, evaluate=False),
])
def test_original_poles_cannot_disappear_behind_zero_or_singleton_determinants(cls, wrapper):
    zero = Add(I, -I, evaluate=False)
    entry = wrapper(Pow(zero, -1, evaluate=False))
    for entries in ([[entry]], [[0, entry], [0, 0]]):
        with pytest.raises(ValueError, match="division by zero"):
            cls(entries).det()


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_changed_retained_complex_children_cannot_publish_stale_determinants(cls, monkeypatch):
    coefficient = Rational(47, 29)
    entry = Mul(coefficient, I, evaluate=False)
    source = cls([[entry, 1], [0, I]])
    monkeypatch.setattr(coefficient, "_value", Integer(3)._value)
    with pytest.raises(RuntimeError, match="retained.*differ"):
        source.det()


@native_only
@pytest.mark.parametrize("cls", [Matrix, SparseMatrix])
def test_source_mutation_during_binding_is_terminal(cls, monkeypatch):
    from sympy.matrices import _definiteness

    source = cls([[1 + I, 1], [0, I]])
    original = _definiteness._bind_gaussian

    def mutate_after_binding(*args):
        original(*args)
        source[0, 0] = 3

    monkeypatch.setattr(_definiteness, "_bind_gaussian", mutate_after_binding)
    with pytest.raises(RuntimeError, match="mutated"):
        source.det()


@native_only
@pytest.mark.parametrize("cls", [Matrix, SparseMatrix])
def test_source_mutation_after_native_checking_prevents_scalar_publication(cls, monkeypatch):
    from sympy.matrices import _determinant

    source = cls([[I, 1], [0, I]])
    original = _determinant._wrap
    visited = []

    def mutate_during_publication(value):
        result = original(value)
        visited.append(True)
        source[0, 0] = 3
        return result

    monkeypatch.setattr(_determinant, "_wrap", mutate_during_publication)
    with pytest.raises(RuntimeError, match="mutated"):
        source.det()
    assert visited == [True]


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_replaced_native_determinant_is_refused_without_invocation(cls, monkeypatch):
    from sympy.matrices import _pseudoinverse

    source = cls([[I]])
    called = []

    def forbidden(*args, **kwargs):
        called.append(True)
        raise AssertionError("replaced native determinant executed")

    monkeypatch.setattr(_pseudoinverse._NATIVE, "gaussian_det", forbidden)
    with pytest.raises(NotImplementedError, match="original native interfaces"):
        source.det()
    assert called == []


@native_only
def test_repeated_native_determinants_keep_one_cumulative_request():
    from sympy.core import SupervisedCallbackSession

    source = Matrix([[I, 1], [0, I]])._native_storage
    measured = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    source.gaussian_det(session=measured)
    cost = measured.consumed_nested_steps
    assert cost > 0
    limited = SupervisedCallbackSession(max_nested_steps=2*cost - 1)
    source.gaussian_det(session=limited)
    assert limited.consumed_nested_steps == cost
    with pytest.raises(ValueError, match="budget exhausted"):
        source.gaussian_det(session=limited)
    spent = limited.consumed_nested_steps
    assert cost <= spent <= limited.max_nested_steps
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        source.gaussian_det(session=limited)
    assert limited.consumed_nested_steps == spent


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_cancelled_public_request_stops_before_binding_original_scalars(cls, monkeypatch):
    from sympy.core import SupervisedCallbackSession, supervised_callback_session
    from sympy.matrices import _definiteness

    source = cls([[I]])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    session.cancel()

    def forbidden(*args, **kwargs):
        raise AssertionError("cancelled determinant request copied original scalars")

    monkeypatch.setattr(_definiteness, "_bind_gaussian", forbidden)
    with supervised_callback_session(session):
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            source.det()


@native_only
def test_cancellation_after_native_checking_prevents_scalar_publication(monkeypatch):
    from sympy.core import supervised_callback_session
    from sympy.matrices import _determinant

    source = Matrix([[I, 1], [0, I]])
    original = _determinant._wrap
    visited = []
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as session:
        def cancel_during_publication(value):
            result = original(value)
            visited.append(session.consumed_nested_steps)
            session.cancel()
            return result

        monkeypatch.setattr(_determinant, "_wrap", cancel_during_publication)
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            source.det()
    assert len(visited) == 1 and visited[0] > 0


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_complex_dimension_bounds_refuse_before_larger_factor_materialization(cls):
    source = cls(33, 33, [
        I if row == column else 0
        for row in range(33) for column in range(33)
    ])
    with pytest.raises(ValueError, match="dimension|limit|32"):
        source.det()


@native_only
@pytest.mark.parametrize("cls", [SparseMatrix, ImmutableSparseMatrix])
def test_large_sparse_complex_sources_refuse_before_dense_allocation(cls):
    source = cls(100_000, 100_000, {(0, 0): I})
    with pytest.raises(ValueError, match="dimension|limit|32"):
        source.det()


@native_only
def test_empty_native_determinant_has_terminal_budget_and_cancel_checks():
    from sympy.core import SupervisedCallbackSession

    source = Matrix(0, 0, [])._native_storage
    limited = SupervisedCallbackSession(max_nested_steps=1)
    with pytest.raises(ValueError, match="budget exhausted"):
        source.gaussian_det(session=limited)
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        source.gaussian_det(session=limited)
