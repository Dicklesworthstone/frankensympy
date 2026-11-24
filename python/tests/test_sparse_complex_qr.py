"""Sparse exact thin QR with pinned classes and original-source admission.

Ordinary fixtures also run against the isolated SymPy 1.14.0 oracle. Native
cases cover the bounded sparse source, publication and cumulative request.
"""

import pytest

from sympy import (
    Add, I, ImmutableSparseMatrix, Integer, Matrix, Mul, Pow, Rational,
    SparseMatrix, eye, simplify, sqrt,
)


CLASSES = (SparseMatrix, ImmutableSparseMatrix)
CANDIDATE = hasattr(Matrix([[1]]), "_native_storage")
native_only = pytest.mark.skipif(not CANDIDATE, reason="native source and request contract")
AMPLE_STEPS = 20_000_000


def assert_exact(left, right):
    assert left.shape == right.shape
    assert all(simplify((a - b).expand()) == 0 for a, b in zip(left, right))


def assert_qr(source, q, r, rank):
    assert type(q) is type(source)
    assert type(r) is type(source)
    assert q.shape == (source.rows, rank)
    assert r.shape == (rank, source.cols)
    assert_exact(q*r, source)
    assert_exact(q.H*q, eye(rank))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entries,rank", [
    ([[1, 2], [3, 4], [5, 6]], 2),
    ([[0, 1, 2, 0], [0, 2, 4, 1]], 2),
    ([[I, 2*I, 0], [1, 2, 0]], 1),
    ([[1, I], [I, -1]], 1),
    ([[1, I], [I, 1]], 2),
    ([[1+I, 2+2*I], [I, 2*I], [3-I, 6-2*I]], 1),
    ([[0, -I, 2*I]], 1),
    ([[I, 1], [0, I], [0, 0]], 2),
])
def test_sparse_rectangular_and_deficient_qr_keep_complete_rank(cls, entries, rank):
    source = cls(entries)
    before = source.copy()
    q, r = source.QRdecomposition()
    assert_qr(source, q, r, rank)
    assert_exact(source, before)
    if CANDIDATE:
        assert source.qr() == (q, r)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("value,phase,magnitude", [
    (I, I, 1), (-3*I, -I, 3),
    (3+4*I, Rational(3, 5)+4*I/5, 5),
])
def test_sparse_scalar_qr_has_positive_norm_and_exact_phase(cls, value, phase, magnitude):
    source = cls([[value]])
    q, r = source.QRdecomposition()
    assert_exact(q, Matrix([[phase]]))
    assert_exact(r, Matrix([[magnitude]]))
    assert_qr(source, q, r, 1)


@pytest.mark.parametrize("cls", CLASSES)
def test_sparse_complex_projection_uses_conjugation(cls):
    source = cls([[1, 1+I], [I, 2]])
    q, r = source.QRdecomposition()
    assert_exact(q, Matrix([
        [sqrt(2)/2, (1+3*I)*sqrt(5)/10],
        [I*sqrt(2)/2, (3-I)*sqrt(5)/10],
    ]))
    assert_exact(r, Matrix([[sqrt(2), (1-I)*sqrt(2)/2], [0, sqrt(5)]]))
    assert_qr(source, q, r, 2)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (4, 0), (3, 4)])
def test_sparse_empty_and_zero_rank_factors_retain_shape(cls, shape):
    source = cls(*shape, {})
    q, r = source.QRdecomposition()
    assert_qr(source, q, r, 0)


@native_only
def test_original_sparse_owner_is_admitted_without_a_public_dense_conversion(monkeypatch):
    source = SparseMatrix([[1], [I]])

    def forbidden(*args):
        raise AssertionError("source escaped through the public dense conversion")

    monkeypatch.setattr(SparseMatrix, "to_dense", forbidden)
    q, r = source.QRdecomposition()
    assert_exact(q, Matrix([[sqrt(2)/2], [I*sqrt(2)/2]]))
    assert_exact(r, Matrix([[sqrt(2)]]))
    assert_qr(source, q, r, 1)


@native_only
def test_sparse_held_source_and_native_only_storage_remain_composable():
    value = Pow(Add(1, I, evaluate=False), -1, evaluate=False)
    source = SparseMatrix([[value]])
    retained = source._surface
    q, r = source.QRdecomposition()
    assert_qr(source, q, r, 1)
    assert source._surface is retained and retained[0, 0] is value
    raw = ImmutableSparseMatrix([[I, 1], [0, I]])
    object.__setattr__(raw, "_surface", None)
    q, r = raw.QRdecomposition()
    assert q._surface is None and r._surface is None
    assert_qr(raw, q, r, 2)


@native_only
@pytest.mark.parametrize("wrapper", [
    lambda pole: Mul(0, pole, evaluate=False),
    lambda pole: Pow(pole, 0, evaluate=False),
])
def test_hidden_sparse_poles_are_not_erased_before_admission(wrapper):
    zero = Add(I, -I, evaluate=False)
    source = SparseMatrix([[wrapper(Pow(zero, -1, evaluate=False))]])
    with pytest.raises(ValueError, match="division by zero"):
        source.QRdecomposition()


@native_only
def test_sparse_scalar_mutation_refuses_stale_factors(monkeypatch):
    coefficient = Rational(19, 13)
    source = SparseMatrix([[Mul(coefficient, I, evaluate=False)]])
    monkeypatch.setattr(coefficient, "_value", Integer(3)._value)
    with pytest.raises(RuntimeError, match="retained.*differ"):
        source.QRdecomposition()


@native_only
def test_mutation_after_sparse_source_binding_prevents_publication(monkeypatch):
    from sympy.matrices import _definiteness

    source = SparseMatrix([[1, I]])
    original = _definiteness._bind_gaussian

    def mutate(*args):
        original(*args)
        source[0, 0] = 2

    monkeypatch.setattr(_definiteness, "_bind_gaussian", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        source.QRdecomposition()


@native_only
@pytest.mark.parametrize("side", ["source", "output"])
def test_replaced_native_conversions_refuse_without_callbacks(side, monkeypatch):
    source = SparseMatrix([[I]])
    native = source._native_storage if side == "source" else Matrix([[I]])._native_storage
    name = "to_dense" if side == "source" else "to_sparse"
    called = []

    def changed(*args):
        called.append(True)
        raise AssertionError("replaced conversion was invoked")

    monkeypatch.setattr(type(native), name, changed)
    with pytest.raises(NotImplementedError, match="original.*native interfaces"):
        source.QRdecomposition()
    assert called == []


@native_only
def test_sparse_source_bound_is_checked_before_rational_copy(monkeypatch):
    from sympy.matrices import _definiteness

    source = SparseMatrix(33, 1, {(0, 0): 1})

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized sparse source was copied")

    monkeypatch.setattr(_definiteness, "_rational_source_snapshot", forbidden)
    with pytest.raises(ValueError, match="dimension|32"):
        source.QRdecomposition()


@native_only
def test_sparse_qr_charges_one_cumulative_public_request():
    from sympy.core import SupervisedCallbackSession, supervised_callback_session

    source = SparseMatrix([[1], [I]])
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as measured:
        source.QRdecomposition()
    cost = measured.consumed_nested_steps
    assert cost > 0
    limited = SupervisedCallbackSession(max_nested_steps=2*cost - 1)
    with supervised_callback_session(limited):
        source.QRdecomposition()
        with pytest.raises(ValueError, match="budget exhausted"):
            source.QRdecomposition()
    assert cost <= limited.consumed_nested_steps <= limited.max_nested_steps


@native_only
def test_cancellation_between_factor_lifts_prevents_returning_the_pair(monkeypatch):
    from sympy.core import supervised_callback_session
    from sympy.matrices import _qr

    source = SparseMatrix([[1], [I]])
    original = _qr._publish
    published = []
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as session:
        def cancel_after_first(*args):
            result = original(*args)
            published.append(result)
            session.cancel()
            return result

        monkeypatch.setattr(_qr, "_publish", cancel_after_first)
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            source.QRdecomposition()
    assert len(published) == 1


@native_only
def test_sparse_subclasses_need_a_supervised_override_lane():
    class Custom(SparseMatrix):
        pass

    with pytest.raises(NotImplementedError, match="built-in"):
        Custom([[I]]).QRdecomposition()
