"""Exact complex thin QR, original-source binding and cumulative requests."""

import pytest

from sympy import (
    Add, Float, I, ImmutableMatrix, Integer, Matrix, Mul, Pow, Rational,
    Symbol, eye, simplify, sqrt, zeros,
)


CLASSES = (Matrix, ImmutableMatrix)
CANDIDATE = hasattr(Matrix([[1]]), "_native_storage")
native_only = pytest.mark.skipif(not CANDIDATE, reason="native source and request contract")
AMPLE_STEPS = 20_000_000


def assert_exact(left, right):
    assert left.shape == right.shape
    assert all(simplify((a - b).expand()) == 0 for a, b in zip(left, right))


def assert_qr(source, q, r, rank):
    assert q.shape == (source.rows, rank)
    assert r.shape == (rank, source.cols)
    assert_exact(q*r, source)
    assert_exact(q.H*q, eye(rank))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("value,phase,magnitude", [
    (I, I, 1),
    (-3*I, -I, 3),
    (3+4*I, Rational(3, 5)+4*I/5, 5),
    (-3-4*I, Rational(-3, 5)-4*I/5, 5),
])
def test_complex_scalar_phase_uses_the_positive_hermitian_norm(cls, value, phase, magnitude):
    source = cls([[value]])
    q, r = source.QRdecomposition()
    assert type(q) is cls
    assert type(r) is cls
    assert_exact(q, Matrix([[phase]]))
    assert_exact(r, Matrix([[magnitude]]))
    assert_qr(source, q, r, 1)
    if CANDIDATE:
        assert source.qr() == (q, r)


@pytest.mark.parametrize("cls", CLASSES)
def test_isotropic_complex_column_does_not_have_zero_hermitian_norm(cls):
    source = cls([[1], [I]])
    q, r = source.QRdecomposition()
    assert_exact(q, Matrix([[sqrt(2)/2], [I*sqrt(2)/2]]))
    assert_exact(r, Matrix([[sqrt(2)]]))
    assert_qr(source, q, r, 1)


@pytest.mark.parametrize("cls", CLASSES)
def test_nonorthogonal_complex_columns_match_hand_derived_projection(cls):
    source = cls([[1, 1+I], [I, 2]])
    q, r = source.QRdecomposition()
    assert_exact(q, Matrix([
        [sqrt(2)/2, (1+3*I)*sqrt(5)/10],
        [I*sqrt(2)/2, (3-I)*sqrt(5)/10],
    ]))
    assert_exact(r, Matrix([[sqrt(2), (1-I)*sqrt(2)/2], [0, sqrt(5)]]))
    assert_qr(source, q, r, 2)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entries,rank", [
    ([[I, 2*I, 0], [1, 2, 0]], 1),
    ([[0, I, 2*I, 0], [0, 1, 2, 1]], 2),
    ([[1, I], [I, -1]], 1),
    ([[1, I], [I, 1]], 2),
    ([[1+I, 2+2*I], [I, 2*I], [3-I, 6-2*I]], 1),
    ([[I, 0, 1]], 1),
    ([[0, -I, 2*I]], 1),
    ([[I, 1], [0, I], [0, 0]], 2),
])
def test_tall_wide_and_rank_deficient_sources_keep_every_original_column(cls, entries, rank):
    source = cls(entries)
    before = source.copy()
    q, r = source.QRdecomposition()
    assert type(q) is cls
    assert type(r) is cls
    assert_qr(source, q, r, rank)
    assert_exact(source, before)


@pytest.mark.parametrize("cls", CLASSES)
def test_gaussian_rational_coefficients_and_global_complex_phase(cls):
    source = cls([[I/2, Rational(1, 3)], [Rational(1, 2), I/3]])
    q, r = source.QRdecomposition()
    assert_qr(source, q, r, 2)
    rotated_q, rotated_r = (I*source).QRdecomposition()
    assert_exact(rotated_q, I*q)
    assert_exact(rotated_r, r)


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_native_unnormalized_api_retains_rational_hermitian_factors(cls):
    from sympy.matrices._qr import gaussian_qr

    source = cls([[1, 1+I], [I, 2]])
    q, r = gaussian_qr(source, normalize=False)
    assert type(q) is cls
    assert type(r) is cls
    assert_exact(q, Matrix([[1, (1+3*I)/2], [I, (3-I)/2]]))
    assert_exact(r, Matrix([[1, (1-I)/2], [0, 1]]))
    assert_exact(q*r, source)
    assert_exact(q.H*q, Matrix([[2, 0], [0, 5]]))


@native_only
def test_defined_held_sources_and_native_only_storage_remain_bound():
    source = Pow(Add(1, I, evaluate=False), -1, evaluate=False)
    matrix = Matrix([[source]])
    retained = matrix._surface
    q, r = matrix.QRdecomposition()
    assert_qr(matrix, q, r, 1)
    assert matrix._surface is retained
    assert matrix._surface[0] is source
    raw = Matrix([[I, 1], [0, I]])
    object.__setattr__(raw, "_surface", None)
    q, r = raw.QRdecomposition()
    assert q._surface is None
    assert r._surface is None
    assert_qr(raw, q, r, 2)


@native_only
@pytest.mark.parametrize("wrapper", [
    lambda pole: Mul(0, pole, evaluate=False),
    lambda pole: Pow(pole, 0, evaluate=False),
])
def test_original_poles_cannot_be_erased_to_zero_rank_or_identity(wrapper):
    zero = Add(I, -I, evaluate=False)
    source = wrapper(Pow(zero, -1, evaluate=False))
    with pytest.raises(ValueError, match="division by zero"):
        Matrix([[source]]).QRdecomposition()


@native_only
@pytest.mark.parametrize("mutation", ["root", "child", "arguments", "surface", "structural"])
def test_retained_complex_cell_mutations_cannot_publish_stale_factors(mutation):
    coefficient = Rational(7, 3)
    child = Mul(coefficient, I, evaluate=False)
    source = Add(1, child, evaluate=False)
    matrix = Matrix([[source]])
    if mutation == "root":
        source._value = Integer(3)._value
    elif mutation == "child":
        coefficient._value = Integer(3)._value
    elif mutation == "arguments":
        child._args = (Integer(3), I)
    elif mutation == "surface":
        child._surface_args = (Integer(3), I)
    else:
        child._struct_args = (Integer(3), I)
    with pytest.raises(RuntimeError, match="retained.*differ"):
        matrix.QRdecomposition()


@native_only
def test_source_mutation_after_binding_is_terminal(monkeypatch):
    from sympy.matrices import _definiteness

    matrix = Matrix([[1, I]])
    bind = _definiteness._bind_gaussian

    def mutate(*args):
        bind(*args)
        matrix[0, 0] = 2

    monkeypatch.setattr(_definiteness, "_bind_gaussian", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        matrix.QRdecomposition()


@native_only
def test_replaced_native_qr_is_rejected_without_executing_it(monkeypatch):
    from sympy.matrices import _qr

    matrix = Matrix([[I]])
    called = []

    def changed(self, **kwargs):
        called.append(True)
        return self, self

    monkeypatch.setattr(type(matrix._native_storage), "gaussian_qr", changed)
    with pytest.raises(NotImplementedError, match="original native interfaces"):
        _qr.gaussian_qr(matrix)
    assert called == []


@native_only
def test_exact_domain_and_resource_refusals_remain_visible():
    for entry in (Symbol("qr_parameter"), Float("1.2"), sqrt(2)+I):
        with pytest.raises(ValueError, match="exact rational entries only"):
            Matrix([[entry]]).QRdecomposition()
    with pytest.raises(ValueError, match="dimension"):
        Matrix(33, 1, [I]*33).QRdecomposition()
    with pytest.raises(ValueError, match="coefficient|bit"):
        Matrix([[Mul(2**4096, I, evaluate=False)]]).QRdecomposition()


@native_only
@pytest.mark.parametrize("shape", [(0, 0), (0, 3), (4, 0), (3, 4)])
def test_empty_and_zero_rank_sources_keep_shape_inside_active_requests(shape):
    from sympy.core import supervised_callback_session

    with supervised_callback_session(max_nested_steps=AMPLE_STEPS):
        q, r = zeros(*shape).QRdecomposition()
    assert q.shape == (shape[0], 0)
    assert r.shape == (0, shape[1])


@native_only
@pytest.mark.parametrize("cancelled", [False, True])
def test_terminal_requests_refuse_before_source_copy(cancelled, monkeypatch):
    from sympy.core import SupervisedCallbackSession, supervised_callback_session
    from sympy.matrices import _definiteness

    matrix = Matrix([[1, I]])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS if cancelled else 0)
    if cancelled:
        session.cancel()

    def forbidden(*args):
        raise AssertionError("terminal request copied a source")

    monkeypatch.setattr(_definiteness, "_bind_gaussian", forbidden)
    with supervised_callback_session(session):
        with pytest.raises(RuntimeError if cancelled else ValueError, match="cancelled|budget exhausted"):
            matrix.QRdecomposition()


@native_only
def test_normalized_native_calls_share_the_caller_request_budget():
    from sympy.core import SupervisedCallbackSession

    native = Matrix([[1, I], [2, 2*I]])._native_storage
    measured = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    native.gaussian_qr(session=measured)
    cost = measured.consumed_nested_steps
    assert cost > 0
    limited = SupervisedCallbackSession(max_nested_steps=2*cost - 1)
    native.gaussian_qr(session=limited)
    assert limited.consumed_nested_steps == cost
    with pytest.raises(ValueError, match="budget exhausted"):
        native.gaussian_qr(session=limited)
    spent = limited.consumed_nested_steps
    assert cost <= spent <= limited.max_nested_steps
    assert limited.cancelled is True
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        native.gaussian_qr(session=limited)
    assert limited.consumed_nested_steps == spent


@native_only
def test_cancellation_after_native_checking_prevents_pair_publication(monkeypatch):
    from sympy.core import supervised_callback_session
    from sympy.matrices import _definiteness

    matrix = Matrix([[1, I]])
    unchanged = _definiteness._unchanged
    visits = []
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as session:
        def cancel_after_native(*args):
            unchanged(*args)
            visits.append(session.consumed_nested_steps)
            if len(visits) == 2:
                session.cancel()

        monkeypatch.setattr(_definiteness, "_unchanged", cancel_after_native)
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            matrix.QRdecomposition()
    assert len(visits) == 2
    assert visits[1] > visits[0]


@native_only
def test_custom_matrix_subclasses_require_an_explicit_override():
    class Custom(Matrix):
        pass

    with pytest.raises(NotImplementedError, match="built-in dense Matrix"):
        Custom([[I]]).QRdecomposition()
