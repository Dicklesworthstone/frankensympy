"""Complete integer column lattices and source-bound Hermite normal forms."""

from unittest.mock import patch

import pytest
import sympy as s
from sympy.core import SupervisedCallbackSession, _native, supervised_callback_session
from sympy.matrices import normalforms
from sympy.matrices.normalforms import hermite_normal_decomposition, hermite_normal_form


CLASSES = (s.Matrix, s.ImmutableMatrix, s.SparseMatrix, s.ImmutableSparseMatrix)
AMPLE_STEPS = 10_000_000


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("values,expected", [
    ([[4, 6], [3, 9]], [[6, 4], [0, 3]]),
    ([[12, 6, 4], [3, 9, 6], [2, 16, 14]], [[10, 0, 2], [0, 15, 3], [0, 0, 2]]),
    ([[2, 4, 6], [4, 8, 12]], [[2], [4]]),
    ([[2, 4], [4, 8], [0, 0]], [[2], [4], [0]]),
    ([[0, 2], [6, 4], [0, 4]], [[0, 2], [6, 4], [0, 4]]),
    ([[0, -12, 18, 30]], [[6]]),
    ([[0], [-12], [18], [30]], [[0], [-12], [18], [30]]),
    ([[-2, 1], [0, 3]], [[2, 1], [0, 3]]),
    ([[6, -1], [0, 3]], [[6, 5], [0, 3]]),
])
def test_canonical_forms_and_complete_lattice_equivalence(cls, values, expected):
    # Expected forms were independently checked in isolated SymPy 1.14.0.
    # Production and normal tests never import an upstream oracle.
    source = cls(values)
    form, right, inverse = hermite_normal_decomposition(source)
    assert type(form) is type(right) is type(inverse) is cls
    assert form == s.Matrix(expected)
    assert form.shape == (source.rows, len(expected[0]))
    assert source * right == s.zeros(source.rows, source.cols - form.cols).row_join(form)
    assert right * inverse == inverse * right == s.eye(source.cols)
    assert all(entry.is_integer is True for matrix in (form, right, inverse) for entry in matrix)
    assert source.hermite_normal_decomposition() == (form, right, inverse)
    standard = hermite_normal_form(source)
    assert type(standard) is s.Matrix
    assert standard == s.Matrix(expected)
    assert source == cls(values)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("shape", [(0, 0), (0, 4), (3, 0), (3, 5), (5, 3)])
def test_zero_lattices_retain_all_rows_and_no_spurious_basis_columns(cls, shape):
    entries = {} if cls in (s.SparseMatrix, s.ImmutableSparseMatrix) else [0] * (shape[0] * shape[1])
    source = cls(*shape, entries)
    form, right, inverse = hermite_normal_decomposition(source)
    assert form.shape == (shape[0], 0)
    assert right == inverse == s.eye(shape[1])
    assert hermite_normal_form(source).shape == (shape[0], 0)


@pytest.mark.parametrize("cls", CLASSES)
def test_column_lattice_is_distinct_from_the_rational_span(cls):
    source = cls([[2, 0], [0, 3]])
    assert hermite_normal_form(source) == s.diag(2, 3)
    assert hermite_normal_form(source) != s.eye(2)
    transform = cls([[1, -7], [0, 1]])
    assert hermite_normal_form(source * transform) == hermite_normal_form(source)
    assert hermite_normal_form(source * 2) == s.diag(4, 6)


@pytest.mark.parametrize("cls", CLASSES)
def test_large_exact_integers_and_native_only_storage(cls):
    large = 1 << 90
    assert hermite_normal_form(cls([[large, -1], [0, large + 1]])) == s.Matrix([[large, large - 1], [0, large + 1]])
    if cls in (s.SparseMatrix, s.ImmutableSparseMatrix):
        native = _native.SparseMatrix(2, 2, {(0, 0): _native.py_integer(4), (0, 1): _native.py_integer(6), (1, 0): _native.py_integer(3), (1, 1): _native.py_integer(9)})
    else:
        native = _native.Matrix(2, 2, [_native.py_integer(n) for n in (4, 6, 3, 9)])
    source = cls(native)
    assert source._surface is None
    assert hermite_normal_form(source) == s.Matrix([[6, 4], [0, 3]])


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entry", [
    s.Rational(1, 2), s.Float(0), s.Symbol("integer", integer=True),
    s.Add(1, -1, evaluate=False),
    s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False),
])
def test_noninteger_or_held_sources_never_disappear_in_rank_shortcuts(cls, entry):
    with pytest.raises((NotImplementedError, ValueError)):
        hermite_normal_form(cls([[entry, 0]]))


@pytest.mark.parametrize("cls", CLASSES)
def test_admission_limits_precede_source_copies(cls):
    oversized = cls([[s.Integer(1 << 4096)]])
    wide = cls(0, 65, {}) if cls in (s.SparseMatrix, s.ImmutableSparseMatrix) else cls(0, 65, [])
    with patch.object(normalforms, "_rational_source_snapshot", side_effect=AssertionError("premature copy")):
        for source in (oversized, wide):
            with pytest.raises(ValueError, match="height|4096|64"):
                hermite_normal_form(source)


def test_optional_algorithm_hints_and_custom_sources_do_not_invoke_callbacks():
    effects = []

    class Hint:
        def __bool__(self):
            effects.append("truth")
            raise AssertionError("unexpected callback")

        def __int__(self):
            effects.append("integer")
            raise AssertionError("unexpected callback")

    class CustomMatrix(s.Matrix):
        def __getitem__(self, index):
            effects.append("index")
            raise AssertionError("unexpected callback")

    with pytest.raises(NotImplementedError, match="built-in"):
        hermite_normal_form(CustomMatrix([[2]]))
    for hint in (Hint(), 2, 0, -2):
        with pytest.raises(NotImplementedError, match="determinant hint"):
            hermite_normal_form(s.Matrix([[2]]), D=hint)
    with pytest.raises(TypeError, match="boolean"):
        hermite_normal_form(s.Matrix([[2]]), check_rank=Hint())
    assert hermite_normal_form(s.Matrix([[2]]), check_rank=True) == s.Matrix([[2]])
    assert effects == []


@pytest.mark.parametrize("cls", CLASSES)
def test_changed_source_owners_and_stale_cells_are_rejected(cls, monkeypatch):
    source = cls([[2]])
    source._surface = {(0, 0): s.Integer(3)} if cls in (s.SparseMatrix, s.ImmutableSparseMatrix) else (s.Integer(3),)
    with pytest.raises(RuntimeError, match="retained cells differ"):
        hermite_normal_form(source)
    source = cls([[2]])

    def forbidden(*args, **kwargs):
        raise AssertionError("changed constructor invoked")

    monkeypatch.setattr(cls, "__init__", forbidden)
    with pytest.raises(NotImplementedError, match="original"):
        hermite_normal_form(source)


def test_mutation_after_verification_prevents_decomposition_publication(monkeypatch):
    source = s.Matrix([[4, 6], [3, 9]])
    original = normalforms._check_rational_source
    calls = []

    def mutate(snapshot):
        calls.append(1)
        if len(calls) == 3:
            source[0, 0] = 5
        original(snapshot)

    monkeypatch.setattr(normalforms, "_check_rational_source", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        hermite_normal_decomposition(source)
    assert len(calls) == 3


@pytest.mark.parametrize("cls", (s.Matrix, s.SparseMatrix))
@pytest.mark.parametrize("cancelled", [False, True])
def test_terminal_requests_stop_before_any_source_copy(cls, cancelled):
    source = cls([[4, 6], [3, 9]])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS if cancelled else 0)
    if cancelled:
        session.cancel()
    with patch.object(normalforms, "_rational_source_snapshot", side_effect=AssertionError("terminal source copy")):
        with supervised_callback_session(session):
            with pytest.raises(RuntimeError if cancelled else ValueError, match="cancelled|budget exhausted"):
                hermite_normal_decomposition(source)


@pytest.mark.parametrize("cls", (s.Matrix, s.SparseMatrix))
def test_generation_verification_and_lifting_share_the_same_active_budget(cls):
    source = cls([[4, 6], [3, 9]])
    native = source._native_storage
    observed = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    native.hermite_normal_decomposition(session=observed)
    cost = observed.consumed_nested_steps
    assert cost > 0
    session = SupervisedCallbackSession(max_nested_steps=2 * cost - 1)
    native.hermite_normal_decomposition(session=session)
    assert session.consumed_nested_steps == cost
    with pytest.raises(ValueError, match="budget exhausted"):
        native.hermite_normal_decomposition(session=session)
    consumed = session.consumed_nested_steps
    assert session.cancelled
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        native.hermite_normal_decomposition(session=session)
    assert session.consumed_nested_steps == consumed

    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as measured:
        hermite_normal_decomposition(source)
    assert measured.consumed_nested_steps > cost


def test_late_cancellation_does_not_publish_accepted_native_result(monkeypatch):
    source = s.Matrix([[4, 6], [3, 9]])
    original = normalforms._check_rational_source
    calls = []
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as session:
        def cancel(snapshot):
            original(snapshot)
            calls.append(1)
            if len(calls) == 4:
                session.cancel()

        monkeypatch.setattr(normalforms, "_check_rational_source", cancel)
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            hermite_normal_decomposition(source)
    assert len(calls) == 4


def test_request_identity_cannot_change_during_publication(monkeypatch):
    source = s.Matrix([[4, 6], [3, 9]])
    original = normalforms._check_rational_source
    calls = []
    foreign = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)

    def switch_request(snapshot):
        original(snapshot)
        calls.append(1)
        if len(calls) == 3:
            monkeypatch.setattr(normalforms, "_active_callback_session", lambda: foreign)

    monkeypatch.setattr(normalforms, "_check_rational_source", switch_request)
    with pytest.raises(RuntimeError, match="request changed"):
        hermite_normal_decomposition(source)
    assert foreign.consumed_nested_steps == 0
