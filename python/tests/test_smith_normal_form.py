"""Canonical integer normal forms, complete transforms and source admission."""

from unittest.mock import patch

import pytest
import sympy as s
from sympy.core import SupervisedCallbackSession, _native, supervised_callback_session
from sympy.matrices import normalforms
from sympy.matrices.normalforms import smith_normal_decomposition, smith_normal_form


CLASSES = (s.Matrix, s.ImmutableMatrix, s.SparseMatrix, s.ImmutableSparseMatrix)
AMPLE_STEPS = 10_000_000


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("values,expected", [
    ([[-6]], [[6]]),
    ([[6, 0], [0, 10]], [[2, 0], [0, 30]]),
    ([[12, 6, 4], [3, 9, 6], [2, 16, 14]], [[1, 0, 0], [0, 10, 0], [0, 0, 30]]),
    ([[2, 4, 4], [6, 6, 12]], [[2, 0, 0], [0, 6, 0]]),
    ([[2, 4, 6], [4, 8, 12], [0, 0, 0]], [[2, 0, 0], [0, 0, 0], [0, 0, 0]]),
])
def test_full_integer_equivalence_and_canonical_invariant_factors(cls, values, expected):
    source = cls(values)
    diagonal, left, right = smith_normal_decomposition(source)
    assert type(diagonal) is type(left) is type(right) is cls
    assert diagonal == cls(expected)
    assert diagonal == left * source * right
    assert abs(left.det()) == abs(right.det()) == 1
    assert left.inv() * left == s.eye(source.rows)
    assert right.inv() * right == s.eye(source.cols)
    assert source.smith_normal_decomposition() == (diagonal, left, right)
    standard = smith_normal_form(source)
    assert type(standard) is s.Matrix
    assert standard == s.Matrix(expected)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("shape", [(0, 0), (0, 4), (3, 0), (3, 5)])
def test_zero_and_empty_rectangular_forms_preserve_shape_and_basis_dimensions(cls, shape):
    source = cls(*shape, {}) if cls in (s.SparseMatrix, s.ImmutableSparseMatrix) else cls(*shape, [0] * (shape[0] * shape[1]))
    diagonal, left, right = smith_normal_decomposition(source)
    assert diagonal.shape == shape
    assert left.shape == (shape[0], shape[0])
    assert right.shape == (shape[1], shape[1])
    assert diagonal == source
    assert left == s.eye(shape[0])
    assert right == s.eye(shape[1])
    assert smith_normal_form(source).shape == shape


def test_diagonal_output_matches_pinned_sympy_114_signed_rectangular_fixtures():
    # Independently executed under isolated SymPy 1.14.0 during development.
    # Keep the expected observations here so normal tests need no upstream
    # installation, subprocess import, or runtime oracle dependency.
    cases = [
        ([[-12]], [[12]]),
        ([[6, 0], [0, 10]], [[2, 0], [0, 30]]),
        ([[-6, 0, 0], [0, 10, 0], [0, 0, -15]], [[1, 0, 0], [0, 30, 0], [0, 0, 30]]),
        ([[12, 6, 4], [3, 9, 6], [2, 16, 14]], [[1, 0, 0], [0, 10, 0], [0, 0, 30]]),
        ([[2, 4, 4], [6, 6, 12]], [[2, 0, 0], [0, 6, 0]]),
        ([[0, -12, 18, 30]], [[6, 0, 0, 0]]),
        ([[0], [-12], [18], [30]], [[6], [0], [0], [0]]),
        ([[2, 4], [4, 8]], [[2, 0], [0, 0]]),
    ]
    for values, expected in cases:
        for cls in CLASSES:
            d = smith_normal_form(cls(values))
            assert type(d) is s.Matrix
            assert d == s.Matrix(expected)


@pytest.mark.parametrize("cls", CLASSES)
def test_native_only_sources_and_exact_large_integer_coefficients(cls):
    large = 1 << 80
    source = cls([[large, 0], [0, large + 1]])
    assert smith_normal_form(source) == s.diag(1, large * (large + 1))
    if cls in (s.SparseMatrix, s.ImmutableSparseMatrix):
        native = _native.SparseMatrix(2, 2, {(0, 0): _native.py_integer(6), (1, 1): _native.py_integer(10)})
    else:
        native = _native.Matrix(2, 2, [_native.py_integer(n) for n in (6, 0, 0, 10)])
    source = cls(native)
    assert source._surface is None
    diagonal, left, right = smith_normal_decomposition(source)
    assert diagonal == left * source * right
    assert diagonal == s.diag(2, 30)


@pytest.mark.parametrize("entry", [
    s.Rational(1, 2), s.Float(0), s.Symbol("integer_source"),
    s.Add(1, -1, evaluate=False),
    s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False),
])
@pytest.mark.parametrize("cls", CLASSES)
def test_noninteger_and_held_original_cells_cannot_disappear_in_zero_outputs(cls, entry):
    source = cls([[entry]])
    with pytest.raises((NotImplementedError, ValueError)):
        smith_normal_form(source)


@pytest.mark.parametrize("cls", CLASSES)
def test_borrowed_limits_are_checked_before_copying_source_coefficients(cls):
    oversized = cls([[s.Integer(1 << 4096)]])
    wide = cls(0, 65, {}) if cls in (s.SparseMatrix, s.ImmutableSparseMatrix) else cls(0, 65, [])
    with patch.object(normalforms, "_rational_source_snapshot", side_effect=AssertionError("copied source before admission")):
        for source in (oversized, wide):
            with pytest.raises(ValueError, match="height|4096|64"):
                smith_normal_form(source)


def test_subclasses_and_domain_objects_are_refused_without_custom_protocols():
    effects = []

    class CustomMatrix(s.Matrix):
        def __getitem__(self, index):
            effects.append("index")
            raise AssertionError("custom indexing")

    class Domain:
        def __bool__(self):
            effects.append("domain truth")
            raise AssertionError("domain coercion")

    with pytest.raises(NotImplementedError, match="built-in"):
        smith_normal_form(CustomMatrix([[2]]))
    for domain in (Domain(), "ZZ", "QQ", 17):
        with pytest.raises(NotImplementedError, match="inferred integer domain"):
            smith_normal_form(s.Matrix([[2]]), domain=domain)
    assert effects == []


@pytest.mark.parametrize("cls", CLASSES)
def test_original_storage_views_and_constructor_owners_are_guarded(cls, monkeypatch):
    source = cls([[2]])
    effects = []

    def forbidden(*args, **kwargs):
        effects.append("called")
        raise AssertionError("replaced implementation invoked")

    monkeypatch.setattr(cls, "__init__", forbidden)
    with pytest.raises(NotImplementedError, match="original"):
        smith_normal_form(source)
    assert effects == []


def test_replaced_native_getter_code_is_refused_without_execution(monkeypatch):
    source = s.Matrix([[2]])
    getter = s.Matrix.__dict__["_native"].fget

    def replacement(self):
        raise AssertionError("replaced native getter executed")

    monkeypatch.setattr(getter, "__code__", replacement.__code__)
    with pytest.raises(NotImplementedError, match="native storage view"):
        smith_normal_form(source)


@pytest.mark.parametrize("cls", CLASSES)
def test_stale_retained_cells_and_shapes_cannot_rebind_original_source(cls):
    stale = cls([[2]])
    stale._surface = {(0, 0): s.Integer(3)} if cls in (s.SparseMatrix, s.ImmutableSparseMatrix) else (s.Integer(3),)
    with pytest.raises(RuntimeError, match="retained cells differ"):
        smith_normal_form(stale)
    malformed = cls([[2]])
    malformed._matrix_shape = (2, 2)
    with pytest.raises(ValueError, match="shapes differ"):
        smith_normal_form(malformed)


def test_late_source_mutation_prevents_publication(monkeypatch):
    source = s.Matrix([[6, 0], [0, 10]])
    original = normalforms._check_rational_source
    calls = []

    def mutate_after_native(snapshot):
        calls.append(1)
        if len(calls) == 3:
            source[0, 0] = 7
        original(snapshot)

    monkeypatch.setattr(normalforms, "_check_rational_source", mutate_after_native)
    with pytest.raises(RuntimeError, match="mutated"):
        smith_normal_decomposition(source)
    assert len(calls) == 3


@pytest.mark.parametrize("cls", (s.ImmutableMatrix, s.ImmutableSparseMatrix))
@pytest.mark.parametrize("name", ("_matrix_shape", "_surface", "_native_storage", "_mutation_generation"))
def test_late_immutable_storage_shadowing_prevents_publication(cls, name, monkeypatch):
    source = cls([[6, 0], [0, 10]])
    original = normalforms._check_rational_source
    calls = []
    effects = []

    def forbidden(self):
        effects.append("shadow invoked")
        raise AssertionError("changed subclass descriptor was invoked")

    def change_after_native(snapshot):
        original(snapshot)
        calls.append(1)
        if len(calls) == 3:
            monkeypatch.setattr(cls, name, property(forbidden))

    monkeypatch.setattr(normalforms, "_check_rational_source", change_after_native)
    with pytest.raises(NotImplementedError, match="storage descriptors"):
        smith_normal_decomposition(source)
    assert len(calls) == 3
    assert effects == []


def test_retained_atom_structure_is_not_erased_by_native_integer_storage(monkeypatch):
    cell = s.Integer(938531)
    source = s.Matrix([[cell]])
    monkeypatch.setattr(cell, "_surface_args", (s.Integer(0),), raising=False)
    with pytest.raises(ValueError, match="literal retained atoms"):
        smith_normal_form(source)


def _native_cost(source):
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    source._native_storage.smith_normal_decomposition(session=session)
    assert session.consumed_nested_steps > 0
    return session.consumed_nested_steps


@pytest.mark.parametrize("cls", (s.Matrix, s.SparseMatrix))
@pytest.mark.parametrize("cancelled", [False, True])
def test_terminal_active_requests_refuse_before_source_copies(cls, cancelled):
    source = cls([[6, 0], [0, 10]])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS if cancelled else 0)
    if cancelled:
        session.cancel()
    expected = RuntimeError if cancelled else ValueError
    with patch.object(normalforms, "_rational_source_snapshot", side_effect=AssertionError("terminal request copied source")):
        with supervised_callback_session(session):
            with pytest.raises(expected, match="cancelled|budget exhausted"):
                smith_normal_decomposition(source)


@pytest.mark.parametrize("cls", (s.Matrix, s.SparseMatrix))
def test_native_generation_verification_and_lifting_share_active_request_allowance(cls):
    source = cls([[6, 0], [0, 10]])
    cost = _native_cost(source)
    session = SupervisedCallbackSession(max_nested_steps=2 * cost - 1)
    native = source._native_storage
    native.smith_normal_decomposition(session=session)
    assert session.consumed_nested_steps == cost
    with pytest.raises(ValueError, match="nested native request budget exhausted"):
        native.smith_normal_decomposition(session=session)
    consumed = session.consumed_nested_steps
    assert cost <= consumed <= session.max_nested_steps
    assert session.cancelled is True
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        native.smith_normal_decomposition(session=session)
    assert session.consumed_nested_steps == consumed


@pytest.mark.parametrize("cls", (s.Matrix, s.SparseMatrix))
def test_public_output_lifting_is_charged_and_late_cancellation_is_terminal(cls):
    source = cls([[6, 0], [0, 10]])
    cost = _native_cost(source)
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as measured:
        smith_normal_decomposition(source)
    assert measured.consumed_nested_steps > cost
    original = normalforms._check_rational_source
    calls = []
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as late:
        def cancel_before_publication(snapshot):
            original(snapshot)
            calls.append(late.consumed_nested_steps)
            if len(calls) == 4:
                late.cancel()

        with patch.object(normalforms, "_check_rational_source", cancel_before_publication):
            with pytest.raises(RuntimeError, match="cancelled|tainted"):
                smith_normal_decomposition(source)
    assert len(calls) == 4
    assert calls[-1] > cost
