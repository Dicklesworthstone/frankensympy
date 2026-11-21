"""Exact inverses over prime and composite residue rings with source retention."""

from math import gcd
from unittest.mock import patch

import pytest
import sympy as s
from sympy.core import (
    Half, NegativeOne, One, Zero, SupervisedCallbackSession, _native,
    supervised_callback_session,
)
from sympy.matrices import normalforms
from sympy.matrices.exceptions import (
    MatrixError, NonInvertibleMatrixError, NonSquareMatrixError, ShapeError,
)


CLASSES = (s.Matrix, s.ImmutableMatrix, s.SparseMatrix, s.ImmutableSparseMatrix)
AMPLE_STEPS = 10_000_000


def _residuals(source, inverse, modulus):
    modulus = int(modulus)
    size = source.rows
    for left, right in ((source, inverse), (inverse, source)):
        for row in range(size):
            for column in range(size):
                value = sum(int(left[row, inner]) * int(right[inner, column])
                            for inner in range(size))
                assert value % modulus == int(row == column)
    assert all(0 <= int(inverse[row, column]) < modulus
               for row in range(size) for column in range(size))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("values,modulus,expected", [
    ([[1, 2], [3, 4]], 5, [[3, 1], [4, 2]]),
    ([[2, 3], [3, 2]], 6, [[2, 3], [3, 2]]),
    ([[2, 3], [3, 2]], 2, [[0, 1], [1, 0]]),
    ([[-3]], 10, [[3]]),
    ([[6, 24, 1], [13, 16, 10], [20, 17, 15]], 26,
     [[8, 5, 10], [21, 8, 21], [21, 12, 8]]),
])
def test_exact_prime_and_composite_inverses_return_mutable_dense_matrix(cls, values, modulus, expected):
    source = cls(values)
    inverse = source.inv_mod(modulus)
    assert type(inverse) is s.Matrix
    assert inverse == s.Matrix(expected)
    _residuals(source, inverse, modulus)


@pytest.mark.parametrize("cls", CLASSES)
def test_composite_modulus_does_not_require_a_unit_pivot_entry(cls):
    # Every entry is a nonunit mod 6; det=-5 is nevertheless a unit.
    source = cls([[2, 3], [3, 2]])
    assert all(gcd(value, 6) != 1 for value in (2, 3))
    assert source.inv_mod(s.Integer(6)) == s.Matrix([[2, 3], [3, 2]])


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("values,modulus", [
    ([[2, 3], [3, 2]], 5),
    ([[2, 0], [0, 1]], 6),
    ([[1, 2], [2, 4]], 7),
    ([[0, 0], [0, 0]], 6),
])
def test_noninvertibility_is_an_exact_ring_decision(cls, values, modulus):
    with pytest.raises(NonInvertibleMatrixError, match="not invertible"):
        cls(values).inv_mod(modulus)


@pytest.mark.parametrize("cls", CLASSES)
def test_empty_shape_and_large_exact_modulus(cls):
    empty = cls(0, 0, {}) if cls in (s.SparseMatrix, s.ImmutableSparseMatrix) else cls(0, 0, [])
    result = empty.inv_mod(6)
    assert type(result) is s.Matrix and result.shape == (0, 0)
    modulus = 2**256 + 1
    source = cls([[modulus - 2, 1], [1, 0]])
    result = source.inv_mod(s.Integer(modulus))
    assert result == s.Matrix([[0, 1], [1, 2]])
    _residuals(source, result, modulus)


@pytest.mark.parametrize("cls", CLASSES)
def test_native_only_integer_sources_are_supported(cls):
    if cls in (s.SparseMatrix, s.ImmutableSparseMatrix):
        native = _native.SparseMatrix(
            2, 2, {(r, c): _native.py_integer(v)
                   for r, row in enumerate(((2, 3), (3, 2)))
                   for c, v in enumerate(row)},
        )
    else:
        native = _native.Matrix(2, 2, [_native.py_integer(v) for v in (2, 3, 3, 2)])
    source = cls(native)
    assert source._surface is None
    assert source.inv_mod(6) == s.Matrix([[2, 3], [3, 2]])


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("shape", [(0, 2), (2, 0), (2, 3)])
def test_nonsquare_shapes_raise_the_standard_matrix_exception(cls, shape):
    source = cls(*shape, {}) if cls in (s.SparseMatrix, s.ImmutableSparseMatrix) else cls(*shape, [0] * (shape[0] * shape[1]))
    with pytest.raises(NonSquareMatrixError):
        source.inv_mod(6)


def test_standard_exception_hierarchy():
    assert issubclass(NonSquareMatrixError, ShapeError)
    assert issubclass(ShapeError, ValueError) and issubclass(ShapeError, MatrixError)
    assert issubclass(NonInvertibleMatrixError, ValueError)
    assert issubclass(NonInvertibleMatrixError, MatrixError)


@pytest.mark.parametrize("modulus", [True, False, 6.0, "6", s.Rational(3, 2), s.Symbol("m")])
def test_noninteger_modulus_types_are_not_coerced(modulus):
    with pytest.raises(TypeError, match="modulus m must be an integer"):
        s.eye(1).inv_mod(modulus)


@pytest.mark.parametrize("modulus", [0, -6, s.Integer(0), s.Integer(-6), 1])
def test_modulus_is_bounded_to_nontrivial_positive_integer_rings(modulus):
    # SymPy 1.14 has an IndexError for nonempty matrices modulo 1. This bounded
    # interface explicitly refuses that degenerate ring instead of emulating it.
    with pytest.raises(ValueError, match="at least two"):
        s.eye(1).inv_mod(modulus)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entry", [
    s.Rational(1, 2), s.Float(0), s.Symbol("integer_entry"),
    s.Add(1, -1, evaluate=False),
    s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False),
])
def test_noninteger_or_held_original_cells_cannot_disappear(cls, entry):
    with pytest.raises(ValueError):
        cls([[0, 0], [0, entry]]).inv_mod(6)


def test_custom_scalar_and_matrix_protocols_are_never_executed():
    effects = []

    class Modulus:
        def __int__(self):
            effects.append("int")
            raise AssertionError("custom int")
        def __index__(self):
            effects.append("index")
            raise AssertionError("custom index")

    class IntegerSubclass(int):
        def bit_length(self):
            effects.append("bits")
            raise AssertionError("custom bits")

    class MatrixSubclass(s.Matrix):
        def __getitem__(self, key):
            effects.append("entry")
            raise AssertionError("custom indexing")

    for value in (Modulus(), IntegerSubclass(6)):
        with pytest.raises(TypeError):
            s.eye(1).inv_mod(value)
    with pytest.raises(NotImplementedError, match="built-in"):
        MatrixSubclass([[1]]).inv_mod(6)
    assert effects == []


@pytest.mark.parametrize("cls", CLASSES)
def test_original_storage_and_source_owners_cannot_be_replaced(cls, monkeypatch):
    source = cls([[2, 3], [3, 2]])
    effects = []
    def forbidden(*args, **kwargs):
        effects.append("replacement")
        raise AssertionError("replaced constructor")
    monkeypatch.setattr(cls, "__init__", forbidden)
    with pytest.raises(NotImplementedError, match="original"):
        source.inv_mod(6)
    assert effects == []


@pytest.mark.parametrize("slot", ("_surface_args", "_struct_args"))
def test_retained_modulus_structure_is_not_erased_by_native_integer(slot, monkeypatch):
    modulus = s.Integer(1009)
    monkeypatch.setattr(modulus, slot, (s.Pow(0, -1, evaluate=False),), raising=False)
    with pytest.raises(ValueError, match="literal integer atom"):
        s.eye(1).inv_mod(modulus)


@pytest.mark.parametrize("cls", CLASSES)
def test_stale_retained_matrix_cells_cannot_rebind_the_source(cls):
    source = cls([[1]])
    source._surface = {(0, 0): s.Integer(2)} if cls in (s.SparseMatrix, s.ImmutableSparseMatrix) else (s.Integer(2),)
    with pytest.raises(RuntimeError, match="retained cells differ"):
        source.inv_mod(6)


def test_modulus_mutation_after_native_acceptance_prevents_publication(monkeypatch):
    source = s.Matrix([[2, 3], [3, 2]])
    modulus = s.Integer(6)
    original = normalforms._check_rational_source
    calls = []
    def mutate(snapshot):
        original(snapshot)
        calls.append(1)
        if len(calls) == 3:
            monkeypatch.setattr(modulus, "_value", _native.py_integer(7))
    monkeypatch.setattr(normalforms, "_check_rational_source", mutate)
    with pytest.raises(RuntimeError, match="modulus mutated"):
        source.inv_mod(modulus)
    assert len(calls) == 3


def test_source_mutation_after_native_acceptance_prevents_publication(monkeypatch):
    source = s.Matrix([[2, 3], [3, 2]])
    original = normalforms._check_rational_source
    calls = []
    def mutate(snapshot):
        calls.append(1)
        if len(calls) == 3:
            source[0, 0] = 7
        original(snapshot)
    monkeypatch.setattr(normalforms, "_check_rational_source", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        source.inv_mod(6)
    assert len(calls) == 3


def test_modulus_and_source_height_are_bounded_before_source_copies():
    source = s.eye(1)
    with patch.object(normalforms, "_rational_source_snapshot", side_effect=AssertionError("copied oversized source")):
        with pytest.raises(ValueError, match="4096"):
            source.inv_mod(1 << 4096)
        with pytest.raises(ValueError, match="height"):
            s.Matrix([[s.Integer(1 << 4096)]]).inv_mod(6)


@pytest.mark.parametrize("cls", (s.Matrix, s.SparseMatrix))
@pytest.mark.parametrize("cancelled", [False, True])
def test_terminal_requests_stop_before_any_matrix_source_copy(cls, cancelled):
    source = cls([[2, 3], [3, 2]])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS if cancelled else 0)
    if cancelled:
        session.cancel()
    with patch.object(normalforms, "_rational_source_snapshot", side_effect=AssertionError("terminal request copied source")):
        with supervised_callback_session(session):
            with pytest.raises(RuntimeError if cancelled else ValueError, match="cancelled|budget exhausted"):
                source.inv_mod(6)


@pytest.mark.parametrize("cls", (s.Matrix, s.SparseMatrix))
def test_native_phases_and_output_lifting_share_one_cumulative_request(cls):
    source = cls([[2, 3], [3, 2]])
    native = source._native_storage
    measured = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    assert native.inverse_mod(6, session=measured) is not None
    cost = measured.consumed_nested_steps
    assert cost > 0
    session = SupervisedCallbackSession(max_nested_steps=2 * cost - 1)
    assert native.inverse_mod(6, session=session) is not None
    assert session.consumed_nested_steps == cost
    with pytest.raises(ValueError, match="budget exhausted"):
        native.inverse_mod(6, session=session)
    consumed = session.consumed_nested_steps
    assert cost <= consumed <= session.max_nested_steps
    assert session.cancelled
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        native.inverse_mod(6, session=session)
    assert session.consumed_nested_steps == consumed
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as public:
        source.inv_mod(6)
    assert public.consumed_nested_steps > cost


@pytest.mark.parametrize("invertible", [False, True])
def test_late_cancellation_cannot_be_relabelled_as_noninvertible(invertible):
    source = s.Matrix([[2, 3], [3, 2]])
    modulus = 6 if invertible else 5
    original = normalforms._check_rational_source
    calls = []
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as session:
        def cancel(snapshot):
            original(snapshot)
            calls.append(1)
            if len(calls) == 3:
                session.cancel()
        with patch.object(normalforms, "_check_rational_source", cancel):
            with pytest.raises(RuntimeError, match="cancelled|tainted"):
                source.inv_mod(modulus)
    assert len(calls) >= 3


@pytest.mark.parametrize("kind", (Zero, One, NegativeOne, Half))
def test_fixed_modulus_constants_cannot_be_forged_as_another_integer(kind, monkeypatch):
    source = s.eye(1)
    modulus = kind()
    monkeypatch.setattr(modulus, "_value", _native.py_integer(6))
    with pytest.raises(ValueError, match="fixed modulus constant"):
        source.inv_mod(modulus)


def test_modulus_owner_and_native_payload_must_agree(monkeypatch):
    source = s.eye(1)
    modulus = s.Integer(1009)
    monkeypatch.setattr(modulus, "_value", s.Rational(3, 2)._value)
    with pytest.raises(ValueError, match="literal owner"):
        source.inv_mod(modulus)
    rational = s.Rational(1009, 1013)
    monkeypatch.setattr(rational, "_value", s.Symbol("forged_modulus")._value)
    with pytest.raises(ValueError, match="literal owner"):
        source.inv_mod(rational)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("kind", (Zero, One, NegativeOne, Half))
def test_fixed_matrix_source_constant_cannot_be_rebound_to_another_integer(cls, kind, monkeypatch):
    cell = kind()
    monkeypatch.setattr(cell, "_value", _native.py_integer(5))
    source = cls([[cell]])
    with pytest.raises(ValueError, match="fixed source constant"):
        source.inv_mod(6)
