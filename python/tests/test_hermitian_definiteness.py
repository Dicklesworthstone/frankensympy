"""Exact complex matrix properties and original-source publication checks."""

from unittest.mock import patch

import pytest
from sympy import (
    Add, I, ImmutableMatrix, ImmutableSparseMatrix, Integer, Matrix, Mul, Pow,
    Rational, SparseMatrix, Symbol,
)


CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)
PROPERTIES = (
    "is_positive_definite", "is_positive_semidefinite", "is_negative_definite",
    "is_negative_semidefinite", "is_indefinite",
)
CANDIDATE = hasattr(Matrix([[1]]), "_native_storage")
native_only = pytest.mark.skipif(not CANDIDATE, reason="native source and request contract")


def properties(matrix):
    return tuple(getattr(matrix, name) for name in PROPERTIES)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entries,expected", [
    ([[3, I, 0], [-I, 3, I], [0, -I, 3]], (True, True, False, False, False)),
    ([[-3, -I, 0], [I, -3, -I], [0, I, -3]], (False, False, True, True, False)),
    ([[1, -I, 1], [I, 1, I], [1, -I, 1]], (False, True, False, False, False)),
    ([[-1, I, -1], [-I, -1, -I], [-1, I, -1]], (False, False, False, True, False)),
    ([[1, I, 1], [-I, 2, 1], [1, 1, 3]], (False, True, False, False, False)),
    ([[1, 2*I, 0], [-2*I, 1, 0], [0, 0, 3]], (False, False, False, False, True)),
    ([[0, I, 0], [-I, 0, 0], [0, 0, 0]], (False, False, False, False, True)),
    ([[3+I, 3*I, 0], [I, 3+2*I, 2*I], [0, 0, 3-3*I]],
     (True, True, False, False, False)),
    ([[I, 2+I, 3*I], [-2+I, -2*I, 1], [3*I, -1, 4*I]],
     (False, True, False, True, False)),
    ([[Rational(1, 2), I/3, 0], [-I/3, Rational(1, 2), 0], [0, 0, Rational(1, 7)]],
     (True, True, False, False, False)),
])
def test_exact_complex_forms_match_pinned_sign_properties(cls, entries, expected):
    assert properties(cls(entries)) == expected


@pytest.mark.parametrize("cls", CLASSES)
def test_tiny_complex_direction_uses_exact_arithmetic(cls):
    delta = Rational(1, 2**1024)
    matrix = cls([[1, I, 0], [-I, 1+delta, 0], [0, 0, 1]])
    assert matrix.is_positive_definite is True
    assert matrix.is_positive_semidefinite is True


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_larger_complex_forms_reach_checked_native_inertia(cls):
    cells = [Integer(0)] * (12*12)
    for i in range(10):
        cells[i*12+i] = Integer(3)
    cells[1], cells[12] = I, -I
    matrix = cls(12, 12, cells)
    assert properties(matrix) == (False, True, False, False, False)
    assert matrix._native_storage.hermitian_form_inertia() == (10, 0, 2)


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_native_only_complex_sources_preserve_both_storage_kinds(cls):
    original = cls([[3, I, 0], [-I, 3, I], [0, -I, 3]])
    matrix = cls(original._native_storage)
    assert matrix._surface is None
    assert properties(matrix) == (True, True, False, False, False)


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_held_gaussian_reciprocal_keeps_its_original_arguments(cls):
    reciprocal = Pow(Add(1, I, evaluate=False), -2, evaluate=False)
    matrix = cls([[Add(1, reciprocal, evaluate=False)]])
    assert properties(matrix) == (True, True, False, False, False)


@native_only
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("form", ("zero_product", "zero_power", "zero_to_zero"))
def test_undefined_held_gaussian_children_never_prove_a_sign(cls, form):
    pole = Pow(0, -1, evaluate=False)
    source = (Mul(0, pole, evaluate=False) if form == "zero_product"
              else Pow(pole, 0, evaluate=False) if form == "zero_power"
              else Pow(0, 0, evaluate=False))
    assert properties(cls([[source]])) == (None,) * 5


@native_only
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("slot", ("_value", "_args", "_surface_args"))
def test_stale_retained_gaussian_cells_are_rejected_at_entry(cls, slot, monkeypatch):
    entry = Add(1, I, evaluate=False)
    matrix = cls([[entry]])
    replacement = Integer(-1)._value if slot == "_value" else (Integer(-1), I)
    monkeypatch.setattr(entry, slot, replacement, raising=False)
    with pytest.raises((ValueError, RuntimeError), match="retained|native"):
        _ = matrix.is_positive_definite


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_hidden_pole_in_replaced_retained_arguments_cannot_use_a_stale_handle(cls, monkeypatch):
    entry = Add(1, I, evaluate=False)
    matrix = cls([[entry]])
    pole = Mul(0, Pow(0, -1, evaluate=False), evaluate=False)
    monkeypatch.setattr(entry, "_args", (Integer(1), pole))
    with pytest.raises((ValueError, RuntimeError), match="retained|native"):
        _ = matrix.is_positive_definite


@native_only
def test_sparse_retained_nonzero_cannot_replace_an_omitted_native_zero():
    matrix = SparseMatrix([[I, 0], [0, 1]])
    matrix._surface[(0, 1)] = Integer(2)
    with pytest.raises(RuntimeError, match="retained cells"):
        _ = matrix.is_positive_definite


@native_only
def test_native_expression_subclasses_cannot_impersonate_exact_scalar_storage(monkeypatch):
    from sympy.core import _native

    class Untrusted(_native.Expr):
        pass

    entry = Add(1, I, evaluate=False)
    matrix = Matrix([[entry]])
    monkeypatch.setattr(entry, "_value", Untrusted("1 + I"))
    with pytest.raises(NotImplementedError, match="original native scalar storage"):
        _ = matrix.is_positive_definite


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_replaced_oversized_scalar_is_bounded_before_native_binding(cls, monkeypatch):
    from sympy.matrices import _definiteness

    entry = Add(1, I, evaluate=False)
    matrix = cls([[entry]])
    monkeypatch.setattr(entry, "_value", Integer(2**4096)._value)
    with patch.object(_definiteness, "_bind_gaussian",
                      side_effect=AssertionError("oversized retained scalar reached binding")):
        with pytest.raises(ValueError, match="4096 bits"):
            _ = matrix.is_positive_definite


@native_only
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("fixed", (False, True))
def test_closed_numeric_owners_match_their_exact_native_payload(cls, fixed, monkeypatch):
    from sympy.core import Zero

    entry = Zero() if fixed else Integer(2)
    replacement = Integer(2) if fixed else Rational(3, 2)
    monkeypatch.setattr(entry, "_value", replacement._value)
    matrix = cls([[Add(entry, I, evaluate=False)]])
    with pytest.raises(RuntimeError, match="retained (constant|atom) differs"):
        _ = matrix.is_positive_definite


@native_only
@pytest.mark.parametrize("cls", (Matrix, SparseMatrix))
def test_matrix_mutation_after_gaussian_capture_is_terminal(cls, monkeypatch):
    from sympy.matrices import _definiteness

    matrix = cls([[3, I], [-I, 3]])
    original = _definiteness._bind_gaussian

    def capture_then_change(*args):
        captured = original(*args)
        matrix[0, 0] = -3
        return captured

    monkeypatch.setattr(_definiteness, "_bind_gaussian", capture_then_change)
    with pytest.raises(RuntimeError, match="mutated"):
        _ = matrix.is_positive_definite


@native_only
@pytest.mark.parametrize("cls", (Matrix, SparseMatrix))
def test_complex_coefficient_envelope_refuses_before_public_source_copy(cls):
    from sympy.matrices import _definiteness

    matrix = cls([[Add(2**4096, I, evaluate=False)]])
    with patch.object(_definiteness, "_bind_gaussian",
                      side_effect=AssertionError("oversized source was copied")):
        with pytest.raises(ValueError, match="coefficient"):
            _ = matrix.is_positive_definite


@native_only
@pytest.mark.parametrize("cls", (Matrix, SparseMatrix))
def test_public_forms_above_native_dimension_remain_unknown_without_copy(cls):
    from sympy.matrices import _definiteness

    for entry in (I, Symbol("large_symbolic_form", positive=True)):
        matrix = cls(33, 33, [entry if i == 0 else 0 for i in range(33*33)])
        with patch.object(_definiteness, "_bind_gaussian",
                          side_effect=AssertionError("oversized source was copied")):
            assert properties(matrix) == (None,) * 5
        with pytest.raises(ValueError, match="dimension"):
            matrix._native_storage.validate_hermitian_inertia_source()


@native_only
@pytest.mark.parametrize("cls", (Matrix, SparseMatrix))
def test_unsupported_symbolic_forms_keep_the_conservative_lane(cls):
    positive = Symbol("hermitian_positive", positive=True)
    unknown = Symbol("hermitian_unknown", real=True)
    assert cls([[positive, I], [I, 1]]).is_positive_definite is True
    assert cls([[unknown, I], [I, 1]]).is_positive_definite is None


@native_only
@pytest.mark.parametrize("cls", (Matrix, SparseMatrix))
def test_native_complex_phases_share_the_original_request_budget(cls):
    from sympy.core import SupervisedCallbackSession

    native = cls([[3, I], [-I, 3]])._native_storage
    measured = SupervisedCallbackSession(max_nested_steps=10_000_000)
    assert native.hermitian_form_inertia(session=measured) == (2, 0, 0)
    cost = measured.consumed_nested_steps
    assert cost > 0
    request = SupervisedCallbackSession(max_nested_steps=2*cost-1)
    assert native.hermitian_form_inertia(session=request) == (2, 0, 0)
    with pytest.raises(ValueError, match="budget exhausted"):
        native.hermitian_form_inertia(session=request)
    assert request.cancelled is True
    consumed = request.consumed_nested_steps
    assert cost <= consumed <= request.max_nested_steps
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        native.hermitian_form_inertia(session=request)
    assert request.consumed_nested_steps == consumed


@native_only
@pytest.mark.parametrize("cls", (Matrix, SparseMatrix))
@pytest.mark.parametrize("cancelled", (False, True))
def test_terminal_complex_requests_refuse_before_retained_copy(cls, cancelled):
    from sympy.core import SupervisedCallbackSession, supervised_callback_session
    from sympy.matrices import _definiteness

    matrix = cls([[3, I], [-I, 3]])
    session = SupervisedCallbackSession(max_nested_steps=10_000_000 if cancelled else 0)
    if cancelled:
        session.cancel()
    with supervised_callback_session(session):
        with patch.object(_definiteness, "_bind_gaussian",
                          side_effect=AssertionError("terminal source was copied")):
            with pytest.raises(RuntimeError if cancelled else ValueError,
                               match="cancelled|budget exhausted"):
                _ = matrix.is_positive_definite


@native_only
@pytest.mark.parametrize("cls", (Matrix, SparseMatrix))
def test_late_complex_request_cancellation_prevents_boolean_publication(cls, monkeypatch):
    from sympy.core import supervised_callback_session
    from sympy.matrices import _definiteness

    matrix = cls([[3, I], [-I, 3]])
    original = _definiteness._inertia_properties
    with supervised_callback_session(max_nested_steps=10_000_000) as session:
        def counts_then_cancel(*args):
            result = original(*args)
            session.cancel()
            return result

        monkeypatch.setattr(_definiteness, "_inertia_properties", counts_then_cancel)
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            _ = matrix.is_positive_definite
