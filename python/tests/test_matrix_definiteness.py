"""Pinned property behavior and exact quadratic-form classification.

The ordinary cases also run under isolated SymPy 1.14.0. Explicit native
envelope/source-binding cases are marked for the FrankenSymPy implementation.
"""

import pytest
from sympy import (
    I, Expr, ImmutableMatrix, ImmutableSparseMatrix, Integer, Matrix, Mul, Pow,
    Rational, SparseMatrix, Symbol, diag, eye, zeros,
)

NAMES = (
    "is_positive_definite", "is_positive_semidefinite", "is_negative_definite",
    "is_negative_semidefinite", "is_indefinite",
)
CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)
CANDIDATE = hasattr(Matrix([[1]]), "_native_storage")
native_only = pytest.mark.skipif(not CANDIDATE, reason="native admission and mutation contract")


def observed(matrix):
    return tuple(getattr(matrix, name) for name in NAMES)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entries,expected", [
    ([[1, 2], [-2, 1]], (True, True, False, False, False)),
    ([[1, 4], [0, 1]], (False, False, False, False, True)),
    ([[1, 2], [2, 4]], (False, True, False, False, False)),
    ([[-1, -2], [-2, -4]], (False, False, False, True, False)),
    ([[0, 0], [0, 0]], (False, True, False, True, False)),
    ([[0, 7], [-7, 0]], (False, True, False, True, False)),
    ([[0, 1], [1, 0]], (False, False, False, False, True)),
    ([[0, 0, 1], [0, 0, 0], [1, 0, 0]], (False, False, False, False, True)),
])
def test_exact_rational_forms_use_the_symmetric_part(cls, entries, expected):
    result = observed(cls(entries))
    assert result == expected
    assert all(type(value) is bool for value in result)


@pytest.mark.parametrize("cls", CLASSES)
def test_empty_and_nonsquare_properties_have_the_pinned_contract(cls):
    assert observed(cls(0, 0, [])) == (True, True, True, True, False)
    for rows, columns in ((0, 2), (2, 0), (1, 2)):
        assert observed(cls(rows, columns, [0] * (rows*columns))) == (False,) * 5
    for name in NAMES:
        assert isinstance(getattr(cls, name), property)


@pytest.mark.parametrize("cls", CLASSES)
def test_tiny_exact_positive_direction_is_not_rounded_to_singularity(cls):
    delta = Rational(1, 2**1024)
    matrix = cls([[1, 1], [1, 1 + delta]])
    assert matrix.is_positive_definite is True
    assert matrix.is_positive_semidefinite is True


@pytest.mark.parametrize("cls", CLASSES)
def test_symbolic_unknown_is_retained_and_declared_signs_are_used(cls):
    x = Symbol("inertia_x", real=True)
    p = Symbol("inertia_p", positive=True)
    z = Symbol("inertia_z")
    assert cls([[x]]).is_positive_definite is None
    assert cls([[x]]).is_positive_semidefinite is None
    assert cls([[z]]).is_positive_definite is None
    assert cls([[p, 0], [0, 1]]).is_positive_definite is True
    assert cls([[0, 0], [0, x]]).is_positive_definite is False
    assert cls([[0, 0], [0, x]]).is_positive_semidefinite is None
    assert cls([[p, 0], [0, -1]]).is_indefinite is True


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entries,expected", [
    ([[I]], (False, True, False, True, False)),
    ([[1, I], [I, 1]], (True, True, False, False, False)),
    ([[1, I], [-I, 1]], (False, True, False, False, False)),
    ([[1, 2*I], [-2*I, 1]], (False, False, False, False, True)),
    ([[1, 1+I], [1-I, 3]], (True, True, False, False, False)),
])
def test_complex_forms_use_adjoint_instead_of_plain_transpose(cls, entries, expected):
    assert observed(cls(entries)) == expected


def test_large_singular_psd_form_does_not_enumerate_principal_subsets():
    # A 24x24 PSD form used to require 2^24 subsets and determinant-size refusals.
    matrix = diag(*([1] * 18 + [0] * 6))
    assert matrix.is_positive_semidefinite is True
    assert matrix.is_positive_definite is False


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_preview_inertia_resolves_a_recorded_oracle_unknown(cls):
    # FSYM-DIFF-MATRIX-INERTIA-001: the pinned oracle's cubic eigenvalue
    # sign predicates remain unknown. This preview has a checked congruence.
    # Keep this explicitly outside the shared parity vectors above.
    matrix = cls([[1, 1, 2], [1, 1, 1], [2, 1, 4]])
    assert observed(matrix) == (False, False, False, False, True)
    assert matrix._native_storage.quadratic_form_inertia() == (2, 1, 0)


@pytest.mark.skipif(CANDIDATE, reason="pinned oracle observation, not preview expectation")
@pytest.mark.parametrize("cls", CLASSES)
def test_pinned_oracle_retains_unknown_for_the_recorded_cubic(cls):
    assert observed(cls([[1, 1, 2], [1, 1, 1], [2, 1, 4]])) == (
        False, False, False, False, None,
    )


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_dimension_and_coefficient_envelopes_are_terminal(cls):
    with pytest.raises(ValueError, match="dimension"):
        _ = cls(eye(65)).is_positive_definite
    with pytest.raises(ValueError, match="coefficient"):
        _ = cls([[2**4096]]).is_positive_semidefinite


@native_only
def test_source_binding_rejects_replaced_retained_rational_storage(monkeypatch):
    entry = Rational(7, 3)
    matrix = Matrix([[entry]])
    monkeypatch.setattr(entry, "_value", Integer(42)._value)
    with pytest.raises(RuntimeError, match="retained cells"):
        _ = matrix.is_positive_definite


@native_only
def test_source_mutation_after_snapshot_cannot_publish_old_inertia(monkeypatch):
    from sympy.matrices import _definiteness
    matrix = Matrix([[1, 0], [0, 1]])
    original = _definiteness._rational_source_snapshot

    def captured_then_changed(source, **options):
        captured = original(source, **options)
        source[0, 0] = -1
        return captured

    monkeypatch.setattr(_definiteness, "_rational_source_snapshot", captured_then_changed)
    with pytest.raises(RuntimeError, match="mutated"):
        _ = matrix.is_positive_definite


@native_only
def test_changed_matrix_accessor_is_rejected_before_invocation(monkeypatch):
    matrix = Matrix([[1]])
    invoked = []

    def replaced(self):
        invoked.append(True)
        return zeros(1)

    monkeypatch.setattr(Matrix, "_native", property(replaced))
    with pytest.raises(NotImplementedError, match="original"):
        _ = matrix.is_positive_definite
    assert not invoked


@native_only
def test_noncommutative_symbols_do_not_enter_scalar_form_rules():
    x = Symbol("noncommutative_inertia_x", commutative=False)
    assert Matrix([[x]]).is_positive_definite is None
    assert Matrix([[x]]).is_positive_semidefinite is None


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_held_source_poles_are_checked_before_symmetrization(cls):
    x = Symbol("inertia_domain_x", real=True)
    reciprocal = Pow(x, -1, evaluate=False)
    held_zero = Mul(0, reciprocal, evaluate=False)
    assert observed(cls([[held_zero]])) == (None,) * 5
    nested = Pow(reciprocal, 0, evaluate=False)
    assert observed(cls([[nested]])) == (None,) * 5


@native_only
def test_changed_symbol_declarations_cannot_prove_a_sign(monkeypatch):
    x = Symbol("inertia_changed_declarations", real=True)
    matrix = Matrix([[x]])
    monkeypatch.setitem(x._assumptions, "positive", True)
    with pytest.raises(ValueError, match="assumptions changed"):
        _ = matrix.is_positive_definite


@native_only
@pytest.mark.parametrize("payload", ("native", "arguments", "assumptions"))
def test_scalar_mutation_during_classification_is_terminal(monkeypatch, payload):
    from sympy.matrices import _definiteness

    x = Symbol("inertia_snapshot_x", positive=True)
    entry = Mul(x, 1, evaluate=False)
    matrix = Matrix([[entry]])
    original = _definiteness._symbolic

    def calculate_then_change(cells, dimension):
        result = original(cells, dimension)
        if payload == "native":
            entry._value = Integer(-1)._value
        elif payload == "arguments":
            entry._args = (Integer(-1),)
        else:
            x._assumptions["positive"] = False
        return result

    monkeypatch.setattr(_definiteness, "_symbolic", calculate_then_change)
    with pytest.raises(RuntimeError, match="mutated"):
        _ = matrix.is_positive_definite


@native_only
@pytest.mark.parametrize("imaginary", [False, True])
def test_changed_scalar_conjugation_is_rejected_before_invocation(monkeypatch, imaginary):
    entry = I if imaginary else Symbol("inertia_conjugation_x", real=True)
    matrix = Matrix([[entry]])
    invoked = []

    def replaced(self):
        invoked.append(True)
        return Integer(1)

    owner = type(I) if imaginary else Expr
    monkeypatch.setattr(owner, "conjugate", replaced, raising=False)
    with pytest.raises(NotImplementedError, match="original scalar arithmetic"):
        _ = matrix.is_positive_definite
    assert not invoked


@native_only
def test_domain_work_exhaustion_cannot_become_an_unknown_predicate(monkeypatch):
    from sympy.solvers import _checksol

    matrix = Matrix([[Symbol("inertia_exhausted_x", real=True)]])
    monkeypatch.setattr(_checksol, "_MAX_WORK", 0)
    with pytest.raises(ValueError, match="domain-check envelope"):
        _ = matrix.is_positive_definite
