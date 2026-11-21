"""Exact Q(i) matrix properties, retained sources, and cumulative budgets.

The ordinary vectors are independent quadratic forms and also run against the
pinned SymPy oracle. Admission, mutation and session cases exercise the actual
FrankenSymPy extension; no native result or scalar operation is mocked.
"""

import pytest

from sympy import (
    Add, Function, I, ImmutableMatrix, ImmutableSparseMatrix, Integer, Matrix,
    Mul, Pow, Rational, SparseMatrix, Symbol,
)


CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)
NAMES = (
    "is_positive_definite", "is_positive_semidefinite", "is_negative_definite",
    "is_negative_semidefinite", "is_indefinite",
)
CANDIDATE = hasattr(Matrix([[1]]), "_native_storage")
native_only = pytest.mark.skipif(not CANDIDATE, reason="native source and budget contract")
AMPLE_STEPS = 20_000_000


def properties(matrix):
    return tuple(getattr(matrix, name) for name in NAMES)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entries,expected", [
    # These positive/semidefinite tridiagonal forms have eigenvalues bounded
    # below by 3-sqrt(2), and exactly {0,1,3}, respectively.
    ([[3, I, 0], [-I, 3, I], [0, -I, 3]], (True, True, False, False, False)),
    ([[1, I, 0], [-I, 2, I], [0, -I, 1]], (False, True, False, False, False)),
    ([[-1, -I, 0], [I, -2, -I], [0, I, -1]], (False, False, False, True, False)),
    ([[-3, -I, 0], [I, -3, -I], [0, I, -3]], (False, False, True, True, False)),
    # The shifted path matrix has eigenvalues 1, 1-sqrt(2), 1+sqrt(2).
    ([[1, I, 0], [-I, 1, I], [0, -I, 1]], (False, False, False, False, True)),
    # Purely skew-Hermitian source: Re(z* A z) vanishes identically.
    ([[I, 2*I, 0], [2*I, -I, 3*I], [0, 3*I, I]], (False, True, False, True, False)),
    # Non-Hermitian source with the same Hermitian part as the first vector.
    ([[3+I, 3*I, 0], [I, 3-2*I, 2*I], [0, 0, 3+I]], (True, True, False, False, False)),
])
def test_complete_complex_forms_beyond_the_previous_two_by_two_slice(cls, entries, expected):
    assert properties(cls(entries)) == expected


@pytest.mark.parametrize("cls", CLASSES)
def test_rational_complex_coefficients_and_tiny_positive_directions_are_exact(cls):
    delta = Rational(1, 2**1024)
    matrix = cls([[1, I, 0], [-I, 1 + delta, 0], [0, 0, Rational(2, 3)]])
    assert matrix.is_positive_definite is True
    assert matrix.is_positive_semidefinite is True


@pytest.mark.parametrize("cls", CLASSES)
def test_symbolic_complex_forms_still_retain_unknown(cls):
    x = Symbol("hermitian_unknown", real=True)
    assert cls([[x, I, 0], [-I, x, I], [0, -I, x]]).is_positive_definite is None


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_native_counts_retain_all_three_signs_and_the_original_complex_dimension(cls):
    matrix = cls([[1, I, 0], [-I, 2, I], [0, -I, 1]])
    native = matrix._native_storage
    assert native.validate_hermitian_inertia_source() is None
    assert native.hermitian_form_inertia() == (2, 0, 1)
    other = cls([[1, I, 0], [-I, 1, I], [0, -I, 1]])
    assert other._native_storage.hermitian_form_inertia() == (2, 1, 0)


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_native_only_matrices_use_the_same_checked_complex_engine(cls):
    matrix = cls([[3, I, 0], [-I, 3, I], [0, -I, 3]])
    object.__setattr__(matrix, "_surface", None)
    assert properties(matrix) == (True, True, False, False, False)


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_larger_complex_blocks_do_not_stop_at_the_symbolic_dimension_limit(cls):
    entries = [[0] * 12 for _ in range(12)]
    for offset in range(0, 12, 3):
        for i, row in enumerate(((1, I, 0), (-I, 2, I), (0, -I, 1))):
            for j, value in enumerate(row):
                entries[offset+i][offset+j] = value
    matrix = cls(entries)
    assert matrix.is_positive_semidefinite is True
    assert matrix.is_positive_definite is False
    assert matrix._native_storage.hermitian_form_inertia() == (8, 0, 4)


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_defined_held_complex_inverse_is_admitted_without_discarding_source(cls):
    reciprocal = Pow(Add(1, I, evaluate=False), -1, evaluate=False)
    matrix = cls([[reciprocal]])
    assert matrix.is_positive_definite is True
    assert matrix._native_storage.hermitian_form_inertia() == (1, 0, 0)


@native_only
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("wrapper", [
    lambda inverse: Mul(0, inverse, evaluate=False),
    lambda inverse: Pow(inverse, 0, evaluate=False),
])
def test_undefined_held_complex_subexpressions_never_publish_zero_or_identity(cls, wrapper):
    zero = Add(I, -I, evaluate=False)
    inverse = Pow(zero, -1, evaluate=False)
    matrix = cls([[wrapper(inverse)]])
    with pytest.raises(ZeroDivisionError, match="division by zero"):
        matrix._native_storage.validate_hermitian_inertia_source()
    assert properties(matrix) == (None,) * 5


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_custom_function_is_not_executed_or_admitted_as_a_closed_complex_scalar(cls):
    class Opaque(Function):
        @classmethod
        def eval(cls, value):
            return None

        def _eval_is_positive(self):
            raise AssertionError("opaque positivity hook was executed")

    x = Symbol("hermitian_opaque")
    assert properties(cls([[Opaque(x)]])) == (None,) * 5


@native_only
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("mutation", ["root", "child", "arguments", "surface_arguments", "structural_arguments"])
def test_original_retained_complex_graph_cannot_be_replaced_before_verification(cls, mutation):
    coefficient = Rational(7, 3)
    child = Mul(coefficient, I, evaluate=False)
    entry = Add(1, child, evaluate=False)
    matrix = cls([[entry]])
    if mutation == "root":
        entry._value = Integer(-1)._value
    elif mutation == "child":
        coefficient._value = Integer(-1)._value
    elif mutation == "arguments":
        child._args = (Integer(1),)
    elif mutation == "surface_arguments":
        child._surface_args = (Integer(1), I)
    else:
        child._struct_args = (Integer(1), I)
    with pytest.raises(RuntimeError, match="retained.*differ"):
        _ = matrix.is_positive_definite


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_matrix_mutation_after_graph_binding_cannot_publish_old_inertia(cls, monkeypatch):
    from sympy.matrices import _definiteness

    matrix = cls([[3, I], [-I, 3]])
    original = _definiteness._bind_gaussian

    def bind_then_change(*args):
        original(*args)
        object.__setattr__(matrix, "_mutation_generation", matrix._mutation_generation + 1)

    monkeypatch.setattr(_definiteness, "_bind_gaussian", bind_then_change)
    with pytest.raises(RuntimeError, match="mutated"):
        _ = matrix.is_positive_definite


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_changed_native_cell_accessor_is_refused_before_it_executes(cls, monkeypatch):
    from sympy.matrices import _definiteness

    matrix = cls([[1, I], [-I, 2]])
    base = Matrix if cls in (Matrix, ImmutableMatrix) else SparseMatrix
    name, original = _definiteness._NATIVE_CELLS[base]
    called = []

    def changed(self):
        called.append(True)
        return original(self)

    monkeypatch.setattr(type(matrix._native_storage), name, changed)
    with pytest.raises(NotImplementedError, match="original native interfaces"):
        _ = matrix.is_positive_definite
    assert not called


@native_only
@pytest.mark.parametrize("cls", (Matrix, SparseMatrix))
def test_native_complex_dimension_and_coefficient_refusals_are_terminal(cls):
    matrix = cls(33, 33, lambda i, j: 1+I if i == j else 0)
    with pytest.raises(ValueError, match="dimension"):
        matrix._native_storage.validate_hermitian_inertia_source()
    with pytest.raises(ValueError, match="dimension"):
        matrix._native_storage.hermitian_form_inertia()
    large = cls([[Mul(2**4096, I, evaluate=False)]])
    with pytest.raises(ValueError, match="coefficient|bit"):
        _ = large.is_positive_definite


@native_only
@pytest.mark.parametrize("cls", (Matrix, SparseMatrix))
@pytest.mark.parametrize("cancelled", (False, True))
def test_terminal_request_refuses_before_complex_source_copy(cls, cancelled, monkeypatch):
    from sympy.core import SupervisedCallbackSession, supervised_callback_session
    from sympy.matrices import _definiteness

    matrix = cls([[3, I], [-I, 3]])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS if cancelled else 0)
    if cancelled:
        session.cancel()

    def forbidden(*args):
        raise AssertionError("terminal request copied a complex source")

    monkeypatch.setattr(_definiteness, "_bind_gaussian", forbidden)
    with supervised_callback_session(session):
        with pytest.raises(RuntimeError if cancelled else ValueError, match="cancelled|budget exhausted"):
            _ = matrix.is_positive_definite


@native_only
@pytest.mark.parametrize("cls", (Matrix, SparseMatrix))
def test_repeated_complex_calls_do_not_reset_the_shared_native_allowance(cls):
    from sympy.core import SupervisedCallbackSession

    native = cls([[3, I], [-I, 3]])._native_storage
    measured = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    assert native.hermitian_form_inertia(session=measured) == (2, 0, 0)
    cost = measured.consumed_nested_steps
    assert cost > 0
    session = SupervisedCallbackSession(max_nested_steps=2*cost-1)
    assert native.hermitian_form_inertia(session=session) == (2, 0, 0)
    assert session.consumed_nested_steps == cost
    with pytest.raises(ValueError, match="nested native request budget exhausted"):
        native.hermitian_form_inertia(session=session)
    spent = session.consumed_nested_steps
    assert cost <= spent <= session.max_nested_steps
    assert session.cancelled is True
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        native.hermitian_form_inertia(session=session)
    assert session.consumed_nested_steps == spent


@native_only
@pytest.mark.parametrize("cls", (Matrix, SparseMatrix))
def test_public_complex_result_checks_cancellation_after_verification(cls, monkeypatch):
    from sympy.core import supervised_callback_session
    from sympy.matrices import _definiteness

    matrix = cls([[3, I], [-I, 3]])
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as measured:
        assert matrix.is_positive_definite is True
    cost = measured.consumed_nested_steps
    original = _definiteness._unchanged
    cancelled = []
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as session:
        def cancel_at_publication(*args):
            original(*args)
            if session.consumed_nested_steps == cost-1:
                cancelled.append(True)
                session.cancel()

        monkeypatch.setattr(_definiteness, "_unchanged", cancel_at_publication)
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            _ = matrix.is_positive_definite
    assert cancelled == [True]
