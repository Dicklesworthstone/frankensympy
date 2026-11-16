"""Matrix classification shares its caller's budget through result publication."""

from unittest.mock import patch

import pytest
from sympy import Matrix, SparseMatrix
from sympy.core import SupervisedCallbackSession, supervised_callback_session
from sympy.matrices import _definiteness


CLASSES = (Matrix, SparseMatrix)
AMPLE_STEPS = 10_000_000


def native_cost(matrix):
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    assert matrix._native_storage.quadratic_form_inertia(session=session) == (2, 0, 0)
    assert session.consumed_nested_steps > 0
    return session.consumed_nested_steps


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("cancelled", [False, True])
def test_terminal_active_requests_refuse_before_rational_source_copy(cls, cancelled):
    matrix = cls([[2, 1], [1, 2]])
    session = SupervisedCallbackSession(max_nested_steps=0 if not cancelled else AMPLE_STEPS)
    if cancelled:
        session.cancel()
    expected = RuntimeError if cancelled else ValueError
    with patch.object(_definiteness, "_rational_source_snapshot",
                      side_effect=AssertionError("terminal request copied a rational source")):
        with supervised_callback_session(session):
            with pytest.raises(expected, match="cancelled|budget exhausted"):
                _ = matrix.is_positive_definite


@pytest.mark.parametrize("cls", CLASSES)
def test_native_phases_and_repeated_calls_share_one_atomic_allowance(cls):
    matrix = cls([[2, 1], [1, 2]])
    cost = native_cost(matrix)
    session = SupervisedCallbackSession(max_nested_steps=2*cost-1)
    native = matrix._native_storage
    assert native.quadratic_form_inertia(session=session) == (2, 0, 0)
    assert session.consumed_nested_steps == cost
    with pytest.raises(ValueError, match="nested native request budget exhausted"):
        native.quadratic_form_inertia(session=session)
    consumed = session.consumed_nested_steps
    assert cost <= consumed <= session.max_nested_steps
    assert session.cancelled is True
    # Catching the first exhaustion cannot create a fresh request allowance.
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        native.quadratic_form_inertia(session=session)
    assert session.consumed_nested_steps == consumed


@pytest.mark.parametrize("cls", CLASSES)
def test_public_property_accounts_for_native_work_and_checks_late_cancellation(cls):
    matrix = cls([[2, 1], [1, 2]])
    cost = native_cost(matrix)
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as measured:
        assert matrix.is_positive_definite is True
    assert measured.consumed_nested_steps > cost

    original_check = _definiteness._check_rational_source
    checked = []
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as late:
        def cancel_after_native_result(snapshot):
            original_check(snapshot)
            checked.append(late.consumed_nested_steps)
            late.cancel()

        with patch.object(_definiteness, "_check_rational_source", cancel_after_native_result):
            with pytest.raises(RuntimeError, match="cancelled|tainted"):
                _ = matrix.is_positive_definite
    assert len(checked) == 1
    assert checked[0] > cost
    assert late.cancelled is True
