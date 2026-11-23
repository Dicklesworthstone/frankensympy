"""Original parameter admission spends the caller's actual native work budget."""

import pytest
import sympy as s


pytestmark = pytest.mark.skipif(
    not hasattr(s.Matrix([[1]]), "_native_storage"),
    reason="native matrix parameter-admission contract",
)


def parameters(cls):
    x, y = s.symbols("parameter_x parameter_y")
    # The original reciprocal invokes an exact polynomial domain check.
    # Counting only its few syntax nodes misses that actual arithmetic.
    base = s.Pow(s.Add(x, y, 1, evaluate=False), 6, evaluate=False)
    return cls([[s.Pow(base, -1, evaluate=False)], [s.sin(x)]])


def session(steps):
    from sympy.core import SupervisedCallbackSession

    return SupervisedCallbackSession(max_nested_steps=steps, max_wall_time_ms=60_000)


@pytest.mark.parametrize("cls,name", [
    (s.Matrix, "validate_least_squares_parameters"),
    (s.Matrix, "validate_sparse_least_squares_parameters"),
    (s.ImmutableMatrix, "validate_sparse_least_squares_parameters"),
    (s.SparseMatrix, "validate_least_squares_parameters"),
    (s.ImmutableSparseMatrix, "validate_least_squares_parameters"),
])
def test_every_native_parameter_admission_debits_cumulative_request_work(cls, name):
    native = parameters(cls)._native_storage
    validate = getattr(native, name)
    measured = session(20_000_000)
    expected = validate(session=measured)
    cost = measured.consumed_nested_steps
    assert cost > 20
    if expected is not None:
        assert cost > sum(expected)
    exact = session(cost)
    assert validate(session=exact) == expected
    assert exact.consumed_nested_steps == cost
    assert not exact.cancelled
    short = session(cost - 1)
    with pytest.raises((ValueError, RuntimeError), match="budget|step|cancel"):
        validate(session=short)
    assert short.cancelled
    cumulative = session(2*cost - 1)
    assert validate(session=cumulative) == expected
    with pytest.raises((ValueError, RuntimeError), match="budget|step|cancel"):
        validate(session=cumulative)
    assert cumulative.cancelled
    assert cost <= cumulative.consumed_nested_steps <= cumulative.max_nested_steps
    cancelled = session(cost)
    cancelled.cancel()
    with pytest.raises((ValueError, RuntimeError), match="cancel"):
        validate(session=cancelled)
    assert cancelled.consumed_nested_steps == 0


@pytest.mark.parametrize("cls", [s.Matrix, s.SparseMatrix])
def test_retained_parameter_forest_uses_the_supplied_request_during_native_validation(cls):
    from sympy.matrices._sparse_family import _ParameterMatrix

    original = parameters(cls)
    measured = session(20_000_000)
    admitted = _ParameterMatrix(original, (2, 1), session=measured)
    admitted.unchanged()
    used = measured.consumed_nested_steps
    assert used > sum(admitted.sizes.values())
    short = session(used - 1)
    with pytest.raises((ValueError, RuntimeError), match="budget|step|cancel"):
        _ParameterMatrix(original, (2, 1), session=short)
    assert short.cancelled


def test_new_session_forwarding_does_not_weaken_original_parameter_domains():
    x = s.Symbol("held_parameter")
    zero = s.Add(x, -x, evaluate=False)
    pole = s.Mul(0, s.Pow(zero, -1, evaluate=False), evaluate=False)
    supplied = s.Matrix([[pole]])
    measured = session(20_000_000)
    with pytest.raises((ValueError, NotImplementedError)):
        supplied._native_storage.validate_least_squares_parameters(session=measured)
    assert measured.consumed_nested_steps > 0
