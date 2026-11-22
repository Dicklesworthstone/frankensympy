"""Exact complex pseudoinverses and the original-source publication boundary."""

import pytest

from sympy import (
    Add, I, ImmutableMatrix, Integer, Matrix, Mul, Pow, Rational, Symbol, zeros,
)


CLASSES = (Matrix, ImmutableMatrix)
CANDIDATE = hasattr(Matrix([[1]]), "_native_storage")
native_only = pytest.mark.skipif(not CANDIDATE, reason="native source and request contract")
AMPLE_STEPS = 20_000_000


def assert_exact(left, right):
    assert left.shape == right.shape
    assert all((a - b).expand() == 0 for a, b in zip(left, right))


def assert_penrose(matrix, inverse):
    assert_exact(matrix * inverse * matrix, matrix)
    assert_exact(inverse * matrix * inverse, inverse)
    assert_exact((matrix * inverse).H, matrix * inverse)
    assert_exact((inverse * matrix).H, inverse * matrix)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entries,expected", [
    ([[I]], [[-I]]),
    ([[1 + I]], [[Rational(1, 2) - I/2]]),
    # A = u*v has ||u||^2 = 2 and ||v||^2 = 5. A+ = v* u* / 10.
    ([[1, 2*I], [I, -2]], [[Rational(1, 10), -I/10], [-I/5, Rational(-1, 5)]]),
    ([[1, I]], [[Rational(1, 2)], [-I/2]]),
    ([[1], [I]], [[Rational(1, 2), -I/2]]),
    ([[1, I], [0, 0], [0, 1]], [[1, 0, -I], [0, 0, 1]]),
    ([[I, 0], [0, 2*I]], [[-I, 0], [0, -I/2]]),
])
def test_exact_complex_singular_rectangular_and_invertible_results(cls, entries, expected):
    matrix = cls(entries)
    inverse = matrix.pinv()
    assert type(inverse) is cls
    assert_exact(inverse, Matrix(expected))
    assert_penrose(matrix, inverse)
    assert_exact(matrix, cls(entries))


@pytest.mark.parametrize("cls", CLASSES)
def test_adjoint_involution_and_nonreal_scaling(cls):
    matrix = cls([[1, I], [2, 2*I], [I, -1]])
    inverse = matrix.pinv()
    assert_exact(inverse.pinv(), matrix)
    assert_exact(matrix.H.pinv(), inverse.H)
    assert_exact((I * matrix).pinv(), -I * inverse)
    assert_exact((Rational(2, 3) * matrix).pinv(), Rational(3, 2) * inverse)


@pytest.mark.parametrize("shape", ((0, 0), (0, 3), (4, 0), (2, 3)))
def test_zero_and_empty_real_inputs_keep_existing_transposed_shape(shape):
    result = zeros(*shape).pinv()
    assert result.shape == tuple(reversed(shape))
    assert result == zeros(*reversed(shape))


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_native_only_complex_results_can_be_used_again(cls):
    matrix = cls([[1, I]])
    object.__setattr__(matrix, "_surface", None)
    result = matrix.pinv()
    assert result._surface is None
    assert_exact(result.pinv(), matrix)


@native_only
@pytest.mark.parametrize("cls", CLASSES)
def test_held_defined_complex_sources_are_checked_without_changing_them(cls):
    source = Pow(Add(1, I, evaluate=False), -1, evaluate=False)
    matrix = cls([[source]])
    original = matrix._surface
    assert_exact(matrix.pinv(), Matrix([[1 + I]]))
    assert matrix._surface is original
    assert matrix._surface[0] is source


@native_only
@pytest.mark.parametrize("wrapper", [
    lambda pole: Mul(0, pole, evaluate=False),
    lambda pole: Pow(pole, 0, evaluate=False),
])
def test_hidden_undefined_complex_sources_do_not_publish_zero_or_identity(wrapper):
    zero = Add(I, -I, evaluate=False)
    pole = Pow(zero, -1, evaluate=False)
    with pytest.raises(ValueError, match="division by zero"):
        Matrix([[wrapper(pole)]]).pinv()


@native_only
@pytest.mark.parametrize("mutation", ["root", "child", "arguments", "surface", "structural"])
def test_retained_complex_source_mutations_cannot_publish_a_stale_answer(mutation):
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
        matrix.pinv()


@native_only
def test_matrix_mutation_after_source_binding_is_terminal(monkeypatch):
    from sympy.matrices import _definiteness

    matrix = Matrix([[1, I]])
    original = _definiteness._bind_gaussian

    def mutate_after_binding(*args):
        original(*args)
        matrix[0, 0] = 2

    monkeypatch.setattr(_definiteness, "_bind_gaussian", mutate_after_binding)
    with pytest.raises(RuntimeError, match="mutated"):
        matrix.pinv()


@native_only
def test_changed_native_entry_reader_is_not_executed(monkeypatch):
    matrix = Matrix([[1, I]])
    called = []

    def changed(self):
        called.append(True)
        return []

    monkeypatch.setattr(type(matrix._native_storage), "flat", changed)
    with pytest.raises(NotImplementedError, match="original native interfaces"):
        matrix.pinv()
    assert called == []


@native_only
def test_symbolic_and_oversized_complex_sources_remain_explicit_refusals():
    with pytest.raises(ValueError, match="exact rational entries only"):
        Matrix([[Symbol("rank_may_change")]]).pinv()
    with pytest.raises(ValueError, match="dimension"):
        Matrix(33, 1, [I] * 33).pinv()
    with pytest.raises(ValueError, match="coefficient|bit"):
        Matrix([[Mul(2**4096, I, evaluate=False)]]).pinv()


@native_only
@pytest.mark.parametrize("cancelled", [False, True])
def test_terminal_request_stops_before_copying_complex_cells(cancelled, monkeypatch):
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
            matrix.pinv()


@native_only
def test_repeated_native_calls_keep_one_cumulative_request_allowance():
    from sympy.core import SupervisedCallbackSession

    native = Matrix([[1, I], [2, 2*I]])._native_storage
    measured = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    native.pinv(session=measured)
    cost = measured.consumed_nested_steps
    assert cost > 0
    limited = SupervisedCallbackSession(max_nested_steps=2*cost - 1)
    native.pinv(session=limited)
    assert limited.consumed_nested_steps == cost
    with pytest.raises(ValueError, match="budget exhausted"):
        native.pinv(session=limited)
    spent = limited.consumed_nested_steps
    assert cost <= spent <= limited.max_nested_steps
    assert limited.cancelled is True
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        native.pinv(session=limited)
    assert limited.consumed_nested_steps == spent


@native_only
def test_publication_cancellation_is_checked_after_independent_verification(monkeypatch):
    from sympy.core import supervised_callback_session
    from sympy.matrices import _definiteness

    matrix = Matrix([[1, I]])
    original = _definiteness._unchanged
    visits = []
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as session:
        def cancel_after_native(*args):
            original(*args)
            visits.append(session.consumed_nested_steps)
            if len(visits) == 2:
                session.cancel()

        monkeypatch.setattr(_definiteness, "_unchanged", cancel_after_native)
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            matrix.pinv()
    assert len(visits) == 2
    assert visits[1] > visits[0]


@native_only
def test_custom_matrix_subclasses_require_their_explicit_override():
    class Custom(Matrix):
        pass

    with pytest.raises(NotImplementedError, match="built-in dense Matrix"):
        Custom([[I]]).pinv()
