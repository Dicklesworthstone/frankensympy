"""Exact integer powers: oracle values/classes and native source obligations.

Ordinary fixtures use closed forms independent of matrix multiplication and
run unchanged against isolated SymPy1.14.0. Native-only cases check the bounded
Q(i) admission, original owners, interface pins and cumulative request ledger.
"""

import pytest

from sympy import (
    Add, Float, I, ImmutableMatrix, ImmutableSparseMatrix, Integer, Matrix,
    Mul, Pow, Rational, SparseMatrix, Symbol,
)


CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)
CANDIDATE = hasattr(Matrix([[1]]), "_native_storage")
native_only = pytest.mark.skipif(not CANDIDATE, reason="native source and request contract")
AMPLE_STEPS = 20_000_000


def assert_exact(actual, expected):
    assert actual.shape == expected.shape
    assert all((left-right).expand() == 0 for left, right in zip(actual, expected))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("exponent", [-7, -2, -1, 0, 1, 2, 13])
def test_gaussian_jordan_powers_preserve_values_class_and_identity(cls, exponent):
    source = cls([[1, I], [0, 1]])
    result = source**exponent
    assert type(result) is cls
    assert_exact(result, Matrix([[1, exponent*I], [0, 1]]))
    assert_exact(source, Matrix([[1, I], [0, 1]]))
    assert (result is source) is (exponent == 1)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("exponent", [-5, -1, 0, 2, 6])
def test_three_by_three_jordan_chain_uses_signed_binomial_coefficients(cls, exponent):
    source = cls([[1, I, 0], [0, 1, I], [0, 0, 1]])
    result = source**exponent
    assert_exact(result, Matrix([
        [1, exponent*I, -exponent*(exponent-1)//2],
        [0, 1, exponent*I], [0, 0, 1],
    ]))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("exponent", [-4, -1, 0, 3, 7])
def test_distinct_eigenvalues_have_exact_rational_negative_powers(cls, exponent):
    source = cls([[2, I], [0, 3]])
    two, three = Integer(2)**exponent, Integer(3)**exponent
    assert_exact(source**exponent, Matrix([[two, I*(three-two)], [0, three]]))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("exponent", [-3, -1, 0, 1, 4])
def test_literal_sympy_integer_exponents_are_supported(cls, exponent):
    source = cls([[0, 1], [-1, 0]])
    forms = ([[1, 0], [0, 1]], [[0, 1], [-1, 0]],
             [[-1, 0], [0, -1]], [[0, -1], [1, 0]])
    result = source**Integer(exponent)
    assert type(result) is cls
    assert_exact(result, Matrix(forms[exponent % 4]))


@pytest.mark.parametrize("cls", CLASSES)
def test_nilpotent_nonnegative_powers_and_singular_negative_refusal(cls):
    source = cls([[0, I], [0, 0]])
    assert_exact(source**0, Matrix([[1, 0], [0, 1]]))
    assert_exact(source**2, Matrix([[0, 0], [0, 0]]))
    with pytest.raises(ValueError):
        source**-1


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("exponent", [-1, 0, 3])
def test_empty_square_matrix_retains_class_and_shape(cls, exponent):
    result = cls(0, 0, [])**exponent
    assert type(result) is cls and result.shape == (0, 0)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("shape,exponent", [((0, 1), 0), ((1, 0), 1), ((2, 3), -1)])
def test_rectangular_sources_refuse_even_zero_and_one_exponents(cls, shape, exponent):
    with pytest.raises(ValueError):
        cls(shape[0], shape[1], [0]*(shape[0]*shape[1]))**exponent


@native_only
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("exponent", [-(1 << 63), (1 << 63)-1])
def test_full_signed_exponent_range_has_no_iteration_overflow(cls, exponent):
    source = cls([[1, I], [0, 1]])
    assert_exact(source**exponent, Matrix([[1, exponent*I], [0, 1]]))


@native_only
@pytest.mark.parametrize("cls", [Matrix, SparseMatrix])
@pytest.mark.parametrize("exponent", [-1, 0, 1, 2])
def test_original_held_pole_cannot_disappear_at_any_exponent(cls, exponent):
    zero = Add(I, -I, evaluate=False)
    hidden = Mul(0, Pow(zero, -1, evaluate=False), evaluate=False)
    with pytest.raises(ValueError, match="division by zero"):
        cls([[hidden]])**exponent


@native_only
@pytest.mark.parametrize("cls", [Matrix, SparseMatrix])
def test_defined_held_cells_and_native_only_results_are_reusable(cls):
    value = Pow(Add(1, I, evaluate=False), -1, evaluate=False)
    source = cls([[value]])
    retained = source._surface
    result = source**-2
    assert_exact(result, Matrix([[2*I]]))
    assert source._surface is retained and result._surface is None
    assert_exact(result**-1, Matrix([[-I/2]]))


@native_only
@pytest.mark.parametrize("cls", [Matrix, SparseMatrix])
def test_mutated_scalar_owner_is_not_a_new_source(cls, monkeypatch):
    scalar = Rational(29, 17)
    source = cls([[Mul(scalar, I, evaluate=False)]])
    monkeypatch.setattr(scalar, "_value", Integer(3)._value)
    with pytest.raises(RuntimeError, match="retained.*differ"):
        source**0


@native_only
@pytest.mark.parametrize("cls", [Matrix, SparseMatrix])
def test_mutation_during_binding_prevents_publication(cls, monkeypatch):
    from sympy.matrices import _definiteness

    source = cls([[I]])
    original = _definiteness._bind_gaussian

    def bind_then_mutate(*args):
        original(*args)
        source[0, 0] = 3

    monkeypatch.setattr(_definiteness, "_bind_gaussian", bind_then_mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        source**3


@native_only
@pytest.mark.parametrize("interface", ["gaussian_power", "to_sparse", "has_gaussian_rational_syntax"])
def test_replaced_native_interfaces_are_not_called(interface, monkeypatch):
    source = SparseMatrix([[I]])
    native_type = type(Matrix([[1]])._native_storage)
    called = []

    def replacement(*args, **kwargs):
        called.append(True)
        raise AssertionError("replaced interface executed")

    monkeypatch.setattr(native_type, interface, replacement)
    with pytest.raises(NotImplementedError, match="original.*native interfaces"):
        source**2
    assert called == []


@native_only
@pytest.mark.parametrize("cls", [SparseMatrix, ImmutableSparseMatrix])
def test_oversized_sparse_source_refuses_before_cell_copy_or_densification(cls, monkeypatch):
    from sympy.matrices import _definiteness

    source = cls(33, 33, {(0, 0): 1})

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized sparse source was copied")

    monkeypatch.setattr(_definiteness, "_rational_source_snapshot", forbidden)
    with pytest.raises(ValueError, match="dimension|32"):
        source**0


@native_only
@pytest.mark.parametrize("cls", [Matrix, SparseMatrix])
@pytest.mark.parametrize("exponent", [True, False, Rational(1, 2), Float(2), Symbol("n"), 1 << 63, -(1 << 63)-1])
def test_checked_integer_lane_rejects_unsupported_exponents(cls, exponent):
    with pytest.raises((TypeError, ValueError)):
        cls([[I]])**exponent


@native_only
def test_exponent_protocol_hooks_are_never_executed():
    called = []

    class CustomInteger(int):
        def __int__(self):
            called.append(True)
            return 2

        def __index__(self):
            called.append(True)
            return 2

    with pytest.raises(TypeError, match="literal integers"):
        SparseMatrix([[I]])**CustomInteger(2)
    assert called == []


@native_only
def test_replaced_integer_extraction_is_refused_without_execution(monkeypatch):
    from sympy.core import _native

    source, exponent = SparseMatrix([[I]]), Integer(17)
    called = []

    def replacement(*args):
        called.append(True)
        return 2

    monkeypatch.setattr(_native.Expr, "exact_numerator", replacement)
    with pytest.raises(NotImplementedError, match="original native integer interfaces"):
        source**exponent
    assert called == []


@native_only
def test_exponent_mutation_during_source_admission_is_terminal(monkeypatch):
    from sympy.matrices import _definiteness

    source, exponent = SparseMatrix([[I]]), Integer(17)
    original = _definiteness._bind_gaussian

    def bind_then_mutate(*args):
        original(*args)
        exponent._value = Integer(19)._value

    monkeypatch.setattr(_definiteness, "_bind_gaussian", bind_then_mutate)
    with pytest.raises(RuntimeError, match="exponent mutated"):
        source**exponent


@native_only
@pytest.mark.parametrize("cls", [Matrix, SparseMatrix])
def test_result_scalar_growth_is_bounded(cls):
    with pytest.raises(ValueError, match="limit|bits|height"):
        cls([[2]])**4096


@native_only
@pytest.mark.parametrize("cancelled", [False, True])
def test_terminal_requests_stop_before_copying_source(cancelled, monkeypatch):
    from sympy.core import SupervisedCallbackSession, supervised_callback_session
    from sympy.matrices import _definiteness

    source = SparseMatrix([[1, 2], [0, 1]])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS if cancelled else 0)
    if cancelled:
        session.cancel()

    def forbidden(*args, **kwargs):
        raise AssertionError("terminal request copied original source")

    monkeypatch.setattr(_definiteness, "_rational_source_snapshot", forbidden)
    with supervised_callback_session(session):
        with pytest.raises(RuntimeError if cancelled else ValueError, match="cancelled|budget exhausted"):
            source**-7


@native_only
@pytest.mark.parametrize("cls", [Matrix, SparseMatrix])
@pytest.mark.parametrize("exponent", [-3, 0, 7])
def test_inverse_generation_checking_and_publication_share_one_request(cls, exponent):
    from sympy.core import SupervisedCallbackSession, supervised_callback_session

    source = cls([[1, I], [0, 1]])
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as measured:
        source**exponent
    cost = measured.consumed_nested_steps
    assert cost > 0
    limited = SupervisedCallbackSession(max_nested_steps=2*cost-1)
    with supervised_callback_session(limited):
        source**exponent
        with pytest.raises(ValueError, match="budget exhausted"):
            source**exponent
    # A terminal one-step public-shell checkpoint records its attempted charge;
    # native multidimensional charges themselves remain atomic.
    assert cost <= limited.consumed_nested_steps <= limited.max_nested_steps + 1


@native_only
@pytest.mark.parametrize("cls", [Matrix, SparseMatrix])
def test_cancellation_at_publication_does_not_return_a_matrix(cls, monkeypatch):
    from sympy.core import supervised_callback_session
    from sympy.matrices import _row_reduction

    source = cls([[I]])
    publish = _row_reduction._publish
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as session:
        def cancel_before_publish(*args):
            session.cancel()
            return publish(*args)

        monkeypatch.setattr(_row_reduction, "_publish", cancel_before_publish)
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            source**2


@native_only
def test_standalone_dense_symbolic_and_large_rational_lanes_are_preserved():
    x = Symbol("power_x")
    assert_exact(Matrix([[x, 1], [0, x]])**2, Matrix([[x*x, 2*x], [0, x*x]]))
    source = Matrix.eye(33)
    assert_exact(source**2, source)


@native_only
def test_inherited_dense_rational_constructor_and_owner_guard_are_preserved():
    constructed = []

    class CustomMatrix(Matrix):
        def _new(self, *args, **kwargs):
            constructed.append(True)
            return type(self)(*args, **kwargs)

    result = CustomMatrix([[2]])**-2
    assert type(result) is CustomMatrix
    assert_exact(result, Matrix([[Rational(1, 4)]]))
    assert constructed == [True]

    class MutatingMatrix(Matrix):
        def _new(self, *args, **kwargs):
            self[0, 0] = 7
            return type(self)(*args, **kwargs)

    with pytest.raises(RuntimeError, match="mutated"):
        MutatingMatrix([[2]])**2


@native_only
def test_new_sparse_subclass_lane_refuses_before_computation():
    class CustomSparse(SparseMatrix):
        pass

    with pytest.raises(NotImplementedError, match="custom sparse matrix subclasses"):
        CustomSparse([[2]])**2
