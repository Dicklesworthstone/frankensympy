"""Ordinary exact Q(i) solves/inverses and original-source publication.

Ordinary fixtures run unchanged against isolated SymPy 1.14.0. Native-only
cases exercise retained source identity, bounded requests and terminal errors.
"""

import pytest

from sympy import (
    Add, I, ImmutableMatrix, ImmutableSparseMatrix, Integer, Matrix, Mul, Pow,
    Rational, SparseMatrix, Symbol, eye, zeros,
)


CLASSES = (Matrix, ImmutableMatrix)
RHS_CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)
CANDIDATE = hasattr(Matrix([[1]]), "_native_storage")
native_only = pytest.mark.skipif(not CANDIDATE, reason="native source and request contract")
AMPLE_STEPS = 20_000_000


def assert_exact(left, right):
    assert left.shape == right.shape
    assert all((a - b).expand() == 0 for a, b in zip(left, right))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("method", ["solve", "LUsolve"])
@pytest.mark.parametrize("entries,rhs,expected", [
    ([[I]], [[1]], [[-I]]),
    ([[1 + I]], [[3 + I]], [[2 - I]]),
    ([[I, 1], [0, 1]], [[3 + 2*I], [1]], [[2 - 2*I], [1]]),
    ([[1, 2], [0, 1]], [[3 + I, 1 - 4*I], [1, I]], [[1 + I, 1 - 6*I], [1, I]]),
])
def test_ordinary_square_methods_solve_exact_complex_systems(cls, method, entries, rhs, expected):
    matrix, right = cls(entries), cls(rhs)
    result = getattr(matrix, method)(right)
    assert type(result) is cls
    assert_exact(result, Matrix(expected))
    assert_exact(matrix * result, right)
    assert_exact(matrix, cls(entries))
    assert_exact(right, cls(rhs))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("entries,expected", [
    ([[I]], [[-I]]),
    ([[1 + I]], [[Rational(1, 2) - I/2]]),
    ([[I, 1], [0, 1]], [[-I, I], [0, 1]]),
    ([[1, I], [I, 1]], [[Rational(1, 2), -I/2], [-I/2, Rational(1, 2)]]),
])
def test_ordinary_inverse_checks_both_exact_products(cls, entries, expected):
    matrix = cls(entries)
    inverse = matrix.inv()
    assert type(inverse) is cls
    assert_exact(inverse, Matrix(expected))
    assert_exact(matrix * inverse, eye(matrix.rows))
    assert_exact(inverse * matrix, eye(matrix.rows))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("rhs_cls", RHS_CLASSES)
@pytest.mark.parametrize("method", ["solve", "LUsolve"])
def test_consistent_tall_systems_preserve_pinned_result_classes(cls, rhs_cls, method):
    matrix = cls([[1], [I]])
    right = rhs_cls([[2 - I, 3 + I], [1 + 2*I, -1 + 3*I]])
    result = getattr(matrix, method)(right)
    assert type(result) is (rhs_cls if method == "LUsolve" else cls)
    assert_exact(result, Matrix([[2 - I, 3 + I]]))
    assert_exact(matrix * result, right)


@pytest.mark.parametrize("method", ["solve", "LUsolve"])
def test_inconsistent_tall_requests_never_return_the_least_squares_projection(method):
    matrix, right = Matrix([[1], [2*I]]), Matrix([[1], [0]])
    assert_exact(matrix.solve_least_squares(right), Matrix([[Rational(1, 5)]]))
    with pytest.raises(ValueError):
        getattr(matrix, method)(right)


@pytest.mark.parametrize("cls", CLASSES)
def test_symbolic_inverse_remains_a_separate_preselected_lane(cls):
    a = Symbol("a")
    matrix = cls([[a, 1], [1, 0]])
    assert_exact(matrix.inv(), Matrix([[0, 1], [1, -a]]))


@pytest.mark.parametrize("cls", CLASSES)
def test_empty_square_inverse_preserves_its_shape(cls):
    result = cls(0, 0, []).inv()
    assert type(result) is cls and result.shape == (0, 0)


@native_only
def test_exact_inverse_alias_and_native_only_results_can_be_used_again():
    matrix = Matrix([[I, 1], [0, 1]])
    object.__setattr__(matrix, "_surface", None)
    inverse = matrix.inverse()
    assert inverse._surface is None
    assert_exact(inverse.inverse(), matrix)
    assert_exact(matrix.LUsolve(inverse), Matrix([[-1, 1 + I], [0, 1]]))


@native_only
@pytest.mark.parametrize("method", ["solve", "LUsolve", "inv"])
def test_singular_complex_requests_remain_terminal_refusals(method):
    matrix = Matrix([[1, I], [2, 2*I]])
    with pytest.raises(ValueError, match="singular"):
        if method == "inv":
            matrix.inv()
        else:
            getattr(matrix, method)(Matrix([[1], [2]]))


@native_only
def test_empty_rhs_retains_full_rank_and_zero_unknowns_retain_consistency():
    matrix = Matrix([[1], [I]])
    assert matrix.LUsolve(zeros(2, 0)).shape == (1, 0)
    with pytest.raises(ValueError):
        matrix.solve(zeros(2, 0))
    assert Matrix(2, 0, []).solve(zeros(2, 2)).shape == (0, 2)
    with pytest.raises(ValueError, match="inconsistent"):
        Matrix(2, 0, []).solve(Matrix([[I], [0]]))


@native_only
@pytest.mark.parametrize("method", ["solve", "LUsolve", "inv"])
def test_held_defined_originals_are_retained_through_publication(method):
    source = Pow(Add(1, I, evaluate=False), -1, evaluate=False)
    right = Pow(Add(1, -I, evaluate=False), -1, evaluate=False)
    matrix, rhs = Matrix([[source]]), Matrix([[right]])
    old_matrix, old_rhs = matrix._surface, rhs._surface
    result = matrix.inv() if method == "inv" else getattr(matrix, method)(rhs)
    assert_exact(result, Matrix([[1 + I if method == "inv" else I]]))
    assert matrix._surface is old_matrix and matrix._surface[0] is source
    assert rhs._surface is old_rhs and rhs._surface[0] is right


@native_only
@pytest.mark.parametrize("method", ["solve", "LUsolve", "inv"])
@pytest.mark.parametrize("wrapper", [
    lambda pole: Mul(0, pole, evaluate=False),
    lambda pole: Pow(pole, 0, evaluate=False),
])
def test_hidden_undefined_originals_cannot_become_zero_or_identity(method, wrapper):
    zero = Add(I, -I, evaluate=False)
    source = wrapper(Pow(zero, -1, evaluate=False))
    matrix = Matrix([[source]])
    with pytest.raises(ValueError, match="division by zero"):
        if method == "inv":
            matrix.inv()
        else:
            getattr(matrix, method)(Matrix([[1]]))


@native_only
@pytest.mark.parametrize("method", ["solve", "LUsolve", "inv"])
@pytest.mark.parametrize("mutation", ["root", "child", "arguments", "surface", "structural"])
def test_mutated_retained_scalar_graphs_are_terminal(method, mutation, monkeypatch):
    coefficient = Rational(19, 13)
    child = Mul(coefficient, I, evaluate=False)
    source = Add(1, child, evaluate=False)
    matrix = Matrix([[source]])
    if mutation == "root":
        monkeypatch.setattr(source, "_value", Integer(3)._value)
    elif mutation == "child":
        monkeypatch.setattr(coefficient, "_value", Integer(3)._value)
    elif mutation == "arguments":
        monkeypatch.setattr(child, "_args", (Integer(3), I))
    elif mutation == "surface":
        monkeypatch.setattr(child, "_surface_args", (Integer(3), I), raising=False)
    else:
        monkeypatch.setattr(child, "_struct_args", (Integer(3), I), raising=False)
    with pytest.raises(RuntimeError, match="retained.*differ"):
        if method == "inv":
            matrix.inv()
        else:
            getattr(matrix, method)(Matrix([[1]]))


@native_only
@pytest.mark.parametrize("rhs_cls", [SparseMatrix, ImmutableSparseMatrix])
def test_sparse_rhs_retained_sources_are_checked_before_conversion(rhs_cls, monkeypatch):
    coefficient = Rational(23, 17)
    rhs = rhs_cls([[Mul(coefficient, I, evaluate=False)]])
    monkeypatch.setattr(coefficient, "_value", Integer(5)._value)
    with pytest.raises(RuntimeError, match="retained.*differ"):
        Matrix([[I]]).LUsolve(rhs)


@native_only
@pytest.mark.parametrize("method", ["solve", "inv", "has_gaussian_rational_syntax"])
def test_replaced_native_interfaces_are_refused_without_execution(method, monkeypatch):
    matrix = Matrix([[I]])
    called = []

    def replaced(*args, **kwargs):
        called.append(True)
        raise AssertionError("replaced native method executed")

    monkeypatch.setattr(type(matrix._native_storage), method, replaced)
    with pytest.raises(NotImplementedError, match="original native interfaces"):
        if method == "solve":
            matrix.solve(Matrix([[1]]))
        else:
            matrix.inv()
    assert called == []


@native_only
@pytest.mark.parametrize("method", ["solve", "LUsolve", "inv"])
def test_source_mutation_after_binding_prevents_exact_publication(method, monkeypatch):
    from sympy.matrices import _definiteness

    matrix = Matrix([[I]])
    original = _definiteness._bind_gaussian

    def bind_then_mutate(*args):
        original(*args)
        matrix[0, 0] = 3

    monkeypatch.setattr(_definiteness, "_bind_gaussian", bind_then_mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        if method == "inv":
            matrix.inv()
        else:
            getattr(matrix, method)(Matrix([[1]]))


@native_only
@pytest.mark.parametrize("method", ["solve", "inv"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_terminal_requests_stop_before_source_copying(method, cancelled, monkeypatch):
    from sympy.core import SupervisedCallbackSession, supervised_callback_session
    from sympy.matrices import _definiteness

    matrix = Matrix([[I]])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS if cancelled else 0)
    if cancelled:
        session.cancel()

    def forbidden(*args):
        raise AssertionError("terminal request copied source cells")

    monkeypatch.setattr(_definiteness, "_bind_gaussian", forbidden)
    with supervised_callback_session(session):
        with pytest.raises(RuntimeError if cancelled else ValueError, match="cancelled|budget exhausted"):
            if method == "inv":
                matrix.inv()
            else:
                matrix.solve(Matrix([[1]]))


@native_only
@pytest.mark.parametrize("method", ["solve", "inv"])
def test_repeated_native_requests_keep_one_cumulative_allowance(method):
    from sympy.core import SupervisedCallbackSession

    matrix = Matrix([[I, 1], [0, 1]])._native_storage
    rhs = Matrix([[1], [I]])._native_storage

    def run(session):
        return matrix.inv(session=session) if method == "inv" else matrix.solve(rhs, session=session)

    measured = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    run(measured)
    cost = measured.consumed_nested_steps
    assert cost > 0
    limited = SupervisedCallbackSession(max_nested_steps=2*cost - 1)
    run(limited)
    assert limited.consumed_nested_steps == cost
    with pytest.raises(ValueError, match="budget exhausted"):
        run(limited)
    spent = limited.consumed_nested_steps
    assert cost <= spent <= limited.max_nested_steps
    with pytest.raises(RuntimeError, match="cancelled|tainted"):
        run(limited)
    assert limited.consumed_nested_steps == spent


@native_only
@pytest.mark.parametrize("method", ["solve", "inv"])
def test_publication_cancellation_after_native_acceptance_is_terminal(method, monkeypatch):
    from sympy.core import supervised_callback_session
    from sympy.matrices import _pseudoinverse

    matrix = Matrix([[I]])
    original = _pseudoinverse._publish
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS) as session:
        def cancel_before_publish(*args):
            session.cancel()
            return original(*args)

        monkeypatch.setattr(_pseudoinverse, "_publish", cancel_before_publish)
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            if method == "inv":
                matrix.inv()
            else:
                matrix.solve(Matrix([[1]]))


@native_only
def test_custom_matrix_subclasses_do_not_enter_the_new_exact_lane():
    class CustomMatrix(Matrix):
        pass

    with pytest.raises(NotImplementedError, match="custom matrix subclasses"):
        CustomMatrix([[I]]).inv()
    with pytest.raises(NotImplementedError, match="custom matrix subclasses"):
        CustomMatrix([[I]]).solve(Matrix([[1]]))


@native_only
def test_plain_literal_rational_subclasses_keep_legacy_values_and_result_construction():
    constructed = []

    class CustomMatrix(Matrix):
        def _new(self, *args, **kwargs):
            constructed.append(True)
            return type(self)(*args, **kwargs)

    matrix = CustomMatrix([[1, 2], [0, 1]])
    rhs = Matrix([[3], [1]])
    inverse = matrix.inv()
    solved = matrix.solve(rhs)
    lu = matrix.LUsolve(rhs)
    assert type(inverse) is type(solved) is type(lu) is CustomMatrix
    assert_exact(inverse, Matrix([[1, -2], [0, 1]]))
    assert_exact(solved, Matrix([[1], [1]]))
    assert_exact(lu, solved)
    assert constructed == [True, True, True]


@native_only
def test_rational_subclass_result_hooks_cannot_rebase_the_original_source():
    class CustomMatrix(Matrix):
        def _new(self, *args, **kwargs):
            self[0, 0] = 7
            return type(self)(*args, **kwargs)

    with pytest.raises(RuntimeError, match="mutated"):
        CustomMatrix([[1]]).inv()


@native_only
def test_plain_subclass_symbolic_inverse_keeps_the_existing_lane():
    class CustomMatrix(Matrix):
        pass

    a = Symbol("a")
    inverse = CustomMatrix([[a, 1], [1, 0]]).inv()
    assert type(inverse) is CustomMatrix
    assert_exact(inverse, Matrix([[0, 1], [1, -a]]))


@native_only
def test_literal_rational_inverse_keeps_its_larger_source_envelope():
    # This original literal source exceeds both the 32-axis Gaussian lane
    # and its 16,384-node syntax walk. The old rational lane still reaches
    # its established RREF work limit rather than the Gaussian source cap.
    with pytest.raises(ValueError, match="RREF exceeds the operation limit of 10000000"):
        zeros(129).inv()


@native_only
@pytest.mark.parametrize("binding", ["_native", "shape", "_new"])
def test_subclass_publication_rechecks_original_accessor_and_constructor_bindings(binding):
    class CustomMatrix(Matrix):
        def _new(self, *args, **kwargs):
            result = type(self)(*args, **kwargs)
            setattr(type(self), binding, property(lambda self: None))
            return result

    with pytest.raises(RuntimeError, match="subclass changed"):
        CustomMatrix([[1]]).inv()


@native_only
@pytest.mark.parametrize("rhs_base", [Matrix, SparseMatrix])
def test_rectangular_literal_custom_rhs_keeps_the_legacy_coefficient_result_class(rhs_base):
    class CustomRhs(rhs_base):
        pass

    result = ImmutableMatrix([[1], [2]]).LUsolve(CustomRhs([[3], [6]]))
    assert type(result) is ImmutableMatrix
    assert_exact(result, Matrix([[3]]))


@native_only
def test_literal_subclasses_do_not_bypass_new_complex_or_active_request_guards():
    from sympy.core import supervised_callback_session

    class CustomMatrix(Matrix):
        pass

    with pytest.raises(NotImplementedError, match="custom matrix subclasses"):
        CustomMatrix([[1]]).solve(Matrix([[I]]))
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS):
        with pytest.raises(NotImplementedError, match="custom matrix subclasses"):
            CustomMatrix([[1]]).inv()
