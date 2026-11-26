"""Exact constant matrices with polynomial right-hand sides.

Mathematical examples and mixed-class result rules are independently checked
against SymPy 1.14.0. Native-only cases pin the bounded original-source and
terminal-request contracts of this implementation.
"""

import pytest

from sympy import (
    Add, Dummy, Float, Function, I, ImmutableMatrix, ImmutableSparseMatrix,
    Integer, Matrix, Mul, Pow, Rational, SparseMatrix, Symbol, sin, symbols, zeros,
)


CLASSES = (Matrix, ImmutableMatrix, SparseMatrix, ImmutableSparseMatrix)
DENSE = (Matrix, ImmutableMatrix)
SPARSE = (SparseMatrix, ImmutableSparseMatrix)
METHODS = ("solve", "LUsolve")
CANDIDATE = hasattr(Matrix([[1]]), "_native_storage")
native_only = pytest.mark.skipif(not CANDIDATE, reason="native source and request contract")
AMPLE_STEPS = 20_000_000


def assert_exact(actual, expected):
    assert actual.shape == expected.shape
    assert all((left-right).expand() == 0 for left, right in zip(actual, expected))


def square_result_class(left, right, method):
    # Isolated SymPy 1.14.0: LUsolve follows the RHS class for every shape.
    return right if method == "LUsolve" or left is SparseMatrix else left


@pytest.mark.parametrize("cls", CLASSES)
def test_solved_polynomials_keep_ordinary_structural_expression_equality(cls):
    x, y = symbols("poly_rhs_canonical_x poly_rhs_canonical_y")
    expected = cls([[x-2*y, 3*I*x**2*y, x+3+2*I, (1+I)*x**2*y]])
    actual = cls([[1]]).LUsolve(expected)
    assert type(actual) is cls
    assert actual.shape == expected.shape
    assert all(value == wanted for value, wanted in zip(actual, expected))


@pytest.mark.parametrize("left", CLASSES)
@pytest.mark.parametrize("right", CLASSES)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("symbolic", (False, True))
def test_square_rhs_keeps_exact_values_and_pinned_result_class(left, right, method, symbolic):
    x, y = (symbols("poly_rhs_square_x poly_rhs_square_y") if symbolic
            else (Integer(3), Integer(5)))
    a, b = left([[2, 1], [1, -1]]), right([[x], [y]])
    before_a, before_b = a.copy(), b.copy()
    result = getattr(a, method)(b)
    assert type(result) is square_result_class(left, right, method)
    assert_exact(result, Matrix([[(x+y)/3], [(x-2*y)/3]]))
    assert_exact(a*result, b)
    assert a == before_a and b == before_b


@pytest.mark.parametrize("left", CLASSES)
@pytest.mark.parametrize("right", CLASSES)
def test_tall_lusolve_checks_all_polynomial_rows_and_preserves_rhs_class(left, right):
    x, y = symbols("poly_rhs_tall_x poly_rhs_tall_y")
    a = left([[1, 0], [0, 2], [1, 2]])
    b = right([[x*x, x*y], [2*y, 2*(x+y)], [x*x+2*y, x*y+2*(x+y)]])
    result = a.LUsolve(b)
    assert type(result) is right
    assert_exact(result, Matrix([[x*x, x*y], [y, x+y]]))
    assert_exact(a*result, b)


@pytest.mark.parametrize("left", DENSE)
@pytest.mark.parametrize("right", (Matrix, ImmutableSparseMatrix))
def test_dense_solve_accepts_a_consistent_tall_polynomial_system(left, right):
    x = Symbol("poly_rhs_tall_solve")
    a, b = left([[1], [-2], [3]]), right([[x**3], [-2*x**3], [3*x**3]])
    result = a.solve(b)
    assert type(result) is left
    assert_exact(result, Matrix([[x**3]]))


@pytest.mark.parametrize("left", SPARSE)
def test_sparse_solve_keeps_its_rectangular_refusal(left):
    x = Symbol("poly_rhs_sparse_shape")
    with pytest.raises(ValueError):
        left([[1], [2]]).solve(Matrix([[x], [2*x]]))


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("method", METHODS)
def test_nonlinear_parameter_polynomials_and_multiple_rhs_are_linear_system_data(cls, method):
    x, y = symbols("poly_rhs_nonlinear_x poly_rhs_nonlinear_y")
    p, q = (x+y)**3, x*y + Rational(1, 3)
    r, t = x*x-y*y, y**4 - 2*x
    a = cls([[2, 1], [1, -1]])
    b = cls([[2*p+q, 2*r+t], [p-q, r-t]])
    result = getattr(a, method)(b)
    assert_exact(result, Matrix([[p, r], [q, t]]))
    assert_exact(a*result, b)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("method", METHODS)
def test_gaussian_coefficients_and_polynomial_rhs_use_exact_complex_arithmetic(cls, method):
    x, y = symbols("poly_rhs_complex_x poly_rhs_complex_y")
    p, q = x*x + I*y, x*y - Rational(2, 3)*I
    a = cls([[I, 1], [1, I]])
    b = cls([[I*p+q], [p+I*q]])
    result = getattr(a, method)(b)
    assert_exact(result, Matrix([[p], [q]]))
    assert_exact(a*result, b)


@pytest.mark.parametrize("method", METHODS)
def test_tiny_exact_pivot_does_not_round_polynomial_coefficients(method):
    x = Symbol("poly_rhs_tiny")
    tiny = Rational(1, 10**200)
    a, b = Matrix([[tiny*I, 0], [0, 1]]), Matrix([[tiny*x], [x*x]])
    assert_exact(getattr(a, method)(b), Matrix([[-I*x], [x*x]]))


@pytest.mark.parametrize("method", METHODS)
def test_inconsistent_polynomial_residual_is_not_accepted_from_sampled_values(method):
    x = Symbol("poly_rhs_inconsistent")
    # This residual vanishes at 0 and 1, but is not the zero polynomial.
    a, b = Matrix([[1], [1]]), Matrix([[x*x], [x*x+x*(x-1)]])
    with pytest.raises(ValueError):
        getattr(a, method)(b)


@pytest.mark.parametrize("method", METHODS)
def test_an_inconsistent_later_rhs_column_rejects_the_whole_result(method):
    x, y = symbols("poly_rhs_inconsistent_x poly_rhs_inconsistent_y")
    a = Matrix([[1, 0], [0, 1], [1, 1]])
    b = Matrix([[x, y], [x*x, y*y], [x+x*x, y+y*y+1]])
    with pytest.raises(ValueError):
        getattr(a, method)(b)


@pytest.mark.parametrize("method", METHODS)
def test_rank_deficiency_does_not_silently_choose_free_parameters(method):
    x = Symbol("poly_rhs_rank")
    with pytest.raises(ValueError):
        getattr(Matrix([[1, 2], [2, 4]]), method)(Matrix([[x], [2*x]]))


def test_wide_polynomial_systems_keep_the_unique_solver_refusals():
    x = Symbol("poly_rhs_wide")
    a, b = Matrix([[1, 2, 3]]), Matrix([[x*x]])
    with pytest.raises(ValueError):
        a.solve(b)
    with pytest.raises(NotImplementedError):
        a.LUsolve(b)


@native_only
@pytest.mark.parametrize("method", METHODS)
def test_zero_unknowns_distinguish_identically_zero_from_symbolic_rhs(method):
    x = Symbol("poly_rhs_zero_unknowns")
    a = Matrix(2, 0, [])
    cancelled = Add(x, Mul(-1, x, evaluate=False), evaluate=False)
    assert getattr(a, method)(Matrix([[cancelled], [0]])).shape == (0, 1)
    with pytest.raises(ValueError):
        getattr(a, method)(Matrix([[x], [0]]))


@native_only
def test_polynomial_rhs_does_not_claim_a_new_gauss_jordan_family_contract():
    x = Symbol("poly_rhs_affine_deferred")
    with pytest.raises((ValueError, NotImplementedError)):
        Matrix([[1, 2]]).gauss_jordan_solve(Matrix([[x]]))
    solution, parameters, free = Matrix([[1, 2]]).gauss_jordan_solve(Matrix([[3]]), freevar=True)
    assert free == [1] and parameters.shape == (1, 1)
    assert_exact(solution, Matrix([[3-2*parameters[0, 0]], [parameters[0, 0]]]))


@pytest.mark.parametrize("method", METHODS)
def test_same_name_symbols_and_distinct_dummies_keep_their_original_identities(method):
    positive = Symbol("poly_rhs_same", positive=True)
    negative = Symbol("poly_rhs_same", negative=True)
    first, second = Dummy("poly_rhs_same"), Dummy("poly_rhs_same")
    a = Matrix([[1, 1], [1, -1]])
    b = Matrix([[positive+negative+first+second], [positive-negative+first-second]])
    result = getattr(a, method)(b)
    assert_exact(result, Matrix([[positive+first], [negative+second]]))
    owners = result[0, 0].free_symbols | result[1, 0].free_symbols
    assert len(owners) == 4
    assert all(any(owner is original for owner in owners)
               for original in (positive, negative, first, second))


@native_only
@pytest.mark.parametrize("cls", (Matrix, ImmutableSparseMatrix))
def test_held_defined_coefficient_and_rhs_sources_survive_success(cls):
    x = Symbol("poly_rhs_held")
    coefficient = Pow(Add(1, I, evaluate=False), -1, evaluate=False)
    polynomial = Mul(Rational(1, 3), Pow(Add(x, 1, evaluate=False), 3, evaluate=False), evaluate=False)
    a, b = cls([[coefficient]]), cls([[polynomial]])
    original = a._surface, b._surface, coefficient.args, polynomial.args
    result = a.LUsolve(b)
    assert_exact(result, Matrix([[(1+I)*(x+1)**3/3]]))
    assert a._surface is original[0] and b._surface is original[1]
    assert coefficient.args is original[2] and polynomial.args is original[3]


@native_only
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("form", ("zero_product", "cancelled", "zero_power", "direct_zero_power"))
def test_original_nonpositive_symbolic_powers_are_refused_before_cancellation(method, form):
    x = Symbol("poly_rhs_denominator")
    reciprocal = Pow(x, -1, evaluate=False)
    value = {
        "zero_product": lambda: Mul(0, reciprocal, evaluate=False),
        "cancelled": lambda: Mul(x, reciprocal, evaluate=False),
        "zero_power": lambda: Pow(reciprocal, 0, evaluate=False),
        "direct_zero_power": lambda: Pow(x, 0, evaluate=False),
    }[form]()
    source = Add(x*x, value, evaluate=False)
    with pytest.raises((ValueError, NotImplementedError)):
        getattr(Matrix([[1]]), method)(Matrix([[source]]))


@native_only
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("form", ("zero_product", "zero_power"))
def test_original_undefined_rhs_children_are_terminal(method, form):
    x = Symbol("poly_rhs_undefined")
    pole = Pow(Add(I, -I, evaluate=False), -1, evaluate=False)
    invalid = Mul(0, pole, evaluate=False) if form == "zero_product" else Pow(pole, 0, evaluate=False)
    source = Add(x, invalid, evaluate=False)
    rhs = Matrix([[source]])
    retained = rhs._surface
    with pytest.raises(ValueError):
        getattr(Matrix([[1]]), method)(rhs)
    assert rhs._surface is retained and rhs[0, 0] is source


@native_only
@pytest.mark.parametrize("value", ("float", "function", "symbolic_coefficient"))
def test_nonpolynomial_data_do_not_enter_the_exact_polynomial_lane(value):
    x = Symbol("poly_rhs_unsupported")
    a = Matrix([[x if value == "symbolic_coefficient" else 1]])
    rhs = Matrix([[x+Float(0.5) if value == "float" else sin(x) if value == "function" else x*x]])
    with pytest.raises((ValueError, NotImplementedError)):
        a.LUsolve(rhs)


@native_only
@pytest.mark.parametrize("native_only_storage", (False, True))
@pytest.mark.parametrize("facts", ({"commutative": False}, {"finite": False}, {"infinite": True}))
def test_noncommuting_or_infinite_symbol_declarations_are_not_erased(facts, native_only_storage):
    x = Symbol("poly_rhs_bad_symbol_" + next(iter(facts)), **facts)
    rhs = Matrix([[x]])
    if native_only_storage:
        object.__setattr__(rhs, "_surface", None)
    with pytest.raises((ValueError, NotImplementedError)):
        Matrix([[1]]).LUsolve(rhs)


@native_only
def test_native_only_rhs_recovers_the_original_typed_symbol_owner():
    x = Symbol("poly_rhs_native_owner", positive=True)
    rhs = ImmutableMatrix([[x*x+1]])
    object.__setattr__(rhs, "_surface", None)
    result = Matrix([[2]]).LUsolve(rhs)
    assert type(result) is ImmutableMatrix
    assert_exact(result, Matrix([[(x*x+1)/2]]))
    assert next(iter(result[0, 0].free_symbols)) is x


@native_only
def test_an_unowned_native_typed_symbol_is_not_fabricated():
    from sympy.core import Expr, _native

    native = _native.py_symbol("poly_rhs_unowned_native", [("positive", "True")])
    rhs = Matrix([[Expr(native)]])
    object.__setattr__(rhs, "_surface", None)
    with pytest.raises((ValueError, NotImplementedError)):
        Matrix([[1]]).solve(rhs)


@native_only
@pytest.mark.parametrize("mutation", ("child", "arguments", "surface"))
def test_stale_retained_polynomial_graphs_cannot_be_rebased(mutation, monkeypatch):
    x = Symbol("poly_rhs_stale_" + mutation)
    coefficient = Rational(19, 13)
    source = Mul(coefficient, Add(x, 1, evaluate=False), evaluate=False)
    rhs = Matrix([[source]])
    if mutation == "child":
        monkeypatch.setattr(coefficient, "_value", Integer(7)._value)
    elif mutation == "arguments":
        monkeypatch.setattr(source, "_args", (Integer(7), x))
    else:
        monkeypatch.setattr(rhs, "_surface", (x,))
    with pytest.raises((RuntimeError, ValueError), match="retained|differ|mutated"):
        Matrix([[1]]).solve(rhs)


@native_only
def test_mutated_symbol_declarations_are_not_reinterpreted(monkeypatch):
    x = Symbol("poly_rhs_stale_declarations")
    rhs = Matrix([[x*x+1]])
    monkeypatch.setitem(x._assumptions, "positive", True)
    with pytest.raises((RuntimeError, ValueError)):
        Matrix([[1]]).solve(rhs)


@native_only
@pytest.mark.parametrize("slot", ("_dummy_name", "_dummy_number"))
def test_mutated_dummy_identity_slots_are_not_reinterpreted(slot, monkeypatch):
    x = Dummy("poly_rhs_stale_dummy")
    rhs = Matrix([[x+1]])
    replacement = "different_poly_rhs_dummy" if slot == "_dummy_name" else x.dummy_index+1
    monkeypatch.setattr(x, slot, replacement)
    with pytest.raises((RuntimeError, ValueError)):
        Matrix([[1]]).solve(rhs)


@native_only
def test_custom_function_sources_cannot_invoke_conversion_or_numeric_hooks():
    calls = []

    class CustomPolynomial(Function):
        def _eval_evalf(self, precision):
            calls.append("evalf")
            return Integer(1)

        def __float__(self):
            calls.append("float")
            return 1.0

        def __str__(self):
            calls.append("printer")
            return "1"

    x = Symbol("poly_rhs_custom")
    rhs = Matrix([[CustomPolynomial(x, evaluate=False)]])
    calls.clear()
    with pytest.raises(NotImplementedError):
        Matrix([[1]]).LUsolve(rhs)
    assert calls == []


@native_only
@pytest.mark.parametrize("side", ("coefficients", "rhs"))
def test_custom_matrix_owners_stay_outside_the_new_exact_lane(side):
    class CustomMatrix(Matrix):
        pass

    x = Symbol("poly_rhs_custom_matrix")
    a = (CustomMatrix if side == "coefficients" else Matrix)([[1]])
    b = (CustomMatrix if side == "rhs" else Matrix)([[x]])
    with pytest.raises(NotImplementedError):
        a.LUsolve(b)


@native_only
@pytest.mark.parametrize("cls", (Matrix, SparseMatrix))
def test_rhs_mutation_after_binding_prevents_publication(cls, monkeypatch):
    from sympy.matrices import _definiteness

    x = Symbol("poly_rhs_bind_mutation")
    a, rhs = cls([[1]]), cls([[x]])
    original = _definiteness._bind_gaussian

    def bind_then_mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        rhs[0, 0] = x+9
        return result

    monkeypatch.setattr(_definiteness, "_bind_gaussian", bind_then_mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        a.solve(rhs)


@native_only
@pytest.mark.parametrize("cls", (Matrix, SparseMatrix))
def test_selection_cannot_rebase_rhs_during_its_session_check(cls, monkeypatch):
    from sympy.matrices import _pseudoinverse

    x = Symbol("poly_rhs_selection_mutation")
    rhs = cls([[x]])
    original = _pseudoinverse._active_callback_session

    def session_then_mutate():
        session = original()
        rhs[0, 0] = x+1
        return session

    monkeypatch.setattr(_pseudoinverse, "_active_callback_session", session_then_mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        _pseudoinverse._polynomial_rhs(rhs, None)


@native_only
@pytest.mark.parametrize("cls", (Matrix, SparseMatrix))
@pytest.mark.parametrize("mutation", ("matrix", "assumptions", "dummy"))
def test_final_output_charge_cannot_publish_mutated_source_owners(cls, mutation, monkeypatch):
    from sympy.matrices import _pseudoinverse

    x = (Dummy("poly_rhs_final_charge") if mutation == "dummy"
         else Symbol("poly_rhs_final_charge_" + mutation))
    a, rhs = cls([[1]]), cls([[x]])
    original = _pseudoinverse._charge_nested_if_active
    calls = [0]

    def counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(_pseudoinverse, "_charge_nested_if_active", counted)
    assert_exact(a.solve(rhs), Matrix([[x]]))
    last_charge = calls[0]
    assert last_charge > 0
    calls[0] = 0

    def mutate_at_last_charge(*args, **kwargs):
        calls[0] += 1
        result = original(*args, **kwargs)
        if calls[0] == last_charge:
            if mutation == "matrix":
                rhs[0, 0] = x+1
            elif mutation == "assumptions":
                monkeypatch.setitem(x._assumptions, "positive", True)
            else:
                monkeypatch.setattr(x, "_dummy_number", x.dummy_index+1)
        return result

    monkeypatch.setattr(_pseudoinverse, "_charge_nested_if_active", mutate_at_last_charge)
    with pytest.raises(RuntimeError, match="mutated"):
        a.solve(rhs)
    assert calls[0] == last_charge


@native_only
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cancelled", (False, True))
def test_terminal_requests_stop_before_retained_polynomial_binding(method, cancelled, monkeypatch):
    from sympy.core import SupervisedCallbackSession, supervised_callback_session
    from sympy.matrices import _definiteness

    x = Symbol("poly_rhs_terminal")
    a, b = Matrix([[1]]), Matrix([[x*x+1]])
    session = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS if cancelled else 0)
    if cancelled:
        session.cancel()

    def forbidden(*args, **kwargs):
        raise AssertionError("terminal request copied or bound polynomial cells")

    monkeypatch.setattr(_definiteness, "_bind_gaussian", forbidden)
    with supervised_callback_session(session):
        with pytest.raises(RuntimeError if cancelled else ValueError, match="cancelled|budget exhausted"):
            getattr(a, method)(b)


@native_only
def test_repeated_public_solve_uses_one_cumulative_request_allowance():
    from sympy.core import SupervisedCallbackSession, supervised_callback_session

    x = Symbol("poly_rhs_cumulative")
    a, b = Matrix([[I]]), Matrix([[x*x+1]])
    measured = SupervisedCallbackSession(max_nested_steps=AMPLE_STEPS)
    with supervised_callback_session(measured):
        a.solve(b)
    cost = measured.consumed_nested_steps
    assert cost > 0
    limited = SupervisedCallbackSession(max_nested_steps=2*cost-1)
    with supervised_callback_session(limited):
        a.solve(b)
        with pytest.raises(ValueError, match="budget exhausted"):
            a.solve(b)
        spent = limited.consumed_nested_steps
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            a.solve(b)
        assert cost <= spent <= limited.max_nested_steps
        assert limited.consumed_nested_steps == spent



@native_only
@pytest.mark.parametrize("cls", (Matrix, SparseMatrix))
@pytest.mark.parametrize("side", ("coefficients", "rhs"))
@pytest.mark.parametrize("method", METHODS)
def test_first_request_charge_cannot_rebase_either_input(cls, side, method, monkeypatch):
    from sympy.matrices import _pseudoinverse

    x = Symbol("poly_rhs_ingress_" + side)
    a, rhs = cls([[1]]), cls([[x]])
    original = _pseudoinverse._charge_nested_if_active
    calls = [0]

    def mutate_on_first_charge(*args, **kwargs):
        calls[0] += 1
        result = original(*args, **kwargs)
        if calls[0] == 1:
            if side == "coefficients":
                a[0, 0] = 2
            else:
                rhs[0, 0] = x+1
        return result

    monkeypatch.setattr(_pseudoinverse, "_charge_nested_if_active", mutate_on_first_charge)
    with pytest.raises(RuntimeError, match="mutated"):
        getattr(a, method)(rhs)
    assert calls[0] == 1


@native_only
@pytest.mark.parametrize("cls", (Matrix, SparseMatrix))
def test_capture_keeps_its_source_before_the_first_request_charge(cls, monkeypatch):
    from sympy.matrices import _pseudoinverse

    x = Symbol("poly_rhs_capture_ingress")
    rhs = cls([[x]])
    original = _pseudoinverse._charge_nested_if_active
    calls = [0]

    def mutate_on_first_charge(*args, **kwargs):
        calls[0] += 1
        result = original(*args, **kwargs)
        if calls[0] == 1:
            rhs[0, 0] = x+1
        return result

    monkeypatch.setattr(_pseudoinverse, "_charge_nested_if_active", mutate_on_first_charge)
    with pytest.raises(RuntimeError, match="mutated"):
        _pseudoinverse._capture(
            rhs, None, 7, "exact linear solve", allow_sparse=True,
            force_gaussian=True, polynomial_rhs=True,
        )
    assert calls[0] == 1


@native_only
@pytest.mark.parametrize("cls", (Matrix, SparseMatrix))
def test_later_rhs_capture_uses_the_original_joint_ingress_snapshot(cls, monkeypatch):
    from sympy.matrices import _pseudoinverse

    x = Symbol("poly_rhs_joint_ingress")
    a, rhs = cls([[1]]), cls([[x]])
    original = _pseudoinverse._capture

    def capture_then_replace_rhs(matrix, *args, **kwargs):
        result = original(matrix, *args, **kwargs)
        if matrix is a:
            rhs[0, 0] = x+1
        return result

    monkeypatch.setattr(_pseudoinverse, "_capture", capture_then_replace_rhs)
    with pytest.raises(RuntimeError, match="mutated"):
        a.solve(rhs)


@native_only
def test_first_request_charge_cannot_rebase_an_inplace_sparse_surface_change(monkeypatch):
    from sympy.matrices import _pseudoinverse

    x = Symbol("poly_rhs_sparse_ingress")
    a, rhs = SparseMatrix([[1]]), SparseMatrix([[x]])
    original = _pseudoinverse._charge_nested_if_active
    calls = [0]

    def mutate_on_first_charge(*args, **kwargs):
        calls[0] += 1
        result = original(*args, **kwargs)
        if calls[0] == 1:
            rhs._surface[(0, 0)] = x+1
        return result

    monkeypatch.setattr(_pseudoinverse, "_charge_nested_if_active", mutate_on_first_charge)
    with pytest.raises(RuntimeError, match="mutated"):
        a.solve(rhs)
    assert calls[0] == 1


@native_only
def test_polynomial_selector_rejects_foreign_sparse_keys_before_hashing(monkeypatch):
    from sympy.matrices import _pseudoinverse

    calls = []

    class ForeignKey:
        def __hash__(self):
            calls.append("hash")
            return 0

    x = Symbol("poly_rhs_foreign_sparse_coordinate")
    rhs = SparseMatrix([[x]])
    monkeypatch.setattr(rhs, "_surface", {ForeignKey(): x})
    calls.clear()
    with pytest.raises(ValueError, match="coordinate"):
        _pseudoinverse._polynomial_rhs(rhs, None)
    assert calls == []
