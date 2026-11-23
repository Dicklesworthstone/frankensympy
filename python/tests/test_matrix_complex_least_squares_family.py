"""Complete Q(i) least-squares families with exact original-source checks.

The explicit matrices are independently derived Hermitian fixtures. The
positive public cases also run against isolated SymPy 1.14.0; native request
and source-admission cases are kept separate from that compatibility evidence.
"""

import pytest
import sympy as s


DENSE = (s.Matrix, s.ImmutableMatrix)
MATRICES = DENSE + (s.SparseMatrix, s.ImmutableSparseMatrix)
CANDIDATE = hasattr(s.Matrix([[1]]), "_native_storage")
native_only = pytest.mark.skipif(not CANDIDATE, reason="native source and request contract")


def equal_matrix(actual, expected):
    assert actual.shape == expected.shape
    assert all(s.expand(actual[index] - expected[index]) == 0 for index in range(len(actual)))


@pytest.mark.parametrize("a,b,expected", [
    ([[1, s.I], [2, 2*s.I]], [[1, 2], [0, s.I]],
     [[s.Rational(1, 10), s.Rational(1, 5) + s.I/5],
      [-s.I/10, s.Rational(1, 5) - s.I/5]]),
    ([[1, s.I, 0]], [[2 + s.I, 4]],
     [[1 + s.I/2, 2], [s.Rational(1, 2) - s.I, -2*s.I], [0, 0]]),
    ([[s.I, 1], [0, 0], [2*s.I, 2]], [[1], [s.I], [0]],
     [[-s.I/10], [s.Rational(1, 10)]]),
    ([[s.I], [1]], [[1], [0]], [[-s.I/2]]),
])
def test_complex_minimum_norm_fixtures(a, b, expected):
    a, b = s.Matrix(a), s.Matrix(b)
    actual = a.pinv_solve(b, s.zeros(a.cols, b.cols))
    equal_matrix(actual, s.Matrix(expected))
    equal_matrix(a.H*(a*actual - b), s.zeros(a.cols, b.cols))


def test_complete_complex_family_has_both_kernel_components_and_fresh_owners():
    a = s.Matrix([[1, s.I], [2, 2*s.I]])
    b = s.Matrix([1, 0])
    family = a.pinv_solve(b)
    names = {parameter.name: parameter for parameter in family.free_symbols}
    assert set(names) == {"w0_0", "w1_0"}
    assert all(type(parameter) is s.Dummy for parameter in names.values())
    u, v = names["w0_0"], names["w1_0"]
    equal_matrix(family, s.Matrix([s.Rational(1, 10) + u/2 - s.I*v/2,
                                  -s.I/10 + s.I*u/2 + v/2]))
    equal_matrix(a.H*(a*family - b), s.zeros(2, 1))
    assert family.free_symbols.isdisjoint(a.pinv_solve(b).free_symbols)
    # A nonreal null direction must survive; transpose-only projection loses it.
    equal_matrix(a.pinv_solve(b, s.Matrix([-s.I, 1])),
                 s.Matrix([s.Rational(1, 10) - s.I, 1 - s.I/10]))


@pytest.mark.parametrize("a_class", DENSE)
@pytest.mark.parametrize("b_class", MATRICES)
@pytest.mark.parametrize("w_class", MATRICES)
def test_all_builtin_rhs_parameter_classes_and_immutability(a_class, b_class, w_class):
    a = a_class([[1, s.I]])
    b = b_class([[1, 2]])
    w = w_class([[0, 0], [0, 0]])
    actual = a.pinv_solve(b, w)
    kinds = (a_class, b_class, w_class)
    expected_class = (s.ImmutableSparseMatrix if s.ImmutableSparseMatrix in kinds else
                      s.ImmutableMatrix if s.ImmutableMatrix in kinds else s.Matrix)
    assert type(actual) is expected_class
    equal_matrix(actual, s.Matrix([[s.Rational(1, 2), 1], [-s.I/2, -s.I]]))


def test_pinv_hint_handles_rank_deficient_complex_system_without_changing_unique_lane():
    a, b = s.Matrix([[1, s.I], [2, 2*s.I]]), s.Matrix([1, 0])
    family = a.solve_least_squares(b, method="PINV")
    assert len(family.free_symbols) == 2
    equal_matrix(a.H*(a*family - b), s.zeros(2, 1))
    with pytest.raises((ValueError, TypeError)):
        a.solve_least_squares(b)


def test_zero_rank_keeps_typed_parameters_and_nonreal_rhs():
    plain = s.Symbol("family_coordinate")
    real = s.Symbol("family_coordinate", real=True)
    first, second = s.Dummy("w"), s.Dummy("w")
    parameters = s.ImmutableSparseMatrix([[plain, real], [first, second]])
    actual = s.zeros(2, 2).pinv_solve(s.Matrix([[s.I, 2], [3, -s.I]]), parameters)
    assert type(actual) is s.ImmutableSparseMatrix
    assert actual.free_symbols == {plain, real, first, second}
    assert all(actual[index] is parameters[index] for index in range(4))


@pytest.mark.parametrize("rows,columns,rhs_columns", [(0, 0, 0), (0, 3, 2),
                                                       (2, 0, 3), (2, 3, 0)])
def test_empty_shapes_preserve_full_family_dimensions(rows, columns, rhs_columns):
    a = s.zeros(rows, columns)
    b = s.Matrix(rows, rhs_columns, [s.I] * (rows*rhs_columns))
    family = a.pinv_solve(b)
    assert family.shape == (columns, rhs_columns)
    assert len(family.free_symbols) == columns*rhs_columns
    assert a.pinv_solve(b, s.zeros(columns, rhs_columns)) == s.zeros(columns, rhs_columns)


@native_only
@pytest.mark.parametrize("side", ["matrix", "rhs"])
def test_original_complex_domain_holes_survive_zero_products(side):
    zero = s.Add(s.I, -s.I, evaluate=False)
    invalid = s.Mul(0, s.Pow(zero, -1, evaluate=False), evaluate=False)
    a, b = s.Matrix([[s.I]]), s.Matrix([[0]])
    if side == "matrix":
        a = s.Matrix([[invalid]])
    else:
        a, b = s.zeros(1, 1), s.Matrix([[invalid]])
    with pytest.raises(ValueError, match="division by zero"):
        a.pinv_solve(b)


@native_only
@pytest.mark.parametrize("side", ["matrix", "rhs", "parameters"])
def test_mutation_during_result_rendering_cannot_publish_a_family(side, monkeypatch):
    from sympy.matrices import _pseudoinverse, _sparse_family

    a, b, w = s.Matrix([[1, s.I]]), s.Matrix([[1]]), s.Matrix([[0], [1]])
    changed = {"matrix": a, "rhs": b, "parameters": w}[side]
    original = _sparse_family._render_family

    def mutate_then_render(*args):
        changed[0, 0] = 7
        return original(*args)

    # The public helper imports the shared renderer at invocation time.
    monkeypatch.setattr(_sparse_family, "_render_family", mutate_then_render)
    with pytest.raises(RuntimeError, match="mutated"):
        _pseudoinverse.pinv_solve(a, b, w)


@native_only
def test_native_family_bridge_keeps_the_active_session_budget():
    from sympy.core import SupervisedCallbackSession

    a, b = s.Matrix([[1, s.I]]), s.Matrix([[1]])
    ample = SupervisedCallbackSession(max_nested_steps=20_000_000, max_wall_time_ms=60_000)
    minimum, projector = a._native_storage.least_squares_family(b._native_storage, session=ample)
    assert minimum.shape == (2, 1) and projector.shape == (2, 2)
    cost = ample.consumed_nested_steps
    assert cost > 1
    repeated = SupervisedCallbackSession(max_nested_steps=2*cost - 1, max_wall_time_ms=60_000)
    a._native_storage.least_squares_family(b._native_storage, session=repeated)
    assert repeated.consumed_nested_steps == cost
    with pytest.raises(ValueError, match="budget"):
        a._native_storage.least_squares_family(b._native_storage, session=repeated)
    assert cost <= repeated.consumed_nested_steps <= repeated.max_nested_steps
    assert repeated.cancelled
    short = SupervisedCallbackSession(max_nested_steps=1, max_wall_time_ms=60_000)
    with pytest.raises((ValueError, RuntimeError), match="budget|step|cancel"):
        a._native_storage.least_squares_family(b._native_storage, session=short)


@native_only
def test_complete_public_operation_has_cumulative_not_reset_budgets():
    from sympy.core import supervised_callback_session

    a, b, w = s.Matrix([[1, s.I]]), s.Matrix([[1]]), s.zeros(2, 1)
    with supervised_callback_session(max_nested_steps=20_000_000, max_wall_time_ms=60_000) as measured:
        result = a.pinv_solve(b, w)
        used = measured.consumed_nested_steps
    equal_matrix(result, s.Matrix([[s.Rational(1, 2)], [-s.I/2]]))
    assert used > 1
    with supervised_callback_session(max_nested_steps=used, max_wall_time_ms=60_000):
        exact = a.pinv_solve(b, w)
    equal_matrix(exact, result)
    with supervised_callback_session(max_nested_steps=used - 1, max_wall_time_ms=60_000):
        with pytest.raises((ValueError, RuntimeError), match="budget|step|cancel"):
            a.pinv_solve(b, w)
    with supervised_callback_session(max_nested_steps=20_000_000, max_wall_time_ms=60_000) as cancelled:
        cancelled.cancel()
        with pytest.raises((ValueError, RuntimeError), match="cancel"):
            a.pinv_solve(b, w)


@native_only
def test_replaced_native_family_method_is_rejected_without_invocation(monkeypatch):
    from sympy.matrices import _pseudoinverse

    a, b = s.Matrix([[s.I]]), s.Matrix([[1]])
    called = []

    def substituted(*args, **kwargs):
        called.append(True)
        raise AssertionError("replaced native method was invoked")

    monkeypatch.setattr(type(a._native_storage), "least_squares_family", substituted)
    with pytest.raises(NotImplementedError, match="original native interfaces"):
        _pseudoinverse.pinv_solve(a, b)
    assert called == []
