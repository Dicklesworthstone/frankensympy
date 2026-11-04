"""Complete exact-rational least-squares families and pinned public ownership."""

import pytest
import sympy as s


def equal_matrix(actual, expected):
    assert actual.shape == expected.shape
    assert all(s.expand(actual[index]-expected[index]) == 0 for index in range(len(actual)))


def test_rank_deficient_inconsistent_system_has_the_complete_dummy_family():
    a = s.Matrix([[1, 1], [2, 2]])
    b = s.Matrix([1, 0])
    result = a.pinv_solve(b)
    parameters = {parameter.name: parameter for parameter in result.free_symbols}
    assert set(parameters) == {"w0_0", "w1_0"}
    assert all(type(parameter) is s.Dummy for parameter in parameters.values())
    u, v = parameters["w0_0"], parameters["w1_0"]
    equal_matrix(result, s.Matrix([u/2-v/2+s.Rational(1, 10), -u/2+v/2+s.Rational(1, 10)]))
    equal_matrix(a.T*(a*result-b), s.zeros(2, 1))
    assert result.free_symbols.isdisjoint(a.pinv_solve(b).free_symbols)


def test_zero_arbitrary_matrix_selects_the_unique_minimum_norm_solution():
    a = s.Matrix([[2, -1], [4, -2], [6, -3]])
    b = s.Matrix([[3, 1], [0, 2], [1, 0]])
    result = a.pinv_solve(b, arbitrary_matrix=s.zeros(2, 2))
    expected = s.Matrix([[s.Rational(6, 35), s.Rational(1, 7)],
                         [-s.Rational(3, 35), -s.Rational(1, 14)]])
    assert result == expected
    assert not result.free_symbols
    assert a.T*(a*result-b) == s.zeros(2, 2)
    assert s.Matrix([[1, 2]])*result == s.zeros(1, 2)


def test_wide_exact_system_keeps_all_free_directions():
    a = s.Matrix([[1, 2, 3]])
    b = s.Matrix([[2, 4]])
    family = a.pinv_solve(b)
    assert family.shape == (3, 2)
    assert len(family.free_symbols) == 6
    equal_matrix(a*family, b)
    minimum = a.pinv_solve(b, s.zeros(3, 2))
    assert minimum == s.Matrix([[s.Rational(1, 7), s.Rational(2, 7)],
                                [s.Rational(2, 7), s.Rational(4, 7)],
                                [s.Rational(3, 7), s.Rational(6, 7)]])


def test_multi_rhs_dummy_names_follow_the_pinned_transposed_grid():
    result = s.zeros(0, 3).pinv_solve(s.zeros(0, 2))
    assert result.shape == (3, 2)
    assert [result[index].name for index in range(len(result))] == [
        "w0_0", "w1_1", "w0_1", "w2_0", "w1_0", "w2_1",
    ]
    assert all(type(result[index]) is s.Dummy for index in range(len(result)))
    assert len(result.free_symbols) == 6


@pytest.mark.parametrize("a_class", [s.Matrix, s.ImmutableMatrix])
@pytest.mark.parametrize("b_class", [s.Matrix, s.ImmutableMatrix, s.SparseMatrix])
@pytest.mark.parametrize("w_class", [s.Matrix, s.ImmutableMatrix, s.SparseMatrix])
def test_result_class_promotes_immutability_from_every_matrix(a_class, b_class, w_class):
    a = a_class([[1, 1], [2, 2]])
    b = b_class([[1, 2], [0, 1]])
    w = w_class([[1, 2], [3, 4]])
    before = tuple(tuple(value[index] for index in range(len(value))) for value in (a, b, w))
    result = a.pinv_solve(b, w)
    expected_class = (s.ImmutableMatrix if s.ImmutableMatrix in (a_class, b_class, w_class)
                      else s.Matrix)
    assert type(result) is expected_class
    assert result == s.Matrix([[-s.Rational(9, 10), -s.Rational(3, 5)],
                               [s.Rational(11, 10), s.Rational(7, 5)]])
    assert tuple(tuple(value[index] for index in range(len(value))) for value in (a, b, w)) == before


def test_default_family_class_tracks_coefficient_and_rhs_immutability():
    for a_class in (s.Matrix, s.ImmutableMatrix):
        for b_class in (s.Matrix, s.ImmutableMatrix, s.SparseMatrix):
            result = a_class([[1, 1]]).pinv_solve(b_class([1]))
            expected = s.ImmutableMatrix if s.ImmutableMatrix in (a_class, b_class) else s.Matrix
            assert type(result) is expected


def test_supplied_symbol_and_dummy_parameters_keep_exact_public_owners():
    plain = s.Symbol("least_squares_same")
    real = s.Symbol("least_squares_same", real=True)
    first = s.Dummy("w0_0")
    second = s.Dummy("w0_0")
    supplied = s.ImmutableMatrix([[plain, real], [first, second]])
    result = s.zeros(2, 2).pinv_solve(s.zeros(2, 2), supplied)
    assert type(result) is s.ImmutableMatrix
    assert result.free_symbols == {plain, real, first, second}
    assert all(result[index] is supplied[index] for index in range(4))
    assert first != second


def test_least_squares_pinv_hint_uses_the_checked_family_and_preserves_default(monkeypatch):
    a = s.Matrix([[1, 2], [2, 3], [3, 4]])
    b = s.Matrix([8, 14, 18])
    expected = s.Matrix([s.Rational(5, 3), s.Rational(10, 3)])
    assert a.solve_least_squares(b) == expected
    assert a.solve_least_squares(b, method="CH") == expected

    def forbidden_old_pseudoinverse(*args, **kwargs):
        raise AssertionError("the complete checked family must use its metered native lane")

    monkeypatch.setattr(s.Matrix, "pinv", forbidden_old_pseudoinverse)
    assert a.solve_least_squares(b, method="PINV") == expected
    singular = s.Matrix([[1, 1], [2, 2]])
    with pytest.raises(ValueError):
        singular.solve_least_squares(s.Matrix([1, 0]))
    result = singular.solve_least_squares(s.Matrix([1, 0]), method="PINV")
    equal_matrix(singular.T*(singular*result-s.Matrix([1, 0])), s.zeros(2, 1))
    assert len(result.free_symbols) == 2
    with pytest.raises(NotImplementedError):
        a.solve_least_squares(b, method="unsupported")


@pytest.mark.parametrize("rows,cols,rhs_cols", [(0, 0, 0), (0, 0, 2), (0, 3, 2),
                                               (4, 0, 3), (3, 2, 2), (2, 3, 0)])
def test_zero_rank_and_zero_area_shapes_have_complete_results(rows, cols, rhs_cols):
    a = s.zeros(rows, cols)
    b = s.ones(rows, rhs_cols)
    result = a.pinv_solve(b)
    assert result.shape == (cols, rhs_cols)
    assert len(result.free_symbols) == cols*rhs_cols
    assert a.pinv_solve(b, s.zeros(cols, rhs_cols)) == s.zeros(cols, rhs_cols)


def test_tiny_exact_coefficients_keep_rank_and_minimum_norm():
    epsilon = s.Rational(1, 10**80)
    a = s.Matrix([[epsilon], [0]])
    assert a.pinv_solve(s.Matrix([epsilon, 1])) == s.Matrix([1])


def test_invalid_shapes_are_rejected_even_when_the_kernel_projector_is_zero():
    a = s.eye(2)
    with pytest.raises(ValueError, match="number of rows"):
        a.pinv_solve(s.zeros(3, 1))
    for wrong in (s.zeros(1, 1), s.zeros(3, 1), s.zeros(2, 2)):
        with pytest.raises(ValueError, match="arbitrary_matrix"):
            a.pinv_solve(s.ones(2, 1), wrong)
    with pytest.raises(TypeError):
        a.pinv_solve([1, 2])
    with pytest.raises(TypeError):
        a.pinv_solve(s.ones(2, 1), [[0], [0]])


def test_all_original_coefficients_and_later_rhs_entries_are_admitted_before_zero():
    x = s.Symbol("least_squares_coefficient")
    hidden = s.Mul(0, x, evaluate=False)
    for bad in (x, hidden, s.I, s.Float(1.0)):
        with pytest.raises(ValueError):
            s.Matrix([[bad]]).pinv_solve(s.zeros(1, 1))
        with pytest.raises(ValueError):
            s.zeros(1, 1).pinv_solve(s.Matrix([[0, bad]]))


def test_opaque_parameters_and_custom_matrix_overrides_stay_outside_the_native_lane():
    x = s.Symbol("least_squares_opaque")
    opaque = s.Function("least_squares_custom")(x)
    parameters = s.Matrix([opaque])
    with pytest.raises(NotImplementedError):
        s.eye(1).pinv_solve(s.ones(1, 1), parameters)
    assert parameters[0] is opaque

    class CustomMatrix(s.Matrix):
        pass

    with pytest.raises(NotImplementedError):
        CustomMatrix([[1]]).pinv_solve(s.ones(1, 1))


@pytest.mark.parametrize("owner,attribute", [
    (s.Basic, "_struct_args"), (s.Add, "_args"), (s.Symbol, "__getattribute__"),
])
def test_parameter_preflight_rejects_modified_accessors_without_calling_them(
        monkeypatch, owner, attribute):
    x = s.Symbol("least_squares_changed_accessor")
    held = s.Add(x, 1, evaluate=False)
    parameters = s.Matrix([held])
    a, b = s.eye(1), s.ones(1, 1)
    calls = []

    def forbidden_accessor(*args):
        calls.append(True)
        raise AssertionError("parameter preflight invoked an untrusted accessor")

    replacement = (forbidden_accessor if attribute == "__getattribute__"
                   else property(forbidden_accessor))
    with monkeypatch.context() as patch:
        patch.setattr(owner, attribute, replacement)
        with pytest.raises(NotImplementedError, match="modified symbolic accessors"):
            a.pinv_solve(b, parameters)
    assert calls == []
    assert parameters[0] is held


@pytest.mark.parametrize("power", [-1, 0])
@pytest.mark.parametrize("literal", [True, False])
def test_held_undefined_parameters_are_checked_even_when_the_projector_is_zero(power, literal):
    x = s.Symbol("least_squares_held_zero")
    zero = s.Integer(0) if literal else s.Add(x, -x, evaluate=False)
    hidden = s.Pow(s.Pow(zero, -1, evaluate=False), power, evaluate=False)
    invalid = s.Mul(0, hidden, evaluate=False)
    parameters = s.Matrix([invalid])
    with pytest.raises(ValueError):
        s.eye(1).pinv_solve(s.ones(1, 1), parameters)
    assert parameters[0] is invalid


@pytest.mark.parametrize("zero_kind", ["literal", "cancellation", "elementary"])
@pytest.mark.parametrize("parent", ["direct", "zero_product", "zeroth_power"])
def test_held_function_poles_cannot_be_erased_by_zero_projection(zero_kind, parent):
    x = s.Symbol("least_squares_held_log_zero")
    zero = {
        "literal": s.Integer(0),
        "cancellation": s.Add(x, -x, evaluate=False),
        "elementary": s.sin(0, evaluate=False),
    }[zero_kind]
    held_log = s.log(zero, evaluate=False)
    invalid = {
        "direct": held_log,
        "zero_product": s.Mul(0, held_log, evaluate=False),
        "zeroth_power": s.Pow(held_log, 0, evaluate=False),
    }[parent]
    parameters = s.Matrix([invalid])
    with pytest.raises(ValueError, match="undefined held function"):
        s.eye(1).pinv_solve(s.ones(1, 1), parameters)
    assert parameters[0] is invalid


def test_formal_logarithm_parameters_keep_their_original_domain():
    x = s.Symbol("least_squares_formal_log")
    held_log = s.log(x, evaluate=False)
    # An unresolved symbolic argument is a formal parameter, not a known pole.
    family = s.zeros(1, 1).pinv_solve(s.zeros(1, 1), s.Matrix([held_log]))
    assert family[0] == held_log
    assert s.eye(1).pinv_solve(s.ones(1, 1), s.Matrix([held_log])) == s.ones(1, 1)


def test_dimension_height_and_nonfinite_parameter_limits_are_explicit():
    for a, b in [(s.zeros(33, 1), s.zeros(33, 1)),
                 (s.zeros(0, 33), s.zeros(0, 1)),
                 (s.zeros(1, 1), s.zeros(1, 33))]:
        with pytest.raises(ValueError, match="32"):
            a.pinv_solve(b)
    with pytest.raises(ValueError):
        s.Matrix([[2**4096]]).pinv_solve(s.zeros(1, 1))
    large_name = s.Symbol("p" * 65_537)
    with pytest.raises(ValueError, match="parameter names"):
        s.eye(1).pinv_solve(s.ones(1, 1), s.Matrix([large_name]))
    for invalid in (s.oo, -s.oo, s.zoo, s.nan,
                    s.Float("inf"), s.Float("-inf"), s.Float("nan")):
        with pytest.raises(ValueError, match="finite"):
            s.eye(1).pinv_solve(s.ones(1, 1), s.Matrix([invalid]))
