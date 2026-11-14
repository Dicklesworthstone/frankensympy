"""Complete sparse least-squares families with pinned classes and retained owners."""

import inspect

import pytest
import sympy as s


_SPARSE = (s.SparseMatrix, s.ImmutableSparseMatrix)
_MATRICES = (s.Matrix, s.ImmutableMatrix, s.SparseMatrix, s.ImmutableSparseMatrix)


def _forbid_dense(*args, **kwargs):
    pytest.fail("sparse least-squares coefficients were converted to dense storage")


def _equal_cells(actual, expected):
    """Exact coefficient/atom comparison; no numerical or sampled equality."""
    assert actual.shape == expected.shape
    for row in range(actual.rows):
        for column in range(actual.cols):
            assert actual[row, column] == expected[row, column]


def _promoted(*classes):
    if s.ImmutableSparseMatrix in classes:
        return s.ImmutableSparseMatrix
    if s.ImmutableMatrix in classes:
        return s.ImmutableMatrix
    return s.Matrix


@pytest.mark.parametrize("a_class", _SPARSE)
@pytest.mark.parametrize("b_class", _MATRICES)
@pytest.mark.parametrize("w_class", _MATRICES)
def test_all_four_rhs_and_parameter_classes_keep_pinned_promotion(a_class, b_class, w_class, monkeypatch):
    # Executed SymPy 1.14.0 oracle, including immutable sparse promotion from W.
    a = a_class([[1, 1], [2, 2]])
    b = b_class([[1, 2], [0, 1]])
    w = w_class([[1, 2], [3, 4]])
    before = [tuple(matrix[index] for index in range(len(matrix))) for matrix in (a, b, w)]
    monkeypatch.setattr(s.SparseMatrix, "to_dense", _forbid_dense)
    result = a.pinv_solve(b, w)
    assert type(result) is _promoted(a_class, b_class, w_class)
    _equal_cells(result, s.Matrix([
        [s.Rational(-9, 10), s.Rational(-3, 5)],
        [s.Rational(11, 10), s.Rational(7, 5)],
    ]))
    assert [tuple(matrix[index] for index in range(len(matrix))) for matrix in (a, b, w)] == before
    assert all(matrix.mutation_generation == 0 for matrix in (a, b, w, result))


@pytest.mark.parametrize("a_class", _SPARSE)
@pytest.mark.parametrize("b_class", _MATRICES)
def test_default_family_class_and_exact_singular_inconsistent_values(a_class, b_class):
    a, b = a_class([[1, 1], [2, 2]]), b_class([1, 0])
    result = a.pinv_solve(b)
    assert type(result) is _promoted(a_class, b_class)
    parameters = {parameter.name: parameter for parameter in result.free_symbols}
    assert set(parameters) == {"w0_0", "w1_0"}
    assert all(type(parameter) is s.Dummy for parameter in parameters.values())
    u, v = parameters["w0_0"], parameters["w1_0"]
    _equal_cells(result, s.Matrix([
        u/2 - v/2 + s.Rational(1, 10),
        -u/2 + v/2 + s.Rational(1, 10),
    ]))
    assert result.free_symbols.isdisjoint(a.pinv_solve(b).free_symbols)


@pytest.mark.parametrize("matrix_type", _SPARSE)
def test_multi_rhs_dummy_names_and_freshness_follow_the_pinned_transposed_grid(matrix_type):
    a, b = matrix_type(0, 3, {}), s.Matrix(0, 2, [])
    result = a.pinv_solve(b)
    assert result.shape == (3, 2)
    assert [result[index].name for index in range(6)] == [
        "w0_0", "w1_1", "w0_1", "w2_0", "w1_0", "w2_1",
    ]
    assert all(type(result[index]) is s.Dummy for index in range(6))
    assert len(result.free_symbols) == 6
    assert result.free_symbols.isdisjoint(a.pinv_solve(b).free_symbols)


def test_wide_multi_rhs_family_preserves_every_nullspace_coefficient():
    a = s.SparseMatrix([[1, 0, 1], [0, 1, 1]])
    b = s.Matrix([[1, 3], [2, -1]])
    result = a.pinv_solve(b)
    parameters = {parameter.name: parameter for parameter in result.free_symbols}
    assert set(parameters) == {f"w{row}_{column}" for row in range(3) for column in range(2)}
    u, v, w = (parameters[name] for name in ("w0_0", "w0_1", "w1_0"))
    p, q, r = (parameters[name] for name in ("w1_1", "w2_0", "w2_1"))
    _equal_cells(result, s.Matrix([
        [u/3 + v/3 - w/3, p/3 + q/3 - r/3 + s.Rational(7, 3)],
        [u/3 + v/3 - w/3 + 1, p/3 + q/3 - r/3 - s.Rational(5, 3)],
        [-u/3 - v/3 + w/3 + 1, -p/3 - q/3 + r/3 + s.Rational(2, 3)],
    ]))


def test_supplied_zero_selects_minimum_norm_and_other_choices_satisfy_normal_equations():
    a = s.SparseMatrix([[2, -1], [4, -2], [6, -3]])
    b = s.Matrix([[3, 1], [0, 2], [1, 0]])
    minimum = a.pinv_solve(b, s.SparseMatrix(2, 2, {}))
    _equal_cells(minimum, s.Matrix([
        [s.Rational(6, 35), s.Rational(1, 7)],
        [s.Rational(-3, 35), s.Rational(-1, 14)],
    ]))
    assert not minimum.free_symbols
    assert s.Matrix([[1, 2]])*minimum == s.zeros(1, 2)
    for w in (s.Matrix([[1, 2], [3, 4]]), s.SparseMatrix([[7, -2], [1, 8]])):
        solution = a.pinv_solve(b, w)
        assert a.T*(a*solution-b) == s.SparseMatrix(2, 2, {})
    assert a.T*(a*minimum-b) == s.SparseMatrix(2, 2, {})


def test_full_column_rank_and_tiny_exact_pivots_have_no_free_component():
    a = s.SparseMatrix([[1, 0], [0, 1], [1, 1]])
    b = s.Matrix([[1, 2], [2, -1], [4, 0]])
    result = a.pinv_solve(b)
    assert not result.free_symbols
    _equal_cells(result, s.Matrix([
        [s.Rational(4, 3), s.Rational(5, 3)],
        [s.Rational(7, 3), s.Rational(-4, 3)],
    ]))
    epsilon = s.Rational(1, 10**80)
    assert s.SparseMatrix([[epsilon], [0]]).pinv_solve(s.Matrix([epsilon, 1])) == s.Matrix([1])


@pytest.mark.parametrize("parameter_type", _MATRICES)
def test_symbol_dummy_and_closed_function_parameters_keep_original_owners(parameter_type):
    plain = s.Symbol("sparse_family_same")
    real = s.Symbol("sparse_family_same", real=True)
    first, second = s.Dummy("w0_0"), s.Dummy("w0_0")
    sine, logarithm = s.sin(real, evaluate=False), s.log(plain, evaluate=False)
    w = parameter_type([[plain, real], [first, second], [sine, logarithm]])
    result = s.SparseMatrix(0, 3, {}).pinv_solve(s.Matrix(0, 2, []), w)
    assert type(result) is _promoted(s.SparseMatrix, s.Matrix, parameter_type)
    assert all(result[index] is w[index] for index in range(6))
    assert result.free_symbols == {plain, real, first, second}
    assert first != second


def test_coupled_projection_retains_distinct_same_name_symbol_and_dummy_owners():
    plain, real = s.Symbol("sparse_projection_same"), s.Symbol("sparse_projection_same", real=True)
    first, second = s.Dummy("sparse_projection_w"), s.Dummy("sparse_projection_w")
    w = s.ImmutableSparseMatrix([[plain, first], [real, second]])
    result = s.SparseMatrix([[1, 1]]).pinv_solve(s.Matrix([[1, 2]]), w)
    _equal_cells(result, s.Matrix([
        [plain/2-real/2+s.Rational(1, 2), first/2-second/2+1],
        [-plain/2+real/2+s.Rational(1, 2), -first/2+second/2+1],
    ]))
    assert result.free_symbols == {plain, real, first, second}
    for original in (plain, real, first, second):
        assert any(symbol is original for symbol in result.free_symbols)


@pytest.mark.parametrize("value", [s.pi, s.E, s.I, s.Float("1.25")])
def test_finite_builtin_constant_parameters_survive_or_are_safely_annihilated(value):
    w = s.SparseMatrix([value])
    family = s.SparseMatrix(1, 1, {}).pinv_solve(s.Matrix([0]), w)
    assert family[0] is value
    assert s.SparseMatrix([[1]]).pinv_solve(s.Matrix([2]), w) == s.Matrix([2])


@pytest.mark.parametrize("matrix_type", _SPARSE)
@pytest.mark.parametrize("rows,columns,rhs_columns", [
    (0, 0, 0), (0, 0, 2), (0, 3, 2), (4, 0, 3), (3, 2, 2), (2, 3, 0),
])
def test_zero_rank_and_zero_area_shapes_keep_the_complete_family(matrix_type, rows, columns, rhs_columns):
    a = matrix_type(rows, columns, {})
    b = s.Matrix(rows, rhs_columns, [1]*(rows*rhs_columns))
    result = a.pinv_solve(b)
    assert result.shape == (columns, rhs_columns)
    assert len(result.free_symbols) == columns*rhs_columns
    minimum = a.pinv_solve(b, s.SparseMatrix(columns, rhs_columns, {}))
    assert minimum.shape == (columns, rhs_columns)
    assert all(minimum[index] == 0 for index in range(len(minimum)))


def test_pinv_least_squares_method_uses_complete_family_and_keeps_default_behavior(monkeypatch):
    assert inspect.signature(s.SparseMatrix.solve_least_squares).parameters["method"].default == "LDL"
    a = s.SparseMatrix([[1, 1], [2, 2]])
    b = s.Matrix([1, 0])
    with pytest.raises(ValueError):
        a.solve_least_squares(b)

    def forbidden_old_inverse(*args, **kwargs):
        pytest.fail("PINV least squares bypassed its accepted native family")

    monkeypatch.setattr(s.SparseMatrix, "pinv", forbidden_old_inverse)
    result = a.solve_least_squares(b, method="PINV")
    parameters = {parameter.name: parameter for parameter in result.free_symbols}
    assert set(parameters) == {"w0_0", "w1_0"}
    u, v = parameters["w0_0"], parameters["w1_0"]
    _equal_cells(result, s.Matrix([u/2-v/2+s.Rational(1, 10), -u/2+v/2+s.Rational(1, 10)]))
    tall = s.SparseMatrix([[1, 0], [0, 1], [1, 1]])
    rhs = s.Matrix([1, 2, 4])
    expected = s.Matrix([s.Rational(4, 3), s.Rational(7, 3)])
    assert tall.solve_least_squares(rhs, method="PINV") == expected
    assert tall.solve_least_squares(rhs, method="LDL") == expected
    assert tall.solve_least_squares(rhs, method="CH") == expected


def test_original_shapes_and_builtin_matrix_ownership_are_checked_before_zero_projection():
    a, b = s.SparseMatrix([[1, 0], [0, 1]]), s.Matrix([1, 2])
    with pytest.raises(ValueError, match="number of rows"):
        a.pinv_solve(s.Matrix([1, 2, 3]))
    for wrong in (s.zeros(1, 1), s.zeros(3, 1), s.zeros(2, 2)):
        with pytest.raises(ValueError, match="arbitrary_matrix"):
            a.pinv_solve(b, wrong)
    with pytest.raises(TypeError):
        a.pinv_solve([1, 2])
    with pytest.raises(TypeError):
        a.pinv_solve(b, [[0], [0]])

    class CustomSparse(s.SparseMatrix):
        pass

    with pytest.raises(NotImplementedError, match="custom matrix subclasses"):
        CustomSparse([[1]]).pinv_solve(s.Matrix([1]))
    with pytest.raises(TypeError):
        s.SparseMatrix([[1]]).pinv_solve(CustomSparse([1]))
    with pytest.raises(TypeError):
        s.SparseMatrix([[1]]).pinv_solve(s.Matrix([1]), CustomSparse([0]))


@pytest.mark.parametrize("bad", [s.Symbol("sparse_family_source"), s.Float(1.0), s.I])
def test_every_original_a_and_b_cell_requires_exact_rational_admission(bad):
    with pytest.raises(ValueError, match="exact rational"):
        s.SparseMatrix([[bad]]).pinv_solve(s.Matrix(1, 0, []))
    with pytest.raises(ValueError, match="exact rational"):
        s.SparseMatrix(1, 1, {}).pinv_solve(s.SparseMatrix([[0, bad]]))


def test_original_held_rational_and_undefined_coefficients_are_not_simplified_into_admission():
    a = s.SparseMatrix([[2]])
    a._surface[(0, 0)] = s.Add(1, 1, evaluate=False)
    with pytest.raises(NotImplementedError, match="literal exact rational"):
        a.pinv_solve(s.Matrix([1]))
    held = s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False)
    with pytest.raises((ValueError, NotImplementedError)):
        s.SparseMatrix([[held]]).pinv_solve(s.Matrix(1, 0, []))


@pytest.mark.parametrize("zero_kind", ["literal", "cancellation", "elementary"])
@pytest.mark.parametrize("parent", ["direct", "zero_product", "zeroth_power"])
def test_held_log_poles_are_rejected_even_when_the_projector_is_zero(zero_kind, parent):
    x = s.Symbol("sparse_family_held_zero")
    zero = {
        "literal": s.Integer(0),
        "cancellation": s.Add(x, -x, evaluate=False),
        "elementary": s.sin(0, evaluate=False),
    }[zero_kind]
    pole = s.log(zero, evaluate=False)
    invalid = {
        "direct": pole,
        "zero_product": s.Mul(0, pole, evaluate=False),
        "zeroth_power": s.Pow(pole, 0, evaluate=False),
    }[parent]
    w = s.SparseMatrix([invalid])
    with pytest.raises(ValueError, match="undefined held function"):
        s.SparseMatrix([[1]]).pinv_solve(s.Matrix([1]), w)
    assert w[0] is invalid


@pytest.mark.parametrize("power", [-1, 0])
def test_held_undefined_power_cannot_be_erased_by_zero_parameters_or_projection(power):
    x = s.Symbol("sparse_family_power_zero")
    zero = s.Add(x, -x, evaluate=False)
    pole = s.Pow(s.Pow(zero, -1, evaluate=False), power, evaluate=False)
    invalid = s.Mul(0, pole, evaluate=False)
    w = s.Matrix([invalid])
    with pytest.raises(ValueError):
        s.SparseMatrix([[1]]).pinv_solve(s.Matrix([1]), w)
    assert w[0] is invalid


@pytest.mark.parametrize("role", ["a", "b", "w"])
def test_retained_original_matrix_cells_cannot_disagree_with_native_storage(role):
    matrices = {"a": s.SparseMatrix([[1]]), "b": s.SparseMatrix([2]), "w": s.SparseMatrix([3])}
    matrices[role]._surface[(0, 0)] = s.Integer(7)
    with pytest.raises(RuntimeError, match="native snapshot"):
        matrices["a"].pinv_solve(matrices["b"], matrices["w"])


@pytest.mark.parametrize("role", ["a", "b", "w"])
def test_source_mutation_during_publication_prevents_a_result(role, monkeypatch):
    from sympy.matrices import sparse

    matrices = {"a": s.SparseMatrix([[1]]), "b": s.SparseMatrix([2]), "w": s.SparseMatrix([3])}
    original = sparse._checked_sparse_lift

    def changed_lift(*args):
        result = original(*args)
        matrices[role][0, 0] = 7
        return result

    monkeypatch.setattr(sparse, "_checked_sparse_lift", changed_lift)
    with pytest.raises(RuntimeError, match="mutated during sparse linear algebra"):
        matrices["a"].pinv_solve(matrices["b"], matrices["w"])


def test_parameter_symbol_assumption_mutation_during_publication_prevents_return(monkeypatch):
    from sympy.matrices import sparse

    x = s.Symbol("sparse_family_mutating_parameter", real=True)
    a, b, w = s.SparseMatrix(1, 1, {}), s.Matrix([0]), s.Matrix([x])
    original = sparse._checked_sparse_lift
    facts = dict(x._assumptions)

    def changed_lift(*args):
        result = original(*args)
        x._assumptions["positive"] = True
        return result

    monkeypatch.setattr(sparse, "_checked_sparse_lift", changed_lift)
    try:
        with pytest.raises(RuntimeError, match="mutated during evaluation"):
            a.pinv_solve(b, w)
    finally:
        x._assumptions.clear()
        x._assumptions.update(facts)


def test_replaced_native_family_is_not_called(monkeypatch):
    from sympy.core import _native

    a, b = s.SparseMatrix([[1, 1]]), s.Matrix([1])
    calls = []

    def replaced(*args):
        calls.append(True)
        raise AssertionError("unadmitted native family")

    monkeypatch.setattr(_native.SparseMatrix, "least_squares_family", replaced)
    with pytest.raises(NotImplementedError, match="original native interfaces"):
        a.pinv_solve(b)
    assert calls == []


@pytest.mark.parametrize("owner,attribute", [(s.Symbol, "__getattribute__"), (s.Add, "_args")])
def test_parameter_accessor_overrides_are_rejected_without_callbacks(owner, attribute, monkeypatch):
    x = s.Symbol("sparse_family_accessor")
    a, b, w = s.SparseMatrix([[1]]), s.Matrix([1]), s.Matrix([s.Add(x, 1, evaluate=False)])
    calls = []

    def changed(*args):
        calls.append(True)
        raise AssertionError("unadmitted symbolic accessor")

    replacement = changed if attribute == "__getattribute__" else property(changed)
    with monkeypatch.context() as patch:
        patch.setattr(owner, attribute, replacement)
        with pytest.raises(NotImplementedError, match="original scalar interfaces"):
            a.pinv_solve(b, w)
    assert calls == []


def test_in_place_arithmetic_code_override_is_rejected_before_projection(monkeypatch):
    a, b = s.SparseMatrix([[1, 1]]), s.Matrix([1])
    w = s.Matrix([s.Symbol("sparse_family_code"), 0])

    def changed(self, other):
        raise AssertionError("unadmitted arithmetic implementation")

    with monkeypatch.context() as patch:
        patch.setattr(s.Expr.__mul__, "__code__", changed.__code__)
        with pytest.raises(NotImplementedError, match="original scalar"):
            a.pinv_solve(b, w)


def test_forged_parameter_storage_is_rejected_before_effectful_length():
    a, b, w = s.SparseMatrix([[1]]), s.Matrix([1]), s.SparseMatrix([0])
    calls = []

    class EffectfulStorage:
        def __len__(self):
            calls.append(True)
            raise AssertionError("unadmitted retained-storage callback")

    original = w._surface
    w._surface = EffectfulStorage()
    try:
        with pytest.raises(ValueError, match="invalid retained cells"):
            a.pinv_solve(b, w)
    finally:
        w._surface = original
    assert calls == []


def test_opaque_function_parameters_require_the_supervised_lane():
    x = s.Symbol("sparse_family_opaque")
    opaque = s.Function("sparse_family_unknown")(x)
    w = s.Matrix([opaque])
    with pytest.raises(NotImplementedError):
        s.SparseMatrix([[1]]).pinv_solve(s.Matrix([1]), w)
    assert w[0] is opaque


def test_large_sparse_family_and_sparse_parameter_grid_avoid_dense_materialization(monkeypatch):
    size = 600
    entries, rhs, parameters, expected = {}, {}, {}, {}
    for row in range(0, size, 2):
        entries.update({(row, row): 1, (row, row+1): 1, (row+1, row): 2, (row+1, row+1): 2})
        rhs.update({(row, 0): 3, (row+1, 0): 4})
        parameters[row, 0] = 2
        expected.update({(row, 0): s.Rational(21, 10), (row+1, 0): s.Rational(1, 10)})
    a, b = s.ImmutableSparseMatrix(size, size, entries), s.SparseMatrix(size, 1, rhs)
    w = s.SparseMatrix(size, 1, parameters)
    monkeypatch.setattr(s.SparseMatrix, "to_dense", _forbid_dense)
    result = a.pinv_solve(b, w)
    assert type(result) is s.ImmutableSparseMatrix
    assert result.shape == (size, 1)
    assert result.todok() == expected
    assert result.nnz == size

    # A sparse W can have a huge logical rectangle and one stored owner.
    x = s.Symbol("sparse_family_large_owner")
    w = s.ImmutableSparseMatrix(size, size, {(size-1, size-1): x})
    result = s.ImmutableSparseMatrix(0, size, {}).pinv_solve(s.SparseMatrix(0, size, {}), w)
    assert result.shape == (size, size)
    assert result.nnz == 1
    assert result[size-1, size-1] is x


def test_dense_parameters_beyond_the_dense_solver_dimension_cap_use_sparse_family_admission():
    # Dense least-squares solving has a separate 32-row envelope. A dense W
    # for an accepted sparse family is admitted by its own 8192-dimension lane.
    values = [s.Integer(index+1) for index in range(64)]
    w = s.Matrix(64, 1, values)
    result = s.SparseMatrix(0, 64, {}).pinv_solve(s.Matrix(0, 1, []), w)
    assert result.shape == (64, 1)
    _equal_cells(result, w)
    assert all(result[index] is values[index] for index in range(64))


@pytest.mark.parametrize("storage", ["dense", "sparse"])
@pytest.mark.parametrize("violation", ["depth", "height"])
def test_native_only_parameter_roots_refuse_at_borrowed_admission(storage, violation):
    from sympy.core import _native

    if violation == "depth":
        value = _native.Expr("sparse_family_native_depth")
        for _ in range(65):
            value = _native.py_function("sin", value)
        reason = "bounded expression envelope"
    else:
        value = _native.py_integer(1 << 4096)
        reason = "coefficient exceeds 4096 bits"
    if storage == "dense":
        native = _native.Matrix(1, 1, [value])
        validate = native.validate_sparse_least_squares_parameters
        w = s.Matrix(native)
    else:
        native = _native.SparseMatrix(1, 1, {(0, 0): value})
        validate = native.validate_least_squares_parameters
        w = s.SparseMatrix(native)
    assert w._surface is None
    # Exercise the borrowing bridge directly: no todok/flat/to_sparse export
    # or Python owner reconstruction is needed to inspect these native roots.
    with pytest.raises(ValueError, match=reason):
        validate()
    with pytest.raises(ValueError, match=reason):
        s.SparseMatrix([[1]]).pinv_solve(s.Matrix([1]), w)
    assert w._surface is None


def test_dimension_parameter_count_height_and_nonfinite_limits_are_terminal(monkeypatch):
    monkeypatch.setattr(s.SparseMatrix, "to_dense", _forbid_dense)
    with pytest.raises(ValueError, match="8192"):
        s.SparseMatrix(0, 8193, {}).pinv_solve(s.SparseMatrix(0, 0, {}))
    with pytest.raises(ValueError, match="fresh parameter count"):
        s.SparseMatrix(0, 4097, {}).pinv_solve(s.SparseMatrix(0, 1, {}))
    with pytest.raises(ValueError, match="height|coefficient|limit|bound"):
        s.SparseMatrix([[2**8192]]).pinv_solve(s.SparseMatrix(1, 0, {}))
    with pytest.raises(ValueError, match="parameter names"):
        s.SparseMatrix([[1]]).pinv_solve(s.Matrix([1]), s.Matrix([s.Symbol("p"*65_537)]))
    for value in (s.oo, -s.oo, s.zoo, s.nan, s.Float("inf"), s.Float("nan")):
        with pytest.raises(ValueError, match="finite"):
            s.SparseMatrix([[1]]).pinv_solve(s.Matrix([1]), s.Matrix([value]))
