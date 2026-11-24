"""Sparse Q(i) pseudoinverses and complete Hermitian least-squares families.

Positive PINV fixtures and result classes also run against isolated SymPy
1.14.0. The CH/LDL Hermitian correction, bounded admission and request checks
are native-only contracts: upstream sparse CH/LDL uses a transpose product.
"""

import pytest
import sympy as s


SPARSE = (s.SparseMatrix, s.ImmutableSparseMatrix)
MATRICES = (s.Matrix, s.ImmutableMatrix) + SPARSE
CANDIDATE = hasattr(s.Matrix([[1]]), "_native_storage")
native_only = pytest.mark.skipif(not CANDIDATE, reason="native source and request contract")
AMPLE_STEPS = 20_000_000


def equal_matrix(actual, expected):
    assert actual.shape == expected.shape
    assert all(s.expand(actual[index] - expected[index]) == 0
               for index in range(len(actual)))


def assert_penrose(matrix, inverse):
    equal_matrix(matrix*inverse*matrix, matrix)
    equal_matrix(inverse*matrix*inverse, inverse)
    equal_matrix((matrix*inverse).H, matrix*inverse)
    equal_matrix((inverse*matrix).H, inverse*matrix)


@pytest.mark.parametrize("cls", SPARSE)
@pytest.mark.parametrize("entries,expected", [
    ([[s.I]], [[-s.I]]),
    ([[1 + s.I]], [[s.Rational(1, 2) - s.I/2]]),
    ([[1, 2*s.I], [s.I, -2]],
     [[s.Rational(1, 10), -s.I/10], [-s.I/5, s.Rational(-1, 5)]]),
    ([[1, s.I]], [[s.Rational(1, 2)], [-s.I/2]]),
    ([[1], [s.I]], [[s.Rational(1, 2), -s.I/2]]),
    ([[1, s.I], [0, 0], [0, 1]], [[1, 0, -s.I], [0, 0, 1]]),
])
def test_sparse_complex_pseudoinverse_preserves_class_and_all_penrose_identities(cls, entries, expected):
    matrix = cls(entries)
    inverse = matrix.pinv()
    assert type(inverse) is cls
    equal_matrix(inverse, s.Matrix(expected))
    assert_penrose(matrix, inverse)
    equal_matrix(matrix, cls(entries))
    equal_matrix(matrix.H.pinv(), inverse.H)


@pytest.mark.parametrize("a_class", SPARSE)
@pytest.mark.parametrize("b_class", MATRICES)
@pytest.mark.parametrize("w_class", MATRICES)
def test_sparse_complex_complete_family_classes_and_minimum_norm(a_class, b_class, w_class):
    matrix = a_class([[1, s.I]])
    rhs = b_class([[1, 2]])
    parameters = w_class([[0, 0], [0, 0]])
    result = matrix.pinv_solve(rhs, parameters)
    kinds = (a_class, b_class, w_class)
    expected_class = (s.ImmutableSparseMatrix if s.ImmutableSparseMatrix in kinds else
                      s.ImmutableMatrix if s.ImmutableMatrix in kinds else s.Matrix)
    assert type(result) is expected_class
    equal_matrix(result, s.Matrix([[s.Rational(1, 2), 1], [-s.I/2, -s.I]]))
    equal_matrix(matrix.H*(matrix*result-rhs), s.zeros(2, 2))
    # Orthogonality to the complete nonreal null direction fixes minimum norm.
    equal_matrix(s.Matrix([[-s.I], [1]]).H*result, s.zeros(1, 2))


@pytest.mark.parametrize("cls", SPARSE)
def test_complete_family_retains_nonreal_null_direction_and_fresh_parameters(cls):
    matrix = cls([[1, s.I], [2, 2*s.I]])
    rhs = s.Matrix([1, 0])
    family = matrix.pinv_solve(rhs)
    names = {parameter.name: parameter for parameter in family.free_symbols}
    assert set(names) == {"w0_0", "w1_0"}
    assert all(type(parameter) is s.Dummy for parameter in names.values())
    u, v = names["w0_0"], names["w1_0"]
    equal_matrix(family, s.Matrix([s.Rational(1, 10) + u/2 - s.I*v/2,
                                   -s.I/10 + s.I*u/2 + v/2]))
    equal_matrix(matrix.H*(matrix*family-rhs), s.zeros(2, 1))
    assert family.free_symbols.isdisjoint(matrix.pinv_solve(rhs).free_symbols)
    equal_matrix(matrix.pinv_solve(rhs, s.SparseMatrix([[-s.I], [1]])),
                 s.Matrix([s.Rational(1, 10)-s.I, 1-s.I/10]))


@native_only
@pytest.mark.parametrize("cls", SPARSE)
def test_sparse_complex_pinv_method_exposes_the_complete_family(cls):
    # This explicit sparse method is a local extension; SymPy 1.14's sparse
    # implementation passes PINV to inv(), which rejects that method name.
    matrix, rhs = cls([[1, s.I], [2, 2*s.I]]), s.Matrix([1, 0])
    hinted = matrix.solve_least_squares(rhs, method="PINV")
    assert len(hinted.free_symbols) == 2
    equal_matrix(matrix.H*(matrix*hinted-rhs), s.zeros(2, 1))


@pytest.mark.parametrize("cls", SPARSE)
def test_rational_sparse_coefficients_with_complex_rhs_keep_typed_parameters(cls):
    plain = s.Symbol("sparse_family_parameter")
    real = s.Symbol("sparse_family_parameter", real=True)
    first, second = s.Dummy("w"), s.Dummy("w")
    parameters = s.ImmutableSparseMatrix([[plain, real], [first, second]])
    rhs = s.Matrix([[s.I, 2], [3, -s.I]])
    family = cls(2, 2, {}).pinv_solve(rhs, parameters)
    assert type(family) is s.ImmutableSparseMatrix
    assert family.free_symbols == {plain, real, first, second}
    assert all(family[index] is parameters[index] for index in range(4))


@pytest.mark.parametrize("cls", SPARSE)
@pytest.mark.parametrize("rows,columns,rhs_columns", [(0, 0, 0), (0, 3, 2),
                                                       (2, 0, 3), (2, 3, 0)])
def test_sparse_empty_families_preserve_every_output_axis(cls, rows, columns, rhs_columns):
    matrix = cls(rows, columns, {})
    rhs = s.Matrix(rows, rhs_columns, [s.I] * (rows*rhs_columns))
    family = matrix.pinv_solve(rhs)
    assert family.shape == (columns, rhs_columns)
    assert len(family.free_symbols) == columns*rhs_columns
    equal_matrix(matrix.pinv_solve(rhs, s.zeros(columns, rhs_columns)),
                 s.zeros(columns, rhs_columns))


@native_only
@pytest.mark.parametrize("a_class", SPARSE)
@pytest.mark.parametrize("b_class", MATRICES)
@pytest.mark.parametrize("method", ["LDL", "CH"])
def test_sparse_unique_least_squares_corrects_transpose_normal_equations(a_class, b_class, method):
    matrix = a_class([[1], [2*s.I]])
    rhs = b_class([[1, 0], [0, 1]])
    result = matrix.solve_least_squares(rhs, method=method)
    expected_class = s.ImmutableSparseMatrix if a_class is s.ImmutableSparseMatrix else b_class
    assert type(result) is expected_class
    # Transpose-only normal equations give -1/3 and 2i/3 here.
    equal_matrix(result, s.Matrix([[s.Rational(1, 5), -2*s.I/5]]))
    equal_matrix(matrix.H*(matrix*result-rhs), s.zeros(1, 2))


@native_only
@pytest.mark.parametrize("cls", SPARSE)
def test_isotropic_column_and_complex_rhs_use_hermitian_norms(cls):
    matrix, rhs = cls([[1], [s.I]]), s.SparseMatrix([[1], [0]])
    equal_matrix(matrix.solve_least_squares(rhs), s.Matrix([[s.Rational(1, 2)]]))
    matrix, rhs = cls([[1], [2]]), s.SparseMatrix([[s.I, 1], [0, s.I]])
    equal_matrix(matrix.solve_least_squares(rhs),
                 s.Matrix([[s.I/5, s.Rational(1, 5)+2*s.I/5]]))


@native_only
@pytest.mark.parametrize("method", ["LDL", "CH"])
def test_unique_sparse_least_squares_refuses_rank_deficiency(method):
    matrix, rhs = s.SparseMatrix([[1, s.I], [2, 2*s.I]]), s.Matrix([1, 0])
    with pytest.raises(ValueError, match="singular|rank|unique"):
        matrix.solve_least_squares(rhs, method=method)
    minimum = matrix.pinv_solve(rhs, s.zeros(2, 1))
    equal_matrix(minimum, s.Matrix([s.Rational(1, 10), -s.I/10]))


@native_only
@pytest.mark.parametrize("operation", ["pinv", "least_squares", "family"])
def test_native_only_sparse_complex_storage_remains_composable(operation):
    matrix, rhs = s.SparseMatrix([[1], [s.I]]), s.SparseMatrix([[1], [0]])
    object.__setattr__(matrix, "_surface", None)
    object.__setattr__(rhs, "_surface", None)
    if operation == "pinv":
        result = matrix.pinv()
        equal_matrix(result, s.Matrix([[s.Rational(1, 2), -s.I/2]]))
        equal_matrix(result.pinv(), matrix)
    elif operation == "least_squares":
        equal_matrix(matrix.solve_least_squares(rhs), s.Matrix([[s.Rational(1, 2)]]))
    else:
        equal_matrix(matrix.pinv_solve(rhs), s.Matrix([[s.Rational(1, 2)]]))


@native_only
@pytest.mark.parametrize("operation", ["pinv", "least_squares", "family"])
def test_defined_held_sparse_complex_sources_retain_original_owners(operation):
    source = s.Pow(s.Add(1, s.I, evaluate=False), -1, evaluate=False)
    matrix = s.SparseMatrix([[source]])
    rhs = s.SparseMatrix([[1]])
    original = matrix._surface
    result = (matrix.pinv() if operation == "pinv" else
              matrix.solve_least_squares(rhs) if operation == "least_squares" else
              matrix.pinv_solve(rhs))
    equal_matrix(result, s.Matrix([[1+s.I]]))
    assert matrix._surface is original and matrix._surface[0, 0] is source


@native_only
@pytest.mark.parametrize("operation", ["pinv", "least_squares", "family"])
@pytest.mark.parametrize("side", ["matrix", "rhs"])
def test_original_hidden_complex_poles_are_not_erased_by_zero_products(operation, side):
    if operation == "pinv" and side == "rhs":
        pytest.skip("pseudoinverse has no right-hand side")
    zero = s.Add(s.I, -s.I, evaluate=False)
    hidden = s.Mul(0, s.Pow(zero, -1, evaluate=False), evaluate=False)
    matrix, rhs = s.SparseMatrix([[s.I]]), s.SparseMatrix([[0]])
    if side == "matrix":
        matrix = s.SparseMatrix([[hidden]])
    else:
        matrix, rhs = s.SparseMatrix([[0]]), s.SparseMatrix([[hidden]])
    with pytest.raises(ValueError, match="division by zero"):
        if operation == "pinv":
            matrix.pinv()
        elif operation == "least_squares":
            matrix.solve_least_squares(rhs)
        else:
            matrix.pinv_solve(rhs)


@native_only
@pytest.mark.parametrize("operation", ["pinv", "least_squares", "family"])
def test_sparse_retained_scalar_mutation_is_terminal(operation, monkeypatch):
    coefficient = s.Rational(19, 13)
    source = s.Mul(coefficient, s.I, evaluate=False)
    matrix, rhs = s.SparseMatrix([[source]]), s.Matrix([[1]])
    monkeypatch.setattr(coefficient, "_value", s.Integer(3)._value)
    with pytest.raises(RuntimeError, match="retained.*differ"):
        if operation == "pinv":
            matrix.pinv()
        elif operation == "least_squares":
            matrix.solve_least_squares(rhs)
        else:
            matrix.pinv_solve(rhs)


@native_only
@pytest.mark.parametrize("side", ["matrix", "rhs", "parameters"])
def test_sparse_complex_family_mutation_during_rendering_blocks_publication(side, monkeypatch):
    from sympy.matrices import _sparse_family

    matrix, rhs, parameters = (s.SparseMatrix([[1, s.I]]),
                               s.SparseMatrix([[1]]), s.SparseMatrix([[0], [1]]))
    changed = {"matrix": matrix, "rhs": rhs, "parameters": parameters}[side]
    original = _sparse_family._render_family

    def mutate_then_render(*args):
        changed[0, 0] = 7
        return original(*args)

    monkeypatch.setattr(_sparse_family, "_render_family", mutate_then_render)
    with pytest.raises(RuntimeError, match="mutated"):
        matrix.pinv_solve(rhs, parameters)


@native_only
@pytest.mark.parametrize("operation", ["pinv", "least_squares", "family"])
def test_changed_sparse_native_conversion_is_never_called(operation, monkeypatch):
    matrix, rhs = s.SparseMatrix([[s.I]]), s.Matrix([[1]])
    called = []

    def changed(*args):
        called.append(True)
        raise AssertionError("replaced sparse conversion ran")

    monkeypatch.setattr(type(matrix._native_storage), "to_dense", changed)
    with pytest.raises(NotImplementedError, match="original.*native interfaces"):
        if operation == "pinv":
            matrix.pinv()
        elif operation == "least_squares":
            matrix.solve_least_squares(rhs)
        else:
            matrix.pinv_solve(rhs)
    assert called == []


@native_only
@pytest.mark.parametrize("operation", ["least_squares", "family"])
def test_mixed_complex_request_bounds_rational_sparse_source_before_copy(operation, monkeypatch):
    from sympy.matrices import _definiteness

    matrix = s.SparseMatrix(33, 1, {(0, 0): 1})
    rhs = s.SparseMatrix(33, 1, {(0, 0): s.I})
    copied = []

    def forbidden(*args, **kwargs):
        copied.append(True)
        raise AssertionError("oversized mixed source was copied")

    monkeypatch.setattr(_definiteness, "_rational_source_snapshot", forbidden)
    with pytest.raises(ValueError, match="dimension|32"):
        if operation == "least_squares":
            matrix.solve_least_squares(rhs)
        else:
            matrix.pinv_solve(rhs)
    assert copied == []


@native_only
def test_large_literal_rational_sparse_operations_keep_existing_lane(monkeypatch):
    from sympy.matrices import _pseudoinverse

    matrix = s.SparseMatrix(33, 1, {(0, 0): 1})
    rhs = s.SparseMatrix(33, 1, {(0, 0): 2})

    def forbidden(*args, **kwargs):
        raise AssertionError("literal rational input entered the bounded dense lane")

    monkeypatch.setattr(_pseudoinverse, "_capture", forbidden)
    inverse = matrix.pinv()
    assert inverse.shape == (1, 33) and inverse[0, 0] == 1
    equal_matrix(matrix.solve_least_squares(rhs), s.Matrix([[2]]))
    equal_matrix(matrix.pinv_solve(rhs), s.Matrix([[2]]))


@native_only
@pytest.mark.parametrize("operation", ["pinv", "least_squares", "family"])
def test_sparse_complex_public_request_is_cumulative_and_checks_final_cancellation(operation):
    from sympy.core import supervised_callback_session

    matrix, rhs = s.SparseMatrix([[1], [s.I]]), s.SparseMatrix([[1], [0]])

    def evaluate():
        if operation == "pinv":
            return matrix.pinv()
        if operation == "least_squares":
            return matrix.solve_least_squares(rhs)
        return matrix.pinv_solve(rhs, s.SparseMatrix([[0]]))

    with supervised_callback_session(max_nested_steps=AMPLE_STEPS, max_wall_time_ms=60_000) as measured:
        result = evaluate()
        cost = measured.consumed_nested_steps
    assert cost > 1
    with supervised_callback_session(max_nested_steps=cost, max_wall_time_ms=60_000):
        exact = evaluate()
    equal_matrix(exact, result)
    with supervised_callback_session(max_nested_steps=cost-1, max_wall_time_ms=60_000):
        with pytest.raises((ValueError, RuntimeError), match="budget|step|cancel"):
            evaluate()
    with supervised_callback_session(max_nested_steps=AMPLE_STEPS, max_wall_time_ms=60_000) as cancelled:
        cancelled.cancel()
        with pytest.raises((ValueError, RuntimeError), match="cancel"):
            evaluate()
