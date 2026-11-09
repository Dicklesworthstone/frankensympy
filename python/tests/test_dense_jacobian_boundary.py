"""Vector shape, coordinate identity and two-input Jacobian snapshot regressions."""

import importlib

import pytest
import sympy as s


@pytest.mark.parametrize("matrix_type", [s.Matrix, s.ImmutableMatrix])
@pytest.mark.parametrize("row_source", [False, True])
@pytest.mark.parametrize("form", ["flat", "row", "column", "dense", "sparse"])
def test_dense_jacobian_vector_spellings_preserve_type_and_values(matrix_type, row_source, form):
    x, y = s.symbols("dense_jacobian_x dense_jacobian_y")
    coordinates = {
        "flat": [x, y], "row": [[x, y]], "column": [[x], [y]],
        "dense": s.Matrix([x, y]), "sparse": s.SparseMatrix([[x, y]]),
    }[form]
    source = matrix_type([[x*x, x*y, 7]] if row_source else [x*x, x*y, 7])
    result = source.jacobian(coordinates)
    assert type(result) is matrix_type
    assert result == matrix_type([[2*x, 0], [y, x], [0, 0]])
    assert source.mutation_generation == 0
    if matrix_type is s.ImmutableMatrix:
        with pytest.raises(TypeError):
            result[0, 0] = 42


def test_jacobian_rejects_nonvectors_instead_of_flattening_them():
    x = s.Symbol("dense_jacobian_nonvector")
    source = s.Matrix([x*x])
    for coordinates in (x, [], s.Matrix(0, 0), [[x, x], [x, x]],
                        s.Matrix([[x, x], [x, x]]), s.SparseMatrix.zeros(2)):
        with pytest.raises(TypeError, match="row or a column"):
            source.jacobian(coordinates)
    for nonvector in (s.Matrix(0, 0), s.Matrix([[x, x], [x, x]])):
        with pytest.raises(TypeError, match="self must"):
            nonvector.jacobian([x])
    with pytest.raises(ValueError, match="Row length mismatch"):
        source.jacobian([[x], []])


def test_empty_vectors_keep_their_exact_jacobian_shape():
    x = s.Symbol("dense_jacobian_empty")
    for matrix_type in (s.Matrix, s.ImmutableMatrix):
        for source in (matrix_type(0, 1), matrix_type(1, 0)):
            assert source.jacobian([x]).shape == (0, 1)
            for coordinates in (s.Matrix(0, 1), s.Matrix(1, 0), [[]]):
                result = source.jacobian(coordinates)
                assert type(result) is matrix_type and result.shape == (0, 0)
        for coordinates in (s.Matrix(0, 1), s.Matrix(1, 0), [[]]):
            assert matrix_type([x, x*x]).jacobian(coordinates).shape == (2, 0)


def test_same_named_typed_coordinates_remain_distinct_and_ordered():
    ordinary = s.Symbol("dense_jacobian_same")
    positive = s.Symbol("dense_jacobian_same", positive=True)
    left, right = s.Dummy("dense_jacobian_same"), s.Dummy("dense_jacobian_same")
    source = s.Matrix([ordinary + 2*positive + 3*left + 4*right])
    coordinates = s.Matrix([[right, ordinary, left, positive, ordinary]])
    assert source.jacobian(coordinates) == s.Matrix([[4, 1, 3, 2, 1]])


@pytest.mark.parametrize("coordinate_type", [s.Matrix, s.SparseMatrix])
def test_derivative_mutation_of_coordinate_matrix_stops_before_the_next_hook(coordinate_type):
    x, y = s.symbols("dense_jacobian_mutation_x dense_jacobian_mutation_y")
    coordinates = coordinate_type([x, y])
    calls = []

    class MutatingLaw(s.Function):
        def fdiff(self, argindex=1):
            calls.append(argindex)
            coordinates[1, 0] = x
            return s.Integer(1)

    source = s.Matrix([MutatingLaw(x), MutatingLaw(y)])
    with pytest.raises(RuntimeError, match="Matrix mutated"):
        source.jacobian(coordinates)
    assert calls == [1]
    assert coordinates[1, 0] is x
    assert source.mutation_generation == 0


@pytest.mark.parametrize("changed", ["source", "coordinates"])
def test_storage_replacement_without_generation_change_is_detected(changed):
    x = s.Symbol("dense_jacobian_replaced")
    coordinates = s.Matrix([x])
    source = None

    class ReplacingLaw(s.Function):
        def fdiff(self, argindex=1):
            target = source if changed == "source" else coordinates
            # The native setter replaces storage without incrementing the
            # public mutation counter; identity checks must still detect it.
            target._native = s.Matrix([2*x])._native
            return s.Integer(1)

    source = s.Matrix([ReplacingLaw(x)])
    with pytest.raises(RuntimeError, match="Matrix mutated"):
        source.jacobian(coordinates)
    assert source.mutation_generation == coordinates.mutation_generation == 0


def test_coordinate_converter_cannot_mutate_the_source_during_capture():
    x = s.Symbol("dense_jacobian_converter")
    source = s.Matrix([x*x])

    class Coordinate:
        def _sympy_(self):
            source[0, 0] = 7
            return x

    with pytest.raises(RuntimeError, match="Matrix mutated"):
        source.jacobian([Coordinate()])
    assert source[0, 0] == 7 and source.mutation_generation == 1


def test_result_construction_cannot_change_captured_coordinates():
    x, y = s.symbols("dense_jacobian_result_x dense_jacobian_result_y")
    coordinates = s.Matrix([x])

    class ResultHookMatrix(s.Matrix):
        def _new(self, *args, **kwargs):
            result = super()._new(*args, **kwargs)
            coordinates[0, 0] = y
            return result

    source = ResultHookMatrix([x*x])
    with pytest.raises(RuntimeError, match="Matrix mutated"):
        source.jacobian(coordinates)
    assert coordinates[0, 0] is y


def test_result_storage_bound_precedes_coordinate_conversion_and_differentiation(monkeypatch):
    x = s.Symbol("dense_jacobian_preflight")
    effects = []

    class Law(s.Function):
        def fdiff(self, argindex=1):
            effects.append("derivative")
            return s.Integer(1)

    class Coordinate:
        def _sympy_(self):
            effects.append("coordinate")
            return x

    source = s.Matrix([Law(x), x, x*x])
    module = importlib.import_module("sympy.matrices.dense")
    monkeypatch.setattr(module, "_MAX_MATRIX_ENTRIES", 4)
    with pytest.raises(ValueError, match="exceeding the limit"):
        source.jacobian([Coordinate(), x])
    assert effects == []


def test_oversized_sparse_coordinate_vector_is_rejected_without_dense_materialization(monkeypatch):
    x = s.Symbol("dense_jacobian_sparse_limit")
    source = s.Matrix([x])
    coordinates = s.SparseMatrix(1_000_000, 1, {(0, 0): x})
    monkeypatch.setattr(
        s.SparseMatrix, "_operation_entries",
        lambda self: pytest.fail("coordinate storage read before shape admission"),
    )
    with pytest.raises(ValueError, match="exceeding the limit"):
        source.jacobian(coordinates)


def test_native_function_names_cannot_recover_opaque_python_classes():
    from sympy.core import _native, _native_expr

    x = s.Symbol("dense_jacobian_native_source")
    calls = []

    class DenseJacobianNativeCollision(s.Function):
        def fdiff(self, argindex=1):
            calls.append(argindex)
            return s.Integer(9)

    native = _native.py_function("DenseJacobianNativeCollision", _native_expr(x))
    source = s.Matrix(_native.Matrix(1, 1, [native]))
    with pytest.raises(NotImplementedError, match="opaque Python function class"):
        source.jacobian([x])
    assert calls == []
