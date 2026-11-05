"""Sparse retained storage, immutable snapshots and bounded coordinate work."""

import copy

import pytest

from sympy import Add, Function, Integer, Matrix, Rational, SparseMatrix, Symbol, nan, oo, sin
from sympy.matrices.sparse import ImmutableSparseMatrix


def test_sparse_snapshot_copy_and_mutation_preserve_actual_custom_applications():
    x = Symbol("x")
    first = type("Law", (Function,), {})(x)
    second = type("Law", (Function,), {})(x)
    source = SparseMatrix(2, 2, {(0, 1): first, (1, 0): second})
    snapshot = source.as_immutable()
    copied = source.copy()
    deep = copy.deepcopy(source)
    assert type(snapshot) is ImmutableSparseMatrix
    assert snapshot[0, 1] is first and snapshot[1, 0] is second
    assert copied[0, 1] is first and deep[1, 0] is second
    assert source._native_storage is None
    assert source.mutation_generation == 0
    source[0, 1] = 7
    assert source.mutation_generation == 1 and source[0, 1] == 7
    assert snapshot[0, 1] is first and copied[0, 1] is first
    with pytest.raises(TypeError):
        snapshot[0, 1] = 9
    thawed = snapshot.as_mutable()
    thawed[0, 1] = 11
    assert snapshot[0, 1] is first
    assert snapshot.as_immutable() is snapshot
    assert copy.copy(snapshot) is snapshot


def test_huge_opaque_sparse_shapes_only_touch_stored_cells(monkeypatch):
    x = Symbol("x")
    value = type("Law", (Function,), {})(x)
    matrix = SparseMatrix(1_000_000, 1_000_000, {(9, 999_999): value})
    snapshot = matrix.as_immutable()
    with monkeypatch.context() as patch:
        patch.setattr(SparseMatrix, "to_dense", lambda self: pytest.fail("dense conversion"))
        assert matrix.todok() == {(9, 999_999): value}
        assert snapshot.T[999_999, 9] is value
        assert snapshot.copy() is snapshot
        assert isinstance(hash(snapshot), int)
    with pytest.raises(ValueError):
        matrix.to_dense()


def test_constructor_shape_storage_and_coordinate_validation():
    for rows, cols, values in ((2, 2, [1]), (0, 0, [1]), (1, 0, [1])):
        with pytest.raises(ValueError):
            SparseMatrix(rows, cols, values)
    for coordinate in ((-1, 0), (0, -1), (2, 0), (0, 2)):
        with pytest.raises(IndexError):
            SparseMatrix(2, 2, {coordinate: 1})
    with pytest.raises(OverflowError):
        SparseMatrix(-1, 2)
    empty = SparseMatrix(0, 3, [])
    assert empty.shape == (0, 3) and empty.nnz == 0
    with pytest.raises(IndexError):
        empty[0] = 1


def test_negative_index_and_zero_mutation_are_consistent():
    matrix = SparseMatrix([[1, 2], [3, 4]])
    assert matrix[-1] == 4 and matrix[-1, -1] == 4
    matrix[-1] = 0
    assert matrix[-1] == 0 and matrix.nnz == 3
    assert matrix.mutation_generation == 1
    for index in (-5, 4, (2, 0), (0, -3)):
        with pytest.raises(IndexError):
            matrix[index] = 99
    assert matrix.mutation_generation == 1


def test_dense_roundtrip_preserves_held_and_opaque_cells():
    x = Symbol("x")
    law = type("Law", (Function,), {})(x)
    held = Add(x, x, evaluate=False)
    sparse = SparseMatrix(Matrix([[law, held]]))
    assert sparse[0, 0] is law and sparse[0, 1] is held
    assert sparse.to_dense()[0, 0] is law
    assert sparse.to_dense()[0, 1] is held
    assert sparse.as_immutable().to_dense()[0, 0] is law
    with pytest.raises(NotImplementedError):
        _ = sparse._native


def test_sparse_mutation_does_not_publish_after_converter_changed_source():
    matrix = SparseMatrix([[1]])

    class Converter:
        def _sympy_(self):
            matrix[0, 0] = 2
            return Integer(3)

    with pytest.raises(RuntimeError, match="mutated"):
        matrix[0, 0] = Converter()
    assert matrix[0, 0] == 2 and matrix.mutation_generation == 1


def test_native_only_conversions_do_not_lift_a_hostile_same_named_function(monkeypatch):
    import importlib

    from sympy.core import FunctionClass, _native, _native_expr

    effects = []

    def forbidden_getter(self, name):
        effects.append(name)
        raise AssertionError("matrix conversion cannot enter Python function getters")

    collision = FunctionClass("SparseNativeCollision", (Function,), {
        "__module__": __name__, "__getattribute__": forbidden_getter,
    })
    value = _native.py_function("SparseNativeCollision", _native_expr(Symbol("x")))
    source = Matrix(_native.Matrix(1, 2, [value, _native.py_integer(0)]))

    def forbidden_wrap(value):
        raise AssertionError("native-only matrix conversion cannot lift expressions")

    for module in ("sympy.matrices.dense", "sympy.matrices.sparse"):
        monkeypatch.setattr(importlib.import_module(module), "_wrap", forbidden_wrap)
    for cls in (SparseMatrix, ImmutableSparseMatrix):
        sparse = cls(source)
        assert sparse._surface is None
        assert sparse._native_storage[0, 0] == value
        dense = sparse.to_dense()
        assert dense._surface is None
        assert dense._native_storage[0, 0] == value
        transposed = sparse.T
        assert type(transposed) is cls and transposed.shape == (2, 1)
        assert transposed._surface is None
        assert transposed._native_storage[0, 0] == value
    assert collision.__name__ == "SparseNativeCollision"
    assert effects == []


@pytest.mark.parametrize("factory", ["eye", "zeros", "ones"])
def test_immutable_sparse_factories_keep_their_class(factory):
    result = getattr(ImmutableSparseMatrix, factory)(2)
    assert type(result) is ImmutableSparseMatrix
    with pytest.raises(TypeError):
        result[0, 0] = 7


def test_sparse_ones_checks_size_before_allocation_and_retains_empty_shapes():
    with pytest.raises(ValueError, match="stored-entry limit"):
        SparseMatrix.ones(1_000_000, 1_000_000)
    empty = ImmutableSparseMatrix.ones(0, 1_000_000)
    assert empty.shape == (0, 1_000_000) and empty.nnz == 0
    with pytest.raises(OverflowError):
        SparseMatrix.ones(-1, -1)


def test_raw_native_coordinates_cannot_invent_python_function_objects():
    from sympy.core import FunctionClass, _native, _native_expr

    effects = []

    def forbidden(self, name):
        effects.append(name)
        raise AssertionError("raw native cells cannot recover a Python class")

    collision = FunctionClass("SparseRawCoordinateCollision", (Function,), {
        "__module__": __name__, "__getattribute__": forbidden,
    })
    x = Symbol("sparse_raw_x")
    raw = _native.py_function("SparseRawCoordinateCollision", _native_expr(x))
    for constructor in (
        lambda: SparseMatrix(1, 1, {(0, 0): raw}),
        lambda: SparseMatrix(1, 1, [raw]),
        lambda: SparseMatrix([[raw]]),
    ):
        with pytest.raises(NotImplementedError, match="opaque Python function class"):
            constructor()
    source = SparseMatrix([[1]])
    generation = source.mutation_generation
    with pytest.raises(NotImplementedError):
        source[0, 0] = raw
    assert source[0, 0] == 1 and source.mutation_generation == generation

    native_only = SparseMatrix(_native.SparseMatrix(1, 1, {(0, 0): raw}))
    with pytest.raises(NotImplementedError):
        native_only[0, 0] = 0
    assert native_only._surface is None
    assert native_only._native_storage[0, 0] == raw
    assert collision.__name__ == "SparseRawCoordinateCollision"
    assert effects == []


def test_raw_native_atoms_keep_exact_sizes_and_known_functions_are_admitted(monkeypatch):
    from sympy.core import _native_expr
    from sympy.core import _opaque_arithmetic

    x = Symbol("sparse_large_atom")
    values = (Integer(1 << 8192), Rational((1 << 8192) + 1, 3), x)

    def forbidden(value):
        raise AssertionError("atomic cell construction is not numerical compilation")

    with monkeypatch.context() as patcher:
        patcher.setattr(_opaque_arithmetic, "_validate_native_region", forbidden)
        source = SparseMatrix(1, 3, {
            (0, index): _native_expr(value) for index, value in enumerate(values)
        })
        assert tuple(source[0, index] for index in range(3)) == values
    known = SparseMatrix(1, 1, {(0, 0): _native_expr(sin(x))})
    assert known[0, 0] == sin(x)


def test_native_only_sparse_algebra_does_not_lift_function_names(monkeypatch):
    import importlib

    from sympy.core import FunctionClass, _native, _native_expr

    collision = FunctionClass("SparseNativeAlgebraCollision", (Function,), {})
    raw = _native.py_function("SparseNativeAlgebraCollision", _native_expr(Symbol("x")))
    source = SparseMatrix(_native.SparseMatrix(1, 1, {(0, 0): raw}))
    identity = ImmutableSparseMatrix.eye(1)
    zero = SparseMatrix.zeros(1)

    def forbidden(value):
        raise AssertionError("native algebra cannot lift a Python application")

    monkeypatch.setattr(importlib.import_module("sympy.matrices.sparse"), "_wrap", forbidden)
    for result in (source + zero, source - zero, source @ identity,
                   source * identity, source*2, 2*source):
        assert result._surface is None
        assert result.shape == (1, 1)
    assert type(source @ identity) is ImmutableSparseMatrix
    assert collision.__name__ == "SparseNativeAlgebraCollision"


def test_exact_rational_algebra_trace_and_solve_keep_native_fast_paths(monkeypatch):
    left = SparseMatrix([[2, 0], [0, 3]])
    right = SparseMatrix([[5, 0], [0, 7]])

    def forbidden(self):
        raise AssertionError("exact rational sparse operation lost its native fast path")

    monkeypatch.setattr(SparseMatrix, "_operation_entries", forbidden)
    assert (left + right).todok() == {(0, 0): 7, (1, 1): 10}
    assert (left - right).todok() == {(0, 0): -3, (1, 1): -4}
    assert (left @ right).todok() == {(0, 0): 10, (1, 1): 21}
    assert (2*left).todok() == {(0, 0): 4, (1, 1): 6}
    assert left.trace() == 5
    frozen = left.as_immutable()
    assert type(frozen.inv()) is ImmutableSparseMatrix
    result = frozen.solve(SparseMatrix([4, 9]))
    assert type(result) is ImmutableSparseMatrix
    assert result.todok() == {(0, 0): 2, (1, 0): 3}


def test_retained_coordinate_algebra_and_trace_never_scan_huge_shapes(monkeypatch):
    x = Symbol("sparse_coordinate_x")
    a = type("SparseCoordinateLaw", (Function,), {})(x)
    b = type("SparseCoordinateLaw", (Function,), {})(x)
    dimension = 1_000_000
    left = SparseMatrix(dimension, dimension, {(9, 22): a, (20, 8): 2})
    right = SparseMatrix(dimension, dimension, {(22, 13): b, (8, 70): 3})
    diagonal = SparseMatrix(dimension, dimension, {(5, 5): a, (15, 15): b})

    def forbidden(*args):
        raise AssertionError("sparse algebra must only touch stored coordinates")

    with monkeypatch.context() as patcher:
        patcher.setattr(SparseMatrix, "to_dense", forbidden)
        patcher.setattr(SparseMatrix, "__getitem__", forbidden)
        combined = left + right
        assert combined.todok()[(9, 22)] is a
        assert combined.todok()[(22, 13)] is b
        assert (left - left).nnz == 0
        assert (left @ right).todok() == {(9, 13): a*b, (20, 70): 6}
        assert (left*3).todok() == {(9, 22): 3*a, (20, 8): 6}
        assert (3*left).todok() == {(9, 22): 3*a, (20, 8): 6}
        assert diagonal.trace() == a+b
    with pytest.raises(ValueError, match="square"):
        SparseMatrix(2, 3, {(0, 1): a}).trace()


def test_sparse_multiplication_preflights_work_before_scalar_hooks(monkeypatch):
    import importlib

    effects = []

    class WatchedProduct(Function):
        def __mul__(self, other):
            effects.append(other)
            return Integer(1)

    value = WatchedProduct(Symbol("x"))
    left = SparseMatrix(2, 2, {(0, 0): value, (0, 1): value,
                               (1, 0): value, (1, 1): value})
    right = SparseMatrix.ones(2)
    module = importlib.import_module("sympy.matrices.sparse")
    monkeypatch.setattr(module, "_MAX_SPARSE_PRODUCTS", 3)
    with pytest.raises(ValueError, match="term-product limit"):
        left @ right
    assert effects == []


def test_sparse_multiplication_preflights_output_fill_before_scalar_hooks(monkeypatch):
    import importlib

    effects = []

    class WatchedFill(Function):
        def __mul__(self, other):
            effects.append(other)
            return Integer(1)

    value = WatchedFill(Symbol("x"))
    left = SparseMatrix([[value]])
    right = SparseMatrix([[1, 2]])
    module = importlib.import_module("sympy.matrices.sparse")
    monkeypatch.setattr(module, "_MAX_SPARSE_ENTRIES", 1)
    with pytest.raises(ValueError, match="stored-entry limit"):
        left @ right
    assert effects == []


@pytest.mark.parametrize("scalar", [oo, -oo, nan])
def test_closed_nonfinite_scalars_keep_implicit_zero_refusal(scalar):
    value = type("SparseInfiniteLaw", (Function,), {})(Symbol("x"))
    for source in (SparseMatrix.zeros(2), SparseMatrix(2, 2, {}),
                   SparseMatrix(2, 2, {(0, 0): value})):
        with pytest.raises(ValueError):
            source*scalar


@pytest.mark.parametrize("row_source", [False, True])
@pytest.mark.parametrize("form", ["flat", "tuple", "row", "column", "dense", "sparse"])
def test_sparse_jacobian_accepts_bounded_vector_spellings(form, row_source):
    x, y = Symbol("sparse_vector_x"), Symbol("sparse_vector_y")
    variables = {
        "flat": [x, y], "tuple": (x, y), "row": [[x, y]],
        "column": [[x], [y]], "dense": Matrix([x, y]),
        "sparse": SparseMatrix([[x, y]]),
    }[form]
    source = SparseMatrix([[x**2, 0, x+y]] if row_source else [x**2, 0, x+y])
    expected = SparseMatrix([[2*x, 0], [0, 0], [1, 1]])
    assert source.jacobian(variables) == expected


def test_sparse_jacobian_rejects_nonvectors_and_validates_zero_residual_coordinates():
    x = Symbol("sparse_invalid_coordinate")
    source = SparseMatrix([0])
    for variables in (x, [], Matrix(0, 0), [[x, x], [x, x]]):
        with pytest.raises(TypeError, match="row or a column"):
            source.jacobian(variables)
    for invalid in (0, 1):
        with pytest.raises((TypeError, ValueError)):
            source.jacobian([invalid])

    class CustomCoordinate(Symbol):
        pass

    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        source.jacobian([CustomCoordinate("custom_coordinate")])
    for nonvector in (SparseMatrix(0, 0), SparseMatrix.zeros(2, 2)):
        with pytest.raises(TypeError, match="self must"):
            nonvector.jacobian([x])
    for variables in (Matrix(0, 1), Matrix(1, 0), [[]]):
        assert source.jacobian(variables).shape == (1, 0)
    for empty in (SparseMatrix(0, 1), SparseMatrix(1, 0)):
        assert empty.jacobian([x]).shape == (0, 1)


def test_sparse_jacobian_work_bound_precedes_variable_coercion_and_derivative_hooks(monkeypatch):
    import importlib

    effects = []
    x = Symbol("sparse_preflight_x")

    class WatchedDerivative(Function):
        def fdiff(self, argindex=1):
            effects.append("differentiate")
            return Integer(1)

    class CoordinateConverter:
        def _sympy_(self):
            effects.append("coerce")
            return x

    source = SparseMatrix([WatchedDerivative(x), x])
    module = importlib.import_module("sympy.matrices.sparse")
    monkeypatch.setattr(module, "_MAX_SPARSE_PRODUCTS", 1)
    with pytest.raises(ValueError, match="differentiation work limit"):
        source.jacobian([CoordinateConverter()])
    assert effects == []


def test_huge_sparse_differentiation_substitution_and_symbols_keep_stored_work(monkeypatch):
    x = Symbol("sparse_transform_x")
    source = ImmutableSparseMatrix(1_000_000, 1_000_000, {(9, 99): x**2})

    def forbidden(*args):
        raise AssertionError("stored symbolic transforms cannot densify")

    monkeypatch.setattr(SparseMatrix, "to_dense", forbidden)
    derivative = source.diff(x)
    assert type(derivative) is ImmutableSparseMatrix
    assert derivative.todok() == {(9, 99): 2*x}
    assert source.subs(x, 3).todok() == {(9, 99): 9}
    assert source.subs(0, 1).todok() == {(9, 99): x**2}
    assert source.free_symbols == {x}


def test_sparse_assignment_rejects_reinitialization_during_result_conversion():
    source = SparseMatrix([[1]])

    class Reinitializer:
        def _sympy_(self):
            source.__init__([[7, 8]])
            return Integer(3)

    with pytest.raises(RuntimeError, match="mutated during callback traversal"):
        source[0, 0] = Reinitializer()
    assert source.shape == (1, 2)
    assert source.todok() == {(0, 0): 7, (0, 1): 8}
