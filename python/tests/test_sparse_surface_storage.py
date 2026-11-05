"""Sparse retained storage, immutable snapshots and bounded coordinate work."""

import copy

import pytest

from sympy import Add, Function, Integer, Matrix, SparseMatrix, Symbol
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
