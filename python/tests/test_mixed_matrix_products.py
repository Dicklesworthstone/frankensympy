"""Sparse/dense products keep the pinned classes without coefficient expansion."""

import importlib
import operator

import pytest
import sympy as s

from sympy.core import _native, _native_expr


def forbid_dense(*args, **kwargs):
    pytest.fail("mixed matrix operation expanded its sparse operand")


@pytest.mark.parametrize("operation", [operator.mul, operator.matmul])
@pytest.mark.parametrize("sparse_cls", [s.SparseMatrix, s.ImmutableSparseMatrix])
@pytest.mark.parametrize("dense_cls", [s.Matrix, s.ImmutableMatrix])
@pytest.mark.parametrize("sparse_first", [True, False])
def test_rectangular_products_match_pinned_result_classes(
    operation, sparse_cls, dense_cls, sparse_first, monkeypatch,
):
    left_cls, right_cls = ((sparse_cls, dense_cls) if sparse_first
                           else (dense_cls, sparse_cls))
    left = left_cls([[1, 0, 2], [0, -3, 0]])
    right = right_cls([[4, 0], [0, 5], [6, 7]])
    cls = (s.ImmutableSparseMatrix if sparse_cls is s.ImmutableSparseMatrix else
           s.ImmutableMatrix if dense_cls is s.ImmutableMatrix else s.Matrix)
    monkeypatch.setattr(s.SparseMatrix, "to_dense", forbid_dense)
    result = operation(left, right)
    assert type(result) is cls
    assert result == cls([[16, 14], [0, -15]])
    assert left.mutation_generation == right.mutation_generation == 0
    assert result.mutation_generation == 0


@pytest.mark.parametrize("sparse_cls", [s.SparseMatrix, s.ImmutableSparseMatrix])
def test_large_coefficients_times_thin_dense_operands_never_scan_sparse_area(sparse_cls, monkeypatch):
    size = 600
    coefficient = sparse_cls(size, size, {(0, size-1): 3, (size-1, 0): -2})
    column = s.Matrix(size, 1, [5] + [0]*(size-2) + [7])
    row = s.Matrix(1, size, [11] + [0]*(size-2) + [13])
    monkeypatch.setattr(s.SparseMatrix, "to_dense", forbid_dense)
    monkeypatch.setattr(s.SparseMatrix, "__getitem__", forbid_dense)
    product = coefficient*column
    reverse = row*coefficient
    expected_cls = s.Matrix if sparse_cls is s.SparseMatrix else s.ImmutableSparseMatrix
    assert type(product) is type(reverse) is expected_cls
    assert product.shape == (size, 1) and reverse.shape == (1, size)
    # Access the owned native result, since sparse scalar scanning is forbidden.
    assert product._native_storage[0, 0] == _native.py_integer(21)
    assert product._native_storage[size-1, 0] == _native.py_integer(-10)
    assert reverse._native_storage[0, 0] == _native.py_integer(-26)
    assert reverse._native_storage[0, size-1] == _native.py_integer(33)


@pytest.mark.parametrize("sparse_first", [True, False])
@pytest.mark.parametrize("shape", [(0, 0, 3), (2, 0, 3), (0, 4, 2), (3, 2, 0)])
def test_zero_and_empty_products_keep_shape_without_dense_conversion(sparse_first, shape, monkeypatch):
    rows, inner, cols = shape
    left_cls, right_cls = ((s.SparseMatrix, s.Matrix) if sparse_first
                           else (s.Matrix, s.SparseMatrix))
    left, right = left_cls.zeros(rows, inner), right_cls.zeros(inner, cols)
    monkeypatch.setattr(s.SparseMatrix, "to_dense", forbid_dense)
    result = left*right
    assert type(result) is s.Matrix and result.shape == (rows, cols)
    assert result == s.Matrix.zeros(rows, cols)


@pytest.mark.parametrize("sparse_first", [True, False])
def test_retained_same_named_functions_and_typed_symbols_keep_owners(sparse_first, monkeypatch):
    x = s.Symbol("mixed_identity_x", positive=True)
    first_cls = type("MixedIdentity", (s.Function,), {})
    second_cls = type("MixedIdentity", (s.Function,), {})
    first, second = first_cls(x, evaluate=False), second_cls(x, evaluate=False)
    if sparse_first:
        left = s.SparseMatrix([[first, second], [x, 0]])
        right = s.Matrix.eye(2)
    else:
        left = s.Matrix([[first, second], [x, 0]])
        right = s.SparseMatrix.eye(2)
    monkeypatch.setattr(s.SparseMatrix, "to_dense", forbid_dense)
    result = left*right
    assert result[0, 0] is first and result[0, 1] is second
    assert result[1, 0] is x
    assert result.free_symbols == {x}


@pytest.mark.parametrize("sparse_first", [True, False])
def test_native_only_function_names_never_invent_python_owners(sparse_first, monkeypatch):
    collision = type("MixedNativeCollision", (s.Function,), {})
    raw = _native.py_function("MixedNativeCollision", _native_expr(s.Symbol("mixed_raw_x")))
    sparse = s.SparseMatrix(_native.SparseMatrix(1, 1, {(0, 0): raw}))
    dense = s.Matrix(_native.Matrix(1, 1, [_native.py_integer(1)]))
    monkeypatch.setattr(s.SparseMatrix, "to_dense", forbid_dense)
    result = sparse*dense if sparse_first else dense*sparse
    assert result._surface is None
    assert result._native_storage[0, 0] == raw
    assert collision.__name__ == "MixedNativeCollision"


@pytest.mark.parametrize("sparse_first", [True, False])
def test_mixed_equality_is_symmetric_and_does_not_expand_storage(sparse_first, monkeypatch):
    sparse = s.SparseMatrix([[0, 2, 0], [3, 0, 4]])
    same = s.Matrix([[0, 2, 0], [3, 0, 4]])
    different = s.Matrix([[0, 2, 1], [3, 0, 4]])
    monkeypatch.setattr(s.SparseMatrix, "to_dense", forbid_dense)
    left, right = (sparse, same) if sparse_first else (same, sparse)
    assert left == right
    assert sparse != different and different != sparse
    assert sparse != s.Matrix.zeros(3, 2)


@pytest.mark.parametrize("target", ["sparse", "dense", "retained"])
def test_scalar_callbacks_cannot_publish_after_source_mutation(target):
    sparse = dense = None

    class MutatingProduct(s.Function):
        def __mul__(self, other):
            if target == "sparse":
                sparse[0, 0] = 9
            elif target == "dense":
                dense[0, 0] = 9
            else:
                sparse._surface[0, 0] = s.Integer(9)
            return s.Integer(3)

    sparse = s.SparseMatrix([[MutatingProduct(s.Symbol("mixed_mutation_x"))]])
    dense = s.Matrix([[2]])
    with pytest.raises(RuntimeError, match="mutated"):
        sparse*dense


def test_resource_preflights_precede_scalar_callbacks(monkeypatch):
    effects = []

    class WatchedProduct(s.Function):
        def __mul__(self, other):
            effects.append(other)
            return s.Integer(1)

    value = WatchedProduct(s.Symbol("mixed_bound_x"))
    sparse = s.SparseMatrix([[value, value]])
    dense = s.Matrix([[1, 2], [3, 4]])
    module = importlib.import_module("sympy.matrices.sparse")
    monkeypatch.setattr(module, "_MAX_SPARSE_PRODUCTS", 3)
    with pytest.raises(ValueError, match="term-product limit"):
        sparse*dense
    assert effects == []


def test_dense_output_area_is_bounded_before_scalar_work():
    class WatchedProduct(s.Function):
        def __mul__(self, other):
            pytest.fail("oversized output reached scalar arithmetic")

    value = WatchedProduct(s.Symbol("mixed_area_x"))
    sparse = s.SparseMatrix(513, 1, {(0, 0): value})
    dense = s.Matrix.ones(1, 512)
    with pytest.raises(ValueError, match="exceeding the limit"):
        sparse*dense


def test_retained_rational_cells_cannot_borrow_a_stale_native_mirror():
    sparse = s.SparseMatrix([[1]])
    dense = s.Matrix([[2]])
    sparse._surface[0, 0] = s.Integer(9)
    with pytest.raises(RuntimeError, match="retained cells differ"):
        sparse*dense


@pytest.mark.parametrize("changed", ["native", "storage", "constructor"])
def test_changed_owners_refuse_before_dispatch(changed, monkeypatch):
    left, right = s.SparseMatrix([[2]]), s.Matrix([[3]])
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("changed owner was dispatched")

    if changed == "native":
        monkeypatch.setattr(_native.SparseMatrix, "__matmul__", forbidden)
    elif changed == "storage":
        monkeypatch.setattr(s.SparseMatrix, "_surface", property(forbidden))
    else:
        monkeypatch.setattr(s.Matrix, "__init__", forbidden)
    with pytest.raises(NotImplementedError, match="original"):
        left*right
    assert calls == []


@pytest.mark.parametrize("sparse_first", [True, False])
@pytest.mark.parametrize("raw", [True, False])
@pytest.mark.parametrize("value", [s.oo, -s.oo, s.nan])
def test_implicit_zero_nonfinite_products_refuse_instead_of_returning_zero(
    sparse_first, raw, value,
):
    sparse = s.SparseMatrix.zeros(1, 1)
    dense = (s.Matrix(_native.Matrix(1, 1, [_native_expr(value)])) if raw else
             s.Matrix([[value]]))
    with pytest.raises(ValueError, match="implicit-zero"):
        sparse*dense if sparse_first else dense*sparse


@pytest.mark.parametrize("sparse_first", [True, False])
def test_skipped_implicit_zeros_cannot_discard_scalar_operator_callbacks(sparse_first):
    calls = []

    class EffectfulProduct(s.Function):
        def __mul__(self, other):
            calls.append("multiply")
            return s.Integer(7)

        def __rmul__(self, other):
            calls.append("reflected multiply")
            return s.Integer(7)

    dense = s.Matrix([[EffectfulProduct(s.Symbol("mixed_zero_effect_x"))]])
    sparse = s.SparseMatrix.zeros(1, 1)
    with pytest.raises(NotImplementedError, match="implicit-zero"):
        sparse*dense if sparse_first else dense*sparse
    assert calls == []
