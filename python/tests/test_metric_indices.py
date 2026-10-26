"""Public exact metric index operations; require the actual native extension."""

import pytest

from sympy import Rational, Symbol
from sympy.tensor.tensor import Metric, Tensor, TensorIndex


def exact_metric():
    return Metric.from_components(
        [[2, 1], [1, 2]],
        [[Rational(2, 3), Rational(-1, 3)], [Rational(-1, 3), Rational(2, 3)]],
    )


def test_rank_two_index_positions_and_roundtrip():
    g = exact_metric()
    t = Tensor("T", 2, [TensorIndex("i"), TensorIndex("j")], [1, 2, 3, 4])
    first = g.lower_index(t, 0)
    second = g.lower_index(t, 1)
    assert first.components == (5, 8, 7, 10)
    assert second.components == (4, 5, 10, 11)
    assert tuple(index.is_up for index in first.indices) == (False, True)
    assert g.raise_index(first, 0, "T") == t
    assert g.lower_index(first, 1).components == (18, 21, 24, 27)


def test_metric_contraction_with_indefinite_zero_diagonal():
    g = Metric.from_components([[0, 1], [1, 0]], [[0, 1], [1, 0]])
    t = Tensor("T", 2, [TensorIndex("i"), TensorIndex("j")], [1, 2, 3, 4])
    assert g.lower_index(t, 1).self_contract("i", "j").components == (5,)


@pytest.mark.parametrize("matrix,inverse", [
    ([[2, 1], [1, 2]], [[1, 0], [0, 1]]),
    ([[1, 1], [0, 1]], [[1, -1], [0, 1]]),
    ([[1, 1], [1, 1]], [[1, 0], [0, 1]]),
    ([[1, 0]], [[1]]),
])
def test_bad_metric_invariants_and_shapes_are_rejected(matrix, inverse):
    with pytest.raises(ValueError):
        Metric.from_components(matrix, inverse)


def test_wrong_variance_and_symbolic_components_refuse():
    g = exact_metric()
    t = Tensor("T", 2, [TensorIndex("i")], [1, Symbol("x")])
    with pytest.raises(ValueError):
        g.lower_index(t, 0)
    with pytest.raises(ValueError):
        g.raise_index(t, 0)


def test_existing_vector_api_uses_validated_exact_metric():
    g = exact_metric()
    v = Tensor("v", 2, [TensorIndex("i")], [2, 3])
    assert g.lower_vector(v).components == (7, 8)
    assert g.raise_covector(g.lower_vector(v)).components == (2, 3)
    assert g.inner_product(v, v) == 38
