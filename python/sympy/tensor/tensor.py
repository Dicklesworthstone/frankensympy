"""Symbolic tensors, index notation, and metrics for FrankenSymPy."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from ..core import Basic, Expr, _native, _native_expr, _wrap


class TensorIndex(Basic):
    """Symbolic tensor index with contravariant (upper) or covariant (lower) variance."""

    __slots__ = ("_native_index",)

    def __new__(cls, name: str, is_up: bool = True):
        obj = object.__new__(cls)
        if _native is not None and hasattr(_native, "TensorIndex"):
            obj._native_index = _native.TensorIndex(name, is_up)
        else:
            raise NotImplementedError("fsym_python TensorIndex is required")
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def name(self) -> str:
        return self._native_index.name

    @property
    def is_up(self) -> bool:
        return self._native_index.is_up

    def flip(self) -> TensorIndex:
        """Return index with inverted variance (upper <-> lower)."""
        obj = object.__new__(TensorIndex)
        obj._native_index = self._native_index.flip()
        return obj

    def __repr__(self) -> str:
        return repr(self._native_index)

    def __str__(self) -> str:
        return str(self._native_index)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, TensorIndex):
            return self._native_index == other._native_index
        return False

    def __hash__(self) -> int:
        return hash((self.name, self.is_up))


def tensor_indices(names: str | Iterable[str], is_up: bool = True) -> list[TensorIndex] | TensorIndex:
    """Generate one or more TensorIndex objects.

    Examples
    --------
    >>> mu, nu = tensor_indices('mu nu')
    >>> mu.is_up
    True
    """
    if isinstance(names, str):
        name_list = names.replace(",", " ").split()
    else:
        name_list = list(names)

    indices = [TensorIndex(name, is_up=is_up) for name in name_list]
    if len(indices) == 1 and not isinstance(names, (list, tuple)):
        return indices[0]
    return indices


class Tensor(Basic):
    """Symbolic tensor expression with named indices and optional concrete components."""

    __slots__ = ("_native_tensor",)

    def __new__(
        cls,
        name: str,
        dimension: int,
        indices: Sequence[TensorIndex],
        components: Sequence[Any] | None = None,
    ):
        obj = object.__new__(cls)
        if _native is not None and hasattr(_native, "Tensor"):
            native_indices = [idx._native_index for idx in indices]
            native_comps = None
            if components is not None:
                native_comps = [_native_expr(c) for c in components]
            obj._native_tensor = _native.Tensor(name, dimension, native_indices, native_comps)
        else:
            raise NotImplementedError("fsym_python Tensor is required")
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @classmethod
    def _from_native(cls, native_t: Any) -> Tensor:
        obj = object.__new__(cls)
        obj._native_tensor = native_t
        return obj

    @property
    def name(self) -> str:
        return self._native_tensor.name

    @property
    def rank(self) -> int:
        return self._native_tensor.rank

    @property
    def dimension(self) -> int:
        return self._native_tensor.dimension

    @property
    def indices(self) -> tuple[TensorIndex, ...]:
        raw_indices = self._native_tensor.indices
        out = []
        for raw in raw_indices:
            idx = object.__new__(TensorIndex)
            idx._native_index = raw
            out.append(idx)
        return tuple(out)

    @property
    def components(self) -> tuple[Expr, ...] | None:
        comps = self._native_tensor.components
        if comps is None:
            return None
        return tuple(_wrap(c) for c in comps)

    def outer_product(self, other: Tensor, new_name: str | None = None) -> Tensor:
        """Outer product (tensor product) with another Tensor."""
        res = self._native_tensor.outer_product(other._native_tensor, new_name)
        return Tensor._from_native(res)

    def self_contract(
        self, upper_name: str, lower_name: str, new_name: str | None = None
    ) -> Tensor:
        """Contract an upper index with a lower index within this tensor."""
        res = self._native_tensor.self_contract(upper_name, lower_name, new_name)
        return Tensor._from_native(res)

    def contract(
        self, other: Tensor, upper_name: str, lower_name: str, new_name: str | None = None
    ) -> Tensor:
        """Contract an index of self with an index of other."""
        res = self._native_tensor.contract(other._native_tensor, upper_name, lower_name, new_name)
        return Tensor._from_native(res)

    def contract_index(self, target_name: str, new_name: str | None = None) -> Tensor:
        """Raise or lower an index symbolically."""
        res = self._native_tensor.contract_index(target_name, new_name)
        return Tensor._from_native(res)

    def __repr__(self) -> str:
        return repr(self._native_tensor)

    def __str__(self) -> str:
        return str(self._native_tensor)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Tensor):
            return self._native_tensor == other._native_tensor
        return False


class Metric(Basic):
    """Metric tensor providing index raising, lowering, and inner products."""

    __slots__ = ("_native_metric",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @classmethod
    def minkowski_4d(cls, name: str = "eta") -> Metric:
        """Standard 4D Minkowski spacetime metric diag(-1, 1, 1, 1)."""
        obj = object.__new__(cls)
        obj._native_metric = _native.Metric.minkowski_4d(name)
        return obj

    @classmethod
    def euclidean(cls, dimension: int, name: str = "delta") -> Metric:
        """Euclidean metric of dimension N diag(1, ..., 1)."""
        obj = object.__new__(cls)
        obj._native_metric = _native.Metric.euclidean(name, dimension)
        return obj

    @classmethod
    def diagonal(cls, diag_entries: Sequence[Any], name: str = "g") -> Metric:
        """Diagonal metric with given nonzero diagonal entries."""
        obj = object.__new__(cls)
        raw = [_native_expr(e) for e in diag_entries]
        obj._native_metric = _native.Metric.diagonal(name, raw)
        return obj

    @property
    def name(self) -> str:
        return self._native_metric.name

    @property
    def dimension(self) -> int:
        return self._native_metric.dimension

    @property
    def matrix(self) -> tuple[Expr, ...]:
        return tuple(_wrap(e) for e in self._native_metric.matrix)

    @property
    def inverse(self) -> tuple[Expr, ...]:
        return tuple(_wrap(e) for e in self._native_metric.inverse)

    def lower_vector(self, vec_tensor: Tensor) -> Tensor:
        """Lower the contravariant vector index: V_mu = g_{mu nu} V^nu."""
        res = self._native_metric.lower_vector(vec_tensor._native_tensor)
        return Tensor._from_native(res)

    def raise_covector(self, covec_tensor: Tensor) -> Tensor:
        """Raise the covariant covector index: W^mu = g^{mu nu} W_nu."""
        res = self._native_metric.raise_covector(covec_tensor._native_tensor)
        return Tensor._from_native(res)

    def inner_product(self, u: Tensor, v: Tensor) -> Expr:
        """Compute inner product <u, v> = g_{mu nu} u^mu v^nu."""
        res = self._native_metric.inner_product(u._native_tensor, v._native_tensor)
        return _wrap(res)

    def norm_squared(self, v: Tensor) -> Expr:
        """Compute norm squared ||v||^2 = g_{mu nu} v^mu v^nu."""
        res = self._native_metric.norm_squared(v._native_tensor)
        return _wrap(res)

    def __repr__(self) -> str:
        return repr(self._native_metric)

    def __str__(self) -> str:
        return str(self._native_metric)


def tensorproduct(*tensors: Tensor) -> Tensor:
    """Compute the outer product of multiple tensors."""
    if not tensors:
        raise ValueError("tensorproduct requires at least one tensor")
    result = tensors[0]
    for t in tensors[1:]:
        result = result.outer_product(t)
    return result


def tensorcontraction(tensor: Tensor, *pairs: tuple[str, str]) -> Tensor:
    """Perform contraction of index pairs in tensor.

    Each pair is (upper_name, lower_name).
    """
    result = tensor
    for up, low in pairs:
        result = result.self_contract(up, low)
    return result
