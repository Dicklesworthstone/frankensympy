"""Hypergeometric and Meijer G-functions for FrankenSymPy."""

from __future__ import annotations
from typing import Any

from ...core import Basic, Expr, Function, Tuple


class hyper(Function):
    """The generalized hypergeometric function: pFq(ap; bq; z)."""

    def __new__(cls, ap: Any, bq: Any, z: Any, **kwargs: Any) -> Any:
        ap = tuple(ap) if isinstance(ap, (list, tuple)) else (ap,)
        bq = tuple(bq) if isinstance(bq, (list, tuple)) else (bq,)
        return super().__new__(cls, Tuple(*ap), Tuple(*bq), z, **kwargs)


class meijerg(Function):
    """The Meijer G-function: G^(m, n)_(p, q)(ap, bq | z)."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        cleaned_args = [Tuple(*a) if isinstance(a, (list, tuple)) else a for a in args]
        return super().__new__(cls, *cleaned_args, **kwargs)


__all__ = [
    "hyper",
    "meijerg",
]
