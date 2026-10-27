"""General utilities (upstream ``sympy.utilities`` subset)."""

from .iterables import (
    default_sort_key, flatten, has_dups, has_variety, ordered, postfixes, prefixes, sift,
    subsets, topological_sort, unflatten, variations,
)
from ..simplify.cse_main import numbered_symbols
from .lambdify import compile_numeric, compile_residual_system, lambdify
from ._certified_numeric import ResidualJacobianEnclosure, RootBoxReceipt

__all__ = [
    "default_sort_key", "flatten", "has_dups", "has_variety", "numbered_symbols", "ordered",
    "postfixes", "prefixes", "sift", "subsets", "topological_sort", "unflatten", "variations",
    "compile_numeric", "compile_residual_system", "lambdify",
    "ResidualJacobianEnclosure", "RootBoxReceipt",
]
