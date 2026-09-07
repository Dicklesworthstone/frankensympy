"""Calculus module for FrankenSymPy."""

from .singularities import (
    is_decreasing,
    is_increasing,
    is_monotonic,
    is_strictly_decreasing,
    is_strictly_increasing,
    singularities,
)
from .util import (
    AccumBounds,
    AccumulationBounds,
    continuous_domain,
    function_range,
    maximum,
    minimum,
    periodicity,
    stationary_points,
)

__all__ = [
    "AccumBounds",
    "AccumulationBounds",
    "continuous_domain",
    "function_range",
    "is_decreasing",
    "is_increasing",
    "is_monotonic",
    "is_strictly_decreasing",
    "is_strictly_increasing",
    "maximum",
    "minimum",
    "periodicity",
    "singularities",
    "stationary_points",
]
