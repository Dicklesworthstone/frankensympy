"""General utilities (upstream ``sympy.utilities`` subset)."""

from .iterables import (
    default_sort_key, flatten, has_dups, has_variety, ordered, postfixes, prefixes, sift,
    subsets, topological_sort, unflatten, variations,
)
from ..simplify.cse_main import numbered_symbols
from .lambdify import (
    EvaluationBudget, EvaluationBudgetExceeded, EvaluationCancelled, NumericRootCandidate,
    compile_numeric, compile_residual_system, load_residual_system, lambdify,
)
from .hybrid import compile_numeric_hybrid
from .hybrid_system import compile_hybrid_residual_system
from .objective import StationarySearchError, compile_objective
from .source_certification import (
    certify_box_extremum, certify_root_box, certify_stationary_point,
    enclose_objective, enclose_residual_system, optimize_box_certified,
)
from ._certified_numeric import ResidualJacobianEnclosure, RootBoxReceipt
from ._root_isolation import RootIsolationReceipt
from ._parametric_root import ParametricRootBoxReceipt
from ._parametric_cover import ParametricRootCoverReceipt
from ._parametric_sensitivity import ParametricSensitivityReceipt
from ._certified_objective import (
    BoxExtremumReceipt, BoxOptimizationReceipt, ObjectiveEnclosure, StationaryPointReceipt,
)

__all__ = [
    "default_sort_key", "flatten", "has_dups", "has_variety", "numbered_symbols", "ordered",
    "postfixes", "prefixes", "sift", "subsets", "topological_sort", "unflatten", "variations",
    "compile_numeric", "compile_numeric_hybrid", "compile_residual_system", "load_residual_system", "compile_objective", "lambdify",
    "compile_hybrid_residual_system",
    "enclose_residual_system", "certify_root_box", "enclose_objective", "certify_stationary_point",
    "certify_box_extremum", "optimize_box_certified",
    "EvaluationBudget", "EvaluationBudgetExceeded", "EvaluationCancelled",
    "NumericRootCandidate",
    "StationarySearchError",
    "ResidualJacobianEnclosure", "RootBoxReceipt",
    "RootIsolationReceipt",
    "ParametricRootBoxReceipt",
    "ParametricRootCoverReceipt",
    "ParametricSensitivityReceipt",
    "ObjectiveEnclosure", "StationaryPointReceipt",
    "BoxExtremumReceipt",
    "BoxOptimizationReceipt",
]
