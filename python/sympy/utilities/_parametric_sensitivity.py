"""Original-source derivative bounds for accepted implicit root families.

Only an immutable native root-family receipt can supply existence and full
parameter coverage. No supplied derivative, numerical sample or root search
is trusted by this separate operation.
"""

from ..core import _native, _wrap
from ._certified_numeric import _Receipt, _bounds, _call_budgeted
from ._parametric_root import _fallback_budget
from .lambdify import _check_evaluation_budget

__all__ = ["ParametricSensitivityReceipt"]


class ParametricSensitivityReceipt(_Receipt):
    """Exact interval bounds on the accepted family's implicit Jacobian.

    solution_jacobian[i][j] encloses dx_i/dp_j throughout parameter_domain.
    Rows follow unknown_variables; columns follow parameters, including
    fixed and unused coordinates. At a closed face these are derivatives
    of the local implicit extension restricted to the declared box.

    The bounds may be conservative. No exact solution formula, requested
    accuracy, or existence outside the original unknown box is asserted.
    """
    __slots__ = ()

    def __init__(self, receipt):
        if hasattr(self, "_receipt"):
            raise AttributeError("accepted numerical receipts are immutable")
        if type(receipt) is not _native.ParametricSensitivityReceipt:
            raise TypeError("an accepted native parametric sensitivity receipt is required")
        object.__setattr__(self, "_receipt", receipt)

    @property
    def unknown_variables(self):
        return tuple(self._receipt.unknown_variables)

    @property
    def parameters(self):
        return tuple(self._receipt.parameters)

    @property
    def unknown_domain(self):
        return _bounds(self._receipt.unknown_domain)

    @property
    def parameter_domain(self):
        return _bounds(self._receipt.parameter_domain)

    @property
    def root_enclosure(self):
        return _bounds(self._receipt.root_enclosure)

    @property
    def solution_jacobian(self):
        return tuple(_bounds(row) for row in self._receipt.solution_jacobian)

    @property
    def contraction_bound(self):
        return _wrap(self._receipt.contraction_bound)

    @property
    def leaf_count(self):
        return self._receipt.leaf_count

    @property
    def claim(self):
        return ("original-source interval bounds on every implicit derivative dx_i/dp_j "
                "of the unique root family throughout the complete declared parameter box")

    def __repr__(self):
        return (f"ParametricSensitivityReceipt(solution_jacobian={self.solution_jacobian!r}, "
                f"leaf_count={self.leaf_count}, parameter_domain={self.parameter_domain!r}, "
                f"claim={self.claim!r})")


def enclose_sensitivity(family, *, budget=None):
    """Independently check a whole accepted family with one cumulative ledger."""
    if type(family) not in (_native.ParametricRootBoxReceipt, _native.ParametricRootCoverReceipt):
        raise TypeError("an accepted native parametric root-family receipt is required")
    _check_evaluation_budget(budget)
    if budget is None:
        budget = _fallback_budget()
    _check_evaluation_budget(budget)
    receipt = _call_budgeted(family.enclose_sensitivity, budget=budget)
    result = ParametricSensitivityReceipt(receipt)
    _check_evaluation_budget(budget)
    return result
