"""Complete parameter coverage through independently checked root families."""

from ..core import _native
from ._certified_numeric import _Receipt, _bounds, _call_budgeted, _native_box
from ._parametric_root import (
    ParametricRootBoxReceipt, _fallback_budget, _prepare_parametric_domains,
)
from .lambdify import _check_evaluation_budget

__all__ = ["ParametricRootCoverReceipt"]


class ParametricRootCoverReceipt(_Receipt):
    """A unique unknown root for every point of the complete parameter box.

    All accepted leaves share exactly the same unknown_domain. Each leaf
    retains its full parameter sub-box, and independent partition replay
    proves that their union covers parameter_domain without missing regions.
    root_enclosure is a hull containing the entire family, not a precision
    promise or an exact symbolic solution function.
    """
    __slots__ = ()

    def __init__(self, receipt):
        if hasattr(self, "_receipt"):
            raise AttributeError("accepted numerical receipts are immutable")
        if type(receipt) is not _native.ParametricRootCoverReceipt:
            raise TypeError("an accepted native parametric root-cover receipt is required")
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
    def leaves(self):
        return tuple(ParametricRootBoxReceipt(receipt) for receipt in self._receipt.leaves)

    @property
    def leaf_count(self):
        return self._receipt.leaf_count

    @property
    def root_enclosure(self):
        return _bounds(self._receipt.root_enclosure)

    @property
    def claim(self):
        return ("for every parameter value in the complete declared parameter box, "
                "one unique nonsingular real root in the fixed declared unknown box")

    def __repr__(self):
        return (f"ParametricRootCoverReceipt(leaf_count={self.leaf_count}, "
                f"root_enclosure={self.root_enclosure!r}, "
                f"unknown_domain={self.unknown_domain!r}, "
                f"parameter_domain={self.parameter_domain!r}, claim={self.claim!r})")


def cover_parametric_roots_certified(
        program, unknown_box, parameter_box, precision, max_leaves, *, budget=None):
    """Search a full parameter partition and independently verify every leaf."""
    _check_evaluation_budget(budget)
    if budget is None:
        budget = _fallback_budget()
    _check_evaluation_budget(budget)
    if type(max_leaves) is not int or not 1 <= max_leaves <= 1024:
        raise ValueError("parametric cover max_leaves must be an integer from 1 through 1024")
    unknown_box, parameter_box = _prepare_parametric_domains(
        program, unknown_box, parameter_box, precision, budget=budget,
    )
    receipt = _call_budgeted(
        program.native_system.cover_parametric_roots,
        _native_box(unknown_box, budget), _native_box(parameter_box, budget),
        precision, max_leaves, budget=budget,
    )
    result = ParametricRootCoverReceipt(receipt)
    _check_evaluation_budget(budget)
    return result
