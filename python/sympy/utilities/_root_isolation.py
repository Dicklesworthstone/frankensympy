"""Exact inputs and immutable receipts for complete real root isolation.

Only the native complete-cover verifier grants acceptance. An unresolved cell
cannot be discarded, and a local root certificate alone never establishes
that the original finite domain contains no other roots.
"""

from ..core import _native
from ._certified_numeric import (
    RootBoxReceipt, _Receipt, _call_budgeted, _native_box, _prepare_box,
)
from .lambdify import _check_evaluation_budget

__all__ = ["RootIsolationReceipt"]


class RootIsolationReceipt(_Receipt):
    """All distinct nonsingular real roots in the declared finite closed box.

    Each member of roots is an independently accepted RootBoxReceipt. The
    complete cover proves there are no additional roots in domain. A zero
    root_count proves the declared domain is root-free. No statement is made
    about roots outside domain or exact symbolic coordinates.
    """
    __slots__ = ()

    def __init__(self, receipt):
        if hasattr(self, "_receipt"):
            raise AttributeError("accepted numerical receipts are immutable")
        if type(receipt) is not _native.RootIsolationReceipt:
            raise TypeError("an accepted native complete root-isolation receipt is required")
        object.__setattr__(self, "_receipt", receipt)

    @property
    def roots(self):
        return tuple(RootBoxReceipt(receipt) for receipt in self._receipt.roots)

    @property
    def root_count(self):
        return self._receipt.root_count

    @property
    def leaf_count(self):
        return self._receipt.leaf_count

    @property
    def excluded_leaf_count(self):
        return self._receipt.excluded_leaf_count

    @property
    def is_root_free(self):
        return self._receipt.is_root_free

    @property
    def claim(self):
        return (f"exactly {self.root_count} distinct nonsingular real roots in "
                "the declared closed box; all roots are covered")

    def __repr__(self):
        return (f"RootIsolationReceipt(root_count={self.root_count}, "
                f"domain={self.domain!r}, leaf_count={self.leaf_count}, "
                f"claim={self.claim!r})")


def isolate_roots_certified(program, box, precision, max_leaves, *, budget=None):
    """Run bounded native discovery and independent complete-cover replay."""
    _check_evaluation_budget(budget)
    if type(max_leaves) is not int or not 1 <= max_leaves <= 1024:
        raise ValueError("root isolation max_leaves must be an integer from 1 through 1024")
    box = _prepare_box(program, box, precision, square=True, budget=budget)
    receipt = _call_budgeted(
        program.native_system.isolate_roots,
        _native_box(box, budget), precision, max_leaves, budget=budget,
    )
    result = RootIsolationReceipt(receipt)
    _check_evaluation_budget(budget)
    return result
