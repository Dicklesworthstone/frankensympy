"""Bounded continuations of original-source global box optimization.

Progress and checkpoints are candidates. Only finish can publish a receipt
from the independent complete-cover and objective-value checker.
"""

from ..core import _native, _wrap
from ._certified_numeric import (
    _bounds, _charge_exact_work, _native_box, _native_number, _rational,
)
from ._certified_objective import BoxOptimizationReceipt, _prepare_box
from .lambdify import _check_evaluation_budget

__all__ = ["BoxOptimizationProgress", "BoxOptimizationSearch"]

_MAX_CHECKPOINT_BYTES = 1024 * 1024


class BoxOptimizationProgress:
    """An immutable candidate snapshot without a global-optimality claim.

    status is searching, ready_for_verification or failed. Being ready means
    candidate search has reached the requested value gap; independent
    original-source verification is still required. Previously returned
    snapshots do not change when a search advances.
    """

    __slots__ = ("_progress",)

    def __init__(self, progress):
        if hasattr(self, "_progress"):
            raise AttributeError("box optimization progress snapshots are immutable")
        if type(progress) is not _native.BoxOptimizationProgress:
            raise TypeError("a native box optimization progress snapshot is required")
        object.__setattr__(self, "_progress", progress)

    def __setattr__(self, name, value):
        raise AttributeError("box optimization progress snapshots are immutable")

    def __delattr__(self, name):
        raise AttributeError("box optimization progress snapshots are immutable")

    @property
    def status(self):
        return self._progress.status

    @property
    def leaf_count(self):
        return self._progress.leaf_count

    @property
    def split_count(self):
        return self._progress.split_count

    @property
    def evaluation_count(self):
        return self._progress.evaluation_count

    @property
    def ready_for_verification(self):
        return self._progress.ready_for_verification

    def __repr__(self):
        return (f"BoxOptimizationProgress(status={self.status!r}, "
                f"leaf_count={self.leaf_count}, split_count={self.split_count}, "
                f"evaluation_count={self.evaluation_count})")


class BoxOptimizationSearch:
    """A source-bound global search with one cumulative evaluation budget.

    Begin with begin_box_optimization or a compiled objective's method, and
    save canonical bytes with checkpoint. Restoration requires the receiver's
    own expected source, typed coordinate order, box, kind, tolerance and
    options. All are fixed for the lifetime of this handle.

    Preparation, bounded advances, checkpoint encoding, restore replay and
    final independent verification spend the same captured budget. An omitted
    budget becomes one private ledger, retained between calls. A refused
    native transition is terminal; its statistics remain readable. Save a
    checkpoint before cancellation or exhaustion to retain a usable
    continuation. A restore retains intrinsic work counters and charges a
    receiving ledger; it cannot authenticate a remote caller's budget history.

    Checkpoints contain candidates, not accepted bounds or Python execution
    state. finish must reconstruct the complete cover and independently
    reevaluate the original source before returning a BoxOptimizationReceipt.
    """

    __slots__ = ("_search",)

    def __init__(self, search):
        if hasattr(self, "_search"):
            raise AttributeError("box optimization search bindings are immutable")
        if type(search) is not _native.BoxOptimizationSearch:
            raise TypeError("a native box optimization search is required")
        object.__setattr__(self, "_search", search)

    def __setattr__(self, name, value):
        raise AttributeError("box optimization search bindings are immutable")

    def __delattr__(self, name):
        raise AttributeError("box optimization search bindings are immutable")

    @property
    def objective(self):
        return _wrap(self._search.objective)

    @property
    def variables(self):
        """Immutable native SymbolBindings in the declared coordinate order."""
        return tuple(self._search.variables)

    @property
    def domain(self):
        return _bounds(self._search.domain)

    @property
    def kind(self):
        return self._search.kind

    @property
    def absolute_tolerance(self):
        return _wrap(self._search.absolute_tolerance)

    @property
    def precision_digits(self):
        return self._search.precision_digits

    @property
    def max_leaves(self):
        return self._search.max_leaves

    @property
    def max_advance_splits(self):
        return self._search.max_advance_splits

    @property
    def max_checkpoint_bytes(self):
        return self._search.max_checkpoint_bytes

    @property
    def budget(self):
        """The captured shared ledger; callers may inspect it or cancel it."""
        return self._search.budget

    @property
    def progress(self):
        return BoxOptimizationProgress(self._search.progress)

    def advance(self, max_splits=1):
        """Perform at most 1..1024 bisections and return candidate progress.

        This bounds splits, not elapsed time. Each split can require several
        original-source interval evaluations. Resource or domain refusal
        grants no partial acceptance and never refreshes the shared budget.
        """
        _check_evaluation_budget(self.budget)
        if type(max_splits) is not int or not 1 <= max_splits <= self.max_advance_splits:
            raise ValueError("box optimization max_splits must be an integer from 1 through 1024")
        result = BoxOptimizationProgress(self._search.advance(max_splits))
        _check_evaluation_budget(self.budget)
        return result

    def checkpoint(self):
        """Export canonical candidate bytes in the native Session format.

        Export spends the captured ledger and refuses a terminal search.
        Save before cancelling or exhausting it; export cannot bypass refusal.
        """
        _check_evaluation_budget(self.budget)
        result = self._search.checkpoint()
        _check_evaluation_budget(self.budget)
        return result

    def finish(self):
        """Independently verify a ready cover and publish a global value gap.

        Searching or failed candidates refuse. Original-source domain checks,
        complete coverage, objective bounds, feasible-point evaluation and a
        final budget checkpoint must succeed before a receipt is returned.
        The receipt bounds objective-value error, not optimizer coordinates.
        """
        _check_evaluation_budget(self.budget)
        result = BoxOptimizationReceipt(self._search.finish())
        _check_evaluation_budget(self.budget)
        return result

    def __repr__(self):
        return (f"BoxOptimizationSearch(kind={self.kind!r}, "
                f"absolute_tolerance={self.absolute_tolerance!r}, "
                f"max_leaves={self.max_leaves}, progress={self.progress!r})")


def _checkpoint_data(data, budget):
    _check_evaluation_budget(budget)
    if type(data) is not bytes:
        raise TypeError("box optimization checkpoint must be exact built-in bytes")
    if len(data) > _MAX_CHECKPOINT_BYTES:
        raise ValueError("box optimization checkpoint exceeds the 1 MiB byte limit")


def _prepare_search(program, box, kind, tolerance, precision, max_leaves, budget):
    from .source_certification import _source_budget

    budget = _source_budget(budget)
    if type(kind) is not str:
        raise TypeError("box-optimization kind must be an exact string")
    if kind not in ("minimum", "maximum"):
        raise ValueError("kind must be 'minimum' or 'maximum'")
    if type(max_leaves) is not int or not 1 <= max_leaves <= 1024:
        raise ValueError("box optimization max_leaves must be an integer from 1 through 1024")
    # Retain the compilation-time assumptions, never recapture mutated Python
    # argument objects when starting or restoring this native source.
    box = _prepare_box(program, box, precision, stationary=False, budget=budget)
    tolerance = _rational(tolerance, budget)
    _charge_exact_work(budget, tolerance)
    if tolerance <= 0:
        raise ValueError("box optimization tolerance must be a positive exact rational")
    return _native_box(box, budget), _native_number(tolerance, budget), budget


def begin_box_optimization(program, box, kind, tolerance, precision, max_leaves, *, budget=None):
    box, tolerance, budget = _prepare_search(
        program, box, kind, tolerance, precision, max_leaves, budget,
    )
    result = BoxOptimizationSearch(program.native_objective.begin_box_optimization(
        box, tolerance, kind, precision, max_leaves, budget=budget,
    ))
    _check_evaluation_budget(budget)
    return result


def restore_box_optimization(program, data, box, kind, tolerance, precision, max_leaves,
                             *, budget=None):
    _checkpoint_data(data, budget)
    box, tolerance, budget = _prepare_search(
        program, box, kind, tolerance, precision, max_leaves, budget,
    )
    result = BoxOptimizationSearch(program.native_objective.restore_box_optimization(
        data, box, tolerance, kind, precision, max_leaves, budget=budget,
    ))
    _check_evaluation_budget(budget)
    return result
