"""Bounded continuations of original-source complete root isolation.

Search progress and serialized checkpoints are untrusted candidates. Only
``finish`` can return the independent complete-cover verifier's receipt.
"""

from ..core import _native, _wrap
from ._certified_numeric import _bounds, _native_box, _prepare_box
from ._root_isolation import RootIsolationReceipt
from .lambdify import _check_evaluation_budget

__all__ = ["RootIsolationProgress", "RootIsolationSearch"]

_MAX_CHECKPOINT_BYTES = 1024 * 1024


class RootIsolationProgress:
    """Immutable candidate statistics, with no accepted root-count claim.

    ``ready_for_verification`` says that the search has no pending cells.
    The candidate root boxes and complete cover still require independent
    original-source verification through ``RootIsolationSearch.finish``.
    A previously returned progress object does not change after another step.
    """

    __slots__ = ("_progress",)

    def __init__(self, progress):
        if hasattr(self, "_progress"):
            raise AttributeError("root isolation progress snapshots are immutable")
        if type(progress) is not _native.RootIsolationProgress:
            raise TypeError("a native root isolation progress snapshot is required")
        object.__setattr__(self, "_progress", progress)

    def __setattr__(self, name, value):
        raise AttributeError("root isolation progress snapshots are immutable")

    def __delattr__(self, name):
        raise AttributeError("root isolation progress snapshots are immutable")

    @property
    def processed_cells(self):
        return self._progress.processed_cells

    @property
    def pending_cells(self):
        return self._progress.pending_cells

    @property
    def candidate_root_count(self):
        return self._progress.candidate_root_count

    @property
    def excluded_cells(self):
        return self._progress.excluded_cells

    @property
    def leaf_count(self):
        return self._progress.leaf_count

    @property
    def ready_for_verification(self):
        return self._progress.ready_for_verification

    def __repr__(self):
        return (f"RootIsolationProgress(processed_cells={self.processed_cells}, "
                f"pending_cells={self.pending_cells}, "
                f"candidate_root_count={self.candidate_root_count}, "
                f"ready_for_verification={self.ready_for_verification})")


class RootIsolationSearch:
    """A source-bound root search sharing one cumulative evaluation budget.

    Create it with a compiled system's ``begin_root_isolation`` or restore
    its canonical checkpoint with ``restore_root_isolation``. The original
    residuals, ordered typed variables, exact domain and search limits remain
    fixed. ``advance`` attempts a bounded number of complete cell transitions;
    a refused transition remains pending and earlier progress is retained.

    All preparation, advances, checkpoint encoding and independent final
    verification spend the same ``budget``. Cancellation is cooperative.
    An exhausted or cancelled ledger is never replaced by another allowance.
    Reading ``progress`` remains useful after a refused advance.

    Checkpoints retain normalized candidate evidence and consumed intrinsic
    search counters. Loading requires the receiver's own expected source,
    ordered typed coordinates, domain and limits, and rechecks the prefix.
    Checkpoints do not contain Python execution state, an accepted receipt,
    or authenticated evidence of a remote caller's budget history.
    """

    __slots__ = ("_search",)

    def __init__(self, search):
        if hasattr(self, "_search"):
            raise AttributeError("root isolation search bindings are immutable")
        if type(search) is not _native.RootIsolationSearch:
            raise TypeError("a native root isolation search is required")
        object.__setattr__(self, "_search", search)

    def __setattr__(self, name, value):
        raise AttributeError("root isolation search bindings are immutable")

    def __delattr__(self, name):
        raise AttributeError("root isolation search bindings are immutable")

    @property
    def residuals(self):
        return tuple(_wrap(value) for value in self._search.residuals)

    @property
    def variables(self):
        """Immutable native SymbolBindings in the declared coordinate order."""
        return tuple(self._search.variables)

    @property
    def domain(self):
        return _bounds(self._search.domain)

    @property
    def precision_digits(self):
        return self._search.precision_digits

    @property
    def max_leaves(self):
        return self._search.max_leaves

    @property
    def max_advance_cells(self):
        return self._search.max_advance_cells

    @property
    def max_checkpoint_bytes(self):
        return self._search.max_checkpoint_bytes

    @property
    def budget(self):
        """The original shared ledger; callers may inspect it or cancel it."""
        return self._search.budget

    @property
    def progress(self):
        return RootIsolationProgress(self._search.progress)

    def advance(self, max_cells=1):
        """Attempt 1..2047 complete cell transitions and return a snapshot.

        This bounds transition count, not elapsed time. Each cell may require
        several original-source interval evaluations. Resource exhaustion,
        cancellation and unresolved cells raise without granting acceptance.
        """
        _check_evaluation_budget(self.budget)
        if type(max_cells) is not int or not 1 <= max_cells <= self.max_advance_cells:
            raise ValueError("root isolation max_cells must be an integer from 1 through 2047")
        result = RootIsolationProgress(self._search.advance(max_cells))
        _check_evaluation_budget(self.budget)
        return result

    def checkpoint(self):
        """Encode bounded canonical candidate bytes for source-bound restore.

        Export spends the captured ledger. Save a checkpoint before cancelling
        or exhausting that ledger; export does not bypass resource refusal.
        The bytes use the native Session root-isolation checkpoint format.
        """
        _check_evaluation_budget(self.budget)
        result = self._search.checkpoint()
        _check_evaluation_budget(self.budget)
        return result

    def finish(self):
        """Independently verify a completed cover and return its receipt.

        Pending cells refuse. A completed search is still only a candidate:
        original-source replay, distinctness, coverage and the final budget
        checkpoint must all succeed before a RootIsolationReceipt is returned.
        """
        _check_evaluation_budget(self.budget)
        result = RootIsolationReceipt(self._search.finish())
        _check_evaluation_budget(self.budget)
        return result

    def __repr__(self):
        return (f"RootIsolationSearch(precision_digits={self.precision_digits}, "
                f"max_leaves={self.max_leaves}, progress={self.progress!r})")


def _prepare_search(program, box, precision, max_leaves, budget):
    from .source_certification import _source_budget

    budget = _source_budget(budget)
    if type(max_leaves) is not int or not 1 <= max_leaves <= 1024:
        raise ValueError("root isolation max_leaves must be an integer from 1 through 1024")
    # The compiled program owns the assumption snapshot. Never recapture
    # potentially mutated Python argument objects during begin or restore.
    box = _prepare_box(program, box, precision, square=True, budget=budget)
    return _native_box(box, budget), budget


def begin_root_isolation(program, box, precision, max_leaves, *, budget=None):
    box, budget = _prepare_search(program, box, precision, max_leaves, budget)
    native = program.native_system.begin_root_isolation(
        box, precision, max_leaves, budget=budget,
    )
    result = RootIsolationSearch(native)
    _check_evaluation_budget(budget)
    return result


def restore_root_isolation(program, data, box, precision, max_leaves, *, budget=None):
    _check_evaluation_budget(budget)
    if type(data) is not bytes:
        raise TypeError("root isolation checkpoint must be exact built-in bytes")
    if len(data) > _MAX_CHECKPOINT_BYTES:
        raise ValueError("root isolation checkpoint exceeds the 1 MiB byte limit")
    box, budget = _prepare_search(program, box, precision, max_leaves, budget)
    native = program.native_system.restore_root_isolation(
        data, box, precision, max_leaves, budget=budget,
    )
    result = RootIsolationSearch(native)
    _check_evaluation_budget(budget)
    return result
