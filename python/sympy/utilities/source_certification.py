"""Exact source-based numerical claims without symbolic derivative compilation.

The native checker independently evaluates the retained source on the supplied
box. A successful enclosure is local to that box; capturing a source or selecting
a Piecewise branch does not establish an unconditional derivative identity.
"""

from fractions import Fraction

from ..core import _native, _symbol_binding
from .lambdify import (
    _argument_layout, _check_evaluation_budget, _functions, _numeric_source, _residual_source,
)

__all__ = [
    "enclose_residual_system", "certify_root_box", "enclose_objective",
    "certify_stationary_point", "certify_box_extremum", "optimize_box_certified",
]


class _SourceContext:
    """Private immutable input to the shared exact-box and receipt adapters."""

    __slots__ = ("_native_source", "_variables", "_has_inexact_source", "_assumptions")

    def __init__(self, native, variables, assumptions):
        if hasattr(self, "_native_source"):
            raise AttributeError("certified source snapshots are immutable")
        object.__setattr__(self, "_native_source", native)
        object.__setattr__(self, "_variables", tuple(variables))
        object.__setattr__(self, "_has_inexact_source", False)
        object.__setattr__(self, "_assumptions", assumptions)

    def __setattr__(self, name, value):
        raise AttributeError("certified source snapshots are immutable")

    def __delattr__(self, name):
        raise AttributeError("certified source snapshots are immutable")

    @property
    def native_system(self):
        return self._native_source

    @property
    def native_objective(self):
        return self._native_source


def _source_budget(budget):
    from ._parametric_root import _fallback_budget

    _check_evaluation_budget(budget)
    # A midpoint enclosure and final certificate are one operation. Neither
    # Python preparation nor a native phase gets a fresh fallback allowance.
    if budget is None:
        budget = _fallback_budget()
    _check_evaluation_budget(budget)
    return budget


def _residual_context(residuals, args, budget):
    _check_evaluation_budget(budget)
    _, variables, bindings, assumptions, sources, inexact = _residual_source(residuals, args)
    if inexact:
        raise ValueError("certified methods require exact source literals; replace Float with explicit Rational")
    native = _native.SourceResidualSystem(sources, bindings)
    return _SourceContext(native, variables, assumptions)


def _objective_context(objective, args, budget):
    from ._certified_numeric import bound_assumptions

    _check_evaluation_budget(budget)
    _, variables = _argument_layout(args)
    bindings = [_symbol_binding(variable) for variable in variables]
    assumptions = bound_assumptions(variables, bindings)
    inexact = [False]
    source = _numeric_source(objective, _functions(), [0], inexact=inexact)
    if inexact[0]:
        raise ValueError("certified methods require exact source literals; replace Float with explicit Rational")
    native = _native.SourceObjective(source, bindings)
    return _SourceContext(native, variables, assumptions)


def enclose_residual_system(residuals, args, box, *, precision=32, budget=None):
    """Enclose exact residual values and Jacobian entries on a supplied box.

    This directly checks the original source with independent first-order
    interval arithmetic. It accepts exact literals beyond binary64 range and
    does not require symbolic derivative compilation. Residual containers
    flatten in row-major order; ``args`` contains typed Symbols/Dummies,
    optionally nested in lists/tuples. ``box`` supplies flat exact rational
    endpoint pairs in argument order, with 1..16 variables and 1..64 residuals.
    Point coordinates are allowed. ``precision`` is an integer from 8 to 128.

    Ordered Piecewise branches are admitted when every visited guard is proved
    stable on a neighborhood of the whole box. Only the selected branch value
    is evaluated. Ambiguous guards, branch seams, missing matching branches and
    unsupported source forms refuse; no global Piecewise derivative is claimed.
    Every evaluated source operation must be C1, including held operations.
    Principal asin and acos require their complete argument intervals strictly
    inside (-1, 1); atan accepts every bounded real argument interval.

    The returned ``ResidualJacobianEnclosure`` retains the original source,
    typed coordinates and box. It makes no root-existence claim. Source
    assumptions, closed built-in classes and hooks use the same admission
    rules as numerical compilation; Float source and Float bounds refuse.
    ``EvaluationBudget`` covers exact-box preparation and all native checking
    cumulatively; when omitted, one private certification budget spans
    the whole operation. Bounded source capture and Python allocation are
    outside it.
    """
    from ._certified_numeric import enclose

    budget = _source_budget(budget)
    context = _residual_context(residuals, args, budget)
    return enclose(context, box, precision, budget=budget)


def certify_root_box(residuals, args, box, *, center=None, preconditioner=None,
                     precision=32, budget=None):
    """Certify a unique nonsingular root from exact source on a real box.

    The system must be square with 1..16 typed variables. Bounds, optional
    center and optional square preconditioner are exact rationals. Each box
    coordinate has positive width. The default center is the box midpoint;
    an omitted preconditioner is proposed by bounded exact inversion of the
    independently enclosed midpoint Jacobian.

    The native checker recomputes the original source, full-box C1 obligations,
    strict interior Krawczyk inclusion and contraction. Piecewise guards must
    remain stable on a neighborhood of the box; unresolved or touched seams
    refuse. Inactive branch values are not evaluated. Symbolic differentiation
    and binary64 compilation are not required or used as proof inputs.

    The immutable ``RootBoxReceipt`` binds the original source and typed box.
    Its claim is local to that box, with no completeness assertion elsewhere.
    The exact-source, assumptions, precision and cumulative-budget contracts
    of ``enclose_residual_system`` apply to preparation and verification alike.
    A false witness, exhausted budget or cancellation returns no receipt.
    """
    from ._certified_numeric import certify_root_box as check

    budget = _source_budget(budget)
    context = _residual_context(residuals, args, budget)
    return check(context, box, center, preconditioner, precision, budget=budget)


def enclose_objective(objective, args, box, *, precision=32, budget=None):
    """Enclose an exact objective, gradient and Hessian on a real box.

    Independent second-order interval arithmetic evaluates the original source;
    symbolic derivative generation and floating-point compilation are not
    required. ``args`` contains 1..16 typed Symbols/Dummies, possibly nested;
    ``box`` supplies flat exact rational endpoint pairs in that order. Point
    coordinates are allowed and ``precision`` ranges from 8 through 128.

    Ordered Piecewise branches require stable visited guards on a neighborhood
    of the complete box. Only selected values are evaluated, with the existing
    C2 requirements on logarithms, square roots, powers and held operations.
    Principal asin and acos require their whole argument intervals strictly
    inside (-1, 1), even under held zero factors or powers. Atan is supported
    on every bounded real argument interval.
    Crossing or touching an unresolved seam, a missing branch, Float source,
    incompatible assumptions or unsupported source grammar refuses explicitly.

    The immutable ``ObjectiveEnclosure`` binds the original source, variables
    and box; it asserts neither stationarity nor global differentiability.
    Exact literals outside binary64 range remain usable. The optional shared
    ``EvaluationBudget`` spans exact-box preparation and native verification;
    one private certification budget is used when none is supplied.
    Bounded source capture and Python allocation remain outside that ledger.
    """
    from ._certified_objective import enclose

    budget = _source_budget(budget)
    context = _objective_context(objective, args, budget)
    return enclose(context, box, precision, budget=budget)


def certify_stationary_point(objective, args, box, *, center=None, preconditioner=None,
                             precision=32, budget=None):
    """Certify a unique nonsingular stationary point from the original source.

    The 1..16 dimensional box, optional center and optional Hessian
    preconditioner use exact rationals. Coordinates require positive width.
    The default center is the midpoint; an omitted preconditioner is proposed
    by bounded exact inversion of the independently enclosed midpoint Hessian.

    Native acceptance recomputes the source and both derivative orders, proves
    the complete C2 domain, strict Krawczyk inclusion and contraction, and may
    classify a strict minimum, strict maximum or saddle. An inconclusive
    curvature test remains ``unclassified``. The receipt claims no global
    optimum or coverage outside the supplied box.

    Piecewise selection must be stable on a neighborhood of the entire box.
    Inactive values are lazy; changing or unresolved seams refuse. All exact
    source, assumptions, precision, admission and cumulative-budget contracts
    of ``enclose_objective`` apply. No symbolic Piecewise derivative certificate
    is generated, and failures or cancellation publish no partial receipt.
    """
    from ._certified_objective import certify_stationary_point as check

    budget = _source_budget(budget)
    context = _objective_context(objective, args, budget)
    return check(context, box, center, preconditioner, precision, budget=budget)


def certify_box_extremum(objective, args, box, point, *, kind="minimum", precision=32,
                         budget=None):
    """Prove that an exact point is a global extremum on a declared box.

    ``kind`` is ``"minimum"`` or ``"maximum"``. The exact rational ``point``
    may lie on a boundary face. Independent original-source C2 evaluation
    checks sufficient convexity/concavity and exact first-order signs on the
    free and constrained coordinates. An arbitrary stationary point is not
    assumed to be a global extremum. Insufficient curvature or a false point
    refuses; nonconvex search is available through ``optimize_box_certified``.

    The immutable ``BoxExtremumReceipt`` retains the original objective,
    variables, box and point, encloses the extremum value, and reports
    uniqueness only when proved. Nothing is claimed outside the declared box.
    Fixed coordinates are allowed. Piecewise guards must be stable on a
    neighborhood of the whole box and inactive branch values remain lazy.

    Exact source, assumptions, closed-class admission, 8..128 precision digits
    and cumulative-budget rules match ``enclose_objective``. Exact source
    constants need not fit binary64; no symbolic derivative compilation is
    required. A private shared certification budget is used when omitted.
    """
    from ._certified_objective import certify_box_extremum as check

    budget = _source_budget(budget)
    context = _objective_context(objective, args, budget)
    return check(context, box, point, kind, precision, budget=budget)


def optimize_box_certified(objective, args, box, *, kind="minimum",
                           tolerance=Fraction(1, 10000), precision=32,
                           max_leaves=512, budget=None):
    """Search exact source for a point with a proved global objective-value gap.

    Bounded branch-and-bound search supports nonconvex objectives on a finite
    box. Independent verification reconstructs a complete dyadic partition,
    recomputes original-source C2 bounds on every leaf, and reevaluates an
    exact feasible point. Candidate values, derivatives and pruning decisions
    have no acceptance authority. No binary64 program or symbolic derivative
    compilation is required.

    The ``BoxOptimizationReceipt`` encloses the global minimum or maximum
    value and proves that the point's value differs from it by at most
    ``gap <= tolerance``. It does not assert coordinate accuracy, an exact
    optimizer, uniqueness or a complete optimizer set. Bounds and the positive
    absolute ``tolerance`` are exact rationals. Fixed coordinates are allowed.
    ``kind`` is ``"minimum"`` or ``"maximum"``, precision is 8..128 digits,
    and ``max_leaves`` is 1..1024.

    Source assumptions apply to the full box. Piecewise guards must have
    stable selections on neighborhoods of all verified cells; unresolved or
    changing seams cannot be bypassed by the cover. Inactive branches remain
    lazy. One caller budget, or one private certification budget when omitted,
    covers exact preparation, search, independent replay and publication.
    Failed bounded search, exhausted resources or cancellation returns no
    partial certificate. Source admission follows ``enclose_objective``.
    """
    from ._box_optimization import optimize_box_certified as check

    budget = _source_budget(budget)
    context = _objective_context(objective, args, budget)
    return check(context, box, kind, tolerance, precision, max_leaves, budget=budget)
