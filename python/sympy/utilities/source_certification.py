"""Exact source-based numerical claims without symbolic derivative compilation.

The native checker independently evaluates the retained source on the supplied
box. A successful enclosure is local to that box; capturing a source or selecting
a Piecewise branch does not establish an unconditional derivative identity.
"""

from fractions import Fraction

from ..core import _native, _symbol_binding
from .lambdify import (
    _argument_layout, _charge, _check_evaluation_budget, _functions, _numeric_source, _residual_source,
)

__all__ = [
    "enclose_residual_system", "certify_root_box", "enclose_objective",
    "certify_stationary_point", "certify_box_extremum", "optimize_box_certified",
    "begin_box_optimization", "restore_box_optimization",
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


def _preserve_source_function_tags(sources):
    """Restore closed surface tags after supervised numerical admission.

    Numerical bytecode uses ``abs`` while the actual built-in Python class is
    ``Abs``. Source-only receipts must retain the latter identity so lifting
    cannot resolve an opaque user function named ``abs``. This pass receives
    only exact native trees already admitted by the closed-class numerical
    lowerer. It preserves held arithmetic and all original children, and never
    changes ordinary compiled programs or their serialized source identities.
    """
    count = [0]

    def visit(source, depth):
        _charge(count, depth)
        if source.is_number or source.is_symbol:
            return source, False
        tag = source.func_name
        children = [visit(child, depth + 1) for child in source.args]
        changed = tag == "abs" or any(changed for _, changed in children)
        if not changed:
            return source, False
        arguments = [child for child, _ in children]
        if source.is_add:
            value = _native.Add(*arguments, evaluate=False).as_expr()
        elif source.is_mul:
            value = _native.Mul(*arguments, evaluate=False).as_expr()
        elif source.is_pow:
            value = _native.Pow(*arguments, evaluate=False).as_expr()
        else:
            value = _native.py_function("Abs" if tag == "abs" else tag, *arguments)
        return value, True

    result = []
    for source in sources:
        # Borrowed bounds precede .args and function-name copies. The shared
        # count additionally bounds all residuals together during this pass.
        _native.CompiledFunction._validate_source(source)
        result.append(visit(source, 0)[0])
    return result


def _residual_context(residuals, args, budget):
    _check_evaluation_budget(budget)
    _, variables, bindings, assumptions, sources, inexact = _residual_source(residuals, args)
    if inexact:
        raise ValueError("certified methods require exact source literals; replace Float with explicit Rational")
    native = _native.SourceResidualSystem(_preserve_source_function_tags(sources), bindings)
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
    native = _native.SourceObjective(_preserve_source_function_tags([source])[0], bindings)
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
    Principal asinh accepts every bounded real argument interval; acosh requires
    the complete interval strictly above one, and atanh strictly inside (-1, 1).
    Abs is admitted when its complete argument enclosure is strictly positive
    or strictly negative. Crossing or touching zero refuses even at a point,
    and all original child-domain obligations still apply.

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
    Asinh has no additional real domain restriction; acosh requires the whole
    argument interval strictly above one, and atanh strictly inside (-1, 1).
    Abs uses the exact locally constant sign on a strictly separated argument
    interval, including the full gradient and Hessian chain rule. An argument
    enclosure containing zero refuses; no nonsmooth extremum is certified.
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
    recomputes original-source value bounds on every leaf, and reevaluates an
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

    Smooth cells use C2 Taylor bounds. Continuous ``Abs`` compositions also
    admit corners through separate Lipschitz value and coordinate-slope
    bounds; these do not assert a derivative, Hessian or stationary point at
    the corner. Other primitives retain their strict source-domain checks,
    including poles hidden by held zero products or zeroth powers. Singular
    non-Lipschitz endpoints such as ``sqrt(Abs(x))`` at zero still refuse.

    Source assumptions apply to the full box. Piecewise guards must have
    stable selections on neighborhoods of all verified cells; unresolved or
    changing seams cannot be bypassed by the cover. Inactive branches remain
    lazy. One caller budget, or one private certification budget when omitted,
    covers exact preparation, search, independent replay and publication.
    Failed bounded search, exhausted resources or cancellation returns no
    partial certificate. ``enclose_objective`` and the root/stationarity
    certifiers retain their stricter C1/C2 requirements at ``Abs`` corners.
    """
    from ._box_optimization import optimize_box_certified as check

    budget = _source_budget(budget)
    context = _objective_context(objective, args, budget)
    return check(context, box, kind, tolerance, precision, max_leaves, budget=budget)


def begin_box_optimization(objective, args, box, *, kind="minimum",
                           tolerance=Fraction(1, 10000), precision=32,
                           max_leaves=512, budget=None):
    """Begin resumable global optimization directly from an exact source.

    No symbolic derivative compilation is required. Like
    optimize_box_certified, this admits nonconvex smooth objectives and
    continuous Abs compositions while retaining original-source domain
    obligations. Exact bounds and tolerance, typed coordinate order, source
    assumptions and options are fixed in the returned BoxOptimizationSearch.

    advance performs bounded bisections; checkpoint saves canonical
    candidates. Only independent finish can publish global value bounds
    and a tolerance-optimal feasible point. All phases share one caller or
    private budget. Save a checkpoint before cancelling or exhausting it.
    """
    from ._box_optimization_search import begin_box_optimization as begin

    budget = _source_budget(budget)
    context = _objective_context(objective, args, budget)
    return begin(context, box, kind, tolerance, precision, max_leaves, budget=budget)


def restore_box_optimization(data, *, objective, args, box, kind="minimum",
                             tolerance=Fraction(1, 10000), precision=32,
                             max_leaves=512, budget=None):
    """Restore candidate bytes against the receiver's exact expected source.

    Source, ordered typed variables, domain, kind, tolerance and options
    must match the checkpoint. The complete saved frontier is reconstructed
    and reevaluated before exposing a continuation. Intrinsic counters are
    retained and replay spends the receiving ledger. Checkpoints contain
    neither Python execution state nor accepted global-optimality evidence;
    a resumed search still requires independent finish verification.
    """
    from ._box_optimization_search import _checkpoint_data, restore_box_optimization as restore

    _checkpoint_data(data, budget)
    budget = _source_budget(budget)
    context = _objective_context(objective, args, budget)
    return restore(context, data, box, kind, tolerance, precision, max_leaves, budget=budget)
