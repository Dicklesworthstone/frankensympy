"""Checked scalar objectives and sparse second derivatives for numerical use.

The native checker admits both differentiation stages and their compiled
instructions. Evaluation returns approximate finite-real binary64 values.
Separate enclosure and stationary-point methods independently check the
original exact source with second-order interval arithmetic.
"""

from fractions import Fraction

from ..core import (
    _native, _native_expr, _require_symbol, _surface_binding_key, _symbol_binding, _wrap,
)
from .lambdify import (
    _argument_layout, _call_arguments, _check_evaluation_budget, _functions, _numeric_source,
)

__all__ = [
    "MinimizationSearchError", "NumericMinimizationCandidate",
    "StationarySearchError", "compile_objective",
]


class NumericMinimizationCandidate:
    """Immutable approximate stationary proposal for one compiled objective.

    Value and gradient diagnostics are freshly evaluated at point. An initial
    stationary maximum or saddle also meets the numerical stopping criterion.
    This object supplies no minimum, uniqueness, global-optimality, rounding
    error or declared-assumption certificate.
    """

    __slots__ = ("_proposal", "_native_objective", "_variables")

    def __init__(self, proposal, native_objective, variables):
        if hasattr(self, "_proposal"):
            raise AttributeError("minimization candidates are immutable")
        if type(proposal) is not _native.NumericMinimizationProposal:
            raise TypeError("an immutable native minimization proposal is required")
        if type(native_objective) is not _native.VerifiedObjective:
            raise TypeError("an immutable native objective is required")
        if native_objective is not proposal.objective:
            raise ValueError("minimization proposal belongs to a different objective")
        from ._certified_numeric import _sequence
        variables = _sequence(
            variables, native_objective.variable_count, "minimization variables",
        )
        # Read the immutable native atom, not a later-mutated assumptions
        # dictionary. Dummy intern identities remain distinct in this handle.
        bindings = tuple(_native_expr(_require_symbol(axis)).symbol_binding for axis in variables)
        if any(binding is None for binding in bindings):
            raise ValueError("minimization variables must retain native symbol atoms")
        supplied = tuple(_surface_binding_key(binding) for binding in bindings)
        expected = tuple(_surface_binding_key(axis) for axis in native_objective.variables)
        if supplied != expected:
            raise ValueError("minimization variables do not match the captured typed coordinate order")
        object.__setattr__(self, "_proposal", proposal)
        object.__setattr__(self, "_native_objective", native_objective)
        object.__setattr__(self, "_variables", variables)

    def __setattr__(self, name, value):
        raise AttributeError("minimization candidates are immutable")

    def __delattr__(self, name):
        raise AttributeError("minimization candidates are immutable")

    @property
    def point(self):
        return self._proposal.point

    @property
    def objective(self):
        """The immutable original source captured by the compiled objective."""
        return _wrap(self._native_objective.source)

    @property
    def variables(self):
        """The caller's captured Symbol/Dummy objects in coordinate order."""
        return self._variables

    @property
    def value(self):
        return self._proposal.value

    @property
    def gradient(self):
        return self._proposal.gradient

    @property
    def gradient_norm(self):
        """Infinity norm of the freshly evaluated binary64 gradient."""
        return self._proposal.gradient_norm

    @property
    def iterations(self):
        return self._proposal.iterations

    @property
    def evaluations(self):
        """Full source/gradient/Hessian evaluations, including rejected trials."""
        return self._proposal.evaluations

    @property
    def regularizations(self):
        """Number of shifted Hessian systems attempted during descent."""
        return self._proposal.regularizations

    @property
    def termination(self):
        return "gradient_tolerance"

    @property
    def backend(self):
        return "frankennumpy"

    @property
    def evidence(self):
        return "approximate"

    def __repr__(self):
        return (f"NumericMinimizationCandidate(point={self.point!r}, value={self.value!r}, "
                f"gradient_norm={self.gradient_norm!r}, iterations={self.iterations!r})")


class StationarySearchError(ValueError):
    """A bounded search did not obtain a local stationary-point certificate.

    "stage" is "candidate" or "certification". "failures" records rejected
    certificate trials and "attempts" gives their number. An unsuccessful
    search makes no assertion that a stationary point is absent. Cancellation
    and exhausted evaluation budgets retain their own exception types instead
    of becoming this mathematical-search outcome.
    """

    def __init__(self, stage, message, *, failures=()):
        self.stage = stage
        self.failures = tuple(failures)
        self.attempts = len(self.failures)
        super().__init__(message)


class MinimizationSearchError(StationarySearchError):
    """A bounded search did not obtain a strict local minimum certificate.

    stage is "candidate", "certification" or "classification". Rejected boxes
    and inconclusive classifications retain their failures and attempt count.
    Failure makes no assertion that a minimum is absent. Cancellation, budget
    exhaustion and allocation failure retain their operational exception types.
    """


class _ObjectiveCallable:
    """Argument layout around an immutable, source-bound native objective."""

    __slots__ = ("_layout", "_native_objective", "_variables", "_has_inexact_source", "_assumptions")

    def __init__(self, layout, native, variables, inexact=False, assumptions=None):
        if hasattr(self, "_native_objective"):
            raise AttributeError("compiled objectives are immutable")
        object.__setattr__(self, "_layout", layout)
        object.__setattr__(self, "_native_objective", native)
        object.__setattr__(self, "_variables", tuple(variables))
        object.__setattr__(self, "_has_inexact_source", inexact)
        object.__setattr__(self, "_assumptions", assumptions)

    def __setattr__(self, name, value):
        raise AttributeError("compiled objectives are immutable")

    def __delattr__(self, name):
        raise AttributeError("compiled objectives are immutable")

    @property
    def native_objective(self):
        return self._native_objective

    @property
    def variables(self):
        """Original argument objects in flat gradient/vector coordinate order."""
        return self._variables

    @property
    def hessian_pattern(self):
        """Immutable, ordered (row, column) coordinates of Hessian nonzeros."""
        return tuple(self.native_objective.hessian_pattern)

    def enclose(self, box, *, precision=32, budget=None):
        """Enclose value, gradient and Hessian from the original exact source.

        ``box`` is a flat list/tuple of exact rational endpoint pairs in
        ``variables`` order, with 1..16 variables and 8..128 precision digits.
        Each original operation must be C2 on the complete real box, including
        operations retained behind a zero multiplier or zero exponent. Source
        assumptions are the snapshots captured during compilation. Float source
        literals remain approximate-only. A supplied ``EvaluationBudget`` is
        cumulative across preparation, native verification and earlier calls.
        """
        from ._certified_objective import enclose
        return enclose(self, box, precision, budget=budget)

    def certify_stationary_point(self, box, *, center=None, preconditioner=None,
                                 precision=32, budget=None):
        """Certify a unique nonsingular stationary point inside one real box.

        Bounds, optional center and optional square preconditioner use exact
        rationals. The default center is the box midpoint. When omitted, the
        preconditioner is proposed by bounded exact midpoint-Hessian inversion.
        Native acceptance independently recomputes both derivative orders from
        the original objective, checks the full C2 domain, strict interior
        inclusion and contraction. Compiled derivatives are not proof inputs.

        The receipt may classify a strict local minimum, strict local maximum
        or saddle; an inconclusive curvature test remains ``unclassified``.
        No global optimum or coverage outside this box is asserted. The same
        optional caller budget covers proposal and verification without reset.
        """
        from ._certified_objective import certify_stationary_point
        return certify_stationary_point(
            self, box, center, preconditioner, precision, budget=budget,
        )

    def certify_box_extremum(self, box, point, *, kind="minimum", precision=32,
                             budget=None):
        """Prove a global minimum or maximum on a declared closed real box.

        The exact rational candidate may lie on a boundary face. Independent
        native second-order evaluation checks the original source on the full
        box, sufficient convexity/concavity, and exact first-order face signs.
        The returned BoxExtremumReceipt encloses the optimum value; it reports
        uniqueness only when proved. No claim extends outside the box.

        Bounds and point coordinates must be exact rational inputs in flat
        variable order. Captured assumptions and one optional cumulative
        EvaluationBudget cover the same stages as the other certified methods.
        """
        from ._certified_objective import certify_box_extremum
        return certify_box_extremum(
            self, box, point, kind, precision, budget=budget,
        )

    def optimize_box_certified(self, box, *, kind="minimum",
                               tolerance=Fraction(1, 10000), precision=32,
                               max_leaves=512, budget=None):
        """Search a finite box and certify a global objective-value error.

        This bounded branch-and-bound search admits nonconvex objectives.
        Native acceptance reconstructs a complete dyadic partition, checks
        original-source C2 bounds on every leaf, and reevaluates an exact
        feasible point. Its BoxOptimizationReceipt encloses the global
        optimum value and proves that the point's objective value is within
        "gap <= tolerance" of it. Coordinate distance to an optimizer,
        uniqueness and complete optimizer sets are not asserted.

        Bounds and tolerance are exact rational inputs. Captured assumptions
        apply to the entire box. Precision is 8..128 digits and max_leaves is
        1..1024; a failed bounded search returns no partial receipt. One
        optional EvaluationBudget covers input preparation, all candidate
        cells, independent verification and final publication cumulatively.
        """
        from ._box_optimization import optimize_box_certified
        return optimize_box_certified(
            self, box, kind, tolerance, precision, max_leaves, budget=budget,
        )

    def solve_stationary_certified(self, initial, *, tol=1e-12, maxiter=200,
                                   precision=32, max_attempts=12, budget=None):
        """Find and independently certify one nonsingular stationary point.

        "initial" is a finite real scalar for one variable, or a flat
        list/tuple in "variables" order, including when ordinary evaluation
        arguments are nested. The captured native gradient and Hessian drive
        bounded, damped Newton iteration with a scaled QR linear solve. Every
        trial evaluates the original objective and both derivative orders.

        The binary64 candidate only proposes exact rational boxes. Acceptance
        uses the independent original-source C2, strict inclusion and
        contraction checks of "certify_stationary_point". The returned
        StationaryPointReceipt may classify a strict local minimum, strict
        local maximum or saddle, or retain an unclassified stationary point.
        This is a local search, with no global optimum or completeness claim.

        Exact source literals and 1..16 variables are required. Certification
        tries 1..24 boxes (default 12), varying width and increasing precision
        from the requested 8..128 digits up to 128. Newton uses a positive
        finite tolerance and 1..10,000 iterations, with additional fixed work
        caps. StationarySearchError distinguishes candidate failure from
        exhausted certificate trials; no floating candidate is returned.
        One optional EvaluationBudget spans all evaluations, exact witness
        proposals, rejected trials and final publication without reset.
        """
        from ._stationary_search import solve_stationary_certified
        return solve_stationary_certified(
            self, initial, tol, maxiter, precision, max_attempts, budget=budget,
        )

    def minimize(self, initial, *, gtol=None, maxiter=None, max_step=None,
                 budget=None):
        """Find an approximate stationary point by bounded Newton descent.

        initial is a finite real scalar for one variable or a flat built-in
        list/tuple in variables order, including with nested evaluation args.
        The checked gradient and Hessian drive equilibrated pinned FrankenNumPy
        linear solves, bounded diagonal regularization and Armijo backtracking.
        Controls default to gtol=1e-10, maxiter=64 and max_step=1e6. The gradient
        tolerance is nonnegative, maxiter is 0..1024, and the positive step
        limit clips the infinity norm before at most 64 line-search trials.

        Success requires a freshly evaluated gradient infinity norm <= gtol.
        A small step or unchanged objective alone cannot succeed. The returned
        immutable NumericMinimizationCandidate contains the point, source,
        value, gradient and work counts. A stationary maximum or saddle can
        satisfy this criterion, so it carries no minimum certificate.

        Every initial, trial and final evaluation retains the full original
        objective and both derivative domains. Mathematical domain/nonfinite
        trial failures backtrack; operational errors remain terminal. One
        supplied EvaluationBudget accounts cumulatively for native buffers,
        failed trials, donor work and final reevaluation. Cancellation is
        cooperative around synchronous donor work. Python object allocation
        and compilation are outside the ledger; no fallback resets its work.
        """
        from ..solvers.numerical import _candidate_options

        _check_evaluation_budget(budget)
        dimension = self.native_objective.variable_count
        if not 1 <= dimension <= 16:
            raise ValueError("minimization requires 1..16 variables")
        gtol, maxiter, max_step = _candidate_options(
            "frankennumpy", 1e-10 if gtol is None else gtol, maxiter,
            max_step, allow_python=False,
        )
        values = initial if type(initial) in (list, tuple) else (initial,)
        inputs = _call_arguments((None,) * dimension, values, native=True)
        proposal = self.native_objective.minimize(
            inputs, gtol=gtol, maxiter=maxiter, max_step=max_step, budget=budget,
        )
        result = NumericMinimizationCandidate(
            proposal, self.native_objective, self.variables,
        )
        _check_evaluation_budget(budget)
        return result

    def minimize_certified(self, initial, *, gtol=None, maxiter=None, max_step=None,
                           precision=32, max_attempts=12, budget=None):
        """Find and independently certify one strict local minimum.

        The bounded minimizer supplies only approximate coordinates. Exact
        rational witness boxes then go through the original-source C2,
        strict inclusion and contraction checker. A result is returned only
        when the immutable native receipt proves strict local minimum
        curvature. Maximum and saddle classifications refuse; unclassified
        boxes retry within the same attempt cap. A singular minimum may remain
        inconclusive. No global optimum or completeness claim is made.

        Exact source literals and 1..16 variables with supported captured
        Boolean assumptions are required. Numerical controls match minimize.
        precision is 8..128 digits and max_attempts is 1..24 (default 12).
        Static eligibility is checked before initial conversion or numerical
        work. MinimizationSearchError identifies candidate, certification or
        classification failure and never returns an unverified candidate.

        One cumulative EvaluationBudget spans numerical search, exact witness
        proposals, failed boxes and final verification. If omitted, a private
        default budget is created once for the entire operation. Cancellation,
        exhausted budgets and allocation failures keep their exception types.
        """
        from ._objective_minimization import minimize_certified
        return minimize_certified(
            self, initial, gtol, maxiter, max_step, precision, max_attempts,
            budget=budget,
        )

    def evaluate_sparse(self, *values, budget=None):
        """Return (value, gradient, sparse Hessian values) after all checks pass.

        Sparse values follow ``hessian_pattern`` in row-major order, retaining
        both mixed-partial entries independently. No dense matrix is allocated.
        A supplied ``EvaluationBudget`` covers all three native stages together
        and retains consumed work if a derivative fails or is cancelled.
        """
        _check_evaluation_budget(budget)
        inputs = _call_arguments(self._layout, values, native=True)
        return self.native_objective.evaluate(inputs, budget=budget)

    def evaluate(self, *values, budget=None):
        """Return (value, gradient, dense Hessian rows), all approximate.

        Source and first-derivative domain checks run before any results return.
        The dense output is limited by the native maximum of 64 variables.
        """
        value, gradient, sparse = self.evaluate_sparse(*values, budget=budget)
        size = self.native_objective.variable_count
        if budget is not None:
            budget._charge(size*size + len(sparse))
        hessian = [[0.0] * size for _ in range(size)]
        for (row, column), entry in zip(self.hessian_pattern, sparse):
            hessian[row][column] = entry
        if budget is not None:
            budget._checkpoint()
        return value, gradient, hessian

    def hessian_vector_product(self, direction, *values, budget=None):
        """Evaluate H(x) times a flat direction without forming a dense Hessian.

        ``direction`` is an exact list/tuple with one real value per variable
        in ``variables`` order, even when ordinary arguments are nested.
        Invalid dimensions, nonfinite arithmetic and source-domain failures
        raise an exception without publishing partial values.
        """
        _check_evaluation_budget(budget)
        size = self.native_objective.variable_count
        if type(direction) not in (list, tuple) or len(direction) != size:
            raise TypeError("Hessian direction must be a list/tuple with one entry per variable")
        vector = _call_arguments((None,) * size, direction, native=True)
        inputs = _call_arguments(self._layout, values, native=True)
        return self.native_objective.hessian_vector_product(inputs, vector, budget=budget)


def compile_objective(objective, args):
    """Compile a scalar objective, checked gradient and checked sparse Hessian.

    ``args`` contains typed Symbols/Dummies, optionally nested in lists/tuples.
    Both differentiation stages are independently replayed, including omitted
    zero entries and compiled instruction provenance. Mixed partials are checked
    in both orders; symmetry is not assumed. The admitted program is immutable.

    ``evaluate`` returns the value, gradient and dense Hessian. Use
    ``evaluate_sparse`` or ``hessian_vector_product`` to avoid dense output.
    Those evaluation methods are synchronous finite-real binary64 programs.
    Separate ``enclose`` and ``certify_stationary_point`` methods independently
    check the original exact source on complete real boxes using second-order
    interval arithmetic, with source-bound immutable receipts and local claims.
    ``solve_stationary_certified`` connects a bounded numerical search to that
    original-source checker, without granting proof authority to convergence.
    Float source literals remain usable only for approximation. Passing a reusable
    ``EvaluationBudget`` to evaluation methods adds cumulative fuel and native
    buffer accounting with cooperative native safe points. Compilation and
    CPython allocation overhead are outside that ledger; no worker is spawned.
    Unsupported functions, opaque hooks, unmatched symbols and oversized work
    refuse explicitly. No upstream SymPy or Python callback executes the math.
    """
    layout, variables = _argument_layout(args)
    from ._certified_numeric import bound_assumptions
    bindings = [_symbol_binding(variable) for variable in variables]
    assumptions = bound_assumptions(variables, bindings)
    inexact = [False]
    source = _numeric_source(objective, _functions(), [0], inexact=inexact)
    native = _native.VerifiedObjective(source, bindings)
    return _ObjectiveCallable(layout, native, variables, inexact[0], assumptions)
