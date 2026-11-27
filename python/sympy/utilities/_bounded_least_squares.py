"""Approximate sparse fitting on an immutable box of coordinate bounds."""

import math

from ..core import Expr, S, _native, _native_expr, _require_symbol
from .lambdify import _surface_binding_key


_NEGATIVE_INFINITY = _native_expr(S.NegativeInfinity)
_POSITIVE_INFINITY = _native_expr(S.Infinity)


class NumericBoundedLeastSquaresCandidate:
    """Immutable source-bound constrained-stationarity proposal.

    The box and numerical values use the actual binary64 inputs. A small
    projected gradient supplies no minimum, root, uniqueness, rounding-error
    or declared-assumption certificate. The full gradient can be nonzero at
    a constrained stationary point. Both vectors follow the original variable
    order and have separately reported infinity norms.
    """

    __slots__ = ("_proposal", "_native_system", "_variables")

    def __init__(self, proposal, native_system, variables):
        if hasattr(self, "_proposal"):
            raise AttributeError("bounded least-squares candidates are immutable")
        if type(proposal) is not _native.NumericBoundedLeastSquaresProposal:
            raise TypeError("an immutable native bounded least-squares proposal is required")
        if type(native_system) is not _native.VerifiedResidualSystem:
            raise TypeError("an immutable native residual system is required")
        if proposal.system is not native_system:
            raise ValueError("bounded proposal belongs to a different residual system")
        from ._certified_numeric import _sequence

        variables = _sequence(variables, native_system.variable_count, "bounded fit variables")
        bindings = tuple(_native_expr(_require_symbol(axis)).symbol_binding for axis in variables)
        if any(binding is None for binding in bindings):
            raise ValueError("bounded fit variables must retain native symbol atoms")
        supplied = tuple(_surface_binding_key(binding) for binding in bindings)
        expected = tuple(_surface_binding_key(binding) for binding in native_system.variables)
        if supplied != expected:
            raise ValueError("bounded fit variables do not match the original typed coordinate order")
        object.__setattr__(self, "_proposal", proposal)
        object.__setattr__(self, "_native_system", native_system)
        object.__setattr__(self, "_variables", variables)

    def __setattr__(self, name, value):
        raise AttributeError("bounded least-squares candidates are immutable")

    def __delattr__(self, name):
        raise AttributeError("bounded least-squares candidates are immutable")

    @property
    def native_system(self):
        return self._native_system

    @property
    def variables(self):
        """The original Symbol/Dummy objects in flat coordinate order."""
        return self._variables

    @property
    def point(self):
        return self._proposal.point

    @property
    def bounds(self):
        """Immutable pairs of the actual binary64 lower/upper endpoints."""
        return tuple(zip(self._proposal.lower_bounds, self._proposal.upper_bounds))

    @property
    def residuals(self):
        return self._proposal.residuals

    @property
    def cost(self):
        """Half the original unweighted squared residual norm."""
        return self._proposal.cost

    @property
    def gradient(self):
        """The full cost gradient Jᵀr as a tuple in the original variable order."""
        return self._proposal.gradient

    @property
    def projected_gradient(self):
        """The gradient with zeroes where a bound blocks the descent direction."""
        return self._proposal.projected_gradient

    @property
    def gradient_norm(self):
        """Infinity norm of the original full J-transpose-r gradient."""
        return self._proposal.gradient_norm

    @property
    def projected_gradient_norm(self):
        """Infinity norm of the bound-projected gradient."""
        return self._proposal.projected_gradient_norm

    @property
    def iterations(self):
        return self._proposal.iterations

    @property
    def evaluations(self):
        return self._proposal.evaluations

    @property
    def termination(self):
        return self._proposal.termination

    @property
    def backend(self):
        return self._proposal.backend

    @property
    def evidence(self):
        return self._proposal.evidence

    def __repr__(self):
        return (f"NumericBoundedLeastSquaresCandidate(point={self.point!r}, "
                f"cost={self.cost!r}, projected_gradient_norm={self.projected_gradient_norm!r}, "
                f"iterations={self.iterations!r})")


def _endpoint(value, lower):
    from .lambdify import _call_arguments

    if value is None:
        return -math.inf if lower else math.inf
    if type(value) is bool:
        raise TypeError("coordinate bounds must be real numbers, not Boolean values")
    if type(value) is Expr:
        # Each symbolic infinity may have a fresh Python wrapper. Compare the
        # closed native atoms after admitting the exact built-in shell class.
        native = _native_expr(value)
        if native == _NEGATIVE_INFINITY:
            number = -math.inf
        elif native == _POSITIVE_INFINITY:
            number = math.inf
        else:
            raise TypeError("symbolic coordinate bounds require an infinity atom")
    elif type(value) is float and math.isinf(value):
        number = value
    else:
        # Closed numeric conversion preserves the ordinary solver's policy:
        # no user __float__, custom subclass, string, or symbolic callback.
        return _call_arguments((None,), (value,), native=True)[0]
    if (lower and number > 0) or (not lower and number < 0):
        raise ValueError("only negative-infinite lower and positive-infinite upper bounds are allowed")
    return number


def solve_bounded_least_squares(program, initial, bounds, gtol, maxiter, max_step_norm, budget):
    from ..solvers.numerical import _candidate_options
    from ._certified_numeric import _sequence
    from .lambdify import _call_arguments, _check_evaluation_budget
    from .source_certification import _source_budget

    budget = _source_budget(budget)
    if any(type(value) is bool for value in (gtol, maxiter, max_step_norm)):
        raise TypeError("bounded least-squares controls must be numeric, not Boolean values")
    n = program.native_system.variable_count
    m = program.native_system.residual_count
    if not 1 <= n <= 256 or not 1 <= m <= 4096:
        raise ValueError("bounded sparse least squares requires 1..256 variables and 1..4096 residuals")
    # Copy the bounded source container before any numeric conversion. Exact
    # pair containers prevent arbitrary iterators or callback properties.
    pairs = _sequence(bounds, n, "bounds")
    pairs = tuple(_sequence(pair, 2, "bound endpoint pair") for pair in pairs)
    lower = [_endpoint(pair[0], True) for pair in pairs]
    upper = [_endpoint(pair[1], False) for pair in pairs]
    gtol, maxiter, max_step_norm = _candidate_options(
        "frankennumpy", 1e-10 if gtol is None else gtol, maxiter,
        max_step_norm, allow_python=False,
    )
    values = initial if type(initial) in (list, tuple) else (initial,)
    initial = _call_arguments((None,) * n, values, native=True)
    proposal = program.native_system.solve_bounded_least_squares(
        initial, lower, upper, gtol=gtol, maxiter=maxiter,
        max_step_norm=max_step_norm, budget=budget,
    )
    result = NumericBoundedLeastSquaresCandidate(proposal, program.native_system, program.variables)
    _check_evaluation_budget(budget)
    return result
