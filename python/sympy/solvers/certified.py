"""Certified existence of real roots in exact, continuous scalar brackets.

This public operation returns evidence of at least one root, including repeated
or nonsmooth roots when an exact endpoint or a sign change proves existence.
It does not claim uniqueness or a complete list of roots.
"""

from fractions import Fraction

from ..core import Dummy, Symbol, _native, _wrap
from ..utilities._certified_numeric import (
    _charge_exact_work, _check_assumptions, _native_number, _rational, _sequence,
    bound_assumptions,
)
from ..utilities.lambdify import _check_evaluation_budget, _functions, _numeric_source
from ..utilities.source_certification import _preserve_source_function_tags, _source_budget

__all__ = ["certified_root_bracket", "ScalarRootBracketReceipt"]


def _root_binding(variable):
    if type(variable) not in (Symbol, Dummy):
        raise TypeError("root variable must be an exact Symbol or Dummy")
    payload = variable._value
    if type(payload) is not _native.Expr:
        raise TypeError("root variable requires an exact native symbol payload")
    binding = payload.symbol_binding
    if binding is None:
        raise TypeError("root variable payload is not a native symbol")
    return binding


def _bracket_statement(expression, variable, binding, lower, upper, tolerance, budget):
    """Capture one exact statement for both candidate search and saved replay."""
    lower, upper, tolerance = (
        _rational(value, budget) for value in (lower, upper, tolerance)
    )
    _charge_exact_work(budget, lower, upper, tolerance, operations=5)
    if lower > upper:
        raise ValueError("root bracket endpoints must be increasing")
    if tolerance <= 0:
        raise ValueError("certified root tolerance must be positive")
    assumptions = bound_assumptions([variable], [binding])
    _check_assumptions([(lower, upper)], assumptions)
    if type(expression) is bool:
        raise TypeError("certified root source requires an exact mathematical expression")
    inexact = [False]
    source = _numeric_source(expression, _functions(), [0], inexact=inexact)
    _check_evaluation_budget(budget)
    if inexact[0]:
        raise ValueError("certified root source requires exact literals; replace Float with explicit Rational")
    # Keep the closed Abs identity when a receipt lifts its source back to
    # Python, even if an opaque function named 'abs' exists in this process.
    source = _preserve_source_function_tags([source])[0]
    _check_evaluation_budget(budget)
    return (
        source, binding, _native_number(lower, budget), _native_number(upper, budget),
        _native_number(tolerance, budget),
    )


class ScalarRootBracketReceipt:
    """Immutable evidence of at least one root in a bounded closed interval.

    The bounds enclose a root of expression and have width at most
    absolute_tolerance. The source is continuous on the entire original
    interval. The midpoint lies within error_bound of at least one
    enclosed root; this is a distance bound, not a residual bound.
    The variable retains the immutable native symbol identity.
    """

    __slots__ = ("_receipt",)

    def __init__(self, receipt):
        if hasattr(self, "_receipt"):
            raise AttributeError("accepted scalar root receipts are immutable")
        if type(receipt) is not _native.ScalarRootBracketReceipt:
            raise TypeError("an accepted native scalar root bracket is required")
        object.__setattr__(self, "_receipt", receipt)

    def __setattr__(self, name, value):
        raise AttributeError("accepted scalar root receipts are immutable")

    def __delattr__(self, name):
        raise AttributeError("accepted scalar root receipts are immutable")

    def to_bytes(self, *, budget=None):
        """Export a canonical source-bound existence bracket, at most 1 MiB.

        The bytes bind the original held expression, typed variable, complete
        original interval, tolerance and default ambient context. Only an exact
        candidate bracket is stored: no accepted flag, cached endpoint values,
        uniqueness or completeness claim. Export uses one supplied cumulative
        EvaluationBudget, or one private bounded ledger when omitted.
        """
        from ._bracket_certificate import export_receipt

        return export_receipt(self, budget=budget)

    @staticmethod
    def from_bytes(data, *, expression, limits, tolerance=Fraction(1, 10000), budget=None):
        """Independently verify a saved bracket against the receiver's source.

        Exact built-in bytes, at most 1 MiB, are admitted before expected-source
        preparation. The exact held expression, typed variable, original bounds
        and requested tolerance must match; equivalent algebra or equal printed
        symbol names do not substitute for identity. Only the default empty
        ambient context is supported, retaining declared symbol facts.

        Replay checks continuity over the entire original interval, containment,
        width and endpoint evidence without running candidate search. It grants
        existence only. One cumulative EvaluationBudget funds exact preparation,
        decoding, independent verification and final publication; cancellation
        or exhaustion never returns a partial receipt.
        """
        from ._bracket_certificate import replay_sources

        return replay_sources(
            data, expression=expression, limits=limits, tolerance=tolerance, budget=budget,
        )

    @property
    def expression(self):
        return _wrap(self._receipt.expression)

    @property
    def variable(self):
        return self._receipt.variable

    @property
    def interval(self):
        return tuple(_wrap(value) for value in self._receipt.interval)

    @property
    def bounds(self):
        return tuple(_wrap(value) for value in self._receipt.bounds)

    @property
    def midpoint(self):
        return _wrap(self._receipt.midpoint)

    @property
    def error_bound(self):
        return _wrap(self._receipt.error_bound)

    @property
    def absolute_tolerance(self):
        return _wrap(self._receipt.absolute_tolerance)

    @property
    def precision_digits(self):
        return self._receipt.precision_digits

    @property
    def endpoint_values(self):
        """Certified value intervals at the accepted bracket's two endpoints."""
        return tuple(tuple(_wrap(value) for value in bounds)
                     for bounds in self._receipt.endpoint_values)

    @property
    def claim(self):
        return "at least one real root in the certified interval"

    def __repr__(self):
        return (f"ScalarRootBracketReceipt(bounds={self.bounds!r}, "
                f"interval={self.interval!r}, error_bound={self.error_bound!r})")


def certified_root_bracket(expression, limits, *, tolerance=Fraction(1, 10000),
                           precision=32, max_bisections=128, budget=None,
                           verification_budget=None):
    """Certify at least one root using continuity and exact endpoint signs.

    The limits are an exact tuple/list (x, lower, upper) with a Symbol or
    Dummy and ordered rational endpoints. Exact int, Rational and Fraction
    values are accepted; Float, bool and custom conversion hooks are refused.
    The tolerance is the maximum complete width of the returned bracket.

    The original expression must be continuous throughout the original
    requested interval, with no undeclared parameters or hidden poles.
    Closed exact arithmetic and supported real elementary functions are
    admitted on verified domains. Abs corners are allowed. Piecewise guards
    must be stable on the original interval. The checker evaluates the
    original source independently of the search.

    An exactly zero endpoint or opposite certified endpoint signs proves
    existence. Midpoint bisection and bounded quarter-point refinement
    narrow the interval. Uncertain signs, unsupported domains, exhausted
    refinement or a same-sign interval with no exact endpoint root raise
    ValueError. An even-multiplicity interior root need not produce a sign
    change and is not generally discoverable by this method.

    Precision ranges from 8 through 128 decimal digits; max_bisections ranges
    from 0 through 1024. A zero limit can accept an already adequate bracket
    or an exact endpoint root. Distinct optional EvaluationBudgets bound
    generation and independent verification, respectively. Passing the same
    budget for both phases shares cumulative limits. Exhaustion/cancellation
    preserves EvaluationBudgetExceeded/EvaluationCancelled and publishes no
    partial receipt. Each omitted phase budget is captured once before exact
    input preparation; preparation consumes the generation ledger. Source
    capture is structurally bounded and Python allocation overhead remains
    outside native-buffer accounting. Both phase budgets are checked again
    after constructing the final public receipt.

    This evidence type is separate from RootBoxReceipt: it carries no
    uniqueness, nonzero-derivative, multiplicity or complete-enumeration claim.

    >>> x = Symbol("x", real=True)
    >>> certified_root_bracket((x - 1)**3, (x, 0, 2)).bounds
    (1, 1)
    """
    _check_evaluation_budget(budget)
    _check_evaluation_budget(verification_budget)
    budget = _source_budget(budget)
    verification_budget = _source_budget(verification_budget)
    variable, lower, upper = _sequence(limits, 3, "root bracket limits")
    binding = _root_binding(variable)
    if type(precision) is not int or not 8 <= precision <= 128:
        raise ValueError("certified precision must be an integer from 8 through 128")
    if type(max_bisections) is not int or not 0 <= max_bisections <= 1024:
        raise ValueError("max_bisections must be an integer from 0 through 1024")
    statement = _bracket_statement(
        expression, variable, binding, lower, upper, tolerance, budget,
    )
    _check_evaluation_budget(budget)
    _check_evaluation_budget(verification_budget)
    receipt = _native.certified_scalar_root_bracket_native(
        *statement, precision, max_bisections,
        budget=budget, verification_budget=verification_budget,
    )
    result = ScalarRootBracketReceipt(receipt)
    _check_evaluation_budget(budget)
    _check_evaluation_budget(verification_budget)
    return result
