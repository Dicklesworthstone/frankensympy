"""Certified proper real integration with exact rational error bounds.

The native generator proposes a bounded adaptive partition. An independent
lower-layer checker evaluates the original source on every closed cell,
including all original domain restrictions, and verifies the requested
absolute enclosure width. No symbolic primitive or sampled error estimate
is required.
"""

from fractions import Fraction

from ..core import Dummy, Symbol, _native, _wrap
from ..utilities._certified_numeric import (
    _charge_exact_work, _check_assumptions, _native_number, _rational, _sequence,
    bound_assumptions,
)
from ..utilities.lambdify import _check_evaluation_budget, _functions, _numeric_source
from ..utilities.source_certification import _preserve_source_function_tags, _source_budget

__all__ = ["certified_integral", "IntegralEnclosureReceipt"]


def _integral_binding(variable):
    """Capture the actual native atom before inspecting mutable shell facts.

    The generic shell binding helper canonicalizes arbitrary fact values.
    A certified API instead checks exact Boolean snapshots against the
    existing immutable payload, without invoking truth/string hooks.
    """
    if type(variable) not in (Symbol, Dummy):
        raise TypeError("integration variable must be an exact Symbol or Dummy")
    payload = variable._value
    if type(payload) is not _native.Expr:
        raise TypeError("integration variable requires an exact native symbol payload")
    binding = payload.symbol_binding
    if binding is None:
        raise TypeError("integration variable payload is not a native symbol")
    return binding


def _integral_statement(integrand, variable, binding, lower, upper, tolerance, budget):
    """Capture the same exact statement for generation and independent replay."""
    lower, upper, tolerance = (
        _rational(value, budget) for value in (lower, upper, tolerance)
    )
    _charge_exact_work(budget, lower, upper, tolerance, operations=5)
    if tolerance <= 0:
        raise ValueError("certified integral tolerance must be positive")
    assumptions = bound_assumptions([variable], [binding])
    _check_assumptions([(min(lower, upper), max(lower, upper))], assumptions)
    if type(integrand) is bool:
        raise TypeError("certified integral source requires an exact mathematical expression")
    inexact = [False]
    source = _numeric_source(integrand, _functions(), [0], inexact=inexact)
    _check_evaluation_budget(budget)
    if inexact[0]:
        raise ValueError("certified integral source requires exact literals; replace Float with explicit Rational")
    # Restore the closed Abs class tag so lifting a receipt cannot resolve a
    # same-named opaque user function. Held arithmetic stays in original order.
    source = _preserve_source_function_tags([source])[0]
    _check_evaluation_budget(budget)
    return (
        source, binding, _native_number(lower, budget), _native_number(upper, budget),
        _native_number(tolerance, budget),
    )


class IntegralEnclosureReceipt:
    """Immutable source-bound integral enclosure with an exact error bound.

    ``bounds`` enclose the proper real integral over ``interval``;
    ``midpoint`` differs from that integral by at most ``error_bound``.
    The complete width is at most ``absolute_tolerance``. ``variable`` is
    an immutable native SymbolBinding, retaining its declared identity.
    """

    __slots__ = ("_receipt",)

    def __init__(self, receipt):
        if hasattr(self, "_receipt"):
            raise AttributeError("accepted integral receipts are immutable")
        if type(receipt) is not _native.IntegralEnclosureReceipt:
            raise TypeError("an accepted native integral enclosure is required")
        object.__setattr__(self, "_receipt", receipt)

    def __setattr__(self, name, value):
        raise AttributeError("accepted integral receipts are immutable")

    def __delattr__(self, name):
        raise AttributeError("accepted integral receipts are immutable")

    def to_bytes(self, *, budget=None):
        """Export a bounded canonical partition for independent verification.

        The bytes bind the original exact source, typed variable, ordered
        endpoints and tolerance. They contain no trusted enclosure or accepted
        flag. Export consumes the supplied cumulative EvaluationBudget.
        """
        from ._certificate import export_receipt

        return export_receipt(self, budget=budget)

    @staticmethod
    def from_bytes(data, *, integrand, limits, tolerance=Fraction(1, 10000), budget=None):
        """Verify a saved partition against the receiver's expected integral.

        Exact built-in bytes, at most 1 MiB, are admitted before expected-source
        preparation. The receiver supplies its own exact integrand, typed axis,
        ordered endpoints and tolerance. Replay checks the complete original
        domain and enclosure width without invoking adaptive generation. A
        shared EvaluationBudget funds preparation, verification and publication;
        cancellation or exhaustion never returns a partial receipt.
        """
        from ._certificate import replay_sources

        return replay_sources(
            data, integrand=integrand, limits=limits, tolerance=tolerance, budget=budget,
        )

    @property
    def integrand(self):
        return _wrap(self._receipt.integrand)

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
    def partition(self):
        return tuple(_wrap(value) for value in self._receipt.partition)

    @property
    def interval_count(self):
        return self._receipt.interval_count

    @property
    def claim(self):
        return "proper real integral enclosed within the requested absolute width"

    def __repr__(self):
        return (f"IntegralEnclosureReceipt(bounds={self.bounds!r}, "
                f"interval={self.interval!r}, error_bound={self.error_bound!r}, "
                f"interval_count={self.interval_count!r})")


def certified_integral(integrand, limits, *, tolerance=Fraction(1, 10000),
                       precision=32, max_intervals=512, budget=None):
    """Enclose a proper real integral without finding an antiderivative.

    ``limits`` is an exact list/tuple ``(x, a, b)`` with a Symbol or Dummy
    and finite exact rational endpoints. Bounds and tolerance admit exact
    int, Rational and fractions.Fraction values, never floating inputs or
    conversion hooks. ``tolerance`` bounds the complete enclosure width,
    so the midpoint's absolute error is at most half that tolerance.

    The supported exact source consists of the declared real variable,
    rational constants, pi/E, arithmetic, real powers on a verified smooth
    domain and the native C2 elementary functions. Continuous Abs compositions
    also admit interior and endpoint corners through independent Lipschitz
    slope bounds, without claiming pointwise derivatives there. Unknown parameters,
    original poles, unstable Piecewise guards, nonreal branches and custom
    symbolic objects refuse. Reversed limits negate the enclosure; equal
    limits return zero only after checking the original source at the point.

    Whole-cell second derivatives refine the signed midpoint remainder,
    retaining exact rational quadratic integrals in one cell. The verifier
    intersects this bound with the independent range and first-derivative
    bounds; a midpoint curvature sample never supplies the remainder. At an
    Abs corner, an absolutely-continuous bound replaces curvature tightening.

    Refinement is deterministic and bounded at 1..1024 intervals, with
    precision 8..128 decimal digits. Failure to establish the requested
    width or source domain raises ValueError and publishes no partial receipt.
    This API is separate from symbolic ``integrate`` and does not assert an
    improper integral or principal value.

    ``budget`` accepts an EvaluationBudget shared with other numerical work.
    Exact input preparation, native proposal and independent verification
    consume one cumulative ledger, including failed attempts. Omitting it
    captures one private bounded ledger for this call. Source capture and
    Python allocation overhead are outside native-buffer accounting. Native
    arithmetic releases the GIL, so another thread can call ``budget.cancel()``.
    Budget exhaustion raises EvaluationBudgetExceeded; cancellation raises
    EvaluationCancelled. Neither is a mathematical domain refusal, and both
    prevent publication of an accepted receipt.
    """
    budget = _source_budget(budget)
    variable, lower, upper = _sequence(limits, 3, "integration limits")
    binding = _integral_binding(variable)
    if type(precision) is not int or not 8 <= precision <= 128:
        raise ValueError("certified precision must be an integer from 8 through 128")
    if type(max_intervals) is not int or not 1 <= max_intervals <= 1024:
        raise ValueError("max_intervals must be an integer from 1 through 1024")
    statement = _integral_statement(
        integrand, variable, binding, lower, upper, tolerance, budget,
    )
    native = _native.certified_integral_native(
        *statement, precision, max_intervals, budget=budget,
    )
    result = IntegralEnclosureReceipt(native)
    _check_evaluation_budget(budget)
    return result
