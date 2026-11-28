"""Uniform original-source root claims over exact parameter boxes.

A midpoint Jacobian inverse only proposes a witness. The independent native
checker retains the full parameter intervals in both the centered residuals
and the unknown Jacobian; only its accepted receipt authorizes publication.
"""

from fractions import Fraction
import math

from ..core import _native, _wrap
from ._certified_numeric import (
    _Receipt, _bounds, _bounded_inverse, _call_budgeted, _charge_exact_work,
    _check_assumptions, _native_box, _native_number, _prepare_box,
    _prepare_root_source, _rational, _sequence,
)
from .lambdify import _check_evaluation_budget

__all__ = ["ParametricRootBoxReceipt"]


class ParametricRootBoxReceipt(_Receipt):
    """Exactly one nonsingular unknown root for every declared parameter value.

    root_enclosure contains the union of these parameter-dependent roots.
    This is a claim within unknown_domain, with no exact symbolic solution
    function, continuation-path claim, or assertion outside that domain.
    """
    __slots__ = ()

    def __init__(self, receipt):
        if hasattr(self, "_receipt"):
            raise AttributeError("accepted numerical receipts are immutable")
        if type(receipt) is not _native.ParametricRootBoxReceipt:
            raise TypeError("an accepted native parametric root-box receipt is required")
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
    def center(self):
        return tuple(_wrap(value) for value in self._receipt.center)

    @property
    def preconditioner(self):
        return tuple(tuple(_wrap(value) for value in row) for row in self._receipt.preconditioner)

    @property
    def root_enclosure(self):
        return _bounds(self._receipt.root_enclosure)

    @property
    def contraction_norm(self):
        """The infinity norm, or the infinity norm scaled by the unknown box radii."""
        return self._receipt.contraction_norm

    @property
    def contraction_bound(self):
        return _wrap(self._receipt.contraction_bound)

    def enclose_sensitivity(self, *, budget=None):
        """Bound dx_i/dp_j throughout this accepted parameter box.

        Rows follow unknown_variables and columns follow parameters. The
        checker differentiates the original source on the complete domain;
        the result is an enclosure, with no requested-accuracy promise.
        """
        from ._parametric_sensitivity import enclose_sensitivity
        return enclose_sensitivity(self._receipt, budget=budget)

    @property
    def claim(self):
        return ("for every parameter value in the declared parameter box, "
                "one unique nonsingular real root in the declared unknown box")

    def __repr__(self):
        return (f"ParametricRootBoxReceipt(root_enclosure={self.root_enclosure!r}, "
                f"unknown_domain={self.unknown_domain!r}, "
                f"parameter_domain={self.parameter_domain!r}, claim={self.claim!r})")


def _fallback_budget():
    """One private ledger, using the native certification fallback envelope."""
    limit = 512 * 1024 * 1024
    return _native.EvaluationBudget(
        max_steps=limit, max_memory_bytes=limit, max_allocations=limit,
    )


def _prepare_parametric_domains(program, unknown_box, parameter_box, precision, *, budget):
    """Capture both exact domains in the compiled unknown/parameter order."""
    dimension = program.native_system.residual_count
    total = program.native_system.variable_count
    if not 1 <= dimension < total <= 16:
        raise ValueError(
            "parametric roots require one unknown per residual, at least one parameter, "
            "and at most 16 total variables"
        )
    unknown_box = _sequence(unknown_box, dimension, "unknown box")
    parameter_box = _sequence(parameter_box, total - dimension, "parameter box")
    product = _prepare_box(
        program, unknown_box + parameter_box, precision, square=False, budget=budget,
    )
    unknown_box, parameter_box = product[:dimension], product[dimension:]
    if any(lo == hi for lo, hi in unknown_box):
        raise ValueError("parametric unknown coordinates require positive width")
    return unknown_box, parameter_box


def solve_parametric_certified(
        program, initial, parameter_box, unknown_radii, tol, maxiter,
        max_step_norm, precision, *, budget=None):
    """Propose one exact unknown box, then check the unchanged whole family."""
    from ..solvers.numerical import RootSearchError
    from .lambdify import NumericParametricRootCandidate

    _check_evaluation_budget(budget)
    if budget is None:
        budget = _fallback_budget()
    _check_evaluation_budget(budget)
    dimension = program.native_system.residual_count
    total = program.native_system.variable_count
    if not 1 <= dimension < total <= 16:
        raise ValueError(
            "parametric roots require one unknown per residual, at least one parameter, "
            "and at most 16 total variables"
        )
    if type(precision) is not int or not 8 <= precision <= 128:
        raise ValueError("certified precision must be an integer from 8 through 128")
    _prepare_root_source(program)
    # The supported facts are real/sign predicates. A positive-width real
    # interval always contains either a positive or a negative subinterval,
    # so these representatives decide static feasibility without constraining
    # the numerical starting guess or proposing the final certificate box.
    for facts in program._assumptions[:dimension]:
        try:
            _check_assumptions(((1, 2),), (facts,))
        except ValueError:
            try:
                _check_assumptions(((-2, -1),), (facts,))
            except ValueError:
                raise ValueError(
                    "captured unknown assumptions do not admit a positive-width real box"
                ) from None
    radii = tuple(_rational(value, budget) for value in
                  _sequence(unknown_radii, dimension, "unknown radii"))
    _charge_exact_work(budget, *radii, operations=dimension)
    if any(radius <= 0 for radius in radii):
        raise ValueError("parametric unknown radii must be positive exact rational values")
    parameters = []
    for pair in _sequence(parameter_box, total - dimension, "parameter box"):
        lo, hi = (_rational(value, budget) for value in _sequence(pair, 2, "parameter bounds"))
        _charge_exact_work(budget, lo, hi)
        if lo > hi:
            raise ValueError("parameter box lower bound exceeds upper bound")
        parameters.append((lo, hi))
    parameter_box = tuple(parameters)
    # Parameter assumptions apply to the complete fixed domain before any
    # floating work. Unknown assumptions apply to the eventual whole box,
    # never to an arbitrary initial guess used only by the numerical proposer.
    _check_assumptions(parameter_box, program._assumptions[dimension:])
    _charge_exact_work(
        budget, *(value for pair in parameter_box for value in pair),
        operations=2 * len(parameter_box),
    )
    midpoints = tuple(_rational((lo + hi) / 2, budget) for lo, hi in parameter_box)
    _charge_exact_work(budget, *midpoints, operations=len(midpoints))
    try:
        approximate_parameters = tuple(float(value) for value in midpoints)
    except OverflowError:
        raise ValueError("parameter midpoints must have finite binary64 approximations") from None
    if any(not math.isfinite(value) for value in approximate_parameters):
        raise ValueError("parameter midpoints must have finite binary64 approximations")
    _check_evaluation_budget(budget)
    try:
        candidate = program.solve_parametric_candidate(
            initial, approximate_parameters, tol=tol, maxiter=maxiter,
            max_step_norm=max_step_norm, budget=budget,
        )
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        raise RootSearchError(
            "candidate", f"certified parametric solve could not generate a candidate: {error}",
        ) from error
    _check_evaluation_budget(budget)
    if (type(candidate) is not NumericParametricRootCandidate
            or candidate.native_system is not program.native_system):
        raise TypeError("a source-bound native parametric root candidate is required")
    values = _sequence(candidate.root, dimension, "parametric root candidate")
    if any(type(value) is not float or not math.isfinite(value) for value in values):
        raise ValueError("parametric root candidate coordinates must be finite binary64 values")
    # Every binary64 coordinate is retained exactly as a rational witness.
    # Neither this conversion nor a small residual grants a root claim.
    _charge_exact_work(budget, bits=1096, operations=4 * dimension)
    center = tuple(Fraction(*value.as_integer_ratio()) for value in values)
    _charge_exact_work(budget, *center, *radii, operations=2 * dimension)
    try:
        unknown_box = tuple(
            (_rational(value - radius, budget), _rational(value + radius, budget))
            for value, radius in zip(center, radii)
        )
        result = certify_parametric_root_box(
            program, unknown_box, parameter_box, center, None, precision, budget=budget,
        )
    except ValueError as error:
        raise RootSearchError(
            "certification", f"the proposed parametric root box was not accepted: {error}",
            failures=(str(error),),
        ) from error
    _check_evaluation_budget(budget)
    return result


def certify_parametric_root_box(
        program, unknown_box, parameter_box, center, preconditioner, precision, *, budget=None):
    """Prepare a bounded proposal, then independently check the entire family."""
    _check_evaluation_budget(budget)
    if budget is None:
        # Proposal and verification must not receive separate default budgets.
        # This ledger also covers exact preparation and final publication.
        budget = _fallback_budget()
    _check_evaluation_budget(budget)
    unknown_box, parameter_box = _prepare_parametric_domains(
        program, unknown_box, parameter_box, precision, budget=budget,
    )
    dimension = len(unknown_box)

    if center is None:
        _charge_exact_work(
            budget, *(value for pair in unknown_box for value in pair), operations=2 * dimension,
        )
        center = tuple(_rational((lo + hi) / 2, budget) for lo, hi in unknown_box)
    else:
        center = tuple(_rational(value, budget) for value in _sequence(center, dimension, "center"))
    _charge_exact_work(
        budget, *center, *(value for pair in unknown_box for value in pair), operations=2 * dimension,
    )
    if any(not lo <= value <= hi for (lo, hi), value in zip(unknown_box, center)):
        raise ValueError("parametric root center must lie inside the declared unknown box")

    if preconditioner is None:
        _charge_exact_work(
            budget, *(value for pair in parameter_box for value in pair),
            operations=2 * len(parameter_box),
        )
        parameter_midpoints = tuple(_rational((lo + hi) / 2, budget) for lo, hi in parameter_box)
        # A parameter midpoint is used only to propose C. The uniform check
        # below receives the full original parameter box, never this point.
        point_box = tuple((value, value) for value in center + parameter_midpoints)
        point = _call_budgeted(
            program.native_system.enclose, _native_box(point_box, budget), precision, budget=budget,
        )
        if type(point) is not _native.ResidualJacobianEnclosure:
            raise TypeError("an accepted native residual enclosure is required for the proposal")
        matrix = []
        for row in point.jacobian:
            values = []
            # Parameter derivative columns do not belong to the square map on X.
            for lo, hi in row[:dimension]:
                lo, hi = _rational(_wrap(lo), budget), _rational(_wrap(hi), budget)
                _charge_exact_work(budget, lo, hi, operations=2)
                values.append((lo + hi) / 2)
            matrix.append(tuple(values))
        preconditioner = _bounded_inverse(tuple(matrix), budget)
    else:
        preconditioner = tuple(
            tuple(_rational(value, budget) for value in _sequence(row, dimension, "preconditioner row"))
            for row in _sequence(preconditioner, dimension, "preconditioner")
        )
    receipt = _call_budgeted(
        program.native_system.certify_parametric_root_box,
        _native_box(unknown_box, budget), _native_box(parameter_box, budget),
        [_native_number(value, budget) for value in center],
        [[_native_number(value, budget) for value in row] for row in preconditioner],
        precision, budget=budget,
    )
    result = ParametricRootBoxReceipt(receipt)
    _check_evaluation_budget(budget)
    return result
