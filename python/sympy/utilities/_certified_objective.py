"""Exact adapters for original-source objective and stationary-point checks.

The native second-order interval checker owns acceptance. An exact inverse
of the midpoint Hessian is only a witness proposal, with no proof authority.
"""

from ..core import _native, _wrap
from .lambdify import _check_evaluation_budget
from ._certified_numeric import (
    _MAX_VARIABLES, _bounds, _bounded_inverse, _call_budgeted, _charge_exact_work,
    _check_assumptions, _native_box, _native_number, _rational, _sequence,
)

__all__ = ["ObjectiveEnclosure", "StationaryPointReceipt"]


def _interval(bounds):
    return tuple(_wrap(value) for value in bounds)


class _ObjectiveReceipt:
    __slots__ = ("_receipt",)

    def __setattr__(self, name, value):
        raise AttributeError("accepted objective receipts are immutable")

    def __delattr__(self, name):
        raise AttributeError("accepted objective receipts are immutable")

    @property
    def objective(self):
        """The exact original source bound to the native accepted receipt."""
        return _wrap(self._receipt.objective)

    @property
    def variables(self):
        """Native SymbolBinding handles in the accepted coordinate order."""
        return tuple(self._receipt.variables)

    @property
    def domain(self):
        return _bounds(self._receipt.domain)

    @property
    def precision_digits(self):
        return self._receipt.precision_digits


class ObjectiveEnclosure(_ObjectiveReceipt):
    """Full-box value, gradient and Hessian bounds on an admitted C2 domain.

    These bounds do not assert stationarity, convexity or a global optimum.
    Only the native original-source checker can construct the wrapped receipt.
    """
    __slots__ = ()

    def __init__(self, receipt):
        if hasattr(self, "_receipt"):
            raise AttributeError("accepted objective receipts are immutable")
        if type(receipt) is not _native.ObjectiveEnclosure:
            raise TypeError("an accepted native objective enclosure is required")
        object.__setattr__(self, "_receipt", receipt)

    @property
    def value(self):
        return _interval(self._receipt.value)

    @property
    def gradient(self):
        return _bounds(self._receipt.gradient)

    @property
    def hessian(self):
        return tuple(_bounds(row) for row in self._receipt.hessian)

    def __repr__(self):
        return (f"ObjectiveEnclosure(value={self.value!r}, gradient={self.gradient!r}, "
                f"hessian={self.hessian!r}, domain={self.domain!r})")


class StationaryPointReceipt(_ObjectiveReceipt):
    """One unique nonsingular stationary point in a declared closed box.

    ``kind`` is ``strict_minimum``, ``strict_maximum``, ``saddle`` or
    ``unclassified``. Classification is local; no global optimum or coverage
    outside this box is asserted. ``unclassified`` means the sufficient
    interval curvature test was inconclusive, not that an extremum is absent.
    """
    __slots__ = ()

    def __init__(self, receipt):
        if hasattr(self, "_receipt"):
            raise AttributeError("accepted objective receipts are immutable")
        if type(receipt) is not _native.StationaryPointReceipt:
            raise TypeError("an accepted native stationary-point receipt is required")
        object.__setattr__(self, "_receipt", receipt)

    @property
    def center(self):
        return tuple(_wrap(value) for value in self._receipt.center)

    @property
    def preconditioner(self):
        return tuple(tuple(_wrap(value) for value in row)
                     for row in self._receipt.preconditioner)

    @property
    def stationary_enclosure(self):
        return _bounds(self._receipt.stationary_enclosure)

    @property
    def objective_enclosure(self):
        """Objective range on the full box, including the stationary value."""
        return _interval(self._receipt.objective_enclosure)

    @property
    def gradient_enclosure(self):
        return _bounds(self._receipt.gradient_enclosure)

    @property
    def hessian_enclosure(self):
        return tuple(_bounds(row) for row in self._receipt.hessian_enclosure)

    @property
    def contraction_bound(self):
        return _wrap(self._receipt.contraction_bound)

    @property
    def kind(self):
        return self._receipt.kind

    @property
    def claim(self):
        return "one unique nonsingular stationary point in the declared closed box"

    def __repr__(self):
        return (f"StationaryPointReceipt(stationary_enclosure={self.stationary_enclosure!r}, "
                f"kind={self.kind!r}, domain={self.domain!r}, claim={self.claim!r})")


def _prepare_box(program, box, precision, *, stationary, budget):
    dimension = program.native_objective.variable_count
    if not 1 <= dimension <= _MAX_VARIABLES:
        raise ValueError("certified objectives require 1..16 variables")
    if type(precision) is not int or not 8 <= precision <= 128:
        raise ValueError("certified precision must be an integer from 8 through 128")
    if program._has_inexact_source:
        raise ValueError("certified methods require exact source literals; replace Float with explicit Rational")
    result = []
    for pair in _sequence(box, dimension, "box"):
        lo, hi = (_rational(value, budget)
                  for value in _sequence(pair, 2, "coordinate bounds"))
        _charge_exact_work(budget, lo, hi)
        if lo > hi:
            raise ValueError("box lower bound exceeds upper bound")
        if stationary and lo == hi:
            raise ValueError("stationary-point box coordinates require positive width")
        result.append((lo, hi))
    result = tuple(result)
    _check_assumptions(result, program._assumptions)
    return result


def enclose(program, box, precision, *, budget=None):
    _check_evaluation_budget(budget)
    box = _prepare_box(program, box, precision, stationary=False, budget=budget)
    receipt = _call_budgeted(
        program.native_objective.enclose, _native_box(box, budget), precision, budget=budget,
    )
    result = ObjectiveEnclosure(receipt)
    _check_evaluation_budget(budget)
    return result


def certify_stationary_point(program, box, center, preconditioner, precision, *, budget=None):
    """Propose exact Hessian preconditioning; native source checking decides."""
    _check_evaluation_budget(budget)
    box = _prepare_box(program, box, precision, stationary=True, budget=budget)
    dimension = len(box)
    if center is None:
        _charge_exact_work(budget, *(value for pair in box for value in pair),
                           operations=2 * dimension)
        center = tuple(_rational((lo + hi) / 2, budget) for lo, hi in box)
    else:
        center = tuple(_rational(value, budget)
                       for value in _sequence(center, dimension, "center"))
    for (lo, hi), value in zip(box, center):
        _charge_exact_work(budget, lo, hi, value, operations=2)
        if not lo <= value <= hi:
            raise ValueError("stationary-point center must lie inside the declared box")
    if preconditioner is None:
        point = _call_budgeted(
            program.native_objective.enclose,
            _native_box(tuple((value, value) for value in center), budget),
            precision, budget=budget,
        )
        matrix = []
        for row in point.hessian:
            values = []
            for lo, hi in row:
                lo, hi = _rational(_wrap(lo), budget), _rational(_wrap(hi), budget)
                _charge_exact_work(budget, lo, hi, operations=2)
                values.append(_rational((lo + hi) / 2, budget))
            matrix.append(tuple(values))
        preconditioner = _bounded_inverse(tuple(matrix), budget)
    else:
        preconditioner = tuple(
            tuple(_rational(value, budget)
                  for value in _sequence(row, dimension, "preconditioner row"))
            for row in _sequence(preconditioner, dimension, "preconditioner")
        )
    receipt = _call_budgeted(
        program.native_objective.certify_stationary_point,
        _native_box(box, budget), [_native_number(value, budget) for value in center],
        [[_native_number(value, budget) for value in row] for row in preconditioner],
        precision, budget=budget,
    )
    result = StationaryPointReceipt(receipt)
    _check_evaluation_budget(budget)
    return result
