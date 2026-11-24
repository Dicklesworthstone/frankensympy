"""Portable integral partitions checked against receiver-owned exact sources.

Only the shared native codec parses witnesses and grants an integral receipt.
The Python boundary retains exact input and declared-assumption admission.
"""

from fractions import Fraction

from ..core import _native
from ..utilities._certified_numeric import _sequence
from ..utilities.lambdify import _check_evaluation_budget
from ..utilities.source_certification import _source_budget
from .certified import IntegralEnclosureReceipt, _integral_binding, _integral_statement

_MAX_BYTES = 1024 * 1024


def _data(data, budget):
    _check_evaluation_budget(budget)
    if type(data) is not bytes:
        raise TypeError("integral certificate data must be exact built-in bytes")
    if len(data) > _MAX_BYTES:
        raise ValueError("integral certificate data exceeds the 1 MiB byte limit")


def export_receipt(receipt, *, budget=None):
    _check_evaluation_budget(budget)
    if type(receipt) is not IntegralEnclosureReceipt:
        raise TypeError("an exact IntegralEnclosureReceipt is required for integral export")
    native = receipt._receipt
    if type(native) is not _native.IntegralEnclosureReceipt:
        raise TypeError("an accepted native integral enclosure is required")
    budget = _source_budget(budget)
    data = native.to_bytes(budget=budget)
    _check_evaluation_budget(budget)
    return data


def replay_sources(data, *, integrand, limits, tolerance=Fraction(1, 10000), budget=None):
    _data(data, budget)
    budget = _source_budget(budget)
    variable, lower, upper = _sequence(limits, 3, "integration limits")
    binding = _integral_binding(variable)
    statement = _integral_statement(
        integrand, variable, binding, lower, upper, tolerance, budget,
    )
    native = _native.replay_integral_certificate_native(data, *statement, budget=budget)
    result = IntegralEnclosureReceipt(native)
    _check_evaluation_budget(budget)
    return result
