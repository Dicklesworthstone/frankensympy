"""Portable scalar existence witnesses checked against receiver-owned sources.

Only the shared native codec parses witnesses and grants a root receipt. This
boundary admits exact bounded inputs and preserves declared typed assumptions.
"""

from fractions import Fraction

from ..core import _native
from ..utilities._certified_numeric import _sequence
from ..utilities.lambdify import _check_evaluation_budget
from ..utilities.source_certification import _source_budget
from .certified import ScalarRootBracketReceipt, _bracket_statement, _root_binding

_MAX_BYTES = 1024 * 1024


def _data(data, budget):
    _check_evaluation_budget(budget)
    if type(data) is not bytes:
        raise TypeError("scalar root bracket data must be exact built-in bytes")
    if len(data) > _MAX_BYTES:
        raise ValueError("scalar root bracket data exceeds the 1 MiB byte limit")


def export_receipt(receipt, *, budget=None):
    _check_evaluation_budget(budget)
    if type(receipt) is not ScalarRootBracketReceipt:
        raise TypeError("an exact ScalarRootBracketReceipt is required for bracket export")
    native = receipt._receipt
    if type(native) is not _native.ScalarRootBracketReceipt:
        raise TypeError("an accepted native scalar root bracket is required")
    budget = _source_budget(budget)
    data = native.to_bytes(budget=budget)
    _check_evaluation_budget(budget)
    return data


def replay_sources(data, *, expression, limits, tolerance=Fraction(1, 10000), budget=None):
    _data(data, budget)
    budget = _source_budget(budget)
    variable, lower, upper = _sequence(limits, 3, "root bracket limits")
    binding = _root_binding(variable)
    statement = _bracket_statement(
        expression, variable, binding, lower, upper, tolerance, budget,
    )
    native = _native.replay_scalar_root_bracket_native(data, *statement, budget=budget)
    result = ScalarRootBracketReceipt(native)
    _check_evaluation_budget(budget)
    return result
