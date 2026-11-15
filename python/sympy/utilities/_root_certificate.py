"""Portable local-root proofs bound to receiver-owned exact source snapshots.

Only the shared native codec parses the bytes and accepts the mathematical
claim. Python retains the public source, declared-assumption and exact-input
admission rules before publishing the immutable shell receipt.
"""

from ..core import _native
from ._certified_numeric import RootBoxReceipt, _bounds, _prepare_box
from .lambdify import _ResidualCallable, _check_evaluation_budget
from .source_certification import _SourceContext, _residual_context, _source_budget

_MAX_BYTES = 1024 * 1024


def _data(data, budget):
    _check_evaluation_budget(budget)
    if type(data) is not bytes:
        raise TypeError("root certificate data must be exact built-in bytes")
    if len(data) > _MAX_BYTES:
        raise ValueError("root certificate data exceeds the 1 MiB byte limit")


def export_receipt(receipt, *, budget=None):
    _check_evaluation_budget(budget)
    if type(receipt) is not RootBoxReceipt:
        raise TypeError("an exact RootBoxReceipt is required for root-proof export")
    native = receipt._receipt
    if type(native) is not _native.RootBoxReceipt:
        raise TypeError("an accepted native root-box receipt is required")
    budget = _source_budget(budget)
    data = native.to_bytes(budget=budget)
    _check_evaluation_budget(budget)
    return data


def replay_program(program, data, *, budget=None):
    _data(data, budget)
    if type(program) not in (_ResidualCallable, _SourceContext):
        raise TypeError("root-proof replay requires a captured exact residual system")
    budget = _source_budget(budget)
    native = program.native_system.replay_root_certificate(data, budget=budget)
    # Root existence is independently checked in native code. Preserve the
    # shell's narrower exact-input and captured-assumption rules as well; an
    # incompatible domain never becomes a returned public receipt.
    _prepare_box(
        program, _bounds(native.domain), native.precision_digits,
        square=True, budget=budget,
    )
    result = RootBoxReceipt(native)
    _check_evaluation_budget(budget)
    return result


def replay_sources(data, *, residuals, args, budget=None):
    _data(data, budget)
    budget = _source_budget(budget)
    program = _residual_context(residuals, args, budget)
    return replay_program(program, data, budget=budget)
