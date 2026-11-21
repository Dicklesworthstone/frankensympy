"""Source-bound compiled-program exchange with native Session and NDJSON APIs."""

__all__ = ["load_portable_residual_system", "load_portable_objective"]
_MAX_BYTES = 1024 * 1024


def _input(data, budget):
    from .source_certification import _source_budget

    budget = _source_budget(budget)
    if type(data) is not bytes:
        raise TypeError("portable program data must be exact built-in bytes")
    if len(data) > _MAX_BYTES:
        raise ValueError("portable program data exceeds the 1 MiB byte limit")
    return budget


def load_portable_residual_system(data, *, residuals, args, budget=None):
    """Load a Python or native Session residual/Jacobian program candidate.

    ``data`` is canonical UTF-8 bytes from ``program.to_portable_bytes()`` or
    the native ``ExportCompiledResidualSystem`` artifact encoded as UTF-8.
    The 1 MiB format contains the original source, ordered typed axes, sparse
    derivative proofs, bytecode and ambient context. The Python API admits
    only the empty ambient context; typed Symbol assumptions still retain
    their identities. A different context is refused, never silently rebased.

    Both receiver-owned ``residuals`` and ``args`` are mandatory. Their exact
    native syntax, source order and typed identities must match. No names are
    rebound, no Python classes or callbacks are deserialized, and no proofs
    are replaced by fresh symbolic differentiation. Independent admission
    rechecks retained derivatives, omitted zeros and executable provenance.

    The supplied argument nesting and captured assumptions are retained.
    Float-origin restrictions are recaptured from the receiving source, so
    these bytes do not turn approximate literals into certified inputs.
    Numerical execution remains approximate; root certificates are separate.

    Native capture, decoding, proof admission and publication consume one
    cumulative ``EvaluationBudget``. Omission uses a bounded private ledger.
    Failure or cancellation publishes no program and retains consumed work.
    Python capture, bounded structural preflight and CPython overhead remain
    outside native accounting. Legacy ``to_bytes``/``load_residual_system``
    artifacts retain their existing distinct format.
    """
    from .lambdify import (
        _ResidualCallable, _check_evaluation_budget, _native, _residual_source,
    )

    budget = _input(data, budget)
    layout, variables, bindings, assumptions, sources, inexact = _residual_source(residuals, args)
    _check_evaluation_budget(budget)
    native = _native.VerifiedResidualSystem.from_portable_bytes(
        data, residuals=sources, variables=bindings, budget=budget,
    )
    result = _ResidualCallable(layout, native, variables, inexact, assumptions)
    _check_evaluation_budget(budget)
    return result


def load_portable_objective(data, *, objective, args, budget=None):
    """Load a Python or native Session objective/gradient/Hessian candidate.

    Accepts canonical UTF-8 bytes from ``program.to_portable_bytes()`` or the
    native ``ExportCompiledObjective`` artifact encoded as UTF-8, up to 1 MiB.
    The receiver's exact objective, ordered typed variables and empty ambient
    context must match before independent gradient/Hessian and bytecode
    admission. Both mixed-partial orders and omitted derivatives are checked.
    Neither an encoded flag nor a producer's receipt grants acceptance.

    Python argument nesting, assumptions and Float-source restrictions are
    recaptured from the supplied objects. No object classes, callbacks, symbol
    names or Dummy identities are recreated from the artifact. Original source
    domains remain in force on every evaluation; binary64 results remain
    approximate. Stationary-point or extremum certificates are separate.

    One cumulative ``EvaluationBudget`` covers native decoding, independent
    admission and publication; omission uses a bounded private ledger.
    Failure or cancellation publishes no program and retains consumed work.
    Python source capture, bounded bridge structural preflight and CPython
    overhead remain outside native accounting. Existing ``to_bytes`` and
    ``load_objective`` retain their context-free candidate format unchanged.
    """
    from .lambdify import _check_evaluation_budget
    from .objective import _ObjectiveCallable, _native, _objective_source

    budget = _input(data, budget)
    layout, variables, bindings, assumptions, source, inexact = _objective_source(objective, args)
    _check_evaluation_budget(budget)
    native = _native.VerifiedObjective.from_portable_bytes(
        data, source=source, variables=bindings, budget=budget,
    )
    result = _ObjectiveCallable(layout, native, variables, inexact, assumptions)
    _check_evaluation_budget(budget)
    return result
