"""Replay the historical numerical ``Abs`` lowering without aliasing sources.

Old Python numerical programs stored the registered Abs class as Function
("abs", ...). New programs retain Function("Abs", ...). The native source
checker remains exact: this adapter first admits the original artifact against
the one historical lowering of the receiver's own closed source, then replays
its retained evidence again against the canonical source. No serialized bytes
are parsed or rewritten in Python until native admission has succeeded.
"""

from ..core import _native

_RESIDUAL_MISMATCH = "program candidate does not match the caller's expected exact source"
_SOURCE_MISMATCH = "program candidate does not match the caller's ordered residual sources"
_PORTABLE_MISMATCH = "compiled program admission refused: " + _SOURCE_MISMATCH
_ROOT_MISMATCH = "root certificate sources or ordered symbol identities differ from the expected problem"
_OLD_TOKEN = b'"Function":["abs",'
_NEW_TOKEN = b'"Function":["Abs",'


def _legacy_sources(sources, budget):
    """Copy only admitted native syntax; preserve order, holds and typed atoms."""
    from .lambdify import _charge

    count = [0]

    def visit(source, depth):
        _charge(count, depth)
        # Native admission bounds names/literals to 4096 bytes/bits; the depth
        # multiplier conservatively accounts for repeated ancestor copies.
        # Bounded .args ingress snapshots follow the existing Python source-
        # capture boundary, outside a preallocation guarantee. Reconstruction
        # and final byte copies share this ledger, with no reset on refusal.
        copies = 2*depth + 3
        budget._charge(copies, memory_bytes=copies*8192, allocations=4*copies)
        if source.is_number or source.is_symbol:
            return source, False
        tag = source.func_name
        children = [visit(child, depth + 1) for child in source.args]
        changed = tag == "Abs" or any(changed for _, changed in children)
        if not changed:
            return source, False
        arguments = [child for child, _ in children]
        if source.is_add:
            result = _native.Add(*arguments, evaluate=False).as_expr()
        elif source.is_mul:
            result = _native.Mul(*arguments, evaluate=False).as_expr()
        elif source.is_pow:
            result = _native.Pow(*arguments, evaluate=False).as_expr()
        else:
            result = _native.py_function("abs" if tag == "Abs" else tag, *arguments)
        return result, True

    result = []
    changed = False
    for source in sources:
        if type(source) is not _native.Expr:
            raise TypeError("legacy transport requires admitted exact native sources")
        # Borrowed native bounds precede all argument/name copies. The shared
        # shell count also bounds the aggregate source list during traversal.
        _native.CompiledFunction._validate_source(source)
        value, altered = visit(source, 0)
        result.append(value)
        changed |= altered
    budget._checkpoint()
    return result if changed else None


def _canonical_bytes(admitted_bytes, budget):
    """Retag only typed Expr tokens in a trusted canonical native export.

    Quoted symbol names, rule text and other strings contain escaped quotes,
    so they cannot match this complete enum-variant prefix. The replacement
    preserves length, all operands, scalar variants, identities and evidence.
    A second independent native replay must accept the result before use.
    """
    budget._charge(len(admitted_bytes) + 1,
                   memory_bytes=len(admitted_bytes), allocations=1)
    result = admitted_bytes.replace(_OLD_TOKEN, _NEW_TOKEN)
    budget._checkpoint()
    return result


def load_program(data, sources, variables, budget, *, objective=False, portable=False):
    """Admit canonical input directly, or migrate one exact historical profile."""
    native_type = _native.VerifiedObjective if objective else _native.VerifiedResidualSystem
    loader = native_type.from_portable_bytes if portable else native_type.from_bytes
    mismatch = (_PORTABLE_MISMATCH if portable else
                _SOURCE_MISMATCH if objective else _RESIDUAL_MISMATCH)

    def load(encoded, expected):
        if objective:
            return loader(encoded, source=expected[0], variables=variables, budget=budget)
        return loader(encoded, residuals=expected, variables=variables, budget=budget)

    try:
        return load(data, sources)
    except ValueError as error:
        # These static native errors mean source binding stopped before proof
        # replay. No malformed encoding, context, proof or operational failure
        # chooses another path. Preserve the actual exception on no migration.
        if type(error) is not ValueError or error.args != (mismatch,):
            raise
        legacy = _legacy_sources(sources, budget)
        if legacy is None:
            raise
        admitted = load(data, legacy)
        encoded = (admitted.to_portable_bytes(budget=budget) if portable
                   else admitted.to_bytes(budget=budget))
        return load(_canonical_bytes(encoded, budget), sources)


def replay_root(program, data, sources, budget):
    """Return a receipt bound to the actual canonical receiving native program."""
    try:
        return program.replay_root_certificate(data, budget=budget)
    except ValueError as error:
        if type(error) is not ValueError or error.args != (_ROOT_MISMATCH,):
            raise
        legacy = _legacy_sources(program.residuals if sources is None else sources, budget)
        if legacy is None:
            raise
        historical = _native.SourceResidualSystem(legacy, program.variables)
        admitted = historical.replay_root_certificate(data, budget=budget)
        encoded = admitted.to_bytes(budget=budget)
        return program.replay_root_certificate(_canonical_bytes(encoded, budget), budget=budget)
