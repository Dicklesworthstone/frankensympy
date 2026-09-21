"""Structural expression representation printer."""

from __future__ import annotations

from typing import Any


def srepr(expr: Any) -> str:
    """Return structural representation matching SymPy srepr."""
    from sympy.core import (
        Add,
        Basic,
        Derivative,
        Dummy,
        Function,
        Float,
        Integer,
        Mul,
        Pow,
        Rational,
        Symbol,
    )

    # Dummy must be checked BEFORE the inherited Symbol._srepr hook, which
    # would otherwise report it as a plain Symbol and break round-tripping
    # (finding 10). Oracle format: Dummy('d', dummy_index=<n>).
    if type(expr).__name__ == "Dummy":
        return f"Dummy({expr.name!r}, dummy_index={expr.dummy_index})"
    if hasattr(expr, "_srepr"):
        return expr._srepr()

    if isinstance(expr, Basic):
        cls_name = type(expr).__name__
        if type(expr) is Symbol:
            return expr._srepr()
        if type(expr) is Dummy:
            return f"Dummy({expr.name!r})"
        if type(expr) is Integer:
            return f"Integer({expr.p})"
        if type(expr) is Float:
            return expr._srepr()
        if type(expr) is Rational:
            return f"Rational({expr.p}, {expr.q})"
        if str(expr) in ("pi", "E", "I", "oo", "zoo", "nan"):
            return str(expr)
        args = expr.args
        if type(expr) is Add:
            # Oracle: srepr orders ALL Adds canonically (monomial-lex,
            # constants last ascending) - held Adds too (oracle:
            # srepr(Add(y, x, evaluate=False)) == Add(x, y)).
            from sympy.core import _add_ordered_terms
            args = _add_ordered_terms(expr)
        elif type(expr) is Mul and hasattr(expr, "_args"):
            # Held Mul: oracle srepr sorts args by sort_key, numbers first
            # (Mul(x, -3, y, evaluate=False) -> Mul(-3, x, y)).
            args = tuple(sorted(expr.args, key=lambda a: a.sort_key()))
        elif type(expr) is Mul and expr.args:
            # Evaluated Mul with a negative leading coefficient: oracle
            # srepr splits it - Mul(-2, w) -> Mul(-1, 2, w) - so the
            # printed structure re-evaluates to the same product.
            first = expr.args[0]
            if isinstance(first, (Integer, Rational)) and first.p < 0:
                parts: list = [Integer(-1)]
                if first.q != 1 or first.p != -1:
                    parts.append(Rational(-first.p, first.q) if first.q != 1 else Integer(-first.p))
                parts.extend(expr.args[1:])
                args = tuple(parts)
        args_s = ", ".join(srepr(a) for a in args)
        return f"{cls_name}({args_s})"

    return repr(expr)
