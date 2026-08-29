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
        Integer,
        Mul,
        Pow,
        Rational,
        Symbol,
    )

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
        if type(expr) is Rational:
            return f"Rational({expr.p}, {expr.q})"
        if str(expr) in ("pi", "E", "I", "oo", "zoo", "nan"):
            return str(expr)
        args_s = ", ".join(srepr(a) for a in expr.args)
        return f"{cls_name}({args_s})"

    return repr(expr)
