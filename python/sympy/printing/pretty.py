"""ASCII pretty printer for the compatibility shell (arithmetic core).

Renders Integer, Rational, Symbol, Add, Mul, and Pow exactly as pinned
SymPy 1.14.0's pretty printer does for the corpus grammar. Multi-line for
Pow (exponent above base, or fraction bar for negative exponents).
Unsupported expression classes raise NotImplementedError.
"""

from __future__ import annotations

from typing import Any

from ..core import Add, ComplexInfinity, Function, Integer, Mul, Pow, Rational, Symbol


def pretty(expr: Any) -> str:
    """Return the pinned-SymPy-1.14.0 ASCII rendering of ``expr``."""
    lines, _bl = _render(expr)
    # Pad all lines to the expression width.
    width = max(len(line) for line in lines) if lines else 0
    return "\n".join(line.ljust(width) for line in lines)


def _render(expr: Any) -> tuple[list[str], int]:
    """Return (lines, baseline_index). Baseline is the visual center row."""
    if isinstance(expr, Integer):
        return [str(expr.p)], 0
    if isinstance(expr, Rational):
        return [f"{expr.p}/{expr.q}"], 0
    if isinstance(expr, Symbol):
        return [expr.name], 0
    if isinstance(expr, Pow):
        return _pretty_pow(expr)
    if isinstance(expr, ComplexInfinity):
        return ["zoo"], 0
    if isinstance(expr, Function):
        name = type(expr).__name__
        args = ",".join(_render(arg)[0][0] for arg in expr.args)
        return [f"{name}({args})"], 0
    if isinstance(expr, Mul):
        return _pretty_mul(expr)
    if isinstance(expr, Add):
        return _pretty_add(expr)
    raise NotImplementedError(
        f"pretty printing is not implemented for {type(expr).__name__}"
    )


def _pretty_pow(expr: Pow) -> tuple[list[str], int]:
    base, exp = expr.args
    base_lines, base_bl = _render(base)
    base_w = max(len(line) for line in base_lines)
    exp_int = _as_int(exp)
    if exp_int is not None and exp_int >= 0:
        exp_line = str(exp_int).rjust(base_w + 1)
        base_padded = [line.ljust(base_w + 1) for line in base_lines]
        return [exp_line] + base_padded, base_bl + 1
    if exp_int is not None and exp_int < 0:
        n = -exp_int
        if n == 1:
            num_lines = ["1"]
        else:
            num_lines = [str(n)]
        num_w = max(len(line) for line in num_lines)
        den_w = max(len(line) for line in base_lines)
        bar_w = max(num_w, den_w)
        num_padded = [line.ljust(bar_w) for line in num_lines]
        bar = "─" * bar_w
        den_padded = [line.ljust(bar_w) for line in base_lines]
        return num_padded + [bar] + den_padded, len(num_lines)
    raise NotImplementedError("non-integer exponent in Pow pretty printing")


def _pretty_mul(expr: Mul) -> tuple[list[str], int]:
    # Check for Mul(-1, x) pattern: absorb into leading minus.
    if len(expr.args) == 2 and isinstance(expr.args[0], Integer) and expr.args[0].p == -1:
        inner_lines, inner_bl = _render(expr.args[1])
        return ["-" + line for line in inner_lines], inner_bl
    operand_strs = []
    for factor in expr.args:
        lines, _bl = _render(factor)
        operand_strs.append("\n".join(lines))
    line = "⋅".join(operand_strs)
    if "\n" in line:
        lines = line.split("\n")
        return lines, 0
    return [line], 0


def _pretty_add(expr: Add) -> tuple[list[str], int]:
    # Oracle print order: non-numeric terms by degree descending (ties:
    # sympy canonical stored order), numeric terms last.
    terms = list(expr.args)
    numeric = [t for t in terms if isinstance(t, (Integer, Rational))]
    non_numeric = [t for t in terms if not isinstance(t, (Integer, Rational))]
    # non_numeric keeps stored order (already degree-grouped by the shell's
    # canonical Add construction).
    def total_degree(term: Any) -> int:
        if isinstance(term, (Integer, Rational)):
            return 0
        if isinstance(term, Symbol):
            return 1
        if isinstance(term, Pow):
            base_deg = total_degree(term.args[0])
            exp_int = _as_int(term.args[1])
            return base_deg * exp_int if exp_int is not None and exp_int >= 0 else base_deg
        if isinstance(term, Mul):
            return sum(total_degree(f) for f in term.args)
        return 1
    non_numeric.sort(key=lambda t: -total_degree(t))
    ordered = non_numeric + numeric

    rendered_parts: list[tuple[list[str], int, bool, bool]] = []
    for i, term in enumerate(ordered):
        lines, bl = _render(term)
        neg = _is_negative(term)
        if neg:
            lines = list(lines)
            lines[bl] = lines[bl].lstrip("-")
        rendered_parts.append((lines, bl, neg, i == 0))

    max_bl = max(bl for _, bl, _, _ in rendered_parts)
    height = max(len(lines) for lines, _, _, _ in rendered_parts)
    padded = []
    for lines, bl, _neg, _first in rendered_parts:
        above = [""] * max(max_bl - bl, 0)
        below = [""] * max(height - len(lines) - max(max_bl - bl, 0), 0)
        padded.append(above + lines + below)

    result = []
    for row in range(height):
        line = ""
        for j, (lines, bl, neg, first) in enumerate(rendered_parts):
            piece = padded[j][row] if row < len(padded[j]) else ""
            if row == max_bl and j > 0:
                line += " - " if neg else " + "
            line += piece
        result.append(line)
    return result, max_bl


def _as_int(expr: Any) -> int | None:
    if isinstance(expr, Integer):
        return expr.p
    return None


def _is_negative(term: Any) -> bool:
    if isinstance(term, (Integer, Rational)):
        return term.p < 0
    if isinstance(term, Mul):
        first = term.args[0]
        if isinstance(first, (Integer, Rational)):
            return first.p < 0
    return False
