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
        p_str, q_str = str(expr.p), str(expr.q)
        # Oracle-pinned rule (verified against SymPy 1.14.0): a Rational
        # renders as a stacked fraction iff, after reduction, both the
        # numerator and the denominator have at least two digits.
        # 13/11 stacks, 6/17 and 1/2 stay inline.
        if len(p_str) >= 2 and len(q_str) >= 2:
            width = max(len(p_str), len(q_str))
            return [p_str.center(width), "─" * width, q_str.center(width)], 1
        return [f"{expr.p}/{expr.q}"], 0
    if isinstance(expr, Symbol):
        return [expr.name], 0
    if isinstance(expr, Pow):
        return _pretty_pow(expr)
    if isinstance(expr, ComplexInfinity):
        return ["zoo"], 0
    if isinstance(expr, Function):
        name = type(expr).__name__
        # Oracle renders trailing digits in function names as unicode
        # subscripts (AdvLaw1 -> AdvLaw₁) and joins args with ", ".
        subscript_map = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
        stripped = name.rstrip("0123456789")
        sub = name[len(stripped):].translate(subscript_map)
        name = stripped + sub
        args = ", ".join(_render(arg)[0][0] for arg in expr.args)
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
        exp_str = str(exp_int)
        # Oracle pads the exponent block to base_w + len(exp_str) so the
        # exponent's last column sits one past the base's last column
        # (x**32 -> ' 32' over 'x  ', not '32' over 'x ').
        total_w = base_w + len(exp_str)
        exp_line = exp_str.rjust(total_w)
        base_padded = [line.ljust(total_w) for line in base_lines]
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
    blocks = [_render(factor) for factor in expr.args]
    if all(len(lines) == 1 for lines, _bl in blocks):
        line = "⋅".join(lines[0] for lines, _bl in blocks)
        return [line], 0
    # Multi-line product: horizontally join blocks aligned on their
    # baselines; the '⋅' separator appears only on the baseline row.
    # Oracle: Mul(3, x**2) -> '   2' / '3⋅x '.
    max_bl = max(bl for _lines, bl in blocks)
    height = max(len(lines) for lines, _bl in blocks)
    padded: list[list[str]] = []
    for lines, bl in blocks:
        width = max(len(line) for line in lines)
        above = [" " * width] * (max_bl - bl)
        below = [" " * width] * (height - len(lines) - (max_bl - bl))
        padded.append(above + lines + below)
    result = []
    for row in range(height):
        pieces = []
        for j, (_lines, bl) in enumerate(blocks):
            piece = padded[j][row]
            if row == max_bl and j > 0:
                piece = "⋅" + piece
            elif j > 0:
                piece = " " + piece
            pieces.append(piece)
        result.append("".join(pieces))
    return result, max_bl


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
    non_numeric.sort(key=lambda t: (-total_degree(t), "\n".join(_render(t)[0])))
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
