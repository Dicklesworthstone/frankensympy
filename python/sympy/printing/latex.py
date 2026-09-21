"""LaTeX printer for the compatibility shell (arithmetic core).

Renders Integer, Rational, Symbol, Add, Mul, and Pow exactly as pinned
SymPy 1.14.0's latex printer does for the corpus grammar, verified
per-fixture against the isolated oracle (fra-rc-surface-nvv). Unsupported
expression classes raise NotImplementedError rather than emitting
approximate output.
"""

from __future__ import annotations

from typing import Any

from ..core import (
    Basic,
    Add,
    ComplexInfinity,
    Function,
    Integer,
    Mul,
    Pow,
    Rational,
    Symbol,
)


def latex(expr: Any) -> str:
    """Return the pinned-SymPy-1.14.0 LaTeX rendering of ``expr``."""
    if isinstance(expr, ComplexInfinity):
        return r"\tilde{\infty}"
    if str(expr) == "nan":
        return "nan"
    if str(expr) == "-oo":
        return r"-\infty"
    if isinstance(expr, Integer):
        return str(expr.p)
    if isinstance(expr, Rational):
        return _latex_rational(expr.p, expr.q)
    if isinstance(expr, Symbol):
        return expr.name
    if isinstance(expr, Pow):
        base, exp = expr.args
        exp_int = _as_int(exp)
        if exp_int is not None and exp_int < 0:
            return _frac("1", _latex_pow_positive(base, -exp_int))
        return _latex_pow_positive(base, exp_int)
    if isinstance(expr, Mul):
        return _latex_mul(expr)
    if isinstance(expr, Add):
        return _latex_add(expr)
    if _is_function_instance(expr):
        return _latex_function(expr)
    struct_args = getattr(expr, "_struct_args", None)
    if struct_args is not None:
        # Oracle: custom Expr subclasses render like applied operators
        # (V(x, 2) -> \operatorname{V}\left(x, 2\right)).
        inner = ", ".join(latex(a) if isinstance(a, Basic) else str(a) for a in struct_args)
        return r"\operatorname{" + type(expr).__name__ + r"}\left(" + inner + r"\right)"
    raise NotImplementedError(
        f"latex printing is not implemented for {type(expr).__name__}"
    )


def _as_int(expr: Any) -> int | None:
    if isinstance(expr, Integer):
        return expr.p
    return None


def _frac(numer: str, denom: str) -> str:
    return rf"\frac{{{numer}}}{{{denom}}}"


def _wrap_atom_sensitive(base_str: str, is_atom: bool) -> str:
    return base_str if is_atom else rf"\left({base_str}\right)"


def _latex_pow_positive(base: Any, exp: int) -> str:
    base_str = _latex_base(base)
    if exp == 1:
        return base_str
    return base_str + rf"^{{{exp}}}"


def _latex_base(base: Any) -> str:
    """Render a Mul/Pow/Atom base with atom-sensitive parenthesisation."""
    if isinstance(base, (Symbol, Integer)):
        return latex(base)
    if isinstance(base, Rational):
        return latex(base)
    if isinstance(base, Pow):
        base_str = _latex_base(base.args[0]) + rf"^{{{base.args[1]}}}"
        return _wrap_atom_sensitive(base_str, isinstance(base.args[0], (Symbol, Integer)))
    if isinstance(base, (Add, Mul)):
        return rf"\left({latex(base)}\right)"
    raise NotImplementedError(f"latex base printing for {type(base).__name__}")


def _latex_rational(p: int, q: int) -> str:
    # q is always positive and (p, q) coprime in the canonical Rational.
    if q == 1:
        return str(p)
    if p < 0:
        return "- " + _frac(str(-p), str(q))
    return _frac(str(p), str(q))


def _latex_mul(expr: Mul) -> str:
    factors = list(expr.args)
    held = not isinstance(factors[0], (Integer, Rational)) and any(
        isinstance(factor, (Integer, Rational)) for factor in factors[1:]
    )
    if held:
        rendered = []
        for factor in factors:
            if isinstance(factor, Integer):
                rendered.append(
                    rf"\left({factor.p}\right)" if factor.p < 0 else str(factor.p)
                )
            else:
                rendered.append(latex(factor))
        return " ".join(rendered)
    numer_terms: list[str] = []
    denom_terms: list[str] = []
    coeff: tuple[int, int] = (1, 1)
    for factor in expr.args:
        if isinstance(factor, Integer):
            coeff = (coeff[0] * factor.p, coeff[1])
            continue
        if isinstance(factor, Rational):
            coeff = (coeff[0] * factor.p, coeff[1] * factor.q)
            continue
        if isinstance(factor, Pow):
            base, exp = factor.args
            exp_int = _as_int(exp)
            if exp_int is not None and exp_int < 0:
                denom_terms.append(_latex_pow_positive(base, -exp_int))
                continue
        numer_terms.append(latex(factor))
    if not numer_terms and coeff[0] == 1 and coeff[1] == 1:
        # Mul(1) collapses at construction; empty here is a caller error.
        raise NotImplementedError("degenerate Mul without factors")

    numer = " ".join(numer_terms)
    sign = ""
    p, q = coeff
    if p < 0:
        sign = "- "
        p = -p
    if q == 1:
        if p != 1:
            coeff_str = str(p)
            numer = f"{coeff_str} {numer}" if numer else coeff_str
        elif not numer:
            numer = "1"
    else:
        denom_terms.insert(0, str(q))
        denom = " ".join(denom_terms)
        if p == 1:
            numer = _frac(numer if numer else "1", denom)
        else:
            numer = _frac(f"{p} {numer}" if numer else str(p), denom)
    return sign + numer


def _latex_add(expr: Add) -> str:
    # Oracle-pinned order (verified across the corpus): non-numeric terms by
    # total degree descending (ties alphabetical by rendering), pure
    # numeric terms last in stored order; sign absorbed into separators.
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
            return sum(total_degree(factor) for factor in term.args)
        if isinstance(term, Add):
            return max(total_degree(part) for part in term.args)
        return 1

    numeric = [term for term in expr.args if isinstance(term, (Integer, Rational))]
    non_numeric = [term for term in expr.args if not isinstance(term, (Integer, Rational))]
    non_numeric.sort(key=lambda term: (-total_degree(term), latex(term)))

    parts: list[str] = []
    ordered = non_numeric + numeric
    for index, term in enumerate(ordered):
        negative = _is_negative_term(term)
        body = _latex_term_body(term)
        if index == 0:
            parts.append(("- " if negative else "") + body)
        else:
            parts.append((" - " if negative else " + ") + body)
    return "".join(parts)


def _is_negative_term(term: Any) -> bool:
    if isinstance(term, (Integer, Rational)):
        return term.p < 0
    if isinstance(term, Mul):
        first = term.args[0]
        if isinstance(first, Integer):
            return first.p < 0
        if isinstance(first, Rational):
            return first.p < 0
    return False


def _latex_term_body(term: Any) -> str:
    """Render a term with any leading numeric sign stripped (Add absorbs
    the sign into its separator)."""
    if isinstance(term, (Integer, Rational)):
        return _strip_leading_sign(latex(term))
    if isinstance(term, Mul):
        stripped = Mul(*[_strip_leading_sign_factor(f) for f in term.args])
        return _strip_leading_sign(latex(stripped))
    return latex(term)


def _strip_leading_sign_factor(factor: Any) -> Any:
    if isinstance(factor, Integer) and factor.p < 0:
        return Integer(-factor.p)
    if isinstance(factor, Rational) and factor.p < 0:
        return Rational(-factor.p, factor.q)
    return factor


def _strip_leading_sign(rendered: str) -> str:
    if rendered.startswith("- "):
        return rendered[2:]
    if rendered.startswith("-"):
        return rendered[1:]
    return rendered


def _is_function_instance(expr: Any) -> bool:
    # Covers Function and every applied Function subclass.
    return isinstance(expr, Function)


def _latex_function_name(name: str) -> str:
    # Pinned convention: a trailing digit run renders as a subscript after
    # \operatorname; single-letter names print bare.
    if len(name) == 1:
        return name
    split = len(name)
    while split > 0 and name[split - 1].isdigit():
        split -= 1
    if split == len(name):
        return r"\operatorname{" + name + "}"
    return r"\operatorname{" + name[:split] + "}_{" + name[split:] + "}"


def _latex_function(expr: Any) -> str:
    name = _latex_function_name(type(expr).__name__)
    args = ",".join(latex(arg) for arg in expr.args)
    return name + rf"{{\left({args} \right)}}"