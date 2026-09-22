"""LaTeX printer for the compatibility shell (arithmetic core).

Renders Integer, Rational, Symbol, Add, Mul, and Pow exactly as pinned
SymPy 1.14.0's latex printer does for the corpus grammar, verified
per-fixture against the isolated oracle (fra-rc-surface-nvv). Unsupported
expression classes raise NotImplementedError rather than emitting
approximate output.
"""

from __future__ import annotations

from typing import Any
import functools

from ..core import (
    Basic,
    Relational,
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
        # Oracle-pinned: a positive-integer power of an applied function
        # renders the exponent between the name and the args
        # (sin(x)**2 -> \sin^{2}{\left(x \right)}).
        exp_int = _as_int(exp)
        if exp_int is not None and exp_int > 0 and _is_function_instance(base):
            fname = _latex_function_name(type(base).__name__)
            args = ",".join(latex(arg) for arg in base.args)
            return fname + "^{" + str(exp_int) + "}" + rf"{{\left({args} \right)}}"
        if exp_int is not None and exp_int < 0:
            return _frac("1", _latex_pow_positive(base, -exp_int))
        if exp_int is not None:
            return _latex_pow_positive(base, exp_int)
        # Rational exponent: q == 2 renders as sqrt (oracle: \sqrt{2});
        # other rationals as \frac{p}{q}-style roots; symbolic as ^{...}.
        if isinstance(exp, Rational):
            if exp.q == 2:
                if exp.p == 1:
                    return r"\sqrt{" + latex(base) + "}"
                if exp.p == -1:
                    # Oracle-pinned: Pow(b, -1/2) -> \frac{\sqrt{b}}{b}.
                    return _frac(r"\sqrt{" + latex(base) + "}", latex(base))
            return _frac(str(exp.p), _latex_pow_positive(base, exp.q))
        # Symbolic exponent: base then ^{latex(exp)}.
        return _latex_pow_positive(base, 1) + "^{" + latex(exp) + "}"
    if isinstance(expr, Mul):
        return _latex_mul(expr)
    if isinstance(expr, Add):
        return _latex_add(expr)
    if _is_function_instance(expr):
        return _latex_function(expr)
    struct_args = getattr(expr, "_struct_args", None)
    if struct_args is not None:
        # Oracle: custom Expr subclasses render like applied operators with
        # trailing name digits as LaTeX subscripts (VecX2(x, 2) ->
        # \operatorname{VecX_{2}}\left(x, 2\right)).
        name = type(expr).__name__
        stripped = name.rstrip("0123456789")
        digits = name[len(stripped):]
        if digits:
            name = f"{stripped}_{{{digits}}}"
        inner = ", ".join(latex(a) if isinstance(a, Basic) else str(a) for a in struct_args)
        return r"\operatorname{" + name + r"}\left(" + inner + r"\right)"
    if type(expr).__name__ == "Interval":
        # Oracle-pinned: \\left[0, 1\\right] bracket style per openness.
        lo, ro = bool(expr.left_open), bool(expr.right_open)
        left = "(" if lo else r"\left["
        right = ")" if ro else r"\right]"
        return f"{left}{latex(expr.start)}, {latex(expr.end)}{right}"
    if isinstance(expr, Relational):
        from ..core import StrictLessThan, StrictGreaterThan, LessThan, GreaterThan, Equality, Unequality
        ops = {
            StrictLessThan: "<",
            StrictGreaterThan: ">",
            LessThan: r"\leq",
            GreaterThan: r"\geq",
            Equality: "=",
            Unequality: r"\neq",
        }
        op = ops[type(expr)]
        return f"{latex(expr.lhs)} {op} {latex(expr.rhs)}"
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
    if _is_function_instance(base):
        # Oracle-pinned: function bases render with their operator form
        # (sin(x) -> \\sin{\\left(x \\right)}).
        return _latex_function(base)
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
    if q == 1 and denom_terms:
        # Negative-integer-exponent factors (Pow(base, -n)) land in
        # denom_terms even when the rational coefficient has q == 1 -
        # dropping the denominator here silently lost factors
        # (Mul(x**2 - 1, (x + 1)**-1) printed as 'x**2 - 1').
        # Oracle-pinned: a fully \left(...\right)-wrapped denominator term
        # renders bare inside the fraction (\frac{x**2 - 1}{x + 1}).
        unwrapped = []
        for term in denom_terms:
            if term.startswith(r"\left(") and term.endswith(r"\right)"):
                term = term[len(r"\left("):-len(r"\right)")]
            unwrapped.append(term)
        denom = " ".join(unwrapped)
        numer_full = f"{p} {numer}".strip() if p != 1 else (numer if numer else "1")
        return sign + _frac(numer_full, denom)
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

    import functools

    def _class_rank(term: Any) -> int:
        # Oracle-pinned: positive-integer Pows lead, then plain symbols,
        # then negative-exponent Pows, then functions.
        if isinstance(term, Pow):
            exp_int = _as_int(term.args[1])
            if exp_int is not None:
                return 0 if exp_int > 0 else 2
            return 3
        if isinstance(term, Symbol):
            return 1
        if _is_function_instance(term):
            return 3
        return 1

    def _term_name(term: Any) -> str:
        if isinstance(term, Symbol):
            return term.name
        if isinstance(term, Pow):
            return str(term.args[0])
        if _is_function_instance(term):
            return type(term).__name__
        return latex(term)

    from ..core import _FUNCTION_CLASS_RANK

    def _pow_info(term: Any):
        """(base, exp_int_or_None) for Pows; (term, None) otherwise."""
        if isinstance(term, Pow):
            return term.args[0], _as_int(term.args[1])
        return term, None

    def _cmp(a: Any, b: Any) -> int:
        ra, rb = _class_rank(a), _class_rank(b)
        if ra != rb:
            return -1 if ra < rb else 1
        ba, ea = _pow_info(a)
        bb, eb = _pow_info(b)
        fa, fb = _is_function_instance(ba), _is_function_instance(bb)
        # Function bases first among same-rank powers, by class rank.
        if ra == 0 and fa != fb:
            return -1 if fa else 1
        if fa and fb and ra in (0, 3):
            na, nb = type(ba).__name__, type(bb).__name__
            ka, kb = _FUNCTION_CLASS_RANK.get(na, 0), _FUNCTION_CLASS_RANK.get(nb, 0)
            if ka != kb:
                return -1 if ka < kb else 1
        na, nb = _term_name(ba), _term_name(bb)
        if na != nb:
            # Negative-exponent powers sort by base name DESCENDING
            # (1/y before 1/x); everything else ascending.
            if ra == 2:
                return -1 if na > nb else 1
            return -1 if na < nb else 1
        da = -_as_int(ea) if ea is not None else 0
        db = -_as_int(eb) if eb is not None else 0
        return da - db

    non_numeric.sort(key=functools.cmp_to_key(_cmp))

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


_LATEX_KNOWN_FUNCTIONS = {
    "sin": r"\sin",
    "cos": r"\cos",
    "tan": r"\tan",
    "cot": r"\cot",
    "sec": r"\sec",
    "csc": r"\csc",
    "log": r"\log",
    "ln": r"\ln",
    "asin": r"\arcsin",
    "acos": r"\arccos",
    "atan": r"\arctan",
    "sinh": r"\sinh",
    "cosh": r"\cosh",
    "tanh": r"\tanh",
}


def _latex_function_name(name: str) -> str:
    # Oracle-pinned: known built-ins render as their backslash forms
    # (\sin, \cos, \tan, \log). A trailing digit run renders as a
    # subscript after \operatorname; single-letter names print bare.
    if name in _LATEX_KNOWN_FUNCTIONS:
        return _LATEX_KNOWN_FUNCTIONS[name]
    if len(name) == 1:
        return name
    split = len(name)
    while split > 0 and name[split - 1].isdigit():
        split -= 1
    if split == len(name):
        return r"\operatorname{" + name + "}"
    return r"\operatorname{" + name[:split] + "}_{" + name[split:] + "}"


def _latex_function(expr: Any) -> str:
    raw_name = type(expr).__name__
    name = _latex_function_name(raw_name)
    args = ",".join(latex(arg) for arg in expr.args)
    # exp(x) renders as e^{x} (oracle-pinned).
    if raw_name == "exp" and len(expr.args) == 1:
        return "e^{" + latex(expr.args[0]) + "}"
    return name + rf"{{\left({args} \right)}}"