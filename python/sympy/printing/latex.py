"""LaTeX printer for the compatibility shell.

A port of upstream SymPy 1.14.0 ``sympy.printing.latex.LatexPrinter`` for
the expression classes the shell provides (numbers, symbols, Add/Mul/Pow,
elementary and special functions, calculus operators, relationals,
booleans, sets, matrices, Piecewise). Term and factor ordering come from
the same ``sort_key`` / ``as_ordered_terms`` port the str printer uses, so
both printers agree with the oracle on ordering. Unsupported classes raise
``NotImplementedError`` rather than emitting approximate output.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

from .str import (
    PRECEDENCE,
    _as_coeff_mul,
    _could_extract_minus_sign,
    _core,
    _kind,
    _num_value,
    as_ordered_terms,
    precedence,
    sort_key,
)

accepted_latex_functions = [
    "arcsin", "arccos", "arctan", "sin", "cos", "tan", "sinh", "cosh", "tanh",
    "sqrt", "ln", "log", "sec", "csc", "cot", "coth", "re", "im", "frac",
    "root", "arg",
]

tex_greek_dictionary = {
    "Alpha": r"\mathrm{A}", "Beta": r"\mathrm{B}", "Gamma": r"\Gamma",
    "Delta": r"\Delta", "Epsilon": r"\mathrm{E}", "Zeta": r"\mathrm{Z}",
    "Eta": r"\mathrm{H}", "Theta": r"\Theta", "Iota": r"\mathrm{I}",
    "Kappa": r"\mathrm{K}", "Lambda": r"\Lambda", "Mu": r"\mathrm{M}",
    "Nu": r"\mathrm{N}", "Xi": r"\Xi", "omicron": "o", "Omicron": r"\mathrm{O}",
    "Pi": r"\Pi", "Rho": r"\mathrm{P}", "Sigma": r"\Sigma", "Tau": r"\mathrm{T}",
    "Upsilon": r"\Upsilon", "Phi": r"\Phi", "Chi": r"\mathrm{X}", "Psi": r"\Psi",
    "Omega": r"\Omega", "lamda": r"\lambda", "Lamda": r"\Lambda", "khi": r"\chi",
    "Khi": r"\mathrm{X}", "varepsilon": r"\varepsilon", "varkappa": r"\varkappa",
    "varphi": r"\varphi", "varpi": r"\varpi", "varrho": r"\varrho",
    "varsigma": r"\varsigma", "vartheta": r"\vartheta",
}

greeks = (
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi", "rho",
    "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
)

other_symbols = {"aleph", "beth", "daleth", "gimel", "ell", "eth", "hbar", "hslash", "mho", "wp"}

modifier_dict = {
    "mathring": lambda s: r"\mathring{" + s + r"}",
    "ddddot": lambda s: r"\ddddot{" + s + r"}",
    "dddot": lambda s: r"\dddot{" + s + r"}",
    "ddot": lambda s: r"\ddot{" + s + r"}",
    "dot": lambda s: r"\dot{" + s + r"}",
    "check": lambda s: r"\check{" + s + r"}",
    "breve": lambda s: r"\breve{" + s + r"}",
    "acute": lambda s: r"\acute{" + s + r"}",
    "grave": lambda s: r"\grave{" + s + r"}",
    "tilde": lambda s: r"\tilde{" + s + r"}",
    "hat": lambda s: r"\hat{" + s + r"}",
    "bar": lambda s: r"\bar{" + s + r"}",
    "vec": lambda s: r"\vec{" + s + r"}",
    "prime": lambda s: "{" + s + "}'",
    "prm": lambda s: "{" + s + "}'",
    "bold": lambda s: r"\boldsymbol{" + s + r"}",
    "bm": lambda s: r"\boldsymbol{" + s + r"}",
    "cal": lambda s: r"\mathcal{" + s + r"}",
    "scr": lambda s: r"\mathscr{" + s + r"}",
    "frak": lambda s: r"\mathfrak{" + s + r"}",
    "norm": lambda s: r"\left\|{" + s + r"}\right\|",
    "avg": lambda s: r"\left\langle{" + s + r"}\right\rangle",
    "abs": lambda s: r"\left|{" + s + r"}\right|",
    "mag": lambda s: r"\left|{" + s + r"}\right|",
}

_between_two_numbers_p = (
    re.compile(r"[0-9][} ]*$"),
    re.compile(r"(\d|\\frac{\d+}{\d+})"),
)

_name_with_digits_p = re.compile(r"^([^\W\d_]+)(\d+)$", re.U)

_INV_TRIG = (
    "asin", "acos", "atan", "acsc", "asec", "acot",
    "asinh", "acosh", "atanh", "acsch", "asech", "acoth",
)

_CONSTANT_TEX = {
    "pi": r"\pi", "E": "e", "I": "i", "oo": r"\infty", "-oo": r"-\infty",
    "zoo": r"\tilde{\infty}", "nan": r"\text{NaN}", "EulerGamma": r"\gamma",
    "Catalan": "G", "GoldenRatio": r"\phi", "TribonacciConstant": r"\text{TribonacciConstant}",
}


def split_super_sub(text: str) -> tuple[str, list, list]:
    """Upstream ``sympy.printing.conventions.split_super_sub``."""
    if not text:
        return text, [], []
    pos = 0
    name = None
    supers: list = []
    subs: list = []
    while pos < len(text):
        start = pos + 1
        if text[pos:pos + 2] == "__":
            start += 1
        pos_hat = text.find("^", start)
        if pos_hat < 0:
            pos_hat = len(text)
        pos_usc = text.find("_", start)
        if pos_usc < 0:
            pos_usc = len(text)
        pos_next = min(pos_hat, pos_usc)
        part = text[pos:pos_next]
        pos = pos_next
        if name is None:
            name = part
        elif part.startswith("^"):
            supers.append(part[1:])
        elif part.startswith("__"):
            supers.append(part[2:])
        elif part.startswith("_"):
            subs.append(part[1:])
    m = _name_with_digits_p.match(name)
    if m:
        name, sub = m.groups()
        subs.insert(0, sub)
    return name, supers, subs


def translate(s: str) -> str:
    """Upstream greek-letter / modifier translation of a name part."""
    tex = tex_greek_dictionary.get(s)
    if tex:
        return tex
    if s.lower() in greeks:
        return "\\" + s.lower()
    if s in other_symbols:
        return "\\" + s
    for key in sorted(modifier_dict.keys(), key=len, reverse=True):
        if s.lower().endswith(key) and len(s) > len(key):
            return modifier_dict[key](translate(s[:-len(key)]))
    return s


def _deal_with_super_sub(string: str) -> str:
    if "{" in string:
        return string
    name, supers, subs = split_super_sub(string)
    name = translate(name)
    supers = [translate(sup) for sup in supers]
    subs = [translate(sub) for sub in subs]
    if supers:
        name += "^{%s}" % " ".join(supers)
    if subs:
        name += "_{%s}" % " ".join(subs)
    return name


def _is_neg_one(e: Any) -> bool:
    return _kind(e) == "Integer" and e.p == -1


class LatexPrinter:
    def __init__(self, settings: dict | None = None) -> None:
        settings = dict(settings or {})
        self._settings = {
            "fold_short_frac": settings.get("fold_short_frac"),
            "mode": settings.get("mode", "plain"),
            "mul_symbol": settings.get("mul_symbol"),
            "inv_trig_style": settings.get("inv_trig_style", "abbreviated"),
            "ln_notation": settings.get("ln_notation", False),
            "mat_delim": settings.get("mat_delim", "["),
            "mat_str": settings.get("mat_str"),
            "root_notation": settings.get("root_notation", True),
            "symbol_names": settings.get("symbol_names", {}),
            "parenthesize_super": settings.get("parenthesize_super", True),
            "imaginary_unit": settings.get("imaginary_unit", "i"),
            "diff_operator": settings.get("diff_operator", "d"),
            "itex": settings.get("itex", False),
        }
        if self._settings["mode"] not in ("inline", "plain", "equation", "equation*"):
            raise ValueError(
                "'mode' must be one of 'inline', 'plain', 'equation' or 'equation*'"
            )
        if self._settings["fold_short_frac"] is None and self._settings["mode"] == "inline":
            self._settings["fold_short_frac"] = True
        table = {None: r" ", "ldot": r" \,.\, ", "dot": r" \cdot ", "times": r" \times "}
        mul = self._settings["mul_symbol"]
        self._sep = table.get(mul, mul)
        self._numbersep = table.get(mul or "dot", mul)
        self._delim = {"(": ")", "[": "]"}

    # -- entry points -----------------------------------------------------

    def doprint(self, expr: Any) -> str:
        tex = self._print(expr)
        mode = self._settings["mode"]
        if mode == "plain":
            return tex
        if mode == "inline":
            return r"$%s$" % tex
        if self._settings["itex"]:
            return r"$$%s$$" % tex
        return r"\begin{%s}%s\end{%s}" % (mode, tex, mode)

    def _print(self, e: Any, exp: str | None = None) -> str:
        c = _core()
        if isinstance(e, bool):
            return r"\text{%s}" % e
        if isinstance(e, int):
            return str(e)
        if isinstance(e, (tuple, list)):
            return self._print_seq(e, isinstance(e, list))
        if isinstance(e, dict):
            items = ["%s : %s" % (self._print(k), self._print(v))
                     for k, v in sorted(e.items(), key=lambda kv: sort_key(kv[0]))]
            return r"\left\{ %s\right\}" % r", \  ".join(items)
        if isinstance(e, (set, frozenset)):
            items = sorted(e, key=sort_key)
            return r"\left\{%s\right\}" % ", ".join(self._print(i) for i in items)
        name = type(e).__name__
        method = getattr(self, "_print_" + name, None)
        if method is not None and name not in ("Function",):
            if exp is not None and name in self._EXP_AWARE:
                return method(e, exp)
            return method(e)
        k = _kind(e)
        if k == "Add":
            return self._print_Add(e)
        if k == "Mul":
            return self._print_Mul(e)
        if k == "Pow":
            return self._print_Pow(e)
        if k == "Integer":
            return str(e.p)
        if k == "Rational":
            return self._print_Rational(e)
        if k == "Float":
            return self._print_Float(e)
        if k == "Symbol":
            return self._print_Symbol(e)
        if k in ("Infinity", "NegativeInfinity", "NaN", "Constant", "ComplexInfinity"):
            text = str(e)
            if text == "I":
                return self._settings["imaginary_unit"]
            return _CONSTANT_TEX.get(text, text)
        if isinstance(e, c.Relational):
            return self._print_Relational(e)
        from ..logic.boolalg import BooleanAtom, BooleanFunction

        if isinstance(e, BooleanAtom):
            return r"\text{%s}" % e
        if isinstance(e, BooleanFunction):
            return self._print_Boolean(e)
        if isinstance(e, c.Derivative):
            return self._print_Derivative(e)
        if k == "Function":
            return self._print_Function(e, exp)
        from ..sets import Set

        if isinstance(e, Set):
            return self._print_Set(e)
        if getattr(e, "_struct_args", None) is not None:
            return self._print_Basic(e)
        raise NotImplementedError(f"latex printing is not implemented for {name}")

    _EXP_AWARE = frozenset({
        "Abs", "log", "exp", "factorial", "subfactorial", "factorial2", "binomial",
        "floor", "ceiling", "conjugate", "re", "im", "Max", "Min", "gamma", "zeta",
        "Mod", "besselj", "bessely", "besseli", "besselk", "LambertW", "Heaviside",
        "DiracDelta",
    })

    # -- helpers ----------------------------------------------------------

    def _precedence(self, item: Any) -> int:
        c = _core()
        name = type(item).__name__
        if name in ("Integral", "Sum", "Product", "Limit") or isinstance(item, c.Derivative):
            return PRECEDENCE["Mul"]
        if name in ("Union", "Intersection", "Complement", "SymmetricDifference", "ProductSet"):
            return PRECEDENCE["Xor"] if "Xor" in PRECEDENCE else 10
        return precedence(item)

    def parenthesize(self, item: Any, level: int, is_neg: bool = False, strict: bool = False) -> str:
        prec = self._precedence(item)
        if is_neg and strict:
            return r"\left(%s\right)" % self._print(item)
        if prec < level or (not strict and prec <= level):
            return r"\left(%s\right)" % self._print(item)
        return self._print(item)

    def parenthesize_super(self, s: str) -> str:
        if "^" in s:
            if self._settings["parenthesize_super"]:
                return r"\left(%s\right)" % s
            return "{%s}" % s
        return s

    def _needs_mul_brackets(self, e: Any, first: bool = False, last: bool = False) -> bool:
        c = _core()
        k = _kind(e)
        if k == "Mul":
            if not first and _could_extract_minus_sign(e):
                return True
        elif self._precedence(e) < PRECEDENCE["Mul"]:
            return True
        elif isinstance(e, c.Relational):
            return True
        if type(e).__name__ == "Piecewise":
            return True
        if self._has(e, ("Mod",)):
            return True
        if not last and self._has(e, ("Integral", "Product", "Sum")):
            return True
        return False

    def _needs_add_brackets(self, e: Any) -> bool:
        c = _core()
        if isinstance(e, c.Relational):
            return True
        if self._has(e, ("Mod",)):
            return True
        return _kind(e) == "Add"

    def _has(self, e: Any, names: tuple) -> bool:
        if type(e).__name__ in names:
            return True
        for a in getattr(e, "args", ()) or ():
            if hasattr(a, "args") and self._has(a, names):
                return True
        return False

    # -- arithmetic -------------------------------------------------------

    def _print_Add(self, e: Any) -> str:
        c = _core()
        terms = as_ordered_terms(e)
        tex = ""
        for i, term in enumerate(terms):
            if i == 0:
                pass
            elif _could_extract_minus_sign(term):
                tex += " - "
                term = -term
            else:
                tex += " + "
            term_tex = self._print(term)
            if self._needs_add_brackets(term):
                term_tex = r"\left(%s\right)" % term_tex
            tex += term_tex
        del c
        return tex

    def _fraction(self, factors: list) -> tuple[list, list]:
        """Upstream ``fraction(expr, exact=True)`` over a factor list."""
        c = _core()
        numer: list = []
        denom: list = []
        for term in factors:
            k = _kind(term)
            if k == "Pow" or (k == "Function" and type(term).__name__ == "exp"):
                if k == "Pow":
                    b, ex = term.args
                else:
                    b, ex = c.E, term.args[0]
                ecoeff, erest = _as_coeff_mul(ex)
                negative = ecoeff is not None and _num_value(ecoeff) < 0 and not erest
                if negative:
                    if _is_neg_one(ex):
                        denom.append(b)
                    else:
                        denom.append(c.Pow(b, -ex) if k == "Pow" else term.func(-ex))
                    continue
                numer.append(term)
            elif k == "Rational":
                if term.p != 1:
                    numer.append(c.Integer(term.p))
                denom.append(c.Integer(term.q))
            else:
                numer.append(term)
        return numer, denom

    def _convert_args(self, args: list) -> str:
        tex = last = ""
        for i, term in enumerate(args):
            term_tex = self._print(term)
            if self._needs_mul_brackets(term, first=(i == 0), last=(i == len(args) - 1)):
                term_tex = r"\left(%s\right)" % term_tex
            if _between_two_numbers_p[0].search(last) and _between_two_numbers_p[1].match(term_tex):
                tex += self._numbersep
            elif tex:
                tex += self._sep
            tex += term_tex
            last = term_tex
        return tex

    def _convert_factors(self, factors: list) -> str:
        c = _core()
        if not factors:
            return "1"
        if len(factors) == 1:
            return self._print(factors[0])
        ordered = sorted(factors, key=sort_key)
        del c
        return self._convert_args(ordered)

    def _print_Mul(self, e: Any) -> str:
        c = _core()
        held = getattr(e, "_args", None)
        if held is not None and type(e) is c.Mul:
            # Upstream: an unevaluated Mul with a leading 1 or a later
            # number prints its args exactly as stored.
            args = list(e.args)
            if args and (
                (_kind(args[0]) == "Integer" and args[0].p == 1)
                or any(_kind(a) in ("Integer", "Rational", "Float") for a in args[1:])
            ):
                return self._convert_args(args)
        coeff, rest = _as_coeff_mul(e)
        factors = ([] if coeff is None else [coeff]) + list(rest)
        tex = ""
        include_parens = False
        if _could_extract_minus_sign(e):
            if coeff is not None and _num_value(coeff) == -1:
                factors = list(rest)
            else:
                neg = -coeff
                factors = [neg] + list(rest)
            tex = "- "
        flat: list = []
        for f in factors:
            if _kind(f) == "Integer" and f.p == 1:
                continue
            flat.append(f)
        numer, denom = self._fraction(flat)
        if not denom:
            tex += self._convert_factors(numer)
        else:
            snumer = self._convert_factors(numer)
            sdenom = self._convert_factors(denom)
            ldenom = len(sdenom.split())
            if self._settings["fold_short_frac"] and ldenom <= 2 and "^" not in sdenom:
                numer_e = numer[0] if len(numer) == 1 else None
                if numer_e is not None and self._needs_mul_brackets(numer_e, last=False):
                    tex += r"\left(%s\right) / %s" % (snumer, sdenom)
                else:
                    tex += r"%s / %s" % (snumer, sdenom)
            else:
                tex += r"\frac{%s}{%s}" % (snumer, sdenom)
        if include_parens:
            tex += ")"
        del c
        return tex

    def _print_Pow(self, e: Any) -> str:
        c = _core()
        base, ex = e.args
        kx = _kind(ex)
        if kx in ("Integer", "Rational"):
            p, q = int(ex.p), int(ex.q)
            if abs(p) == 1 and q != 1 and self._settings["root_notation"]:
                b = self._print(base)
                tex = r"\sqrt{%s}" % b if q == 2 else r"\sqrt[%d]{%s}" % (q, b)
                return r"\frac{1}{%s}" % tex if p < 0 else tex
            if p < 0:
                if _kind(base) == "Integer" and base.p == 1:
                    return r"%s^{%s}" % (self._print(base), self._print(ex))
                if _kind(base) in ("Integer", "Rational"):
                    bp, bq = int(base.p), int(base.q)
                    if bp * bq == abs(bq):
                        if p == -1 and q == 1:
                            return r"\frac{1}{\frac{%s}{%s}}" % (bp, bq)
                        return r"\frac{1}{(\frac{%s}{%s})^{%s}}" % (bp, bq, self._print(-ex))
                return self._print_pow_as_frac(base, ex)
        if _kind(base) == "Function" and not isinstance(base, c.Derivative):
            return self._print(base, exp=self._print(ex))
        return self._standard_power(base, ex)

    def _print_pow_as_frac(self, base: Any, ex: Any) -> str:
        c = _core()
        pos = -ex
        denom = base if (_kind(pos) == "Integer" and pos.p == 1) else c.Pow(base, pos)
        sdenom = self._print(denom)
        if self._settings["fold_short_frac"] and len(sdenom.split()) <= 2 and "^" not in sdenom:
            return r"1 / %s" % sdenom
        return r"\frac{1}{%s}" % sdenom

    def _standard_power(self, base: Any, ex: Any) -> str:
        exp_tex = self._print(ex)
        btex = self.parenthesize(base, PRECEDENCE["Pow"])
        if _kind(base) == "Symbol":
            btex = self.parenthesize_super(btex)
        elif _kind(base) == "Float":
            btex = "{%s}" % btex
        return r"%s^{%s}" % (btex, exp_tex)

    def _print_Rational(self, e: Any) -> str:
        p, q = int(e.p), int(e.q)
        if q == 1:
            return str(p)
        sign = ""
        if p < 0:
            sign = "- "
            p = -p
        if self._settings["fold_short_frac"]:
            return r"%s%d / %d" % (sign, p, q)
        return r"%s\frac{%d}{%d}" % (sign, p, q)

    def _print_Float(self, e: Any) -> str:
        text = str(e)
        mant, sep, ex = text.partition("e")
        if "." in mant:
            # mpmath to_str(strip_zeros=True): keep one digit after '.'.
            mant = mant.rstrip("0")
            if mant.endswith("."):
                mant += "0"
        text = mant + sep + ex
        if "e" in text:
            mant, ex = text.split("e")
            if ex.startswith("+"):
                ex = ex[1:]
            return r"%s%s10^{%s}" % (mant, self._numbersep, ex)
        if text in ("inf", "+inf"):
            return r"\infty"
        if text == "-inf":
            return r"- \infty"
        return text

    def _print_Symbol(self, e: Any) -> str:
        name = self._settings["symbol_names"].get(e)
        if name is not None:
            return name
        return _deal_with_super_sub(e.name)

    _print_Dummy = _print_Symbol

    # -- functions --------------------------------------------------------

    def _hprint_Function(self, func: str) -> str:
        func = _deal_with_super_sub(func)
        sup = func.find("^")
        sub = func.find("_")
        if func in accepted_latex_functions:
            return "\\%s" % func
        if len(func) == 1 or func.startswith("\\") or sub == 1 or sup == 1:
            return func
        if sup > 0 and sub > 0:
            cut = min(sub, sup)
            return r"\operatorname{%s}%s" % (func[:cut], func[cut:])
        if sup > 0:
            return r"\operatorname{%s}%s" % (func[:sup], func[sup:])
        if sub > 0:
            return r"\operatorname{%s}%s" % (func[:sub], func[sub:])
        return r"\operatorname{%s}" % func

    def _print_Function(self, e: Any, exp: str | None = None) -> str:
        c = _core()
        func = type(e).__name__
        special = getattr(self, "_print_" + func, None)
        if special is not None and not isinstance(e, c.AppliedUndef):
            return special(e, exp) if func in self._EXP_AWARE else special(e)
        args = [self._print(a) for a in e.args]
        style = self._settings["inv_trig_style"]
        power_case = False
        if func in _INV_TRIG:
            if style == "full":
                func = ("ar" if func[-1] == "h" else "arc") + func[1:]
            elif style == "power":
                func = func[1:]
                power_case = True
        if power_case:
            if func in accepted_latex_functions:
                name = r"\%s^{-1}" % func
            else:
                name = r"\operatorname{%s}^{-1}" % func
        elif exp is not None:
            ftex = self.parenthesize_super(self._hprint_Function(func))
            name = r"%s^{%s}" % (ftex, exp)
        else:
            name = self._hprint_Function(func)
        name += r"{\left(%s \right)}"
        if power_case and exp is not None:
            name += r"^{%s}" % exp
        return name % ",".join(args)

    def _do_exponent(self, tex: str, exp: str | None) -> str:
        if exp is not None:
            return r"\left(%s\right)^{%s}" % (tex, exp)
        return tex

    def _with_exp(self, tex: str, exp: str | None) -> str:
        return r"%s^{%s}" % (tex, exp) if exp is not None else tex

    def _print_exp(self, e: Any, exp: str | None = None) -> str:
        return self._do_exponent(r"e^{%s}" % self._print(e.args[0]), exp)

    def _print_log(self, e: Any, exp: str | None = None) -> str:
        fn = r"\ln" if self._settings["ln_notation"] else r"\log"
        return self._with_exp(r"%s{\left(%s \right)}" % (fn, self._print(e.args[0])), exp)

    def _print_Abs(self, e: Any, exp: str | None = None) -> str:
        return self._with_exp(r"\left|{%s}\right|" % self._print(e.args[0]), exp)

    def _print_re(self, e: Any, exp: str | None = None) -> str:
        return self._do_exponent(
            r"\operatorname{re}{%s}" % self.parenthesize(e.args[0], PRECEDENCE["Atom"]), exp)

    def _print_im(self, e: Any, exp: str | None = None) -> str:
        return self._do_exponent(
            r"\operatorname{im}{%s}" % self.parenthesize(e.args[0], PRECEDENCE["Atom"]), exp)

    def _print_conjugate(self, e: Any, exp: str | None = None) -> str:
        return self._with_exp(r"\overline{%s}" % self._print(e.args[0]), exp)

    def _print_factorial(self, e: Any, exp: str | None = None) -> str:
        return self._with_exp(r"%s!" % self.parenthesize(e.args[0], PRECEDENCE["Func"]), exp)

    def _print_factorial2(self, e: Any, exp: str | None = None) -> str:
        return self._with_exp(r"%s!!" % self.parenthesize(e.args[0], PRECEDENCE["Func"]), exp)

    def _print_subfactorial(self, e: Any, exp: str | None = None) -> str:
        tex = r"!%s" % self.parenthesize(e.args[0], PRECEDENCE["Func"])
        return r"\left(%s\right)^{%s}" % (tex, exp) if exp is not None else tex

    def _print_binomial(self, e: Any, exp: str | None = None) -> str:
        tex = r"{\binom{%s}{%s}}" % (self._print(e.args[0]), self._print(e.args[1]))
        return self._with_exp(tex, exp)

    def _print_floor(self, e: Any, exp: str | None = None) -> str:
        return self._with_exp(r"\left\lfloor{%s}\right\rfloor" % self._print(e.args[0]), exp)

    def _print_ceiling(self, e: Any, exp: str | None = None) -> str:
        return self._with_exp(r"\left\lceil{%s}\right\rceil" % self._print(e.args[0]), exp)

    def _print_gamma(self, e: Any, exp: str | None = None) -> str:
        tex = r"\left(%s\right)" % self._print(e.args[0])
        return r"\Gamma^{%s}%s" % (exp, tex) if exp is not None else r"\Gamma%s" % tex

    def _print_zeta(self, e: Any, exp: str | None = None) -> str:
        if len(e.args) == 2:
            tex = r"\left(%s, %s\right)" % tuple(self._print(a) for a in e.args)
        else:
            tex = r"\left(%s\right)" % self._print(e.args[0])
        return r"\zeta^{%s}%s" % (exp, tex) if exp is not None else r"\zeta%s" % tex

    def _hprint_bessel(self, e: Any, exp: str | None, sym: str) -> str:
        tex = sym if exp is None else r"%s^{%s}" % (sym, exp)
        return r"%s_{%s}\left(%s\right)" % (tex, self._print(e.args[0]), self._print(e.args[1]))

    def _print_besselj(self, e: Any, exp: str | None = None) -> str:
        return self._hprint_bessel(e, exp, "J")

    def _print_bessely(self, e: Any, exp: str | None = None) -> str:
        return self._hprint_bessel(e, exp, "Y")

    def _print_besseli(self, e: Any, exp: str | None = None) -> str:
        return self._hprint_bessel(e, exp, "I")

    def _print_besselk(self, e: Any, exp: str | None = None) -> str:
        return self._hprint_bessel(e, exp, "K")

    def _print_LambertW(self, e: Any, exp: str | None = None) -> str:
        sup = r"^{%s}" % exp if exp is not None else ""
        if len(e.args) == 1:
            return r"W%s\left(%s\right)" % (sup, self._print(e.args[0]))
        return "W{0}_{{{1}}}\\left({2}\\right)".format(
            sup, self._print(e.args[1]), self._print(e.args[0]))

    def _print_Heaviside(self, e: Any, exp: str | None = None) -> str:
        tex = r"\theta\left(%s\right)" % self._print(e.args[0])
        return r"\left(%s\right)^{%s}" % (tex, exp) if exp else tex

    def _print_DiracDelta(self, e: Any, exp: str | None = None) -> str:
        if len(e.args) == 1 or str(e.args[1]) == "0":
            tex = r"\delta\left(%s\right)" % self._print(e.args[0])
        else:
            tex = r"\delta^{\left( %s \right)}\left( %s \right)" % (
                self._print(e.args[1]), self._print(e.args[0]))
        return r"\left(%s\right)^{%s}" % (tex, exp) if exp else tex

    def _hprint_variadic(self, e: Any, exp: str | None = None) -> str:
        args = sorted(e.args, key=sort_key)
        tex = r"\%s\left(%s\right)" % (type(e).__name__.lower(), ", ".join(self._print(a) for a in args))
        return self._with_exp(tex, exp)

    _print_Max = _print_Min = _hprint_variadic

    def _print_Mod(self, e: Any, exp: str | None = None) -> str:
        a = self.parenthesize(e.args[0], PRECEDENCE["Mul"], strict=True)
        b = self.parenthesize(e.args[1], PRECEDENCE["Mul"], strict=True)
        if exp is not None:
            return r"\left(%s \bmod %s\right)^{%s}" % (a, b, exp)
        return r"%s \bmod %s" % (a, b)

    def _print_Order(self, e: Any) -> str:
        args = e.args
        s = self._print(args[0])
        if len(args) == 3 and str(args[2]) != "0":
            s += "; %s\\rightarrow %s" % (self._print(args[1]), self._print(args[2]))
        return r"O\left(%s\right)" % s

    def _print_Subs(self, e: Any) -> str:
        subs = r"\\ ".join(
            self._print(v) + "=" + self._print(p) for v, p in zip(e.variables, e.point)
        )
        return r"\left. %s \right|_{\substack{ %s }}" % (self._print(e.expr), subs)

    def _print_Tuple(self, e: Any) -> str:
        return self._print_seq(tuple(e.args), False)

    def _print_seq(self, items: Any, is_list: bool) -> str:
        if is_list:
            return r"\left[ %s\right]" % r", \  ".join(self._print(i) for i in items)
        if len(items) == 1:
            return r"\left( %s,\right)" % self._print(items[0])
        return r"\left( %s\right)" % r", \  ".join(self._print(i) for i in items)

    def _print_Basic(self, e: Any) -> str:
        name = _deal_with_super_sub(type(e).__name__)
        args = getattr(e, "_struct_args", None) or e.args
        if args:
            return r"\operatorname{%s}\left(%s\right)" % (name, ", ".join(self._print(a) for a in args))
        return r"\text{%s}" % name

    # -- calculus operators ----------------------------------------------

    def _print_Integral(self, e: Any) -> str:
        limits = [tuple(lim) if isinstance(lim, (tuple, list)) else
                  (tuple(lim.args) if type(lim).__name__ == "Tuple" else (lim,))
                  for lim in e.limits]
        d = self._settings["diff_operator"]
        tex = ""
        symbols: list = []
        if len(limits) <= 4 and all(len(lim) == 1 for lim in limits):
            tex = r"\i" + "i" * (len(limits) - 1) + "nt"
            symbols = [r"\, %s%s" % (d, self._print(lim[0])) for lim in limits]
        else:
            for lim in reversed(limits):
                tex += r"\int"
                if len(lim) > 1:
                    if self._settings["mode"] != "inline" and not self._settings["itex"]:
                        tex += r"\limits"
                    if len(lim) == 3:
                        tex += "_{%s}^{%s}" % (self._print(lim[1]), self._print(lim[2]))
                    if len(lim) == 2:
                        tex += "^{%s}" % self._print(lim[1])
                symbols.insert(0, r"\, %s%s" % (d, self._print(lim[0])))
        neg = _could_extract_minus_sign(e.function)
        body = self.parenthesize(e.function, PRECEDENCE["Mul"], is_neg=neg, strict=True)
        return r"%s %s%s" % (tex, body, "".join(symbols))

    def _requires_partial(self, e: Any) -> bool:
        free = getattr(e, "free_symbols", set())
        return sum(1 for s in free if not getattr(s, "is_integer", False)) > 1

    def _print_Derivative(self, e: Any) -> str:
        expr = e.args[0]
        diff_symbol = r"\partial" if self._requires_partial(expr) else self._settings["diff_operator"]
        groups: list = []
        for v in e.args[1:]:
            if groups and groups[-1][0] == v:
                groups[-1][1] += 1
            else:
                groups.append([v, 1])
        tex = ""
        dim = 0
        for v, num in reversed(groups):
            dim += num
            if num == 1:
                tex += r"%s %s" % (diff_symbol, self._print(v))
            else:
                tex += r"%s %s^{%s}" % (diff_symbol, self.parenthesize_super(self._print(v)), num)
        if dim == 1:
            tex = r"\frac{%s}{%s}" % (diff_symbol, tex)
        else:
            tex = r"\frac{%s^{%s}}{%s}" % (diff_symbol, dim, tex)
        neg = any(_could_extract_minus_sign(a) for a in e.args)
        return r"%s %s" % (tex, self.parenthesize(expr, PRECEDENCE["Mul"], is_neg=neg, strict=True))

    def _concrete(self, e: Any, op: str) -> str:
        limits = [tuple(t.args) if hasattr(t, "args") and not isinstance(t, tuple) else tuple(t)
                  for t in getattr(e, "limits", e.args[1:])]
        if len(limits) == 1:
            tex = r"\%s_{%s=%s}^{%s} " % ((op,) + tuple(self._print(i) for i in limits[0]))
        else:
            parts = [r"%s \leq %s \leq %s" % tuple(self._print(s) for s in (lim[1], lim[0], lim[2]))
                     for lim in limits]
            tex = r"\%s_{\substack{%s}} " % (op, "\\\\".join(parts))
        f = e.args[0]
        if _kind(f) == "Add":
            tex += r"\left(%s\right)" % self._print(f)
        else:
            tex += self._print(f)
        return tex

    def _print_Sum(self, e: Any) -> str:
        return self._concrete(e, "sum")

    def _print_Product(self, e: Any) -> str:
        return self._concrete(e, "prod")

    def _print_Limit(self, e: Any) -> str:
        expr, z, z0, d = e.args
        tex = r"\lim_{%s \to " % self._print(z)
        if str(d) == "+-" or str(z0) in ("oo", "-oo"):
            tex += r"%s}" % self._print(z0)
        else:
            tex += r"%s^%s}" % (self._print(z0), str(d))
        if _kind(expr) in ("Add", "Mul"):
            return r"%s\left(%s\right)" % (tex, self._print(expr))
        return r"%s %s" % (tex, self._print(expr))

    # -- relations, logic, piecewise, matrices, sets ----------------------

    def _print_Relational(self, e: Any) -> str:
        charmap = {"==": "=", ">": ">", "<": "<", ">=": r"\geq", "<=": r"\leq", "!=": r"\neq"}
        return "%s %s %s" % (self._print(e.lhs), charmap[e.rel_op], self._print(e.rhs))

    def _print_Boolean(self, e: Any) -> str:
        from ..logic.boolalg import And, Equivalent, Implies, Not, Or, Xor

        if isinstance(e, Not):
            arg = e.args[0]
            if isinstance(arg, Equivalent):
                return self._logop(sorted(arg.args, key=sort_key), r"\not\Leftrightarrow")
            if isinstance(arg, Implies):
                return self._logop(arg.args, r"\not\Rightarrow")
            if self._is_boolean(arg):
                return r"\neg \left(%s\right)" % self._print(arg)
            return r"\neg %s" % self._print(arg)
        table = {And: r"\wedge", Or: r"\vee", Xor: r"\veebar", Equivalent: r"\Leftrightarrow"}
        for cls, char in table.items():
            if isinstance(e, cls):
                return self._logop(sorted(e.args, key=self._bool_key), char)
        if isinstance(e, Implies):
            return self._logop(e.args, r"\Rightarrow")
        return self._print_Basic(e)

    def _bool_key(self, a: Any) -> tuple:
        from ..logic.boolalg import _bool_sort_key

        return _bool_sort_key(a)

    def _is_boolean(self, a: Any) -> bool:
        from ..logic.boolalg import Boolean

        c = _core()
        return isinstance(a, (Boolean, c.Relational))

    def _logop(self, args: Any, char: str) -> str:
        from ..logic.boolalg import Not

        out = []
        for i, arg in enumerate(args):
            if self._is_boolean(arg) and not isinstance(arg, Not):
                t = r"\left(%s\right)" % self._print(arg)
            else:
                t = self._print(arg)
            out.append(t if i == 0 else r" %s %s" % (char, t))
        return "".join(out)

    def _print_Piecewise(self, e: Any) -> str:
        from ..logic.boolalg import true

        pairs = [tuple(p.args) if hasattr(p, "args") and not isinstance(p, tuple) else tuple(p)
                 for p in e.args]
        lines = [r"%s & \text{for}\: %s" % (self._print(p[0]), self._print(p[1])) for p in pairs[:-1]]
        last = pairs[-1]
        if last[1] == true or last[1] is True:
            lines.append(r"%s & \text{otherwise}" % self._print(last[0]))
        else:
            lines.append(r"%s & \text{for}\: %s" % (self._print(last[0]), self._print(last[1])))
        return r"\begin{cases} %s \end{cases}" % r" \\".join(lines)

    def _print_matrix(self, m: Any) -> str:
        rows = [" & ".join(self._print(m[i, j]) for j in range(m.cols)) for i in range(m.rows)]
        mat_str = self._settings["mat_str"]
        if mat_str is None:
            if self._settings["mode"] == "inline":
                mat_str = "smallmatrix"
            elif m.cols <= 10:
                mat_str = "matrix"
            else:
                mat_str = "array"
        out = r"\begin{%s}%s\end{%s}" % (mat_str, "%s", mat_str)
        if mat_str == "array":
            out = out.replace("%s", "{" + "c" * m.cols + "}%s")
        out = out % r"\\".join(rows)
        delim = self._settings["mat_delim"]
        if delim:
            out = r"\left" + delim + out + r"\right" + self._delim[delim]
        return out

    _print_MutableDenseMatrix = _print_ImmutableDenseMatrix = _print_Matrix = _print_matrix
    _print_DenseMatrix = _print_MatrixBase = _print_matrix

    def _print_Set(self, s: Any) -> str:
        from ..sets import EmptySet, FiniteSet, Interval, Union

        name = type(s).__name__
        if name == "Reals":
            return r"\mathbb{R}"
        if name in ("Integers", "Naturals", "Naturals0", "Rationals", "Complexes"):
            return {"Integers": r"\mathbb{Z}", "Naturals": r"\mathbb{N}",
                    "Naturals0": r"\mathbb{N}_0", "Rationals": r"\mathbb{Q}",
                    "Complexes": r"\mathbb{C}"}[name]
        if isinstance(s, EmptySet) or name == "EmptySet":
            return r"\emptyset"
        if isinstance(s, Interval):
            if s.start == s.end:
                return r"\left\{%s\right\}" % self._print(s.start)
            left = "(" if s.left_open else "["
            right = ")" if s.right_open else "]"
            return r"\left%s%s, %s\right%s" % (left, self._print(s.start), self._print(s.end), right)
        if isinstance(s, FiniteSet):
            items = sorted(s.args, key=sort_key)
            return r"\left\{%s\right\}" % ", ".join(self._print(i) for i in items)
        joiner = {"Union": r" \cup ", "Intersection": r" \cap ", "Complement": r" \setminus "}.get(name)
        if joiner is not None:
            return joiner.join(self.parenthesize(a, 10) for a in s.args)
        if isinstance(s, Union):
            return r" \cup ".join(self._print(a) for a in s.args)
        raise NotImplementedError(f"latex printing is not implemented for {name}")


def latex(expr: Any, **settings: Any) -> str:
    """Upstream ``latex``: the LaTeX rendering of ``expr``."""
    return LatexPrinter(settings).doprint(expr)


def print_latex(expr: Any, **settings: Any) -> None:
    print(latex(expr, **settings))


__all__ = ["LatexPrinter", "latex", "print_latex"]
