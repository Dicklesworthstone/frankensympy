"""2D pretty printer for the compatibility shell.

A port of upstream SymPy 1.14.0 ``sympy.printing.pretty.pretty.PrettyPrinter``
for the classes the shell provides, built on the vendored upstream layout
engine (``stringpict`` / ``pretty_symbology``). Term and factor ordering
come from the shared ``sort_key`` / ``as_ordered_terms`` port, so the str,
LaTeX and pretty printers agree with each other and with the oracle.
"""

from __future__ import annotations

from typing import Any

from .pretty_symbology import (
    U,
    greek_unicode,
    hobj,
    pretty_atom,
    pretty_symbol,
    pretty_try_use_unicode,
    pretty_use_unicode,
    root as nth_root,
    vobj,
    xobj,
    xsym,
    center_pad,
)
from .str import (
    PRECEDENCE,
    _as_coeff_mul,
    _core,
    _could_extract_minus_sign,
    _kind,
    _num_value,
    _strip_float,
    as_ordered_terms,
    precedence,
    sort_key,
)
from .stringpict import prettyForm, stringPict

pprint_use_unicode = pretty_use_unicode
pprint_try_use_unicode = pretty_try_use_unicode

_ATOM_NAMES = {
    "pi": "Pi", "E": "Exp1", "I": "ImaginaryUnit", "oo": "Infinity",
    "-oo": "NegativeInfinity", "zoo": "ComplexInfinity", "nan": "NaN",
}


class PrettyPrinter:
    """Printer converting an expression into a 2D ASCII/Unicode figure."""

    _default_settings = {
        "order": None,
        "full_prec": "auto",
        "use_unicode": None,
        "wrap_line": True,
        "num_columns": None,
        "use_unicode_sqrt_char": True,
        "root_notation": True,
        "mat_symbol_style": "plain",
        "imaginary_unit": "i",
        "perm_cyclic": True,
    }

    def __init__(self, settings: dict | None = None) -> None:
        self._settings = dict(self._default_settings)
        for k, v in (settings or {}).items():
            if k not in self._default_settings:
                raise TypeError("Unknown setting '%s'." % k)
            self._settings[k] = v
        if self._settings["imaginary_unit"] not in ("i", "j"):
            raise ValueError(
                "'imaginary_unit' must be either 'i' or 'j', not '{}'".format(
                    self._settings["imaginary_unit"]
                )
            )
        self._print_level = 0

    @property
    def _use_unicode(self) -> bool:
        if self._settings["use_unicode"]:
            return True
        return pretty_use_unicode()

    def doprint(self, expr: Any) -> str:
        return self._print(expr).render(**self._settings)

    def emptyPrinter(self, expr: Any) -> prettyForm:
        return prettyForm(str(expr))

    # -- dispatch ---------------------------------------------------------

    def _print(self, e: Any, **kwargs: Any) -> Any:
        self._print_level += 1
        try:
            return self._dispatch(e, **kwargs)
        finally:
            self._print_level -= 1

    def _dispatch(self, e: Any, **kwargs: Any) -> Any:
        c = _core()
        if isinstance(e, (stringPict,)):
            return e
        if isinstance(e, str):
            return prettyForm(e)
        if isinstance(e, bool):
            return prettyForm(str(e))
        if isinstance(e, int):
            return prettyForm(str(e))
        if isinstance(e, list):
            return self._print_seq(e, "[", "]")
        if isinstance(e, tuple):
            return self._print_tuple(e)
        if isinstance(e, dict):
            return self._print_dict(e)
        if isinstance(e, (set, frozenset)):
            if not e:
                return prettyForm("set()")
            items = sorted(e, key=sort_key)
            return prettyForm(*self._print_seq(items).parens("{", "}", ifascii_nougly=True))
        name = type(e).__name__
        method = getattr(self, "_print_" + name, None)
        if method is not None and name not in ("Function",):
            return method(e)
        k = _kind(e)
        if k == "Add":
            return self._print_Add(e)
        if k == "Mul":
            return self._print_Mul(e)
        if k == "Pow":
            return self._print_Pow(e)
        if k in ("Integer", "Rational"):
            return self._print_Rational(e)
        if k == "Float":
            return self._print_Float(e)
        if k == "Symbol":
            return self._print_Symbol(e)
        if k in ("Infinity", "NegativeInfinity", "NaN", "Constant", "ComplexInfinity"):
            return self._print_Atom(e)
        if isinstance(e, c.Relational):
            return self._print_Relational(e)
        from ..logic.boolalg import BooleanAtom, BooleanFunction

        if isinstance(e, BooleanAtom):
            return prettyForm(str(e))
        if isinstance(e, BooleanFunction):
            return self._print_Boolean(e)
        if isinstance(e, c.Derivative):
            return self._print_Derivative(e)
        if k == "Function":
            return self._print_Function(e)
        from ..sets import Set

        if isinstance(e, Set):
            return self._print_Set(e)
        if getattr(e, "_struct_args", None) is not None:
            return self.emptyPrinter(e)
        return self.emptyPrinter(e)

    # -- atoms ------------------------------------------------------------

    def _print_Atom(self, e: Any) -> prettyForm:
        text = str(e)
        named = {"EulerGamma": ("gamma", "EulerGamma"), "GoldenRatio": ("phi", "GoldenRatio"),
                 "Catalan": ("G", "G")}.get(text)
        if named is not None:
            return prettyForm(pretty_symbol(named[0] if self._use_unicode else named[1]))
        atom = _ATOM_NAMES.get(text)
        if atom is None:
            return self.emptyPrinter(e)
        try:
            return prettyForm(pretty_atom(atom, printer=self))
        except KeyError:
            return self.emptyPrinter(e)

    def _print_Symbol(self, e: Any, bold_name: bool = False) -> prettyForm:
        return prettyForm(pretty_symbol(e.name, bold_name))

    _print_Dummy = _print_Symbol

    def _print_Float(self, e: Any) -> prettyForm:
        full_prec = self._settings["full_prec"]
        if full_prec == "auto":
            full_prec = self._print_level == 1
        text = str(e)
        return prettyForm(text if full_prec else _strip_float(text))

    def _numer_denom(self, p: int, q: int) -> Any:
        if q == 1:
            if p < 0:
                return prettyForm(str(p), binding=prettyForm.NEG)
            return prettyForm(str(p))
        if abs(p) >= 10 and abs(q) >= 10:
            if p < 0:
                return prettyForm(str(p), binding=prettyForm.NEG) / prettyForm(str(q))
            return prettyForm(str(p)) / prettyForm(str(q))
        return None

    def _print_Rational(self, e: Any) -> prettyForm:
        result = self._numer_denom(int(e.p), int(e.q))
        if result is not None:
            return result
        return self.emptyPrinter(e)

    # -- arithmetic -------------------------------------------------------

    def _print_Add(self, expr: Any) -> prettyForm:
        c = _core()
        terms = as_ordered_terms(expr)
        pforms: list = []
        indices: list = []

        def pretty_negative(pform: Any, index: int) -> prettyForm:
            if index == 0:
                pform_neg = "- " if pform.height() > 1 else "-"
            else:
                pform_neg = " - "
            if pform.binding > prettyForm.NEG or pform.binding == prettyForm.ADD:
                p = stringPict(*pform.parens())
            else:
                p = pform
            p = stringPict.next(pform_neg, p)
            return prettyForm(binding=prettyForm.NEG, *p)

        for i, term in enumerate(terms):
            k = _kind(term)
            if k == "Mul" and _could_extract_minus_sign(term):
                coeff, other = _as_coeff_mul(term)
                if coeff is not None and _num_value(coeff) == -1:
                    negterm = c.Mul(*other, evaluate=False) if len(other) > 1 else other[0]
                else:
                    negterm = c.Mul(-coeff, *other, evaluate=False)
                pforms.append(pretty_negative(self._print(negterm), i))
            elif k == "Rational":
                pforms.append(None)
                indices.append(i)
            elif k in ("Integer", "Float", "NegativeInfinity") and _could_extract_minus_sign(term):
                pforms.append(pretty_negative(self._print(-term), i))
            elif isinstance(term, c.Relational):
                pforms.append(prettyForm(*self._print(term).parens()))
            else:
                pforms.append(self._print(term))
        if indices:
            large = any(p is not None and p.height() > 1 for p in pforms)
            for i in indices:
                term = terms[i]
                negative = term.p < 0
                if negative:
                    term = -term
                if large:
                    pform = prettyForm(str(term.p)) / prettyForm(str(term.q))
                else:
                    pform = self._print(term)
                if negative:
                    pform = pretty_negative(pform, i)
                pforms[i] = pform
        return prettyForm.__add__(*pforms)

    def _ordered_factors(self, product: Any) -> list:
        """Upstream ``as_ordered_factors`` (via ``args_cnc``): a negative
        coefficient other than -1 is split into -1 and its magnitude."""
        c = _core()
        coeff, rest = _as_coeff_mul(product)
        out: list = []
        if coeff is not None:
            v = _num_value(coeff)
            if v is not None and v < 0 and v != -1:
                out = [c.Integer(-1), -coeff]
            else:
                out = [coeff]
        return out + sorted(rest, key=sort_key)

    def _print_Mul(self, product: Any) -> prettyForm:
        c = _core()
        args = list(product.args)
        held = getattr(product, "_args", None) is not None
        if held and args and (
            (_kind(args[0]) == "Integer" and args[0].p == 1)
            or any(_kind(a) in ("Integer", "Rational", "Float") for a in args[1:])
        ):
            strargs = [self._print(a) for a in args]
            negone = strargs[0] == "-1"
            if negone:
                strargs[0] = prettyForm("1", 0, 0)
            obj = prettyForm.__mul__(*strargs)
            if negone:
                obj = prettyForm("-" + obj.s, obj.baseline, obj.binding)
            return obj
        a: list = []
        b: list = []
        for item in self._ordered_factors(product):
            k = _kind(item)
            if k == "Pow":
                base, ex = item.args
                if _kind(ex) in ("Integer", "Rational") and ex.p < 0:
                    if not (_kind(ex) == "Integer" and ex.p == -1):
                        b.append(c.Pow(base, -ex, evaluate=False))
                    else:
                        b.append(base)
                    continue
                a.append(item)
            elif k in ("Integer", "Rational"):
                if item.p != 1:
                    a.append(c.Integer(item.p))
                if item.q != 1:
                    b.append(c.Integer(item.q))
            else:
                a.append(item)
        pa = [self._print(ai) for ai in a]
        pb = [self._print(bi) for bi in b]
        if not pb:
            return prettyForm.__mul__(*pa)
        if not pa:
            pa.append(self._print(c.Integer(1)))
        return prettyForm.__mul__(*pa) / prettyForm.__mul__(*pb)

    def _print_nth_root(self, base: Any, root: Any) -> prettyForm:
        bpretty = self._print(base)
        is_nonneg_int = _kind(base) == "Integer" and base.p >= 0
        if (
            self._settings["use_unicode_sqrt_char"]
            and self._use_unicode
            and root == 2
            and bpretty.height() == 1
            and (bpretty.width() == 1 or is_nonneg_int)
        ):
            return prettyForm(*bpretty.left(nth_root[2]))
        _zZ = xobj("/", 1)
        rootsign = xobj("\\", 1) + _zZ
        rpretty = self._print(root)
        if rpretty.height() != 1:
            c = _core()
            return self._print(base) ** self._print(c.Integer(1) / root)
        exp = "" if root == 2 else str(rpretty).ljust(2)
        if len(exp) > 2:
            rootsign = " " * (len(exp) - 2) + rootsign
        rootsign = stringPict(exp + "\n" + rootsign)
        rootsign.baseline = 0
        linelength = bpretty.height() - 1
        diagonal = stringPict(
            "\n".join(" " * (linelength - i - 1) + _zZ + " " * i for i in range(linelength))
        )
        diagonal.baseline = linelength - 1
        rootsign = prettyForm(*rootsign.right(diagonal))
        rootsign.baseline = max(1, bpretty.baseline)
        s = prettyForm(hobj("_", 2 + bpretty.width()))
        s = prettyForm(*bpretty.above(s))
        s = prettyForm(*s.left(rootsign))
        return s

    def _print_Pow(self, power: Any) -> prettyForm:
        c = _core()
        b, e = power.args
        # Upstream Pow.as_base_exp: (1/q)**e is displayed as q**(-e).
        if _kind(b) == "Rational" and b.p == 1 and b.q > 1:
            b, e = c.Integer(b.q), -e
        ke = _kind(e)
        if ke == "Integer" and e.p == -1:
            return prettyForm("1") / self._print(b)
        # fraction(e): numerator 1 over an atom denominator -> root notation
        if self._settings["root_notation"]:
            if ke == "Rational" and e.p == 1:
                return self._print_nth_root(b, c.Integer(e.q))
            if ke == "Pow" and _kind(e.args[1]) == "Integer" and e.args[1].p == -1 \
                    and _kind(e.args[0]) == "Symbol":
                return self._print_nth_root(b, e.args[0])
        if ke in ("Integer", "Rational") and e.p < 0:
            return prettyForm("1") / self._print(c.Pow(b, -e, evaluate=False))
        if isinstance(b, c.Relational):
            return prettyForm(*self._print(b).parens()).__pow__(self._print(e))
        return self._print(b) ** self._print(e)

    # -- functions --------------------------------------------------------

    def _helper_print_function(
        self, func_name: str, args: Any, sort: bool = False, delimiter: str = ", ",
        left: str = "(", right: str = ")",
    ) -> prettyForm:
        if sort:
            args = sorted(args, key=sort_key)
        pretty_func = prettyForm(pretty_symbol(func_name))
        pretty_args = prettyForm(
            *self._print_seq(args, delimiter=delimiter).parens(left=left, right=right)
        )
        pform = prettyForm(binding=prettyForm.FUNC, *stringPict.next(pretty_func, pretty_args))
        pform.prettyFunc = pretty_func
        pform.prettyArgs = pretty_args
        return pform

    def _print_Function(self, e: Any, sort: bool = False, func_name: str | None = None,
                        left: str = "(", right: str = ")") -> prettyForm:
        c = _core()
        name = type(e).__name__
        special = getattr(self, "_print_" + name, None)
        if special is not None and func_name is None and not isinstance(e, c.AppliedUndef):
            return special(e)
        return self._helper_print_function(func_name or name, e.args, sort=sort,
                                           left=left, right=right)

    def _print_exp(self, e: Any) -> prettyForm:
        base = prettyForm(pretty_atom("Exp1", "e"))
        return base ** self._print(e.args[0])

    def _print_Abs(self, e: Any) -> prettyForm:
        return prettyForm(*self._print(e.args[0]).parens("|", "|"))

    def _print_conjugate(self, e: Any) -> prettyForm:
        pform = self._print(e.args[0])
        return prettyForm(*pform.above(hobj("_", pform.width())))

    def _print_floor(self, e: Any) -> prettyForm:
        if self._use_unicode:
            return prettyForm(*self._print(e.args[0]).parens("lfloor", "rfloor"))
        return self._helper_print_function("floor", e.args)

    def _print_ceiling(self, e: Any) -> prettyForm:
        if self._use_unicode:
            return prettyForm(*self._print(e.args[0]).parens("lceil", "rceil"))
        return self._helper_print_function("ceiling", e.args)

    def _postfix(self, e: Any, mark: str, prefix: bool = False) -> prettyForm:
        x = e.args[0]
        pform = self._print(x)
        if not ((_kind(x) == "Integer" and x.p >= 0) or _kind(x) == "Symbol"):
            pform = prettyForm(*pform.parens())
        return prettyForm(*(pform.left(mark) if prefix else pform.right(mark)))

    def _print_factorial(self, e: Any) -> prettyForm:
        return self._postfix(e, "!")

    def _print_factorial2(self, e: Any) -> prettyForm:
        return self._postfix(e, "!!")

    def _print_subfactorial(self, e: Any) -> prettyForm:
        return self._postfix(e, "!", prefix=True)

    def _print_binomial(self, e: Any) -> prettyForm:
        n, k = e.args
        n_pform = self._print(n)
        k_pform = self._print(k)
        bar = " " * max(n_pform.width(), k_pform.width())
        pform = prettyForm(*k_pform.above(bar))
        pform = prettyForm(*pform.above(n_pform))
        pform = prettyForm(*pform.parens("(", ")"))
        pform.baseline = (pform.baseline + 1) // 2
        return pform

    def _print_gamma(self, e: Any) -> prettyForm:
        name = greek_unicode["Gamma"] if self._use_unicode else "Gamma"
        return self._helper_print_function(name, e.args)

    def _print_LambertW(self, e: Any) -> prettyForm:
        return self._helper_print_function("W", e.args)

    def _print_Heaviside(self, e: Any) -> prettyForm:
        name = greek_unicode["theta"] if self._use_unicode else "Heaviside"
        if len(e.args) == 1:
            pform = prettyForm(*self._print(e.args[0]).parens())
            return prettyForm(*pform.left(name))
        return self._helper_print_function(name, e.args)

    def _print_DiracDelta(self, e: Any) -> prettyForm:
        if not self._use_unicode:
            return self._helper_print_function("DiracDelta", e.args)
        if len(e.args) == 2:
            a = prettyForm(greek_unicode["delta"])
            b = prettyForm(*self._print(e.args[1]).parens())
            cc = prettyForm(*self._print(e.args[0]).parens())
            pform = a ** b
            pform = prettyForm(*pform.right(" "))
            return prettyForm(*pform.right(cc))
        pform = prettyForm(*self._print(e.args[0]).parens())
        return prettyForm(*pform.left(greek_unicode["delta"]))

    def _print_atan2(self, e: Any) -> prettyForm:
        pform = prettyForm(*self._print_seq(e.args).parens())
        return prettyForm(*pform.left("atan2"))

    def _print_Mod(self, e: Any) -> prettyForm:
        pform = self._print(e.args[0])
        if pform.binding > prettyForm.MUL:
            pform = prettyForm(*pform.parens())
        pform = prettyForm(*pform.right(" mod "))
        pform = prettyForm(*pform.right(self._print(e.args[1])))
        pform.binding = prettyForm.OPEN
        return pform

    def _print_Order(self, e: Any) -> prettyForm:
        args = e.args
        pform = self._print(args[0])
        if len(args) == 3 and str(args[2]) != "0":
            pform = prettyForm(*pform.right("; "))
            pform = prettyForm(*pform.right(self._print(args[1])))
            arrow = " %s " % pretty_atom("Arrow") if self._use_unicode else " -> "
            pform = prettyForm(*pform.right(arrow))
            pform = prettyForm(*pform.right(self._print(args[2])))
        pform = prettyForm(*pform.parens())
        return prettyForm(*pform.left("O"))

    def _print_Tuple(self, e: Any) -> prettyForm:
        return self._print_tuple(tuple(e.args))

    # -- calculus ---------------------------------------------------------

    def _requires_partial(self, e: Any) -> bool:
        free = getattr(e, "free_symbols", set())
        return sum(1 for s in free if not getattr(s, "is_integer", False)) > 1

    def _print_Derivative(self, deriv: Any) -> prettyForm:
        expr = deriv.args[0]
        if self._requires_partial(expr) and self._use_unicode:
            deriv_symbol = U("PARTIAL DIFFERENTIAL")
        else:
            deriv_symbol = "d"
        groups: list = []
        for v in deriv.args[1:]:
            if groups and groups[-1][0] == v:
                groups[-1][1] += 1
            else:
                groups.append([v, 1])
        x = None
        total = 0
        for sym, num in reversed(groups):
            s = self._print(sym)
            ds = prettyForm(*s.left(deriv_symbol))
            total += num
            if num > 1:
                ds = ds ** prettyForm(str(num))
            if x is None:
                x = ds
            else:
                x = prettyForm(*x.right(" "))
                x = prettyForm(*x.right(ds))
        f = prettyForm(binding=prettyForm.FUNC, *self._print(expr).parens())
        pform = prettyForm(deriv_symbol)
        if total > 1:
            pform = pform ** prettyForm(str(total))
        pform = prettyForm(*pform.below(stringPict.LINE, x))
        pform.baseline = pform.baseline + 1
        pform = prettyForm(*stringPict.next(pform, f))
        pform.binding = prettyForm.MUL
        return pform

    def _print_Integral(self, integral: Any) -> prettyForm:
        f = integral.function
        limits = [tuple(lim) for lim in integral.limits]
        prettyF = self._print(f)
        if _kind(f) == "Add":
            prettyF = prettyForm(*prettyF.parens())
        arg = prettyF
        for x in limits:
            prettyArg = self._print(x[0])
            if prettyArg.width() > 1:
                prettyArg = prettyForm(*prettyArg.parens())
            arg = prettyForm(*arg.right(" d", prettyArg))
        firstterm = True
        s = None
        for lim in limits:
            h = arg.height()
            H = h + 2
            ascii_mode = not self._use_unicode
            if ascii_mode:
                H += 2
            vint = vobj("int", H)
            pform = prettyForm(vint)
            pform.baseline = arg.baseline + (H - h) // 2
            if len(lim) > 1:
                if len(lim) == 2:
                    prettyA = prettyForm("")
                    prettyB = self._print(lim[1])
                if len(lim) == 3:
                    prettyA = self._print(lim[1])
                    prettyB = self._print(lim[2])
                if ascii_mode:
                    spc = max(1, 3 - prettyB.width())
                    prettyB = prettyForm(*prettyB.left(" " * spc))
                    spc = max(1, 4 - prettyA.width())
                    prettyA = prettyForm(*prettyA.right(" " * spc))
                pform = prettyForm(*pform.above(prettyB))
                pform = prettyForm(*pform.below(prettyA))
            if not ascii_mode:
                pform = prettyForm(*pform.right(" "))
            if firstterm:
                s = pform
                firstterm = False
            else:
                s = prettyForm(*s.left(pform))
        pform = prettyForm(*arg.left(s))
        pform.binding = prettyForm.MUL
        return pform

    def _sum_product_limits(self, lim: tuple) -> tuple:
        op = prettyForm(" " + xsym("==") + " ")
        lower = prettyForm(*stringPict.next(self._print(lim[0]), op, self._print(lim[1])))
        return lower, self._print(lim[2])

    def _print_Sum(self, expr: Any) -> prettyForm:
        ascii_mode = not self._use_unicode

        def asum(hrequired: int, use_ascii: bool) -> tuple:
            h = max(hrequired, 2)
            d = h // 2
            w = d + 1
            more = hrequired % 2
            lines = []
            if use_ascii:
                lines.append("_" * w + " ")
                lines.append(r"\%s`" % (" " * (w - 1)))
                for i in range(1, d):
                    lines.append("%s\\%s" % (" " * i, " " * (w - i)))
                if more:
                    lines.append("%s)%s" % (" " * d, " " * (w - d)))
                for i in reversed(range(1, d)):
                    lines.append("%s/%s" % (" " * i, " " * (w - i)))
                lines.append("/" + "_" * (w - 1) + ",")
                return d, h + more, lines, more
            w = w + more
            d = d + more
            vsum = vobj("sum", 4)
            lines.append("_" * w)
            for i in range(0, d):
                lines.append("%s%s%s" % (" " * i, vsum[2], " " * (w - i - 1)))
            for i in reversed(range(0, d)):
                lines.append("%s%s%s" % (" " * i, vsum[4], " " * (w - i - 1)))
            lines.append(vsum[8] * w)
            return d, h + 2 * more, lines, more

        f = expr.function
        prettyF = self._print(f)
        if _kind(f) == "Add":
            prettyF = prettyForm(*prettyF.parens())
        H = prettyF.height() + 2
        first = True
        max_upper = 0
        sign_height = 0
        adjustment = 0
        for lim in expr.limits:
            prettyLower, prettyUpper = self._sum_product_limits(lim)
            max_upper = max(max_upper, prettyUpper.height())
            d, h, slines, adjustment = asum(H, ascii_mode)
            prettySign = stringPict("")
            prettySign = prettyForm(*prettySign.stack(*slines))
            if first:
                sign_height = prettySign.height()
            prettySign = prettyForm(*prettySign.above(prettyUpper))
            prettySign = prettyForm(*prettySign.below(prettyLower))
            if first:
                prettyF.baseline -= d - (prettyF.height() // 2 - prettyF.baseline)
                first = False
            pad = stringPict("")
            pad = prettyForm(*pad.stack(*[" "] * h))
            prettySign = prettyForm(*prettySign.right(pad))
            prettyF = prettyForm(*prettySign.right(prettyF))
        ascii_adjustment = ascii_mode if not adjustment else 0
        prettyF.baseline = max_upper + sign_height // 2 + ascii_adjustment
        prettyF.binding = prettyForm.MUL
        return prettyF

    def _print_Product(self, expr: Any) -> prettyForm:
        pretty_func = self._print(expr.function)
        horizontal_chr = xobj("_", 1)
        corner_chr = xobj("_", 1)
        vertical_chr = xobj("|", 1)
        if self._use_unicode:
            horizontal_chr = xobj("-", 1)
            corner_chr = xobj("UpTack", 1)
        func_height = pretty_func.height()
        first = True
        max_upper = 0
        sign_height = 0
        for lim in expr.limits:
            pretty_lower, pretty_upper = self._sum_product_limits(lim)
            width = (func_height + 2) * 5 // 3 - 2
            sign_lines = [horizontal_chr + corner_chr + (horizontal_chr * (width - 2))
                          + corner_chr + horizontal_chr]
            for _ in range(func_height + 1):
                sign_lines.append(" " + vertical_chr + (" " * (width - 2)) + vertical_chr + " ")
            pretty_sign = stringPict("")
            pretty_sign = prettyForm(*pretty_sign.stack(*sign_lines))
            max_upper = max(max_upper, pretty_upper.height())
            if first:
                sign_height = pretty_sign.height()
            pretty_sign = prettyForm(*pretty_sign.above(pretty_upper))
            pretty_sign = prettyForm(*pretty_sign.below(pretty_lower))
            if first:
                pretty_func.baseline = 0
                first = False
            height = pretty_sign.height()
            padding = stringPict("")
            padding = prettyForm(*padding.stack(*[" "] * (height - 1)))
            pretty_sign = prettyForm(*pretty_sign.right(padding))
            pretty_func = prettyForm(*pretty_sign.right(pretty_func))
        pretty_func.baseline = max_upper + sign_height // 2
        pretty_func.binding = prettyForm.MUL
        return pretty_func

    def _print_Limit(self, lim: Any) -> prettyForm:
        e, z, z0, d = lim.args
        E = self._print(e)
        if precedence(e) <= PRECEDENCE["Mul"]:
            E = prettyForm(*E.parens("(", ")"))
        Lim = prettyForm("lim")
        LimArg = self._print(z)
        if self._use_unicode:
            LimArg = prettyForm(*LimArg.right(f"{xobj('-', 1)}{pretty_atom('Arrow')}"))
        else:
            LimArg = prettyForm(*LimArg.right("->"))
        LimArg = prettyForm(*LimArg.right(self._print(z0)))
        if str(d) == "+-" or str(z0) in ("oo", "-oo"):
            dtext = ""
        elif self._use_unicode:
            dtext = pretty_atom("SuperscriptPlus") if str(d) == "+" else pretty_atom("SuperscriptMinus")
        else:
            dtext = str(d)
        LimArg = prettyForm(*LimArg.right(self._print(dtext)))
        Lim = prettyForm(*Lim.below(LimArg))
        return prettyForm(*Lim.right(E), binding=prettyForm.MUL)

    # -- relations, logic, piecewise, matrices, sets ----------------------

    def _print_Relational(self, e: Any) -> prettyForm:
        op = prettyForm(" " + xsym(e.rel_op) + " ")
        return prettyForm(*stringPict.next(self._print(e.lhs), op, self._print(e.rhs)),
                          binding=prettyForm.OPEN)

    def _print_Boolean(self, e: Any) -> prettyForm:
        from ..logic.boolalg import (
            And, Boolean, Equivalent, Implies, Not, Or, Xor, _bool_sort_key,
        )

        c = _core()

        def is_boolean(a: Any) -> bool:
            return isinstance(a, (Boolean, c.Relational))

        def join(args: Any, char: str, sort: bool = True) -> prettyForm:
            args = sorted(args, key=_bool_sort_key) if sort else list(args)
            first = args[0]
            pform = self._print(first)
            if is_boolean(first) and not isinstance(first, Not) and not isinstance(first, c.Symbol):
                pform = prettyForm(*pform.parens())
            for arg in args[1:]:
                pa = self._print(arg)
                if is_boolean(arg) and not isinstance(arg, Not) and not isinstance(arg, c.Symbol):
                    pa = prettyForm(*pa.parens())
                pform = prettyForm(*pform.right(" %s " % char))
                pform = prettyForm(*pform.right(pa))
            return pform

        if not self._use_unicode:
            return self._helper_print_function(type(e).__name__, e.args, sort=not isinstance(e, Implies))
        if isinstance(e, Not):
            arg = e.args[0]
            if isinstance(arg, Equivalent):
                return join(arg.args, pretty_atom("NotEquiv"))
            if isinstance(arg, Implies):
                return join(arg.args, pretty_atom("NotArrow"), sort=False)
            pform = self._print(arg)
            if is_boolean(arg) and not isinstance(arg, Not) and not isinstance(arg, c.Symbol):
                pform = prettyForm(*pform.parens())
            return prettyForm(*pform.left(pretty_atom("Not")))
        for cls, atom in ((And, "And"), (Or, "Or"), (Xor, "Xor"), (Equivalent, "Equiv")):
            if isinstance(e, cls):
                return join(e.args, pretty_atom(atom))
        if isinstance(e, Implies):
            return join(e.args, pretty_atom("Arrow"), sort=False)
        return self._helper_print_function(type(e).__name__, e.args)

    def _print_Piecewise(self, pexpr: Any) -> prettyForm:
        from ..logic.boolalg import true

        pairs = [tuple(p.args) if hasattr(p, "args") and not isinstance(p, tuple) else tuple(p)
                 for p in pexpr.args]
        P = {}
        for n, (ex, cond) in enumerate(pairs):
            P[n, 0] = self._print(ex)
            if cond == true or cond is True:
                P[n, 1] = prettyForm("otherwise")
            else:
                P[n, 1] = prettyForm(*prettyForm("for ").right(self._print(cond)))
        hsep, vsep = 2, 1
        maxw = [max(P[i, j].width() for i in range(len(pairs))) for j in range(2)]
        D = None
        for i in range(len(pairs)):
            D_row = None
            for j in range(2):
                p = P[i, j]
                wdelta = maxw[j] - p.width()
                wleft = wdelta // 2
                wright = wdelta - wleft
                p = prettyForm(*p.right(" " * wright))
                p = prettyForm(*p.left(" " * wleft))
                if D_row is None:
                    D_row = p
                    continue
                D_row = prettyForm(*D_row.right(" " * hsep))
                D_row = prettyForm(*D_row.right(p))
            if D is None:
                D = D_row
                continue
            for _ in range(vsep):
                D = prettyForm(*D.below(" "))
            D = prettyForm(*D.below(D_row))
        D = prettyForm(*D.parens("{", ""))
        D.baseline = D.height() // 2
        D.binding = prettyForm.OPEN
        return D

    def _print_matrix_contents(self, M: Any) -> prettyForm:
        Ms = {}
        for i in range(M.rows):
            for j in range(M.cols):
                Ms[i, j] = self._print(M[i, j])
        hsep, vsep = 2, 1
        maxw = [max([Ms[i, j].width() for i in range(M.rows)] or [0]) for j in range(M.cols)]
        D = None
        for i in range(M.rows):
            D_row = None
            for j in range(M.cols):
                s = Ms[i, j]
                left, right = center_pad(s.width(), maxw[j])
                s = prettyForm(*s.right(right))
                s = prettyForm(*s.left(left))
                if D_row is None:
                    D_row = s
                    continue
                D_row = prettyForm(*D_row.right(" " * hsep))
                D_row = prettyForm(*D_row.right(s))
            if D is None:
                D = D_row
                continue
            for _ in range(vsep):
                D = prettyForm(*D.below(" "))
            D = prettyForm(*D.below(D_row))
        if D is None:
            D = prettyForm("")
        return D

    def _print_MatrixBase(self, e: Any) -> prettyForm:
        D = self._print_matrix_contents(e)
        D.baseline = D.height() // 2
        return prettyForm(*D.parens("[", "]"))

    _print_Matrix = _print_MutableDenseMatrix = _print_ImmutableDenseMatrix = _print_MatrixBase
    _print_DenseMatrix = _print_MatrixBase

    def _print_Set(self, s: Any) -> prettyForm:
        from ..sets import FiniteSet, Interval

        name = type(s).__name__
        if name in ("EmptySet", "Reals", "Integers", "Naturals", "Naturals0", "Rationals",
                    "Complexes", "UniversalSet") or getattr(s, "is_empty", None) is True:
            key = "EmptySet" if getattr(s, "is_empty", None) is True and name != "Reals" else name
            if key == "UniversalSet":
                return prettyForm(pretty_atom("Universe") if self._use_unicode else "UniversalSet")
            if key == "Reals" and not self._use_unicode:
                return self._print_seq(["-oo", "oo"], "(", ")")
            try:
                return prettyForm(pretty_atom(key, printer=self))
            except KeyError:
                return prettyForm(name)
        if isinstance(s, Interval):
            if s.start == s.end:
                return self._print_seq([s.start], "{", "}")
            left = "(" if s.left_open else "["
            right = ")" if s.right_open else "]"
            return self._print_seq([s.start, s.end], left, right)
        if isinstance(s, FiniteSet):
            return self._print_seq(sorted(s.args, key=sort_key), "{", "}", ", ")
        delim = {
            "Union": " %s " % pretty_atom("Union", "U"),
            "Intersection": " %s " % pretty_atom("Intersection", "n"),
            "Complement": r" \ ",
        }.get(name)
        if delim is not None:
            return self._print_seq(list(s.args), None, None, delim,
                                   parenthesize=lambda a: type(a).__name__ in
                                   ("Union", "Intersection", "Complement", "ProductSet")
                                   and type(a).__name__ != name)
        return self.emptyPrinter(s)

    # -- sequences --------------------------------------------------------

    def _print_seq(self, seq: Any, left: str | None = None, right: str | None = None,
                   delimiter: str = ", ", parenthesize: Any = lambda x: False,
                   ifascii_nougly: bool = True) -> prettyForm:
        pforms: list = []
        for item in seq:
            pform = self._print(item)
            if parenthesize(item):
                pform = prettyForm(*pform.parens())
            if pforms:
                pforms.append(delimiter)
            pforms.append(pform)
        if not pforms:
            s = stringPict("")
        else:
            s = prettyForm(*stringPict.next(*pforms))
        return prettyForm(*s.parens(left, right, ifascii_nougly=ifascii_nougly))

    def _print_tuple(self, t: tuple) -> prettyForm:
        if len(t) == 1:
            ptuple = prettyForm(*stringPict.next(self._print(t[0]), ","))
            return prettyForm(*ptuple.parens("(", ")", ifascii_nougly=True))
        return self._print_seq(t, "(", ")")

    def _print_dict(self, d: dict) -> prettyForm:
        items = []
        for k in sorted(d.keys(), key=sort_key):
            items.append(prettyForm(*stringPict.next(self._print(k), ": ", self._print(d[k]))))
        return self._print_seq(items, "{", "}")


def pretty(expr: Any, **settings: Any) -> str:
    """Return a string containing the prettified form of ``expr``."""
    pp = PrettyPrinter(settings)
    uflag = pretty_use_unicode(pp._settings["use_unicode"])
    try:
        return pp.doprint(expr)
    finally:
        pretty_use_unicode(uflag)


def pretty_print(expr: Any, **kwargs: Any) -> None:
    """Print ``expr`` in pretty form."""
    print(pretty(expr, **kwargs))


pprint = pretty_print

# Upstream enables unicode output when the terminal can encode it.
pretty_try_use_unicode()

__all__ = ["PrettyPrinter", "pprint", "pprint_use_unicode", "pretty", "pretty_print"]
