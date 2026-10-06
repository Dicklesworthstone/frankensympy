"""Plain ``str`` printer: a port of the pinned upstream StrPrinter.

Covers the arithmetic core (``Add``, ``Mul``, ``Pow``, numbers, named
constants and function applications) with upstream's precedence table,
``as_ordered_terms`` term order, ``as_ordered_factors`` factor order and
numerator/denominator rendering (``x/y``, ``1/sqrt(x)``, ``-2*x/(3*y)``).

Objects outside that core (held ``evaluate=False`` forms, Dummy, Derivative
and other structural classes) are rendered by the caller-supplied fallback so
their oracle-pinned forms are preserved.
"""

from __future__ import annotations

import cmath
from fractions import Fraction
from typing import Any, Callable

PRECEDENCE = {
    "Lambda": 1,
    "Xor": 10,
    "Or": 20,
    "And": 30,
    "Relational": 35,
    "Add": 40,
    "Mul": 50,
    "Pow": 60,
    "Func": 70,
    "Not": 100,
    "Atom": 1000,
}

_CONSTANT_CLASS = {
    "pi": "Pi",
    "E": "Exp1",
    "I": "ImaginaryUnit",
    "EulerGamma": "EulerGamma",
    "GoldenRatio": "GoldenRatio",
    "Catalan": "Catalan",
    "TribonacciConstant": "TribonacciConstant",
}

_FUNCTION_RANK = {
    "exp": 10, "log": 11, "sin": 20, "cos": 21, "tan": 22, "cot": 23,
    "sinh": 30, "cosh": 31, "tanh": 32, "coth": 33,
    "conjugate": 40, "re": 41, "im": 42, "arg": 43,
}


def _core():
    from .. import core

    return core


# ---------------------------------------------------------------------------
# Classification helpers over shell objects
# ---------------------------------------------------------------------------


def _kind(e: Any) -> str:
    """One of: Integer, Rational, Float, Infinity, NaN, ComplexInfinity,
    Constant, Symbol, Add, Mul, Pow, Function, Other."""
    c = _core()
    t = type(e)
    if t is c.Add:
        return "Add"
    if t is c.Mul:
        return "Mul"
    if t is c.Pow:
        return "Pow"
    if t is c.Float:
        return "Float"
    if isinstance(e, c.Rational):
        return "Integer" if e.q == 1 else "Rational"
    if t is c.ComplexInfinity:
        return "ComplexInfinity"
    if isinstance(e, c.Symbol):
        return "Symbol"
    if isinstance(e, c.Function):
        return "Function"
    if t is c.Expr and not e.args:
        s = str(e._value)
        if s == "oo":
            return "Infinity"
        if s == "-oo":
            return "NegativeInfinity"
        if s == "nan":
            return "NaN"
        if s in _CONSTANT_CLASS:
            return "Constant"
    return "Other"


def _is_number_kind(k: str) -> bool:
    return k in ("Integer", "Rational", "Float", "Infinity", "NegativeInfinity", "NaN", "ComplexInfinity")


def _num_value(e: Any) -> Any:
    k = _kind(e)
    if k in ("Integer", "Rational"):
        return Fraction(int(e.p), int(e.q))
    if k == "Float":
        return float(e)
    if k == "Infinity":
        return float("inf")
    if k == "NegativeInfinity":
        return float("-inf")
    return None


def _is_held(e: Any) -> bool:
    c = _core()
    return type(e) in (c.Add, c.Mul) and getattr(e, "_args", None) is not None


def _as_coeff_mul(e: Any) -> tuple[Any, list]:
    """Upstream ``as_coeff_Mul``: (Number coefficient or None, other factors)."""
    if _kind(e) == "Mul":
        args = list(e.args)
        if args and _kind(args[0]) in ("Integer", "Rational", "Float"):
            return args[0], args[1:]
        for i, a in enumerate(args):
            if _kind(a) in ("Infinity", "NegativeInfinity"):
                return a, args[:i] + args[i + 1:]
        return None, args
    if _kind(e) in ("Integer", "Rational", "Float"):
        return e, []
    return None, [e]


def _could_extract_minus_sign(e: Any) -> bool:
    k = _kind(e)
    if k in ("Integer", "Rational", "Float"):
        return _num_value(e) < 0
    if k == "NegativeInfinity":
        return True
    if k == "Mul":
        coeff, _ = _as_coeff_mul(e)
        return coeff is not None and _num_value(coeff) < 0
    if k == "Add":
        negative = sum(1 for a in e.args if _could_extract_minus_sign(a))
        positive = len(e.args) - negative
        if positive != negative:
            return negative > positive
        terms = as_ordered_terms(e)
        return bool(terms) and _could_extract_minus_sign(terms[0])
    return False


def precedence(e: Any) -> int:
    k = _kind(e)
    if k == "Integer":
        return PRECEDENCE["Add"] if e.p < 0 else PRECEDENCE["Atom"]
    if k == "Rational":
        return PRECEDENCE["Add"] if e.p < 0 else PRECEDENCE["Mul"]
    if k == "Float":
        return PRECEDENCE["Add"] if float(e) < 0 else PRECEDENCE["Atom"]
    if k in ("Add", "NegativeInfinity"):
        return PRECEDENCE["Add"]
    if k == "Mul":
        return PRECEDENCE["Add"] if _could_extract_minus_sign(e) else PRECEDENCE["Mul"]
    if k == "Pow":
        return PRECEDENCE["Pow"]
    if k == "Function":
        return PRECEDENCE["Func"]
    c = _core()
    if isinstance(e, c.Relational):
        return PRECEDENCE["Relational"]
    return PRECEDENCE["Atom"]


# ---------------------------------------------------------------------------
# Upstream default_sort_key
# ---------------------------------------------------------------------------


def _class_key(e: Any) -> tuple:
    k = _kind(e)
    if _is_number_kind(k):
        return (1, 0, "Number")
    if k == "Symbol":
        return (2, 0, type(e).__name__ if type(e).__name__ != "Symbol" else "Symbol")
    if k == "Constant":
        return (2, 0, _CONSTANT_CLASS[str(e._value)])
    if k == "Mul":
        return (3, 0, "Mul")
    if k == "Add":
        return (3, 1, "Add")
    if k == "Pow":
        return (3, 2, "Pow")
    if k == "Function":
        # Upstream Function.class_key: table rank, else 0 for variadic
        # (undefined) functions and 10000 for fixed-arity known ones.
        name = type(e).__name__
        c = _core()
        default = 0 if isinstance(e, c.AppliedUndef) else 10000
        return (4, _FUNCTION_RANK.get(name, default), name)
    return (5, 0, type(e).__name__)


_ONE_KEY = ((1, 0, "Number"), (0, ()), (), Fraction(1))


def sort_key(e: Any) -> tuple:
    """Port of upstream ``Expr.sort_key`` / ``default_sort_key``."""
    k = _kind(e)
    if _is_number_kind(k):
        v = _num_value(e)
        if v is None:
            v = float("nan") if k == "NaN" else float("inf")
        return ((1, 0, "Number"), (0, ()), (), v)
    if k in ("Symbol", "Constant"):
        return (_class_key(e), (1, (str(e),)), _ONE_KEY, Fraction(1))
    coeff, rest = _as_coeff_mul(e)
    coeff_v = Fraction(1) if coeff is None else _num_value(coeff)
    c = _core()
    if coeff is not None:
        if not rest:
            body = c.Integer(1)
        elif len(rest) == 1:
            body = rest[0]
        else:
            body = _MulView(rest)
    else:
        body = e if k != "Mul" else _MulView(rest)
    bk = _kind(body) if not isinstance(body, _MulView) else "Mul"
    exp_key = _ONE_KEY
    if bk == "Pow":
        base, ex = body.args
        exp_key = sort_key(ex)
        body = base
        bk = _kind(body)
    if bk in ("Symbol", "Constant") or (_is_number_kind(bk)):
        args: tuple = (str(body),)
    elif isinstance(body, _MulView):
        args = tuple(sort_key(a) for a in sorted(body.factors, key=sort_key))
    elif bk == "Add":
        args = tuple(sort_key(a) for a in as_ordered_terms(body))
    elif bk == "Mul":
        args = tuple(sort_key(a) for a in sorted(body.args, key=sort_key))
    else:
        args = tuple(sort_key(a) for a in body.args)
    ck = (3, 0, "Mul") if isinstance(body, _MulView) else _class_key(body)
    return (ck, (len(args), args), exp_key, coeff_v)


class _MulView:
    """Product of factors without constructing a shell Mul."""

    __slots__ = ("factors",)

    def __init__(self, factors: list) -> None:
        self.factors = factors


# ---------------------------------------------------------------------------
# Upstream as_ordered_terms (order=None)
# ---------------------------------------------------------------------------


def _complex_value(e: Any) -> complex | None:
    """Numeric value of a symbol-free factor, or None."""
    k = _kind(e)
    if k in ("Integer", "Rational"):
        return complex(e.p / e.q)
    if k == "Float":
        return complex(float(e))
    if k == "Infinity":
        return complex(float("inf"))
    if k == "NegativeInfinity":
        return complex(float("-inf"))
    if k == "Constant":
        name = str(e._value)
        if name == "I":
            return 1j
    if k in ("Symbol", "NaN", "ComplexInfinity", "Other"):
        return None
    if getattr(e, "free_symbols", None):
        return None
    c = _core()
    if any(isinstance(a, c.AppliedUndef) for a in _preorder(e)):
        return None
    try:
        re_part = float(e)
        return complex(re_part)
    except Exception:
        pass
    try:
        value = complex(e.evalf())
        return value
    except Exception:
        return None



def _preorder(e: Any):
    yield e
    for a in getattr(e, "args", ()):
        yield from _preorder(a)


def _decompose_power(f: Any) -> tuple[Any, Any]:
    """Upstream ``decompose_power`` returning (base key object, exponent)."""
    k = _kind(f)
    c = _core()
    if k == "Pow":
        base, ex = f.args
        ek = _kind(ex)
        if ek == "Integer":
            return base, int(ex.p)
        if ek == "Rational":
            return ("rootpow", base, int(ex.q)), int(ex.p)
        if ek == "Float":
            return f, 1
        coeff, rest = _as_coeff_mul(ex)
        if coeff is not None and _kind(coeff) in ("Integer", "Rational"):
            tail = rest[0] if len(rest) == 1 else _MulView(rest)
            if coeff.p == -1 and coeff.q == 1:
                return ("pow", base, tail), -1
            if not (coeff.p == 1 and coeff.q == 1):
                return ("pow", base, ("scaled", int(coeff.q), tail)), int(coeff.p)
        return f, 1
    if k == "Function" and type(f).__name__ == "exp":
        arg = f.args[0]
        coeff, rest = _as_coeff_mul(arg)
        if coeff is not None and _kind(coeff) in ("Integer", "Rational") and rest:
            tail = rest[0] if len(rest) == 1 else _MulView(rest)
            if coeff.p == -1 and coeff.q == 1:
                return ("exp", tail), -1
            if not (coeff.p == 1 and coeff.q == 1):
                return ("exp", ("scaled", int(coeff.q), tail)), int(coeff.p)
        return f, 1
    del c
    return f, 1


def _gen_key(g: Any) -> tuple:
    if isinstance(g, tuple):
        tag = g[0]
        if tag == "rootpow":
            _, base, qd = g
            return ((3, 2, "Pow"), (1, (sort_key(base),)), ((1, 0, "Number"), (0, ()), (), Fraction(1, qd)), Fraction(1))
        if tag == "pow":
            _, base, tail = g
            return ((3, 2, "Pow"), (1, (sort_key(base),)), _gen_key(tail), Fraction(1))
        if tag == "exp":
            _, tail = g
            return ((4, 10, "exp"), (1, (_gen_key(tail),)), _ONE_KEY, Fraction(1))
        if tag == "scaled":
            _, qd, tail = g
            inner = _gen_key(tail)
            return inner[:3] + (Fraction(1, qd),)
    if isinstance(g, _MulView):
        args = tuple(sort_key(a) for a in sorted(g.factors, key=sort_key))
        return ((3, 0, "Mul"), (len(args), args), _ONE_KEY, Fraction(1))
    return sort_key(g)


def _gen_identity(g: Any) -> Any:
    if isinstance(g, tuple):
        return tuple(_gen_identity(x) for x in g)
    if isinstance(g, _MulView):
        return ("mul",) + tuple(sorted((_gen_identity(f) for f in g.factors), key=repr))
    return g


def as_ordered_terms(e: Any) -> list:
    """Port of upstream ``Expr.as_ordered_terms(order=None)`` for an Add."""
    if _kind(e) != "Add":
        return [e]
    args = list(e.args)
    # Special case: Add(Number, Mul(Number, expr)), first positive, second
    # negative, keeps the number first ("1 - x", "2 - sqrt(3)").
    numbers = [a for a in args if _is_number_kind(_kind(a)) or _kind(a) == "Constant"]
    others = [a for a in args if not (_is_number_kind(_kind(a)) or _kind(a) == "Constant")]
    ordered = numbers + others
    if (
        len(ordered) == 2
        and (_is_number_kind(_kind(ordered[0])) or _kind(ordered[0]) == "Constant")
        and _kind(ordered[1]) == "Mul"
    ):
        margs = list(ordered[1].args)
        mnums = [a for a in margs if _is_number_kind(_kind(a))]
        mrest = [a for a in margs if not _is_number_kind(_kind(a))]
        msorted = mnums + mrest
        first = ordered[0]
        first_positive = (
            _kind(first) == "Constant" and str(first._value) != "I"
        ) or (_num_value(first) is not None and _num_value(first) > 0)
        if (
            len(msorted) == 2
            and _kind(msorted[0]) in ("Integer", "Rational", "Float")
            and first_positive
            and _num_value(msorted[0]) < 0
        ):
            return ordered

    orders = [a for a in args if _kind(a) == "Function" and type(a).__name__ == "Order"]
    if orders:
        rest = [a for a in args if not (_kind(a) == "Function" and type(a).__name__ == "Order")]
        # Upstream: with Order terms present both groups sort descending.
        return _ordered_plain_terms(rest, reverse=True) + orders
    return _ordered_plain_terms(args)


def _ordered_plain_terms(args: list, reverse: bool = False) -> list:
    terms = []
    gens: dict = {}
    for term in args:
        coeff, rest = _as_coeff_mul(term)
        cval = complex(_num_value(coeff)) if coeff is not None else complex(1)
        cpart: dict = {}
        for factor in rest:
            v = _complex_value(factor)
            if v is not None:
                cval *= v
                continue
            base, ex = _decompose_power(factor)
            ident = _gen_identity(base)
            cpart[ident] = ex
            gens.setdefault(ident, base)
        terms.append((term, cval, cpart))
    gen_list = sorted(gens.items(), key=lambda kv: _gen_key(kv[1]))
    idents = [ident for ident, _ in gen_list]

    def key(t: tuple) -> tuple:
        _, cval, cpart = t
        monom = tuple(-cpart.get(ident, 0) for ident in idents)
        re_, im_ = cval.real, cval.imag
        return (monom, ((bool(im_), im_), (re_, im_)))

    return [t[0] for t in sorted(terms, key=key, reverse=reverse)]


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _strip_float(text: str) -> str:
    if "e" in text:
        mant, exp = text.split("e", 1)
        if "." in mant:
            mant = mant.rstrip("0")
            if mant.endswith("."):
                mant += "0"
        return mant + "e" + exp
    if "." in text:
        text = text.rstrip("0")
        if text.endswith("."):
            text += "0"
    return text


class StrPrinter:
    def __init__(self, fallback: Callable[[Any], str]) -> None:
        self._fallback = fallback
        self._level = 0

    def doprint(self, e: Any) -> str:
        return self._print(e)

    def _print(self, e: Any) -> str:
        self._level += 1
        try:
            return self._dispatch(e)
        finally:
            self._level -= 1

    def parenthesize(self, item: Any, level: float, strict: bool = False) -> str:
        p = precedence(item)
        if p < level or (not strict and p <= level):
            return "(%s)" % self._print(item)
        return self._print(item)

    def _dispatch(self, e: Any) -> str:
        c = _core()
        if not isinstance(e, c.Basic):
            return str(e)
        if _is_held(e):
            return self._fallback(e)
        k = _kind(e)
        if k == "Add":
            return self._print_Add(e)
        if k == "Mul":
            return self._print_Mul(e)
        if k == "Pow":
            return self._print_Pow(e)
        if k == "Float":
            text = self._fallback(e)
            return _strip_float(text) if self._level > 1 else text
        if k == "Function" and type(e).__name__ not in ("Derivative",):
            return self._print_Function(e)
        return self._fallback(e)

    def _print_Function(self, e: Any) -> str:
        if getattr(e, "_struct_args", None) is not None:
            return self._fallback(e)
        name = type(e).__name__
        if name == "Order":
            args = e.args
            if len(args) == 3:
                return "O(%s, (%s, %s))" % (self._print(args[0]), self._print(args[1]), self._print(args[2]))
            return "O(%s)" % self._print(args[0])
        return name + "(%s)" % ", ".join(self._print(a) for a in e.args)

    def _print_Add(self, e: Any) -> str:
        terms = as_ordered_terms(e)
        prec = PRECEDENCE["Add"]
        parts: list[str] = []
        for term in terms:
            t = self._print(term)
            if t.startswith("-") and _kind(term) != "Add":
                sign = "-"
                t = t[1:]
            else:
                sign = "+"
            if precedence(term) < prec or _kind(term) == "Add":
                parts.extend([sign, "(%s)" % t])
            else:
                parts.extend([sign, t])
        sign = parts.pop(0)
        if sign == "+":
            sign = ""
        return sign + " ".join(parts)

    def _print_Mul(self, e: Any) -> str:
        c = _core()
        prec = precedence(e)
        args = list(e.args)
        coeff, rest = _as_coeff_mul(e)
        sign = ""
        cval = None
        if coeff is not None:
            cval = _num_value(coeff)
            if cval < 0:
                sign = "-"
                cval = -cval
        factors: list = []
        if cval == float("inf"):
            factors.append(c.oo)
        elif cval is not None and cval != 1:
            if isinstance(cval, Fraction):
                factors.append(c.Rational(cval.numerator, cval.denominator))
            else:
                factors.append(c.Float(cval, coeff._dps) if hasattr(coeff, "_dps") else c.Float(cval))
        factors.extend(sorted(rest, key=sort_key))
        if not factors:
            factors = [c.Integer(1)]
        del args

        a: list = []
        b: list = []
        pow_paren: list = []
        for item in factors:
            k = _kind(item)
            if k == "Pow":
                base, ex = item.args
                ecoeff, erest = _as_coeff_mul(ex)
                ecv = _num_value(ecoeff) if ecoeff is not None else None
                if ecv is not None and ecv < 0:
                    if _kind(ex) == "Integer" and ex.p == -1:
                        if _kind(base) in ("Mul", "Pow") and len(base.args) != 1:
                            pow_paren.append(base)
                        b.append(base)
                    else:
                        b.append(_PowView(base, _negated_exponent(ex)))
                    continue
                a.append(item)
            elif k in ("Integer", "Rational"):
                if item.p != 1:
                    a.append(c.Integer(item.p))
                if item.q != 1:
                    b.append(c.Integer(item.q))
            else:
                a.append(item)
        if not a:
            a = [c.Integer(1)]

        if len(a) == 1 and sign == "-":
            a_str = [self._paren_any(a[0], 0.5 * (PRECEDENCE["Pow"] + PRECEDENCE["Mul"]))]
        else:
            a_str = [self._paren_any(x, prec) for x in a]
        b_str = [self._paren_any(x, prec) for x in b]
        for base in pow_paren:
            for i, x in enumerate(b):
                if x is base and not (b_str[i].startswith("(") and b_str[i].endswith(")")):
                    b_str[i] = "(%s)" % b_str[i]
        if not b:
            return sign + "*".join(a_str)
        if len(b) == 1:
            return sign + "*".join(a_str) + "/" + b_str[0]
        return sign + "*".join(a_str) + "/(%s)" % "*".join(b_str)

    def _paren_any(self, item: Any, level: float) -> str:
        if isinstance(item, _PowView):
            p = PRECEDENCE["Pow"]
            text = self._print_pow_parts(item.base, item.exp)
            return "(%s)" % text if p <= level else text
        return self.parenthesize(item, level)

    def _print_Pow(self, e: Any) -> str:
        base, ex = e.args
        return self._print_pow_parts(base, ex)

    def _print_pow_parts(self, base: Any, ex: Any) -> str:
        prec = PRECEDENCE["Pow"]
        if isinstance(ex, _ExpView):
            exp_text = self._print_exp_view(ex)
            exp_prec = ex.precedence()
            exp_str = "(%s)" % exp_text if exp_prec <= prec else exp_text
            return "%s**%s" % (self.parenthesize(base, prec), exp_str)
        k = _kind(ex)
        if k == "Rational" and ex.p == 1 and ex.q == 2:
            return "sqrt(%s)" % self._print(base)
        if k == "Rational" and ex.p == -1 and ex.q == 2:
            return "1/sqrt(%s)" % self._print(base)
        if k == "Integer" and ex.p == -1:
            return "1/%s" % self.parenthesize(base, prec)
        return "%s**%s" % (self.parenthesize(base, prec), self.parenthesize(ex, prec))

    def _print_exp_view(self, ex: "_ExpView") -> str:
        return ex.render(self)


class _PowView:
    """``base**exp`` with a (possibly unconstructed) negated exponent."""

    __slots__ = ("base", "exp")

    def __init__(self, base: Any, exp: Any) -> None:
        self.base = base
        self.exp = exp


class _ExpView:
    """A negated product exponent ``-c*rest`` rendered without construction."""

    __slots__ = ("coeff", "rest")

    def __init__(self, coeff: Fraction | float | None, rest: list) -> None:
        self.coeff = coeff
        self.rest = rest

    def precedence(self) -> int:
        if not self.rest:
            if isinstance(self.coeff, Fraction) and self.coeff.denominator != 1:
                return PRECEDENCE["Mul"]
            return PRECEDENCE["Atom"]
        if len(self.rest) == 1 and (self.coeff is None or self.coeff == 1):
            return precedence(self.rest[0])
        return PRECEDENCE["Mul"]

    def render(self, printer: StrPrinter) -> str:
        c = _core()
        if len(self.rest) == 1 and (self.coeff is None or self.coeff == 1):
            return printer._print(self.rest[0])
        if not self.rest:
            v = self.coeff
            if isinstance(v, Fraction):
                return printer._print(c.Rational(v.numerator, v.denominator))
            return printer._print(c.Float(v))
        mul = _MulView(([] if self.coeff in (None, 1) else [self.coeff]) + list(self.rest))
        return _render_mulview(printer, mul)


def _render_mulview(printer: StrPrinter, mul: _MulView) -> str:
    c = _core()
    factors = []
    for f in mul.factors:
        if isinstance(f, Fraction):
            factors.append(c.Rational(f.numerator, f.denominator))
        elif isinstance(f, float):
            factors.append(c.Float(f))
        else:
            factors.append(f)
    return printer._print(c.Mul(*factors))


def _negated_exponent(ex: Any) -> Any:
    """Exponent ``-ex`` for a negative-coefficient exponent."""
    c = _core()
    k = _kind(ex)
    if k in ("Integer", "Rational"):
        return c.Rational(-ex.p, ex.q)
    if k == "Float":
        return c.Float(-float(ex))
    coeff, rest = _as_coeff_mul(ex)
    v = -_num_value(coeff)
    return _ExpView(v, rest)


def sstr(expr: Any, fallback: Callable[[Any], str] | None = None) -> str:
    if fallback is None:
        fallback = str
    return StrPrinter(fallback).doprint(expr)


__all__ = ["StrPrinter", "as_ordered_terms", "precedence", "sort_key", "sstr"]
