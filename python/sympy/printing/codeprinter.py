"""Code printers (upstream ``pycode``, ``ccode``, ``fcode``, ``jscode``,
``octave_code``, ``rust_code``, ``julia_code``, ``mathematica_code``).

One recursive printer with per-language hooks for numbers, powers,
function names, constants, ``Piecewise`` and ``Max``/``Min``. Products
follow upstream ``CodePrinter._print_Mul`` (numerator/denominator split,
rational coefficients printed as ratios); sums use the canonical term
order. Unsupported constructs raise NotImplementedError rather than
emitting code that means something else.
"""

from __future__ import annotations

from typing import Any

from .str import as_ordered_terms, precedence, PRECEDENCE


def _balanced(text: str) -> bool:
    depth = 0
    for ch in text:
        depth += ch == "("
        depth -= ch == ")"
        if depth < 0:
            return False
    return depth == 0


class CodePrinter:
    language = "generic"
    functions: dict = {}
    constants: dict = {}
    power = "{base}**{exp}"
    sqrt_name = "sqrt"

    def __init__(self, settings: dict | None = None) -> None:
        self.settings = dict(settings or {})
        self.preamble: list = []

    # -- entry -------------------------------------------------------------
    def doprint(self, expr: Any, assign_to: Any = None) -> str:
        from ..core import sympify

        body = self._print(sympify(expr))
        if assign_to is not None:
            body = self._assign(str(assign_to), body)
        return body

    def _assign(self, target: str, body: str) -> str:
        return "%s = %s" % (target, body)

    # -- helpers -----------------------------------------------------------
    def parenthesize(self, item: Any, level: float, strict: bool = False) -> str:
        p = precedence(item)
        text = self._print(item)
        if p < level or (not strict and p <= level):
            if text.startswith("(") and text.endswith(")") and _balanced(text[1:-1]):
                return text
            return "(%s)" % text
        return text

    def _print(self, e: Any) -> str:
        from ..core import Add, Float, Function, Integer, Mul, Pow, Rational, Symbol

        name = type(e).__name__
        hook = getattr(self, "_print_" + name, None)
        if hook is not None:
            return hook(e)
        if isinstance(e, Integer):
            return str(int(e))
        if isinstance(e, Rational):
            return self._rational(int(e.p), int(e.q))
        if isinstance(e, Float):
            return self._float(e)
        if isinstance(e, Symbol):
            return e.name
        if isinstance(e, Add):
            return self._print_Add(e)
        if isinstance(e, Mul):
            return self._print_Mul(e)
        if isinstance(e, Pow):
            return self._print_Pow(e)
        text = str(e)
        if text in self.constants:
            return self.constants[text]
        if isinstance(e, Function):
            return self._print_Function(e)
        if text in ("oo", "-oo", "zoo", "nan"):
            return self._print_special(text)
        raise NotImplementedError("%s: unsupported expression %s" % (self.language, text))

    def _print_special(self, text: str) -> str:
        raise NotImplementedError("%s: %s has no code form" % (self.language, text))

    def _rational(self, p: int, q: int) -> str:
        return "%d/%d" % (p, q)

    def _float(self, e: Any) -> str:
        return str(e)

    def _print_Add(self, e: Any) -> str:
        prec = precedence(e)
        parts = []
        for i, t in enumerate(as_ordered_terms(e)):
            text = self._print(t) if precedence(t) >= prec else "(%s)" % self._print(t)
            if i == 0:
                parts.append(text)
            elif text.startswith("-"):
                parts.append("- " + text[1:])
            else:
                parts.append("+ " + text)
        return " ".join(parts)

    def _print_Mul(self, e: Any) -> str:
        from ..core import Integer, Mul, Pow, Rational

        c, rest = e.as_coeff_Mul()
        sign = ""
        if isinstance(c, Rational) and c < 0:
            sign = "-"
            e = (-c) * rest if rest != 1 else -c
            if not isinstance(e, Mul):
                return sign + self.parenthesize(e, PRECEDENCE["Mul"], strict=True)
        prec = precedence(e)
        from .str import sort_key as _sk

        factors = list(e.args) if isinstance(e, Mul) else [e]
        lead = [f for f in factors if isinstance(f, Rational)]
        factors = lead + sorted([f for f in factors if not isinstance(f, Rational)], key=_sk)
        num, den = [], []
        for f in factors:
            if isinstance(f, Pow) and isinstance(f.args[1], Rational) and f.args[1] < 0:
                den.append(f.args[0] if f.args[1] == -1 else Pow(f.args[0], -f.args[1]))
            elif isinstance(f, Rational) and not isinstance(f, Integer) and self._split_rational_mul(f):
                if f.p != 1:
                    num.append(Integer(f.p))
                den.append(Integer(f.q))
            else:
                num.append(f)
        num = num or [Integer(1)]
        a = [self.parenthesize(x, prec) for x in num]
        b = [self.parenthesize(x, prec) for x in den]
        if not den:
            return sign + self._mul_join(num, a)
        div = self._div(den)
        if len(den) == 1:
            return sign + self._mul_join(num, a) + div + b[0]
        return sign + self._mul_join(num, a) + div + "(%s)" % self._mul_join(den, b)

    def _mul_join(self, items: list, texts: list) -> str:
        return "*".join(texts)

    def _split_rational_mul(self, r: Any) -> bool:
        return False

    def _div(self, den: list) -> str:
        return "/"

    def _print_Pow(self, e: Any) -> str:
        from ..core import Rational

        base, ex = e.args
        if ex == Rational(1, 2):
            return "%s(%s)" % (self._fn(self.sqrt_name), self._print(base))
        if ex == -1:
            return "%s/%s" % (self._one(), self.parenthesize(base, PRECEDENCE["Mul"]))
        return self._pow(base, ex)

    def _one(self) -> str:
        return "1"

    def _pow(self, base: Any, ex: Any) -> str:
        return "%s**%s" % (self.parenthesize(base, PRECEDENCE["Pow"]), self.parenthesize(ex, PRECEDENCE["Pow"]))

    def _fn(self, name: str) -> str:
        return name

    def _print_Function(self, e: Any) -> str:
        name = type(e).__name__
        target = self.functions.get(name)
        if target is None:
            raise NotImplementedError("%s: function %s is not supported" % (self.language, name))
        return "%s(%s)" % (target, ", ".join(self._print(a) for a in e.args))

    def _print_exp(self, e: Any) -> str:
        return "%s(%s)" % (self.functions.get("exp", "exp"), self._print(e.args[0]))


# ---------------------------------------------------------------------------


class PythonCodePrinter(CodePrinter):
    language = "Python"
    _module = "math"
    _names = {
        "sin": "sin", "cos": "cos", "tan": "tan", "asin": "asin", "acos": "acos", "atan": "atan",
        "atan2": "atan2", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "asinh": "asinh",
        "acosh": "acosh", "atanh": "atanh", "exp": "exp", "log": "log", "floor": "floor",
        "ceiling": "ceil", "erf": "erf", "erfc": "erfc", "gamma": "gamma", "factorial": "factorial",
        "loggamma": "lgamma",
    }

    def __init__(self, settings: dict | None = None) -> None:
        super().__init__(settings)
        self.functions = {k: "%s.%s" % (self._module, v) for k, v in self._names.items()}
        self.functions.update(Abs="abs", Max="max", Min="min")
        self.constants = {"pi": "%s.pi" % self._module, "E": "%s.e" % self._module}

    def _fn(self, name: str) -> str:
        return "%s.%s" % (self._module, name)

    def _print_special(self, text: str) -> str:
        return {"oo": "math.inf", "-oo": "-math.inf", "nan": "math.nan"}.get(text) or super()._print_special(text)

    def _print_Piecewise(self, e: Any) -> str:
        parts = []
        args = list(e.args)
        last = args[-1]
        out = "(%s)" % self._print(last.expr)
        for pair in reversed(args[:-1]):
            out = "((%s) if (%s) else %s)" % (self._print(pair.expr), self._cond(pair.cond), out)
        if len(args) == 1:
            return out
        return out if out.startswith("((") else "(%s)" % out

    def _cond(self, c: Any) -> str:
        return str(c)


class CCodePrinter(CodePrinter):
    language = "C"
    functions = {
        "sin": "sin", "cos": "cos", "tan": "tan", "asin": "asin", "acos": "acos", "atan": "atan",
        "atan2": "atan2", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "asinh": "asinh",
        "acosh": "acosh", "atanh": "atanh", "exp": "exp", "log": "log", "floor": "floor",
        "ceiling": "ceil", "Abs": "fabs", "erf": "erf", "erfc": "erfc", "gamma": "tgamma",
        "Max": "fmax", "Min": "fmin",
    }
    constants = {"pi": "M_PI", "E": "M_E"}

    def _rational(self, p: int, q: int) -> str:
        return "%d.0/%d.0" % (p, q)

    def _one(self) -> str:
        return "1.0"

    def _pow(self, base: Any, ex: Any) -> str:
        return "pow(%s, %s)" % (self._print(base), self._print(ex))

    def _print_special(self, text: str) -> str:
        return {"oo": "HUGE_VAL", "-oo": "-HUGE_VAL", "nan": "NAN"}.get(text) or super()._print_special(text)

    def _print_Piecewise(self, e: Any) -> str:
        args = list(e.args)
        out = self._print(args[-1].expr)
        for pair in reversed(args[:-1]):
            out = "((%s) ? (\n   %s\n)\n: (\n   %s\n))" % (str(pair.cond), self._print(pair.expr), out)
        return out

    def _assign(self, target: str, body: str) -> str:
        return "%s = %s;" % (target, body)


class JavascriptCodePrinter(CCodePrinter):
    language = "JavaScript"
    functions = {k: "Math." + v for k, v in {
        "sin": "sin", "cos": "cos", "tan": "tan", "asin": "asin", "acos": "acos", "atan": "atan",
        "atan2": "atan2", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "asinh": "asinh",
        "acosh": "acosh", "atanh": "atanh", "exp": "exp", "log": "log", "floor": "floor",
        "ceiling": "ceil", "Abs": "abs", "Max": "max", "Min": "min",
    }.items()}
    constants = {"pi": "Math.PI", "E": "Math.E"}
    sqrt_name = "Math.sqrt"

    def _rational(self, p: int, q: int) -> str:
        return "%d/%d" % (p, q)

    def _one(self) -> str:
        return "1"

    def _pow(self, base: Any, ex: Any) -> str:
        return "Math.pow(%s, %s)" % (self._print(base), self._print(ex))

    def _print_special(self, text: str) -> str:
        return {"oo": "Number.POSITIVE_INFINITY", "-oo": "Number.NEGATIVE_INFINITY", "nan": "NaN"}.get(text) or CodePrinter._print_special(self, text)


class FCodePrinter(CodePrinter):
    language = "Fortran"
    functions = {
        "sin": "sin", "cos": "cos", "tan": "tan", "asin": "asin", "acos": "acos", "atan": "atan",
        "atan2": "atan2", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "exp": "exp", "log": "log",
        "Abs": "abs", "Max": "max", "Min": "min", "floor": "floor", "ceiling": "ceiling",
        "erf": "erf", "gamma": "gamma",
    }

    def _rational(self, p: int, q: int) -> str:
        return "%d.0d0/%d.0d0" % (p, q)

    def _one(self) -> str:
        return "1d0"

    def _print(self, e: Any) -> str:
        if str(e) == "pi":
            self.preamble.append("parameter (pi = 3.1415926535897932d0)")
            return "pi"
        return super()._print(e)

    def _print_Pow(self, e: Any) -> str:
        from ..core import Rational

        base, ex = e.args
        if isinstance(ex, Rational) and not ex.q == 1:
            if ex == Rational(1, 2):
                return "sqrt(%s)" % self._print(base)
            return "%s**(%s)" % (self.parenthesize(base, PRECEDENCE["Pow"]), self._rational(int(ex.p), int(ex.q)))
        if ex == -1:
            return "1d0/%s" % self.parenthesize(base, PRECEDENCE["Mul"])
        exs = self._print(ex)
        if exs.startswith("-"):
            exs = "(%s)" % exs
        return "%s**%s" % (self.parenthesize(base, PRECEDENCE["Pow"]), exs)

    def doprint(self, expr: Any, assign_to: Any = None) -> str:
        body = super().doprint(expr, assign_to)
        lines = []
        for p in dict.fromkeys(self.preamble):
            lines.append("      " + p)
        lines.append("      " + body)
        return "\n".join(lines)


class OctaveCodePrinter(CodePrinter):
    language = "Octave"
    functions = {
        "sin": "sin", "cos": "cos", "tan": "tan", "asin": "asin", "acos": "acos", "atan": "atan",
        "atan2": "atan2", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "exp": "exp", "log": "log",
        "Abs": "abs", "Max": "max", "Min": "min", "floor": "floor", "ceiling": "ceil",
        "erf": "erf", "gamma": "gamma",
    }
    constants = {"pi": "pi", "E": "exp(1)"}

    def _split_rational_mul(self, r: Any) -> bool:
        return True

    def _div(self, den: list) -> str:
        from ..core import Rational

        return "/" if all(isinstance(d, Rational) for d in den) else "./"

    def _mul_join(self, items: list, texts: list) -> str:
        from ..core import Rational

        out = texts[0]
        for prev, t in zip(items, texts[1:]):
            out += ("*" if isinstance(prev, Rational) else ".*") + t
        return out

    def _print_Pow(self, e: Any) -> str:
        from ..core import Rational

        base, ex = e.args
        if ex == Rational(1, 2):
            return "sqrt(%s)" % self._print(base)
        if ex == -1:
            return "1./%s" % self.parenthesize(base, PRECEDENCE["Mul"])
        exs = self._print(ex)
        if exs.startswith("-") or (isinstance(ex, Rational) and ex.q != 1):
            exs = "(%s)" % exs
        return "%s.^%s" % (self.parenthesize(base, PRECEDENCE["Pow"]), exs)

    def _print_Piecewise(self, e: Any) -> str:
        args = list(e.args)
        out = "(%s)" % self._print(args[-1].expr)
        for pair in reversed(args[:-1]):
            c = str(pair.cond)
            out = "((%s).*(%s) + (~(%s)).*%s)" % (c, self._print(pair.expr), c, out)
        return out



class JuliaCodePrinter(OctaveCodePrinter):
    language = "Julia"

    def _mul_join(self, items: list, texts: list) -> str:
        from ..core import Rational

        out = texts[0]
        for prev, t in zip(items, texts[1:]):
            out += (" * " if isinstance(prev, Rational) else " .* ") + t
        return out

    def _split_rational_mul(self, r: Any) -> bool:
        return r.p == 1

    def _rational(self, p: int, q: int) -> str:
        return "(%d // %d)" % (p, q)

    def _print_Pow(self, e: Any) -> str:
        from ..core import Rational

        base, ex = e.args
        if ex == Rational(1, 2):
            return "sqrt(%s)" % self._print(base)
        if ex == -1:
            return "1 ./ %s" % self.parenthesize(base, PRECEDENCE["Mul"])
        if isinstance(ex, Rational) and ex.q != 1:
            exs = "(%d // %d)" % (ex.p, ex.q)
        else:
            exs = self._print(ex)
            if exs.startswith("-"):
                exs = "(%s)" % exs
        return "%s .^ %s" % (self.parenthesize(base, PRECEDENCE["Pow"]), exs)

    def _div(self, den: list) -> str:
        from ..core import Rational

        return " / " if all(isinstance(d, Rational) for d in den) else " ./ "

    def _print_Piecewise(self, e: Any) -> str:
        args = list(e.args)
        out = "(%s)" % self._print(args[-1].expr)
        for pair in reversed(args[:-1]):
            out = "((%s) ? (%s) : %s)" % (str(pair.cond), self._print(pair.expr), out)
        return out


class MathematicaCodePrinter(CodePrinter):
    language = "Mathematica"
    functions = {
        "sin": "Sin", "cos": "Cos", "tan": "Tan", "asin": "ArcSin", "acos": "ArcCos", "atan": "ArcTan",
        "sinh": "Sinh", "cosh": "Cosh", "tanh": "Tanh", "exp": "Exp", "log": "Log", "Abs": "Abs",
        "Max": "Max", "Min": "Min", "gamma": "Gamma", "erf": "Erf",
    }
    constants = {"pi": "Pi", "E": "E"}

    def _print_Function(self, e: Any) -> str:
        name = type(e).__name__
        if name == "atan2":
            y, x = e.args
            return "ArcTan[%s, %s]" % (self._print(x), self._print(y))
        target = self.functions.get(name, name)
        return "%s[%s]" % (target, ", ".join(self._print(a) for a in e.args))

    def _print_exp(self, e: Any) -> str:
        return "Exp[%s]" % self._print(e.args[0])

    def _print_Pow(self, e: Any) -> str:
        base, ex = e.args
        return "%s^%s" % (self.parenthesize(base, PRECEDENCE["Pow"]), self.parenthesize(ex, PRECEDENCE["Pow"]))


class RustCodePrinter(CodePrinter):
    language = "Rust"
    _methods = {
        "sin": "sin", "cos": "cos", "tan": "tan", "asin": "asin", "acos": "acos", "atan": "atan",
        "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "exp": "exp", "log": "ln", "Abs": "abs",
        "floor": "floor", "ceiling": "ceil", "sqrt": "sqrt",
    }
    constants = {"pi": "PI", "E": "E"}

    def _recv(self, a: Any) -> str:
        s = self._print(a)
        from ..core import Symbol

        return s if isinstance(a, Symbol) else "(%s)" % s

    def _print_Function(self, e: Any) -> str:
        name = type(e).__name__
        if name == "atan2":
            y, x = e.args
            return "%s.atan2(%s)" % (self._recv(y), self._print(x))
        m = self._methods.get(name)
        if m is None:
            raise NotImplementedError("Rust: function %s is not supported" % name)
        return "%s.%s()" % (self._recv(e.args[0]), m)

    def _print_exp(self, e: Any) -> str:
        return "%s.exp()" % self._recv(e.args[0])

    def _rational(self, p: int, q: int) -> str:
        return "%d_f64/%d.0" % (p, q)

    def _print_Mul(self, e: Any) -> str:
        from ..core import Mul, Pow, Rational

        factors = list(e.args) if isinstance(e, Mul) else [e]
        den = [f for f in factors if isinstance(f, Pow) and f.args[1] == -1]
        num = [f for f in factors if f not in den]
        if den and len(num) > 1:
            # Upstream rust_code: x*y.recip() once the numerator is a product.
            return "*".join([self.parenthesize(f, PRECEDENCE["Mul"]) for f in num] + [self._print(d) for d in den])
        return super()._print_Mul(e)

    def _print_Pow(self, e: Any) -> str:
        from ..core import Integer, Rational

        base, ex = e.args
        if ex == Rational(1, 2):
            return "%s.sqrt()" % self._recv(base)
        if ex == -1:
            return "%s.recip()" % self._recv(base)
        if isinstance(ex, Integer):
            return "%s.powi(%s)" % (self._recv(base), self._print(ex))
        return "%s.powf(%s)" % (self._recv(base), self._print(ex))


def _make(cls: Any) -> Any:
    def printer(expr: Any, assign_to: Any = None, **settings: Any) -> str:
        return cls(settings).doprint(expr, assign_to)

    return printer


pycode = _make(PythonCodePrinter)
ccode = _make(CCodePrinter)
fcode = _make(FCodePrinter)
jscode = _make(JavascriptCodePrinter)
octave_code = _make(OctaveCodePrinter)
julia_code = _make(JuliaCodePrinter)
mathematica_code = _make(MathematicaCodePrinter)
rust_code = _make(RustCodePrinter)


def print_ccode(expr: Any, **settings: Any) -> None:
    print(ccode(expr, **settings))


def print_fcode(expr: Any, **settings: Any) -> None:
    print(fcode(expr, **settings))


__all__ = [
    "ccode", "fcode", "jscode", "julia_code", "mathematica_code", "octave_code", "print_ccode",
    "print_fcode", "pycode", "rust_code",
]
