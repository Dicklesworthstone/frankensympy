"""Polynomial tools and representations for FrankenSymPy (WS08, WS09)."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, List, Optional, Sequence, Tuple, Union
from .polyerrors import PolynomialError
from ..core import (
    Basic,
    Mul,
    Pow,
    Expr,
    Function,
    Integer,
    Rational,
    Symbol,
    _native,
    _native_expr,
    _native_symbol_key,
    _parse_result,
    _require_symbol,
    _wrap,
)


_GENS_ORDER = {
    "a": 301, "b": 302, "c": 303, "d": 304, "e": 305, "f": 306, "g": 307,
    "h": 308, "i": 309, "j": 310, "k": 311, "l": 312, "m": 313, "n": 314,
    "o": 315, "p": 216, "q": 217, "r": 218, "s": 219, "t": 220, "u": 221,
    "v": 222, "w": 223, "x": 124, "y": 125, "z": 126,
}


def _sort_gens(gens: Any) -> list:
    """Upstream ``_sort_gens`` default order: x, y, z first, then p..w,
    then a..o, then other names; a trailing number orders numerically."""
    import re as _re

    def key(g: Any) -> tuple:
        name, index = _re.match(r"^(.*?)(\d*)$", str(g)).groups()
        return (_GENS_ORDER.get(name, 1000), name, int(index) if index else 0)

    return sorted(gens, key=key)


class Poly(Basic):
    """Exact polynomial representation over rational field QQ."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __new__(cls, expr: Any, *gens: Any, **kwargs: Any) -> "Poly":
        if isinstance(expr, Poly):
            if not gens and "gens" not in kwargs:
                return expr
            kwargs.setdefault("_extra", expr._domain_syms())
            expr = expr.as_expr()

        if not gens and "gens" in kwargs and kwargs["gens"] is not None:
            gens_kw = kwargs["gens"]
            if isinstance(gens_kw, (list, tuple)):
                gens = tuple(gens_kw)
            else:
                gens = (gens_kw,)

        wrapped_expr = _wrap(_native_expr(expr))
        if gens:
            if len(gens) == 1 and isinstance(gens[0], (list, tuple)):
                generator_list = [_require_symbol(g) for g in gens[0]]
            else:
                generator_list = [_require_symbol(g) for g in gens]
        else:
            free = wrapped_expr.free_symbols
            if len(free) == 0:
                generator_list = [Symbol("x")]
            elif len(free) == 1:
                generator_list = [next(iter(free))]
            else:
                generator_list = _sort_gens(free)

        obj = object.__new__(cls)
        obj._expr = wrapped_expr
        obj._gens = tuple(generator_list)
        obj._domain = kwargs.get("domain", "QQ")
        obj._domain_given = "domain" in kwargs
        # Coefficient symbols the domain carries beyond the current
        # coefficients (upstream Poly keeps its domain through diff/div).
        obj._extra_syms = frozenset(kwargs.get("_extra", ())) - set(generator_list)
        return obj

    def _domain_syms(self) -> frozenset:
        return frozenset(self._expr.free_symbols - set(self._gens)) | self._extra_syms

    def _like(self, expr: Any, other: Any = None) -> "Poly":
        extra = self._domain_syms()
        if isinstance(other, Poly):
            extra = extra | other._domain_syms()
        return Poly(expr, *self._gens, _extra=extra)

    @property
    def gen(self) -> Symbol:
        return self._gens[0]

    @property
    def gens(self) -> Tuple[Symbol, ...]:
        return self._gens

    def as_expr(self) -> Expr:
        return self._expr

    @property
    def _parametric(self) -> bool:
        """Coefficients involve symbols outside the generators (domain
        ZZ[a, ...] / QQ(a, ...)); such polynomials take the Python lanes."""
        return bool(self._expr.free_symbols - set(self._gens))

    def _gen_index(self, gen: Any) -> int:
        return gen if isinstance(gen, int) else self._gens.index(_require_symbol(gen))

    def _univariate_coeffs(self) -> List[Any]:
        from ..core import Integer as _Integer

        terms = self.terms()
        deg = max(m[0] for m, _ in terms)
        out = [_Integer(0)] * (deg + 1)
        for m, c in terms:
            out[deg - m[0]] = out[deg - m[0]] + c
        return out

    def total_degree(self) -> Any:
        from ..core import S as _S

        if self.as_expr() == 0:
            return _S.NegativeInfinity
        return max(sum(m) for m, _ in self.terms())

    def all_coeffs(self) -> List[Any]:
        """Return all coefficients of the polynomial in descending degree order."""
        if self._parametric:
            return self._univariate_coeffs()
        raw = _native.poly_coeffs_expr(str(_native_expr(self._expr)), _native_symbol_key(self.gen))
        return [_parse_result(c) for c in raw]

    def coeffs(self) -> List[Any]:
        """Return all non-zero coefficients in descending degree order."""
        return [c for c in self.all_coeffs() if c != 0]

    def degree(self, gen: Any = 0) -> Optional[int]:
        """Return polynomial degree."""
        if self._parametric:
            from ..core import S as _S

            if self.as_expr() == 0:
                return _S.NegativeInfinity
            i = self._gen_index(gen)
            return max(m[i] for m, _ in self.terms())
        if len(self._gens) > 1:
            var_names = [_native_symbol_key(g) for g in self._gens]
            target_key = None
            if gen is not None:
                target_gen = self._gens[gen] if isinstance(gen, int) else _require_symbol(gen)
                target_key = _native_symbol_key(target_gen)
            return _native.poly_multivariate_degree_expr(
                str(_native_expr(self._expr)), var_names, target_key
            )
        target_gen = self._gens[gen] if isinstance(gen, int) else _require_symbol(gen)
        return _native.poly_degree_expr(str(_native_expr(self._expr)), _native_symbol_key(target_gen))

    def leading_coeff(self) -> Any:
        """Return the leading coefficient."""
        if self._parametric:
            return self.terms()[0][1]
        raw = _native.poly_leading_coeff_expr(str(_native_expr(self._expr)), _native_symbol_key(self.gen))
        return _parse_result(raw)

    def LC(self) -> Any:
        """Alias for leading_coeff."""
        return self.leading_coeff()

    def trailing_coeff(self) -> Any:
        """Return the trailing coefficient."""
        coeffs = self.all_coeffs()
        if coeffs:
            return coeffs[-1]
        from ..core import Integer
        return Integer(0)

    def TC(self) -> Any:
        """Alias for trailing_coeff."""
        return self.trailing_coeff()

    def EC(self) -> Any:
        """Alias for trailing_coeff."""
        return self.trailing_coeff()

    def nth(self, *coords: int) -> Any:
        """Return coefficient of the given monomial degree."""
        if len(coords) == 1:
            n = int(coords[0])
            deg = self.degree()
            if deg is None or n < 0 or n > deg:
                from ..core import Integer
                return Integer(0)
            coeffs = self.all_coeffs()
            return coeffs[deg - n]
        raise NotImplementedError("multivariate nth is not yet implemented")

    def eval(self, *args: Any) -> Any:
        """Evaluate polynomial at the given points."""
        if len(args) == 1:
            return self.as_expr().subs(self.gen, args[0])
        elif len(args) == 2:
            var, val = args
            return self.as_expr().subs(var, val)
        raise TypeError(f"eval takes 1 or 2 arguments, got {len(args)}")

    def diff(self, *specs: Any) -> "Poly":
        """Differentiate polynomial."""
        from ..core import diff
        return self._like(diff(self.as_expr(), *(specs or (self.gen,))))

    def integrate(self, *specs: Any) -> "Poly":
        """Integrate polynomial."""
        from ..integrals import integrate
        return Poly(integrate(self.as_expr(), *(specs or (self.gen,))), *self._gens)

    def subs(self, *args: Any, **kwargs: Any) -> Any:
        """Substitute into polynomial expression."""
        return self.as_expr().subs(*args, **kwargs)

    @property
    def free_symbols(self) -> set[Any]:
        return self.as_expr().free_symbols

    @property
    def is_linear(self) -> bool:
        return self.degree() == 1

    @property
    def is_quadratic(self) -> bool:
        return self.degree() == 2

    @property
    def is_zero(self) -> bool:
        if self.as_expr() == 0 or self.as_expr() == Integer(0):
            return True
        deg = self.degree()
        if deg is None:
            return True
        if len(self._gens) == 1:
            coeffs = self.all_coeffs()
            return len(coeffs) == 1 and coeffs[0] == 0
        return False

    @property
    def is_one(self) -> bool:
        if self.as_expr() == 1 or self.as_expr() == Integer(1):
            return True
        deg = self.degree()
        if deg != 0:
            return False
        if len(self._gens) == 1:
            coeffs = self.all_coeffs()
            return len(coeffs) == 1 and coeffs[0] == 1
        return False

    @property
    def is_monic(self) -> bool:
        return bool(self.leading_coeff() == 1)

    @property
    def is_univariate(self) -> bool:
        return len(self._gens) == 1

    @property
    def is_multivariate(self) -> bool:
        return len(self._gens) > 1

    @property
    def is_irreducible(self) -> bool:
        deg = self.degree()
        if deg is None or deg <= 0:
            return False
        if deg == 1:
            return True
        try:
            scale, factors = self.factor_list()
            return len(factors) == 1 and factors[0][1] == 1 and factors[0][0].degree() == deg
        except Exception:
            return False

    def content(self) -> Any:
        """Compute the content (GCD of coefficients) of this polynomial."""
        from functools import reduce
        from ..core import Add, Integer, Rational
        if self._parametric:
            g = Integer(0)
            for _, c in self.terms():
                g = c if g == 0 else gcd(g, c)
            return g
        if len(self._gens) > 1:
            expr = self._expr
            terms = expr.args if isinstance(expr, Add) else [expr]
            coeffs = [t.as_coeff_Mul(rational=True)[0] for t in terms]
        else:
            try:
                coeffs = self.all_coeffs()
            except Exception:
                expr = self._expr
                terms = expr.args if isinstance(expr, Add) else [expr]
                coeffs = [t.as_coeff_Mul(rational=True)[0] for t in terms]
        if not coeffs or all(c == 0 for c in coeffs):
            return Integer(0)
        denoms = []
        for c in coeffs:
            if isinstance(c, Rational):
                denoms.append(c.q)
            elif hasattr(c, "q"):
                denoms.append(int(c.q))
            else:
                denoms.append(1)
        common_denom = reduce(lambda a, b: (a * b) // math.gcd(a, b), denoms, 1)
        int_numers = []
        for c, d in zip(coeffs, denoms):
            mult = common_denom // d
            if isinstance(c, Rational):
                val = c.p * mult
            elif isinstance(c, Integer):
                val = int(c) * mult
            else:
                val = int(c) * mult
            int_numers.append(abs(val))
        common_gcd = reduce(math.gcd, int_numers)
        if common_denom == 1:
            return Integer(common_gcd)
        return Rational(common_gcd, common_denom)

    def primitive(self) -> Tuple[Any, "Poly"]:
        """Compute the content and primitive form of this polynomial."""
        from ..core import Integer, expand
        cont = self.content()
        if cont == 0:
            return (Integer(0), Poly(0, *self._gens))
        prim_expr = expand(cancel(self.as_expr() / cont)) if self._parametric else expand(self.as_expr() / cont)
        return (cont, self._like(prim_expr))

    @classmethod
    def from_list(cls, coeffs: Sequence[Any], gens: Any = None) -> "Poly":
        """Construct a polynomial from a list of coefficients in descending degree order."""
        if gens is None:
            gens = [Symbol("x")]
        elif isinstance(gens, (list, tuple)):
            gens = [_require_symbol(g) for g in gens]
        else:
            gens = [_require_symbol(gens)]
        gen = gens[0]
        deg = len(coeffs) - 1
        terms = []
        for i, c in enumerate(coeffs):
            d = deg - i
            c_val = _wrap(_native_expr(c))
            if c_val != 0:
                if d == 0:
                    terms.append(c_val)
                elif d == 1:
                    terms.append(c_val * gen)
                else:
                    terms.append(c_val * (gen**d))
        if not terms:
            from ..core import Integer
            expr = Integer(0)
        else:
            from functools import reduce
            expr = reduce(lambda a, b: a + b, terms)
        return cls(expr, *gens)

    @classmethod
    def from_expr(cls, expr: Any, *gens: Any, **kwargs: Any) -> "Poly":
        """Construct a Poly from an expression."""
        return cls(expr, *gens, **kwargs)

    @classmethod
    def from_poly(cls, poly: "Poly", *gens: Any, **kwargs: Any) -> "Poly":
        """Construct a Poly from another Poly."""
        return cls(poly.as_expr(), *(gens or poly.gens), **kwargs)

    def monic(self) -> "Poly":
        """Return the monic associate of this polynomial."""
        if self._parametric:
            lc = self.leading_coeff()
            out = Integer(0)
            for m, c in self.terms():
                mono = Integer(1)
                for g, k in zip(self._gens, m):
                    mono = mono * g**k
                out = out + cancel(c / lc) * mono
            return Poly(out, *self._gens)
        raw = _native.poly_monic_expr(str(_native_expr(self._expr)), _native_symbol_key(self.gen))
        return Poly(_parse_result(raw), *self._gens)

    def div(self, other: Any) -> Tuple["Poly", "Poly"]:
        """Polynomial division with remainder returning (quotient, remainder)."""
        other_expr = other.as_expr() if isinstance(other, Poly) else _wrap(_native_expr(other))
        other_gens = other._gens if isinstance(other, Poly) else ()
        if len(self._gens) == 1 and len(other_gens) <= 1 and (
            self._parametric or (other_expr.free_symbols - set(self._gens))
        ):
            q, r = _parametric_div(self._expr, other_expr, self.gen)
            return self._like(q, other), self._like(r, other)
        if len(self._gens) > 1 or len(other_gens) > 1:
            all_gens = tuple(dict.fromkeys(self._gens + other_gens))
            var_names = [_native_symbol_key(g) for g in all_gens]
            q_raw, r_raw = _native.poly_multivariate_div_rem_expr(
                str(_native_expr(self._expr)), str(other_expr), var_names
            )
            return (
                Poly(_parse_result(q_raw), *all_gens),
                Poly(_parse_result(r_raw), *all_gens),
            )
        q_raw, r_raw = _native.poly_div_rem_expr(
            str(_native_expr(self._expr)), str(other_expr), _native_symbol_key(self.gen)
        )
        return (
            Poly(_parse_result(q_raw), *self._gens),
            Poly(_parse_result(r_raw), *self._gens),
        )

    def rem(self, other: Any) -> "Poly":
        return self.div(other)[1]

    def quo(self, other: Any) -> "Poly":
        return self.div(other)[0]

    def sturm(self) -> List["Poly"]:
        return sturm(self)

    def decompose(self) -> List["Poly"]:
        return decompose(self, polys=True)

    def resultant(self, other: Any) -> Any:
        """Resultant with respect to the primary generator."""
        other_expr = other.as_expr() if isinstance(other, Poly) else _wrap(_native_expr(other))
        raw = _native.poly_resultant_expr(
            str(_native_expr(self._expr)), str(other_expr), _native_symbol_key(self.gen)
        )
        return _parse_result(raw)

    def discriminant(self) -> Any:
        """Discriminant with respect to the primary generator."""
        raw = _native.poly_discriminant_expr(str(_native_expr(self._expr)), _native_symbol_key(self.gen))
        return _parse_result(raw)

    def gcd(self, other: Any) -> "Poly":
        """Greatest common divisor (monic)."""
        other_expr = other.as_expr() if isinstance(other, Poly) else _wrap(_native_expr(other))
        other_gens = other._gens if isinstance(other, Poly) else ()
        if len(self._gens) > 1 or len(other_gens) > 1:
            all_gens = tuple(dict.fromkeys(self._gens + other_gens))
            var_names = [_native_symbol_key(g) for g in all_gens]
            raw = _native.poly_multivariate_gcd_expr(
                str(_native_expr(self._expr)), str(other_expr), var_names
            )
            return Poly(_parse_result(raw), *all_gens)
        raw = _native.poly_gcd_expr(
            str(_native_expr(self._expr)), str(other_expr), _native_symbol_key(self.gen)
        )
        return Poly(_parse_result(raw), *self._gens)

    def lcm(self, other: Any) -> "Poly":
        """Least common multiple (monic)."""
        other_expr = other.as_expr() if isinstance(other, Poly) else _wrap(_native_expr(other))
        other_gens = other._gens if isinstance(other, Poly) else ()
        if len(self._gens) > 1 or len(other_gens) > 1:
            all_gens = tuple(dict.fromkeys(self._gens + other_gens))
            var_names = [_native_symbol_key(g) for g in all_gens]
            raw = _native.poly_multivariate_lcm_expr(
                str(_native_expr(self._expr)), str(other_expr), var_names
            )
            return Poly(_parse_result(raw), *all_gens)
        raw = _native.poly_lcm_expr(
            str(_native_expr(self._expr)), str(other_expr), _native_symbol_key(self.gen)
        )
        return Poly(_parse_result(raw), *self._gens)

    def gcdex(self, other: Any) -> Tuple["Poly", "Poly", "Poly"]:
        """Extended Euclidean algorithm: returns (s, t, h) such that s*self + t*other = h."""
        other_poly = other if isinstance(other, Poly) else Poly(other, *self._gens)
        raw_s, raw_t, raw_h = _native.poly_gcdex_expr(
            str(_native_expr(self._expr)), str(other_poly._expr), _native_symbol_key(self.gen)
        )
        return (
            Poly(_parse_result(raw_s), *self._gens),
            Poly(_parse_result(raw_t), *self._gens),
            Poly(_parse_result(raw_h), *self._gens),
        )

    def half_gcdex(self, other: Any) -> Tuple["Poly", "Poly"]:
        """Half extended Euclidean algorithm: returns (s, h) such that s*self == h (mod other)."""
        s, _, h = self.gcdex(other)
        return s, h

    def compose(self, other: Any) -> "Poly":
        """Compute polynomial composition self(other)."""
        other_poly = other if isinstance(other, Poly) else Poly(other, *self._gens)
        raw = _native.poly_compose_expr(
            str(_native_expr(self._expr)), str(other_poly._expr), _native_symbol_key(self.gen)
        )
        return Poly(_parse_result(raw), *self._gens)

    def shift(self, a: Any) -> "Poly":
        """Compute polynomial shift self(x + a)."""
        raw = _native.poly_shift_expr(
            str(_native_expr(self._expr)), str(a), _native_symbol_key(self.gen)
        )
        return Poly(_parse_result(raw), *self._gens)

    def sqf_list(self) -> Tuple[Any, List[Tuple["Poly", int]]]:
        """Square-free factorization returning (scale, [(factor, multiplicity), ...])."""
        scale_raw, factors_raw = _native.poly_sqf_list_expr(
            str(_native_expr(self._expr)), _native_symbol_key(self.gen)
        )
        scale = _parse_result(scale_raw)
        factors = [
            (Poly(_parse_result(f), *self._gens), mult) for f, mult in factors_raw
        ]
        return (scale, factors)

    def sqf_part(self) -> "Poly":
        """Square-free part of the polynomial."""
        scale, factors = self.sqf_list()
        res = scale
        for f, _ in factors:
            res = res * f.as_expr()
        return Poly(res, *self._gens)

    def factor_list(self) -> Tuple[Any, List[Tuple["Poly", int]]]:
        """Factorization over Q returning (scale, [(factor, multiplicity), ...])."""
        if self._parametric or len(self._gens) > 1:
            res = _native_factor_multivariate(
                self._expr, list(self._gens) + _sort_gens(self._expr.free_symbols - set(self._gens))
            )
            if res is not None:
                scale, fl = res
                out = []
                for f, m in fl:
                    if f.free_symbols & set(self._gens):
                        out.append((self._like(f), m))
                    else:
                        scale = scale * f**m
                return scale, out
        scale_raw, factors_raw = _native.poly_factor_list_expr(
            str(_native_expr(self._expr)), _native_symbol_key(self.gen)
        )
        scale = _parse_result(scale_raw)
        factors = [
            (Poly(_parse_result(f), *self._gens), mult) for f, mult in factors_raw
        ]
        return (scale, factors)

    def factor(self) -> Expr:
        """Factor polynomial into a product of irreducible rational factors."""
        scale, factors = self.factor_list()
        if not factors:
            return scale
        terms = []
        for f, mult in factors:
            f_expr = f.as_expr()
            if mult == 1:
                terms.append(f_expr)
            else:
                terms.append(f_expr**mult)
        from ..core import _keep_coeff

        return _keep_coeff(scale, Mul(*terms))

    def roots(self) -> dict[Any, int]:
        """Compute polynomial roots over Q with multiplicities."""
        if self._parametric:
            return _parametric_roots(self)
        roots_raw = _native.poly_roots_expr(
            str(_native_expr(self._expr)), _native_symbol_key(self.gen)
        )
        return {_parse_result(r): mult for r, mult in roots_raw}

    def __add__(self, other: Any) -> "Poly":
        other_expr = other.as_expr() if isinstance(other, Poly) else other
        return self._like(self.as_expr() + other_expr, other)

    def __radd__(self, other: Any) -> "Poly":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Poly":
        other_expr = other.as_expr() if isinstance(other, Poly) else other
        return self._like(self.as_expr() - other_expr, other)

    def __rsub__(self, other: Any) -> "Poly":
        other_expr = other.as_expr() if isinstance(other, Poly) else other
        return self._like(other_expr - self.as_expr(), other)

    def __mul__(self, other: Any) -> "Poly":
        other_expr = other.as_expr() if isinstance(other, Poly) else other
        return self._like(self.as_expr() * other_expr, other)

    def __rmul__(self, other: Any) -> "Poly":
        return self.__mul__(other)

    def __neg__(self) -> "Poly":
        return self._like(-self.as_expr())

    def __pos__(self) -> "Poly":
        return self

    def __pow__(self, n: int) -> "Poly":
        if not isinstance(n, int) or n < 0:
            raise ValueError("Polynomial exponent must be a non-negative integer")
        from ..core import expand
        return self._like(expand(self.as_expr() ** n))

    def __floordiv__(self, other: Any) -> "Poly":
        return self.div(other)[0]

    def __mod__(self, other: Any) -> "Poly":
        return self.div(other)[1]

    def __divmod__(self, other: Any) -> Tuple["Poly", "Poly"]:
        return self.div(other)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Poly):
            if self._gens != other._gens:
                return False
            if self.as_expr() == other.as_expr():
                return True
            try:
                return self.all_coeffs() == other.all_coeffs()
            except Exception:
                return False
        return False

    def __hash__(self) -> int:
        return hash((self._expr, self._gens))

    def terms(self) -> list:
        """(monomial exponents, coefficient) pairs, lex-descending in gens."""
        from ..core import Add as _Add, Integer as _Integer, Mul as _Mul, Pow as _Pow, expand

        gens = list(self._gens)
        e = expand(self._expr)
        acc: dict = {}
        for t in (e.args if isinstance(e, _Add) else (e,)):
            if t == 0:
                continue
            exps = [0] * len(gens)
            coeff = _Integer(1)
            for f in (t.args if isinstance(t, _Mul) else (t,)):
                base, k = (f.args[0], f.args[1]) if isinstance(f, _Pow) else (f, _Integer(1))
                if base in gens and isinstance(k, _Integer) and k.p > 0:
                    exps[gens.index(base)] += int(k.p)
                else:
                    coeff = coeff * f
            key = tuple(exps)
            acc[key] = acc.get(key, _Integer(0)) + coeff
        items = [(m, c) for m, c in acc.items() if c != 0]
        items.sort(key=lambda mc: mc[0], reverse=True)
        return items or [(tuple([0] * len(gens)), _Integer(0))]

    def as_dict(self) -> dict:
        return {m: c for m, c in reversed(self.terms()) if c != 0}

    def _infer_domain(self) -> str:
        from ..core import Float as _Float, Integer as _Integer, Rational as _Rational

        coeffs = [c for _, c in self.terms()]
        carried = sorted(getattr(self, "_extra_syms", ()), key=lambda v: v.name)
        if all(isinstance(c, _Integer) for c in coeffs):
            return "ZZ[%s]" % ",".join(map(str, carried)) if carried else "ZZ"
        if all(isinstance(c, _Rational) for c in coeffs):
            return "QQ[%s]" % ",".join(map(str, carried)) if carried else "QQ"
        if all(isinstance(c, (_Rational, _Float)) for c in coeffs):
            return "RR"
        extra = set(getattr(self, "_extra_syms", ()))
        numeric_ok = True
        integral = True
        for c in coeffs:
            extra |= set(c.free_symbols)
            sub = Poly(c, *sorted(c.free_symbols, key=lambda v: v.name)) if c.free_symbols else None
            parts = [cc for _, cc in sub.terms()] if sub is not None else [c]
            for cc in parts:
                if isinstance(cc, _Integer):
                    continue
                if isinstance(cc, _Rational):
                    integral = False
                    continue
                numeric_ok = False
        if not extra:
            return "EX"
        if not numeric_ok:
            from .. import fraction as _fraction

            if all(
                not _fraction(c)[1].free_symbols or True
                for c in coeffs
            ) and all(_is_rational_function(c) for c in coeffs):
                names = ",".join(str(v) for v in sorted(extra, key=lambda v: v.name))
                return "ZZ(%s)" % names
            return "EX"
        names = ",".join(str(v) for v in sorted(extra, key=lambda v: v.name))
        return "%s[%s]" % ("ZZ" if integral else "QQ", names)

    @property
    def domain(self) -> Any:
        return self._domain if self._domain_given else self._infer_domain()

    def __repr__(self) -> str:
        """Upstream ``StrPrinter._print_Poly``."""
        from ..core import Add as _Add, Integer as _Integer

        gens = [str(g) for g in self._gens]
        terms: list = []
        for monom, coeff in self.terms():
            parts = []
            for i, ex in enumerate(monom):
                if ex > 0:
                    parts.append(gens[i] if ex == 1 else gens[i] + "**%d" % ex)
            s_monom = "*".join(parts)
            if isinstance(coeff, _Add):
                s_coeff = "(" + str(coeff) + ")" if s_monom else str(coeff)
            else:
                if s_monom:
                    if coeff == 1:
                        terms.extend(["+", s_monom])
                        continue
                    if coeff == -1:
                        terms.extend(["-", s_monom])
                        continue
                s_coeff = str(coeff)
            s_term = s_coeff if not s_monom else s_coeff + "*" + s_monom
            if s_term.startswith("-"):
                terms.extend(["-", s_term[1:]])
            else:
                terms.extend(["+", s_term])
        if terms and terms[0] in ("-", "+"):
            modifier = terms.pop(0)
            if modifier == "-":
                terms[0] = "-" + terms[0]
        return "Poly(%s, %s, domain='%s')" % (" ".join(terms), ", ".join(gens), self.domain)

    def __str__(self) -> str:
        return self.__repr__()


def _is_rational_function(c: Any) -> bool:
    from .. import fraction as _fraction

    n, d = _fraction(c)
    for part in (n, d):
        syms = sorted(part.free_symbols, key=lambda v: v.name)
        if not syms:
            continue
        try:
            for _, cc in Poly(part, *syms).terms():
                if not isinstance(cc, Rational):
                    return False
        except Exception:
            return False
    return True


def _parametric_div(f: Any, g: Any, x: Any) -> tuple:
    """Univariate division over the coefficient fraction field."""
    from ..core import expand as _expand

    fc = Poly(f, x)
    gc = Poly(g, x)
    dg = gc.degree()
    lg = gc.leading_coeff()
    q = Integer(0)
    r = _expand(f)
    while r != 0:
        pr = Poly(r, x)
        dr = pr.degree()
        if dr < dg:
            break
        t = cancel(pr.leading_coeff() / lg) * x ** (dr - dg)
        q = q + t
        r = _expand(cancel(r - t * g))
    del fc
    return _expand(q), r


def _sqrt_squares_out(d: Any) -> Any:
    """Upstream roots_quadratic ``_sqrt``: even powers leave the radical
    (both signs are roots, so no absolute value is introduced)."""
    from ..core import sqrt as _sqrt

    co, other = [], []
    for di in Mul.make_args(d):
        if isinstance(di, Pow) and isinstance(di.args[1], Integer) and int(di.args[1]) % 2 == 0:
            co.append(di.args[0] ** (int(di.args[1]) // 2))
        else:
            other.append(di)
    if co:
        return Mul(*co) * _sqrt(Mul(*other))
    return _sqrt(d)


def _quadratic_roots(c2: Any, c1: Any, c0: Any) -> list:
    """Upstream ``roots_quadratic`` shapes."""
    if c0 == 0:
        return [Integer(0), -c1 / c2]
    if c1 == 0:
        r = _sqrt_squares_out(cancel(-c0 / c2))
        return [-r, r]
    d = c1**2 - 4 * c2 * c0
    A = 2 * c2
    B = -c1 / A
    D = _sqrt_squares_out(d) / A
    return [B - D, B + D]


def _binomial_roots(cn: Any, c0: Any, n: int) -> list:
    """``cn*x**n + c0``: base * every n-th root of unity (upstream
    ``roots_binomial`` order: 1 first, then conjugate pairs)."""
    from ..core import I as _I, Rational as _R, expand as _expand, pi as _pi
    from ..functions import cos as _cos, sin as _sin

    alpha = cancel(-c0 / cn)
    co, other = [], []
    for f in Mul.make_args(alpha):
        if isinstance(f, Pow) and isinstance(f.args[1], Integer) and int(f.args[1]) % n == 0:
            co.append(f.args[0] ** (int(f.args[1]) // n))
        else:
            other.append(f)
    base = Mul(*co) * Mul(*other) ** _R(1, n) if co else alpha ** _R(1, n)
    order = [0]
    for k in range(1, n // 2 + 1):
        order.extend([n - k, k] if n - k != k else [k])
    out = []
    for k in order:
        ang = _R(2 * k, n) * _pi
        zeta = _expand(_cos(ang) + _I * _sin(ang))
        out.append(base if zeta == 1 else base * zeta)
    return out


def _root_heuristics(f: Any, x: Any) -> list:
    cs = Poly(f, x).all_coeffs()
    n = len(cs) - 1
    nonzero = [c for c in cs if c != 0]
    if n == 1:
        return [cancel(-cs[1] / cs[0])]
    if len(nonzero) == 2 and cs[-1] != 0:
        return _quadratic_roots(*cs) if n == 2 else _binomial_roots(cs[0], cs[-1], n)
    if n == 2:
        return [cancel(r) for r in _quadratic_roots(*cs)]
    raise NotImplementedError("roots of a degree-%d parametric factor" % n)


def _parametric_roots(p: "Poly") -> dict:
    """Roots of a univariate polynomial with symbolic coefficients
    (upstream ``roots``): linear and two-term polynomials by formula,
    otherwise factor over all symbols and solve each factor."""
    from ..core import Add

    x = p.gen
    cs = p.all_coeffs()
    k = 0
    while len(cs) > 1 and cs[-1] == 0:
        cs.pop()
        k += 1
    out: dict = {}
    f = Add(*[c * x ** (len(cs) - 1 - i) for i, c in enumerate(cs)])
    n = len(cs) - 1
    nonzero = [c for c in cs if c != 0]
    if n >= 1:
        if n == 1 or len(nonzero) == 2:
            for r in _root_heuristics(f, x):
                out[r] = out.get(r, 0) + 1
        else:
            # Upstream roots factors with the variable replaced by a Dummy
            # ``_x0``, which sorts after every named generator.
            res = _native_factor_multivariate(f, _sort_gens(f.free_symbols - {x}) + [x])
            factors = [(g, m) for g, m in (res[1] if res is not None else []) if x in g.free_symbols]
            if len(factors) <= 1 and n == 2:
                for r in _quadratic_roots(*cs):
                    out[r] = out.get(r, 0) + 1
            else:
                for g, m in factors or [(f, 1)]:
                    for r in _root_heuristics(g, x):
                        out[r] = out.get(r, 0) + m
    if k:
        out[Integer(0)] = out.get(Integer(0), 0) + k
    return out


def degree(f: Any, gen: Any = 0) -> Optional[int]:
    """Return polynomial degree."""
    if not isinstance(f, Poly):
        p = Poly(f, gen) if isinstance(gen, Symbol) else Poly(f)
    else:
        p = f
    return p.degree(gen)


def _coeffs_in(f: Any, x: Any) -> list:
    """Coefficients of ``f`` as a polynomial in ``x`` (descending), with
    arbitrary x-free coefficients."""
    p = Poly(f, x)
    terms = p.terms()
    deg = terms[0][0][0]
    out = [0] * (deg + 1)
    for (k,), c in terms:
        out[deg - k] = c
    from ..core import sympify as _sympify

    return [_sympify(c) for c in out]


def _leading(f: Any, gens: tuple) -> tuple:
    """(coefficient, monomial expr) of the lex-leading term."""
    from ..core import Integer as _Integer

    p = f if isinstance(f, Poly) else Poly(f, *gens)
    monom, coeff = p.terms()[0]
    m = _Integer(1)
    for g, k in zip(p.gens, monom):
        if k:
            m = m * g ** k
    return coeff, m


def LC(f: Any, *gens: Any) -> Any:
    """Leading coefficient (lex order in the generators)."""
    return _leading(f, gens)[0]


def LM(f: Any, *gens: Any) -> Any:
    """Leading monomial."""
    return _leading(f, gens)[1]


def LT(f: Any, *gens: Any) -> Any:
    """Leading term."""
    c, m = _leading(f, gens)
    return c * m


def invert(f: Any, g: Any, *gens: Any) -> Any:
    """Inverse of ``f`` modulo ``g`` (integers or univariate polynomials)."""
    from ..core import Integer as _Integer, sympify as _sympify

    if isinstance(f, (int, _Integer)) and isinstance(g, (int, _Integer)):
        return _Integer(pow(int(f), -1, int(g)))
    x = gens[0] if gens else None
    s_, _t, h = gcdex(f, g, x)
    h = _sympify(h)
    if h.free_symbols:
        raise ValueError("zero divisor: %s is not invertible modulo %s" % (f, g))
    return cancel(_sympify(s_) / h)


def interpolate(data: Any, x: Any) -> Any:
    """Lagrange interpolating polynomial: values at 1..n, (x, y) pairs or a
    dict {x: y} (upstream ``interpolate``)."""
    from ..core import Integer as _Integer, expand, sympify as _sympify

    if isinstance(data, dict):
        pts = list(data.items())
    else:
        data = list(data)
        if data and isinstance(data[0], (tuple, list)):
            pts = [tuple(p) for p in data]
        else:
            pts = [(_Integer(i + 1), v) for i, v in enumerate(data)]
    pts = [(_sympify(a), _sympify(b)) for a, b in pts]
    total = _Integer(0)
    for i, (xi, yi) in enumerate(pts):
        term = yi
        for j, (xj, _) in enumerate(pts):
            if i != j:
                term = term * (x - xj) / (xi - xj)
        total = total + term
    return expand(total)


def trailing_coeff(f: Any, *gens: Any) -> Any:
    """Return polynomial trailing coefficient."""
    p = f if isinstance(f, Poly) else Poly(f, *gens)
    return p.trailing_coeff()


def TC(f: Any, *gens: Any) -> Any:
    """Alias for trailing_coeff."""
    return trailing_coeff(f, *gens)


def EC(f: Any, *gens: Any) -> Any:
    """Alias for trailing_coeff."""
    return trailing_coeff(f, *gens)


def monic(f: Any) -> Any:
    """Return monic polynomial."""
    if not isinstance(f, Poly):
        return Poly(f).monic().as_expr()
    return f.monic()


def gcd(a: Any, b: Any) -> Any:
    """Compute polynomial or integer greatest common divisor."""
    if isinstance(a, int) and isinstance(b, int):
        return math.gcd(a, b)
    if isinstance(a, Poly):
        return a.gcd(b)
    if isinstance(b, Poly):
        return b.gcd(a)
    multi = _multivariate_gcd_lcm(a, b, lcm=False)
    if multi is not None:
        return multi
    p_a = Poly(a)
    res = p_a.gcd(b)
    return res.as_expr()


def _multivariate_gcd_lcm(a: Any, b: Any, lcm: bool) -> Any:
    """gcd/lcm over ZZ[x, y, ...] from the two complete factorizations:
    shared irreducible factors at the min (max) multiplicity times the gcd
    (lcm) of the integer contents, expanded as upstream returns it."""
    from ..core import expand as _expand, sympify as _s

    mgens = _sort_gens(_s(a).free_symbols | _s(b).free_symbols)
    if not mgens:
        return None
    fa = _native_factor_multivariate(a, mgens)
    fb = _native_factor_multivariate(b, mgens)
    if fa is None or fb is None:
        return None
    (ca, la), (cb, lb) = fa, fb
    if not (isinstance(ca, Integer) and isinstance(cb, Integer)) or ca == 0 or cb == 0:
        return None
    ia, ib = abs(int(ca)), abs(int(cb))
    content = (ia * ib // math.gcd(ia, ib)) if lcm else math.gcd(ia, ib)
    mult_b = {f: m for f, m in lb}
    mult_a = {f: m for f, m in la}
    out = Integer(content)
    keys = list(mult_a) + [f for f in mult_b if f not in mult_a]
    for f in keys:
        ma, mb = mult_a.get(f, 0), mult_b.get(f, 0)
        k = max(ma, mb) if lcm else min(ma, mb)
        if k:
            out = out * f**k
    return _expand(out)


def lcm(a: Any, b: Any) -> Any:
    """Compute polynomial or integer least common multiple."""
    if isinstance(a, int) and isinstance(b, int):
        return abs(a * b) // math.gcd(a, b) if a and b else 0
    if isinstance(a, Poly):
        return a.lcm(b)
    if isinstance(b, Poly):
        return b.lcm(a)
    multi = _multivariate_gcd_lcm(a, b, lcm=True)
    if multi is not None:
        return multi
    p_a = Poly(a)
    res = p_a.lcm(b)
    return res.as_expr()


def gcdex(f: Any, g: Any, x: Any = None) -> Tuple[Any, Any, Any]:
    """Extended Euclidean algorithm: returns (s, t, h) such that s*f + t*g = h = gcd(f, g)."""
    p_f = Poly(f, x) if x is not None else (f if isinstance(f, Poly) else Poly(f))
    p_g = Poly(g, x) if x is not None else (g if isinstance(g, Poly) else Poly(g, *p_f._gens))
    s, t, h = p_f.gcdex(p_g)
    if isinstance(f, Poly) or isinstance(g, Poly):
        return s, t, h
    return s.as_expr(), t.as_expr(), h.as_expr()


def half_gcdex(f: Any, g: Any, x: Any = None) -> Tuple[Any, Any]:
    """Half extended Euclidean algorithm: returns (s, h) such that s*f == h (mod g)."""
    s, _, h = gcdex(f, g, x)
    return s, h


def _sylvester_resultant(p: Any, q: Any, x: Any) -> Any:
    from ..core import expand
    from ..matrices import Matrix

    a = _coeffs_in(p, x)
    b = _coeffs_in(q, x)
    m, n = len(a) - 1, len(b) - 1
    if m == 0 and n == 0:
        return 1
    if m == 0:
        return expand(a[0] ** n)
    if n == 0:
        return expand(b[0] ** m)
    size = m + n
    rows = []
    for i in range(n):
        rows.append([0] * i + a + [0] * (size - m - 1 - i))
    for i in range(m):
        rows.append([0] * i + b + [0] * (size - n - 1 - i))
    return expand(Matrix(rows).det())


def _has_parameters(f: Any, x: Any) -> bool:
    from ..core import sympify as _sympify

    expr = f.as_expr() if isinstance(f, Poly) else _sympify(f)
    return bool(set(expr.free_symbols) - {x})


def resultant(p: Any, q: Any, x: Any = None) -> Any:
    """Resultant of two polynomials in ``x`` (Sylvester determinant when
    the coefficients are symbolic)."""
    if x is not None and (_has_parameters(p, x) or _has_parameters(q, x)):
        return _sylvester_resultant(p, q, x)
    poly_p = Poly(p, x) if x is not None else (p if isinstance(p, Poly) else Poly(p))
    return poly_p.resultant(q)


def _decompose_power(f: Any) -> tuple:
    """``f = base**e`` with integer ``e`` (upstream ``decompose_power``):
    ``exp(3*x)`` is ``(exp(x), 3)``, ``x**(3/2)`` is ``(sqrt(x), 3)``;
    a negative power flips to the reciprocal generator."""
    from ..core import Pow as _Pow, Rational as _Rat, expand as _expand
    from ..functions import exp as _exp

    if isinstance(f, _Pow):
        base, e = f.args
    elif type(f).__name__ == "exp":
        base, e = None, f.args[0]
    else:
        return f, 1
    if isinstance(e, _Rat):
        if base is None:
            return f, 1
        if e.q != 1:
            base = _Pow(base, _Rat(1, e.q))
        k = int(e.p)
    else:
        c = _Rat(1)
        tail = e
        if type(e).__name__ == "Mul" and isinstance(e.args[0], _Rat):
            c = e.args[0]
            tail = _expand(e / c)
        k = int(c.p)
        tail = tail / int(c.q)
        base = _exp(tail) if base is None else _Pow(base, tail)
    if k < 0:
        return 1 / base, -k
    return base, k


def terms_gcd(f: Any, *gens: Any, clear: bool = True, deep: bool = False, expand: bool = True, **_: Any) -> Any:
    """Remove the GCD of the terms of ``f`` (upstream ``terms_gcd``): the
    common monomial and numeric content come out as a factor. Generators
    are read structurally (``exp(3*x)`` is ``exp(x)**3``); nothing is
    expanded, so ``expand`` only matters for upstream signature parity."""
    from fractions import Fraction as _F
    from math import gcd as _gcd, lcm as _lcm
    from ..core import Add as _Add, Eq as _Eq, Integer as _Int, Mul as _Mul, Rational as _Rat, sympify as _s

    del gens, expand
    orig = _s(f)
    if type(orig).__name__ in ("Equality",) or isinstance(orig, _Eq):
        return _Eq(*(terms_gcd(a, clear=clear, deep=deep) for a in (orig.lhs, orig.rhs)))
    args = getattr(orig, "args", ())
    if not args:
        return orig
    if deep:
        new = orig.func(*[terms_gcd(a, clear=clear, deep=True) for a in args])
        return terms_gcd(new, clear=clear)
    if not isinstance(orig, _Add):
        return orig
    rows = []
    for t in orig.args:
        coeff = _F(1)
        mono: dict = {}
        for fac in _Mul.make_args(t):
            if isinstance(fac, _Rat):
                coeff *= _F(int(fac.p), int(fac.q))
                continue
            if not fac.free_symbols and not isinstance(fac, _Rat) and fac.is_number and type(fac).__name__ in ("Float", "ImaginaryUnit"):
                return orig
            b, k = _decompose_power(fac)
            mono[b] = mono.get(b, 0) + k
        rows.append((coeff, mono))
    common = dict(rows[0][1])
    for _, mono in rows[1:]:
        common = {b: min(k, mono[b]) for b, k in common.items() if b in mono}
    common = {b: k for b, k in common.items() if k > 0}
    den = 1
    for c, _ in rows:
        den = _lcm(den, c.denominator)
    num = 0
    for c, _ in rows:
        num = _gcd(num, int(c * den))
    content = _F(num, den)
    if content == 1 and not common:
        return orig
    term = _Mul(*[b ** k for b, k in common.items()])
    reduced = []
    for c, mono in rows:
        rest = _Mul(*[b ** (k - common.get(b, 0)) for b, k in mono.items()])
        q = c / content
        reduced.append(_Rat(q.numerator, q.denominator) * rest)
    poly_part = _Add(*reduced)
    k = _Rat(content.numerator, content.denominator)
    if not clear and content.denominator != 1:
        # clear=False (upstream ``_keep_coeff``) distributes the fractional
        # content back when that leaves some term an integer coefficient.
        if any(c.denominator == 1 for c, _ in rows):
            poly_part = _Add(*[k * r for r in reduced])
            k = _Int(1)
    return _Mul(k, term, poly_part)


def discriminant(p: Any, x: Any = None) -> Any:
    """Discriminant ``(-1)**(n(n-1)/2) * res(p, p') / lc(p)``."""
    from ..core import diff as _diff, expand, sympify as _sympify

    if x is not None and _has_parameters(p, x):
        expr = p.as_expr() if isinstance(p, Poly) else _sympify(p)
        a = _coeffs_in(expr, x)
        n = len(a) - 1
        r = _sylvester_resultant(expr, _diff(expr, x), x)
        sign = -1 if (n * (n - 1) // 2) % 2 else 1
        return expand(cancel(sign * r / a[0]))
    poly_p = Poly(p, x) if x is not None else (p if isinstance(p, Poly) else Poly(p))
    return poly_p.discriminant()


def sqf_list(p: Any, x: Any = None) -> Tuple[Any, List[Tuple[Any, int]]]:
    """Compute square-free factorization returning (scale, [(factor, multiplicity), ...])."""
    if not isinstance(p, Poly):
        mgens = _multivariate_gens(p, (x,) if x is not None else ())
        if mgens is not None:
            res = _native_factor_multivariate(p, mgens)
            if res is not None:
                # The square-free parts are the products of the irreducible
                # factors sharing a multiplicity (upstream returns them
                # expanded, by increasing multiplicity).
                from ..core import expand as _expand

                scale, factors = res
                groups: dict = {}
                for f, m in factors:
                    groups[m] = groups.get(m, Integer(1)) * f
                return scale, [(_expand(groups[m]), m) for m in sorted(groups)]
    poly_p = Poly(p, x) if x is not None else (p if isinstance(p, Poly) else Poly(p))
    scale, factors = poly_p.sqf_list()
    return (scale, [(f.as_expr(), mult) for f, mult in factors])


def sqf_part(p: Any, x: Any = None) -> Any:
    """Compute the square-free part of a polynomial."""
    poly_p = Poly(p, x) if x is not None else (p if isinstance(p, Poly) else Poly(p))
    return poly_p.sqf_part().as_expr()


def sqf(p: Any, x: Any = None) -> Any:
    """Square-free factorization of a polynomial into expression form."""
    scale, factors = sqf_list(p, x)
    if not factors:
        return scale
    terms = [f ** mult if mult != 1 else f for f, mult in factors]
    prod = terms[0]
    for t in terms[1:]:
        prod = prod * t
    if scale != 1:
        prod = scale * prod
    return prod


class GroebnerBasis:
    """Reduced Groebner basis (upstream ``GroebnerBasis`` surface): a
    sequence of polynomial expressions with generators, domain and order."""

    def __init__(self, exprs: list, gens: tuple, domain: str, order: str) -> None:
        self.exprs = list(exprs)
        self.gens = tuple(gens)
        self.domain = domain
        self.order = order

    @property
    def args(self) -> tuple:
        return (tuple(self.exprs), self.gens)

    def __iter__(self):
        return iter(self.exprs)

    def __len__(self) -> int:
        return len(self.exprs)

    def __getitem__(self, i: Any) -> Any:
        return self.exprs[i]

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, GroebnerBasis):
            return self.exprs == other.exprs and self.gens == other.gens
        if isinstance(other, (list, tuple)):
            return self.exprs == list(other)
        return NotImplemented

    def __hash__(self) -> int:
        return hash((tuple(self.exprs), self.gens, self.order))

    def _ordered_str(self, e: Any) -> str:
        """``e`` printed with its terms in the basis' monomial order."""
        from ..core import Add as _Add, Integer as _Integer

        if self.order == "lex" or not isinstance(e, _Add):
            return str(e)
        p = Poly(e, *self.gens)

        def key(mc: tuple) -> tuple:
            m = mc[0]
            if self.order == "grlex":
                return (sum(m), m)
            return (sum(m), tuple(-k for k in reversed(m)))

        out = ""
        for i, (m, c) in enumerate(sorted(p.terms(), key=key, reverse=True)):
            mono = _Integer(1)
            for g, k in zip(self.gens, m):
                if k:
                    mono = mono * g ** k
            t = str(c * mono)
            if i == 0:
                out = t
            elif t.startswith("-"):
                out += " - " + t[1:]
            else:
                out += " + " + t
        return out

    def __repr__(self) -> str:
        args = ["[%s]" % ", ".join(self._ordered_str(e) for e in self.exprs)] + [str(g) for g in self.gens]
        args += ["domain='%s'" % self.domain, "order='%s'" % self.order]
        return "GroebnerBasis(%s)" % ", ".join(args)

    __str__ = __repr__


def _primitive_integer(e: Any, gens: tuple) -> Any:
    """Scale to integer coefficients with positive leading coefficient
    and unit content (upstream ZZ-domain basis normalization)."""
    from math import gcd as _igcd, lcm as _ilcm
    from ..core import Integer as _Integer, Rational as _Rational, expand

    p = Poly(e, *gens)
    terms = p.terms()
    if not all(isinstance(c, _Rational) for _, c in terms):
        return e
    den = 1
    for _, c in terms:
        den = _ilcm(den, int(c.q))
    num = 0
    for _, c in terms:
        num = _igcd(num, abs(int(c.p * (den // c.q))))
    scale = _Rational(den, num or 1)
    if terms[0][1] < 0:
        scale = -scale
    return expand(e * scale)


def groebner(F: Sequence[Any], *gens: Any, order: str = "lex", **args: Any) -> Any:
    """Reduced Groebner basis of ``F`` (orders lex, grlex, grevlex)."""
    if len(gens) == 1 and isinstance(gens[0], (list, tuple)):
        var_list = list(gens[0])
    else:
        var_list = list(gens)
    if not var_list:
        # Collect free symbols across all input polynomials
        symbols = set()
        for e in F:
            wrapped = _wrap(_native_expr(e))
            symbols.update(wrapped.free_symbols)
        var_list = sorted(list(symbols), key=lambda s: s.name)

    eq_sources = [str(_native_expr(e)) for e in F]
    var_names = [_native_symbol_key(_require_symbol(g)) for g in var_list]
    if order not in ("lex", "grlex", "grevlex"):
        raise ValueError("unknown monomial order %r" % order)
    raw = _native.groebner_basis_expr(eq_sources, var_names, order)
    gens_t = tuple(var_list)
    exprs = [_primitive_integer(_parse_result(r), gens_t) for r in raw]
    from ..printing.str import sort_key as _sk

    def lead_key(e: Any) -> tuple:
        return tuple(Poly(e, *gens_t).terms()[0][0])

    if order == "lex":
        exprs.sort(key=lead_key, reverse=True)
    domain = "ZZ" if all(Poly(e, *gens_t)._infer_domain() == "ZZ" for e in exprs) else "QQ"
    return GroebnerBasis(exprs, gens_t, domain, order)


def _dmp_rep(expr: Any, gens: list) -> Any:
    """Dense recursive coefficient list (upstream DMP ``rep``), used only
    as the factor ordering key."""
    from ..core import expand as _expand, sympify

    if not gens:
        v = sympify(expr)
        return Fraction(int(v.p), int(v.q)) if isinstance(v, Rational) else 0
    x, rest = gens[0], gens[1:]
    e = _expand(expr)
    coeffs: dict = {}
    for t in (e.args if type(e).__name__ == "Add" else (e,)):
        k = 0
        others = []
        for fct in (t.args if type(t).__name__ == "Mul" else (t,)):
            if fct == x:
                k += 1
            elif type(fct).__name__ == "Pow" and fct.args[0] == x:
                k += int(fct.args[1])
            else:
                others.append(fct)
        c = Integer(1)
        for o in others:
            c = c * o
        coeffs[k] = coeffs.get(k, 0) + c
    coeffs = {k: c for k, c in coeffs.items() if c != 0}
    if not coeffs:
        return []  # the zero polynomial (upstream DMP strips it to [])
    deg = max(coeffs)
    return [_dmp_rep(coeffs.get(k, 0), rest) for k in range(deg, -1, -1)]


def _native_factor_multivariate(expr: Any, gens: list) -> Tuple[Any, List[Tuple[Any, int]]] | None:
    """Complete multivariate factorization over QQ (native Kronecker /
    homogeneous lane with exact division checks), ordered as upstream
    ``factor_list``: by degree in the first generator, multiplicity, rep.
    None when the input is not a rational-coefficient polynomial in
    ``gens`` or lies outside the native regime."""
    try:
        scale_raw, raw = _native.poly_factor_multivariate_expr(
            str(_native_expr(expr)), [_native_symbol_key(g) for g in gens]
        )
    except Exception:
        return None
    factors = [(_parse_result(f), m) for f, m in raw]

    def key(fm: Any) -> tuple:
        rep = _dmp_rep(fm[0], list(gens))
        return (len(rep), fm[1], rep)

    try:
        factors.sort(key=key)
    except TypeError:
        pass
    return _parse_result(scale_raw), factors


def _multivariate_gens(p: Any, gens: tuple) -> list | None:
    syms = _sort_gens(getattr(_wrap(_native_expr(p)), "free_symbols", set()))
    if gens:
        extra = [s for s in syms if s not in gens]
        out = list(gens) + extra
    else:
        out = syms
    return out if len(out) > 1 else None


def factor_list(p: Any, *gens: Any) -> Tuple[Any, List[Tuple[Any, int]]]:
    """Compute polynomial factor list returning (scale, [(factor, multiplicity), ...])."""
    if not isinstance(p, Poly):
        mgens = _multivariate_gens(p, gens)
        if mgens is not None:
            res = _native_factor_multivariate(p, mgens)
            if res is not None:
                return res
    poly_p = p if isinstance(p, Poly) else Poly(p, *gens)
    scale, factors = poly_p.factor_list()
    return (scale, [(f.as_expr(), mult) for f, mult in factors])


def _factor_list_any(e: Any) -> tuple | None:
    """(coefficient, [(irreducible, multiplicity)]) of a polynomial in its
    free symbols over QQ, or None when ``e`` is not such a polynomial."""
    from ..core import sympify as _s

    e = _s(e)
    syms = _sort_gens(e.free_symbols)
    if not syms:
        return (e, [])
    if len(syms) == 1:
        try:
            scale, fl = Poly(e, syms[0]).factor_list()
        except Exception:
            return None
        return scale, [(f.as_expr(), m) for f, m in fl]
    return _native_factor_multivariate(e, syms)


def _factor_rational(p: Any) -> Any:
    """Upstream factor of a rational function: numerator and denominator
    factored separately, one coefficient kept outside
    (factor(1/(a**2 + 2*a + 1)) -> (a + 1)**(-2))."""
    from ..core import _keep_coeff, sympify as _s
    from .. import fraction as _fraction

    expr = _wrap(_native_expr(p))
    if not getattr(expr, "free_symbols", None):
        return None
    try:
        n, d = _fraction(together(expr))
    except Exception:
        return None
    if d == 1 or not _s(d).free_symbols:
        return None
    fn, fd = _factor_list_any(n), _factor_list_any(d)
    if fn is None or fd is None:
        return None
    coeff = fn[0] / fd[0]
    num = Mul(*[f if m == 1 else f**m for f, m in fn[1]])
    den = Mul(*[f if m == 1 else f**m for f, m in fd[1]])
    return _keep_coeff(coeff, num / den)


def factor(p: Any, *gens: Any) -> Any:
    # Oracle-pinned: factor cancels rational functions
    # (factor((x**2 - 1)/(x + 1)) -> x - 1).
    from ..core import Mul as _Mul, Pow as _Pow, Add as _Add, Integer as _Integer
    if not isinstance(p, Poly) and not gens:
        rational = _factor_rational(p)
        if rational is not None:
            return rational
    if isinstance(p, _Mul):
        for f in p.args:
            if (
                isinstance(f, _Pow)
                and isinstance(f.args[1], _Integer)
                and f.args[1].p < 0
                and isinstance(f.args[0], _Add)
            ):
                return cancel(p)
    # Oracle-pinned: an all-negative-power Add fuses over the common
    # denominator (factor(1/x + 1/y) -> Pow(x, -1) * Pow(y, -1) * (x + y)).
    if isinstance(p, _Add) and p.args and all(
        isinstance(t, _Pow)
        and isinstance(t.args[0], Symbol)
        and isinstance(t.args[1], _Integer)
        and t.args[1].p < 0
        for t in p.args
    ):
        return together(p)
    """Factor polynomial into irreducible factors."""
    if isinstance(p, Poly):
        return p.factor()
    if not gens:
        abstracted = _factor_function_generators(p)
        if abstracted is not None:
            return abstracted
    mgens = _multivariate_gens(p, gens)
    if mgens is not None:
        res = _native_factor_multivariate(p, mgens)
        if res is not None:
            from ..core import _keep_coeff

            scale, factors = res
            # One n-ary Mul: a pairwise 2*(x - y) would distribute.
            return _keep_coeff(scale, Mul(*[f if m == 1 else f**m for f, m in factors]))
    if not gens:
        wrapped = _wrap(_native_expr(p))
        syms = _sort_gens(getattr(wrapped, "free_symbols", set()))
        if len(syms) > 1:
            result = _factor_multivariate(wrapped, syms)
            if result is not None:
                return result
    try:
        poly_p = Poly(p, *gens)
        return poly_p.factor()
    except Exception:
        return _wrap(_native_expr(p))


def _function_generators(e, out):
    from ..core import Function as _Function

    if isinstance(e, _Function) and e.free_symbols:
        out.add(e)
        return
    for a in getattr(e, "args", ()) or ():
        if hasattr(a, "free_symbols"):
            _function_generators(a, out)


def _factor_function_generators(p):
    """Upstream factors over function generators too
    (factor(sin(x)**2 + 2*sin(x)*cos(x) + cos(x)**2) -> (sin(x) + cos(x))**2):
    each maximal applied function becomes a fresh generator."""
    from ..core import Dummy as _Dummy, sympify as _sympify

    from fractions import Fraction as _Fraction
    from math import gcd as _igcd, lcm as _ilcm
    from ..core import Rational as _Rational
    from ..functions import exp as _exp

    p = _sympify(p)
    gens = set()
    _function_generators(p, gens)
    if not gens:
        return None
    # exp(k*u) for rational k share the generator exp(g*u), g = gcd of k.
    exp_groups: dict = {}
    plain = []
    for g in gens:
        if type(g).__name__ == "exp":
            arg = g.args[0]
            coeff, rest = arg.as_coeff_Mul() if hasattr(arg, "as_coeff_Mul") else (_Rational(1), arg)
            if not isinstance(coeff, _Rational):
                coeff, rest = _Rational(1), arg
            exp_groups.setdefault(rest, []).append((g, _Fraction(int(coeff.p), int(coeff.q))))
        else:
            plain.append(g)
    replacements = []  # (old expr, new expr in dummies), back-map
    back = []
    index = 0
    for g in sorted(plain, key=str):
        d = _Dummy("g%d" % index)
        index += 1
        replacements.append((g, d))
        back.append((d, g))
    for rest, members in sorted(exp_groups.items(), key=lambda kv: str(kv[0])):
        num = 0
        den = 1
        for _, c in members:
            num = _igcd(num, abs(c.numerator))
            den = _ilcm(den, c.denominator)
        # Upstream treats exp(k*u) as exp(u)**k: the generator carries
        # the unit numerator (exp(2*x) - 1 = (exp(x) - 1)*(exp(x) + 1)).
        step = _Fraction(1, den)
        d = _Dummy("g%d" % index)
        index += 1
        for g, c in members:
            k = c / step
            replacements.append((g, d ** int(k)))
        back.append((d, _exp(_Rational(step.numerator, step.denominator) * rest)))
    q = p
    for g, d in replacements:
        q = q.subs(g, d)
    probe = set()
    _function_generators(q, probe)
    if probe:
        return None
    f = factor(q)
    if f == q:
        return None
    f = _normalize_factor_signs(f, [d for d, _ in back])
    for d, g in back:
        f = f.subs(d, g)
    return f


def _normalize_factor_signs(f, syms):
    """Upstream factor form: every polynomial factor has a positive leading
    coefficient (lex in ``syms``) and the overall sign is pulled out."""
    from ..core import Add as _Add, Integer as _Integer, Mul as _Mul, Pow as _Pow, expand

    factors = list(f.args) if isinstance(f, _Mul) else [f]
    sign = 1
    out = []
    for fac in factors:
        base, k = (fac.args[0], fac.args[1]) if isinstance(fac, _Pow) else (fac, _Integer(1))
        if isinstance(base, _Add) and isinstance(k, _Integer):
            terms = expand(base).args
            parsed = [_term_monomial(t, syms) for t in terms]
            if all(pm is not None for pm in parsed):
                lead = max(parsed, key=lambda cm: tuple(cm[1].get(s, 0) for s in syms))
                if lead[0] < 0:
                    base = -base
                    if k.p % 2:
                        sign = -sign
                    out.append(base ** k)
                    continue
        out.append(fac)
    return _Mul(_Integer(sign), *out)


def _term_monomial(term, syms):
    """(rational coefficient, {symbol: exponent}) of an expanded term, or None."""
    from ..core import Mul as _Mul, Pow as _Pow, Integer as _Integer, Rational as _Rational
    coeff = _Rational(1)
    mon = {}
    factors = term.args if isinstance(term, _Mul) else (term,)
    for f in factors:
        if isinstance(f, _Rational):
            coeff = coeff * f
        elif f in syms:
            mon[f] = mon.get(f, 0) + 1
        elif isinstance(f, _Pow) and f.args[0] in syms and isinstance(f.args[1], _Integer) and f.args[1].p > 0:
            mon[f.args[0]] = mon.get(f.args[0], 0) + f.args[1].p
        else:
            return None
    return coeff, mon


def _factor_multivariate(expr, syms):
    """Multivariate factoring by content extraction: numeric and monomial
    content, the content of a bivariate polynomial with respect to its main
    variable (a univariate gcd in the other), and homogeneous bivariate
    primitive parts through dehomogenization. Returns None when the input
    is not a polynomial in ``syms`` with rational coefficients."""
    from ..core import Add as _Add, Mul as _Mul, Integer as _Integer, Rational as _Rational, expand
    from math import gcd as _igcd

    e = expand(expr)
    terms = e.args if isinstance(e, _Add) else (e,)
    parsed = []
    for t in terms:
        tm = _term_monomial(t, syms)
        if tm is None:
            return None
        parsed.append(tm)
    if not parsed:
        return None
    # numeric content (sign follows the leading term in sorted order)
    num = 0
    den = 1
    for c, _ in parsed:
        num = _igcd(num, abs(c.p))
        den = den * c.q // _igcd(den, c.q)
    content = _Rational(num, den)
    lead = max(parsed, key=lambda cm: tuple(cm[1].get(s, 0) for s in syms))
    if lead[0].p < 0:
        content = -content
    mono = {s: min(m.get(s, 0) for _, m in parsed) for s in syms}
    rest = _Integer(0)
    for c, m in parsed:
        term = c / content
        for s in syms:
            k = m.get(s, 0) - mono[s]
            if k:
                term = term * s**k
        rest = rest + term
    outer = content
    for s in syms:
        if mono[s]:
            outer = outer * s**mono[s]
    inner = _factor_primitive(rest, syms)
    result = outer * inner
    return result


def _factor_primitive(rest, syms):
    from ..core import Add as _Add, Integer as _Integer, expand
    present = [s for s in syms if s in rest.free_symbols]
    if len(present) <= 1:
        try:
            return factor(rest) if present else rest
        except Exception:
            return rest
    if len(present) != 2:
        return rest
    x, y = present
    by_degree: dict = {}
    for t in (rest.args if isinstance(rest, _Add) else (rest,)):
        tm = _term_monomial(t, [x, y])
        if tm is None:
            return rest
        c, m = tm
        by_degree[m.get(x, 0)] = by_degree.get(m.get(x, 0), _Integer(0)) + c * y ** m.get(y, 0)
    coeffs = [c for _, c in sorted(by_degree.items(), reverse=True) if c != 0]
    from . import polytools as _pt
    cont = _Integer(1)
    if all(c.free_symbols for c in coeffs):
        cont = coeffs[0]
        for c in coeffs[1:]:
            try:
                cont = _pt.gcd(cont, c)
            except Exception:
                cont = _Integer(1)
                break
    factored = _Integer(1)
    if cont.free_symbols:
        quotient = cancel(rest / cont)
        factored = factor(cont)
        rest = expand(quotient)
    # homogeneous primitive part: dehomogenize at y = 1
    terms = rest.args if isinstance(rest, _Add) else (rest,)
    degrees = set()
    for t in terms:
        tm = _term_monomial(t, [x, y])
        if tm is None:
            return factored * rest
        degrees.add(tm[1].get(x, 0) + tm[1].get(y, 0))
    if len(degrees) == 1 and rest.free_symbols >= {x, y}:
        total = degrees.pop()
        uni = rest.subs(y, 1)
        try:
            scale, flist = Poly(uni, x).factor_list()
        except Exception:
            return factored * rest
        out = scale
        used = 0
        for f, mult in flist:
            fx = f.as_expr()
            d = Poly(fx, x).degree()
            used += d * mult
            homog = expand(fx.subs(x, x / y) * y**d)
            out = out * homog**mult
        if total > used:
            out = out * y ** (total - used)
        return factored * out
    return factored * rest


def roots(p: Any, *gens: Any) -> dict[Any, int]:
    """Compute roots of a polynomial with their multiplicities."""
    poly_p = p if isinstance(p, Poly) else Poly(p, *gens)
    return poly_p.roots()


def content(f: Any, *gens: Any) -> Any:
    """Compute the content (GCD of coefficients) of a polynomial."""
    p = f if isinstance(f, Poly) else Poly(f, *gens)
    return p.content()


def primitive(f: Any, *gens: Any) -> Tuple[Any, Any]:
    """Compute the content and primitive form of a polynomial."""
    p = f if isinstance(f, Poly) else Poly(f, *gens)
    cont, prim = p.primitive()
    if isinstance(f, Poly):
        return cont, prim
    return cont, prim.as_expr()


def cancel(f: Any, *gens: Any) -> Any:
    """Cancel common factors in a rational function f = p / q."""
    from ..core import Integer, expand
    is_poly = isinstance(f, Poly)
    wrapped = f.as_expr() if is_poly else _wrap(_native_expr(f))
    if not hasattr(wrapped, "as_numer_denom"):
        return f
    numer, denom = wrapped.as_numer_denom()
    if denom == 1 or denom == Integer(1):
        # Upstream cancel returns the polynomial expanded
        # (cancel(x*(x + 1)) -> x**2 + x).
        return f if is_poly else expand(numer)

    try:
        if gens:
            all_gens = tuple(gens)
        elif is_poly:
            all_gens = f.gens
        else:
            all_gens = tuple(sorted(list(numer.free_symbols | denom.free_symbols), key=lambda s: s.name))

        if not all_gens:
            res = numer / denom
            return Poly(res) if is_poly else res

        p_poly = Poly(numer, *all_gens)
        q_poly = Poly(denom, *all_gens)

        p_cont, p_prim = p_poly.primitive()
        q_cont, q_prim = q_poly.primitive()

        if q_cont == 0:
            return f if is_poly else wrapped

        c = p_cont / q_cont
        c_num = c.p if hasattr(c, "p") else c
        c_den = c.q if hasattr(c, "q") else 1

        try:
            g = p_prim.gcd(q_prim)
            p_div = p_prim.div(g)[0]
            q_div = q_prim.div(g)[0]
            num_final = expand(p_div.as_expr() * c_num)
            den_final = expand(q_div.as_expr() * c_den)
        except Exception:
            num_final = expand(p_prim.as_expr() * c_num)
            den_final = expand(q_prim.as_expr() * c_den)

        # Upstream cancel keeps the denominator's lex-leading coefficient
        # positive: -1/(-a - 1) -> 1/(a + 1).
        try:
            lead_c, _ = _leading(den_final, all_gens)
            if lead_c < 0:
                num_final, den_final = expand(-num_final), expand(-den_final)
        except Exception:
            pass
        if den_final == 1 or den_final == Integer(1):
            res = num_final
        else:
            res = num_final / den_final

        if is_poly:
            return Poly(res, *all_gens)
        return _split_combined_denominator(res) if not is_poly else res
    except Exception:
        return f if is_poly else wrapped

def _split_combined_denominator(expr: Any) -> Any:
    """Oracle-pinned normalization: Pow(Mul(f1, f2), -n) splits into
    per-factor negative Pows (together(1/x + 1/y) renders
    Pow(x, -1) * Pow(y, -1) * (x + y), never Pow(x*y, -1) * (x + y))."""
    if not isinstance(expr, Mul):
        return expr
    new_factors: list[Any] = []
    changed = False
    for f in expr.args:
        if (
            isinstance(f, Pow)
            and isinstance(f.args[0], Mul)
            and isinstance(f.args[1], Integer)
            and f.args[1].p < 0
        ):
            n = -f.args[1].p
            for sub in f.args[0].args:
                if isinstance(sub, Integer) and sub.p == 1:
                    continue
                new_factors.append(pow(sub, Integer(-n)))
            changed = True
        else:
            new_factors.append(f)
    if not changed:
        return expr
    return Mul(*new_factors)


def together(expr: Any) -> Any:
    """Combine symbolic expressions into a single rational function.

    Examples
    ========
    >>> from sympy import together, symbols
    >>> x, y = symbols('x y')
    >>> together(1/x + 1/y)
    (x + y)/(x*y)
    >>> together(1/x + 1)
    (x + 1)/x
    """
    from ..core import Add, expand
    is_poly = isinstance(expr, Poly)
    wrapped = expr.as_expr() if is_poly else _wrap(_native_expr(expr))

    if isinstance(wrapped, Add):
        terms = list(wrapped.args)
        if not terms:
            return expr
        num_acc, den_acc = terms[0].as_numer_denom()
        for term in terms[1:]:
            n, d = term.as_numer_denom()
            num_acc = expand(num_acc * d + n * den_acc)
            den_acc = expand(den_acc * d)

        combined = num_acc / den_acc if den_acc != 1 else num_acc
        res = cancel(combined)
        return Poly(res, *expr._gens) if is_poly else _split_combined_denominator(res)

    elif hasattr(wrapped, "args") and wrapped.args:
        new_args = [together(a) for a in wrapped.args]
        if new_args != list(wrapped.args):
            res = type(wrapped)(*new_args)
            return Poly(res, *expr._gens) if is_poly else res

    return _split_combined_denominator(expr) if not is_poly else expr


def _x_coeffs(expr: Any, x: Any) -> dict | None:
    """``{k: coefficient}`` of a polynomial in ``x`` with x-free symbolic
    coefficients, or None when ``expr`` is not polynomial in ``x``."""
    from ..core import Add as _Add, expand as _expand

    out: dict = {}
    e = _expand(expr)
    for t in (e.args if isinstance(e, _Add) else (e,)):
        k = 0
        rest = []
        for f in (t.args if isinstance(t, Mul) else (t,)):
            if f == x:
                k += 1
            elif isinstance(f, Pow) and f.args[0] == x and isinstance(f.args[1], Integer) and f.args[1] > 0:
                k += int(f.args[1])
            elif x in getattr(f, "free_symbols", set()):
                return None
            else:
                rest.append(f)
        out[k] = out.get(k, Integer(0)) + Mul(*rest)
    return out


def _apart_parametric(numer: Any, denom: Any, x: Any) -> Any:
    """Partial fractions of ``numer/denom`` in ``x`` over QQ(params):
    the denominator factors over ZZ[x, params]; the polynomial part and the
    numerators ``A_ij`` (deg < deg q_i) come from one linear system in
    undetermined coefficients, solved generically as upstream does."""
    from ..core import Add as _Add, Dummy as _Dummy, expand as _expand

    gens = [x] + _sort_gens((numer.free_symbols | denom.free_symbols) - {x})
    fl = _native_factor_multivariate(denom, gens)
    if fl is None:
        return None
    scale, factors = fl
    unit = scale
    parts = []
    for f, m in factors:
        if x in f.free_symbols:
            parts.append((f, m))
        else:
            unit = unit * f**m
    nc = _x_coeffs(numer, x)
    if nc is None or not parts:
        return None
    dx = _expand(Mul(*[f**m for f, m in parts]))
    dc = _x_coeffs(dx, x)
    deg_n, deg_d = max(nc), max(dc)
    unknowns = []
    poly_part = Integer(0)
    for k in range(deg_n - deg_d + 1):
        a = _Dummy("a")
        unknowns.append(a)
        poly_part = poly_part + a * x**k
    pieces = []
    total = _expand(poly_part * dx)
    for f, m in parts:
        dq = max(_x_coeffs(f, x))
        for j in range(1, m + 1):
            num = Integer(0)
            for k in range(dq):
                a = _Dummy("a")
                unknowns.append(a)
                num = num + a * x**k
            cof = cancel(dx / f**j)
            pieces.append((num, f, j))
            total = total + _expand(num * cof)
    target = _expand(numer / unit)
    eqs = []
    tc, gc = _x_coeffs(total, x), _x_coeffs(target, x)
    if tc is None or gc is None:
        return None
    for k in set(tc) | set(gc):
        eqs.append(_expand(tc.get(k, 0) - gc.get(k, 0)))
    from ..solvers.solvers import linsolve as _linsolve

    sol = _linsolve(eqs, unknowns)
    if not sol:
        return None
    vals = dict(zip(unknowns, next(iter(sol))))
    if any(v.free_symbols & set(unknowns) for v in vals.values()):
        return None
    out = [_expand(poly_part.subs(vals))]
    for num, f, j in pieces:
        n = factor(cancel(num.subs(vals)))
        if n != 0:
            out.append(n / f**j)
    return _Add(*out)


def apart(expr: Any, x: Any = None) -> Any:
    """Compute partial fraction decomposition of a rational function.

    Examples
    ========
    >>> from sympy import apart, symbols
    >>> x = symbols('x')
    >>> apart(1/(x**2 - 1), x)
    1/(2*(x - 1)) - 1/(2*(x + 1))
    >>> apart((x + 2)/(x + 1), x)
    1 + 1/(x + 1)
    """
    from ..core import Add, Integer, Symbol, _require_symbol
    is_poly = isinstance(expr, Poly)
    wrapped = expr.as_expr() if is_poly else _wrap(_native_expr(expr))

    if x is None:
        # Oracle-pinned no-gens boundary (probed on SymPy 1.14.0):
        # - bare rationals: identity
        # - non-rational constants (sqrt(8)): PolynomialError
        # - 2+ free symbols: NotImplementedError multivariate
        # - single Function node with polynomial args (sin(2*x)): identity
        # - function-bearing Add/Mul/Pow (sin**2 + cos**2, tan*cos): multivariate
        free = wrapped.free_symbols
        if not free:
            if isinstance(expr, (Integer, Rational)):
                return expr
            raise PolynomialError(f"Cannot construct polynomials from {expr}, 1")
        if len(free) > 1:
            raise NotImplementedError("multivariate partial fraction decomposition")
        fn_atoms = wrapped.atoms(Function)
        if fn_atoms:
            # A single applied function node with polynomial content
            # (sin(2*x)) passes through as identity, matching the oracle.
            if len(fn_atoms) == 1 and isinstance(wrapped, Function):
                return expr
            raise NotImplementedError("multivariate partial fraction decomposition")
        x_sym = next(iter(free))
    elif isinstance(x, str):
        x_sym = Symbol(x)
    else:
        x_sym = _require_symbol(x)

    if isinstance(wrapped, Add):
        terms = list(wrapped.args)
        decomposed_terms = []
        for t in terms:
            decomposed_terms.append(apart(t, x_sym))
        res = Add(*decomposed_terms)
        return Poly(res, x_sym) if is_poly else res

    if not hasattr(wrapped, "as_numer_denom"):
        return expr

    numer, denom = wrapped.as_numer_denom()
    if x_sym not in denom.free_symbols:
        return expr

    if (numer.free_symbols | denom.free_symbols) - {x_sym}:
        res = _apart_parametric(numer, denom, x_sym)
        if res is not None:
            return Poly(res, x_sym) if is_poly else res

    try:
        p_poly = Poly(numer, x_sym)
        q_poly = Poly(denom, x_sym)
    except Exception:
        return expr

    # 1. Division with remainder: P(x) = S(x)*Q(x) + R(x)
    try:
        s_poly, r_poly = p_poly.div(q_poly)
    except Exception:
        return expr

    s_expr = s_poly.as_expr()
    if r_poly.is_zero:
        return Poly(s_expr, x_sym) if is_poly else s_expr

    # 2. Factor Q(x) over QQ
    try:
        scale, factors = q_poly.factor_list()
    except Exception:
        return expr

    if scale != 1 and scale != Integer(1):
        r_poly = Poly(r_poly.as_expr() / scale, x_sym)

    if not factors:
        res = s_expr + r_poly.as_expr()
        return Poly(res, x_sym) if is_poly else res

    # 3. Helper to expand rem / (base ** exp) using base-adic expansion
    def _decompose_power(rem: Poly, base: Poly, exp: int) -> list:
        res = []
        cur = rem
        for i in range(exp, 0, -1):
            if cur.is_zero:
                break
            q, r = cur.div(base)
            if not r.is_zero:
                d = base.as_expr() ** i if i > 1 else base.as_expr()
                res.append(r.as_expr() / d)
            cur = q
        return res

    # 4. Helper for coprime factors
    def _decompose_coprime(rem: Poly, f_tuples: list) -> list:
        if not f_tuples:
            return []
        if len(f_tuples) == 1:
            b, e = f_tuples[0]
            return _decompose_power(rem, b, e)

        b1, e1 = f_tuples[0]
        d1 = b1 ** e1

        rest = f_tuples[1:]
        d_rest = rest[0][0] ** rest[0][1]
        for b, e in rest[1:]:
            d_rest = d_rest * (b ** e)

        s, t, h = d1.gcdex(d_rest)
        if not h.is_one:
            lc = h.leading_coeff()
            s = Poly(s.as_expr() / lc, x_sym)
            t = Poly(t.as_expr() / lc, x_sym)

        rv = rem * t
        ru = rem * s
        _, r1 = rv.div(d1)
        _, r_rest = ru.div(d_rest)

        return _decompose_power(r1, b1, e1) + _decompose_coprime(r_rest, rest)

    pf_terms = _decompose_coprime(r_poly, factors)
    if s_expr != 0 and s_expr != Integer(0):
        total = Add(s_expr, *pf_terms)
    else:
        total = Add(*pf_terms) if len(pf_terms) > 1 else (pf_terms[0] if pf_terms else Integer(0))

    return Poly(total, x_sym) if is_poly else total


def poly(expr: Any, *gens: Any, **args: Any) -> Poly:
    """Construct a Poly from an expression."""
    return Poly(expr, *gens, **args)


def div(f: Any, g: Any, *gens: Any, **kwargs: Any) -> Tuple[Any, Any]:
    """Polynomial division with remainder returning (quotient, remainder)."""
    is_poly = isinstance(f, Poly) or isinstance(g, Poly)
    p_f = f if isinstance(f, Poly) else Poly(f, *gens, **kwargs)
    p_g = g if isinstance(g, Poly) else Poly(g, *p_f.gens)
    q, r = p_f.div(p_g)
    if is_poly or kwargs.get("polys", False):
        return (q, r)
    return (q.as_expr(), r.as_expr())


def rem(f: Any, g: Any, *gens: Any, **kwargs: Any) -> Any:
    """Polynomial remainder of f divided by g."""
    return div(f, g, *gens, **kwargs)[1]


def quo(f: Any, g: Any, *gens: Any, **kwargs: Any) -> Any:
    """Polynomial quotient of f divided by g."""
    return div(f, g, *gens, **kwargs)[0]


def sturm(f: Any, *gens: Any, **kwargs: Any) -> List[Poly]:
    """Compute the Sturm sequence of a polynomial."""
    p0 = f if isinstance(f, Poly) else Poly(f, *gens, **kwargs)
    if p0.is_zero:
        return []
    p1 = p0.diff()
    if p1.is_zero:
        return [p0]
    seq = [p0, p1]
    while True:
        r = seq[-2].rem(seq[-1])
        if r.is_zero:
            break
        p_next = -r
        seq.append(p_next)
        if p_next.degree() == 0:
            break
    if not isinstance(f, Poly) and not kwargs.get("polys", True):
        return [p.as_expr() for p in seq]
    return seq


def compose(f: Any, g: Any, *gens: Any, **kwargs: Any) -> Any:
    """Compute functional composition f(g)."""
    is_poly = isinstance(f, Poly) or isinstance(g, Poly)
    p_f = f if isinstance(f, Poly) else Poly(f, *gens, **kwargs)
    res = p_f.compose(g)
    if is_poly or kwargs.get("polys", False):
        return res
    return res.as_expr()


def _find_candidate_h(f_poly: Poly, s: int, x: Any) -> Poly:
    n = f_poly.degree()
    if n is None:
        return Poly(x**s, x)
    r = n // s
    h = Poly(x**s, x)
    for k in range(1, s):
        h_pow = h
        for _ in range(r - 1):
            h_pow = h_pow * h
        diff = f_poly - h_pow
        coeff = diff.nth(n - k)
        c = coeff / r
        if c != 0:
            h = h + Poly(c * x**(s - k), x)
    return h


def _decompose_poly_core(f: Poly) -> List[Poly]:
    if len(f.gens) != 1:
        return [f]
    deg = f.degree()
    if deg is None or deg <= 1:
        return [f]
    x = f.gen
    lc = f.leading_coeff()
    f_monic = f if lc == 1 else Poly(f.as_expr() / lc, x)
    n = f_monic.degree()
    if n is None:
        return [f]
    for s in range(2, n):
        if n % s != 0:
            continue
        r = n // s
        h = _find_candidate_h(f_monic, s, x)
        q = f_monic
        coeffs = []
        possible = True
        for _ in range(r):
            q, remainder = q.div(h)
            if remainder.degree() is not None and remainder.degree() > 0:
                possible = False
                break
            coeffs.append(remainder.nth(0))
        if not possible:
            continue
        if q.degree() is not None and q.degree() > 0:
            continue
        coeffs.append(q.nth(0))
        g_expr = sum(c * x**i for i, c in enumerate(coeffs))
        if lc != 1:
            g_expr = lc * g_expr
        g = Poly(g_expr, x)
        return _decompose_poly_core(g) + _decompose_poly_core(h)
    return [f]


def decompose(f: Any, *gens: Any, **kwargs: Any) -> List[Any]:
    """Compute the functional decomposition of a polynomial."""
    is_poly = isinstance(f, Poly)
    p_f = f if isinstance(f, Poly) else Poly(f, *gens, **kwargs)
    res = _decompose_poly_core(p_f)
    if is_poly or kwargs.get("polys", False):
        return res
    return [p.as_expr() for p in res]


__all__ = [
    "EC",
    "LC",
    "Poly",
    "TC",
    "apart",
    "cancel",
    "compose",
    "content",
    "decompose",
    "degree",
    "discriminant",
    "div",
    "factor",
    "factor_list",
    "gcd",
    "gcdex",
    "groebner",
    "half_gcdex",
    "lcm",
    "monic",
    "poly",
    "primitive",
    "quo",
    "rem",
    "resultant",
    "roots",
    "sqf",
    "sqf_list",
    "sqf_part",
    "sturm",
    "together",
    "terms_gcd",
    "trailing_coeff",
]
