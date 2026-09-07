"""Polynomial tools and representations for FrankenSymPy (WS08, WS09)."""

import math
from typing import Any, List, Optional, Sequence, Tuple, Union
from ..core import (
    Basic,
    Expr,
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


class Poly(Basic):
    """Exact polynomial representation over rational field QQ."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __new__(cls, expr: Any, *gens: Any, **kwargs: Any) -> "Poly":
        if isinstance(expr, Poly):
            if not gens and "gens" not in kwargs:
                return expr
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
                generator_list = sorted(list(free), key=lambda s: s.name)

        obj = object.__new__(cls)
        obj._expr = wrapped_expr
        obj._gens = tuple(generator_list)
        obj._domain = kwargs.get("domain", "QQ")
        return obj

    @property
    def domain(self) -> Any:
        return self._domain

    @property
    def gen(self) -> Symbol:
        return self._gens[0]

    @property
    def gens(self) -> Tuple[Symbol, ...]:
        return self._gens

    def as_expr(self) -> Expr:
        return self._expr

    def all_coeffs(self) -> List[Any]:
        """Return all coefficients of the polynomial in descending degree order."""
        raw = _native.poly_coeffs_expr(str(self._expr), _native_symbol_key(self.gen))
        return [_parse_result(c) for c in raw]

    def coeffs(self) -> List[Any]:
        """Return all non-zero coefficients in descending degree order."""
        return [c for c in self.all_coeffs() if c != 0]

    def degree(self, gen: int = 0) -> Optional[int]:
        """Return polynomial degree."""
        target_gen = self._gens[gen] if isinstance(gen, int) else _require_symbol(gen)
        return _native.poly_degree_expr(str(self._expr), _native_symbol_key(target_gen))

    def leading_coeff(self) -> Any:
        """Return the leading coefficient."""
        raw = _native.poly_leading_coeff_expr(str(self._expr), _native_symbol_key(self.gen))
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
        return Poly(diff(self.as_expr(), *(specs or (self.gen,))), *self._gens)

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
        deg = self.degree()
        if deg is None:
            return True
        coeffs = self.all_coeffs()
        return len(coeffs) == 1 and coeffs[0] == 0

    @property
    def is_one(self) -> bool:
        deg = self.degree()
        if deg != 0:
            return False
        coeffs = self.all_coeffs()
        return len(coeffs) == 1 and coeffs[0] == 1

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
        from ..core import Integer, Rational
        coeffs = self.all_coeffs()
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
        prim_expr = expand(self.as_expr() / cont)
        return (cont, Poly(prim_expr, *self._gens))

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
        raw = _native.poly_monic_expr(str(self._expr), _native_symbol_key(self.gen))
        return Poly(_parse_result(raw), *self._gens)

    def div(self, other: Any) -> Tuple["Poly", "Poly"]:
        """Polynomial division with remainder returning (quotient, remainder)."""
        other_expr = other.as_expr() if isinstance(other, Poly) else _wrap(_native_expr(other))
        q_raw, r_raw = _native.poly_div_rem_expr(
            str(self._expr), str(other_expr), _native_symbol_key(self.gen)
        )
        return (
            Poly(_parse_result(q_raw), *self._gens),
            Poly(_parse_result(r_raw), *self._gens),
        )

    def rem(self, other: Any) -> "Poly":
        return self.div(other)[1]

    def resultant(self, other: Any) -> Any:
        """Resultant with respect to the primary generator."""
        other_expr = other.as_expr() if isinstance(other, Poly) else _wrap(_native_expr(other))
        raw = _native.poly_resultant_expr(
            str(self._expr), str(other_expr), _native_symbol_key(self.gen)
        )
        return _parse_result(raw)

    def discriminant(self) -> Any:
        """Discriminant with respect to the primary generator."""
        raw = _native.poly_discriminant_expr(str(self._expr), _native_symbol_key(self.gen))
        return _parse_result(raw)

    def gcd(self, other: Any) -> "Poly":
        """Greatest common divisor (monic)."""
        other_expr = other.as_expr() if isinstance(other, Poly) else _wrap(_native_expr(other))
        raw = _native.poly_gcd_expr(
            str(self._expr), str(other_expr), _native_symbol_key(self.gen)
        )
        return Poly(_parse_result(raw), *self._gens)

    def lcm(self, other: Any) -> "Poly":
        """Least common multiple (monic)."""
        other_expr = other.as_expr() if isinstance(other, Poly) else _wrap(_native_expr(other))
        raw = _native.poly_lcm_expr(
            str(self._expr), str(other_expr), _native_symbol_key(self.gen)
        )
        return Poly(_parse_result(raw), *self._gens)

    def gcdex(self, other: Any) -> Tuple["Poly", "Poly", "Poly"]:
        """Extended Euclidean algorithm: returns (s, t, h) such that s*self + t*other = h."""
        other_poly = other if isinstance(other, Poly) else Poly(other, *self._gens)
        raw_s, raw_t, raw_h = _native.poly_gcdex_expr(
            str(self._expr), str(other_poly._expr), _native_symbol_key(self.gen)
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
            str(self._expr), str(other_poly._expr), _native_symbol_key(self.gen)
        )
        return Poly(_parse_result(raw), *self._gens)

    def shift(self, a: Any) -> "Poly":
        """Compute polynomial shift self(x + a)."""
        raw = _native.poly_shift_expr(
            str(self._expr), str(a), _native_symbol_key(self.gen)
        )
        return Poly(_parse_result(raw), *self._gens)

    def sqf_list(self) -> Tuple[Any, List[Tuple["Poly", int]]]:
        """Square-free factorization returning (scale, [(factor, multiplicity), ...])."""
        scale_raw, factors_raw = _native.poly_sqf_list_expr(
            str(self._expr), _native_symbol_key(self.gen)
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
        scale_raw, factors_raw = _native.poly_factor_list_expr(
            str(self._expr), _native_symbol_key(self.gen)
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
        prod = terms[0]
        for t in terms[1:]:
            prod = prod * t
        if scale != 1:
            prod = scale * prod
        return prod

    def roots(self) -> dict[Any, int]:
        """Compute polynomial roots over Q with multiplicities."""
        roots_raw = _native.poly_roots_expr(
            str(self._expr), _native_symbol_key(self.gen)
        )
        return {_parse_result(r): mult for r, mult in roots_raw}

    def __add__(self, other: Any) -> "Poly":
        other_expr = other.as_expr() if isinstance(other, Poly) else other
        return Poly(self.as_expr() + other_expr, *self._gens)

    def __radd__(self, other: Any) -> "Poly":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Poly":
        other_expr = other.as_expr() if isinstance(other, Poly) else other
        return Poly(self.as_expr() - other_expr, *self._gens)

    def __rsub__(self, other: Any) -> "Poly":
        other_expr = other.as_expr() if isinstance(other, Poly) else other
        return Poly(other_expr - self.as_expr(), *self._gens)

    def __mul__(self, other: Any) -> "Poly":
        other_expr = other.as_expr() if isinstance(other, Poly) else other
        return Poly(self.as_expr() * other_expr, *self._gens)

    def __rmul__(self, other: Any) -> "Poly":
        return self.__mul__(other)

    def __neg__(self) -> "Poly":
        return Poly(-self.as_expr(), *self._gens)

    def __pos__(self) -> "Poly":
        return self

    def __pow__(self, n: int) -> "Poly":
        if not isinstance(n, int) or n < 0:
            raise ValueError("Polynomial exponent must be a non-negative integer")
        from ..core import expand
        return Poly(expand(self.as_expr() ** n), *self._gens)

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

    def __repr__(self) -> str:
        gen_str = ", ".join(str(g) for g in self._gens)
        return f"Poly({self.as_expr()}, {gen_str}, domain='QQ')"

    def __str__(self) -> str:
        return self.__repr__()


def degree(f: Any, gen: Any = 0) -> Optional[int]:
    """Return polynomial degree."""
    if not isinstance(f, Poly):
        p = Poly(f, gen) if isinstance(gen, Symbol) else Poly(f)
    else:
        p = f
    return p.degree(gen)


def LC(f: Any) -> Any:
    """Return polynomial leading coefficient."""
    if not isinstance(f, Poly):
        p = Poly(f)
    else:
        p = f
    return p.LC()


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
    p_a = Poly(a)
    res = p_a.gcd(b)
    return res.as_expr()


def lcm(a: Any, b: Any) -> Any:
    """Compute polynomial or integer least common multiple."""
    if isinstance(a, int) and isinstance(b, int):
        return abs(a * b) // math.gcd(a, b) if a and b else 0
    if isinstance(a, Poly):
        return a.lcm(b)
    if isinstance(b, Poly):
        return b.lcm(a)
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


def resultant(p: Any, q: Any, x: Any = None) -> Any:
    """Compute the resultant of two polynomials."""
    poly_p = Poly(p, x) if x is not None else (p if isinstance(p, Poly) else Poly(p))
    return poly_p.resultant(q)


def discriminant(p: Any, x: Any = None) -> Any:
    """Compute the discriminant of a polynomial."""
    poly_p = Poly(p, x) if x is not None else (p if isinstance(p, Poly) else Poly(p))
    return poly_p.discriminant()


def sqf_list(p: Any, x: Any = None) -> Tuple[Any, List[Tuple[Any, int]]]:
    """Compute square-free factorization returning (scale, [(factor, multiplicity), ...])."""
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


def groebner(F: Sequence[Any], *gens: Any) -> List[Any]:
    """Compute a Groebner basis under Lexicographical order."""
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

    eq_sources = [str(_wrap(_native_expr(e))) for e in F]
    var_names = [_native_symbol_key(_require_symbol(g)) for g in var_list]
    raw = _native.groebner_basis_expr(eq_sources, var_names)
    return [_parse_result(r) for r in raw]


def factor_list(p: Any, *gens: Any) -> Tuple[Any, List[Tuple[Any, int]]]:
    """Compute polynomial factor list returning (scale, [(factor, multiplicity), ...])."""
    poly_p = p if isinstance(p, Poly) else Poly(p, *gens)
    scale, factors = poly_p.factor_list()
    return (scale, [(f.as_expr(), mult) for f, mult in factors])


def factor(p: Any, *gens: Any) -> Any:
    """Factor polynomial into irreducible factors."""
    if isinstance(p, Poly):
        return p.factor()
    try:
        poly_p = Poly(p, *gens)
        return poly_p.factor()
    except Exception:
        return _wrap(_native_expr(p))


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
        return f if is_poly else numer

    try:
        p_poly = Poly(numer, *gens) if gens else Poly(numer)
        q_poly = Poly(denom, *p_poly.gens)
    except Exception:
        return f if is_poly else wrapped

    g = p_poly.gcd(q_poly)
    p_div = p_poly.div(g)[0]
    q_div = q_poly.div(g)[0]

    p_cont, p_prim = p_div.primitive()
    q_cont, q_prim = q_div.primitive()

    if q_cont == 0:
        return f if is_poly else wrapped

    c = p_cont / q_cont
    num_final = expand(p_prim.as_expr() * (c.p if hasattr(c, "p") else c))
    den_final = expand(q_prim.as_expr() * (c.q if hasattr(c, "q") else 1))

    if den_final == 1 or den_final == Integer(1):
        res = num_final
    else:
        res = num_final / den_final

    if is_poly:
        return Poly(res, *p_poly.gens)
    return res


def poly(expr: Any, *gens: Any, **args: Any) -> Poly:
    """Construct a Poly from an expression."""
    return Poly(expr, *gens, **args)


__all__ = [
    "EC",
    "LC",
    "Poly",
    "TC",
    "cancel",
    "content",
    "degree",
    "discriminant",
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
    "resultant",
    "roots",
    "sqf",
    "sqf_list",
    "sqf_part",
    "trailing_coeff",
]
