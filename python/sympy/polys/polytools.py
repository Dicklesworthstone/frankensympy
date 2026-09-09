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

    def degree(self, gen: Any = 0) -> Optional[int]:
        """Return polynomial degree."""
        if len(self._gens) > 1:
            var_names = [_native_symbol_key(g) for g in self._gens]
            target_key = None
            if gen is not None:
                target_gen = self._gens[gen] if isinstance(gen, int) else _require_symbol(gen)
                target_key = _native_symbol_key(target_gen)
            return _native.poly_multivariate_degree_expr(
                str(self._expr), var_names, target_key
            )
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
        other_gens = other._gens if isinstance(other, Poly) else ()
        if len(self._gens) > 1 or len(other_gens) > 1:
            all_gens = tuple(dict.fromkeys(self._gens + other_gens))
            var_names = [_native_symbol_key(g) for g in all_gens]
            q_raw, r_raw = _native.poly_multivariate_div_rem_expr(
                str(self._expr), str(other_expr), var_names
            )
            return (
                Poly(_parse_result(q_raw), *all_gens),
                Poly(_parse_result(r_raw), *all_gens),
            )
        q_raw, r_raw = _native.poly_div_rem_expr(
            str(self._expr), str(other_expr), _native_symbol_key(self.gen)
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
        other_gens = other._gens if isinstance(other, Poly) else ()
        if len(self._gens) > 1 or len(other_gens) > 1:
            all_gens = tuple(dict.fromkeys(self._gens + other_gens))
            var_names = [_native_symbol_key(g) for g in all_gens]
            raw = _native.poly_multivariate_gcd_expr(
                str(self._expr), str(other_expr), var_names
            )
            return Poly(_parse_result(raw), *all_gens)
        raw = _native.poly_gcd_expr(
            str(self._expr), str(other_expr), _native_symbol_key(self.gen)
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
                str(self._expr), str(other_expr), var_names
            )
            return Poly(_parse_result(raw), *all_gens)
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

        if den_final == 1 or den_final == Integer(1):
            res = num_final
        else:
            res = num_final / den_final

        if is_poly:
            return Poly(res, *all_gens)
        return res
    except Exception:
        return f if is_poly else wrapped


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
        return Poly(res, *expr._gens) if is_poly else res

    elif hasattr(wrapped, "args") and wrapped.args:
        new_args = [together(a) for a in wrapped.args]
        if new_args != list(wrapped.args):
            res = type(wrapped)(*new_args)
            return Poly(res, *expr._gens) if is_poly else res

    return expr


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
        free = wrapped.free_symbols
        if not free:
            return expr
        if len(free) == 1:
            x_sym = next(iter(free))
        else:
            x_sym = sorted(list(free), key=lambda s: s.name)[0]
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
    "trailing_coeff",
]
