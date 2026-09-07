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
            if not gens:
                return expr
            expr = expr.as_expr()

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
        return obj

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


__all__ = [
    "EC",
    "LC",
    "Poly",
    "TC",
    "degree",
    "discriminant",
    "factor",
    "factor_list",
    "gcd",
    "groebner",
    "lcm",
    "monic",
    "resultant",
    "roots",
    "sqf",
    "sqf_list",
    "sqf_part",
    "trailing_coeff",
]
