"""Polynomial tools and representations for FrankenSymPy (WS08, WS09)."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any, List, Optional, Sequence, Tuple, Union
from .polyerrors import DomainError, ExactQuotientFailed, NotInvertible, PolynomialError
from . import _finitefield
from ..core import (
    Basic,
    Mul,
    Pow,
    Expr,
    Function,
    Integer,
    Rational,
    Symbol,
    _lift_native_result,
    _native,
    _native_expr,
    _native_symbol_key,
    _parse_result,
    _require_symbol,
    _symbol_binding,
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
    """Exact polynomial representation over QQ or a bounded prime field.

    ``modulus=p`` and ``domain='GF(p)'`` select checked prime-field arithmetic
    (prime 2..65521, total degree at most 64, up to eight generators). Domain
    and typed generator identity survive every supported operation. The
    factorization and GCD families currently require one generator.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __new__(cls, expr: Any, *gens: Any, **kwargs: Any) -> "Poly":
        if _finitefield.requested(expr, kwargs):
            return _finitefield.construct(cls, expr, gens, kwargs)
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
        if _finitefield.is_finite(self):
            _finitefield.modulus(self)
            return frozenset()
        return frozenset(self._expr.free_symbols - set(self._gens)) | self._extra_syms

    def _like(self, expr: Any, other: Any = None) -> "Poly":
        if _finitefield.is_finite(self):
            return type(self)(expr, *self._gens, modulus=_finitefield.modulus(self),
                              symmetric=self._symmetric)
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
        if _finitefield.is_finite(self):
            return _finitefield.expression(self)
        return self._expr

    def get_modulus(self) -> int:
        """Return the characteristic of a prime-field polynomial."""
        if not _finitefield.is_finite(self):
            raise DomainError("not a polynomial over a prime field")
        return _finitefield.modulus(self)

    @property
    def _parametric(self) -> bool:
        """Coefficients involve symbols outside the generators (domain
        ZZ[a, ...] / QQ(a, ...)); such polynomials take the Python lanes."""
        if _finitefield.is_finite(self):
            _finitefield.modulus(self)
            return False
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
        if _finitefield.is_finite(self):
            return _finitefield.total_degree(self)
        from ..core import S as _S

        if self.as_expr() == 0:
            return _S.NegativeInfinity
        return max(sum(m) for m, _ in self.terms())

    def all_coeffs(self) -> List[Any]:
        """Return all coefficients of the polynomial in descending degree order."""
        if _finitefield.is_finite(self):
            return _finitefield.coefficients(self)
        if self._parametric:
            return self._univariate_coeffs()
        raw = _native.poly_coeffs_native(_native_expr(self._expr), _symbol_binding(self.gen))
        return [_lift_native_result(c, self._expr, *self._gens) for c in raw]

    def coeffs(self, order: Any = None) -> List[Any]:
        """Return all non-zero coefficients in descending degree order."""
        if _finitefield.is_finite(self):
            return [c for _, c in _finitefield.terms(self, order)]
        if order not in (None, "lex"):
            raise NotImplementedError("non-lexicographic coefficient queries require a prime-field Poly")
        return [c for c in self.all_coeffs() if c != 0]

    def degree(self, gen: Any = 0) -> Optional[int]:
        """Return polynomial degree."""
        if _finitefield.is_finite(self):
            return _finitefield.degree(self, gen)
        if self._parametric:
            from ..core import S as _S

            if self.as_expr() == 0:
                return _S.NegativeInfinity
            i = self._gen_index(gen)
            return max(m[i] for m, _ in self.terms())
        if len(self._gens) > 1:
            var_bindings = [_symbol_binding(g) for g in self._gens]
            target_binding = None
            if gen is not None:
                target_gen = self._gens[gen] if isinstance(gen, int) else _require_symbol(gen)
                target_binding = _symbol_binding(target_gen)
            return _native.poly_multivariate_degree_native(
                _native_expr(self._expr), var_bindings, target_binding
            )
        target_gen = self._gens[gen] if isinstance(gen, int) else _require_symbol(gen)
        return _native.poly_degree_native(_native_expr(self._expr), _symbol_binding(target_gen))

    def leading_coeff(self) -> Any:
        """Return the leading coefficient."""
        if _finitefield.is_finite(self):
            return _finitefield.terms(self)[0][1]
        if self._parametric:
            return self.terms()[0][1]
        raw = _native.poly_leading_coeff_native(_native_expr(self._expr), _symbol_binding(self.gen))
        return _lift_native_result(raw, self._expr, *self._gens)

    def LC(self) -> Any:
        """Alias for leading_coeff."""
        return self.leading_coeff()

    def trailing_coeff(self) -> Any:
        """Return the trailing coefficient."""
        if _finitefield.is_finite(self):
            return _finitefield.trailing_coefficient(self)
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
        if _finitefield.is_finite(self):
            return _finitefield.terms(self)[-1][1]
        return self.trailing_coeff()

    def nth(self, *coords: int) -> Any:
        """Return coefficient of the given monomial degree."""
        if _finitefield.is_finite(self):
            return _finitefield.nth(self, coords)
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
        if _finitefield.is_finite(self):
            return _finitefield.evaluate(self, args)
        if len(args) == 1:
            return self.as_expr().subs(self.gen, args[0])
        elif len(args) == 2:
            var, val = args
            return self.as_expr().subs(var, val)
        raise TypeError(f"eval takes 1 or 2 arguments, got {len(args)}")

    def diff(self, *specs: Any) -> "Poly":
        """Differentiate polynomial."""
        if _finitefield.is_finite(self):
            return _finitefield.derivative(self, specs)
        from ..core import diff
        return self._like(diff(self.as_expr(), *(specs or (self.gen,))))

    def integrate(self, *specs: Any) -> "Poly":
        """Integrate polynomial."""
        if _finitefield.is_finite(self):
            return _finitefield.unsupported(self, "integration")
        from ..integrals import integrate
        return Poly(integrate(self.as_expr(), *(specs or (self.gen,))), *self._gens)

    def subs(self, *args: Any, **kwargs: Any) -> Any:
        """Substitute into polynomial expression."""
        if _finitefield.is_finite(self):
            return _finitefield.unsupported(self, "general substitution; use eval or compose")
        return self.as_expr().subs(*args, **kwargs)

    @property
    def free_symbols(self) -> set[Any]:
        return self.as_expr().free_symbols

    @property
    def is_linear(self) -> bool:
        if _finitefield.is_finite(self):
            return self.total_degree() <= 1
        return self.degree() == 1

    @property
    def is_quadratic(self) -> bool:
        if _finitefield.is_finite(self):
            return self.total_degree() <= 2
        return self.degree() == 2

    @property
    def is_zero(self) -> bool:
        if _finitefield.is_finite(self):
            return _finitefield.is_constant(self, 0)
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
        if _finitefield.is_finite(self):
            return _finitefield.is_constant(self, 1)
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
        if _finitefield.is_finite(self):
            _, factors = _finitefield.factors(self)
            return len(factors) == 1 and factors[0][1] == 1
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
        if _finitefield.is_finite(self):
            return Integer(0 if self.is_zero else 1)
        from functools import reduce
        from ..core import Add, Rational
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
        if _finitefield.is_finite(self):
            normalized = _finitefield.operation(self, "normalize")
            expression = object.__getattribute__(normalized, "__dict__")["_expr"]
            return (Integer(0 if expression == 0 else 1), normalized)
        from ..core import expand
        cont = self.content()
        if cont == 0:
            return (Integer(0), Poly(0, *self._gens))
        prim_expr = expand(cancel(self.as_expr() / cont)) if self._parametric else expand(self.as_expr() / cont)
        return (cont, self._like(prim_expr))

    @classmethod
    def from_list(cls, coeffs: Sequence[Any], gens: Any = None, **kwargs: Any) -> "Poly":
        """Construct a polynomial from a list of coefficients in descending degree order."""
        if _finitefield.requested(None, kwargs):
            return _finitefield.from_list(cls, coeffs, gens, kwargs)
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
        return cls(expr, *gens, **kwargs)

    @classmethod
    def from_expr(cls, expr: Any, *gens: Any, **kwargs: Any) -> "Poly":
        """Construct a Poly from an expression."""
        return cls(expr, *gens, **kwargs)

    @classmethod
    def from_poly(cls, poly: "Poly", *gens: Any, **kwargs: Any) -> "Poly":
        """Construct a Poly from another Poly."""
        if _finitefield.requested(poly, kwargs):
            return cls(poly, *gens, **kwargs)
        return cls(poly.as_expr(), *(gens or poly.gens), **kwargs)

    def monic(self) -> "Poly":
        """Return the monic associate of this polynomial."""
        if _finitefield.is_finite(self):
            return _finitefield.operation(self, "monic")
        if self._parametric:
            lc = self.leading_coeff()
            out = Integer(0)
            for m, c in self.terms():
                mono = Integer(1)
                for g, k in zip(self._gens, m):
                    mono = mono * g**k
                out = out + cancel(c / lc) * mono
            return Poly(out, *self._gens)
        raw = _native.poly_monic_native(_native_expr(self._expr), _symbol_binding(self.gen))
        return Poly(_lift_native_result(raw, self._expr, *self._gens), *self._gens)

    def div(self, other: Any) -> Tuple["Poly", "Poly"]:
        """Polynomial division with remainder returning (quotient, remainder)."""
        if _finitefield.is_finite(self):
            return _finitefield.operation(self, "divrem", other)
        if _finitefield.is_finite(other):
            from .polyerrors import UnificationFailed
            raise UnificationFailed("cannot unify rational and prime-field polynomials")
        other_expr = other.as_expr() if isinstance(other, Poly) else _wrap(_native_expr(other))
        other_gens = other._gens if isinstance(other, Poly) else ()
        if len(self._gens) == 1 and len(other_gens) <= 1 and (
            self._parametric or (other_expr.free_symbols - set(self._gens))
        ):
            q, r = _parametric_div(self._expr, other_expr, self.gen)
            return self._like(q, other), self._like(r, other)
        if len(self._gens) > 1 or len(other_gens) > 1:
            all_gens = tuple(dict.fromkeys(self._gens + other_gens))
            var_bindings = [_symbol_binding(g) for g in all_gens]
            q_raw, r_raw = _native.poly_multivariate_div_rem_native(
                _native_expr(self._expr), _native_expr(other_expr), var_bindings
            )
            return (
                Poly(_lift_native_result(q_raw, self._expr, other_expr, *all_gens), *all_gens),
                Poly(_lift_native_result(r_raw, self._expr, other_expr, *all_gens), *all_gens),
            )
        q_raw, r_raw = _native.poly_div_rem_native(
            _native_expr(self._expr), _native_expr(other_expr), _symbol_binding(self.gen)
        )
        return (
            Poly(_lift_native_result(q_raw, self._expr, other_expr, *self._gens), *self._gens),
            Poly(_lift_native_result(r_raw, self._expr, other_expr, *self._gens), *self._gens),
        )

    def rem(self, other: Any) -> "Poly":
        return self.div(other)[1]

    def quo(self, other: Any) -> "Poly":
        return self.div(other)[0]

    def exquo(self, other: Any) -> "Poly":
        """Return an exact polynomial quotient, refusing a nonzero remainder."""
        quotient, remainder = self.div(other)
        if not remainder.is_zero:
            raise ExactQuotientFailed("polynomial division has a nonzero remainder")
        return quotient

    def invert(self, other: Any) -> "Poly":
        """Return the multiplicative inverse modulo another polynomial."""
        if _finitefield.is_finite(self):
            return _finitefield.operation(self, "invert", other)
        s, _, h = self.gcdex(other)
        if h.is_zero or h.degree() != 0:
            raise NotInvertible("zero divisor")
        return self._like(s.as_expr() / h.as_expr()).rem(other)

    def sturm(self) -> List["Poly"]:
        if _finitefield.is_finite(self):
            return _finitefield.unsupported(self, "Sturm sequences")
        return sturm(self)

    def decompose(self) -> List["Poly"]:
        if _finitefield.is_finite(self):
            return _finitefield.unsupported(self, "decomposition")
        return decompose(self, polys=True)

    def resultant(self, other: Any) -> Any:
        """Resultant with respect to the primary generator."""
        if _finitefield.is_finite(self):
            return _finitefield.scalar_operation(self, "resultant", other)
        if _finitefield.is_finite(other):
            from .polyerrors import UnificationFailed
            raise UnificationFailed("cannot unify rational and prime-field polynomials")
        other_expr = other.as_expr() if isinstance(other, Poly) else _wrap(_native_expr(other))
        raw = _native.poly_resultant_native(
            _native_expr(self._expr), _native_expr(other_expr), _symbol_binding(self.gen)
        )
        return _lift_native_result(raw, self._expr, other_expr, *self._gens)

    def discriminant(self) -> Any:
        """Discriminant with respect to the primary generator."""
        if _finitefield.is_finite(self):
            return _finitefield.scalar_operation(self, "discriminant")
        raw = _native.poly_discriminant_native(_native_expr(self._expr), _symbol_binding(self.gen))
        return _lift_native_result(raw, self._expr, *self._gens)

    def gcd(self, other: Any) -> "Poly":
        """Greatest common divisor (monic)."""
        if _finitefield.is_finite(self):
            return _finitefield.operation(self, "gcd", other)
        if _finitefield.is_finite(other):
            return _finitefield.operation(other, "gcd", self)
        other_expr = other.as_expr() if isinstance(other, Poly) else _wrap(_native_expr(other))
        other_gens = other._gens if isinstance(other, Poly) else ()
        if len(self._gens) > 1 or len(other_gens) > 1:
            all_gens = tuple(dict.fromkeys(self._gens + other_gens))
            var_bindings = [_symbol_binding(g) for g in all_gens]
            raw = _native.poly_multivariate_gcd_native(
                _native_expr(self._expr), _native_expr(other_expr), var_bindings
            )
            return Poly(_lift_native_result(raw, self._expr, other_expr, *all_gens), *all_gens)
        raw = _native.poly_gcd_native(
            _native_expr(self._expr), _native_expr(other_expr), _symbol_binding(self.gen)
        )
        return Poly(_lift_native_result(raw, self._expr, other_expr, *self._gens), *self._gens)

    def lcm(self, other: Any) -> "Poly":
        """Least common multiple (monic)."""
        if _finitefield.is_finite(self):
            return _finitefield.operation(self, "lcm", other)
        if _finitefield.is_finite(other):
            return _finitefield.operation(other, "lcm", self)
        other_expr = other.as_expr() if isinstance(other, Poly) else _wrap(_native_expr(other))
        other_gens = other._gens if isinstance(other, Poly) else ()
        if len(self._gens) > 1 or len(other_gens) > 1:
            all_gens = tuple(dict.fromkeys(self._gens + other_gens))
            var_bindings = [_symbol_binding(g) for g in all_gens]
            raw = _native.poly_multivariate_lcm_native(
                _native_expr(self._expr), _native_expr(other_expr), var_bindings
            )
            return Poly(_lift_native_result(raw, self._expr, other_expr, *all_gens), *all_gens)
        raw = _native.poly_lcm_native(
            _native_expr(self._expr), _native_expr(other_expr), _symbol_binding(self.gen)
        )
        return Poly(_lift_native_result(raw, self._expr, other_expr, *self._gens), *self._gens)

    def gcdex(self, other: Any) -> Tuple["Poly", "Poly", "Poly"]:
        """Extended Euclidean algorithm: returns (s, t, h) such that s*self + t*other = h."""
        if _finitefield.is_finite(self):
            return _finitefield.operation(self, "gcdex", other)
        if _finitefield.is_finite(other):
            from .polyerrors import UnificationFailed
            raise UnificationFailed("cannot unify rational and prime-field polynomials")
        other_poly = other if isinstance(other, Poly) else Poly(other, *self._gens)
        raw_s, raw_t, raw_h = _native.poly_gcdex_native(
            _native_expr(self._expr), _native_expr(other_poly._expr), _symbol_binding(self.gen)
        )
        return (
            Poly(_lift_native_result(raw_s, self._expr, other_poly._expr, *self._gens), *self._gens),
            Poly(_lift_native_result(raw_t, self._expr, other_poly._expr, *self._gens), *self._gens),
            Poly(_lift_native_result(raw_h, self._expr, other_poly._expr, *self._gens), *self._gens),
        )

    def half_gcdex(self, other: Any) -> Tuple["Poly", "Poly"]:
        """Half extended Euclidean algorithm: returns (s, h) such that s*self == h (mod other)."""
        s, _, h = self.gcdex(other)
        return s, h

    def compose(self, other: Any) -> "Poly":
        """Compute polynomial composition self(other)."""
        if _finitefield.is_finite(self):
            return _finitefield.operation(self, "compose", other)
        if _finitefield.is_finite(other):
            from .polyerrors import UnificationFailed
            raise UnificationFailed("cannot unify rational and prime-field polynomials")
        other_poly = other if isinstance(other, Poly) else Poly(other, *self._gens)
        raw = _native.poly_compose_native(
            _native_expr(self._expr), _native_expr(other_poly._expr), _symbol_binding(self.gen)
        )
        return Poly(_lift_native_result(raw, self._expr, other_poly._expr, *self._gens), *self._gens)

    def shift(self, a: Any) -> "Poly":
        """Compute polynomial shift self(x + a)."""
        if _finitefield.is_finite(self):
            return _finitefield.shift(self, a)
        raw = _native.poly_shift_native(
            _native_expr(self._expr), _native_expr(a), _symbol_binding(self.gen)
        )
        return Poly(_lift_native_result(raw, self._expr, a, *self._gens), *self._gens)

    def sqf_list(self) -> Tuple[Any, List[Tuple["Poly", int]]]:
        """Square-free factorization returning (scale, [(factor, multiplicity), ...])."""
        if _finitefield.is_finite(self):
            return _finitefield.factors(self, "sqf")
        scale_raw, factors_raw = _native.poly_sqf_list_native(
            _native_expr(self._expr), _symbol_binding(self.gen)
        )
        scale = _lift_native_result(scale_raw, self._expr, *self._gens)
        factors = [
            (Poly(_lift_native_result(f, self._expr, *self._gens), *self._gens), mult)
            for f, mult in factors_raw
        ]
        return (scale, factors)

    def sqf_part(self) -> "Poly":
        """Square-free part of the polynomial."""
        if _finitefield.is_finite(self):
            return _finitefield.factors(self, "sqf_part")
        scale, factors = self.sqf_list()
        res = scale
        for f, _ in factors:
            res = res * f.as_expr()
        return Poly(res, *self._gens)

    def factor_list(self) -> Tuple[Any, List[Tuple["Poly", int]]]:
        """Factorization over Q returning (scale, [(factor, multiplicity), ...])."""
        if _finitefield.is_finite(self):
            return _finitefield.factors(self)
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
        scale_raw, factors_raw = _native.poly_factor_list_native(
            _native_expr(self._expr), _symbol_binding(self.gen)
        )
        scale = _lift_native_result(scale_raw, self._expr, *self._gens)
        factors = [
            (Poly(_lift_native_result(f, self._expr, *self._gens), *self._gens), mult)
            for f, mult in factors_raw
        ]
        return (scale, factors)

    def factor(self) -> Expr:
        """Factor polynomial into a product of irreducible rational factors."""
        if _finitefield.is_finite(self):
            return _finitefield.factors(self, expression=True)
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

    def all_roots(self, multiple: bool = True, radicals: bool = True) -> list:
        if _finitefield.is_finite(self):
            return _finitefield.unsupported(self, "algebraic roots")
        from .rootoftools import all_roots as _ar

        return _ar(self, multiple=multiple, radicals=radicals)

    def real_roots(self, multiple: bool = True, radicals: bool = True) -> list:
        if _finitefield.is_finite(self):
            return _finitefield.unsupported(self, "real roots")
        from .rootoftools import real_roots as _rr

        return _rr(self, multiple=multiple, radicals=radicals)

    def nroots(self, n: int = 15, maxsteps: int = 50, cleanup: bool = True) -> list:
        if _finitefield.is_finite(self):
            return _finitefield.unsupported(self, "numerical roots")
        from .. import nroots as _nroots

        return _nroots(self, n=n, maxsteps=maxsteps, cleanup=cleanup)

    def roots(self) -> dict[Any, int]:
        """Compute polynomial roots over Q with multiplicities."""
        if _finitefield.is_finite(self):
            return _finitefield.unsupported(self, "roots")
        if self._parametric:
            return _parametric_roots(self)
        roots_raw = _native.poly_roots_native(
            _native_expr(self._expr), _symbol_binding(self.gen)
        )
        return {_lift_native_result(r, self._expr, *self._gens): mult for r, mult in roots_raw}

    def __add__(self, other: Any) -> "Poly":
        if _finitefield.is_finite(self):
            return _finitefield.operation(self, "add", other)
        if _finitefield.is_finite(other):
            return _finitefield.operation(other, "add", self)
        other_expr = other.as_expr() if isinstance(other, Poly) else other
        return self._like(self.as_expr() + other_expr, other)

    def __radd__(self, other: Any) -> "Poly":
        return self.__add__(other)

    def __sub__(self, other: Any) -> "Poly":
        if _finitefield.is_finite(self):
            return _finitefield.operation(self, "sub", other)
        if _finitefield.is_finite(other):
            return _finitefield.operation(other, "sub", self, reverse=True)
        other_expr = other.as_expr() if isinstance(other, Poly) else other
        return self._like(self.as_expr() - other_expr, other)

    def __rsub__(self, other: Any) -> "Poly":
        if _finitefield.is_finite(self):
            return _finitefield.operation(self, "sub", other, reverse=True)
        other_expr = other.as_expr() if isinstance(other, Poly) else other
        return self._like(other_expr - self.as_expr(), other)

    def __mul__(self, other: Any) -> "Poly":
        if _finitefield.is_finite(self):
            return _finitefield.operation(self, "mul", other)
        if _finitefield.is_finite(other):
            return _finitefield.operation(other, "mul", self)
        other_expr = other.as_expr() if isinstance(other, Poly) else other
        return self._like(self.as_expr() * other_expr, other)

    def __rmul__(self, other: Any) -> "Poly":
        return self.__mul__(other)

    def __neg__(self) -> "Poly":
        if _finitefield.is_finite(self):
            return _finitefield.operation(self, "neg")
        return self._like(-self.as_expr())

    def __pos__(self) -> "Poly":
        if _finitefield.is_finite(self):
            return _finitefield.operation(self, "normalize")
        return self

    def __pow__(self, n: int) -> "Poly":
        if _finitefield.is_finite(self):
            return _finitefield.power(self, n)
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

    add = __add__
    sub = __sub__
    mul = __mul__
    neg = __neg__
    pow = __pow__

    def __eq__(self, other: Any) -> bool:
        if _finitefield.is_finite(self):
            return _finitefield.equal(self, other)
        if _finitefield.is_finite(other):
            return False
        if not isinstance(other, Poly):
            # Upstream converts the other side into this ring first
            # (Poly(x**2, x) == x**2 is True).
            try:
                other = type(self)(other, *self._gens)
            except Exception:
                return False
        if isinstance(other, Poly):
            if self._gens != other._gens and not (
                isinstance(self, PurePoly) and len(self._gens) == len(other._gens)
            ):
                return False
            if isinstance(self, PurePoly) and self._gens != other._gens:
                other = Poly(other.as_expr().subs(dict(zip(other._gens, self._gens))), *self._gens)
            if self.as_expr() == other.as_expr():
                return True
            try:
                return self.all_coeffs() == other.all_coeffs()
            except Exception:
                return False
        return False

    def __hash__(self) -> int:
        if _finitefield.is_finite(self):
            return _finitefield.identity_hash(self)
        return hash((self._expr, self._gens))

    def terms(self, order: Any = None) -> list:
        """(monomial exponents, coefficient) pairs, lex-descending in gens."""
        if _finitefield.is_finite(self):
            return _finitefield.terms(self, order)
        if order not in (None, "lex"):
            raise NotImplementedError("non-lexicographic term queries require a prime-field Poly")
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

    def monoms(self, order: Any = None) -> list:
        """Monomial exponent tuples in the requested supported order."""
        return [monomial for monomial, _ in self.terms(order)]

    def degree_list(self) -> tuple:
        """Degree in each ordered polynomial generator."""
        if _finitefield.is_finite(self):
            return _finitefield.degree_list(self)
        return tuple(self.degree(index) for index in range(len(self._gens)))

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
        if _finitefield.is_finite(self):
            return "GF(%d)" % _finitefield.modulus(self)
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
        if _finitefield.is_finite(self):
            return "Poly(%s, %s, modulus=%d)" % (" ".join(terms), ", ".join(gens), self.get_modulus())
        return "Poly(%s, %s, domain='%s')" % (" ".join(terms), ", ".join(gens), self.domain)

    def __str__(self) -> str:
        return self.__repr__()


class PurePoly(Poly):
    """A polynomial whose identity ignores the generator names (upstream
    ``PurePoly``: charpoly results compare equal across symbols)."""

    def __repr__(self) -> str:
        return "Pure" + super().__repr__()

    __str__ = __repr__

    def __hash__(self) -> int:
        if _finitefield.is_finite(self):
            return _finitefield.identity_hash(self)
        return hash(("PurePoly", tuple(c for _, c in self.terms())))


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


def invert(f: Any, g: Any, *gens: Any, **kwargs: Any) -> Any:
    """Inverse of ``f`` modulo ``g`` (integers or univariate polynomials)."""
    modular = _finitefield.binary_entry(f, g, gens, kwargs, "invert")
    if modular is not None:
        return modular
    if kwargs:
        raise TypeError("unsupported inverse options outside a prime field")
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
    if _finitefield.is_finite(f):
        return f.EC()
    return trailing_coeff(f, *gens)


def monic(f: Any) -> Any:
    """Return monic polynomial."""
    if not isinstance(f, Poly):
        return Poly(f).monic().as_expr()
    return f.monic()


def gcd(a: Any, b: Any, *gens: Any, **kwargs: Any) -> Any:
    """Compute polynomial or integer greatest common divisor."""
    modular = _finitefield.binary_entry(a, b, gens, kwargs, "gcd")
    if modular is not None:
        return modular
    if gens or kwargs:
        raise TypeError("explicit gcd options currently require a prime field")
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


def lcm(a: Any, b: Any, *gens: Any, **kwargs: Any) -> Any:
    """Compute polynomial or integer least common multiple."""
    modular = _finitefield.binary_entry(a, b, gens, kwargs, "lcm")
    if modular is not None:
        return modular
    if gens or kwargs:
        raise TypeError("explicit lcm options currently require a prime field")
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


def gcdex(f: Any, g: Any, x: Any = None, **kwargs: Any) -> Tuple[Any, Any, Any]:
    """Extended Euclidean algorithm: returns (s, t, h) such that s*f + t*g = h = gcd(f, g)."""
    modular = _finitefield.binary_entry(f, g, (x,) if x is not None else (), kwargs, "gcdex")
    if modular is not None:
        return modular
    if kwargs:
        raise TypeError("explicit gcdex options currently require a prime field")
    p_f = Poly(f, x) if x is not None else (f if isinstance(f, Poly) else Poly(f))
    p_g = Poly(g, x) if x is not None else (g if isinstance(g, Poly) else Poly(g, *p_f._gens))
    s, t, h = p_f.gcdex(p_g)
    if isinstance(f, Poly) or isinstance(g, Poly):
        return s, t, h
    return s.as_expr(), t.as_expr(), h.as_expr()


def half_gcdex(f: Any, g: Any, x: Any = None, **kwargs: Any) -> Tuple[Any, Any]:
    """Half extended Euclidean algorithm: returns (s, h) such that s*f == h (mod g)."""
    s, _, h = gcdex(f, g, x, **kwargs)
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


def resultant(p: Any, q: Any, x: Any = None, **kwargs: Any) -> Any:
    """Resultant of two polynomials in ``x`` in their declared coefficient domain."""
    modular = _finitefield.binary_entry(p, q, (x,) if x is not None else (), kwargs, "resultant")
    if modular is not None:
        return modular
    if kwargs:
        raise TypeError("explicit resultant options currently require a prime field")
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


def discriminant(p: Any, x: Any = None, **kwargs: Any) -> Any:
    """Discriminant specialized to the declared coefficient domain."""
    modular = _finitefield.unary_entry(p, (x,) if x is not None else (), kwargs, "discriminant")
    if modular is not None:
        return modular
    if kwargs:
        raise TypeError("explicit discriminant options currently require a prime field")
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
    if _finitefield.is_finite(p):
        if x is not None:
            _finitefield.degree(p, x)
        return p.sqf_list()
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
    if _finitefield.is_finite(p):
        if x is not None:
            _finitefield.degree(p, x)
        return p.sqf_part()
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


def _rational_ideal_source(value: Any) -> Any:
    """Unwrap a Poly only after admitting its retained coefficient domain.

    An expression lifted from GF(p) is an integer representative, not the
    original polynomial over QQ. Losing that distinction changes ideals.
    Read exact built-in storage without invoking an overridable as_expr.
    An explicit None domain requests inference; native source admission still
    requires exact rational coefficients in the declared generators.
    """
    if not isinstance(value, Poly):
        return value
    if type(value) is not Poly and type(value) is not PurePoly:
        raise TypeError("checked ideal operations require exact Poly operands")
    namespace = object.__getattribute__(value, "__dict__")
    domain = namespace.get("_domain")
    if ("_modulus" in namespace or "_domain" not in namespace
            or (domain is not None
                and (type(domain) is not str or domain not in ("ZZ", "QQ")))):
        raise NotImplementedError("checked ideal operations support only ZZ and QQ Poly inputs")
    if "as_expr" in namespace:
        raise TypeError("checked ideal operations do not invoke overridden Poly.as_expr")
    return namespace["_expr"]


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

    @property
    def polys(self) -> list:
        from . import _finitefield_groebner as finite

        if finite.has_field_marker(self):
            return finite.polys(self)
        return [Poly(expression, *self.gens, domain=self.domain) for expression in self.exprs]

    def __iter__(self):
        from . import _finitefield_groebner as finite

        if finite.has_field_marker(self):
            return finite.iterate(self)
        return iter(self.exprs)

    def __len__(self) -> int:
        from . import _finitefield_groebner as finite

        if finite.has_field_marker(self):
            return finite.length(self)
        return len(self.exprs)

    def __getitem__(self, i: Any) -> Any:
        from . import _finitefield_groebner as finite

        if finite.has_field_marker(self):
            return finite.getitem(self, i)
        return self.exprs[i]

    def __eq__(self, other: Any) -> bool:
        from . import _finitefield_groebner as finite

        if finite.has_field_marker(self):
            return finite.equals(self, other)
        if isinstance(other, GroebnerBasis):
            if finite.has_field_marker(other):
                return False
            return (self.exprs == other.exprs and self.gens == other.gens
                    and self.domain == other.domain and self.order == other.order)
        if isinstance(other, (list, tuple)):
            return self.exprs == list(other)
        return NotImplemented

    def __hash__(self) -> int:
        from . import _finitefield_groebner as finite

        if finite.has_field_marker(self):
            return finite.basis_hash(self)
        return hash((tuple(self.exprs), self.gens, self.domain, self.order))

    def _native_basis(self) -> tuple:
        from . import _finitefield_groebner as finite

        if finite.has_field_marker(self):
            finite.unsupported(self)
        if str(self.domain) not in ("ZZ", "QQ"):
            raise NotImplementedError("checked Groebner operations currently support ZZ and QQ")
        return ([_native_expr(_rational_ideal_source(e)) for e in self.exprs],
                [_symbol_binding(g) for g in self.gens], self.order)

    @property
    def is_zero_dimensional(self) -> bool:
        """Whether pure leading powers bound every generator in the quotient.

        The current basis is independently checked before this criterion is
        applied, including when its public expression list has been modified.
        """
        from . import _finitefield_groebner as finite

        if finite.has_field_marker(self):
            return finite.is_zero_dimensional(self)
        return _native.groebner_is_zero_dimensional_native(*self._native_basis())

    def reduce(self, expr: Any, auto: bool = True) -> tuple:
        """Return exact quotients and the reduced remainder in the basis order.

        The default lifts ZZ coefficients to QQ, as in upstream. The native
        checker verifies both the coefficient identity and complete reduction.
        Integer-ring division with ``auto=False`` is not implemented.
        """
        from . import _finitefield_groebner as finite

        if finite.has_field_marker(self):
            return finite.reduce(self, expr, auto)
        if not auto and str(self.domain) == "ZZ":
            raise NotImplementedError("Groebner reduction over ZZ with auto=False is not implemented")
        if isinstance(expr, Poly):
            expression = _rational_ideal_source(expr)
            if expr.gens != self.gens:
                raise ValueError("polynomial generators do not match the Groebner basis")
            expr = expression
        raw_quotients, raw_remainder = _native.groebner_reduce_native(
            *self._native_basis(), _native_expr(expr), str(self.domain) == "ZZ"
        )
        context = (expr, *self.exprs, *self.gens)
        return ([_lift_native_result(q, *context) for q in raw_quotients],
                _lift_native_result(raw_remainder, *context))

    def contains(self, expr: Any) -> bool:
        """Test ideal membership using a checked complete basis and normal form."""
        from . import _finitefield_groebner as finite

        if finite.has_field_marker(self):
            return finite.contains(self, expr)
        return self.reduce(expr)[1] == 0

    def fglm(self, order: str) -> GroebnerBasis:
        """Convert a finite rational quotient to another monomial order.

        Supports lex, grlex and grevlex with at most 64 standard monomials.
        Conversion uses exact quotient-space dependencies and independently
        checks both ideal inclusions and the target basis S-pairs.
        """
        from . import _finitefield_groebner as finite

        if finite.has_field_marker(self):
            finite.unsupported(self)
        if order not in ("lex", "grlex", "grevlex"):
            raise ValueError("unknown monomial order %r" % order)
        if order == self.order:
            return self
        if not self.is_zero_dimensional:
            raise NotImplementedError("FGLM conversion requires a zero-dimensional ideal")
        raw = _native.groebner_fglm_native(*self._native_basis(), order)
        exprs = [_lift_native_result(value, *self.exprs, *self.gens) for value in raw]
        if str(self.domain) == "ZZ":
            exprs = [_primitive_integer(value, self.gens, order) for value in exprs]
        return GroebnerBasis(exprs, self.gens, self.domain, order)

    def _ideal_inputs(self) -> tuple:
        """Admit current expressions as generators of an ideal over QQ."""
        from . import _finitefield_groebner as finite

        if finite.has_field_marker(self):
            finite.unsupported(self)
        if str(self.domain) not in ("ZZ", "QQ"):
            raise NotImplementedError("checked ideal operations support rational coefficients")
        if len(self.gens) > 8:
            raise ValueError("checked ideal operations support at most eight generators")
        gens = tuple(_require_symbol(g) for g in self.gens)
        for generator in gens:
            _native.ideal_source_preflight_native(_native_expr(generator))
        if len(set(gens)) != len(gens):
            raise ValueError("checked ideal operations require distinct typed generators")
        sources = []
        for expression in self.exprs:
            if len(sources) == 64:
                raise ValueError("checked ideal operations support at most 64 polynomials")
            sources.append(_ideal_polynomial_source(expression))
        return sources, gens, [_native_expr(source) for source in sources], [_symbol_binding(g) for g in gens]

    def eliminate(self, *generators: Any) -> GroebnerBasis:
        """Return the complete elimination ideal in lexicographic order.

        This FrankenSymPy extension eliminates the specified typed generators,
        including generators outside the original leading block. Remaining
        generators keep their original order. Eliminating all variables yields
        the constant ring, represented by an empty basis or the unit basis.
        ZZ input coefficients are explicitly extended to the rational field.
        A retained GF(p) basis keeps its characteristic and output policies.
        """
        from . import _finitefield_groebner as finite

        if finite.has_field_marker(self):
            from . import _finitefield_ideals
            return _finitefield_ideals.eliminate(self, generators)
        if len(generators) == 1 and isinstance(generators[0], (list, tuple)):
            generators = tuple(generators[0])
        if len(generators) > 8:
            raise ValueError("too many elimination generators")
        eliminated = tuple(_require_symbol(g) for g in generators)
        sources, gens, equations, bindings = self._ideal_inputs()
        if len(set(eliminated)) != len(eliminated) or any(g not in gens for g in eliminated):
            raise ValueError("elimination requires distinct generators from the original typed ring")
        raw = _native.ideal_eliminate_native(equations, bindings, [_symbol_binding(g) for g in eliminated])
        remaining = tuple(g for g in gens if g not in eliminated)
        values = [_lift_native_result(value, *sources, *gens) for value in raw]
        return GroebnerBasis(values, remaining, "QQ", "lex")

    def intersection(self, other: GroebnerBasis) -> GroebnerBasis:
        """Return the complete intersection of two polynomial ideals.

        This FrankenSymPy extension uses an independently checked auxiliary
        ideal and complete elimination. Both ideals must use the same typed
        variables, but their variable and monomial orders may differ. The
        result uses this basis's variable order and QQ coefficients in lex order.
        Two GF(p) operands instead retain their common characteristic and this
        basis's coefficient lift and polynomial output policies.
        """
        from . import _finitefield_groebner as finite

        if finite.has_field_marker(self):
            from . import _finitefield_ideals
            return _finitefield_ideals.intersection(self, other)
        if type(other) is not GroebnerBasis:
            raise TypeError("ideal intersection requires a GroebnerBasis")
        if finite.has_field_marker(other):
            finite._BasisState(other)
            from .polyerrors import UnificationFailed
            raise UnificationFailed("ideal intersection requires the same coefficient field")
        sources, gens, equations, bindings = self._ideal_inputs()
        other_sources, other_gens, other_equations, _ = other._ideal_inputs()
        if set(gens) != set(other_gens):
            raise ValueError("ideal intersection requires the same typed polynomial ring")
        raw = _native.ideal_intersection_native(equations, other_equations, bindings)
        values = [_lift_native_result(value, *sources, *other_sources, *gens) for value in raw]
        return GroebnerBasis(values, gens, "QQ", "lex")

    def quotient(self, other: GroebnerBasis) -> GroebnerBasis:
        """Return the complete finite ideal quotient I:J in its coefficient field.

        This FrankenSymPy extension returns all polynomials whose product with
        every member of J belongs to I. Multiplicity is retained: <x**3*y>:<x>
        is <x**2*y>, whereas saturation by x yields <y>. Both operands must
        declare the same typed polynomial ring; their variable and monomial
        orders may differ. The result uses this operand's variable order, QQ
        coefficients, and lex order. ZZ coefficients are extended to QQ.
        Two GF(p) operands instead retain their common characteristic and this
        basis's coefficient lift and polynomial output policies.

        Each principal quotient uses a complete checked intersection and
        independently checked exact division; intersections combine all
        divisor constraints. Nonzero divisors support at most seven variables
        and 63 numerator generators, with at most 64 divisor generators over QQ.
        Over GF(p), these counts are 31 numerator and 32 divisor generators;
        an all-zero divisor list permits eight variables and 32 numerator
        generators. Intermediate modular intersections remain bounded too.
        Unsupported inputs, mutation, resource exhaustion and verifier failure
        raise rather than returning a partial ideal.
        """
        from . import _finitefield_groebner as finite

        if type(self) is not GroebnerBasis or type(other) is not GroebnerBasis:
            raise TypeError("ideal quotient requires exact GroebnerBasis operands")
        if finite.has_field_marker(self):
            from . import _finitefield_ideals
            return _finitefield_ideals.quotient(self, other)
        if finite.has_field_marker(other):
            finite._BasisState(other)
            from .polyerrors import UnificationFailed
            raise UnificationFailed("ideal quotient requires the same coefficient field")
        snapshot = _IdealQuotientSnapshot(self, other)
        left, right = snapshot.inputs()
        sources, gens, equations, bindings = left
        other_sources, other_gens, other_equations, _ = right
        if set(gens) != set(other_gens):
            raise ValueError("ideal quotient requires the same typed polynomial ring")
        raw = _native.ideal_quotient_native(equations, other_equations, bindings)
        snapshot.unchanged()
        values = [_lift_native_result(value, *sources, *other_sources, *gens) for value in raw]
        snapshot.unchanged()
        return GroebnerBasis(values, gens, "QQ", "lex")

    def syzygies(self) -> list:
        """Return generators of all polynomial relations on the current exprs.

        This FrankenSymPy extension returns a list of tuples over QQ. A tuple
        ``(a_1, ..., a_m)`` means ``sum(a_i*exprs[i]) == 0`` for this basis's
        current ordered expression list, retaining duplicate and zero entries.
        The tuples generate the complete relation module; they are not promised
        to form a minimal or reduced module basis. ZZ coefficients extend to QQ.

        Generation and independent verification use the declared lex, grlex or
        grevlex order. The bounded lane supports eight variables, 64 input
        polynomials and 16384 output polynomial cells. Unsupported sources or
        exhausted work raise without publishing a partial relation module.
        """
        snapshot = _IdealQuotientSnapshot(self)
        (source_data,) = snapshot.inputs()
        sources, gens, equations, bindings = source_data
        order = snapshot.bases[0][2][3]
        raw = _native.polynomial_syzygies_native(equations, bindings, order)
        snapshot.unchanged()
        relations = [tuple(_lift_native_result(value, *sources, *gens) for value in row)
                     for row in raw]
        snapshot.unchanged()
        return relations

    def invert(self, polynomial: Any) -> Any:
        """Return the reduced multiplicative inverse modulo this ideal over QQ.

        This FrankenSymPy extension supports multivariate and positive-
        dimensional quotient rings. A complete checked augmented ideal proves
        whether an inverse exists; accepted inverses are independently checked
        and reduced in this basis's declared monomial order. Current exprs need
        not already form a Groebner basis. ZZ coefficients extend to QQ.

        ``NotInvertible`` means the element is proved to be a nonunit. Resource,
        admission and verification failures remain distinct exceptions. In the
        zero quotient ring (the unit ideal), the inverse representative is zero,
        including for the zero element. There are at most eight variables and
        63 ideal generators in the bounded lane.
        """
        # A second retained source container includes the element in the same
        # immutable snapshot without invoking its public Poly.as_expr protocol.
        element_basis = GroebnerBasis([polynomial], (), "QQ", "lex")
        snapshot = _IdealQuotientSnapshot(self, element_basis)
        source_data, element_data = snapshot.inputs()
        sources, gens, equations, bindings = source_data
        element_sources, _, element_equations, _ = element_data
        order = snapshot.bases[0][2][3]
        raw = _native.ideal_inverse_native(equations, element_equations[0], bindings, order)
        snapshot.unchanged()
        if raw is None:
            raise NotInvertible("polynomial is not invertible modulo the ideal over QQ")
        inverse = _lift_native_result(raw, *sources, *element_sources, *gens)
        snapshot.unchanged()
        return inverse

    def quotient_basis(self) -> tuple:
        """Return the complete monomial basis of the finite quotient QQ[gens]/I.

        This FrankenSymPy extension returns standard monomials in ascending
        order for this basis's declared lex, grlex or grevlex order. The ideal
        is generated by the current exprs, which are independently checked
        rather than assumed to remain a Groebner basis. Nilpotents retain their
        multiplicity. ZZ coefficients are explicitly extended to QQ.

        At most 64 monomials, eight variables and 64 ideal generators are
        supported. The unit ideal has the empty basis; the constant ring QQ
        has basis (1,). Infinite-dimensional quotients and exhausted work
        bounds raise without returning an incomplete basis.
        """
        snapshot, context, monomials, _ = _quotient_operator_data(self, Integer(0))
        result = tuple(_lift_native_result(value, *context) for value in monomials)
        snapshot.unchanged()
        return result

    def multiplication_matrix(self, element: Any) -> Any:
        """Return the exact matrix of multiplication by element modulo I.

        Column j contains the coordinates of element*quotient_basis()[j].
        The row and column basis is the ascending standard-monomial basis for
        the declared order, including nilpotent directions. Independent exact
        coefficient checks cover every column and the complete quotient basis.
        The matrix acts on coefficient column vectors over QQ.

        This FrankenSymPy extension has the same finite-dimensional bounds as
        quotient_basis(). The unit ideal yields a 0 by 0 matrix. Unsupported
        sources, mutation and operational failures raise; no partial matrix or
        numerical rank approximation is returned.
        """
        from ..matrices import Matrix

        snapshot, context, monomials, rows = _quotient_operator_data(self, element)
        dimension = len(monomials)
        if len(rows) != dimension or any(len(row) != dimension for row in rows):
            raise ValueError("checked quotient multiplication returned an invalid matrix shape")
        entries = [_lift_native_result(value, *context) for row in rows for value in row]
        result = Matrix(dimension, dimension, entries)
        snapshot.unchanged()
        return result

    def minimal_polynomial(self, element: Any, gen: Any = None) -> Any:
        """Return the monic minimal polynomial of element in the full quotient.

        This FrankenSymPy extension computes the complete kernel of
        QQ[gen] -> QQ[gens]/I, gen -> element. A None result proves that no
        nonzero polynomial annihilates the element. Positive-dimensional
        quotient rings are supported, and nilpotent multiplicities are retained;
        this does not select one root or algebraic embedding of the ideal.

        gen must be an exact Symbol or Dummy distinct from every typed ring
        generator; omitting it creates a fresh Dummy. The unit ideal returns
        1. ZZ coefficients extend to QQ. At most seven ring variables and 63
        ideal generators are supported. Admission, mutation, resource and
        verifier failures raise rather than being reported as transcendence.
        """
        from ..core import Dummy, _surface_binding_key

        parameter = Dummy("t") if gen is None else gen
        if type(parameter) is not Symbol and type(parameter) is not Dummy:
            raise TypeError("quotient minimal polynomial requires an exact Symbol or Dummy")
        element_basis = GroebnerBasis([element], (), "QQ", "lex")
        parameter_basis = GroebnerBasis([], (parameter,), "QQ", "lex")
        snapshot = _IdealQuotientSnapshot(self, element_basis, parameter_basis)
        if len(snapshot.bases[0][3]) > 63 or len(snapshot.bases[0][4]) > 7:
            raise ValueError("quotient minimal polynomials support seven variables and 63 ideal generators")
        source_data, element_data, parameter_data = snapshot.inputs()
        sources, gens, equations, bindings = source_data
        element_sources, _, element_equations, _ = element_data
        _, _, _, parameter_bindings = parameter_data
        raw_parameter = snapshot.slot(Basic, "_value").__get__(parameter, Basic)
        if type(raw_parameter) is not _native.Expr or not raw_parameter.is_symbol:
            raise ValueError("quotient minimal polynomial has an invalid output symbol")
        if (_surface_binding_key(raw_parameter.symbol_binding)
                != _surface_binding_key(parameter_bindings[0])):
            raise ValueError("quotient minimal polynomial output symbol has inconsistent typed assumptions")
        if any(raw_parameter == _native_expr(symbol) for symbol in gens):
            raise ValueError("quotient minimal polynomial requires a distinct typed output variable")
        raw = _native.quotient_minimal_polynomial_native(
            equations, element_equations[0], bindings, parameter_bindings[0],
        )
        snapshot.unchanged()
        if raw is None:
            return None
        result = _lift_native_result(raw, *sources, *element_sources, *gens, parameter)
        snapshot.unchanged()
        return result

    minpoly = minimal_polynomial

    def saturate(self, polynomial: Any) -> GroebnerBasis:
        """Return the ideal saturation I:polynomial**infinity.

        This FrankenSymPy extension returns the algebraic closure of the part
        away from the polynomial's zero set. It is not a domain-filtered set
        of solutions: saturating <x*y> by x gives <y>, whose variety still
        contains the origin. Denominator exclusions must be retained separately
        by callers solving rational equations. Results use QQ and lex order.
        A retained GF(p) basis instead keeps its prime field and output policies;
        the ideal is not restricted to points with coordinates in GF(p).
        """
        from . import _finitefield_groebner as finite

        if finite.has_field_marker(self):
            from . import _finitefield_ideals
            return _finitefield_ideals.saturate(self, polynomial)
        sources, gens, equations, bindings = self._ideal_inputs()
        factor = _ideal_polynomial_source(polynomial)
        raw = _native.ideal_saturation_native(equations, bindings, _native_expr(factor))
        values = [_lift_native_result(value, *sources, factor, *gens) for value in raw]
        return GroebnerBasis(values, gens, "QQ", "lex")

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


_IDEAL_ABSENT = object()


class _IdealQuotientSnapshot:
    """Retain bounded source storage for checked polynomial operations.

    Native inputs are immutable. The mutable public basis lists and retained
    Python children must still describe those inputs when a result is lifted.
    Native-owned children remain borrowed; this never calls Python ``args``.
    """

    def __init__(self, *bases):
        from ..core import Add, Dummy, Half, NegativeOne, One, Zero
        from . import _finitefield_groebner as finite

        self.bases = []
        self.slots = []
        self.dictionaries = []
        self.fields = []
        pending = []
        for basis in bases:
            if type(basis) is not GroebnerBasis:
                raise TypeError("ideal quotient requires exact GroebnerBasis operands")
            if finite.has_field_marker(basis):
                finite.unsupported(basis)
            namespace = object.__getattribute__(basis, "__dict__")
            if type(namespace) is not dict:
                raise TypeError("ideal quotient requires built-in source dictionaries")
            values = tuple(namespace.get(name, _IDEAL_ABSENT)
                           for name in ("exprs", "gens", "domain", "order"))
            expressions, generators, domain, order = values
            if ((type(expressions) is not list and type(expressions) is not tuple)
                    or (type(generators) is not list and type(generators) is not tuple)):
                raise TypeError("ideal quotient requires built-in expression and generator containers")
            if len(expressions) > 64 or len(generators) > 8:
                raise ValueError("ideal quotient exceeds its input count limits")
            if type(domain) is not str or domain not in ("QQ", "ZZ"):
                raise NotImplementedError("checked ideal operations support rational coefficients")
            if type(order) is not str or order not in ("lex", "grlex", "grevlex"):
                raise ValueError("unknown ideal monomial order")
            expressions, generators = tuple(expressions), tuple(generators)
            self.bases.append((basis, namespace, values, expressions, generators))
            pending.extend((node, 0) for node in (*expressions, *generators))

        atoms = (Integer, Rational, Zero, One, NegativeOne, Half, Symbol, Dummy)
        seen = set()
        count = 0
        while pending:
            node, depth = pending.pop()
            count += 1
            if count > 65_536 or depth > 64:
                raise ValueError("ideal quotient source exceeds its bounded expression envelope")
            if id(node) in seen:
                continue
            seen.add(id(node))
            kind = type(node)
            if kind is int:
                if node.bit_length() > 4096:
                    raise ValueError("ideal coefficient exceeds 4096 bits")
                continue
            if kind is Poly or kind is PurePoly:
                namespace = object.__getattribute__(node, "__dict__")
                if type(namespace) is not dict:
                    raise TypeError("ideal quotient requires built-in source dictionaries")
                expression = _rational_ideal_source(node)
                self.dictionaries.append((namespace, tuple(namespace.items())))
                self.fields.append((node, namespace, "_expr", expression))
                pending.append((expression, depth + 1))
                continue
            arithmetic = any(kind is cls for cls in (Add, Mul, Pow))
            if not any(kind is cls for cls in atoms) and not arithmetic:
                raise ValueError("ideal operations require exact rational polynomial expressions")
            for name in ("_value", "_struct_args", "_surface_args"):
                value = self.retain(node, Basic, name)
                if name != "_value" and value is not _IDEAL_ABSENT and value is not None:
                    if type(value) is not tuple:
                        raise ValueError("ideal source requires exact original argument tuples")
                    if len(value) > 65_536 - count - len(pending):
                        raise ValueError("ideal quotient source exceeds its bounded expression envelope")
                    pending.extend((child, depth + 1) for child in value)
            if arithmetic:
                children = self.retain(node, kind, "_args")
                if children is not _IDEAL_ABSENT:
                    if type(children) is not tuple:
                        raise ValueError("ideal source requires exact original argument tuples")
                    if len(children) > 65_536 - count - len(pending):
                        raise ValueError("ideal quotient source exceeds its bounded expression envelope")
                    pending.extend((child, depth + 1) for child in children)
            if kind is Symbol or kind is Dummy:
                assumptions = self.retain(node, Symbol, "_assumptions")
                if (type(assumptions) is not dict or len(assumptions) > 128
                        or any(type(key) is not str or len(key) > 4096
                               or type(value) is not bool for key, value in assumptions.items())):
                    raise ValueError("ideal quotient requires bounded exact symbol assumptions")
                self.dictionaries.append((assumptions, tuple(assumptions.items())))
                if kind is Dummy:
                    self.retain(node, Dummy, "_dummy_name")
                    self.retain(node, Dummy, "_dummy_number")
            if len(pending) > 65_536 - count:
                raise ValueError("ideal quotient source exceeds its bounded expression envelope")

    @staticmethod
    def slot(owner, name):
        from types import MemberDescriptorType

        descriptor = type.__dict__["__dict__"].__get__(owner).get(name)
        if (type(descriptor) is not MemberDescriptorType or descriptor.__objclass__ is not owner
                or descriptor.__name__ != name):
            raise TypeError("ideal quotient requires original source storage descriptors")
        return descriptor

    def retain(self, node, owner, name):
        slot = self.slot(owner, name)
        try:
            value = slot.__get__(node, owner)
        except AttributeError:
            value = _IDEAL_ABSENT
        self.slots.append((node, owner, name, slot, value))
        return value

    def inputs(self):
        from ..core import Add

        # Copy only the bounded containers, preserving each original typed
        # scalar object for source admission and result identity restoration.
        # Poly has an instance dictionary: invoking its public as_expr could
        # execute an override unrelated to the retained exact source. Unwrap
        # the captured _expr directly, without calling any Poly protocol.
        polynomials = {id(node): expression for node, _, _, expression in self.fields}
        values = [GroebnerBasis([polynomials.get(id(node), node) for node in expressions],
                               generators, original[2], original[3])._ideal_inputs()
                  for _, _, original, expressions, generators in self.bases]
        self.unchanged()
        for node, _, name, _, children in self.slots:
            if name not in ("_args", "_struct_args", "_surface_args") or type(children) is not tuple:
                continue
            operator = next((op for cls, op in ((Add, "add"), (Mul, "mul"), (Pow, "pow"))
                             if type(node) is cls), None)
            if operator is not None:
                _native.ideal_source_arguments_native(
                    _native_expr(node), operator, [_native_expr(child) for child in children]
                )
        self.unchanged()
        return values

    def unchanged(self):
        for basis, namespace, original, expressions, generators in self.bases:
            if (object.__getattribute__(basis, "__dict__") is not namespace
                    or any(namespace.get(name, _IDEAL_ABSENT) is not previous
                           for name, previous in zip(("exprs", "gens", "domain", "order"), original))
                    or len(original[0]) != len(expressions) or len(original[1]) != len(generators)
                    or any(a is not b for a, b in zip(original[0], expressions))
                    or any(a is not b for a, b in zip(original[1], generators))):
                raise RuntimeError("ideal quotient source mutated during exact evaluation")
        for node, namespace, name, previous in self.fields:
            if (object.__getattribute__(node, "__dict__") is not namespace
                    or namespace.get(name, _IDEAL_ABSENT) is not previous):
                raise RuntimeError("ideal quotient source mutated during exact evaluation")
        for mapping, previous in self.dictionaries:
            if len(mapping) != len(previous):
                raise RuntimeError("ideal quotient source mutated during exact evaluation")
            current = tuple(mapping.items())
            if any(a is not c or b is not d for (a, b), (c, d) in zip(current, previous)):
                raise RuntimeError("ideal quotient source mutated during exact evaluation")
        for node, owner, name, slot, previous in self.slots:
            if self.slot(owner, name) is not slot:
                raise RuntimeError("ideal quotient source storage changed during exact evaluation")
            try:
                current = slot.__get__(node, owner)
            except AttributeError:
                current = _IDEAL_ABSENT
            if current is previous:
                continue
            # Lifting may refresh the native handle of an interned Symbol;
            # equality here compares the complete immutable typed atom.
            if (type(node) is Symbol and owner is Basic and name == "_value"
                    and type(current) is _native.Expr and type(previous) is _native.Expr
                    and current.is_symbol and previous.is_symbol and current == previous):
                continue
            raise RuntimeError("ideal quotient source mutated during exact evaluation")


def _quotient_operator_data(basis: GroebnerBasis, element: Any) -> tuple:
    """Admit and retain the complete request around one checked operator call."""
    element_basis = GroebnerBasis([element], (), "QQ", "lex")
    snapshot = _IdealQuotientSnapshot(basis, element_basis)
    source_data, element_data = snapshot.inputs()
    sources, gens, equations, bindings = source_data
    element_sources, _, element_equations, _ = element_data
    order = snapshot.bases[0][2][3]
    monomials, rows = _native.quotient_multiplication_native(
        equations, element_equations[0], bindings, order,
    )
    snapshot.unchanged()
    return snapshot, (*sources, *element_sources, *gens), monomials, rows


def _ideal_polynomial_source(expression: Any) -> Any:
    """Keep exact held polynomial sources intact until native admission."""
    from ..core import Add, Dummy, Half, NegativeOne, One, Zero

    expression = _rational_ideal_source(expression)
    if type(expression) is int:
        if expression.bit_length() > 4096:
            raise ValueError("ideal coefficient exceeds 4096 bits")
        expression = Integer(expression)
    pending = [(expression, 0)]
    visited = 0
    atoms = (Integer, Rational, Zero, One, NegativeOne, Half, Symbol, Dummy)
    while pending:
        node, depth = pending.pop()
        visited += 1
        if visited > 4096 or depth > 64:
            raise ValueError("ideal source exceeds its bounded expression envelope")
        kind = type(node)
        if kind not in atoms and kind not in (Add, Mul, Pow):
            raise ValueError("ideal operations require exact rational polynomial expressions")
        # Basic lowering borrows the existing native value. Inspect its whole
        # tree without exporting coefficients or lifting native children.
        try:
            native_node = _native_expr(node)
        except NotImplementedError as error:
            # An exact held container can retain an opaque custom child.
            # Preserve this polynomial boundary's source-admission error.
            raise ValueError("ideal operations require exact rational polynomial expressions") from error
        _native.ideal_source_preflight_native(native_node)
        if kind in atoms:
            continue
        # Native-owned subtrees contain no Python subclasses and were already
        # checked above. Only held forms retain additional original surface
        # children that need class admission. Calling `.args` here would deep
        # clone native subtrees, including repeated large symbol names, before
        # the native operation's cumulative allocation meter can bound them.
        children = getattr(node, "_args", None)
        if children is None:
            children = getattr(node, "_struct_args", None)
        if children is None:
            continue
        if type(children) is not tuple:
            raise ValueError("ideal source requires exact original argument tuples")
        if len(children) > 4096 - visited - len(pending):
            raise ValueError("ideal source exceeds its bounded expression envelope")
        pending.extend((child, depth + 1) for child in children)
    return expression


def _primitive_integer(e: Any, gens: tuple, order: str = "lex") -> Any:
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
    def leading_key(term: tuple) -> tuple:
        powers = term[0]
        if order == "grlex":
            return (sum(powers), powers)
        if order == "grevlex":
            return (sum(powers), tuple(-power for power in reversed(powers)))
        return powers

    if max(terms, key=leading_key)[1] < 0:
        scale = -scale
    return expand(e * scale)


def groebner(F: Sequence[Any], *gens: Any, order: str = "lex", **args: Any) -> Any:
    """Checked reduced Groebner basis over ZZ/QQ or a bounded prime field.

    Supports lex, grlex and grevlex. Prime fields use a separate checked
    Buchberger engine with original-source admission, both ideal inclusions
    and complete S-pair verification. Unsupported coefficient domains and
    controls raise before mathematical work. method=None selects Buchberger.
    """
    from . import _finitefield_groebner
    if _finitefield_groebner.requested(F, args):
        return _finitefield_groebner.construct(F, gens, order, args)
    for name in args:
        if name not in ("domain", "modulus", "polys", "method"):
            raise NotImplementedError("checked Groebner construction does not support option %r" % name)
    if "polys" in args and args["polys"] is not False:
        raise NotImplementedError("checked Groebner construction supports only polys=False")
    method = args.get("method")
    if (method is not None
            and (type(method) is not str or method != "buchberger")):
        raise NotImplementedError("checked Groebner construction supports only the Buchberger method")
    requested_domain = args.get("domain")
    if (requested_domain is not None and
            (type(requested_domain) is not str or requested_domain not in ("ZZ", "QQ"))):
        raise NotImplementedError("native Groebner bases currently support ZZ and QQ")
    sources = []
    for expression in F:
        if len(sources) == 64:
            raise ValueError("checked Groebner construction supports at most 64 polynomials")
        source = _rational_ideal_source(expression)
        sources.append(source)
    F = sources
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
        def generator_key(symbol: Any) -> tuple:
            binding = _symbol_binding(symbol)
            return symbol.name, binding.name, binding.identity_hex

        var_list = sorted(symbols, key=generator_key)

    eq_natives = [_native_expr(e) for e in F]
    gens_t = tuple(_require_symbol(g) for g in var_list)
    if len(gens_t) > 8 or len(set(gens_t)) != len(gens_t):
        raise ValueError("checked Groebner construction requires at most eight distinct generators")
    var_bindings = [_symbol_binding(g) for g in gens_t]
    if order not in ("lex", "grlex", "grevlex"):
        raise ValueError("unknown monomial order %r" % order)
    # Admit the original typed expressions before Poly's domain inference can
    # expand them or erase held zero/undefined children. The native result is
    # published only after complete independent ideal/basis verification.
    raw = _native.groebner_basis_native(eq_natives, var_bindings, order)
    inferred = "ZZ" if all(Poly(e, *gens_t)._infer_domain() == "ZZ" for e in F) else "QQ"
    domain = requested_domain or inferred
    if domain == "ZZ" and inferred != "ZZ":
        raise ValueError("noninteger coefficients cannot be represented in ZZ")
    exprs = [_lift_native_result(r, *F, *gens_t) for r in raw]
    if domain == "ZZ":
        exprs = [_primitive_integer(e, gens_t, order) for e in exprs]

    def lead_key(e: Any) -> tuple:
        return tuple(Poly(e, *gens_t).terms()[0][0])

    if order == "lex":
        exprs.sort(key=lead_key, reverse=True)
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
        scale_raw, raw = _native.poly_factor_multivariate_native(
            _native_expr(expr), [_symbol_binding(_require_symbol(g)) for g in gens]
        )
    except Exception:
        return None
    factors = [(_lift_native_result(f, expr, *gens), m) for f, m in raw]

    def key(fm: Any) -> tuple:
        rep = _dmp_rep(fm[0], list(gens))
        return (len(rep), fm[1], rep)

    try:
        factors.sort(key=key)
    except TypeError:
        pass
    return _lift_native_result(scale_raw, expr, *gens), factors


def _multivariate_gens(p: Any, gens: tuple) -> list | None:
    syms = _sort_gens(getattr(_wrap(_native_expr(p)), "free_symbols", set()))
    if gens:
        extra = [s for s in syms if s not in gens]
        out = list(gens) + extra
    else:
        out = syms
    return out if len(out) > 1 else None


def _prime_field_factor_list(
    p: Any, gens: tuple, modulus: Any, *, for_factor: bool = False
) -> Tuple[Any, List[Tuple[Any, int]]]:
    """Checked univariate modular factorization with exact exterior content.

    The native checker replays the original source and every proposed integer
    component. The numerator/denominator transformation is a compatibility
    choice, not a trusted proof of the modular result.

    This bounded lane refuses nonzero exterior numerators divisible by the
    characteristic, including literal constants, and requires the exterior
    denominator to be the original canonical coefficient denominator. It does
    not provide general Poly objects over finite fields.
    """
    if isinstance(modulus, bool) or not isinstance(modulus, (int, Integer)):
        raise ValueError("modulus must be an exact prime integer")
    prime = int(modulus)
    if not 2 <= prime <= 65521:
        raise ValueError("modular factorization requires a prime in 2..65521")
    if isinstance(p, Poly):
        raise NotImplementedError("modular factorization of Poly objects is not supported")
    raw = _native_expr(p)
    _native.prime_field_source_preflight_native(raw)
    expr = _wrap(raw)
    if gens:
        if len(gens) != 1:
            raise NotImplementedError("modular factorization currently supports one generator")
        variable = _require_symbol(gens[0])
    else:
        symbols = _sort_gens(expr.free_symbols)
        if len(symbols) > 1:
            raise NotImplementedError("modular factorization currently supports one generator")
        variable = symbols[0] if symbols else Symbol("_modular_constant")
    binding = _symbol_binding(variable)
    # This full native admission precedes shell normalization, including held
    # zero products and powers whose invalid children could otherwise vanish.
    _native.prime_field_source_admit_native(raw, binding, prime)
    normalized = together(expr)
    if for_factor:
        # Pinned factor extracts content from together(expr) itself, whereas
        # factor_list first takes its numerator and denominator. The latter
        # can redistribute integer content into an Add: over F_3, factor
        # preserves 2*(x+1), while factor_list(2*x+2) has modular unit -1.
        numerator, content = normalized, Integer(1)
    else:
        numerator, denominator = normalized.as_numer_denom()
        if not isinstance(denominator, Integer) or denominator == 0:
            raise ValueError("modular factorization requires a polynomial with exact rational coefficients")
        content = Rational(1, int(denominator))
    components = []
    for piece in Mul.make_args(numerator):
        if isinstance(piece, Rational):
            content *= piece
            continue
        base, exponent = piece.as_base_exp()
        if not isinstance(exponent, Integer) or not 1 <= int(exponent) <= 64:
            raise ValueError("modular component powers must be integers in 1..64")
        components.append((_native_expr(base), int(exponent)))
    scale_raw, factors_raw = _native.prime_field_factor_source_native(
        raw, binding, prime, _native_expr(content), components
    )
    return (
        _lift_native_result(scale_raw, expr, variable),
        [(_lift_native_result(value, expr, variable), multiplicity)
         for value, multiplicity in factors_raw],
    )


def factor_list(p: Any, *gens: Any, modulus: Any = None) -> Tuple[Any, List[Tuple[Any, int]]]:
    """Compute polynomial factor list returning (scale, [(factor, multiplicity), ...])."""
    if _finitefield.is_finite(p):
        return _finitefield.factor_entry(p, gens, modulus)
    if modulus is not None:
        return _prime_field_factor_list(p, gens, modulus)
    from . import _opaque
    if _opaque.applies(p):
        return _opaque.factor(p, gens, listing=True)
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


def factor(p: Any, *gens: Any, modulus: Any = None) -> Any:
    if _finitefield.is_finite(p):
        return _finitefield.factor_entry(p, gens, modulus, expression=True)
    if modulus is not None:
        from ..core import _keep_coeff

        scale, factors = _prime_field_factor_list(p, gens, modulus, for_factor=True)
        return _keep_coeff(scale, Mul(*[value if multiplicity == 1 else value**multiplicity
                                        for value, multiplicity in factors]))
    from . import _opaque
    if _opaque.applies(p):
        return _opaque.factor(p, gens)
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
    from . import _opaque
    if _opaque.applies(f):
        return _opaque.cancel(f, gens)
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


def _gt_parse(t: Any) -> tuple:
    """Term -> (rational coefficient, {base: exp} numerator, {base: exp}
    denominator), upstream ``Factors`` style: ``exp(u)`` is ``E**u`` and a
    negative (or minus-signed) exponent goes to the denominator."""
    from fractions import Fraction as _F
    from ..core import E as _E

    coeff = _F(1)
    num: dict = {}
    den: dict = {}
    for f in Mul.make_args(t):
        if isinstance(f, Rational):
            coeff *= _F(int(f.p), int(f.q))
            continue
        if isinstance(f, Pow):
            base, e = f.args
        elif type(f).__name__ == "exp":
            base, e = _E, f.args[0]
        else:
            base, e = f, Integer(1)
        neg = (isinstance(e, Rational) and e < 0) or (
            not isinstance(e, Rational) and Mul.make_args(e)[0] is not None
            and isinstance(Mul.make_args(e)[0], Rational) and Mul.make_args(e)[0] < 0
        )
        target = den if neg else num
        e = -e if neg else e
        target[base] = target.get(base, Integer(0)) + e
    return coeff, num, den


def _gt_merge(d1: dict, d2: dict, pick: Any) -> dict:
    """Per-base min (gcd) or max (lcm) of exponents; bases whose exponents
    differ by a non-number are kept apart by the caller's ``pick``."""
    from ..core import expand as _expand

    out = {}
    for base in set(d1) | set(d2):
        e1, e2 = d1.get(base), d2.get(base)
        if e1 is None or e2 is None:
            if pick == "max":
                out[base] = e1 if e2 is None else e2
            continue
        difference = e1 - e2
        diff = difference if isinstance(difference, Rational) else _expand(difference)
        if isinstance(diff, Rational):
            out[base] = (e1 if diff >= 0 else e2) if pick == "max" else (e2 if diff >= 0 else e1)
        elif pick == "max":
            out[base] = _expand(e1 + e2)
    return out


def _gt_sub(d1: dict, d2: dict) -> dict:
    from ..core import expand as _expand

    out = {}
    for base, e in d1.items():
        difference = e - d2.get(base, 0)
        r = difference if isinstance(difference, Rational) else _expand(difference)
        if r != 0:
            out[base] = r
    return out


def _gt_expr(d: dict) -> list:
    return [b**e for b, e in d.items()]


def _gcd_terms_fraction(terms: list) -> Any:
    """Upstream ``gcd_terms(terms, fraction=True)``: common content out,
    numerators over the structural lcm of the denominators, unexpanded."""
    from fractions import Fraction as _F
    from math import gcd as _g, lcm as _l
    from ..core import Add as _Add, _keep_coeff

    parsed = [_gt_parse(t) for t in terms if t != 0]
    if not parsed:
        return Integer(0)

    def fgcd(a: _F, b: _F) -> _F:
        return _F(_g(a.numerator, b.numerator), _l(a.denominator, b.denominator))

    cc, cn, cd = parsed[0]
    cc = abs(cc)
    for c, n, d in parsed[1:]:
        cc = fgcd(cc, abs(c))
        cn = _gt_merge(cn, n, "min")
        cd = _gt_merge(cd, d, "min")
    reduced = [(c / cc, _gt_sub(n, cn), _gt_sub(d, cd)) for c, n, d in parsed]
    den = reduced[0][2]
    for _, _, d in reduced[1:]:
        den = _gt_merge(den, d, "max")
    numers = []
    for c, n, d in reduced:
        cof = _gt_sub(den, d)
        numers.append(Mul(Rational(c.numerator, c.denominator), *_gt_expr(n), *_gt_expr(cof)))
    numer = _Add(*numers)
    if isinstance(numer, _Add):
        # Add.primitive: positive rational content of the numerator.
        content = _F(0)
        for t in numer.args:
            k = Mul.make_args(t)[0]
            v = _F(int(k.p), int(k.q)) if isinstance(k, Rational) else _F(1)
            content = fgcd(content, abs(v)) if content else abs(v)
        if content and content != 1:
            numer = _Add(*[t / Rational(content.numerator, content.denominator) for t in numer.args])
            cc *= content
    body = Mul(*_gt_expr(cn), numer, *[b ** (-e) for b, e in cd.items()], *[b ** (-e) for b, e in den.items()])
    return _keep_coeff(Rational(cc.numerator, cc.denominator), body)


def together(expr: Any, deep: bool = False, fraction: bool = True) -> Any:
    """Combine an expression over a common denominator (upstream
    ``together``): denominators are combined structurally (their lcm as
    products of powers, nothing is expanded or cancelled) and function
    arguments are left alone unless ``deep``.

    Examples
    ========
    >>> from sympy import together, symbols
    >>> x, y = symbols('x y')
    >>> together(1/x + 1/y)
    (x + y)/(x*y)
    >>> together(1/x + 1)
    (x + 1)/x
    """
    from . import _opaque
    if _opaque.applies(expr):
        return _opaque.together(expr, deep=deep, fraction=fraction)
    from ..core import Add, Function as _Function

    is_poly = isinstance(expr, Poly)
    wrapped = expr.as_expr() if is_poly else _wrap(_native_expr(expr))

    def _together(e: Any) -> Any:
        args = getattr(e, "args", ())
        if not args:
            return e
        if isinstance(e, _Function) and not deep:
            return e
        if isinstance(e, Add):
            return _gcd_terms_fraction([_together(t) for t in e.args])
        if isinstance(e, Pow):
            return Pow(_together(e.args[0]), e.args[1])
        try:
            return e.func(*[_together(a) for a in args])
        except Exception:
            return e

    res = _together(wrapped)
    return Poly(res, *expr._gens) if is_poly else res


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
                # Upstream apart factors each numerator: -(x + 2)/3.
                n = r.as_expr()
                if n.free_symbols:
                    n = factor(n)
                res.append(n / d)
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
    modular = _finitefield.binary_entry(f, g, gens, kwargs, "divrem")
    if modular is not None:
        return modular
    is_poly = isinstance(f, Poly) or isinstance(g, Poly)
    p_f = f if isinstance(f, Poly) else Poly(f, *gens, **kwargs)
    p_g = g if isinstance(g, Poly) else Poly(g, *p_f.gens)
    q, r = p_f.div(p_g)
    if is_poly or kwargs.get("polys", False):
        return (q, r)
    return (q.as_expr(), r.as_expr())


def reduced(f: Any, G: Any, *gens: Any, **kwargs: Any) -> Tuple[list, Any]:
    """Reduce a prime-field polynomial by a complete ordered divisor list.

    The checked bounded lane supports lex, grlex and grevlex with at most
    eight generators and 32 divisors. It independently verifies the complete
    quotient/remainder identity and that no divisor leading monomial divides
    any remainder term. Other coefficient domains currently refuse.
    """
    return _finitefield.reduced_entry(f, G, gens, kwargs)


def rem(f: Any, g: Any, *gens: Any, **kwargs: Any) -> Any:
    """Polynomial remainder of f divided by g."""
    return div(f, g, *gens, **kwargs)[1]


def quo(f: Any, g: Any, *gens: Any, **kwargs: Any) -> Any:
    """Polynomial quotient of f divided by g."""
    return div(f, g, *gens, **kwargs)[0]


def sturm(f: Any, *gens: Any, **kwargs: Any) -> List[Poly]:
    """Compute the Sturm sequence of a polynomial."""
    p0 = f if isinstance(f, Poly) else Poly(f, *gens, **kwargs)
    if _finitefield.is_finite(p0):
        return _finitefield.unsupported(p0, "Sturm sequences")
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
    modular = _finitefield.binary_entry(f, g, gens, kwargs, "compose")
    if modular is not None:
        return modular
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
    if _finitefield.is_finite(f) or _finitefield.requested(None, kwargs):
        polynomial = f if _finitefield.is_finite(f) else Poly(f, *gens, **kwargs)
        return _finitefield.unsupported(polynomial, "decomposition")
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
    "reduced",
    "rem",
    "resultant",
    "roots",
    "sqf",
    "sqf_list",
    "sqf_part",
    "sturm",
    "together",
    "PurePoly",
    "terms_gcd",
    "trailing_coeff",
]
