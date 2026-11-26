"""Indexed roots of univariate rational polynomials (upstream ``CRootOf``).

Real-root identity, order, multiplicity and decimal refinement use complete
native exact-rational isolation. A separate reference checker verifies the
original polynomial's complete real-root count and every supplied interval.
The factor association checks the full monic factor product and exact interval
membership; it does not infer identity from numerical proximity.

Real ``evalf`` supports up to 512 correctly rounded significant digits using
certified endpoint agreement. The existing shell decimal-Float representation
preserves those rendered digits; subsequent Float arithmetic still uses the
shell's binary64 backend. Nonreal root ordering/evaluation retains the bounded
binary64 lane, with an explicit refusal when its real-root classification
conflicts with the exact count. This is not certified complex-root isolation.

The defining generator stays bound inside symbolic arithmetic, substitutions
and derivatives. Rational expressions in one exact real root use native
algebraic reduction and rational enclosures for numerical evaluation. Original
denominator exclusions remain in force. Multiple distinct primitive roots and
transcendental root-containing composites are outside that bounded evaluator.
"""

from __future__ import annotations

import math
from typing import Any

from ..core import Basic, Function, Integer, NegativeOne, One, Rational, Symbol, Zero, _wrap, sympify


def _poly_of(f: Any, x: Any = None) -> tuple:
    from .polytools import Poly

    if isinstance(f, Poly):
        p = f
    else:
        f = sympify(f)
        if x is None:
            syms = sorted(f.free_symbols, key=lambda s: s.name)
            if len(syms) != 1:
                raise ValueError("CRootOf needs a univariate polynomial")
            x = syms[0]
        p = Poly(f, x)
    if len(p.gens) != 1 or p._parametric:
        raise NotImplementedError("CRootOf supports rational-coefficient univariate polynomials")
    # Poly stores the original typed tree. Validate it before any degree,
    # coefficient extraction or factorization can exponentiate a held power.
    _native_coefficients(p)
    return p, p.gen


def _native_coefficients(p: Any) -> list:
    """Ascending typed literal coefficients; no printed expression transport."""
    from ..core import _native, _native_expr, _symbol_binding

    return _native.real_root_coefficients_native(
        _native_expr(p.as_expr()), _symbol_binding(p.gen)
    )


def _real_intervals(p: Any) -> list:
    from ..core import _native

    return _native.real_roots_isolate_native(_native_coefficients(p))


def _real_count(p: Any) -> int:
    return sum(multiplicity for _, _, multiplicity in _real_intervals(p))


def _factor_data(p: Any) -> tuple:
    from ..core import _native

    coefficients = _native_coefficients(p)
    _, factors = p.factor_list()
    # Native checking binds the entire factor list (including multiplicities)
    # to the exact original polynomial before any root identity is returned.
    real = _native.indexed_real_roots_native(
        coefficients,
        [_native_coefficients(fac) for fac, _ in factors],
        [int(mult) for _, mult in factors],
    )
    return factors, real


def _order(zs: list) -> list:
    """Legacy numerical order, used only in the explicitly nonreal lane."""
    scale = max([1.0] + [abs(z) for z in zs])
    real = sorted(z.real for z in zs if abs(z.imag) <= 1e-10 * scale)
    cplx = sorted(
        (z for z in zs if abs(z.imag) > 1e-10 * scale),
        key=lambda z: (round(z.real, 9), z.imag),
    )
    return [complex(r, 0) for r in real] + cplx


def _approximate_factor_roots(p: Any, real_count: int) -> list:
    from .. import _numeric_poly_roots

    coefficients = [complex(float(c)) for c in p.all_coeffs()]
    if not all(math.isfinite(c.real) and math.isfinite(c.imag) for c in coefficients):
        raise NotImplementedError("nonreal CRootOf coefficients exceed the binary64 lane")
    roots = _order(_numeric_poly_roots(coefficients))
    if len(roots) != p.degree() or not all(
        math.isfinite(z.real) and math.isfinite(z.imag) for z in roots
    ):
        raise NotImplementedError("nonreal CRootOf numerical root set is incomplete")
    if sum(z.imag == 0 for z in roots) != real_count:
        raise NotImplementedError(
            "nonreal CRootOf numerical classification conflicts with the exact real-root count"
        )
    return roots


def _indexed_roots(p: Any) -> list:
    """[(numeric value or None, factor Poly, local index)] in global order.

    None identifies an exactly isolated real root. It deliberately avoids
    computing a floating approximation merely to choose that root's identity.
    """
    factors, real = _factor_data(p)
    real_entries = []
    real_counts = [0] * len(factors)
    for factor, local, multiplicity in real:
        real_entries.extend([(None, factors[factor][0], local)] * multiplicity)
        real_counts[factor] += 1
    nonreal = []
    for factor, ((fac, mult), count) in enumerate(zip(factors, real_counts)):
        if count == fac.degree():
            continue
        roots = _approximate_factor_roots(fac, count)
        for local in range(count, len(roots)):
            nonreal.extend([(roots[local], fac, local)] * mult)
    nonreal.sort(key=lambda entry: (round(entry[0].real, 9), entry[0].imag))
    return real_entries + nonreal


def _numeric_roots(p: Any) -> list:
    """Approximate values, retaining exact identification of real roots."""
    from ..core import _native

    result = []
    for value, factor, index in _indexed_roots(p):
        if value is None:
            value = complex(float(_native.real_root_decimal_native(_native_coefficients(factor), index, 17)), 0)
        result.append(value)
    return result


def _radical_roots(fac: Any) -> list | None:
    """Numerical ordering of explicit nonreal quadratic/binomial roots."""
    from .polytools import _binomial_roots, _quadratic_roots

    cs = fac.all_coeffs()
    if len(cs) == 3:
        roots = _quadratic_roots(*cs)
    elif all(c == 0 for c in cs[1:-1]):
        from ..core import expand

        roots = [expand(r) for r in _binomial_roots(cs[0], cs[-1], len(cs) - 1)]
    else:
        return None

    def key(root: Any) -> tuple:
        value = complex(root)
        if abs(value.imag) < 1e-12:
            return (0, value.real, 0.0)
        return (1, round(value.real, 9), value.imag)

    return sorted(roots, key=key)


def _real_radical(fac: Any, index: int) -> Any:
    """Real quadratic/binomial formulas ordered by exact coefficient signs."""
    from fractions import Fraction
    from ..core import _exact_ratio

    # Keep construction on the bounded exact converter used to admit the
    # source and its factors. The general Poly coefficient API is unmetered.
    cs = [_wrap(value) for value in reversed(_native_coefficients(fac))]
    exact = [Fraction(*_exact_ratio(c)) for c in cs]
    if len(cs) == 3:
        discriminant = exact[1]**2 - 4 * exact[0] * exact[2]
        if discriminant <= 0:
            return None

        def rational(value):
            return Rational(value.numerator, value.denominator)

        # These coefficients are already exact rationals. Keep their
        # arithmetic in that domain instead of invoking generic cancellation
        # and expansion merely to normalize a rational radicand.
        if exact[2] == 0:
            return rational(sorted((Fraction(0), -exact[1] / exact[0]))[index])
        if exact[1] == 0:
            magnitude = rational(-exact[2] / exact[0]) ** Rational(1, 2)
            return -magnitude if index == 0 else magnitude
        denominator = 2 * exact[0]
        center = rational(-exact[1] / denominator)
        radius = rational(discriminant) ** Rational(1, 2) / rational(denominator)
        roots = [center - radius, center + radius]
        if exact[0] < 0:
            roots.reverse()
        return roots[index]
    if len(cs) > 2 and all(c == 0 for c in exact[1:-1]):
        degree = len(cs) - 1
        alpha = -exact[-1] / exact[0]
        magnitude = Rational(abs(alpha.numerator), alpha.denominator) ** Rational(1, degree)
        if degree % 2:
            return -magnitude if alpha < 0 else magnitude
        if alpha > 0:
            return -magnitude if index == 0 else magnitude
    return None


def _make_root(cls: type, fac: Any, gen: Any, local: int, radicals: bool, real: bool) -> Any:
    coefficients = [_wrap(value) for value in reversed(_native_coefficients(fac))]
    if len(coefficients) == 2:
        return -coefficients[1] / coefficients[0]
    if radicals:
        if real:
            explicit = _real_radical(fac, local)
            if explicit is not None:
                return explicit
        else:
            explicit = _radical_roots(fac)
            if explicit is not None:
                return explicit[local]
    expr = fac.as_expr()
    # The generator is bound. The pinned profile retains its printed name
    # but discards assumptions; input validation above used its full identity.
    bound_gen = Symbol(gen.name)
    if fac.gen != bound_gen:
        expr = expr.subs(fac.gen, bound_gen)
    return Function.__new__(cls, expr, Integer(local), evaluate=False)


class CRootOf(Function):
    """The indexed root of a univariate polynomial; real indices are exact."""

    def __new__(cls, f: Any, x: Any = None, index: Any = None, radicals: bool = False, expand: bool = True):
        if index is None and x is not None and not isinstance(x, Symbol):
            x, index = None, x
        p, gen = _poly_of(f, x)
        degree = len(_native_coefficients(p)) - 1
        index = sympify(index)
        if type(index) not in (Integer, Zero, One, NegativeOne):
            raise ValueError("expected an integer root index")
        k = int(index)
        if not -degree <= k < degree:
            raise IndexError("root index out of [%d, %d] range, got %d" % (-degree, degree - 1, k))
        if k < 0:
            k += degree
        # Do not enter the approximate complex lane for a real index, even
        # when the same polynomial also has nonreal roots.
        factors, real = _factor_data(p)
        offset = 0
        for factor, local, multiplicity in real:
            if offset <= k < offset + multiplicity:
                return _make_root(cls, factors[factor][0], gen, local, radicals, True)
            offset += multiplicity
        value, fac, local = _indexed_roots(p)[k]
        return _make_root(cls, fac, gen, local, radicals, value is None)

    @classmethod
    def eval(cls, *args: Any) -> Any:
        return None

    @property
    def poly(self) -> Any:
        from .polytools import PurePoly

        return PurePoly(self.args[0])

    @property
    def index(self) -> int:
        return int(self.args[1])

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        return self.index == other.index and self.poly == other.poly

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self) -> int:
        from ..core import _exact_ratio, _wrap

        # The defining factor already belongs to the bounded exact univariate
        # lane. Its full coefficient vector (including zero coefficients)
        # agrees with PurePoly equality while ignoring the bound generator's
        # name. Hash those literal rational pairs directly: generic terms()
        # expansion is unnecessary and would re-enter the symbolic runtime.
        coefficients = tuple(
            _exact_ratio(_wrap(coefficient))
            for coefficient in _native_coefficients(self.poly)
        )
        return hash((type(self), coefficients, self.index))

    @property
    def free_symbols(self) -> set:
        return set()

    def subs(self, *args: Any, **kwargs: Any) -> Any:
        # Basic normalizes sequential/simultaneous and whole-object
        # replacements; its exact built-in root leaf keeps the generator
        # bound when the replacement does not match this root itself.
        return Basic.subs(self, *args, **kwargs)

    def _root_value(self) -> complex:
        from ..core import _native

        p, _ = _poly_of(self.args[0])
        real_count = _real_count(p)
        if self.index < real_count:
            text = _native.real_root_decimal_native(_native_coefficients(p), self.index, 17)
            return complex(float(text), 0)
        return _approximate_factor_roots(p, real_count)[self.index]

    @property
    def is_real(self) -> bool:
        p, _ = _poly_of(self.args[0])
        return self.index < _real_count(p)

    def evalf(self, n: int = 15, **kwargs: Any) -> Any:
        budget = kwargs.get("budget")
        if budget is not None:
            return _core._budgeted_evalf(self, n, budget)
        from ..core import I, _ExactDecimalFloat, _format_decimal, _native
        from .. import _float_digits

        if type(n) is not int or not 1 <= n <= 512:
            raise ValueError("CRootOf precision must be between 1 and 512 decimal digits")
        p, _ = _poly_of(self.args[0])
        if self.index < _real_count(p):
            text = _native.real_root_decimal_native(_native_coefficients(p), self.index, n)
            return _ExactDecimalFloat(_format_decimal(text, n), n)
        if n > 15:
            raise NotImplementedError(
                "precision-honest nonreal CRootOf: binary64 roots support at most 15 digits; requested %d" % n
            )
        value = self._root_value()
        real = _float_digits(value.real, n) if abs(value.real) > 1e-14 else Integer(0)
        return real + _float_digits(value.imag, n) * I

    def __float__(self) -> float:
        if not self.is_real:
            raise TypeError("cannot convert a non-real root to float")
        return self._root_value().real

    def __complex__(self) -> complex:
        return self._root_value()


from .. import core as _core

_core._bound_algebraic_root_type = CRootOf


rootof = CRootOf
ComplexRootOf = CRootOf
RootOf = CRootOf


def _algebraic_expression_value(expression: Any, digits: int):
    """Evaluate bounded rational arithmetic in genuine indexed real roots.

    Root identity comes from the registered built-in class, never its printed
    name. Held native constructors preserve every original inverse and power
    until the native source/domain checker has admitted them. Several distinct
    roots use a checked common representation with exact embedding selection.
    """
    from ..core import Add, Half, Mul, Pow, Dummy, _native, _native_expr, _symbol_binding

    if type(digits) is not int or not 1 <= digits <= 512:
        raise ValueError("algebraic expression precision must be between 1 and 512 decimal digits")
    root_type = _core._bound_algebraic_root_type
    roots = []
    pending = [(expression, 0)]
    visited = 0
    atoms = (Integer, Rational, Zero, One, NegativeOne, Half)
    while pending:
        node, depth = pending.pop()
        visited += 1
        if visited > 2048 or depth > 64:
            raise ValueError("algebraic expression exceeds its bounded source envelope")
        kind = type(node)
        if kind is root_type:
            if node not in roots:
                if len(roots) == 8:
                    raise ValueError("algebraic evaluation supports at most eight distinct roots")
                roots.append(node)
            continue
        if kind in atoms:
            continue
        if kind not in (Add, Mul, Pow):
            raise NotImplementedError(
                "exact algebraic evaluation requires rational arithmetic on real roots"
            )
        children = node.args
        if len(children) > 256 or len(children) > 2048 - visited - len(pending):
            raise ValueError("algebraic expression exceeds its bounded source envelope")
        pending.extend((child, depth + 1) for child in children)
    if not roots:
        raise ValueError("algebraic expression contains no genuine indexed root")

    parameters = [Dummy("_fsym_algebraic_value") for _ in roots]
    native_parameters = [_native_expr(parameter) for parameter in parameters]

    def lower(node):
        kind = type(node)
        if kind is root_type:
            return native_parameters[roots.index(node)]
        if kind in atoms:
            return _native_expr(node)
        args = [lower(child) for child in node.args]
        if kind is Add:
            return _native.Add(*args, evaluate=False).as_expr()
        if kind is Mul:
            return _native.Mul(*args, evaluate=False).as_expr()
        return _native.Pow(*args, evaluate=False).as_expr()

    source = lower(expression)
    if len(roots) == 1:
        result = _native.evaluate_real_algebraic_native(
            source,
            _symbol_binding(parameters[0]),
            _native_coefficients(roots[0].poly),
            roots[0].index,
            digits,
        )
    else:
        result = _native.evaluate_real_algebraic_expression_native(
            source,
            [_symbol_binding(parameter) for parameter in parameters],
            [_native_coefficients(root.poly) for root in roots],
            [root.index for root in roots],
            digits,
        )
    if result is None:
        raise NotImplementedError(
            "exact algebraic evaluation requires supported bounded rational arithmetic on real roots"
        )
    return result


def _algebraic_comparison(lhs: Any, rhs: Any) -> int:
    """Exact sign of a supported algebraic difference.

    Indexed roots may have different defining polynomials. Rational
    expressions retain both original trees, including domain obligations,
    until the native evaluator has checked them in a common representation. The
    caller keeps unsupported or exhausted decisions unknown.
    """
    from ..core import Add, Half, Mul, _exact_ratio, _native, _native_expr, _wrap

    root_type = _core._bound_algebraic_root_type
    atoms = (Integer, Rational, Zero, One, NegativeOne, Half)

    def descriptor(value):
        if type(value) is root_type:
            if not value.is_real:
                raise TypeError("Invalid comparison of a non-real algebraic root")
            return _native_coefficients(value.poly), value.index
        if type(value) in atoms:
            # A rational is the sole real root of t - value.
            return [_native_expr(-value), _native_expr(Integer(1))], 0
        return None

    left, right = descriptor(lhs), descriptor(rhs)
    if left is not None and right is not None:
        return _native.compare_real_roots_native(left[0], left[1], right[0], right[1])

    difference = Add(lhs, Mul(-1, rhs, evaluate=False), evaluate=False)
    value = _algebraic_expression_value(difference, 1)
    lower = _exact_ratio(_wrap(value.lower))
    upper = _exact_ratio(_wrap(value.upper))
    if lower is None or upper is None:
        raise ValueError("algebraic comparison did not receive exact rational bounds")
    if lower[0] > 0:
        return 1
    if upper[0] < 0:
        return -1
    if lower[0] == 0 and upper[0] == 0:
        return 0
    raise ValueError("the verified algebraic enclosure does not establish a sign")


def all_roots(f: Any, multiple: bool = True, radicals: bool = True, extension: bool = False) -> list:
    # Rational coefficients are already in the admitted exact domain whether
    # extension is requested or not; algebraic coefficient extensions refuse.
    p, gen = _poly_of(f)
    result = []
    previous = None
    for value, fac, local in _indexed_roots(p):
        key = (fac.as_expr(), local)
        if not multiple and key == previous:
            root, multiplicity = result[-1]
            result[-1] = (root, multiplicity + 1)
            continue
        root = _make_root(CRootOf, fac, gen, local, radicals, value is None)
        result.append(root if multiple else (root, 1))
        previous = key
    return result


def real_roots(f: Any, multiple: bool = True, radicals: bool = True, extension: bool = False) -> list:
    p, gen = _poly_of(f)
    factors, real = _factor_data(p)
    result = []
    for factor, local, multiplicity in real:
        root = _make_root(CRootOf, factors[factor][0], gen, local, radicals, True)
        if multiple:
            result.extend([root] * multiplicity)
        else:
            result.append((root, multiplicity))
    return result


# Capture the implementation owners when this eagerly imported module loads,
# before a first nroots call can install replacement polynomial protocols.
from .polytools import GroebnerBasis as _NrootsBasis, Poly as _NrootsPoly, PurePoly as _NrootsPurePoly
from ._evaluation_owners import require_original_owners as _nroots_scalar_owners
from ..core._function_capabilities import _owner_matches as _nroots_owner_matches
from ..core._function_capabilities import _owner_record as _nroots_owner_record
from ..core._function_capabilities import _implementation as _nroots_implementation

_NROOTS_MRO = type.__dict__["__mro__"]
_NROOTS_OWNERS = tuple(
    (owner, _NROOTS_MRO.__get__(owner), _nroots_owner_record(owner))
    for owner in (_NrootsPoly, _NrootsPurePoly, _NrootsBasis)
)
_NROOTS_POLY_METHODS = tuple(
    (owner, name, _nroots_implementation(type.__dict__["__dict__"].__get__(owner).get(name)))
    for owner, names in (
        (_NrootsPoly, ("as_expr", "nroots", "gen", "gens", "_domain_syms", "_parametric")),
        (_NrootsBasis, ("_ideal_inputs",)),
    ) for name in names
)
_NROOTS_NATIVE = tuple(
    (name, getattr(_core._native, name)) for name in (
        "real_root_coefficients_native", "real_roots_isolate_native",
        "real_root_decimal_native", "ideal_source_preflight_native",
        "ideal_source_arguments_native", "evalf_complex_decimal_native",
        "Mul", "Pow", "py_integer",
    )
)
_NROOTS_FLOAT = _core._ExactDecimalFloat


def _nroots_original_interfaces():
    _nroots_scalar_owners()
    _core._require_evalf_owners()
    for owner, parents, record in _NROOTS_OWNERS:
        current = _NROOTS_MRO.__get__(owner)
        if (len(current) != len(parents)
                or any(a is not b for a, b in zip(current, parents))
                or not _nroots_owner_matches(owner, record)):
            raise TypeError("high-precision nroots requires original polynomial implementations")
    for owner, name, recorded in _NROOTS_POLY_METHODS:
        current = _nroots_implementation(type.__dict__["__dict__"].__get__(owner).get(name))
        if (len(current) != len(recorded)
                or any(a is not b for a, b in zip(current, recorded))):
            raise TypeError("high-precision nroots requires original polynomial implementations")
    if _core._ExactDecimalFloat is not _NROOTS_FLOAT:
        raise TypeError("high-precision nroots requires the original decimal Float implementation")
    if any(vars(_core._native).get(name) is not function for name, function in _NROOTS_NATIVE):
        raise TypeError("high-precision nroots requires original native root interfaces")


def _high_precision_real_nroots(source, digits, maxsteps):
    """Complete all-real numerical roots using existing exact native checks.

    No factorization, binary64 candidate, proximity ordering or numerical
    imaginary-part cleanup participates in this selected lane. Each native
    operation retains its existing fixed work/growth/refinement envelope;
    this adapter does not claim a caller-owned cumulative native budget.
    """
    from .polytools import _IdealQuotientSnapshot, _rational_ideal_source
    from ..core import Dummy, _surface_binding_key

    if digits > 512:
        raise NotImplementedError("high-precision all-real nroots supports at most 512 digits")
    if maxsteps != 50:
        raise NotImplementedError(
            "high-precision nroots accepts only the default maxsteps=50; "
            "exact native isolation and refinement use their own bounded limits"
        )
    _nroots_original_interfaces()
    kind = type(source)
    domain = None
    if kind is _NrootsPoly or kind is _NrootsPurePoly:
        namespace = object.__getattribute__(source, "__dict__")
        if (type(namespace) is not dict or len(namespace) > 32
                or any(type(key) is not str or len(key) > 128 for key in namespace)):
            raise TypeError("high-precision nroots requires bounded built-in Poly storage")
        _rational_ideal_source(source)
        domain = namespace.get("_domain")
        generators = namespace.get("_gens")
        extra = namespace.get("_extra_syms")
        if type(generators) is not tuple or len(generators) != 1:
            raise ValueError("high-precision nroots requires a univariate polynomial")
        if type(extra) is not frozenset or extra:
            raise NotImplementedError("high-precision nroots requires numeric QQ/ZZ coefficients")
        if type(namespace.get("_domain_given")) is not bool:
            raise ValueError("high-precision nroots requires an exact domain declaration")
    else:
        generators = ()

    snapshot = _IdealQuotientSnapshot(_NrootsBasis([source], generators, "QQ", "lex"))
    classes = [(source, kind)]
    seen = {id(source)}
    for node, *_ in snapshot.slots:
        if id(node) not in seen:
            classes.append((node, type(node)))
            seen.add(id(node))

    def unchanged():
        # Any ambient publication charge precedes the callback-free source
        # checks, so a mutable original cannot publish stale numerical roots.
        _core._charge_nested_if_active(1)
        _nroots_original_interfaces()
        if any(type(node) is not owner for node, owner in classes):
            raise RuntimeError("high-precision nroots source class changed during evaluation")
        snapshot.unchanged()

    unchanged()
    (data,) = snapshot.inputs()
    sources, generators, equations, bindings = data
    unchanged()
    if not generators:
        symbols = sources[0].free_symbols
        if len(symbols) > 1:
            raise ValueError("high-precision nroots requires a univariate polynomial")
        generator = next(iter(symbols)) if symbols else Dummy("_fsym_nroots_constant")
        binding = _core._symbol_binding(generator)
    else:
        generator, binding = generators[0], bindings[0]
    if type(generator) is not Symbol and type(generator) is not Dummy:
        raise TypeError("high-precision nroots requires an exact Symbol or Dummy generator")
    raw_generator = _core.Basic.__dict__["_value"].__get__(generator, _core.Basic)
    if (type(raw_generator) is not _core._native.Expr or not raw_generator.is_symbol
            or _surface_binding_key(raw_generator.symbol_binding) != _surface_binding_key(binding)):
        raise ValueError("high-precision nroots generator disagrees with its typed assumptions")
    unchanged()
    coefficients = _core._native.real_root_coefficients_native(equations[0], binding)
    unchanged()
    if domain == "ZZ" and any(coefficient.exact_denominator() != 1 for coefficient in coefficients):
        raise ValueError("high-precision nroots ZZ coefficients must be exact integers")
    roots = _core._native.real_roots_isolate_native(coefficients)
    unchanged()
    degree = len(coefficients) - 1
    if sum(multiplicity for _, _, multiplicity in roots) != degree:
        raise NotImplementedError(
            "high-precision nroots requires the complete polynomial root set to be real; "
            "mixed or nonreal roots currently support at most 15 digits"
        )
    result = []
    for index, (_, _, multiplicity) in enumerate(roots):
        if degree == 1:
            # The sole root is exactly -c0/c1. Avoid absolute-interval
            # bisection work for rational roots far outside binary64's range;
            # the complete isolation above has already checked multiplicity.
            native = _core._native
            minus_one = native.py_integer(-1)
            inverse = native.Pow(coefficients[1], minus_one, evaluate=False).as_expr()
            quotient = native.Mul(minus_one, coefficients[0], inverse, evaluate=False).as_expr()
            pair = native.evalf_complex_decimal_native(quotient, digits)
            if pair is None or pair[1] != "0":
                raise RuntimeError("exact linear-root refinement did not return a real decimal")
            text = pair[0]
        else:
            text = _core._native.real_root_decimal_native(coefficients, index, digits)
        unchanged()
        value = _NROOTS_FLOAT(_core._format_decimal(text, digits), digits)
        result.extend([value] * multiplicity)
    unchanged()
    return result


__all__ = ["CRootOf", "ComplexRootOf", "RootOf", "all_roots", "real_roots", "rootof"]
