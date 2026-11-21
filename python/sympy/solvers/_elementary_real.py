"""Bounded complete real inversion of one elementary affine argument.

Exact coefficient identities establish ``A*f(a*x+b)+B`` before inversion;
the real range and injectivity (or the two cosh branches) determine every root.
Original inverse operands remain separate domain obligations, including ones
hidden by cancellation or zero products. Unknown membership/definedness stays
an intersection/condition. This shell rule catalog is not a proof certificate.
"""

from fractions import Fraction

from ..core import (
    Add, Dummy, Eq, Expr, Function, Half, Integer, Mul, NegativeOne, One,
    Pow, Rational, Symbol, Zero, _charge_nested_if_active, _native, _native_expr,
    _require_closed_function_class, _surface_binding_key, _symbol_binding, _wrap,
)
from ..functions import acosh, asinh, atanh, cosh, exp, log, sinh, tanh
from ..sets import EmptySet, FiniteSet, Intersection, Reals, Union
from ._checksol import _SolveSetSourceCheck
from ._piecewise_real import _ABSENT, _Capture
from ._real_domain import _ATOMS, _Domain, _SET_TYPES, _is_exact


_FUNCTIONS = {"exp": exp, "log": log, "sinh": sinh, "cosh": cosh,
              "tanh": tanh, "asinh": asinh, "acosh": acosh, "atanh": atanh}
_MAX_DEGREE = 32
_MAX_TERMS = 256
_MAX_BITS = 8192
_MAX_WORK = 65_536


class _UnsupportedShape(Exception):
    pass


class _UndefinedSource(Exception):
    pass


class _ElementaryCapture(_Capture):
    """Freeze native values and every retained spelling before arithmetic."""

    def __init__(self):
        super().__init__()
        self.guards = []

    def touch(self, depth):
        super().touch(depth)
        _charge_nested_if_active(1)

    def unchanged(self):
        super().unchanged()
        for function in _FUNCTIONS.values():
            _require_closed_function_class(function)

    @staticmethod
    def atomic_storage(groups):
        # Literal/symbol owners have no source operands. Generic native Expr
        # wrappers may only delegate to their immutable native tree; a second
        # retained graph must not hide a pole that the native value omitted.
        for children in groups:
            if children is _ABSENT or children is None:
                continue
            if type(children) is not tuple or children:
                raise ValueError("elementary real atom has inconsistent retained arguments")

    def symbol(self, source):
        super().symbol(source)
        self.atomic_storage(self.basic(source))
        native = _native_expr(source)
        if (not native.is_symbol or _surface_binding_key(native.symbol_binding)
                != _surface_binding_key(_symbol_binding(source))):
            raise ValueError("elementary real solving requires consistent typed symbol assumptions")

    def read(self, source, depth=0):
        self.touch(depth)
        kind = type(source)
        if kind is int:
            if source.bit_length() > 4096:
                raise ValueError("elementary real source exceeds its scalar bound")
            source, kind = Integer(source), Integer
        native = _native_expr(source)
        if kind is _native.Expr:
            return self.read_native(native, depth)
        if kind is Symbol or kind is Dummy:
            self.symbol(source)
            return ("atom", native)
        if _is_exact(source, _ATOMS):
            self.atomic_storage(self.basic(source))
            valid = (native.is_integer if kind in (Integer, Zero, One, NegativeOne)
                     else native.is_integer or native.is_rational)
            if not valid:
                raise ValueError("elementary real scalar does not match its native owner")
            fixed = {Zero: (0, 1), One: (1, 1), NegativeOne: (-1, 1), Half: (1, 2)}.get(kind)
            if fixed is not None and (native.exact_numerator(), native.exact_denominator()) != fixed:
                raise ValueError("elementary real constant does not match its native owner")
            return ("atom", native)
        if kind is Expr:
            self.atomic_storage(self.basic(source))
            return self.read_native(native, depth)
        tag = next((name for cls, name in ((Add, "Add"), (Mul, "Mul"), (Pow, "Pow"))
                    if kind is cls), None)
        if tag is None:
            tag = next((name for name, cls in _FUNCTIONS.items() if kind is cls), None)
        if tag is None:
            # Closed unsupported built-ins retain their existing public lane.
            # Closed lowering has already rejected opaque/custom owners.
            self.supported = False
            return self.read_native(native, depth)
        retained = self.basic(source)
        original = self.retain(source, Function if tag in _FUNCTIONS else kind, "_args")
        if original is not _ABSENT:
            retained.append(original)
        expected = native.args
        if native.func_name != tag:
            raise ValueError("elementary real source owner does not match its native operator")
        parsed, groups = None, []
        for children in retained:
            if children is _ABSENT or children is None:
                continue
            if type(children) is not tuple or len(children) > 256:
                raise ValueError("elementary real source requires bounded argument tuples")
            if any(children is previous for previous in groups):
                continue
            groups.append(children)
            current = tuple(_native_expr(child) for child in children)
            if (len(current) != len(expected)
                    or any(left != right for left, right in zip(current, expected))):
                raise ValueError("elementary real source does not match its retained arguments")
            values = tuple(self.read(child, depth + 1) for child in children)
            if parsed is None:
                parsed = values
        if parsed is None:
            parsed = tuple(self.read_native(child, depth + 1) for child in expected)
        return self.node(native, parsed)

    def read_native(self, native, depth):
        self.touch(depth)
        if native.is_symbol:
            self.symbol(_wrap(native))
            return ("atom", native)
        if native.is_integer or native.is_rational:
            return ("atom", native)
        children = native.args
        if len(children) > 256:
            raise ValueError("elementary real source exceeds its arity bound")
        return self.node(native, tuple(self.read_native(child, depth + 1) for child in children))

    def node(self, native, children):
        tag = native.func_name
        if tag == "Pow" and len(children) == 2:
            exponent = native.args[1]
            if exponent.is_integer:
                if abs(exponent.exact_numerator()) > _MAX_DEGREE:
                    raise ValueError("elementary real source exceeds its power bound")
                if exponent.exact_numerator() < 0:
                    self.guards.append(_wrap(native))
                return (tag, children)
        elif tag in ("Add", "Mul") and children:
            return (tag, children)
        elif tag in _FUNCTIONS and len(children) == 1:
            return (tag, children)
        self.supported = False
        return ("unsupported", ())


class _Algebra:
    """Bounded coefficient identities without a symbolic simplifier.

Monomials are (coordinate degree, elementary-value degree). Arithmetic shares
one allowance across recognition and the final identity check.
"""

    def __init__(self, variable):
        self.variable = _native_expr(variable)
        self.remaining = _MAX_WORK
        self.elementary = None

    def charge(self, amount=1):
        if amount > self.remaining:
            raise ValueError("elementary real solving exceeds its algebra work bound")
        self.remaining -= amount
        _charge_nested_if_active(amount)

    def scalar(self, value):
        self.charge()
        if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > _MAX_BITS:
            raise ValueError("elementary real solving exceeds its coefficient bound")
        return value

    def preflight(self, left, right):
        bits = max(abs(left.numerator).bit_length(), left.denominator.bit_length())
        bits += max(abs(right.numerator).bit_length(), right.denominator.bit_length()) + 2
        if bits > _MAX_BITS:
            raise ValueError("elementary real solving exceeds its coefficient growth bound")
        self.charge(max(1, (bits + 63)//64))

    def binary(self, left, right, multiply=False):
        self.preflight(left, right)
        return self.scalar(left*right if multiply else left+right)

    def divide(self, left, right):
        self.preflight(left, right)
        return self.scalar(left/right)

    def const(self, value):
        value = self.scalar(Fraction(value))
        return {(0, 0): value} if value else {}

    def add(self, left, right):
        self.charge(len(left) + len(right) + 1)
        out = dict(left)
        for monomial, value in right.items():
            combined = self.binary(out.get(monomial, Fraction(0)), value)
            if combined:
                out[monomial] = combined
            else:
                out.pop(monomial, None)
        if len(out) > _MAX_TERMS:
            raise ValueError("elementary real solving exceeds its term bound")
        return out

    def mul(self, left, right):
        self.charge(len(left)*len(right) + 1)
        out = {}
        for (x1, f1), a in left.items():
            for (x2, f2), b in right.items():
                monomial = (x1+x2, f1+f2)
                if sum(monomial) > _MAX_DEGREE:
                    raise ValueError("elementary real solving exceeds its degree bound")
                value = self.binary(out.get(monomial, Fraction(0)), self.binary(a, b, True))
                if value:
                    out[monomial] = value
                else:
                    out.pop(monomial, None)
                if len(out) > _MAX_TERMS:
                    raise ValueError("elementary real solving exceeds its term bound")
        return out

    def power(self, poly, exponent):
        result = self.const(1)
        while exponent:
            if exponent & 1:
                result = self.mul(result, poly)
            exponent //= 2
            if exponent:
                poly = self.mul(poly, poly)
        return result

    def affine(self, numerator, denominator, axis):
        """Prove P = Q*(a*u+b) by coefficients, without numerical sampling."""
        if not denominator:
            raise _UndefinedSource
        if not numerator:
            return Fraction(0), Fraction(0)
        key = max(denominator, key=lambda monomial: (monomial[axis], monomial[1-axis]))
        degree = max(monomial[axis] for monomial in numerator)
        shifted = (key[0]+(axis == 0), key[1]+(axis == 1))
        slope = (self.divide(numerator.get(shifted, Fraction(0)), denominator[key])
                 if degree == key[axis]+1 else Fraction(0))
        linear = {(int(axis == 0), int(axis == 1)): -slope} if slope else {}
        remainder = self.add(numerator, self.mul(denominator, linear))
        offset = self.divide(remainder.get(key, Fraction(0)), denominator[key])
        if remainder != self.mul(denominator, self.const(offset)):
            raise _UnsupportedShape
        return slope, offset

    def read(self, tree, inner=False):
        self.charge()
        tag, children = tree
        one = self.const(1)
        if tag == "atom":
            value = children
            if value.is_integer or value.is_rational:
                return self.const(Fraction(value.exact_numerator(), value.exact_denominator())), one
            if value.is_symbol and value == self.variable:
                return {(1, 0): Fraction(1)}, one
            raise _UnsupportedShape
        if tag in _FUNCTIONS:
            if inner:
                raise _UnsupportedShape
            a, b = self.affine(*self.read(children[0], inner=True), 0)
            if not a:
                raise _UnsupportedShape
            identity = (tag, a, b)
            if self.elementary is not None and self.elementary != identity:
                raise _UnsupportedShape
            self.elementary = identity
            return {(0, 1): Fraction(1)}, one
        if tag == "Pow":
            if children[1][0] != "atom" or not children[1][1].is_integer:
                raise _UnsupportedShape
            numerator, denominator = self.read(children[0], inner)
            exponent = children[1][1].exact_numerator()
            if exponent < 0:
                if not numerator:
                    raise _UndefinedSource
                numerator, denominator = denominator, numerator
            return self.power(numerator, abs(exponent)), self.power(denominator, abs(exponent))
        if tag not in ("Add", "Mul"):
            raise _UnsupportedShape
        numerator, denominator = (self.const(0) if tag == "Add" else one), one
        for child in children:
            p, q = self.read(child, inner)
            if tag == "Add":
                numerator = self.add(self.mul(numerator, q), self.mul(p, denominator))
            else:
                numerator = self.mul(numerator, p)
            denominator = self.mul(denominator, q)
        return numerator, denominator


def _rational(value):
    return Rational(value.numerator, value.denominator)


def _inverse_values(name, target):
    """All real preimages, with exact range checks before constructing them."""
    value = _rational(target)
    if name == "exp":
        return [] if target <= 0 else [log(value)]
    if name == "log":
        return [exp(value)]
    if name == "sinh":
        return [asinh(value)]
    if name == "cosh":
        if target < 1:
            return []
        return [Integer(0)] if target == 1 else [-acosh(value), acosh(value)]
    if name == "tanh":
        return [] if not -1 < target < 1 else [atanh(value)]
    if name == "asinh":
        return [sinh(value)]
    if name == "acosh":
        return [] if target < 0 else [cosh(value)]
    if name == "atanh":
        return [tanh(value)]
    raise ValueError("unknown elementary real inversion rule")


def solve_elementary_real(expression, variable, domain):
    """A complete bounded set, or None for another source grammar.

Only grammar rejection selects another algorithm. Resource limits, source
mutation and rejected storage/owner checks remain terminal on this route.
"""
    if not (_is_exact(domain, _SET_TYPES) or type(domain) is Reals):
        return None
    capture = _ElementaryCapture()
    sides = capture.sides(expression) if type(expression) is Eq else (expression, Integer(0))
    natives = tuple(_native_expr(side) for side in sides)
    if not any(native.has_function_bounded(name) for native in natives for name in _FUNCTIONS):
        return None
    capture.unchanged()
    for native in natives:
        _native.real_constraint_source_preflight_native(native)
    trees = tuple(capture.read(side) for side in sides)
    if variable is None:
        if len(capture.symbols) != 1:
            return None
        variable = capture.symbols[0]
    if type(variable) is not Symbol and type(variable) is not Dummy:
        raise TypeError("elementary real solving requires an exact Symbol")
    capture.symbol(variable)
    if not capture.supported or any(symbol != variable for symbol in capture.symbols):
        return None
    snapshot = _Domain(domain)
    if not snapshot.real:
        return None

    def unchanged():
        capture.unchanged()
        snapshot.unchanged()

    unchanged()
    algebra = _Algebra(variable)
    try:
        lp, lq = algebra.read(trees[0])
        rp, rq = algebra.read(trees[1])
        numerator = algebra.add(algebra.mul(lp, rq),
                                algebra.mul(algebra.mul(rp, lq), algebra.const(-1)))
        scale, offset = algebra.affine(numerator, algebra.mul(lq, rq), 1)
    except _UnsupportedShape:
        unchanged()
        return None
    except _UndefinedSource:
        unchanged()
        return EmptySet()
    if algebra.elementary is None:
        return None
    original = _SolveSetSourceCheck(expression)
    if not scale:
        result = original.condition_set(variable, domain) if not offset else EmptySet()
        unchanged()
        return result
    name, slope, shift = algebra.elementary
    values = _inverse_values(name, algebra.divide(-offset, scale))
    roots = [(value - _rational(shift))/_rational(slope) for value in values]
    unchanged()
    # The real inversion rule proves every elementary function defined at these
    # preimages. Check every original rational inverse separately, including
    # nested poles and those inside the admitted affine argument.
    conditional = False
    if capture.guards:
        roots, conditional = _SolveSetSourceCheck(tuple(capture.guards)).finite(roots, variable)
    known, uncertain = [], []
    for root in roots:
        descriptor = snapshot.scalar(_native_expr(root))
        membership = True if type(domain) is Reals else snapshot.contains(descriptor)
        if membership is not False:
            (known if membership is True else uncertain).append(root)
    result = FiniteSet(*known)
    if uncertain:
        result = Union(result, Intersection(domain, FiniteSet(*uncertain)))
    if conditional:
        result = original.condition_set(variable, result)
    unchanged()
    return result
