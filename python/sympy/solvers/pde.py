"""Checked characteristic families for bounded first-order linear PDEs.

The shell retains the original dependent-function owner and typed coordinates.
Only exact rational coefficients and polynomial forcing cross the native
boundary. The native checker establishes the characteristic identities and
the particular-solution residual before an arbitrary function is attached.
"""

from fractions import Fraction
from types import GetSetDescriptorType, MemberDescriptorType

from ..core import (
    Add, AppliedUndef, Basic, Derivative, Dummy, Eq, Function, Integer, Mul,
    Pow, Rational, S, Symbol, UndefinedFunction, _dummy_intern_name,
    _is_closed_function_class, _lift_native_result, _native, _native_expr,
    _require_default_function_identity, _surface_binding_key, _symbol_binding,
    _undefined_function_records,
)
from ..functions.elementary.exponential import exp
from ._ode_system import _require_original_surface
from .ode import _checkodesol_children, _checkodesol_function_protocol


_NUMBERS = (Integer, Rational, type(S.Zero), type(S.One),
            type(S.NegativeOne), type(S.Half))
_BASES = type.__dict__["__bases__"]
_DICT = type.__dict__["__dict__"]
_NAME = type.__dict__["__name__"]
_ABSENT = object()
_MAX_BITS = 4096
_MAX_DEGREE = 16
_MAX_TERMS = 192
_MAX_NODES = 4096
_MAX_DEPTH = 64
_MAX_TEXT = 16_384
_FUNCTION_MEMBERS = frozenset({
    "__module__", "__dict__", "__weakref__", "__doc__", "_nargs", "_kwargs",
    "_undefined_function_options", "_undefined_function_identity",
    "_undefined_function_arity_list",
})


def _plain_function(cls, arity):
    """A registered factory declaration is authority; its spelling is not."""
    if type(cls) is not UndefinedFunction:
        return False
    record = _undefined_function_records.get(id(cls))
    if record is None or record[0] is not cls or record[4] is not False:
        return False
    bases = _BASES.__get__(cls)
    if len(bases) != 1 or bases[0] is not AppliedUndef:
        return False
    declaration = record[3]
    if any(name != "nargs" for name, _ in declaration):
        return False
    namespace = _DICT.__get__(cls)
    if (len(namespace) > len(_FUNCTION_MEMBERS)
            or any(type(name) is not str or name not in _FUNCTION_MEMBERS for name in namespace)):
        return False
    name = _NAME.__get__(cls)
    module = namespace.get("__module__", _ABSENT)
    if (type(name) is not str or len(name) > _MAX_TEXT or name != record[1]
            or type(module) is not str or module != "sympy.core.function"
            or namespace.get("__doc__", _ABSENT) is not None):
        return False
    for field in ("__dict__", "__weakref__"):
        if field in namespace:
            descriptor = namespace[field]
            if (type(descriptor) is not GetSetDescriptorType
                    or descriptor.__objclass__ is not cls or descriptor.__name__ != field):
                return False
    if (namespace.get("_undefined_function_identity", _ABSENT) is not record[2]
            or namespace.get("_undefined_function_options", _ABSENT) is not declaration
            or namespace.get("_undefined_function_arity_list", _ABSENT) is not False):
        return False
    options = namespace.get("_kwargs", _ABSENT)
    if (type(options) is not dict or len(options) != len(declaration)
            or any(type(key) is not str for key in options)
            or any(options.get(key, _ABSENT) is not value for key, value in declaration)):
        return False
    declared = declaration[0][1] if declaration else None
    arities = namespace.get("_nargs", _ABSENT)
    if declared is None:
        return arities is None
    expected = tuple(sorted(set(declared if type(declared) is tuple else (declared,))))
    return (type(arities) is tuple and len(arities) == len(expected)
            and all(type(value) is int and value.bit_length() <= _MAX_BITS for value in arities)
            and arity in expected and arities == expected)


def _protocol():
    _require_original_surface()
    _checkodesol_function_protocol()
    _require_default_function_identity()
    if not _is_closed_function_class(exp):
        raise NotImplementedError("pdsolve requires the original exponential implementation")


def _bounded(value):
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > _MAX_BITS:
        raise ValueError("pdsolve: rational coefficient growth exceeds 4096 bits")
    return value


def _slot(owner, name):
    descriptor = _DICT.__get__(owner).get(name)
    if (type(descriptor) is not MemberDescriptorType
            or descriptor.__objclass__ is not owner or descriptor.__name__ != name):
        raise TypeError("pdsolve requires original symbolic storage descriptors")
    return descriptor


class _Source:
    """Bounded original graph and storage snapshot, without public getters."""

    def __init__(self):
        self.nodes = 0
        self.text = 0
        self.slots = []
        self.mappings = []
        self.functions = []
        self.classes = []
        self.seen = set()
        self.arguments = {}

    def retain(self, node, owner, name):
        descriptor = _slot(owner, name)
        try:
            value = descriptor.__get__(node, owner)
        except AttributeError:
            value = _ABSENT
        self.slots.append((node, type(node), owner, name, descriptor, value))
        if name in ("_surface_args", "_struct_args", "_args", "_derivative_args", "_relational_sides"):
            self.arguments[id(node)].append(value)
        return value

    def snapshot(self, node, kind):
        if id(node) in self.seen:
            return
        self.seen.add(id(node))
        self.arguments[id(node)] = []
        for name in ("_value", "_surface_args", "_struct_args"):
            self.retain(node, Basic, name)
        if any(kind is owner for owner in (Add, Mul, Pow)):
            self.retain(node, kind, "_args")
        elif kind is Derivative:
            self.retain(node, Derivative, "_derivative_args")
        elif kind is Eq:
            from ..core import Relational
            self.retain(node, Relational, "_relational_sides")
        elif type(kind) is UndefinedFunction:
            self.retain(node, Function, "_args")
        if kind is Symbol or kind is Dummy:
            facts = self.retain(node, Symbol, "_assumptions")
            if (type(facts) is not dict or len(facts) > 64
                    or any(type(key) is not str or len(key) > 256 or type(value) is not bool
                           for key, value in facts.items())):
                raise ValueError("pdsolve requires bounded exact symbol assumptions")
            if facts.get("commutative") is False or facts.get("finite") is False or facts.get("infinite") is True:
                raise NotImplementedError("pdsolve requires finite commuting coordinates")
            self.mappings.append((facts, tuple(facts.items())))
            if kind is Dummy:
                self.retain(node, Dummy, "_dummy_name")
                self.retain(node, Dummy, "_dummy_number")

    def arguments_match(self, node, children):
        for retained in self.arguments[id(node)]:
            if retained is _ABSENT or retained is None:
                continue
            if (type(retained) is not tuple or len(retained) != len(children)
                    or any(first is not second for first, second in zip(retained, children))):
                raise ValueError("pdsolve: original source has inconsistent retained arguments")

    def read(self, node, depth=0):
        self.nodes += 1
        if self.nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("pdsolve: original source exceeds its traversal bound")
        kind = type(node)
        if kind is int:
            if node.bit_length() > _MAX_BITS:
                raise ValueError("pdsolve: rational coefficient exceeds 4096 bits")
            return ("number", Fraction(node), (), node)
        number = any(kind is owner for owner in _NUMBERS)
        tag = next((name for owner, name in (
            (Symbol, "symbol"), (Dummy, "symbol"), (Add, "add"), (Mul, "mul"),
            (Pow, "pow"), (Derivative, "derivative"), (Eq, "equation"),
        ) if kind is owner), None)
        if not number and tag is None:
            if not _plain_function(kind, 2):
                raise NotImplementedError("pdsolve requires exact rational linear source expressions")
            tag = "function"
            self.classes.append((kind, 2))
        if tag == "equation" and depth:
            raise NotImplementedError("pdsolve does not admit equations inside arithmetic")
        self.snapshot(node, kind)
        try:
            native = _slot(Basic, "_value").__get__(node, Basic)
        except AttributeError:
            raise ValueError("pdsolve: missing original native source storage") from None
        # Borrowed preflight precedes integer materialization or native-child
        # lifting. A printed native function cannot acquire a Python owner.
        if native is not None:
            if type(native) is not _native.Expr:
                raise ValueError("pdsolve: malformed native source storage")
            _native.ideal_source_preflight_native(native)
        if number:
            self.arguments_match(node, ())
            if native is None or not (native.is_integer or native.is_rational):
                raise ValueError("pdsolve: malformed exact rational source")
            return ("number", _bounded(Fraction(native.exact_numerator(), native.exact_denominator())), (), node)
        if tag == "symbol":
            self.arguments_match(node, ())
            if native is None or not native.is_symbol:
                raise ValueError("pdsolve: malformed coordinate source")
            binding = _symbol_binding(node)
            if _surface_binding_key(binding) != _surface_binding_key(native.symbol_binding):
                raise ValueError("pdsolve: coordinate assumptions disagree with its native identity")
            if kind is Dummy:
                name = _slot(Dummy, "_dummy_name").__get__(node, Dummy)
                index = _slot(Dummy, "_dummy_number").__get__(node, Dummy)
                if (type(name) is not str or len(name) > _MAX_TEXT
                        or type(index) not in (int, bool) or int(index).bit_length() > _MAX_BITS
                        or _dummy_intern_name(name, int(index)) != binding.name):
                    raise ValueError("pdsolve: Dummy declaration disagrees with its native identity")
            self.text += len(binding.name)
            if self.text > _MAX_TEXT:
                raise ValueError("pdsolve: original source exceeds its text bound")
            return (tag, _surface_binding_key(binding), (), node)
        children = _checkodesol_children(node)
        if (type(children) is not tuple or len(children) > 256
                or self.nodes + len(children) > _MAX_NODES):
            raise ValueError("pdsolve: original arguments exceed their bounded shape")
        self.arguments_match(node, children)
        tree = tuple(self.read(child, depth + 1) for child in children)
        if tag == "function":
            if native is not None or len(tree) != 2 or any(child[0] != "symbol" for child in tree):
                raise NotImplementedError("pdsolve requires an ordinary function of two coordinate symbols")
            if tree[0][1] == tree[1][1]:
                raise ValueError("pdsolve requires two distinct typed coordinates")
            self.text += len(_NAME.__get__(kind))
            if self.text > _MAX_TEXT:
                raise ValueError("pdsolve: original source exceeds its text bound")
            self.functions.append((kind, tuple(child[1] for child in tree), node, tree))
        if native is not None and tag in ("add", "mul", "pow"):
            _native.ideal_source_arguments_native(native, tag, [_native_expr(child) for child in children])
        return (tag, kind, tree, node)

    def unchanged(self):
        _protocol()
        if any(not _plain_function(owner, arity) for owner, arity in self.classes):
            raise RuntimeError("pdsolve: a function declaration changed during evaluation")
        for node, kind, owner, name, descriptor, previous in self.slots:
            if type(node) is not kind or _slot(owner, name) is not descriptor:
                raise RuntimeError("pdsolve: original source storage changed during evaluation")
            try:
                current = descriptor.__get__(node, owner)
            except AttributeError:
                current = _ABSENT
            if current is not previous:
                raise RuntimeError("pdsolve: original source changed during evaluation")
        for mapping, previous in self.mappings:
            if len(mapping) != len(previous) or any(mapping.get(key, _ABSENT) is not value for key, value in previous):
                raise RuntimeError("pdsolve: coordinate assumptions changed during evaluation")


class _Polynomials:
    def __init__(self, degree=_MAX_DEGREE):
        self.degree = degree
        self.work = 0

    def step(self, amount=1):
        self.work += amount
        if self.work > 200_000:
            raise ValueError("pdsolve: exact source arithmetic exceeds its work bound")

    @staticmethod
    def constant(value):
        return {(0, 0): value} if value else {}

    @staticmethod
    def scalar(value):
        if any(power != (0, 0) for power in value):
            raise NotImplementedError("pdsolve requires constant rational derivative and function coefficients")
        return value.get((0, 0), Fraction(0))

    def add(self, left, right):
        self.step(len(left) + len(right))
        out = left.copy()
        for power, value in right.items():
            total = _bounded(out.get(power, Fraction(0)) + value)
            if total:
                out[power] = total
            else:
                out.pop(power, None)
        if len(out) > _MAX_TERMS:
            raise ValueError("pdsolve: polynomial source exceeds 192 terms")
        return out

    def scale(self, polynomial, scalar):
        self.step(len(polynomial))
        return {power: _bounded(value*scalar) for power, value in polynomial.items() if scalar}

    def multiply(self, left, right):
        self.step(len(left)*len(right))
        out = {}
        for (i, j), first in left.items():
            for (k, m), second in right.items():
                power = (i+k, j+m)
                if sum(power) > self.degree:
                    raise NotImplementedError("pdsolve: polynomial degree exceeds its bounded regime")
                value = _bounded(out.get(power, Fraction(0)) + _bounded(first*second))
                if value:
                    out[power] = value
                else:
                    out.pop(power, None)
                if len(out) > _MAX_TERMS:
                    raise ValueError("pdsolve: polynomial source exceeds 192 terms")
        return out

    def power(self, base, power):
        result = self.constant(Fraction(1))
        while power:
            if power & 1:
                result = self.multiply(result, base)
            power >>= 1
            if power:
                base = self.multiply(base, base)
        return result


class _LinearSource(_Polynomials):
    def __init__(self, function, axes):
        super().__init__()
        self.function = function
        self.axes = axes

    def constant_row(self, polynomial):
        return [polynomial, Fraction(0), Fraction(0), Fraction(0), False]

    def scale_row(self, row, value):
        return [self.scale(row[0], value), *(_bounded(item*value) for item in row[1:4]), row[4]]

    def visit(self, tree):
        self.step()
        tag, value, children, _ = tree
        if tag == "number":
            return self.constant_row(self.constant(value))
        if tag == "symbol":
            if value not in self.axes:
                raise NotImplementedError("pdsolve: forcing contains a foreign parameter")
            return self.constant_row({(1, 0) if value == self.axes[0] else (0, 1): Fraction(1)})
        if tag == "function":
            if value is not self.function or tuple(child[1] for child in children) != self.axes:
                raise ValueError("pdsolve: dependent-function applications do not share one typed coordinate pair")
            return [{}, Fraction(0), Fraction(0), Fraction(1), True]
        if tag == "derivative":
            if (len(children) != 2 or children[0][0] != "function"
                    or children[1][0] != "symbol" or children[1][1] not in self.axes):
                raise NotImplementedError("pdsolve supports first derivatives of its dependent function only")
            self.visit(children[0])
            row = self.constant_row({})
            row[1 + self.axes.index(children[1][1])] = Fraction(1)
            row[4] = True
            return row
        # Every child is visited before multiplication, cancellation or a
        # zeroth power can discard any original source obligation.
        rows = [self.visit(child) for child in children]
        if tag in ("add", "equation"):
            if tag == "equation" and len(rows) != 2:
                raise ValueError("pdsolve: malformed equation")
            result = self.constant_row({})
            for index, row in enumerate(rows):
                if tag == "equation" and index == 1:
                    row = self.scale_row(row, Fraction(-1))
                result = [self.add(result[0], row[0]),
                          *(_bounded(a+b) for a, b in zip(result[1:4], row[1:4])),
                          result[4] or row[4]]
            return result
        if tag == "mul":
            result = self.constant_row(self.constant(Fraction(1)))
            for row in rows:
                if result[4] and row[4]:
                    raise NotImplementedError("pdsolve requires a linear dependent function and derivatives")
                if row[4]:
                    result = self.scale_row(row, self.scalar(result[0]))
                elif result[4]:
                    result = self.scale_row(result, self.scalar(row[0]))
                else:
                    result = self.constant_row(self.multiply(result[0], row[0]))
            return result
        if tag != "pow" or len(rows) != 2 or children[1][0] != "number":
            raise NotImplementedError("pdsolve requires literal bounded integer powers")
        power = children[1][1]
        if power.denominator != 1 or abs(power.numerator) > 64:
            raise NotImplementedError("pdsolve requires integer powers between -64 and 64")
        power = power.numerator
        base = rows[0]
        if base[4]:
            if power == 1:
                return base
            raise NotImplementedError("pdsolve does not admit powers of the dependent function or its derivatives")
        if power <= 0:
            if children[0][0] != "number" or children[0][1] == 0:
                raise NotImplementedError("pdsolve requires nonzero rational literal bases for nonpositive powers")
            scalar = children[0][1]
            if max(abs(scalar.numerator).bit_length(), scalar.denominator.bit_length())*abs(power) > _MAX_BITS:
                raise ValueError("pdsolve: power coefficient growth exceeds 4096 bits")
            return self.constant_row(self.constant(_bounded(scalar**power)))
        return self.constant_row(self.power(base[0], power))


def _prepare(equation, function=None):
    _protocol()
    source = _Source()
    tree = source.read(equation)
    original = tuple(source.functions)
    if not original:
        raise ValueError("pdsolve could not find a dependent function of two variables")
    if function is None:
        owner, axes, function, arguments = original[0]
    else:
        supplied = source.read(function)
        if supplied[0] != "function":
            raise ValueError("pdsolve requires an applied function of two coordinate symbols")
        owner, axes, _, arguments = source.functions[-1]
    if any(other is not owner or coordinates != axes for other, coordinates, _, _ in original):
        raise ValueError("pdsolve requires one dependent function with unchanged typed coordinates")
    parser = _LinearSource(owner, axes)
    polynomial, a, b, c, _ = parser.visit(tree)
    if not a and not b:
        raise NotImplementedError("pdsolve requires a nonzero first-derivative coefficient")
    source.unchanged()
    return source, function, tuple(child[3] for child in arguments), (a, b, c), parser.scale(polynomial, Fraction(-1))


def _polynomial_expression(polynomial, axes):
    result = Integer(0)
    for (i, j), coefficient in sorted(polynomial.items()):
        term = Rational(coefficient.numerator, coefficient.denominator)
        if i:
            term = term*axes[0]**i
        if j:
            term = term*axes[1]**j
        result = result + term
    return result


def _arbitrary_function(supplied, dependent, axes):
    if supplied is not None:
        if supplied is type(dependent) or not _plain_function(supplied, 1):
            raise NotImplementedError("pdsolve requires a separate ordinary unary arbitrary function")
        return supplied
    reserved = {_NAME.__get__(type(dependent)), *(axis.name for axis in axes)}
    for index in range(256):
        name = "F" if index == 0 else f"F{index}"
        if name not in reserved:
            candidate = Function(name)
            if _plain_function(candidate, 1):
                return candidate
    raise ValueError("pdsolve: no fresh arbitrary-function name is available within its bound")


def pdsolve(eq, func=None, hint="default", dict=False, solvefun=None, **kwargs):
    """Return a checked family for a*u_x + b*u_y + c*u = f(x, y).

    a, b and c are exact rational constants with (a, b) != (0, 0).
    The forcing is an exact polynomial of total degree at most 16. The result
    is Eq(u, exp(-c*s)*F(t) + q): the native checker verifies the affine
    characteristic map, its invertibility, and the original polynomial
    residual. F is an arbitrary differentiable function; setting
    solvefun=Function('Phi') supplies its name and exact Python owner.

    Coordinates retain Symbol/Dummy identity and assumptions. Input is read
    before cancellation; original reciprocals of variables, inexact values,
    custom hooks, higher derivatives and variable coefficients are refused.
    This bounded solution family imposes no initial or boundary data and
    makes no claim about other PDE classes or compatibility-profile closure.
    """
    if type(hint) is not str or hint not in (
        "default", "1st_linear_constant_coeff", "1st_linear_constant_coeff_homogeneous",
    ):
        raise NotImplementedError("pdsolve supports constant-coefficient first-order linear PDE hints")
    if type(dict) is not bool:
        raise TypeError("pdsolve dict must be a bool")
    if dict or kwargs:
        raise NotImplementedError("pdsolve does not support hint dictionaries or additional options")
    source, function, axes, coefficients, polynomial = _prepare(eq, func)
    if hint == "1st_linear_constant_coeff_homogeneous" and polynomial:
        raise ValueError("pdsolve: the homogeneous hint requires zero forcing")
    arbitrary = _arbitrary_function(solvefun, function, axes)
    source.classes.append((arbitrary, 1))
    forcing = _polynomial_expression(polynomial, axes)
    source.unchanged()
    raw = _native.linear_first_order_pde_native(
        *(_native_expr(Rational(value.numerator, value.denominator)) for value in coefficients),
        _native_expr(forcing), *(_symbol_binding(axis) for axis in axes),
    )
    source.unchanged()
    if type(raw) is not tuple or len(raw) != 4 or any(type(value) is not _native.Expr for value in raw):
        raise ValueError("pdsolve: checked native family has an invalid result shape")
    _, invariant, particular, multiplier = (_lift_native_result(value, *axes) for value in raw)
    result = Eq(function, multiplier*arbitrary(invariant, evaluate=False) + particular, evaluate=False)
    source.unchanged()
    return result


__all__ = ["pdsolve"]
