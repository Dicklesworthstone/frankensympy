"""Bounded real-axis continuity domains assembled from exact sign charts.

The shell retains and checks the original source before deriving a conjunction
of rational conditions. One independently checked native partition then decides
every condition together. This is a compatibility operation, not a new portable
continuity certificate or a general transcendental inequality solver.

Rational powers follow the pinned SymPy 1.14 domain convention: an odd
denominator introduces no sign restriction, while a negative power excludes
zero. That convention does not establish that a principal odd root is real on
negative inputs. Such nodes are never used as proved-real inputs to a
branch-sensitive outer function.
"""

from fractions import Fraction

from ..core import (
    Abs, Add, Basic, Dummy, Expr, Function, Half, Integer, Mul, NegativeOne,
    One, Pow, Rational, Symbol, Zero, _charge_nested_if_active, _native,
    _native_expr, _require_closed_function_class, _surface_binding_key,
    _symbol_binding, _wrap,
)
from ..functions.elementary.exponential import exp, log
from ..functions.elementary.hyperbolic import (
    acosh, acoth, acsch, asech, asinh, atanh, cosh, coth, csch, sech, sinh,
    tanh,
)
from ..functions.elementary.trigonometric import (
    acos, acot, acsc, asec, asin, atan, cos, sin,
)
from ..sets import EmptySet, Reals
from ._rational_range import _require_original_surface


_NUMBERS = (Integer, Rational, Zero, One, NegativeOne, Half)
_FUNCTIONS = (
    (Abs, "Abs"), (exp, "exp"), (log, "log"),
    (sin, "sin"), (cos, "cos"), (sinh, "sinh"), (cosh, "cosh"),
    (tanh, "tanh"), (sech, "sech"), (coth, "coth"), (csch, "csch"),
    (asin, "asin"), (acos, "acos"), (atan, "atan"), (acot, "acot"),
    (asec, "asec"), (acsc, "acsc"), (asinh, "asinh"), (acosh, "acosh"),
    (atanh, "atanh"), (acoth, "acoth"), (asech, "asech"), (acsch, "acsch"),
)
_ARITHMETIC = ((Add, "Add"), (Mul, "Mul"), (Pow, "Pow"))
_FIXED_NUMBERS = ((Zero, 0, 1), (One, 1, 1), (NegativeOne, -1, 1), (Half, 1, 2))


def _unsupported():
    raise NotImplementedError(
        "continuous_domain requires bounded rational conditions on the original real source"
    )


def _integer(value):
    return _native.py_integer(value)


def _held_add(left, right):
    return _native.Add(left, right, evaluate=False).as_expr()


def _held_mul(left, right):
    return _native.Mul(left, right, evaluate=False).as_expr()


def _subtract_integer(source, value):
    return _held_add(source, _integer(-value))


class _Node:
    __slots__ = ("tag", "native", "children", "rational", "real", "power")

    def __init__(self, tag, native, children=()):
        self.tag, self.native, self.children = tag, native, children
        self.real = False
        self.power = None
        if tag in ("number", "symbol"):
            self.rational = True
        elif tag in ("Add", "Mul"):
            self.rational = all(child.rational for child in children)
        elif tag == "Pow":
            if len(children) != 2 or children[1].tag != "number":
                _unsupported()
            exponent = children[1].native
            self.power = Fraction(exponent.exact_numerator(), exponent.exact_denominator())
            if abs(self.power.numerator) > 64 or self.power.denominator > 64:
                raise ValueError("continuous_domain rational exponent exceeds its bounded envelope")
            self.rational = children[0].rational and self.power.denominator == 1
        else:
            self.rational = False


class _Source:
    """Capture actual storage and typed owners, not printed function names."""

    def __init__(self):
        # Solver helpers load here, after the package finishes initializing.
        # Their own owner records, and _rational_range's records, were frozen
        # during package import before callers could alter implementations.
        from ..solvers._piecewise_real import _Capture

        self.capture = _Capture()
        self.function_owners = []

    def touch(self, depth):
        self.capture.touch(depth)
        _charge_nested_if_active(1)

    def function(self, owner):
        _require_closed_function_class(owner)
        if not any(owner is previous for previous in self.function_owners):
            self.function_owners.append(owner)

    def symbol(self, source):
        self.capture.symbol(source)
        self.empty_arguments(self.capture.basic(source))
        native = _native_expr(source)
        if (not native.is_symbol or _surface_binding_key(native.symbol_binding)
                != _surface_binding_key(_symbol_binding(source))):
            raise ValueError("continuous_domain symbol has inconsistent typed assumptions")
        facts = self.capture.dictionaries[-1][0]
        if (facts.get("commutative") is False or facts.get("finite") is False
                or facts.get("infinite") is True or facts.get("real") is False
                or facts.get("imaginary") is True):
            _unsupported()

    @staticmethod
    def empty_arguments(retained):
        from ..solvers._piecewise_real import _ABSENT

        if any(value is not _ABSENT and value is not None
               and (type(value) is not tuple or value)
               for value in retained):
            raise ValueError("continuous_domain atomic source has inconsistent arguments")

    def unchanged(self):
        from ..solvers.inequalities import _constraint_owners_unchanged

        _require_original_surface()
        _constraint_owners_unchanged()
        for owner in self.function_owners:
            _require_closed_function_class(owner)
        self.capture.unchanged()

    def read(self, source, depth=0):
        from ..solvers._piecewise_real import _ABSENT

        self.touch(depth)
        kind = type(source)
        if kind is int:
            if source.bit_length() > 4096:
                raise ValueError("continuous_domain coefficient exceeds 4096 bits")
            source, kind = Integer(source), None
            kind = type(source)
        number = any(kind is owner for owner in _NUMBERS)
        tag = next((name for owner, name in _ARITHMETIC if kind is owner), None)
        function = next(((owner, name) for owner, name in _FUNCTIONS if kind is owner), None)
        if (not number and tag is None and function is None
                and not any(kind is owner for owner in (Symbol, Dummy, Expr, _native.Expr))):
            _unsupported()
        if function is not None:
            self.function(function[0])
            tag = function[1]
        native = _native_expr(source)
        # This borrowed traversal bounds every child before native .args can
        # copy it, even when its grammar is not rational. Retained children
        # receive the same check in case their handles have become stale.
        _native.real_constraint_source_preflight_native(native)
        if kind is _native.Expr:
            return self.read_native(native, depth)
        if kind in (Symbol, Dummy):
            self.symbol(source)
            return _Node("symbol", native)
        if number:
            self.empty_arguments(self.capture.basic(source))
            integer_owner = any(kind is owner for owner in (Integer, Zero, One, NegativeOne))
            if not (native.is_integer if integer_owner else native.is_integer or native.is_rational):
                raise ValueError("continuous_domain scalar owner disagrees with its native value")
            for owner, numerator, denominator in _FIXED_NUMBERS:
                if kind is owner and (native.exact_numerator(), native.exact_denominator()) != (numerator, denominator):
                    raise ValueError("continuous_domain singleton has an inconsistent value")
            return _Node("number", native)
        retained = self.capture.basic(source)
        if kind is Expr:
            result = self.read_native(native, depth)
            expected = native.args
            for children in retained:
                if children is _ABSENT or children is None:
                    continue
                if (type(children) is not tuple or len(children) > 256
                        or len(children) != len(expected)):
                    raise ValueError("continuous_domain source disagrees with its retained arguments")
                parsed = tuple(self.read(child, depth + 1) for child in children)
                if any(child.native != original for child, original in zip(parsed, expected)):
                    raise ValueError("continuous_domain source disagrees with its retained arguments")
            return result
        original = self.capture.retain(source, Function if function is not None else kind, "_args")
        if original is not _ABSENT:
            retained.append(original)
        groups = []
        for children in retained:
            if children is _ABSENT or children is None:
                continue
            if type(children) is not tuple or len(children) > 256:
                raise ValueError("continuous_domain requires bounded original argument tuples")
            if not any(children is previous for previous in groups):
                groups.append(children)
        expected = native.args
        native_tag = "Abs" if native.func_name == "abs" else native.func_name
        if native_tag != tag:
            raise ValueError("continuous_domain source owner disagrees with its native operator")
        parsed = None
        for children in groups:
            if len(children) != len(expected):
                raise ValueError("continuous_domain source disagrees with its retained arguments")
            current = tuple(self.read(child, depth + 1) for child in children)
            if any(child.native != expected_child for child, expected_child in zip(current, expected)):
                raise ValueError("continuous_domain source disagrees with its retained arguments")
            if parsed is None:
                parsed = current
        if parsed is None:
            parsed = tuple(self.read_native(child, depth + 1) for child in expected)
        return self.node(tag, native, parsed)

    def read_native(self, native, depth):
        self.touch(depth)
        if native.is_symbol:
            self.symbol(_wrap(native))
            return _Node("symbol", native)
        if native.is_integer or native.is_rational:
            return _Node("number", native)
        tag = "Abs" if native.func_name == "abs" else native.func_name
        owner = next((owner for owner, name in _FUNCTIONS if tag == name), None)
        if tag not in ("Add", "Mul", "Pow") and owner is None:
            _unsupported()
        if owner is not None:
            self.function(owner)
        children = native.args
        if len(children) > 256:
            raise ValueError("continuous_domain source exceeds its arity bound")
        return self.node(tag, native, tuple(self.read_native(child, depth + 1) for child in children))

    @staticmethod
    def node(tag, native, children):
        if tag not in ("Add", "Mul", "Pow") and len(children) != 1:
            _unsupported()
        return _Node(tag, native, children)


class _Conditions:
    """Derive one flat rational conjunction without evaluating held sources."""

    def __init__(self):
        self.sources = []
        self.operations = []

    def add(self, source, operation):
        _charge_nested_if_active(1)
        if not _native.real_constraint_source_preflight_native(source):
            _unsupported()
        for previous, previous_operation in zip(self.sources, self.operations):
            if operation == previous_operation and source == previous:
                return
        if len(self.sources) == 32:
            raise ValueError("continuous_domain exceeds 32 independent rational conditions")
        self.sources.append(source)
        self.operations.append(operation)

    def constant(self, value, operation, bound):
        passed = {
            "==": value == bound, "!=": value != bound,
            "<": value < bound, "<=": value <= bound,
            ">": value > bound, ">=": value >= bound,
        }[operation]
        if not passed:
            self.add(_integer(0), ">")

    def require(self, node, operation, bound=0):
        _charge_nested_if_active(1)
        if node.rational:
            self.add(_subtract_integer(node.native, bound), operation)
            return
        if node.tag == "exp":
            if bound == 0:
                # exp never vanishes, including on nonreal admitted sources.
                if operation in ("==", "!=") or node.children[0].real:
                    self.constant(1, operation, bound)
                    return
            elif bound == 1 and node.children[0].real:
                self.require(node.children[0], operation)
                return
        if node.tag == "log" and bound == 0:
            self.require(node.children[0], operation, 1)
            return
        if node.tag == "cosh" and bound == 0 and node.children[0].real:
            self.constant(1, operation, 0)
            return
        if bound == 0 and node.real and node.tag in ("Pow", "Abs"):
            base = node.children[0]
            if node.tag == "Pow":
                power = node.power
                if power == 0:
                    self.constant(1, operation, 0)
                    return
                if power.denominator == 1 and power.numerator % 2:
                    self.require(base, operation)
                    return
                if power < 0:
                    self.constant(1, operation, 0)
                    return
            # Positive even powers, even roots and Abs are nonnegative, and
            # vanish precisely when their admitted original argument vanishes.
            if operation in (">=", "<"):
                self.constant(1, operation, 0)
            elif operation in (">", "!="):
                self.require(base, "!=")
            else:
                self.require(base, "==")
            return
        _unsupported()

    def walk(self, node):
        _charge_nested_if_active(1)
        if node.rational:
            # Native source reduction visits children before identities:
            # neither 0*f nor f**0 erases an original reciprocal pole.
            self.add(_held_mul(_integer(0), node.native), ">=")
            node.real = True
            return
        for child in node.children:
            self.walk(child)
        if node.tag in ("Add", "Mul"):
            node.real = all(child.real for child in node.children)
            return
        if node.tag == "Pow":
            base, exponent = node.children
            power = node.power
            if power.denominator == 1:
                if power < 0:
                    self.require(base, "!=")
                node.real = base.real or power == 0
            elif power.denominator % 2 == 0:
                if not base.real:
                    _unsupported()
                self.require(base, ">" if power < 0 else ">=")
                node.real = True
            else:
                # Preserve the pinned odd-denominator domain convention
                # without asserting that a negative principal root is real.
                if not base.real:
                    _unsupported()
                if power < 0:
                    self.require(base, "!=")
                node.real = False
            return
        argument = node.children[0]
        if node.tag in ("exp", "sin", "cos", "sinh", "cosh"):
            node.real = argument.real
            return
        if node.tag == "Abs":
            node.real = True
            return
        if not argument.real:
            _unsupported()
        if node.tag in ("atan", "asinh", "tanh", "sech"):
            pass
        elif node.tag == "log":
            self.require(argument, ">")
        elif node.tag in ("asin", "acos"):
            self.require(argument, ">=", -1)
            self.require(argument, "<=", 1)
        elif node.tag == "acosh":
            self.require(argument, ">=", 1)
        elif node.tag == "atanh":
            self.require(argument, ">", -1)
            self.require(argument, "<", 1)
        elif node.tag in ("acot", "acsch", "coth", "csch"):
            self.require(argument, "!=")
        elif node.tag == "asech":
            self.require(argument, ">")
            self.require(argument, "<=", 1)
        elif node.tag in ("asec", "acsc", "acoth"):
            if not argument.rational:
                _unsupported()
            squared = _native.Pow(argument.native, _integer(2), evaluate=False).as_expr()
            self.add(_subtract_integer(squared, 1), ">" if node.tag == "acoth" else ">=")
        else:
            _unsupported()
        node.real = True


def elementary_continuous_domain(expression, symbol=None, domain=None):
    """Compute a bounded domain, retaining unsupported and terminal failures."""
    from ..solvers._piecewise_real import _restrict
    from ..solvers._real_domain import _Domain, _SET_TYPES
    from ..solvers.inequalities import _constraint_owners_unchanged, _lift_constraint_chart

    _require_original_surface()
    _constraint_owners_unchanged()
    if type(expression) is str:
        if len(expression) > 4096:
            raise ValueError("continuous_domain source text exceeds 4096 characters")
        expression = _wrap(_native_expr(expression))
    if domain is None or domain is Reals:
        domain = Reals()
    elif domain is EmptySet:
        domain = EmptySet()
    if not any(type(domain) is owner for owner in (*_SET_TYPES, Reals)):
        _unsupported()
    snapshot = _Domain(domain)
    if not snapshot.real or any(descriptor is None for _, descriptor in snapshot.domain_values):
        _unsupported()
    capture = _Source()
    tree = capture.read(expression)
    source_symbols = tuple(capture.capture.symbols)
    if symbol is None:
        if len(source_symbols) > 1:
            raise ValueError("continuous_domain requires one variable")
        symbol = source_symbols[0] if source_symbols else Dummy("_fsym_continuity_coordinate")
    if type(symbol) is not Symbol and type(symbol) is not Dummy:
        raise TypeError("continuous_domain variable must be an exact Symbol or Dummy")
    capture.symbol(symbol)
    if any(coordinate != symbol for coordinate in source_symbols):
        _unsupported()

    def unchanged():
        capture.unchanged()
        snapshot.unchanged()

    unchanged()
    conditions = _Conditions()
    conditions.walk(tree)
    unchanged()
    checked = _native.solve_real_rational_constraints_native(
        conditions.sources, _symbol_binding(symbol), conditions.operations,
    )
    unchanged()
    if checked is None:
        raise NotImplementedError("admitted continuous_domain conditions could not be completed")
    selected = _lift_constraint_chart(checked, symbol)
    unchanged()
    result = _restrict(domain, selected)
    _Domain(result)
    unchanged()
    return result
