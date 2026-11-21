"""Complete real solution sets for one principal square root over QQ(x).

On the original common domain, P + Q*sqrt(R) vanishes exactly when
P**2 - Q**2*R == 0 and P*Q <= 0. If Q is nonzero the first condition forces
R >= 0. If Q vanishes it forces P == 0, including finite negative radicands:
a principal complex square root multiplied by zero is still defined.

A bounded structural generator proposes P and Q. A separate formal rational
identity check binds them to the original source, and independently checked
rational sign charts establish the complete set. A rational domain shadow
retains every original pole before any cancellation, zero factor or power.
This is an in-process shell construction, not a portable proof certificate.
"""

from ..core import (
    Add, Basic, Dummy, Eq, Expr, Half, Integer, Mul, NegativeOne, One, Pow,
    Symbol, Zero, _charge_nested_if_active, _native, _native_expr,
    _surface_binding_key, _symbol_binding, _wrap,
)
from ..polys._opaque import rational_zero
from ..sets import ConditionSet, Reals
from ._piecewise_real import _ABSENT, _Capture, _checked_domain, _restrict
from ._real_domain import _ATOMS, _Domain, _SET_TYPES, _is_exact
from .inequalities import _constraint_owners_unchanged, _lift_constraint_chart


_MAX_WORK = 65_536
_MAX_OUTPUT_NODES = 2048
_MAX_DEPTH = 64


class _UnsupportedShape(Exception):
    pass


class _Tree:
    __slots__ = ("native", "children", "root", "dependent", "nodes", "depth")

    def __init__(self, native, children=(), root=False):
        self.native = native
        self.children = children
        self.root = root
        self.dependent = root or any(child.dependent for child in children)
        self.nodes = 1 + sum(child.nodes for child in children)
        self.depth = 1 + max((child.depth for child in children), default=0)


class _Value:
    __slots__ = ("native", "nodes", "depth")

    def __init__(self, native, nodes=1, depth=1):
        self.native, self.nodes, self.depth = native, nodes, depth


_ZERO = _Value(_native.py_integer(0))
_ONE = _Value(_native.py_integer(1))
_NEGATIVE_ONE = _Value(_native.py_integer(-1))


def _literal(value, integer):
    return value.native.is_integer and value.native.exact_numerator() == integer


def _value(tree):
    return _Value(tree.native, tree.nodes, tree.depth)


class _Work:
    """One allowance for coefficient construction, replay and publication."""

    def __init__(self):
        self.remaining = _MAX_WORK

    def charge(self, amount=1):
        if type(amount) is not int or amount < 0 or amount > self.remaining:
            raise ValueError("principal-square-root solving exceeds its work bound")
        self.remaining -= amount
        _charge_nested_if_active(amount)

    def make(self, operation, values, *, prune=True):
        values = tuple(values)
        if prune and operation == "Add":
            values = tuple(value for value in values if not _literal(value, 0))
            if not values:
                return _ZERO
            if len(values) == 1:
                return values[0]
        if prune and operation == "Mul":
            if any(_literal(value, 0) for value in values):
                return _ZERO
            values = tuple(value for value in values if not _literal(value, 1))
            if not values:
                return _ONE
            if len(values) == 1:
                return values[0]
        nodes = 1 + sum(value.nodes for value in values)
        depth = 1 + max((value.depth for value in values), default=0)
        if nodes > _MAX_OUTPUT_NODES or depth > _MAX_DEPTH or len(values) > 256:
            raise ValueError("principal-square-root construction exceeds its expression bound")
        self.charge(nodes)
        raw = tuple(value.native for value in values)
        # Held constructors avoid unrecorded simplification and preserve every
        # original rational inverse. Only literal zero/one pruning above is
        # used for the proposed coefficients; the domain shadow never prunes.
        if operation == "Add":
            native = _native.Add(*raw, evaluate=False).as_expr()
        elif operation == "Mul":
            native = _native.Mul(*raw, evaluate=False).as_expr()
        elif operation == "Pow" and len(raw) == 2:
            native = _native.Pow(*raw, evaluate=False).as_expr()
        else:
            raise ValueError("invalid principal-square-root construction operator")
        return _Value(native, nodes, depth)

    def negative(self, value):
        return self.make("Mul", (_NEGATIVE_ONE, value))

    def square(self, value):
        return self.make("Pow", (value, _Value(_native.py_integer(2))))


def _contains_square_root(natives):
    for native in natives:
        # The borrowed presence walker bounds the entire stored tree, even
        # after finding a function. Square roots themselves are Pow nodes,
        # so inspect their operands only after this shared source admission.
        native.has_function_bounded("sqrt")
        pending = [native]
        while pending:
            node = pending.pop()
            children = node.args
            if node.is_pow and len(children) == 2:
                exponent = children[1]
                if (exponent.is_rational and exponent.exact_numerator() == 1
                        and exponent.exact_denominator() == 2):
                    return True
            pending.extend(children)
    return False


class _RadicalCapture(_Capture):
    """Freeze exact owners and every retained spelling before decomposition."""

    def __init__(self):
        super().__init__()
        self.radicand = None

    def touch(self, depth):
        super().touch(depth)
        _charge_nested_if_active(1)

    def unchanged(self):
        _constraint_owners_unchanged()
        super().unchanged()

    @staticmethod
    def atomic_storage(groups):
        for children in groups:
            if children is _ABSENT or children is None:
                continue
            if type(children) is not tuple or children:
                raise ValueError("principal-square-root atom has inconsistent retained arguments")

    def symbol(self, source):
        super().symbol(source)
        self.atomic_storage(self.basic(source))
        native = _native_expr(source)
        if (not native.is_symbol or _surface_binding_key(native.symbol_binding)
                != _surface_binding_key(_symbol_binding(source))):
            raise ValueError("principal-square-root source has inconsistent typed assumptions")

    def read(self, source, depth=0):
        self.touch(depth)
        kind = type(source)
        if kind is int:
            if source.bit_length() > 4096:
                raise ValueError("principal-square-root source exceeds its scalar bound")
            source, kind = Integer(source), Integer
        native = _native_expr(source)
        if kind is _native.Expr:
            return self.read_native(native, depth)
        if kind is Symbol or kind is Dummy:
            self.symbol(source)
            return _Tree(native)
        if _is_exact(source, _ATOMS):
            self.atomic_storage(self.basic(source))
            valid = (native.is_integer if kind in (Integer, Zero, One, NegativeOne)
                     else native.is_integer or native.is_rational)
            if not valid:
                raise ValueError("principal-square-root scalar does not match its native owner")
            fixed = {Zero: (0, 1), One: (1, 1), NegativeOne: (-1, 1), Half: (1, 2)}.get(kind)
            if fixed is not None and (native.exact_numerator(), native.exact_denominator()) != fixed:
                raise ValueError("principal-square-root constant does not match its native owner")
            return _Tree(native)
        if kind is Expr:
            self.atomic_storage(self.basic(source))
            return self.read_native(native, depth)
        operator = next((name for cls, name in ((Add, "Add"), (Mul, "Mul"), (Pow, "Pow"))
                         if kind is cls), None)
        if operator is None:
            self.supported = False
            if isinstance(source, Basic):
                self.basic(source)
            return self.read_native(native, depth)
        groups = self.basic(source)
        original = self.retain(source, kind, "_args")
        if original is not _ABSENT:
            groups.append(original)
        if native.func_name != operator:
            raise ValueError("principal-square-root source owner does not match its native operator")
        expected = native.args
        parsed, seen = None, []
        for children in groups:
            if children is _ABSENT or children is None:
                continue
            if type(children) is not tuple or len(children) > 256:
                raise ValueError("principal-square-root source requires bounded argument tuples")
            if any(children is previous for previous in seen):
                continue
            seen.append(children)
            current = tuple(_native_expr(child) for child in children)
            if (len(current) != len(expected)
                    or any(left != right for left, right in zip(current, expected))):
                raise ValueError("principal-square-root source does not match its retained arguments")
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
            return _Tree(native)
        if native.is_integer or native.is_rational:
            return _Tree(native)
        children = native.args
        if len(children) > 256:
            raise ValueError("principal-square-root source exceeds its arity bound")
        return self.node(native, tuple(self.read_native(child, depth + 1) for child in children))

    def node(self, native, children):
        root = False
        if native.is_pow and len(children) == 2:
            exponent = children[1].native
            if exponent.is_integer:
                power = exponent.exact_numerator()
                if abs(power) > 64:
                    raise ValueError("principal-square-root source exceeds its power bound")
                if children[0].dependent and power not in (0, 1):
                    self.supported = False
            elif (exponent.is_rational and exponent.exact_numerator() == 1
                  and exponent.exact_denominator() == 2):
                root = True
                if children[0].dependent:
                    self.supported = False
                if self.radicand is None:
                    self.radicand = children[0]
                elif self.radicand.native != children[0].native:
                    self.supported = False
            else:
                self.supported = False
        elif not (native.is_add or native.is_mul):
            self.supported = False
        return _Tree(native, children, root)


def _decompose(tree, work):
    """Propose rational coefficients of a structurally affine root value."""
    work.charge()
    if not tree.dependent:
        return _value(tree), _ZERO
    if tree.root:
        return _ZERO, _ONE
    if tree.native.is_add:
        parts = [_decompose(child, work) for child in tree.children]
        return (work.make("Add", (part[0] for part in parts)),
                work.make("Add", (part[1] for part in parts)))
    if tree.native.is_mul:
        dependent = [child for child in tree.children if child.dependent]
        if len(dependent) != 1:
            raise _UnsupportedShape
        p, q = _decompose(dependent[0], work)
        coefficient = work.make("Mul", (_value(child) for child in tree.children
                                        if not child.dependent))
        return work.make("Mul", (coefficient, p)), work.make("Mul", (coefficient, q))
    if tree.native.is_pow:
        exponent = tree.children[1].native
        if exponent.is_integer and exponent.exact_numerator() == 0:
            return _ONE, _ZERO
        if exponent.is_integer and exponent.exact_numerator() == 1:
            return _decompose(tree.children[0], work)
    raise _UnsupportedShape


def _decompose_equation(trees, work):
    left, right = (_decompose(tree, work) for tree in trees)
    return (work.make("Add", (left[0], work.negative(right[0]))),
            work.make("Add", (left[1], work.negative(right[1]))))


def _replay(tree, replacement, work):
    """Independent structural replacement, without coefficient extraction."""
    work.charge()
    if tree.root:
        return _value(tree.children[0]) if replacement is None else replacement
    if not tree.dependent:
        return _value(tree)
    return work.make(tree.native.func_name,
                     (_replay(child, replacement, work) for child in tree.children),
                     prune=False)


def _verify_shadow(tree, candidate, work):
    """The domain shadow changes only sqrt(R) into its original rational R."""
    work.charge()
    if tree.root:
        if candidate != tree.children[0].native:
            raise ValueError("principal-square-root domain shadow changed its original radicand")
        return
    if not tree.dependent:
        if candidate != tree.native:
            raise ValueError("principal-square-root domain shadow changed an original operand")
        return
    if candidate.func_name != tree.native.func_name:
        raise ValueError("principal-square-root domain shadow changed an original operator")
    children = candidate.args
    if len(children) != len(tree.children):
        raise ValueError("principal-square-root domain shadow changed its original arity")
    for original, child in zip(tree.children, children):
        _verify_shadow(original, child, work)


def _verify_coefficient_domains(trees, parts, variable, work):
    """A proposed coefficient cannot invent a pole absent from the source."""
    allowed = []
    pending = list(trees)
    while pending:
        tree = pending.pop()
        work.charge()
        if tree.native.is_pow and tree.children[1].native.is_integer:
            if tree.children[1].native.exact_numerator() < 0:
                if tree.children[0].dependent:
                    raise ValueError("a radical denominator reached rational coefficient verification")
                allowed.append(tree.children[0].native)
        pending.extend(tree.children)
    coordinate = _native_expr(variable)
    for part in parts:
        if type(part) is not _Value or type(part.native) is not _native.Expr:
            raise ValueError("principal-square-root coefficients require immutable native expressions")
        if not _native.real_constraint_source_preflight_native(part.native):
            raise ValueError("principal-square-root coefficients are not rational expressions")
        pending = [part.native]
        while pending:
            node = pending.pop()
            work.charge()
            if node.is_symbol and node != coordinate:
                raise ValueError("principal-square-root coefficient introduced another coordinate")
            children = node.args
            if node.is_pow and children[1].exact_numerator() < 0:
                base = children[0]
                work.charge(len(allowed) + 1)
                if not any(base == original for original in allowed):
                    raise ValueError("principal-square-root coefficient introduced an original-domain hole")
            pending.extend(children)


def _verify_decomposition(trees, parts, variable, work):
    """Check the original equation as an identity in an independent formal u."""
    if type(parts) is not tuple or len(parts) != 2:
        raise ValueError("principal-square-root decomposition must contain two coefficients")
    _verify_coefficient_domains(trees, parts, variable, work)
    parameter = Dummy("_fsym_radical_value")
    formal = _Value(_native_expr(parameter))
    left, right = (_replay(tree, formal, work) for tree in trees)
    p, q = parts
    difference = work.make("Add", (
        left, work.negative(right), work.negative(p),
        work.negative(work.make("Mul", (q, formal))),
    ))
    if rational_zero(_wrap(difference.native)) is not True:
        raise ValueError("principal-square-root coefficients do not reconstruct the original equation")


def solve_radical_real(expression, variable, domain):
    """Complete bounded real sets; unsupported radical grammars stay conditional."""
    if not (_is_exact(domain, _SET_TYPES) or type(domain) is Reals):
        return None
    capture = _RadicalCapture()
    sides = capture.sides(expression) if type(expression) is Eq else (expression, Integer(0))
    natives = tuple(_native_expr(side) for side in sides)
    if not _contains_square_root(natives):
        return None
    capture.unchanged()
    for native in natives:
        _native.real_constraint_source_preflight_native(native)
    trees = tuple(capture.read(side) for side in sides)
    source_symbols = tuple(capture.symbols)
    if variable is None:
        if len(source_symbols) > 1:
            raise ValueError("at least one solve variable is required")
        variable = source_symbols[0] if source_symbols else Dummy("_fsym_radical_equation")
    if type(variable) is not Symbol and type(variable) is not Dummy:
        raise TypeError("principal-square-root solving requires an exact Symbol")
    capture.symbol(variable)
    snapshot = _Domain(domain)
    if not snapshot.real:
        return None

    def unchanged():
        capture.unchanged()
        snapshot.unchanged()

    def conditional():
        unchanged()
        condition = expression if type(expression) is Eq else Eq(expression, 0, evaluate=False)
        result = ConditionSet(variable, condition, domain)
        unchanged()
        return result

    unchanged()
    if not capture.supported or any(symbol != variable for symbol in source_symbols):
        return conditional()
    work = _Work()
    try:
        parts = _decompose_equation(trees, work)
    except _UnsupportedShape:
        return conditional()
    _verify_decomposition(trees, parts, variable, work)
    unchanged()
    shadows = [_replay(tree, None, work) for tree in trees]
    for tree, shadow in zip(trees, shadows):
        if not _native.real_constraint_source_preflight_native(shadow.native):
            raise ValueError("principal-square-root domain shadow is not rational")
        _verify_shadow(tree, shadow.native, work)
    p, q = parts
    r = _value(capture.radicand)
    residual = work.make("Add", (work.square(p),
                                work.negative(work.make("Mul", (work.square(q), r)))))
    sign = work.make("Mul", (p, q))
    # Multiplication by zero is deliberately held. Native source admission
    # retains poles even when P or Q is zero, or the squared equation is an
    # identity. A square root of a finite negative radicand remains defined.
    defined = [work.make("Mul", (_ZERO, shadow), prune=False) for shadow in shadows]
    unchanged()
    checked = _native.solve_real_rational_constraints_native(
        [value.native for value in (residual, sign, *defined)],
        _symbol_binding(variable), ["==", "<=", "==", "=="],
    )
    if checked is None:
        raise NotImplementedError("an admitted principal-square-root equation could not be completed")
    unchanged()
    result = _lift_constraint_chart(checked, variable)
    unchanged()
    result = _restrict(domain, result)
    _checked_domain(result)
    unchanged()
    return result
