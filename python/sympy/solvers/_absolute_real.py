"""Complete bounded real solving for rational expressions containing Abs.

Each absolute value contributes the disjoint cases ``argument >= 0`` and
``argument < 0``. Every case is sent, together with the original equation
or inequality, to the independently checked rational constraint engine.
Nested absolute values use the same case assignment throughout a branch.
Held arithmetic preserves poles under cancellation, zero factors and powers.
"""

from itertools import product

from ..core import (
    Abs, Add, Basic, Dummy, Eq, Expr, Function, Ge, Gt, Half, Integer, Le, Lt,
    Mul, Ne, NegativeOne, One, Pow, Symbol, Zero, _charge_nested_if_active,
    _native, _native_expr,
    _require_closed_function_class, _surface_binding_key, _symbol_binding, _wrap,
)
from ..sets import ConditionSet, EmptySet, FiniteSet, Reals, Set, Union
from ._piecewise_real import _ABSENT, _Capture, _checked_domain, _restrict
from ._real_domain import _ATOMS, _Domain, _SET_TYPES, _is_exact, _slot
from .inequalities import _constraint_owners_unchanged, _lift_constraint_chart


_MAX_ABSOLUTE_VALUES = 4
_MAX_REBUILT_NODES = 65_536
_RELATIONS = ((Eq, "=="), (Ne, "!="), (Lt, "<"), (Le, "<="),
              (Gt, ">"), (Ge, ">="))


class _AbsoluteCapture(_Capture):
    """Bind the retained Python tree to immutable, typed native operands."""

    def __init__(self):
        super().__init__()
        self.absolute_values = []
        self.absolute_sources = []

    def touch(self, depth):
        super().touch(depth)
        _charge_nested_if_active(1)

    def unchanged(self):
        _constraint_owners_unchanged()
        _require_closed_function_class(Abs)
        super().unchanged()

    def symbol(self, value):
        super().symbol(value)
        native = _native_expr(value)
        if (not native.is_symbol or _surface_binding_key(native.symbol_binding)
                != _surface_binding_key(_symbol_binding(value))):
            raise ValueError("absolute-value source has inconsistent typed symbol assumptions")

    def read(self, source, depth=0):
        self.touch(depth)
        kind = type(source)
        if kind is int:
            source, kind = Integer(source), Integer
        native = _native_expr(source)
        if kind is _native.Expr:
            return self.read_native(native, depth)
        if kind is Symbol or kind is Dummy:
            self.symbol(source)
            return ("atom", native)
        if _is_exact(source, _ATOMS):
            self.basic(source)
            valid = (native.is_integer if kind in (Integer, Zero, One, NegativeOne)
                     else native.is_integer or native.is_rational)
            if not valid:
                raise ValueError("absolute-value scalar owner does not match its native value")
            fixed = {Zero: (0, 1), One: (1, 1), NegativeOne: (-1, 1), Half: (1, 2)}.get(kind)
            if fixed is not None and (native.exact_numerator(), native.exact_denominator()) != fixed:
                raise ValueError("absolute-value constant owner does not match its native value")
            return ("atom", native)
        if kind is Expr:
            self.basic(source)
            return self.read_native(native, depth)

        arithmetic = next((tag for cls, tag in ((Add, "Add"), (Mul, "Mul"),
                                                (Pow, "Pow")) if kind is cls), None)
        if arithmetic is not None or kind is Abs:
            retained = self.basic(source)
            owner = Function if kind is Abs else kind
            original = self.retain(source, owner, "_args")
            if original is not _ABSENT:
                retained.append(original)
            groups = []
            for children in retained:
                if children is _ABSENT or children is None:
                    continue
                if type(children) is not tuple or len(children) > 256:
                    raise ValueError("absolute-value source requires bounded retained argument tuples")
                if not any(children is previous for previous in groups):
                    groups.append(children)
            expected = native.args
            correct_owner = (native.func_name in ("Abs", "abs") if kind is Abs
                             else native.func_name == arithmetic)
            if not correct_owner:
                raise ValueError("absolute-value source owner does not match its native operator")
            parsed = None
            for children in groups:
                # Check every spelling, including shadowed surface tuples,
                # before using any candidate or sign chart.
                current = tuple(_native_expr(child) for child in children)
                if (len(current) != len(expected)
                        or any(left != right for left, right in zip(current, expected))):
                    raise ValueError("absolute-value source does not match its retained arguments")
                values = tuple(self.read(child, depth + 1) for child in children)
                if parsed is None:
                    parsed = values
            if parsed is None:
                parsed = tuple(self.read_native(child, depth + 1) for child in expected)
            return self.node(native, parsed)

        # Closed nonrational sources remain conditional. Foreign functions
        # and scalar subclasses have already been refused by closed lowering.
        mro = type.__dict__["__mro__"].__get__(kind)
        if any(base is Basic for base in mro):
            self.basic(source)
            if any(base is Function for base in mro):
                children = self.retain(source, Function, "_args")
                if children is not _ABSENT:
                    if type(children) is not tuple or len(children) > 256:
                        raise ValueError("absolute-value source has malformed function arguments")
                    for child in children:
                        self.read(child, depth + 1)
        self.supported = False
        return self.read_native(native, depth)

    def read_native(self, native, depth):
        self.touch(depth)
        if native.is_symbol:
            self.symbol(_wrap(native))
            return ("atom", native)
        if native.is_integer or native.is_rational:
            return ("atom", native)
        children = native.args
        if len(children) > 256:
            raise ValueError("absolute-value source exceeds its arity bound")
        parsed = tuple(self.read_native(child, depth + 1) for child in children)
        return self.node(native, parsed)

    def node(self, native, children):
        tag = native.func_name
        if tag in ("Abs", "abs") and len(children) == 1:
            for index, previous in enumerate(self.absolute_sources):
                if native == previous:
                    return ("abs", index)
            if len(self.absolute_values) == _MAX_ABSOLUTE_VALUES:
                raise ValueError("exact real solving supports at most four distinct absolute values")
            index = len(self.absolute_values)
            self.absolute_sources.append(native)
            self.absolute_values.append(children[0])
            return ("abs", index)
        if tag in ("Add", "Mul"):
            return (tag, children)
        if tag == "Pow" and len(children) == 2 and native.args[1].is_integer:
            return (tag, children)
        self.supported = False
        return ("unsupported", native)


def _rebuild(tree, signs, absolute_values, work):
    work[0] += 1
    if work[0] > _MAX_REBUILT_NODES:
        raise ValueError("absolute-value solving exceeds its total branch-work envelope")
    _charge_nested_if_active(1)
    tag, value = tree
    if tag == "atom":
        return value
    if tag == "abs":
        argument = _rebuild(absolute_values[value], signs, absolute_values, work)
        if signs[value] > 0:
            return argument
        return _native.Mul(_native.py_integer(-1), argument, evaluate=False).as_expr()
    children = tuple(_rebuild(child, signs, absolute_values, work) for child in value)
    if tag == "Add":
        return _native.Add(*children, evaluate=False).as_expr()
    if tag == "Mul":
        return _native.Mul(*children, evaluate=False).as_expr()
    if tag == "Pow":
        return _native.Pow(*children, evaluate=False).as_expr()
    raise ValueError("unsupported source reached an absolute-value branch")


def _verify_replacement(tree, candidate, signs, absolute_values, work):
    """Check one substitution structurally without calling its constructor.

    This discharges only the branch identity: under the listed exact sign
    guards the rational candidate equals the original Abs expression. Root
    completeness and the sign guards themselves belong to the native checker.
    """
    work[0] += 1
    if work[0] > _MAX_REBUILT_NODES:
        raise ValueError("absolute-value verification exceeds its branch-work envelope")
    _charge_nested_if_active(1)
    if type(candidate) is not _native.Expr:
        raise ValueError("absolute-value branch is not an immutable native expression")
    tag, value = tree
    if tag == "atom":
        if candidate != value:
            raise ValueError("absolute-value branch disagrees with its original source")
        return
    children = candidate.args
    if tag == "abs":
        if signs[value] < 0:
            if (not candidate.is_mul or len(children) != 2 or not children[0].is_integer
                    or children[0].exact_numerator() != -1):
                raise ValueError("absolute-value branch lost its negative sign")
            candidate = children[1]
        _verify_replacement(absolute_values[value], candidate, signs, absolute_values, work)
        return
    if candidate.func_name != tag or len(children) != len(value):
        raise ValueError("absolute-value branch changed an original operator")
    for original, child in zip(value, children):
        _verify_replacement(original, child, signs, absolute_values, work)


def solve_absolute_real(expression, variable, domain):
    """Complete bounded Abs equations/inequalities on a closed real domain.

    None selects another surface grammar. Unsupported Abs expressions retain
    their original condition. Admitted solver/resource failures propagate;
    branches which happened to finish first cannot become partial answers.
    """
    if not (_is_exact(domain, _SET_TYPES) or type(domain) is Reals):
        return None
    operation = next((op for cls, op in _RELATIONS if type(expression) is cls), "==")
    capture = _AbsoluteCapture()
    sides = (capture.sides(expression) if any(type(expression) is cls for cls, _ in _RELATIONS)
             else (expression, Integer(0)))
    natives = tuple(_native_expr(side) for side in sides)
    present = [native.has_function_bounded("Abs") or native.has_function_bounded("abs")
               for native in natives]
    if not any(present):
        return None
    capture.unchanged()
    # Borrowed traversal bounds unsupported nodes before .args clones
    # children. Its grammar result is separate from the broader Abs grammar.
    for native in natives:
        _native.real_constraint_source_preflight_native(native)
    trees = tuple(capture.read(side) for side in sides)
    source_symbols = tuple(capture.symbols)
    if variable is None:
        if len(source_symbols) > 1:
            raise ValueError("at least one solve variable is required")
        variable = source_symbols[0] if source_symbols else Dummy("_fsym_absolute_equation")
    if type(variable) is not Symbol and type(variable) is not Dummy:
        raise TypeError("absolute-value solving requires an exact Symbol")
    capture.symbol(variable)
    binding = _symbol_binding(variable)
    snapshot = _Domain(domain)
    if not snapshot.real:
        return None

    def unchanged():
        capture.unchanged()
        snapshot.unchanged()

    unchanged()
    if not capture.supported or any(symbol != variable for symbol in source_symbols):
        condition = (expression if any(type(expression) is cls for cls, _ in _RELATIONS)
                     else Eq(expression, 0, evaluate=False))
        result = ConditionSet(variable, condition, domain)
        unchanged()
        return result

    results, work, verification_work, covered = [], [0], [0], set()
    for signs in product((1, -1), repeat=len(capture.absolute_values)):
        if (type(signs) is not tuple or len(signs) != len(capture.absolute_values)
                or any(type(sign) is not int or sign not in (1, -1) for sign in signs)
                or signs in covered):
            raise ValueError("absolute-value branch coverage is malformed")
        covered.add(signs)
        guards = [_rebuild(tree, signs, capture.absolute_values, work)
                  for tree in capture.absolute_values]
        left, right = (_rebuild(tree, signs, capture.absolute_values, work) for tree in trees)
        for original, candidate in zip((*capture.absolute_values, *trees), (*guards, left, right)):
            _verify_replacement(original, candidate, signs, capture.absolute_values, verification_work)
        residual = _native.Add(
            left, _native.Mul(_native.py_integer(-1), right, evaluate=False).as_expr(),
            evaluate=False,
        ).as_expr()
        operations = [">=" if sign > 0 else "<" for sign in signs]
        checked = _native.solve_real_rational_constraints_native(
            [*guards, residual], binding, [*operations, operation],
        )
        if checked is None:
            raise NotImplementedError("an admitted absolute-value branch could not be completed")
        branch = _lift_constraint_chart(checked, variable)
        unchanged()
        branch = _restrict(domain, branch)
        _checked_domain(branch)
        results.append(branch)
        unchanged()

    if len(covered) != 1 << len(capture.absolute_values):
        raise ValueError("absolute-value branch coverage is incomplete")
    if all(_is_exact(result, (EmptySet, FiniteSet)) for result in results):
        values = []
        for result in results:
            if type(result) is FiniteSet:
                native = _slot(Set, "_native_set").__get__(result, Set)
                values.extend(_wrap(value) for value in native.elements)
        result = FiniteSet(*values)
    else:
        result = Union(*results)
    _checked_domain(result)
    unchanged()
    return result
