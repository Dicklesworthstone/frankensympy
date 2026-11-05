"""Complete bounded rational Piecewise equations on exact real domains.

Every branch is selected by its original first-match condition, before solving
that branch's expression. An undefined selected expression never falls through
to a later branch. Unsupported guards and reachable undefined guard points keep
the entire original query conditional, while admitted operational errors escape.
"""

from ..core import (
    Add, Basic, Dummy, Eq, Ge, Gt, Le, Lt, Mul, Ne, Pow, Relational, Symbol,
    _native, _native_expr, _require_closed_function_class, _symbol_binding, _wrap,
)
from ..functions.elementary.piecewise import Piecewise
from ..logic.boolalg import (
    And, BooleanFalse, BooleanFunction, BooleanTrue, Equivalent, Implies, Not,
    Or, Xor,
)
from ..sets import (
    Complement, ConditionSet, EmptySet, FiniteSet, Intersection, Interval, Reals,
    Set, Union,
)
from ._real_domain import (
    _ATOMS, _Domain, _SET_TYPES, _is_exact, _rational_source, _slot,
    solve_in_real_domain,
)


_ABSENT = object()
_RELATIONS = ((Eq, "=="), (Ne, "!="), (Lt, "<"), (Le, "<="),
              (Gt, ">"), (Ge, ">="))


class _Capture:
    """Snapshot bounded original slots without executing foreign protocols."""

    def __init__(self):
        self.nodes = 0
        self.guard_nodes = 0
        self.supported = True
        self.symbols = []
        self.leaves = []
        self.stamps = []

    def touch(self, depth):
        self.nodes += 1
        if self.nodes > 4096 or depth > 64:
            raise ValueError("rational Piecewise source exceeds its bounded envelope")

    def retain(self, value, owner, name):
        slot = _slot(owner, name)
        try:
            stored = slot.__get__(value, owner)
        except AttributeError:
            stored = _ABSENT
        self.stamps.append((value, slot, stored))
        return stored

    def basic(self, value):
        for name in ("_value", "_struct_args", "_surface_args"):
            self.retain(value, Basic, name)

    def symbol(self, value):
        if value not in self.symbols:
            self.symbols.append(value)

    def scalar(self, source, depth):
        pending = [(source, depth)]
        while pending:
            node, level = pending.pop()
            self.touch(level)
            kind = type(node)
            if kind is Symbol or kind is Dummy:
                self.basic(node)
                self.symbol(node)
                continue
            if _is_exact(node, _ATOMS):
                self.basic(node)
                continue
            if _is_exact(node, (Add, Mul, Pow)):
                self.basic(node)
                children = node.args
                if len(children) > 256 or len(pending) + len(children) > 4096:
                    raise ValueError("rational Piecewise source exceeds its bounded envelope")
                pending.extend((child, level + 1) for child in children)
                continue
            mro = type.__dict__["__mro__"].__get__(kind)
            if any(base is Basic for base in mro):
                self.basic(node)
            elif kind is int:
                if node.bit_length() > 4096:
                    raise ValueError("rational Piecewise source exceeds its coefficient limit")
            elif kind is not bool and kind is not float and kind is not _native.Expr:
                raise TypeError("rational Piecewise requires built-in scalar sources")
            # The closed lowering gate rejects custom classes before any
            # foreign args, conversion, arithmetic, or printing protocol.
            native = _native_expr(node)
            native_pending = [(native, level)]
            while native_pending:
                item, native_depth = native_pending.pop()
                self.touch(native_depth)
                if item.is_symbol:
                    self.symbol(_wrap(item))
                children = item.args
                if len(children) > 256 or len(native_pending) + len(children) > 4096:
                    raise ValueError("rational Piecewise source exceeds its bounded envelope")
                native_pending.extend((child, native_depth + 1) for child in children)
        index = len(self.leaves)
        self.leaves.append(source)
        return index

    def sides(self, relation):
        self.basic(relation)
        values = self.retain(relation, Relational, "_relational_sides")
        if type(values) is not tuple or len(values) != 2:
            raise NotImplementedError("rational Piecewise requires retained relational sides")
        return values

    def condition(self, value, depth=0):
        self.touch(depth)
        self.guard_nodes += 1
        if self.guard_nodes > 64:
            raise ValueError("rational Piecewise has too many guard nodes")
        kind = type(value)
        if value is True or kind is BooleanTrue:
            return ("true",)
        if value is False or kind is BooleanFalse:
            return ("false",)
        operation = next((op for cls, op in _RELATIONS if kind is cls), None)
        if operation is not None:
            left, right = self.sides(value)
            return ("relation", operation, self.scalar(left, depth + 1),
                    self.scalar(right, depth + 1))
        if _is_exact(value, (And, Or, Not, Xor, Implies, Equivalent)):
            self.basic(value)
            children = self.retain(value, BooleanFunction, "_args")
            if type(children) is not tuple or len(children) > 64:
                raise ValueError("invalid bounded rational Piecewise Boolean guard")
            operation = next((name for cls, name in ((And, "and"), (Or, "or"), (Not, "not"))
                              if kind is cls), "unknown")
            if operation == "unknown":
                self.supported = False
            if kind is Not and len(children) != 1:
                raise ValueError("invalid rational Piecewise Not guard")
            return (operation, tuple(self.condition(child, depth + 1) for child in children))
        # A symbolic Boolean (even Symbol("True")) is not a proved condition.
        # Capture its typed symbols, and still reject any opaque source.
        self.scalar(value, depth + 1)
        self.supported = False
        return ("unknown",)

    def freeze(self, variable):
        if variable is None:
            if len(self.symbols) > 1:
                raise ValueError("at least one solve variable is required")
            variable = self.symbols[0] if self.symbols else Dummy("_fsym_piecewise_equation")
        else:
            # The real-domain gate checks exact type/MRO before _require_symbol.
            variable = _rational_source(0, variable)[1]
        sources = []
        eligible = self.supported
        for source in self.leaves:
            admitted = _rational_source(source, variable)
            if admitted is None:
                eligible = False
                sources.append(None)
            else:
                # All scalar payloads are captured before the first solver or
                # sign-chart call. Later work never rereads the mutable surface.
                sources.append(_wrap(_native_expr(admitted[0])))
        return variable, sources, eligible

    def unchanged(self):
        _require_closed_function_class(Piecewise)
        for value, slot, previous in self.stamps:
            if _slot(slot.__objclass__, slot.__name__) is not slot:
                raise RuntimeError("Piecewise source storage changed during exact solving")
            try:
                current = slot.__get__(value, slot.__objclass__)
            except AttributeError:
                current = _ABSENT
            if current is not previous:
                # Native lifting refreshes the cached handle on an interned
                # plain Symbol. Its complete typed atom must remain equal;
                # every retained structural slot still requires identity.
                if (type(value) is Symbol and slot.__objclass__ is Basic
                        and slot.__name__ == "_value"
                        and type(current) is _native.Expr and type(previous) is _native.Expr
                        and current.is_symbol and previous.is_symbol
                        and current == previous):
                    continue
                raise RuntimeError("Piecewise source mutated during exact solving")


def _checked_domain(domain):
    snapshot = _Domain(domain)
    if not snapshot.real:
        raise NotImplementedError("rational Piecewise requires an exact real domain")
    return snapshot


def _restrict(domain, selected, exclude=False):
    """Compose branch sets without discarding proved algebraic aliases."""
    source = _checked_domain(domain)
    target = _checked_domain(selected)
    if source.tree[0] == "EmptySet":
        return EmptySet()
    if source.tree[0] == "FiniteSet":
        known, unknown = [], []
        for native, descriptor in zip(source.native.elements, source.tree[1]):
            inside = target.contains(descriptor)
            value = _wrap(native)
            if inside is None:
                unknown.append(value)
            elif inside is not exclude:
                known.append(value)
        result = FiniteSet(*known)
        if unknown:
            uncertain = FiniteSet(*unknown)
            part = (Complement(uncertain, selected) if exclude
                    else Intersection(uncertain, selected))
            result = Union(result, part)
        return result
    if exclude and target.tree[0] == "FiniteSet":
        # Reuse the original endpoint/member spellings as exclusions as well
        # as the canonical roots, including differently named bound symbols.
        return source.exclude(list(zip(target.tree[1], target.native.elements)))
    if target.tree[0] == "FiniteSet":
        known, unknown = [], []
        for native, descriptor in zip(target.native.elements, target.tree[1]):
            inside = source.contains(descriptor)
            if inside is not False:
                (known if inside is True else unknown).append(_wrap(native))
        result = FiniteSet(*known)
        return (Union(result, Intersection(domain, FiniteSet(*unknown)))
                if unknown else result)
    return Complement(domain, selected) if exclude else Intersection(domain, selected)


def _solve(source, variable, domain):
    if type(domain) is Reals:
        # Keep the public unrestricted-Reals refusal policy separate. This
        # admitted Piecewise operation must propagate every native failure.
        domain = object.__new__(Interval)
        _slot(Set, "_native_set").__set__(domain, _slot(Set, "_native_set").__get__(Reals(), Set))
    result = solve_in_real_domain(source, variable, domain)
    if result is None:
        raise NotImplementedError("admitted rational Piecewise branch could not be completed")
    return result


def _sign_set(source, variable, operation):
    from ..polys.polytools import Poly
    from ..polys.rootoftools import CRootOf, _real_radical, real_roots
    from .inequalities import _lift_sign_chart

    checked = _native.solve_real_rational_inequality_native(
        _native_expr(source), _symbol_binding(variable), operation,
    )
    if checked is None:
        raise NotImplementedError("admitted rational Piecewise guard could not be completed")
    polynomial, gaps, points = checked
    count = len(points)
    if (count > 64 or len(gaps) != count + 1
            or any(type(value) is not bool for value in (*gaps, *points))):
        raise ValueError("invalid complete rational Piecewise sign chart")
    roots = []
    if count:
        associated = real_roots(Poly(_wrap(polynomial), variable), multiple=False, radicals=False)
        if (len(associated) != count or any(m != 1 for _, m in associated)
                or len({root for root, _ in associated}) != count):
            raise ValueError("rational Piecewise guard root association is incomplete")
        coefficients = tuple(_native.real_root_coefficients_native(polynomial, _symbol_binding(variable)))
        order = _checked_domain(Reals())
        for index, (root, _) in enumerate(associated):
            if not _is_exact(root, _ATOMS) and type(root) is not CRootOf:
                raise ValueError("rational Piecewise guard returned an unsupported root")
            descriptor = order.scalar(_native_expr(root))
            if order.compare(descriptor, ("root", coefficients, index, count)) != 0:
                raise ValueError("rational Piecewise guard root association changed an ordered root")
            explicit = _real_radical(root.poly, root.index) if type(root) is CRootOf else None
            roots.append(root if explicit is None else explicit)
    return _lift_sign_chart(roots, gaps, points)


def _guard(condition, sources, variable, remaining):
    kind = condition[0]
    if kind == "true":
        return remaining
    if kind == "false":
        return EmptySet()
    if kind == "unknown":
        return None
    if kind == "relation":
        operation, left, right = condition[1:]
        source = Add(sources[left], Mul(-1, sources[right], evaluate=False), evaluate=False)
        defined = _solve(Mul(0, source, evaluate=False), variable, remaining)
        # An undefined or undecidably reached guard is not Boolean false.
        # Retain the whole original query rather than publish a partial set.
        if defined != remaining:
            return None
        if operation in ("==", "!="):
            zero = _solve(source, variable, remaining)
            return _restrict(remaining, zero, operation == "!=")
        return _restrict(remaining, _sign_set(source, variable, operation))
    children = [_guard(child, sources, variable, remaining) for child in condition[1]]
    if any(child is None for child in children):
        return None
    if kind == "not":
        result = _restrict(remaining, children[0], True)
    elif kind == "and":
        result = _restrict(remaining, Intersection(*children))
    else:
        result = _restrict(remaining, Union(*children))
    _checked_domain(result)
    return result


def solve_piecewise_real(expression, variable, domain):
    """Solve a top-level Piecewise, or one side of an original equality."""
    if not (_is_exact(domain, _SET_TYPES) or type(domain) is Reals):
        return None
    capture = _Capture()
    other, reverse = None, False
    if type(expression) is Eq:
        left, right = capture.sides(expression)
        if type(left) is Piecewise and type(right) is not Piecewise:
            piecewise, other = left, right
        elif type(right) is Piecewise and type(left) is not Piecewise:
            piecewise, other, reverse = right, left, True
        else:
            return None
    elif type(expression) is Piecewise:
        piecewise = expression
    else:
        return None
    _require_closed_function_class(Piecewise)
    capture.basic(piecewise)
    branches = capture.retain(piecewise, Piecewise, "_piecewise_branches")
    if type(branches) is not tuple or not 1 <= len(branches) <= 16:
        raise ValueError("rational Piecewise requires one to sixteen retained branches")
    common = capture.scalar(other, 0) if other is not None else None
    prepared = []
    for pair in branches:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("invalid retained Piecewise branch")
        prepared.append((capture.scalar(pair[0], 0), capture.condition(pair[1])))
    variable, sources, eligible = capture.freeze(variable)
    original_domain = _Domain(domain)
    if not original_domain.real:
        return None

    def conditional():
        capture.unchanged()
        original_domain.unchanged()
        condition = expression if type(expression) is Eq else Eq(expression, 0, evaluate=False)
        return ConditionSet(variable, condition, domain)

    if not eligible:
        return conditional()
    remaining, solutions = domain, []
    for branch, condition in prepared:
        if _checked_domain(remaining).tree[0] == "EmptySet":
            break
        selected = _guard(condition, sources, variable, remaining)
        if selected is None:
            return conditional()
        selected_domain = _checked_domain(selected)
        source = sources[branch]
        if common is not None:
            left, right = (sources[common], source) if reverse else (source, sources[common])
            source = Add(left, Mul(-1, right, evaluate=False), evaluate=False)
        if selected_domain.tree[0] != "EmptySet":
            solutions.append(_solve(source, variable, selected))
        # Subtract the matching condition itself, even when its selected
        # expression has no zeros or is undefined at some selected points.
        remaining = _restrict(remaining, selected, True)
        _checked_domain(remaining)
    capture.unchanged()
    original_domain.unchanged()
    if all(_is_exact(result, (EmptySet, FiniteSet)) for result in solutions):
        # Generic set union deliberately leaves some algebraic finite sets
        # structural. These checked branch sets are already complete, so join
        # their explicit members directly without requesting another solver.
        values = []
        for result in solutions:
            if type(result) is FiniteSet:
                native = _slot(Set, "_native_set").__get__(result, Set)
                values.extend(_wrap(value) for value in native.elements)
        return FiniteSet(*values)
    return Union(*solutions)
