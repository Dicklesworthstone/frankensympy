"""Checked rational equations restricted to closed real set expressions.

The equation engine proves the complete finite/cofinite real solution set.
This boundary snapshots a bounded built-in domain and filters its roots with
the independent indexed-real-root comparator. Unknown membership remains an
exact intersection; neither a float comparison nor Python truth coercion can
discard a possible root. The established unrestricted Reals route is separate.
"""

from fractions import Fraction
from types import MemberDescriptorType

from ..core import (
    Add, Basic, Dummy, Eq, Expr, Half, Integer, Mul, NegativeOne, One, Pow,
    Rational, Symbol, Zero, _native, _native_expr, _require_closed_function_class,
    _require_symbol, _symbol_binding, _wrap,
)
from ..sets import (
    Complement, EmptySet, FiniteSet, Intersection, Interval, Set, Union,
)


_ATOMS = (Integer, Rational, Zero, One, NegativeOne, Half)
_SET_TYPES = (EmptySet, FiniteSet, Interval, Union, Intersection, Complement)
_NEG_INF = _native.Expr("-oo")
_POS_INF = _native.Expr("oo")


def _is_exact(value, classes):
    return any(type(value) is cls for cls in classes)


def _slot(owner, name):
    namespace = type.__dict__["__dict__"].__get__(owner)
    slot = namespace.get(name)
    if (type(slot) is not MemberDescriptorType or slot.__objclass__ is not owner
            or slot.__name__ != name):
        raise NotImplementedError("exact real solving requires original storage descriptors")
    return slot


class _Domain:
    """One immutable native domain with bounded scalar/comparison caches."""

    def __init__(self, source):
        self.source = source
        self.slot = _slot(Set, "_native_set")
        self.native = self.slot.__get__(source, Set)
        self.nodes = 0
        self.scalar_nodes = 0
        self.expression_nodes = 0
        self.coefficient_steps = 0
        self.comparisons = 0
        self.scalars = {}
        self.orders = {}
        self.domain_values = []
        if type(self.native) is not _native.SymSet:
            self.tree, self.real = None, False
        else:
            # Borrowed native traversal checks the complete stored graph
            # before any set/endpoint getter clones a child into Python.
            self.native.real_domain_preflight()
            self.tree, self.real = self._read(self.native, 0)

    def unchanged(self):
        if self.slot.__get__(self.source, Set) is not self.native:
            raise RuntimeError("solveset domain mutated during exact evaluation")

    def _read(self, native, depth):
        self.nodes += 1
        if self.nodes > 128 or depth > 16:
            raise ValueError("exact real solveset domain exceeds its bounded set envelope")
        kind = native.kind
        if kind == "EmptySet":
            return (kind,), True
        if kind == "UniversalSet":
            return (kind,), False
        if kind == "Interval":
            start, end = native.start, native.end
            lower, upper = self.scalar(start), self.scalar(end)
            self.domain_values.extend(((start, lower), (end, upper)))
            order = self.compare(lower, upper)
            if order is not None and (order > 0 or order == 0 and (native.left_open or native.right_open)):
                return ("EmptySet",), True
            return (kind, lower, upper, native.left_open, native.right_open), True
        if kind == "FiniteSet":
            elements = native.elements
            if len(elements) > 128:
                raise ValueError("exact real solveset domain has too many finite points")
            values = tuple(self.scalar(value) for value in elements)
            self.domain_values.extend(zip(elements, values))
            real = all(value is not None and value[0] != "infinity" for value in values)
            return (kind, values), real
        if kind not in ("Union", "Intersection", "Complement"):
            raise NotImplementedError("unsupported exact real set expression")
        children = native.args
        if len(children) > 128 - self.nodes:
            raise ValueError("exact real solveset domain exceeds its bounded set envelope")
        parts = [self._read(child, depth + 1) for child in children]
        if kind == "Complement":
            if len(parts) != 1:
                raise ValueError("invalid native complement domain")
            real = False  # Native complements are relative to the universe.
        elif kind == "Intersection":
            real = any(part[1] for part in parts)
        else:
            real = all(part[1] for part in parts)
        return (kind, tuple(part[0] for part in parts)), real

    def scalar(self, native):
        if type(native) is not _native.Expr:
            raise ValueError("invalid native real-domain endpoint")
        # Inspect the structural payload before hashing, lifting, or invoking
        # a root algorithm. In particular, an unknown endpoint remains an
        # unknown expression rather than becoming an invented Python class.
        pending = [(native, 0)]
        while pending:
            item, depth = pending.pop()
            self.expression_nodes += 1
            if self.expression_nodes > 16384 or depth > 64:
                raise ValueError("exact real solveset domain exceeds its scalar source envelope")
            children = item.args
            if len(children) > 256:
                raise ValueError("exact real solveset endpoint has too many operands")
            pending.extend((child, depth + 1) for child in children)
        if native in self.scalars:
            return self.scalars[native]
        self.scalar_nodes += 1
        if self.scalar_nodes > 256:
            raise ValueError("exact real solveset domain has too many scalar endpoints")
        if native.is_integer or native.is_rational:
            numerator, denominator = native.exact_numerator(), native.exact_denominator()
            if max(numerator.bit_length(), denominator.bit_length()) > 4096:
                raise ValueError("exact real solveset endpoint exceeds the coefficient limit")
            result = ("rational", Fraction(numerator, denominator))
        elif native == _NEG_INF or native == _POS_INF:
            result = ("infinity", -1 if native == _NEG_INF else 1)
        elif native.func_name == "CRootOf":
            result = self._indexed_root(native)
        elif native.is_add or native.is_mul or native.is_pow:
            result = self._arithmetic_scalar(native)
        else:
            # Intervals are real domains even when their exact endpoints are
            # symbolic. Such membership must stay conditional. In a finite
            # set an unknown element does not establish a real-only domain.
            result = None
        self.scalars[native] = result
        return result

    def _fraction(self, value):
        self.coefficient_steps += 1
        if (self.coefficient_steps > 32768
                or max(value.numerator.bit_length(), value.denominator.bit_length()) > 4096):
            raise ValueError("exact real solveset endpoint arithmetic exceeds its limit")
        return value

    def _arithmetic_scalar(self, native):
        children = [self.scalar(child) for child in native.args]
        if any(child is None or child[0] == "infinity" for child in children):
            return None
        if native.is_pow:
            if len(children) != 2 or any(child[0] != "rational" for child in children):
                return None
            base, exponent = (child[1] for child in children)
            if abs(exponent.numerator) > 64 or exponent.denominator > 64:
                raise ValueError("exact real solveset endpoint power exceeds its degree limit")
            if not base:
                if exponent <= 0:
                    raise ValueError("undefined power in exact real solveset domain")
                return ("rational", Fraction(0))
            if exponent.denominator != 1 and base < 0:
                return None  # Principal complex powers are not real roots.
            if max(base.numerator.bit_length(), base.denominator.bit_length()) * abs(exponent.numerator) > 4096:
                raise ValueError("exact real solveset endpoint arithmetic exceeds its limit")
            value = self._fraction(base ** exponent.numerator)
            degree = exponent.denominator
            if degree == 1:
                return ("rational", value)
            coefficients = [Fraction(0)] * (degree + 1)
            coefficients[0], coefficients[-1] = -value, Fraction(1)
            # A positive binomial has one positive real root, and also one
            # negative real root exactly when the degree is even.
            count = 2 if degree % 2 == 0 else 1
            return ("root", tuple(self._native_fraction(c) for c in coefficients), count - 1, count)
        roots = [child for child in children if child[0] == "root"]
        if len(roots) > 1:
            return None
        rationals = [child[1] for child in children if child[0] == "rational"]
        value = Fraction(0 if native.is_add else 1)
        for rational in rationals:
            value = self._fraction(value + rational if native.is_add else value * rational)
        if not roots:
            return ("rational", value)
        return self._affine_root(roots[0], Fraction(1) if native.is_add else value,
                                 value if native.is_add else Fraction(0))

    @staticmethod
    def _native_fraction(value):
        return _native.py_rational(value.numerator, value.denominator)

    def _affine_root(self, root, scale, offset):
        if not scale:
            return ("rational", offset)
        coefficients = [Fraction(c.exact_numerator(), c.exact_denominator()) for c in root[1]]
        # Horner substitution p((z-offset)/scale) independently preserves the
        # complete root set; a negative scale reverses its exact real order.
        result = [coefficients[-1]]
        for coefficient in reversed(coefficients[:-1]):
            expanded = [Fraction(0)] * (len(result) + 1)
            for index, entry in enumerate(result):
                divided = self._fraction(entry / scale)
                expanded[index] = self._fraction(expanded[index] - divided * offset)
                expanded[index + 1] = self._fraction(expanded[index + 1] + divided)
            expanded[0] = self._fraction(expanded[0] + coefficient)
            result = expanded
        index = root[2] if scale > 0 else root[3] - 1 - root[2]
        return ("root", tuple(self._native_fraction(c) for c in result), index, root[3])

    def _indexed_root(self, native):
        from ..polys.rootoftools import CRootOf, _native_coefficients

        _require_closed_function_class(CRootOf)
        children = native.args
        if len(children) != 2 or not children[1].is_integer:
            raise ValueError("invalid indexed real-domain endpoint")
        index = children[1].exact_numerator()
        if not 0 <= index < 64:
            raise ValueError("indexed real-domain endpoint exceeds the root limit")
        bindings = children[0].free_symbol_bindings
        if len(bindings) != 1:
            raise ValueError("indexed real-domain endpoint requires one polynomial coordinate")
        coefficients = _native.real_root_coefficients_native(children[0], bindings[0])
        intervals = _native.real_roots_isolate_native(coefficients)
        if index >= sum(multiplicity for _, _, multiplicity in intervals):
            return None
        # Canonicalize the factor and its local index with the checked root
        # association route. A raw CRootOf tag must not confuse an index among
        # repeated roots with the comparator's distinct irreducible-root index.
        root = CRootOf(_wrap(children[0]), index, radicals=False)
        if type(root) is not CRootOf:
            return self.scalar(_native_expr(root))
        values = _native_coefficients(root.poly)
        count = sum(multiplicity for _, _, multiplicity in _native.real_roots_isolate_native(values))
        return ("root", tuple(values), root.index, count)

    def compare(self, left, right):
        if left is None or right is None:
            return None
        if left == right:
            return 0
        if left[0] == "infinity":
            return left[1]
        if right[0] == "infinity":
            return -right[1]
        if left[0] == right[0] == "rational":
            return (left[1] > right[1]) - (left[1] < right[1])
        key = (left, right)
        if key in self.orders:
            return self.orders[key]
        if self.comparisons == 256:
            raise ValueError("exact real solveset exceeds its root-comparison limit")
        self.comparisons += 1

        def descriptor(value):
            if value[0] == "root":
                return list(value[1]), value[2]
            fraction = value[1]
            constant = _native.py_rational(-fraction.numerator, fraction.denominator)
            return [constant, _native.py_integer(1)], 0

        lp, li = descriptor(left)
        rp, ri = descriptor(right)
        order = _native.compare_real_roots_native(lp, li, rp, ri)
        if type(order) is not int or order not in (-1, 0, 1):
            raise ValueError("invalid checked real-root ordering")
        self.orders[key] = order
        self.orders[(right, left)] = -order
        return order

    def contains(self, value, tree=None):
        tree = self.tree if tree is None else tree
        kind = tree[0]
        if kind == "EmptySet":
            return False
        if kind == "UniversalSet":
            return True
        if kind == "Interval":
            lower, upper = self.compare(value, tree[1]), self.compare(value, tree[2])
            if lower is not None and (lower < 0 or lower == 0 and tree[3]):
                return False
            if upper is not None and (upper > 0 or upper == 0 and tree[4]):
                return False
            return True if lower is not None and upper is not None else None
        if kind == "FiniteSet":
            orders = [self.compare(value, member) for member in tree[1]]
            if any(order == 0 for order in orders):
                return True
            return None if any(order is None for order in orders) else False
        if kind == "Complement":
            result = self.contains(value, tree[1][0])
            return None if result is None else not result
        results = [self.contains(value, part) for part in tree[1]]
        decisive = kind == "Union"
        if any(result is decisive for result in results):
            return decisive
        return None if any(result is None for result in results) else not decisive

    def exclude(self, excluded):
        """Apply proved exclusions without losing algebraic equality in sets.

        Native symbolic set normalization may not recognize CRootOf and its
        equivalent radical, or roots whose bound polynomial names differ.
        A finite domain is filtered directly. Other domains retain all matching
        original endpoint/point spellings as explicit exclusions, so a proved
        hole cannot turn back into undecidable membership during publication.
        """
        if self.tree[0] == "EmptySet":
            return EmptySet()
        if self.tree[0] == "FiniteSet":
            values = self.native.elements
            retained = []
            for native, descriptor in zip(values, self.tree[1]):
                if not any(self.compare(descriptor, hole) == 0 for hole, _ in excluded):
                    retained.append(_wrap(native))
            return FiniteSet(*retained)
        if self.tree[0] == "Interval" and self.compare(self.tree[1], self.tree[2]) == 0:
            if any(self.compare(self.tree[1], hole) == 0 for hole, _ in excluded):
                return EmptySet()
        removed = [native for _, native in excluded]
        for native, descriptor in self.domain_values:
            if (descriptor is not None and native not in removed
                    and any(self.compare(descriptor, hole) == 0 for hole, _ in excluded)):
                removed.append(native)
        return Complement(self.source, FiniteSet(*(_wrap(value) for value in removed)))


def _rational_source(expression, variable):
    """Admit the original surface before subtraction, expansion, or lowering."""
    if type(expression) is bool:
        expression = Integer(int(expression))
    elif type(expression) is int:
        if expression.bit_length() > 4096:
            raise ValueError("exact real solveset source exceeds its coefficient limit")
        expression = Integer(expression)
    if type(expression) is Expr or type(expression) is _native.Expr:
        native = _native_expr(expression)
        pending = [(native, 0)]
        visited = 0
        while pending:
            node, depth = pending.pop()
            visited += 1
            if visited > 2048 or depth > 64:
                raise ValueError("exact real solveset source exceeds its bounded envelope")
            if node.func_name not in ("Integer", "Rational", "Symbol", "Add", "Mul", "Pow"):
                return None
            children = node.args
            if node.is_pow and not children[1].is_integer:
                return None
            if len(children) > 256 or len(pending) + len(children) > 2048:
                raise ValueError("exact real solveset source exceeds its bounded envelope")
            pending.extend((child, depth + 1) for child in children)
        expression = _wrap(native)
    pending = [(expression, 0)]
    variables = []
    visited = 0
    eligible = True
    while pending:
        node, depth = pending.pop()
        visited += 1
        if visited > 2048 or depth > 64:
            raise ValueError("exact real solveset source exceeds its bounded envelope")
        if _is_exact(node, _ATOMS):
            continue
        if type(node) is Symbol or type(node) is Dummy:
            if node not in variables:
                variables.append(node)
            continue
        if type(node) is Eq and node is expression:
            children = node.args
        elif _is_exact(node, (Add, Mul, Pow)):
            children = node.args
            if type(node) is Pow and not _is_exact(children[1], (Integer, Zero, One, NegativeOne)):
                eligible = False
        else:
            mro = type.__dict__["__mro__"].__get__(type(node))
            if any(base is Basic for base in mro):
                # Closed nonrational built-ins may use the existing public
                # lane. Opaque/custom sources must refuse before Eq's ordinary
                # subtraction could execute or erase their Python behavior.
                _native_expr(node)
            elif type(node) is not float and type(node) is not str:
                raise TypeError("exact real solveset requires built-in scalar sources")
            eligible = False
            continue
        if len(children) > 256 or len(pending) + len(children) > 2048:
            raise ValueError("exact real solveset source exceeds its bounded envelope")
        pending.extend((child, depth + 1) for child in children)
    if not eligible:
        return None
    if variable is None:
        if len(variables) > 1:
            raise ValueError("at least one solve variable is required")
        variable = variables[0] if variables else Dummy("_fsym_real_equation")
    else:
        if type(variable) is not Symbol and type(variable) is not Dummy:
            mro = type.__dict__["__mro__"].__get__(type(variable))
            if any(base is Symbol for base in mro):
                raise NotImplementedError("custom Symbol subclasses require a supervised override lane")
            raise TypeError("exact real solveset variable must be a Symbol")
        variable = _require_symbol(variable)
    if any(symbol != variable for symbol in variables):
        return None
    if type(expression) is Eq:
        expression = Add(expression.lhs, Mul(-1, expression.rhs, evaluate=False), evaluate=False)
    return expression, variable


def solve_in_real_domain(expression, variable, domain):
    """Complete rational solutions in an admitted restricted real domain.

    None means the source/domain grammar belongs to another public lane. Once
    admitted, native, association, ordering, and resource failures propagate;
    they do not trigger another algorithm or become a mathematical condition.
    """
    if not _is_exact(domain, _SET_TYPES):
        return None
    source = _rational_source(expression, variable)
    if source is None:
        return None
    snapshot = _Domain(domain)
    if not snapshot.real:
        return None
    expression, variable = source
    checked = _native.solve_real_rational_native(
        _native_expr(expression), _symbol_binding(variable),
    )
    if checked is None:
        raise NotImplementedError("unsupported typed rational source in restricted real solveset")
    complement, polynomial, count = checked
    if type(complement) is not bool or type(count) is not int or not 0 <= count <= 64:
        raise ValueError("invalid complete real equation result")
    if not count:
        result = domain if complement and snapshot.tree[0] != "EmptySet" else EmptySet()
    else:
        from ..polys.polytools import Poly
        from ..polys.rootoftools import CRootOf, _real_radical, real_roots

        roots = real_roots(Poly(_wrap(polynomial), variable), multiple=False, radicals=False)
        if len(roots) != count or any(multiplicity != 1 for _, multiplicity in roots):
            raise ValueError("real equation root association is incomplete")
        if len({root for root, _ in roots}) != count:
            raise ValueError("distinct real roots lost during association")
        coefficients = tuple(_native.real_root_coefficients_native(
            polynomial, _symbol_binding(variable),
        ))
        certain, undecided, exclusions = [], [], []
        for index, (root, _) in enumerate(roots):
            if not _is_exact(root, _ATOMS) and type(root) is not CRootOf:
                raise ValueError("real equation association returned an unsupported root")
            value_descriptor = snapshot.scalar(_native_expr(root))
            if value_descriptor is None or value_descriptor[0] == "infinity":
                raise ValueError("real equation association returned a nonreal root")
            if snapshot.compare(value_descriptor, ("root", coefficients, index, count)) != 0:
                raise ValueError("real equation root association changed an ordered root")
            membership = snapshot.contains(value_descriptor)
            if membership is False:
                continue
            explicit = _real_radical(root.poly, root.index) if type(root) is CRootOf else None
            value = root if explicit is None else explicit
            (certain if membership is True else undecided).append(value)
            exclusions.append((value_descriptor, _native_expr(value)))
        if complement:
            result = snapshot.exclude(exclusions)
        else:
            known = FiniteSet(*certain)
            result = (Union(known, Intersection(FiniteSet(*undecided), domain))
                      if undecided else known)
    snapshot.unchanged()
    return result
