"""Bounded, signed Boolean reasoning for the Python assumptions boundary.

The alternatives below are conjunctions, not a bag of asserted descendants.
Every surviving alternative must support an answer. Keys retain Python surface
structure and typed symbol identity; printed expressions never identify facts.
This is a compatibility inference lane, not a mathematical proof receipt.
"""

from functools import lru_cache

from ..core import (
    Add, Dummy, Mul, Pow, Symbol, _charge_nested_if_active, _contains_opaque,
    _native, _native_expr, _wrap,
)
from ..core._opaque_arithmetic import _key
from .assume import AppliedPredicate, Predicate


NATIVE_PREDICATES = frozenset({
    "complex", "real", "rational", "integer", "algebraic", "transcendental",
    "positive", "negative", "nonnegative", "nonpositive", "zero", "nonzero",
    "prime", "even", "odd", "finite", "infinite",
})
_PROPERTY_PREDICATES = NATIVE_PREDICATES | {
    "imaginary", "irrational", "composite", "extended_real", "commutative",
}
_MAX_WORK = 65_536
_MAX_DEPTH = 64
_MAX_BRANCHES = 128
_MAX_EXPRESSIONS = 128
_MAX_LITERALS = 1024

# These complements require their stated domain. In particular not-zero
# alone does not imply Python Q.nonzero, which denotes a nonzero real value.
_DOMAIN_RULES = (
    (("real", True), ("zero", False), "nonzero"),
    (("real", True), ("nonzero", False), "zero"),
    (("real", True), ("positive", False), "nonpositive"),
    (("real", True), ("negative", False), "nonnegative"),
    (("real", True), ("nonnegative", False), "negative"),
    (("real", True), ("nonpositive", False), "positive"),
    (("nonnegative", True), ("nonzero", True), "positive"),
    (("nonpositive", True), ("nonzero", True), "negative"),
    (("nonnegative", True), ("nonpositive", True), "zero"),
    (("integer", True), ("even", False), "odd"),
    (("integer", True), ("odd", False), "even"),
)


class UnsupportedAssumptions(Exception):
    """An unsupported inference remains unknown, never a partial answer."""


@lru_cache(maxsize=128)
def _consequences(name):
    """Adapt the native implication vocabulary to the pinned Python Q API.

    Native Finite/Infinite are not Python's complex-number predicates. In the
    Python profile finite alone does not establish complex, and infinite
    excludes complex. Their native implications must not cross this boundary.
    """
    if name not in NATIVE_PREDICATES or name in ("finite", "infinite"):
        positive = {name}
        negative = set()
    else:
        positive = set(_native.predicate_closure(name))
        negative = set(_native.predicate_contradictions(name))
    if "complex" in positive:
        positive.add("finite")
    if name == "nonzero":
        positive.update(("real", "complex", "finite"))
    if name == "odd":
        positive.add("nonzero")
        negative.add("zero")
    if name == "finite":
        negative.add("infinite")
    elif name == "infinite":
        negative.update(NATIVE_PREDICATES - {"infinite"})
    if name in ("complex", "real", "algebraic", "transcendental"):
        negative.add("infinite")
    return frozenset(positive), frozenset(negative)


class QueryProblem:
    """One cumulative work budget and source universe for an assumptions call."""

    def __init__(self):
        self.remaining = _MAX_WORK
        self.expressions = {}
        self._object_keys = {}

    def charge(self, amount=1, depth=0):
        if depth > _MAX_DEPTH or amount < 0 or amount > self.remaining:
            raise ValueError("assumptions query exceeds its work/depth limit")
        self.remaining -= amount
        _charge_nested_if_active(amount)

    def expression_key(self, expr):
        found = self._object_keys.get(id(expr))
        if found is not None:
            return found[1]
        self.charge()
        if len(self._object_keys) >= _MAX_EXPRESSIONS:
            raise ValueError("assumptions query exceeds its expression limit")
        try:
            key = _key(expr)
        except NotImplementedError as error:
            raise UnsupportedAssumptions from error
        self._object_keys[id(expr)] = (expr, key)
        self.expressions.setdefault(key, expr)
        return key

    def literal(self, expr, name, value=True):
        if type(name) is not str or not name or len(name) > 256:
            raise ValueError("assumptions predicate name exceeds its bound")
        key = self.expression_key(expr)
        if not value:
            return {(key, name): False}
        positive, negative = _consequences(name)
        branch = {(key, fact): True for fact in positive}
        branch.update(((key, fact), False) for fact in negative)
        self.charge(len(branch))
        return branch

    def merge(self, left, right):
        self.charge(len(left) + len(right) + 1)
        if len(left) + len(right) > 2 * _MAX_LITERALS:
            raise ValueError("assumptions query exceeds its literal limit")
        result = left.copy()
        for key, value in right.items():
            if key in result and result[key] is not value:
                return None
            result[key] = value
        groups = {}
        for (key, name), truth in result.items():
            groups.setdefault(key, {})[name] = truth
        for key, facts in groups.items():
            changed = True
            while changed:
                changed = False
                self.charge(len(_DOMAIN_RULES))
                for first, second, conclusion in _DOMAIN_RULES:
                    if facts.get(first[0]) is not first[1] or facts.get(second[0]) is not second[1]:
                        continue
                    positive, negative = _consequences(conclusion)
                    for names, truth in ((positive, True), (negative, False)):
                        for name in names:
                            if name in facts:
                                if facts[name] is not truth:
                                    return None
                            else:
                                facts[name] = truth
                                result[(key, name)] = truth
                                changed = True
        if len(result) > _MAX_LITERALS:
            raise ValueError("assumptions query exceeds its literal limit")
        return result

    def conjunction(self, left, right):
        self.charge(len(left) * len(right) + 1)
        result = []
        for a in left:
            for b in right:
                merged = self.merge(a, b)
                if merged is not None:
                    result.append(merged)
                    if len(result) > _MAX_BRANCHES:
                        raise ValueError("assumptions query exceeds its Boolean alternative limit")
        return result

    def disjunction(self, left, right):
        self.charge(len(left) + len(right) + 1)
        if len(left) + len(right) > _MAX_BRANCHES:
            raise ValueError("assumptions query exceeds its Boolean alternative limit")
        return left + right

    def branches(self, formula, negated=False, depth=0):
        from ..logic.boolalg import (
            And, BooleanFalse, BooleanFunction, BooleanTrue, Equivalent,
            ITE, Implies, Nand, Nor, Not, Or, Xnor, Xor,
        )

        self.charge(depth=depth)
        cls = type(formula)
        if formula is None:
            return [] if negated else [{}]
        if cls is bool or cls is BooleanTrue or cls is BooleanFalse:
            value = formula if cls is bool else cls is BooleanTrue
            return [{}] if value is not negated else []
        if cls is AppliedPredicate:
            predicate = AppliedPredicate.__dict__["_predicate"].__get__(formula)
            expr = AppliedPredicate.__dict__["_expr"].__get__(formula)
            if type(predicate) is not Predicate:
                raise UnsupportedAssumptions
            name = Predicate.__dict__["_name"].__get__(predicate)
            if type(name) is not str:
                raise UnsupportedAssumptions
            return [self.literal(expr, name, not negated)]
        if any(cls is kind for kind in (list, tuple, set, frozenset)):
            args, operation = tuple(formula), And
        elif any(cls is kind for kind in (And, Or, Not, Implies, Equivalent, Xor, Nand, Nor, Xnor, ITE)):
            args = BooleanFunction.__dict__["_args"].__get__(formula)
            if type(args) is not tuple:
                raise ValueError("assumptions Boolean children must be a tuple")
            operation = cls
        else:
            # Do not inspect arbitrary .args, descriptors or extension hooks.
            raise UnsupportedAssumptions
        self.charge(len(args), depth)
        if operation is Not:
            if len(args) != 1:
                raise ValueError("Not assumptions require exactly one operand")
            return self.branches(args[0], not negated, depth + 1)
        if operation in (And, Or, Nand, Nor):
            is_and = operation in (And, Nand)
            invert = negated ^ (operation in (Nand, Nor))
            combine_and = is_and != invert
            result = [{}] if combine_and else []
            combine = self.conjunction if combine_and else self.disjunction
            for arg in args:
                # Read every operand even when an earlier one decides the
                # formula, so limits/unsupported tails cannot be hidden.
                result = combine(result, self.branches(arg, invert, depth + 1))
            return result
        if operation is Implies:
            if len(args) != 2:
                raise ValueError("Implies assumptions require two operands")
            left = self.branches(args[0], not negated, depth + 1)
            right = self.branches(args[1], negated, depth + 1)
            return self.conjunction(left, right) if negated else self.disjunction(left, right)
        if operation is ITE:
            if len(args) != 3:
                raise ValueError("ITE assumptions require three operands")
            yes = self.conjunction(
                self.branches(args[0], False, depth + 1),
                self.branches(args[1], negated, depth + 1),
            )
            no = self.conjunction(
                self.branches(args[0], True, depth + 1),
                self.branches(args[2], negated, depth + 1),
            )
            return self.disjunction(yes, no)
        if operation is Equivalent:
            if len(args) < 2:
                return [] if negated else [{}]
            first_yes = self.branches(args[0], False, depth + 1)
            first_no = self.branches(args[0], True, depth + 1)
            if negated:
                result = []
                for arg in args[1:]:
                    unequal = self.disjunction(
                        self.conjunction(first_yes, self.branches(arg, True, depth + 1)),
                        self.conjunction(first_no, self.branches(arg, False, depth + 1)),
                    )
                    result = self.disjunction(result, unequal)
                return result
            all_yes, all_no = first_yes, first_no
            for arg in args[1:]:
                all_yes = self.conjunction(all_yes, self.branches(arg, False, depth + 1))
                all_no = self.conjunction(all_no, self.branches(arg, True, depth + 1))
            return self.disjunction(all_yes, all_no)
        # XOR/XNOR retain every argument, including arities greater than two.
        even, odd = [{}], []
        for arg in args:
            yes = self.branches(arg, False, depth + 1)
            no = self.branches(arg, True, depth + 1)
            even, odd = (
                self.disjunction(self.conjunction(even, no), self.conjunction(odd, yes)),
                self.disjunction(self.conjunction(even, yes), self.conjunction(odd, no)),
            )
        return even if negated ^ (operation is Xnor) else odd

    def logical_value(self, expr, name, branch):
        key = (self.expression_key(expr), name)
        if key in branch:
            return branch[key]
        if self.merge(branch, self.literal(expr, name, True)) is None:
            return False
        if self.merge(branch, self.literal(expr, name, False)) is None:
            return True
        return None

    def answer(self, expr, name, branches):
        if not branches:
            raise ValueError("inconsistent assumptions")
        answers = []
        for branch in branches:
            self.charge()
            value = self.logical_value(expr, name, branch)
            if value is None:
                value = self.mathematical_value(expr, name, branch)
            answers.append(value)
        first = answers[0]
        return first if first is not None and all(value is first for value in answers) else None

    def mathematical_value(self, expr, name, branch):
        """Infer only from an admitted, lossless closed source and typed facts."""
        try:
            if _contains_opaque(expr):
                return None
            native = _native_expr(expr)
            if self.expression_key(expr) != _key(_wrap(native)):
                return None
        except (NotImplementedError, TypeError):
            return None
        symbols = tuple(expr.free_symbols)
        if any(type(symbol) is not Symbol and type(symbol) is not Dummy for symbol in symbols):
            return None
        bindings = {}
        declarations = {}
        constrained = False
        for symbol in symbols:
            key = self.expression_key(symbol)
            binding, declared = typed_symbol(symbol)
            bindings[key] = binding
            declarations[key] = declared
            constrained |= any(target == key for target, _ in branch)
        if not symbols and name in NATIVE_PREDICATES and any(
            native == _native.Expr(token) for token in ("oo", "-oo", "zoo")
        ):
            # Python Q.infinite excludes every finite numeric-tower predicate.
            return name == "infinite"
        context = _native.AssumptionsContext()
        effective_branch = branch.copy()
        safe_properties = not constrained and self.property_source_admitted(expr)
        for key, binding in bindings.items():
            supplied = {literal: truth for literal, truth in branch.items() if literal[0] == key}
            declared_facts = {}
            for predicate, truth in declarations[key].items():
                declared_facts = self.merge(
                    declared_facts,
                    self.literal(self.expressions[key], predicate, truth),
                )
                if declared_facts is None:
                    raise ValueError("inconsistent symbol assumptions")
            combined = self.merge(supplied, declared_facts)
            if combined is None:
                # Explicit premises retain their established precedence over
                # conflicting constructor facts. Compatible premises augment
                # the declaration instead of erasing it (positive x remains
                # positive when the caller additionally states integer(x)).
                combined = supplied
            effective_branch.update(combined)
            stated = [(predicate, truth) for (_, predicate), truth in combined.items()]
            if self.expression_key(expr) == key:
                declared_answer = self.logical_value(expr, name, combined)
                if declared_answer is not None:
                    return declared_answer
            if any(predicate not in NATIVE_PREDICATES for predicate in declarations[key]):
                # Legacy Symbol properties forward every positive declaration
                # to the native predicate parser. Shell-only facts such as
                # commutative must stay in this signed, typed query boundary,
                # including when a compound property visits that symbol.
                safe_properties = False
            complex_value = ("complex", True) in stated
            if (("infinite", True) in stated
                    or ("finite", True) in stated and not complex_value):
                # Legacy Symbol/compound properties still use the native
                # finite vocabulary. They cannot override the adapted facts.
                safe_properties = False
            for predicate, truth in stated:
                if truth is True and predicate in NATIVE_PREDICATES - {"finite", "infinite"}:
                    context.assume_binding(binding, predicate)
                elif truth is True and predicate == "finite" and complex_value:
                    # Finite->Complex in the native vocabulary is safe only
                    # when complex has already been established independently.
                    context.assume_binding(binding, predicate)
        admitted = self.native_domain_admitted(native, context)
        # Closed atomic properties include Python's infinity/NaN behavior;
        # composite property handlers require the same domain preflight as
        # native inference (e.g. a held atan(I) must not become finite).
        if safe_properties and name in _PROPERTY_PREDICATES and (admitted or not expr.args):
            value = getattr(expr, "is_" + name, None)
            if type(value) is bool:
                return value
        if not admitted or name not in NATIVE_PREDICATES:
            return None
        realness = self.known_realness(native, context, effective_branch)
        if realness is False and name in {
            "real", "rational", "integer", "positive", "negative", "nonnegative",
            "nonpositive", "zero", "nonzero", "even", "odd", "prime",
        }:
            return False
        answer = context.is_true(native, name)
        if name == "nonzero" and answer is True and realness is not True:
            # Native NonZero includes non-real complex values; Python Q.nonzero
            # requires a real value. Non-vanishing exp(z) alone is insufficient.
            return None
        return answer

    def property_source_admitted(self, expr, depth=0):
        """Canonical-only heuristics must not classify deliberately held forms."""
        self.charge(depth=depth)
        cls = type(expr)
        if cls is Add or cls is Mul or cls is Pow:
            try:
                cls.__dict__["_args"].__get__(expr)
            except AttributeError:
                pass
            else:
                return False
        return all(self.property_source_admitted(arg, depth + 1) for arg in expr.args)

    def known_realness(self, native, context, branch, depth=0):
        """Keep non-realness separate from the native non-vanishing predicate."""
        self.charge(depth=depth)
        if context.is_true(native, "real") is True:
            return True
        if native == _native.Expr("I"):
            return False
        if native.is_symbol:
            if (self.logical_value(_wrap(native), "real", branch) is False
                    and context.is_true(native, "complex") is True):
                return False
            return None
        kind = native.func_name
        if kind not in ("Add", "Mul"):
            return None
        args = native.args
        parts = [self.known_realness(arg, context, branch, depth + 1) for arg in args]
        if any(value is None for value in parts) or parts.count(False) != 1:
            return None
        if kind == "Add" or all(
            value is False or context.is_true(arg, "nonzero") is True
            for arg, value in zip(args, parts)
        ):
            return False
        return None

    def native_domain_admitted(self, native, context, depth=0):
        """Fence structural inference to operators with checked source domains.

        In particular finite arguments do not exclude complex poles of atan or
        tanh. No enclosing arithmetic operation may erase that obligation.
        """
        self.charge(depth=depth)
        kind = native.func_name
        if kind in ("Symbol", "Integer", "Rational"):
            return True
        if kind == "Constant":
            return not any(native == _native.Expr(token) for token in ("oo", "-oo", "zoo", "nan"))
        args = native.args
        if not all(self.native_domain_admitted(arg, context, depth + 1) for arg in args):
            return False
        if kind in ("Add", "Mul"):
            return True
        if kind == "Pow" and len(args) == 2:
            base, exponent = args
            if exponent.is_integer or exponent.is_rational:
                numerator = exponent.exact_numerator()
                denominator = exponent.exact_denominator()
                if denominator == 1:
                    return numerator > 0 or context.is_true(base, "nonzero") is True
                predicate = "nonnegative" if numerator > 0 else "positive"
                return context.is_true(base, predicate) is True
            return (context.is_true(base, "positive") is True
                    and context.is_true(exponent, "real") is True)
        if len(args) != 1:
            return False
        if kind in ("exp", "sin", "cos", "sinh", "cosh", "erf", "erfc", "Abs", "abs"):
            return True
        if kind in ("atan", "tanh", "asinh"):
            return context.is_true(args[0], "real") is True
        if kind in ("log", "ln"):
            return context.is_true(args[0], "positive") is True
        if kind == "sqrt":
            return context.is_true(args[0], "nonnegative") is True
        return False


def typed_symbol(symbol):
    """Read the immutable native identity and reject mutated declaration state."""
    if type(symbol) is not Symbol and type(symbol) is not Dummy:
        raise TypeError("assumptions require an exact Symbol or Dummy")
    native = _native_expr(symbol)
    binding = native.symbol_binding
    declared = Symbol.__dict__["_assumptions"].__get__(symbol)
    if type(declared) is not dict or any(
        type(name) is not str or type(value) is not bool for name, value in declared.items()
    ):
        raise ValueError("symbol assumptions are not a canonical declaration")
    expected = _native.SymbolBinding(
        binding.name, [(name, "True" if value else "False") for name, value in declared.items()],
    )
    if (expected.plain != binding.plain
            or not expected.plain and expected.identity_hex != binding.identity_hex):
        raise ValueError("symbol assumptions changed after its typed identity was constructed")
    return binding, declared.copy()


def prepare_problem(items, domains=(), *, allow_unsupported=False):
    problem = QueryProblem()
    branches = [{}]
    if domains:
        native = _native.AssumptionsContext()
        symbols = {}
        for symbol, domain in domains:
            problem.charge()
            binding, _ = typed_symbol(symbol)
            native.assume_domain_binding(binding, domain)
            symbols[problem.expression_key(symbol)] = (symbol, binding)
        for symbol, binding in symbols.values():
            for name in native.deductions_binding(binding):
                branches = problem.conjunction(branches, [problem.literal(symbol, name)])
    from ..logic.boolalg import And, BooleanFunction

    unsupported = False

    def add_conjunct(item, depth=0):
        nonlocal branches, unsupported
        problem.charge(depth=depth)
        cls = type(item)
        if any(cls is kind for kind in (list, tuple, set, frozenset)):
            children = tuple(item)
        elif cls is And:
            children = BooleanFunction.__dict__["_args"].__get__(item)
            if type(children) is not tuple:
                raise ValueError("assumptions Boolean children must be a tuple")
        else:
            try:
                alternatives = problem.branches(item, depth=depth)
            except UnsupportedAssumptions:
                unsupported = True
                return
            branches = problem.conjunction(branches, alternatives)
            return
        problem.charge(len(children), depth)
        for child in children:
            add_conjunct(child, depth + 1)

    add_conjunct(tuple(items))
    if not branches:
        raise ValueError("inconsistent assumptions")
    if unsupported and not allow_unsupported:
        raise UnsupportedAssumptions
    return problem, branches
