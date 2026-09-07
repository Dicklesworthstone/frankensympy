"""Boolean algebra and propositional logic expressions for FrankenSymPy."""

from __future__ import annotations

from typing import Any

from ..core import Expr, Symbol, _native


class Boolean(Expr):
    """A boolean expression."""

    __slots__ = ()

    def __and__(self, other: Any) -> "Boolean":
        return And(self, other)

    def __rand__(self, other: Any) -> "Boolean":
        return And(other, self)

    def __or__(self, other: Any) -> "Boolean":
        return Or(self, other)

    def __ror__(self, other: Any) -> "Boolean":
        return Or(other, self)

    def __invert__(self) -> "Boolean":
        return Not(self)

    def __rshift__(self, other: Any) -> "Boolean":
        return Implies(self, other)

    def __rrshift__(self, other: Any) -> "Boolean":
        return Implies(other, self)

    def __xor__(self, other: Any) -> "Boolean":
        return Xor(self, other)

    def __rxor__(self, other: Any) -> "Boolean":
        return Xor(other, self)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, bool):
            return False
        if type(self) is not type(other):
            return False
        return self.args == getattr(other, "args", None)

    def __hash__(self) -> int:
        return hash((type(self), self.args))


class BooleanAtom(Boolean):
    """A boolean atom (true, false)."""

    __slots__ = ()


class BooleanTrue(BooleanAtom):
    """The singleton true."""

    __slots__ = ()
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        if other is True:
            return True
        if other is False:
            return False
        from ..core import Number
        if isinstance(other, (Number, int, float)):
            return False
        return type(other) is type(self)

    def __hash__(self) -> int:
        return 1

    def __repr__(self) -> str:
        return "True"

    def __str__(self) -> str:
        return "True"


class BooleanFalse(BooleanAtom):
    """The singleton false."""

    __slots__ = ()
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        if other is False:
            return True
        if other is True:
            return False
        from ..core import Number
        if isinstance(other, (Number, int, float)):
            return False
        return type(other) is type(self)

    def __hash__(self) -> int:
        return 0

    def __repr__(self) -> str:
        return "False"

    def __str__(self) -> str:
        return "False"


true = BooleanTrue()
false = BooleanFalse()


class BooleanFunction(Boolean):
    """A composite boolean function."""

    __slots__ = ("_args",)

    def __init__(self, *args: Any):
        pass

    @property
    def args(self) -> tuple[Any, ...]:
        return getattr(self, "_args", ())


class And(BooleanFunction):
    """Logical AND."""

    __slots__ = ()

    def __new__(cls, *args: Any):
        flat: list[Any] = []
        for a in args:
            if isinstance(a, bool):
                a = true if a else false
            if isinstance(a, And):
                flat.extend(a.args)
            else:
                flat.append(a)

        res_args: list[Any] = []
        for a in flat:
            if a is false or a is False:
                return false
            if a is true or a is True:
                continue
            if a not in res_args:
                res_args.append(a)

        if not res_args:
            return true
        if len(res_args) == 1:
            return res_args[0]

        res_args.sort(key=str)
        obj = object.__new__(cls)
        obj._args = tuple(res_args)
        return obj

    def __repr__(self) -> str:
        return f"And({', '.join(repr(a) for a in self.args)})"

    def __str__(self) -> str:
        return f"({' & '.join(str(a) for a in self.args)})"


class Or(BooleanFunction):
    """Logical OR."""

    __slots__ = ()

    def __new__(cls, *args: Any):
        flat: list[Any] = []
        for a in args:
            if isinstance(a, bool):
                a = true if a else false
            if isinstance(a, Or):
                flat.extend(a.args)
            else:
                flat.append(a)

        res_args: list[Any] = []
        for a in flat:
            if a is true or a is True:
                return true
            if a is false or a is False:
                continue
            if a not in res_args:
                res_args.append(a)

        if not res_args:
            return false
        if len(res_args) == 1:
            return res_args[0]

        res_args.sort(key=str)
        obj = object.__new__(cls)
        obj._args = tuple(res_args)
        return obj

    def __repr__(self) -> str:
        return f"Or({', '.join(repr(a) for a in self.args)})"

    def __str__(self) -> str:
        return f"({' | '.join(str(a) for a in self.args)})"


class Not(BooleanFunction):
    """Logical NOT."""

    __slots__ = ()

    def __new__(cls, arg: Any):
        if isinstance(arg, bool):
            arg = true if arg else false
        if arg is true:
            return false
        if arg is false:
            return true
        if isinstance(arg, Not):
            return arg.args[0]

        obj = object.__new__(cls)
        obj._args = (arg,)
        return obj

    def __repr__(self) -> str:
        return f"Not({repr(self.args[0])})"

    def __str__(self) -> str:
        return f"~{self.args[0]}"


class Xor(BooleanFunction):
    """Logical XOR."""

    __slots__ = ()

    def __new__(cls, *args: Any):
        if len(args) == 2:
            a, b = args
            if isinstance(a, bool):
                a = true if a else false
            if isinstance(b, bool):
                b = true if b else false
            if a is true:
                return Not(b)
            if a is false:
                return b
            if b is true:
                return Not(a)
            if b is false:
                return a
            if a == b:
                return false
        obj = object.__new__(cls)
        obj._args = tuple(args)
        return obj

    def __repr__(self) -> str:
        return f"Xor({', '.join(repr(a) for a in self.args)})"

    def __str__(self) -> str:
        return f"({' ^ '.join(str(a) for a in self.args)})"


class Implies(BooleanFunction):
    """Logical Implication."""

    __slots__ = ()

    def __new__(cls, a: Any, b: Any):
        if isinstance(a, bool):
            a = true if a else false
        if isinstance(b, bool):
            b = true if b else false
        if a is false or b is true:
            return true
        if a is true:
            return b
        if b is false:
            return Not(a)
        if a == b:
            return true
        obj = object.__new__(cls)
        obj._args = (a, b)
        return obj

    def __repr__(self) -> str:
        return f"Implies({repr(self.args[0])}, {repr(self.args[1])})"

    def __str__(self) -> str:
        return f"({self.args[0]} >> {self.args[1]})"


class Equivalent(BooleanFunction):
    """Logical Equivalence."""

    __slots__ = ()

    def __new__(cls, *args: Any):
        obj = object.__new__(cls)
        obj._args = tuple(args)
        return obj

    def __repr__(self) -> str:
        return f"Equivalent({', '.join(repr(a) for a in self.args)})"

    def __str__(self) -> str:
        return f"({' <=> '.join(str(a) for a in self.args)})"


class Nand(BooleanFunction):
    """Logical NAND."""

    __slots__ = ()

    def __new__(cls, *args: Any):
        a = And(*args)
        if a is true:
            return false
        if a is false:
            return true
        obj = object.__new__(cls)
        obj._args = tuple(args)
        return obj

    def __repr__(self) -> str:
        return f"Nand({', '.join(repr(a) for a in self.args)})"


class Nor(BooleanFunction):
    """Logical NOR."""

    __slots__ = ()

    def __new__(cls, *args: Any):
        o = Or(*args)
        if o is true:
            return false
        if o is false:
            return true
        obj = object.__new__(cls)
        obj._args = tuple(args)
        return obj

    def __repr__(self) -> str:
        return f"Nor({', '.join(repr(a) for a in self.args)})"


class Xnor(BooleanFunction):
    """Logical XNOR (logical biconditional / equivalence)."""

    __slots__ = ()

    def __new__(cls, *args: Any):
        if len(args) == 2:
            a, b = args
            if a == b:
                return true
            if a is true:
                return b
            if b is true:
                return a
            if a is false:
                return Not(b)
            if b is false:
                return Not(a)
        obj = object.__new__(cls)
        obj._args = tuple(args)
        return obj

    def __repr__(self) -> str:
        return f"Xnor({', '.join(repr(a) for a in self.args)})"


class ITE(BooleanFunction):
    """If-Then-Else conditional operator on boolean expressions."""

    __slots__ = ()

    def __new__(cls, c: Any, a: Any, b: Any):
        if isinstance(c, bool):
            c = true if c else false
        if isinstance(a, bool):
            a = true if a else false
        if isinstance(b, bool):
            b = true if b else false

        if c is true:
            return a
        if c is false:
            return b
        if a == b:
            return a
        if a is true and b is false:
            return c
        if a is false and b is true:
            return Not(c)
        if a is true:
            return Or(c, b)
        if b is false:
            return And(c, a)
        if a is false:
            return And(Not(c), b)
        if b is true:
            return Or(Not(c), a)

        obj = object.__new__(cls)
        obj._args = (c, a, b)
        return obj

    def __repr__(self) -> str:
        return f"ITE({repr(self.args[0])}, {repr(self.args[1])}, {repr(self.args[2])})"


NAND = Nand
NOR = Nor
XNOR = Xnor


def _native_bool(expr: Any) -> _native.BoolExpr:
    """Converts a surface expression to a native BoolExpr."""
    if isinstance(expr, bool):
        return _native.BoolExpr.bool_const(expr)
    if isinstance(expr, BooleanTrue) or expr is true:
        return _native.BoolExpr.bool_const(True)
    if isinstance(expr, BooleanFalse) or expr is false:
        return _native.BoolExpr.bool_const(False)
    if isinstance(expr, Not):
        return _native.BoolExpr.bool_not(_native_bool(expr.args[0]))
    if isinstance(expr, And):
        return _native.BoolExpr.bool_and([_native_bool(a) for a in expr.args])
    if isinstance(expr, Or):
        return _native.BoolExpr.bool_or([_native_bool(a) for a in expr.args])
    if isinstance(expr, Implies):
        return _native.BoolExpr.bool_implies(_native_bool(expr.args[0]), _native_bool(expr.args[1]))
    if isinstance(expr, Equivalent):
        return _native.BoolExpr.bool_equivalent(_native_bool(expr.args[0]), _native_bool(expr.args[1]))
    if isinstance(expr, Xor):
        return _native.BoolExpr.bool_xor(_native_bool(expr.args[0]), _native_bool(expr.args[1]))
    if isinstance(expr, Nand):
        return _native.BoolExpr.bool_not(_native.BoolExpr.bool_and([_native_bool(a) for a in expr.args]))
    if isinstance(expr, Nor):
        return _native.BoolExpr.bool_not(_native.BoolExpr.bool_or([_native_bool(a) for a in expr.args]))
    if isinstance(expr, Xnor):
        return _native.BoolExpr.bool_not(_native.BoolExpr.bool_xor(_native_bool(expr.args[0]), _native_bool(expr.args[1])))
    if isinstance(expr, ITE):
        c, a, b = [_native_bool(x) for x in expr.args]
        return _native.BoolExpr.bool_or([
            _native.BoolExpr.bool_and([c, a]),
            _native.BoolExpr.bool_and([_native.BoolExpr.bool_not(c), b]),
        ])
    if isinstance(expr, Symbol):
        return _native.BoolExpr.bool_var(expr.name)
    if isinstance(expr, str):
        return _native.BoolExpr.bool_var(expr)
    raise TypeError(f"cannot convert {type(expr).__name__} to BoolExpr")


def _from_native_bool(nb: _native.BoolExpr) -> Any:
    """Converts a native BoolExpr to a surface expression."""
    kind = nb.kind()
    if kind == "Const":
        return true if nb.const_value() else false
    if kind == "Var":
        return Symbol(nb.var_name())
    if kind == "Not":
        args = nb.args()
        return Not(_from_native_bool(args[0]))
    if kind == "And":
        args = nb.args()
        return And(*[_from_native_bool(a) for a in args])
    if kind == "Or":
        args = nb.args()
        return Or(*[_from_native_bool(a) for a in args])
    if kind == "Implies":
        args = nb.args()
        return Implies(_from_native_bool(args[0]), _from_native_bool(args[1]))
    if kind == "Equivalent":
        args = nb.args()
        return Equivalent(*[_from_native_bool(a) for a in args])
    raise ValueError(f"unknown BoolExpr kind: {kind}")


def to_cnf(expr: Any, simplify: bool = False) -> Any:
    """Converts a boolean expression to Conjunctive Normal Form (CNF)."""
    if expr is true or expr is True:
        return true
    if expr is false or expr is False:
        return false
    nb = _native_bool(expr)
    res = _from_native_bool(nb.to_cnf())
    if simplify:
        return simplify_logic(res)
    return res


def to_dnf(expr: Any, simplify: bool = False) -> Any:
    """Converts a boolean expression to Disjunctive Normal Form (DNF)."""
    if expr is true or expr is True:
        return true
    if expr is false or expr is False:
        return false
    nb = _native_bool(expr)
    res = _from_native_bool(nb.to_dnf())
    if simplify:
        return simplify_logic(res)
    return res


def simplify_logic(expr: Any, form: str = "cnf", deep: bool = True) -> Any:
    """Simplifies a boolean expression."""
    if expr is true or expr is True:
        return true
    if expr is false or expr is False:
        return false
    nb = _native_bool(expr)
    return _from_native_bool(nb.simplify())


def is_nnf(expr: Any) -> bool:
    """Return True if expr is in Negation Normal Form, False otherwise."""
    if isinstance(expr, (bool, BooleanAtom, Symbol)):
        return True
    if isinstance(expr, Not):
        return isinstance(expr.args[0], (Symbol, BooleanAtom))
    if isinstance(expr, (And, Or)):
        return all(is_nnf(a) for a in expr.args)
    return False


def to_nnf(expr: Any, simplify: bool = True) -> Any:
    """Converts a boolean expression to Negation Normal Form (NNF)."""
    if expr is true or expr is True:
        return true
    if expr is false or expr is False:
        return false
    if isinstance(expr, Symbol):
        return expr
    if isinstance(expr, Not):
        arg = expr.args[0]
        if isinstance(arg, (Symbol, BooleanAtom)):
            return expr
        if isinstance(arg, Not):
            return to_nnf(arg.args[0], simplify=simplify)
        if isinstance(arg, And):
            return Or(*[to_nnf(Not(a), simplify=simplify) for a in arg.args])
        if isinstance(arg, Or):
            return And(*[to_nnf(Not(a), simplify=simplify) for a in arg.args])
        if isinstance(arg, Implies):
            return And(to_nnf(arg.args[0], simplify=simplify), to_nnf(Not(arg.args[1]), simplify=simplify))
        if isinstance(arg, Equivalent):
            p, q = arg.args
            return Or(
                And(to_nnf(p, simplify=simplify), to_nnf(Not(q), simplify=simplify)),
                And(to_nnf(Not(p), simplify=simplify), to_nnf(q, simplify=simplify)),
            )
        if isinstance(arg, Xor):
            p, q = arg.args
            return Or(
                And(to_nnf(p, simplify=simplify), to_nnf(q, simplify=simplify)),
                And(to_nnf(Not(p), simplify=simplify), to_nnf(Not(q), simplify=simplify)),
            )
        if isinstance(arg, Nand):
            return And(*[to_nnf(a, simplify=simplify) for a in arg.args])
        if isinstance(arg, Nor):
            return Or(*[to_nnf(a, simplify=simplify) for a in arg.args])
        if isinstance(arg, Xnor):
            p, q = arg.args
            return to_nnf(Xor(p, q), simplify=simplify)
        if isinstance(arg, ITE):
            c, a, b = arg.args
            return to_nnf(Or(And(c, Not(a)), And(Not(c), Not(b))), simplify=simplify)
        return Not(to_nnf(arg, simplify=simplify))
    if isinstance(expr, Implies):
        return Or(to_nnf(Not(expr.args[0]), simplify=simplify), to_nnf(expr.args[1], simplify=simplify))
    if isinstance(expr, Equivalent):
        p, q = expr.args
        return And(
            Or(to_nnf(Not(p), simplify=simplify), to_nnf(q, simplify=simplify)),
            Or(to_nnf(Not(q), simplify=simplify), to_nnf(p, simplify=simplify)),
        )
    if isinstance(expr, Xor):
        p, q = expr.args
        return Or(
            And(to_nnf(p, simplify=simplify), to_nnf(Not(q), simplify=simplify)),
            And(to_nnf(Not(p), simplify=simplify), to_nnf(q, simplify=simplify)),
        )
    if isinstance(expr, Nand):
        return Or(*[to_nnf(Not(a), simplify=simplify) for a in expr.args])
    if isinstance(expr, Nor):
        return And(*[to_nnf(Not(a), simplify=simplify) for a in expr.args])
    if isinstance(expr, Xnor):
        p, q = expr.args
        return to_nnf(Equivalent(p, q), simplify=simplify)
    if isinstance(expr, ITE):
        c, a, b = expr.args
        return Or(
            And(to_nnf(c, simplify=simplify), to_nnf(a, simplify=simplify)),
            And(to_nnf(Not(c), simplify=simplify), to_nnf(b, simplify=simplify)),
        )
    if isinstance(expr, And):
        res = And(*[to_nnf(a, simplify=simplify) for a in expr.args])
        return simplify_logic(res) if simplify else res
    if isinstance(expr, Or):
        res = Or(*[to_nnf(a, simplify=simplify) for a in expr.args])
        return simplify_logic(res) if simplify else res
    return expr


def truth_table(expr: Any, variables: list[Any]) -> Any:
    """Generate a truth table for the boolean expression.

    Yields:
        (combination, result): list of 0/1 for each variable, and bool result.
    """
    import itertools
    from .inference import pl_true

    variables = list(variables)
    n = len(variables)
    for combination in itertools.product([0, 1], repeat=n):
        model = {var: bool(val) for var, val in zip(variables, combination)}
        res = bool(pl_true(expr, model))
        yield list(combination), res


def SOPform(variables: list[Any], minterms: list[Any], dontcares: list[Any] = None) -> Any:
    """Return a Sum of Products (SOP) boolean expression."""
    if not minterms:
        return false
    n = len(variables)
    cubes = []
    for m in minterms:
        if isinstance(m, int):
            bits = [(m >> (n - 1 - i)) & 1 for i in range(n)]
        else:
            bits = list(m)
        literals = [variables[i] if bits[i] else Not(variables[i]) for i in range(n)]
        cubes.append(And(*literals) if len(literals) > 1 else literals[0])
    res = Or(*cubes) if len(cubes) > 1 else cubes[0]
    return simplify_logic(res)


def POSform(variables: list[Any], minterms: list[Any], dontcares: list[Any] = None) -> Any:
    """Return a Product of Sums (POS) boolean expression."""
    if not minterms:
        return true
    n = len(variables)
    cubes = []
    for m in minterms:
        if isinstance(m, int):
            bits = [(m >> (n - 1 - i)) & 1 for i in range(n)]
        else:
            bits = list(m)
        literals = [Not(variables[i]) if bits[i] else variables[i] for i in range(n)]
        cubes.append(Or(*literals) if len(literals) > 1 else literals[0])
    res = And(*cubes) if len(cubes) > 1 else cubes[0]
    return simplify_logic(res)


__all__ = [
    "And",
    "Boolean",
    "BooleanAtom",
    "BooleanFalse",
    "BooleanFunction",
    "BooleanTrue",
    "Equivalent",
    "ITE",
    "Implies",
    "NAND",
    "NOR",
    "Nand",
    "Nor",
    "Not",
    "Or",
    "POSform",
    "SOPform",
    "XNOR",
    "Xnor",
    "Xor",
    "false",
    "is_nnf",
    "simplify_logic",
    "to_cnf",
    "to_dnf",
    "to_nnf",
    "true",
    "truth_table",
]
