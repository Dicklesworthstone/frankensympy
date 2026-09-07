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


__all__ = [
    "And",
    "Boolean",
    "BooleanAtom",
    "BooleanFalse",
    "BooleanFunction",
    "BooleanTrue",
    "Equivalent",
    "Implies",
    "Not",
    "Or",
    "Xor",
    "false",
    "simplify_logic",
    "to_cnf",
    "to_dnf",
    "true",
]
