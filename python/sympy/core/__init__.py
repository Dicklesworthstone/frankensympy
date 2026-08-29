"""Small, honest SymPy-compatible surface backed by the native kernel.

This module intentionally exposes only the implemented vertical slice. It
never substitutes placeholder Python classes when the native extension is
missing: importing an unusable symbolic shell would make capability checks lie.
"""

from __future__ import annotations

import sys
from typing import Any, Iterable

_DUMMY_PREFIX = "__fsymDummy_"
_dummy_next = 1

try:
    import fsym_python as _native
except ImportError as exc:
    _native = None
    if sys.modules.get("fsym_python", -1) is not None:
        import importlib.machinery
        import importlib.util
        import os
        from pathlib import Path

        search_dirs = []
        if "CARGO_TARGET_DIR" in os.environ:
            target = Path(os.environ["CARGO_TARGET_DIR"])
            search_dirs.extend([target / "debug", target / "release", target])
        search_dirs.extend([
            Path("target/debug"),
            Path("target/release"),
            Path("../target/debug"),
            Path("../target/release"),
            Path(__file__).resolve().parent.parent.parent.parent / "target" / "debug",
            Path("/data/tmp/cargo-target/debug"),
        ])
        names = ("fsym_python.so", "libfsym_python.so", "fsym_python.pyd", "libfsym_python.dylib")
        for directory in search_dirs:
            if not directory.is_dir():
                continue
            for name in names:
                so = directory / name
                if not so.is_file():
                    continue
                try:
                    loader = importlib.machinery.ExtensionFileLoader("fsym_python", str(so.resolve()))
                    spec = importlib.util.spec_from_loader("fsym_python", loader)
                    if spec is not None and spec.loader is not None:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules["fsym_python"] = module
                        spec.loader.exec_module(module)
                        _native = module
                        break
                except Exception:
                    continue
            if _native is not None:
                break

    if _native is None:
        raise ImportError(
            "FrankenSymPy requires its fsym_python native extension; "
            "build or install the package with maturin before importing sympy"
        ) from exc


def _exact_surface_types():
    return (
        Basic,
        Atom,
        AtomicExpr,
        Expr,
        Symbol,
        Dummy,
        Number,
        Integer,
        Rational,
        ComplexInfinity,
        Add,
        Mul,
        Pow,
        Derivative,
        AppliedUndef,
        Application,
        Function,
    )


def _native_expr(value: Any):
    if isinstance(value, Basic):
        if type(value) not in _exact_surface_types() and not isinstance(value, Function):
            raise NotImplementedError(
                "custom symbolic subclasses require a Python override; "
                "the native fast path accepts exact built-in classes only"
            )
        return value._value
    if isinstance(value, _native.Expr):
        return value
    if isinstance(value, int):
        return _native.py_integer(value)
    if isinstance(value, float):
        from fractions import Fraction

        fract = Fraction(value)
        return _native.py_rational(fract.numerator, fract.denominator)
    if isinstance(value, str):
        return _native.Expr(value)
    raise TypeError(f"cannot convert {type(value).__name__} to native Expr")


def _dummy_intern_name(name: str, number: int) -> str:
    return f"{_DUMMY_PREFIX}_{number}_{name}"


def _parse_dummy_intern_name(name: str) -> tuple[int, str] | None:
    if not name.startswith(_DUMMY_PREFIX + "_"):
        return None
    rest = name[len(_DUMMY_PREFIX) + 1 :]
    parts = rest.split("_", 1)
    if len(parts) != 2:
        return None
    try:
        number = int(parts[0])
    except ValueError:
        return None
    return number, parts[1]


def _note_dummy_number(number: int) -> None:
    global _dummy_next
    if number >= _dummy_next:
        _dummy_next = number + 1


def _allocate_dummy_number() -> int:
    global _dummy_next
    number = _dummy_next
    _dummy_next += 1
    return number


def _symbol_from_intern_name(name: str) -> "Symbol":
    parsed = _parse_dummy_intern_name(name)
    if parsed is None:
        return Symbol(name)
    number, printed = parsed
    return Dummy._from_intern(printed, number)


def _native_symbol_key(symbol: "Symbol") -> str:
    return str(_native_expr(symbol))


def _wrap(value: Any) -> "Basic":
    if isinstance(value, Basic):
        return value
    if not isinstance(value, _native.Expr):
        raise TypeError(f"native expression required, got {type(value).__name__}")
    if value.func_name == "Symbol":
        parsed = _parse_dummy_intern_name(str(value))
        if parsed is not None:
            number, name = parsed
            return Dummy._from_intern(name, number, value)
    if value.func_name == "Constant":
        if str(value) == "zoo":
            return zoo
        return Expr(str(value))
    cls = {
        "Symbol": Symbol,
        "Integer": Integer,
        "Rational": Rational,
        "Add": Add,
        "Mul": Mul,
        "Pow": Pow,
        "Derivative": Derivative,
        "Constant": Expr,
    }.get(value.func_name)
    if cls is None:
        func_cls = Function(value.func_name)
        obj = object.__new__(func_cls)
        obj._value = value
        return obj
    obj = object.__new__(cls)
    obj._value = value
    return obj


def _parse_result(value: str) -> "Expr":
    return _wrap(_native.Expr(value))


def _restore_nary(cls, args):
    return cls(*args, evaluate=False)


def _restore_pow(cls, base, exponent):
    return cls(base, exponent, evaluate=False)


def _exact_integer_argument(value: Any) -> int:
    """Apply the pinned built-in conversions without invoking user hooks."""
    if type(value) is int:
        return value
    if type(value) is bool:
        return 1 if value else 0
    if type(value) is float:
        return int(value)
    if type(value) is str:
        return int(value)
    if type(value) is Integer:
        return value.p
    raise TypeError("admitted built-in number or Integer required")


def _exact_rational_argument(value: Any) -> tuple[int, int]:
    """Return an exact ratio for admitted built-ins and exact shell numbers."""
    if type(value) is int:
        return value, 1
    if type(value) is bool:
        return (1 if value else 0), 1
    if type(value) is float:
        return value.as_integer_ratio()
    if type(value) is Integer:
        return value.p, 1
    if type(value) is Rational:
        return value.p, value.q
    raise TypeError("exact built-in number, Integer, or Rational required")


class Basic:
    """Base class for all SymPy objects in the compatibility shell."""

    __slots__ = ("_value",)

    def __init__(self, src: Any = None):
        if isinstance(src, Basic):
            self._value = src._value
        elif isinstance(src, _native.Expr):
            self._value = src
        elif src is None or isinstance(src, str):
            self._value = _native.Expr(src)
        else:
            self._value = _native_expr(src)

    @property
    def args(self) -> tuple["Basic", ...]:
        return tuple(_wrap(arg) for arg in _native_expr(self).args)

    @property
    def func(self):
        if type(self) not in _exact_surface_types():
            return type(self)
        return type(_wrap(self._value))

    @property
    def free_symbols(self) -> set["Symbol"]:
        return {_symbol_from_intern_name(name) for name in _native_expr(self).free_symbols}

    def has(self, pattern: Any) -> bool:
        return _native_expr(self).has(_native_expr(pattern))

    def subs(self, old: Any, new: Any) -> "Basic":
        return _wrap(_native_expr(self).subs(_native_expr(old), _native_expr(new)))

    def _repr_latex_(self) -> str:
        return self._value._repr_latex_()

    def pretty(self) -> str:
        return self._value.pretty()

    def __str__(self) -> str:
        return str(self._value)

    def __repr__(self) -> str:
        return repr(self._value)

    def __hash__(self) -> int:
        if type(self) not in _exact_surface_types() and not isinstance(self, Function):
            return object.__hash__(self)
        return hash(self._value)

    def __eq__(self, other: object) -> bool:
        if type(self) not in _exact_surface_types() and not isinstance(self, Function):
            return self is other
        try:
            return self._value == _native_expr(other)
        except (TypeError, NotImplementedError):
            return False

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __deepcopy__(self, memo) -> "Basic":
        del memo
        return self

    def __reduce__(self):
        if type(self) is Dummy:
            return _restore_dummy, (self.name, self._dummy_number)
        if isinstance(self, AppliedUndef):
            return _restore_applied_undef, (self._value.func_name, self.args)
        if isinstance(self, Symbol):
            return type(self), (self.name,)
        if isinstance(self, Integer):
            return type(self), (self.p,)
        if isinstance(self, Rational):
            return type(self), (self.p, self.q)
        if isinstance(self, (Add, Mul, Derivative)):
            return _restore_nary, (type(self), self.args)
        if isinstance(self, Pow):
            return _restore_pow, (type(self), *self.args)
        return type(self), (str(self),)


class Atom(Basic):
    """A basic object that has no subexpressions (args == ())."""

    __slots__ = ()


class Expr(Basic):
    """Python-visible wrapper for a native exact mathematical expression."""

    __slots__ = ()

    @property
    def is_integer(self) -> bool:
        return self._value.is_integer

    @property
    def is_rational(self) -> bool:
        return self._value.is_rational

    @property
    def is_symbol(self) -> bool:
        return self._value.is_symbol

    @property
    def is_Add(self) -> bool:
        return self._value.is_add

    @property
    def is_Mul(self) -> bool:
        return self._value.is_mul

    @property
    def is_Pow(self) -> bool:
        return self._value.is_pow

    @property
    def is_number(self) -> bool:
        return self._value.is_number

    def diff(self, *variables: Any) -> "Expr":
        return diff(self, *variables)

    def simplify(self) -> "Expr":
        return simplify(self)

    def expand(self) -> "Expr":
        return expand(self)

    def evalf(self) -> float:
        return _native_expr(self).evalf()

    def __lt__(self, other: Any) -> bool:
        return _native_expr(self) < _native_expr(other)

    def __le__(self, other: Any) -> bool:
        return _native_expr(self) <= _native_expr(other)

    def __gt__(self, other: Any) -> bool:
        return _native_expr(self) > _native_expr(other)

    def __ge__(self, other: Any) -> bool:
        return _native_expr(self) >= _native_expr(other)

    def __add__(self, other: Any) -> "Expr":
        return _wrap(_native_expr(self) + _native_expr(other))

    def __radd__(self, other: Any) -> "Expr":
        return _wrap(_native_expr(other) + _native_expr(self))

    def __sub__(self, other: Any) -> "Expr":
        return _wrap(_native_expr(self) - _native_expr(other))

    def __rsub__(self, other: Any) -> "Expr":
        return _wrap(_native_expr(other) - _native_expr(self))

    def __mul__(self, other: Any) -> "Expr":
        return _wrap(_native_expr(self) * _native_expr(other))

    def __rmul__(self, other: Any) -> "Expr":
        return _wrap(_native_expr(other) * _native_expr(self))

    def __truediv__(self, other: Any) -> "Expr":
        reciprocal = _native.py_pow(_native_expr(other), _native.py_integer(-1))
        return _wrap(_native_expr(self) * reciprocal)

    def __rtruediv__(self, other: Any) -> "Expr":
        reciprocal = _native.py_pow(_native_expr(self), _native.py_integer(-1))
        return _wrap(_native_expr(other) * reciprocal)

    def __pow__(self, exponent: Any, modulo: Any = None) -> "Expr":
        if modulo is not None:
            raise TypeError("modular symbolic exponentiation is not implemented")
        return _wrap(_native.py_pow(_native_expr(self), _native_expr(exponent)))

    def __neg__(self) -> "Expr":
        return _wrap(-_native_expr(self))


class AtomicExpr(Expr, Atom):
    """An expression that is also an Atom."""

    __slots__ = ()


class Symbol(AtomicExpr):
    __slots__ = ("_assumptions",)

    def __init__(self, name: str, **assumptions: Any):
        if not isinstance(name, str):
            raise TypeError("Symbol name must be a string")
        if name.startswith(_DUMMY_PREFIX):
            raise ValueError("Symbol name collides with Dummy intern encoding")
        self._assumptions = {k: v for k, v in assumptions.items() if v is not None}
        self._value = _native.py_symbol(name)

    @property
    def name(self) -> str:
        return str(self)

    @property
    def is_positive(self) -> bool | None:
        return self._assumptions.get("positive")

    @property
    def is_real(self) -> bool | None:
        return self._assumptions.get("real")

    @property
    def is_integer(self) -> bool | None:
        return self._assumptions.get("integer")

    @property
    def is_negative(self) -> bool | None:
        return self._assumptions.get("negative")

    @property
    def is_zero(self) -> bool | None:
        return self._assumptions.get("zero")

    @property
    def is_nonnegative(self) -> bool | None:
        return self._assumptions.get("nonnegative")

    @property
    def is_nonpositive(self) -> bool | None:
        return self._assumptions.get("nonpositive")

    def __repr__(self) -> str:
        return self.name

    def _srepr(self) -> str:
        if self._assumptions:
            items = ", ".join(f"{k}={v!r}" for k, v in sorted(self._assumptions.items()))
            return f"Symbol({self.name!r}, {items})"
        return f"Symbol({self.name!r})"


class Dummy(Symbol):
    """Dummy symbol whose identity is distinct across constructor calls."""

    __slots__ = ("_dummy_name", "_dummy_number")

    def __init__(self, name: str = "Dummy", **assumptions: Any):
        if assumptions:
            raise NotImplementedError("symbol assumptions are not implemented in this profile")
        if not isinstance(name, str):
            raise TypeError("Dummy name must be a string")
        number = _allocate_dummy_number()
        self._dummy_name = name
        self._dummy_number = number
        self._assumptions = {}
        self._value = _native.py_symbol(_dummy_intern_name(name, number))

    @classmethod
    def _from_intern(cls, name: str, number: int, value: Any = None) -> "Dummy":
        dummy = object.__new__(cls)
        dummy._dummy_name = name
        dummy._dummy_number = number
        dummy._value = (
            value if value is not None else _native.py_symbol(_dummy_intern_name(name, number))
        )
        _note_dummy_number(number)
        return dummy

    @property
    def name(self) -> str:
        return self._dummy_name

    @property
    def dummy_index(self) -> int:
        return self._dummy_number

    def __repr__(self) -> str:
        return f"Dummy({self.name!r})"


def _restore_dummy(name: str, number: int) -> Dummy:
    return Dummy._from_intern(name, number)


class Number(AtomicExpr):
    """Base class for exact numbers in the compatibility shell."""

    __slots__ = ()


class Rational(Number):
    __slots__ = ()

    def __init__(self, numerator: int, denominator: int):
        numerator_p, numerator_q = _exact_rational_argument(numerator)
        denominator_p, denominator_q = _exact_rational_argument(denominator)
        self._value = _native.py_rational(
            numerator_p * denominator_q,
            numerator_q * denominator_p,
        )

    @property
    def p(self) -> int:
        return self._value.exact_numerator()

    @property
    def q(self) -> int:
        return self._value.exact_denominator()


class Integer(Rational):
    __slots__ = ()

    def __init__(self, value: int):
        self._value = _native.py_integer(_exact_integer_argument(value))

    @property
    def p(self) -> int:
        return self._value.exact_numerator()

    @property
    def q(self) -> int:
        return 1


class ComplexInfinity(AtomicExpr):
    """Complex infinity (zoo) singleton."""

    __slots__ = ()

    def __init__(self, name: str = "zoo"):
        self._value = _native.Expr("zoo")

    @property
    def is_number(self) -> bool:
        return True

    def __repr__(self) -> str:
        return "zoo"

    def __str__(self) -> str:
        return "zoo"

    def _repr_latex_(self) -> str:
        return r"\tilde{\infty}"

    def _srepr(self) -> str:
        return "zoo"


zoo = ComplexInfinity("zoo")


class Add(Expr):
    __slots__ = ()

    def __new__(cls, *args: Any, evaluate: bool = True):
        native_args = [_native_expr(arg) for arg in args]
        val = _native.Add(*native_args, evaluate=evaluate).as_expr()
        if evaluate:
            return _wrap(val)
        obj = object.__new__(cls)
        obj._value = val
        return obj

    def __init__(self, *args: Any, evaluate: bool = True):
        pass


class Mul(Expr):
    __slots__ = ()

    def __new__(cls, *args: Any, evaluate: bool = True):
        native_args = [_native_expr(arg) for arg in args]
        val = _native.Mul(*native_args, evaluate=evaluate).as_expr()
        if evaluate:
            return _wrap(val)
        obj = object.__new__(cls)
        obj._value = val
        return obj

    def __init__(self, *args: Any, evaluate: bool = True):
        pass


class Pow(Expr):
    __slots__ = ()

    def __new__(cls, base: Any, exponent: Any, evaluate: bool = True):
        val = _native.Pow(
            _native_expr(base), _native_expr(exponent), evaluate=evaluate
        ).as_expr()
        if evaluate:
            return _wrap(val)
        obj = object.__new__(cls)
        obj._value = val
        return obj

    def __init__(self, base: Any, exponent: Any, evaluate: bool = True):
        pass


class Derivative(Expr):
    __slots__ = ()

    def __init__(self, expression: Any, *variables: Any, evaluate: bool = False):
        native_vars = [_native_expr(var) for var in variables]
        self._value = _native.Derivative(
            _native_expr(expression), *native_vars, evaluate=evaluate
        ).as_expr()


class FunctionClass(type):
    """Metaclass for all SymPy Function classes."""

    def __repr__(cls) -> str:
        return cls.__name__


class Application(Expr):
    """Application of a mathematical function."""

    __slots__ = ()


class Function(Application, metaclass=FunctionClass):
    """Base class for applied mathematical functions."""

    __slots__ = ("_args",)

    def __new__(cls, *args: Any, **options: Any):
        if cls is Function:
            if not args or not isinstance(args[0], str):
                raise TypeError("Function name must be a string")
            if not args[0]:
                raise ValueError("Function name must be non-empty")
            name = args[0]
            return UndefinedFunction(name)

        # Classmethod eval hook
        eval_method = getattr(cls, "eval", None)
        if eval_method is not None:
            evaluated = eval_method(*args)
            if evaluated is not None:
                return _wrap(_native_expr(evaluated))

        obj = object.__new__(cls)
        wrapped_args = tuple(_wrap(a) if not isinstance(a, Basic) else a for a in args)
        obj._args = wrapped_args
        name = cls.__name__
        native_args = [_native_expr(arg) for arg in args]
        obj._value = _native.py_function(name, *native_args)
        return obj

    def __init__(self, *args: Any, **options: Any):
        pass

    @property
    def args(self) -> tuple[Basic, ...]:
        if hasattr(self, "_args"):
            return self._args
        return super().args

    @property
    def func(self):
        return type(self)

    def __repr__(self) -> str:
        arg_strs = ", ".join(repr(a) for a in self.args)
        return f"{type(self).__name__}({arg_strs})"

    def __str__(self) -> str:
        return repr(self)


class AppliedUndef(Function):
    """Base class for applied undefined functions."""

    __slots__ = ()

    def __new__(cls, *args: Any, **options: Any):
        if cls is AppliedUndef:
            if not args or not isinstance(args[0], str):
                raise TypeError("AppliedUndef requires function name as first argument")
            name = args[0]
            fn = Function(name)
            return fn(*args[1:])
        return super().__new__(cls, *args, **options)


_undefined_functions: dict[str, Any] = {}


class UndefinedFunction(FunctionClass):
    """Metaclass/callable for undefined functions like Function('f')."""

    def __new__(mcls, name: str, bases=(AppliedUndef,), namespace=None):
        if name in _undefined_functions:
            return _undefined_functions[name]
        if namespace is None:
            namespace = {}
        cls = super().__new__(mcls, name, bases, namespace)
        cls.__module__ = "sympy.core.function"
        _undefined_functions[name] = cls
        return cls

    def __repr__(cls) -> str:
        return cls.__name__

    def __eq__(cls, other: object) -> bool:
        return type(other) is UndefinedFunction and cls.__name__ == other.__name__

    def __hash__(cls) -> int:
        return hash(("UndefinedFunction", cls.__name__))


def _restore_applied_undef(name: str, args: tuple[Any, ...]) -> AppliedUndef:
    return Function(name)(*args)


def symbols(names: str | Iterable[str], **assumptions: Any):
    """Create one or more symbols without silently discarding assumptions."""
    if isinstance(names, str):
        parts = [part for part in names.replace(",", " ").split() if part]
    else:
        parts = list(names)
    if not parts:
        raise ValueError("at least one symbol name is required")
    result = tuple(Symbol(name, **assumptions) for name in parts)
    return result if len(result) != 1 else result[0]


def _require_symbol(value: Any) -> Symbol:
    if type(value) is Symbol or type(value) is Dummy:
        return value
    if isinstance(value, Symbol):
        raise NotImplementedError(
            "custom Symbol subclasses require a supervised Python override lane"
        )
    raise TypeError("differentiation variable must be a Symbol")


def diff(expression: Any, *variables: Any) -> Expr:
    """Differentiate through the implemented exact native rule set."""
    if not variables:
        raise TypeError("at least one differentiation variable is required")
    result = _wrap(_native_expr(expression))
    for variable in variables:
        symbol = _require_symbol(variable)
        result = _parse_result(
            _native.diff_expr(str(result), _native_symbol_key(symbol))
        )
    return result


def expand(expression: Any) -> Expr:
    return _parse_result(_native.expand_expr(str(_wrap(_native_expr(expression)))))


def simplify(expression: Any) -> Expr:
    return _parse_result(_native.simplify_expr(str(_wrap(_native_expr(expression)))))


def pretty(expression: Any) -> str:
    """Unicode-math printer view. Not a semantic identity."""
    return _native_expr(expression).pretty()


Basic.__module__ = "sympy.core.basic"
Atom.__module__ = "sympy.core.basic"
Expr.__module__ = "sympy.core.expr"
AtomicExpr.__module__ = "sympy.core.expr"
Symbol.__module__ = "sympy.core.symbol"
Dummy.__module__ = "sympy.core.symbol"
Number.__module__ = "sympy.core.numbers"
Rational.__module__ = "sympy.core.numbers"
Integer.__module__ = "sympy.core.numbers"
Add.__module__ = "sympy.core.add"
Mul.__module__ = "sympy.core.mul"
Pow.__module__ = "sympy.core.power"
Derivative.__module__ = "sympy.core.function"
FunctionClass.__module__ = "sympy.core.function"
Application.__module__ = "sympy.core.function"
Function.__module__ = "sympy.core.function"
UndefinedFunction.__module__ = "sympy.core.function"
AppliedUndef.__module__ = "sympy.core.function"
ComplexInfinity.__module__ = "sympy.core.numbers"
_restore_nary.__module__ = "sympy.core.basic"
_restore_pow.__module__ = "sympy.core.basic"
_restore_dummy.__module__ = "sympy.core.basic"
_restore_applied_undef.__module__ = "sympy.core.basic"

import types as _types

for _mod_name, _mod_items in [
    ("sympy.core.basic", (Basic, Atom, _restore_nary, _restore_pow, _restore_dummy, _restore_applied_undef)),
    ("sympy.core.expr", (Expr, AtomicExpr)),
    ("sympy.core.symbol", (Symbol, Dummy, symbols)),
    ("sympy.core.numbers", (Number, Rational, Integer, ComplexInfinity)),
    ("sympy.core.add", (Add,)),
    ("sympy.core.mul", (Mul,)),
    ("sympy.core.power", (Pow,)),
    ("sympy.core.function", (Function, UndefinedFunction, AppliedUndef, Derivative, Application, FunctionClass, diff)),
]:
    _mod = sys.modules.get(_mod_name)
    if _mod is None:
        _mod = _types.ModuleType(_mod_name)
        sys.modules[_mod_name] = _mod
    for _item in _mod_items:
        setattr(_mod, getattr(_item, "__name__", str(_item)), _item)


__all__ = [
    "Add",
    "Application",
    "AppliedUndef",
    "Atom",
    "AtomicExpr",
    "Basic",
    "ComplexInfinity",
    "Derivative",
    "Dummy",
    "Expr",
    "Function",
    "FunctionClass",
    "Integer",
    "Mul",
    "Number",
    "Pow",
    "Rational",
    "Symbol",
    "UndefinedFunction",
    "diff",
    "expand",
    "pretty",
    "simplify",
    "symbols",
    "zoo",
]
