"""Small, honest SymPy-compatible surface backed by the native kernel.

This module intentionally exposes only the implemented vertical slice. It
never substitutes placeholder Python classes when the native extension is
missing: importing an unusable symbolic shell would make capability checks lie.
"""

from __future__ import annotations

import math
import struct
import sys
from fractions import Fraction
from typing import Any, Iterable

# The Python str printer recurses ~3 frames per nesting level and admissible
# trees go to MAX_EXPR_DEPTH=4096, so keep the interpreter limit safely above
# the worst case (a deeper construction is refused by the bridge anyway).
sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

_DUMMY_PREFIX = "__fsymDummy_"
_FLOAT_INTERN = "__fsymFloat"
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
    types = [
        Basic,
        Atom,
        AtomicExpr,
        Expr,
        Symbol,
        Dummy,
        Number,
        Integer,
        Rational,
        Float,
        ComplexInfinity,
        Add,
        Mul,
        Pow,
        Derivative,
        Zero,
        One,
        NegativeOne,
        AppliedUndef,
        Application,
        Function,
    ]
    for name in ("Relational", "Eq", "Ne", "Lt", "Le", "Gt", "Ge"):
        cls = globals().get(name)
        if cls is not None:
            types.append(cls)
    return tuple(types)


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
        return _float_intern(_ieee_bits(value))
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
    if value.func_name == _FLOAT_INTERN:
        return Float._from_native(value)
    cls = {
        "Symbol": Symbol,
        "Integer": Integer,
        "Rational": Rational,
        "Add": Add,
        "Mul": Mul,
        "Pow": Pow,
        "Derivative": Derivative,
        "Eq": Eq,
        "Ne": Ne,
        "Lt": Lt,
        "Le": Le,
        "Gt": Gt,
        "Ge": Ge,
        "Constant": Expr,
    }.get(value.func_name)
    if cls is None:
        func_cls = Function(value.func_name)
        obj = object.__new__(func_cls)
        obj._value = value
        return obj
    obj = object.__new__(cls)
    obj._value = value
    if isinstance(obj, Symbol):
        # Symbols recovered from native results bypass Symbol.__init__; the
        # native kernel carries no assumption payload, so the recovered atom
        # is assumption-free (oracle: Symbol('x').is_positive is None and
        # srepr prints plain Symbol('x')). Leaving the slot unset made every
        # _srepr/is_* access raise AttributeError (gauntlet bead
        # fra-shell-atom-assumptions-bypasses-7o3).
        obj._assumptions = {}
    if cls is Integer:
        # Native-recovered integers route to the same singletons the
        # constructors produce (bead fra-shell-number-canonical-construction-qf6).
        p = obj.p
        if p == 0:
            return _ZERO
        if p == 1:
            return _ONE
        if p == -1:
            return _NEGATIVE_ONE
    return obj


def _parse_result(value: str) -> "Expr":
    return _wrap(_native.Expr(value))


def _restore_nary(cls, args):
    return cls(*args, evaluate=False)


def _restore_pow(cls, base, exponent):
    return cls(base, exponent, evaluate=False)


def _is_sympy_operand(value: Any) -> bool:
    """True for objects that are themselves SymPy-surface values (shell
    expressions, matrices, ...) — their reflected operators deserve a
    NotImplemented round-trip instead of a hard TypeError (finding 4)."""
    return isinstance(value, Basic) or hasattr(value, "_value") or hasattr(value, "_native")
    return cls(base, exponent, evaluate=False)


def _admitted_exact_int(value: Any) -> int | None:
    """Non-raising exact-integer probe for construction folds (returns None
    for anything that is not a shell Integer or a built-in int)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, Integer):
        return int(value.p)
    if isinstance(value, int):
        return value
    return None

def _ieee_bits(value: float) -> int:
    return int.from_bytes(struct.pack(">d", value), "big")


def _bits_to_float(bits: int) -> float:
    value = int(bits)
    if value < 0 or value >= 1 << 64:
        raise ValueError("malformed Float intern encoding")
    return struct.unpack(">d", value.to_bytes(8, "big"))[0]


def _float_intern(bits: int):
    return _native.py_function(_FLOAT_INTERN, _native.py_integer(bits))


def _admitted_python_float(value: Any) -> float:
    """Pinned built-in conversions without invoking user ``__float__`` hooks."""
    if type(value) is float:
        return value
    if type(value) is bool:
        return 1.0 if value else 0.0
    if type(value) is int:
        return float(value)
    if type(value) is str:
        return float(value)
    if type(value) is Integer:
        return float(value.p)
    if type(value) is Rational:
        return value.p / value.q
    if type(value) is Float:
        return value._as_python_float()
    raise TypeError("admitted built-in number, Integer, Rational, or Float required")


def _maybe_python_float(value: Any) -> float | None:
    try:
        return _admitted_python_float(value)
    except (TypeError, ValueError, OverflowError):
        return None

def _exact_ratio(value: Any) -> tuple[int, int] | None:
    """Canonical (p, q) for admitted numeric atoms. Non-finite floats are None."""
    if isinstance(value, Zero):
        return 0, 1
    if isinstance(value, bool):
        return (1 if value else 0), 1
    if isinstance(value, int):
        return int(value), 1
    if isinstance(value, Integer):
        return value.p, 1
    if isinstance(value, Rational):
        return value.p, value.q
    if type(value) is float:
        try:
            return value.as_integer_ratio()
        except (OverflowError, ValueError):
            return None
    if type(value) is Float:
        try:
            return value._as_python_float().as_integer_ratio()
        except (OverflowError, ValueError):
            return None
    return None


def _is_numeric_coeff(value: Any, rational: bool) -> bool:
    if type(value) is Integer or type(value) is Rational:
        return True
    return (not rational) and type(value) is Float


def _combine_mul(factors: list["Expr"]) -> "Expr":
    if not factors:
        return Integer(1)
    if len(factors) == 1:
        return factors[0]
    return Mul(*factors)


def _combine_add(terms: list["Expr"]) -> "Expr":
    if not terms:
        return Integer(0)
    if len(terms) == 1:
        return terms[0]
    return Add(*terms)


def _raise_integer_power(base: "Expr", power: int) -> "Expr":
    """Raise a split numerator/denominator. Fold ±1 and 0 so we don't emit 1**n."""
    if power == 1:
        return base
    if type(base) is Integer:
        value = base.p
        if value == 0:
            return Integer(0)
        if value == 1:
            return Integer(1)
        if value == -1:
            return Integer(-1 if power % 2 else 1)
    return base**power


def _unsigned_term(term: "Expr") -> tuple["Expr", int]:
    """Peel a numeric leading minus. Returns (unsigned term, +1/-1)."""
    coeff, rest = term.as_coeff_Mul(rational=False)
    ratio = _exact_ratio(coeff)
    if ratio is None:
        return term, 1
    numer, denom = ratio
    if numer == 0:
        return term, 1
    if (numer < 0) == (denom < 0):
        return term, 1
    abs_numer, abs_denom = abs(numer), abs(denom)
    if type(coeff) is Float:
        positive: Expr = Float(abs(coeff._as_python_float()))
    elif type(coeff) is Integer or abs_denom == 1:
        positive = Integer(abs_numer)
    else:
        positive = Rational(abs_numer, abs_denom)
    if rest == Integer(1):
        return positive, -1
    if positive == Integer(1):
        return rest, -1
    return positive * rest, -1


def _add_tie_term_key(term: "Expr") -> tuple:
    """Term key emulating the pinned SymPy 1.14 tie-break ordering.

    Upstream resolves Add sign ties through sort_key() whose term order flows
    through as_terms()/monomial-key: degree DESCENDING, then generator names
    ascending, numeric coefficient (sign) last. Validated against the pinned
    oracle on: x-1, 1-x, x-y, y-x, 3-sqrt(2), -3+sqrt(2), x-sqrt(2),
    sqrt(2)-x, x-3, 3-x.
    """
    if isinstance(term, Number):
        ratio = _exact_ratio(term)
        coeff = Fraction(ratio[0], ratio[1]) if ratio else Fraction(0)
        return (0, (), coeff)
    if type(term) is Symbol or type(term) is Dummy:
        return (1, (term.name,), Fraction(1))
    if type(term) is Pow:
        base, exp = term.as_base_exp()
        base_degree, base_names, _ = _add_tie_term_key(base)
        exp_ratio = _exact_ratio(exp) if isinstance(exp, Number) else None
        degree = base_degree + (exp_ratio[0] // exp_ratio[1] if exp_ratio and exp_ratio[1] != 0 else 0)
        return (degree, base_names + (str(exp),), Fraction(1))
    if type(term) is Mul:
        coeff = Fraction(1)
        degree = 0
        names: tuple[str, ...] = ()
        for a in term.args:
            if isinstance(a, Number):
                r = _exact_ratio(a)
                if r:
                    coeff *= Fraction(r[0], r[1])
            else:
                d, nm, c = _add_tie_term_key(a)
                degree += d
                names += nm
                coeff *= c
        return (degree, names, coeff)
    return (1, (str(term),), Fraction(1))

def _add_tie_less(expr: "Expr") -> bool:
    """bool(expr.sort_key() < (-expr).sort_key()) as the pinned oracle computes it."""
    negated = Add(*(-arg for arg in expr.args))
    self_keys = sorted((-k[0], k[1], k[2]) for k in (_add_tie_term_key(a) for a in expr.args))
    neg_keys = sorted((-k[0], k[1], k[2]) for k in (_add_tie_term_key(a) for a in negated.args))
    return self_keys < neg_keys


def _number_is_extended_negative(n: "Expr") -> bool:
    """SymPy 1.14 Number.is_extended_negative semantics for concrete numbers.

    Rational keeps canonical q > 0, so p < 0 decides; Float follows binary64
    comparison (so -0.0 is not negative, matching upstream).
    """
    if type(n) is Float:
        return n._as_python_float() < 0.0
    if isinstance(n, Rational):
        return n.p < 0
    return False


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
    if type(value) is Float:
        return int(value._as_python_float())
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

    def __bool__(self) -> bool:
        # SymPy 1.14.0: bool(expr) is False only when the expression is
        # *definitely* zero; unknown (None) stays truthy (finding 5).
        is_zero = getattr(self, "is_zero", None)
        return is_zero is not True

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

    def atoms(self, *types: type) -> set["Basic"]:
        """Collect subexpressions. With no types, only atoms (empty args)."""
        found: set[Basic] = set()
        stack = [self]
        while stack:
            node = stack.pop()
            if not isinstance(node, Basic):
                continue
            args = node.args
            if types:
                if isinstance(node, types):
                    found.add(node)
                stack.extend(args)
                continue
            if args:
                stack.extend(args)
            else:
                found.add(node)
        return found

    def doit(self, **hints: Any) -> "Basic":
        """Evaluate held constructors one layer. Derivative evaluates; relationals stay held."""
        del hints
        if type(self) is Derivative:
            args = self.args
            if len(args) < 2:
                return self
            return Derivative(args[0], *args[1:], evaluate=True)
        if isinstance(self, Relational) or not self.args:
            return self
        new_args = tuple(arg.doit() if isinstance(arg, Basic) else arg for arg in self.args)
        if new_args == self.args:
            return self
        return self.func(*new_args)

    def find(self, query: Any) -> set["Basic"]:
        """Collect nodes matching a type or an exact expression."""
        found: set[Basic] = set()
        stack: list[Basic] = [self]
        while stack:
            node = stack.pop()
            if isinstance(query, type):
                if isinstance(node, query):
                    found.add(node)
            elif node == query:
                found.add(node)
            stack.extend(arg for arg in node.args if isinstance(arg, Basic))
        return found

    def xreplace(self, rule: Any) -> "Basic":
        """Replace exact nodes. Unlike ``subs``, this does not rewrite algebraically."""
        if isinstance(rule, dict):
            mapping = rule
        else:
            try:
                mapping = dict(rule)
            except (TypeError, ValueError) as exc:
                raise TypeError("xreplace mapping must be a dict or iterable of pairs") from exc
        for old, new in mapping.items():
            try:
                if self == old:
                    return _wrap(_native_expr(new))
            except (TypeError, NotImplementedError):
                continue
        args = self.args
        if not args:
            return self
        replaced = []
        changed = False
        for arg in args:
            if isinstance(arg, Basic):
                next_arg = arg.xreplace(mapping)
                if next_arg != arg:
                    changed = True
                replaced.append(next_arg)
            else:
                replaced.append(arg)
        if not changed:
            return self
        return self.func(*replaced)

    def _repr_latex_(self) -> str:
        return self._value._repr_latex_()

    def pretty(self) -> str:
        return self._value.pretty()

    def __str__(self) -> str:
        # SymPy 1.14.0 plain str printer (printer-parity bead qxr): Python
        # rendering over wrapped args; the native Display wraps every Add in
        # parentheses and keeps ' + -1' terms, which diverges from the oracle.
        return _str_expr(self)

    def __repr__(self) -> str:
        # Upstream SymPy: repr == str for core expressions (both StrPrinter).
        return _str_expr(self)

    def __hash__(self) -> int:
        if type(self) not in _exact_surface_types() and not isinstance(self, Function):
            return object.__hash__(self)
        ratio = _exact_ratio(self)
        if ratio is not None:
            from fractions import Fraction

            return hash(Fraction(ratio[0], ratio[1]))
        return hash(self._value)

    def __eq__(self, other: object) -> bool:
        if type(self) not in _exact_surface_types() and not isinstance(self, Function):
            return self is other
        left = _exact_ratio(self)
        right = _exact_ratio(other)
        if left is not None and right is not None:
            return left == right
        try:
            return self._value == _native_expr(other)
        except (TypeError, NotImplementedError):
            return False

    def __ne__(self, other: object) -> bool:
        return not self == other

    def sort_key(self, order: Any = None) -> tuple:
        """Canonical ordering key. Not a mathematical comparison and not TermId."""
        del order
        if type(self) not in _exact_surface_types() and not isinstance(self, Function):
            return (0, type(self).__name__)
        ratio = _exact_ratio(self)
        if ratio is not None:
            return (1, ratio, type(self).__name__)
        if type(self) is Float:
            # Non-finite floats reach here (finite ones returned a ratio
            # above). Use the numeric value itself in a 1-tuple so the key
            # stays comparable with the finite (p, q) 2-tuples (finding 11:
            # the old str-tuple crashed mixed sorts).
            return (1, (self._as_python_float(),), "Float")
        args = self.args
        if not args:
            return (2, type(self).__name__, (str(self),))
        return (
            3,
            type(self).__name__,
            tuple(arg.sort_key() for arg in args),
        )

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
        if type(self) is Float:
            return _restore_float, (self._as_python_float(), self.dps)
        if isinstance(self, Rational):
            return type(self), (self.p, self.q)
        if isinstance(self, (Add, Mul, Derivative)):
            return _restore_nary, (type(self), self.args)
        if isinstance(self, Pow):
            return _restore_pow, (type(self), *self.args)
        if isinstance(self, Relational):
            return type(self), self.args
        return type(self), (str(self),)


class Atom(Basic):
    """A basic object that has no subexpressions (args == ())."""

    __slots__ = ()


def _str_number(value: "Rational") -> tuple[bool, str]:
    """(is_negative, unsigned_body) for exact numbers."""
    p, q = value.p, value.q
    if q == 1:
        return p < 0, str(abs(p))
    return p < 0, f"{abs(p)}/{q}"


def _str_float_value(raw: float) -> str:
    # SymPy 1.14 str(Float): fixed decimal in the human range, otherwise
    # e-notation with the full 15-digit payload.
    if raw == 0:
        return "0" if math.copysign(1.0, raw) > 0 else "-0.000000000000000"
    if 1e-4 <= abs(raw) < 1e16:
        text = f"{raw:.15f}"
        return text
    return f"{raw:.14e}".replace("e-", "e-").replace("e+", "e+")


def _str_parenthesize(text: str, expr: "Expr") -> str:
    if (
        type(expr) is Add
        or type(expr) is Mul
        or (isinstance(expr, Rational) and (value := _str_number(expr))[0])
    ):
        return f"({text})"
    if isinstance(expr, Rational) and expr.q != 1:
        return f"({text})"
    return text


def _expr_degree(expr: "Expr") -> int:
    """Total formal degree for polynomial-shaped terms; 0 otherwise.
    Mirrors the dominant case of the oracle's monomial ordering."""
    if isinstance(expr, Symbol):
        return 1
    if isinstance(expr, Rational):
        return 0
    if type(expr) is Pow:
        base, exp = expr.args
        b = _expr_degree(base)
        n = _admitted_exact_int(exp)
        return b * n if (b is not None and n is not None and n > 0) else 0
    if type(expr) is Mul:
        return sum(_expr_degree(a) for a in expr.args)
    return 0


def _str_expr(expr: "Expr") -> str:
    """SymPy 1.14.0-faithful plain str printer (printer-parity bead qxr)."""
    neg, body = _str_term(expr)
    return ("-" + body) if neg else body

def _str_term(expr: "Expr") -> tuple[bool, str]:
    """Render one additive term as (negative, unsigned_body)."""
    if isinstance(expr, Rational):
        return _str_number(expr)
    if type(expr) is Float:
        text = _str_float_value(expr._as_python_float())
        return text.startswith("-"), text.lstrip("-")
    if type(expr) is Pow:
        base, exp = expr.args
        return False, f"{_str_parenthesize(_str_expr(base), base)}**{_str_parenthesize(_str_expr(exp), exp)}"
    if type(expr) is Mul:
        # Self-contained coefficient split (no as_coeff_Mul dependency).
        coeff_p, coeff_q = 1, 1
        rest = []
        for a in expr.args:
            if isinstance(a, Rational):
                coeff_p *= a.p
                coeff_q *= a.q
            else:
                rest.append(a)
        neg = coeff_p < 0
        coeff_p = abs(coeff_p)
        rest_strs = [
            _str_parenthesize(_str_expr(a), a)
            for a in sorted(rest, key=lambda a: a.sort_key())
        ]
        body = "*".join(rest_strs)
        if not rest:
            return (neg, f"{coeff_p}/{coeff_q}") if coeff_q != 1 else (neg, str(coeff_p))
        if coeff_q != 1:
            if coeff_p == 1:
                return neg, f"{body}/{coeff_q}"
            return neg, f"{coeff_p}*{body}/{coeff_q}"
        if coeff_p == 1:
            return neg, body
        return neg, f"{coeff_p}*{body}"
    if type(expr) is Add:
        const = None
        rest = []
        for a in expr.args:
            if isinstance(a, Rational):
                const = a if const is None else Rational(const.p * a.q + a.p * const.q, const.q * a.q)
            else:
                rest.append(a)
        if not isinstance(const, Rational):
            const = None
        if not rest:
            # Pure-number Add (e.g. produced by held-form folding): the
            # summed constant is the whole rendering.
            if isinstance(const, Rational):
                return _str_number(const)
            return False, str(_native_expr(expr))
        rendered = [_str_term(t) for t in sorted(rest, key=lambda a: (-_expr_degree(a), a.sort_key()))]
        out = rendered[0][1]
        for neg, body in rendered[1:]:
            out += (f" - {body}") if neg else (f" + {body}")
        if isinstance(const, Rational) and (const.p != 0 or not rendered):
            cneg, cbody = _str_number(const)
            out += (f" - {cbody}") if cneg else (f" + {cbody}")
        return False, out
    return False, str(_native_expr(expr))


def sort_key_of(a):
    return a.sort_key()


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

    def evalf(self, n: int = 15) -> "Float":
        if type(n) is not int or n < 1:
            raise TypeError("evalf dps must be a positive int")
        return Float(_native_expr(self).evalf(), n)

    def as_ordered_terms(self) -> tuple["Expr", ...]:
        """Addends in sort_key order. Non-Add expressions are a one-term tuple."""
        if type(self) is Add:
            return tuple(sorted(self.args, key=lambda term: term.sort_key()))
        return (self,)

    def as_coeff_Mul(self, rational: bool = True) -> tuple["Expr", "Expr"]:
        """Split a numeric multiplicative coefficient from the rest."""
        if type(self) is Mul:
            coeff: Expr = Integer(1)
            rest: list[Expr] = []
            for arg in self.args:
                if _is_numeric_coeff(arg, rational):
                    coeff = coeff * arg
                else:
                    rest.append(arg)
            if not rest:
                return coeff, Integer(1)
            if len(rest) == 1:
                return coeff, rest[0]
            return coeff, Mul(*rest, evaluate=False)
        if _is_numeric_coeff(self, rational):
            return self, Integer(1)
        return Integer(1), self

    def as_coeff_Add(self, rational: bool = True) -> tuple["Expr", "Expr"]:
        """Split a numeric additive coefficient from the rest."""
        if type(self) is Add:
            coeff: Expr = Integer(0)
            rest: list[Expr] = []
            for arg in self.args:
                if _is_numeric_coeff(arg, rational):
                    coeff = coeff + arg
                else:
                    rest.append(arg)
            if not rest:
                return coeff, Integer(0)
            if len(rest) == 1:
                return coeff, rest[0]
            return coeff, Add(*rest, evaluate=False)
        if _is_numeric_coeff(self, rational):
            return self, Integer(0)
        return Integer(0), self

    def as_base_exp(self) -> tuple["Expr", "Expr"]:
        """Split a power into ``(base, exp)``. Non-Pow expressions are ``(self, 1)``."""
        if type(self) is Pow:
            base, exp = self.args
            return base, exp
        return self, Integer(1)

    def as_numer_denom(self) -> tuple["Expr", "Expr"]:
        """Split into ``(numerator, denominator)``. Conservative on mixed-denominator Adds."""
        if isinstance(self, Rational):
            return Integer(self.p), Integer(self.q)
        if type(self) is Float:
            return self, Integer(1)
        if type(self) is Pow:
            base, exp = self.args
            ratio = _exact_ratio(exp)
            if ratio is None or ratio[1] != 1:
                return self, Integer(1)
            numer, denom = base.as_numer_denom()
            power = ratio[0]
            if power < 0:
                numer, denom = denom, numer
                power = -power
            if power == 0:
                return Integer(1), Integer(1)
            return _raise_integer_power(numer, power), _raise_integer_power(denom, power)
        if type(self) is Mul:
            numers: list[Expr] = []
            denoms: list[Expr] = []
            for arg in self.args:
                numer, denom = arg.as_numer_denom()
                numers.append(numer)
                denoms.append(denom)
            return _combine_mul(numers), _combine_mul(denoms)
        if type(self) is Add:
            parts = [arg.as_numer_denom() for arg in self.args]
            denoms = [denom for _, denom in parts]
            if denoms and all(d == denoms[0] for d in denoms[1:]):
                return _combine_add([numer for numer, _ in parts]), denoms[0]
            if denoms and all(type(denom) is Integer for denom in denoms):
                lcm = math.lcm(*(abs(denom.p) for denom in denoms))
                if lcm == 0:
                    return self, Integer(1)
                scaled: list[Expr] = []
                for numer, denom in parts:
                    scale = lcm // abs(denom.p)
                    if denom.p < 0:
                        scale = -scale
                    if scale == 1:
                        scaled.append(numer)
                    else:
                        scaled.append(numer * Integer(scale))
                return _combine_add(scaled), Integer(lcm)
            return self, Integer(1)
        return self, Integer(1)

    def could_extract_minus_sign(self) -> bool:
        """Profile-correct vs SymPy 1.14.0.

        Mirrors sympy/core/numbers.py (Number -> extended negativity),
        sympy/core/mul.py (leading Number factor, zoo self-negation guard), and
        sympy/core/add.py:_could_extract_minus_sign (majority count with
        sort_key tie-break against the negated form). Default False.
        """
        if type(self) is Mul:
            if self == (-self):
                return False
            # Upstream canonical Mul carries its Number coefficient in args[0];
            # the native kernel may keep it in any slot, so fold all Number
            # factors (product sign == coefficient sign, magnitude irrelevant).
            saw_number = False
            coefficient_negative = False
            for a in self.args:
                if isinstance(a, Number):
                    saw_number = True
                    if _number_is_extended_negative(a):
                        coefficient_negative = not coefficient_negative
            return saw_number and coefficient_negative
        if type(self) is Add:
            negative_args = sum(1 for i in self.args if i.could_extract_minus_sign())
            positive_args = len(self.args) - negative_args
            if positive_args > negative_args:
                return False
            if positive_args < negative_args:
                return True
            return _add_tie_less(self)
        if isinstance(self, Number):
            return _number_is_extended_negative(self)
        return False

    def __lt__(self, other: Any) -> bool:
        return _native_expr(self) < _native_expr(other)

    def __le__(self, other: Any) -> bool:
        return _native_expr(self) <= _native_expr(other)

    def __gt__(self, other: Any) -> bool:
        return _native_expr(self) > _native_expr(other)

    def __ge__(self, other: Any) -> bool:
        return _native_expr(self) >= _native_expr(other)

    def __add__(self, other: Any) -> "Expr":
        try:
            return _wrap(_native_expr(self) + _native_expr(other))
        except TypeError:
            if _is_sympy_operand(other):
                return NotImplemented
            raise

    def __radd__(self, other: Any) -> "Expr":
        try:
            return _wrap(_native_expr(other) + _native_expr(self))
        except TypeError:
            if _is_sympy_operand(other):
                return NotImplemented
            raise

    def __sub__(self, other: Any) -> "Expr":
        try:
            return _wrap(_native_expr(self) - _native_expr(other))
        except TypeError:
            if _is_sympy_operand(other):
                return NotImplemented
            raise

    def __rsub__(self, other: Any) -> "Expr":
        try:
            return _wrap(_native_expr(other) - _native_expr(self))
        except TypeError:
            if _is_sympy_operand(other):
                return NotImplemented
            raise

    def __mul__(self, other: Any) -> "Expr":
        try:
            return _wrap(_native_expr(self) * _native_expr(other))
        except TypeError:
            if _is_sympy_operand(other):
                return NotImplemented
            raise

    def __rmul__(self, other: Any) -> "Expr":
        try:
            return _wrap(_native_expr(other) * _native_expr(self))
        except TypeError:
            if _is_sympy_operand(other):
                return NotImplemented
            raise

    def __truediv__(self, other: Any) -> "Expr":
        try:
            reciprocal = _native.py_pow(_native_expr(other), _native.py_integer(-1))
            return _wrap(_native_expr(self) * reciprocal)
        except TypeError:
            if _is_sympy_operand(other):
                return NotImplemented
            raise

    def __rtruediv__(self, other: Any) -> "Expr":
        try:
            reciprocal = _native.py_pow(_native_expr(self), _native.py_integer(-1))
            return _wrap(_native_expr(other) * reciprocal)
        except TypeError:
            if _is_sympy_operand(other):
                return NotImplemented
            raise

    def __pow__(self, exponent: Any, modulo: Any = None) -> "Expr":
        if modulo is not None:
            raise TypeError("modular symbolic exponentiation is not implemented")
        # SymPy 1.14.0 power identities (bead
        # fra-shell-number-canonical-construction-qf6): the native lane keeps
        # Pow(x, 0) etc. applied, the oracle folds them at construction.
        base_int = _admitted_exact_int(self)
        exp_int = _admitted_exact_int(exponent)
        if exp_int is not None:
            if exp_int == 0:
                return _ONE
            if exp_int == 1:
                return self
            if base_int is not None:
                if base_int == 1:
                    return _ONE
                if base_int == 0:
                    if exp_int > 0:
                        return _ZERO
                    return zoo
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

    @classmethod
    def _from_intern(cls, name: str, number: int, value: Any = None) -> "Dummy":
        dummy = object.__new__(cls)
        dummy._dummy_name = name
        dummy._dummy_number = number
        dummy._assumptions = {}
        dummy._value = (
            value if value is not None else _native.py_symbol(_dummy_intern_name(name, number))
        )
        _note_dummy_number(number)
        return dummy


def _restore_dummy(name: str, number: int) -> Dummy:
    return Dummy._from_intern(name, number)


class Number(AtomicExpr):
    """Base class for exact numbers in the compatibility shell."""

    __slots__ = ()


class Rational(Number):
    __slots__ = ()

    def __new__(cls, numerator: int, denominator: int):
        if cls is Rational:
            # SymPy 1.14.0 construction semantics (bead
            # fra-shell-number-canonical-construction-qf6): zero denominator
            # is zoo; exact integers promote to Integer (routing the 0/1/-1
            # singletons); everything else normalizes sign into the numerator.
            numerator_p, numerator_q = _exact_rational_argument(numerator)
            denominator_p, denominator_q = _exact_rational_argument(denominator)
            num = numerator_p * denominator_q
            den = numerator_q * denominator_p
            if den == 0:
                return zoo
            common = math.gcd(abs(num), abs(den))
            if common:
                num //= common
                den //= common
            if den < 0:
                num, den = -num, -den
            if den == 1:
                return Integer(num)
            obj = object.__new__(cls)
            obj._value = _native.py_rational(num, den)
            return obj
        return object.__new__(cls)
    @property
    def p(self) -> int:
        return self._value.exact_numerator()

    @property
    def q(self) -> int:
        return self._value.exact_denominator()

    def __float__(self) -> float:
        # Truncating int division matches SymPy 1.14 float(Rational(-3, 2)).
        return float(self.p) / float(self.q)

    def __int__(self) -> int:
        return int(float(self))

    def __init__(self, *args: Any, **kwargs: Any):
        # __new__ builds the native value (or routes to a promoted singleton);
        # type(instance).__init__ receives the ORIGINAL caller args regardless
        # of which class __new__ returned, so this must accept anything.
        pass

    @property
    def p(self) -> int:
        return self._value.exact_numerator()

    @property
    def q(self) -> int:
        return self._value.exact_denominator()

    # Tri-valued assumption properties, pinned against SymPy 1.14.0 for
    # concrete rationals (gauntlet bead fra-shell-atom-assumptions-bypasses-7o3,
    # fresh-eyes finding 7): the native value is exact, so every answer is a
    # concrete True/False — never None.
    @property
    def is_positive(self) -> bool:
        return self.p > 0

    @property
    def is_negative(self) -> bool:
        return self.p < 0

    @property
    def is_zero(self) -> bool:
        return self.p == 0

    @property
    def is_nonnegative(self) -> bool:
        return self.p >= 0

    @property
    def is_nonpositive(self) -> bool:
        return self.p <= 0

    @property
    def is_real(self) -> bool:
        return True

    @property
    def is_integer(self) -> bool:
        return self.q == 1


class Integer(Rational):
    __slots__ = ()

    def __new__(cls, value: int):
        if cls is Integer:
            # Construction routing to the SymPy 1.14.0 integer singletons
            # (bead fra-shell-number-canonical-construction-qf6).
            v = _exact_integer_argument(value)
            if v == 0:
                return _ZERO
            if v == 1:
                return _ONE
            if v == -1:
                return _NEGATIVE_ONE
        obj = object.__new__(cls)
        obj._value = _native.py_integer(_exact_integer_argument(value))
        return obj

    def __init__(self, *args: Any, **kwargs: Any):
        # __new__ sets the native value (or routes to a promoted singleton);
        # type(instance).__init__ gets the ORIGINAL caller args regardless of
        # which class __new__ returned, so this must accept anything.
        pass

    @property
    def p(self) -> int:
        return self._value.exact_numerator()

    @property
    def q(self) -> int:
        return 1

class Zero(Integer):
    """The singleton integer zero (SymPy 1.14: sympy.core.numbers.Zero).

    Profile identity: type name ``Zero``, module ``sympy.core.numbers``,
    ``is_Zero`` True, repr ``0``. Arithmetic results are not required to stay
    in this class; the registry hands out this single instance.
    """

    __slots__ = ()
    is_Zero = True

    def __new__(cls):
        obj = object.__new__(cls)
        obj._value = _native.py_integer(0)
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __reduce__(self):
        return _restore_zero, ()

    def _srepr(self) -> str:
        # Oracle parity: sympy.srepr(S.Zero) == "Integer(0)".
        return "Integer(0)"


Zero.__module__ = "sympy.core.numbers"
_ZERO = Zero()


class One(Integer):
    """The singleton integer one (SymPy 1.14: sympy.core.numbers.One)."""

    __slots__ = ()
    is_One = True

    def __new__(cls):
        obj = object.__new__(cls)
        obj._value = _native.py_integer(1)
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __reduce__(self):
        return _restore_one, ()

    def _srepr(self) -> str:
        # Oracle parity: sympy.srepr(S.One) == "Integer(1)".
        return "Integer(1)"

One.__module__ = "sympy.core.numbers"
_ONE = One()


class NegativeOne(Integer):
    """The singleton integer minus one (SymPy 1.14: NegativeOne)."""

    __slots__ = ()
    is_NegativeOne = True

    def __new__(cls):
        obj = object.__new__(cls)
        obj._value = _native.py_integer(-1)
        return obj

    def __init__(self, *args, **kwargs):
        pass

    def __reduce__(self):
        return _restore_negative_one, ()

    def _srepr(self) -> str:
        # Oracle parity: sympy.srepr(S.NegativeOne) == "Integer(-1)".
        return "Integer(-1)"


NegativeOne.__module__ = "sympy.core.numbers"
_NEGATIVE_ONE = NegativeOne()


def _restore_zero():
    return _ZERO


def _restore_one():
    return _ONE


def _restore_negative_one():
    return _NEGATIVE_ONE

class Float(Number):
    """Profile-compatible binary64 float. Distinct from Rational and from RealBall."""

    __slots__ = ("_dps",)

    def __init__(self, value: Any = 0, dps: int = 15):
        if type(dps) is not int or dps < 1:
            raise TypeError("Float dps must be a positive int")
        self._dps = dps
        self._value = _float_intern(_ieee_bits(_admitted_python_float(value)))

    @classmethod
    def _from_native(cls, value: Any) -> "Float":
        obj = object.__new__(cls)
        obj._value = value
        obj._dps = 15
        return obj

    def _as_python_float(self) -> float:
        payload = self._value.args
        if len(payload) != 1:
            raise ValueError("malformed Float intern encoding")
        return _bits_to_float(payload[0].exact_numerator())

    # Tri-valued assumption properties pinned against SymPy 1.14.0: a nan
    # answers None for everything; every other binary64 value answers
    # concretely (Float(0.0).is_integer is True, Float(2.5).is_integer False).
    @property
    def is_positive(self) -> bool | None:
        value = self._as_python_float()
        return None if math.isnan(value) else value > 0

    @property
    def is_negative(self) -> bool | None:
        value = self._as_python_float()
        return None if math.isnan(value) else value < 0

    @property
    def is_zero(self) -> bool | None:
        value = self._as_python_float()
        return None if math.isnan(value) else value == 0.0

    @property
    def is_real(self) -> bool | None:
        return None if math.isnan(self._as_python_float()) else True

    @property
    def is_nonnegative(self) -> bool | None:
        value = self._as_python_float()
        return None if math.isnan(value) else value >= 0

    @property
    def is_nonpositive(self) -> bool | None:
        value = self._as_python_float()
        return None if math.isnan(value) else value <= 0
    @property
    def is_integer(self) -> bool | None:
        value = self._as_python_float()
        return None if math.isnan(value) else value.is_integer()

    @property
    def dps(self) -> int:
        return self._dps

    @property
    def args(self) -> tuple:
        return ()

    @property
    def func(self):
        return Float

    @property
    def is_number(self) -> bool:
        return True

    @property
    def is_rational(self) -> bool:
        return False

    @property
    def is_symbol(self) -> bool:
        return False

    def evalf(self, n: int = 15) -> "Float":
        if type(n) is not int or n < 1:
            raise TypeError("evalf dps must be a positive int")
        if n == self._dps:
            return self
        return Float(self._as_python_float(), n)

    def __float__(self) -> float:
        return self._as_python_float()

    def __int__(self) -> int:
        # Truncates toward zero like SymPy 1.14; nan/inf raise via float().
        return int(self._as_python_float())

    def __abs__(self) -> "Float":
        return Float(abs(self._as_python_float()), self._dps)

    def __str__(self) -> str:
        # Oracle str(Float('0.75')) == '0.750000000000000' (15 fixed digits).
        return _str_float_value(self._as_python_float())

    def __repr__(self) -> str:
        return f"Float({self._as_python_float()!r})"

    def _srepr(self) -> str:
        return f"Float({self._as_python_float()!r})"

    def __add__(self, other: Any) -> "Expr":
        rhs = _maybe_python_float(other)
        if rhs is not None:
            return Float(self._as_python_float() + rhs)
        return Expr.__add__(self, other)

    def __radd__(self, other: Any) -> "Expr":
        rhs = _maybe_python_float(other)
        if rhs is not None:
            return Float(rhs + self._as_python_float())
        return Expr.__radd__(self, other)

    def __sub__(self, other: Any) -> "Expr":
        rhs = _maybe_python_float(other)
        if rhs is not None:
            return Float(self._as_python_float() - rhs)
        return Expr.__sub__(self, other)

    def __rsub__(self, other: Any) -> "Expr":
        rhs = _maybe_python_float(other)
        if rhs is not None:
            return Float(rhs - self._as_python_float())
        return Expr.__rsub__(self, other)

    def __mul__(self, other: Any) -> "Expr":
        rhs = _maybe_python_float(other)
        if rhs is not None:
            return Float(self._as_python_float() * rhs)
        return Expr.__mul__(self, other)

    def __rmul__(self, other: Any) -> "Expr":
        rhs = _maybe_python_float(other)
        if rhs is not None:
            return Float(rhs * self._as_python_float())
        return Expr.__rmul__(self, other)

    def __truediv__(self, other: Any) -> "Expr":
        rhs = _maybe_python_float(other)
        if rhs is not None:
            return Float(self._as_python_float() / rhs)
        return Expr.__truediv__(self, other)

    def __rtruediv__(self, other: Any) -> "Expr":
        rhs = _maybe_python_float(other)
        if rhs is not None:
            return Float(rhs / self._as_python_float())
        return Expr.__rtruediv__(self, other)


def _restore_float(value: float, dps: int) -> Float:
    return Float(value, dps)


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


class Relational(Expr):
    """Held comparison. Not a Boolean proof and not a mathematical order."""

    __slots__ = ()
    rel_op = "=="

    def __init__(self, lhs: Any, rhs: Any):
        self._value = _native.py_function(type(self).__name__, _native_expr(lhs), _native_expr(rhs))

    @property
    def lhs(self) -> Basic:
        return self.args[0]

    @property
    def rhs(self) -> Basic:
        return self.args[1]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.lhs!r}, {self.rhs!r})"

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.lhs}, {self.rhs})"


class Eq(Relational):
    rel_op = "=="


class Ne(Relational):
    rel_op = "!="


class Lt(Relational):
    rel_op = "<"


class Le(Relational):
    rel_op = "<="


class Gt(Relational):
    rel_op = ">"


class Ge(Relational):
    rel_op = ">="



class _SingletonRegistry:
    """Exact-atom registry. ``S(float)`` constructs compatibility ``Float``."""

    @property
    def Zero(self) -> "Zero":
        return _ZERO

    @property
    def One(self) -> "One":
        return _ONE

    @property
    def NegativeOne(self) -> "NegativeOne":
        return _NEGATIVE_ONE

    @property
    def Half(self) -> "Rational":
        return Rational(1, 2)

    @property
    def Infinity(self) -> Expr:
        return Expr("oo")

    @property
    def NegativeInfinity(self) -> Expr:
        return Expr("-oo")

    @property
    def ComplexInfinity(self) -> ComplexInfinity:
        return zoo
    @property
    def NaN(self) -> Expr:
        return Expr("nan")

    @property
    def Pi(self) -> Expr:
        return Expr("pi")

    @property
    def Exp1(self) -> Expr:
        return Expr("E")

    @property
    def ImaginaryUnit(self) -> Expr:
        return Expr("I")

    def __call__(self, value: Any) -> Basic:
        if isinstance(value, Basic):
            return value
        if type(value) is bool:
            return Integer(1 if value else 0)
        if type(value) is int:
            return Integer(value)
        if type(value) is float:
            return Float(value)
        raise TypeError(f"cannot convert {type(value).__name__} through S")


S = _SingletonRegistry()




class AssocOp(Basic):
    """Associative operator base (SympPy 1.14: sympy.core.operations.AssocOp).

    Present so the shell MRO matches the pinned oracle's for Add/Mul; the
    oracle MRO is Add -> Expr -> AssocOp -> Basic.
    """

    __slots__ = ()


AssocOp.__module__ = "sympy.core.operations"


class Add(Expr, AssocOp):
    __slots__ = ("_args",)

    def __new__(cls, *args: Any, evaluate: bool = True):
        native_args = [_native_expr(arg) for arg in args]
        val = _native.Add(*native_args, evaluate=evaluate).as_expr()
        if evaluate:
            result = _wrap(val)
            if isinstance(result, Add) and len(result.args) == len(args):
                # The native kernel reorders non-combining terms; the oracle
                # preserves insertion order for them (Add(-8, w).args ==
                # (-8, w)). When no folding happened (same multiset of args),
                # surface the caller's order over the native value.
                given = [a if isinstance(a, Basic) else _wrap(_native_expr(a)) for a in args]
                remaining = list(result.args)
                matched = True
                for g in given:
                    for i, r in enumerate(remaining):
                        if g == r:
                            remaining.pop(i)
                            break
                    else:
                        matched = False
                        break
                if matched:
                    obj = object.__new__(cls)
                    obj._args = tuple(given)
                    obj._value = result._value
                    return obj
            return result
        obj = object.__new__(cls)
        # Held forms preserve the caller's argument order exactly (oracle:
        # Add(-8, w, evaluate=False).args == (-8, w)); the native lane may
        # reorder, so the surface args come from the construction tuple.
        obj._args = tuple(a if isinstance(a, Basic) else _wrap(_native_expr(a)) for a in args)
        obj._value = val
        return obj

    def __init__(self, *args: Any, evaluate: bool = True):
        pass

    @property
    def args(self) -> tuple["Basic", ...]:
        if hasattr(self, "_args"):
            return self._args
        return super().args

    def __neg__(self) -> "Expr":
        # Profile-correct vs SymPy 1.14.0 Add.__neg__: distribute over terms.
        # -(x - 1) must canonicalize to (1, -x), never an unfolded Mul(-1, Add).
        return Add(*(-arg for arg in self.args))


class Mul(Expr, AssocOp):
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

    def __neg__(self) -> "Expr":
        # Profile-correct vs SymPy 1.14.0 Mul.__neg__: flip the leading Number
        # factor; a leading zoo self-negates (zoo == -zoo); otherwise prepend -1.
        args = self.args
        if args and args[0] is zoo:
            return self
        if args and isinstance(args[0], Number):
            return Mul(-args[0], *args[1:])
        return Mul(Integer(-1), *args)


class Pow(Expr):
    __slots__ = ()

    def __new__(cls, base: Any, exponent: Any, evaluate: bool = True):
        if evaluate:
            # SymPy 1.14.0 construction folds (bead
            # fra-shell-number-canonical-construction-qf6): 1**x -> 1,
            # x**0 -> 1, 0**0 -> 1, 0**positive -> 0, 0**negative -> zoo.
            base_int = _admitted_exact_int(base)
            exp_int = _admitted_exact_int(exponent)
            if base_int is not None and exp_int is not None:
                if base_int == 1:
                    return _ONE
                if exp_int == 0:
                    return _ONE
                if base_int == 0:
                    if exp_int > 0:
                        return _ZERO
                    if exp_int < 0:
                        return zoo
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
            if name == _FLOAT_INTERN or name.startswith(_DUMMY_PREFIX):
                raise ValueError("Function name collides with native intern encoding")
            return UndefinedFunction(name)

        # Sympify raw Python values first (fresh-eyes finding 3): upstream
        # sympifies every argument before dispatch, so user eval hooks and
        # the native lane both see shell expressions, never raw ints/floats.
        sympified = tuple(
            a
            if isinstance(a, Basic)
            else _wrap(_native_expr(a))
            if isinstance(a, (int, float, str))
            else a
            for a in args
        )
        evaluate = options.get("evaluate", True)

        # Classmethod eval hook — skipped for evaluate=False (finding 8).
        eval_method = getattr(cls, "eval", None)
        if eval_method is not None and evaluate:
            evaluated = eval_method(*sympified)
            if evaluated is not None:
                return _wrap(_native_expr(evaluated))

        obj = object.__new__(cls)
        obj._args = sympified
        name = cls.__name__
        native_args = [_native_expr(arg) for arg in sympified]
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


def _machin_pi_floor(total_digits: int) -> int:
    """floor(pi * 10**total_digits) via Machin's formula in pure integer
    arithmetic with 15 guard digits (bead
    fra-fra-native-evalf-precision-honesty-ke8)."""
    guard = 15
    scale = 10 ** (total_digits + guard)
    def atan_inv(x: int) -> int:
        total = 0
        k = 0
        term = scale // x
        x2 = x * x
        while term != 0:
            total += term if k % 2 == 0 else -term
            k += 1
            term = scale // ((2 * k + 1) * x * x2 ** k)
        return total
    return (16 * atan_inv(5) - 4 * atan_inv(239)) // 10 ** guard


_MAX_HONEST_DIGITS = 10_000
_F64_HONEST_DIGITS = 15


class _ExactDecimalFloat(Float):
    """A Float carrying its exact decimal rendering (precision-honest N of
    symbolic constants). Arithmetic re-enters the binary64 backend, so the
    exact digits describe THIS value only — matching the documented
    not-a-certified-enclosure contract."""

    __slots__ = ("_decimal",)

    def __init__(self, decimal: str, dps: int):
        self._decimal = decimal
        self._dps = dps
        self._value = _float_intern(_ieee_bits(_admitted_python_float(float(decimal))))

    def __str__(self) -> str:
        return self._decimal

    def __repr__(self) -> str:
        return self._decimal

    def _srepr(self) -> str:
        return f"Float('{self._decimal}', precision={self._dps})"


_ExactDecimalFloat.__name__ = "Float"
_ExactDecimalFloat.__qualname__ = "Float"
_ExactDecimalFloat.__module__ = "sympy.core.numbers"


def N(expression: Any, n: Any = None) -> Basic:
    """Evaluate to a compatibility Float. Precision-honest (bead
    fra-fra-native-evalf-precision-honesty-ke8): ``N(pi, d)`` computes the
    true decimal digits via a Machin series for 1 <= d <= 10_000; every other
    expression beyond the binary64-honest 15 significant digits raises
    NotImplementedError naming the limit — an f64 is never extended with
    unjustified digits. Not a certified enclosure."""
    explicit = n is not None
    if not explicit:
        n = _F64_HONEST_DIGITS
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"N precision must be an integer, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"N precision must be >= 1, got {n}")
    is_pi = isinstance(expression, Expr) and str(expression) == "pi"
    if is_pi:
        if n > _MAX_HONEST_DIGITS:
            raise NotImplementedError(
                f"precision-honest N: pi is computed via a bounded Machin "
                f"series supporting at most {_MAX_HONEST_DIGITS} significant "
                f"digits; requested {n}"
            )
        # Upstream convention (pinned oracle): d significant digits, ROUNDED.
        m = _machin_pi_floor(n)
        rounded = (m + 5) // 10
        digits = str(rounded)
        if len(digits) != n:
            raise ValueError(f"pi digit stream length {len(digits)} != {n}")
        decimal = digits[0] + "." + digits[1:]
        return _ExactDecimalFloat(decimal, n)
    if isinstance(expression, Basic) and n > _F64_HONEST_DIGITS:
        raise NotImplementedError(
            f"precision-honest N: this shell evaluates through binary64 and "
            f"is honest to at most {_F64_HONEST_DIGITS} significant digits; "
            f"requested {n}. Refusing to emit unjustified digits."
        )
    if isinstance(expression, Basic):
        return expression.evalf(n)
    if n > _F64_HONEST_DIGITS:
        raise NotImplementedError(
            f"precision-honest N: this shell evaluates through binary64 and "
            f"is honest to at most {_F64_HONEST_DIGITS} significant digits; "
            f"requested {n}. Refusing to emit unjustified digits."
        )
    return Float(_admitted_python_float(expression), n)


Basic.__module__ = "sympy.core.basic"
Atom.__module__ = "sympy.core.basic"
Expr.__module__ = "sympy.core.expr"
AtomicExpr.__module__ = "sympy.core.expr"
Symbol.__module__ = "sympy.core.symbol"
Dummy.__module__ = "sympy.core.symbol"
Number.__module__ = "sympy.core.numbers"
Rational.__module__ = "sympy.core.numbers"
Integer.__module__ = "sympy.core.numbers"
Float.__module__ = "sympy.core.numbers"
Relational.__module__ = "sympy.core.relational"
Eq.__module__ = "sympy.core.relational"
Ne.__module__ = "sympy.core.relational"
Lt.__module__ = "sympy.core.relational"
Le.__module__ = "sympy.core.relational"
Gt.__module__ = "sympy.core.relational"
Ge.__module__ = "sympy.core.relational"
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
_SingletonRegistry.__module__ = "sympy.core.singleton"
S.__module__ = "sympy.core.singleton"
_restore_nary.__module__ = "sympy.core.basic"
_restore_pow.__module__ = "sympy.core.basic"
_restore_dummy.__module__ = "sympy.core.basic"
_restore_applied_undef.__module__ = "sympy.core.basic"
_restore_float.__module__ = "sympy.core.numbers"

import types as _types

for _mod_name, _mod_items in [
    ("sympy.core.basic", (Basic, Atom, _restore_nary, _restore_pow, _restore_dummy, _restore_applied_undef)),
    ("sympy.core.expr", (Expr, AtomicExpr)),
    ("sympy.core.symbol", (Symbol, Dummy, symbols)),
    ("sympy.core.numbers", (Number, Rational, Integer, Zero, One, NegativeOne, Float, ComplexInfinity, _restore_float, _restore_zero, _restore_one, _restore_negative_one)),
    ("sympy.core.relational", (Relational, Eq, Ne, Lt, Le, Gt, Ge)),
    ("sympy.core.add", (Add,)),
    ("sympy.core.mul", (Mul,)),
    ("sympy.core.operations", (AssocOp,)),
    ("sympy.core.power", (Pow,)),
    ("sympy.core.function", (Function, UndefinedFunction, AppliedUndef, Derivative, Application, FunctionClass, diff)),
]:
    _mod = sys.modules.get(_mod_name)
    if _mod is None:
        _mod = _types.ModuleType(_mod_name)
        sys.modules[_mod_name] = _mod
    for _item in _mod_items:
        setattr(_mod, getattr(_item, "__name__", str(_item)), _item)

_singleton_mod = sys.modules.get("sympy.core.singleton")
if _singleton_mod is None:
    _singleton_mod = _types.ModuleType("sympy.core.singleton")
    sys.modules["sympy.core.singleton"] = _singleton_mod
_singleton_mod.S = S


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
    "Eq",
    "Expr",
    "Float",
    "Function",
    "FunctionClass",
    "Ge",
    "Gt",
    "Integer",
    "Le",
    "Lt",
    "Mul",
    "Ne",
    "N",
    "Number",
    "Pow",
    "Rational",
    "S",
    "Symbol",
    "UndefinedFunction",
    "diff",
    "expand",
    "pretty",
    "simplify",
    "symbols",
    "zoo",
]
