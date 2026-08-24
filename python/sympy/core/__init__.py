"""SymPy core compatibility module powered by FrankenSymPy."""

try:
    import fsym_python as _native
except ImportError:
    _native = None

if _native is not None:
    Expr = _native.Expr
    Symbol = _native.Symbol
    Integer = _native.Integer
    Rational = _native.Rational
    Add = _native.Add
    Mul = _native.Mul
    Pow = _native.Pow
    Derivative = _native.Derivative
    diff = _native.diff_expr
    expand = _native.expand_expr
    simplify = _native.simplify_expr
else:
    class Expr:
        pass
    class Symbol(Expr):
        pass
    class Integer(Expr):
        pass
    class Rational(Expr):
        pass
    class Add(Expr):
        pass
    class Mul(Expr):
        pass
    class Pow(Expr):
        pass
    class Derivative(Expr):
        pass

def symbols(names, **kwargs):
    """Create multiple symbols from space/comma-separated string or sequence."""
    if isinstance(names, str):
        parts = [p.strip() for p in names.replace(',', ' ').split() if p.strip()]
        syms = [Symbol(p) for p in parts]
        return tuple(syms) if len(syms) > 1 else syms[0]
    return tuple(Symbol(n) for n in names)

__all__ = [
    "Expr", "Symbol", "Integer", "Rational", "Add", "Mul", "Pow", "Derivative",
    "symbols", "diff", "expand", "simplify",
]
