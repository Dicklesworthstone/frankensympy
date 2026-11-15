"""Lambda and Subs (upstream sympy.core.function).

Registered into the synthetic sympy.core.function module by
sympy/__init__.py so upstream import paths resolve.
"""

from __future__ import annotations

from typing import Any

from . import Basic, Dummy, Expr, Function, Symbol, Tuple, _charge_nested_if_active, _native


class BadSignatureError(TypeError):
    """A Lambda declaration has duplicate or non-symbol parameters."""


class BadArgumentsError(TypeError):
    """Arguments do not have the declared Lambda tuple shape."""


_MAX_BINDING_NODES = 4096
_MAX_BINDING_DEPTH = 64
_VALUE = Basic.__dict__["_value"]
_FUNCTION_ARGS = Function.__dict__["_args"]


def _tuple_items(value: Any, *, lists: bool = False) -> tuple | None:
    """Inspect built-in containers without accepting a same-named class."""
    if type(value) is tuple:
        if len(value) > _MAX_BINDING_NODES:
            raise ValueError("binding tuple exceeds its work/depth limit")
        return value
    if lists and type(value) is list:
        if len(value) > _MAX_BINDING_NODES:
            raise ValueError("binding tuple exceeds its work/depth limit")
        return tuple(value)
    if type(value) is Tuple:
        children = _FUNCTION_ARGS.__get__(value, Function)
        if type(children) is not tuple:
            raise ValueError("binding tuple has invalid retained arguments")
        if len(children) > _MAX_BINDING_NODES:
            raise ValueError("binding tuple exceeds its work/depth limit")
        return children
    return None


def _lambda_signature(signature: Any) -> tuple[Any, tuple[Symbol, ...]]:
    """Build an immutable, bounded signature using actual typed symbols.

    The symbolic Tuple is part of the surface graph. A raw Python tuple in
    args would hide its parameters from structural replacement and make
    nested Lambdas unusable as retained expression children.
    """
    from . import sympify

    remaining = _MAX_BINDING_NODES
    active: set[int] = set()
    seen = set()
    variables = []

    def visit(value, depth):
        nonlocal remaining
        remaining -= 1
        _charge_nested_if_active(1)
        if remaining < 0 or depth > _MAX_BINDING_DEPTH:
            raise ValueError("Lambda signature exceeds its work/depth limit")
        children = _tuple_items(value, lists=True)
        if children is not None:
            if id(value) in active:
                raise ValueError("cyclic Lambda signature")
            if len(children) > remaining:
                raise ValueError("Lambda signature exceeds its work/depth limit")
            active.add(id(value))
            result = Tuple(*(visit(child, depth + 1) for child in children))
            active.remove(id(value))
            return result
        if type(value) in (str, int, float, bool):
            value = sympify(value)
        if type(value) not in (Symbol, Dummy):
            raise BadSignatureError("Lambda signature should contain only tuples and symbols")
        native = _VALUE.__get__(value, Basic)
        if type(native) is not _native.Expr or not native.is_symbol:
            raise ValueError("Lambda signature has an invalid typed symbol")
        key = (type(value), native)
        if key in seen:
            raise BadSignatureError("Duplicate symbol in Lambda signature")
        seen.add(key)
        variables.append(value)
        return value

    root = signature if _tuple_items(signature, lists=True) is not None else (signature,)
    return visit(root, 0), tuple(variables)


class Lambda(Expr):
    """Lambda(x, expr) / Lambda((x, y), expr): an anonymous function."""

    __slots__ = ()
    is_Function = True

    def __new__(cls, signature: Any, expr: Any):
        from . import sympify

        sig, _ = _lambda_signature(signature)
        obj = object.__new__(cls)
        obj._struct_args = (sig, sympify(expr))
        return obj

    @property
    def signature(self):
        return self._struct_args[0]

    @property
    def variables(self):
        return _lambda_signature(self.signature)[1]

    bound_symbols = variables

    @property
    def nargs(self):
        from ..sets import FiniteSet

        return FiniteSet(len(_tuple_items(self.signature)))

    @property
    def expr(self):
        return self._struct_args[1]

    @property
    def free_symbols(self) -> set:
        return set(self.expr.free_symbols) - set(self.variables)

    def __call__(self, *args: Any) -> Any:
        signature, _ = _lambda_signature(self.signature)
        parameters = _tuple_items(signature)
        if len(args) != len(parameters):
            count = len(parameters)
            raise BadArgumentsError(
                f"{self} takes exactly {count} argument{'s' if count != 1 else ''} "
                f"({len(args)} given)"
            )
        replacements = {}

        def match(parameters, values):
            if len(parameters) != len(values):
                raise BadArgumentsError("Lambda arguments do not match its nested tuple signature")
            for parameter, value in zip(parameters, values):
                children = _tuple_items(parameter)
                if children is None:
                    replacements[parameter] = value
                else:
                    supplied = _tuple_items(value)
                    if supplied is None:
                        raise BadArgumentsError("Lambda arguments do not match its nested tuple signature")
                    match(children, supplied)

        match(parameters, args)
        # Lambda application in the pinned profile is structural and
        # simultaneous: a replacement is never consumed by a later parameter.
        # In particular this must not dispatch algebraic _eval_subs hooks or
        # replace the sequential substitutions used by the distinct Subs API.
        return self.expr.xreplace(replacements)

    def _eval_subs(self, old: Any, new: Any):
        signature = self.signature.subs(old, new)
        expr = self.expr.subs(old, new)
        if signature == self.signature and expr is self.expr:
            return self
        return self.func(signature, expr)

    def __str__(self) -> str:
        def render(signature):
            items = _tuple_items(signature)
            if items is None:
                return str(signature)
            return "(" + ", ".join(render(item) for item in items) + ("," if len(items) == 1 else "") + ")"

        sig = _tuple_items(self.signature)
        sig_s = (render(sig[0]) if len(sig) == 1 and _tuple_items(sig[0]) is None
                 else render(self.signature))
        return f"Lambda({sig_s}, {self.expr})"

    __repr__ = __str__


class Subs(Function):
    """Unevaluated substitution with retained expression, variables and points."""

    __slots__ = ()

    def __new__(cls, expr: Any, variables: Any, point: Any, **options: Any):
        from . import sympify

        def items(v: Any) -> tuple:
            values = _tuple_items(v, lists=True)
            values = (v,) if values is None else values
            if len(values) > _MAX_BINDING_NODES:
                raise ValueError("Subs exceeds its binding limit")
            return values

        vs = tuple(sympify(v) for v in items(variables))
        ps = tuple(sympify(p) for p in items(point))
        if len(vs) != len(ps):
            raise ValueError("Number of point values must be the same as the number of variables.")
        if len(set(vs)) != len(vs):
            raise ValueError("The following expressions appear more than once in Subs variables")
        expr = sympify(expr)
        if not vs:
            return expr
        if type(expr) is Subs:
            vs, ps = expr.variables + vs, expr.point + ps
            if len(vs) > _MAX_BINDING_NODES:
                raise ValueError("Subs exceeds its binding limit")
            expr = expr.expr
        # Even an initially absent variable can occur in an earlier point.
        # Retain the full ordered program and every point's free symbols.
        return Function.__new__(
            cls, expr, Tuple(*vs), Tuple(*ps), evaluate=False
        )

    @property
    def expr(self):
        return self.args[0]

    @property
    def variables(self):
        return tuple(self.args[1].args)

    @property
    def point(self):
        return tuple(self.args[2].args)

    @property
    def free_symbols(self) -> set:
        syms = set(self.expr.free_symbols) - set(self.variables)
        for p in self.point:
            syms |= set(getattr(p, "free_symbols", set()))
        return syms

    def _eval_subs(self, old: Any, new: Any):
        variables, points = self.variables, list(self.point)
        if old in variables:
            # A fresh atomic spelling may rename this bound coordinate. If
            # the replacement is already in the program it instead affects
            # this point and subsequent points, preserving execution order.
            if not new.args and not any(arg.has(new) for arg in self.args):
                return self.xreplace({old: new})
            start = variables.index(old)
            for index in range(start, len(points)):
                points[index] = points[index].subs(old, new)
            return self.func(self.expr, variables, points)
        changed = tuple(variable.subs(old, new) for variable in variables)
        if changed != variables:
            return self.func(self.expr, (*variables, old), (*points, new))
        return self.func(
            self.expr.subs(old, new), variables,
            tuple(point.subs(old, new) for point in points),
        )

    def doit(self, **hints: Any) -> Any:
        e = self.expr
        if hints.get("deep", True) and hasattr(e, "doit"):
            e = e.doit(**hints)
        # Unlike Lambda, Subs executes in the declared order. A dict sorts
        # by expression complexity/name and changes overlapping replacements.
        return e.subs(list(zip(self.variables, self.point)))

    def __str__(self) -> str:
        def tup(t):
            return str(t[0]) if len(t) == 1 else "(" + ", ".join(str(a) for a in t) + ")"
        return f"Subs({self.expr}, {tup(self.variables)}, {tup(self.point)})"

    __repr__ = __str__
