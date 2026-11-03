"""Bounded differentiation of expressions retaining opaque Python objects.

Closed regions still use native differentiation. Product, sum, power and
built-in chain rules compose their results around the original applications.
User hooks run through the existing supervised bridge. An unresolved custom
application remains a formal derivative of that actual object, with no name
lookup or invented native function identity and no verified-proof claim.
"""

from __future__ import annotations

from . import (
    Add, Basic, Derivative, Dummy, Function, Mul, Pow, Symbol,
    _ZERO, _ONE, _charge_nested_if_active, _diff_custom_node,
    _diff_single_supervised, _lift_native_result, _native, _native_expr,
    _node_has_custom_diff_hook, _surface_binding_key, _symbol_binding,
)
from ._opaque_arithmetic import contains_opaque, stored_args

_VALUE = Basic.__dict__["_value"]
_MAX_WORK = 16_384
_MAX_DEPTH = 128


class _Work:
    __slots__ = ("remaining",)

    def __init__(self):
        self.remaining = _MAX_WORK

    def charge(self, amount=1):
        if amount < 0 or amount > self.remaining:
            raise ValueError("surface differentiation exceeds its work limit")
        _charge_nested_if_active(amount)
        self.remaining -= amount


def _children(node):
    children = stored_args(node)
    if children is not None:
        return children
    return Basic.__dict__["args"].__get__(node, Basic)


def _is_zero(value):
    return value is _ZERO


def _depends_on(node, symbol, work):
    """Read actual children and typed native bindings, never a custom getter."""
    target = _surface_binding_key(_symbol_binding(symbol))
    stack = [node]
    while stack:
        current = stack.pop()
        work.charge()
        if not isinstance(current, Basic):
            continue
        if type(current) is Derivative:
            stack.append(current.expr)
            continue
        children = stored_args(current)
        if children is not None:
            stack.extend(children)
            continue
        native = _VALUE.__get__(current, Basic)
        if type(native) is not _native.Expr:
            raise NotImplementedError("formal differentiation requires retained source children")
        if any(_surface_binding_key(binding) == target for binding in native.free_symbol_bindings):
            return True
    return False


def _replace_template(node, replacements, work, depth=0):
    """Restore a closed built-in template without reconstructing user classes."""
    work.charge()
    if depth > _MAX_DEPTH:
        raise ValueError("surface differentiation exceeds its depth limit")
    for dummy, original in replacements:
        if type(node) is Dummy and node == dummy:
            return original
    children = _children(node)
    if not children:
        return node
    restored = tuple(_replace_template(child, replacements, work, depth + 1)
                     for child in children)
    if all(a is b for a, b in zip(children, restored)):
        return node
    cls = type(node)
    if cls is Add or cls is Mul or cls is Pow:
        return cls(*restored)
    if cls is Derivative:
        return Derivative(*restored, evaluate=False)
    # Only registered intrinsic templates are produced by this path. Do not
    # lift a user Function from a name or run its constructor during restore.
    from ..utilities.lambdify import _functions

    if any(cls is builtin for builtin in _functions()):
        return cls(*restored)
    raise NotImplementedError("native derivative template contains an unsupported surface operator")


def differentiate(expression, symbol):
    """Differentiate one retained expression inside its owning callback scope."""
    work = _Work()

    def visit(node, depth=0):
        if _node_has_custom_diff_hook(node):
            # Hook admission owns cancellation/unknown-effect precedence.
            # Do not mask its precise terminal outcome with a speculative
            # nested-native-work charge before the requested invocation.
            return _diff_custom_node(node, symbol)
        work.charge()
        if depth > _MAX_DEPTH:
            raise ValueError("surface differentiation exceeds its depth limit")
        if not contains_opaque(node):
            return _diff_single_supervised(node, symbol)
        if not _depends_on(node, symbol, work):
            return _ZERO
        cls = type(node)
        if cls is Derivative:
            return Derivative(node.expr, *node.variables, symbol, evaluate=False)
        children = _children(node)
        if cls is Add:
            return Add(*(visit(child, depth + 1) for child in children))
        if cls is Mul:
            work.charge(len(children) * len(children))
            terms = []
            for index, child in enumerate(children):
                derivative = visit(child, depth + 1)
                if not _is_zero(derivative):
                    terms.append(Mul(*children[:index], derivative, *children[index + 1:]))
            return Add(*terms)
        if cls is Pow:
            base, exponent = children
            d_base, d_exponent = visit(base, depth + 1), visit(exponent, depth + 1)
            if _is_zero(d_exponent):
                return _ZERO if _is_zero(d_base) else exponent * Pow(base, exponent - _ONE) * d_base
            if _is_zero(base):
                # log(0) cannot justify the variable-exponent chain rule.
                # Keep the original singular source as a formal derivative.
                return Derivative(node, symbol, evaluate=False)
            from ..functions.elementary.exponential import log

            terms = [d_exponent * log(base)]
            if not _is_zero(d_base):
                terms.append(exponent * d_base / base)
            return node * Add(*terms)
        if isinstance(node, Function):
            from ..utilities.lambdify import _functions

            intrinsic = next((name for builtin, (name, _) in _functions().items()
                              if cls is builtin), None)
            if intrinsic is None:
                # This preserves unresolved composition without fabricating a
                # custom application with dummy arguments or running __new__.
                return Derivative(node, symbol, evaluate=False)
            dummies = tuple(Dummy("surface_chain") for _ in children)
            replacements = tuple(zip(dummies, children))
            native = _native.py_function(type.__dict__["__name__"].__get__(cls),
                                         *(_native_expr(dummy) for dummy in dummies))
            parts = []
            for dummy, child in replacements:
                d_child = visit(child, depth + 1)
                if _is_zero(d_child):
                    continue
                work.charge(len(children) + 1)
                partial = _lift_native_result(native.diff(_symbol_binding(dummy)), *dummies)
                restored = _replace_template(partial, replacements, work)
                parts.append(restored * d_child)
            return Add(*parts)
        raise NotImplementedError("surface differentiation requires an admitted operator or supervised hook")

    result = visit(expression)
    # Validate the complete output before returning it; malformed/cyclic hook
    # results and excessive retained graphs cannot escape as partial success.
    contains_opaque(result)
    return result
