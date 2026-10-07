"""Common subexpression elimination (upstream ``cse``).

Two passes, as upstream's ``opt_cse``/``tree_cse``:

1. common arguments: two ``Add`` (or two ``Mul``) nodes sharing two or
   more arguments get that shared sub-sum (sub-product) factored into a
   placeholder (``exp(x*y)`` and ``2*x*y`` share ``x*y``);
2. every non-atomic subexpression seen at least twice is replaced by the
   next symbol, inner subexpressions first (postorder).

Each replacement is exact: substituting the definitions back in order
reproduces the inputs.
"""

from __future__ import annotations

from typing import Any, Iterator


def numbered_symbols(prefix: str = "x", cls: Any = None, start: int = 0, exclude: Any = (), *args: Any, **assumptions: Any) -> Iterator[Any]:
    """Infinite generator of symbols ``prefix0, prefix1, ...``."""
    from ..core import Symbol

    cls = cls or Symbol
    exclude = set(exclude or ())
    i = start
    while True:
        name = "%s%d" % (prefix, i)
        s = cls(name, *args, **assumptions)
        if s not in exclude:
            yield s
        i += 1


def _is_atom(e: Any) -> bool:
    return not getattr(e, "args", ())


def _common_args(exprs: list) -> tuple:
    """Factor shared argument sets of Add/Mul nodes into placeholders."""
    from ..core import Add, Dummy, Mul
    from ..core.match import preorder_traversal

    placeholders: dict = {}
    for op in (Mul, Add):
        nodes = []
        for e in exprs:
            for n in preorder_traversal(e):
                if isinstance(n, op) and n not in nodes:
                    nodes.append(n)
        commons: list = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                inter = [a for a in nodes[i].args if a in nodes[j].args]
                if len(inter) >= 2 and len(inter) < max(len(nodes[i].args), len(nodes[j].args)):
                    key = frozenset(inter)
                    if key not in commons:
                        commons.append(key)
        commons.sort(key=len, reverse=True)
        for key in commons:
            common = op(*key)
            d = Dummy("cse")
            placeholders[d] = common

            def rebuild(e: Any) -> Any:
                args = getattr(e, "args", ())
                if not args:
                    return e
                new_args = [rebuild(a) for a in args]
                if isinstance(e, op) and all(k in new_args for k in key):
                    rest = [a for a in new_args if a not in key]
                    return op(d, *rest) if rest else d
                if new_args != list(args):
                    try:
                        return e.func(*new_args)
                    except Exception:
                        return e
                return e

            exprs = [rebuild(e) for e in exprs]
    return exprs, placeholders


def cse(exprs: Any, symbols: Any = None, optimizations: Any = None, postprocess: Any = None,
        order: str = "canonical", ignore: Any = (), list: bool = True) -> tuple:
    """Return ``(replacements, reduced_exprs)``."""
    from ..core import sympify
    from ..core.match import postorder_traversal

    import builtins

    single = not isinstance(exprs, (builtins.list, tuple))
    items = [sympify(exprs)] if single else [sympify(e) for e in exprs]
    if symbols is None:
        symbols = numbered_symbols("x", exclude=set().union(*[e.free_symbols for e in items]))
    else:
        symbols = iter(symbols)

    work, placeholders = _common_args(items)

    # Upstream tree_cse: a node seen again is repeated and its children
    # are not revisited (they are only counted through distinct parents).
    seen: set = set()
    repeated_set: set = set()

    def find(e: Any) -> None:
        if _is_atom(e):
            if e in placeholders:
                if e in seen:
                    repeated_set.add(e)
                seen.add(e)
            return
        if any(e.has(i) for i in ignore):
            return
        if e in seen:
            repeated_set.add(e)
            return
        seen.add(e)
        for a in e.args:
            find(a)

    for e in work:
        find(e)
    order_seen: Any = builtins.list()
    for e in work:
        for n in postorder_traversal(e):
            if n in repeated_set and n not in order_seen:
                order_seen.append(n)
    counts = {n: 2 for n in repeated_set}

    repeated = [n for n in order_seen if counts.get(n, 0) > 1]
    replacements: Any = builtins.list()
    mapping: dict = {}

    def sub(e: Any) -> Any:
        if e in mapping:
            return mapping[e]
        args = getattr(e, "args", ())
        if not args:
            if e in placeholders:
                return sub(placeholders[e])
            return e
        new = [sub(a) for a in args]
        if new != [a for a in args]:
            try:
                return e.func(*new)
            except Exception:
                return e
        return e

    for n in repeated:
        value = sub(placeholders[n]) if n in placeholders else sub(n)
        sym = next(symbols)
        replacements.append((sym, value))
        mapping[n] = sym
        if n in placeholders:
            mapping[placeholders[n]] = sym
    reduced = [sub(e) for e in work]
    if postprocess is not None:
        return postprocess(replacements, reduced)
    return replacements, reduced


__all__ = ["cse", "numbered_symbols"]
