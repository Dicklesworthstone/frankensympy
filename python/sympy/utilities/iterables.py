"""Iteration helpers (upstream ``sympy.utilities.iterables``)."""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any, Callable, Iterable, Iterator


def flatten(iterable: Any, levels: int | None = None, cls: Any = None) -> list:
    """Recursively flatten nested lists/tuples (``levels`` deep)."""
    if levels is not None:
        if levels < 0:
            raise ValueError("expected non-negative number of levels, got %s" % levels)
        if levels == 0:
            return list(iterable)
    out = []
    for el in iterable:
        nested = isinstance(el, (list, tuple)) if cls is None else isinstance(el, cls)
        if nested:
            out.extend(flatten(el, None if levels is None else levels - 1, cls))
        else:
            out.append(el)
    return out


def sift(seq: Iterable, keyfunc: Callable, binary: bool = False) -> Any:
    """Group ``seq`` by ``keyfunc``; ``binary=True`` gives (true, false)."""
    if not binary:
        m: Any = defaultdict(list)
        for i in seq:
            m[keyfunc(i)].append(i)
        return m
    t, f = [], []
    for i in seq:
        k = keyfunc(i)
        if k is True:
            t.append(i)
        elif k is False:
            f.append(i)
        else:
            raise ValueError("keyfunc gave non-binary output")
    return t, f


def default_sort_key(item: Any, order: Any = None) -> Any:
    """Canonical sort key of expressions, numbers, strings and containers."""
    from ..core import Basic, sympify
    from ..printing.str import sort_key

    if isinstance(item, Basic):
        return sort_key(item)
    if isinstance(item, (int, float)) and not isinstance(item, bool):
        return sort_key(sympify(item))
    if isinstance(item, str):
        return ((1, 0, "str"), (len(item), item), ((1, 0, "Number"), (0, ()), (), 1), 1)
    if isinstance(item, (list, tuple, set, frozenset, dict)):
        if isinstance(item, (set, frozenset)):
            args = sorted(default_sort_key(a) for a in item)
        elif isinstance(item, dict):
            args = sorted(default_sort_key(kv) for kv in item.items())
        else:
            args = [default_sort_key(a) for a in item]
        return ((10, 0, type(item).__name__), (len(args), tuple(args)), ((1, 0, "Number"), (0, ()), (), 1), 1)
    return ((100, 0, type(item).__name__), (1, (str(item),)), ((1, 0, "Number"), (0, ()), (), 1), 1)


def _nodes(e: Any) -> int:
    from ..core import Basic
    from ..core.match import preorder_traversal

    if isinstance(e, Basic):
        return sum(1 for _ in preorder_traversal(e))
    if isinstance(e, (list, tuple, set, frozenset)):
        return 1 + sum(_nodes(a) for a in e)
    return 1


def ordered(seq: Iterable, keys: Any = None, default: bool = True, warn: bool = False) -> Iterator:
    """Yield ``seq`` in a canonical order (``keys`` first, then
    ``default_sort_key``)."""
    seq = list(seq)
    key_funcs = list(keys) if isinstance(keys, (list, tuple)) else ([keys] if keys else [])
    if not key_funcs:
        key_funcs.append(_nodes)  # upstream: node count, then default_sort_key
    if default:
        key_funcs.append(default_sort_key)

    def k(x: Any) -> tuple:
        return tuple(f(x) for f in key_funcs)

    yield from sorted(seq, key=k)


def subsets(seq: Iterable, k: int | None = None, repetition: bool = False) -> Iterator:
    seq = list(seq)
    if k is None:
        if repetition:
            raise ValueError("k must be given with repetition")
        for r in range(len(seq) + 1):
            yield from itertools.combinations(seq, r)
        return
    if repetition:
        yield from itertools.combinations_with_replacement(seq, k)
    else:
        yield from itertools.combinations(seq, k)


def variations(seq: Iterable, n: int, repetition: bool = False) -> Iterator:
    seq = list(seq)
    if repetition:
        yield from itertools.product(seq, repeat=n)
    else:
        yield from itertools.permutations(seq, n)


def has_dups(seq: Iterable) -> bool:
    seen = []
    for x in seq:
        if x in seen:
            return True
        seen.append(x)
    return False


def has_variety(seq: Iterable) -> bool:
    seq = list(seq)
    return any(x != seq[0] for x in seq[1:])


def unflatten(iter: Iterable, n: int = 2) -> list:
    items = list(iter)
    if n < 1 or len(items) % n:
        raise ValueError("iter length is not a multiple of %i" % n)
    return [tuple(items[i:i + n]) for i in range(0, len(items), n)]


def topological_sort(graph: tuple, key: Any = None) -> list:
    """Kahn's algorithm over ``(vertices, edges)``; ValueError on a cycle."""
    V, E = graph
    V = list(V)
    incoming: dict = {v: set() for v in V}
    for a, b in E:
        incoming[b].add(a)
    out: list = []
    ready = sorted([v for v in V if not incoming[v]], key=key or (lambda v: V.index(v)))
    while ready:
        v = ready.pop(0)
        out.append(v)
        for w in V:
            if v in incoming[w]:
                incoming[w].discard(v)
                if not incoming[w]:
                    ready.append(w)
                    ready.sort(key=key or (lambda u: V.index(u)))
    if len(out) != len(V):
        raise ValueError("cycle detected")
    return out


def postfixes(seq: Any) -> Iterator:
    for i in range(len(seq)):
        yield seq[-i - 1:]


def prefixes(seq: Any) -> Iterator:
    for i in range(len(seq)):
        yield seq[:i + 1]


__all__ = [
    "default_sort_key", "flatten", "has_dups", "has_variety", "ordered", "postfixes",
    "prefixes", "sift", "subsets", "topological_sort", "unflatten", "variations",
]
