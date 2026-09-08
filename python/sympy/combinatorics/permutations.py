"""Permutations and cycle representations for FrankenSymPy."""

from __future__ import annotations
import math
from typing import Any, Iterable, Sequence

from ..core import Basic


class Cycle(dict):
    """A dictionary-based representation of disjoint permutation cycles."""

    def __init__(self, *args: Any) -> None:
        super().__init__()
        if not args:
            return
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = tuple(args[0])
        for i in range(len(args)):
            self[args[i]] = args[(i + 1) % len(args)]

    def __call__(self, *args: Any) -> "Cycle":
        if not args:
            return self
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = tuple(args[0])
        for i in range(len(args)):
            self[args[i]] = args[(i + 1) % len(args)]
        return self


class Permutation(Basic):
    """A permutation of a finite set {0, 1, ..., n - 1}."""

    __slots__ = ("_array_form", "_size")

    def __new__(cls, *args: Any, size: int | None = None, **kwargs: Any) -> "Permutation":
        if len(args) == 1 and isinstance(args[0], Permutation):
            p = args[0]
            if size is None or size == p._size:
                return p
            if size < p._size:
                raise ValueError("size cannot be smaller than existing permutation size")
            new_arr = list(p._array_form) + list(range(p._size, size))
            obj = object.__new__(cls)
            obj._array_form = tuple(new_arr)
            obj._size = size
            return obj

        arr: list[int] = []
        if not args:
            n = size if size is not None else 0
            arr = list(range(n))
        elif len(args) == 1 and isinstance(args[0], (list, tuple)):
            first = args[0]
            if first and isinstance(first[0], (list, tuple)):
                # List of cycles: e.g. [[0, 1], [2, 3]]
                max_elem = -1
                for c in first:
                    for x in c:
                        if x > max_elem:
                            max_elem = x
                n = max(max_elem + 1, size if size is not None else 0)
                arr = list(range(n))
                for c in first:
                    for i in range(len(c)):
                        arr[c[i]] = c[(i + 1) % len(c)]
            else:
                # Array form: e.g. [1, 0, 2]
                arr = [int(x) for x in first]
                n = len(arr)
                if size is not None:
                    if size < n:
                        raise ValueError("size cannot be smaller than array length")
                    arr.extend(range(n, size))
        elif len(args) == 1 and isinstance(args[0], Cycle):
            c_dict = args[0]
            max_elem = max(c_dict.keys(), default=-1)
            n = max(max_elem + 1, size if size is not None else 0)
            arr = list(range(n))
            for k, v in c_dict.items():
                arr[k] = v
        else:
            # Arguments are a single cycle: e.g. Permutation(0, 1, 2)
            c = [int(x) for x in args]
            max_elem = max(c, default=-1)
            n = max(max_elem + 1, size if size is not None else 0)
            arr = list(range(n))
            for i in range(len(c)):
                arr[c[i]] = c[(i + 1) % len(c)]

        # Validate arr is a bijection of {0, ..., len(arr) - 1}
        n = len(arr)
        if sorted(arr) != list(range(n)):
            raise ValueError(f"Invalid permutation array: {arr}")

        obj = object.__new__(cls)
        obj._array_form = tuple(arr)
        obj._size = n
        return obj

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def array_form(self) -> list[int]:
        return list(self._array_form)

    @property
    def size(self) -> int:
        return self._size

    @property
    def cardinality(self) -> int:
        return self._size

    @property
    def cyclic_form(self) -> list[list[int]]:
        visited = [False] * self._size
        cycles = []
        for i in range(self._size):
            if visited[i]:
                continue
            cur = i
            cycle = []
            while not visited[cur]:
                visited[cur] = True
                cycle.append(cur)
                cur = self._array_form[cur]
            if len(cycle) > 1:
                min_idx = cycle.index(min(cycle))
                norm_cycle = cycle[min_idx:] + cycle[:min_idx]
                cycles.append(norm_cycle)
        cycles.sort(key=lambda c: c[0])
        return cycles

    def support(self) -> list[int]:
        return [i for i in range(self._size) if self._array_form[i] != i]

    def length(self) -> int:
        return len(self.support())

    @property
    def is_identity(self) -> bool:
        return all(self._array_form[i] == i for i in range(self._size))

    def order(self) -> int:
        cycles = self.cyclic_form
        if not cycles:
            return 1
        res = 1
        for c in cycles:
            res = (res * len(c)) // math.gcd(res, len(c))
        return res

    def inversions(self) -> int:
        inv = 0
        arr = self._array_form
        n = self._size
        for i in range(n):
            for j in range(i + 1, n):
                if arr[i] > arr[j]:
                    inv += 1
        return inv

    def parity(self) -> int:
        return self.inversions() % 2

    @property
    def is_even(self) -> bool:
        return self.parity() == 0

    @property
    def is_odd(self) -> bool:
        return self.parity() == 1

    def signature(self) -> int:
        return 1 if self.is_even else -1

    def ascents(self) -> list[int]:
        return [i for i in range(self._size - 1) if self._array_form[i] < self._array_form[i + 1]]

    def descents(self) -> list[int]:
        return [i for i in range(self._size - 1) if self._array_form[i] > self._array_form[i + 1]]

    def rank(self) -> int:
        arr = list(self._array_form)
        n = len(arr)
        r = 0
        fact = math.factorial(n - 1)
        digits = list(range(n))
        for i in range(n - 1):
            idx = digits.index(arr[i])
            r += idx * fact
            digits.pop(idx)
            fact //= (n - 1 - i)
        return r

    @classmethod
    def unrank_lex(cls, size: int, rank: int) -> "Permutation":
        digits = list(range(size))
        fact = math.factorial(size - 1)
        arr = []
        for i in range(size - 1):
            idx = rank // fact
            rank %= fact
            arr.append(digits.pop(idx))
            fact //= (size - 1 - i)
        arr.append(digits[0])
        return Permutation(arr)

    def __call__(self, *args: Any) -> Any:
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, (int, Basic)) and not isinstance(arg, (list, tuple)):
                i = int(arg)
                if 0 <= i < self._size:
                    return self._array_form[i]
                return i
            if isinstance(arg, (list, tuple)):
                return type(arg)(arg[self._array_form[i]] if i < self._size else arg[i] for i in range(len(arg)))
        c = [int(x) for x in args]
        cycs = self.cyclic_form
        cycs.append(c)
        max_elem = max([self._size - 1] + c)
        return Permutation(cycs, size=max_elem + 1)

    def __mul__(self, other: Any) -> "Permutation":
        if not isinstance(other, Permutation):
            return NotImplemented
        n = max(self._size, other._size)
        res = [0] * n
        for i in range(n):
            pi = self._array_form[i] if i < self._size else i
            res[i] = other._array_form[pi] if pi < other._size else pi
        return Permutation(res)

    def __invert__(self) -> "Permutation":
        inv = [0] * self._size
        for i, val in enumerate(self._array_form):
            inv[val] = i
        return Permutation(inv)

    def __pow__(self, exponent: int) -> "Permutation":
        if exponent == 0:
            return Permutation(size=self._size)
        if exponent < 0:
            return (~self) ** (-exponent)
        cur = self
        res = Permutation(size=self._size)
        p = exponent
        while p > 0:
            if p % 2 == 1:
                res = res * cur
            cur = cur * cur
            p //= 2
        return res

    def commutator(self, other: "Permutation") -> "Permutation":
        return self * other * (~self) * (~other)

    def commutes_with(self, other: "Permutation") -> bool:
        return (self * other) == (other * self)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Permutation):
            return False
        n = max(self._size, other._size)
        for i in range(n):
            a = self._array_form[i] if i < self._size else i
            b = other._array_form[i] if i < other._size else i
            if a != b:
                return False
        return True

    def __hash__(self) -> int:
        arr = list(self._array_form)
        while arr and arr[-1] == len(arr) - 1:
            arr.pop()
        return hash(tuple(arr))

    def __repr__(self) -> str:
        cycs = self.cyclic_form
        if not cycs:
            return f"Permutation({self._size})" if self._size > 0 else "Permutation()"
        res = "".join(f"({', '.join(str(x) for x in c)})" for c in cycs)
        if self._size > 0 and (not cycs or max(max(c) for c in cycs) < self._size - 1):
            return f"Permutation({self._size - 1}){res}"
        return f"Permutation{res}"

    def __str__(self) -> str:
        return repr(self)


__all__ = [
    "Cycle",
    "Permutation",
]
