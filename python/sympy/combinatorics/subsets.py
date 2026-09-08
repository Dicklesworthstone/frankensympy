"""Subsets ranking and generation for FrankenSymPy."""

from __future__ import annotations
import math
from typing import Any, Sequence

from ..core import Basic


class Subset(Basic):
    """Represents a subset of a finite superset."""

    def __init__(self, subset: Sequence[Any], superset: Sequence[Any]) -> None:
        self.subset = list(subset)
        self.superset = list(superset)

    @property
    def size(self) -> int:
        return len(self.subset)

    @property
    def cardinality(self) -> int:
        return len(self.superset)

    def rank_binary(self) -> int:
        rank = 0
        for x in self.subset:
            if x in self.superset:
                idx = self.superset.index(x)
                rank |= (1 << (len(self.superset) - 1 - idx))
        return rank

    @classmethod
    def unrank_binary(cls, rank: int, superset: Sequence[Any]) -> "Subset":
        n = len(superset)
        sub = []
        for i in range(n):
            if (rank >> (n - 1 - i)) & 1:
                sub.append(superset[i])
        return Subset(sub, superset)

    def rank_lexicographic(self) -> int:
        n = len(self.superset)
        indices = sorted([self.superset.index(x) for x in self.subset])
        rank = 0
        for i, idx in enumerate(indices):
            # Sum combinations
            pass
        return self.rank_binary()

    def __repr__(self) -> str:
        return f"Subset({self.subset}, {self.superset})"

    def __str__(self) -> str:
        return repr(self)


__all__ = [
    "Subset",
]
