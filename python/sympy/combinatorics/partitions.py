"""Set and integer partitions for FrankenSymPy."""

from __future__ import annotations
from collections import Counter
from typing import Any, Iterable

from ..core import Basic


class Partition(Basic):
    """A partition of a finite set into non-empty disjoint subsets."""

    def __init__(self, *partition: Any) -> None:
        if len(partition) == 1 and isinstance(partition[0], (list, tuple, set)):
            raw_blocks = partition[0]
        else:
            raw_blocks = partition
        blocks = [sorted(list(b)) for b in raw_blocks]
        blocks.sort(key=lambda b: (min(b) if b else -1, len(b)))
        self._blocks = blocks
        members = set()
        for b in blocks:
            for x in b:
                if x in members:
                    raise ValueError(f"Disjoint blocks required; {x} repeated")
                members.add(x)
        self._members = sorted(list(members))

    @property
    def partition(self) -> list[list[Any]]:
        return [list(b) for b in self._blocks]

    @property
    def members(self) -> list[Any]:
        return list(self._members)

    @property
    def RGS(self) -> list[int]:
        """Restricted growth string representation."""
        elem_to_block = {}
        for block_idx, block in enumerate(self._blocks):
            for elem in block:
                elem_to_block[elem] = block_idx
        return [elem_to_block[elem] for elem in self._members]

    def rank(self) -> int:
        rgs = self.RGS
        r = 0
        for i, val in enumerate(rgs):
            r = r * (i + 1) + val
        return r

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Partition):
            return False
        return self._blocks == other._blocks

    def __hash__(self) -> int:
        return hash(tuple(tuple(b) for b in self._blocks))

    def __repr__(self) -> str:
        blocks_str = ", ".join(f"{{{', '.join(str(x) for x in b)}}}" for b in self._blocks)
        return f"Partition({blocks_str})"

    def __str__(self) -> str:
        return repr(self)


class IntegerPartition(Basic):
    """An integer partition of a positive integer n."""

    def __init__(self, partition: Any, integer: int | None = None) -> None:
        if isinstance(partition, (int, Basic)) and not isinstance(partition, (list, tuple)):
            n = int(partition)
            parts = (n,) if n > 0 else ()
            self._integer = n
            self._partition = parts
        else:
            parts = tuple(sorted([int(x) for x in partition], reverse=True))
            self._partition = parts
            self._integer = sum(parts) if integer is None else int(integer)

    @property
    def partition(self) -> tuple[int, ...]:
        return self._partition

    @property
    def integer(self) -> int:
        return self._integer

    def as_dict(self) -> dict[int, int]:
        return dict(Counter(self._partition))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, IntegerPartition):
            return False
        return self._partition == other._partition

    def __hash__(self) -> int:
        return hash(self._partition)

    def __repr__(self) -> str:
        return f"IntegerPartition({list(self._partition)})"

    def __str__(self) -> str:
        return repr(self)


__all__ = [
    "IntegerPartition",
    "Partition",
]
