"""Gray code generation and ranking for FrankenSymPy."""

from __future__ import annotations
from typing import Any, Iterator

from ..core import Basic


class GrayCode(Basic):
    """A Gray code generator of length n."""

    def __init__(self, n: int, *args: Any, current: str | None = None, **kwargs: Any) -> None:
        self.n = int(n)
        self.current = current if current is not None else "0" * self.n

    def generate_gray(self, start: str | None = None) -> Iterator[str]:
        n = self.n
        total = 1 << n
        start_rank = self.rank(start) if start is not None else 0
        for i in range(start_rank, total):
            yield self.unrank(n, i)

    @classmethod
    def unrank(cls, n: int, rank: int) -> str:
        gray = rank ^ (rank >> 1)
        return format(gray, f"0{n}b")

    @classmethod
    def rank(cls, gray_str: str) -> int:
        g = int(gray_str, 2)
        b = 0
        while g > 0:
            b ^= g
            g >>= 1
        return b

    def __repr__(self) -> str:
        return f"GrayCode({self.n}, current='{self.current}')"

    def __str__(self) -> str:
        return repr(self)


__all__ = [
    "GrayCode",
]
