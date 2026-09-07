"""Modular arithmetic and Chinese Remainder Theorem for FrankenSymPy (WS18)."""

from typing import Any, Optional, Sequence, Tuple
from ..core import Integer, _native


def mod_inverse(a: Any, m: Any) -> Integer:
    """Compute the modular multiplicative inverse of a modulo m."""
    a_int = int(a)
    m_int = int(m)
    if m_int <= 1:
        raise ValueError(f"modulus must be > 1, got {m_int}")
    res_str = _native.mod_inverse_fn(str(a_int), str(m_int))
    return Integer(int(res_str))


def crt(
    m: Sequence[Any],
    r: Sequence[Any],
    symmetric: bool = False,
    check: bool = True,
) -> Optional[Tuple[Integer, Integer]]:
    """Chinese Remainder Theorem.

    Solves x = r_i (mod m_i) for pairwise coprime moduli m_i.
    Returns (result, combined_modulus) or None if no solution exists.
    """
    if len(m) != len(r) or len(m) == 0:
        return None
    m_strs = [str(int(x)) for x in m]
    r_strs = [str(int(x)) for x in r]
    res = _native.crt_fn(m_strs, r_strs)
    if res is None:
        return None
    sol_str, mod_str = res
    sol = int(sol_str)
    mod = int(mod_str)
    if symmetric and sol > mod // 2:
        sol -= mod
    return Integer(sol), Integer(mod)


__all__ = [
    "crt",
    "mod_inverse",
]
