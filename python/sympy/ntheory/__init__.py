"""Number theory algorithms and functions for FrankenSymPy."""

from typing import Any
from ..core import _native
from .. import (
    divisor_count,
    divisor_sigma,
    factorint,
    isprime,
    jacobi_symbol,
    mobius,
    totient,
)
from .modular import crt, mod_inverse


def legendre_symbol(a: Any, p: Any) -> int:
    """Compute Legendre symbol (a / p) for integer a and odd prime p."""
    return _native.legendre_symbol_fn(int(a), int(p))


def is_squarefree(n: Any) -> bool:
    """Return True if n has no repeated prime factors, False otherwise."""
    return _native.is_square_free_fn(int(n))


is_square_free = is_squarefree


def primenu(n: Any) -> int:
    """Return the number of distinct prime factors of n: ω(n)."""
    return _native.prime_omega_fn(int(n))


prime_omega = primenu


def primeomega(n: Any) -> int:
    """Return the total number of prime factors of n with multiplicity: Ω(n)."""
    return _native.prime_big_omega_fn(int(n))


prime_big_omega = primeomega


def is_perfect(n: Any) -> bool:
    """Return True if n is a perfect number (sum of proper divisors equals n)."""
    return _native.is_perfect_number_fn(int(n))


def carmichael(n: Any) -> int:
    """Return the Carmichael lambda function λ(n)."""
    return _native.carmichael_fn(int(n))


reduced_totient = carmichael


def is_primitive_root(g: Any, p: Any) -> bool:
    """Return True if g is a primitive root modulo p, False otherwise."""
    return _native.is_primitive_root_fn(int(g), int(p))


def integer_nthroot(y: Any, n: Any):
    """Return (root, exact) where root = floor(y**(1/n)) and exact is a boolean."""
    y_int = int(y)
    n_int = int(n)
    if y_int < 0:
        raise ValueError("y must be non-negative")
    if n_int <= 0:
        raise ValueError("n must be positive")
    root = _native.integer_nth_root_fn(y_int, n_int)
    exact = (root ** n_int == y_int)
    return root, exact


__all__ = [
    "carmichael",
    "crt",
    "divisor_count",
    "divisor_sigma",
    "factorint",
    "integer_nthroot",
    "is_perfect",
    "is_primitive_root",
    "is_square_free",
    "is_squarefree",
    "isprime",
    "jacobi_symbol",
    "legendre_symbol",
    "mobius",
    "mod_inverse",
    "primenu",
    "prime_big_omega",
    "prime_omega",
    "primeomega",
    "reduced_totient",
    "totient",
]
