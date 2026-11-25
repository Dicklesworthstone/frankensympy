"""Number theory algorithms and functions for FrankenSymPy."""

from __future__ import annotations

from typing import Any
from ..core import _active_callback_session, _native
from .. import (
    _exact_integer_value,
    _not_integer_message,
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
    """Decide whether a unit generates the complete group modulo p.

    Prime powers and twice odd prime powers are supported. Nonunits and
    moduli <= 1 are domain errors; an exhausted native search raises an error.
    """
    g, p = _modular_integer(g), _modular_integer(p)
    if p <= 1:
        raise ValueError("p should be an integer greater than 1")
    return _native.is_primitive_root_fn(g % p, p)


def n_order(a: Any, n: Any) -> int:
    """Return the least positive k with a**k == 1 modulo n.

    The exact native lane admits n through 2**64 - 1 and requires a unit.
    """
    a, n = _modular_integer(a), _modular_integer(n)
    if n <= 1:
        raise ValueError("n should be an integer greater than 1")
    return _native.multiplicative_order_fn(a % n, n)


def primitive_root(p: Any, smallest: bool = True):
    """Return a generator modulo p, or None when its unit group is noncyclic.

    By default the least positive generator is returned. ``smallest=False``
    permits faster lifting from an odd prime. Moduli are bounded by 2**64 - 1;
    bounded search exhaustion raises an error instead of returning None.
    """
    p = _modular_integer(p)
    if p <= 1:
        raise ValueError("p should be an integer greater than 1")
    return _native.primitive_root_fn(p, bool(smallest))


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


def primefactors(n: Any) -> list[int]:
    """Return a sorted list of the distinct prime factors of n."""
    n = abs(int(n))
    if n in (0, 1):
        return []
    factors = factorint(n)
    return sorted(factors.keys())


def divisors(n: Any, generator: bool = False):
    """Return all positive divisors of n sorted in increasing order."""
    n = abs(int(n))
    if n == 0:
        return [] if not generator else iter([])
    if n == 1:
        return [1] if not generator else iter([1])
    factors = factorint(n)
    divs = [1]
    for p, e in factors.items():
        powers = [p ** k for k in range(e + 1)]
        divs = [d * pk for d in divs for pk in powers]
    divs.sort()
    return divs if not generator else iter(divs)


def proper_divisors(n: Any, generator: bool = False):
    """Return all positive divisors of n except n itself."""
    divs = divisors(n)
    res = divs[:-1] if len(divs) > 1 else []
    return res if not generator else iter(res)


def nextprime(n: Any, ith: int = 1) -> int:
    """Return the ith smallest prime strictly greater than n."""
    n = int(n)
    ith = int(ith)
    if ith < 1:
        raise ValueError("ith must be a positive integer")
    p = max(2, n + 1)
    if p > 2 and p % 2 == 0:
        p += 1
    step = 1 if p == 2 else 2
    found = 0
    while True:
        if isprime(p):
            found += 1
            if found == ith:
                return p
        p += step
        if p == 3:
            step = 2


def prevprime(n: Any) -> int:
    """Return the largest prime strictly smaller than n."""
    n = int(n)
    if n <= 2:
        raise ValueError("no preceding primes")
    if n == 3:
        return 2
    p = n - 1
    if p % 2 == 0:
        p -= 1
    while p >= 3:
        if isprime(p):
            return p
        p -= 2
    return 2


def is_quad_residue(a: Any, p: Any) -> bool:
    """Test square-root existence modulo a positive integer, including composites.

    Exact prime-power conditions avoid enumerating a potentially large root set.
    The native lane supports moduli through 2**64 - 1 and refuses exhausted work.
    """
    a = _modular_integer(a)
    p = _modular_integer(p)
    if p <= 0:
        raise ValueError("p must be > 0")
    return _native.is_quadratic_residue_fn(a % p, p)


def _modular_integer(value: Any) -> int:
    """Admit exact built-in integers without lossy or custom conversion hooks."""
    integer = _exact_integer_value(value)
    if integer is None:
        raise ValueError(_not_integer_message(value))
    return integer


def prime(nth: Any) -> int:
    """Return the nth prime (1-indexed: prime(1) == 2, prime(2) == 3, etc.)."""
    nth = int(nth)
    if nth < 1:
        raise ValueError("nth must be a positive integer")
    p = 2
    for _ in range(nth - 1):
        p = nextprime(p)
    return p


def primepi(n: Any) -> int:
    """Return the number of primes less than or equal to n."""
    n = int(n)
    if n < 2:
        return 0
    count = 1
    p = 2
    while True:
        p = nextprime(p)
        if p <= n:
            count += 1
        else:
            break
    return count


def multiplicity(p: Any, n: Any) -> int:
    """Return the multiplicity of p in n (the largest integer k such that p**k divides n)."""
    p = int(p)
    n = int(n)
    if p <= 1:
        raise ValueError("p must be greater than 1")
    if n == 0:
        raise ValueError("n cannot be zero")
    n = abs(n)
    count = 0
    while n % p == 0:
        count += 1
        n //= p
    return count


def primerange(a: Any, b: Any = None):
    """Generate all prime numbers in the range [a, b)."""
    if b is None:
        start = 2
        stop = int(a)
    else:
        start = int(a)
        stop = int(b)
    if stop <= 2 or start >= stop:
        return
    p = 2 if start <= 2 else nextprime(start - 1)
    while p < stop:
        yield p
        p = nextprime(p)


def perfect_power(n: Any):
    """Return (a, b) such that a**b == n with b > 1, or False if n is not a perfect power."""
    n = int(n)
    if n in (0, 1):
        return False
    sgn = -1 if n < 0 else 1
    abs_n = abs(n)
    import math
    max_b = int(math.log2(abs_n))
    for b in range(max_b, 1, -1):
        if sgn == -1 and b % 2 == 0:
            continue
        a, exact = integer_nthroot(abs_n, b)
        if exact and a > 1:
            return (a * sgn, b)
    return False


def sqrt_mod(a: Any, p: Any, all_roots: bool = False):
    """Solve x**2 == a modulo an integer, including composite prime powers.

    With ``all_roots=True``, return every distinct root in increasing order.
    Otherwise return one root at most abs(p)//2, or None when no root exists.
    Enumeration has an explicit native output bound; the one-root operation
    does not enumerate all roots. Moduli are bounded by 2**64 - 1.
    """
    a = _modular_integer(a)
    p = abs(_modular_integer(p))
    # Preserve the profile's zero-modulus exception and negative-modulus rule.
    a = a % p
    if all_roots:
        return _native.modular_square_roots_fn(a, p)
    return _native.modular_square_root_fn(a, p)


def _nthroot_charge(session: Any, steps: int) -> None:
    """Keep the shortcut paths on the captured request and retain its error."""
    if session is None:
        return
    try:
        session.charge_nested_steps(steps)
    except Exception:
        # The public callback counter predates atomic native admission. Mark
        # its refusal terminal too, without replacing the original exception.
        try:
            session.cancel()
        except Exception:
            pass
        raise


def nthroot_mod(a: Any, n: Any, p: Any, all_roots: bool = False):
    """Solve x**n == a modulo a positive integer, with complete exact roots.

    The all-roots result is sorted and contains every distinct residue. The
    scalar result is the least root, except that degree two preserves the
    pinned sqrt_mod selection rule. No solution returns [] or None.

    Degrees above two admit u64 degrees and moduli, at most 65536 materialized
    roots, and a bounded odd-group logarithm table. Exhausted native work or
    output bounds raise RuntimeError, never a false empty result. A zero scalar
    root and a linear congruence require no enumeration. Active callback work
    budgets remain cumulative; synchronous square-root and factorization
    helpers reserve bounded work and observe cancellation at their boundaries.
    """
    session = _active_callback_session()
    _nthroot_charge(session, 0)
    a, p = _modular_integer(a), _modular_integer(p)
    # Keep the pinned zero-modulus exception and its reduction order.
    a %= p
    n = _modular_integer(n)
    if n < 1:
        raise ValueError("n should be positive")
    if p < 1:
        raise ValueError("p should be positive")
    all_roots = bool(all_roots)
    if n == 1:
        result = [a] if all_roots else a
        _nthroot_charge(session, 1)
        return result
    if n == 2:
        # Preserve the specialized scalar selection and no-enumeration path.
        # Its fixed factorization, root-generation, and reference-check bounds
        # fit this conservative reservation; it is not incrementally metered.
        _nthroot_charge(session, 64_000_000)
        result = sqrt_mod(a, p, all_roots)
        _nthroot_charge(session, max(1, len(result)) if all_roots else 1)
        return result
    if all_roots:
        return _native.modular_nth_roots_fn(a, n, p, session=session)
    return _native.modular_nth_root_fn(a, n, p, session=session)


def quadratic_residues(p: Any) -> list[int]:
    """Return the sorted list of all quadratic residues modulo p."""
    p = int(p)
    if p <= 0:
        raise ValueError("p must be positive")
    return sorted(list({(x * x) % p for x in range(p)}))


def is_nthpow_residue(a: Any, n: Any, m: Any) -> bool:
    """Decide whether x**n == a modulo m without enumerating residue classes.

    Prime powers, nonunits, and mixed composites use exact native valuation
    and unit-group criteria. Positive degrees and moduli through 2**64 - 1
    are admitted; exhausted work raises an error instead of returning False.
    """
    a, n, m = _modular_integer(a), _modular_integer(n), _modular_integer(m)
    if m <= 0:
        raise ValueError("m must be positive")
    if n < 0:
        raise ValueError("n must be non-negative")
    a = a % m
    if n == 0:
        # The pinned public profile declares degree 0 modulo 1 false. Preserve
        # that convention here, apart from native congruence arithmetic.
        return a == 1
    return _native.is_nth_power_residue_fn(a, n, m)


def discrete_log(n: Any, a: Any, b: Any, order: Any = None) -> int:
    """Return the least nonnegative x with b**x == a modulo positive n.

    Unit and nonunit bases are supported. A supplied positive ``order``, when
    needed, must annihilate the reduced unit base. It is reduced to an exact,
    independently checked multiplicative order before bounding the search.
    Moduli are bounded by 2**64 - 1. Native work or table exhaustion raises
    RuntimeError; a fully established nonexistent logarithm raises ValueError.
    """
    n, a, b = _modular_integer(n), _modular_integer(a), _modular_integer(b)
    if n < 1:
        raise ValueError("n should be positive")
    if order is not None:
        order = _modular_integer(order)
        if order <= 0:
            raise ValueError("order should be positive")
    return _native.discrete_log_fn(n, a % n, b % n, order)


def npartitions(n: Any) -> int:
    """Return the number of integer partitions of n via Euler's pentagonal recurrence."""
    n = int(n)
    if n < 0:
        return 0
    if n == 0:
        return 1
    p = [0] * (n + 1)
    p[0] = 1
    for i in range(1, n + 1):
        s = 0
        k = 1
        while True:
            g1 = k * (3 * k - 1) // 2
            if g1 > i:
                break
            sign = 1 if (k % 2 == 1) else -1
            s += sign * p[i - g1]

            g2 = k * (3 * k + 1) // 2
            if g2 <= i:
                s += sign * p[i - g2]
            k += 1
        p[i] = s
    return p[n]


__all__ = [
    "carmichael",
    "crt",
    "discrete_log",
    "divisor_count",
    "divisor_sigma",
    "divisors",
    "factorint",
    "integer_nthroot",
    "nthroot_mod",
    "is_nthpow_residue",
    "is_perfect",
    "is_primitive_root",
    "is_quad_residue",
    "is_square_free",
    "is_squarefree",
    "isprime",
    "jacobi_symbol",
    "legendre_symbol",
    "mobius",
    "mod_inverse",
    "multiplicity",
    "n_order",
    "nextprime",
    "npartitions",
    "perfect_power",
    "prevprime",
    "prime",
    "prime_big_omega",
    "prime_omega",
    "primefactors",
    "primenu",
    "primeomega",
    "primepi",
    "primerange",
    "primitive_root",
    "proper_divisors",
    "quadratic_residues",
    "reduced_totient",
    "sqrt_mod",
    "totient",
]
