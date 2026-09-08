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
    """Return True if a is a quadratic residue modulo p, False otherwise."""
    a = int(a)
    p = int(p)
    if p <= 0:
        raise ValueError("p must be a positive integer")
    if p in (1, 2):
        return True
    a = a % p
    if a == 0:
        return True
    return legendre_symbol(a, p) == 1


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
    """Compute the square root of a modulo p: x**2 == a (mod p) via Tonelli-Shanks."""
    a = int(a)
    p = int(p)
    if p <= 0:
        raise ValueError("p must be positive")
    a = a % p
    if a == 0:
        return [0] if all_roots else 0
    if p == 2:
        return [a] if all_roots else a
    if pow(a, (p - 1) // 2, p) != 1:
        return [] if all_roots else None
    if p % 4 == 3:
        r = pow(a, (p + 1) // 4, p)
    else:
        q = p - 1
        s = 0
        while q % 2 == 0:
            q //= 2
            s += 1
        z = 2
        while pow(z, (p - 1) // 2, p) != p - 1:
            z += 1
        m = s
        c = pow(z, q, p)
        t = pow(a, q, p)
        r = pow(a, (q + 1) // 2, p)
        while True:
            if t == 0:
                r = 0
                break
            if t == 1:
                break
            i = 0
            temp = t
            while temp != 1 and i < m:
                temp = pow(temp, 2, p)
                i += 1
            if i == m:
                return [] if all_roots else None
            b = pow(c, 1 << (m - i - 1), p)
            m = i
            c = (b * b) % p
            t = (t * c) % p
            r = (r * b) % p

    roots = sorted(list({r, (p - r) % p}))
    if all_roots:
        return roots
    return min(roots)


def quadratic_residues(p: Any) -> list[int]:
    """Return the sorted list of all quadratic residues modulo p."""
    p = int(p)
    if p <= 0:
        raise ValueError("p must be positive")
    return sorted(list({(x * x) % p for x in range(p)}))


def is_nthpow_residue(a: Any, n: Any, m: Any) -> bool:
    """Return True if x**n == a (mod m) has an integer solution."""
    import math
    a = int(a)
    n = int(n)
    m = int(m)
    if m <= 0:
        raise ValueError("m must be positive")
    if n < 0:
        raise ValueError("n must be non-negative")
    a = a % m
    if a == 0:
        return True
    if n == 0:
        return a == 1
    if n == 1 or m == 1:
        return True
    if n == 2:
        return is_quad_residue(a, m)
    if m <= 10000:
        for x in range(m):
            if pow(x, n, m) == a:
                return True
        return False
    if isprime(m):
        g = math.gcd(n, m - 1)
        return pow(a, (m - 1) // g, m) == 1
    factors = factorint(m)
    for p, e in factors.items():
        mod = p ** e
        a_p = a % mod
        found = False
        for x in range(mod):
            if pow(x, n, mod) == a_p:
                found = True
                break
        if not found:
            return False
    return True


def discrete_log(n: Any, a: Any, b: Any, order: Any = None) -> int:
    """Compute discrete logarithm x such that b**x == a (mod n) using baby-step giant-step."""
    import math
    n = int(n)
    a = int(a) % n
    b = int(b) % n
    if a == 1:
        return 0
    m = int(math.isqrt(int(order) if order is not None else n)) + 1
    tbl: dict[int, int] = {}
    cur = 1
    for j in range(m):
        tbl[cur] = j
        cur = (cur * b) % n
    b_m = pow(b, m, n)
    b_m_inv = pow(b_m, -1, n)
    gamma = a
    for i in range(m + 1):
        if gamma in tbl:
            return i * m + tbl[gamma]
        gamma = (gamma * b_m_inv) % n
    raise ValueError("Log does not exist")


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
    "proper_divisors",
    "quadratic_residues",
    "reduced_totient",
    "sqrt_mod",
    "totient",
]
