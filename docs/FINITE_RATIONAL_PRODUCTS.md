# Checked finite rational products

The native finite-product path computes bounded exact products in `Q(k)` and
checks the result independently against the original source. It is exposed by
`fsym_calculus::summation::checked_product` and the Python compatibility shell's
`product` and `Product.doit` entry points. This is implemented, uncertified
functionality; it does not establish a complete SymPy compatibility profile.

In the repository compatibility shell:

```python
from sympy import Symbol, Product, Rational, product

k = Symbol("k", integer=True)
product(k / (k + 1), (k, 1, 10**12))
# Rational(1, 1000000000001)

product(k * (k + 2) / (k + 1)**2, (k, 1, 10**9))
# Rational(500000001, 1000000001)

product(((k + 1)**2 + 1) / (k*k + 1), (k, 0, 10**9))
# 1000000002000000002

Product(k, (k, 5, 1)).doit()
# Rational(1, 24)
```

Short ranges are independently recomputed with exact rational arithmetic.
Longer ranges use complete rational-linear and irreducible-quadratic
factorizations to find a multiplicative telescope
`f(k) = c * R(k + 1) / R(k)`. The proof kernel checks the full polynomial
identity by coefficient arithmetic, reconstructs every supplied factorization,
excludes integer poles, and checks the endpoint quotient. A separate zero-factor
witness proves that the entire source remains defined before returning zero.

Reversed ranges use `product(a, b) = 1 / product(b + 1, a - 1)`; adjacent
endpoints give the empty product. A reversed range containing a zero factor is
refused. Every original inverse is retained, including inverses hidden by held
cancellation, multiplication by zero, and zeroth powers. A first finite
`Product.doit` range therefore preserves the original source before native
admission, including when `deep=True` and a later limit is symbolic.

The rational path is bounded to 4,096 source nodes, depth 64, degree 64 input
polynomials, 4,096-bit input coefficients/endpoints, degree 128 telescopers,
integer-shift spans at most 64, and 16,384-bit exact arithmetic results. Direct
evaluation covers at most 2,001 factors. Larger unsupported products remain
unevaluated; for example, this path does not materialize a factorial whose
value exceeds its exact output envelope. Source admission also precedes empty
products, so an explicitly nowhere-defined source is refused rather than
canonicalized away.

The native metered API accepts separate generator and verifier meters and
preserves cumulative charges and cancellation. Domain errors, exhausted
resources, and failed certificates are terminal errors; they do not trigger an
unguarded retry. Python propagates those errors as `ValueError`. The receipt is
an immutable in-process exact check, not a portable proof transport, an
infinite-product convergence certificate, or a symbolic-bound theorem.

Focused native targets are `rational_product` in both `fsym-calculus` and
`fsym-proof-kernel`; public regressions are in
`python/tests/test_rational_products.py`.
