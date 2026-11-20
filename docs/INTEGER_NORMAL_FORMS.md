# Exact integer normal forms and linear systems

The bounded integer matrix lane computes Smith normal forms and solves
`A*X = B` over the integers. Every accepted result is checked by the independent
proof kernel against the original inputs. These are native capabilities; they
do not certify a complete SymPy matrix compatibility profile.

## Smith normal form

```python
from sympy import Matrix
from sympy.matrices.normalforms import (
    smith_normal_form, smith_normal_decomposition,
)

A = Matrix([[4, 6], [3, 9]])
D = smith_normal_form(A)
assert D == Matrix([[1, 0], [0, 18]])

D, U, V = smith_normal_decomposition(A)
assert D == U*A*V
```

The convention is `D = U*A*V`. The nonzero diagonal entries of `D` are positive,
each divides its successor, and all remaining entries are zero. `U` and `V`
are integer matrices with integer inverses. The transformation matrices are
not canonical; their particular entries are not an upstream compatibility
promise.

`smith_normal_form(A, domain=None)` infers the integer domain from literal
integer entries and returns a mutable dense `Matrix`, including for immutable
or sparse inputs. Explicit domain objects and other coefficient rings are
outside this slice. The native extension `smith_normal_decomposition(A)` and
`A.smith_normal_decomposition()` preserve the input's built-in matrix class.
Rectangular, singular, zero, and zero-area matrices retain their exact shapes.

## Complete integer solutions

```python
A = Matrix([[2, 4]])
P, K = A.solve_integer(Matrix([[6]]))
assert P == Matrix([[3], [0]])
assert K == Matrix([[-2], [1]])
assert A*P == Matrix([[6]])
assert A*K == Matrix([[0]])

# A rational solution exists here, but no integer solution does.
assert A.solve_integer(Matrix([[1]])) is None
```

For an `m` by `n` matrix `A` and an `m` by `k` right-hand side `B`, a consistent
result is `(P, K)`. The matrix `P` has shape `n` by `k`, and `K` has shape `n`
by `n-rank(A)`. The complete solution set is

```text
X = P + K*T, for every integer matrix T of shape (n-rank(A), k).
```

In the example, every solution is `(x, y) = (3-2*t, t)` for an integer `t`.
The kernel basis is primitive and complete over the integers. Merely checking
`A*K = 0` and the rational rank would not prove this: replacing the example's
kernel column by `(-4, 2)` would omit valid integer solutions.

Both operands may use any of the four built-in mutable/immutable dense/sparse
matrix classes. Outputs preserve `A`'s class. Multiple right-hand-side columns
are solved together, with independent integer parameters for each column. If
any column is inconsistent, the result is `None`; partially solved columns
are not published. Empty dimensions and zero right-hand-side columns are
supported.

## Why the result is accepted

The generator uses deterministic Euclidean row and column operations and
maintains the two integer inverse transformations. The independent verifier
uses bounded reference matrix products. It checks all of the following:

- Original integer source identity, shapes, canonical sparse storage, and
  coefficient bounds.
- Both inverse identities for each transformation, with integer coefficients.
- `U*A*V = D`, positive invariant factors, divisibility, and trailing zeros.
- For a linear system, `C = U*B`; every nonzero Smith factor must divide the
  corresponding entries of `C`, and every remaining row of `C` must be zero.
- The canonical particular solution `P = V*Y`, with zero free Smith coordinates,
  and the complete kernel consisting of the remaining columns of `V`.

A checked inconsistency identifies a nondivisible entry or a nonzero entry in
a zero Smith row. The row coordinate is in `U*B`, not necessarily in the
original right-hand side.

Native Rust callers can retain the exact sources, witness, and receipt for
replay. Receipts bind the accepted input and output matrices and have no public
constructor or deserializer that could turn an untrusted flag into acceptance.
Portable witness decoding is bounded and rejects extra authority fields.

## Bounds and failure behavior

Each input dimension is at most 64. Original entries are literal integers of
at most 4096 bits; intermediate and witness coefficients are bounded by 8192
bits. Generation and verification have separate cumulative 20-million-step
and 256-MiB allocation envelopes. Active Python request accounting also spans
admission, computation, verification, lifting, and publication.

Unsupported coefficient domains, modified built-in implementations, stale or
mutated sources, invalid certificates, cancellation, and exhausted resources
remain terminal outcomes. They do not fall back to upstream SymPy, floating
point arithmetic, or an unchecked answer. An intrinsic work bound may refuse
a difficult input even when its shape and original coefficients fit the input
envelope; this is not a claim of complete algorithmic coverage within those
two bounds alone.

The implementation and its targeted tests do not close the broad WS10/WS19
validation gates or establish a performance win.
