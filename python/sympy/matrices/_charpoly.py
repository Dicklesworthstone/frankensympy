"""Characteristic polynomial dispatch for bounded exact structured matrices."""

from ..core import (
    Add, Dummy, Mul, Symbol, _charge_nested_if_active, _native, _native_expr,
    _symbol_binding, _wrap,
)
from ..polys.polytools import PurePoly
from . import sparse
from .exceptions import NonSquareMatrixError

_MAX_DIMENSION = 256
_MAX_BLOCK_INPUT_BITS = 16_384


def generator_name(x):
    if x is None:
        return "lambda"
    if type(x) is str:
        return x
    if type(x) in (Symbol, Dummy):
        return str(x)
    raise TypeError("characteristic polynomial generator must be a string, Symbol or Dummy")


def _components(n, entries):
    """Strongly connected components of the stored graph in topological order."""
    forward = [[] for _ in range(n)]
    reverse = [[] for _ in range(n)]
    for row, col in entries:
        if row != col:
            forward[row].append(col)
            reverse[col].append(row)

    # Kosaraju's finish times use an explicit stack, even on long paths.
    seen = set()
    finished = []
    for vertex in range(n):
        if vertex in seen:
            continue
        seen.add(vertex)
        frames = [(vertex, iter(forward[vertex]))]
        while frames:
            current, neighbours = frames[-1]
            child = next(neighbours, None)
            if child is None:
                finished.append(current)
                frames.pop()
            elif child not in seen:
                seen.add(child)
                frames.append((child, iter(forward[child])))

    seen.clear()
    blocks = []
    for vertex in reversed(finished):
        if vertex in seen:
            continue
        block = []
        stack = [vertex]
        seen.add(vertex)
        while stack:
            current = stack.pop()
            block.append(current)
            for child in reverse[current]:
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        blocks.append(sorted(block))
    return blocks


def _tridiagonal(n, entries, gen):
    # The determinant checker specializes xI-A at degree + 1 distinct points
    # and checks every rational result against the original band by sparse LU.
    _charge_nested_if_active(1 + n * (n + len(entries)))
    shifted = {ij: -value for ij, value in entries.items()}
    var = _native_expr(gen)
    for i in range(n):
        shifted[i, i] = var + shifted[i, i] if (i, i) in shifted else var
    polynomial = sparse._NativeSparseMatrix(n, n, shifted).tridiagonal_polynomial_det()
    if polynomial is None:
        raise RuntimeError("rational tridiagonal characteristic polynomial was not admitted")
    return _wrap(polynomial)


def _expression(coefficients, gen):
    """Lift descending rational coefficients once, directly into a single sum."""
    n = len(coefficients) - 1
    zero = sparse._REDUCTION_NATIVE_ZERO
    return Add(*(_wrap(c) * gen**(n-i) for i, c in enumerate(coefficients) if c != zero))


def _block_factor(n, entries, gen):
    if n == 1:
        diagonal = entries.get((0, 0))
        return gen if diagonal is None else _wrap(_native_expr(gen) - diagonal)
    if n <= 64 and all(abs(row - col) <= 1 for row, col in entries):
        return _tridiagonal(n, entries, gen)
    if n > 32:
        raise NotImplementedError(
            "irreducible characteristic-polynomial block exceeds the checked dense or tridiagonal limits")
    _charge_nested_if_active(1 + n**4)
    block = sparse._NativeSparseMatrix(n, n, entries).to_dense()
    return _expression(block.checked_char_poly(), gen)


def _checked_blocks(n, entries):
    _charge_nested_if_active(1 + 5*(n + len(entries)))
    blocks = _components(n, entries)
    # A candidate partition need not be maximally fine. A permutation of
    # every source index and zero below its block diagonal suffice for
    # det(xI-A) to equal the product of the diagonal block determinants.
    labels = {}
    for number, block in enumerate(blocks):
        if not block:
            raise RuntimeError("empty characteristic-polynomial block")
        for local, original in enumerate(block):
            if not 0 <= original < n or original in labels:
                raise RuntimeError("invalid characteristic-polynomial block permutation")
            labels[original] = (number, local)
    if len(labels) != n:
        raise RuntimeError("incomplete characteristic-polynomial block permutation")
    diagonal = [{} for _ in blocks]
    height = 0
    for (row, col), entry in entries.items():
        source, i = labels[row]
        target, j = labels[col]
        if source > target:
            raise RuntimeError("source entry below characteristic-polynomial block diagonal")
        if source == target:
            # Bound the original rational before exporting its integers. The
            # sum of their heights bounds denominator clearing in the product.
            entry.polynomial_source_size()
            height += entry.exact_numerator().bit_length() + entry.exact_denominator().bit_length()
            if height > _MAX_BLOCK_INPUT_BITS:
                raise ValueError("characteristic-polynomial diagonal blocks exceed coefficient height limit")
            diagonal[source][i, j] = entry
    return [(len(block), cells) for block, cells in zip(blocks, diagonal)], height


def _structured_polynomial(n, entries, gen):
    if not n:
        return PurePoly(1, gen)
    blocks, height = _checked_blocks(n, entries)
    factors = [_block_factor(size, cells, gen) for size, cells in blocks]
    if len(factors) == 1:
        return PurePoly(factors[0], gen)
    # Native univariate multiplication collects the exact product over QQ.
    # All factors are monic and their degrees sum to the original dimension.
    _charge_nested_if_active(1 + n*n*(1 + height//64))
    coefficients = _native.poly_coeffs_native(_native_expr(Mul(*factors)), _symbol_binding(gen))
    if len(coefficients) != n+1 or coefficients[0] != _native_expr(1):
        raise RuntimeError("invalid degree or leading coefficient in characteristic-polynomial product")
    return PurePoly(_expression(coefficients, gen), gen)


def _capture(matrix, dense):
    rows, cols = matrix.shape
    if rows != cols:
        raise NonSquareMatrixError("Characteristic polynomial requires a square matrix")
    if rows > _MAX_DIMENSION:
        raise ValueError(f"structured characteristic polynomial supports dimensions up to {_MAX_DIMENSION}")
    snapshot = sparse._structural_sources(matrix)
    _charge_nested_if_active(1 + rows)
    rational = sparse._rational_source_snapshot(
        matrix, dense=dense, context="characteristic polynomial")
    entries = None
    if rational is None:
        if dense or rows > 32:
            raise NotImplementedError(
                "large characteristic polynomial requires literal rational coefficients")
    else:
        native, retained = rational
        if dense:
            native = sparse._SOLVE_DENSE_MEMBERS["to_sparse"](native)
        entries = sparse._REDUCTION_NATIVE_MEMBERS["todok"](native)

    def unchanged():
        sparse._check_structural_sources(snapshot)
        if rational is not None:
            sparse._check_rational_source(retained)

    unchanged()
    return rows, entries, unchanged


def characteristic_polynomial(matrix, x, *, dense):
    """Return det(xI-A) from an exact rational source or bounded symbolic block."""
    rows, entries, unchanged = _capture(matrix, dense)
    name = generator_name(x)
    if entries is None:
        # Preserve source coefficient identity and generator-name collision rules.
        result = matrix.to_dense().charpoly(x)
    else:
        result = _structured_polynomial(rows, entries, Symbol(name))
    _charge_nested_if_active()
    unchanged()
    return result


def _block_eigenvalues(size, entries):
    if size == 1:
        return [_wrap(entries.get((0, 0), sparse._REDUCTION_NATIVE_ZERO))]
    if size <= 32:
        # The existing dense radical and indexed-real-root lanes handle a
        # single diagonal block. Cross-block couplings have already been checked.
        _charge_nested_if_active(1 + size**4)
        native = sparse._NativeSparseMatrix(size, size, entries).to_dense()
        return sparse.Matrix(native).eigenvalues()

    gen = Symbol("lambda")
    poly = PurePoly(_block_factor(size, entries, gen), gen)
    # Keep exactly expressible complex radicals. For an unresolved higher
    # degree factor only exact real isolation can supply indexed identities.
    radical = poly.roots()
    if sum(radical.values()) == size:
        return [root for root, multiplicity in radical.items() for _ in range(multiplicity)]
    from ..polys.rootoftools import real_roots

    return real_roots(poly)


def eigenvalue_multiplicities(matrix, *, dense):
    """Compose complete diagonal-block spectra with algebraic multiplicities."""
    rows, entries, unchanged = _capture(matrix, dense)
    if entries is None:
        result = matrix.to_dense().eigenvals()
    else:
        blocks, _ = _checked_blocks(rows, entries)
        result = {}
        for size, cells in blocks:
            roots = _block_eigenvalues(size, cells)
            if len(roots) != size:
                raise ValueError("unable to establish the complete exact spectrum of a diagonal block")
            for root in roots:
                result[root] = result.get(root, 0) + 1
            _charge_nested_if_active(1 + size)
    _charge_nested_if_active()
    unchanged()
    return result
