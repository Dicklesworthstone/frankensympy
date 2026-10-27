"""Exact-input adapters for native residual enclosures and local root claims.

The native checker owns acceptance. Rational midpoint inversion here is only a
bounded witness proposal and carries no mathematical acceptance authority.
"""

from fractions import Fraction
import math

from ..core import Integer, Rational, _exact_surface_types, _native, _wrap

__all__ = ["ResidualJacobianEnclosure", "RootBoxReceipt"]

_MAX_BITS = 4096
_MAX_VARIABLES = 16
_MAX_RESIDUALS = 64
_MAX_ROOT_SEARCH_ATTEMPTS = 24


def _sequence(value, length, label):
    if type(value) not in (tuple, list) or len(value) != length:
        raise TypeError(f"{label} must be an exact list/tuple with {length} entries")
    return value


def _rational(value):
    cls = type(value)
    if cls is int:
        p, q = value, 1
    elif cls is Fraction:
        p, q = value.numerator, value.denominator
    elif cls in _exact_surface_types() and isinstance(value, (Integer, Rational)):
        p, q = value.p, value.q
    else:
        raise TypeError("certified inputs must be exact int, Rational or fractions.Fraction values")
    if max(abs(p).bit_length(), q.bit_length()) > _MAX_BITS:
        raise ValueError("certified rational input exceeds the 4096-bit limit")
    return Fraction(p, q)


def _native_number(value):
    return _native.py_rational(value.numerator, value.denominator)


def _native_box(box):
    return [(_native_number(lo), _native_number(hi)) for lo, hi in box]


def _bounds(values):
    return tuple((_wrap(lo), _wrap(hi)) for lo, hi in values)


class _Receipt:
    __slots__ = ("_receipt",)

    def __setattr__(self, name, value):
        raise AttributeError("accepted numerical receipts are immutable")

    @property
    def residuals(self):
        return tuple(_wrap(value) for value in self._receipt.residuals)

    @property
    def variables(self):
        """Immutable native SymbolBinding handles, including typed identity."""
        return tuple(self._receipt.variables)

    @property
    def domain(self):
        return _bounds(self._receipt.domain)

    @property
    def precision_digits(self):
        return self._receipt.precision_digits


class ResidualJacobianEnclosure(_Receipt):
    """Source-bound closed real enclosures; this is not a root-existence claim."""
    __slots__ = ()

    def __init__(self, receipt):
        if hasattr(self, "_receipt"):
            raise AttributeError("accepted numerical receipts are immutable")
        if type(receipt) is not _native.ResidualJacobianEnclosure:
            raise TypeError("an accepted native residual enclosure is required")
        object.__setattr__(self, "_receipt", receipt)

    @property
    def residual_values(self):
        return _bounds(self._receipt.residual_values)

    @property
    def jacobian(self):
        return tuple(_bounds(row) for row in self._receipt.jacobian)

    def __repr__(self):
        return (f"ResidualJacobianEnclosure(residual_values={self.residual_values!r}, "
                f"jacobian={self.jacobian!r}, domain={self.domain!r})")


class RootBoxReceipt(_Receipt):
    """One unique nonsingular real root in the declared closed box only."""
    __slots__ = ()

    def __init__(self, receipt):
        if hasattr(self, "_receipt"):
            raise AttributeError("accepted numerical receipts are immutable")
        if type(receipt) is not _native.RootBoxReceipt:
            raise TypeError("an accepted native root-box receipt is required")
        object.__setattr__(self, "_receipt", receipt)

    @property
    def center(self):
        return tuple(_wrap(value) for value in self._receipt.center)

    @property
    def preconditioner(self):
        return tuple(tuple(_wrap(value) for value in row) for row in self._receipt.preconditioner)

    @property
    def root_enclosure(self):
        return _bounds(self._receipt.root_enclosure)

    @property
    def contraction_bound(self):
        return _wrap(self._receipt.contraction_bound)

    @property
    def claim(self):
        return "one unique nonsingular real root in the declared closed box"

    def __repr__(self):
        return (f"RootBoxReceipt(root_enclosure={self.root_enclosure!r}, "
                f"domain={self.domain!r}, contraction_bound={self.contraction_bound!r}, "
                f"claim={self.claim!r})")


def bound_assumptions(variables, bindings):
    """Capture facts matching the immutable handles actually sent to native.

    The shell's private dictionaries can mutate. Never reread them after native
    compilation and silently attach the new facts to an older typed identity.
    A concurrent inconsistent capture disables certification, not evaluation.
    """
    snapshots = []
    for variable, binding in zip(variables, bindings):
        facts = variable._assumptions
        if type(facts) is not dict or len(facts) > 64:
            return None
        try:
            copied = tuple(facts.items())
        except RuntimeError:
            return None
        if any(type(key) is not str or not key or len(key) > 128 or type(value) is not bool
               for key, value in copied):
            return None
        snapshot = tuple(sorted(copied))
        captured = _native.SymbolBinding(
            binding.name, [(key, "True" if value else "False") for key, value in snapshot]
        )
        if (captured.plain != binding.plain
                or (not captured.plain and captured.identity_hex != binding.identity_hex)):
            return None
        snapshots.append(snapshot)
    return tuple(snapshots)


def _check_assumptions(box, snapshots):
    if snapshots is None or len(snapshots) != len(box):
        raise ValueError("certified boxes require bounded Boolean assumptions matching captured symbol bindings")
    for coordinate, ((lo, hi), facts) in enumerate(zip(box, snapshots)):
        predicates = {
            "positive": lo > 0, "negative": hi < 0,
            "nonnegative": lo >= 0, "nonpositive": hi <= 0,
            "nonzero": lo > 0 or hi < 0, "zero": lo == hi == 0,
        }
        negations = {
            "positive": hi <= 0, "negative": lo >= 0,
            "nonnegative": hi < 0, "nonpositive": lo > 0,
            "nonzero": lo == hi == 0, "zero": lo > 0 or hi < 0,
        }
        for key, value in facts:
            sign = key[9:] if key in (
                "extended_positive", "extended_negative", "extended_nonnegative",
                "extended_nonpositive", "extended_nonzero",
            ) else key
            if sign in predicates:
                compatible = (predicates if value else negations)[sign]
            elif key in ("real", "extended_real", "complex", "finite", "commutative", "hermitian"):
                compatible = value
            elif key in ("imaginary", "infinite"):
                compatible = not value
            else:
                raise NotImplementedError("certified real boxes do not support one or more declared symbol assumptions")
            if not compatible:
                raise ValueError(f"box coordinate {coordinate} violates its declared {key} assumption")


def _prepare_box(program, box, precision, *, square):
    columns, rows = program.native_system.variable_count, program.native_system.residual_count
    if not 1 <= columns <= _MAX_VARIABLES or not 1 <= rows <= _MAX_RESIDUALS:
        raise ValueError("certified enclosures require 1..16 variables and 1..64 residuals")
    if square and rows != columns:
        raise ValueError("root-box certification requires a square system")
    if type(precision) is not int or not 8 <= precision <= 128:
        raise ValueError("certified precision must be an integer from 8 through 128")
    if program._has_inexact_source:
        raise ValueError("certified methods require exact source literals; replace Float with explicit Rational")
    result = []
    for pair in _sequence(box, columns, "box"):
        lo, hi = (_rational(value) for value in _sequence(pair, 2, "coordinate bounds"))
        if lo > hi:
            raise ValueError("box lower bound exceeds upper bound")
        if square and lo == hi:
            raise ValueError("root-box coordinates require positive width")
        result.append((lo, hi))
    result = tuple(result)
    _check_assumptions(result, program._assumptions)
    return result


def root_search_options(precision, max_attempts):
    if type(precision) is not int or not 8 <= precision <= 128:
        raise ValueError("certified precision must be an integer from 8 through 128")
    if type(max_attempts) is not int or not 1 <= max_attempts <= _MAX_ROOT_SEARCH_ATTEMPTS:
        raise ValueError("certified root search max_attempts must be an integer from 1 through 24")


def prepare_root_search(program, precision, max_attempts):
    """Reject static certificate ineligibility before candidate iteration."""
    root_search_options(precision, max_attempts)
    columns, rows = program.native_system.variable_count, program.native_system.residual_count
    if not 1 <= columns <= _MAX_VARIABLES or rows != columns:
        raise ValueError("certified root search requires a square system with 1..16 variables")
    if program._has_inexact_source:
        raise ValueError("certified methods require exact source literals; replace Float with explicit Rational")
    if program._assumptions is None or len(program._assumptions) != columns:
        raise ValueError("certified boxes require bounded Boolean assumptions matching captured symbol bindings")
    supported = {
        "positive", "negative", "nonnegative", "nonpositive", "nonzero", "zero",
        "extended_positive", "extended_negative", "extended_nonnegative",
        "extended_nonpositive", "extended_nonzero", "real", "extended_real",
        "complex", "finite", "commutative", "hermitian", "imaginary", "infinite",
    }
    for facts in program._assumptions:
        if any(key not in supported for key, _ in facts):
            raise NotImplementedError("certified real boxes do not support one or more declared symbol assumptions")


def search_root_box(program, candidate, precision, max_attempts):
    """Propose bounded exact witnesses; only the native checker can accept.

    Floating convergence is deliberately irrelevant here. Every proposal is
    checked against the same immutable original source. The one cumulative
    trial counter includes failed box, assumption, inverse and native checks;
    changing widths or precision never resets it. Each native call also has
    its existing fixed arithmetic/allocation bounds. These are synchronous
    finite work envelopes, not a supervised runtime cancellation claim.
    """
    from ..solvers.numerical import RootSearchError

    prepare_root_search(program, precision, max_attempts)
    dimension = program.native_system.variable_count
    if (type(candidate) not in (tuple, list) or len(candidate) != dimension
            or any(type(value) is not float or not math.isfinite(value) for value in candidate)):
        raise RootSearchError("candidate", "certified root search received an invalid finite-real candidate")
    # Exact conversion does not upgrade a float's evidence: these rationals
    # are merely witness centers, checked independently against the source.
    center = tuple(Fraction(*value.as_integer_ratio()) for value in candidate)
    radii = tuple((abs(value) or Fraction(1)) / (1 << 20) for value in center)
    failures = []
    for attempt in range(max_attempts):
        # Start near the candidate, then alternate shrinking and widening.
        # Narrow boxes resolve local domain/dependency excess; wider boxes can
        # include a root just outside an inaccurate binary64 proposal. Neither
        # direction grants acceptance, and no global search is implied.
        exponent = 0 if attempt == 0 else 4 * ((attempt + 1) // 2) * (-1 if attempt % 2 else 1)
        scale = Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)
        box = tuple((value - radius * scale, value + radius * scale)
                    for value, radius in zip(center, radii))
        digits = min(128, precision << (attempt // 2))
        try:
            return certify_root_box(program, box, center, None, digits)
        except ValueError as error:
            # Keep the bounded diagnostic for each rejected trial. Unsupported
            # source/type errors remain explicit and never become a fallback.
            failures.append(str(error))
    raise RootSearchError(
        "certification",
        f"no root certificate accepted after {max_attempts} bounded trials; last rejection: {failures[-1]}",
        failures=failures,
    )


def enclose(program, box, precision):
    box = _prepare_box(program, box, precision, square=False)
    return ResidualJacobianEnclosure(program.native_system.enclose(_native_box(box), precision))


def _bounded_inverse(matrix):
    """Bounded exact Gaussian elimination proposing an inverse, without floats."""
    n = len(matrix)
    rows = [list(row) + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(matrix)]
    operations = 0

    def op(left, right, operation):
        nonlocal operations
        operations += 1
        # Each operand is already <=4096 bits: pre-normalization multiplication
        # and addition have at most 8193 bits. No unbounded intermediate work.
        if operations > 32768:
            raise ValueError("preconditioner proposal exhausted its arithmetic budget")
        if max(abs(left.numerator).bit_length(), left.denominator.bit_length(),
               abs(right.numerator).bit_length(), right.denominator.bit_length()) > _MAX_BITS:
            raise ValueError("preconditioner proposal exceeds its rational bit limit")
        value = operation(left, right)
        if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > _MAX_BITS:
            raise ValueError("preconditioner proposal exceeds its rational bit limit")
        return value

    for column in range(n):
        pivot = next((i for i in range(column, n) if rows[i][column]), None)
        if pivot is None:
            raise ValueError("midpoint Jacobian is singular; supply a valid exact preconditioner")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [op(value, divisor, lambda a, b: a / b) for value in rows[column]]
        for row in range(n):
            if row == column or not rows[row][column]:
                continue
            scale = rows[row][column]
            rows[row] = [op(left, op(scale, right, lambda a, b: a * b), lambda a, b: a - b)
                         for left, right in zip(rows[row], rows[column])]
    return tuple(tuple(row[n:]) for row in rows)


def certify_root_box(program, box, center, preconditioner, precision):
    box = _prepare_box(program, box, precision, square=True)
    dimension = len(box)
    if center is None:
        center = tuple((lo + hi) / 2 for lo, hi in box)
    else:
        center = tuple(_rational(value) for value in _sequence(center, dimension, "center"))
    if any(not lo <= value <= hi for (lo, hi), value in zip(box, center)):
        raise ValueError("root-box center must lie inside the declared box")
    if preconditioner is None:
        point = program.native_system.enclose(_native_box(tuple((x, x) for x in center)), precision)
        matrix = tuple(tuple((_rational(_wrap(lo)) + _rational(_wrap(hi))) / 2
                             for lo, hi in row) for row in point.jacobian)
        preconditioner = _bounded_inverse(matrix)
    else:
        preconditioner = tuple(tuple(_rational(value) for value in _sequence(row, dimension, "preconditioner row"))
                               for row in _sequence(preconditioner, dimension, "preconditioner"))
    receipt = program.native_system.certify_root_box(
        _native_box(box), [_native_number(x) for x in center],
        [[_native_number(q) for q in row] for row in preconditioner], precision,
    )
    return RootBoxReceipt(receipt)
