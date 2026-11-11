"""Exact-input adapters for native residual enclosures and local root claims.

The native checker owns acceptance. Rational midpoint inversion here is only a
bounded witness proposal and carries no mathematical acceptance authority.
"""

from fractions import Fraction
import math

from ..core import Integer, Rational, _exact_surface_types, _native, _wrap
from .lambdify import _check_evaluation_budget

__all__ = ["ResidualJacobianEnclosure", "RootBoxReceipt"]

_MAX_BITS = 4096
_MAX_VARIABLES = 16
_MAX_RESIDUALS = 64
_MAX_ROOT_SEARCH_ATTEMPTS = 24


def _sequence(value, length, label):
    if type(value) not in (tuple, list) or len(value) != length:
        raise TypeError(f"{label} must be an exact list/tuple with {length} entries")
    # Copy only the admitted index range: Python may switch threads between
    # the length check and construction even without an explicit GIL release.
    try:
        snapshot = tuple(value[index] for index in range(length))
    except IndexError:
        raise ValueError(f"{label} length changed during its snapshot") from None
    if len(value) != length:
        raise ValueError(f"{label} length changed during its snapshot")
    return snapshot


def _charge_exact_work(budget, *values, bits=None, operations=1):
    if budget is not None:
        if bits is None:
            bits = max((max(abs(value.numerator).bit_length(), value.denominator.bit_length())
                        for value in values), default=1)
        limbs = (bits + 63) // 64 + 1
        # Logical, limb-weighted rational work. Python objects/containers are
        # outside native-buffer accounting; the native checker charges its
        # own arithmetic and conservative interval-work reservations as well.
        budget._charge(64 * limbs * limbs * operations)


def _call_budgeted(function, *args, budget):
    return function(*args) if budget is None else function(*args, budget=budget)


def _rational(value, budget=None):
    cls = type(value)
    if cls is int:
        p, q = value, 1
    elif cls is Fraction:
        p, q = value.numerator, value.denominator
    elif cls in _exact_surface_types() and isinstance(value, (Integer, Rational)):
        p, q = value.p, value.q
    else:
        raise TypeError("certified inputs must be exact int, Rational or fractions.Fraction values")
    bits = max(abs(p).bit_length(), q.bit_length())
    if bits > _MAX_BITS:
        raise ValueError("certified rational input exceeds the 4096-bit limit")
    _charge_exact_work(budget, bits=bits)
    return Fraction(p, q)


def _native_number(value, budget=None):
    _charge_exact_work(budget, value)
    return _native.py_rational(value.numerator, value.denominator)


def _native_box(box, budget=None):
    return [(_native_number(lo, budget), _native_number(hi, budget)) for lo, hi in box]


def _bounds(values):
    return tuple((_wrap(lo), _wrap(hi)) for lo, hi in values)


class _Receipt:
    __slots__ = ("_receipt",)

    def __setattr__(self, name, value):
        raise AttributeError("accepted numerical receipts are immutable")

    def __delattr__(self, name):
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


def _prepare_box(program, box, precision, *, square, budget=None):
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
        lo, hi = (_rational(value, budget) for value in _sequence(pair, 2, "coordinate bounds"))
        _charge_exact_work(budget, lo, hi)
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


def search_root_box(program, candidate, precision, max_attempts, *, budget=None):
    """Propose bounded exact witnesses; only the native checker can accept.

    Floating convergence is deliberately irrelevant here. Every proposal is
    checked against the same immutable original source. The one cumulative
    trial counter includes failed box, assumption, inverse and native checks;
    changing widths or precision never resets it. A supplied caller ledger
    also spans every rational proposal, native enclosure and verification.
    Without one, native calls retain their fixed safety envelopes. Native
    work releases the GIL and observes cooperative cancellation safe points.
    """
    from ..solvers.numerical import RootSearchError

    _check_evaluation_budget(budget)
    prepare_root_search(program, precision, max_attempts)
    dimension = program.native_system.variable_count
    try:
        candidate = _sequence(candidate, dimension, "root candidate")
    except (TypeError, ValueError) as error:
        raise RootSearchError("candidate", "certified root search received an invalid finite-real candidate") from error
    if any(type(value) is not float or not math.isfinite(value) for value in candidate):
        raise RootSearchError("candidate", "certified root search received an invalid finite-real candidate")
    # Exact conversion does not upgrade a float's evidence: these rationals
    # are merely witness centers, checked independently against the source.
    _charge_exact_work(budget, bits=1096, operations=4 * dimension)
    center = tuple(Fraction(*value.as_integer_ratio()) for value in candidate)
    relative_radii = tuple((abs(value) or Fraction(1)) / (1 << 20) for value in center)
    absolute_radii = tuple(max(abs(value), Fraction(1)) / (1 << 20) for value in center)
    failures = []
    for attempt in range(max_attempts):
        # An absolute scale lets tiny nonzero candidates enclose a zero root:
        # relative widening alone can stop just short of that root. Retain
        # relative shrinking for tiny roots whose domains exclude zero.
        # Neither proposal grants acceptance, and no global search is implied.
        radii = relative_radii if attempt % 2 else absolute_radii
        exponent = 0 if attempt == 0 else 4 * ((attempt + 1) // 2) * (-1 if attempt % 2 else 1)
        _charge_exact_work(budget, *center, *radii, bits=None, operations=4 * dimension + 1)
        scale = Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)
        box = tuple((value - radius * scale, value + radius * scale)
                    for value, radius in zip(center, radii))
        digits = min(128, precision << (attempt // 2))
        try:
            result = _call_budgeted(certify_root_box, program, box, center, None, digits, budget=budget)
            _check_evaluation_budget(budget)
            return result
        except ValueError as error:
            # Keep the bounded diagnostic for each rejected trial. Unsupported
            # source/type errors remain explicit and never become a fallback.
            failures.append(str(error))
    raise RootSearchError(
        "certification",
        f"no root certificate accepted after {max_attempts} bounded trials; last rejection: {failures[-1]}",
        failures=failures,
    )


def enclose(program, box, precision, *, budget=None):
    _check_evaluation_budget(budget)
    box = _prepare_box(program, box, precision, square=False, budget=budget)
    receipt = _call_budgeted(program.native_system.enclose, _native_box(box, budget), precision,
                            budget=budget)
    result = ResidualJacobianEnclosure(receipt)
    _check_evaluation_budget(budget)
    return result


def _bounded_inverse(matrix, budget=None):
    """Bounded exact Gaussian elimination proposing an inverse, without floats."""
    n = len(matrix)
    if budget is not None:
        budget._charge(2 * n * n)
    rows = [list(row) + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(matrix)]
    operations = 0

    def op(left, right, operation):
        nonlocal operations
        operations += 1
        # Each operand is already <=4096 bits: pre-normalization multiplication
        # and addition have at most 8193 bits. No unbounded intermediate work.
        if operations > 32768:
            raise _native.EvaluationBudgetExceeded("preconditioner proposal exhausted its arithmetic budget")
        if max(abs(left.numerator).bit_length(), left.denominator.bit_length(),
               abs(right.numerator).bit_length(), right.denominator.bit_length()) > _MAX_BITS:
            raise _native.EvaluationBudgetExceeded("preconditioner proposal exceeds its rational work limit")
        _charge_exact_work(budget, left, right)
        value = operation(left, right)
        if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > _MAX_BITS:
            raise _native.EvaluationBudgetExceeded("preconditioner proposal exceeds its rational work limit")
        return value

    for column in range(n):
        if budget is not None:
            budget._charge(n - column)
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
    result = tuple(tuple(row[n:]) for row in rows)
    _check_evaluation_budget(budget)
    return result


def certify_root_box(program, box, center, preconditioner, precision, *, budget=None):
    _check_evaluation_budget(budget)
    box = _prepare_box(program, box, precision, square=True, budget=budget)
    dimension = len(box)
    if center is None:
        _charge_exact_work(budget, *(value for pair in box for value in pair), operations=2 * dimension)
        center = tuple((lo + hi) / 2 for lo, hi in box)
    else:
        center = tuple(_rational(value, budget) for value in _sequence(center, dimension, "center"))
    if any(not lo <= value <= hi for (lo, hi), value in zip(box, center)):
        raise ValueError("root-box center must lie inside the declared box")
    if preconditioner is None:
        point = _call_budgeted(
            program.native_system.enclose, _native_box(tuple((x, x) for x in center), budget), precision,
            budget=budget,
        )
        matrix = []
        for row in point.jacobian:
            values = []
            for lo, hi in row:
                lo, hi = _rational(_wrap(lo), budget), _rational(_wrap(hi), budget)
                _charge_exact_work(budget, lo, hi, operations=2)
                values.append((lo + hi) / 2)
            matrix.append(tuple(values))
        preconditioner = _bounded_inverse(tuple(matrix), budget)
    else:
        preconditioner = tuple(tuple(_rational(value, budget) for value in _sequence(row, dimension, "preconditioner row"))
                               for row in _sequence(preconditioner, dimension, "preconditioner"))
    receipt = _call_budgeted(
        program.native_system.certify_root_box,
        _native_box(box, budget), [_native_number(x, budget) for x in center],
        [[_native_number(q, budget) for q in row] for row in preconditioner], precision, budget=budget,
    )
    result = RootBoxReceipt(receipt)
    _check_evaluation_budget(budget)
    return result
