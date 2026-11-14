"""Bounded binary64 polynomial-root iteration.

This is an approximate numerical lane, not a certified root isolator. Root
updates, scaled residuals and reconstruction of the complete coefficient
vector must agree before a result is returned. NoConvergence is a local
ArithmeticError; it does not impersonate mpmath's exception class.
"""

from __future__ import annotations

import cmath
import math
import sys


class NoConvergence(ArithmeticError):
    """The binary64 root candidate did not meet the numerical admission checks."""


def _finite(value: complex) -> bool:
    return (
        math.isfinite(value.real) and math.isfinite(value.imag)
        and math.isfinite(math.hypot(value.real, value.imag))
    )


def _measure(coefficients: list[complex], z: complex) -> tuple[float, complex | None]:
    """Scaled residual and Newton correction without large powers of z.

    For |z| > 1, evaluate q(w) = w**d * p(1/w) at w = 1/z.
    Then p/p' = z*q/(d*q - w*q'). The normalization denominator is
    sum(abs(a[k]) * abs(z)**(d-k)), evaluated in the same direction.
    In particular it never includes a unit floor that could admit p(0) != 0.
    """
    degree = len(coefficients) - 1
    reverse = abs(z) > 1.0
    argument = 1 / z if reverse else z
    source = reversed(coefficients) if reverse else coefficients
    value = derivative = 0j
    magnitude = 0.0
    for coefficient in source:
        derivative = derivative * argument + value
        value = value * argument + coefficient
        magnitude = magnitude * abs(argument) + abs(coefficient)
    if not _finite(value) or not _finite(derivative) or not math.isfinite(magnitude):
        return math.inf, None
    if magnitude == 0:
        return math.inf, None
    residual = abs(value) / magnitude
    if value == 0:
        return residual, 0j
    denominator = degree * value - argument * derivative if reverse else derivative
    if denominator == 0 or not _finite(denominator):
        return residual, None
    correction = value / denominator
    if reverse:
        correction *= z
    return residual, correction if _finite(correction) else None


def _initial_roots(coefficients: list[complex]) -> list[complex]:
    """Use the coefficient Newton polygon to seed each root-magnitude group.

    One common starting circle can spend the entire iteration budget moving
    from a large root's scale to much smaller roots. The upper hull of the
    log-magnitude coefficients supplies separate circles without changing
    the polynomial or discarding its smaller coefficients.
    """
    ascending = list(reversed(coefficients))
    hull = []
    for power, coefficient in enumerate(ascending):
        if coefficient == 0:
            continue
        point = (power, math.log(abs(coefficient)))
        while len(hull) >= 2:
            left, middle = hull[-2:]
            old_slope = (middle[1] - left[1]) / (middle[0] - left[0])
            new_slope = (point[1] - middle[1]) / (point[0] - middle[0])
            if old_slope > new_slope:
                break
            hull.pop()
        hull.append(point)
    roots = []
    for (left, left_log), (right, right_log) in zip(hull, hull[1:]):
        count = right - left
        try:
            radius = math.exp((left_log - right_log) / count)
        except OverflowError:
            raise NoConvergence("nroots: a starting root scale exceeds binary64") from None
        if radius == 0 or not math.isfinite(radius):
            raise NoConvergence("nroots: a starting root scale exceeds binary64")
        phase = cmath.phase(-ascending[left]) - cmath.phase(ascending[right])
        for index in range(count):
            angle = (phase + 2 * math.pi * (index + 0.125)) / count
            roots.append(radius * complex(math.cos(angle), math.sin(angle)))
    return roots


def _coefficient_agreement(
    coefficients: list[complex], roots: list[complex], tolerance: float
) -> bool:
    """Reject a set of individually good roots that omits another root.

    Multiply normalized linear factors and retain an absolute convolution
    envelope. Cross-products compare the resulting coefficient vector with
    the input up to an overall scalar. The envelope permits rounding in
    cancelling coefficients without giving zero coefficients a unit floor.
    """
    if len(roots) != len(coefficients) - 1:
        return False
    product = [1 + 0j]
    envelope = [1.0]
    for root in sorted(roots, key=abs):
        factor_scale = max(1.0, abs(root))
        leading, constant = 1 / factor_scale, -root / factor_scale
        updated = [0j] * (len(product) + 1)
        bounds = [0.0] * len(updated)
        for index, (coefficient, bound) in enumerate(zip(product, envelope)):
            updated[index] += coefficient * leading
            updated[index + 1] += coefficient * constant
            bounds[index] += bound * leading
            bounds[index + 1] += bound * abs(constant)
        scale = max(bounds)
        if scale == 0 or not math.isfinite(scale) or any(not _finite(c) for c in updated):
            return False
        product = [c / scale for c in updated]
        envelope = [bound / scale for bound in bounds]
    pivot = max(range(len(coefficients)), key=lambda index: abs(coefficients[index]))
    limit = min(0.25, 16 * tolerance * max(1, len(roots)))
    reliability = 16 * sys.float_info.epsilon * max(1, len(roots))
    if abs(product[pivot]) <= reliability * envelope[pivot]:
        return False
    for original, candidate, bound in zip(coefficients, product, envelope):
        error = abs(original * product[pivot] - candidate * coefficients[pivot])
        scale = abs(original) * abs(product[pivot]) + bound * abs(coefficients[pivot])
        if error > limit * scale:
            return False
    return True


def _accurate(coefficients: list[complex], root: complex, tolerance: float) -> bool:
    if not _finite(root) or root == 0:
        # Exact zero roots have already been removed from the constant term.
        return False
    residual, correction = _measure(coefficients, root)
    return (
        residual <= tolerance
        and correction is not None
        and abs(correction) / abs(root) <= tolerance
    )


def _cleanup_root(coefficients: list[complex], root: complex, tolerance: float) -> complex:
    """Project onto an axis only when that root's own error checks admit it."""
    for projected in (complex(root.real, 0), complex(0, root.imag)):
        if (
            projected != 0
            and abs(root - projected) / abs(root) <= tolerance
            and _accurate(coefficients, projected, tolerance)
        ):
            return projected
    return root


def numeric_roots(
    coefficients, *, maxsteps: int = 500, cleanup: bool = True, digits: int = 15
) -> list[complex]:
    """Return finite approximate roots or raise an explicit failure.

    maxsteps bounds simultaneous Aberth-Ehrlich iteration sweeps; there are
    no uncounted polishing steps or restarts. Exact trailing zero roots and
    a linear factor need no iterative work. cleanup=False retains every
    computed component, including small nonzero imaginary or real parts.
    """
    if type(maxsteps) is not int or maxsteps < 0:
        raise ValueError("nroots maxsteps must be a nonnegative integer")
    if type(digits) is not int or not 1 <= digits <= 15:
        raise ValueError("binary64 nroots digits must be an integer from 1 to 15")
    if type(cleanup) is not bool:
        raise TypeError("nroots cleanup must be a boolean")
    coefficients = [complex(coefficient) for coefficient in coefficients]
    if any(not _finite(coefficient) for coefficient in coefficients):
        raise ValueError("nroots requires finite numeric coefficients")
    while coefficients and coefficients[0] == 0:
        coefficients.pop(0)
    if len(coefficients) <= 1:
        return []
    zeros = 0
    while coefficients[-1] == 0:
        coefficients.pop()
        zeros += 1
    if len(coefficients) == 1:
        return [0j] * zeros
    scale = max(max(abs(c.real), abs(c.imag)) for c in coefficients)
    normalized = [coefficient / scale for coefficient in coefficients]
    for original, value in zip(coefficients, normalized):
        for source_part, part in zip((original.real, original.imag), (value.real, value.imag)):
            if source_part != 0 and (part == 0 or abs(part) < sys.float_info.min):
                raise ValueError("nroots coefficient normalization loses binary64 precision")
    coefficients = normalized
    tolerance = max(8 * sys.float_info.epsilon, 10.0 ** (-digits))
    degree = len(coefficients) - 1
    if degree == 1:
        roots = [-coefficients[1] / coefficients[0]]
    else:
        roots = _initial_roots(coefficients)
        for _ in range(maxsteps):
            updated = []
            small_updates = True
            for index, root in enumerate(roots):
                _, correction = _measure(coefficients, root)
                if correction is None:
                    raise NoConvergence("nroots: a Newton correction is singular or nonfinite")
                repulsion = 0j
                if correction != 0:
                    for other_index, other in enumerate(roots):
                        if index == other_index:
                            continue
                        difference = root - other
                        if difference == 0:
                            raise NoConvergence("nroots: simultaneous root candidates collided")
                        repulsion += 1 / difference
                denominator = 1 - correction * repulsion
                if denominator == 0 or not _finite(denominator):
                    raise NoConvergence("nroots: an Aberth correction is singular or nonfinite")
                step = correction / denominator
                candidate = root - step
                if not _finite(step) or not _finite(candidate) or candidate == 0:
                    raise NoConvergence("nroots: a nonzero root exceeds the binary64 lane")
                if abs(step) / abs(candidate) > tolerance:
                    small_updates = False
                updated.append(candidate)
            roots = updated
            if small_updates and all(_accurate(coefficients, root, tolerance) for root in roots):
                break
        else:
            raise NoConvergence(
                "convergence to root failed; try n < %s or maxsteps > %s" % (digits, maxsteps)
            )
    if not all(_accurate(coefficients, root, tolerance) for root in roots):
        raise NoConvergence("nroots: final candidates failed the scaled residual checks")
    if cleanup:
        roots = [_cleanup_root(coefficients, root, tolerance) for root in roots]
    if not _coefficient_agreement(coefficients, roots, tolerance):
        raise NoConvergence("nroots: candidate roots failed complete coefficient reconstruction")
    return roots + [0j] * zeros
