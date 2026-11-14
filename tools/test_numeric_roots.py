"""Execute the production numerical-root kernel without loading the native bridge."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "_frankensympy_numeric_roots", ROOT / "python/sympy/polys/_numeric_roots.py"
)
assert SPEC is not None and SPEC.loader is not None
KERNEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(KERNEL)


class NumericRootsTests(unittest.TestCase):
    def assert_root_set(self, actual, expected, relative=5e-13):
        remaining = list(actual)
        self.assertEqual(len(remaining), len(expected))
        for root in expected:
            nearest = min(range(len(remaining)), key=lambda i: abs(remaining[i] - root))
            candidate = remaining.pop(nearest)
            if root == 0:
                self.assertEqual(candidate, 0)
            else:
                self.assertLessEqual(abs(candidate - root) / abs(root), relative)

    def test_known_complex_and_real_roots(self):
        radius = 2 ** 0.25
        cases = [
            ([1, 0, 0, -1], [1, complex(-0.5, math.sqrt(3) / 2), complex(-0.5, -math.sqrt(3) / 2)]),
            ([1, 0, 0, 0, -2], [radius, -radius, radius * 1j, -radius * 1j]),
            ([1, 0, -2, -5], [2.0945514815423265,
                              complex(-1.0472757407711633, 1.1359398890889283),
                              complex(-1.0472757407711633, -1.1359398890889283)]),
        ]
        for coefficients, expected in cases:
            with self.subTest(coefficients=coefficients):
                self.assert_root_set(KERNEL.numeric_roots(coefficients, maxsteps=50), expected)

    def test_large_real_root_does_not_erase_unrelated_complex_roots(self):
        for large in (1e10, 1e20, 1e100):
            with self.subTest(large=large):
                roots = KERNEL.numeric_roots([1, -large, 1, -large], maxsteps=50)
                self.assert_root_set(roots, [large, 1j, -1j])
                self.assertEqual(sum(root.imag == 0 for root in roots), 1)

    def test_tiny_imaginary_roots_are_not_replaced_with_zero(self):
        for magnitude in (1e-12, 1e-50, 1e-150):
            with self.subTest(magnitude=magnitude):
                roots = KERNEL.numeric_roots([1, 0, magnitude * magnitude], maxsteps=50)
                self.assert_root_set(roots, [magnitude * 1j, -magnitude * 1j])
                self.assertTrue(all(root.real == 0 and root.imag != 0 for root in roots))

    def test_reciprocal_horner_preserves_disparate_real_root_scales(self):
        for large in (1e50, 1e150):
            with self.subTest(large=large):
                self.assert_root_set(
                    KERNEL.numeric_roots([1, -large, 1], maxsteps=50),
                    [large, 1 / large],
                )

    def test_common_real_and_complex_coefficient_scales_preserve_roots(self):
        baseline = [1, 0, -2, -5]
        expected = KERNEL.numeric_roots(baseline, maxsteps=50)
        for scalar in (1e-200, 1e200, complex(3, -4), -7):
            with self.subTest(scalar=scalar):
                actual = KERNEL.numeric_roots([scalar * c for c in baseline], maxsteps=50)
                self.assert_root_set(actual, expected)

    def test_ordinary_higher_degree_multiset(self):
        expected = [-5, -2, 1, 3, 7]
        # (x+5)(x+2)(x-1)(x-3)(x-7), expanded independently.
        self.assert_root_set(
            KERNEL.numeric_roots([1, -4, -36, 86, 163, -210], maxsteps=50),
            expected,
        )

    def test_maxsteps_exhaustion_is_terminal(self):
        for limit in (0, 1):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(KERNEL.NoConvergence, "maxsteps"):
                    KERNEL.numeric_roots([1, 0, -2, -5], maxsteps=limit)
        self.assertEqual(len(KERNEL.numeric_roots([1, 0, -2, -5], maxsteps=50)), 3)

    def test_cleanup_false_preserves_small_computed_components(self):
        source = [1, complex(-1, -1e-17)]
        raw = KERNEL.numeric_roots(source, cleanup=False)
        clean = KERNEL.numeric_roots(source, cleanup=True)
        self.assertEqual(raw, [complex(1, 1e-17)])
        self.assertEqual(clean, [complex(1, 0)])

    def test_nonzero_constant_has_unit_relative_residual_at_zero(self):
        residual, _ = KERNEL._measure([1 + 0j, 0j, 1e-24 + 0j], 0j)
        self.assertEqual(residual, 1.0)
        self.assertFalse(KERNEL._accurate([1 + 0j, 0j, 1e-24 + 0j], 0j, 1e-15))

    def test_duplicate_exact_roots_do_not_establish_complete_multiset(self):
        coefficients = [1 + 0j, 0j, 0j, -1 + 0j]
        self.assertTrue(all(KERNEL._measure(coefficients, root)[0] == 0
                            for root in [1 + 0j] * 3))
        self.assertFalse(KERNEL._coefficient_agreement(coefficients, [1 + 0j] * 3, 1e-15))

    def test_exact_zero_multiplicities_and_linear_factors_need_no_sweeps(self):
        self.assert_root_set(KERNEL.numeric_roots([1, -2, 0, 0], maxsteps=0), [2, 0, 0])
        self.assertEqual(KERNEL.numeric_roots([1, 0, 0, 0], maxsteps=0), [0j, 0j, 0j])
        self.assertEqual(KERNEL.numeric_roots([0, 0]), [])
        self.assertEqual(KERNEL.numeric_roots([7]), [])

    def test_nonfinite_coefficients_are_refused(self):
        for value in (math.inf, -math.inf, math.nan, complex(0, math.inf)):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite"):
                    KERNEL.numeric_roots([1, value])

    def test_normalization_cannot_erase_a_nonzero_coefficient_or_component(self):
        cases = [
            [1e308, 1e-308],
            [1e308 + 0j, complex(1e308, 1e-300)],
            [1, 1e-321],
        ]
        for coefficients in cases:
            with self.subTest(coefficients=coefficients):
                with self.assertRaisesRegex(ValueError, "binary64 precision"):
                    KERNEL.numeric_roots(coefficients)

    def test_invalid_iteration_precision_and_cleanup_options(self):
        for limit in (-1, 1.5, True):
            with self.subTest(maxsteps=limit):
                with self.assertRaises(ValueError):
                    KERNEL.numeric_roots([1, -1], maxsteps=limit)
        for digits in (0, -1, 16, 1.5, True):
            with self.subTest(digits=digits):
                with self.assertRaises(ValueError):
                    KERNEL.numeric_roots([1, -1], digits=digits)
        with self.assertRaises(TypeError):
            KERNEL.numeric_roots([1, -1], cleanup=1)

    def test_requested_low_precision_still_returns_the_complete_root_set(self):
        for digits in (1, 2, 5, 10, 15):
            with self.subTest(digits=digits):
                roots = KERNEL.numeric_roots([1, 0, 0, -1], digits=digits, maxsteps=50)
                self.assert_root_set(
                    roots,
                    [1, complex(-0.5, math.sqrt(3) / 2), complex(-0.5, -math.sqrt(3) / 2)],
                    relative=max(5e-13, 10 ** (-digits)),
                )


if __name__ == "__main__":
    unittest.main()
