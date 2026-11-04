"""Generalized series preserve exact coefficients and source symbol identity."""

import unittest

import sympy as sp


class GeneralizedSeriesTests(unittest.TestCase):
    def test_analytic_pole_keeps_cross_term(self):
        x = sp.Symbol("x")
        actual = sp.series((1 / x + sp.sin(x)) ** 2, x, 0, 1).removeO()
        self.assertEqual(actual, x ** -2 + 2)

    def test_laurent_power_retains_terms_brought_back_by_poles(self):
        x = sp.Symbol("x")
        actual = sp.series((1 / x + x ** 2) ** 4, x, 0, 2).removeO()
        self.assertEqual(actual, x ** -4 + 4 / x)

    def test_logarithm_placeholder_name_is_a_normal_user_parameter(self):
        x = sp.Symbol("x")
        a, b = sp.symbols("_fsym_log_t _fsym_log_t_1")
        actual = sp.series(a + b + sp.exp(x), x, 0, 3).removeO()
        self.assertEqual(actual, a + b + 1 + x + x ** 2 / 2)
        self.assertEqual(actual.free_symbols, {a, b, x})

    def test_series_coordinate_name_is_a_normal_user_parameter(self):
        x = sp.Symbol("x")
        a = sp.Symbol("_fsym_series_t")
        actual = sp.series(a + sp.sin(x), x, 0, 3).removeO()
        self.assertEqual(actual, a + x)

    def test_series_center_can_share_the_preferred_coordinate_name(self):
        x = sp.Symbol("x")
        a = sp.Symbol("_fsym_series_t")
        actual = sp.series(sp.sin(x), x, a, 2).removeO()
        expected = sp.sin(a) + sp.cos(a) * (x - a)
        self.assertEqual(sp.expand(actual - expected), 0)

    def test_expansion_variable_can_share_the_logarithm_placeholder_name(self):
        x = sp.Symbol("_fsym_log_t")
        actual = sp.series(sp.log(x) + x, x, 0, 2).removeO()
        self.assertEqual(actual, sp.log(x) + x)

    def test_limit_scale_names_do_not_capture_positive_parameters(self):
        x = sp.Symbol("x")
        for name in ("_fsym_w1", "_fsym_w1_1", "_fsym_log_t"):
            with self.subTest(name=name):
                parameter = sp.Symbol(name, positive=True)
                self.assertEqual(sp.limit(parameter + sp.exp(-x), x, sp.oo), parameter)


if __name__ == "__main__":
    unittest.main()
