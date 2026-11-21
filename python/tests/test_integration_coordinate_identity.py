"""Integration substitution coordinates never capture original parameters."""

import unittest

import sympy as s


class IntegrationCoordinateIdentityTests(unittest.TestCase):
    def check_primitive(self, coordinate, parameters):
        coefficient = s.Add(*parameters)
        expression = coordinate * (coefficient + s.cos(coordinate**2))
        primitive = s.integrate(expression, coordinate)
        expected = coefficient * coordinate**2 / 2 + s.sin(coordinate**2) / 2
        self.assertNotIsInstance(primitive, s.Integral)
        self.assertEqual(s.expand(primitive - expected), 0)
        self.assertEqual(s.simplify(s.diff(primitive, coordinate) - expression), 0)
        self.assertEqual(primitive.free_symbols, {coordinate, *parameters})

    def test_parameter_named_like_the_first_substitution_coordinate(self):
        self.check_primitive(s.Symbol("x"), [s.Symbol("_fsym_sub1")])

    def test_all_original_suffix_names_are_reserved(self):
        parameters = [s.Symbol(name) for name in (
            "_fsym_sub1", "_fsym_sub1_0", "_fsym_sub1_1", "_fsym_sub1_2",
        )]
        self.check_primitive(s.Symbol("x"), parameters)

    def test_integration_coordinate_can_have_the_internal_preferred_name(self):
        self.check_primitive(s.Symbol("_fsym_sub1"), [s.Symbol("a")])

    def test_same_printed_parameter_names_keep_distinct_assumptions(self):
        parameters = [s.Symbol("_fsym_sub1"),
                      s.Symbol("_fsym_sub1", positive=True),
                      s.Symbol("_fsym_sub1", negative=True)]
        self.check_primitive(s.Symbol("x"), parameters)

    def test_dummy_parameter_keeps_its_identity(self):
        self.check_primitive(s.Symbol("x"), [s.Dummy("_fsym_sub1")])

    def test_same_printed_coordinate_and_parameter_are_not_conflated(self):
        coordinate = s.Symbol("_fsym_sub1")
        parameter = s.Symbol("_fsym_sub1", positive=True)
        self.check_primitive(coordinate, [parameter])


if __name__ == "__main__":
    unittest.main()
