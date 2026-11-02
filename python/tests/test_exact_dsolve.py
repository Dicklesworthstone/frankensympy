"""Public exact first-order ODEs composed with the native gradient checker."""

import unittest
from unittest.mock import patch

from sympy import (
    Add, Derivative, Dummy, Eq, Float, Function, Integer, Mul, Pow, Rational,
    Subs, Symbol, checkodesol, dsolve, expand,
)


class ExactDsolveTests(unittest.TestCase):
    def setUp(self):
        self.x = Symbol("x")
        self.f = Function("exact_flow")
        self.y = self.f(self.x)
        self.derivative = Derivative(self.y, self.x)

    def quadratic(self):
        x, y = self.x, self.y
        return 2*x + y + (x + 2*y)*self.derivative, x*x + x*y + y*y

    def quintic(self):
        x, y = self.x, self.y
        return 2*x + y + (x + 5*y**4)*self.derivative, x*x + x*y + y**5

    def test_default_and_exact_hint_compose_the_checked_potential(self):
        equation, potential = self.quadratic()
        for options in ({"simplify": False}, {"hint": "1st_exact", "simplify": False}):
            with self.subTest(options=options):
                relation = dsolve(equation, self.y, **options)
                self.assertIsInstance(relation, Eq)
                self.assertEqual(expand(relation.lhs - potential), 0)
                self.assertEqual(relation.rhs, Symbol("C1"))
                self.assertEqual(
                    checkodesol(equation, relation, solve_for_func=False), (True, 0)
                )
                self.assertEqual(
                    checkodesol(Eq(equation, 0), relation, func=self.y), (True, 0)
                )

    def test_quintic_relation_is_retained_without_partial_root_selection(self):
        equation, potential = self.quintic()
        relation = dsolve(equation, self.y)
        self.assertIsInstance(relation, Eq)
        self.assertEqual(expand(relation.lhs - potential), 0)
        self.assertEqual(relation.rhs, Symbol("C1"))
        self.assertEqual(checkodesol(equation, relation, self.y), (True, 0))

    def test_quadratic_lifting_preserves_both_verified_graphs(self):
        equation, potential = self.quadratic()
        solutions = dsolve(equation, self.y)
        self.assertIsInstance(solutions, list)
        self.assertEqual(len(solutions), 2)
        self.assertNotEqual(solutions[0].rhs, solutions[1].rhs)
        for solution in solutions:
            self.assertEqual(solution.lhs, self.y)
            self.assertEqual(
                expand(potential.subs(self.y, solution.rhs) - Symbol("C1")).simplify(), 0
            )
            self.assertEqual(checkodesol(equation, solution, self.y), (True, 0))

    def test_exact_initial_values_bind_the_implicit_level(self):
        equation, potential = self.quintic()
        relation = dsolve(equation, self.y, ics={self.f(0): 1, self.f(1): 0})
        self.assertEqual(expand(relation.lhs - potential), 0)
        self.assertEqual(relation.rhs, 1)
        self.assertEqual(checkodesol(equation, relation, self.y), (True, 0))
        with self.assertRaises(ValueError):
            dsolve(equation, self.y, ics={self.f(0): 1, self.f(1): 1})

    def test_initial_value_selects_the_quadratic_branch(self):
        equation, potential = self.quadratic()
        solution = dsolve(equation, self.y, ics={self.f(0): 1})
        self.assertIsInstance(solution, Eq)
        self.assertEqual(solution.lhs, self.y)
        self.assertEqual(solution.rhs.subs(self.x, 0), 1)
        self.assertEqual(expand(potential.subs(self.y, solution.rhs) - 1).simplify(), 0)
        self.assertEqual(checkodesol(equation, solution, self.y), (True, 0))
        implicit = dsolve(equation, self.y, simplify=False, ics={self.f(0): 1})
        self.assertEqual(expand(implicit.lhs - potential), 0)
        self.assertEqual(implicit.rhs, 1)

    def test_quadratic_graph_check_uses_an_exact_identity_without_sampling(self):
        equation, _ = self.quadratic()
        solutions = dsolve(equation, self.y)
        for solution in solutions:
            with self.subTest(solution=solution), patch(
                "sympy.core.Expr.evalf",
                side_effect=AssertionError("numerical samples cannot verify an ODE graph"),
            ), patch(
                "sympy.core.Expr.simplify",
                side_effect=AssertionError("checked polynomial reduction must verify this graph"),
            ):
                self.assertEqual(checkodesol(equation, solution, self.y), (True, 0))
                self.assertEqual(checkodesol(equation, solution.rhs), (True, 0))
                self.assertEqual(
                    checkodesol(equation, Eq(solution.rhs, self.y, evaluate=False)),
                    (True, 0),
                )

    def test_quadratic_graph_check_rejects_shifted_and_wrong_coefficient_branches(self):
        equation, _ = self.quadratic()
        solution = dsolve(equation, self.y)[0].rhs
        for candidate in (solution + 1, 2*solution,
                          solution + Rational(1, 10**60)*self.x):
            with self.subTest(candidate=candidate):
                self.assertFalse(checkodesol(equation, Eq(self.y, candidate), self.y)[0])

    def test_quadratic_graph_check_rechecks_the_proposed_potential(self):
        from sympy.core import _native_expr

        equation, _ = self.quadratic()
        solution = dsolve(equation, self.y)[0]
        with patch(
            "sympy.solvers.ode._native.exact_ode_potential_native",
            return_value=_native_expr(Integer(0)),
        ):
            self.assertFalse(checkodesol(equation, solution, self.y)[0])

    def test_quadratic_graph_checked_reduction_refusal_is_terminal(self):
        equation, _ = self.quadratic()
        solution = dsolve(equation, self.y)[0]
        with patch(
            "sympy.solvers.ode._native.groebner_reduce_native",
            side_effect=ValueError("bounded checked graph reduction exhausted test budget"),
        ):
            with self.assertRaisesRegex(ValueError, "exhausted test budget"):
                checkodesol(equation, solution, self.y)

    def test_quadratic_graph_keeps_typed_coordinates_and_fresh_root_identity(self):
        for x in (Symbol("exact_ode_root"), Dummy("exact_ode_root"),
                  Symbol("x", positive=True)):
            with self.subTest(variable=x):
                y = self.f(x)
                equation = 2*x + y + (x + 2*y)*Derivative(y, x)
                for solution in dsolve(equation, y):
                    self.assertEqual(checkodesol(equation, solution, y), (True, 0))
        typed_x = Symbol("x", positive=True)
        typed_y = self.f(typed_x)
        typed_equation = 2*typed_x + typed_y + (typed_x + 2*typed_y)*Derivative(typed_y, typed_x)
        plain_branch = dsolve(self.quadratic()[0], self.y)[0].rhs
        self.assertFalse(checkodesol(typed_equation, Eq(typed_y, plain_branch), typed_y)[0])

    def test_quadratic_graph_original_held_domains_cannot_be_erased(self):
        equation, _ = self.quadratic()
        solution = dsolve(equation, self.y)[0].rhs
        hidden = Mul(0, Pow(self.x, -1, evaluate=False), evaluate=False)
        candidate = Eq(self.y, Add(solution, hidden, evaluate=False), evaluate=False)
        self.assertFalse(checkodesol(equation, candidate, self.y)[0])
        original = Eq(Add(equation, hidden, evaluate=False), hidden, evaluate=False)
        self.assertFalse(checkodesol(original, Eq(self.y, solution), self.y)[0])

    def test_quadratic_graph_large_held_powers_refuse_before_potential_generation(self):
        equation, _ = self.quadratic()
        solution = dsolve(equation, self.y)[0].rhs
        for exponent in (10**9, -10**9, Rational(10**9 + 1, 2), -Rational(10**9 + 1, 2)):
            candidate = Eq(self.y, Add(solution, Pow(2, exponent, evaluate=False), evaluate=False),
                           evaluate=False)
            with self.subTest(exponent=exponent), patch(
                "sympy.solvers.ode._native.exact_ode_potential_native",
                side_effect=AssertionError("unsupported powers must be refused before generation"),
            ):
                with self.assertRaisesRegex(ValueError, "bounded exact ODE graph"):
                    checkodesol(equation, candidate, self.y)

    def test_quadratic_graph_original_large_power_refuses_before_eq_subtraction(self):
        equation, _ = self.quadratic()
        solution = dsolve(equation, self.y)[0]
        original = Eq(Add(equation, Pow(2, 10**9, evaluate=False), evaluate=False), 0,
                      evaluate=False)
        with patch(
            "sympy.solvers.ode._ode_to_symbols",
            side_effect=AssertionError("original source must be admitted before normalization"),
        ):
            for candidate, args in ((solution, (self.y,)), (solution.rhs, ())):
                with self.subTest(explicit_function=bool(args)):
                    with self.assertRaisesRegex(ValueError, "bounded exact polynomial ODE"):
                        checkodesol(original, candidate, *args)

    def test_nonpolynomial_radical_graph_keeps_the_legacy_checker(self):
        from sympy import sqrt
        from sympy.solvers.ode import _check_quadratic_exact_ode_graph

        candidate = sqrt(Symbol("C1") / self.x)
        equation = 2*self.x*self.y*self.derivative + self.y**2
        self.assertIsNone(_check_quadratic_exact_ode_graph(equation, candidate, self.y))

    def test_nonconstant_leading_coefficient_stays_implicit(self):
        x, y = self.x, self.y
        equation = y*y + 1 + 2*x*y*self.derivative
        relation = dsolve(equation, y, hint="1st_exact")
        self.assertEqual(expand(relation.lhs - (x*y*y + x)), 0)
        self.assertEqual(checkodesol(equation, relation, y), (True, 0))

    def test_native_checker_rejects_false_and_singular_factor_relations(self):
        equation, potential = self.quadratic()
        constant = Symbol("C1")
        for bad in (potential + self.x, potential + Rational(1, 10**60)*self.y,
                    self.x*potential, 0):
            with self.subTest(candidate=bad):
                accepted, _ = checkodesol(
                    equation, Eq(bad, constant, evaluate=False), self.y,
                    solve_for_func=False,
                )
                self.assertFalse(accepted)
        self.assertFalse(checkodesol(equation, Eq(0, 0, evaluate=False), self.y)[0])
        scaled = Eq(-Rational(3, 2)*potential + 7, Symbol("K"), evaluate=False)
        self.assertEqual(checkodesol(equation, scaled, self.y), (True, 0))
        self.assertEqual(
            checkodesol(equation, Eq(constant, potential, evaluate=False), self.y),
            (True, 0),
        )

    def test_original_held_domain_exclusions_cannot_cancel_into_exact_lane(self):
        equation, potential = self.quadratic()
        hole = Pow(self.x, -1, evaluate=False)
        hidden = Mul(0, hole, evaluate=False)
        sources = (
            Eq(Add(equation, hidden, evaluate=False), hidden, evaluate=False),
            Eq(Add(equation, Pow(hole, 0, evaluate=False), evaluate=False), 1,
               evaluate=False),
        )
        for source in sources:
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    dsolve(source, self.y, hint="1st_exact", simplify=False)
                self.assertFalse(checkodesol(
                    source, Eq(potential, Symbol("C1"), evaluate=False), self.y
                )[0])
        for forged in (
            Eq(Add(potential, hidden, evaluate=False), Symbol("C1"), evaluate=False),
            Eq(Add(potential, hidden, evaluate=False), hidden, evaluate=False),
        ):
            self.assertFalse(checkodesol(equation, forged, self.y)[0])

    def test_nonexact_nonlinear_derivative_and_inexact_inputs_refuse_exact_hint(self):
        x, y, dy = self.x, self.y, self.derivative
        for equation in (
            y + dy, y + dy**2,
            2*x + y + (x + 2*y + Symbol("a"))*dy,
            Float(2.0)*x + y + (x + 2*y)*dy,
        ):
            with self.subTest(equation=equation):
                with self.assertRaises(ValueError):
                    dsolve(equation, y, hint="1st_exact", simplify=False)

    def test_implicit_derivative_and_inexact_initial_values_refuse(self):
        equation, _ = self.quintic()
        for data in (
            {Subs(self.derivative, self.x, 0): 1},
            {self.f(0): Float(1.0)},
            {self.f(Rational(1, 2)): Symbol("a")},
        ):
            with self.subTest(data=data):
                with self.assertRaises(NotImplementedError):
                    dsolve(equation, self.y, ics=data)

    def test_placeholders_and_constant_names_preserve_coordinate_identity(self):
        for x in (Symbol("_fsym_y0"), Dummy("_fsym_y0"), Symbol("C1")):
            with self.subTest(variable=x):
                y = self.f(x)
                equation = 2*x + y + (x + 2*y)*Derivative(y, x)
                relation = dsolve(equation, y, simplify=False)
                self.assertEqual(expand(relation.lhs - (x*x + x*y + y*y)), 0)
                self.assertEqual(relation.lhs.free_symbols, {x})
                self.assertEqual(relation.rhs, Symbol("C2" if x.name == "C1" else "C1"))
                self.assertEqual(checkodesol(equation, relation, y), (True, 0))

    def test_operational_native_refusals_do_not_fall_back(self):
        equation, _ = self.quadratic()
        with patch(
            "sympy.solvers.exact_ode._native.exact_ode_potential_native",
            side_effect=ValueError("bounded exact polynomial ODE: exhausted test budget"),
        ):
            with self.assertRaisesRegex(ValueError, "exhausted test budget"):
                dsolve(equation, self.y)

    def test_existing_linear_and_separable_routes_remain_usable(self):
        x, y, dy = self.x, self.y, self.derivative
        for equation in (dy - y, dy - x*y*y, y*dy - x):
            with self.subTest(equation=equation):
                with patch(
                    "sympy.solvers.exact_ode._native.exact_ode_potential_native",
                    side_effect=AssertionError("existing successful route must retain precedence"),
                ), patch(
                    "sympy.solvers.ode._exact_ode_source",
                    side_effect=AssertionError("successful legacy route must keep its scope"),
                ):
                    solution = dsolve(equation, y)
                branches = solution if isinstance(solution, list) else [solution]
                for branch in branches:
                    self.assertEqual(branch.lhs, y)
                    self.assertEqual(checkodesol(equation, branch, y), (True, 0))

    def test_exact_source_growth_refuses_before_symbol_substitution(self):
        equation, potential = self.quadratic()
        x, y = self.x, self.y
        oversized = (
            Pow(2, 10**9, evaluate=False),
            Pow(Pow(2, 64, evaluate=False), 64, evaluate=False),
            Mul(*([Add(x, y, evaluate=False)] * 18), evaluate=False),
            Pow(Add(1, x, y, evaluate=False), 16, evaluate=False),
            Integer(1 << 4096),
        )
        relation = Eq(potential, Symbol("C1"), evaluate=False)
        for extra in oversized:
            source = Eq(Add(equation, extra, evaluate=False), 0, evaluate=False)
            with self.subTest(extra=extra):
                with patch(
                    "sympy.solvers.ode._ode_to_symbols",
                    side_effect=AssertionError("resource refusal must precede substitution"),
                ):
                    with self.assertRaisesRegex(ValueError, "bounded exact polynomial ODE"):
                        dsolve(source, self.y, hint="1st_exact", simplify=False)
                    with self.assertRaisesRegex(ValueError, "bounded exact polynomial ODE"):
                        checkodesol(source, relation, self.y)

    def test_candidate_growth_refuses_before_native_verification(self):
        equation, potential = self.quadratic()
        candidate = Eq(
            Add(potential, Pow(2, 10**9, evaluate=False), evaluate=False),
            Symbol("C1"), evaluate=False,
        )
        with patch(
            "sympy.solvers.ode._native.verify_exact_ode_potential_native",
            side_effect=AssertionError("candidate preflight must precede verification"),
        ):
            with self.assertRaisesRegex(ValueError, "bounded exact polynomial ODE"):
                checkodesol(equation, candidate, self.y)

    def test_initial_coordinate_growth_refuses_before_evaluation(self):
        from sympy.solvers.ode import _apply_exact_ode_ics

        equation, _ = self.quintic()
        relation = dsolve(equation, self.y, hint="1st_exact", simplify=False)
        for value in (1 << 1000, Rational(1, 1 << 1000), 1 << 5000):
            with self.subTest(value_bits=value.bit_length() if type(value) is int else 1001):
                with patch(
                    "sympy.core.Basic.subs",
                    side_effect=AssertionError("initial-data preflight must precede evaluation"),
                ):
                    with self.assertRaisesRegex(ValueError, "bounded exact polynomial ODE"):
                        _apply_exact_ode_ics(relation, {self.f(0): value})

    def test_degree_seventeen_potential_remains_inside_checked_bound(self):
        equation = self.x**16 + self.derivative
        relation = dsolve(equation, self.y, hint="1st_exact", simplify=False)
        self.assertEqual(expand(relation.lhs - self.x**17/17 - self.y), 0)
        self.assertEqual(checkodesol(equation, relation, self.y), (True, 0))

    def test_default_inference_preserves_cancelled_forcing_functions(self):
        g = Function("cancelled_forcing")
        equation = Eq(self.derivative + self.y + g(self.x), g(self.x))
        with patch(
            "sympy.solvers.ode._exact_ode_source",
            side_effect=AssertionError("successful legacy route must keep its scope"),
        ):
            solution = dsolve(equation)
        self.assertEqual(solution.lhs, self.y)
        self.assertEqual(checkodesol(equation, solution), (True, 0))


if __name__ == "__main__":
    unittest.main()
