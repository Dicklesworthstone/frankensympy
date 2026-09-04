"""Executable checks for the Python compatibility boundary."""

from __future__ import annotations

import copy
import os
import pickle
import subprocess
import sys
import unittest
from pathlib import Path

import sympy


class SurfaceTests(unittest.TestCase):
    def test_classes_are_usable_and_operations_preserve_surface_kind(self):
        x = sympy.Symbol("x")
        two = sympy.Integer(2)

        self.assertIsInstance(x, sympy.Expr)
        self.assertIsInstance(two, sympy.Expr)
        expression = x + two
        self.assertIsInstance(expression, sympy.Add)
        self.assertEqual(expression.func(*expression.args), expression)
        self.assertEqual(expression.free_symbols, {x})
        self.assertEqual(expression.subs(x, 3), sympy.Integer(5))

    def test_held_forms_copy_and_pickle_without_collapsing_identity(self):
        x = sympy.Symbol("x")
        held = sympy.Add(x, x, evaluate=False)

        self.assertIsInstance(held, sympy.Add)
        self.assertEqual(len(held.args), 2)
        self.assertIs(copy.deepcopy(held), held)
        restored = pickle.loads(pickle.dumps(held))  # ubs:ignore — trusted in-process bytes
        self.assertIsInstance(restored, sympy.Add)
        self.assertEqual(restored.args, held.args)

    def test_function_wrappers_return_expressions_not_wire_strings(self):
        x = sympy.Symbol("x")

        self.assertEqual(sympy.diff(x**2, x), 2 * x)
        self.assertIsInstance(sympy.simplify(x + 0), sympy.Expr)
        self.assertEqual(sympy.integrate(2 * x, (x, 0, 1)), sympy.Integer(1))
        self.assertEqual(sympy.solve(2 * x - 4, x), [sympy.Integer(2)])
        self.assertEqual(sympy.solve(sympy.Eq(2 * x, 4), x), [sympy.Integer(2)])
        with self.assertRaises(NotImplementedError):
            sympy.dsolve(x)

    def test_atoms_and_held_equality(self):
        x, y = sympy.symbols("x y")
        expr = x + 2 * y + 1
        self.assertEqual(expr.atoms(sympy.Symbol), {x, y})
        self.assertIn(sympy.Integer(1), expr.atoms(sympy.Integer))
        self.assertTrue(all(arg.args == () for arg in expr.atoms()))

        relation = sympy.Eq(x, 2)
        self.assertIs(type(relation), sympy.Eq)
        self.assertEqual(relation.lhs, x)
        self.assertEqual(relation.rhs, sympy.Integer(2))
        restored = pickle.loads(pickle.dumps(relation))  # ubs:ignore — trusted in-process bytes
        self.assertIs(type(restored), sympy.Eq)
        self.assertEqual(restored.lhs, x)
        self.assertEqual(expr.atoms(sympy.Add), {expr})

        unequal = sympy.Ne(x, y)
        self.assertIs(type(unequal), sympy.Ne)
        self.assertEqual(unequal.lhs, x)
        self.assertEqual(sympy.Lt(x, 1).rel_op, "<")
        self.assertEqual(sympy.Le(x, 1).rel_op, "<=")
        self.assertEqual(sympy.Gt(x, 1).rel_op, ">")
        self.assertEqual(sympy.Ge(x, 1).rel_op, ">=")
        restored_ne = pickle.loads(pickle.dumps(unequal))  # ubs:ignore — trusted in-process bytes
        self.assertIs(type(restored_ne), sympy.Ne)

        held = sympy.Derivative(x**2, x, evaluate=False)
        self.assertEqual(held.doit(), 2 * x)
        terms = sympy.Add(y, x, sympy.Integer(1), evaluate=False).as_ordered_terms()
        self.assertEqual(set(terms), {x, y, sympy.Integer(1)})
        self.assertEqual(terms, tuple(sorted(terms, key=lambda term: term.sort_key())))

        coeff, rest = (2 * x * y).as_coeff_Mul()
        self.assertEqual(coeff, sympy.Integer(2))
        self.assertEqual(rest, x * y)
        add_coeff, add_rest = (x + 3).as_coeff_Add()
        self.assertEqual(add_coeff, sympy.Integer(3))
        self.assertEqual(add_rest, x)
        self.assertEqual((x**2).as_base_exp(), (x, sympy.Integer(2)))
        self.assertEqual(x.as_base_exp(), (x, sympy.Integer(1)))
        self.assertEqual(expr.find(sympy.Symbol), {x, y})
        self.assertEqual(expr.find(x), {x})

    def test_as_numer_denom_and_could_extract_minus_sign(self):
        x, y, z = sympy.symbols("x y z")
        one = sympy.Integer(1)
        two = sympy.Integer(2)
        three = sympy.Integer(3)

        self.assertEqual(two.as_numer_denom(), (two, one))
        self.assertEqual(sympy.Integer(-3).as_numer_denom(), (sympy.Integer(-3), one))
        self.assertEqual(sympy.Rational(2, 3).as_numer_denom(), (two, three))
        self.assertEqual(sympy.Rational(-2, 3).as_numer_denom(), (sympy.Integer(-2), three))
        float_one = sympy.Float(1.5)
        self.assertEqual(float_one.as_numer_denom(), (float_one, one))
        self.assertEqual(x.as_numer_denom(), (x, one))
        self.assertEqual((x**2).as_numer_denom(), (x**2, one))
        self.assertEqual((1 / x).as_numer_denom(), (one, x))
        self.assertEqual((x ** (-2)).as_numer_denom(), (one, x**2))
        self.assertEqual((x / y).as_numer_denom(), (x, y))
        self.assertEqual((2 * x / 3).as_numer_denom(), (2 * x, three))
        self.assertEqual((x + 1).as_numer_denom(), (x + 1, one))

        half_sum_n, half_sum_d = (x / 2 + sympy.Rational(1, 2)).as_numer_denom()
        self.assertEqual(half_sum_d, two)
        self.assertEqual(half_sum_n, x + 1)
        mixed_n, mixed_d = (x / 2 + y / 3).as_numer_denom()
        self.assertEqual(mixed_d, sympy.Integer(6))
        self.assertEqual(mixed_n, 3 * x + 2 * y)
        same_n, same_d = (x / y + z / y).as_numer_denom()
        self.assertEqual(same_d, y)
        self.assertEqual(same_n, x + z)
        conservative = x / y + 1
        self.assertEqual(conservative.as_numer_denom(), (conservative, one))

        self.assertFalse(two.could_extract_minus_sign())
        self.assertTrue(sympy.Integer(-3).could_extract_minus_sign())
        self.assertFalse(sympy.Rational(2, 3).could_extract_minus_sign())
        self.assertTrue(sympy.Rational(-2, 3).could_extract_minus_sign())
        self.assertFalse(sympy.Float(1.5).could_extract_minus_sign())
        self.assertTrue(sympy.Float(-1.5).could_extract_minus_sign())
        self.assertFalse(x.could_extract_minus_sign())
        self.assertTrue((-x).could_extract_minus_sign())
        self.assertTrue((-2 * x).could_extract_minus_sign())
        self.assertFalse((2 * x).could_extract_minus_sign())
        self.assertFalse((x - y).could_extract_minus_sign())
        self.assertTrue((y - x).could_extract_minus_sign())
        self.assertFalse((x - 1).could_extract_minus_sign())
        self.assertTrue((1 - x).could_extract_minus_sign())
        self.assertFalse((x * (y - x)).could_extract_minus_sign())
        self.assertFalse(sympy.Integer(0).could_extract_minus_sign())

    def test_constants_are_native_constants_not_spoofed_symbols(self):
        for constant in (sympy.pi, sympy.E, sympy.I, sympy.oo, sympy.zoo, sympy.nan):
            self.assertIsInstance(constant, sympy.Expr)
            self.assertNotIsInstance(constant, sympy.Symbol)
            self.assertFalse(constant.is_symbol)

    def test_custom_subclasses_are_not_silently_collapsed_to_native_nodes(self):
        class CustomSymbol(sympy.Symbol):
            pass

        custom = CustomSymbol("custom")
        self.assertIs(type(custom), CustomSymbol)
        with self.assertRaisesRegex(NotImplementedError, "exact built-in classes only"):
            custom + 1

    def test_number_theory_wrappers_admit_only_exact_integers(self):
        with self.assertRaisesRegex(ValueError, r"^2\.9 is not an integer$"):
            sympy.isprime(2.9)
        with self.assertRaisesRegex(ValueError, r"^2\.9 is not an integer$"):
            sympy.factorint(2.9)
        with self.assertRaisesRegex(TypeError, r"^n should be an integer$"):
            sympy.totient(2.9)

        class LossyInteger:
            calls = 0

            def __int__(self):
                self.calls += 1
                return 2

            def __str__(self):
                self.calls += 1
                return "2"

            def __repr__(self):
                self.calls += 1
                return "2"

            def __format__(self, format_spec):
                del format_spec
                self.calls += 1
                return "2"

        lossy = LossyInteger()
        with self.assertRaises(ValueError):
            sympy.isprime(lossy)
        with self.assertRaises(ValueError):
            sympy.factorint(lossy)
        with self.assertRaises(TypeError):
            sympy.Integer(lossy)
        with self.assertRaises(TypeError):
            sympy.Rational(lossy, 1)
        self.assertEqual(lossy.calls, 0)

        self.assertEqual(sympy.Integer(2.9), sympy.Integer(2))
        self.assertEqual(sympy.Integer(True), sympy.Integer(1))
        self.assertEqual(
            sympy.Rational(1.9, 2),
            sympy.Rational(4278419646001971, 4503599627370496),
        )

    def test_numeric_bridge_preserves_values_beyond_machine_and_decimal_limits(self):
        huge = 1 << 20_000
        integer = sympy.Integer(huge)
        self.assertEqual(integer.p, huge)
        self.assertEqual(integer.q, 1)

        rational = sympy.Rational(huge, 3)
        self.assertEqual(rational.p, huge)
        self.assertEqual(rational.q, 3)
        trusted_payload = pickle.dumps(rational)
        self.assertEqual(
            pickle.loads(trusted_payload),  # nosec B301  # ubs:ignore — trusted in-process bytes
            rational,
        )

        normalized = sympy.Rational(-(1 << 63), -1)
        self.assertEqual(normalized.p, 1 << 63)
        self.assertEqual(normalized.q, 1)

    def test_custom_symbol_variable_refuses_before_running_overrides(self):
        effects = []

        class EffectfulSymbol(sympy.Symbol):
            @property
            def name(self):
                effects.append("name")
                return "x"

        variable = EffectfulSymbol("x")
        with self.assertRaisesRegex(
            NotImplementedError, "supervised Python override lane"
        ):
            sympy.diff(sympy.Symbol("x"), variable)
        self.assertEqual(effects, [])

    def test_number_theory_wrappers_preserve_signed_and_zero_domains(self):
        self.assertTrue(sympy.isprime(sympy.Integer(2)))
        self.assertFalse(sympy.isprime(-2))
        self.assertEqual(sympy.factorint(-12), {2: 2, 3: 1, -1: 1})
        self.assertEqual(sympy.factorint(0), {0: 1})
        with self.assertRaisesRegex(ValueError, r"^n should be a positive integer$"):
            sympy.totient(0)

    def test_missing_native_extension_fails_closed(self):
        package_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(package_root)
        completed = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "import sys; sys.modules['fsym_python'] = None; import sympy",
            ],
            cwd=package_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("requires its fsym_python native extension", completed.stderr)

    def test_dummy_symbols_are_distinct_and_serialize(self):
        d1 = sympy.Dummy("x")
        d2 = sympy.Dummy("x")
        s = sympy.Symbol("x")

        self.assertIsInstance(d1, sympy.Dummy)
        self.assertIsInstance(d1, sympy.Symbol)
        self.assertEqual(d1.name, "x")
        self.assertEqual(d2.name, "x")
        self.assertNotEqual(d1, d2)
        self.assertNotEqual(d1, s)
        self.assertNotEqual(d1.dummy_index, d2.dummy_index)

        # Free symbols
        expr = d1 + d2 + s
        self.assertEqual(expr.free_symbols, {d1, d2, s})

        # Calculus differentiation
        self.assertEqual(sympy.diff(d1**2, d1), 2 * d1)
        self.assertEqual(sympy.diff(d1**2, d2), sympy.Integer(0))

        # Serialization round-trip
        restored = pickle.loads(pickle.dumps(d1))  # ubs:ignore — trusted in-process bytes
        self.assertIsInstance(restored, sympy.Dummy)
        self.assertEqual(restored, d1)
        self.assertEqual(restored.dummy_index, d1.dummy_index)

        # Reserved prefix collision rejection
        with self.assertRaises(ValueError):
            sympy.Symbol("__fsymDummy_1_x")

    def test_undefined_function_application_round_trips(self):
        x = sympy.Symbol("x")
        f = sympy.Function("f")
        applied = f(x)

        self.assertIsInstance(applied, sympy.Expr)
        self.assertEqual(applied.func, f)
        self.assertEqual(applied.args, (x,))
        self.assertEqual(applied.func(*applied.args), applied)
        self.assertEqual(applied.free_symbols, {x})

        summed = applied + 1
        self.assertTrue(any(type(arg) is type(applied) for arg in summed.args))
        restored = pickle.loads(pickle.dumps(applied))  # ubs:ignore — trusted in-process bytes
        self.assertEqual(restored, applied)
        self.assertEqual(restored.func, f)

        with self.assertRaises(ValueError):
            sympy.Function("")
        with self.assertRaises(TypeError):
            sympy.Function(1)

    def test_class_hierarchy_and_mro(self):
        self.assertTrue(issubclass(sympy.Symbol, sympy.AtomicExpr))
        self.assertTrue(issubclass(sympy.Dummy, sympy.Symbol))
        self.assertTrue(issubclass(sympy.AtomicExpr, sympy.Expr))
        self.assertTrue(issubclass(sympy.AtomicExpr, sympy.Atom))
        self.assertTrue(issubclass(sympy.Atom, sympy.Basic))
        self.assertTrue(issubclass(sympy.Expr, sympy.Basic))
        self.assertTrue(issubclass(sympy.Number, sympy.AtomicExpr))
        self.assertTrue(issubclass(sympy.Rational, sympy.Number))
        self.assertTrue(issubclass(sympy.Integer, sympy.Rational))
        self.assertTrue(issubclass(sympy.Float, sympy.Number))
        self.assertFalse(issubclass(sympy.Float, sympy.Rational))
        self.assertTrue(issubclass(sympy.Add, sympy.Expr))
        self.assertTrue(issubclass(sympy.Mul, sympy.Expr))
        self.assertTrue(issubclass(sympy.Pow, sympy.Expr))
        self.assertTrue(issubclass(sympy.Derivative, sympy.Expr))
        self.assertTrue(issubclass(sympy.AppliedUndef, sympy.Expr))

    def test_symbols_utility_function(self):
        x, y, z = sympy.symbols("x y z")
        self.assertIsInstance(x, sympy.Symbol)
        self.assertIsInstance(y, sympy.Symbol)
        self.assertIsInstance(z, sympy.Symbol)
        self.assertEqual(x.name, "x")
        self.assertEqual(y.name, "y")
        self.assertEqual(z.name, "z")

        a, b = sympy.symbols("a, b")
        self.assertEqual(a.name, "a")
        self.assertEqual(b.name, "b")

        single = sympy.symbols("single")
        self.assertIsInstance(single, sympy.Symbol)
        self.assertEqual(single.name, "single")

        with self.assertRaises(ValueError):
            sympy.symbols("")
        with self.assertRaises(ValueError):
            sympy.symbols("  ")

    def test_basic_methods_has_and_subs(self):
        x, y = sympy.symbols("x y")
        expr = x + 2 * y + 1

        self.assertTrue(expr.has(x))
        self.assertTrue(expr.has(y))
        self.assertFalse(expr.has(sympy.Symbol("z")))

        substituted = expr.subs(x, 2)
        expected = sympy.Integer(2) + 2 * y + 1
        self.assertEqual(substituted, expected)

    def test_xreplace_replaces_exact_nodes_without_inventing_algebra(self):
        x, y = sympy.symbols("x y")
        expr = x + 1
        self.assertEqual(expr.xreplace({expr: y}), y)
        self.assertEqual(x.xreplace({x: y}), y)
        self.assertEqual(x.xreplace({y: 1}), x)
        self.assertEqual((x + 2).xreplace({x: sympy.Integer(3)}), sympy.Integer(5))

        held = sympy.Add(x, x, evaluate=False)
        unchanged = held.xreplace({y: 1})
        self.assertIs(type(unchanged), sympy.Add)
        self.assertEqual(len(unchanged.args), 2)

        with self.assertRaises(TypeError):
            x.xreplace(1)

    def test_singleton_registry_exposes_exact_atoms_and_constructs_float(self):
        self.assertEqual(sympy.S.Zero, sympy.Integer(0))
        self.assertEqual(sympy.S.One, sympy.Integer(1))
        self.assertEqual(sympy.S.NegativeOne, sympy.Integer(-1))
        self.assertEqual(sympy.S.Half, sympy.Rational(1, 2))
        self.assertEqual(sympy.S(2), sympy.Integer(2))
        self.assertEqual(sympy.S(True), sympy.Integer(1))
        one = sympy.S.One
        self.assertIs(sympy.S(one), one)
        self.assertEqual(str(sympy.S.Pi), "pi")
        self.assertEqual(str(sympy.S.Infinity), "oo")
        self.assertEqual(str(sympy.S.ComplexInfinity), "zoo")
        self.assertIsInstance(sympy.S(1.5), sympy.Float)
        with self.assertRaises(TypeError):
            sympy.S(object())

    def test_float_is_a_number_atom_not_a_rational(self):
        value = sympy.Float(1.5)
        self.assertIs(type(value), sympy.Float)
        self.assertIsInstance(value, sympy.Number)
        self.assertNotIsInstance(value, sympy.Rational)
        self.assertEqual(value.args, ())
        self.assertIs(value.func, sympy.Float)
        self.assertFalse(value.is_symbol)
        self.assertTrue(value.is_number)
        self.assertFalse(value.is_integer)
        self.assertFalse(value.is_rational)
        self.assertAlmostEqual(value.evalf(), 1.5)
        self.assertEqual(value, sympy.Rational(3, 2))
        self.assertEqual(sympy.Float(1.0), sympy.Integer(1))
        self.assertEqual(sympy.Integer(1), 1.0)
        self.assertEqual(hash(sympy.Float(1.0)), hash(sympy.Integer(1)))
        self.assertEqual(hash(sympy.Integer(1)), hash(1))
        self.assertNotEqual(sympy.Float(0.1), sympy.Rational(1, 10))
        self.assertEqual(value, sympy.Float(1.5))
        self.assertEqual(sympy.Float(3) + sympy.Float(0.5), sympy.Float(3.5))
        self.assertEqual(sympy.Integer(sympy.Float(2.9)), sympy.Integer(2))

        x = sympy.Symbol("x")
        summed = x + 1.5
        self.assertIsInstance(summed, sympy.Add)
        self.assertTrue(any(type(arg) is sympy.Float for arg in summed.args))

        restored = pickle.loads(pickle.dumps(value))  # ubs:ignore — trusted in-process bytes
        self.assertIs(type(restored), sympy.Float)
        self.assertEqual(restored, value)

        with self.assertRaises(TypeError):
            sympy.Float(object())
        with self.assertRaises(ValueError):
            sympy.Function("__fsymFloat")

        class LossyFloat:
            def __float__(self):
                return 1.5

        with self.assertRaises(TypeError):
            sympy.Float(LossyFloat())

    def test_float_atom_pins_ieee_binary64_contract(self) -> None:
        # Float is profile-compatible binary64. The intern encoding stores the
        # IEEE binary64 bit pattern as a reserved function payload, so a
        # future change to (a) the bit-packing, (b) the reserved name, or
        # (c) the dps validation would silently break the wire and persistence
        # contract. Pin each surface so any change is loud.
        # Round-trip: Float(value).evalf() returns the same Python float.
        for value in [0.0, 1.0, -1.0, 1.5, -1.5, 1.0e100, 1.0e-100, float("inf"), -float("inf")]:
            self.assertEqual(float(sympy.Float(value).evalf()), float(value))
        # The intern encoding stores IEEE binary64 bits in big-endian order.
        import struct
        for value, expected_bits in [(1.5, 0x3FF8000000000000), (0.0, 0), (-0.0, 0x8000000000000000), (float("inf"), 0x7FF0000000000000)]:
            self.assertEqual(sympy.Float(value)._value.args[0].exact_numerator(), expected_bits)
        with self.assertRaises(TypeError):
            sympy.Float(1.0, dps=0)
        with self.assertRaises(TypeError):
            sympy.Float(1.0, dps=-1)
        # Default dps is 15 (matches Python's repr(float) at full precision).
        self.assertEqual(sympy.Float(1.0).dps, 15)
        # __fsymFloat is the reserved intern name; user Function cannot collide.
        with self.assertRaises(ValueError):
            sympy.Function("__fsymFloat")
        # Float is a Number, not a Rational.
        self.assertTrue(issubclass(sympy.Float, sympy.Number))
        self.assertFalse(issubclass(sympy.Float, sympy.Rational))


    def test_latex_and_evalf_representations(self) -> None:
        x = sympy.Symbol("x")
        expr = x / 2
        latex_repr = expr._repr_latex_()
        self.assertIsInstance(latex_repr, str)
        self.assertTrue(len(latex_repr) > 0)

        two = sympy.Integer(2)
        self.assertIsInstance(two.evalf(), sympy.Float)
        self.assertEqual(two.evalf(), sympy.Float(2.0))
        self.assertAlmostEqual(float(sympy.pi.evalf()), 3.1415926535, places=4)
        self.assertEqual(sympy.N(2), sympy.Float(2.0))
        self.assertIsInstance(sympy.N(sympy.pi), sympy.Float)

        x, y = sympy.symbols("x y")
        ordered = sorted([y, sympy.Integer(1), x + 1, x], key=lambda expr: expr.sort_key())
        self.assertEqual(ordered[0], sympy.Integer(1))
        self.assertEqual({elt.name for elt in ordered[1:] if type(elt) is sympy.Symbol}, {"x", "y"})
        self.assertLess(sympy.Integer(1).sort_key(), x.sort_key())
        self.assertLess(x.sort_key(), (x + 1).sort_key())

        powered = x**2
        self.assertEqual(sympy.pretty(powered), "x²")
        self.assertEqual(powered.pretty(), "x²")
        self.assertEqual(sympy.pretty(x - 1), "x − 1")

    def test_calculus_and_solvers_facades(self):
        x = sympy.Symbol("x")
        self.assertEqual(sympy.diff(x**3, x), 3 * x**2)
        self.assertEqual(sympy.diff(x**3, x, x), 6 * x)
        self.assertEqual(sympy.integrate(x**2, x), sympy.Rational(1, 3) * x**3)
        self.assertEqual(sympy.integrate(x, (x, 0, 2)), sympy.Integer(2))
        self.assertEqual(sympy.integrate(x, x, 0, 2), sympy.Integer(2))
        with self.assertRaises(ValueError):
            sympy.integrate(x, (x, 0))
        with self.assertRaises(TypeError):
            sympy.integrate(x, x, 0)

    def test_elementary_functions_use_native_identity_folds(self):
        x = sympy.Symbol("x")
        self.assertEqual(sympy.sin(0), sympy.Integer(0))
        self.assertEqual(sympy.cos(0), sympy.Integer(1))
        self.assertEqual(sympy.exp(0), sympy.Integer(1))
        self.assertEqual(sympy.log(1), sympy.Integer(0))
        self.assertEqual(sympy.sin(x).func, sympy.Function("sin"))
        self.assertEqual(sympy.sin(x).args, (x,))

    def test_matrix_surface_operations(self):
        # 1. Constructor patterns
        m = sympy.Matrix([[1, 2], [3, 4]])
        self.assertEqual(m.shape, (2, 2))
        self.assertEqual(m.rows, 2)
        self.assertEqual(m.cols, 2)
        self.assertTrue(m.is_square)
        self.assertFalse(m.is_symmetric)
        self.assertEqual(m[0, 0], sympy.Integer(1))
        self.assertEqual(m[0, 1], sympy.Integer(2))
        self.assertEqual(m[1, 0], sympy.Integer(3))
        self.assertEqual(m[1, 1], sympy.Integer(4))

        # 2. Transpose, Trace, Determinant
        mt = m.T
        self.assertEqual(mt[0, 1], sympy.Integer(3))
        self.assertEqual(mt[1, 0], sympy.Integer(2))
        self.assertEqual(m.trace(), sympy.Integer(5))
        self.assertEqual(m.det(), sympy.Integer(-2))

        # 3. Inverse and Adjugate
        inv = m.inv()
        self.assertEqual(inv.shape, (2, 2))
        ident = m @ inv
        self.assertEqual(ident[0, 0], sympy.Integer(1))
        self.assertEqual(ident[0, 1], sympy.Integer(0))
        self.assertEqual(ident[1, 0], sympy.Integer(0))
        self.assertEqual(ident[1, 1], sympy.Integer(1))

        adj = m.adjugate()
        self.assertEqual(adj[0, 0], sympy.Integer(4))
        self.assertEqual(adj[0, 1], sympy.Integer(-2))
        self.assertEqual(adj[1, 0], sympy.Integer(-3))
        self.assertEqual(adj[1, 1], sympy.Integer(1))

        # 4. Arithmetic
        m2 = m + m
        self.assertEqual(m2[0, 0], sympy.Integer(2))
        self.assertEqual(m2[1, 1], sympy.Integer(8))
        diff_m = m - m
        self.assertEqual(diff_m[0, 0], sympy.Integer(0))
        self.assertEqual(diff_m[1, 1], sympy.Integer(0))

        # Scalar multiplication
        scaled = m * 2
        self.assertEqual(scaled[0, 0], sympy.Integer(2))
        self.assertEqual(scaled[1, 1], sympy.Integer(8))

        # Matrix power
        m_sq = m ** 2
        m_mul = m @ m
        self.assertEqual(m_sq.tolist(), m_mul.tolist())

        # 5. Helpers: eye, zeros, diag
        e = sympy.eye(3)
        self.assertEqual(e.shape, (3, 3))
        self.assertTrue(e.is_diagonal)
        self.assertTrue(e.is_symmetric)
        self.assertEqual(e.trace(), sympy.Integer(3))

        z = sympy.zeros(2, 3)
        self.assertEqual(z.shape, (2, 3))
        self.assertEqual(z.rank(), 0)

        d = sympy.diag(1, 2, 3)
        self.assertEqual(d.shape, (3, 3))
        self.assertTrue(d.is_diagonal)
        self.assertEqual(d.det(), sympy.Integer(6))

        # 6. RREF and Nullspace
        sing = sympy.Matrix([[1, 2], [2, 4]])
        self.assertEqual(sing.rank(), 1)
        rref_m, pivots = sing.rref()
        self.assertEqual(pivots, (0,))
        ns = sing.nullspace()
        self.assertEqual(len(ns), 1)
        self.assertEqual((sing @ ns[0])[0, 0], sympy.Integer(0))
        self.assertEqual((sing @ ns[0])[1, 0], sympy.Integer(0))

        # 7. LaTeX rendering
        latex_str = m._repr_latex_()
        self.assertIn("begin{matrix}", latex_str)
        self.assertIn("end{matrix}", latex_str)

    def test_zero_singleton_identity_and_module(self):
        z = sympy.S.Zero
        self.assertIs(z, sympy.S.Zero)
        self.assertEqual(type(z).__name__, "Zero")
        self.assertEqual(type(z).__module__, "sympy.core.numbers")
        self.assertTrue(type(z).is_Zero)
        self.assertEqual(z, sympy.Integer(0))
        self.assertEqual(z, 0)
        self.assertEqual(hash(z), hash(0))
        self.assertEqual(repr(z), "0")
        self.assertEqual(str(z * 5), "0")

    def test_custom_subclass_zero_collapse_matches_oracle(self):
        # Mirrors tools/conformance-lab fixture subclass/ConstitutiveLawZero_zero_collapse:
        # eval folds on literal-zero first arg and keeps the applied form otherwise.
        x, k = sympy.Symbol("x"), sympy.Symbol("k")

        def eval_(cls, *a):
            if len(a) == 2 and a[0] == 0:
                return sympy.S.Zero
            return None

        cls = type(
            "ConstitutiveLawZeroPin",
            (sympy.Function,),
            {"eval": classmethod(eval_), "nargs": (2,)},
        )
        collapsed = cls(0, k)
        self.assertEqual(type(collapsed).__name__, "Zero")
        self.assertEqual(collapsed, 0)
        applied = cls(x, k)
        self.assertEqual(applied.func.__name__, "ConstitutiveLawZeroPin")
        self.assertEqual(applied.args, (x, k))

    def test_deep_chain_refuses_instead_of_crashing(self):
        # Gauntlet bead fra-native-drop-depth-bound-9mk: a deep exact-arithmetic
        # chain used to SIGSEGV the interpreter (recursive derived Clone in the
        # native kernel at depth ~8000). The bridge now refuses beyond
        # FSYM_MAX_EXPR_DEPTH with RecursionError and the process survives.
        a = sympy.Integer(2)
        with self.assertRaises(RecursionError):
            for i in range(1, 6000):
                a = a * sympy.Integer(i) + sympy.Rational(1, i)
        # The interpreter is alive and ordinary arithmetic still works at
        # moderate depth.
        b = sympy.Integer(2)
        for i in range(1, 500):
            b = b * sympy.Integer(i) + sympy.Rational(1, i)
        self.assertGreater(len(str(b)), 1000)

    def test_depth_bound_env_override(self):
        # The bound is configurable; a bound of 1 refuses any compound operand
        # (depth 2 > 1).
        env = dict(os.environ, FSYM_MAX_EXPR_DEPTH="1")
        code = (
            "import sympy\n"
            "x = sympy.Symbol('x')\n"
            "try:\n"
            "    e = (x + 1) + (x + 2)\n"
            "    print('NO-REFUSAL', e)\n"
            "except RecursionError:\n"
            "    print('REFUSED')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, env=env
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("REFUSED", proc.stdout)

    def test_wrap_recovered_symbols_carry_assumptions(self):
        # Gauntlet bead fra-shell-atom-assumptions-bypasses-7o3: symbols and
        # dummies recovered from native results bypass __init__ and previously
        # left the _assumptions slot unset, crashing srepr and is_* access.
        x = sympy.Symbol("x")
        recovered = sympy.simplify(x + x - x)
        self.assertEqual(sympy.srepr(recovered), "Symbol('x')")
        d = sympy.Dummy("d")
        expr = d + x
        for atom in expr.free_symbols:
            if isinstance(atom, sympy.Dummy):
                # Exact Dummy srepr form is finding 10 (seams bead); this pin
                # only requires the recovered Dummy to answer is_* and srepr
                # without AttributeError.
                self.assertIsInstance(atom.is_integer, (bool, type(None)))
                # Must not raise; exact form is finding 10 (seams bead).
                sympy.srepr(atom)

    def test_number_assumption_properties_match_oracle(self):
        # Tri-valued semantics pinned against SymPy 1.14.0 (fresh-eyes
        # finding 7): concrete numbers answer concretely, nan answers None.
        self.assertTrue(sympy.Integer(5).is_positive)
        self.assertFalse(sympy.Integer(0).is_positive)
        self.assertTrue(sympy.Integer(0).is_zero)
        self.assertTrue(sympy.Integer(-7).is_negative)
        self.assertTrue(sympy.Integer(-7).is_nonpositive)
        self.assertTrue(sympy.Rational(1, 2).is_positive)
        self.assertFalse(sympy.Rational(1, 2).is_integer)
        self.assertTrue(sympy.Integer(3).is_integer)
        self.assertTrue(sympy.Rational(1, 2).is_real)
        self.assertTrue(sympy.Float(2.5).is_positive)
        self.assertFalse(sympy.Float(2.5).is_integer)
        self.assertTrue(sympy.Float(0.0).is_integer)
        self.assertTrue(sympy.Float(0.0).is_zero)
        nan = sympy.Float("nan")
        self.assertIsNone(nan.is_positive)
        self.assertIsNone(nan.is_zero)
        self.assertIsNone(nan.is_integer)

    def test_seam_pack_findings_are_closed(self):
        # Gauntlet bead fra-fra-shell-seams-pack-c9g (fresh-eyes findings
        # 3, 4, 5, 6, 8, 9, 10, 11; finding 12 verified fixed externally).
        x = sympy.Symbol("x")
        # 3 + 8: raw args sympified; eval hook sympified + honors evaluate.
        self.assertEqual(str(sympy.Function("f")(3)), "f(3)")
        log = []

        class Traced(sympy.Function):
            @classmethod
            def eval(cls, *a):
                log.append(a)
                return None

        traced = Traced(x, evaluate=False)
        self.assertEqual(log, [])  # evaluate=False skips the hook
        fired = Traced(3)
        self.assertEqual(len(log), 1)
        self.assertEqual(type(log[0][0]).__name__, "Integer")
        self.assertEqual(str(fired), "Traced(3)")
        # 4: reflected matrix ops via NotImplemented.
        m = sympy.Matrix([[1, 2], [3, 4]])
        self.assertEqual(2 * m, m * 2)
        # 5: truthiness honors numeric zero.
        self.assertFalse(bool(sympy.Integer(0)))
        self.assertTrue(bool(sympy.Symbol("x")))
        # 6: numeric conversions.
        self.assertEqual(float(sympy.Rational(3, 2)), 1.5)
        self.assertEqual(int(sympy.Rational(-3, 2)), -1)
        self.assertEqual(float(sympy.Integer(2)), 2.0)
        self.assertEqual(int(sympy.Float(2.5)), 2)
        # 9: zoo is the ComplexInfinity singleton.
        self.assertIs(sympy.zoo, sympy.S.ComplexInfinity)
        self.assertEqual(type(sympy.zoo).__name__, "ComplexInfinity")
        # 10: Dummy survives srepr.
        self.assertIn("Dummy", sympy.srepr(sympy.Dummy("d")))
        # 11: mixed finite/non-finite sorts do not crash.
        ordered = sorted(
            [sympy.Float(2.0), sympy.Float(float("inf"))],
            key=lambda e: e.sort_key(),
        )
        self.assertEqual([str(v) for v in ordered], ["2", "inf"])


if __name__ == "__main__":
    unittest.main()
