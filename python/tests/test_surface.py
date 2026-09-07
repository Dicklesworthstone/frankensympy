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
        self.assertEqual(sympy.diff(x**4, x, 2), 12 * x**2)
        self.assertEqual(sympy.diff(x**4, (x, 2)), 12 * x**2)
        self.assertEqual(sympy.diff(x**4, x, 0), x**4)
        self.assertEqual((x**4).diff(x, 2), 12 * x**2)
        self.assertEqual(sympy.Derivative(x**4, x, 2, evaluate=True), 12 * x**2)
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
        evaluated = sympy.Derivative(x**2, x, evaluate=True)
        self.assertEqual(evaluated, 2 * x)
        self.assertNotEqual(type(evaluated), sympy.Derivative)

        # 0-ary and 1-ary Add and Mul (oracle parity)
        self.assertEqual(sympy.Add(), sympy.Integer(0))
        self.assertEqual(sympy.Add(x), x)
        self.assertEqual(sympy.Add(x, evaluate=False), x)
        self.assertEqual(sympy.Mul(), sympy.Integer(1))
        self.assertEqual(sympy.Mul(x), x)
        self.assertEqual(sympy.Mul(x, evaluate=False), x)
        held_mul = sympy.Mul(x, sympy.Integer(2), evaluate=False)
        self.assertEqual(held_mul.args, (x, sympy.Integer(2)))

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
        # SymPy 1.14.0 parity (bead fra-fra-shell-float-zero-eq-structural-crx):
        # Float equality is Float-vs-Float only; cross-type is structurally
        # False even where the values are numerically equal — while the hash
        # collision with Integer remains (oracle: hash matches, eq does not).
        self.assertNotEqual(value, sympy.Rational(3, 2))
        self.assertNotEqual(sympy.Float(1.0), sympy.Integer(1))
        self.assertNotEqual(sympy.Integer(1), 1.0)
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
        # Pinned oracle: pretty() is ASCII multi-line, linear forms use the
        # ASCII hyphen-minus (bead fra-fra-shell-printer-parity-pack-qxr).
        self.assertEqual(sympy.pretty(powered), " 2\nx ")
        self.assertEqual(powered.pretty(), " 2\nx ")
        self.assertEqual(sympy.pretty(x - 1), "x - 1")

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

        # Negation
        neg_m = -m
        self.assertEqual(neg_m[0, 0], sympy.Integer(-1))
        self.assertEqual(neg_m[1, 1], sympy.Integer(-4))
        self.assertEqual(m + neg_m, sympy.zeros(2, 2))

        # Scalar multiplication and right-multiplication
        scaled = m * 2
        r_scaled = 2 * m
        self.assertEqual(scaled[0, 0], sympy.Integer(2))
        self.assertEqual(scaled[1, 1], sympy.Integer(8))
        self.assertEqual(scaled, r_scaled)

        # Scalar division
        div_m = m / 2
        self.assertEqual(div_m[0, 0], sympy.Rational(1, 2))
        self.assertEqual(div_m[0, 1], sympy.Integer(1))

        # Equality
        self.assertEqual(m, sympy.Matrix([[1, 2], [3, 4]]))
        self.assertNotEqual(m, sympy.Matrix([[1, 2], [3, 5]]))

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
        self.assertEqual([str(v) for v in ordered], ["2.00000000000000", "oo"])

    def test_extended_solvers_transforms_functions_and_submodules(self):
        x, t, s, w = sympy.symbols("x t s w")

        # 1. Quadratic solving and auto-variable detection
        solutions = sympy.solve(x**2 - 4, x)
        self.assertEqual(set(solutions), {sympy.Integer(2), sympy.Integer(-2)})
        auto_sol = sympy.solve(2 * x - 6)
        self.assertEqual(auto_sol, [sympy.Integer(3)])

        # 2. Series expansion (module-level and method-level)
        sin_series = sympy.series(sympy.sin(x), x, 0, 4)
        self.assertEqual(sin_series, sympy.sin(x).series(x, 0, 4))
        self.assertEqual(str(sin_series), "x - (x**3/6)")

        # 3. Integral transforms
        laplace_res = sympy.laplace_transform(sympy.exp(t), t, s)
        self.assertEqual(str(laplace_res), "(s - 1)**(-1)")
        fourier_res = sympy.fourier_transform(sympy.Integer(5), t, w)
        self.assertEqual(str(fourier_res), "10*pi*dirac(w)")

        # 4. Number theory functions
        self.assertEqual(sympy.mobius(1), 1)
        self.assertEqual(sympy.mobius(6), 1)
        self.assertEqual(sympy.mobius(4), 0)
        self.assertEqual(sympy.divisor_count(12), 6)
        self.assertEqual(sympy.divisor_sigma(12, 1), 28)
        self.assertEqual(sympy.jacobi_symbol(2, 5), -1)

        # 5. Functions
        self.assertEqual(sympy.tan(0), sympy.Integer(0))
        self.assertEqual(sympy.asin(0), sympy.Integer(0))
        self.assertEqual(sympy.atan(0), sympy.Integer(0))
        self.assertEqual(sympy.sinh(0), sympy.Integer(0))
        self.assertEqual(sympy.cosh(0), sympy.Integer(1))
        self.assertEqual(sympy.tanh(0), sympy.Integer(0))
        self.assertEqual(sympy.floor(sympy.Rational(5, 2)), sympy.Integer(2))
        self.assertEqual(sympy.ceiling(sympy.Rational(5, 2)), sympy.Integer(3))
        self.assertEqual(sympy.factorial(5), sympy.Integer(120))
        self.assertEqual(sympy.gamma(5), sympy.Integer(24))
        self.assertEqual(sympy.fibonacci(10), sympy.Integer(55))

        # 6. Submodule imports
        from sympy.solvers import dsolve as sol_dsolve, solve as sol_solve
        from sympy.functions import (
            asin as f_asin,
            atan as f_atan,
            ceiling as f_ceiling,
            cos as f_cos,
            cosh as f_cosh,
            exp as f_exp,
            factorial as f_factorial,
            fibonacci as f_fibonacci,
            floor as f_floor,
            gamma as f_gamma,
            log as f_log,
            sin as f_sin,
            sinh as f_sinh,
            tan as f_tan,
            tanh as f_tanh,
        )
        from sympy.ntheory import (
            divisor_count as n_divisor_count,
            divisor_sigma as n_divisor_sigma,
            factorint as n_factorint,
            isprime as n_isprime,
            jacobi_symbol as n_jacobi_symbol,
            mobius as n_mobius,
            totient as n_totient,
        )
        from sympy.series import limit as ser_limit, series as ser_series
        from sympy.integrals import (
            fourier_transform as int_fourier,
            integrate as int_integrate,
            laplace_transform as int_laplace,
        )

        self.assertIs(sol_solve, sympy.solve)
        self.assertIs(sol_dsolve, sympy.dsolve)
        self.assertIs(f_tan, sympy.tan)
        self.assertIs(n_mobius, sympy.mobius)
        self.assertIs(ser_series, sympy.series)
        self.assertIs(int_integrate, sympy.integrate)

        # Module-level package access
        self.assertIs(sympy.solvers.solve, sympy.solve)
        self.assertIs(sympy.functions.tan, sympy.tan)
        self.assertIs(sympy.ntheory.mobius, sympy.mobius)
        self.assertIs(sympy.integrals.integrate, sympy.integrate)

    def test_matrix_advanced_linear_algebra_and_as_expr(self):
        from sympy.matrices import (
            Matrix,
            hadamard_product,
            kronecker_product,
            matrix_multiply_elementwise,
        )

        x = sympy.Symbol("x")
        self.assertIs(x.as_expr(), x)
        two = sympy.Integer(2)
        self.assertIs(two.as_expr(), two)

        A = Matrix([[1, 2], [3, 4]])
        b = Matrix([5, 11])

        # 1. Linear solve
        sol = A.solve(b)
        self.assertEqual(sol, Matrix([[1], [2]]))
        self.assertEqual(A.LUsolve(b), sol)
        self.assertEqual(A * sol, b)

        # 2. Least-squares solve
        ls_sol = A.solve_least_squares(b)
        self.assertEqual(ls_sol, sol)

        # 3. Characteristic polynomial
        cp_default = A.charpoly()
        lam = sympy.Symbol("lambda")
        self.assertEqual(cp_default, lam**2 - 5 * lam - 2)
        cp_x = A.charpoly(x)
        self.assertEqual(cp_x, x**2 - 5 * x - 2)
        self.assertEqual(cp_x.as_expr(), x**2 - 5 * x - 2)

        # 4. LU decomposition
        L, U, P = A.LUdecomposition()
        self.assertEqual(L * U, P * A)
        p_mat, l_mat, u_mat = A.lu()
        self.assertEqual(l_mat, L)
        self.assertEqual(u_mat, U)
        self.assertEqual(p_mat, P)

        # 5. QR decomposition
        Q, R = A.QRdecomposition()
        self.assertEqual(Q * R, A)
        q_mat, r_mat = A.qr()
        self.assertEqual(q_mat, Q)
        self.assertEqual(r_mat, R)

        # 6. Hadamard product
        B = Matrix([[5, 6], [7, 8]])
        had = A.hadamard(B)
        expected_had = Matrix([[5, 12], [21, 32]])
        self.assertEqual(had, expected_had)
        self.assertEqual(hadamard_product(A, B), expected_had)
        self.assertEqual(matrix_multiply_elementwise(A, B), expected_had)

        # 7. Kronecker product
        kron = A.kron(B)
        expected_kron = Matrix([
            [5, 6, 10, 12],
            [7, 8, 14, 16],
            [15, 18, 20, 24],
            [21, 24, 28, 32],
        ])
        self.assertEqual(kron, expected_kron)
        self.assertEqual(A.kronecker_product(B), expected_kron)
        self.assertEqual(kronecker_product(A, B), expected_kron)

    def test_sqrt_and_sympify_behavior(self):
        x = sympy.Symbol("x")
        self.assertEqual(sympy.sqrt(0), sympy.Integer(0))
        self.assertEqual(sympy.sqrt(1), sympy.Integer(1))
        self.assertEqual(sympy.sqrt(4), sympy.Integer(2))
        self.assertEqual(sympy.sqrt(25), sympy.Integer(5))
        self.assertEqual(sympy.sqrt(8), 2 * sympy.sqrt(2))
        self.assertEqual(sympy.sqrt(-1), sympy.I)
        self.assertEqual(sympy.sqrt(-4), 2 * sympy.I)
        self.assertEqual(sympy.sqrt(sympy.Rational(4, 9)), sympy.Rational(2, 3))
        self.assertEqual(sympy.sqrt(sympy.Rational(1, 4)), sympy.Rational(1, 2))
        self.assertEqual(sympy.sqrt(x), sympy.Pow(x, sympy.S.Half))

        # Held evaluate=False
        held_sqrt = sympy.sqrt(4, evaluate=False)
        self.assertIsInstance(held_sqrt, sympy.Pow)
        self.assertEqual(held_sqrt.args, (sympy.Integer(4), sympy.S.Half))

        # Pow square root fold
        self.assertEqual(sympy.Pow(sympy.Integer(25), sympy.S.Half), sympy.Integer(5))
        self.assertEqual(sympy.Pow(sympy.Rational(4, 9), sympy.S.Half), sympy.Rational(2, 3))

        # sympify
        self.assertEqual(sympy.sympify(42), sympy.Integer(42))
        self.assertEqual(sympy.sympify("x + 1"), x + 1)
        self.assertEqual(sympy.sympify(True), sympy.Integer(1))
        self.assertEqual(sympy.sympify(False), sympy.Integer(0))
        self.assertIsInstance(sympy.sympify(3.14), sympy.Float)

        # Module / function facades
        from sympy.functions import sqrt as fn_sqrt
        self.assertIs(fn_sqrt, sympy.sqrt)
        from sympy.core.sympify import sympify as mod_sympify, SympifyError
        self.assertIs(mod_sympify, sympy.sympify)
        self.assertIs(SympifyError, sympy.SympifyError)

    def test_logic_and_boolean_algebra(self):
        from sympy import (
            And,
            Equivalent,
            Implies,
            Not,
            Or,
            Symbol,
            Xor,
            false,
            satisfiable,
            simplify_logic,
            to_cnf,
            to_dnf,
            true,
        )
        from sympy.logic import And as LAnd, satisfiable as lsat
        from sympy.logic.boolalg import to_cnf as lto_cnf
        from sympy.logic.inference import satisfiable as isat

        self.assertIs(LAnd, And)
        self.assertIs(lsat, satisfiable)
        self.assertIs(isat, satisfiable)
        self.assertIs(lto_cnf, to_cnf)

        x = Symbol("x")
        y = Symbol("y")
        z = Symbol("z")

        # Operators and classes
        self.assertIsInstance(x & y, And)
        self.assertIsInstance(x | y, Or)
        self.assertIsInstance(~x, Not)
        self.assertIsInstance(x >> y, Implies)
        self.assertIsInstance(x ^ y, Xor)

        # Singletons and boolean evaluation
        self.assertIs(sympy.S.true, true)
        self.assertIs(sympy.S.false, false)
        self.assertTrue(bool(true))
        self.assertFalse(bool(false))

        # Basic simplifications
        self.assertEqual(And(x, true), x)
        self.assertEqual(And(x, false), false)
        self.assertEqual(Or(x, true), true)
        self.assertEqual(Or(x, false), x)
        self.assertEqual(Not(true), false)
        self.assertEqual(Not(false), true)
        self.assertEqual(Not(Not(x)), x)

        # Tautology / Contradiction via simplify_logic
        self.assertEqual(simplify_logic(x | ~x), true)
        self.assertEqual(simplify_logic(true), true)
        self.assertEqual(simplify_logic(false), false)

        # Satisfiability via DPLL
        sat_model = satisfiable(x & y)
        self.assertEqual(sat_model, {x: True, y: True})
        self.assertFalse(satisfiable(x & ~x))
        self.assertEqual(satisfiable(true), {})
        self.assertFalse(satisfiable(false))

        # CNF and DNF conversions
        cnf_expr = to_cnf(x | (y & z))
        self.assertIsInstance(cnf_expr, (And, Symbol))
        dnf_expr = to_dnf((x | y) & z)
        self.assertIsInstance(dnf_expr, (Or, Symbol))

    def test_geometry_surface(self):
        from sympy import (
            Circle,
            Line,
            Line2D,
            Point,
            Point2D,
            Point3D,
            Polygon,
            Rational,
            Ray,
            Ray2D,
            Segment,
            Segment2D,
            Triangle,
            pi,
        )
        import sympy.geometry as sgeom

        self.assertIs(sgeom.Point, Point)
        self.assertIs(sgeom.Point2D, Point2D)
        self.assertIs(sgeom.Point3D, Point3D)
        self.assertIs(sgeom.Line, Line)
        self.assertIs(sgeom.Circle, Circle)
        self.assertIs(sgeom.Triangle, Triangle)
        self.assertIs(sgeom.Polygon, Polygon)

        p1 = Point(0, 0)
        p2 = Point(3, 4)
        self.assertIsInstance(p1, Point2D)
        self.assertEqual(p1.x, 0)
        self.assertEqual(p1.y, 0)
        self.assertEqual(p1.distance(p2), 5)

        p3d1 = Point(1, 2, 3)
        p3d2 = Point(4, 6, 3)
        self.assertIsInstance(p3d1, Point3D)
        self.assertEqual(p3d1.z, 3)
        self.assertEqual(p3d1.distance(p3d2), 5)

        seg = Segment(Point(0, 0), Point(4, 4))
        self.assertIsInstance(seg, Segment2D)
        self.assertEqual(seg.midpoint, Point2D(2, 2))

        l1 = Line(Point(0, 0), Point(2, 2))
        l2 = Line(Point(0, 2), Point(2, 0))
        self.assertIsInstance(l1, Line2D)
        self.assertEqual(l1.intersection(l2), [Point2D(1, 1)])

        circ = Circle(Point(0, 0), 5)
        self.assertEqual(circ.area, 25 * pi)
        self.assertEqual(circ.circumference, 10 * pi)

        tri = Triangle(Point(0, 0), Point(3, 0), Point(0, 4))
        self.assertEqual(tri.area, 6)
        self.assertEqual(tri.centroid, Point2D(1, Rational(4, 3)))
        self.assertTrue(tri.is_right())
        self.assertFalse(tri.is_equilateral())

        poly = Polygon(Point(0, 0), Point(4, 0), Point(4, 3), Point(0, 3))
        self.assertEqual(poly.area, 12)
        self.assertEqual(poly.centroid, Point2D(2, Rational(3, 2)))
        self.assertTrue(poly.is_convex())

    def test_sets_surface(self):
        from sympy import (
            Complement,
            EmptySet,
            FiniteSet,
            Integer,
            Intersection,
            Interval,
            S,
            Set,
            Union,
            UniversalSet,
        )
        import sympy.sets as ssets

        self.assertIs(ssets.Set, Set)
        self.assertIs(ssets.Interval, Interval)
        self.assertIs(ssets.FiniteSet, FiniteSet)
        self.assertIs(ssets.Union, Union)
        self.assertIs(ssets.Intersection, Intersection)
        self.assertIs(ssets.Complement, Complement)
        self.assertIs(ssets.EmptySet, EmptySet)
        self.assertIs(ssets.UniversalSet, UniversalSet)

        # Singletons
        self.assertIs(S.EmptySet, EmptySet())
        self.assertIs(S.UniversalSet, UniversalSet())
        self.assertTrue(S.EmptySet.is_empty)
        self.assertFalse(S.UniversalSet.is_empty)
        self.assertEqual(S.EmptySet.measure, Integer(0))
        self.assertEqual(len(S.EmptySet), 0)
        self.assertFalse(bool(S.EmptySet))
        self.assertTrue(bool(S.UniversalSet))

        # Interval
        iv = Interval(0, 5)
        self.assertEqual(iv.start, 0)
        self.assertEqual(iv.end, 5)
        self.assertEqual(iv.left, 0)
        self.assertEqual(iv.right, 5)
        self.assertFalse(iv.left_open)
        self.assertFalse(iv.right_open)
        self.assertEqual(iv.measure, 5)
        self.assertFalse(iv.is_open)
        self.assertTrue(iv.is_closed)
        self.assertTrue(iv.is_compact)
        self.assertTrue(2 in iv)
        self.assertTrue(0 in iv)
        self.assertTrue(5 in iv)
        self.assertFalse(6 in iv)

        # Open interval
        open_iv = Interval(0, 5, left_open=True, right_open=True)
        self.assertTrue(open_iv.left_open)
        self.assertTrue(open_iv.right_open)
        self.assertTrue(open_iv.is_open)
        self.assertFalse(open_iv.is_closed)
        self.assertFalse(0 in open_iv)
        self.assertFalse(5 in open_iv)
        self.assertTrue(3 in open_iv)

        # Degenerate & empty interval
        self.assertIs(Interval(5, 5, left_open=True), S.EmptySet)
        with self.assertRaises(ValueError):
            Interval(5, 0)

        # Topology
        self.assertEqual(iv.interior, open_iv)
        self.assertEqual(iv.closure, iv)
        self.assertEqual(iv.boundary, FiniteSet(0, 5))

        # FiniteSet
        fs = FiniteSet(1, 2, 3)
        self.assertEqual(len(fs), 3)
        self.assertTrue(2 in fs)
        self.assertFalse(4 in fs)
        self.assertTrue(fs.is_subset(iv))
        self.assertFalse(fs.is_subset(Interval(10, 20)))
        self.assertTrue(fs.is_disjoint(Interval(10, 20)))

        # Set algebra operators
        u = iv | FiniteSet(7)
        self.assertIsInstance(u, Union)
        self.assertTrue(7 in u)
        self.assertTrue(2 in u)

        inter = iv & FiniteSet(2, 3, 7)
        self.assertIsInstance(inter, Intersection)
        self.assertTrue(2 in inter)
        self.assertTrue(3 in inter)
        self.assertFalse(7 in inter)

        diff = FiniteSet(1, 2, 3) - FiniteSet(2)
        self.assertTrue(1 in diff)
        self.assertFalse(2 in diff)
        self.assertTrue(3 in diff)

        sym_diff = FiniteSet(1, 2) ^ FiniteSet(2, 3)
        self.assertTrue(1 in sym_diff)
        self.assertFalse(2 in sym_diff)
        self.assertTrue(3 in sym_diff)

    def test_solveset_and_checksol(self):
        from sympy.solvers import solveset as s_solveset, checksol as s_checksol
        self.assertIs(s_solveset, sympy.solveset)
        self.assertIs(s_checksol, sympy.checksol)

        x = sympy.Symbol("x")
        # Quadratic solveset
        sol_quad = sympy.solveset(x**2 - 9, x)
        self.assertIsInstance(sol_quad, sympy.FiniteSet)
        self.assertEqual(sol_quad, sympy.FiniteSet(-3, 3))

        # Linear solveset with auto-variable
        sol_lin = sympy.solveset(2 * x - 6)
        self.assertEqual(sol_lin, sympy.FiniteSet(3))

        # Equation solveset
        sol_eq = sympy.solveset(sympy.Eq(x**2, 16), x)
        self.assertEqual(sol_eq, sympy.FiniteSet(-4, 4))

        # Inconsistent equation -> EmptySet
        sol_empty = sympy.solveset(sympy.Integer(1), x)
        self.assertIs(sol_empty, sympy.S.EmptySet)

        # checksol verification
        self.assertTrue(sympy.checksol(x**2 - 9, x, 3))
        self.assertTrue(sympy.checksol(x**2 - 9, x, -3))
        self.assertFalse(sympy.checksol(x**2 - 9, x, 4))
        self.assertTrue(sympy.checksol(x - 5, {x: 5}))
        self.assertFalse(sympy.checksol(x - 5, {x: 4}))
        self.assertTrue(sympy.checksol(sympy.Eq(2 * x, 10), x, 5))

    def test_geometry_3d_and_sequence_access(self):
        from sympy import (
            Point,
            Point2D,
            Point3D,
            Line,
            Line2D,
            Line3D,
            Segment,
            Segment2D,
            Segment3D,
            Plane,
            Sphere,
        )

        # Sequence access and hashing on Point2D
        p2 = Point2D(3, 7)
        self.assertEqual(len(p2), 2)
        self.assertEqual(p2[0], 3)
        self.assertEqual(p2[1], 7)
        self.assertEqual(p2[-1], 7)
        self.assertEqual(p2[-2], 3)
        with self.assertRaises(IndexError):
            _ = p2[2]
        self.assertEqual(hash(p2), hash(Point2D(3, 7)))
        self.assertEqual({p2, Point2D(3, 7)}, {Point2D(3, 7)})

        # Sequence access and hashing on Point3D
        p3 = Point3D(1, 4, 9)
        self.assertEqual(len(p3), 3)
        self.assertEqual(p3[0], 1)
        self.assertEqual(p3[1], 4)
        self.assertEqual(p3[2], 9)
        self.assertEqual(p3[-1], 9)
        self.assertEqual(p3[-2], 4)
        self.assertEqual(p3[-3], 1)
        with self.assertRaises(IndexError):
            _ = p3[3]
        self.assertEqual(hash(p3), hash(Point3D(1, 4, 9)))

        # Segment3D
        origin = Point3D(0, 0, 0)
        seg = Segment(origin, Point3D(2, 4, 6))
        self.assertIsInstance(seg, Segment3D)
        self.assertEqual(seg.midpoint, Point3D(1, 2, 3))
        self.assertEqual(seg.p1, origin)
        self.assertEqual(seg.p2, Point3D(2, 4, 6))

        # Line3D
        l3 = Line(origin, Point3D(0, 0, 5))
        self.assertIsInstance(l3, Line3D)
        self.assertEqual(l3.direction, Point3D(0, 0, 5))

        # Plane from point + normal
        pl1 = Plane(origin, Point3D(0, 0, 1))
        self.assertEqual(pl1.point, origin)
        self.assertEqual(pl1.normal_vector, Point3D(0, 0, 1))
        self.assertEqual(pl1.eval_at_point(Point3D(2, 3, 0)), 0)
        self.assertEqual(pl1.eval_at_point(Point3D(2, 3, 5)), 5)

        # Plane from 3 points
        pl2 = Plane(Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 0))
        self.assertEqual(pl2.normal_vector, Point3D(0, 0, 1))
        self.assertTrue(pl1.is_parallel(pl2))

        # Plane perpendicularity
        pl_perp = Plane(origin, Point3D(1, 0, 0))
        self.assertTrue(pl1.is_perpendicular(pl_perp))

        # Sphere
        sp = Sphere(origin, 3)
        self.assertEqual(sp.center, origin)
        self.assertEqual(sp.radius, 3)
        self.assertEqual(sp.surface_area, 36 * sympy.pi)
        self.assertEqual(sp.area, 36 * sympy.pi)
        self.assertEqual(sp.volume, 36 * sympy.pi)

    def test_numeric_exact_division(self):
        from sympy import Integer, Rational, Float, zoo

        # Integer / Integer
        self.assertEqual(Integer(6) / Integer(2), Integer(3))
        self.assertIsInstance(Integer(6) / Integer(2), Integer)
        self.assertEqual(Integer(1) / Integer(2), Rational(1, 2))

        # Integer / Rational
        self.assertEqual(Integer(6) / Rational(3, 2), Integer(4))
        self.assertIsInstance(Integer(6) / Rational(3, 2), Integer)

        # Rational / Rational
        self.assertEqual(Rational(1, 3) / Rational(2, 5), Rational(5, 6))

        # Division by zero
        self.assertEqual(Integer(1) / Integer(0), zoo)

        # Float preserves float behavior
        f_res = Float(2.5) / Integer(2)
        self.assertIsInstance(f_res, Float)

    def test_polys_module(self):
        from sympy.polys import (
            LC,
            Poly,
            degree,
            discriminant,
            gcd,
            groebner,
            lcm,
            monic,
            resultant,
            sqf_list,
            sqf_part,
        )
        self.assertIs(Poly, sympy.Poly)
        self.assertIs(degree, sympy.degree)
        self.assertIs(gcd, sympy.gcd)
        self.assertIs(lcm, sympy.lcm)
        self.assertIs(resultant, sympy.resultant)
        self.assertIs(discriminant, sympy.discriminant)
        self.assertIs(sqf_list, sympy.sqf_list)
        self.assertIs(sqf_part, sympy.sqf_part)
        self.assertIs(groebner, sympy.groebner)

        x, y = sympy.symbols("x y")
        p = Poly(3 * x**2 - 4, x)
        self.assertEqual(p.degree(), 2)
        self.assertEqual(degree(3 * x**2 - 4, x), 2)
        self.assertEqual(p.all_coeffs(), [sympy.Integer(3), sympy.Integer(0), sympy.Integer(-4)])
        self.assertEqual(p.coeffs(), [sympy.Integer(3), sympy.Integer(-4)])
        self.assertEqual(p.leading_coeff(), sympy.Integer(3))
        self.assertEqual(LC(p), sympy.Integer(3))
        self.assertFalse(p.is_monic)

        p_monic = Poly(2 * x**2 - 8, x).monic()
        self.assertTrue(p_monic.is_monic)
        self.assertEqual(p_monic.all_coeffs(), [sympy.Integer(1), sympy.Integer(0), sympy.Integer(-4)])

        # Division with remainder
        p1 = Poly(x**2 - 1, x)
        p2 = Poly(x - 1, x)
        q, r = p1.div(p2)
        self.assertEqual(q.as_expr(), x + 1)
        self.assertEqual(r.as_expr(), sympy.Integer(0))
        self.assertEqual(p1.rem(p2).as_expr(), sympy.Integer(0))

        # Resultant and discriminant
        self.assertEqual(resultant(x - 2, x - 3, x), sympy.Integer(-1))
        self.assertEqual(discriminant(x**2 - 4, x), sympy.Integer(16))

        # GCD and LCM
        self.assertEqual(gcd(x**2 - 1, x - 1), x - 1)
        self.assertEqual(lcm(x - 1, x + 1), x**2 - 1)
        self.assertEqual(gcd(12, 18), 6)
        self.assertEqual(lcm(12, 18), 36)

        # Square-free factorization
        scale, factors = sqf_list((x - 1)**2 * (x + 2), x)
        self.assertEqual(scale, sympy.Integer(1))
        factor_map = {f: mult for f, mult in factors}
        self.assertEqual(factor_map[x - 1], 2)
        self.assertEqual(factor_map[x + 2], 1)

        # Groebner basis
        gb = groebner([x * y - 2 * y, 2 * y**2 - x**2], x, y)
        self.assertTrue(len(gb) >= 2)

    def test_extended_ode_solvers(self):
        from sympy.solvers.ode import (
            dsolve_cauchy_euler,
            dsolve_const_coeff_second_order,
            dsolve_const_coeff_second_order_nonhomogeneous,
            dsolve_linear_first_order,
            dsolve_separable_linear,
        )
        self.assertIs(dsolve_cauchy_euler, sympy.dsolve_cauchy_euler)
        self.assertIs(dsolve_const_coeff_second_order, sympy.dsolve_const_coeff_second_order)
        self.assertIs(
            dsolve_const_coeff_second_order_nonhomogeneous,
            sympy.dsolve_const_coeff_second_order_nonhomogeneous,
        )
        self.assertIs(dsolve_linear_first_order, sympy.dsolve_linear_first_order)
        self.assertIs(dsolve_separable_linear, sympy.dsolve_separable_linear)

        x = sympy.Symbol("x")
        # 1st-order linear
        sol_lin1 = dsolve_linear_first_order(0, 2 * x, x)
        self.assertTrue("C1" in str(sol_lin1))

        # 2nd-order constant coefficient
        sol_hom2 = dsolve_const_coeff_second_order(1, -3, 2, x)
        self.assertTrue("C1" in str(sol_hom2) and "C2" in str(sol_hom2))

        # 2nd-order nonhomogeneous
        sol_nonhom2 = dsolve_const_coeff_second_order_nonhomogeneous(1, -3, 2, 4, x)
        self.assertTrue("C1" in str(sol_nonhom2) and "C2" in str(sol_nonhom2))

        # Cauchy-Euler
        sol_ce = dsolve_cauchy_euler(1, -1, 1, x)
        self.assertTrue("C1" in str(sol_ce) and "C2" in str(sol_ce))

        # Separable linear
        sol_sep = dsolve_separable_linear(2 * x, x)
        self.assertTrue("C1" in str(sol_sep))

    def test_poly_system_solvers(self):
        from sympy.solvers import nonlinsolve, solve_poly_system
        self.assertIs(solve_poly_system, sympy.solve_poly_system)
        self.assertIs(nonlinsolve, sympy.nonlinsolve)

        x, y = sympy.symbols("x y")
        # solve_poly_system
        sols = solve_poly_system([x + y - 5, x - y - 1], x, y)
        self.assertEqual(sols, [(sympy.Integer(3), sympy.Integer(2))])

        # nonlinsolve
        n_sols = nonlinsolve([x + y - 5, x - y - 1], x, y)
        self.assertIsInstance(n_sols, sympy.FiniteSet)
        sol_tuple = list(n_sols)[0]
        self.assertEqual(sol_tuple[0], sympy.Integer(3))
        self.assertEqual(sol_tuple[1], sympy.Integer(2))

        # solve with system of equations
        sys_sol = sympy.solve([x + y - 5, x - y - 1], [x, y])
        self.assertEqual(sys_sol, {x: sympy.Integer(3), y: sympy.Integer(2)})

        # solve with auto-detected symbols
        sys_auto = sympy.solve([x + y - 5, x - y - 1])
        self.assertEqual(sys_auto, {x: sympy.Integer(3), y: sympy.Integer(2)})

    def test_series_order_and_tuple(self):
        from sympy.series import O as s_O, Order as s_Order
        self.assertIs(s_O, sympy.O)
        self.assertIs(s_Order, sympy.Order)

        x = sympy.Symbol("x")
        o1 = sympy.Order(x**3)
        self.assertEqual(str(o1), "Order(x**3)")
        o2 = sympy.O(x**3)
        self.assertEqual(str(o2), "Order(x**3)")

        t = sympy.Tuple(1, 2, 3)
        self.assertEqual(len(t), 3)
        self.assertEqual(t[0], sympy.Integer(1))
        self.assertEqual(t[1], sympy.Integer(2))
        self.assertEqual(t[2], sympy.Integer(3))
        self.assertEqual(list(t), [sympy.Integer(1), sympy.Integer(2), sympy.Integer(3)])

    def test_tensor_and_metric(self):
        from sympy.tensor import (
            Metric,
            Tensor,
            TensorIndex,
            tensor_indices,
            tensorcontraction,
            tensorproduct,
        )
        self.assertIs(Metric, sympy.Metric)
        self.assertIs(Tensor, sympy.Tensor)
        self.assertIs(TensorIndex, sympy.TensorIndex)
        self.assertIs(tensor_indices, sympy.tensor_indices)
        self.assertIs(tensorcontraction, sympy.tensorcontraction)
        self.assertIs(tensorproduct, sympy.tensorproduct)

        # Tensor indices
        mu, nu = tensor_indices("mu nu")
        self.assertTrue(mu.is_up)
        self.assertEqual(mu.name, "mu")
        mu_low = mu.flip()
        self.assertFalse(mu_low.is_up)

        # 4D Minkowski metric
        eta = Metric.minkowski_4d("eta")
        self.assertEqual(eta.dimension, 4)
        self.assertEqual(len(eta.matrix), 16)
        self.assertEqual(eta.matrix[0], sympy.Integer(-1))
        self.assertEqual(eta.matrix[5], sympy.Integer(1))

        # Vector v^\mu = (3, 0, 0, 4)
        v = Tensor("v", 4, [mu], [3, 0, 0, 4])
        self.assertEqual(v.rank, 1)
        self.assertEqual(v.dimension, 4)
        self.assertEqual(v.components, (sympy.Integer(3), sympy.Integer(0), sympy.Integer(0), sympy.Integer(4)))

        # Lower vector v_\mu = eta_{\mu\nu} v^\nu = (-3, 0, 0, 4)
        v_low = eta.lower_vector(v)
        self.assertEqual(v_low.rank, 1)
        self.assertFalse(v_low.indices[0].is_up)
        self.assertEqual(v_low.components, (sympy.Integer(-3), sympy.Integer(0), sympy.Integer(0), sympy.Integer(4)))

        # Norm squared: -3^2 + 0 + 0 + 4^2 = 7
        ns = eta.norm_squared(v)
        self.assertEqual(ns, sympy.Integer(7))

        # Inner product with itself
        ip = eta.inner_product(v, v)
        self.assertEqual(ip, sympy.Integer(7))

        # Raise covector back to vector
        v_recov = eta.raise_covector(v_low)
        self.assertTrue(v_recov.indices[0].is_up)
        self.assertEqual(v_recov.components, (sympy.Integer(3), sympy.Integer(0), sympy.Integer(0), sympy.Integer(4)))

        # Outer product and contraction
        prod = tensorproduct(v, v_low)
        self.assertEqual(prod.rank, 2)
        contracted = tensorcontraction(prod, ("mu", "mu"))
        self.assertEqual(contracted.rank, 0)
        self.assertEqual(contracted.components, (sympy.Integer(7),))

    def test_assumptions_and_deductive_predicates(self):
        from sympy.assumptions import (
            AppliedPredicate,
            AssumptionsContext,
            Predicate,
            Q,
            ask,
        )
        self.assertIs(Q, sympy.Q)
        self.assertIs(ask, sympy.ask)

        x = sympy.Symbol("x")

        # Inherent facts on concrete numbers
        self.assertIs(ask(Q.positive(sympy.Integer(5))), True)
        self.assertIs(ask(Q.negative(sympy.Integer(5))), False)
        self.assertIs(ask(Q.even(sympy.Integer(4))), True)
        self.assertIs(ask(Q.odd(sympy.Integer(4))), False)
        self.assertIs(ask(Q.zero(sympy.Integer(0))), True)

        # Assumptions queries with facts
        self.assertIs(ask(Q.positive(x), Q.positive(x)), True)
        self.assertIs(ask(Q.real(x), Q.positive(x)), True)
        self.assertIs(ask(Q.complex(x), Q.positive(x)), True)
        self.assertIs(ask(Q.nonnegative(x), Q.positive(x)), True)
        self.assertIs(ask(Q.negative(x), Q.positive(x)), False)
        self.assertIs(ask(Q.zero(x), Q.positive(x)), False)
        self.assertIs(ask(Q.integer(x), Q.positive(x)), None)

        # Inherent deductive properties on Symbol
        x_pos = sympy.Symbol("x", positive=True)
        self.assertIs(x_pos.is_positive, True)
        self.assertIs(x_pos.is_real, True)
        self.assertIs(x_pos.is_complex, True)
        self.assertIs(x_pos.is_nonnegative, True)
        self.assertIs(x_pos.is_nonzero, True)
        self.assertIs(x_pos.is_negative, False)
        self.assertIs(x_pos.is_zero, False)
        self.assertIs(x_pos.is_integer, None)

        n_int = sympy.Symbol("n", integer=True)
        self.assertIs(n_int.is_integer, True)
        self.assertIs(n_int.is_rational, True)
        self.assertIs(n_int.is_real, True)
        self.assertIs(n_int.is_complex, True)
        self.assertIs(n_int.is_positive, None)

        # AssumptionsContext
        ctx = AssumptionsContext()
        ctx.assume(x, Q.positive)
        self.assertIs(ctx.is_true(x, Q.real), True)
        self.assertIs(ctx.is_true(x, Q.negative), False)
        self.assertIs(ctx.is_true(x, Q.integer), None)
        self.assertEqual(ctx.query(x, Q.real), "True")
        self.assertEqual(ctx.query(x, Q.negative), "False")
        self.assertEqual(ctx.query(x, Q.integer), "Unknown")


if __name__ == "__main__":
    unittest.main()

