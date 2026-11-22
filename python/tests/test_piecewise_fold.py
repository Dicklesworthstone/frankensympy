"""Ordered Piecewise folding, pinned to isolated SymPy 1.14.0 observations.

The public tests run unchanged against the oracle and the actual native shell.
Comparisons are structural, including class and argument order; no algebraic
simplification or numeric sampling substitutes for the profile comparator.
The final class checks the shell's explicit resource/effect boundary.
"""

import contextlib
import unittest
from unittest.mock import patch

import sympy as s


def _request(**limits):
    supervisor = getattr(s.core, "supervised_callback_session", None)
    return supervisor(**limits) if supervisor is not None else contextlib.nullcontext()


def _supervised():
    active = getattr(s.core, "_active_callback_session", None)
    if active is not None:
        assert active() is not None, "Piecewise reconstruction escaped supervision"


class PiecewiseFoldTests(unittest.TestCase):
    def setUp(self):
        self.x, self.y, self.z = s.symbols("x y z", real=True)

    def same(self, actual, expected):
        self.assertIs(type(actual), type(expected))
        self.assertEqual(actual, expected)
        if isinstance(actual, s.Basic):
            self.assertEqual(len(actual.args), len(expected.args))
            for left, right in zip(actual.args, expected.args):
                self.same(left, right)

    def test_non_piecewise_inputs_are_exact_identity_operations(self):
        x = self.x
        held = s.Mul(2, s.Add(x, -x, evaluate=False), evaluate=False)
        for value in (x, held, 3, None, [x], (x,)):
            for mode in (True, False, None):
                self.assertIs(s.piecewise_fold(value, evaluate=mode), value)

    def test_arithmetic_product_keeps_factorization(self):
        x = self.x
        p = s.Piecewise((x, x < 0), (x*x, True))
        expected = s.Piecewise((x*(x + 1), x < 0), (x*x*(x + 1), True))
        self.same(s.piecewise_fold((x + 1)*p), expected)

    def test_independent_piecewise_operands_form_ordered_cartesian_branches(self):
        x, y = self.x, self.y
        p = s.Piecewise((x, x < 0), (1, True))
        q = s.Piecewise((y, y < 0), (2, True))
        for operation in (s.Add, s.Mul):
            expected = s.Piecewise(
                (operation(x, y), s.And(x < 0, y < 0)),
                (operation(x, 2), x < 0),
                (operation(1, y), y < 0),
                (operation(1, 2), True),
            )
            for mode in (True, False, None):
                self.same(s.piecewise_fold(operation(p, q), evaluate=mode), expected)

    def test_identical_guards_align_before_the_cartesian_product(self):
        x, y = self.x, self.y
        p = s.Piecewise((x, x < 0), (1, True))
        q = s.Piecewise((y, x < 0), (2, True))
        for operation in (s.Add, s.Mul):
            expected = s.Piecewise((operation(x, y), x < 0), (operation(1, 2), True))
            for mode in (True, False, None):
                self.same(s.piecewise_fold(operation(p, q), evaluate=mode), expected)

    def test_functions_and_powers_rebuild_each_branch(self):
        x = self.x
        p = s.Piecewise((x, x < 0), (x*x, True))
        self.same(s.piecewise_fold(s.sin(p)),
                  s.Piecewise((s.sin(x), x < 0), (s.sin(x*x), True)))
        self.same(s.piecewise_fold(s.Pow(p, 3, evaluate=False)),
                  s.Piecewise((x**3, x < 0), (x**6, True)))

    def test_relational_values_keep_their_boolean_type(self):
        x, y = self.x, self.y
        p = s.Piecewise((x, x < 0), (x*x, True))
        for relation in (s.Lt, s.Eq):
            self.same(s.piecewise_fold(relation(p, y)),
                      s.Piecewise((relation(x, y), x < 0), (relation(x*x, y), True)))

    def test_nested_values_combine_guards_in_source_order(self):
        x, y, z = self.x, self.y, self.z
        p = s.Piecewise((x, x < 0), (x*x, True))
        source = s.Piecewise((p, y < 0), (z, True), evaluate=False)
        expected = s.Piecewise((x, s.And(x < 0, y < 0)), (x*x, y < 0), (z, True))
        for mode in (True, False, None):
            self.same(s.piecewise_fold(source, evaluate=mode), expected)

    def test_direct_nested_piecewise_retains_its_complementary_inner_guard(self):
        x, y = self.x, self.y
        inner = s.Piecewise((1, x < 0), (2, x >= 0), evaluate=False)
        source = s.Piecewise((inner, y < 0), (3, True), evaluate=False)
        expected = s.Piecewise((1, s.And(x < 0, y < 0)),
                              (2, s.And(x >= 0, y < 0)), (3, True), evaluate=False)
        for mode in (True, False, None):
            self.same(s.piecewise_fold(source, evaluate=mode), expected)

    def test_duplicate_guard_modes(self):
        x = self.x
        source = s.Piecewise((1, x < 0), (2, x < 0), (3, True), evaluate=False)
        self.same(s.piecewise_fold(source, evaluate=False), source)
        expected = s.Piecewise((1, x < 0), (3, True), evaluate=False)
        self.same(s.piecewise_fold(source), expected)
        self.same(s.piecewise_fold(source, evaluate=None), expected)

    def test_exact_relational_complements_only_normalize_in_full_mode(self):
        x = self.x
        source = s.Piecewise((1, x < 0), (2, x >= 0), evaluate=False)
        self.same(s.piecewise_fold(source), s.Piecewise((1, x < 0), (2, True)))
        for mode in (False, None):
            self.same(s.piecewise_fold(source, evaluate=mode), source)

    def test_held_symbolic_equality_is_decided_only_by_full_normalization(self):
        x = self.x
        source = s.Piecewise((1, s.Eq(x, x, evaluate=False)), (2, True), evaluate=False)
        self.same(s.piecewise_fold(source), s.Integer(1))
        for mode in (False, None):
            self.same(s.piecewise_fold(source, evaluate=mode), source)

    def test_contradictory_nested_guards_preserve_raw_empty_piecewise(self):
        x = self.x
        source = s.Piecewise((s.Piecewise((1, x < 0)), x >= 0), evaluate=False)
        self.assertIs(s.piecewise_fold(source), s.nan)
        for mode in (False, None):
            result = s.piecewise_fold(source, evaluate=mode)
            self.assertIs(type(result), s.Piecewise)
            self.assertEqual(result.args, ())
            with self.assertRaises(TypeError):
                s.piecewise_fold(result, evaluate=mode)

    def test_adjacent_equal_values_only_merge_in_full_mode(self):
        x, y, z = self.x, self.y, self.z
        source = s.Piecewise((x, x < 0), (x, y < 0), (z, True), evaluate=False)
        expected = s.Piecewise((x, s.Or(x < 0, y < 0)), (z, True))
        self.same(s.piecewise_fold(source), expected)
        for mode in (False, None):
            self.same(s.piecewise_fold(source, evaluate=mode), source)

    def test_merged_guard_shadows_a_later_branch_after_normalization(self):
        x, y = self.x, self.y
        source = s.Piecewise((1, x < 0), (1, y < 0),
                             (2, s.Or(x < 0, y < 0)), (3, True), evaluate=False)
        expected = s.Piecewise((1, s.Or(x < 0, y < 0)), (3, True))
        self.same(s.piecewise_fold(source), expected)
        self.same(s.piecewise_fold(source, evaluate=False), source)

    def test_recursive_operands_use_default_evaluation_before_raw_outer_fold(self):
        x = self.x
        p = s.Piecewise((1, x < 0), (2, x >= 0), evaluate=False)
        q = s.Piecewise((3, x < 0), (4, True), evaluate=False)
        source = s.Add(p, q, evaluate=False)
        expected = s.Piecewise((4, x < 0), (5, x < 0), (5, x < 0), (6, True), evaluate=False)
        self.same(s.piecewise_fold(source, evaluate=False), expected)
        for mode in (True, None):
            self.same(s.piecewise_fold(source, evaluate=mode),
                      s.Piecewise((4, x < 0), (6, True)))

    def test_ordered_guards_remove_known_false_factors_without_inequality_guessing(self):
        x, y = self.x, self.y
        source = s.Piecewise((1, x < 0), (2, s.And(x >= 0, y < 0)),
                             (3, x < 1), (4, True), evaluate=False)
        expected = s.Piecewise((1, x < 0), (2, y < 0), (3, x < 1), (4, True))
        self.same(s.piecewise_fold(source), expected)

    def test_rational_relational_coefficients_are_canonicalized(self):
        x = self.x
        guard = s.Lt(-2*x + 1, 0, evaluate=False)
        source = s.Piecewise((x, guard), (0, True), evaluate=False)
        self.same(s.piecewise_fold(source), s.Piecewise((x, x > s.Rational(1, 2)), (0, True)))
        self.same(s.piecewise_fold(source, evaluate=False), source)

    def test_ite_guards_are_rewritten_to_the_pinned_cnf(self):
        x, y = self.x, self.y
        source = s.Piecewise((x, s.ITE(x < 0, y < 0, y > 1)), (0, True), evaluate=False)
        guard = s.And(s.Or(x >= 0, y < 0), s.Or(y > 1, x < 0))
        for mode in (True, False, None):
            self.same(s.piecewise_fold(source, evaluate=mode), s.Piecewise((x, guard), (0, True)))

    def test_complete_boolean_piecewise_condition_is_folded(self):
        x, y = self.x, self.y
        condition = s.Piecewise((x < 0, y < 0), (x > 0, True), evaluate=False)
        source = s.Piecewise((1, condition), (2, True), evaluate=False)
        guard = s.And(s.Or(y >= 0, x < 0), s.Or(x > 0, y < 0))
        self.same(s.piecewise_fold(source), s.Piecewise((1, guard), (2, True)))

    def test_held_undefined_branch_and_untouched_children_keep_identity(self):
        x = self.x
        undefined = s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False)
        source = s.Piecewise((undefined, x < 0), (1, True), evaluate=False)
        for mode in (True, False, None):
            result = s.piecewise_fold(source, evaluate=mode)
            self.same(result, source)
            self.assertIs(result.args[0].expr, undefined)

    def test_typed_same_name_symbols_and_dummies_remain_distinct(self):
        plain, positive = s.Symbol("same"), s.Symbol("same", positive=True)
        first, second = s.Dummy("same"), s.Dummy("same")
        guard = s.Symbol("True")
        source = s.Piecewise((plain + first, guard), (positive + second, True), evaluate=False)
        result = s.piecewise_fold(s.sin(source))
        expected = s.Piecewise((s.sin(plain + first), guard), (s.sin(positive + second), True))
        self.same(result, expected)
        self.assertEqual(result.free_symbols, {plain, positive, first, second, guard})

    def test_custom_function_reconstruction_preserves_actual_class(self):
        x = self.x
        first = type("SameFunction", (s.Function,), {})
        second = type("SameFunction", (s.Function,), {})
        p = s.Piecewise((first(x), x < 0), (second(x), True))
        result = s.piecewise_fold(s.sin(p))
        self.assertIs(type(result.args[0].expr.args[0]), first)
        self.assertIs(type(result.args[1].expr.args[0]), second)

    def test_custom_eval_runs_in_the_existing_supervised_lane(self):
        x = self.x
        calls = []

        class Shift(s.Function):
            @classmethod
            def eval(cls, argument):
                _supervised()
                calls.append(argument)
                return argument + 1

        p = s.Piecewise((x, x < 0), (x*x, True))
        source = Shift(p, evaluate=False)
        result = s.piecewise_fold(source)
        self.same(result, s.Piecewise((x + 1, x < 0), (x*x + 1, True)))
        self.assertEqual(calls, [x, x*x])

    def test_folded_result_connects_to_substitution_and_math_compilation(self):
        x = self.x
        p = s.Piecewise((1/x, s.Ne(x, 0)), (7, True))
        result = s.piecewise_fold(p + 2)
        self.same(result, s.Piecewise((2 + 1/x, s.Ne(x, 0)), (9, True)))
        self.assertEqual(result.subs(x, 0), 9)
        function = s.lambdify(x, result, modules="math")
        self.assertEqual([function(value) for value in (-2, 0, 2)], [1.5, 9, 2.5])


@unittest.skipUnless(hasattr(s.core, "supervised_callback_session"), "native shell boundary")
class PiecewiseFoldBoundaryTests(unittest.TestCase):
    def test_non_boolean_evaluate_is_refused_without_truth_conversion(self):
        calls = []

        class Mode:
            def __bool__(self):
                calls.append("truth")
                return True

        x = s.Symbol("fold_mode")
        with self.assertRaisesRegex(TypeError, "True, False, or None"):
            s.piecewise_fold(s.Piecewise((x, x < 0), (1, True)), evaluate=Mode())
        self.assertEqual(calls, [])

    def test_malformed_branch_storage_is_refused_without_iterating_user_objects(self):
        calls = []

        class Branches:
            def __iter__(self):
                calls.append("iterate")
                return iter(())

        x = s.Symbol("fold_branch_storage")
        source = s.Piecewise((x, x < 0), (1, True))
        s.Piecewise.__dict__["_piecewise_branches"].__set__(source, Branches())
        with self.assertRaisesRegex(ValueError, "stored branch pairs"):
            s.piecewise_fold(source)
        self.assertEqual(calls, [])

    def test_independent_branch_growth_is_refused_before_user_reconstruction(self):
        calls = []

        class Observe(s.Function):
            @classmethod
            def eval(cls, *arguments):
                calls.append(arguments)

        symbols = s.symbols("fold_limit0:10")
        operands = [s.Piecewise((v, v < 0), (1, True)) for v in symbols]
        source = Observe(*operands, evaluate=False)
        original = source.args
        with self.assertRaisesRegex(ValueError, "branch limit"):
            s.piecewise_fold(source)
        self.assertEqual(calls, [])
        self.assertEqual(source.args, original)

    def test_guard_products_are_planned_before_user_reconstruction(self):
        calls = []

        class Observe(s.Function):
            @classmethod
            def eval(cls, *arguments):
                calls.append(arguments)

        symbols = s.symbols("fold_guard_limit0:10")
        operands = [s.Piecewise((v, v < 0), (1, True)) for v in symbols]
        guard = s.Gt(Observe(*operands, evaluate=False), 0, evaluate=False)
        source = s.Piecewise((1, guard), (2, True), evaluate=False)
        with self.assertRaisesRegex(ValueError, "branch limit"):
            s.piecewise_fold(source)
        self.assertEqual(calls, [])

    def test_ite_growth_is_planned_before_value_reconstruction(self):
        calls = []

        class Observe(s.Function):
            @classmethod
            def eval(cls, value):
                calls.append(value)

        symbols = s.symbols("fold_ite_limit0:31")
        condition = symbols[0]
        for index in range(1, 30, 2):
            condition = s.ITE(condition, symbols[index], symbols[index + 1])
        value = Observe(s.Piecewise((1, symbols[0]), (2, True)), evaluate=False)
        source = s.Piecewise((value, condition), (3, True), evaluate=False)
        with self.assertRaisesRegex(ValueError, "Boolean expansion limit"):
            s.piecewise_fold(source)
        self.assertEqual(calls, [])

    def test_modified_builtin_owners_are_refused_before_their_effects_run(self):
        from sympy.functions.elementary.piecewise import ExprCondPair

        x = s.Symbol("fold_injected_owner")
        source = s.Piecewise((x, x < 0), (1, True))
        calls = []

        def injected(*args, **kwargs):
            calls.append("effect")
            raise AssertionError("modified builtin escaped owner admission")

        cases = (
            (s.Basic, "args", property(injected)),
            (s.Symbol, "__eq__", injected),
            (s.core.Relational, "canonical", property(injected)),
            (ExprCondPair, "__new__", staticmethod(injected)),
        )
        for owner, name, replacement in cases:
            with self.subTest(owner=owner.__name__, name=name):
                with patch.object(owner, name, replacement):
                    with self.assertRaisesRegex(NotImplementedError, "built-in owners"):
                        s.piecewise_fold(source)
        self.assertEqual(calls, [])

    def test_shared_guard_grouping_prevents_exponential_growth(self):
        x = s.Symbol("fold_shared_limit")
        operands = [s.Piecewise((i*x, x < 0), (i, True)) for i in range(1, 12)]
        result = s.piecewise_fold(s.Add(*operands, evaluate=False))
        self.assertEqual(result, s.Piecewise((66*x, x < 0), (66, True)))
        self.assertEqual(len(result.args), 2)

    def test_descriptor_reads_and_reconstruction_are_supervised(self):
        reads = []

        class Retained(s.Expr):
            @property
            def args(self):
                _supervised()
                reads.append("args")
                return s.Basic.__dict__["_struct_args"].__get__(self, type(self))

            @property
            def func(self):
                _supervised()
                reads.append("func")
                return type(self)

        x = s.Symbol("fold_descriptor")
        p = s.Piecewise((x, x < 0), (1, True))
        source = Retained(p)
        result = s.piecewise_fold(source)
        self.assertIs(type(result), s.Piecewise)
        self.assertEqual(reads.count("func"), 2)
        self.assertTrue(reads.count("args") >= 1)

    def test_callback_failure_propagates_and_taints_the_owning_request(self):
        calls = []

        class Failing(s.Function):
            @classmethod
            def eval(cls, argument):
                _supervised()
                calls.append(argument)
                raise RuntimeError("fold callback failed")

        x = s.Symbol("fold_failure")
        source = Failing(s.Piecewise((x, x < 0), (1, True)), evaluate=False)
        original = source.args
        with _request() as session:
            with self.assertRaisesRegex(RuntimeError, "fold callback failed"):
                s.piecewise_fold(source)
            self.assertTrue(session.unknown_effect_tainted)
        self.assertEqual(calls, [x])
        self.assertEqual(source.args, original)

    def test_callback_invocation_limit_refuses_before_descriptor_effect(self):
        calls = []

        class Retained(s.Expr):
            @property
            def args(self):
                calls.append("args")
                return s.Basic.__dict__["_struct_args"].__get__(self, type(self))

        x = s.Symbol("fold_budget")
        source = Retained(s.Piecewise((x, x < 0), (1, True)))
        with _request(max_invocations=1) as session:
            session.invoke_fresh("test", "already-used", 1, lambda: None, ())
            with self.assertRaisesRegex(ValueError, "invocation count"):
                s.piecewise_fold(source)
        self.assertEqual(calls, [])

    def test_unsupported_held_constant_guard_normalization_is_explicit(self):
        guard = s.Eq(1, 1, evaluate=False)
        source = s.Piecewise((1, guard), (2, True), evaluate=False)
        with self.assertRaisesRegex(NotImplementedError, "held constant relational"):
            s.piecewise_fold(source)
        self.assertEqual(s.piecewise_fold(source, evaluate=False).args, source.args)

    def test_cyclic_source_is_refused_without_publishing_a_result(self):
        class Cyclic(s.Basic):
            pass

        source = Cyclic()
        s.Basic.__dict__["_struct_args"].__set__(source, (source,))
        with self.assertRaisesRegex(ValueError, "graph|limit"):
            s.piecewise_fold(source)

    def test_incomplete_boolean_piecewise_condition_is_not_filled_with_false(self):
        x, y = s.symbols("fold_partial_x fold_partial_y")
        condition = s.Piecewise((x < 0, y < 0), evaluate=False)
        source = s.Piecewise((1, condition), (2, True), evaluate=False)
        with self.assertRaisesRegex(ValueError, "cover all"):
            s.piecewise_fold(source)


if __name__ == "__main__":
    unittest.main()
