"""Typed Lambda application and ordered unevaluated substitution.

The portable expectations are anchored to isolated SymPy 1.14.0. In that
profile Lambda performs simultaneous structural replacement, including under
nested binders, whereas Subs evaluates its retained mappings sequentially.
Resource and callback tests additionally exercise this shell's boundaries.
"""

import copy
import pickle

import pytest
import sympy as s

from sympy.core.lambdify_subs import BadArgumentsError, BadSignatureError


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_parameter_swaps_never_consume_a_preceding_replacement(order):
    variables = s.symbols("lambda_swap_x lambda_swap_y")
    x, y = (variables[index] for index in order)
    assert s.Lambda((x, y), x - y)(y, x) == y - x
    assert s.Lambda((x, y), x / y)(y, x) == y / x
    assert s.Lambda((x, y), x + 2*y)(y, x) == y + 2*x


def test_three_way_cycles_and_replacements_with_free_parameters():
    x, y, z = s.symbols("lambda_cycle_x lambda_cycle_y lambda_cycle_z")
    function = s.Lambda((x, y, z), x + 2*y + 3*z)
    assert function(y, z, x) == y + 2*z + 3*x
    assert function(x + y, y + z, z + x) == 4*x + 3*y + 5*z


def test_zero_and_pole_arguments_are_installed_together():
    x, y = s.symbols("lambda_zero_x lambda_zero_y")
    assert s.Lambda((x, y), x / y)(0, 0) is s.nan
    assert s.Lambda((x, y), x / y)(2, 0) is s.zoo


def test_exact_projection_returns_the_actual_replacement_object():
    x, y = s.symbols("lambda_projection_x lambda_projection_y")
    marker = object()
    assert s.Lambda((x, y), x)(marker, 3) is marker


@pytest.mark.parametrize("typed", ["assumptions", "dummy"])
def test_same_printed_parameter_names_keep_their_distinct_identity(typed):
    if typed == "assumptions":
        first, second = s.Symbol("lambda_owner", positive=True), s.Symbol("lambda_owner")
    else:
        first, second = s.Dummy("lambda_owner"), s.Dummy("lambda_owner")
    function = s.Lambda((first, second), first + 2*second)
    assert function.variables[0] is first
    assert function.variables[1] is second
    assert function(3, 5) == 13
    assert function(second, first) == second + 2*first
    assert s.Lambda(first, first + second)(3) == 3 + second


def test_nested_tuple_signatures_unpack_without_flattening_the_call_shape():
    x, y, z, t = s.symbols("lambda_nested_x lambda_nested_y lambda_nested_z lambda_nested_t")
    function = s.Lambda(((x, y), (z, (t,))), x + 2*y + 3*z + 4*t)
    assert function((1, 2), (3, (4,))) == 30
    assert function(s.Tuple(1, 2), s.Tuple(3, s.Tuple(4))) == 30
    assert function.variables == (x, y, z, t)
    assert function.bound_symbols == (x, y, z, t)
    assert function.nargs == s.FiniteSet(2)
    assert function.signature == s.Tuple(s.Tuple(x, y), s.Tuple(z, s.Tuple(t)))
    assert function.free_symbols == set()


def test_single_tuple_argument_and_zero_argument_lambdas():
    x, y, free = s.symbols("lambda_single_x lambda_single_y lambda_single_free")
    function = s.Lambda(((x, y),), x - y + free)
    assert function((3, 2)) == 1 + free
    assert function.free_symbols == {free}
    assert str(function).startswith("Lambda(((lambda_single_x, lambda_single_y),),")
    constant = s.Lambda((), free)
    assert constant() is free
    assert constant.variables == ()
    assert constant.nargs == s.FiniteSet(0)


@pytest.mark.parametrize("signature", ["flat", "nested"])
def test_duplicate_parameters_are_rejected_before_application(signature):
    x = s.Symbol("lambda_duplicate")
    declaration = (x, x) if signature == "flat" else ((x,), x)
    with pytest.raises(BadSignatureError, match="Duplicate"):
        s.Lambda(declaration, x)


@pytest.mark.parametrize("kind", ["number", "expression", "function"])
def test_non_symbol_parameters_are_rejected(kind):
    x = s.Symbol("lambda_invalid")
    value = {"number": 1, "expression": x + 1, "function": s.sin(x)}[kind]
    with pytest.raises(BadSignatureError):
        s.Lambda((x, value), x)


@pytest.mark.parametrize("arguments", [(), (1,), (1, 2, 3), ((1,), 2), ((1, 2, 3), 2), (1, 2)])
def test_bad_outer_and_nested_argument_shapes_refuse(arguments):
    x, y, z = s.symbols("lambda_shape_x lambda_shape_y lambda_shape_z")
    function = s.Lambda(((x, y), z), x + y + z)
    with pytest.raises(BadArgumentsError):
        function(*arguments)


def test_same_named_tuple_impostor_does_not_acquire_container_semantics():
    x, y = s.symbols("lambda_tuple_x lambda_tuple_y")

    class Tuple:
        @property
        def args(self):
            raise AssertionError("a name must not authorize reading custom args")

        def __iter__(self):
            raise AssertionError("a name must not authorize an iterator")

    with pytest.raises(BadSignatureError):
        s.Lambda(Tuple(), x)
    with pytest.raises(BadArgumentsError):
        s.Lambda(((x, y),), x + y)(Tuple())


def test_signature_size_depth_and_cycles_have_explicit_limits():
    from sympy.core.lambdify_subs import _MAX_BINDING_DEPTH, _MAX_BINDING_NODES

    x = s.Symbol("lambda_bounded")
    with pytest.raises(ValueError, match="limit"):
        s.Lambda([x] * (_MAX_BINDING_NODES + 1), x)
    nested = x
    for _ in range(_MAX_BINDING_DEPTH + 2):
        nested = (nested,)
    with pytest.raises(ValueError, match="limit"):
        s.Lambda(nested, x)
    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cyclic"):
        s.Lambda(cyclic, x)


def test_nested_binders_follow_structural_application_in_the_pinned_profile():
    x, y = s.symbols("lambda_binder_x lambda_binder_y")
    # Structural replacement intentionally enters nested binder declarations;
    # kernel alpha-renaming must not silently change this Python contract.
    assert s.Lambda(x, s.Lambda(y, x + y))(y) == s.Lambda(y, 2*y)
    integral = s.Integral(x*y, (y, 0, 1))
    assert s.Lambda(x, integral)(y) == s.Integral(y*y, (y, 0, 1))


def test_lambda_substitution_retains_symbolic_signature_and_body_together():
    x, y, z = s.symbols("lambda_subs_x lambda_subs_y lambda_subs_z")
    function = s.Lambda(x, x + y)
    assert function.subs(x, z) == s.Lambda(z, z + y)
    assert function.subs(y, x) == s.Lambda(x, 2*x)
    assert function.subs(z, 3) is function
    assert function.xreplace({x: z}) == s.Lambda(z, z + y)


def test_unchanged_held_nodes_retain_order_and_changed_parents_reconstruct():
    x, y, z = s.symbols("lambda_held_x lambda_held_y lambda_held_z")
    source = s.Add(x, x, y, evaluate=False)
    assert s.Lambda(z, source)(3) is source
    assert source.args == (x, x, y)
    assert s.Lambda(x, source)(z) == 2*z + y
    assert source.args == (x, x, y)


def test_application_does_not_run_algebraic_substitution_hooks():
    x, y = s.symbols("lambda_hook_x lambda_hook_y")

    class NoAlgebraicSubstitution(s.Function):
        def _eval_subs(self, old, new):
            raise AssertionError("Lambda application must not call _eval_subs")

    source = NoAlgebraicSubstitution(x, evaluate=False)
    result = s.Lambda((x, y), source)(y, x)
    assert result.func is NoAlgebraicSubstitution
    assert result.args == (y,)


def test_custom_function_evaluation_stays_in_the_supervised_callback_lane():
    from sympy.core import _active_callback_session, supervised_callback_session

    x = s.Symbol("lambda_supervised")
    calls = []

    class Evaluated(s.Function):
        @classmethod
        def eval(cls, value):
            assert _active_callback_session() is not None
            calls.append(value)
            if value == 3:
                return s.Integer(7)

    source = Evaluated(x, evaluate=False)
    with supervised_callback_session():
        assert s.Lambda(x, source)(3) == 7
    assert calls == [s.Integer(3)]


def test_application_never_uses_custom_printers_or_argument_properties():
    x, y = s.symbols("lambda_guard_x lambda_guard_y")

    class Guarded(s.Function):
        @property
        def args(self):
            raise AssertionError("retained storage must be read directly")

        def __str__(self):
            raise AssertionError("printed text must not be parameter identity")

    source = Guarded(x, evaluate=False)
    result = s.Lambda((x, y), source)(y, x)
    assert result.func is Guarded
    assert s.Function.__dict__["_args"].__get__(result, s.Function) == (y,)


def test_lambda_reconstruction_copy_and_pickle_preserve_nested_signature():
    x, y = s.symbols("lambda_copy_x lambda_copy_y")
    source = s.Lambda(((x, y),), x + 2*y)
    for restored in (source.func(*source.args), copy.copy(source), copy.deepcopy(source),
                     pickle.loads(pickle.dumps(source))):
        assert restored == source
        assert restored((3, 5)) == 13
        assert restored.signature == source.signature


def test_subs_retains_points_needed_by_later_ordered_replacements():
    x, y = s.symbols("subs_cascade_x subs_cascade_y")
    source = s.Subs(x, (x, y), (y, 2))
    assert source.variables == (x, y)
    assert source.point == (y, s.Integer(2))
    assert source.free_symbols == {y}
    assert source.doit() == 2


def test_subs_preserves_declared_order_instead_of_sorting_a_mapping():
    x, y = s.symbols("subs_order_x subs_order_y")
    assert s.Subs(x - y, (y, x), (x, 1)).doit() == 0
    assert s.Subs(x - y, (x, y), (y, 1)).doit() == 0
    assert s.Subs(x, (y, x), (2, y)).doit() == y
    # This contrast is an intentional, isolated 1.14.0 observation.
    assert s.Lambda((x, y), x - y)(y, x) == y - x
    assert s.Subs(x - y, (x, y), (y, x)).doit() == 0


def test_absent_and_self_substitutions_remain_observable_until_doit():
    x, y, z = s.symbols("subs_retained_x subs_retained_y subs_retained_z")
    source = s.Subs(x, y, z)
    assert type(source) is s.Subs
    assert source.variables == (y,)
    assert source.point == (z,)
    assert source.free_symbols == {x, z}
    assert source.doit() is x
    identity = s.Subs(x, x, x)
    assert type(identity) is s.Subs
    assert identity.free_symbols == {x}
    assert identity.doit() == x
    assert s.Subs(x, (), ()) is x


def test_nested_subs_flatten_without_changing_evaluation_order():
    x, y = s.symbols("subs_nested_x subs_nested_y")
    source = s.Subs(s.Subs(x, x, y), y, 2)
    assert source.expr is x
    assert source.variables == (x, y)
    assert source.point == (y, s.Integer(2))
    assert source.doit() == 2


def test_duplicate_subs_variables_refuse_but_distinct_typed_symbols_survive():
    x = s.Symbol("subs_duplicate")
    with pytest.raises(ValueError, match="more than once"):
        s.Subs(x, (x, x), (1, 2))
    first, second = s.Symbol("subs_same", positive=True), s.Symbol("subs_same")
    source = s.Subs(first + 2*second, (second, first), (3, 5))
    assert source.variables == (second, first)
    assert source.doit() == 11


def test_subs_substitution_preserves_bound_coordinates_and_point_dependencies():
    x, y, z = s.symbols("subs_scope_x subs_scope_y subs_scope_z")
    source = s.Subs(x, x, y)
    renamed = source.subs(x, z)
    assert renamed.expr is z
    assert renamed.variables == (z,)
    assert renamed.point == (y,)
    assert source.subs(x, y).args == source.args
    replaced = source.subs(y, x)
    assert replaced.variables == (x,)
    assert replaced.point == (x,)
    assert replaced.doit() == x
    assert s.Subs(x*y, x, 0).subs(y, 2).doit() == 0


def test_substitution_inside_a_compound_bound_expression_is_deferred():
    x, y, z = s.symbols("subs_compound_x subs_compound_y subs_compound_z")
    source = s.Subs(x*y, x*y, z)
    changed = source.subs(x, 1)
    assert changed.variables == (x*y, x)
    assert changed.point == (z, s.Integer(1))
    assert changed.doit() == z


def test_subs_copy_and_pickle_retain_the_whole_ordered_program():
    x, y = s.symbols("subs_copy_x subs_copy_y")
    source = s.Subs(x, (x, y), (y, 2))
    for restored in (source.func(*source.args), copy.copy(source), copy.deepcopy(source),
                     pickle.loads(pickle.dumps(source))):
        assert restored.args == source.args
        assert restored.doit() == 2


def test_subs_evaluates_a_retained_derivative_before_substitution():
    x = s.Symbol("subs_derivative_x")
    source = s.Subs(s.Derivative(x*x, x, evaluate=False), x, 2)
    assert source.doit() == 4


def test_lambda_errors_have_the_standard_function_module_identity():
    from sympy.core.function import BadArgumentsError as public_arguments
    from sympy.core.function import BadSignatureError as public_signature

    assert public_arguments is BadArgumentsError
    assert public_signature is BadSignatureError
    assert public_arguments.__module__ == "sympy.core.function"
    assert public_signature.__module__ == "sympy.core.function"
