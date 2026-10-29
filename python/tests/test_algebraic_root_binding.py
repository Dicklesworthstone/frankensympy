"""Bound root constants inside genuine arithmetic and user function collisions."""

import pytest
import sympy as s
from sympy import core as _core


def algebraic_root(name="bound_root_t"):
    generator = s.Symbol(name)
    return s.CRootOf(generator**5 - generator + 1, generator, 0)


def test_root_and_nested_arithmetic_have_no_defining_free_symbol():
    root = algebraic_root()
    for expression in [root, 1 + root**2, (1 + root**2)**3, s.sin(root) + root/3]:
        assert expression.free_symbols == set()


def test_a_defining_generator_can_remain_free_outside_the_root():
    x, y = s.symbols("bound_root_same bound_root_target")
    root = algebraic_root(x.name)
    for expression in [x + root, (x + root)**2, s.sin(x*root)]:
        assert expression.free_symbols == {x}
        replaced = expression.subs(x, y)
        assert replaced.free_symbols == {y}
        assert replaced.atoms(s.CRootOf) == {root}
    assert root.subs(x, y) == root
    assert root.poly.gen == x


def test_typed_assumptions_and_dummy_identities_survive_composite_binding():
    plain = s.Symbol("bound_root_shared")
    positive = s.Symbol("bound_root_shared", positive=True)
    first, second = s.Dummy("bound_root_shared"), s.Dummy("bound_root_shared")
    destination = s.Symbol("bound_root_destination", integer=True)
    root = s.CRootOf(positive**5 - positive + 1, positive, 0)
    expression = plain + positive + first + second + root
    assert expression.free_symbols == {plain, positive, first, second}
    actual = {id(symbol) for symbol in expression.free_symbols}
    assert actual == {id(plain), id(positive), id(first), id(second)}
    replaced = expression.subs(first, destination).subs(positive, 2)
    assert replaced.free_symbols == {plain, second, destination}
    assert replaced.atoms(s.CRootOf) == {root}
    assert root.subs(plain, destination) == root
    assert root.subs(positive, destination) == root


def test_held_equation_substitution_preserves_the_bound_root_on_the_other_side():
    root = algebraic_root()
    generator = root.poly.gen
    replacement = s.Symbol("root_equation_variable")
    equation = s.Eq(generator, root, evaluate=False)
    assert equation.free_symbols == {generator}
    substituted = equation.subs(generator, replacement)
    assert substituted.lhs == replacement
    assert substituted.rhs == root
    assert substituted.free_symbols == {replacement}


def test_dummy_defining_generator_remains_bound_when_it_is_free_elsewhere():
    variable = s.Dummy("root_dummy")
    other = s.Dummy("root_dummy")
    root = s.CRootOf(variable**5 - variable + 1, variable, 0)
    expression = variable + other*root
    assert expression.free_symbols == {variable, other}
    replaced = expression.subs(variable, 3)
    assert replaced.free_symbols == {other}
    assert replaced.atoms(s.CRootOf) == {root}


def test_whole_root_replacement_does_not_descend_into_its_polynomial():
    root = algebraic_root()
    x, y = s.symbols("root_replacement_x root_replacement_y")
    assert root.subs(root, x) is x
    assert (1 + root**2).subs(root, x) == 1 + x**2
    assert (x + root).subs({root: y, x: 2}) == y + 2
    assert root.subs(root.poly.gen, 0) == root
    assert root.subs(root.args[0], y) == root


def test_alpha_equivalent_root_replacement_respects_value_identity():
    root = algebraic_root("root_alpha_first")
    renamed = algebraic_root("root_alpha_second")
    x = s.Symbol("root_alpha_value")
    assert root == renamed
    assert root.subs(renamed, x) is x
    assert (1 + root**2).subs(renamed, x) == 1 + x**2


def test_sequential_and_simultaneous_root_substitutions_preserve_scope():
    root = algebraic_root()
    x, y = s.symbols("root_subs_x root_subs_y")
    assert root.subs([(root, x), (x, y)]) is y
    assert root.subs([(root, x), (x, y)], simultaneous=True) is x
    assert (x + root).subs({root: x, x: y}, simultaneous=True) == x + y
    assert (x + root).subs({root.poly.gen: y, x: 3}) == root + 3


def test_algebraic_constants_differentiate_to_zero_inside_arithmetic():
    root = algebraic_root()
    x = root.poly.gen
    other = s.Symbol("root_derivative_other")
    for variable in [x, other]:
        assert s.diff(root, variable) == 0
        assert s.diff(1 + root**2, variable) == 0
    assert s.diff((x + root)**3, x) == 3*(x + root)**2
    assert s.diff(root*x**2 + x, x) == 2*root*x + 1
    assert s.diff(s.sin(root*x), x) == root*s.cos(root*x)
    assert s.diff((x + root)**3, x, 2) == 6*(x + root)


def test_differentiation_preserves_distinct_typed_symbols():
    plain = s.Symbol("root_diff_shared")
    assumed = s.Symbol("root_diff_shared", real=True)
    dummy = s.Dummy("root_diff_shared")
    root = s.CRootOf(assumed**5 - assumed + 1, assumed, 0)
    expression = root*plain + assumed**2 + dummy
    assert s.diff(expression, plain) == root
    assert s.diff(expression, assumed) == 2*assumed
    assert s.diff(expression, dummy) == 1


def test_supervised_derivative_hook_receives_the_original_algebraic_constant():
    root = algebraic_root()
    x = s.Symbol("root_hook_x")
    arguments = []

    class RootCoefficient(s.Function):
        nargs = 2

        @classmethod
        def eval(cls, variable, coefficient):
            assert not isinstance(coefficient, s.Dummy)
            return None

        def fdiff(self, argindex=1):
            arguments.append((argindex, self.args))
            if argindex == 1:
                assert type(self.args[1]) is s.CRootOf
                return self.args[1]
            if argindex == 2:
                return self.args[0]
            raise ValueError("unexpected derivative argument")

    assert s.diff(RootCoefficient(x, root), x) == root
    assert arguments == [(1, (x, root))]


@pytest.mark.parametrize("module", [__name__, "sympy.untrusted_root_module"])
def test_same_name_user_function_cannot_capture_root_lifting_or_become_constant(monkeypatch, module):
    x, y, z = s.symbols("opaque_root_x opaque_root_y opaque_root_z")
    root = algebraic_root("opaque_root_generator")
    # Restore the known-function registry after the intentionally spoofed
    # module case; its spelling alone must not establish a trusted binder.
    monkeypatch.setitem(_core._known_functions, "CRootOf", s.CRootOf)
    custom_class = type("CRootOf", (s.Function,), {"__module__": module})
    custom = custom_class(x + y, 0)
    assert type(custom) is custom_class
    assert custom.free_symbols == {x, y}
    replaced = custom.subs(x, z)
    assert type(replaced) is custom_class
    assert replaced.free_symbols == {y, z}
    with pytest.raises(NotImplementedError, match="same-name native intrinsic"):
        _core._native_expr(custom)
    with pytest.raises(NotImplementedError, match="same-name native intrinsic"):
        s.Add(custom, root, evaluate=False)
    genuine_composite = 1 + root**2
    assert genuine_composite.free_symbols == set()
    assert genuine_composite.atoms(s.CRootOf) == {root}
    assert s.diff(genuine_composite, root.poly.gen) == 0
