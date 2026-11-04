"""Bound CRootOf identity must remain congruent inside ordered expressions."""

import importlib

import pytest
import sympy as s

from sympy.core import _native_expr, _wrap


@pytest.fixture
def roots():
    x, y = s.symbols("root_identity_x root_identity_y")
    left = [s.CRootOf(x**3 - 3*x + 1, x, index) for index in range(3)]
    right = [s.CRootOf(y**3 - 3*y + 1, y, index) for index in range(3)]
    return left, right


def test_equal_bound_roots_keep_equality_and_hash_under_sqrt_and_ratios(roots):
    left, right = roots
    assert left == right
    assert [hash(root) for root in left] == [hash(root) for root in right]
    for a, b in [
        (s.sqrt(left[2]), s.sqrt(right[2])),
        (s.sqrt(left[2]) / s.sqrt(left[1]), s.sqrt(right[2]) / s.sqrt(right[1])),
    ]:
        assert a == b
        assert b == a
        assert not a != b
        assert hash(a) == hash(b)
        assert {a: "same exact constant"}[b] == "same exact constant"
        assert len({a, b}) == 1


def test_native_lifted_and_held_same_shapes_have_one_identity(roots):
    left, right = roots
    original = s.sqrt(left[2])
    lifted = _wrap(_native_expr(original))
    held = s.Pow(right[2], s.Rational(1, 2), evaluate=False)
    assert original == lifted == held
    assert hash(original) == hash(lifted) == hash(held)
    z = s.Symbol("root_identity_outer")
    a = s.Add(original, z, evaluate=False)
    b = s.Add(held, z, evaluate=False)
    assert a == b
    assert hash(a) == hash(b)


def test_whole_composite_substitution_matches_an_alpha_renamed_root(roots):
    left, right = roots
    z, replacement = s.symbols("root_identity_z root_identity_replacement")
    expression = z + s.sqrt(left[2])
    assert expression.subs(s.sqrt(right[2]), replacement) == z + replacement


def test_distinct_indices_polynomials_and_zero_coefficient_positions_stay_distinct(roots):
    left, right = roots
    assert s.sqrt(left[2]) != s.sqrt(right[1])
    x, y = s.symbols("root_identity_coeff_x root_identity_coeff_y")
    # Equal lists of nonzero coefficients would lose the different zero slots.
    first = s.CRootOf(x**3 + x + 1, x, 0)
    second = s.CRootOf(y**3 + y*y + 1, y, 0)
    a = s.Pow(first, 2, evaluate=False)
    b = s.Pow(second, 2, evaluate=False)
    assert a != b
    assert len({a, b}) == 2
    assert s.Pow(left[2], 2, evaluate=False) != s.Pow(right[2], 3, evaluate=False)


def test_free_symbol_assumptions_and_dummy_identity_are_never_erased(roots):
    left, right = roots
    plain = s.Symbol("root_identity_free")
    positive = s.Symbol("root_identity_free", positive=True)
    first, second = s.Dummy("root_identity_dummy"), s.Dummy("root_identity_dummy")
    for a, b in [(plain, positive), (first, second)]:
        assert s.Add(s.sqrt(left[2]), a, evaluate=False) != s.Add(
            s.sqrt(right[2]), b, evaluate=False,
        )
        same_left = s.Add(s.sqrt(left[2]), a, evaluate=False)
        same_right = s.Add(s.sqrt(right[2]), a, evaluate=False)
        assert same_left == same_right
        assert hash(same_left) == hash(same_right)
    expression = s.Add(s.sqrt(left[2]), positive, evaluate=False)
    original_hash, facts = hash(expression), dict(positive._assumptions)
    try:
        positive._assumptions.clear()
        assert hash(expression) == original_hash
        assert expression != s.Add(s.sqrt(right[2]), plain, evaluate=False)
    finally:
        positive._assumptions.update(facts)


def test_bound_generator_is_still_free_when_used_outside_its_root(roots):
    left, right = roots
    x = s.Symbol("root_identity_x")
    y = s.Symbol("root_identity_external_y")
    expression = s.Add(s.sqrt(left[2]), x, evaluate=False)
    alias = s.Add(s.sqrt(right[2]), x, evaluate=False)
    assert expression == alias
    assert expression.free_symbols == {x}
    assert expression.subs(x, y) == s.sqrt(right[2]) + y
    assert expression != s.Add(s.sqrt(right[2]), y, evaluate=False)


def test_true_function_class_identity_survives_equal_root_arguments(roots):
    left, right = roots
    first = type("RootIdentityApplication", (s.Function,), {})
    second = type("RootIdentityApplication", (s.Function,), {})
    a = first(left[2], evaluate=False)
    b = first(right[2], evaluate=False)
    different = second(right[2], evaluate=False)
    assert a == b
    assert hash(a) == hash(b)
    assert a != different
    assert len({a, b, different}) == 2
    assert s.sin(left[2]) == s.sin(right[2])
    assert hash(s.sin(left[2])) == hash(s.sin(right[2]))
    assert s.sin(left[2]) != s.cos(right[2])
    # Equality support grants no permission to lower the opaque application.
    with pytest.raises(NotImplementedError):
        _native_expr(a)


def test_identity_does_not_evaluate_approximate_or_print_root_values(roots, monkeypatch):
    left, right = roots
    a, b = s.sqrt(left[2]), s.sqrt(right[2])

    def forbidden(*args, **kwargs):
        raise AssertionError("identity must not sample or print an algebraic root")

    for name in ("evalf", "__float__", "__str__"):
        monkeypatch.setattr(s.CRootOf, name, forbidden)
    assert a == b
    assert hash(a) == hash(b)


def test_custom_identity_hooks_refuse_without_executing_them(roots):
    left, right = roots

    class CustomIdentity(s.Function):
        def __eq__(self, other):
            raise AssertionError("custom equality must not run")

        def __hash__(self):
            raise AssertionError("custom hashing must not run")

    a = s.Pow(CustomIdentity(left[2], evaluate=False), 2, evaluate=False)
    b = s.Pow(CustomIdentity(right[2], evaluate=False), 2, evaluate=False)
    for operation in (lambda: a == b, lambda: hash(a)):
        with pytest.raises(NotImplementedError, match="custom equality or hashing"):
            operation()


def test_retained_args_are_read_without_user_properties(roots):
    left, right = roots

    class GuardedArgs(s.Function):
        @property
        def args(self):
            raise AssertionError("a user args property must not execute")

    a = GuardedArgs(left[2], evaluate=False)
    b = GuardedArgs(right[2], evaluate=False)
    assert a == b
    assert hash(a) == hash(b)


def test_held_order_and_zero_or_unit_factors_remain_structural(roots):
    left, right = roots
    z = s.Symbol("root_identity_order_z")
    a = s.Add(left[2], z, evaluate=False)
    b = s.Add(right[2], z, evaluate=False)
    reversed_order = s.Add(z, right[2], evaluate=False)
    assert a == b
    assert hash(a) == hash(b)
    assert a != reversed_order
    assert s.Mul(0, left[2], evaluate=False) != 0
    assert s.Mul(1, left[2], evaluate=False) != right[2]
    assert s.Pow(left[2], 1, evaluate=False) != right[2]


def test_shared_retained_dags_do_not_expand_repeated_subtrees(roots):
    left, right = roots
    function = type("RootIdentityDag", (s.Function,), {})
    a, b = left[2], right[2]
    for _ in range(24):
        a = function(a, a, evaluate=False)
        b = function(b, b, evaluate=False)
    assert a == b
    assert hash(a) == hash(b)


def test_identity_work_refusal_is_never_converted_to_false(roots, monkeypatch):
    left, right = roots
    a, b = s.sqrt(left[2]), s.sqrt(right[2])
    module = importlib.import_module("sympy.core._algebraic_identity")
    monkeypatch.setattr(module, "_MAX_NODES", 1)
    for operation in (lambda: a == b, lambda: hash(a)):
        with pytest.raises(ValueError, match="structural work limit"):
            operation()


def test_cyclic_retained_graph_is_rejected(roots):
    left, _ = roots
    function = type("RootIdentityCycle", (s.Function,), {})
    node = function(left[2], evaluate=False)
    s.Function.__dict__["_args"].__set__(node, (node, left[2]))
    with pytest.raises(ValueError, match="cyclic retained"):
        hash(node)


def test_generic_closed_expr_wrappers_keep_native_equality_and_hash(roots):
    left, right = roots
    ordinary = s.sqrt(left[2])
    generic = s.Expr(_native_expr(s.sqrt(right[2])))
    assert ordinary == generic
    assert generic == ordinary
    assert hash(ordinary) == hash(generic)
    assert {ordinary: 1}[generic] == 1
    for unrelated in (0, None, "root"):
        assert ordinary != unrelated


@pytest.mark.parametrize("tag", ["Add", "Mul", "Pow", "Constant"])
def test_native_function_names_cannot_forge_typed_core_node_identity(roots, tag):
    left, _ = roots
    core = importlib.import_module("sympy.core")
    arguments = (_native_expr(left[2]), _native_expr(s.Integer(2)))
    impostor = core._native.py_function(tag, *arguments)
    # This deliberately bypasses ordinary lowering admission to exercise the
    # identity boundary on a raw typed native Function, not an Add/Mul/Pow.
    wrapper = object.__new__(s.Expr)
    s.Basic.__dict__["_value"].__set__(wrapper, impostor)
    ordinary = {
        "Add": lambda: left[2] + 2,
        "Mul": lambda: left[2] * 2,
        "Pow": lambda: s.Pow(left[2], 2),
        "Constant": lambda: s.sqrt(left[2]),
    }[tag]()
    for operation in (lambda: wrapper == ordinary, lambda: hash(wrapper)):
        with pytest.raises(NotImplementedError, match="impersonate"):
            operation()

def test_piecewise_owned_branches_preserve_root_alpha_identity_and_hash(roots):
    left, right = roots
    z = s.Symbol("root_identity_piecewise_z")

    def expression(root, condition_root):
        condition = s.And(s.Ne(z, 0), s.Gt(condition_root, z, evaluate=False))
        return s.Piecewise((s.sqrt(root), condition),
                           (s.sqrt(root) / 2, True), evaluate=False)

    a = expression(left[2], left[1])
    b = expression(right[2], right[1])
    generic = s.Expr(_native_expr(b))
    assert a == b == generic
    assert generic == a
    assert hash(a) == hash(b) == hash(generic)
    assert {a: "branch identity"}[b] == "branch identity"
    assert a != expression(right[2], right[0])


def test_piecewise_owner_and_condition_tags_ignore_same_named_user_classes(roots):
    left, right = roots
    core = importlib.import_module("sympy.core")
    z = s.Symbol("root_identity_piecewise_shadow_z")
    condition = s.And(s.Ne(z, 0), s.Gt(z, -2))
    a = s.Piecewise((s.sqrt(left[2]), condition), (0, True), evaluate=False)
    b = s.Piecewise((s.sqrt(right[2]), condition), (0, True), evaluate=False)
    expected_hash = hash(a)
    names = ("Piecewise", "Tuple", "Unequality", "And")
    previous = {name: core._user_functions.get(name) for name in names}
    try:
        classes = [type(name, (s.Function,), {}) for name in names]
        lookalike = classes[0](right[2], evaluate=False)
        assert a == b
        assert hash(a) == hash(b) == expected_hash
        assert a != lookalike
    finally:
        for name, original in previous.items():
            if original is None:
                core._user_functions.pop(name, None)
            else:
                core._user_functions[name] = original


def test_piecewise_owned_branch_and_condition_arities_are_validated(roots):
    left, _ = roots
    core = importlib.import_module("sympy.core")
    root = _native_expr(left[2])
    invalid_pairs = [
        core._native.py_function("Tuple", root),
        core._native.py_function(
            "Tuple", root, core._native.py_function("Unequality", root),
        ),
        core._native.py_function(
            "Tuple", root, core._native.py_function("Not", root, root),
        ),
    ]
    for pair in invalid_pairs:
        wrapper = object.__new__(s.Expr)
        s.Basic.__dict__["_value"].__set__(
            wrapper, core._native.py_function("Piecewise", pair),
        )
        with pytest.raises(ValueError, match="invalid native Piecewise"):
            hash(wrapper)


def test_piecewise_subclass_native_payload_cannot_erase_its_true_class(roots):
    left, right = roots

    class CustomPiecewise(s.Piecewise):
        pass

    z = s.Symbol("root_identity_piecewise_subclass_z")
    ordinary = s.Piecewise((s.sqrt(left[2]), s.Ne(z, 0)), (0, True), evaluate=False)
    custom = CustomPiecewise((s.sqrt(right[2]), s.Ne(z, 0)), (0, True), evaluate=False)
    for operation in (lambda: ordinary == custom, lambda: custom == ordinary,
                      lambda: hash(custom)):
        with pytest.raises(NotImplementedError, match="retained arguments"):
            operation()


def test_modified_bound_root_identity_refuses_without_running_hooks(roots, monkeypatch):
    left, right = roots
    a, b = s.sqrt(left[2]), s.sqrt(right[2])

    def forbidden(*args, **kwargs):
        raise AssertionError("modified root identity hook must not execute")

    for name in ("__eq__", "__hash__"):
        with monkeypatch.context() as patch:
            patch.setattr(s.CRootOf, name, forbidden)
            for operation in (lambda: a == b, lambda: hash(a)):
                with pytest.raises(NotImplementedError, match="custom or opaque"):
                    operation()



