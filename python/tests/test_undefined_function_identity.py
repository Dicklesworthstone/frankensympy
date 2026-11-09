"""Undefined functions retain their declaration, distinct from native tags.

The portable observations are checked against isolated SymPy 1.14.0. Native
admission assertions additionally ensure that a familiar name grants no
closed-function capability to the new Python application.
"""

from __future__ import annotations

import copy
import pickle

import pytest
import sympy as s


class NamedUserFunction(s.Function):
    @classmethod
    def eval(cls, value):
        if value == 0:
            return s.Integer(17)


class FactorySubclass(s.Function("factory_parent", nargs=1)):
    @classmethod
    def eval(cls, value):
        if value == 0:
            return s.Integer(23)


@pytest.mark.parametrize("name", ["sin", "cos", "exp", "log", "Abs", "acosh"])
def test_factory_names_never_select_native_intrinsics(name):
    x = s.Symbol("undefined_intrinsic_x")
    function = s.Function(name)
    intrinsic = getattr(s, name)
    assert function is not intrinsic
    assert function != intrinsic
    assert function(0).func is function
    assert function(0).args == (s.Integer(0),)
    assert function(x) != intrinsic(x)
    assert function is s.Function(name)
    assert function(x).func(*function(x).args) == function(x)


def test_factory_names_do_not_capture_registered_user_classes():
    x = s.Symbol("undefined_custom_x")
    function = s.Function("NamedUserFunction")
    assert function is not NamedUserFunction
    assert function(0).func is function
    assert NamedUserFunction(0) == 17
    assert function(x) != NamedUserFunction(x)
    assert function is s.Function("NamedUserFunction")


def test_undefined_intrinsic_spelling_keeps_a_formal_derivative():
    x = s.Symbol("undefined_derivative_x")
    function = s.Function("sin")
    node = function(x)
    derivative = s.diff(node, x)
    assert derivative == s.Derivative(node, x)
    assert derivative != s.cos(x)
    assert derivative.expr.func is function


@pytest.mark.parametrize("declaration, accepted", [
    (0, (0,)), (1, (1,)), (2, (2,)), ((1, 2), (1, 2)),
    ((2, 1, 2), (1, 2)), ([1, 2], (1, 2)), (s.Integer(2), (2,)),
])
def test_declared_arity_controls_construction_and_public_nargs(declaration, accepted):
    function = s.Function("arity_function", nargs=declaration)
    x = s.Symbol("arity_x")
    assert function.nargs == s.FiniteSet(*accepted)
    for count in range(4):
        if count in accepted:
            node = function(*([x] * count))
            assert node.args == (x,) * count
            assert node.nargs == function.nargs
        else:
            with pytest.raises(TypeError):
                function(*([x] * count))


def test_arity_declarations_retain_identity_beyond_the_accepted_count_set():
    declarations = [2, (2,), (2, 2), None]
    classes = [s.Function("arity_identity", nargs=value) for value in declarations]
    classes.append(s.Function("arity_identity"))
    assert len(set(classes)) == len(classes)
    for declaration, function in zip(declarations, classes):
        assert function == s.Function("arity_identity", nargs=declaration)
        assert hash(function) == hash(s.Function("arity_identity", nargs=declaration))
    assert classes[-1].nargs == s.S.Naturals0
    assert classes[-2].nargs == s.S.Naturals0
    x = s.Symbol("arity_identity_x")
    assert classes[0](x, x) != classes[1](x, x)


def test_list_arity_retains_the_oracle_unhashable_declaration():
    function = s.Function("list_arity", nargs=[1, 2])
    node = function(s.Symbol("list_arity_x"))
    assert node.nargs == s.FiniteSet(1, 2)
    assert function != s.Function("list_arity", nargs=(1, 2))
    with pytest.raises(TypeError):
        hash(function)
    assert isinstance(hash(node), int)
    assert pickle.loads(pickle.dumps(function)) == function
    with pytest.raises(TypeError):
        pickle.dumps(node)


@pytest.mark.parametrize("value", [True, 1.0, "2", {1, 2}, [], ()])
def test_invalid_arities_fail_before_a_class_is_cached(value):
    with pytest.raises(ValueError):
        s.Function("bad_arity", nargs=value)
    assert s.Function("bad_arity")(1).args == (s.Integer(1),)


@pytest.mark.parametrize("declarations, expected", [
    ({"real": True}, {"real": True, "complex": True, "finite": True,
                      "imaginary": False, "positive": None}),
    ({"real": False}, {"real": False, "integer": False, "positive": False,
                       "zero": False, "complex": None}),
    ({"positive": True}, {"real": True, "finite": True, "positive": True,
                          "negative": False, "zero": False, "nonzero": True}),
    ({"integer": True, "even": False}, {"integer": True, "odd": True,
                                       "nonzero": True, "rational": True}),
    ({"real": True, "nonzero": False}, {"real": True, "zero": True,
                                        "even": True, "positive": False}),
    ({"imaginary": True}, {"imaginary": True, "real": False, "finite": True,
                           "nonzero": False, "zero": False}),
    ({"finite": False}, {"finite": False, "infinite": True, "complex": False,
                         "real": False, "positive": False}),
    ({"commutative": False}, {"commutative": False, "real": False,
                              "complex": False, "integer": False}),
    ({"finite": True, "real": False}, {"extended_real": False, "real": False}),
    ({"extended_real": True, "real": False}, {"infinite": True, "finite": False}),
    ({"finite": True, "positive": False}, {"extended_positive": None, "positive": False}),
    ({"polar": True}, {"polar": True, "real": None}),
])
def test_function_assumptions_remain_attached_to_the_defining_class(declarations, expected):
    x = s.Symbol("function_fact_x")
    function = s.Function("fact_function", **declarations)
    node = function(x)
    for name, value in expected.items():
        assert getattr(node, "is_" + name) is value
    assert node.func is function
    assert node != s.Function("fact_function")(x)
    assert function == s.Function("fact_function", **declarations)


def test_redundant_and_unknown_assumption_declarations_keep_logical_identity():
    x = s.Symbol("function_fact_identity_x")
    positive = s.Function("positive_function", positive=True)
    redundant = s.Function("positive_function", positive=True, real=True, finite=True)
    assert positive is not redundant
    assert positive == redundant
    assert hash(positive) == hash(redundant)
    assert positive(x) == redundant(x)
    assert positive(x) + 1 == redundant(x) + 1
    assert hash(positive(x) + 1) == hash(redundant(x) + 1)
    assert positive(x) - redundant(x) == 0
    assert s.Function("unknown_function", real=None) == s.Function("unknown_function")
    assert s.Function("unknown_function", real=False) != s.Function("unknown_function")


def test_different_function_facts_do_not_merge_same_named_applications():
    x = s.Symbol("function_order_x")
    real = s.Function("ordered_function", real=True)(x)
    imaginary = s.Function("ordered_function", imaginary=True)(x)
    assert real != imaginary
    assert real - imaginary != 0
    assert (real + imaginary).args == (real, imaginary)
    assert (imaginary + real).args == (imaginary, real)
    assert real + imaginary != imaginary + real


@pytest.mark.parametrize("declarations", [
    {"positive": True, "negative": True},
    {"real": True, "imaginary": True},
    {"finite": True, "infinite": True},
    {"even": True, "odd": True},
    {"commutative": False, "integer": True},
])
def test_inconsistent_function_declarations_cannot_publish_a_class(declarations):
    with pytest.raises(ValueError):
        s.Function("inconsistent_function", **declarations)
    assert s.Function("inconsistent_function")(1).func is s.Function("inconsistent_function")


def test_immutable_user_metadata_is_preserved_and_contributes_to_identity():
    first = s.Function("metadata_function", label="left", dimensions=(2, 3))
    second = s.Function("metadata_function", label="right", dimensions=(2, 3))
    assert first.label == "left"
    assert first.dimensions == (2, 3)
    assert first != second
    assert first == s.Function("metadata_function", dimensions=(2, 3), label="left")


@pytest.mark.parametrize("ingress", ["name", "option"])
def test_name_subclasses_cannot_run_hooks_during_later_structural_operations(ingress):
    """The bounded shell excludes callbacks the upstream oracle permits."""
    calls = []

    class Name(str):
        armed = False

        def __hash__(self):
            if self.armed:
                calls.append("hash")
            return str.__hash__(self)

        def __eq__(self, other):
            if self.armed:
                calls.append("eq")
            return str.__eq__(self, other)

    name = Name("hook_safe_name" if ingress == "name" else "label")
    function = (s.Function(name) if ingress == "name" else
                s.Function("hook_safe_option", **{name: "metadata"}))
    name.armed = True
    x = s.Symbol("hook_safe_x")
    first, second = function(x), function(x + 1)
    assert first != second
    assert first + 1 == 1 + first
    assert hash(first) == hash(function(x))
    assert hash(first + 1) == hash(1 + first)
    assert calls == []


@pytest.mark.parametrize("options", [
    {}, {"nargs": 2}, {"nargs": (1, 2)}, {"real": True},
    {"positive": True, "nargs": 2}, {"label": "snapshot", "dimensions": (2, 3)},
])
def test_pickle_and_copy_preserve_undefined_function_declarations(options):
    function = s.Function("pickle_function", **options)
    x, y = s.symbols("pickle_x pickle_y")
    node = function(x, y)
    for protocol in (2, pickle.HIGHEST_PROTOCOL):
        restored_class = pickle.loads(pickle.dumps(function, protocol=protocol))
        restored = pickle.loads(pickle.dumps(node, protocol=protocol))
        assert restored_class == function
        if "real" in options or "positive" in options:
            # The oracle pickles the closed assumption declaration, so this
            # is an equal class from a distinct factory cache key.
            assert restored_class is not function
        else:
            assert restored_class is function
        assert restored.func == function
        assert restored == node
        assert restored.args == node.args
        assert hash(restored) == hash(node)
        assert restored.nargs == node.nargs
        assert restored.is_real is node.is_real
    assert copy.copy(node) == node
    assert copy.deepcopy(node) == node


def test_factory_subclasses_remain_ordinary_python_classes():
    x = s.Symbol("factory_subclass_x")
    assert FactorySubclass(0) == 23
    node = FactorySubclass(x)
    assert node.func is FactorySubclass
    assert FactorySubclass is not s.Function("FactorySubclass")
    with pytest.raises(TypeError):
        FactorySubclass(x, x)


def test_factory_subclass_body_can_replace_the_inherited_arity():
    class Unary(s.Function("binary_parent", nargs=2)):
        nargs = 1

    x = s.Symbol("factory_subclass_arity_x")
    assert Unary(x).args == (x,)
    assert Unary.nargs == s.FiniteSet(1)
    assert Unary(x).nargs == Unary.nargs
    with pytest.raises(TypeError):
        Unary(x, x)


def test_undefined_function_custom_metaclasses_preserve_arity_dispatch():
    from sympy.core.function import UndefinedFunction

    class CustomMeta(UndefinedFunction):
        pass

    class Unconstrained(s.Function("custom_meta_parent"), metaclass=CustomMeta):
        pass

    class Bounded(Unconstrained):
        nargs = (1, 2)

    x = s.Symbol("custom_meta_x")
    assert Unconstrained.nargs == s.S.Naturals0
    for count in (0, 1, 3):
        node = Unconstrained(*([x] * count))
        assert node.func is Unconstrained
        assert node.args == (x,) * count
    assert Bounded.nargs == s.FiniteSet(1, 2)
    for count in (1, 2):
        assert Bounded(*([x] * count)).args == (x,) * count
    for count in (0, 3):
        with pytest.raises(TypeError):
            Bounded(*([x] * count))


def test_native_capabilities_cannot_be_acquired_from_an_undefined_name_or_fact():
    from sympy.core import Basic, _native_expr

    x = s.Symbol("undefined_admission_x")
    for function in (s.Function("sin"), s.Function("exp", real=True),
                     s.Function("log", positive=True, nargs=1)):
        node = function(x)
        assert Basic.__dict__["_value"].__get__(node, Basic) is None
        with pytest.raises(NotImplementedError):
            _native_expr(node)
        with pytest.raises((NotImplementedError, TypeError, ValueError)):
            s.compile_numeric(x, node)
        assert (node + s.sin(x)).has(node)
    assert s.compile_numeric(x, s.sin(x))(0) == 0


def test_copied_factory_metadata_does_not_capture_a_custom_class_identity():
    from sympy.core import UndefinedFunction, _native_expr

    x = s.Symbol("factory_spoof_x")
    factory = s.Function("factory_spoof", real=True)
    custom = UndefinedFunction("factory_spoof", (s.Function,), {
        "__module__": __name__,
        "_undefined_function_identity": factory.__dict__["_undefined_function_identity"],
    })
    assert custom is not factory
    assert custom(x) != factory(x)
    assert custom(x) - factory(x) != 0
    with pytest.raises(NotImplementedError):
        _native_expr(custom(x))


def test_noncommutative_functions_refuse_commutative_native_slotting():
    from sympy.core import _native_expr

    x = s.Symbol("undefined_noncommutative_x")
    node = s.Function("noncommuting_function", commutative=False)(x)
    assert node.is_commutative is False
    with pytest.raises(NotImplementedError):
        _native_expr(node)
    with pytest.raises(NotImplementedError):
        node + 1


def test_declaration_metadata_limits_refuse_without_executing_user_hooks():
    class Hostile:
        def __hash__(self):
            raise AssertionError("metadata hash executed")

        def __bool__(self):
            raise AssertionError("metadata truth conversion executed")

    with pytest.raises(NotImplementedError):
        s.Function("hostile_metadata", label=Hostile())
    with pytest.raises(TypeError):
        s.Function("hostile_assumption", real=Hostile())
    with pytest.raises(ValueError):
        s.Function("oversized_arity", nargs=tuple(range(65)))
