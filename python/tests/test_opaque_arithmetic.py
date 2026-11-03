"""Ordinary arithmetic keeps exact Python function identities and effects."""

from __future__ import annotations

import gc
import weakref

import pytest
import sympy as s

from sympy.core import Basic, FunctionClass, _native_expr


def _same_named():
    first = FunctionClass("OpaqueSame", (s.Function,), {"__module__": __name__})
    second = FunctionClass("OpaqueSame", (s.Function,), {"__module__": __name__})
    return first, second


def _observations():
    """Portable structural observations also executed with SymPy 1.14.0."""
    x, y = s.symbols("opaque_x opaque_y")
    first, second = _same_named()
    a, b = first(x), second(x)

    def tree(value):
        if type(value) is first:
            return ["first", *(tree(child) for child in value.args)]
        if type(value) is second:
            return ["second", *(tree(child) for child in value.args)]
        return ([type(value).__name__, *(tree(child) for child in value.args)]
                if value.args else str(value))

    cases = {
        "sum": a + b,
        "reverse_sum": b + a,
        "difference": a - b,
        "reverse_difference": b - a,
        "same_application": a + first(x),
        "collected_coefficients": 2 * a + 3 * first(x),
        "different_arguments": first(y) + a,
        "symbol_and_application": x + a,
        "power_and_application": x**2 + a,
        "application_square": a**2 + a,
        "product": a * b,
        "nested_product": (2 * a) * b,
        "flat_product": s.Mul(2, a, b),
        "nested_sum": (1 + a) + b,
        "flat_sum": s.Add(1, a, b),
        "quotient": a / b,
        "reciprocal": 1 / a,
        "distribution": 2 * (a + 1),
        "outer_builtin": s.sin(a) + s.cos(b),
        "held_sum": s.Add(a, b, evaluate=False),
        "held_product": s.Mul(b, a, evaluate=False),
        "held_power": s.Pow(a, b, evaluate=False),
    }
    return {
        "trees": {name: tree(value) for name, value in cases.items()},
        "equalities": {
            "class_distinction": a != b,
            "sum_distinction": a + 1 != b + 1,
            "held_distinction": s.Add(a, 1, evaluate=False) != s.Add(b, 1, evaluate=False),
            "equal_application": a == first(x),
            "equal_hash": hash(a) == hash(first(x)),
            "equal_sum_hash": hash(a + 1) == hash(first(x) + 1),
            "same_name_hash": hash(a) == hash(b),
            "mapping_lookup": {a + 1: 7}[first(x) + 1] == 7,
            "held_canonical_equal": s.Add(1, a, evaluate=False) == a + 1,
            "held_canonical_hash": hash(s.Add(1, a, evaluate=False)) == hash(a + 1),
            "reverse_order_distinct": a + b != b + a,
        },
        "reuse": {
            "add_zero": a + 0 is a,
            "reverse_add_zero": 0 + a is a,
            "subtract_zero": a - 0 is a,
            "multiply_one": a * 1 is a,
            "reverse_multiply_one": 1 * a is a,
            "divide_one": a / 1 is a,
            "unit_power": a**1 is a,
            "constructor_add": s.Add(a, 0) is a,
            "constructor_mul": s.Mul(a, 1) is a,
        },
    }


def test_same_name_classes_are_never_merged_by_arithmetic():
    x = s.Symbol("opaque_identity_x")
    first, second = _same_named()
    a, b = first(x), second(x)
    assert a + b != 2 * a
    assert a + b != 2 * b
    assert a - b != 0
    assert (a + b).args == (a, b)
    assert (b + a).args == (b, a)
    assert len({a + 1: "first", b + 1: "second"}) == 2
    assert _observations()["equalities"] == {key: True for key in _observations()["equalities"]}
    assert all(_observations()["reuse"].values())


@pytest.mark.parametrize("builder", [
    lambda a, b: a + b,
    lambda a, b: a - b,
    lambda a, b: a * b,
    lambda a, b: a / b,
    lambda a, b: a**b,
    lambda a, b: s.Add(a, b, evaluate=False),
    lambda a, b: s.Mul(a, b, evaluate=False),
    lambda a, b: s.Pow(a, b, evaluate=False),
])
def test_closed_native_admission_refuses_every_retained_opaque_operator(builder):
    x = s.Symbol("opaque_closed_x")
    first, second = _same_named()
    source = builder(first(x), second(x))
    assert Basic.__dict__["_value"].__get__(source, Basic) is None
    with pytest.raises(NotImplementedError, match="opaque"):
        _native_expr(source)
    with pytest.raises((NotImplementedError, ValueError, TypeError)):
        s.compile_numeric(x, source)
    with pytest.raises((NotImplementedError, ValueError, TypeError)):
        s.compile_residual_system([source], [x])


def test_normal_arithmetic_reaches_distinct_hybrid_callbacks():
    x = s.Symbol("opaque_numeric_x")
    first, second = _same_named()
    calls = []
    source = first(x) - second(x) + 1
    program = s.compile_numeric_hybrid(x, source, callbacks={
        first: lambda value: calls.append(("first", value)) or value + 10,
        second: lambda value: calls.append(("second", value)) or value * 2,
    })
    assert calls == []
    assert program(3) == 8
    assert calls == [("second", 3.0), ("first", 3.0)]
    assert program.evidence == "approximate_with_user_callbacks"


def test_same_named_builtin_shadow_does_not_change_either_application():
    x = s.Symbol("opaque_builtin_x")
    custom = FunctionClass("sin", (s.Function,), {"__module__": __name__})
    node = custom(x)
    source = node + s.sin(x)
    assert any(child is node for child in source.args)
    program = s.compile_numeric_hybrid(x, source, callbacks={custom: lambda value: 5})
    assert program(0) == 5
    assert s.compile_numeric(x, s.sin(x))(0) == 0


def test_normalization_does_not_reconstruct_custom_objects_or_read_their_hooks():
    state = {"armed": False, "calls": []}

    def guard(name):
        if state["armed"]:
            state["calls"].append(name)
            raise AssertionError("opaque normalization invoked " + name)

    class Guarded(s.Function):
        def __new__(cls, *args, **kwargs):
            guard("new")
            return super().__new__(cls, *args, **kwargs)

        def __setattr__(self, name, value):
            guard("setattr")
            object.__setattr__(self, name, value)

        @property
        def args(self):
            guard("args")
            return super().args

        def __str__(self):
            guard("str")
            return super().__str__()

    x = s.Symbol("opaque_guard_x")
    node = Guarded(x, evaluate=False)
    state["armed"] = True
    try:
        result = 2 * (node + 1) - node
        assert any(child is node for child in result.args)
        program = s.compile_numeric_hybrid(x, result, callbacks={Guarded: lambda value: 4})
        assert program(2) == 6
        assert state["calls"] == []
    finally:
        state["armed"] = False


def test_native_slot_spelling_cannot_capture_a_user_symbol():
    x = s.Symbol("__fsym_operation_slot_00000000")
    cls, _ = _same_named()
    node = cls(x)
    source = node + x
    assert source.free_symbols == {x}
    assert source.args[0] is x
    assert source.args[1] is node
    assert s.compile_numeric_hybrid(x, source, callbacks={cls: lambda value: 2 * value})(3) == 9


def test_operation_slots_do_not_retain_custom_classes_after_graph_collection():
    def build():
        cls = FunctionClass("OpaqueCollectible", (s.Function,), {"__module__": __name__})
        node = cls(s.Symbol("opaque_gc_x"))
        graph = (node + 1) * (node + 2)
        return graph, weakref.ref(node), weakref.ref(cls)

    graph, node_ref, class_ref = build()
    assert node_ref() is not None
    del graph
    gc.collect()
    assert node_ref() is None
    assert class_ref() is None


def test_cyclic_retained_graph_is_rejected_before_normalization():
    from sympy.core._opaque_arithmetic import contains_opaque

    cls, _ = _same_named()
    node = cls(s.Symbol("opaque_cycle_x"))
    s.Function.__dict__["_args"].__set__(node, (node,))
    with pytest.raises(ValueError, match="cyclic"):
        contains_opaque(node)
    with pytest.raises(ValueError, match="cyclic"):
        node + 1


def test_explicit_noncommutative_functions_are_not_normalized_as_commutative():
    class Noncommutative(s.Function):
        is_commutative = False

    node = Noncommutative(s.Symbol("opaque_noncommutative_x"))
    with pytest.raises(NotImplementedError, match="commutativity"):
        node * node


@pytest.mark.parametrize("hook", ["__eq__", "__hash__", "_hashable_content"])
def test_custom_equality_and_hash_contracts_refuse_without_running_hooks(hook):
    calls = []

    def trap(*args):
        calls.append(hook)
        raise AssertionError("normalization invoked " + hook)

    cls = FunctionClass("OpaqueCustomIdentity", (s.Function,), {
        "__module__": __name__, hook: trap,
    })
    node = cls(s.Symbol("opaque_custom_identity_x"), evaluate=False)
    with pytest.raises(NotImplementedError, match="equality or hashing"):
        node + 1
    with pytest.raises(NotImplementedError, match="equality or hashing"):
        s.sin(node) + 1
    assert calls == []
    # Held source is still a valid explicit callback graph: no algebraic
    # identity or grouping decision is made by that constructor.
    held = s.Add(node, 1, evaluate=False)
    assert s.compile_numeric_hybrid(held.args[0].args[0], held, callbacks={cls: lambda value: 3})(2) == 4
    with pytest.raises(NotImplementedError, match="equality or hashing"):
        hash(held)
    with pytest.raises(NotImplementedError, match="equality or hashing"):
        held == s.Add(node, 1, evaluate=False)
    assert calls == []


@pytest.mark.parametrize("held, canonical", [
    (lambda x: s.Add(1, x, evaluate=False), lambda x: 1 + x),
    (lambda x: s.Mul(2, x, evaluate=False), lambda x: 2 * x),
    (lambda x: s.Pow(x, 2, evaluate=False), lambda x: x**2),
])
def test_function_child_equality_does_not_depend_on_storage_lane(held, canonical):
    x = s.Symbol("opaque_child_equality_x")
    cls, _ = _same_named()
    first, second = cls(held(x)), cls(canonical(x))
    assert first == second
    assert hash(first) == hash(second)
    assert {first: "same"}[second] == "same"
    assert first + second == 2 * first


def test_spoofed_module_and_builtin_name_cannot_admit_a_native_proof():
    from sympy.core import diff_with_receipt

    x = s.Symbol("opaque_module_spoof_x")
    custom = FunctionClass("sin", (s.Function,), {"__module__": "sympy.not_a_builtin"})
    node = custom(x, evaluate=False)
    assert Basic.__dict__["_value"].__get__(node, Basic) is None
    with pytest.raises(NotImplementedError):
        diff_with_receipt(node, x)
    with pytest.raises(NotImplementedError):
        _native_expr(node)
    assert s.compile_numeric(x, s.sin(x))(0) == 0
    assert s.compile_numeric_hybrid(x, node + 1, callbacks={custom: lambda value: 4})(2) == 5


def test_custom_symbol_class_is_not_merged_with_its_plain_namesake():
    class CustomSymbol(s.Symbol):
        pass

    cls, _ = _same_named()
    custom, plain = CustomSymbol("opaque_symbol"), s.Symbol("opaque_symbol")
    first, second = cls(custom), cls(plain)
    with pytest.raises(NotImplementedError, match="exact built-in"):
        first + second
    assert first.args[0] is custom
    assert second.args[0] is plain
    with pytest.raises(NotImplementedError):
        _native_expr(custom)


def test_shared_surface_graph_hashing_does_not_expand_every_occurrence():
    cls, _ = _same_named()
    node = cls(s.Symbol("opaque_shared_x"), evaluate=False)
    for _ in range(50):
        node = cls(node, node, evaluate=False)
    first = hash(node)
    assert hash(node) == first
    assert (node + 1).args[1] is node


def test_later_cyclic_operand_is_checked_even_after_an_opaque_first_operand():
    cls, _ = _same_named()
    first = cls(s.Symbol("opaque_first_x"))
    cyclic = object.__new__(s.Add)
    Basic.__dict__["_value"].__set__(cyclic, None)
    Basic.__dict__["_surface_args"].__set__(cyclic, (cyclic,))
    with pytest.raises(ValueError, match="cyclic"):
        first + cyclic


def test_shell_function_operations_have_no_missing_native_handle_errors():
    cls, _ = _same_named()
    node = cls(s.Symbol("opaque_shell_x"))
    assert abs(node).args == (node,)
    assert s.exp(node).args == (node,)
    assert s.log(node).args == (node,)
    assert node._repr_latex_().startswith("$")
    for operation in (len, iter, lambda value: value[0]):
        with pytest.raises(TypeError):
            operation(node)


def test_unknown_native_function_tags_remain_part_of_retained_child_identity():
    from sympy.core import _native

    x = s.Symbol("opaque_native_tag_x")
    cls, _ = _same_named()
    first_native = _native.py_function("NativeUnknownU", _native_expr(x))
    second_native = _native.py_function("NativeUnknownV", _native_expr(x))
    first = cls(s.Expr(first_native) + 1)
    second = cls(s.Expr(second_native) + 1)
    assert first != second
    assert len({first: "first", second: "second"}) == 2
    held_child = cls(s.Add(1, s.Expr(first_native), evaluate=False))
    assert first == held_child
    assert hash(first) == hash(held_child)
