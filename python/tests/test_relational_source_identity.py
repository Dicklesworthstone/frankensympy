"""Retained relation sides and numerical admission without class-name lifting."""

import copy
import pickle

import pytest
import sympy as s

from sympy.core import Basic, FunctionClass, Relational, _native, _native_expr


RELATIONS = (s.Eq, s.Ne, s.Lt, s.Le, s.Gt, s.Ge)


def _builtin_relational_observations():
    """Portable exact observations, also exercised under SymPy 1.14.0."""
    x = s.Symbol("x")
    observations = {}
    for constructor in (s.Eq, s.Ne, s.Lt, s.Le, s.Gt, s.Ge):
        relation = constructor(x, 2, evaluate=False)
        duplicate = constructor(x, 2, evaluate=False)
        retained = relation.args
        observations[constructor.__name__] = {
            "class": type(relation).__name__,
            "text": str(relation),
            "repr": repr(relation),
            "args": [str(value) for value in retained],
            "lhs_is_original": relation.lhs is x,
            "rhs_is_arg": relation.rhs is retained[1],
            "stable_args": relation.args is retained,
            "equal_duplicate": relation == duplicate,
            "equal_hash": hash(relation) == hash(duplicate),
            "mapping_lookup": {relation: "same"}[duplicate],
            "held_literal_class": type(constructor(1, 2, evaluate=False)).__name__,
            "decided": [bool(constructor(1, 2)), bool(constructor(2, 1)),
                        bool(constructor(2, 2))],
        }
    return observations


@pytest.mark.parametrize("constructor", RELATIONS)
def test_relations_retain_exact_sides_across_same_name_registration(constructor):
    x = s.Symbol("relation_original_x")
    first = FunctionClass("RelationOriginal", (s.Function,), {"__module__": __name__})
    original = first(x, evaluate=False)
    relation = constructor(original, 2, evaluate=False)
    calls = []

    def hostile_setattr(self, name, value):
        calls.append(name)
        raise AssertionError("reading a relation must not reconstruct a custom class")

    second = FunctionClass("RelationOriginal", (s.Function,), {
        "__module__": __name__, "__setattr__": hostile_setattr,
    })
    retained = relation.args
    for _ in range(3):
        assert relation.args is retained
        assert relation.lhs is original and relation.args[0] is original
        assert relation.rhs is retained[1] and relation.rhs == 2
    assert type(relation.lhs) is first and type(relation.lhs) is not second
    assert calls == []
    duplicate = constructor(original, 2, evaluate=False)
    assert relation == duplicate and hash(relation) == hash(duplicate)
    with pytest.raises(NotImplementedError, match="opaque"):
        _native_expr(original)
    other = object.__new__(second)
    Basic.__dict__["_value"].__set__(other, None)
    s.Function.__dict__["_args"].__set__(other, (x,))
    assert relation != constructor(other, 2, evaluate=False)
    assert calls == []


def test_builtin_relational_observations_match_pinned_oracle():
    text_and_decisions = {
        "Equality": ("Eq(x, 2)", [False, False, True]),
        "Unequality": ("Ne(x, 2)", [True, True, False]),
        "StrictLessThan": ("x < 2", [True, False, False]),
        "LessThan": ("x <= 2", [True, False, True]),
        "StrictGreaterThan": ("x > 2", [False, True, False]),
        "GreaterThan": ("x >= 2", [False, True, True]),
    }
    expected = {
        name: {
            "class": name, "text": text, "repr": text, "args": ["x", "2"],
            "lhs_is_original": True, "rhs_is_arg": True, "stable_args": True,
            "equal_duplicate": True, "equal_hash": True, "mapping_lookup": "same",
            "held_literal_class": name, "decided": decisions,
        }
        for name, (text, decisions) in text_and_decisions.items()
    }
    assert _builtin_relational_observations() == expected


@pytest.mark.parametrize("constructor", RELATIONS)
def test_relational_copy_pickle_and_typed_symbol_order_remain_structural(constructor):
    x = s.Symbol("relation_typed_x", positive=True)
    y = s.Dummy("relation_typed_x")
    relation = constructor(x, y, evaluate=False)
    assert relation.lhs is x and relation.rhs is y
    for restored in (copy.copy(relation), copy.deepcopy(relation), pickle.loads(pickle.dumps(relation))):
        assert type(restored) is constructor and restored == relation
        assert restored.args == (x, y)
        assert hash(restored) == hash(relation)


def test_relational_storage_and_lhs_rhs_bypass_shadowed_slots_and_public_args():
    x, y = s.symbols("relation_shadow_x relation_shadow_y")
    calls = []

    class ShadowedEq(s.Eq):
        @property
        def _relational_sides(self):
            raise AssertionError("do not dispatch a user slot getter")

        @_relational_sides.setter
        def _relational_sides(self, value):
            raise AssertionError("do not dispatch a user slot setter")

        @property
        def args(self):
            calls.append("args")
            return (s.Integer(99),)

        def __setattr__(self, name, value):
            calls.append(name)
            object.__setattr__(self, name, value)

    relation = ShadowedEq(x, y, evaluate=False)
    # Keep the existing constructor hook. Subsequent access must neither
    # reconstruct an object nor dispatch through shadowed side/args getters.
    assert calls == ["_value"]
    calls.clear()
    assert relation.lhs is x and relation.rhs is y
    assert calls == []
    assert relation.args == (99,) and calls == ["args"]
    assert Relational.__dict__["args"].__get__(relation, Relational) == (x, y)


def test_hybrid_equation_uses_original_classes_and_never_public_side_properties(monkeypatch):
    x = s.Symbol("hybrid_equation_original_x")
    first = FunctionClass("EquationOriginal", (s.Function,), {"__module__": __name__})
    original = first(x, evaluate=False)
    equation = s.Eq(original, 1, evaluate=False)
    effects = []

    def hostile_setattr(self, name, value):
        effects.append(name)
        raise AssertionError("class-name reconstruction escaped numerical admission")

    second = FunctionClass("EquationOriginal", (s.Function,), {
        "__module__": __name__, "__setattr__": hostile_setattr,
    })

    def forbidden(self):
        raise AssertionError("hybrid admission must read the trusted retained pair")

    for name in ("lhs", "rhs", "args"):
        monkeypatch.setattr(s.Eq, name, property(forbidden))
    calls = []
    program = s.compile_hybrid_residual_system(
        equation, x, jacobian=[[1]], callbacks={
            first: lambda value: calls.append("original") or 10,
            second: lambda value: calls.append("replacement") or 20,
        },
    )
    assert effects == calls == []
    assert program.callback_slots[0][1] is first
    assert program.residuals(3) == [9]
    assert calls == ["original"] and effects == []
    with pytest.raises(NotImplementedError, match="exact-class"):
        s.compile_hybrid_residual_system(equation, x, jacobian=[[1]], callbacks={second: lambda value: 20})
    assert effects == []


def test_equation_callbacks_keep_left_before_right_order_and_fresh_occurrences():
    x = s.Symbol("hybrid_equation_order_x")

    class Law(s.Function):
        pass

    calls = []
    equation = s.Eq(Law(x, evaluate=False), Law(x, evaluate=False), evaluate=False)
    program = s.compile_hybrid_residual_system(
        equation, x, jacobian=[[Law(x, evaluate=False)]],
        callbacks={Law: lambda value: calls.append(value) or len(calls)},
    )
    assert program.evaluate(3) == ([-1], [[3]])
    assert calls == [3.0, 3.0, 3.0]
    assert program.residuals(4) == [-1] and calls[-2:] == [4.0, 4.0]


@pytest.mark.parametrize("name", ["ln", "abs", "sqrt"])
def test_retained_custom_intrinsic_alias_requires_its_exact_callback(name):
    x = s.Symbol("hybrid_equation_alias_x")
    custom = FunctionClass(name, (s.Function,), {"__module__": __name__})
    node = custom(x, evaluate=False)
    equation = s.Eq(node, 1, evaluate=False)
    assert equation.lhs is node
    with pytest.raises(NotImplementedError, match="exact-class"):
        s.compile_hybrid_residual_system(equation, x, jacobian=[[1]], callbacks={})
    calls = []
    program = s.compile_hybrid_residual_system(
        equation, x, jacobian=[[1]],
        callbacks={custom: lambda value: calls.append(value) or 7},
    )
    assert calls == [] and program.residuals(-4) == [6]
    assert calls == [-4.0] and program.callback_slots[0][1] is custom


@pytest.mark.parametrize("backend", ["python", "frankennumpy", "frankenscipy"])
@pytest.mark.parametrize("name", ["ln", "abs", "sqrt"])
def test_certified_solver_refuses_retained_custom_alias_before_candidate_work(name, backend):
    x = s.Symbol("certified_equation_alias_x")
    effects = []

    def record_setattr(self, key, value):
        effects.append(key)
        object.__setattr__(self, key, value)

    custom = FunctionClass(name, (s.Function,), {
        "__module__": __name__, "__setattr__": record_setattr,
    })
    equation = s.Eq(custom(x, evaluate=False), 0, evaluate=False)
    effects.clear()
    with pytest.raises(NotImplementedError, match="custom or opaque"):
        s.nsolve_certified(equation, x, 1, backend=backend)
    assert effects == []


def _native_only_equation(payload):
    equation = object.__new__(s.Eq)
    Basic.__dict__["_value"].__set__(equation, payload)
    return equation


@pytest.mark.parametrize("name", ["UnretainedLaw", "sin", "ln", "abs", "sqrt", "__fsymFloat"])
@pytest.mark.parametrize("bind_callback", [False, True])
def test_native_only_function_equations_refuse_without_class_reconstruction(name, bind_callback):
    x = s.Symbol("hybrid_equation_native_x")
    effects = []

    def forbidden(self, key, value):
        effects.append(key)
        raise AssertionError("native-only equations must not reconstruct custom objects")

    custom = FunctionClass(name, (s.Function,), {"__module__": __name__, "__setattr__": forbidden})
    side = _native.py_function(name, _native_expr(x))
    equation = _native_only_equation(_native.py_function("Equality", side, _native.py_integer(0)))
    callbacks = {custom: lambda value: 12} if bind_callback else {}
    with pytest.raises(NotImplementedError, match="retained Python sides"):
        s.compile_hybrid_residual_system(equation, x, jacobian=[[1]], callbacks=callbacks)
    assert effects == []


@pytest.mark.parametrize("wrapper", [lambda value: value, s.Expr])
def test_retaining_a_generic_native_side_does_not_recover_erased_function_identity(wrapper):
    x = s.Symbol("hybrid_equation_generic_x")
    opaque = wrapper(_native.py_function("sin", _native_expr(x)))
    equation = s.Eq(opaque, 0, evaluate=False)
    with pytest.raises(NotImplementedError, match="retained Python sides"):
        s.compile_hybrid_residual_system(equation, x, jacobian=[[1]], callbacks={})


@pytest.mark.parametrize("nest", [
    lambda value: s.Add(value, 1, evaluate=False),
    lambda value: s.Mul(value, 2, evaluate=False),
    lambda value: s.Pow(value, 2, evaluate=False),
])
def test_held_arithmetic_cannot_hide_an_already_erased_native_function_side(nest):
    x = s.Symbol("hybrid_equation_nested_erasure_x")
    opaque = s.Expr(_native.py_function("sin", _native_expr(x)))
    equation = s.Eq(nest(opaque), 0, evaluate=False)
    with pytest.raises(NotImplementedError, match="retained Python sides"):
        s.compile_hybrid_residual_system(equation, x, jacobian=[[1]], callbacks={})


def test_equation_with_held_arithmetic_retains_actual_builtin_function_children():
    x = s.Symbol("hybrid_equation_held_builtin_x")
    side = s.Add(s.sin(x), s.Mul(2, s.exp(x), evaluate=False), evaluate=False)
    equation = s.Eq(side, 2, evaluate=False)
    program = s.compile_hybrid_residual_system(equation, x, jacobian=[[1]], callbacks={})
    assert program.residuals(0) == [0]


@pytest.mark.parametrize("tag,arity", [("Equality", 0), ("Equality", 1), ("Equality", 3), ("LessThan", 2), ("Other", 2)])
def test_native_only_equation_validates_tag_and_arity_before_lowering(tag, arity):
    x = s.Symbol("hybrid_equation_malformed_x")
    equation = _native_only_equation(_native.py_function(tag, *([_native_expr(x)] * arity)))
    with pytest.raises(ValueError, match="equality|two sides"):
        s.compile_hybrid_residual_system(equation, x, jacobian=[[1]], callbacks={})


def test_native_only_equation_borrowed_preflight_precedes_side_traversal(monkeypatch):
    import sympy.utilities.hybrid_system as hybrid_system

    x = s.Symbol("hybrid_equation_preflight_x")
    equation = _native_only_equation(_native.py_function(
        "Equality", _native.py_integer(1 << 4096), _native_expr(x),
    ))

    def forbidden(*args):
        raise AssertionError("borrowed whole-source admission must run first")

    monkeypatch.setattr(hybrid_system, "_arithmetic_equation_side", forbidden)
    with pytest.raises(ValueError, match="literal|bit"):
        s.compile_hybrid_residual_system(equation, x, jacobian=[[1]], callbacks={})


def test_corrupt_retained_equation_pair_refuses_instead_of_lifting_native_payload():
    x = s.Symbol("hybrid_equation_corrupt_x")
    for sides in ([x, 0], (), (x,), (x, 0, 1)):
        equation = s.Eq(x, 0, evaluate=False)
        Relational.__dict__["_relational_sides"].__set__(equation, sides)
        with pytest.raises(ValueError, match="exact captured pair"):
            s.compile_hybrid_residual_system(equation, x, jacobian=[[1]], callbacks={})


@pytest.mark.parametrize("tag", ["Equality", "Eq"])
def test_native_only_arithmetic_equation_keeps_original_domain_and_typed_axes(tag):
    x = s.Symbol("hybrid_equation_arithmetic_x", positive=True)
    y = s.Dummy("hybrid_equation_arithmetic_x")
    native_x, native_y = _native_expr(x), _native_expr(y)
    left = _native.Add(native_x, native_y, evaluate=False).as_expr()
    equation = _native_only_equation(_native.py_function(tag, left, _native.py_integer(3)))
    program = s.compile_hybrid_residual_system(equation, (x, y), jacobian=[[1, 1]], callbacks={})
    assert program.variables[0] is x and program.variables[1] is y
    assert program.evaluate(1, 2) == ([0], [[1, 1]])
    reciprocal = _native.Pow(native_x, _native.py_integer(-1), evaluate=False).as_expr()
    hidden_pole = _native.Mul(_native.py_integer(0), reciprocal, evaluate=False).as_expr()
    equation = _native_only_equation(_native.py_function(tag, hidden_pole, _native.py_integer(0)))
    program = s.compile_hybrid_residual_system(equation, x, jacobian=[[0]], callbacks={})
    assert program.residuals(2) == [0]
    with pytest.raises(ValueError):
        program.residuals(0)


def test_retained_builtin_equations_keep_constants_float_literals_and_domain_guards():
    x = s.Symbol("hybrid_equation_builtins_x")
    equations = [s.Eq(s.sin(x), 0, evaluate=False), s.Eq(s.exp(x), 1, evaluate=False),
                 s.Eq(s.log(x), 0, evaluate=False), s.Eq(x, s.Float(1.25), evaluate=False),
                 s.Eq(x, s.pi, evaluate=False), s.Eq(x, s.E, evaluate=False)]
    program = s.compile_hybrid_residual_system(equations, x, jacobian=[[1]] * len(equations), callbacks={})
    import math
    assert program.residuals(1) == pytest.approx([
        math.sin(1), math.exp(1) - 1, 0, -0.25, 1 - math.pi, 1 - math.e,
    ])
    with pytest.raises(ValueError):
        program.residuals(0)
    equation = s.Eq(s.Pow(s.Pow(x, -1, evaluate=False), 0, evaluate=False), 1, evaluate=False)
    program = s.compile_hybrid_residual_system(equation, x, jacobian=[[0]], callbacks={})
    assert program.residuals(2) == [0]
    with pytest.raises(ValueError):
        program.residuals(0)
