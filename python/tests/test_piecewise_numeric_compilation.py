"""Lazy, closed Piecewise compilation against the actual native extension."""

import math

import pytest
import sympy as s

from sympy.core import Basic, _native, _native_expr


BACKENDS = ("math", "frankensympy")


@pytest.mark.parametrize("backend", BACKENDS)
def test_only_selected_piecewise_value_is_evaluated(backend):
    x = s.Symbol("piecewise_x")
    expression = s.Piecewise(
        (s.log(x), x > 0), (s.sqrt(-x), x < 0), (s.Integer(0), True),
    )
    function = s.lambdify(x, expression, modules=backend)
    assert function(-4) == 2.0
    assert function(0) == 0
    assert function(math.e) == pytest.approx(1)
    reciprocal = s.lambdify(x, s.Piecewise((1/x, s.Ne(x, 0)), (0, True)), modules=backend)
    assert reciprocal(0) == 0
    assert reciprocal(-2) == -.5


@pytest.mark.parametrize("backend", BACKENDS)
def test_source_order_and_nested_piecewise_inside_arithmetic(backend):
    x = s.Symbol("piecewise_order")
    overlap = s.Piecewise((1, x < 2), (2, x < 4), (3, True))
    nested = s.Piecewise((s.Piecewise((s.sqrt(x), x > 1), (7, True)), x > 0), (9, True))
    function = s.lambdify(x, [overlap + 10, (nested, 2 * overlap)], modules=backend)
    assert function(-1) == [11, (9, 2)]
    assert function(.5) == [11, (7, 2)]
    assert function(4) == [13, (2, 6)]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("relation,values", [
    (s.Eq, [False, True, False]), (s.Ne, [True, False, True]),
    (s.Lt, [True, False, False]), (s.Le, [True, True, False]),
    (s.Gt, [False, False, True]), (s.Ge, [False, True, True]),
])
def test_all_comparison_boundaries(backend, relation, values):
    x = s.Symbol("piecewise_boundary")
    expression = s.Piecewise((11, relation(x, 0)), (22, True))
    function = s.lambdify(x, expression, modules=backend)
    assert [function(value) for value in (-1, 0, 1)] == [11 if value else 22 for value in values]


@pytest.mark.parametrize("backend", BACKENDS)
def test_and_or_not_conditions_short_circuit_domain_failures(backend):
    x = s.Symbol("piecewise_boolean")
    # Operand order is the shell's deterministic Boolean argument order; it
    # agrees with the pinned SymPy math printer for these inequalities.
    conjunction = s.And(x < 0, s.log(-x) < 1)
    disjunction = s.Or(x < 0, s.log(x) < -1)
    first = s.lambdify(x, s.Piecewise((1, conjunction), (2, True)), modules=backend)
    second = s.lambdify(x, s.Piecewise((1, disjunction), (2, True)), modules=backend)
    assert first(0) == 2
    assert first(-1) == 1
    assert second(-1) == 1
    assert second(1) == 2
    negation = s.Not(s.And(x < 0, x > -2), evaluate=False)
    third = s.lambdify(x, s.Piecewise((1, negation), (2, True)), modules=backend)
    assert [third(value) for value in (-3, -1, 1)] == [1, 2, 1]


def test_math_no_match_matches_sympy_114_and_native_refuses():
    x = s.Symbol("piecewise_no_match")
    expression = s.Piecewise((x + 1, x > 0))
    assert s.lambdify(x, expression, modules="math")(-1) is None
    native = s.compile_numeric(x, expression)
    with pytest.raises(ValueError, match="[Pp]iecewise|branch"):
        native(-1)
    assert native(2) == 3.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_condition_coordinates_keep_assumptions_and_dummy_identity(backend):
    plain = s.Symbol("piecewise_shared")
    positive = s.Symbol("piecewise_shared", positive=True)
    first, second = s.Dummy("piecewise_dummy"), s.Dummy("piecewise_dummy")
    expression = s.Piecewise((first, plain < positive), (second, True))
    function = s.lambdify((plain, positive, first, second), expression, modules=backend)
    assert function(1, 2, 7, 9) == 7
    assert function(3, 2, 7, 9) == 9
    with pytest.raises(ValueError):
        s.lambdify((plain, first, second), expression, modules=backend)


def test_piecewise_matrix_snapshots_and_batches():
    x = s.Symbol("piecewise_matrix")
    expression = s.Piecewise((1/x, s.Ne(x, 0)), (0, True))
    matrix = s.Matrix([[expression, 2]])
    function = s.compile_numeric(x, matrix)
    matrix[0, 0] = 99
    assert function(0) == [[0.0, 2.0]]
    assert function(-2) == [[-.5, 2.0]]
    assert function.evaluate_batch([(0,), (2,)]) == [[[0.0, 2.0]], [[.5, 2.0]]]


@pytest.mark.parametrize("backend", BACKENDS)
def test_held_false_and_true_branches_remain_lazy(backend):
    x = s.Symbol("piecewise_held")
    expression = s.Piecewise((1/x, False), (5, True), (s.log(x), True), evaluate=False)
    function = s.lambdify(x, expression, modules=backend)
    assert function(0) == 5
    assert len(expression.args) == 3


@pytest.mark.parametrize("backend", BACKENDS)
def test_native_piecewise_does_not_reconstruct_condition_classes(backend, monkeypatch):
    x = s.Symbol("piecewise_native")
    expression = s.Piecewise((x, x > 0), (-x, True))
    raw = _native_expr(expression)
    effects = []

    def forbidden(cls, *args, **kwargs):
        effects.append("new")
        raise AssertionError("numeric admission must not call a modified constructor")

    monkeypatch.setattr(s.Gt, "__new__", forbidden)
    for source in (raw, s.Expr(raw), _native.Add(raw, _native.py_integer(1), evaluate=False).as_expr()):
        with pytest.raises(NotImplementedError):
            s.lambdify(x, source, modules=backend)
    assert effects == []


@pytest.mark.parametrize("backend", BACKENDS)
def test_custom_and_mutated_conditions_are_refused_without_callbacks(backend, monkeypatch):
    x = s.Symbol("piecewise_custom")
    expression = s.Piecewise((1, x > 0), (2, True))
    effects = []

    class Impostor(s.Gt):
        @property
        def args(self):
            effects.append("args")
            raise AssertionError("opaque condition must not execute")

    impostor = object.__new__(Impostor)
    Basic.__dict__["_value"].__set__(impostor, _native.py_function("StrictGreaterThan", _native_expr(x), _native.py_integer(0)))
    s.Piecewise.__dict__["_piecewise_branches"].__set__(expression, ((1, impostor), (2, True)))
    with pytest.raises(NotImplementedError):
        s.lambdify(x, expression, modules=backend)
    assert effects == []
    normal = s.Piecewise((1, x > 0), (2, True))
    monkeypatch.setattr(s.Gt, "rel_op", "<")
    with pytest.raises(NotImplementedError):
        s.lambdify(x, normal, modules=backend)
    assert effects == []


@pytest.mark.parametrize("name", ["True", "False", "ordinary_boolean_name"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_symbolic_conditions_remain_valid_but_cannot_be_laundered_into_boolean_atoms(name, backend):
    for symbol in (s.Symbol(name), s.Symbol(name, real=True), s.Dummy(name)):
        expression = s.Piecewise((1, symbol), (2, True), evaluate=False)
        assert expression.args[0].cond is symbol
        for source in (expression, expression + 3, expression * 2, s.Pow(expression, 2, evaluate=False)):
            with pytest.raises(NotImplementedError):
                s.lambdify(symbol, source, modules=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_inactive_branch_still_requires_closed_source(backend):
    x = s.Symbol("piecewise_admission")
    opaque = _native.py_function("UnregisteredBranch", _native_expr(x))
    raw = _native.py_function("Piecewise", _native.py_function("Tuple", opaque, _native.Expr("False")),
                              _native.py_function("Tuple", _native.py_integer(1), _native.Expr("True")))
    with pytest.raises(NotImplementedError):
        s.lambdify(x, raw, modules=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_unsupported_boolean_grammar_is_not_treated_as_numerical(backend):
    x, y = s.Symbol("piecewise_xor_x"), s.Symbol("piecewise_xor_y")
    expression = s.Piecewise((1, s.Xor(x > 0, y > 0)), (0, True))
    with pytest.raises(NotImplementedError):
        s.lambdify((x, y), expression, modules=backend)


def test_piecewise_scalar_support_does_not_grant_derivative_certificates():
    x = s.Symbol("piecewise_seam")
    expression = s.Piecewise((x, x < 0), (2*x, True))
    assert s.compile_numeric(x, expression)(0) == 0.0
    with pytest.raises((ValueError, NotImplementedError)):
        s.compile_residual_system([expression], x)
    with pytest.raises((ValueError, NotImplementedError)):
        s.compile_objective(expression, x)


def test_native_comparison_is_explicitly_binary64_while_math_keeps_integer_values():
    x = s.Symbol("piecewise_rounding")
    expression = s.Piecewise((1, x > 2**53), (0, True))
    assert s.lambdify(x, expression, modules="math")(2**53 + 1) == 1
    assert s.compile_numeric(x, expression)(2**53 + 1) == 0.0


@pytest.mark.parametrize("backend", BACKENDS)
def test_compiled_callable_is_a_snapshot_of_condition_semantics(backend, monkeypatch):
    x = s.Symbol("piecewise_snapshot")
    expression = s.Piecewise((3, x > 0), (7, True))
    compiled = s.lambdify(x, expression, modules=backend)
    monkeypatch.setattr(s.Gt, "rel_op", "<")
    assert compiled(1) == 3
    assert compiled(-1) == 7
    with pytest.raises(NotImplementedError):
        s.lambdify(x, expression, modules=backend)


@pytest.mark.parametrize("backend", BACKENDS)
def test_symbolic_boolean_trees_and_custom_boolean_classes_keep_surface_identity(backend):
    from sympy.logic.boolalg import Boolean
    x = s.Symbol("piecewise_logic_symbol")
    y = s.Symbol("piecewise_logic_other")

    class CustomBoolean(Boolean):
        pass

    for condition in (s.And(x, y), s.Not(x), CustomBoolean()):
        expression = s.Piecewise((1, condition), (2, True), evaluate=False)
        assert expression.args[0].cond is condition
        with pytest.raises(NotImplementedError):
            s.lambdify((x, y), expression, modules=backend)


@pytest.mark.parametrize("function,expected", [("gamma", math.gamma), ("erf", math.erf), ("erfc", math.erfc)])
def test_math_native_function_children_keep_lazy_held_piecewise(function, expected):
    x = s.Symbol("piecewise_native_function")
    held = s.Piecewise((1/x, False), (4, True), (s.log(x), True), evaluate=False)
    raw = _native.py_function(function, _native_expr(held))
    assert s.lambdify(x, raw, modules="math")(0) == expected(4)


@pytest.mark.parametrize("outer", ["gamma", "erf", "erfc", "UnregisteredPiecewiseOuter"])
def test_math_native_function_descent_never_calls_condition_constructors(outer, monkeypatch):
    x = s.Symbol("piecewise_raw_descendants")
    raw = _native.py_function(outer, _native_expr(s.Piecewise((x, x > 0), (-x, True))))
    calls = []

    def forbidden(cls, *args, **kwargs):
        calls.append("relational constructor")
        raise AssertionError("numeric source admission cannot reconstruct conditions")

    monkeypatch.setattr(s.Gt, "__new__", forbidden)
    with pytest.raises(NotImplementedError):
        s.lambdify(x, raw, modules="math")
    assert calls == []


def test_math_native_function_descent_admits_even_unreachable_held_branches():
    x = s.Symbol("piecewise_raw_held")
    held = _native.py_function(
        "Piecewise",
        _native.py_function("Tuple", _native.py_integer(4), _native.Expr("True")),
        _native.py_function("Tuple", _native.py_function("UnregisteredInactiveBranch", _native_expr(x)),
                            _native.Expr("True")),
    )
    raw = _native.py_function("gamma", held)
    with pytest.raises(NotImplementedError):
        s.lambdify(x, raw, modules="math")


@pytest.mark.parametrize("backend", BACKENDS)
def test_symbols_named_boolean_literals_remain_numeric_coordinates_outside_guard_position(backend):
    first, second = s.Symbol("True"), s.Symbol("False")
    expression = s.Piecewise((first + second, first > second), (second, True))
    for source in (expression, _native_expr(expression)):
        compiled = s.lambdify((first, second), source, modules=backend)
        assert compiled(3, 2) == 5
        assert compiled(1, 2) == 2


@pytest.mark.parametrize("name", ["True", "False", "piecewise_condition"])
def test_symbolic_guard_roundtrips_preserve_arguments_printing_and_pickle(name):
    import pickle
    from sympy.core import _wrap

    x = s.Symbol("piecewise_roundtrip_value")
    for guard in (s.Symbol(name), s.Symbol(name, real=True), s.Dummy(name)):
        expression = s.Piecewise((x, guard), (2, True), evaluate=False)
        lifted = _wrap(_native_expr(expression))
        rebuilt = expression.func(*expression.args)
        restored = pickle.loads(pickle.dumps(expression))
        for result in (expression, lifted, rebuilt, restored):
            assert result.args[0].cond == guard
            assert type(result.args[0].cond) is type(guard)
            assert result.args[1].cond is s.true
            assert str(result) == str(expression)
            assert "_FsymBooleanSymbol" not in str(result)
        replacement = s.Symbol("piecewise_replaced_condition")
        substituted = expression.subs(guard, replacement)
        assert substituted.args[0].cond is replacement
        assert substituted.args[1].cond is s.true
        assert expression.subs(x, 7).args[0].cond is guard


@pytest.mark.parametrize("name", ["True", "False", "piecewise_substitution_guard"])
def test_nested_arithmetic_substitution_keeps_guard_and_fallback_distinct(name):
    x, replacement = s.Symbol("piecewise_substitution_value"), s.Symbol("piecewise_new_guard")
    guard = s.Symbol(name)
    expression = s.Piecewise((x, guard), (0, True))
    for wrap in (lambda value: value, lambda value: value + 1, lambda value: 3*value):
        substituted = wrap(expression).subs(guard, replacement)
        assert substituted == wrap(s.Piecewise((x, replacement), (0, True)))
        assert substituted.subs(replacement, s.true) == wrap(x)
        assert substituted.subs(replacement, s.false) == wrap(s.Integer(0))
        for symbol in (s.Symbol("True"), s.Symbol("False"), s.Dummy("True")):
            result = wrap(expression).subs(guard, symbol)
            assert result == wrap(s.Piecewise((x, symbol), (0, True)))
        relation = x > 0
        result = wrap(expression).subs(guard, relation)
        assert result == wrap(s.Piecewise((x, relation), (0, True)))
        assert result.subs(x, -2) == wrap(s.Integer(0))
        assert result.subs(x, 2) == wrap(s.Integer(2))
        for backend in BACKENDS:
            compiled = s.lambdify(x, result, modules=backend)
            assert compiled(-2) == float(wrap(s.Integer(0)))
            assert compiled(2) == float(wrap(s.Integer(2)))


def test_substitution_of_numeric_true_named_symbol_does_not_change_boolean_fallback():
    value, coordinate = s.Symbol("piecewise_numeric_value"), s.Symbol("True")
    expression = s.Piecewise((coordinate, value > 0), (4, True)) + 1
    substituted = expression.subs(coordinate, 9)
    for backend in BACKENDS:
        compiled = s.lambdify(value, substituted, modules=backend)
        assert compiled(2) == 10
        assert compiled(-2) == 5
