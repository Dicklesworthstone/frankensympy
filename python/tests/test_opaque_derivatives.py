"""Actual custom applications remain identifiable through symbolic calculus."""

import copy
import pickle

import pytest

import sympy as s
from sympy.core import (
    Basic, Function, _native_expr, diff_with_receipt, replay_diff_receipt,
    supervised_callback_session,
)


class FormalLaw(Function):
    nargs = 1


def same_named_laws():
    return type("DerivativeCollision", (Function,), {}), type("DerivativeCollision", (Function,), {})


def test_formal_derivative_retains_the_source_class_and_typed_variables():
    x = s.Symbol("x", positive=True)
    A, B = same_named_laws()
    a, b = A(x), B(x)
    da, db = s.Derivative(a, x), s.Derivative(b, x)
    assert da.expr is a
    assert db.expr is b
    assert da.variables == (x,)
    assert da.variables[0] is x
    assert da != db
    assert len({da: "first", db: "second"}) == 2
    assert da == s.Derivative(A(x), x)
    assert hash(da) == hash(s.Derivative(A(x), x))
    assert s.diff(a, x) == da
    assert s.diff(b, x) == db
    assert Basic.__dict__["_value"].__get__(da, Basic) is None


def test_derivative_variables_do_not_create_spurious_free_symbols():
    x, y = s.symbols("x y")
    held = s.Derivative(FormalLaw(y), x)
    assert held.free_symbols == {y}
    assert held.doit() == 0
    assert s.diff(held, x) == 0
    plain = s.Symbol("coordinate")
    positive = s.Symbol("coordinate", positive=True)
    expression = FormalLaw(positive)
    assert s.diff(expression, plain) == 0
    assert s.diff(expression, positive).expr is expression


def test_formal_derivatives_compose_and_reuse_the_original_application():
    x, y = s.symbols("x y")
    application = FormalLaw(x + y)
    first = s.diff(application, x)
    second = s.diff(first, x)
    assert first.expr is application
    assert second.expr is application
    assert second.variables == (x, x)
    assert second.variable_count == ((x, 2),)
    assert s.Derivative(first, (y, 2)).variables == (x, y, y)
    assert s.Derivative(first, y).expr is application


@pytest.mark.parametrize("operation", [s.sin, s.cos, s.exp, s.log])
def test_builtin_chain_rule_keeps_the_exact_custom_application(operation):
    x = s.Symbol("x")
    application = FormalLaw(x)
    result = s.diff(operation(application), x)
    formal = s.Derivative(application, x)
    expected = {
        s.sin: lambda: s.cos(application) * formal,
        s.cos: lambda: -s.sin(application) * formal,
        s.exp: lambda: s.exp(application) * formal,
        s.log: lambda: formal / application,
    }[operation]()
    assert result == expected
    assert all(node is application for node in result.atoms(FormalLaw))
    assert all(node.expr is application for node in result.atoms(s.Derivative))


def test_sum_product_and_power_rules_preserve_two_same_named_laws():
    x = s.Symbol("x")
    A, B = same_named_laws()
    a, b = A(x), B(x)
    da, db = s.Derivative(a, x), s.Derivative(b, x)
    assert s.diff(a + b, x) == da + db
    assert s.diff(a - b, x) == da - db
    assert s.diff(a * b, x) == da*b + a*db
    assert s.diff(a**3, x) == 3*a**2*da
    assert s.diff(a**b, x) == a**b * (db*s.log(a) + b*da/a)


def test_unresolved_composition_is_retained_without_running_a_custom_constructor():
    x = s.Symbol("x")
    calls = []

    class Unresolved(Function):
        def __new__(cls, *args, **options):
            calls.append(args)
            return super().__new__(cls, *args, **options)

    application = Unresolved(x**2)
    calls.clear()
    result = s.diff(application, x)
    assert type(result) is s.Derivative
    assert result.expr is application
    assert result.variables == (x,)
    assert calls == []


def test_supervised_derivative_results_feed_an_ordinary_matrix_residual_solve():
    x = s.Symbol("x")
    effects = []

    def derivative_a(self, variable):
        effects.append(("a", self, variable))
        return 2 * s.diff(self.args[0], variable)

    def derivative_b(self, variable):
        effects.append(("b", self, variable))
        return s.diff(self.args[0], variable)

    A = type("SameResidualLaw", (Function,), {"_eval_derivative": derivative_a})
    B = type("SameResidualLaw", (Function,), {"_eval_derivative": derivative_b})
    a, b = A(x), B(x)
    residuals = s.Matrix([a - b - 3])
    jacobian = residuals.jacobian([x])
    assert jacobian == s.Matrix([[1]])
    assert len(effects) == 2
    assert {entry[0] for entry in effects} == {"a", "b"}
    assert all(entry[1] is (a if entry[0] == "a" else b) and entry[2] is x
               for entry in effects)
    program = s.compile_hybrid_residual_system(
        residuals, [x], jacobian=jacobian, callbacks={A: lambda value: 2*value, B: lambda value: value},
    )
    assert program.residuals(3) == [0.0]
    assert program.jacobian(3) == [[1.0]]
    root = program.solve([0])
    assert root.shape == (1, 1)
    assert float(root[0, 0]) == pytest.approx(3.0)
    assert program.evidence == "approximate_with_user_callbacks"


def test_none_from_a_supervised_hook_keeps_the_existing_owned_effect_contract():
    x = s.Symbol("x")
    seen = []

    class Pending(Function):
        def _eval_derivative(self, variable):
            seen.append((self, variable))
            return None

    application = Pending(x)
    with supervised_callback_session() as session:
        first = s.diff(application, x)
        second = s.diff(application, x)
    assert first.expr is application and second.expr is application
    assert seen == [(application, x)]
    assert [entry[3] for entry in session.trace()
            if entry[0] == "_eval_derivative"] == ["ok", "cached_hit"]


def test_copy_and_pickle_keep_formal_derivatives_in_the_python_identity_lane():
    x = s.Symbol("x", real=True)
    source = FormalLaw(x)
    expression = s.Derivative(source, (x, 2))
    for result in [copy.copy(expression), copy.deepcopy(expression), pickle.loads(pickle.dumps(expression))]:
        assert result == expression
        assert type(result.expr) is FormalLaw
        assert result.variables == (x, x)
        assert Basic.__dict__["_value"].__get__(result, Basic) is None


def test_opaque_derivatives_cannot_be_lowered_or_issued_a_native_proof():
    x = s.Symbol("x")
    application = FormalLaw(x)
    formal = s.Derivative(application, x)
    for expression in [application, formal, 2*formal + s.sin(application)]:
        with pytest.raises(NotImplementedError, match="opaque"):
            _native_expr(expression)
        with pytest.raises(NotImplementedError, match="opaque"):
            diff_with_receipt(expression, x)
        with pytest.raises(NotImplementedError):
            s.compile_numeric([x], expression)


def test_order_and_cycle_limits_refuse_before_unbounded_formal_work():
    x = s.Symbol("x")
    application = FormalLaw(x)
    with pytest.raises(ValueError, match="order limit"):
        s.Derivative(application, (x, 10**9))
    with pytest.raises(ValueError, match="non-negative"):
        s.Derivative(application, (x, -1))
    Function.__dict__["_args"].__set__(application, (application,))
    with pytest.raises(ValueError, match="cyclic"):
        s.diff(application, x)


@pytest.mark.parametrize("coordinate", [lambda x: s.Integer(1), lambda x: x + 1,
                                         lambda x: s.Float(0.5)])
def test_formal_derivatives_refuse_invalid_coordinates(coordinate):
    x = s.Symbol("x")
    with pytest.raises(ValueError, match="non-symbolic"):
        s.Derivative(FormalLaw(x), coordinate(x))


def test_unresolved_partial_at_a_composite_argument_remains_a_whole_source_derivative():
    x = s.Symbol("x")

    class PendingPartial(Function):
        def fdiff(self, argindex=1):
            return None

    application = PendingPartial(x**2)
    result = s.diff(application, x)
    assert result.expr is application
    assert result.variables == (x,)


def test_variable_exponent_at_zero_keeps_the_original_singular_source():
    x = s.Symbol("x")

    class ExponentLaw(Function):
        def _eval_derivative(self, variable):
            return s.Integer(7)

    source = s.Pow(0, ExponentLaw(x), evaluate=False)
    result = s.diff(source, x)
    assert type(result) is s.Derivative
    assert result.expr is source
    assert result.variables == (x,)


def test_closed_derivative_evaluation_and_typed_native_receipts_are_unchanged():
    x = s.Symbol("x", positive=True)
    source = x**4
    held = s.Derivative(source, (x, 2))
    assert held.expr is source
    assert all(variable is x for variable in held.variables)
    assert held.doit() == 12*x**2
    result, receipt = diff_with_receipt(source, x)
    assert result == 4*x**3
    assert replay_diff_receipt(receipt)
