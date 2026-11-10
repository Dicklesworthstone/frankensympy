"""Public elementary PDE families through the native original-source checker."""

from unittest.mock import patch

import pytest
import sympy as s

from sympy.core import _native, _native_expr, _wrap


def assert_zero(expression):
    # The public trig search supplies exact angle/parity identities beyond
    # construction's deliberately incomplete normalization of balanced sums.
    reduced = s.simplify(s.expand(expression))
    assert reduced == 0 or s.trigsimp(reduced) == 0


def assert_family(answer, u, x, y, a, b, c, forcing, particular):
    assert type(answer) is s.Eq
    assert answer.lhs is u
    applications = answer.rhs.atoms(s.AppliedUndef)
    assert len(applications) == 1
    application = next(iter(applications))
    assert len(application.args) == 1
    invariant = application.args[0]
    assert_zero(a*s.diff(invariant, x) + b*s.diff(invariant, y))
    assert_zero(answer.rhs.subs(application, 0) - particular)
    assert answer.rhs.free_symbols <= {x, y}
    for symbol in answer.rhs.free_symbols:
        assert symbol is x or symbol is y
    # These exact differentiated representatives supplement the native
    # coefficient/characteristic proof; they do not assert PDE completeness.
    for choice in (s.Integer(0), invariant**2 + 1, s.exp(invariant)):
        candidate = answer.rhs.subs(application, choice)
        assert_zero(a*s.diff(candidate, x) + b*s.diff(candidate, y)
                    + c*candidate - forcing)


@pytest.mark.parametrize("a,b,c,forcing,particular", [
    (
        1, 2, 3,
        lambda x, y: s.exp(2*x-y+1)*(
            (1+3*x+4*y)*s.cos(2*x+y+2) + (2+3*y-4*x)*s.sin(2*x+y+2)),
        lambda x, y: s.exp(2*x-y+1)*(x*s.cos(2*x+y+2) + y*s.sin(2*x+y+2)),
    ),
    (
        2, 1, 2,
        lambda x, y: x*s.exp(2*x-3*y+4),
        lambda x, y: (x/3-s.Rational(2, 9))*s.exp(2*x-3*y+4),
    ),
    (
        1, 0, 0,
        lambda x, y: s.cos(x) + s.sin(x),
        lambda x, y: s.sin(x) - s.cos(x),
    ),
    (
        1, 0, 2,
        lambda x, y: s.cos(x) + s.sin(x),
        lambda x, y: (s.cos(x) + 3*s.sin(x))/5,
    ),
    (
        s.Rational(1, 2), s.Rational(-2, 3), s.Rational(3, 5),
        lambda x, y: s.exp(x/3+y/4)*(
            (s.Rational(1, 2)+3*x/5-y)*s.cos(2*x+3*y)
            + (s.Rational(-2, 3)+3*y/5+x)*s.sin(2*x+3*y)),
        lambda x, y: s.exp(x/3+y/4)*(x*s.cos(2*x+3*y) + y*s.sin(2*x+3*y)),
    ),
])
def test_elementary_forcing_has_exact_hand_derived_public_particulars(
        a, b, c, forcing, particular):
    x, y = s.symbols("pde_modes_x pde_modes_y")
    u = s.Function("pde_modes_u")(x, y)
    load = forcing(x, y)
    equation = s.Eq(a*s.diff(u, x) + b*s.diff(u, y) + c*u, load, evaluate=False)
    answer = s.pdsolve(equation, u)
    assert_family(answer, u, x, y, a, b, c, load, particular(x, y))


@pytest.mark.parametrize("a,b,c,forcing,particular", [
    (
        1, 1, 2,
        lambda x, y: s.exp(-x-y+1)*(x*s.cos(x-y+3) + y*s.sin(x-y+3)),
        lambda x, y: s.exp(-x-y+1)*(
            x**2*s.cos(x-y+3)/2 + (x*y-x**2/2)*s.sin(x-y+3)),
    ),
    (
        0, 2, -1,
        lambda x, y: x*y*s.exp(x+y/2+2)*s.cos(3*x+1),
        lambda x, y: x*y**2*s.exp(x+y/2+2)*s.cos(3*x+1)/4,
    ),
    (
        1, 1, 0,
        lambda x, y: x**16*s.exp(x-y)*s.cos(x-y),
        lambda x, y: x**17*s.exp(x-y)*s.cos(x-y)/17,
    ),
])
def test_resonance_uses_transport_integration_including_the_degree_boundary(
        a, b, c, forcing, particular):
    x, y = s.symbols("pde_resonance_x pde_resonance_y")
    u = s.Function("pde_resonance_u")(x, y)
    load = forcing(x, y)
    equation = a*s.diff(u, x) + b*s.diff(u, y) + c*u - load
    answer = s.pdsolve(equation)
    assert_family(answer, u, x, y, a, b, c, load, particular(x, y))


def test_canonical_e_constant_signed_phases_and_exponential_products():
    x, y = s.symbols("pde_canonical_x pde_canonical_y")
    u = s.Function("pde_canonical_u")(x, y)
    load = (1+s.E)*s.cos(x)
    answer = s.pdsolve(s.Eq(s.diff(u, x), load, evaluate=False), u)
    assert_family(answer, u, x, y, 1, 0, 0, load, (1+s.E)*s.sin(x))
    # Native lifting currently creates a fresh exact Expr for Const(E).
    # Admission must inspect that exact constant rather than object identity.
    load = s.exp(1)*s.cos(x)
    answer = s.pdsolve(s.diff(u, x)-load, u)
    assert_family(answer, u, x, y, 1, 0, 0, load, s.E*s.sin(x))
    load = _wrap(_native_expr(s.E*s.cos(x)))
    answer = s.pdsolve(s.diff(u, x)-load, u)
    assert_family(answer, u, x, y, 1, 0, 0, load, s.E*s.sin(x))
    load = x+y+x*s.exp(x)*s.exp(-y)+2*s.sin(y-x)
    expected = x+y-2+(x-1)*s.exp(x-y)-2*s.sin(x-y)
    answer = s.pdsolve(s.diff(u, x)+s.diff(u, y)+u-load, u)
    assert_family(answer, u, x, y, 1, 1, 1, load, expected)
    load = s.exp(2)*(x*s.cos(-3)+y*s.sin(-3))
    answer = s.pdsolve(s.diff(u, x)-load, u)
    expected = s.exp(2)*(x**2*s.cos(3)/2-x*y*s.sin(3))
    assert_family(answer, u, x, y, 1, 0, 0, load, expected)


@pytest.mark.parametrize("axes", [
    lambda: (s.Symbol("pde_mode_axis", positive=True),
             s.Symbol("pde_mode_axis", negative=True)),
    lambda: (s.Dummy("pde_mode_dummy", real=True), s.Dummy("pde_mode_dummy", real=True)),
])
def test_elementary_arguments_preserve_exact_typed_coordinate_identity(axes):
    x, y = axes()
    u = s.Function("pde_mode_typed_u")(x, y)
    load = x*s.exp(y)*s.sin(y)
    answer = s.pdsolve(s.diff(u, x)-load, u)
    assert answer.lhs.args[0] is x
    assert answer.lhs.args[1] is y
    assert_family(answer, u, x, y, 1, 0, 0, load, x**2*s.exp(y)*s.sin(y)/2)


def test_held_unsupported_elementary_sources_refuse_before_native_solving():
    x, y = s.symbols("pde_mode_guard_x pde_mode_guard_y")
    u = s.Function("pde_mode_guard_u")(x, y)
    dx = s.Derivative(u, x)
    terms = [
        s.sin(x*x, evaluate=False),
        s.exp(s.Pow(x, -1, evaluate=False), evaluate=False),
        s.exp(s.E*x, evaluate=False),
        s.sin(s.exp(x), evaluate=False),
        s.Function("exp")(x),
        s.Pow(s.exp(x), 0, evaluate=False),
        s.Pow(s.cos(x), 2, evaluate=False),
        s.Mul(s.sin(x), s.cos(y), evaluate=False),
    ]
    with patch.object(_native, "linear_first_order_pde_native",
                      side_effect=AssertionError("unsupported original mode reached native solving")):
        for term in terms:
            for source in (
                s.Add(dx, term, evaluate=False),
                s.Add(dx, s.Mul(0, term, evaluate=False), evaluate=False),
                s.Eq(s.Add(dx, term, evaluate=False), term, evaluate=False),
            ):
                with pytest.raises((NotImplementedError, ValueError)):
                    s.pdsolve(source, u)
        for source in (
            s.exp(x)*dx + s.Derivative(u, y),
            dx + s.exp(x)*u,
            s.Add(dx, s.Mul(s.cos(0, evaluate=False), s.sin(x),
                           evaluate=False), evaluate=False),
        ):
            with pytest.raises((NotImplementedError, ValueError)):
                s.pdsolve(source, u)


def test_retained_elementary_argument_mutation_is_not_hidden_by_native_value():
    x, y = s.symbols("pde_mode_stale_x pde_mode_stale_y")
    u = s.Function("pde_mode_stale_u")(x, y)
    term = s.sin(x, evaluate=False)
    source = s.Add(s.Derivative(u, x), term, evaluate=False)
    slot = s.Function.__dict__["_args"]
    original = slot.__get__(term, s.Function)
    try:
        slot.__set__(term, (y,))
        with pytest.raises(ValueError):
            s.pdsolve(source, u)
    finally:
        slot.__set__(term, original)


def test_elementary_source_mutation_during_native_call_blocks_publication():
    x, y = s.symbols("pde_mode_race_x pde_mode_race_y")
    u = s.Function("pde_mode_race_u")(x, y)
    term = s.cos(x, evaluate=False)
    source = s.Add(s.Derivative(u, x), term, evaluate=False)
    original_native = _native.linear_first_order_pde_native
    slot = s.Function.__dict__["_args"]
    original = slot.__get__(term, s.Function)

    def mutate(*args):
        result = original_native(*args)
        slot.__set__(term, (y,))
        return result

    try:
        with patch.object(_native, "linear_first_order_pde_native", side_effect=mutate):
            with pytest.raises(RuntimeError):
                s.pdsolve(source, u)
    finally:
        slot.__set__(term, original)


def test_trigonometric_implementation_mutation_blocks_lifting_without_callback():
    x, y = s.symbols("pde_mode_protocol_x pde_mode_protocol_y")
    u = s.Function("pde_mode_protocol_u")(x, y)
    source = s.Derivative(u, x) - s.sin(x)
    original_native = _native.linear_first_order_pde_native
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("called")
        raise AssertionError("changed sine implementation executed")

    changed = patch.object(s.sin, "eval", classmethod(forbidden), create=True)

    def mutate(*args):
        result = original_native(*args)
        changed.start()
        return result

    try:
        with patch.object(_native, "linear_first_order_pde_native", side_effect=mutate):
            with pytest.raises(NotImplementedError):
                s.pdsolve(source, u)
    finally:
        changed.stop()
    assert calls == []


def test_canonical_e_requires_its_original_native_storage():
    x, y = s.symbols("pde_mode_e_x pde_mode_e_y")
    u = s.Function("pde_mode_e_u")(x, y)
    source = s.Add(s.Derivative(u, x), s.Mul(s.E, s.cos(x), evaluate=False),
                   evaluate=False)
    slot = s.Basic.__dict__["_value"]
    original = slot.__get__(s.E, s.Basic)
    try:
        slot.__set__(s.E, _native_expr(s.Integer(2)))
        with pytest.raises((NotImplementedError, ValueError)):
            s.pdsolve(source, u)
    finally:
        slot.__set__(s.E, original)


def test_other_constants_and_arbitrary_expr_aliases_do_not_gain_e_authority():
    x, y = s.symbols("pde_mode_alias_x pde_mode_alias_y")
    u = s.Function("pde_mode_alias_u")(x, y)
    dx = s.Derivative(u, x)
    aliases = (s.pi, s.I, s.oo, s.Expr(s.Integer(2)), s.Expr(x), s.Expr(s.sin(x)))
    with patch.object(_native, "linear_first_order_pde_native",
                      side_effect=AssertionError("non-E Expr acquired constant authority")):
        for alias in aliases:
            source = s.Add(dx, alias, evaluate=False)
            with pytest.raises((NotImplementedError, ValueError)):
                s.pdsolve(source, u)
