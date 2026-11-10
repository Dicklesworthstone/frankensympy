"""Public characteristic PDE families through the actual native bridge.

Residual identities use several explicit differentiable arbitrary-function
choices. They supplement the native coefficient and characteristic checks;
sampled choices alone do not establish a general PDE-completeness claim.
"""

from unittest.mock import patch

import pytest
import sympy as s

from sympy.core import _native
from sympy.solvers import pdsolve as exported_pdsolve
from sympy.solvers.pde import pdsolve as module_pdsolve


def assert_zero(expression):
    assert s.simplify(s.expand(expression)) == 0


def arbitrary_application(solution):
    applications = solution.rhs.atoms(s.AppliedUndef)
    assert len(applications) == 1
    application = next(iter(applications))
    assert len(application.args) == 1
    return application


def assert_family(solution, function, x, y, a, b, c, forcing):
    assert type(solution) is s.Eq
    assert solution.lhs is function
    application = arbitrary_application(solution)
    assert type(application) is not type(function)
    invariant = application.args[0]
    assert solution.rhs.free_symbols <= {x, y}
    for symbol in solution.rhs.free_symbols:
        assert symbol is x or symbol is y
    assert_zero(a*s.diff(invariant, x) + b*s.diff(invariant, y))
    for choice in (s.Integer(0), s.Integer(1), invariant,
                   invariant**3 - 2*invariant + 5, s.exp(invariant)):
        candidate = solution.rhs.subs(application, choice)
        assert_zero(a*s.diff(candidate, x) + b*s.diff(candidate, y) + c*candidate - forcing)
    return application


def test_pdsolve_is_exported_through_both_public_namespaces():
    assert s.pdsolve is exported_pdsolve is module_pdsolve


@pytest.mark.parametrize("a,b,c,forcing", [
    (1, 0, 0, lambda x, y: 0),
    (0, 2, 0, lambda x, y: x*y + 3),
    (0, -3, 2, lambda x, y: x**2 + y),
    (1, 2, 0, lambda x, y: x*y + x**2),
    (2, -3, 5, lambda x, y: x**2*y + 2*y**2 - 7),
    (s.Rational(2, 3), s.Rational(-5, 7), s.Rational(-3, 2),
     lambda x, y: s.Rational(3, 5)*x*y + s.Rational(7, 11)),
])
def test_transport_reaction_and_polynomial_forcing_have_exact_families(a, b, c, forcing):
    x, y = s.symbols("pde_transport_x pde_transport_y")
    u = s.Function("pde_transport_u")(x, y)
    load = forcing(x, y)
    source = s.Eq(a*s.diff(u, x) + b*s.diff(u, y) + c*u, load, evaluate=False)
    explicit = s.pdsolve(source, u)
    inferred = s.pdsolve(source)
    assert explicit == inferred
    assert_family(explicit, u, x, y, a, b, c, load)


def test_each_original_equation_side_and_retained_rational_coefficient_is_used():
    x, y = s.symbols("pde_original_x pde_original_y")
    u = s.Function("pde_original_u")(x, y)
    half = s.Pow(s.Integer(2), -1, evaluate=False)
    left = s.Add(s.Mul(half, s.Derivative(u, x), evaluate=False), 3*u, x*y, evaluate=False)
    right = s.Add(-2*s.Derivative(u, y), 1, evaluate=False)
    source = s.Eq(left, right, evaluate=False)
    answer = s.pdsolve(source, u)
    assert_family(answer, u, x, y, s.Rational(1, 2), 2, 3, 1 - x*y)


def test_maximum_forcing_degree_reaches_the_degree_seventeen_particular():
    x, y = s.symbols("pde_degree_x pde_degree_y")
    u = s.Function("pde_degree_u")(x, y)
    answer = s.pdsolve(s.diff(u, x) - x**16, u)
    application = assert_family(answer, u, x, y, 1, 0, 0, x**16)
    assert_zero(answer.rhs.subs(application, 0) - x**17/17)


@pytest.mark.parametrize("declaration", [{}, {"nargs": 2}, {"nargs": (1, 2)}])
def test_exact_factory_declarations_and_explicit_arbitrary_owner_are_preserved(declaration):
    x, y = s.symbols("pde_declared_x pde_declared_y")
    owner = s.Function("pde_declared_u", **declaration)
    u = owner(x, y)
    phi = s.Function("pde_declared_Phi", nargs=1)
    answer = s.pdsolve(s.diff(u, x) - s.diff(u, y) + u - x, u, solvefun=phi)
    application = assert_family(answer, u, x, y, 1, -1, 1, x)
    assert type(application) is phi
    assert type(answer.lhs) is owner


@pytest.mark.parametrize("name", ["sin", "exp", "Derivative"])
def test_a_formal_function_name_never_grants_intrinsic_authority(name):
    x, y = s.symbols("pde_spelling_x pde_spelling_y")
    owner = s.Function(name, nargs=2)
    u = owner(x, y)
    answer = s.pdsolve(s.diff(u, x) + s.diff(u, y) - y, u)
    assert type(answer.lhs) is owner
    assert_family(answer, u, x, y, 1, 1, 0, y)


def test_default_arbitrary_function_avoids_dependent_and_coordinate_names():
    x, y = s.Symbol("F"), s.Symbol("F1")
    u = s.Function("F2", nargs=2)(x, y)
    answer = s.pdsolve(s.diff(u, x) + s.diff(u, y), u)
    application = arbitrary_application(answer)
    assert type(application).__name__ not in {"F", "F1", "F2"}
    assert_family(answer, u, x, y, 1, 1, 0, 0)


@pytest.mark.parametrize("axes", [
    lambda: (s.Symbol("pde_same", positive=True), s.Symbol("pde_same", negative=True)),
    lambda: (s.Dummy("pde_dummy", real=True), s.Dummy("pde_dummy", real=True)),
    lambda: (s.Dummy("pde_mixed", positive=True), s.Symbol("pde_mixed", real=True)),
])
def test_same_spelling_and_dummy_coordinates_keep_their_exact_identity(axes):
    x, y = axes()
    u = s.Function("pde_typed_u")(x, y)
    answer = s.pdsolve(2*s.diff(u, x) - 3*s.diff(u, y) + 4*u - x*y, u)
    assert answer.lhs.args[0] is x
    assert answer.lhs.args[1] is y
    assert_family(answer, u, x, y, 2, -3, 4, x*y)


def test_constant_coefficient_hints_have_explicit_scope():
    x, y = s.symbols("pde_hint_x pde_hint_y")
    u = s.Function("pde_hint_u")(x, y)
    source = s.diff(u, x) + s.diff(u, y)
    for hint in ("default", "1st_linear_constant_coeff", "1st_linear_constant_coeff_homogeneous"):
        assert_family(s.pdsolve(source, u, hint=hint), u, x, y, 1, 1, 0, 0)
    with pytest.raises(ValueError):
        s.pdsolve(source - 1, u, hint="1st_linear_constant_coeff_homogeneous")
    for options in ({"hint": "all"}, {"dict": True}, {"ics": {u: 0}}):
        with pytest.raises(NotImplementedError):
            s.pdsolve(source, u, **options)
    with pytest.raises(TypeError):
        s.pdsolve(source, u, dict=1)


def test_nonpolynomial_nonlinear_higher_order_and_variable_coefficients_refuse_before_solve():
    x, y, parameter = s.symbols("pde_refuse_x pde_refuse_y pde_parameter")
    u = s.Function("pde_refuse_u")(x, y)
    dx, dy = s.Derivative(u, x), s.Derivative(u, y)
    sources = [
        dx + s.sin(x*x), dx + s.Float(0.5), dx + parameter,
        x*dx + dy, dx + x*u, dx + u*u, dx*dy + u,
        s.Derivative(u, x, y), s.Derivative(u, x, 2),
        dx + s.Pow(x, s.Rational(1, 2), evaluate=False),
        dx + s.Function("foreign_forcing")(x),
        dx + s.Derivative(x*x, x, evaluate=False),
        u - x*y,
    ]
    with patch.object(_native, "linear_first_order_pde_native",
                      side_effect=AssertionError("unsupported original PDE reached native solve")):
        for source in sources:
            with pytest.raises((NotImplementedError, ValueError)):
                s.pdsolve(source, u)


def test_cancelled_or_zeroed_original_domains_and_opaque_terms_are_not_erased():
    x, y = s.symbols("pde_guard_x pde_guard_y")
    u = s.Function("pde_guard_u")(x, y)
    dx = s.Derivative(u, x)
    bad = [
        s.Pow(x, -1, evaluate=False),
        s.Pow(x, 0, evaluate=False),
        s.Pow(s.Integer(0), 0, evaluate=False),
        s.Pow(x, s.Rational(1, 2), evaluate=False),
        s.Function("hidden_pde_forcing")(x),
    ]
    with patch.object(_native, "linear_first_order_pde_native",
                      side_effect=AssertionError("erased original source reached native solve")):
        for term in bad:
            expressions = [
                s.Add(dx, s.Mul(0, term, evaluate=False), evaluate=False),
                s.Eq(s.Add(dx, term, evaluate=False), term, evaluate=False),
                s.Add(dx, s.Pow(term, 0, evaluate=False), evaluate=False),
            ]
            for source in expressions:
                with pytest.raises((NotImplementedError, ValueError)):
                    s.pdsolve(source, u)


def test_foreign_function_owner_and_shifted_or_reordered_arguments_refuse():
    x, y, z = s.symbols("pde_owner_x pde_owner_y pde_owner_z")
    owner = s.Function("pde_owner_u")
    u = owner(x, y)
    dx = s.Derivative(u, x)
    sources = [
        dx + owner(y, x), dx + owner(x + 1, y),
        dx + s.Function("pde_owner_u", nargs=2)(x, y),
        dx + s.Derivative(u, z),
    ]
    for source in sources:
        with pytest.raises((NotImplementedError, ValueError)):
            s.pdsolve(source, u)
    with pytest.raises((NotImplementedError, ValueError)):
        s.pdsolve(dx, s.Function("pde_other_u")(x, y))
    for function in (owner(x), owner(x, x), owner(x, y, z)):
        with pytest.raises((NotImplementedError, ValueError)):
            s.pdsolve(s.Derivative(function, x), function)


def test_custom_function_or_symbol_getters_and_hooks_are_not_called():
    x, y = s.symbols("pde_callback_x pde_callback_y")
    calls = []

    class Custom(s.Function):
        nargs = 2

    u = Custom(x, y, evaluate=False)
    source = s.Derivative(u, x)
    def forbidden(*args, **kwargs):
        calls.append("called")
        raise AssertionError("custom callback executed")
    with patch.object(Custom, "args", property(forbidden)):
        with pytest.raises(NotImplementedError):
            s.pdsolve(source, u)

    class Coordinate(s.Symbol):
        pass

    foreign = Coordinate("pde_custom_coordinate")
    source = s.Derivative(s.Function("pde_custom_u")(foreign, y), x)
    with patch.object(Coordinate, "name", property(forbidden)):
        with pytest.raises(NotImplementedError):
            s.pdsolve(source)
    assert calls == []


def test_modified_factory_or_inherited_implementation_cannot_gain_admission():
    x, y = s.symbols("pde_mutated_owner_x pde_mutated_owner_y")
    owner = s.Function("pde_mutated_owner", nargs=2)
    u = owner(x, y)
    source = s.Derivative(u, x)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("called")
        raise AssertionError("modified implementation executed")

    with patch.object(owner, "eval", classmethod(forbidden), create=True):
        with pytest.raises(NotImplementedError):
            s.pdsolve(source, u)
    with patch.object(s.AppliedUndef, "args", property(forbidden)):
        with pytest.raises(NotImplementedError):
            s.pdsolve(source, u)
    assert calls == []


def test_mutated_coordinate_assumptions_and_dummy_declarations_refuse():
    x, y = s.symbols("pde_stale_x pde_stale_y")
    u = s.Function("pde_stale_u")(x, y)
    source = s.Derivative(u, x)
    original = x._assumptions.copy()
    try:
        x._assumptions["positive"] = True
        with pytest.raises(ValueError):
            s.pdsolve(source, u)
    finally:
        x._assumptions.clear()
        x._assumptions.update(original)
    d = s.Dummy("pde_stale_dummy")
    u = s.Function("pde_stale_dummy_u")(d, y)
    source = s.Derivative(u, d)
    original_name = d._dummy_name
    try:
        d._dummy_name = "pde_changed_dummy"
        with pytest.raises(ValueError):
            s.pdsolve(source, u)
    finally:
        d._dummy_name = original_name


def test_stale_native_and_conflicting_retained_arguments_refuse():
    x, y = s.symbols("pde_stale_args_x pde_stale_args_y")
    u = s.Function("pde_stale_args_u")(x, y)
    held = s.Pow(x, 2, evaluate=False)
    source = s.Add(s.Derivative(u, x), held, evaluate=False)
    slot = s.Pow.__dict__["_args"]
    original = slot.__get__(held, s.Pow)
    try:
        slot.__set__(held, (y, original[1]))
        with pytest.raises(ValueError):
            s.pdsolve(source, u)
    finally:
        slot.__set__(held, original)
    surface = s.Basic.__dict__["_surface_args"]
    derivative = s.Derivative(u, x)
    try:
        surface.__set__(u, (y, x))
        with pytest.raises(ValueError):
            s.pdsolve(derivative, u)
    finally:
        surface.__delete__(u)


def test_native_failures_remain_terminal_without_fallback():
    x, y = s.symbols("pde_failure_x pde_failure_y")
    u = s.Function("pde_failure_u")(x, y)
    source = s.Derivative(u, x)
    failure = RuntimeError("PDE budget exhausted")
    with patch.object(_native, "linear_first_order_pde_native", side_effect=failure):
        with pytest.raises(RuntimeError) as caught:
            s.pdsolve(source, u)
    assert caught.value is failure


def test_original_function_mutation_during_native_evaluation_blocks_publication():
    x, y = s.symbols("pde_race_x pde_race_y")
    u = s.Function("pde_race_u")(x, y)
    source = s.Derivative(u, x) + s.Derivative(u, y)
    original_native = _native.linear_first_order_pde_native
    slot = s.Function.__dict__["_args"]
    original = slot.__get__(u, s.Function)

    def mutate(*args):
        result = original_native(*args)
        slot.__set__(u, (y, x))
        return result

    try:
        with patch.object(_native, "linear_first_order_pde_native", side_effect=mutate):
            with pytest.raises(RuntimeError):
                s.pdsolve(source, u)
    finally:
        slot.__set__(u, original)


def test_exponential_implementation_mutation_during_native_call_blocks_lifting():
    x, y = s.symbols("pde_exp_race_x pde_exp_race_y")
    u = s.Function("pde_exp_race_u")(x, y)
    source = s.Derivative(u, x) + u
    original_native = _native.linear_first_order_pde_native
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("called")
        raise AssertionError("changed exponential implementation executed")

    changed = patch.object(s.exp, "eval", classmethod(forbidden), create=True)

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


def test_arbitrary_function_requires_its_own_unmodified_unary_owner():
    x, y = s.symbols("pde_arbitrary_x pde_arbitrary_y")
    owner = s.Function("pde_arbitrary_u")
    u = owner(x, y)
    source = s.Derivative(u, x)
    with pytest.raises(NotImplementedError):
        s.pdsolve(source, u, solvefun=owner)
    with pytest.raises(NotImplementedError):
        s.pdsolve(source, u, solvefun=s.Function("pde_wrong_arity", nargs=2))
    calls = []

    class Custom(s.Function):
        @classmethod
        def eval(cls, arg):
            calls.append(arg)
            raise AssertionError("custom arbitrary function evaluated")

    with pytest.raises(NotImplementedError):
        s.pdsolve(source, u, solvefun=Custom)
    assert calls == []


def test_native_function_spelling_is_not_an_original_formal_source():
    from sympy.core import _native_expr, _wrap

    x, y = s.symbols("pde_native_name_x pde_native_name_y")
    owner = s.Function("pde_native_name_u", nargs=2)
    forged = _wrap(_native.py_function("pde_native_name_u", _native_expr(x), _native_expr(y)))
    with pytest.raises((NotImplementedError, ValueError)):
        s.pdsolve(s.Derivative(forged, x), forged)
    assert type(forged) is owner


def test_degree_power_text_and_original_source_bounds_fail_explicitly():
    x, y = s.symbols("pde_bounds_x pde_bounds_y")
    u = s.Function("pde_bounds_u")(x, y)
    dx = s.Derivative(u, x)
    sources = [
        s.Add(dx, x**17, evaluate=False),
        s.Add(dx, s.Pow(x, 65, evaluate=False), evaluate=False),
        s.Add(dx, s.Integer(1 << 4096), evaluate=False),
    ]
    for source in sources:
        with pytest.raises((NotImplementedError, ValueError)):
            s.pdsolve(source, u)
    huge = s.Symbol("p" * 16_385)
    target = s.Function("pde_long_u")(huge, y)
    with pytest.raises(ValueError):
        s.pdsolve(s.Derivative(target, huge), target)
