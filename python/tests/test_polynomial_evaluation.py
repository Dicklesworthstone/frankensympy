"""Typed partial evaluation, exact domains and retained-source adversaries."""

import pytest
import sympy as sp
from sympy.polys.polyerrors import CoercionFailed, DomainError, GeneratorsError, UnificationFailed


def assert_polynomial(value, cls, expression, generators, domain):
    assert type(value) is cls
    assert value.gens == generators
    assert str(value.domain) == domain
    assert value.as_expr() == expression


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
@pytest.mark.parametrize("domain", ["ZZ", "QQ"])
@pytest.mark.parametrize("selector", ["default", "symbol", "index", "negative"])
def test_profile_partial_evaluation_preserves_class_domain_and_remaining_ring(cls, domain, selector):
    x, y = sp.symbols("x y")
    polynomial = cls(x*y + x + y, x, y, domain=domain)
    if selector == "default":
        result = polynomial.eval(2)
    else:
        result = polynomial.eval({"symbol": x, "index": 0, "negative": -2}[selector], 2)
    assert_polynomial(result, cls, 3*y + 2, (y,), domain)
    assert_polynomial(polynomial.eval(y, 3), cls, 4*x + 3, (x,), domain)
    assert polynomial.eval([2, 3]) == 11


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
@pytest.mark.parametrize("container", [list, tuple])
def test_profile_sequence_assignments_preserve_unassigned_order_and_return_exact_scalars(cls, container):
    x, y, z = sp.symbols("x y z")
    polynomial = cls(x*y + x + y + z/2, x, y, z, domain="QQ")
    assert_polynomial(polynomial.eval(container([2, 3])), cls, 11 + z/2, (z,), "QQ")
    assert polynomial.eval(container([2, 3, sp.Rational(1, 3)])) == sp.Rational(67, 6)


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
def test_profile_mapping_indices_follow_the_remaining_generator_list(cls):
    x, y, z = sp.symbols("x y z")
    polynomial = cls(x + 10*y + 100*z, x, y, z, domain="ZZ")
    assert_polynomial(polynomial.eval({0: 2, 1: 3}), cls, 10*y + 302, (y,), "ZZ")
    assert_polynomial(polynomial.eval({1: 2, 0: 3}), cls, 100*z + 23, (z,), "ZZ")
    assert_polynomial(polynomial.eval({-1: 2, -2: 3}), cls, 10*y + 203, (y,), "ZZ")
    assert polynomial.eval({z: 2, x: 3, y: 4}) == 243


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
@pytest.mark.parametrize("expression", ["zero", "constant", "variable"])
def test_profile_rational_assignments_promote_integer_domains_even_for_constant_results(cls, expression):
    x, y = sp.symbols("x y")
    value = {"zero": sp.Integer(0), "constant": sp.Integer(2), "variable": x*y + x + y}[expression]
    expected = {"zero": sp.Integer(0), "constant": sp.Integer(2), "variable": 3*y/2 + sp.Rational(1, 2)}[expression]
    polynomial = cls(value, x, y, domain="ZZ")
    assert_polynomial(polynomial.eval(sp.Rational(1, 2)), cls, expected, (y,), "QQ")
    with pytest.raises(DomainError):
        polynomial.eval(sp.Rational(1, 2), auto=False)
    with pytest.raises(DomainError):
        polynomial.eval(x, sp.Rational(1, 2), False)
    assert_polynomial(polynomial.eval(x, 2, False), cls, value.subs(x, 2), (y,), "ZZ")


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
def test_profile_explicit_rational_domain_is_retained_and_implicit_domain_is_inferred(cls):
    x, y = sp.symbols("x y")
    for value in [0, 2, x + y]:
        explicit = cls(value, x, y, domain="QQ").eval(2)
        implicit = cls(value, x, y).eval(2)
        expected = sp.sympify(value).subs(x, 2)
        assert_polynomial(explicit, cls, expected, (y,), "QQ")
        assert_polynomial(implicit, cls, expected, (y,), "ZZ")
    assert_polynomial(cls(x/2 + y, x, y).eval(2), cls, y + 1, (y,), "QQ")


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
@pytest.mark.parametrize("empty", [{}, [], ()])
def test_profile_empty_assignments_preserve_identity(cls, empty):
    x, y = sp.symbols("x y")
    polynomial = cls(x*y + 1, x, y, domain="QQ")
    assert polynomial.eval(empty) is polynomial
    assert polynomial.eval(empty, auto=False) is polynomial


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
def test_profile_projection_combines_columns_without_removing_unused_axes(cls):
    x, y, unused = sp.symbols("x y unused")
    polynomial = cls((x - 1)**2*y + 2, x, y, unused, domain="ZZ")
    result = polynomial.eval(1)
    assert_polynomial(result, cls, sp.Integer(2), (y, unused), "ZZ")
    assert result.eval([0, 99]) == 2
    zero = cls((x - 1)*y, x, y, unused, domain="ZZ").eval(1)
    assert_polynomial(zero, cls, sp.Integer(0), (y, unused), "ZZ")


def test_profile_partial_evaluation_composes_with_gcd_resultant_and_groebner():
    parameter, x, y = sp.symbols("parameter x y")
    first = sp.Poly((x + y)*(x + parameter), parameter, x, y, domain="QQ").eval(parameter, 2)
    second = sp.Poly((x + y)*(y + parameter), parameter, x, y, domain="QQ").eval(parameter, 2)
    gcd = first.gcd(second)
    assert_polynomial(gcd, sp.Poly, x + y, (x, y), "QQ")
    assert first.resultant(second).as_expr() == 0
    basis = sp.groebner([first.as_expr(), second.as_expr()], x, y, domain="QQ")
    assert basis.reduce(first.as_expr())[1] == 0
    assert basis.reduce(second.as_expr())[1] == 0


def test_profile_univariate_evaluation_returns_exact_integer_or_rational():
    x = sp.Symbol("x")
    polynomial = sp.Poly(3*x**3 - 2*x + 5, x, domain="QQ")
    assert polynomial.eval(2) == 25
    assert type(polynomial.eval(2)) is sp.Integer
    assert polynomial.eval(sp.Rational(1, 2)) == sp.Rational(35, 8)
    assert polynomial.eval(x=x, a=sp.Rational(-1, 3), auto=False) == sp.Rational(50, 9)


def test_typed_same_name_symbols_and_dummy_owners_survive_partial_publication():
    x = sp.Symbol("same_name", positive=True)
    other_x = sp.Symbol("same_name", negative=True)
    unused = sp.Dummy("unused")
    polynomial = sp.Poly(x + 2*other_x, x, other_x, unused, domain="QQ")
    result = polynomial.eval(x, 3)
    assert result.gens[0] is other_x
    assert result.gens[1] is unused
    assert result.as_expr() == 2*other_x + 3
    with pytest.raises(GeneratorsError):
        polynomial.eval(sp.Symbol("same_name"), 2)
    with pytest.raises(GeneratorsError):
        polynomial.eval("same_name", 2)


def test_degree_256_evaluation_does_not_use_generic_expression_expansion():
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x**254*y + x**256 + x**65, x, y, domain="ZZ")
    assert_polynomial(polynomial.eval(-1), sp.Poly, y, (y,), "ZZ")


def test_invalid_selectors_and_excess_values_are_refused_without_custom_coercion():
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x + y, x, y, domain="ZZ")
    for selector in [2, -3, sp.Integer(0), 0.0, sp.Symbol("unknown")]:
        with pytest.raises(GeneratorsError):
            polynomial.eval(selector, 2)
    for values in ([1, 2, 3], (1, 2, 3), {x: 1, y: 2, 0: 3}):
        with pytest.raises(ValueError, match="too many"):
            polynomial.eval(values)
    with pytest.raises(GeneratorsError):
        polynomial.eval({0: 2, x: 3})
    for invalid in [True, object()]:
        with pytest.raises(UnificationFailed):
            polynomial.eval(invalid)

    class Value:
        def _sympy_(self):
            raise AssertionError("custom conversion executed")

    with pytest.raises(UnificationFailed):
        polynomial.eval(Value())


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
def test_profile_existing_nonrational_univariate_scalar_substitutions_are_preserved(cls):
    x, y = sp.symbols("x y")
    polynomial = cls(x**2, x)
    assert polynomial.eval(y) == y**2
    assert polynomial.eval(sp.sqrt(2)) == 2
    assert polynomial.eval(sp.Float("1.5")) == sp.Float("2.25")
    assert polynomial.eval(1.5) == sp.Float("2.25")
    assert polynomial.eval(x, sp.sin(y)) == sp.sin(y)**2
    with pytest.raises(DomainError):
        polynomial.eval(y, auto=False)
    with pytest.raises(DomainError):
        polynomial.eval(sp.Float("1.5"), auto=False)


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
def test_profile_existing_parameter_coefficient_scalar_substitutions_are_preserved(cls):
    x, y = sp.symbols("x y")
    polynomial = cls(x + y, x)
    assert polynomial.eval(2) == y + 2
    assert polynomial.eval(2, auto=False) == y + 2


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
def test_profile_existing_string_scalar_selectors_are_preserved(cls):
    x = sp.Symbol("x")
    polynomial = cls(x**2 + 1, x)
    assert polynomial.eval("x", 2) == 5
    assert polynomial.eval("x", sp.Rational(1, 2)) == sp.Rational(5, 4)


def test_legacy_scalar_route_is_preselected_and_never_calls_exact_map(monkeypatch):
    from sympy.core import _native

    x, y = sp.symbols("x y")

    def unexpected(*args, **kwargs):
        raise AssertionError("legacy scalar request reached the checked rational map")

    monkeypatch.setattr(_native, "polynomial_evaluate_native", unexpected)
    assert sp.Poly(x**2, x).eval(y) == y**2
    assert sp.Poly(x + y, x).eval(2) == y + 2
    assert sp.Poly(x**2, x).eval("x", 2) == 4


@pytest.mark.parametrize("value", [2, 1.5])
def test_legacy_selection_does_not_invoke_mutated_metadata_truthiness(value):
    x = sp.Symbol("x")
    polynomial = sp.Poly(x + 1, x)

    class Metadata:
        def __bool__(self):
            raise AssertionError("mutable coefficient metadata was invoked")

        def __len__(self):
            raise AssertionError("mutable coefficient metadata was inspected")

    polynomial._extra_syms = Metadata()
    with pytest.raises(DomainError, match="metadata"):
        polynomial.eval(value)


@pytest.mark.parametrize("owner", [sp.Basic, sp.Symbol, sp.Dummy, sp.Integer, sp.Add, sp.Mul, sp.Pow])
@pytest.mark.parametrize("name", ["__hash__", "__eq__"])
def test_changed_builtin_identity_hooks_are_refused_without_invocation(monkeypatch, owner, name):
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x*y + 1, x, y)
    calls = []

    def unexpected(*args, **kwargs):
        calls.append(True)
        raise AssertionError("altered builtin identity hook executed")

    with monkeypatch.context() as patch:
        patch.setattr(owner, name, unexpected)
        with pytest.raises(TypeError, match="original built-in scalar owners"):
            polynomial.eval(x, 2)
        with pytest.raises(TypeError, match="original built-in scalar owners"):
            polynomial.eval(y)
    assert calls == []


def test_changed_in_place_builtin_implementation_is_refused(monkeypatch):
    x = sp.Symbol("x")
    polynomial = sp.Poly(x + 1, x)

    def unexpected(self):
        raise AssertionError("modified implementation body executed")

    with monkeypatch.context() as patch:
        patch.setattr(sp.Basic.__hash__, "__code__", unexpected.__code__)
        with pytest.raises(TypeError, match="original built-in scalar owners"):
            polynomial.eval(2)


@pytest.mark.parametrize("name", ["free_symbols", "subs"])
def test_changed_legacy_source_protocols_are_refused_without_invocation(monkeypatch, name):
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x**2, x)

    def unexpected(*args, **kwargs):
        raise AssertionError("altered source protocol executed")

    replacement = property(unexpected) if name == "free_symbols" else unexpected
    with monkeypatch.context() as patch:
        patch.setattr(sp.Basic, name, replacement)
        with pytest.raises(TypeError, match="original built-in scalar owners"):
            polynomial.eval(y)


def test_builtin_owners_remain_bound_through_native_publication(monkeypatch):
    from sympy.core import _native

    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x*y + 1, x, y)
    original = _native.polynomial_evaluate_native

    def unexpected(*args, **kwargs):
        raise AssertionError("altered builtin hash executed during publication")

    with monkeypatch.context() as patch:
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            patch.setattr(sp.Symbol, "__hash__", unexpected)
            return result

        patch.setattr(_native, "polynomial_evaluate_native", mutate)
        with pytest.raises(TypeError, match="original built-in scalar owners"):
            polynomial.eval(2)


def test_subclasses_and_overridden_source_protocols_do_not_enter_native_lane():
    x = sp.Symbol("x")

    class Custom(sp.Poly):
        def as_expr(self):
            raise AssertionError("custom source executed")

    with pytest.raises(TypeError):
        Custom(x + 1, x).eval(2)
    polynomial = sp.Poly(x + 1, x)
    polynomial.as_expr = lambda: (_ for _ in ()).throw(AssertionError("override executed"))
    with pytest.raises(TypeError):
        polynomial.eval(2)


def test_false_integer_declaration_and_unsupported_domains_remain_terminal():
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x/2 + y, x, y, domain="ZZ")
    with pytest.raises(CoercionFailed):
        polynomial.eval(0)
    with pytest.raises(CoercionFailed):
        polynomial.eval({})
    unsupported = sp.Poly(x + y, x, y)
    unsupported._domain = "RR"
    with pytest.raises(NotImplementedError):
        unsupported.eval(2)


@pytest.mark.parametrize("field,value", [("_domain", "ZZ"), ("_gens", ()), ("_domain_given", False), ("_extra_syms", frozenset({"changed"}))])
def test_source_mutation_after_native_work_prevents_publication(monkeypatch, field, value):
    from sympy.core import _native

    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x*y + 1, x, y, domain="QQ")
    original = _native.polynomial_evaluate_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        setattr(polynomial, field, value)
        return result

    monkeypatch.setattr(_native, "polynomial_evaluate_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        polynomial.eval(2)


def test_sequence_is_retained_before_scalar_conversion(monkeypatch):
    from sympy.polys import _evaluation

    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x*y + 1, x, y, domain="QQ")
    values = [sp.Integer(2)]
    original = _evaluation._native_expr

    def mutate(*args, **kwargs):
        values[0] = sp.Integer(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(_evaluation, "_native_expr", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        polynomial.eval(values)


def test_mapping_remains_bound_through_result_lifting(monkeypatch):
    from sympy.polys import _evaluation

    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x*y + 1, x, y, domain="QQ")
    values = {x: 2}
    original = _evaluation._lift_native_result

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        values[y] = 3
        return result

    monkeypatch.setattr(_evaluation, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="mapping mutated"):
        polynomial.eval(values)


def test_empty_assignment_still_admits_the_complete_original_source():
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x + y, x, y, domain="QQ")
    polynomial._expr = sp.sin(x)
    with pytest.raises((ValueError, NotImplementedError)):
        polynomial.eval({})


def test_large_exact_values_are_bounded_even_when_the_polynomial_is_zero():
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(0, x, y, domain="QQ")
    with pytest.raises(ValueError, match="4096"):
        polynomial.eval(1 << 4096)
    with pytest.raises(ValueError, match="4096"):
        polynomial.eval(sp.Rational(1, 1 << 4096))


def test_native_operational_failure_does_not_fall_back_to_substitution(monkeypatch):
    from sympy.core import _native

    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x*y + 1, x, y, domain="QQ")
    failure = ValueError("polynomial evaluation verification budget exhausted")

    def exhausted(*args, **kwargs):
        raise failure

    monkeypatch.setattr(_native, "polynomial_evaluate_native", exhausted)
    with pytest.raises(ValueError) as caught:
        polynomial.eval(2)
    assert caught.value is failure


def test_raw_bridge_admits_held_sources_and_all_values_before_zero_evaluation():
    from sympy.core import _native, _native_expr, _symbol_binding

    x, y = sp.symbols("x y")
    variables = [_symbol_binding(x), _symbol_binding(y)]
    for source in (
        sp.Mul(0, sp.sin(y), evaluate=False),
        sp.Pow(sp.sin(y), 0, evaluate=False),
        sp.Pow(0, 0, evaluate=False),
        sp.Pow(x, 257, evaluate=False),
        sp.Mul(0, sp.Pow(y, -1, evaluate=False), evaluate=False),
    ):
        with pytest.raises(ValueError):
            _native.polynomial_evaluate_native(_native_expr(source), variables, [_native_expr(0), None])
    for invalid in (_native_expr(sp.Float(0.5)), _native_expr(1 << 4096), _native_expr(x)):
        with pytest.raises(ValueError):
            _native.polynomial_evaluate_native(_native_expr(0), variables, [invalid, None])
    with pytest.raises(ValueError):
        _native.polynomial_evaluate_native(_native_expr(0), variables, [None])
