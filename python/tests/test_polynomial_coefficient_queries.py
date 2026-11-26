"""Exact coefficient projections, monomial ordering and retained query sources.

Profile fixtures run unchanged against isolated SymPy 1.14.0. Native-only
cases cover deliberately malformed retained sources and resource boundaries.
"""

import pytest
import sympy as sp
from sympy.polys.polyerrors import CoercionFailed, PolynomialError


NATIVE = hasattr(sp.Integer(1), "_value")
native_only = pytest.mark.skipif(not NATIVE, reason="FrankenSymPy retained-source boundary")
CLASSES = (sp.Poly, sp.PurePoly)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("domain", ("QQ", "ZZ"))
@pytest.mark.parametrize("order", ("lex", "grlex", "grevlex"))
def test_profile_multivariate_queries_share_the_ordered_exact_support(cls, domain, order):
    x, y = sp.symbols("x y")
    polynomial = cls(2*x**2*y + 3*x*y**4 + 5*y + 7, x, y, domain=domain)
    expected = [((2, 1), 2), ((1, 4), 3), ((0, 1), 5), ((0, 0), 7)]
    if order != "lex":
        expected[0], expected[1] = expected[1], expected[0]
    assert polynomial.terms(order=order) == expected
    assert polynomial.monoms(order=order) == [powers for powers, _ in expected]
    assert polynomial.coeffs(order=order) == [coefficient for _, coefficient in expected]
    assert polynomial.LC(order=order) == expected[0][1]
    assert polynomial.EC(order=order) == expected[-1][1]
    assert polynomial.TC() == 7
    assert polynomial.nth(2, 1) == 2
    assert polynomial.nth(sp.Integer(1), sp.Integer(4)) == 3
    assert polynomial.nth(2, 2) == 0
    assert polynomial.coeff_monomial(x**2*y) == 2
    assert polynomial.coeff_monomial((1, 4)) == 3
    assert polynomial.coeff_monomial([0, 1]) == 5
    assert polynomial.coeff_monomial(1) == 7


@pytest.mark.parametrize("cls", CLASSES)
def test_profile_reverse_graded_order_uses_the_last_differing_axis(cls):
    x, y, z = sp.symbols("x y z")
    polynomial = cls(x**2*y + 2*x*z**2 + 3*y**3, x, y, z, domain="QQ")
    lexicographic = [((2, 1, 0), 1), ((1, 0, 2), 2), ((0, 3, 0), 3)]
    reversed_graded = [((2, 1, 0), 1), ((0, 3, 0), 3), ((1, 0, 2), 2)]
    assert polynomial.terms(order="lex") == lexicographic
    assert polynomial.terms(order="grlex") == lexicographic
    assert polynomial.terms(order="grevlex") == reversed_graded
    assert polynomial.EC(order="lex") == 3
    assert polynomial.EC(order="grevlex") == 2


@pytest.mark.parametrize("cls", CLASSES)
def test_profile_unexpanded_powers_and_rational_coefficients_are_projected_exactly(cls):
    x, y = sp.symbols("x y")
    polynomial = cls((x+y)**4/7 + x*y/13, x, y, domain="QQ")
    assert polynomial.terms() == [
        ((4, 0), sp.Rational(1, 7)), ((3, 1), sp.Rational(4, 7)),
        ((2, 2), sp.Rational(6, 7)), ((1, 3), sp.Rational(4, 7)),
        ((1, 1), sp.Rational(1, 13)), ((0, 4), sp.Rational(1, 7)),
    ]
    assert polynomial.coeff_monomial(x*y) == sp.Rational(1, 13)
    assert polynomial.nth(2, 2) == sp.Rational(6, 7)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("count", (1, 3))
@pytest.mark.parametrize("value", (0, 5))
def test_profile_zero_and_constant_polynomials_keep_their_declared_axes(cls, count, value):
    generators = sp.symbols("x y z")[:count]
    polynomial = cls(value, *generators, domain="QQ")
    zero_powers = (0,) * count
    assert polynomial.terms() == [(zero_powers, value)]
    assert polynomial.monoms() == [zero_powers]
    assert polynomial.coeffs() == [value]
    assert polynomial.LC() == polynomial.EC() == polynomial.TC() == value
    assert polynomial.nth(*zero_powers) == value
    assert polynomial.nth(*((10**100,) * count)) == 0
    assert polynomial.coeff_monomial(1) == value
    if count == 1:
        assert polynomial.all_coeffs() == [value]
    else:
        with pytest.raises(PolynomialError):
            polynomial.all_coeffs()


@pytest.mark.parametrize("cls", CLASSES)
def test_profile_dense_univariate_coefficients_include_holes_and_ec_is_not_tc(cls):
    x = sp.Symbol("x")
    polynomial = cls(x**4 + 2*x**2, x, domain="ZZ")
    assert polynomial.all_coeffs() == [1, 0, 2, 0, 0]
    assert polynomial.coeffs() == [1, 2]
    assert polynomial.LC() == 1
    assert sp.LC(polynomial) == 1
    assert polynomial.EC() == 2
    assert polynomial.TC() == 0
    assert polynomial.nth(3) == polynomial.coeff_monomial(x**3) == 0
    assert polynomial.nth(4) == 1
    with pytest.raises(IndexError):
        polynomial.nth(-1)


@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("denominator", (1, 3))
def test_profile_explicit_none_domain_matches_automatic_coefficients_and_roots(cls, denominator):
    x = sp.Symbol("x")
    expression = (x - 1)*(x + 2)/denominator
    automatic = cls(expression, x)
    explicit_none = cls(expression, x, domain=None)
    assert str(explicit_none.domain) == str(automatic.domain) == (
        "ZZ" if denominator == 1 else "QQ"
    )
    assert explicit_none.all_coeffs() == automatic.all_coeffs() == [
        sp.Rational(1, denominator), sp.Rational(1, denominator),
        sp.Rational(-2, denominator),
    ]
    assert explicit_none.terms() == automatic.terms()
    assert explicit_none.nroots() == automatic.nroots() == [sp.Float(-2), sp.Float(1)]


@native_only
@pytest.mark.parametrize("cls", CLASSES)
@pytest.mark.parametrize("operation", ("all_coeffs", "nroots"))
def test_mutated_none_domain_is_refused_after_polynomial_construction(cls, operation):
    x = sp.Symbol("x")
    polynomial = cls(x**2 + x - 2, x)
    polynomial._domain = None
    with pytest.raises(NotImplementedError, match="supported exact coefficient domain"):
        getattr(polynomial, operation)()


@pytest.mark.parametrize("cls", CLASSES)
def test_profile_reordered_and_unused_axes_define_the_coefficient_coordinates(cls):
    x, y, unused = sp.symbols("x y unused")
    polynomial = cls(2*x**2*y + 3*y + 7, y, unused, x, domain="QQ")
    assert polynomial.terms() == [((1, 0, 2), 2), ((1, 0, 0), 3), ((0, 0, 0), 7)]
    assert polynomial.nth(1, 0, 2) == 2
    assert polynomial.nth(1, 1, 2) == 0
    assert polynomial.coeff_monomial(x**2*y) == 2
    assert polynomial.coeff_monomial(unused) == 0
    assert polynomial.gens == (y, unused, x)


@pytest.mark.parametrize("cls", CLASSES)
def test_profile_typed_same_name_symbols_and_distinct_dummies_do_not_alias(cls):
    positive = sp.Symbol("query_same", positive=True)
    negative = sp.Symbol("query_same", negative=True)
    first, unused = sp.Dummy("query_same"), sp.Dummy("query_same")
    axes = (positive, negative, first, unused)
    polynomial = cls(2*positive**2*negative + 3*first + 7, *axes, domain="QQ")
    assert polynomial.terms() == [
        ((2, 1, 0, 0), 2), ((0, 0, 1, 0), 3), ((0, 0, 0, 0), 7),
    ]
    assert polynomial.coeff_monomial(positive**2*negative) == 2
    assert polynomial.coeff_monomial(first) == 3
    assert polynomial.coeff_monomial(unused) == 0
    assert all(actual is original for actual, original in zip(polynomial.gens, axes))
    with pytest.raises((ValueError, PolynomialError)):
        polynomial.coeff_monomial(sp.Symbol("query_same"))


def test_profile_explicit_dummy_index_keeps_the_original_coordinate_owner():
    variable = sp.Dummy("query_index", dummy_index=True)
    polynomial = sp.Poly(3*variable+2, variable, domain="QQ")
    assert polynomial.gens[0] is variable
    assert polynomial.terms() == [((1,), 3), ((0,), 2)]
    assert polynomial.coeff_monomial(variable) == 3


@pytest.mark.parametrize("cls", CLASSES)
def test_profile_prime_field_queries_keep_the_existing_modular_coefficients(cls):
    x, y = sp.symbols("x y")
    polynomial = cls(7*x**2*y - 3*y + 5, x, y, modulus=5)
    assert polynomial.terms() == [((2, 1), 2), ((0, 1), 2)]
    assert polynomial.coeffs(order="grevlex") == [2, 2]
    assert polynomial.LC(order="grlex") == 2
    assert polynomial.EC() == 2 and polynomial.TC() == 0
    assert polynomial.nth(2, 1) == polynomial.coeff_monomial(x**2*y) == 2
    with pytest.raises(ValueError):
        polynomial.coeff_monomial(6*x**2*y)


@pytest.mark.parametrize("domain", (None, "ZZ[y]", "QQ(y)"))
def test_profile_existing_parameter_coefficient_surface_and_domain_are_preserved(domain):
    x, y = sp.symbols("x y")
    options = {} if domain is None else {"domain": domain}
    polynomial = sp.Poly(x*y + 1, x, **options)
    assert polynomial.terms() == [((1,), y), ((0,), 1)]
    assert polynomial.coeffs() == [y, 1]
    assert polynomial.LC() == polynomial.nth(1) == polynomial.coeff_monomial(x) == y
    assert polynomial.TC() == polynomial.EC() == 1
    assert str(polynomial.domain) == (domain or "ZZ[y]")
    assert "domain=" in repr(polynomial)


@pytest.mark.parametrize("coefficient", (sp.I, sp.pi, sp.sqrt(2)))
def test_profile_existing_inferred_generic_numeric_coefficients_still_render(coefficient):
    x = sp.Symbol("x")
    polynomial = sp.Poly(coefficient*x+1, x)
    assert polynomial.terms() == [((1,), coefficient), ((0,), 1)]
    assert polynomial.LC() == coefficient
    assert polynomial.TC() == 1
    assert "domain=" in repr(polynomial)


def test_profile_existing_function_parameter_coefficients_still_render():
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(sp.sin(y)*x+1, x)
    assert polynomial.terms() == [((1,), sp.sin(y)), ((0,), 1)]
    assert polynomial.coeffs() == [sp.sin(y), 1]
    assert polynomial.LC() == sp.sin(y)
    assert "domain=" in repr(polynomial)


@pytest.mark.parametrize("query", ("coeffs", "terms", "monoms", "LC", "EC"))
def test_profile_invalid_order_is_refused(query):
    x, y = sp.symbols("x y")
    with pytest.raises(ValueError):
        getattr(sp.Poly(x+y, x, y), query)(order="unknown")


@pytest.mark.parametrize("coordinates", ((1,), (1, 0, 0)))
def test_profile_nth_requires_one_exponent_per_axis(coordinates):
    x, y = sp.symbols("x y")
    with pytest.raises(ValueError):
        sp.Poly(x+y, x, y).nth(*coordinates)


@pytest.mark.parametrize("selector", ("zero", "multiple", "scaled", "negative", "fractional"))
def test_profile_coeff_monomial_requires_one_coefficient_one_monomial(selector):
    x, y = sp.symbols("x y")
    selected = {
        "zero": 0, "multiple": x+y, "scaled": 2*x,
        "negative": x**-1, "fractional": x**sp.Rational(1, 2),
    }[selector]
    with pytest.raises((ValueError, PolynomialError)):
        sp.Poly(x+y, x, y, domain="QQ").coeff_monomial(selected)


@native_only
def test_existing_module_ec_extension_agrees_with_the_poly_method():
    from sympy.polys.polytools import EC, TC

    x = sp.Symbol("x")
    polynomial = sp.Poly(x**2+2*x, x, domain="QQ")
    assert EC(polynomial) == EC(x**2+2*x, x) == 2
    assert TC(polynomial) == 0


@native_only
def test_checked_queries_do_not_use_generic_python_expansion(monkeypatch):
    from sympy.polys import _coefficient_queries

    x, y = sp.symbols("x y")
    polynomial = sp.Poly((x+y)**4 + x**256, x, y, domain="QQ")

    def forbidden(*args, **kwargs):
        raise AssertionError("checked coefficient query used the legacy expansion path")

    monkeypatch.setattr(_coefficient_queries, "_legacy_terms", forbidden)
    assert polynomial.nth(256, 0) == 1
    assert polynomial.nth(2, 2) == 6
    assert polynomial.coeff_monomial(x*y**3) == 4


@native_only
@pytest.mark.parametrize("operation", ("terms", "coeffs", "LC", "EC", "TC", "nth", "coeff_monomial"))
def test_false_integer_declaration_cannot_be_hidden_by_an_absent_requested_coefficient(operation):
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x/2 + y, x, y, domain="QQ")
    polynomial._domain = "ZZ"
    args = (99, 99) if operation == "nth" else (x**99,) if operation == "coeff_monomial" else ()
    with pytest.raises(CoercionFailed):
        getattr(polynomial, operation)(*args)


@native_only
@pytest.mark.parametrize("domain", ("RR", "EX", "QQ<sqrt(2)>"))
def test_existing_nonrational_declared_surfaces_do_not_enter_the_exact_map(monkeypatch, domain):
    from sympy.polys import _coefficient_queries

    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x+y, x, y, domain="QQ")
    polynomial._domain = domain

    def forbidden(*args, **kwargs):
        raise AssertionError("legacy coefficient surface entered the exact QQ map")

    monkeypatch.setattr(_coefficient_queries, "_project", forbidden)
    assert polynomial.terms() == [((1, 0), 1), ((0, 1), 1)]
    assert polynomial.coeffs() == [1, 1]
    assert polynomial.LC() == polynomial.EC() == 1
    assert polynomial.TC() == 0
    assert str(polynomial.domain) == domain
    assert "domain=" in repr(polynomial)


@native_only
def test_unknown_declared_domain_is_not_reinterpreted_as_qq():
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x+y, x, y, domain="QQ")
    polynomial._domain = "UNSUPPORTED"
    with pytest.raises(NotImplementedError):
        polynomial.coeffs()
    with pytest.raises(NotImplementedError):
        polynomial.coeff_monomial(x)


@native_only
def test_explicit_qq_does_not_silently_gain_coefficient_parameters():
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x*y+1, x, domain="QQ")
    with pytest.raises(ValueError):
        polynomial.terms()


@native_only
@pytest.mark.parametrize("forged", ("x+y", [1, 2], object()))
def test_forged_source_storage_is_refused_before_preselection_conversion(monkeypatch, forged):
    from sympy.polys import _coefficient_queries

    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x+y, x, y)
    polynomial._expr = forged

    def forbidden(*args, **kwargs):
        raise AssertionError("forged Poly source reached native conversion")

    monkeypatch.setattr(_coefficient_queries, "_native_expr", forbidden)
    with pytest.raises(TypeError):
        polynomial.terms()


@native_only
@pytest.mark.parametrize("wrapper", ("zero_product", "zero_power", "cancelled"))
def test_original_held_undefined_sources_are_refused_before_coefficient_projection(wrapper):
    x, y = sp.symbols("x y")
    pole = sp.Pow(sp.Integer(0), sp.Integer(-1), evaluate=False)
    expression = {
        "zero_product": sp.Mul(0, pole, evaluate=False),
        "zero_power": sp.Pow(pole, 0, evaluate=False),
        "cancelled": sp.Add(pole, sp.Mul(-1, pole, evaluate=False), evaluate=False),
    }[wrapper]
    polynomial = sp.Poly(0, x, y, domain="QQ")
    polynomial._expr = expression
    for operation, args in (("terms", ()), ("TC", ()), ("nth", (99, 99)), ("coeff_monomial", (x,))):
        with pytest.raises((ValueError, ZeroDivisionError)):
            getattr(polynomial, operation)(*args)


@native_only
def test_custom_controls_and_owners_are_refused_without_invoking_conversion(monkeypatch):
    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x+y, x, y, domain="QQ")
    calls = []

    class Control:
        def __int__(self):
            calls.append("int")
            return 1

        def _sympy_(self):
            calls.append("sympy")
            return x

    with pytest.raises(CoercionFailed):
        polynomial.nth(Control(), 0)
    with pytest.raises(ValueError):
        polynomial.coeff_monomial(Control())
    for value in (True, 1.0, "1", sp.Rational(3, 2)):
        with pytest.raises(CoercionFailed):
            polynomial.nth(value, 0)

    class CustomPoly(sp.Poly):
        def as_expr(self):
            calls.append("as_expr")
            return x

    with pytest.raises(TypeError):
        CustomPoly(x+y, x, y).coeffs()

    def forbidden(*args, **kwargs):
        calls.append("hash")
        raise AssertionError("changed Symbol hash executed")

    with monkeypatch.context() as patch:
        patch.setattr(sp.Symbol, "__hash__", forbidden)
        with pytest.raises(TypeError):
            polynomial.coeffs()
    assert calls == []


@native_only
@pytest.mark.parametrize("field", ("_domain", "_gens", "_expr"))
def test_original_polynomial_state_is_bound_through_native_projection(monkeypatch, field):
    from sympy.polys import _coefficient_queries

    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x+y, x, y, domain="QQ")
    original = _coefficient_queries._project

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        setattr(polynomial, field, {"_domain": "ZZ", "_gens": (y, x), "_expr": x-y}[field])
        return result

    monkeypatch.setattr(_coefficient_queries, "_project", mutate)
    with pytest.raises(RuntimeError, match="mutat"):
        polynomial.terms()


@native_only
def test_monomial_query_retains_both_sources_before_either_native_projection(monkeypatch):
    from sympy.polys import _coefficient_queries

    x, y = sp.symbols("x y")
    polynomial = sp.Poly(2*x*y+1, x, y, domain="QQ")
    original = _coefficient_queries._project

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        polynomial._expr = 3*x*y+1
        return result

    monkeypatch.setattr(_coefficient_queries, "_project", mutate)
    with pytest.raises(RuntimeError, match="mutat"):
        polynomial.coeff_monomial(x*y)


@native_only
def test_exponent_list_is_retained_until_the_result_is_published(monkeypatch):
    from sympy.polys import _coefficient_queries

    x, y = sp.symbols("x y")
    polynomial = sp.Poly(2*x+3*y, x, y, domain="QQ")
    coordinates = [1, 0]
    original = _coefficient_queries._project

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        coordinates[:] = [0, 1]
        return result

    monkeypatch.setattr(_coefficient_queries, "_project", mutate)
    with pytest.raises(RuntimeError, match="mutat"):
        polynomial.coeff_monomial(coordinates)


@native_only
@pytest.mark.parametrize("explicit", (False, True))
@pytest.mark.parametrize("kind", ("assumptions", "dummy_name", "dummy_index"))
def test_stale_typed_generator_metadata_is_not_rebound(kind, explicit, monkeypatch):
    if kind == "assumptions":
        variable = sp.Symbol("query_stale", positive=True)
    else:
        variable = sp.Dummy("query_stale")
    polynomial = sp.Poly(variable+1, variable, **({"domain": "QQ"} if explicit else {}))
    if kind == "assumptions":
        monkeypatch.setitem(variable._assumptions, "positive", False)
    elif kind == "dummy_name":
        monkeypatch.setattr(variable, "_dummy_name", "changed")
    else:
        monkeypatch.setattr(variable, "_dummy_number", variable._dummy_number+1)
    with pytest.raises(ValueError):
        polynomial.coeffs()


@native_only
def test_operational_failure_is_terminal_and_never_retries_the_legacy_lane(monkeypatch):
    from sympy.polys import _coefficient_queries

    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x+y, x, y, domain="QQ")
    failure = RuntimeError("coefficient request cancelled")

    def fail(*args, **kwargs):
        raise failure

    def fallback(*args, **kwargs):
        raise AssertionError("native refusal retried through legacy expansion")

    monkeypatch.setattr(_coefficient_queries, "_project", fail)
    monkeypatch.setattr(_coefficient_queries, "_legacy_terms", fallback)
    with pytest.raises(RuntimeError) as caught:
        polynomial.coeffs()
    assert caught.value is failure


@native_only
@pytest.mark.parametrize("cancelled", (False, True))
def test_zero_or_cancelled_request_cannot_reach_native_projection(monkeypatch, cancelled):
    from sympy.core import supervised_callback_session
    from sympy.polys import _coefficient_queries

    x, y = sp.symbols("x y")
    polynomial = sp.Poly(x+y, x, y, domain="QQ")

    def forbidden(*args, **kwargs):
        raise AssertionError("exhausted request reached native polynomial projection")

    monkeypatch.setattr(_coefficient_queries, "_project", forbidden)
    with supervised_callback_session(max_nested_steps=1_000_000 if cancelled else 0) as session:
        if cancelled:
            session.cancel()
        with pytest.raises((RuntimeError, ValueError)):
            polynomial.coeffs()


@native_only
def test_native_projection_work_accumulates_across_queries_on_one_request():
    from sympy.core import supervised_callback_session

    x, y = sp.symbols("x y")
    polynomial = sp.Poly((x+y)**6 + x*y/7, x, y, domain="QQ")
    polynomial.terms()
    with supervised_callback_session(max_nested_steps=20_000_000) as measured:
        polynomial.terms()
        cost = measured.consumed_nested_steps
    assert cost > 0
    with supervised_callback_session(max_nested_steps=2*cost-1) as limited:
        result = polynomial.terms()
        assert limited.consumed_nested_steps > 0
        with pytest.raises((RuntimeError, ValueError)):
            polynomial.terms()
        assert limited.cancelled or limited.unknown_effect_tainted
        with pytest.raises((RuntimeError, ValueError)):
            polynomial.coeffs()
    assert len(result) == 8

