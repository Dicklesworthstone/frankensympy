"""Prime-field elimination through the checked public polynomial interface.

Algebraic identities and public result conventions are also runnable against
the pinned SymPy 1.14.0 oracle. Six observed upstream algebraic failures are
strict expected divergences; they are never compatibility passes. Native-only cases exercise original sources,
typed rings, interface integrity and the explicitly bounded native lane.
"""

import pytest
import sympy as sp

from sympy.polys.polyerrors import CoercionFailed, GeneratorsError, UnificationFailed

try:
    from sympy.core import _native, _native_expr, _symbol_binding
    from sympy.polys import _finitefield
except ImportError:
    _native = None
    _finitefield = None


native_only = pytest.mark.skipif(_native is None, reason="requires the FrankenSymPy extension")


def polynomial(value, generators, prime, symmetric=True, cls=sp.Poly):
    return cls(value, *generators, modulus=prime, symmetric=symmetric)


@pytest.mark.parametrize("prime", [2, 3, 5, 7, 65521])
@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
def test_quadratic_linear_resultant_keeps_the_coefficient_ring(prime, cls):
    x, a, b, c, d, e = sp.symbols("ff_resultant_x ff_a ff_b ff_c ff_d ff_e")
    gens = (x, a, b, c, d, e)
    left = polynomial(a*x**2 + b*x + c, gens, prime, cls=cls)
    right = polynomial(d*x + e, gens, prime, cls=cls)
    expected = polynomial(a*e**2 - b*d*e + c*d**2, gens[1:], prime, cls=cls)
    for result in (left.resultant(right), sp.resultant(left, right)):
        assert type(result) is cls
        assert result.gens == gens[1:]
        assert result.get_modulus() == prime
        assert result == expected
    assert sp.resultant(left, right, polys=False) == expected.as_expr()
    assert sp.resultant(left, right, polys=True) == expected


@pytest.mark.parametrize("prime", [2, 3, 5, 7, 65521])
@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
def test_quadratic_discriminant_is_the_universal_coefficient_invariant(prime, cls, request):
    if _native is None and prime == 2:
        request.node.add_marker(pytest.mark.xfail(
            strict=True, raises=ZeroDivisionError,
            reason="SymPy 1.14.0 divides by zero for a nonmonic GF(2) quadratic discriminant"))
    x, a, b, c = sp.symbols("ff_disc_x ff_disc_a ff_disc_b ff_disc_c")
    gens = (x, a, b, c)
    source = polynomial(a*x**2 + b*x + c, gens, prime, cls=cls)
    expected = polynomial(b**2 - 4*a*c, gens[1:], prime, cls=cls)
    for result in (source.discriminant(), sp.discriminant(source)):
        assert type(result) is cls and result.gens == gens[1:]
        assert result.get_modulus() == prime and result == expected
    assert sp.discriminant(source, polys=False) == expected.as_expr()
    # In characteristic two the derivative is constant, but the universal
    # discriminant is b**2, without a spurious inverse leading coefficient.
    if prime == 2:
        assert expected.as_expr() == b**2


@pytest.mark.parametrize("prime", [2, 3, 5, 7])
@pytest.mark.parametrize("degree", [3, 4])
def test_cubic_and_quartic_discriminants_survive_derivative_degree_drop(prime, degree, request):
    if _native is None and degree % prime == 0:
        from sympy.polys.polyerrors import PolynomialDivisionFailed
        request.node.add_marker(pytest.mark.xfail(
            strict=True, raises=PolynomialDivisionFailed,
            reason="SymPy 1.14.0 fails polynomial division after derivative degree drops"))
    x, a, b, c = sp.symbols("ff_higher_x ff_higher_a ff_higher_b ff_higher_c")
    source = polynomial(a*x**degree + b*x + c, (x, a, b, c), prime)
    formula = -4*a*b**3 - 27*a**2*c**2 if degree == 3 else 256*a**3*c**3 - 27*a**2*b**4
    expected = polynomial(formula, (a, b, c), prime)
    assert source.discriminant() == expected


@pytest.mark.parametrize("prime", [2, 3, 5])
def test_resultant_reversal_sign_and_multiplicativity(prime, request):
    if _native is None and prime != 2:
        request.node.add_marker(pytest.mark.xfail(
            strict=True, raises=AssertionError,
            reason="SymPy 1.14.0 loses the odd-degree resultant reversal sign"))
    x, y, z = sp.symbols("ff_sign_x ff_sign_y ff_sign_z")
    gens = (x, y, z)
    left = polynomial(x**3 + y*x + z, gens, prime)
    right = polynomial(x + y, gens, prime)
    expected = polynomial(y**3 + y**2 - z, (y, z), prime)
    assert left.resultant(right) == expected
    assert right.resultant(left) == -expected
    second = polynomial(x + z, gens, prime)
    assert (left*second).resultant(right) == left.resultant(right)*second.resultant(right)
    assert (left*right).resultant(right).is_zero


@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
def test_result_types_generator_owners_and_signed_lifts(symmetric, cls):
    x = sp.Symbol("ff_elimination_identity")
    positive = sp.Symbol("ff_elimination_identity", positive=True)
    unused = sp.Dummy("ff_elimination_identity")
    gens = (x, positive, unused)
    left = polynomial(x + positive, gens, 5, symmetric, cls)
    right = polynomial(x, gens, 5, symmetric, cls)
    result = left.resultant(right)
    assert type(result) is cls
    assert result.gens[0] is positive and result.gens[1] is unused
    assert result.as_expr() == (-1 if symmetric else 4)*positive
    assert result.get_modulus() == 5
    assert x not in result.as_expr().free_symbols


@pytest.mark.parametrize("polys", [False, True])
@pytest.mark.parametrize("domain_option", ["modulus", "domain"])
def test_expression_entries_accept_explicit_and_inferred_full_rings(polys, domain_option):
    x, y, z = sp.symbols("aa_ff_x bb_ff_y cc_ff_z")
    options = {"modulus": 5} if domain_option == "modulus" else {"domain": "GF(5)"}
    left, right = x**2 + y*x + z, x + y
    for args in ((), (x, y, z), ([x, y, z],)):
        result = sp.resultant(left, right, *args, polys=polys, **options)
        disc = sp.discriminant(left, *args, polys=polys, **options)
        if polys:
            assert type(result) is sp.Poly and result.gens == (y, z)
            assert result.as_expr() == z
            assert type(disc) is sp.Poly and disc.gens == (y, z)
            assert disc.as_expr() == y**2 + z
        else:
            assert result == z and disc == y**2 + z


def test_declared_primary_axis_controls_elimination():
    x, y = sp.symbols("ff_axis_x ff_axis_y")
    for gens, value in [((x, y), y**2 + y), ((y, x), x - x**2)]:
        result = polynomial(x**2 + y, gens, 5).resultant(polynomial(x + y, gens, 5))
        assert result.gens == gens[1:]
        assert result == polynomial(value, gens[1:], 5)


@pytest.mark.parametrize("left,right,expected", [
    (0, 0, 0), (0, 3, 0), (2, 0, 0), (2, 3, 1),
])
def test_zero_and_constant_resultants_keep_unused_axes(left, right, expected):
    x, y, z = sp.symbols("ff_const_x ff_const_y ff_const_z")
    gens = (x, y, z)
    result = polynomial(left, gens, 5).resultant(polynomial(right, gens, 5))
    assert result.gens == (y, z) and result.as_expr() == expected


@pytest.mark.parametrize("prime", [2, 5])
def test_symbolic_constant_resultants_and_linear_discriminants(prime):
    x, y, z = sp.symbols("ff_coefficient_x ff_coefficient_y ff_coefficient_z")
    gens = (x, y, z)
    constant = polynomial(y, gens, prime)
    cubic = polynomial(z*x**3 + y*x + 1, gens, prime)
    expected = polynomial(y**3, (y, z), prime)
    assert constant.resultant(cubic) == expected
    assert cubic.resultant(constant) == expected
    assert constant.resultant(polynomial(z, gens, prime)).as_expr() == 1
    assert constant.discriminant().as_expr() == 0
    assert polynomial(0, gens, prime).discriminant().as_expr() == 0
    assert polynomial(y*x + z, gens, prime).discriminant().as_expr() == 1


@pytest.mark.parametrize("polys", [False, True])
def test_univariate_elimination_remains_a_scalar(polys):
    x = sp.Symbol("ff_scalar_x")
    source = sp.Poly(x**2 + 2*x + 3, x, modulus=5)
    assert source.resultant(sp.Poly(x, x, modulus=5)) == -2
    assert source.discriminant() == 2
    assert sp.resultant(source, sp.Poly(x, x, modulus=5), polys=polys) == -2
    assert sp.discriminant(source, polys=polys) == 2


@native_only
@pytest.mark.parametrize("operation", ["resultant", "discriminant"])
def test_multivariate_operations_do_not_use_rational_or_univariate_fallbacks(monkeypatch, operation):
    x, y = sp.symbols("ff_route_x ff_route_y")
    source = polynomial(x**2 + y*x + 1, (x, y), 5)
    other = polynomial(x + y, (x, y), 5)

    def forbidden(*args, **kwargs):
        raise AssertionError("multivariate prime-field elimination entered another arithmetic lane")

    for name in ("prime_field_polynomial_native", "poly_resultant_native", "poly_discriminant_native"):
        monkeypatch.setattr(_native, name, forbidden)
    result = source.resultant(other) if operation == "resultant" else source.discriminant()
    assert result.as_expr() == (1 if operation == "resultant" else y**2 + 1)


@native_only
def test_mismatched_fields_and_ordered_typed_rings_refuse():
    x, y = sp.symbols("ff_mismatch_x ff_mismatch_y")
    shadow = sp.Symbol("ff_mismatch_x", positive=True)
    source = polynomial(x + y, (x, y), 5)
    for other in (sp.Poly(x + y, x, y), polynomial(x + y, (x, y), 7),
                  polynomial(x + y, (y, x), 5), polynomial(shadow + y, (shadow, y), 5)):
        with pytest.raises(UnificationFailed):
            source.resultant(other)
    with pytest.raises(UnificationFailed):
        source.resultant(x + y)


@native_only
@pytest.mark.parametrize("operation", ["resultant", "discriminant"])
@pytest.mark.parametrize("kind", ["reciprocal", "float", "function"])
def test_original_unsupported_children_cannot_disappear_behind_zero(operation, kind):
    x, y = sp.symbols("ff_hidden_x ff_hidden_y")
    bad = {"reciprocal": sp.Pow(y, -1, evaluate=False),
           "float": sp.Float("1.0"), "function": sp.sin(y)}[kind]
    source = polynomial(0, (x, y), 5)
    other = polynomial(1, (x, y), 5)
    source._expr = sp.Mul(0, bad, evaluate=False)
    with pytest.raises((ValueError, CoercionFailed, NotImplementedError)):
        source.resultant(other) if operation == "resultant" else source.discriminant()


@native_only
@pytest.mark.parametrize("operation", ["resultant", "discriminant"])
def test_original_unused_axis_cannot_be_erased_by_explicit_options(operation):
    x, unused = sp.Symbol("ff_retained_x"), sp.Dummy("ff_retained_unused")
    source = polynomial(x**2, (x, unused), 5)
    unused._assumptions["commutative"] = False
    with pytest.raises((GeneratorsError, UnificationFailed)):
        if operation == "resultant":
            sp.resultant(source, 1, x, modulus=5)
        else:
            sp.discriminant(source, x, modulus=5)


@native_only
@pytest.mark.parametrize("operation", ["resultant", "discriminant"])
def test_substituted_native_interface_is_not_executed(monkeypatch, operation):
    x, y = sp.symbols("ff_interface_x ff_interface_y")
    source = polynomial(x**2 + y, (x, y), 5)
    other = polynomial(x, (x, y), 5)
    calls = []

    def replacement(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("a replacement verifier must not execute")

    monkeypatch.setattr(_native, "prime_field_multivariate_native", replacement)
    with pytest.raises(NotImplementedError, match="original native interface"):
        source.resultant(other) if operation == "resultant" else source.discriminant()
    assert calls == []


@native_only
@pytest.mark.parametrize("target", ["source", "other", "ring", "modulus", "symmetric", "container", "interface"])
@pytest.mark.parametrize("polys", [False, True])
def test_source_and_interface_are_retained_through_publication(monkeypatch, target, polys):
    x, y = sp.symbols("ff_publish_x ff_publish_y")
    source = polynomial(x**2 + y, (x, y), 5)
    other = polynomial(x + 1, (x, y), 5)
    generators = [x, y]
    original = _finitefield._lift_native_result

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if target == "source":
            source._expr = x
        elif target == "other":
            other._expr = x
        elif target == "ring":
            source._gens = (y, x)
        elif target == "modulus":
            source._modulus = 7
        elif target == "symmetric":
            source._symmetric = False
        elif target == "container":
            generators.reverse()
        else:
            monkeypatch.setattr(_native, "prime_field_multivariate_native", lambda *a, **k: [])
        return result

    monkeypatch.setattr(_finitefield, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="mutated|interface changed"):
        sp.resultant(source, other, generators, polys=polys)


@native_only
@pytest.mark.parametrize("operation", ["resultant", "discriminant"])
def test_active_callback_cannot_start_an_untracked_elimination_budget(operation):
    from sympy.core import SupervisedCallbackSession, supervised_callback_session

    x, y = sp.symbols("ff_callback_x ff_callback_y")
    source = polynomial(x**2 + y, (x, y), 5)
    other = polynomial(x, (x, y), 5)
    session = SupervisedCallbackSession(max_nested_steps=20_000_000)
    with supervised_callback_session(session):
        with pytest.raises(NotImplementedError, match="supervised callback lane"):
            source.resultant(other) if operation == "resultant" else source.discriminant()


@native_only
def test_sylvester_dimension_bound_and_native_controls_are_explicit():
    x, y = sp.symbols("ff_bound_x ff_bound_y")
    left = polynomial(x**5 + y, (x, y), 5)
    right = polynomial(x**4 + y + 1, (x, y), 5)
    with pytest.raises(ValueError, match="dimension|Sylvester"):
        left.resultant(right)
    with pytest.raises(ValueError, match="dimension|Sylvester"):
        left.discriminant()
    bindings = [_symbol_binding(x), _symbol_binding(y)]
    source = _native_expr(x**2 + y)
    for operation, sources in [("resultant", [source, source]), ("discriminant", [source])]:
        for bad in ([], sources + [source]):
            with pytest.raises(ValueError):
                _native.prime_field_multivariate_native(bad, bindings, 5, operation)
        for options in ({"parameter": 1}, {"derivative_orders": [0, 0]}, {"order": "grlex"}):
            with pytest.raises(ValueError):
                _native.prime_field_multivariate_native(sources, bindings, 5, operation, **options)
