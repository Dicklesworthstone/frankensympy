"""Complete QQ/ZZ composition, public ring policy and original-source binding.

The profile tests also run unchanged in an isolated SymPy 1.14.0 process.
Native-only tests cover bounded publication and retained-source adversaries.
"""

import pytest
import sympy as sp


def assert_polynomial(value, cls, expression, generators, domain):
    assert type(value) is cls
    assert value.gens == generators
    assert str(value.domain) == domain
    assert value.as_expr() == expression


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
@pytest.mark.parametrize("domain", ["ZZ", "QQ"])
@pytest.mark.parametrize("reverse", [False, True])
def test_profile_composition_substitutes_first_axis_and_retains_class_domain(cls, domain, reverse):
    x, y, unused = sp.symbols("x y unused")
    generators = (y, x, unused) if reverse else (x, y, unused)
    source = cls(x*x + x*y + y, *generators, domain=domain)
    replacement = sp.Poly(x + y, *generators, domain=domain)
    expected = 2*x*x + x*y + x + y if reverse else x*x + 3*x*y + 2*y*y + y
    result = source.compose(replacement)
    assert_polynomial(result, cls, expected, generators, domain)


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
def test_profile_rational_coefficients_cancel_exactly_without_losing_qq(cls):
    x, y = sp.symbols("x y")
    source = cls(x*x/2 - x*y + y*y/2 - sp.Rational(3, 7), x, y, domain="QQ")
    replacement = sp.Poly(x/3 + y, x, y, domain="QQ")
    assert_polynomial(source.compose(replacement), cls, x*x/18 - sp.Rational(3, 7), (x, y), "QQ")
    assert_polynomial(source.compose(sp.Poly(y, x, y)), cls, -sp.Rational(3, 7), (x, y), "QQ")


@pytest.mark.parametrize("source_value", ["zero", "constant", "inactive"])
def test_profile_constant_compositions_admit_both_rings_and_keep_unused_axes(source_value):
    x, y, unused = sp.symbols("x y unused")
    expression = {"zero": sp.Integer(0), "constant": sp.Integer(3), "inactive": y*y + 2}[source_value]
    source = sp.Poly(expression, x, y, unused, domain="ZZ")
    replacement = sp.Poly(x/2 + y, x, y, unused, domain="QQ")
    assert_polynomial(source.compose(replacement), sp.Poly, expression, (x, y, unused), "QQ")


def test_profile_named_rings_unify_without_renaming_coefficient_axes():
    x, y, z = sp.symbols("x y z")
    result = sp.Poly(x*x + y, x, y).compose(sp.Poly(x-y+z, x, z, y))
    # The shared axes follow the left ring; the new z precedes shared y.
    assert_polynomial(result, sp.Poly, x*x - 2*x*y + 2*x*z + y*y - 2*y*z + z*z + y,
                      (x, z, y), "ZZ")
    # A right-only leading coordinate can become primary during unification;
    # an outer polynomial independent of that coordinate then stays unchanged.
    reordered = sp.Poly(x*x + y, x, y).compose(sp.Poly(x-y+z, z, y, x))
    assert_polynomial(reordered, sp.Poly, x*x + y, (z, x, y), "ZZ")
    foreign = sp.Poly(x*x, x).compose(sp.Poly(y+1, y))
    assert_polynomial(foreign, sp.Poly, y*y + 2*y + 1, (x, y), "ZZ")


def test_profile_pure_poly_uses_positional_axes_but_ordinary_poly_keeps_names():
    x, y, a, b = sp.symbols("x y a b")
    outer = sp.PurePoly(x*x + y, x, y, domain="QQ")
    inner = sp.Poly(a + b, a, b, domain="ZZ")
    assert_polynomial(outer.compose(inner), sp.PurePoly, x*x + 2*x*y + y*y + y, (x, y), "QQ")
    result = sp.Poly(x*x + y, x, y).compose(sp.PurePoly(a+b, a, b))
    assert_polynomial(result, sp.Poly, a*a + 2*a*b + b*b + y, (x, y, a, b), "ZZ")


@pytest.mark.parametrize("polys", [False, True])
@pytest.mark.parametrize("poly_inputs", [False, True])
def test_profile_module_result_policy_honors_explicit_polys(polys, poly_inputs):
    x, y, z = sp.symbols("x y z")
    outer, inner = x*x*y + z, x + y*z
    expected = x*x*y + 2*x*y*y*z + y**3*z*z + z
    if poly_inputs:
        outer, inner = sp.Poly(outer, x, y, z), sp.Poly(inner, x, y, z)
    result = sp.compose(outer, inner, x, y, z, polys=polys)
    if polys:
        assert_polynomial(result, sp.Poly, expected, (x, y, z), "ZZ")
    else:
        assert result == expected and not isinstance(result, sp.Poly)


def test_profile_module_infers_generators_from_both_original_inputs():
    x, y = sp.symbols("x y")
    assert sp.compose(x*x, x+y) == x*x + 2*x*y + y*y
    assert_polynomial(sp.compose(x*x, sp.Poly(y+1, y)), sp.Poly,
                      y*y + 2*y + 1, (x, y), "ZZ")
    assert_polynomial(sp.compose(sp.Poly(x, x, domain="QQ"), x+1), sp.Poly,
                      x+1, (x,), "ZZ")


def test_profile_module_explicit_generator_order_and_domain_are_respected():
    x, y = sp.symbols("x y")
    result = sp.compose(sp.Poly(x*x+y, x, y), sp.Poly(x-y, y, x), y, x)
    assert_polynomial(result, sp.Poly, x*x + x - y, (y, x), "ZZ")
    result = sp.compose(sp.Poly(x, x, domain="QQ"), sp.Poly(x+1, x, domain="QQ"), domain="ZZ")
    assert_polynomial(result, sp.Poly, x+1, (x,), "ZZ")
    assert_polynomial(sp.compose(x*x+y, x+y, gens=(x, y), field=True, polys=True), sp.Poly,
                      x*x + 2*x*y + y*y + y, (x, y), "QQ")


def test_profile_composition_interoperates_with_exact_evaluation_and_division():
    x, y = sp.symbols("x y")
    outer = sp.Poly(x*x + 2*x*y + y*y, x, y, domain="QQ")
    inner = sp.Poly(x-y, x, y, domain="QQ")
    result = outer.compose(inner)
    assert_polynomial(result, sp.Poly, x*x, (x, y), "QQ")
    assert result.eval([sp.Rational(2, 3), 99]) == sp.Rational(4, 9)
    quotient, remainder = result.div(sp.Poly(x, x, y, domain="QQ"))
    assert quotient.as_expr() == x and remainder.as_expr() == 0


def test_typed_same_name_symbols_and_unused_dummy_owners_are_preserved():
    x = sp.Symbol("composition_same", positive=True)
    other = sp.Symbol("composition_same", negative=True)
    unused = sp.Dummy("composition_same")
    source = sp.Poly(x*x + other, x, other, unused, domain="QQ")
    replacement = sp.Poly(x+other, x, other, unused)
    result = source.compose(replacement)
    assert result.as_expr() == x*x + 2*x*other + other*other + other
    assert all(actual is original for actual, original in zip(result.gens, (x, other, unused)))


def test_existing_expression_replacement_convenience_is_retained():
    x, y = sp.symbols("x y")
    result = sp.Poly(x*x + y, x, y).compose(x+y)
    assert_polynomial(result, sp.Poly, x*x + 2*x*y + y*y + y, (x, y), "ZZ")
    assert sp.Poly(x*x+1, x).compose(x+2).as_expr() == x*x + 4*x + 5


def test_sparse_high_degree_composition_keeps_coefficient_degrees_in_the_bound():
    x, y = sp.symbols("x y")
    replacement = sp.Poly(y*y, x, y)
    assert sp.Poly(x**128, x, y).compose(replacement).as_expr() == y**256
    with pytest.raises(ValueError, match="degree"):
        sp.Poly(x**128*y, x, y).compose(replacement)
    assert sp.Poly(y+1, x, y).compose(sp.Poly(x**256, x, y)).as_expr() == y+1


@pytest.mark.parametrize("side", ["outer", "inner"])
@pytest.mark.parametrize("wrapper", ["zero_product", "zero_power"])
def test_original_held_poles_cannot_disappear_in_constant_composition(side, wrapper):
    x, y = sp.symbols("x y")
    outer, inner = sp.Poly(0, x, y), sp.Poly(0, x, y)
    pole = sp.Pow(x, -1, evaluate=False)
    held = sp.Mul(0, pole, evaluate=False) if wrapper == "zero_product" else sp.Pow(pole, 0, evaluate=False)
    operand = outer if side == "outer" else inner
    operand._expr = held
    with pytest.raises(ValueError):
        outer.compose(inner)


@pytest.mark.parametrize("side", ["outer", "inner"])
def test_original_declared_domains_are_checked_before_operation_domain_extension(side):
    x, y = sp.symbols("x y")
    outer, inner = sp.Poly(0, x, y, domain="QQ"), sp.Poly(0, x, y, domain="QQ")
    malformed = outer if side == "outer" else inner
    malformed._domain = "ZZ"
    malformed._expr = x/2 + y
    with pytest.raises(ValueError, match="ZZ"):
        sp.compose(outer, inner, x, y, domain="QQ")


def test_original_unused_ring_declarations_cannot_be_replaced_by_operation_generators():
    x, y = sp.symbols("x y")
    wrong = sp.Poly(0, x, domain="QQ")
    wrong._expr = y
    with pytest.raises(ValueError):
        sp.compose(sp.Poly(0, x, y), wrong, x, y)
    duplicate = sp.Poly(0, x, y)
    duplicate._gens = (x, x)
    with pytest.raises((ValueError, TypeError)):
        sp.Poly(0, x, y).compose(duplicate)


@pytest.mark.parametrize("side,field", [("outer", "_domain"), ("inner", "_gens"), ("inner", "_expr")])
def test_mutation_during_lifting_prevents_complete_result_publication(monkeypatch, side, field):
    from sympy.polys import _composition

    x, y = sp.symbols("x y")
    outer, inner = sp.Poly(x*x+y, x, y), sp.Poly(x+y, x, y)
    original = _composition._lift_native_result

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        operand = outer if side == "outer" else inner
        setattr(operand, field, {"_domain": "ZZ", "_gens": (y, x), "_expr": sp.Integer(0)}[field])
        return result

    monkeypatch.setattr(_composition, "_lift_native_result", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        outer.compose(inner)


def test_changed_native_entry_and_custom_poly_owners_are_not_executed(monkeypatch):
    from sympy.core import _native

    x, y = sp.symbols("x y")
    outer, inner = sp.Poly(x*x+y, x, y), sp.Poly(x+y, x, y)

    def forbidden(*args, **kwargs):
        raise AssertionError("replacement entry executed")

    monkeypatch.setattr(_native, "polynomial_compose_native", forbidden)
    with pytest.raises(TypeError, match="original checked native"):
        outer.compose(inner)


def test_native_coordinate_remapping_requires_the_complete_original_inner_ring():
    from sympy.core import _native, _native_expr, _symbol_binding

    x, y, a, b = sp.symbols("x y a b")
    with pytest.raises(ValueError, match="equal original and operation"):
        _native.polynomial_compose_native(
            _native_expr(x*x+y), _native_expr(a+b),
            [_symbol_binding(x), _symbol_binding(y)],
            [_symbol_binding(x), _symbol_binding(y)], [_symbol_binding(a), _symbol_binding(b)],
            "QQ", "QQ", "QQ", [_symbol_binding(b), _symbol_binding(a)],
        )


def test_prime_field_composition_still_uses_its_existing_field_domain():
    x, y = sp.symbols("x y")
    expected = sp.Poly(x*x + 3*x*y + 2*y*y+y, x, y, modulus=5)
    result = sp.compose(sp.Poly(x*x+x*y+y, x, y, modulus=5), sp.Poly(x+y, x, y, modulus=5))
    assert result == expected and result.get_modulus() == 5
