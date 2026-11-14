"""Source-bound polynomial elimination through the actual public extension.

The fixed results are exact resultant/discriminant identities. Public class,
ring, coefficient-domain and elimination-axis conventions follow the pinned
SymPy 1.14.0 Poly protocol. These are functional regressions, not a claim that
the existing symbolic determinant lane issues mathematical certificates.
"""

import pytest

import sympy as sp
from sympy.core import _native, _native_expr
from sympy.polys import polytools
from sympy.polys.polyerrors import GeneratorsNeeded, PolynomialError, UnificationFailed


def expression(value):
    return value.as_expr() if isinstance(value, sp.Poly) else value


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
@pytest.mark.parametrize("domain", ["ZZ", "QQ"])
def test_resultant_eliminates_only_the_primary_axis_and_preserves_result_type(cls, domain):
    x, y, z = sp.symbols("resultant_x resultant_y resultant_z")
    left = cls(x**2 - y, x, y, z, domain=domain)
    right = sp.Poly(x - z, x, y, z, domain=domain)
    for result in (left.resultant(right), sp.resultant(left, right)):
        assert type(result) is cls
        assert result.gens == (y, z)
        assert result.domain == domain
        assert result.as_expr() == z**2 - y
    assert sp.resultant(left, right, polys=False) == z**2 - y


def test_distinct_poly_rings_unify_before_selecting_the_eliminated_axis():
    x, y, z = sp.symbols("union_x union_y union_z")
    left = sp.Poly(x**2 - y, x, y)
    right = sp.Poly(x - z, x, y, z)
    result = left.resultant(right)
    assert result.gens == (y, z)
    assert result.as_expr() == z**2 - y
    # The right-only z precedes the first shared axis x in the merged ring.
    left = sp.Poly(x + y, x, y)
    right = sp.Poly(z - x, z, x)
    result = left.resultant(right)
    assert result.gens == (x, y)
    assert result.as_expr() == x + y


def test_purepoly_unifies_formal_axes_positionally_without_merging_names():
    x, y, u, v = sp.symbols("pure_x pure_y pure_u pure_v")
    left = sp.PurePoly(x**2 - y, x, y)
    right = sp.Poly(u + 2*v, v, u)
    result = left.resultant(right)
    assert type(result) is sp.PurePoly
    assert result.gens == (y,)
    assert result.as_expr() == y**2 - 4*y
    with pytest.raises(UnificationFailed):
        left.resultant(sp.Poly(u + 1, u))


@pytest.mark.parametrize("polys", [False, True])
def test_module_expression_calls_accept_inferred_explicit_and_container_generators(polys):
    x, a, b = sp.symbols("x a b")
    expected = a - b
    for args in [(), (x, a, b), ([x, a, b],)]:
        result = sp.resultant(x - a, x - b, *args, polys=polys)
        assert expression(result) == expected
        if polys:
            assert type(result) is sp.Poly and result.gens == (a, b)
        else:
            assert not isinstance(result, sp.Poly)
    # An explicit single axis returns an expression in its coefficient field.
    assert sp.resultant(x - a, x - b, x, polys=True) == expected


@pytest.mark.parametrize("poly_on_left", [False, True])
def test_mixed_module_operands_infer_the_joint_ring_and_base_poly_class(poly_on_left):
    x, y = sp.symbols("mixed_x mixed_y")
    left, right = x**2 - y, x - 1
    if poly_on_left:
        left = sp.PurePoly(left, x, y, domain="QQ")
    else:
        right = sp.PurePoly(right, x, y, domain="QQ")
    result = sp.resultant(left, right)
    assert type(result) is sp.Poly
    assert result.gens == (y,)
    assert result.domain == "ZZ"
    assert result.as_expr() == 1 - y


@pytest.mark.parametrize("cls", [sp.Poly, sp.PurePoly])
@pytest.mark.parametrize("domain", ["ZZ", "QQ"])
def test_quadratic_discriminant_preserves_nonconstant_leading_coefficient(cls, domain):
    x, a, b, c = sp.symbols("disc_x disc_a disc_b disc_c")
    source = cls(a*x**2 + b*x + c, x, a, b, c, domain=domain)
    for result in (source.discriminant(), sp.discriminant(source)):
        assert type(result) is cls
        assert result.gens == (a, b, c)
        assert result.domain == domain
        assert result.as_expr() == b**2 - 4*a*c
    assert sp.discriminant(source, polys=False) == b**2 - 4*a*c
    assert sp.discriminant(source.as_expr(), x) == b**2 - 4*a*c


@pytest.mark.parametrize("degree", [3, 4])
def test_higher_discriminants_have_exact_multiplicities_and_signs(degree):
    x, a, b = sp.symbols("higher_x higher_a higher_b")
    if degree == 3:
        source, expected = x**3 + a*x + b, -4*a**3 - 27*b**2
    else:
        source, expected = x**4 + a*x**2 + b, 16*b*(a**2 - 4*b)**2
    result = sp.discriminant(source, x, a, b, polys=True)
    assert result.gens == (a, b)
    assert result.as_expr() == sp.expand(expected)


def test_rational_coefficient_domain_survives_an_integral_discriminant():
    x, y, z = sp.symbols("rational_x rational_y rational_z")
    source = sp.Poly(x**2/2 + y*x + z, x, y, z, domain="QQ")
    result = source.discriminant()
    assert result.gens == (y, z) and result.domain == "QQ"
    assert result.as_expr() == y**2 - 2*z
    assert sp.discriminant(source.as_expr(), x, y, z, domain="QQ", polys=True) == result


@pytest.mark.parametrize("source,other,expected", [
    (0, 0, 0), (0, 3, 0), (2, 0, 0), (2, 3, 1),
])
def test_resultant_zero_and_constant_conventions_survive_remaining_axes(source, other, expected):
    x, y = sp.symbols("constant_x constant_y")
    result = sp.Poly(source, x, y).resultant(sp.Poly(other, x, y))
    assert type(result) is sp.Poly and result.gens == (y,)
    assert result.as_expr() == expected
    assert sp.resultant(source, other, x) == expected


def test_symbolic_constants_and_zero_polynomials_have_correct_resultants():
    x, y = sp.symbols("symbolic_constant_x symbolic_constant_y")
    assert sp.resultant(y, x**3 + x + 1, x) == y**3
    assert sp.resultant(x**3 + x + 1, y, x) == y**3
    assert sp.resultant(y, y + 1, x) == 1
    assert sp.resultant(0, y + 1, x) == 0
    assert sp.resultant(x**2 - y, x**2 - y, x) == 0


@pytest.mark.parametrize("source,expected", [(0, 0), (3, 0)])
def test_constant_discriminants_preserve_the_remaining_ring(source, expected):
    x, y = sp.symbols("constant_disc_x constant_disc_y")
    result = sp.Poly(source, x, y, domain="QQ").discriminant()
    assert type(result) is sp.Poly and result.gens == (y,) and result.domain == "QQ"
    assert result.as_expr() == expected
    assert sp.discriminant(source, x) == expected
    assert sp.discriminant(y, x) == 0
    assert sp.discriminant(y*x + 1, x) == 1


def test_polynomial_methods_coerce_only_coefficients_from_the_left_domain():
    x, y = sp.symbols("scalar_x scalar_y")
    source = sp.Poly(x**2 + y, x, y, domain="QQ")
    result = source.resultant(sp.Rational(3, 2))
    assert result.gens == (y,) and result.domain == "QQ"
    assert result.as_expr() == sp.Rational(9, 4)
    for scalar in (x + 1, y):
        with pytest.raises(UnificationFailed):
            source.resultant(scalar)
    with pytest.raises(UnificationFailed):
        sp.Poly(x**2 + y, x, y, domain="ZZ").resultant(sp.Rational(1, 2))


def test_constant_expression_requests_require_an_explicit_axis():
    with pytest.raises(GeneratorsNeeded):
        sp.resultant(4, 2)
    with pytest.raises(GeneratorsNeeded):
        sp.discriminant(4)


def test_rational_native_lane_retains_its_larger_determinant_support(monkeypatch):
    x, unused = sp.symbols("native_x native_unused")
    def forbidden(*args, **kwargs):
        raise AssertionError("rational coefficients entered the symbolic determinant")
    monkeypatch.setattr(polytools, "_sylvester_resultant", forbidden)
    result = sp.Poly(x**5 - 1, x, unused).resultant(sp.Poly(x**5 + 1, x, unused))
    assert result.gens == (unused,) and result.as_expr() == 32


def test_typed_same_name_symbols_and_unused_dummy_axes_remain_original_owners():
    x = sp.Symbol("elimination_identity")
    positive = sp.Symbol("elimination_identity", positive=True)
    unused = sp.Dummy("elimination_identity")
    source = sp.Poly(x**2 - positive, x, positive, unused)
    result = source.resultant(sp.Poly(x - positive, x, positive, unused))
    assert result.gens[0] is positive and result.gens[1] is unused
    assert result.as_expr() == positive**2 - positive
    assert x not in result.free_symbols


@pytest.mark.parametrize("operation", ["resultant", "discriminant"])
@pytest.mark.parametrize("domain,kind", [("ZZ", "fractional"), ("ZZ", "foreign"), ("QQ", "foreign")])
def test_original_domain_declarations_cannot_be_repaired_by_generator_union(operation, domain, kind, monkeypatch):
    x, y = sp.symbols("declaration_x declaration_y")
    source = sp.Poly(x, x, domain=domain)
    source._expr = x/2 if kind == "fractional" else x*y
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid original declaration reached symbolic computation")
    monkeypatch.setattr(polytools, "_sylvester_resultant", forbidden)
    with pytest.raises(ValueError):
        if operation == "resultant":
            sp.resultant(source, sp.Poly(x + y, x, y), x, y)
        else:
            sp.discriminant(source, x, y)


@pytest.mark.parametrize("invalid", ["function", "pole", "float"])
def test_hidden_invalid_original_children_are_admitted_before_constant_shortcuts(invalid):
    x, y = sp.symbols("hidden_x hidden_y")
    child = {"function": sp.sin(y), "pole": sp.Pow(y, -1, evaluate=False),
             "float": sp.Float("1.0")}[invalid]
    source = sp.Poly(0, x, y)
    source._expr = sp.Mul(0, child, evaluate=False)
    for operation in (lambda: source.resultant(sp.Poly(1, x, y)), source.discriminant):
        with pytest.raises((ValueError, TypeError, NotImplementedError)):
            operation()


@pytest.mark.parametrize("target", ["source", "ring", "container", "assumptions"])
def test_source_mutation_during_native_admission_is_terminal(target, monkeypatch):
    x = sp.Symbol("mutation_x", positive=True)
    y = sp.Symbol("mutation_y")
    source = sp.Poly(x**2 - y, x, y)
    generators = [x, y]
    original = _native.poly_recursive_div_native
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        if target == "source":
            monkeypatch.setattr(source, "_expr", x)
        elif target == "ring":
            monkeypatch.setattr(source, "_gens", (y, x))
        elif target == "container":
            generators.reverse()
        else:
            monkeypatch.setitem(x._assumptions, "positive", False)
        return result
    monkeypatch.setattr(_native, "poly_recursive_div_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        sp.resultant(source, sp.Poly(x - 1, x, y), generators)


@pytest.mark.parametrize("during", ["determinant", "lifting"])
def test_source_mutation_after_admission_is_terminal(during, monkeypatch):
    x, y = sp.symbols("late_x late_y")
    source = sp.Poly(x**2 - y, x, y)
    owner, name = (sp.Matrix, "det") if during == "determinant" else (polytools, "_lift_native_result")
    original = getattr(owner, name)
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setattr(source, "_expr", x)
        return result
    monkeypatch.setattr(owner, name, mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        source.resultant(sp.Poly(x - 1, x, y))


def test_native_budget_or_verifier_refusal_never_retries_an_unchecked_path(monkeypatch):
    x, y = sp.symbols("refusal_x refusal_y")
    calls = []
    def fail(*args, **kwargs):
        calls.append(True)
        raise ValueError("injected source verifier exhaustion")
    monkeypatch.setattr(_native, "poly_recursive_div_native", fail)
    with pytest.raises(ValueError, match="verifier exhaustion"):
        sp.resultant(x**2 - y, x - 1, x, y)
    assert calls == [True]


@pytest.mark.parametrize("kind", ["dimension", "expansion"])
def test_symbolic_growth_is_refused_before_allocating_matrix_products(kind, monkeypatch):
    x, y, z = sp.symbols("growth_x growth_y growth_z")
    if kind == "dimension":
        left, right = x**5 - y, x**4 + z
    else:
        coefficient = (y + z + 1)**5
        left, right = x**4 + coefficient, x**4 - coefficient
    def forbidden(*args, **kwargs):
        raise AssertionError("oversized symbolic resultant reached matrix multiplication")
    monkeypatch.setattr(sp.Matrix, "det", forbidden)
    with pytest.raises(ValueError, match="dimension|expansion"):
        sp.resultant(left, right, x, y, z)


@pytest.mark.parametrize("operation", ["resultant", "discriminant"])
def test_noncommutative_generators_are_refused(operation):
    x = sp.Symbol("noncommutative_elimination", commutative=False)
    with pytest.raises((ValueError, TypeError, PolynomialError)):
        if operation == "resultant":
            sp.resultant(x + 1, x - 1, x)
        else:
            sp.discriminant(x + 1, x)


@pytest.mark.parametrize("prime", [5, 7])
def test_existing_prime_field_lanes_are_unchanged(prime):
    x = sp.Symbol("modular_elimination")
    source = sp.Poly(x**2 + 1, x, modulus=prime)
    other = sp.Poly(x - 1, x, modulus=prime)
    assert source.resultant(other) == sp.resultant(source, other) == 2
    expected = 1 if prime == 5 else 3
    assert source.discriminant() == sp.discriminant(source) == expected


@pytest.mark.parametrize("operation", ["resultant", "discriminant"])
@pytest.mark.parametrize("options", [
    {"polys": "yes"}, {"domain": "RR"}, {"includePRS": True},
    {"_method": True}, {"discriminant": True},
])
def test_controls_cannot_select_a_different_internal_operation(operation, options):
    x = sp.Symbol("elimination_options")
    with pytest.raises((TypeError, NotImplementedError)):
        if operation == "resultant":
            sp.resultant(x + 1, x - 1, x, **options)
        else:
            sp.discriminant(x**2 + 1, x, **options)


def test_purepoly_renaming_cannot_erase_a_left_coefficient_parameter():
    x, z, y, t = sp.symbols("parameter_left_x parameter_left_z parameter_y parameter_t")
    left = sp.PurePoly(x + z + y, x, z)
    right = sp.PurePoly(y + t, y, t)
    # Renaming the right formal y to x does not rename the left coefficient y.
    # The exact answer belongs to ZZ[y][z], outside this bounded QQ/ZZ output.
    with pytest.raises(NotImplementedError, match="coefficient parameters"):
        left.resultant(right)


@pytest.mark.parametrize("operation", ["resultant", "discriminant"])
def test_inferred_polynomial_parameters_are_distinct_from_explicit_numeric_domains(operation):
    x, a, b = sp.symbols("parameter_x parameter_a parameter_b")
    source = sp.Poly(a*x**2 + b, x)
    if operation == "resultant":
        assert source.resultant(sp.Poly(x - 1, x)) == a + b
    else:
        assert source.discriminant() == -4*a*b
    falsely_declared = sp.Poly(a*x**2 + b, x, domain="QQ")
    with pytest.raises(ValueError):
        if operation == "resultant":
            falsely_declared.resultant(sp.Poly(x - 1, x))
        else:
            falsely_declared.discriminant()


def test_legacy_explicit_axis_coefficient_fields_are_selected_before_polynomial_admission(monkeypatch):
    x, y = sp.symbols("legacy_x legacy_y")
    def forbidden(*args, **kwargs):
        raise AssertionError("legacy coefficient field entered the new polynomial admission lane")
    monkeypatch.setattr(polytools, "_polynomial_elimination", forbidden)
    assert sp.resultant(x/y, x + 1, x) == 1/y
    assert sp.discriminant(x**2/y + 1, x) == -4/y
    assert sp.resultant(sp.sin(y)*x + 1, x - 1, x) == -sp.sin(y) - 1
    assert sp.discriminant(sp.sin(y)*x**2 + 1, x) == -4*sp.sin(y)
    assert sp.resultant(sp.Poly(x/y, x), x + 1, x) == 1/y
    assert sp.discriminant(sp.Poly(x**2/y + 1, x), x) == -4/y
    assert sp.resultant(0, 1/y, x) == 0
    assert sp.resultant(1/y, sp.sin(y), x) == 1
    assert sp.discriminant(1/y, x) == 0


@pytest.mark.parametrize("operation", ["resultant", "discriminant"])
def test_explicit_axis_compatibility_never_bypasses_a_numeric_poly_declaration(operation, monkeypatch):
    x, y = sp.symbols("legacy_domain_x legacy_domain_y")
    source = sp.Poly(x*y + 1, x, domain="QQ")
    def forbidden(*args, **kwargs):
        raise AssertionError("false original Poly declaration entered legacy computation")
    monkeypatch.setattr(polytools, "_legacy_sylvester_resultant", forbidden)
    with pytest.raises(ValueError):
        if operation == "resultant":
            sp.resultant(source, x - 1, x)
        else:
            sp.discriminant(source, x)
