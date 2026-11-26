"""Complete exact all-real numerical roots without binary64 intermediates."""

from decimal import Decimal, localcontext

import pytest
import sympy as s
from sympy.core import _ExactDecimalFloat


def values(result):
    assert all(type(value) is _ExactDecimalFloat for value in result)
    return [Decimal(str(value)) for value in result]


@pytest.mark.parametrize("digits", [30, 70, 100])
@pytest.mark.parametrize("method", [False, True])
def test_irrational_roots_match_independent_decimal_square_roots(digits, method):
    x = s.Symbol("high_precision_root")
    p = s.Poly(x**2 - 2, x)
    result = p.nroots(n=digits) if method else s.nroots(p, n=digits)
    with localcontext() as context:
        context.prec = digits
        positive = Decimal(2).sqrt()
    assert values(result) == [positive.copy_negate(), positive]
    assert all(root.dps == digits and root.is_real for root in result)


@pytest.mark.parametrize("cleanup", [False, True])
@pytest.mark.parametrize("domain", ["QQ", "ZZ", None])
def test_complete_order_and_original_multiplicities_are_retained(cleanup, domain):
    x = s.Symbol("multiple_real_roots")
    p = s.Poly((x**2 - 2)**2 * x**3 * (x - 1), x, domain=domain)
    result = p.nroots(n=30, cleanup=cleanup)
    with localcontext() as context:
        context.prec = 30
        root = Decimal(2).sqrt()
    negative = root.copy_negate()
    assert values(result) == [negative, negative, Decimal(0), Decimal(0), Decimal(0), Decimal(1), root, root]
    assert len(result) == 8
    assert all(value.dps == 30 for value in result)


def test_near_coincident_roots_remain_distinct_and_exactly_ordered():
    x = s.Symbol("close_real_roots")
    scale = 10**40
    result = s.nroots((scale*x - scale)**2 - 2, n=70)
    with localcontext() as context:
        context.prec = 110
        displacement = Decimal(2).sqrt()/Decimal(scale)
        expected = [Decimal(1)-displacement, Decimal(1)+displacement]
        context.prec = 70
        expected = [+value for value in expected]
    assert values(result) == expected
    assert values(result)[0] < Decimal(1) < values(result)[1]


@pytest.mark.parametrize("root", [s.Rational(1, 10**400), s.Integer(10**400)])
def test_roots_outside_binary64_preserve_nonzero_finite_decimal_values(root):
    x = s.Symbol("extreme_real_root")
    result = s.nroots(s.Poly(x-root, x), n=40)
    expected = Decimal("1e-400") if root < 1 else Decimal("1e400")
    assert values(result) == [expected]
    assert result[0].is_zero is False
    assert result[0].is_finite is True


def test_new_lane_never_calls_binary64_conversion_or_candidate_finding(monkeypatch):
    x = s.Symbol("native_exact_only")
    def forbidden(*args, **kwargs):
        raise AssertionError("high-precision roots selected a binary64 operation")
    monkeypatch.setattr(s, "_numeric_poly_roots", forbidden)
    monkeypatch.setattr(s, "_nroots_coefficients", forbidden)
    assert len(s.nroots(x**4 - 5*x**2 + 6, n=40)) == 4


def test_full_precision_envelope_and_pure_polynomial_adapter():
    x = s.Symbol("full_precision_linear_root")
    result = s.PurePoly(3*x-1, x).nroots(n=512)
    assert str(result[0]) == "0." + "3"*512
    assert result[0].dps == 512


@pytest.mark.parametrize("offset, rounded_offset", [(5, 0), (15, 20)])
def test_linear_rational_half_even_ties_remain_exact(offset, rounded_offset):
    x = s.Symbol("exact_rounding_boundary")
    root = s.Rational(10**30 + offset, 10**30)
    result = s.nroots(x-root, n=30)
    with localcontext() as context:
        context.prec = 60
        expected = Decimal(1) + Decimal(rounded_offset)/Decimal(10**30)
    assert values(result) == [expected]


@pytest.mark.parametrize("polynomial", [lambda x: x**2+1, lambda x: (x-2)*(x**2+1), lambda x: x**3-2])
def test_nonreal_or_mixed_roots_refuse_before_any_decimal_publication(monkeypatch, polynomial):
    import sympy.core as core
    x = s.Symbol("incomplete_real_subset")
    source = polynomial(x)
    def forbidden(*args, **kwargs):
        raise AssertionError("a partial real-root list reached decimal publication")
    monkeypatch.setattr(core, "_format_decimal", forbidden)
    with pytest.raises(NotImplementedError, match="complete.*real"):
        s.nroots(source, n=30)


@pytest.mark.parametrize("maxsteps", [0, 1, 49, 51, 1000])
def test_high_precision_custom_iteration_controls_refuse_explicitly(maxsteps):
    x = s.Symbol("exact_root_controls")
    with pytest.raises(NotImplementedError, match="default maxsteps=50"):
        s.nroots(x**2-2, n=30, maxsteps=maxsteps)


def test_exact_native_precision_degree_and_coefficient_bounds_remain_terminal():
    x = s.Symbol("exact_root_bounds")
    with pytest.raises(NotImplementedError, match="512"):
        s.nroots(x-1, n=513)
    for source in [s.Pow(x, 65, evaluate=False), x+s.Integer(1 << 4096)]:
        with pytest.raises((ValueError, NotImplementedError)):
            s.nroots(source, n=30)
    with pytest.raises(ValueError, match="zero polynomial"):
        s.nroots(s.Poly(0, x), n=30)
    assert s.nroots(s.Poly(7, x), n=30) == []


@pytest.mark.parametrize("form", ["pole", "undefined_power", "nonpolynomial"])
def test_original_held_source_admission_cannot_erase_invalid_children(form):
    x = s.Symbol("original_root_source")
    if form == "pole":
        source = s.Add(x, s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False), evaluate=False)
    elif form == "undefined_power":
        source = s.Add(x, s.Mul(0, s.Pow(0, 0, evaluate=False), evaluate=False), evaluate=False)
    else:
        source = s.Pow(s.Pow(x, -1, evaluate=False), -1, evaluate=False)
    with pytest.raises((ValueError, NotImplementedError)):
        s.nroots(source, n=30)


@pytest.mark.parametrize("domain", ["EX", "RR", "GF(5)"])
def test_foreign_coefficient_domains_are_not_reinterpreted_as_rationals(domain):
    x = s.Symbol("root_domain")
    with pytest.raises((ValueError, NotImplementedError)):
        s.nroots(s.Poly(x**2-1, x, domain=domain), n=30)


def test_inexact_coefficients_parameters_and_multiple_generators_refuse():
    x, y = s.symbols("numeric_root_x numeric_root_y")
    for source in [s.Poly(x+s.Float(0.5), x), s.Poly(x-y, x), s.Poly(x+y, x, y)]:
        with pytest.raises((ValueError, NotImplementedError)):
            s.nroots(source, n=30)
    with pytest.raises(ValueError, match="exact integers"):
        s.nroots(s.Poly(x+s.Rational(1, 2), x, domain="ZZ"), n=30)


def test_custom_source_getters_are_not_called():
    effects = []
    class Opaque(s.Expr):
        @property
        def args(self):
            effects.append("args")
            raise AssertionError("opaque getter ran")
        @property
        def free_symbols(self):
            effects.append("free_symbols")
            raise AssertionError("opaque getter ran")
    source = object.__new__(Opaque)
    with pytest.raises((ValueError, TypeError)):
        s.nroots(source, n=30)
    assert effects == []


def test_retained_source_mismatch_refuses_before_root_finding():
    x = s.Symbol("retained_root_source")
    source = s.Add(x, -1, evaluate=False)
    s.Add.__dict__["_args"].__set__(source, (x, s.Integer(-2)))
    with pytest.raises(ValueError):
        s.nroots(source, n=30)


@pytest.mark.parametrize("change", ["domain", "class", "source"])
def test_source_mutation_during_result_lifting_never_publishes(monkeypatch, change):
    import sympy.core as core
    x = s.Symbol("mutating_root_source")
    polynomial = s.Poly(x**2-2, x)
    original = dict(polynomial.__dict__)
    formatter = core._format_decimal
    def mutate(*args):
        result = formatter(*args)
        if change == "domain":
            polynomial._domain = "EX"
        elif change == "source":
            polynomial._expr = x**2-3
        else:
            polynomial.__class__ = s.PurePoly
        return result
    monkeypatch.setattr(core, "_format_decimal", mutate)
    try:
        with pytest.raises(RuntimeError, match="mutated|class changed"):
            polynomial.nroots(n=30)
    finally:
        polynomial.__class__ = s.Poly
        polynomial.__dict__.clear()
        polynomial.__dict__.update(original)


def test_changed_polynomial_implementation_refuses_without_calling_it(monkeypatch):
    x = s.Symbol("patched_root_owner")
    source = s.Poly(x**2-2, x)
    def forbidden(*args):
        raise AssertionError("patched polynomial getter ran")
    monkeypatch.setattr(s.Poly, "as_expr", forbidden)
    with pytest.raises(TypeError, match="original polynomial"):
        s.nroots(source, n=30)
