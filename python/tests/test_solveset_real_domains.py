"""Complete rational roots in restricted real domains, on the native shell."""

import pytest

import sympy as s
from sympy.core import _native, _native_expr, _symbol_binding
from sympy.polys import rootoftools


def _forbid_approximation(*args, **kwargs):
    raise AssertionError("restricted exact solving must not approximate roots")


def test_irreducible_quintic_is_filtered_exactly_by_rational_intervals(monkeypatch):
    x = s.Symbol("x")
    polynomial = x**5 - x + 1
    root = s.CRootOf(polynomial, 0)
    monkeypatch.setattr(s, "_numeric_poly_roots", _forbid_approximation)
    monkeypatch.setattr(rootoftools, "_approximate_factor_roots", _forbid_approximation)
    monkeypatch.setattr(s.CRootOf, "evalf", _forbid_approximation)
    assert s.solveset(polynomial, x, s.Interval(-2, -1)) == s.FiniteSet(root)
    assert s.solveset(polynomial, x, s.Interval(0, 2)) == s.S.EmptySet
    assert s.solveset(polynomial, x, s.Interval.open(-s.oo, 0)) == s.FiniteSet(root)


@pytest.mark.parametrize("left_open,right_open,indices", [
    (False, False, (0, 1)),
    (True, False, (1,)),
    (False, True, (0,)),
    (True, True, ()),
])
def test_algebraic_interval_boundaries_obey_exact_openness(left_open, right_open, indices):
    x = s.Symbol("x")
    polynomial = x**4 - x - 1
    roots = [s.CRootOf(polynomial, index) for index in (0, 1)]
    domain = s.Interval(*roots, left_open=left_open, right_open=right_open)
    assert s.solveset(polynomial, x, domain) == s.FiniteSet(*(roots[i] for i in indices))


def test_positive_radical_and_affine_radical_endpoints_use_exact_root_order():
    x = s.Symbol("x")
    root = s.sqrt(2)
    assert s.solveset(x**2 - 2, x, s.Interval(-root, root)) == s.FiniteSet(-root, root)
    assert s.solveset(x**2 - 2, x, s.Interval.open(-root, root)) == s.S.EmptySet
    polynomial = (x - 1)**2 - 2
    assert s.solveset(polynomial, x, s.Interval.Lopen(1-root, 1+root)) == s.FiniteSet(1+root)
    assert s.solveset(x**3 - 2, x, s.FiniteSet(s.root(2, 3), 0)) == s.FiniteSet(s.root(2, 3))


def test_union_of_intervals_and_isolated_points_keeps_every_admitted_root():
    x = s.Symbol("x")
    quintic = x**5 - x + 1
    polynomial = quintic * (x**2 - 2) * (x - 3)**2
    domain = s.Union(s.Interval(-2, -1), s.Interval(1, 2), s.FiniteSet(3, 4))
    assert s.solveset(polynomial, x, domain) == s.FiniteSet(
        s.CRootOf(quintic, 0), -s.sqrt(2), s.sqrt(2), 3,
    )
    domain = s.Union(s.Interval.open(-2, -1), s.Interval(1, 2))
    assert s.solveset((x-1)*(x+1), x, domain) == s.FiniteSet(1)


def test_finite_exact_real_domain_filters_algebraic_and_rational_members():
    x = s.Symbol("x")
    polynomial = x**5 - x + 1
    root = s.CRootOf(polynomial, 0)
    assert s.solveset(polynomial, x, s.FiniteSet(root, -2, 0, 1)) == s.FiniteSet(root)
    assert s.solveset(polynomial, x, s.FiniteSet(-2, 0, 1)) == s.S.EmptySet
    assert s.solveset(x**2 - 2, x, s.FiniteSet(-s.sqrt(2), 0)) == s.FiniteSet(-s.sqrt(2))


def test_intersections_and_complements_retain_the_exact_real_bound():
    x = s.Symbol("x")
    polynomial = x**5 - x + 1
    root = s.CRootOf(polynomial, 0)
    without_root = s.Complement(s.S.Reals, s.FiniteSet(root))
    domain = s.Intersection(s.Interval(-2, 2), without_root)
    assert s.solveset(polynomial * (x - 1), x, domain) == s.FiniteSet(1)
    punctured = s.Complement(s.Interval(-2, 2), s.Interval.open(-1, 1))
    assert s.solveset(x * (x-1) * (x+1), x, punctured) == s.FiniteSet(-1, 1)


def test_original_equation_sides_keep_a_pole_before_subtraction():
    x = s.Symbol("x")
    inverse = s.Pow(x, -1, evaluate=False)
    original = s.Eq(inverse, inverse, evaluate=False)
    domain = s.Interval(-1, 1)
    result = s.solveset(original, x, domain)
    assert result == s.Complement(domain, s.FiniteSet(0))
    assert result.contains(0) is False
    assert result.contains(-1) is True
    assert result.contains(1) is True
    assert s.solveset(original, domain=domain) == result


def test_original_algebraic_holes_are_excluded_even_from_an_identity():
    x = s.Symbol("x")
    quintic = x**5 - x + 1
    inverse = s.Pow(quintic * (x**2 - 2), -1, evaluate=False)
    original = s.Eq(inverse, inverse, evaluate=False)
    domain = s.Interval(-2, 2)
    holes = s.FiniteSet(s.CRootOf(quintic, 0), -s.sqrt(2), s.sqrt(2))
    assert s.solveset(original, x, domain) == s.Complement(domain, holes)
    for hole in holes:
        assert s.solveset(original, x, s.FiniteSet(hole)) == s.S.EmptySet


def test_cofinite_finite_domains_keep_proved_root_radical_equality():
    x, bound = s.symbols("x bound")
    lower = s.CRootOf(bound**2-2, 0, radicals=False)
    upper = s.CRootOf(bound**2-2, 1, radicals=False)
    inverse = s.Pow(x**2-2, -1, evaluate=False)
    identity = s.Eq(inverse, inverse, evaluate=False)
    assert s.solveset(identity, x, s.FiniteSet(lower, upper)) == s.S.EmptySet
    assert s.solveset(identity, x, s.FiniteSet(lower, 0, upper)) == s.FiniteSet(0)
    assert s.solveset(identity, x, s.Interval(upper, upper)) == s.S.EmptySet


def test_algebraic_interval_holes_retain_both_original_and_radical_spellings():
    x, bound = s.symbols("x bound")
    lower = s.CRootOf(bound**2-2, 0, radicals=False)
    upper = s.CRootOf(bound**2-2, 1, radicals=False)
    inverse = s.Pow(x**2-2, -1, evaluate=False)
    identity = s.Eq(inverse, inverse, evaluate=False)
    result = s.solveset(identity, x, s.Interval(lower, upper))
    for hole in (lower, upper, -s.sqrt(2), s.sqrt(2)):
        assert result.contains(hole) is False
    assert result.contains(0) is not False
    assert s.solveset(0, x, s.Interval.open(lower, lower)) == s.S.EmptySet


def test_nested_reciprocal_and_zero_power_holes_survive_restricted_solving():
    x = s.Symbol("x")
    inverse = s.Pow(x, -1, evaluate=False)
    assert s.solveset(s.Pow(inverse, -1, evaluate=False), x, s.Interval(-1, 1)) == s.S.EmptySet
    held_zero = s.Mul(0, s.Pow(inverse, 0, evaluate=False), evaluate=False)
    domain = s.Union(s.Interval(-2, -1), s.FiniteSet(0, 1))
    assert s.solveset(held_zero, x, domain) == s.Complement(domain, s.FiniteSet(0))
    nowhere = s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False)
    assert s.solveset(nowhere, x, domain) == s.S.EmptySet


def test_held_numerator_denominator_cancellation_cannot_reintroduce_a_root():
    x = s.Symbol("x")
    quintic = x**5 - x + 1
    expression = s.Mul(
        quintic * (x**2-2)**2 * (x-3)**2,
        s.Pow((x**2-2)*(x-3), -1, evaluate=False), evaluate=False,
    )
    assert s.solveset(expression, x, s.Interval(-2, 4)) == s.FiniteSet(s.CRootOf(quintic, 0))


def test_unknown_endpoint_order_stays_an_exact_intersection():
    x, endpoint = s.symbols("x endpoint")
    polynomial = x**5 - x + 1
    root = s.CRootOf(polynomial, 0)
    domain = s.Interval(endpoint, 0)
    result = s.solveset(polynomial, x, domain)
    assert result == s.Intersection(s.FiniteSet(root), domain)
    assert result != s.S.EmptySet
    assert result != s.FiniteSet(root)


def test_constants_and_empty_domains_keep_their_exact_meaning():
    x = s.Symbol("x")
    domain = s.Union(s.Interval.open(-2, -1), s.FiniteSet(1))
    assert s.solveset(0, x, domain) is domain
    assert s.solveset(0, domain=domain) is domain
    assert s.solveset(False, x, domain) is domain
    assert s.solveset(True, x, domain) == s.S.EmptySet
    assert s.solveset(7, x, domain) == s.S.EmptySet
    assert s.solveset(x**2-1, x, s.S.EmptySet) == s.S.EmptySet


def test_assumed_symbols_and_dummies_retain_full_typed_binding(monkeypatch):
    native_solve = _native.solve_real_rational_native
    seen = []

    def capture(source, binding):
        seen.append((binding.name, binding.identity_hex))
        return native_solve(source, binding)

    monkeypatch.setattr(_native, "solve_real_rational_native", capture)
    for x in (s.Symbol("x", positive=True), s.Dummy("x", real=True)):
        expected = _symbol_binding(x)
        assert s.solveset(x**2 - 4, x, s.Interval(-3, 3)) == s.FiniteSet(-2, 2)
        assert seen[-1] == (expected.name, expected.identity_hex)


@pytest.mark.parametrize("failure", [ValueError("resource exhausted"),
                                        OverflowError("coefficient limit"),
                                        MemoryError("allocation refused"),
                                        NotImplementedError("unsupported source")])
def test_restricted_kernel_failures_are_terminal_and_keep_exception_identity(monkeypatch, failure):
    calls = []

    def fail(*args):
        calls.append(args)
        raise failure

    monkeypatch.setattr(_native, "solve_real_rational_native", fail)
    monkeypatch.setattr(_native, "solve_native", _forbid_approximation)
    x = s.Symbol("x")
    with pytest.raises(type(failure)) as error:
        s.solveset(x**2-2, x, s.Interval(-2, 2))
    assert error.value is failure
    assert len(calls) == 1


def test_exact_root_comparison_failures_cannot_drop_a_possible_root(monkeypatch):
    failure = ValueError("checked comparison resource exhausted")

    def fail(*args):
        raise failure

    monkeypatch.setattr(_native, "compare_real_roots_native", fail)
    monkeypatch.setattr(_native, "solve_native", _forbid_approximation)
    x = s.Symbol("x")
    with pytest.raises(ValueError) as error:
        s.solveset(x**2-2, x, s.Interval(-2, 2))
    assert error.value is failure


def test_incomplete_or_duplicate_root_association_is_not_published(monkeypatch):
    x = s.Symbol("x")
    domain = s.Interval(-2, 2)
    for roots in ([(s.Integer(1), 1)], [(s.Integer(1), 1), (s.Integer(1), 1)],
                  [(s.Integer(1), 1), (s.Integer(2), 1)]):
        monkeypatch.setattr(rootoftools, "real_roots", lambda *args, **kwargs: roots)
        with pytest.raises(ValueError, match="association|distinct"):
            s.solveset(x**2-2, x, domain)


def test_domain_replacement_during_native_evaluation_is_not_published(monkeypatch):
    x = s.Symbol("x")
    domain = s.Interval(-2, 2)
    native_solve = _native.solve_real_rational_native

    def mutate(*args):
        result = native_solve(*args)
        domain._native_set = s.Interval(10, 20)._native_set
        return result

    monkeypatch.setattr(_native, "solve_real_rational_native", mutate)
    with pytest.raises(RuntimeError, match="domain mutated"):
        s.solveset(x**2-2, x, domain)


def test_custom_symbol_source_refuses_before_equation_arithmetic_or_printing():
    class CustomSymbol(s.Symbol):
        def __sub__(self, other):
            raise AssertionError("custom arithmetic must not run")

        def __str__(self):
            raise AssertionError("custom printer must not run")

    x = CustomSymbol("custom")
    equation = s.Eq(x, x, evaluate=False)
    with pytest.raises(NotImplementedError, match="custom|opaque"):
        s.solveset(equation, s.Symbol("x"), s.Interval(-1, 1))


def test_custom_domain_class_never_enters_the_checked_closed_set_lane(monkeypatch):
    class CustomInterval(s.Interval):
        pass

    monkeypatch.setattr(_native, "solve_real_rational_native", _forbid_approximation)
    x = s.Symbol("x")
    assert s.solveset(x, x, CustomInterval(-1, 1)) == s.FiniteSet(0)


def test_foreign_class_conversion_and_printing_hooks_do_not_run():
    class Foreign:
        @property
        def __class__(self):
            raise AssertionError("foreign class getter must not run")

        def _sympy_(self):
            raise AssertionError("foreign conversion must not run")

        def __str__(self):
            raise AssertionError("foreign printer must not run")

    with pytest.raises(TypeError, match="built-in scalar"):
        s.solveset(Foreign(), s.Symbol("x"), s.Interval(-1, 1))
    with pytest.raises(TypeError, match="variable must be a Symbol"):
        s.solveset(s.Symbol("x"), Foreign(), s.Interval(-1, 1))


def test_closed_native_sources_keep_typed_coordinates_and_original_holes():
    x = s.Symbol("x", real=True)
    domain = s.Interval(-2, 2)
    polynomial = x**5-x+1
    assert s.solveset(_native_expr(polynomial), x, domain) == s.FiniteSet(s.CRootOf(polynomial, 0))
    inverse = s.Pow(x, -1, evaluate=False)
    source = s.Mul(0, inverse, evaluate=False)
    assert s.solveset(_native_expr(source), x, domain) == s.Complement(domain, s.FiniteSet(0))


def test_domain_and_source_bounds_refuse_before_native_work(monkeypatch):
    x = s.Symbol("x")
    domain = s.FiniteSet(*range(129))
    monkeypatch.setattr(_native, "solve_real_rational_native", _forbid_approximation)
    with pytest.raises(ValueError, match="too many finite points"):
        s.solveset(x, x, domain)
    oversized = s.Add(*(x for _ in range(257)), evaluate=False)
    with pytest.raises(ValueError, match="bounded envelope"):
        s.solveset(oversized, x, s.Interval(-1, 1))
    with pytest.raises(ValueError, match="coefficient limit"):
        s.solveset(1 << 4096, x, s.Interval(-1, 1))


@pytest.mark.parametrize("oversized", [
    "points", "set_nodes", "set_depth", "scalar_operands", "scalar_depth", "coefficient",
    "scalar_total", "name_bytes",
])
def test_native_domain_preflight_bounds_the_stored_graph_before_getters(oversized):
    atom = _native.py_integer(1)
    if oversized == "points":
        domain = _native.SymSet.finite([_native.py_integer(i) for i in range(129)])
    elif oversized == "set_nodes":
        domain = _native.SymSet.union_many([
            _native.SymSet.interval(_native.py_integer(3*i), _native.py_integer(3*i+1))
            for i in range(128)
        ])
    elif oversized == "set_depth":
        domain = _native.SymSet.finite([atom])
        for _ in range(17):
            domain = _native.SymSet.complement_of(domain)
    elif oversized == "scalar_operands":
        domain = _native.SymSet.finite([_native.py_function("unknown", *([atom]*257))])
    elif oversized == "scalar_depth":
        for _ in range(65):
            atom = _native.py_function("unknown", atom)
        domain = _native.SymSet.finite([atom])
    elif oversized == "coefficient":
        domain = _native.SymSet.finite([_native.py_integer(1 << 4096)])
    elif oversized == "scalar_total":
        domain = _native.SymSet.finite([
            _native.py_function("unknown%d" % i, *([atom]*128)) for i in range(128)
        ])
    else:
        domain = _native.SymSet.finite([_native.py_function("f"*65537, atom)])
    with pytest.raises(ValueError, match="exact real solveset"):
        domain.real_domain_preflight()


def test_native_domain_preflight_is_structural_and_does_not_evaluate_sources():
    x = s.Symbol("x")
    undefined = _native_expr(s.Pow(0, -1, evaluate=False))
    unknown = _native.py_function("unknown", _native_expr(x))
    domain = _native.SymSet.finite([undefined, unknown])
    assert domain.real_domain_preflight() is None
    boundary = _native.SymSet.finite([_native.py_integer(i) for i in range(128)])
    assert boundary.real_domain_preflight() is None


def test_native_root_indices_with_multiplicity_are_canonicalized_before_filtering():
    x = s.Symbol("x")
    polynomial = (x-1)**2 * (x-2)**2
    raw = _native.py_function("CRootOf", _native_expr(polynomial), _native.py_integer(1))
    domain = s.Interval(raw, 3)
    assert s.solveset(x-1, x, domain) == s.FiniteSet(1)


def test_unrestricted_real_refusal_contract_and_complex_routes_are_preserved():
    x = s.Symbol("x")
    expression = s.Pow(x, 65, evaluate=False)
    assert isinstance(s.solveset(expression, x, s.S.Reals), s.ConditionSet)
    with pytest.raises(ValueError, match="resource limit"):
        s.solveset(expression, x, s.Interval(-1, 1))
    assert s.solveset(x**2+1, x) == s.FiniteSet(-s.I, s.I)
    assert s.solveset(x**2+1, x, s.FiniteSet(s.I, 1)) == s.FiniteSet(s.I)
