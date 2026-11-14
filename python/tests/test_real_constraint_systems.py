"""Complete conjunctions through the actual Python/native shared partition."""

import importlib

import pytest
import sympy as s
from sympy.core import Basic, _native, _native_expr, _symbol_binding


def _forbid(*args, **kwargs):
    raise AssertionError("exact real constraint systems must keep their checked native lane")


def _atoms(expression):
    if isinstance(expression, s.Relational):
        return {expression.canonical}
    result = set()
    for child in expression.args:
        result.update(_atoms(child))
    return result


def test_common_algebraic_partition_decides_empty_intersections_without_approximation(monkeypatch):
    monkeypatch.setattr(s, "_numeric_poly_roots", _forbid)
    monkeypatch.setattr(s.CRootOf, "evalf", _forbid)
    monkeypatch.setattr(_native, "solve_inequality_native", _forbid)
    x = s.Symbol("constraint_empty_x")
    for relations in (
        [s.Le(x*x, 2, evaluate=False), s.Ge(x*x, 3, evaluate=False)],
        [s.Le(x**4-x-1, 0, evaluate=False), s.Ge(x, 2, evaluate=False)],
        [s.Eq(x**5-x+1, 0, evaluate=False), s.Gt(x, 0, evaluate=False)],
    ):
        assert s.reduce_inequalities(relations, x) is s.S.false


def test_common_partition_removes_foreign_algebraic_bounds_instead_of_retaining_intersections(monkeypatch):
    monkeypatch.setattr(s.CRootOf, "evalf", _forbid)
    x = s.Symbol("constraint_quartic_x")
    polynomial = x**4-x-1
    upper = s.CRootOf(polynomial, 1)
    result = s.reduce_inequalities([
        s.Le(polynomial, 0, evaluate=False), s.Ge(x, 0, evaluate=False),
    ], x)
    assert _atoms(result) == {
        s.Le(0, x, evaluate=False).canonical,
        s.Le(x, upper, evaluate=False).canonical,
    }
    assert s.solve([s.Le(polynomial, 0, evaluate=False), s.Ge(x, 0, evaluate=False)], x) == result


def test_equalities_and_disequalities_filter_complete_algebraic_point_sets():
    x = s.Symbol("constraint_equality_x")
    polynomial = x**4-x-1
    upper = s.CRootOf(polynomial, 1)
    assert s.reduce_inequalities([
        s.Eq(polynomial, 0, evaluate=False), s.Ge(x, 0, evaluate=False),
    ], x) == s.Eq(x, upper, evaluate=False)
    assert s.reduce_inequalities([
        s.Eq(x*x-2, 0, evaluate=False), s.Ne(x*x-2, 0, evaluate=False),
    ], x) is s.S.false
    selected = s.reduce_inequalities([
        s.Eq((x-1)*(x+1), 0, evaluate=False), s.Ne(x-1, 0, evaluate=False),
    ], x)
    assert selected == s.Eq(x, -1, evaluate=False)


def test_disequality_keeps_cancelled_and_held_original_domain_holes():
    x = s.Symbol("constraint_holes_x")
    reciprocal = s.Pow(x, -1, evaluate=False)
    held_one = s.Mul(x, reciprocal, evaluate=False)
    held_zero = s.Add(held_one, -1, evaluate=False)
    assert s.reduce_inequalities(s.Ne(held_zero, 0, evaluate=False), x) is s.S.false
    result = s.reduce_inequalities([
        s.Ne(held_one, 0, evaluate=False), s.Ge(x, 0, evaluate=False),
    ], x)
    assert _atoms(result) == {
        s.Lt(0, x, evaluate=False).canonical,
        s.Lt(x, s.oo, evaluate=False).canonical,
    }
    nested = s.Pow(reciprocal, 0, evaluate=False)
    assert s.reduce_inequalities([
        s.Ne(nested, 0, evaluate=False), s.Eq(x, 0, evaluate=False),
    ], x) is s.S.false


def test_held_relational_sides_preserve_their_domains_before_subtraction():
    x = s.Symbol("constraint_sides_x")
    reciprocal = s.Pow(x-1, -1, evaluate=False)
    result = s.reduce_inequalities([
        s.Eq(reciprocal, reciprocal, evaluate=False), s.Ge(x, 1, evaluate=False),
    ], x)
    assert _atoms(result) == {
        s.Lt(1, x, evaluate=False).canonical,
        s.Lt(x, s.oo, evaluate=False).canonical,
    }
    assert s.reduce_inequalities(s.Ne(reciprocal, reciprocal, evaluate=False), x) is s.S.false


def test_true_cells_join_across_an_unused_internal_critical_point():
    x = s.Symbol("constraint_join_x")
    result = s.reduce_inequalities([
        s.Le(x*x-2, 0, evaluate=False), s.Ge(x*x, 0, evaluate=False),
    ], x)
    assert _atoms(result) == {
        s.Le(-s.sqrt(2), x, evaluate=False).canonical,
        s.Le(x, s.sqrt(2), evaluate=False).canonical,
    }


def test_constant_relation_proofs_do_not_leak_an_inferred_dummy_variable():
    for relation in (s.Eq(1, 1, evaluate=False), s.Ne(1, 0, evaluate=False),
                     s.Gt(1, 0, evaluate=False)):
        assert s.reduce_inequalities([relation]) is s.S.true
    for relation in (s.Eq(1, 0, evaluate=False), s.Ne(1, 1, evaluate=False),
                     s.Gt(0, 1, evaluate=False)):
        assert s.reduce_inequalities([relation]) is s.S.false
    undefined = s.Mul(0, s.Pow(0, -1, evaluate=False), evaluate=False)
    assert s.reduce_inequalities([s.Ne(undefined, 0, evaluate=False)]) is s.S.false


def test_boolean_constants_never_hide_source_limits_or_unsupported_relations(monkeypatch):
    x = s.Symbol("constraint_constants_x")
    calls = []
    original = _native.solve_real_rational_constraints_native

    def checked(sources, variable, operations, conjunction_false=False):
        calls.append((len(sources), conjunction_false))
        return original(sources, variable, operations, conjunction_false)

    monkeypatch.setattr(_native, "solve_real_rational_constraints_native", checked)
    assert s.reduce_inequalities([s.S.false, s.Ge(x, 0, evaluate=False)], x) is s.S.false
    assert calls == [(1, True)]
    for source in (s.Pow(x, 65, evaluate=False), s.Mul(0, s.Integer(1 << 4096), evaluate=False)):
        for constants in ([s.S.false], [s.S.true]):
            with pytest.raises(NotImplementedError, match="resource limit"):
                s.reduce_inequalities([*constants, s.Ge(source, 0, evaluate=False)], x)
    with pytest.raises(NotImplementedError, match="resource limit"):
        s.reduce_inequalities([
            s.S.false, s.Ge(s.sin(x), 0, evaluate=False),
            s.Ge(s.Pow(x, 65, evaluate=False), 0, evaluate=False),
        ], x)
    with pytest.raises(NotImplementedError, match="disequality"):
        s.reduce_inequalities([s.S.false, s.Ne(s.sin(x), 0, evaluate=False)], x)
    assert s.reduce_inequalities([s.S.true, s.S.true], x) is s.S.true
    assert s.reduce_inequalities([s.S.true, s.S.false], x) is s.S.false


def test_bridge_returns_one_complete_common_basis_and_validates_original_inputs():
    x = s.Symbol("constraint_bridge_x")
    sources = [_native_expr(x*x-2), _native_expr(x*x-3)]
    polynomial, gaps, points = _native.solve_real_rational_constraints_native(
        sources, _symbol_binding(x), [">=", "<="],
    )
    assert s.Poly(polynomial, x) == s.Poly(x**4-5*x*x+6, x)
    assert gaps == [False, True, False, True, False]
    assert points == [True, True, True, True]
    for operations in ([">="], [">=", "bad"]):
        with pytest.raises(ValueError):
            _native.solve_real_rational_constraints_native(sources, _symbol_binding(x), operations)
    with pytest.raises(ValueError, match="32"):
        _native.solve_real_rational_constraints_native([_native_expr(x)] * 33, _symbol_binding(x), ["=="] * 33)
    with pytest.raises(ValueError, match="resource limit"):
        _native.solve_real_rational_constraints_native(
            [_native_expr(s.sin(x)), _native_expr(s.Pow(x, 65, evaluate=False))],
            _symbol_binding(x), [">", ">"], True,
        )


def test_typed_symbol_identity_dummy_inference_and_plain_symbol_collision_are_retained():
    x = s.Dummy("constraint_dummy", positive=True)
    result = s.reduce_inequalities([s.Le(x*x, 4, evaluate=False), s.Ge(x, 0, evaluate=False)])
    assert _atoms(result) == {
        s.Le(0, x, evaluate=False).canonical,
        s.Le(x, 2, evaluate=False).canonical,
    }
    plain = s.Symbol("constraint_identity")
    positive = s.Symbol("constraint_identity", positive=True)
    assert _native.solve_real_rational_constraints_native(
        [_native_expr(positive)], _symbol_binding(plain), [">"],
    ) is None
    with pytest.raises((ValueError, NotImplementedError)):
        s.reduce_inequalities([s.Ge(plain, 0, evaluate=False), s.Le(positive, 1, evaluate=False)])


def test_recognized_native_refusal_and_corrupted_root_order_cannot_fall_back(monkeypatch):
    x = s.Symbol("constraint_refusal_x")
    relation = s.Le(x**4-x-1, 0, evaluate=False)
    monkeypatch.setattr(_native, "solve_inequality_native", _forbid)
    with monkeypatch.context() as patch:
        patch.setattr(_native, "solve_real_rational_constraints_native", lambda *args: None)
        with pytest.raises(NotImplementedError, match="admitted real constraints"):
            s.reduce_inequalities([relation], x)
    roots = importlib.import_module("sympy.polys.rootoftools")
    original = roots.real_roots

    def reversed_roots(*args, **kwargs):
        return list(reversed(original(*args, **kwargs)))

    monkeypatch.setattr(roots, "real_roots", reversed_roots)
    with pytest.raises(NotImplementedError, match="ordered root"):
        s.reduce_inequalities([relation], x)


def test_source_mutation_during_native_work_is_detected_before_publication(monkeypatch):
    x = s.Symbol("constraint_mutation_x")
    source = s.Add(x, 1, evaluate=False)
    relation = s.Ge(source, 0, evaluate=False)
    original = _native.solve_real_rational_constraints_native

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        s.Add.__dict__["_args"].__set__(source, (x, s.Integer(2)))
        return result

    monkeypatch.setattr(_native, "solve_real_rational_constraints_native", mutate)
    with pytest.raises(RuntimeError, match="mutated"):
        s.reduce_inequalities([relation], x)


def test_final_relational_publication_rechecks_sources_and_builtin_owners(monkeypatch):
    x = s.Symbol("constraint_publication_x")
    source = s.Add(x, 1, evaluate=False)
    relation = s.Ge(source, 0, evaluate=False)
    original = s._set_as_relational

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        s.Add.__dict__["_args"].__set__(source, (x, s.Integer(2)))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(s, "_set_as_relational", mutate)
        with pytest.raises(RuntimeError, match="mutated"):
            s.reduce_inequalities([relation], x)
    fresh = s.Ge(x, 0, evaluate=False)
    monkeypatch.setattr(s.Interval, "__new__", staticmethod(_forbid))
    with pytest.raises(NotImplementedError, match="original built-in"):
        s.reduce_inequalities([fresh], x)


def test_stale_scalar_owner_and_opaque_relations_are_rejected_without_callbacks():
    x = s.Symbol("constraint_owner_x")
    source = object.__new__(s.Integer)
    Basic.__dict__["_value"].__set__(source, _native_expr(x))
    with pytest.raises(NotImplementedError, match="scalar owner"):
        s.reduce_inequalities(s.Ge(source, 0, evaluate=False), x)

    class ForeignRelation(s.Ge):
        @property
        def free_symbols(self):
            raise AssertionError("custom relation hooks must not run")

    relation = ForeignRelation(x, 0, evaluate=False)
    with pytest.raises(NotImplementedError, match="built-in relations"):
        s.reduce_inequalities([s.S.false, relation], x)


def test_existing_closed_elementary_and_piecewise_equation_fallbacks_remain_available():
    x = s.Symbol("constraint_fallback_x")
    elementary = s.reduce_inequalities([s.Ge(s.Abs(x), 1, evaluate=False)], x)
    assert elementary is not s.S.false
    piecewise = s.Piecewise((x, s.Ge(x, 0, evaluate=False)), (0, True), evaluate=False)
    equation = s.Eq(piecewise, 0, evaluate=False)
    assert s.reduce_inequalities(equation) == s.reduce_inequalities(equation, x)
