"""Semantic free symbols survive arithmetic around conditional expressions."""

import pytest
import sympy as s


@pytest.mark.parametrize("wrap", [
    pytest.param(lambda p, z: p, id="direct"),
    pytest.param(lambda p, z: -p, id="negative"),
    pytest.param(lambda p, z: -s.Rational(2, 3)*p, id="scaled"),
    pytest.param(lambda p, z: p**2, id="power"),
    pytest.param(lambda p, z: s.sin(p), id="closed-function"),
    pytest.param(lambda p, z: s.exp(-p), id="nested-function"),
    pytest.param(lambda p, z: p + z, id="sum"),
    pytest.param(lambda p, z: (p + z)**2, id="nested-power"),
    pytest.param(lambda p, z: s.Mul(-1, p, evaluate=False), id="held-product"),
    pytest.param(lambda p, z: s.Add(p, z, evaluate=False), id="held-sum"),
    pytest.param(lambda p, z: s.Pow(p, 2, evaluate=False), id="held-power"),
])
def test_wrapped_conditional_primitive_has_only_its_semantic_symbols(wrap):
    x, a, b, z = s.symbols("piecewise_fs_x piecewise_fs_a piecewise_fs_b piecewise_fs_z")
    conditional = s.Piecewise(
        (s.cos(a*x)/(3*a), s.Ne(a, 0)), (x*s.sin(b), True),
    )
    expression = wrap(conditional, z)
    expected = {x, a, b}
    if expression.has(z):
        expected.add(z)
    assert expression.free_symbols == expected
    assert {id(symbol) for symbol in expression.free_symbols} == {
        id(symbol) for symbol in expected
    }


@pytest.mark.parametrize("wrap", [
    pytest.param(lambda p: -p, id="negative"),
    pytest.param(lambda p: 3*p, id="scaled"),
    pytest.param(lambda p: p**2, id="power"),
    pytest.param(lambda p: s.sin(p), id="closed-function"),
])
def test_true_named_value_is_a_real_free_symbol(wrap):
    x, y = s.symbols("piecewise_value_x piecewise_value_y")
    value = s.Symbol("True")
    expression = wrap(s.Piecewise((value, x > 0), (y, True)))
    assert expression.free_symbols == {value, x, y}
    assert any(symbol is value for symbol in expression.free_symbols)


@pytest.mark.parametrize("guard_factory", [s.Symbol, s.Dummy])
@pytest.mark.parametrize("wrap", [
    pytest.param(lambda p: -p, id="negative"),
    pytest.param(lambda p: s.sin(p), id="closed-function"),
])
def test_true_named_symbolic_guard_survives_default_marker(wrap, guard_factory):
    x, y = s.symbols("piecewise_guard_x piecewise_guard_y")
    guard = guard_factory("True")
    expression = wrap(s.Piecewise((x, guard), (y, True)))
    assert expression.free_symbols == {guard, x, y}
    assert any(symbol is guard for symbol in expression.free_symbols)


def test_same_named_assumed_symbols_and_dummies_keep_their_identities():
    plain = s.Symbol("True")
    real = s.Symbol("True", real=True)
    first, second = s.Dummy("True"), s.Dummy("True")
    conditional = s.Piecewise((plain + first, second), (real, True))
    expression = s.sin(-conditional) + second
    expected = {plain, real, first, second}
    assert len(expected) == 4
    assert expression.free_symbols == expected
    assert {id(symbol) for symbol in expression.free_symbols} == {
        id(symbol) for symbol in expected
    }


def test_bound_algebraic_root_generator_stays_bound_inside_a_branch():
    x, t = s.symbols("piecewise_root_x piecewise_root_t")
    root = s.CRootOf(t**3 - t - 1, t, 0)
    conditional = s.Piecewise((root, x > 0), (x, True))
    assert (-conditional + t).free_symbols == {x, t}
    assert s.sin(conditional).free_symbols == {x}


def test_integral_binder_keeps_its_existing_symbol_scope():
    x, y = s.symbols("piecewise_bound_x piecewise_bound_y")
    integrand = s.Piecewise((x, y > 0), (y, True))
    integral = s.Integral(integrand, (x, 0, 1))
    assert integral.free_symbols == {y}
    assert (2*integral).free_symbols == {y}


def test_opaque_function_keeps_its_existing_surface_children():
    x, y = s.symbols("piecewise_opaque_x piecewise_opaque_y")
    opaque = s.Function("piecewise_opaque_owner")
    conditional = s.Piecewise((x, y > 0), (y, True))
    expression = s.Mul(2, opaque(conditional), evaluate=False)
    assert expression.free_symbols == {x, y}


@pytest.mark.skipif(not hasattr(s.core, "_native"), reason="preview native admission boundary")
def test_preview_changed_free_symbols_property_keeps_native_binding_path(monkeypatch):
    x, y = s.symbols("piecewise_changed_property_x piecewise_changed_property_y")
    conditional = s.Piecewise((x, y > 0), (y, True))
    expression = s.Mul(-1, conditional, evaluate=False)
    calls = []

    def changed(self):
        calls.append(self)
        return set()

    monkeypatch.setattr(s.Piecewise, "free_symbols", property(changed))
    # This is intentionally the legacy native symbol result: the changed
    # semantic hook must not run, including through the opaque fallback.
    assert expression.free_symbols == {x, y, s.Symbol("True")}
    assert calls == []


@pytest.mark.skipif(not hasattr(s.core, "_native"), reason="preview native admission boundary")
def test_preview_changed_free_symbols_body_keeps_native_binding_path(monkeypatch):
    x, y = s.symbols("piecewise_changed_body_x piecewise_changed_body_y")
    conditional = s.Piecewise((x, y > 0), (y, True))
    expression = s.Mul(-1, conditional, evaluate=False)
    original = s.Piecewise.__dict__["free_symbols"].fget
    calls = []
    namespace = {}
    exec("def changed(self):\n    _preview_free_symbols_calls.append(self)\n    return set()", namespace)
    monkeypatch.setitem(original.__globals__, "_preview_free_symbols_calls", calls)
    monkeypatch.setattr(original, "__code__", namespace["changed"].__code__)
    assert expression.free_symbols == {x, y, s.Symbol("True")}
    assert calls == []
