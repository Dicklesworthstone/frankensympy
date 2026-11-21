"""Signed constructor facts retain their domains and typed symbol owners.

The public expectations are pinned against an isolated SymPy 1.14.0 process.
Native-only tests additionally check that deduction changes no declaration,
typed identity, cache owner, or restored Dummy identity.
"""

from __future__ import annotations

import copy
import pickle

import pytest
import sympy as s


@pytest.mark.parametrize("constructor", [s.Symbol, s.Dummy])
@pytest.mark.parametrize("declarations, expected", [
    ({"real": True, "positive": False},
     {"nonpositive": True, "positive": False, "negative": None, "zero": None}),
    ({"real": True, "negative": False},
     {"nonnegative": True, "negative": False, "positive": None, "zero": None}),
    ({"real": True, "nonnegative": False},
     {"negative": True, "positive": False, "zero": False, "nonzero": True}),
    ({"real": True, "nonpositive": False},
     {"positive": True, "negative": False, "zero": False, "nonzero": True}),
    ({"integer": True, "even": False},
     {"odd": True, "even": False, "rational": True, "zero": False, "nonzero": True}),
    ({"integer": True, "odd": False},
     {"even": True, "odd": False, "rational": True, "zero": None, "nonzero": None}),
    ({"real": True, "zero": False},
     {"nonzero": True, "zero": False, "positive": None, "negative": None}),
    ({"real": True, "nonzero": False},
     {"zero": True, "integer": True, "even": True, "positive": False, "negative": False}),
    ({"positive": False},
     {"positive": False, "negative": None, "nonpositive": None, "real": None}),
    ({"zero": False},
     {"zero": False, "nonzero": None, "positive": None, "real": None}),
    ({"rational": False},
     {"rational": False, "integer": False, "real": None, "complex": None}),
    ({"real": False},
     {"real": False, "positive": False, "nonzero": False, "zero": False,
      "integer": False, "complex": None, "finite": None}),
    ({"finite": True},
     {"finite": True, "infinite": False, "complex": None, "real": None}),
    ({"complex": False},
     {"complex": False, "real": False, "integer": False, "finite": None}),
    ({"commutative": True},
     {"commutative": True, "real": None, "integer": None, "finite": None}),
    ({"positive": True, "commutative": True},
     {"positive": True, "real": True, "complex": True, "finite": True, "odd": None}),
])
def test_signed_constructor_deductions_match_the_pinned_scalar_domains(constructor, declarations, expected):
    symbol = constructor("signed_constructor_x", **declarations)
    for name, value in expected.items():
        assert getattr(symbol, "is_" + name) is value


@pytest.mark.parametrize("declarations, expected", [
    ({"imaginary": True},
     {"imaginary": True, "real": False, "complex": True, "finite": True,
      "nonzero": False, "zero": False, "irrational": False}),
    ({"real": True, "rational": False},
     {"irrational": True, "rational": False, "integer": False, "imaginary": False}),
    ({"composite": True},
     {"composite": True, "prime": False, "positive": True, "integer": True}),
    ({"even": True, "positive": True, "prime": False},
     {"composite": True, "prime": False, "even": True, "odd": False}),
    ({"infinite": True},
     {"infinite": True, "finite": False, "complex": False, "real": False,
      "zero": False, "nonzero": False, "positive": False}),
    ({"finite": False},
     {"infinite": True, "finite": False, "complex": False, "integer": False}),
    ({"commutative": False},
     {"commutative": False, "complex": False, "real": False, "integer": False,
      "finite": None}),
])
def test_symbol_only_declarations_do_not_enter_the_native_predicate_parser(declarations, expected):
    symbol = s.Symbol("signed_extra_x", **declarations)
    for name, value in expected.items():
        assert getattr(symbol, "is_" + name) is value


@pytest.mark.parametrize("constructor", [s.Symbol, s.Dummy])
@pytest.mark.parametrize("declarations", [
    {"positive": True, "negative": True},
    {"real": True, "complex": False},
    {"integer": True, "rational": False},
    {"even": True, "odd": True},
    {"integer": True, "even": False, "odd": False},
    {"real": True, "positive": False, "negative": False, "zero": False},
])
def test_inconsistent_constructor_facts_fail_in_both_declaration_orders(constructor, declarations):
    for ordered in (declarations, dict(reversed(tuple(declarations.items())))):
        with pytest.raises(ValueError):
            constructor("contradictory_constructor_x", **ordered)
    ordinary = constructor("contradictory_constructor_x")
    assert ordinary.is_real is None
    assert ordinary.is_positive is None


@pytest.mark.parametrize("constructor", [s.Symbol, s.Dummy])
def test_derived_facts_reach_sign_and_parity_operations(constructor):
    odd = constructor("derived_odd_x", integer=True, even=False)
    nonpositive = constructor("derived_nonpositive_x", real=True, positive=False)
    nonnegative = constructor("derived_nonnegative_x", real=True, negative=False)
    assert (odd + 1).is_even is True
    assert (3 * odd).is_odd is True
    assert s.Abs(nonpositive) == -nonpositive
    assert s.Abs(nonnegative) == nonnegative


@pytest.mark.parametrize("constructor", [s.Symbol, s.Dummy])
def test_local_declarations_are_independent_of_ambient_assumptions(constructor):
    symbol = constructor("ambient_constructor_x", positive=False)
    assert symbol.is_nonpositive is None
    with s.assuming(s.Q.real(symbol)):
        assert s.ask(s.Q.real(symbol)) is True
        assert symbol.is_nonpositive is None
        assert symbol.is_real is None
    assert symbol.is_nonpositive is None


@pytest.mark.parametrize("constructor", [s.Symbol, s.Dummy])
def test_same_named_atoms_do_not_share_signed_answers(constructor):
    odd = constructor("same_name_signed_x", integer=True, even=False)
    even = constructor("same_name_signed_x", integer=True, odd=False)
    unknown = constructor("same_name_signed_x", positive=False)
    for _ in range(3):
        assert odd.is_odd is True
        assert even.is_odd is False
        assert unknown.is_nonpositive is None
        assert odd != even
        assert odd != unknown


@pytest.mark.parametrize("constructor", [s.Symbol, s.Dummy])
def test_copy_and_pickle_preserve_signed_assumptions(constructor):
    symbol = constructor("serialized_signed_x", integer=True, even=False)
    for restored in (copy.copy(symbol), copy.deepcopy(symbol), pickle.loads(pickle.dumps(symbol))):
        assert restored == symbol
        assert hash(restored) == hash(symbol)
        assert restored.is_odd is True
        assert restored.is_even is False
        assert restored.is_nonzero is True
        if constructor is s.Dummy:
            assert restored.dummy_index == symbol.dummy_index


_NATIVE = hasattr(s.core, "_symbol_binding")


@pytest.mark.skipif(not _NATIVE, reason="native declaration and publication boundary")
@pytest.mark.parametrize("constructor", [s.Symbol, s.Dummy])
def test_deduction_does_not_change_native_identity_or_declared_payload(constructor):
    from sympy.core import _symbol_binding, _wrap

    symbol = constructor("typed_signed_x", integer=True, even=False)
    declarations = dict(symbol._assumptions)
    before = _symbol_binding(symbol)
    native_before = symbol._value
    assert symbol.is_odd is True
    assert symbol.is_nonzero is True
    assert symbol.is_finite is True
    after = _symbol_binding(symbol)
    assert symbol._assumptions == declarations == {"integer": True, "even": False}
    assert (before.name, before.identity_hex, before.plain) == (after.name, after.identity_hex, after.plain)
    assert symbol._value is native_before
    assert _wrap(native_before) is symbol
    if constructor is s.Symbol:
        assert symbol is s.Symbol(symbol.name, integer=1, even=0)


@pytest.mark.skipif(not _NATIVE, reason="native declaration and publication boundary")
def test_failed_construction_cannot_publish_a_native_handle_or_symbol_owner(monkeypatch):
    from sympy import core

    calls = []

    def unexpected_native_symbol(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("inconsistent declaration reached native construction")

    before_cache = dict(core._SYMBOL_CACHE)
    before_owners = dict(core._surface_symbols)
    monkeypatch.setattr(core._native, "py_symbol", unexpected_native_symbol)
    for constructor in (s.Symbol, s.Dummy):
        with pytest.raises(ValueError, match="inconsistent symbol assumptions"):
            constructor("unpublished_inconsistent_x", integer=True, even=False, odd=False)
    with pytest.raises(ValueError, match="inconsistent symbol assumptions"):
        core._restore_dummy(
            "unpublished_inconsistent_x", 123,
            (("integer", True), ("even", False), ("odd", False)),
        )
    assert calls == []
    assert core._SYMBOL_CACHE == before_cache
    assert dict(core._surface_symbols) == before_owners


@pytest.mark.skipif(not _NATIVE, reason="native private-declaration mutation guard")
def test_cached_deduction_does_not_hide_a_changed_or_contradictory_declaration():
    symbol = s.Symbol("mutated_signed_x", real=True, positive=False)
    original = symbol._assumptions
    assert symbol.is_nonpositive is True
    try:
        symbol._assumptions = {"positive": False}
        assert symbol.is_nonpositive is None
        symbol._assumptions = {"positive": True, "negative": True}
        with pytest.raises(ValueError, match="inconsistent symbol assumptions"):
            _ = symbol.is_positive
    finally:
        symbol._assumptions = original
    assert symbol.is_nonpositive is True


@pytest.mark.skipif(not _NATIVE, reason="existing bounded Dummy domain")
@pytest.mark.parametrize("declarations", [
    {"infinite": True}, {"finite": False}, {"commutative": False},
    {"extended_real": True}, {"imaginary": True},
])
def test_signed_closure_does_not_expand_the_existing_dummy_admission_boundary(declarations):
    with pytest.raises(NotImplementedError):
        s.Dummy("still_unsupported_dummy", **declarations)
