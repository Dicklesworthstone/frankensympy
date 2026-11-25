"""Complete modular nth roots, pinned public selection, and terminal refusals."""

import pytest

import sympy as sp
from sympy.core import _native, supervised_callback_session
from sympy.ntheory import nthroot_mod as package_nthroot_mod
from sympy.ntheory.residue_ntheory import nthroot_mod as module_nthroot_mod


def test_public_module_paths_refer_to_the_same_implementation():
    assert sp.nthroot_mod is package_nthroot_mod is module_nthroot_mod


@pytest.mark.parametrize("a,n,p,roots", [
    (11, 4, 19, [8, 11]),
    (68, 3, 109, [23, 32, 54]),
    (8, 3, 27, [2, 11, 20]),
    (8, 3, 125, [2]),
    (1, 6, 64, [1, 31, 33, 63]),
    (8, 3, 72, [2, 8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68]),
    (-1, 3, 7, [3, 5, 6]),
    (9, 3, 27, []),
    (2, 4, 19, []),
    (0, 3, 1, [0]),
])
def test_exact_prime_power_nonunit_and_composite_families(a, n, p, roots):
    actual = sp.nthroot_mod(a, n, p, all_roots=True)
    assert actual == roots
    assert sp.nthroot_mod(a, n, p) == (roots[0] if roots else None)
    assert sp.nthroot_mod(a + 17*p, n, p, True) == roots
    assert all(type(root) is int for root in actual)


def test_small_moduli_match_an_independent_complete_residue_enumeration():
    for modulus in range(1, 36):
        for degree in range(3, 8):
            for value in range(modulus):
                expected = [
                    root for root in range(modulus)
                    if pow(root, degree, modulus) == value
                ]
                assert sp.nthroot_mod(value, degree, modulus, True) == expected
                assert sp.nthroot_mod(value, degree, modulus) == (
                    expected[0] if expected else None
                )


def test_degree_two_keeps_the_pinned_sqrt_mod_scalar_selection():
    # The upstream degree-two shortcut selects 38 here, although 17 is the
    # least root. Preserve its observable selection instead of the docstring's
    # general least-root rule.
    assert sp.nthroot_mod(79, 2, 105) == sp.sqrt_mod(79, 105) == 38
    assert sp.nthroot_mod(79, 2, 105, True) == [17, 32, 38, 52, 53, 67, 73, 88]
    assert sp.nthroot_mod(3, 2, 7) is None
    assert sp.nthroot_mod(3, 2, 7, True) == []


def test_linear_congruences_and_zero_scalar_roots_need_no_enumeration():
    assert sp.nthroot_mod(-17, 1, 13) == 9
    assert sp.nthroot_mod(-17, 1, 13, True) == [9]
    assert sp.nthroot_mod(0, 3, 2**63) == 0
    assert sp.nthroot_mod(1, 1, 1, True) == [0]


def test_exact_sympy_integers_and_large_native_boundary_values():
    assert sp.nthroot_mod(sp.Integer(68), sp.Integer(3), sp.Integer(109)) == 23
    modulus = 2**63
    assert sp.nthroot_mod(729, 3, modulus, True) == [9]
    assert sp.nthroot_mod(531441, 6, modulus, True) == [
        9, 2**62 - 9, 2**62 + 9, modulus - 9,
    ]
    assert sp.nthroot_mod(modulus - 1, 2**64 - 1, modulus) == modulus - 1
    assert sp.nthroot_mod(8, 3, 18446744073709551557, True) == [2]


@pytest.mark.parametrize("degree", [0, -1, -10])
def test_nonpositive_degree_is_rejected(degree):
    with pytest.raises(ValueError, match="n should be positive"):
        sp.nthroot_mod(8, degree, 27)


@pytest.mark.parametrize("modulus", [-1, -27])
def test_negative_modulus_is_rejected(modulus):
    with pytest.raises(ValueError, match="p should be positive"):
        sp.nthroot_mod(8, 3, modulus)


def test_zero_modulus_preserves_the_profile_modulo_exception():
    with pytest.raises(ZeroDivisionError):
        sp.nthroot_mod(8, 3, 0)


@pytest.mark.parametrize("values", [
    (1.5, 3, 7), (8, 3.5, 7), (8, 3, 7.5),
    (sp.Rational(1, 2), 3, 7), (8, sp.Rational(3, 2), 7),
    ("8", 3, 7),
])
def test_lossy_noninteger_sources_are_not_coerced(values):
    with pytest.raises(ValueError):
        sp.nthroot_mod(*values)


def test_bounded_failures_never_become_false_nonexistence():
    with pytest.raises(RuntimeError, match="131072"):
        sp.nthroot_mod(0, 3, 2**26, True)
    with pytest.raises(RuntimeError, match="65537"):
        sp.nthroot_mod(8, 3, 4294967311, True)
    with pytest.raises(RuntimeError, match="65537"):
        sp.nthroot_mod(8, 3, 4294967311)
    # The later nonresidue at seven wins over a huge local root count at two.
    assert sp.nthroot_mod(2 * 2**40, 3, 7 * 2**40, True) == []
    assert sp.nthroot_mod(2 * 2**40, 3, 7 * 2**40) is None


def test_native_failure_is_terminal_without_square_or_oracle_fallback(monkeypatch):
    from sympy.core import _native

    def stopped(*_args, **_kwargs):
        raise RuntimeError("injected nth-root resource refusal")

    monkeypatch.setattr(_native, "modular_nth_roots_fn", stopped)
    monkeypatch.setattr(_native, "modular_nth_root_fn", stopped)
    for all_roots in [False, True]:
        with pytest.raises(RuntimeError, match="injected nth-root resource refusal"):
            sp.nthroot_mod(68, 3, 109, all_roots)


@pytest.mark.parametrize("all_roots", [False, True])
def test_active_request_exhaustion_is_terminal_and_cumulative(all_roots):
    with supervised_callback_session(max_nested_steps=1000) as session:
        with pytest.raises(ValueError, match="nested native request budget exhausted"):
            sp.nthroot_mod(68, 3, 109, all_roots)
        consumed = session.consumed_nested_steps
        assert consumed <= session.max_nested_steps
        assert session.cancelled
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            sp.nthroot_mod(68, 3, 109, all_roots)
        assert session.consumed_nested_steps == consumed


def test_successful_native_calls_share_the_active_request_ledger():
    with supervised_callback_session(max_nested_steps=600_000_000) as session:
        assert sp.nthroot_mod(8, 3, 27, True) == [2, 11, 20]
        first_cost = session.consumed_nested_steps
        assert first_cost > 0
        assert sp.nthroot_mod(8, 3, 27) == 2
        assert session.consumed_nested_steps > first_cost
        assert not session.cancelled


@pytest.mark.parametrize("function,a,n,p,expected", [
    (_native.modular_nth_root_fn, 0, 3, 17, 0),
    (_native.modular_nth_root_fn, 9, 3, 27, None),
    (_native.modular_nth_roots_fn, 9, 3, 27, []),
    (_native.modular_nth_roots_fn, 8, 3, 27, [2, 11, 20]),
])
def test_result_admission_cannot_reset_a_spent_native_request(function, a, n, p, expected):
    with supervised_callback_session(max_nested_steps=600_000_000) as measured:
        assert function(a, n, p, session=measured) == expected
        cost = measured.consumed_nested_steps
    assert cost > 0
    # The deterministic native phases fit, but publication has one fewer
    # step than the successful request. No value, None, or empty set escapes.
    with supervised_callback_session(max_nested_steps=cost - 1) as session:
        with pytest.raises(ValueError, match="nested native request budget exhausted"):
            function(a, n, p, session=session)
        consumed = session.consumed_nested_steps
        assert consumed <= session.max_nested_steps
        assert session.cancelled
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            function(a, n, p, session=session)
        assert session.consumed_nested_steps == consumed


@pytest.mark.parametrize("n,all_roots", [
    (1, False), (1, True), (2, False), (2, True), (3, False), (3, True),
])
def test_cancelled_request_rejects_fast_paths_and_families(n, all_roots):
    with supervised_callback_session(max_nested_steps=600_000_000) as session:
        session.cancel()
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            sp.nthroot_mod(0, n, 27, all_roots)
        assert session.consumed_nested_steps == 0


@pytest.mark.parametrize("n", [1, 2])
def test_shortcut_budget_refusal_also_marks_the_request_terminal(n):
    with supervised_callback_session(max_nested_steps=0) as session:
        with pytest.raises(ValueError, match="nested native request budget exhausted"):
            sp.nthroot_mod(8, n, 27)
        assert session.cancelled
        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            sp.nthroot_mod(8, n, 27)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_cancellation_during_flag_admission_cannot_publish_a_result(n):
    with supervised_callback_session(max_nested_steps=600_000_000) as session:
        class CancellingFlag:
            def __bool__(self):
                session.cancel()
                return True

        with pytest.raises(RuntimeError, match="cancelled|tainted"):
            sp.nthroot_mod(0, n, 27, CancellingFlag())
