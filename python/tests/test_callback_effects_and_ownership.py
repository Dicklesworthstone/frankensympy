"""Supervised Python callback capability routing, effect discipline, and interpreter ownership tests (WS05 / fra-rc-effects-dpc)."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import threading
import time
import warnings
import weakref
from pathlib import Path

import pytest

import sympy as sp
from sympy import (
    Add,
    Derivative,
    Expr,
    Float,
    Function,
    FunctionClass,
    ImmutableDenseMatrix,
    Integer,
    Matrix,
    MutableDenseMatrix,
    Rational,
    S,
    Symbol,
    cos,
    diff,
    exp,
    sin,
    srepr,
    symbols,
)
from sympy.core import (
    SupervisedCallbackSession,
    check_interpreter_support,
    supervised_callback_session,
)

ORACLE_PYTHON = Path("/home/ubuntu/.venvs/fsym-oracle-sympy-1.14.0/bin/python")
PYTHON_ROOT = Path(__file__).resolve().parents[1]


def test_hero_constitutive_law_held_matrix_jacobian_differential() -> None:
    """Connected lower/differentiate/subs/evalf/Jacobian path on held + custom Function + Matrix matches SymPy 1.14.0."""
    script = """
import json, warnings
import sympy as sp
from sympy import Function, Add, MutableDenseMatrix, symbols, sin, exp, S, srepr

a, b, c, d, e, k, p, q, x, y = symbols("a b c d e k p q x y")

class ConstitutiveLaw(Function):
    nargs = 2
    eval_log = []
    diff_log = []
    evalf_log = []

    @classmethod
    def eval(cls, x_arg, k_arg):
        cls.eval_log.append((str(x_arg), str(k_arg)))
        if x_arg.is_zero:
            return S.Zero

    def _eval_derivative(self, s):
        type(self).diff_log.append((str(self.args[0]), str(self.args[1]), str(s)))
        x_arg, k_arg = self.args
        return x_arg.diff(s) * (k_arg * x_arg**2 + 1) + k_arg.diff(s) * (x_arg**3 / 3)

    def _eval_evalf(self, prec):
        type(self).evalf_log.append((str(self.args[0]), str(self.args[1]), int(prec)))
        x_arg, k_arg = self.args
        xv = x_arg._evalf(prec)
        kv = k_arg._evalf(prec)
        if xv.is_Number and kv.is_Number:
            return (xv + kv * (xv**3) / 3)._evalf(prec)
        return None

held = Add(a*x**2, -a*x**2, b*x*y, evaluate=False)
r0 = Add(held, (x - y)*(x**2 + x*y + y**2), sin(y), ConstitutiveLaw(x, k), -c, evaluate=False)
r1 = (x + y)*(x**2 - 2*x*y + y**2) + exp(x) - d
r2 = p*(x**2 - y**2) + q*(x - y)**2 + ConstitutiveLaw(y, k) - e

mutable_res = MutableDenseMatrix([r0, r1, r2])
snap = mutable_res.as_immutable()
# Mutating the source matrix after snapshot must not affect the immutable snapshot.
mutable_res[0, 0] = S.Zero

eval_after_construct = len(ConstitutiveLaw.eval_log)
J = snap.jacobian([x, y])
diff_after_jacobian = len(ConstitutiveLaw.diff_log)

subs_map = {a: 2, b: 3, c: 1, d: 2, e: 1, k: S(1)/2, p: 1, q: 2, x: 1, y: 2}
subbed_r = snap.subs(subs_map)
subbed_J = J.subs(subs_map)
eval_after_subs = len(ConstitutiveLaw.eval_log)

num_r = subbed_r.evalf(15)
evalf_after_r = list(ConstitutiveLaw.evalf_log)
num_J = subbed_J.evalf(15)

print(json.dumps({
    "held_srepr": srepr(held),
    "snap0_has_held": held in snap[0, 0].args,
    "eval_after_construct": eval_after_construct,
    "diff_after_jacobian": diff_after_jacobian,
    "diff_log": ConstitutiveLaw.diff_log,
    "eval_after_subs": eval_after_subs,
    "eval_log": ConstitutiveLaw.eval_log,
    "evalf_after_r": evalf_after_r,
    "subbed_J": [str(v) for v in subbed_J],
    "num_r": [str(v) for v in num_r],
    "num_J": [str(v) for v in num_J],
}))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PYTHON_ROOT)
    fsym_proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    fsym_data = json.loads(fsym_proc.stdout.strip())

    if ORACLE_PYTHON.exists():
        oracle_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        oracle_proc = subprocess.run(
            [str(ORACLE_PYTHON), "-c", script],
            env=oracle_env,
            capture_output=True,
            text=True,
            check=True,
        )
        oracle_data = json.loads(oracle_proc.stdout.strip())
        assert fsym_data == oracle_data


def test_exactly_once_callback_execution_and_session_trace() -> None:
    """Effectful hooks execute at most once per (hook_kind, target_key, args_key) within a session."""
    x, y = symbols("x y")
    calls = {"diff": 0, "subs": 0, "evalf": 0}

    class TrackedLaw(Function):
        nargs = 2

        def _eval_derivative(self, s):
            calls["diff"] += 1
            return (self.args[0] * self.args[1]).diff(s)

        def _eval_subs(self, old, new):
            calls["subs"] += 1
            if old == x:
                return TrackedLaw(new, self.args[1])
            return None

        def _eval_evalf(self, prec):
            calls["evalf"] += 1
            return Float(4.25, 15)

    # Repeated subexpression in outer product/sum:
    t = TrackedLaw(x, y)
    expr = t + x * (t ** 2)

    with supervised_callback_session() as session:
        d = diff(expr, x)
        # Even if a second diff on the same node is requested in the same session,
        # the hook result is served from the session's executed_effects cache:
        d2 = diff(t, x)
        assert d2 == y
        assert d == y + t ** 2 + 2 * x * y * t
        assert calls["diff"] == 1
        trace = session.trace()
        outcomes = [entry[3] for entry in trace if entry[0] == "_eval_derivative"]
        assert outcomes == ["ok", "cached_hit", "cached_hit"]
        assert session.cached_effect_count() == 1
        assert session.consumed_nested_steps > 0

    with supervised_callback_session() as session:
        s1 = expr.subs(x, 3)
        s2 = t.subs(x, 3)
        assert s2 == TrackedLaw(3, y)
        assert s1 == TrackedLaw(3, y) + 3 * (TrackedLaw(3, y) ** 2)
        assert calls["subs"] == 1

    with supervised_callback_session() as session:
        v1 = t.evalf(15)
        v2 = t.evalf(15)
        assert v1 == Float(4.25, 15)
        assert v2 == Float(4.25, 15)
        assert calls["evalf"] == 1


def test_cancellation_and_unknown_effect_taint_refuse_auto_retry() -> None:
    """Cancellation before/during callback or hook failure taints the session and forbids auto-retry."""
    x = Symbol("x")

    # 1. Cancellation before commit runs 0 callbacks.
    pre_calls = 0

    class PreCancelFn(Function):
        nargs = 1

        def _eval_derivative(self, s):
            nonlocal pre_calls
            pre_calls += 1
            return S.One

    with supervised_callback_session() as session:
        session.cancel()
        with pytest.raises(RuntimeError, match="cancelled before hook invocation"):
            diff(PreCancelFn(x), x)
        assert pre_calls == 0

    # 2. Cancellation in flight taints the session; subsequent retry is refused without calling hook again.
    inflight_calls = 0

    class InFlightCancelFn(Function):
        nargs = 1

        def _eval_derivative(self, s):
            nonlocal inflight_calls
            inflight_calls += 1
            session_ref.cancel()
            return S.One

    with supervised_callback_session() as session_ref:
        with pytest.raises(RuntimeError, match="unknown effect outcome cannot auto-retry"):
            diff(InFlightCancelFn(x), x)
        assert session_ref.unknown_effect_tainted is True
        assert inflight_calls == 1
        with pytest.raises(RuntimeError, match="unknown callback effect outcome cannot auto-retry"):
            diff(InFlightCancelFn(x), x)
        assert inflight_calls == 1

    # 3. Raising hook taints the session and preserves exact exception type/message; retry is refused.
    err_calls = 0

    class RaisingFn(Function):
        nargs = 1

        def _eval_derivative(self, s):
            nonlocal err_calls
            err_calls += 1
            warnings.warn("custom derivative warning before raise", UserWarning, stacklevel=2)
            raise ValueError("singular constitutive branch")

    with supervised_callback_session() as err_session:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="singular constitutive branch"):
                diff(RaisingFn(x), x)
        assert len(caught) == 1
        assert issubclass(caught[0].category, UserWarning)
        assert "custom derivative warning before raise" in str(caught[0].message)
        assert err_session.unknown_effect_tainted is True
        assert err_calls == 1
        with pytest.raises(RuntimeError, match="unknown callback effect outcome cannot auto-retry"):
            diff(RaisingFn(x), x)
        assert err_calls == 1


def test_slow_non_cooperative_callback_and_budget_bounds() -> None:
    """Enforce wall-time, reentrancy depth, invocation count, input/output node, and nested step bounds."""
    x = Symbol("x")

    # 1. Slow non-cooperative callback exceeding wall-time bound is tainted post-return and cannot retry.
    slow_calls = 0

    class SlowNonCooperativeFn(Function):
        nargs = 1

        def _eval_derivative(self, s):
            nonlocal slow_calls
            slow_calls += 1
            time.sleep(0.03)
            return S.One

    with supervised_callback_session(max_wall_time_ms=5) as slow_session:
        with pytest.raises(RuntimeError, match="exceeded wall-time bound"):
            diff(SlowNonCooperativeFn(x), x)
        assert slow_session.unknown_effect_tainted is True
        assert slow_calls == 1
        with pytest.raises(RuntimeError, match="unknown callback effect outcome cannot auto-retry"):
            diff(SlowNonCooperativeFn(x), x)
        assert slow_calls == 1

    # 2. Recursive callback exceeding max_reentrancy_depth refuses deterministically.
    class RecursiveFn(Function):
        nargs = 1

        def _eval_derivative(self, s):
            return diff(RecursiveFn(self.args[0] + S.One), s)

    with supervised_callback_session(max_reentrancy_depth=3) as rec_session:
        with pytest.raises(RuntimeError, match="reentrancy depth"):
            diff(RecursiveFn(x), x)
        assert rec_session.unknown_effect_tainted is True

    # 3. Input AST node bound refusal before callback execution.
    input_calls = 0

    class BoundedInputFn(Function):
        nargs = 1

        def _eval_derivative(self, s):
            nonlocal input_calls
            input_calls += 1
            return S.One

    big_arg = Add(*[x ** i for i in range(1, 15)], evaluate=False)
    with supervised_callback_session(max_input_nodes=5):
        with pytest.raises(ValueError, match="callback input node count"):
            diff(BoundedInputFn(big_arg), x)
    assert input_calls == 0

    # 4. Output AST node bound refusal after callback return.
    class HugeOutputFn(Function):
        nargs = 1

        def _eval_derivative(self, s):
            return Add(*[x ** i for i in range(1, 20)], evaluate=False)

    with supervised_callback_session(max_output_nodes=5) as out_session:
        with pytest.raises(ValueError, match="callback output node count"):
            diff(HugeOutputFn(x), x)
        assert out_session.unknown_effect_tainted is True

    # 5. Monotonic nested step budget exhaustion inside reentrant callback.
    class HeavyNestedFn(Function):
        nargs = 1

        def _eval_derivative(self, s):
            poly = (s ** 4 + 3 * s ** 3 - 2 * s + 1) ** 3
            return diff(poly, s) + diff(poly * s, s)

    with supervised_callback_session(max_nested_steps=3) as nested_session:
        with pytest.raises(ValueError, match="nested native request budget exhausted"):
            diff(HeavyNestedFn(x), x)
        assert nested_session.unknown_effect_tainted is True


def test_metaclass_subclass_override_and_dynamic_capability_change() -> None:
    """Verify custom FunctionClass metaclass, fdiff override, _imp_ override, and dynamic capability updates."""
    x, y = symbols("x y")
    meta_records: list[str] = []

    class CustomMeta(FunctionClass):
        def __new__(mcls, name, bases, ns):
            meta_records.append(name)
            return super().__new__(mcls, name, bases, ns)

    class BaseSpecial(Function, metaclass=CustomMeta):
        nargs = 1

        def fdiff(self, argindex=1):
            return cos(self.args[0]) + Integer(1)

    class SubSpecial(BaseSpecial):
        def fdiff(self, argindex=1):
            return Integer(2) * super().fdiff(argindex)

        @classmethod
        def _imp_(cls, arg):
            return float(arg) * 2.5

    assert "BaseSpecial" in meta_records
    assert "SubSpecial" in meta_records
    assert diff(BaseSpecial(x ** 2), x) == 2 * x * (cos(x ** 2) + 1)
    assert diff(SubSpecial(x ** 2), x) == 2 * x * (2 * cos(x ** 2) + 2)
    assert SubSpecial(2).evalf(15) == Float(5.0, 15)

    # Dynamic capability modification on a Function subclass must take effect immediately:
    class DynamicFn(Function):
        nargs = 1

    assert isinstance(diff(DynamicFn(x), x), Derivative)
    DynamicFn._eval_derivative = lambda self, s: Integer(7) * self.args[0].diff(s)
    assert diff(DynamicFn(x ** 2), x) == 14 * x
    delattr(DynamicFn, "_eval_derivative")
    assert isinstance(diff(DynamicFn(x), x), Derivative)


def test_matrix_alias_snapshot_and_mid_traversal_mutation_guard() -> None:
    """Verify Matrix input list isolation, immutable snapshot isolation, and mid-traversal mutation refusal."""
    x, y = symbols("x y")
    held = Add(x, x, evaluate=False)
    row = [held, y]
    m = MutableDenseMatrix([row])
    assert m.mutation_generation == 0
    row[0] = Integer(99)
    assert srepr(m[0, 0]) == srepr(held)

    snap = m.as_immutable()
    assert isinstance(snap, ImmutableDenseMatrix)
    assert snap.mutation_generation == 0

    m[0, 0] = Integer(42)
    assert m.mutation_generation == 1
    assert m[0, 0] == Integer(42)
    assert srepr(snap[0, 0]) == srepr(held)

    # Mid-traversal mutation inside applyfunc, row_op, or jacobian must raise RuntimeError:
    def mutating_apply(val):
        m[0, 1] = Integer(100)
        return val

    with pytest.raises(RuntimeError, match="Matrix mutated during callback traversal"):
        m.applyfunc(mutating_apply)

    def mutating_row_op(val, col_idx):
        m[0, 0] = Integer(77)
        return val

    with pytest.raises(RuntimeError, match="Matrix mutated during callback traversal"):
        m.row_op(0, mutating_row_op)

    class MatrixMutatingFn(Function):
        nargs = 1

        def _eval_derivative(self, s):
            target_mat[0, 0] = Integer(0)
            return S.One

    target_mat = MutableDenseMatrix([MatrixMutatingFn(x), y])
    with pytest.raises(RuntimeError, match="Matrix mutated during callback traversal"):
        target_mat.jacobian([x])


def test_gc_cycle_weakref_finalizer_and_no_class_retention() -> None:
    """User Function subclasses, cyclic graphs, weakrefs, and finalizers are collected on gc.collect()."""
    x = Symbol("x")
    finalized: list[str] = []

    def make_scoped_objects():
        class EphemeralLaw(Function):
            nargs = 1

            @classmethod
            def eval(cls, arg):
                if arg == 0:
                    return S.Zero

        inst = EphemeralLaw(x)
        cycle_partner = [inst]
        inst._cycle_ref = cycle_partner
        cycle_partner.append(cycle_partner)
        weakref.finalize(EphemeralLaw, finalized.append, "class_finalized")
        weakref.finalize(inst, finalized.append, "instance_finalized")
        return weakref.ref(EphemeralLaw), weakref.ref(inst)

    cls_ref, inst_ref = make_scoped_objects()
    gc.collect()
    gc.collect()
    assert cls_ref() is None
    assert inst_ref() is None
    assert "class_finalized" in finalized
    assert "instance_finalized" in finalized


def test_interpreter_and_thread_ownership_adversaries() -> None:
    """Cross-thread session access, subinterpreters, and free-threaded mode are explicitly refused."""
    session = SupervisedCallbackSession()
    assert check_interpreter_support() == session.owner_interpreter_id

    # Cross-thread ownership violation:
    thread_err: list[ BaseException ] = []

    def worker():
        try:
            session.verify_ownership()
        except BaseException as exc:  # noqa: BLE001
            thread_err.append(exc)

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert len(thread_err) == 1
    assert isinstance(thread_err[0], RuntimeError)
    assert "ownership violation" in str(thread_err[0])

    # Simulated subinterpreter refusal in fresh subprocess:
    env_sub = os.environ.copy()
    env_sub["PYTHONPATH"] = str(PYTHON_ROOT)
    env_sub["FSYM_SIMULATE_SUBINTERPRETER_ID"] = "7"
    proc_sub = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sympy.core import check_interpreter_support; check_interpreter_support()",
        ],
        env=env_sub,
        capture_output=True,
        text=True,
    )
    assert proc_sub.returncode != 0
    assert "unsupported subinterpreter configuration" in proc_sub.stderr

    # Simulated free-threaded Python refusal in fresh subprocess:
    env_ft = os.environ.copy()
    env_ft["PYTHONPATH"] = str(PYTHON_ROOT)
    env_ft["FSYM_FORCE_FREE_THREADED_REFUSAL"] = "1"
    proc_ft = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sympy.core import SupervisedCallbackSession; SupervisedCallbackSession()",
        ],
        env=env_ft,
        capture_output=True,
        text=True,
    )
    assert proc_ft.returncode != 0
    assert "unsupported free-threaded Python configuration" in proc_ft.stderr
