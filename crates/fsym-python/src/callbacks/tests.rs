//! Execute the actual PyO3 callback supervisor, including reentrant Python effects.

use super::*;
use pyo3::types::PyDict;
use std::ffi::CString;

fn run(source: &str, max_depth: usize, max_invocations: usize, steps: u64, millis: u64) {
    Python::initialize();
    Python::attach(|py| {
        let session = Py::new(
            py,
            PySupervisedCallbackSession::new(py, max_depth, max_invocations, steps, 32, 32, millis)
                .unwrap(),
        )
        .unwrap();
        let namespace = PyDict::new(py);
        namespace.set_item("session", &session).unwrap();
        namespace
            .set_item("trace_limit", MAX_CALLBACK_TRACE_RECORDS)
            .unwrap();
        namespace
            .set_item("key_limit", MAX_CALLBACK_KEY_BYTES)
            .unwrap();
        py.run(
            CString::new(source).unwrap().as_c_str(),
            Some(&namespace),
            Some(&namespace),
        )
        .unwrap();
        let session = session.borrow(py);
        assert_eq!(session.current_reentrancy_depth(), 0);
        let state = session.state.lock().unwrap();
        assert!(state.in_flight_effects.is_empty());
        assert!(state.effect_trace.len() <= MAX_CALLBACK_TRACE_RECORDS);
    });
}

#[test]
fn ownership_queries_the_actual_cpython_interpreter_module() {
    Python::initialize();
    Python::attach(|py| {
        // CPython 3.11/3.12 use _xxsubinterpreters; later runtimes may expose
        // _interpreters. Exercise the installed real module without global mocks.
        let module = py
            .import("_xxsubinterpreters")
            .or_else(|_| py.import("_interpreters"))
            .unwrap();
        let (current, main) = interpreter_ids_from_module(&module).unwrap();
        assert_eq!(raw_interpreter_ids(py).unwrap(), (current, main));
        assert_eq!(require_main_interpreter(current, main).unwrap(), main);
        assert_eq!(init_primary_interpreter(py).unwrap(), main);
        assert_eq!(check_interpreter_support(py).unwrap(), main);
    });
}

#[test]
fn missing_interpreter_capabilities_and_malformed_ids_do_not_mean_main() {
    Python::initialize();
    Python::attach(|py| {
        let missing = PyModule::new(py, "no_interpreter_capability").unwrap();
        assert!(interpreter_ids_from_module(&missing).is_err());
        assert!(interpreter_id(py.None().bind(py)).is_err());
        let invalid = PyTuple::new(py, [-1i64, 0]).unwrap();
        assert!(interpreter_id(invalid.as_any()).is_err());
        assert!(require_main_interpreter(1, 0).is_err());
        assert!(require_main_interpreter(7, 0).is_err());
    });
}

#[test]
fn fresh_occurrences_execute_again_and_share_the_invocation_limit() {
    run(
        r#"
calls = []
def callback():
    calls.append(len(calls) + 1)
    return calls[-1]
assert [session.invoke_fresh('numeric', 'slot', 1, callback, ()) for _ in range(3)] == [1, 2, 3]
assert session.invocations == 3
assert session.cached_effect_count() == 0
assert [row[2] for row in session.trace()] == ['fresh:1', 'fresh:2', 'fresh:3']
try:
    session.invoke_fresh('numeric', 'slot', 1, callback, ())
except ValueError:
    pass
else:
    raise AssertionError('a fresh call reset the invocation allowance')
assert calls == [1, 2, 3]
assert session.unknown_effect_tainted
"#,
        4,
        3,
        100,
        5000,
    );
}

#[test]
fn cached_effect_keys_are_tuples_and_do_not_collide_at_delimiters() {
    run(
        r#"
calls = []
def callback():
    calls.append(len(calls) + 1)
    return calls[-1]
assert session.invoke_once('a::b', 'c', 'd', 1, callback, ()) == 1
assert session.invoke_once('a', 'b::c', 'd', 1, callback, ()) == 2
assert session.invoke_once('a::b', 'c', 'd', 1, callback, ()) == 1
assert session.invoke_once('a', 'b::c', 'd', 1, callback, ()) == 2
assert calls == [1, 2]
assert session.invocations == session.cached_effect_count() == 2
assert [row[3] for row in session.trace()] == ['ok', 'ok', 'cached_hit', 'cached_hit']
"#,
        4,
        8,
        100,
        5000,
    );
}

#[test]
fn recursion_cannot_execute_an_in_flight_effect_twice_or_publish_after_catching_it() {
    run(
        r#"
calls = []
def callback():
    calls.append(True)
    try:
        session.invoke_once('hook', 'target', 'args', 1, callback, ())
    except RuntimeError as error:
        assert 'in flight' in str(error)
    return 17
try:
    session.invoke_once('hook', 'target', 'args', 1, callback, ())
except RuntimeError as error:
    assert 'unknown effect outcome cannot auto-retry' in str(error)
else:
    raise AssertionError('a tainted outer result escaped')
assert calls == [True]
assert session.invocations == 1
assert session.cached_effect_count() == 0
assert session.unknown_effect_tainted
"#,
        4,
        8,
        100,
        5000,
    );
}

#[test]
fn fresh_reentrancy_keeps_one_session_and_distinct_occurrence_ids() {
    run(
        r#"
depths = []
def inner():
    depths.append(session.current_reentrancy_depth)
    session.charge_nested_steps(2)
    return 3
def outer():
    depths.append(session.current_reentrancy_depth)
    session.charge_nested_steps(1)
    return 1 + session.invoke_fresh('numeric', 'inner', 1, inner, ())
assert session.invoke_fresh('numeric', 'outer', 1, outer, ()) == 4
assert depths == [1, 2]
assert session.invocations == 2
assert session.consumed_nested_steps == 3
assert session.cached_effect_count() == 0
assert [row[2] for row in session.trace()] == ['fresh:2', 'fresh:1']
"#,
        2,
        8,
        3,
        5000,
    );
}

#[test]
fn result_counter_exception_keeps_ownership_depth_and_exact_exception_identity() {
    run(
        r#"
marker = LookupError('counter failure')
calls = []
def callback():
    calls.append(True)
    return object()
def counter(value):
    assert session.current_reentrancy_depth == 1
    raise marker
try:
    session.invoke_fresh('numeric', 'slot', 1, callback, (), counter)
except LookupError as error:
    assert error is marker
else:
    raise AssertionError('counter failure was hidden')
assert calls == [True]
assert session.unknown_effect_tainted
try:
    session.invoke_fresh('numeric', 'slot', 1, callback, ())
except RuntimeError:
    pass
else:
    raise AssertionError('post-effect failure was retried')
assert calls == [True]
"#,
        4,
        8,
        100,
        5000,
    );
}

#[test]
fn counter_conversion_failure_taints_before_discarding_returned_python_objects() {
    run(
        r#"
observed = []
marker = ValueError('not a count')
class CounterResult:
    def __index__(self):
        raise marker
    def __del__(self):
        observed.append(session.unknown_effect_tainted)
def callback():
    return 7
def counter(value):
    return CounterResult()
try:
    session.invoke_once('hook', 'target', 'args', 1, callback, (), counter)
except ValueError as error:
    assert error is marker
else:
    raise AssertionError('an invalid output count escaped')
assert session.unknown_effect_tainted
assert session.cached_effect_count() == 0
# An exception traceback may retain CounterResult until the marker is cleared.
marker.__traceback__ = None
assert observed == [True]
"#,
        4,
        8,
        100,
        5000,
    );
}

#[test]
fn counter_finalization_and_cancellation_precede_publication() {
    run(
        r#"
finalized = []
class CounterResult:
    def __index__(self):
        return 1
    def __del__(self):
        finalized.append(session.current_reentrancy_depth)
        session.cancel()
def callback():
    return 11
def counter(value):
    return CounterResult()
try:
    session.invoke_fresh('numeric', 'slot', 1, callback, (), counter)
except RuntimeError as error:
    assert 'unknown effect outcome cannot auto-retry' in str(error)
else:
    raise AssertionError('finalizer cancellation was published')
assert finalized == [1]
assert session.cancelled and session.unknown_effect_tainted
"#,
        4,
        8,
        100,
        5000,
    );
}

#[test]
fn counter_wall_time_is_part_of_the_supervised_invocation() {
    run(
        r#"
import time
counted = []
def counter(value):
    counted.append(True)
    assert session.current_reentrancy_depth == 1
    time.sleep(0.03)
    return 1
try:
    session.invoke_fresh('numeric', 'slot', 1, lambda: 1, (), counter)
except RuntimeError as error:
    assert 'exceeded wall-time bound' in str(error)
else:
    raise AssertionError('time spent admitting the result was not counted')
assert session.cancelled and session.unknown_effect_tainted
assert counted == [True]
"#,
        4,
        8,
        100,
        5,
    );
}

#[test]
fn caught_nested_budget_failures_and_counter_reentrancy_cannot_publish() {
    run(
        r#"
def callback():
    try:
        session.charge_nested_steps(4)
    except ValueError:
        pass
    return 3
try:
    session.invoke_fresh('numeric', 'slot', 1, callback, ())
except RuntimeError:
    pass
else:
    raise AssertionError('catching a nested budget failure reset admission')
assert session.consumed_nested_steps == 4
assert session.unknown_effect_tainted
"#,
        1,
        8,
        3,
        5000,
    );
    run(
        r#"
calls = []
def inner():
    calls.append(True)
    return 1
def counter(value):
    try:
        session.invoke_fresh('numeric', 'nested', 1, inner, ())
    except RuntimeError:
        pass
    return 1
try:
    session.invoke_fresh('numeric', 'outer', 1, lambda: 1, (), counter)
except RuntimeError:
    pass
else:
    raise AssertionError('output admission escaped the depth limit')
assert calls == []
assert session.invocations == 1
assert session.unknown_effect_tainted
"#,
        1,
        8,
        100,
        5000,
    );
}

#[test]
fn nested_step_overflow_is_a_refusal_even_at_the_largest_configured_limit() {
    run(
        r#"
def callback():
    assert session.charge_nested_steps(2**64 - 1) == 2**64 - 1
    try:
        session.charge_nested_steps(1)
    except ValueError:
        pass
    else:
        raise AssertionError('counter saturation bypassed exhaustion')
    return 1
try:
    session.invoke_fresh('numeric', 'slot', 1, callback, ())
except RuntimeError:
    pass
else:
    raise AssertionError('an overflowed ledger published a value')
assert session.consumed_nested_steps == 2**64 - 1
assert session.unknown_effect_tainted
"#,
        4,
        8,
        u64::MAX,
        5000,
    );
}

#[test]
fn exception_rendering_does_not_run_inside_the_session_mutex() {
    run(
        r#"
rendered = []
class ReentrantError(Exception):
    def __str__(self):
        rendered.append(session.current_reentrancy_depth)
        return 'safe to render after refusal'
marker = ReentrantError()
def callback():
    raise marker
try:
    session.invoke_fresh('numeric', 'slot', 1, callback, ())
except ReentrantError as error:
    assert error is marker
    assert rendered == []
    assert str(error) == 'safe to render after refusal'
else:
    raise AssertionError('original exception was replaced')
assert rendered == [0]
assert session.trace()[0][3] == 'error:callback_or_result_admission'
"#,
        4,
        8,
        100,
        5000,
    );
}

#[test]
fn keys_and_trace_growth_are_bounded_before_additional_effects() {
    run(
        r#"
calls = []
def callback():
    calls.append(True)
    return 1
try:
    session.invoke_once('k' * (key_limit + 1), '', '', 1, callback, ())
except ValueError:
    pass
else:
    raise AssertionError('unbounded key accepted')
assert session.invocations == 0 and session.trace() == []
for _ in range(trace_limit):
    assert session.invoke_once('hook', 'target', 'args', 1, callback, ()) == 1
assert calls == [True]
try:
    session.invoke_fresh('numeric', 'slot', 1, callback, ())
except ValueError as error:
    assert 'trace' in str(error)
else:
    raise AssertionError('trace growth was unbounded')
assert len(session.trace()) == trace_limit
assert session.invocations == 1
"#,
        4,
        8,
        100,
        5000,
    );
}

#[test]
fn nested_calls_cannot_steal_the_outer_terminal_trace_reservation() {
    run(
        r#"
for _ in range(trace_limit - 1):
    session.invoke_once('hook', 'target', 'args', 1, lambda: 1, ())
def outer():
    try:
        session.invoke_fresh('numeric', 'inner', 1, lambda: 2, ())
    except ValueError:
        pass
    return 3
try:
    session.invoke_fresh('numeric', 'outer', 1, outer, ())
except RuntimeError:
    pass
else:
    raise AssertionError('caught trace exhaustion published an outer result')
assert len(session.trace()) == trace_limit
assert session.invocations == 2
assert session.unknown_effect_tainted
"#,
        4,
        8,
        100,
        5000,
    );
}

#[test]
fn fresh_calls_refuse_precancellation_and_foreign_thread_ownership() {
    run(
        r#"
import threading
calls = []
errors = []
def callback():
    calls.append(True)
    return 1
def worker():
    try:
        session.invoke_fresh('numeric', 'slot', 1, callback, ())
    except RuntimeError as error:
        errors.append(str(error))
thread = threading.Thread(target=worker)
thread.start()
thread.join()
assert len(errors) == 1 and 'ownership violation' in errors[0]
assert calls == [] and session.invocations == 0
session.cancel()
try:
    session.invoke_fresh('numeric', 'slot', 1, callback, ())
except RuntimeError as error:
    assert 'cancelled before hook invocation' in str(error)
else:
    raise AssertionError('a cancelled callback executed')
assert calls == [] and session.invocations == 0
"#,
        4,
        8,
        100,
        5000,
    );
}
