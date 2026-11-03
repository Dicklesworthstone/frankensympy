//! Supervised Python callback capability routing and interpreter ownership (WS05 / fra-rc-effects-dpc).
//!
//! Arbitrary Python callbacks are not assumed pure, deterministic, terminating,
//! thread-safe, reentrant, or honest (`AGENTS.md` §18). This module enforces:
//! - Interpreter and GIL ownership verification (`check_interpreter_support`),
//!   explicitly refusing free-threaded (`sys._is_gil_enabled() == False`) and
//!   subinterpreter configurations rather than inheriting certification.
//! - Thread and interpreter context binding on [`PySupervisedCallbackSession`],
//!   refusing cross-thread or cross-interpreter reuse.
//! - Exactly-once execution per `(hook_kind, target_key, args_key)` within a
//!   request session across fallback/race paths.
//! - Taint-on-unknown-effect (`unknown_effect_tainted`): if a callback raises,
//!   exceeds its wall-time bound, exceeds output node bounds, or is cancelled
//!   while in flight, the session is tainted and any auto-retry or fallback
//!   invocation is refused.
//! - Preflight input/output AST node bounds, reentrancy depth bounds, total
//!   invocation bounds, and monotonic nested native step accounting (fallback
//!   never resets accounting).
//! - Explicit documentation and post-return enforcement of the CPython
//!   non-cooperative callback forced-cancellation limitation.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use pyo3::exceptions::{PyImportError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Primary interpreter ID recorded when `fsym_python` initializes.
static PRIMARY_INTERPRETER_ID: OnceLock<i64> = OnceLock::new();

/// Combined UTF-8 payload of the diagnostic key fields for one callback.
const MAX_CALLBACK_KEY_BYTES: usize = 4096;
/// Includes successful calls, cache hits and refused attempts. In-flight calls
/// reserve their terminal record before executing any Python code.
const MAX_CALLBACK_TRACE_RECORDS: usize = 4096;

type EffectKey = (String, String, String);

fn interpreter_id(value: &Bound<'_, PyAny>) -> PyResult<i64> {
    if let Ok(id) = value.extract::<i64>()
        && id >= 0
    {
        return Ok(id);
    }
    // Newer CPython exposes an (ID, whence) tuple, while the older module
    // returns an integer-like InterpreterID. Neither case needs repr parsing.
    if let Ok(tuple) = value.cast::<PyTuple>()
        && tuple.len() == 2
        && let Ok(id) = tuple.get_item(0)?.extract::<i64>()
        && id >= 0
    {
        return Ok(id);
    }
    Err(PyRuntimeError::new_err(
        "unsupported CPython interpreter identifier format",
    ))
}

fn interpreter_ids_from_module(module: &Bound<'_, PyModule>) -> PyResult<(i64, i64)> {
    let current = module
        .call_method0("get_current")
        .map_err(|_| PyRuntimeError::new_err("cannot verify the current CPython interpreter"))?;
    let main = module
        .call_method0("get_main")
        .map_err(|_| PyRuntimeError::new_err("cannot verify the main CPython interpreter"))?;
    Ok((interpreter_id(&current)?, interpreter_id(&main)?))
}

/// Query real CPython ownership. Both supported module spellings expose the
/// independently known main interpreter; an unavailable API never implies ID 0.
fn raw_interpreter_ids(py: Python<'_>) -> PyResult<(i64, i64)> {
    for name in ["_interpreters", "_xxsubinterpreters"] {
        match py.import(name) {
            Ok(module) => return interpreter_ids_from_module(&module),
            Err(error) if error.is_instance_of::<PyImportError>(py) => continue,
            Err(error) => return Err(error),
        }
    }
    Err(PyRuntimeError::new_err(
        "unsupported Python configuration: cannot verify interpreter ownership",
    ))
}

fn require_main_interpreter(current: i64, main: i64) -> PyResult<i64> {
    if current != main {
        return Err(PyRuntimeError::new_err(format!(
            "unsupported subinterpreter configuration: main interpreter is {main}, current is {current}"
        )));
    }
    Ok(main)
}

/// Queries the current Python interpreter ID (honoring `FSYM_SIMULATE_SUBINTERPRETER_ID` for adversarial tests).
fn current_interpreter_id(actual: i64) -> PyResult<i64> {
    if let Ok(sim) = std::env::var("FSYM_SIMULATE_SUBINTERPRETER_ID")
        && !sim.is_empty()
    {
        return sim.parse::<i64>().map_err(|e| {
            PyValueError::new_err(format!("invalid FSYM_SIMULATE_SUBINTERPRETER_ID: {e}"))
        });
    }
    Ok(actual)
}

/// Queries the current Python thread identifier via `threading.get_ident()`.
fn current_thread_ident(py: Python<'_>) -> PyResult<u64> {
    let threading = py.import("threading")?;
    threading.call_method0("get_ident")?.extract::<u64>()
}

/// Initializes the primary interpreter ID at module import time.
pub fn init_primary_interpreter(py: Python<'_>) -> PyResult<i64> {
    let (current, main) = raw_interpreter_ids(py)?;
    let id = require_main_interpreter(current, main)?;
    let _ = PRIMARY_INTERPRETER_ID.set(id);
    Ok(*PRIMARY_INTERPRETER_ID.get().unwrap_or(&id))
}

/// Verifies that the active Python runtime is a supported GIL-enabled primary interpreter.
///
/// Explicitly refuses:
/// - Free-threaded CPython configurations (`sys._is_gil_enabled() == False` or
///   `FSYM_FORCE_FREE_THREADED_REFUSAL=1`).
/// - Subinterpreters whose interpreter ID differs from the primary interpreter.
#[pyfunction]
pub fn check_interpreter_support(py: Python<'_>) -> PyResult<i64> {
    if std::env::var("FSYM_FORCE_FREE_THREADED_REFUSAL").as_deref() == Ok("1") {
        return Err(PyRuntimeError::new_err(
            "unsupported free-threaded Python configuration: GIL must be enabled for certified bridge execution",
        ));
    }
    let sys = py.import("sys")?;
    if let Ok(is_gil_enabled_fn) = sys.getattr("_is_gil_enabled")
        && is_gil_enabled_fn.is_callable()
    {
        let gil_enabled: bool = is_gil_enabled_fn.call0()?.extract()?;
        if !gil_enabled {
            return Err(PyRuntimeError::new_err(
                "unsupported free-threaded Python configuration: GIL must be enabled for certified bridge execution",
            ));
        }
    }
    let (actual, main) = raw_interpreter_ids(py)?;
    require_main_interpreter(actual, main)?;
    let current = current_interpreter_id(actual)?;
    let primary = *PRIMARY_INTERPRETER_ID.get_or_init(|| main);
    if current != primary {
        return Err(PyRuntimeError::new_err(format!(
            "unsupported subinterpreter configuration: primary interpreter is {primary}, current is {current}"
        )));
    }
    Ok(current)
}

#[derive(Default)]
struct CallbackSessionState {
    current_reentrancy_depth: usize,
    invocations: usize,
    consumed_nested_steps: u64,
    cancelled: bool,
    unknown_effect_tainted: bool,
    executed_effects: BTreeMap<EffectKey, Py<PyAny>>,
    in_flight_effects: BTreeSet<EffectKey>,
    effect_trace: Vec<(String, String, String, String)>,
}

/// Supervised callback session enforcing interpreter ownership, reentrancy bounds,
/// nested request budgets, and exactly-once effect execution.
///
/// Forced cancellation limitation: arbitrary non-cooperative Python callbacks
/// holding the CPython GIL cannot be forcibly preempted mid-bytecode without
/// corrupting interpreter state. When a callback exceeds `max_wall_time_ms` or
/// is cancelled in flight, [`PySupervisedCallbackSession`] immediately marks
/// `unknown_effect_tainted = true` upon return, discards the untrusted result,
/// and refuses any subsequent retry or fallback.
#[pyclass(name = "SupervisedCallbackSession", module = "fsym_python")]
pub struct PySupervisedCallbackSession {
    #[pyo3(get)]
    pub owner_thread_id: u64,
    #[pyo3(get)]
    pub owner_interpreter_id: i64,
    #[pyo3(get)]
    pub max_reentrancy_depth: usize,
    #[pyo3(get)]
    pub max_invocations: usize,
    #[pyo3(get)]
    pub max_nested_steps: u64,
    #[pyo3(get)]
    pub max_input_nodes: usize,
    #[pyo3(get)]
    pub max_output_nodes: usize,
    #[pyo3(get)]
    pub max_wall_time_ms: u64,
    state: Mutex<CallbackSessionState>,
}

#[pymethods]
impl PySupervisedCallbackSession {
    #[new]
    #[pyo3(signature = (
        max_reentrancy_depth=8,
        max_invocations=128,
        max_nested_steps=10_000,
        max_input_nodes=2048,
        max_output_nodes=2048,
        max_wall_time_ms=5000
    ))]
    pub fn new(
        py: Python<'_>,
        max_reentrancy_depth: usize,
        max_invocations: usize,
        max_nested_steps: u64,
        max_input_nodes: usize,
        max_output_nodes: usize,
        max_wall_time_ms: u64,
    ) -> PyResult<Self> {
        let owner_interpreter_id = check_interpreter_support(py)?;
        let owner_thread_id = current_thread_ident(py)?;
        if max_reentrancy_depth == 0
            || max_invocations == 0
            || max_input_nodes == 0
            || max_output_nodes == 0
        {
            return Err(PyValueError::new_err(
                "SupervisedCallbackSession bounds must all be positive",
            ));
        }
        Ok(Self {
            owner_thread_id,
            owner_interpreter_id,
            max_reentrancy_depth,
            max_invocations,
            max_nested_steps,
            max_input_nodes,
            max_output_nodes,
            max_wall_time_ms,
            state: Mutex::new(CallbackSessionState::default()),
        })
    }

    #[getter]
    pub fn current_reentrancy_depth(&self) -> usize {
        self.state
            .lock()
            .expect("session state lock")
            .current_reentrancy_depth
    }

    #[getter]
    pub fn invocations(&self) -> usize {
        self.state.lock().expect("session state lock").invocations
    }

    #[getter]
    pub fn consumed_nested_steps(&self) -> u64 {
        self.state
            .lock()
            .expect("session state lock")
            .consumed_nested_steps
    }

    #[getter]
    pub fn cancelled(&self) -> bool {
        self.state.lock().expect("session state lock").cancelled
    }

    #[getter]
    pub fn unknown_effect_tainted(&self) -> bool {
        self.state
            .lock()
            .expect("session state lock")
            .unknown_effect_tainted
    }

    /// Verifies that the caller is on the owning interpreter and thread.
    pub fn verify_ownership(&self, py: Python<'_>) -> PyResult<()> {
        let interp = check_interpreter_support(py)?;
        if interp != self.owner_interpreter_id {
            return Err(PyRuntimeError::new_err(format!(
                "interpreter ownership violation: session owned by interpreter {}, accessed from {}",
                self.owner_interpreter_id, interp
            )));
        }
        let thread_id = current_thread_ident(py)?;
        if thread_id != self.owner_thread_id {
            return Err(PyRuntimeError::new_err(format!(
                "interpreter/thread ownership violation: session owned by thread {}, accessed from {}",
                self.owner_thread_id, thread_id
            )));
        }
        Ok(())
    }

    /// Requests cancellation of the callback session.
    ///
    /// If a callback is currently in flight (`current_reentrancy_depth > 0`),
    /// the session is also marked `unknown_effect_tainted = true` so that no
    /// auto-retry or fallback can execute after cancellation.
    pub fn cancel(&self, py: Python<'_>) -> PyResult<()> {
        self.verify_ownership(py)?;
        let mut st = self.state.lock().expect("session state lock");
        st.cancelled = true;
        if st.current_reentrancy_depth > 0 {
            st.unknown_effect_tainted = true;
        }
        Ok(())
    }

    /// Monotonically charges nested native evaluation steps against this session.
    ///
    /// Fallback never resets accounting. Exceeding `max_nested_steps` taints
    /// the session when inside a callback and raises `ValueError`.
    pub fn charge_nested_steps(&self, py: Python<'_>, steps: u64) -> PyResult<u64> {
        self.verify_ownership(py)?;
        let mut st = self.state.lock().expect("session state lock");
        if st.unknown_effect_tainted || st.cancelled {
            return Err(PyRuntimeError::new_err(
                "cancelled or tainted callback session cannot perform nested native work",
            ));
        }
        let consumed = st.consumed_nested_steps.checked_add(steps);
        st.consumed_nested_steps = consumed.unwrap_or(u64::MAX);
        if consumed.is_none() || st.consumed_nested_steps > self.max_nested_steps {
            if st.current_reentrancy_depth > 0 {
                st.unknown_effect_tainted = true;
            }
            return Err(PyValueError::new_err(format!(
                "nested native request budget exhausted: consumed {} > max {}",
                st.consumed_nested_steps, self.max_nested_steps
            )));
        }
        Ok(st.consumed_nested_steps)
    }

    /// Returns the ordered trace of `(hook_kind, target_key, args_key, outcome)` records.
    pub fn trace(&self) -> Vec<(String, String, String, String)> {
        self.state
            .lock()
            .expect("session state lock")
            .effect_trace
            .clone()
    }

    /// Returns the number of distinct executed effects cached in this session.
    pub fn cached_effect_count(&self) -> usize {
        self.state
            .lock()
            .expect("session state lock")
            .executed_effects
            .len()
    }

    /// Invokes `callback(*cb_args)` at most once for `(hook_kind, target_key, args_key)`.
    ///
    /// - If the same `(hook_kind, target_key, args_key)` already succeeded in this
    ///   session, returns the cached result without re-executing `callback`.
    /// - If `unknown_effect_tainted` is set (from a prior exception, timeout,
    ///   cancellation-in-flight, or output-bound overflow), refuses immediately
    ///   without executing `callback`.
    #[pyo3(signature = (hook_kind, target_key, args_key, input_nodes, callback, cb_args, node_counter=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn invoke_once(
        &self,
        py: Python<'_>,
        hook_kind: &str,
        target_key: &str,
        args_key: &str,
        input_nodes: usize,
        callback: &Bound<'_, PyAny>,
        cb_args: &Bound<'_, PyTuple>,
        node_counter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        self.invoke_supervised(
            py,
            hook_kind,
            target_key,
            Some(args_key),
            input_nodes,
            callback,
            cb_args,
            node_counter,
        )
    }

    /// Execute one fresh unknown-effect callback occurrence without caching its result.
    ///
    /// Equal arguments or callable identities never suppress another explicit occurrence.
    /// The same owning session still supplies all invocation, depth, nested-work and wall-time
    /// bounds. Result admission is part of the supervised invocation: validate a numerical
    /// result inside `callback`, or supply a bounded `node_counter`. No result is published
    /// after cancellation, timeout, or a nested failure caught by the callback itself.
    #[pyo3(signature = (hook_kind, target_key, input_nodes, callback, cb_args, node_counter=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn invoke_fresh(
        &self,
        py: Python<'_>,
        hook_kind: &str,
        target_key: &str,
        input_nodes: usize,
        callback: &Bound<'_, PyAny>,
        cb_args: &Bound<'_, PyTuple>,
        node_counter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        self.invoke_supervised(
            py,
            hook_kind,
            target_key,
            None,
            input_nodes,
            callback,
            cb_args,
            node_counter,
        )
    }
}

impl PySupervisedCallbackSession {
    #[allow(clippy::too_many_arguments)]
    fn invoke_supervised(
        &self,
        py: Python<'_>,
        hook_kind: &str,
        target_key: &str,
        args_key: Option<&str>,
        input_nodes: usize,
        callback: &Bound<'_, PyAny>,
        cb_args: &Bound<'_, PyTuple>,
        node_counter: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        self.verify_ownership(py)?;
        // Fresh calls reserve space for a diagnostic sequence number, not a semantic ID.
        // Neither path copies unbounded user keys or concatenates ambiguous cache keys.
        let key_bytes = hook_kind
            .len()
            .checked_add(target_key.len())
            .and_then(|bytes| bytes.checked_add(args_key.map_or(32, str::len)));
        if key_bytes.is_none_or(|bytes| bytes > MAX_CALLBACK_KEY_BYTES) {
            let mut st = self.state.lock().expect("session state lock");
            st.unknown_effect_tainted |= st.current_reentrancy_depth != 0;
            return Err(PyValueError::new_err(
                "callback key payload exceeds the 4096-byte bound",
            ));
        }

        let trace_args;
        let effect_id;

        {
            let mut st = self.state.lock().expect("session state lock");
            if st
                .effect_trace
                .len()
                .saturating_add(st.current_reentrancy_depth)
                >= MAX_CALLBACK_TRACE_RECORDS
            {
                st.unknown_effect_tainted |= st.current_reentrancy_depth != 0;
                return Err(PyValueError::new_err(
                    "callback trace exhausted its 4096-record bound",
                ));
            }
            trace_args = match args_key {
                Some(key) => key.to_owned(),
                None => format!("fresh:{}", st.invocations.saturating_add(1)),
            };
            if st.unknown_effect_tainted {
                Self::record_trace(
                    &mut st,
                    hook_kind,
                    target_key,
                    &trace_args,
                    "refused:unknown_effect_tainted",
                );
                return Err(PyRuntimeError::new_err(
                    "cancellation or failure with unknown callback effect outcome cannot auto-retry",
                ));
            }

            if st.cancelled {
                Self::record_trace(
                    &mut st,
                    hook_kind,
                    target_key,
                    &trace_args,
                    "refused:cancelled",
                );
                return Err(PyRuntimeError::new_err(
                    "callback session cancelled before hook invocation",
                ));
            }

            effect_id =
                args_key.map(|key| (hook_kind.to_owned(), target_key.to_owned(), key.to_owned()));
            if let Some(key) = &effect_id {
                if st.in_flight_effects.contains(key) {
                    st.unknown_effect_tainted = true;
                    Self::record_trace(
                        &mut st,
                        hook_kind,
                        target_key,
                        &trace_args,
                        "refused:effect_in_flight",
                    );
                    return Err(PyRuntimeError::new_err(
                        "callback effect is already in flight; unknown effect outcome cannot auto-retry",
                    ));
                }
                if let Some(cached) = st.executed_effects.get(key) {
                    let cloned = cached.clone_ref(py);
                    Self::record_trace(&mut st, hook_kind, target_key, &trace_args, "cached_hit");
                    return Ok(cloned);
                }
            }

            if input_nodes > self.max_input_nodes {
                st.unknown_effect_tainted |= st.current_reentrancy_depth != 0;
                Self::record_trace(
                    &mut st,
                    hook_kind,
                    target_key,
                    &trace_args,
                    "refused:input_bound_exceeded",
                );
                return Err(PyValueError::new_err(format!(
                    "callback input node count {input_nodes} exceeds bound {}",
                    self.max_input_nodes
                )));
            }

            if st.current_reentrancy_depth >= self.max_reentrancy_depth {
                st.unknown_effect_tainted = true;
                Self::record_trace(
                    &mut st,
                    hook_kind,
                    target_key,
                    &trace_args,
                    "refused:reentrancy_depth_exceeded",
                );
                return Err(PyRuntimeError::new_err(format!(
                    "callback reentrancy depth {} exceeds maximum {}",
                    st.current_reentrancy_depth, self.max_reentrancy_depth
                )));
            }

            if st.invocations >= self.max_invocations {
                st.unknown_effect_tainted = true;
                Self::record_trace(
                    &mut st,
                    hook_kind,
                    target_key,
                    &trace_args,
                    "refused:invocation_budget_exceeded",
                );
                return Err(PyValueError::new_err(format!(
                    "callback invocation count {} exceeds budget {}",
                    st.invocations, self.max_invocations
                )));
            }

            st.current_reentrancy_depth += 1;
            st.invocations += 1;
            if let Some(key) = &effect_id {
                st.in_flight_effects.insert(key.clone());
            }
        }

        let start = Instant::now();
        let value = match callback.call1(cb_args) {
            Ok(value) => value,
            Err(err) => {
                self.finish_failed_call(hook_kind, target_key, &trace_args, effect_id.as_ref());
                return Err(err);
            }
        };

        // Retain the returned object until failure has tainted the session and released its
        // mutex. Its finalizer may execute Python, including reentrant callback attempts.
        let mut counted_output = None;
        let admission = (|| {
            self.check_in_flight(start)?;
            if let Some(counter_fn) = node_counter {
                counted_output = Some(counter_fn.call1((&value,))?);
                let output_nodes: usize = counted_output
                    .as_ref()
                    .expect("counter returned a value")
                    .extract()?;
                if output_nodes > self.max_output_nodes {
                    return Err(PyValueError::new_err(format!(
                        "callback output node count {output_nodes} exceeds bound {}",
                        self.max_output_nodes
                    )));
                }
            }
            Ok(())
        })();
        if let Err(err) = admission {
            self.finish_failed_call(hook_kind, target_key, &trace_args, effect_id.as_ref());
            return Err(err);
        }
        // A custom counter result may have an effectful finalizer. Dispose of it while
        // the callback is still supervised, then check cancellation/taint before commit.
        drop(counted_output);
        if let Err(err) = self.check_in_flight(start) {
            self.finish_failed_call(hook_kind, target_key, &trace_args, effect_id.as_ref());
            return Err(err);
        }

        {
            let mut st = self.state.lock().expect("session state lock");
            st.current_reentrancy_depth -= 1;
            if let Some(key) = &effect_id {
                st.in_flight_effects.remove(key);
            }
            if let Some(key) = effect_id {
                st.executed_effects.insert(key, value.clone().unbind());
            }
            Self::record_trace(&mut st, hook_kind, target_key, &trace_args, "ok");
        }
        Ok(value.unbind())
    }

    fn check_in_flight(&self, start: Instant) -> PyResult<()> {
        let elapsed_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);
        let mut st = self.state.lock().expect("session state lock");
        if st.cancelled || elapsed_ms > self.max_wall_time_ms {
            st.unknown_effect_tainted = true;
            st.cancelled = true;
            return Err(PyRuntimeError::new_err(format!(
                "callback exceeded wall-time bound ({elapsed_ms}ms > {}ms) or cancelled in flight; unknown effect outcome cannot auto-retry",
                self.max_wall_time_ms
            )));
        }
        if st.unknown_effect_tainted {
            return Err(PyRuntimeError::new_err(
                "nested callback supervision failed; unknown effect outcome cannot auto-retry",
            ));
        }
        Ok(())
    }

    fn finish_failed_call(
        &self,
        hook_kind: &str,
        target_key: &str,
        args_key: &str,
        effect_id: Option<&EffectKey>,
    ) {
        let mut st = self.state.lock().expect("session state lock");
        st.current_reentrancy_depth -= 1;
        st.unknown_effect_tainted = true;
        if let Some(key) = effect_id {
            st.in_flight_effects.remove(key);
        }
        // Formatting a PyErr may invoke an arbitrary exception.__str__. Never do that
        // while holding the session mutex; retain the original exception for the caller.
        Self::record_trace(
            &mut st,
            hook_kind,
            target_key,
            args_key,
            "error:callback_or_result_admission",
        );
    }

    fn record_trace(
        state: &mut CallbackSessionState,
        hook_kind: &str,
        target_key: &str,
        args_key: &str,
        outcome: &str,
    ) {
        debug_assert!(state.effect_trace.len() < MAX_CALLBACK_TRACE_RECORDS);
        state.effect_trace.push((
            hook_kind.to_owned(),
            target_key.to_owned(),
            args_key.to_owned(),
            outcome.to_owned(),
        ));
    }
}

#[cfg(test)]
#[path = "callbacks/tests.rs"]
mod tests;
