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

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Primary interpreter ID recorded when `fsym_python` initializes.
static PRIMARY_INTERPRETER_ID: OnceLock<i64> = OnceLock::new();

/// Queries the raw CPython interpreter ID (0 for main interpreter).
fn raw_interpreter_id(py: Python<'_>) -> i64 {
    if let Ok(interp_mod) = py.import("_interpreters")
        && let Ok(cur) = interp_mod.call_method0("get_current")
    {
        if let Ok(id) = cur.extract::<i64>() {
            return id;
        }
        if let Ok((id, _)) = cur.extract::<(i64, Py<PyAny>)>() {
            return id;
        }
    }
    0
}

/// Queries the current Python interpreter ID (honoring `FSYM_SIMULATE_SUBINTERPRETER_ID` for adversarial tests).
fn current_interpreter_id(py: Python<'_>) -> PyResult<i64> {
    if let Ok(sim) = std::env::var("FSYM_SIMULATE_SUBINTERPRETER_ID")
        && !sim.is_empty()
    {
        return sim.parse::<i64>().map_err(|e| {
            PyValueError::new_err(format!("invalid FSYM_SIMULATE_SUBINTERPRETER_ID: {e}"))
        });
    }
    Ok(raw_interpreter_id(py))
}

/// Queries the current Python thread identifier via `threading.get_ident()`.
fn current_thread_ident(py: Python<'_>) -> PyResult<u64> {
    let threading = py.import("threading")?;
    threading.call_method0("get_ident")?.extract::<u64>()
}

/// Initializes the primary interpreter ID at module import time.
pub fn init_primary_interpreter(py: Python<'_>) -> PyResult<i64> {
    let id = raw_interpreter_id(py);
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
    let current = current_interpreter_id(py)?;
    let primary = *PRIMARY_INTERPRETER_ID.get_or_init(|| current);
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
    executed_effects: BTreeMap<String, Py<PyAny>>,
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
        st.consumed_nested_steps = st.consumed_nested_steps.saturating_add(steps);
        if st.consumed_nested_steps > self.max_nested_steps {
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
        self.verify_ownership(py)?;
        let effect_id = format!("{hook_kind}::{target_key}::{args_key}");

        {
            let mut st = self.state.lock().expect("session state lock");
            if st.unknown_effect_tainted {
                st.effect_trace.push((
                    hook_kind.to_string(),
                    target_key.to_string(),
                    args_key.to_string(),
                    "refused:unknown_effect_tainted".to_string(),
                ));
                return Err(PyRuntimeError::new_err(
                    "cancellation or failure with unknown callback effect outcome cannot auto-retry",
                ));
            }

            if st.cancelled {
                st.effect_trace.push((
                    hook_kind.to_string(),
                    target_key.to_string(),
                    args_key.to_string(),
                    "refused:cancelled".to_string(),
                ));
                return Err(PyRuntimeError::new_err(
                    "callback session cancelled before hook invocation",
                ));
            }

            if let Some(cached) = st.executed_effects.get(&effect_id) {
                let cloned = cached.clone_ref(py);
                st.effect_trace.push((
                    hook_kind.to_string(),
                    target_key.to_string(),
                    args_key.to_string(),
                    "cached_hit".to_string(),
                ));
                return Ok(cloned);
            }

            if input_nodes > self.max_input_nodes {
                st.effect_trace.push((
                    hook_kind.to_string(),
                    target_key.to_string(),
                    args_key.to_string(),
                    "refused:input_bound_exceeded".to_string(),
                ));
                return Err(PyValueError::new_err(format!(
                    "callback input node count {input_nodes} exceeds bound {}",
                    self.max_input_nodes
                )));
            }

            if st.current_reentrancy_depth >= self.max_reentrancy_depth {
                st.unknown_effect_tainted = true;
                st.effect_trace.push((
                    hook_kind.to_string(),
                    target_key.to_string(),
                    args_key.to_string(),
                    "refused:reentrancy_depth_exceeded".to_string(),
                ));
                return Err(PyRuntimeError::new_err(format!(
                    "callback reentrancy depth {} exceeds maximum {}",
                    st.current_reentrancy_depth, self.max_reentrancy_depth
                )));
            }

            if st.invocations >= self.max_invocations {
                st.unknown_effect_tainted = true;
                st.effect_trace.push((
                    hook_kind.to_string(),
                    target_key.to_string(),
                    args_key.to_string(),
                    "refused:invocation_budget_exceeded".to_string(),
                ));
                return Err(PyValueError::new_err(format!(
                    "callback invocation count {} exceeds budget {}",
                    st.invocations, self.max_invocations
                )));
            }

            st.current_reentrancy_depth += 1;
            st.invocations += 1;
        }

        let start = Instant::now();
        let call_result = callback.call1(cb_args);
        let elapsed_ms = u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX);

        let value = match call_result {
            Ok(v) => {
                let mut st = self.state.lock().expect("session state lock");
                st.current_reentrancy_depth = st.current_reentrancy_depth.saturating_sub(1);
                if st.cancelled || elapsed_ms > self.max_wall_time_ms {
                    st.unknown_effect_tainted = true;
                    st.cancelled = true;
                    st.effect_trace.push((
                        hook_kind.to_string(),
                        target_key.to_string(),
                        args_key.to_string(),
                        format!("refused:timeout_or_cancelled({elapsed_ms}ms)"),
                    ));
                    return Err(PyRuntimeError::new_err(format!(
                        "callback exceeded wall-time bound ({elapsed_ms}ms > {}ms) or cancelled in flight; unknown effect outcome cannot auto-retry",
                        self.max_wall_time_ms
                    )));
                }
                v
            }
            Err(err) => {
                let mut st = self.state.lock().expect("session state lock");
                st.current_reentrancy_depth = st.current_reentrancy_depth.saturating_sub(1);
                st.unknown_effect_tainted = true;
                st.effect_trace.push((
                    hook_kind.to_string(),
                    target_key.to_string(),
                    args_key.to_string(),
                    format!("error:{err}"),
                ));
                return Err(err);
            }
        };

        if let Some(counter_fn) = node_counter {
            let output_nodes: usize = counter_fn.call1((&value,))?.extract()?;
            if output_nodes > self.max_output_nodes {
                let mut st = self.state.lock().expect("session state lock");
                st.unknown_effect_tainted = true;
                st.effect_trace.push((
                    hook_kind.to_string(),
                    target_key.to_string(),
                    args_key.to_string(),
                    format!("refused:output_bound_exceeded({output_nodes})"),
                ));
                return Err(PyValueError::new_err(format!(
                    "callback output node count {output_nodes} exceeds bound {}",
                    self.max_output_nodes
                )));
            }
        }

        let unbound = value.unbind();
        let mut st = self.state.lock().expect("session state lock");
        st.executed_effects.insert(effect_id, unbound.clone_ref(py));
        st.effect_trace.push((
            hook_kind.to_string(),
            target_key.to_string(),
            args_key.to_string(),
            "ok".to_string(),
        ));
        Ok(unbound)
    }
}
