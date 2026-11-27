//! Robot-first NDJSON session CLI.
//!
//! Reads one JSON request envelope per stdin line, writes exactly one JSON
//! response per line to stdout, and exits cleanly on EOF. Stdout carries
//! protocol responses only; diagnostics go to stderr. Exit code 0 on a
//! clean session (including mid-stream typed refusals), 2 on usage errors.

#![forbid(unsafe_code)]

use frankensympy::{BudgetLimits, MAX_ENVELOPE_BYTES, MeterError, Session};
use fsym_budget::Dimension;
use std::io::{BufRead, Write};
use std::process::ExitCode;
use std::time::{Duration, Instant};

/// Drain one frame while retaining at most the payload limit plus CRLF.
/// An oversized frame never consumes bytes belonging to the next request.
fn read_frame(reader: &mut impl BufRead, frame: &mut Vec<u8>) -> std::io::Result<Option<bool>> {
    frame.clear();
    let mut oversized = false;
    loop {
        let chunk = reader.fill_buf()?;
        if chunk.is_empty() {
            return Ok((oversized || !frame.is_empty())
                .then_some(oversized || frame.len() > MAX_ENVELOPE_BYTES));
        }
        let newline = chunk.iter().position(|byte| *byte == b'\n');
        let consumed = newline.map_or(chunk.len(), |position| position + 1);
        let retained = consumed.min((MAX_ENVELOPE_BYTES + 2).saturating_sub(frame.len()));
        frame.extend_from_slice(&chunk[..retained]);
        oversized |= retained < consumed;
        reader.consume(consumed);
        if newline.is_some() {
            if !oversized {
                frame.pop();
                if frame.last() == Some(&b'\r') {
                    frame.pop();
                }
            }
            return Ok(Some(oversized || frame.len() > MAX_ENVELOPE_BYTES));
        }
    }
}

const USAGE: &str = "frankensympy [--compute-steps N] [--memory-bytes N] [--allocations N] [--verifier-steps N] [--checkpoint-dir PATH] [--request-timeout-ms N]\n\
Reads one typed JSON request per line and flushes one JSON response per request.\n\
Compute, memory, allocation and verifier allowances stay cumulative.\n\
Defaults: 2000000 ordinary steps, 1073741824 logical memory bytes,\n\
1000000 allocations and 100000000 protected verification steps.\n\
Exact source certification reserves bounded scratch space on the same ledger.\n\
--checkpoint-dir explicitly enables local candidate checkpoint save/restore.\n\
--request-timeout-ms enables cooperative per-request deadlines for certification\n\
operations; legacy operations explicitly refuse in this mode. Zero stops before\n\
decoding. Input wait and output I/O are outside the deadline. Elapsed request\n\
time is reported separately from the cumulative session resource ledger.\n\
Use --help to display this text.\n";

struct Options {
    limits: BudgetLimits,
    checkpoint_directory: Option<std::path::PathBuf>,
    request_timeout_ms: Option<u64>,
}

fn arguments() -> Result<Option<Options>, String> {
    let mut limits = BudgetLimits::uniform(0, 100_000_000);
    limits.dimensions[Dimension::ComputeSteps.index()] = 2_000_000;
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits.dimensions[Dimension::AllocationCount.index()] = 1_000_000;
    limits.dimensions[Dimension::DepthLimit.index()] = 256;
    let mut checkpoint_directory = None;
    let mut request_timeout_ms = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--help" || argument == "-h" {
            return Ok(None);
        }
        if argument == "--checkpoint-dir" {
            if checkpoint_directory.is_some() {
                return Err("--checkpoint-dir may be supplied only once".into());
            }
            let path = arguments
                .next()
                .filter(|path| !path.is_empty() && !path.starts_with("--"))
                .ok_or("--checkpoint-dir requires a directory path")?;
            checkpoint_directory = Some(std::path::PathBuf::from(path));
            continue;
        }
        if argument == "--request-timeout-ms" {
            if request_timeout_ms.is_some() {
                return Err("--request-timeout-ms may be supplied only once".into());
            }
            let value = arguments
                .next()
                .ok_or("--request-timeout-ms requires a nonnegative integer")?;
            request_timeout_ms = Some(
                value
                    .parse::<u64>()
                    .map_err(|_| "--request-timeout-ms requires a nonnegative integer")?,
            );
            continue;
        }
        let slot = match argument.as_str() {
            "--compute-steps" => &mut limits.dimensions[Dimension::ComputeSteps.index()],
            "--memory-bytes" => &mut limits.dimensions[Dimension::MemoryBytes.index()],
            "--allocations" => &mut limits.dimensions[Dimension::AllocationCount.index()],
            "--verifier-steps" => &mut limits.verifier_pool,
            _ => return Err(format!("unknown option: {argument}")),
        };
        let value = arguments
            .next()
            .ok_or_else(|| format!("{argument} requires a nonnegative integer"))?;
        *slot = value
            .parse::<u64>()
            .map_err(|_| format!("{argument} requires a nonnegative integer"))?;
    }
    Ok(Some(Options {
        limits,
        checkpoint_directory,
        request_timeout_ms,
    }))
}

/// Host-owned timing for one dispatched frame, separate from Session's ledger.
/// The callback decides whether to stop. The final sample only records elapsed
/// time: it must never turn an already published result into a timeout.
struct RequestDeadline {
    allowance_ms: u64,
    elapsed: Duration,
    triggered: bool,
}

impl RequestDeadline {
    fn new(allowance_ms: u64) -> Self {
        Self {
            allowance_ms,
            elapsed: Duration::ZERO,
            triggered: false,
        }
    }

    fn record_elapsed(&mut self, elapsed: Duration) {
        self.elapsed = self.elapsed.max(elapsed);
    }

    fn checkpoint(&mut self, elapsed: Duration) -> Result<(), MeterError> {
        self.record_elapsed(elapsed);
        self.triggered |= self.elapsed >= Duration::from_millis(self.allowance_ms);
        if self.triggered {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }

    fn finish(&mut self, response: String, elapsed: Duration) -> String {
        self.record_elapsed(elapsed);
        // This is bounded JSON produced by Session, never an unparsed request.
        let mut response: serde_json::Value =
            serde_json::from_str(&response).expect("Session returns a JSON response object");
        if self.triggered && response["status"] == "error" && response["code"] == "cancelled" {
            response["code"] = serde_json::json!("timed_out");
            response["error"] =
                serde_json::json!("request exceeded its cooperative execution deadline");
        }
        response["request_timing"] = serde_json::json!({
            "allowance_ms": self.allowance_ms,
            "elapsed_us": u64::try_from(self.elapsed.as_micros()).unwrap_or(u64::MAX),
            "deadline_triggered": self.triggered,
        });
        response.to_string()
    }
}

/// The clock seam drives the real serialized certification path in tests.
/// Production supplies monotonic elapsed time after admitting a complete frame.
fn handle_envelope_with_deadline(
    session: &mut Session,
    line: &str,
    allowance_ms: u64,
    mut elapsed: impl FnMut() -> Duration,
) -> String {
    let mut deadline = RequestDeadline::new(allowance_ms);
    let response = session.handle_envelope_with_checkpoint(line, || deadline.checkpoint(elapsed()));
    deadline.finish(response, elapsed())
}

fn main() -> ExitCode {
    let options = match arguments() {
        Ok(Some(options)) => options,
        Ok(None) => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(error) => {
            eprintln!("frankensympy: {error}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let mut session = Session::with_budget_limits(options.limits, 256);
    if let Some(directory) = options.checkpoint_directory {
        session = match session.with_checkpoint_directory(directory) {
            Ok(session) => session,
            Err(error) => {
                eprintln!("frankensympy: {error}");
                return ExitCode::from(2);
            }
        };
    }
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut input = stdin.lock();
    let mut frame = Vec::with_capacity(MAX_ENVELOPE_BYTES + 2);
    loop {
        match read_frame(&mut input, &mut frame) {
            Ok(None) => break,
            Ok(Some(oversized)) => {
                let response = if oversized {
                    serde_json::json!({
                        "status": "error",
                        "code": "request_too_large",
                        "error": "request exceeds the session envelope limit",
                    })
                    .to_string()
                } else {
                    let line = match std::str::from_utf8(&frame) {
                        Ok(line) => line,
                        Err(error) => {
                            eprintln!("frankensympy: stdin read failed: {error}");
                            out.flush().ok();
                            return ExitCode::from(2);
                        }
                    };
                    match options.request_timeout_ms {
                        Some(allowance_ms) => {
                            let started = Instant::now();
                            handle_envelope_with_deadline(&mut session, line, allowance_ms, || {
                                started.elapsed()
                            })
                        }
                        None => session.handle_envelope(line),
                    }
                };
                if writeln!(out, "{response}").is_err() {
                    // stdout closed by the peer: end the session cleanly.
                    out.flush().ok();
                    return ExitCode::SUCCESS;
                }
                // Stateful clients need this response before they can construct
                // the next verification/export request; do not wait for EOF.
                if out.flush().is_err() {
                    return ExitCode::SUCCESS;
                }
            }
            Err(error) => {
                eprintln!("frankensympy: stdin read failed: {error}");
                out.flush().ok();
                return ExitCode::from(2);
            }
        }
    }
    out.flush().ok();
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use frankensympy::{ResidualSystemId, RootCertificateId};
    use serde_json::{Value, json};

    fn session() -> Session {
        Session::with_budget_limits(BudgetLimits::uniform(1 << 40, 1 << 40), 256)
    }

    fn system() -> ResidualSystemId {
        ResidualSystemId::new("system:deadline").unwrap()
    }

    fn certificate() -> RootCertificateId {
        RootCertificateId::new("certificate:deadline").unwrap()
    }

    fn registration() -> String {
        json!({"id":"request","type":"RegisterResidualSystem","payload":{
            "schema_version":1,"system":"system:deadline",
            "residuals":["x^2-2"],"variables":["x"]
        }})
        .to_string()
    }

    fn certify() -> String {
        json!({"id":"request","type":"CertifyRootBox","payload":{
            "schema_version":1,"system":"system:deadline","certificate":"certificate:deadline",
            "witness":{
                "domain":[{"lower":"7/5","upper":"3/2"}],
                "center":["29/20"],"preconditioner":[["1/3"]],"precision_digits":16
            }
        }})
        .to_string()
    }

    fn response(value: String) -> Value {
        serde_json::from_str(&value).unwrap()
    }

    fn registered() -> Session {
        let mut session = session();
        assert_eq!(
            response(session.handle_envelope(&registration()))["status"],
            "success"
        );
        session
    }

    fn checkpoint_count(session: &mut Session, request: &str) -> usize {
        let mut samples = 0;
        let result = response(handle_envelope_with_deadline(session, request, 1, || {
            samples += 1;
            Duration::ZERO
        }));
        assert_eq!(result["status"], "success", "{result}");
        assert!(samples > 5, "the deadline must reach actual execution");
        // The last sample records elapsed time after Session returns.
        samples - 1
    }

    #[test]
    fn exact_deadline_boundary_latches_and_large_receipts_saturate() {
        let mut deadline = RequestDeadline::new(1);
        assert_eq!(deadline.checkpoint(Duration::from_nanos(999_999)), Ok(()));
        assert_eq!(
            deadline.checkpoint(Duration::from_millis(1)),
            Err(MeterError::Cancelled)
        );
        assert_eq!(
            deadline.checkpoint(Duration::ZERO),
            Err(MeterError::Cancelled)
        );
        let result = response(deadline.finish(
            json!({"status":"error","code":"cancelled","id":17}).to_string(),
            Duration::from_nanos(1_000_999),
        ));
        assert_eq!(result["code"], "timed_out");
        assert_eq!(result["id"], 17);
        assert_eq!(result["request_timing"]["elapsed_us"], 1000);
        assert_eq!(result["request_timing"]["deadline_triggered"], true);

        let mut maximum = RequestDeadline::new(u64::MAX);
        let elapsed = Duration::from_millis(u64::MAX - 1);
        assert_eq!(maximum.checkpoint(elapsed), Ok(()));
        let result =
            response(maximum.finish(json!({"status":"success","result":1}).to_string(), elapsed));
        assert_eq!(result["status"], "success");
        assert_eq!(result["request_timing"]["allowance_ms"], u64::MAX);
        assert_eq!(result["request_timing"]["elapsed_us"], u64::MAX);
        assert_eq!(result["request_timing"]["deadline_triggered"], false);
    }

    #[test]
    fn final_timing_does_not_reclassify_other_terminal_causes() {
        let mut deadline = RequestDeadline::new(1);
        assert_eq!(deadline.checkpoint(Duration::ZERO), Ok(()));
        let cancelled = json!({"status":"error","code":"cancelled","error":"owner cancelled"});
        let result =
            response(deadline.finish(cancelled.clone().to_string(), Duration::from_secs(1)));
        assert_eq!(result["code"], cancelled["code"]);
        assert_eq!(result["error"], cancelled["error"]);
        assert_eq!(result["request_timing"]["deadline_triggered"], false);
        assert_eq!(result["request_timing"]["elapsed_us"], 1_000_000);

        for code in [
            "budget_exhausted",
            "malformed_request",
            "domain_not_certified",
        ] {
            let mut deadline = RequestDeadline::new(0);
            assert_eq!(
                deadline.checkpoint(Duration::ZERO),
                Err(MeterError::Cancelled)
            );
            let original = json!({"status":"error","code":code,"error":"original cause","id":23});
            let result = response(deadline.finish(original.clone().to_string(), Duration::ZERO));
            assert_eq!(result["code"], original["code"]);
            assert_eq!(result["error"], original["error"]);
            assert_eq!(result["id"], 23);
        }
    }

    #[test]
    fn zero_deadline_stops_before_decoding_without_admission_or_publication() {
        for line in [registration(), "{broken".into()] {
            let mut session = session();
            let before = session.budget_snapshot();
            let result = response(handle_envelope_with_deadline(
                &mut session,
                &line,
                0,
                || Duration::ZERO,
            ));
            assert_eq!(result["code"], "timed_out");
            assert!(result.get("id").is_none());
            assert_eq!(result["request_timing"]["elapsed_us"], 0);
            assert_eq!(session.budget_snapshot(), before);
            assert!(session.residual_problem(&system()).is_none());
        }
    }

    #[test]
    fn mid_request_deadlines_keep_registration_atomic_and_charges_cumulative() {
        let request = registration();
        let count = checkpoint_count(&mut session(), &request);
        for target in 1..=count {
            let mut session = session();
            let before = session.budget_snapshot();
            let mut samples = 0;
            let result = response(handle_envelope_with_deadline(
                &mut session,
                &request,
                1,
                || {
                    samples += 1;
                    if samples == target {
                        Duration::from_millis(1)
                    } else {
                        Duration::ZERO
                    }
                },
            ));
            assert_eq!(result["code"], "timed_out", "{result}");
            assert_eq!(result["request_timing"]["elapsed_us"], 1000);
            assert_eq!(
                samples,
                target + 1,
                "only the final elapsed sample follows expiry"
            );
            assert!(session.residual_problem(&system()).is_none());
            if target == 1 {
                assert!(result.get("id").is_none());
            } else {
                assert_eq!(result["id"], "request");
            }
            let stopped = session.budget_snapshot();
            if target > 2 {
                assert!(
                    stopped.dimensions[Dimension::ComputeSteps.index()]
                        < before.dimensions[Dimension::ComputeSteps.index()]
                );
            }
            assert_eq!(
                stopped.dimensions[Dimension::TimeBudget.index()],
                before.dimensions[Dimension::TimeBudget.index()]
            );
            let retried = response(handle_envelope_with_deadline(
                &mut session,
                &request,
                1,
                || Duration::ZERO,
            ));
            assert_eq!(retried["status"], "success", "{retried}");
            assert!(session.residual_problem(&system()).is_some());
            for dimension in Dimension::ALL {
                assert!(
                    session.budget_snapshot().dimensions[dimension.index()]
                        <= stopped.dimensions[dimension.index()]
                );
            }
        }
    }

    #[test]
    fn deadlines_interrupt_protected_verification_and_final_publication() {
        let mut completed = registered();
        let before = completed.budget_snapshot();
        let request = certify();
        let count = checkpoint_count(&mut completed, &request);
        let full_work = before.verifier_pool - completed.budget_snapshot().verifier_pool;
        assert!(full_work > 0);
        let mut interrupted_verifier = false;
        for target in [count / 4, count / 2, count * 3 / 4, count - 1, count] {
            let mut session = registered();
            let before = session.budget_snapshot();
            let mut samples = 0;
            let result = response(handle_envelope_with_deadline(
                &mut session,
                &request,
                1,
                || {
                    samples += 1;
                    if samples == target {
                        Duration::from_millis(1)
                    } else {
                        Duration::ZERO
                    }
                },
            ));
            assert_eq!(result["code"], "timed_out", "{result}");
            if target == 1 {
                assert!(result.get("id").is_none());
            } else {
                assert_eq!(result["id"], "request");
            }
            assert!(session.root_certificate(&certificate()).is_none());
            let stopped = session.budget_snapshot();
            let work = before.verifier_pool - stopped.verifier_pool;
            interrupted_verifier |= work > 0 && work < full_work;
            let retried = response(handle_envelope_with_deadline(
                &mut session,
                &request,
                1,
                || Duration::ZERO,
            ));
            assert_eq!(retried["status"], "success", "{retried}");
            assert!(session.root_certificate(&certificate()).is_some());
            assert!(session.budget_snapshot().verifier_pool < stopped.verifier_pool);
        }
        assert!(
            interrupted_verifier,
            "the timer must reach real verifier work"
        );
    }

    #[test]
    fn time_observed_after_publication_is_reported_without_rejecting_success() {
        let request = registration();
        let count = checkpoint_count(&mut session(), &request);
        let mut session = session();
        let mut samples = 0;
        let result = response(handle_envelope_with_deadline(
            &mut session,
            &request,
            1,
            || {
                samples += 1;
                if samples > count {
                    Duration::from_millis(5)
                } else {
                    Duration::ZERO
                }
            },
        ));
        assert_eq!(result["status"], "success", "{result}");
        assert_eq!(result["request_timing"]["elapsed_us"], 5000);
        assert_eq!(result["request_timing"]["deadline_triggered"], false);
        assert!(session.residual_problem(&system()).is_some());
    }

    #[test]
    fn request_timing_leaves_all_session_resource_ledgers_unchanged() {
        let mut ordinary = session();
        let mut timed = session();
        let initial_time = timed.budget_snapshot().dimensions[Dimension::TimeBudget.index()];
        for request in [registration(), certify()] {
            let expected = response(ordinary.handle_envelope(&request));
            let mut actual = response(handle_envelope_with_deadline(
                &mut timed,
                &request,
                1,
                || Duration::from_micros(500),
            ));
            assert_eq!(actual["status"], "success", "{actual}");
            assert_eq!(actual["request_timing"]["elapsed_us"], 500);
            actual.as_object_mut().unwrap().remove("request_timing");
            assert_eq!(actual, expected);
            assert_eq!(timed.budget_snapshot(), ordinary.budget_snapshot());
            assert_eq!(timed.context_digest(), ordinary.context_digest());
        }
        assert_eq!(
            timed.budget_snapshot().dimensions[Dimension::TimeBudget.index()],
            initial_time
        );
    }
}
