//! Robot-first NDJSON session CLI.
//!
//! Reads one JSON request envelope per stdin line, writes exactly one JSON
//! response per line to stdout, and exits cleanly on EOF. Stdout carries
//! protocol responses only; diagnostics go to stderr. Exit code 0 on a
//! clean session (including mid-stream typed refusals), 2 on usage errors.

#![forbid(unsafe_code)]

use frankensympy::{BudgetLimits, MAX_ENVELOPE_BYTES, Session};
use fsym_budget::Dimension;
use std::io::{BufRead, Write};
use std::process::ExitCode;

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

const USAGE: &str = "frankensympy [--compute-steps N] [--memory-bytes N] [--allocations N] [--verifier-steps N]\n\
Reads one typed JSON request per line and flushes one JSON response per request.\n\
Allowances are cumulative for the entire session; requests cannot reset them.\n\
Defaults: 2000000 ordinary steps, 1073741824 logical memory bytes,\n\
1000000 allocations and 100000000 protected verification steps.\n\
Exact source certification reserves bounded scratch space on the same ledger.\n\
Use --help to display this text.\n";

fn arguments() -> Result<Option<BudgetLimits>, String> {
    let mut limits = BudgetLimits::uniform(0, 100_000_000);
    limits.dimensions[Dimension::ComputeSteps.index()] = 2_000_000;
    limits.dimensions[Dimension::MemoryBytes.index()] = 1024 * 1024 * 1024;
    limits.dimensions[Dimension::AllocationCount.index()] = 1_000_000;
    limits.dimensions[Dimension::DepthLimit.index()] = 256;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--help" || argument == "-h" {
            return Ok(None);
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
    Ok(Some(limits))
}

fn main() -> ExitCode {
    let limits = match arguments() {
        Ok(Some(limits)) => limits,
        Ok(None) => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        Err(error) => {
            eprintln!("frankensympy: {error}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let mut session = Session::with_budget_limits(limits, 256);
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
                    session.handle_envelope(line)
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
