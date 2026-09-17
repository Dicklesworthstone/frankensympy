//! Robot-first NDJSON session CLI.
//!
//! Reads one JSON request envelope per stdin line, writes exactly one JSON
//! response per line to stdout, and exits cleanly on EOF. Stdout carries
//! protocol responses only; diagnostics go to stderr. Exit code 0 on a
//! clean session (including mid-stream typed refusals), 2 on usage errors.

#![forbid(unsafe_code)]

use frankensympy::{MAX_ENVELOPE_BYTES, Session, SessionBudgets};
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

fn main() -> ExitCode {
    let budgets = SessionBudgets::default();
    let mut session = Session::new(budgets);
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
