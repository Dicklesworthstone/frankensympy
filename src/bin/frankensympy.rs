//! Robot-first NDJSON session CLI.
//!
//! Reads one JSON request envelope per stdin line, writes exactly one JSON
//! response per line to stdout, and exits cleanly on EOF. Stdout carries
//! protocol responses only; diagnostics go to stderr. Exit code 0 on a
//! clean session (including mid-stream typed refusals), 2 on usage errors.

#![forbid(unsafe_code)]

use frankensympy::{Session, SessionBudgets};
use std::io::{BufRead, Write};
use std::process::ExitCode;

fn main() -> ExitCode {
    let budgets = SessionBudgets::default();
    let mut session = Session::new(budgets);
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        match line {
            Ok(line) => {
                let response = session.handle_envelope(&line);
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
