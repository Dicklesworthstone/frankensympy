#![forbid(unsafe_code)]

//! Live native-mathematics differential capture CLI for FrankenSymPy.

use fsym_conformance::{
    PROFILE_ID, corpus, default_python, run_conformance, write_evidence_ndjson,
};
use std::path::PathBuf;
use std::process::ExitCode;

fn usage() {
    println!(
        "usage: capture_sympy_oracle [--python PATH] [--output PATH]\n\
         Runs the fixed scalar corpus against live SymPy 1.14.0 and writes NDJSON evidence.\n\
         Exit 0 means every outcome matched the corpus expectation; typed expected refusals\n\
         remain nonconformant capability gaps and are reported separately."
    );
}

fn parse_args() -> Result<Option<(String, PathBuf)>, String> {
    let mut python = default_python();
    let mut output = PathBuf::from("target/conformance/live-oracle.ndjson");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--python" => {
                python = args
                    .next()
                    .ok_or_else(|| "--python requires a path".to_string())?;
            }
            "--output" => {
                output = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--output requires a path".to_string())?,
                );
            }
            "-h" | "--help" => return Ok(None),
            other => return Err(format!("unknown argument `{other}`")),
        }
    }
    Ok(Some((python, output)))
}

fn run() -> Result<bool, String> {
    let Some((python, output)) = parse_args()? else {
        usage();
        return Ok(true);
    };
    let report = run_conformance(&corpus(), &python)?;
    write_evidence_ndjson(&report, &output)
        .map_err(|e| format!("writing `{}` failed: {e}", output.display()))?;

    let conformant = report
        .iter()
        .filter(|case| case.verdict.is_conformant())
        .count();
    let expected_refusals = report
        .iter()
        .filter(|case| matches!(case.verdict, fsym_conformance::Verdict::ExpectedRefusal))
        .count();
    let unexpected: Vec<_> = report
        .iter()
        .filter(|case| !case.verdict.is_expected_outcome())
        .collect();
    println!(
        "profile={PROFILE_ID} cases={} conformant={conformant} expected_refusals={expected_refusals} unexpected={} output={}",
        report.len(),
        unexpected.len(),
        output.display()
    );
    for case in &unexpected {
        eprintln!("{}: {:?}", case.case_id, case.verdict);
    }
    Ok(unexpected.is_empty())
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("capture_sympy_oracle: {error}");
            ExitCode::from(2)
        }
    }
}
