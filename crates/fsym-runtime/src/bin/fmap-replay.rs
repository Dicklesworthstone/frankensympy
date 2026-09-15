//! No-mock FMAP replay CLI: reads a canonical FMAP bundle from disk,
//! replays the workspace transaction in this fresh process, and exits
//! nonzero unless the replayed certificate equals the recorded one and the
//! expected result root agrees.
//!
//! Usage: `fmap-replay <bundle.json>`
//!
//! Output: one JSON object `{"recomputed_root": "...", "expected_root":
//! "...", "match": true|false}` on success; a diagnostic on stderr and a
//! nonzero exit on any refusal.

#![forbid(unsafe_code)]

use fsym_runtime::fmap::{FmapBundle, MAX_FMAP_BYTES, replay_fmap_bundle};
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: fmap-replay <bundle.json>");
        return ExitCode::from(2);
    }
    let bytes = match std::fs::read(&args[1]) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("fmap-replay: cannot read {}: {error}", args[1]);
            return ExitCode::from(2);
        }
    };
    if bytes.len() > MAX_FMAP_BYTES {
        eprintln!("fmap-replay: bundle exceeds the declared size limit of {MAX_FMAP_BYTES} bytes");
        return ExitCode::from(2);
    }
    let bundle = match FmapBundle::from_json(&bytes) {
        Ok(bundle) => bundle,
        Err(error) => {
            eprintln!("fmap-replay: bundle refused: {error}");
            return ExitCode::from(2);
        }
    };
    match replay_fmap_bundle(&bundle) {
        Ok(outcome) => {
            let expected = bundle.replay_metadata.expected_result_root;
            println!(
                "{{\"recomputed_root\":\"{}\",\"expected_root\":\"{}\",\"match\":{}}}",
                hex(&outcome.result_root),
                hex(&expected),
                outcome.result_root == expected,
            );
            if outcome.result_root == expected {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            eprintln!("fmap-replay: replay refused: {error}");
            ExitCode::FAILURE
        }
    }
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
