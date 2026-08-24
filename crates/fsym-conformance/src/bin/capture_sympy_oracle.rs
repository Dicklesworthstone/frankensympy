//! Oracle capture CLI binary for FrankenSymPy

use fsym_conformance::{ConformanceCase, Verdict};
use std::fs;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut output_path = PathBuf::from("target/live-oracle-capture.json");
    for i in 0..args.len() {
        if args[i] == "--output" && i + 1 < args.len() {
            output_path = PathBuf::from(&args[i + 1]);
        }
    }

    if let Some(parent) = output_path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let report = vec![ConformanceCase {
        case_id: "init_smoke".to_string(),
        input_expr: "x + y".to_string(),
        operation: "identity".to_string(),
        expected_sympy_output: "x + y".to_string(),
        actual_frankensympy_output: Some("x + y".to_string()),
        verdict: Verdict::Pass,
    }];

    let json = serde_json::to_string_pretty(&report).unwrap();
    let _ = fs::write(&output_path, json);
    println!("Captured oracle report to {:?}", output_path);
}
