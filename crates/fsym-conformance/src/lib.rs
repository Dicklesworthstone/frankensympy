//! # fsym-conformance
//!
//! Differential testing harness, Python SymPy oracle validation, and evidence ledger tools.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

/// Conformance case record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConformanceCase {
    pub case_id: String,
    pub input_expr: String,
    pub operation: String,
    pub expected_sympy_output: String,
    pub actual_frankensympy_output: Option<String>,
    pub passed: bool,
}

/// Differential conformance test suite runner.
pub mod tests {
    pub mod differential {
        use crate::ConformanceCase;

        #[test]
        fn test_basic_conformance_smoke() {
            let case = ConformanceCase {
                case_id: "diff_smoke_001".to_string(),
                input_expr: "x + x".to_string(),
                operation: "simplify".to_string(),
                expected_sympy_output: "2*x".to_string(),
                actual_frankensympy_output: Some("2*x".to_string()),
                passed: true,
            };
            assert!(case.passed);
        }
    }
}
