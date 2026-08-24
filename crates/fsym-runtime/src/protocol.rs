//! NDJSON agent-native protocol for symbolic computation and workspace synchronization (WS14).

#![forbid(unsafe_code)]

use crate::workspace::{MergeReceipt, SemanticWorkspace};
use fsym_calculus::diff;
use fsym_core::Symbol;
use fsym_simplify::simplify;
use serde::{Deserialize, Serialize};

/// Agent-native NDJSON request packet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
pub enum AgentRequest {
    Bind { symbol: String, expr: String },
    Eval { expr: String },
    Simplify { expr: String },
    Diff { expr: String, var: String },
    Fork { branch_name: String },
    Merge { branch: SemanticWorkspace },
}

/// Agent-native NDJSON response packet.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "data")]
pub enum AgentResponse {
    Success { result: String },
    Receipt { receipt: MergeReceipt },
    WorkspaceForked { branch_name: String },
    Error { error: String },
}

/// Dispatches a single NDJSON line against the active semantic workspace.
pub fn handle_agent_ndjson(line: &str, workspace: &mut SemanticWorkspace) -> String {
    let req: AgentRequest = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(e) => {
            let resp = AgentResponse::Error {
                error: format!("Malformed request: {e}"),
            };
            return serde_json::to_string(&resp).unwrap();
        }
    };

    let resp = match req {
        AgentRequest::Bind { symbol, expr } => match fsym_core::parse(&expr) {
            Ok(e) => {
                workspace.bind(Symbol::new(symbol.clone()), e.clone());
                AgentResponse::Success {
                    result: format!("Bound {} = {}", symbol, e),
                }
            }
            Err(e) => AgentResponse::Error {
                error: format!("Parse error: {e}"),
            },
        },
        AgentRequest::Eval { expr } => match fsym_core::parse(&expr) {
            Ok(e) => {
                let res = workspace.eval(&e);
                AgentResponse::Success {
                    result: res.to_string(),
                }
            }
            Err(e) => AgentResponse::Error {
                error: format!("Parse error: {e}"),
            },
        },
        AgentRequest::Simplify { expr } => match fsym_core::parse(&expr) {
            Ok(e) => {
                let res = simplify(&workspace.eval(&e));
                AgentResponse::Success {
                    result: res.to_string(),
                }
            }
            Err(e) => AgentResponse::Error {
                error: format!("Parse error: {e}"),
            },
        },
        AgentRequest::Diff { expr, var } => match fsym_core::parse(&expr) {
            Ok(e) => {
                let res = diff(&e, &Symbol::new(var));
                AgentResponse::Success {
                    result: res.to_string(),
                }
            }
            Err(e) => AgentResponse::Error {
                error: format!("Parse error: {e}"),
            },
        },
        AgentRequest::Fork { branch_name } => {
            let _forked = workspace.fork(branch_name.clone());
            AgentResponse::WorkspaceForked { branch_name }
        }
        AgentRequest::Merge { branch } => match workspace.merge(&branch) {
            Ok(receipt) => AgentResponse::Receipt { receipt },
            Err(e) => AgentResponse::Error {
                error: format!("Merge rejected: {e}"),
            },
        },
    };

    serde_json::to_string(&resp).unwrap()
}
