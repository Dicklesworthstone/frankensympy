//! Provisional bounded NDJSON facade for the pre-WS14 runtime slice.
//!
//! This is not the versioned agent-native protocol described by the WS14
//! contract: it has no stable request IDs, universe manifest, event stream, or
//! semantic-patch transport. Operations that require those facilities are
//! refused rather than reported as successful.

#![forbid(unsafe_code)]

use crate::workspace::SemanticWorkspace;
use fsym_calculus::diff;
use fsym_core::{Expr, Symbol};
use serde::{Deserialize, Serialize};
use std::fmt::{self, Write as _};
use std::io;

/// Maximum encoded size of one provisional request envelope.
pub const MAX_AGENT_NDJSON_REQUEST_BYTES: usize = 256 * 1024;
/// Maximum source size accepted by a string-expression operation.
pub const MAX_AGENT_EXPRESSION_BYTES: usize = 64 * 1024;
/// Maximum encoded size of one provisional response envelope.
pub const MAX_AGENT_NDJSON_RESPONSE_BYTES: usize = 1024 * 1024;
const MAX_AGENT_NAME_BYTES: usize = 1024;

const GENERAL_EXPRESSION_LIMITS: ExpressionLimits = ExpressionLimits {
    max_nodes: 8_192,
    max_depth: 256,
    max_fanout: 4_096,
    max_payload_bytes: MAX_AGENT_EXPRESSION_BYTES,
};
const EVALUATED_EXPRESSION_LIMITS: ExpressionLimits = ExpressionLimits {
    max_nodes: 100_000,
    max_depth: 512,
    max_fanout: 4_096,
    max_payload_bytes: MAX_AGENT_NDJSON_RESPONSE_BYTES,
};
const DIFFERENTIATION_LIMITS: ExpressionLimits = ExpressionLimits {
    max_nodes: 512,
    max_depth: 128,
    max_fanout: 64,
    max_payload_bytes: MAX_AGENT_EXPRESSION_BYTES,
};

const OUTPUT_LIMIT_RESPONSE: &str = r#"{"status":"Error","data":{"code":"output_limit_exceeded","error":"response exceeded protocol output limit"}}"#;
const SERIALIZATION_FAILURE_RESPONSE: &str = r#"{"status":"Error","data":{"code":"internal_serialization","error":"response serialization failed"}}"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ExpressionLimits {
    max_nodes: usize,
    max_depth: usize,
    max_fanout: usize,
    max_payload_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExpressionValidationError {
    StructuralLimit,
    PayloadLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputFailure {
    SizeLimit,
    Allocation,
}

struct BoundedOutputWriter {
    bytes: Vec<u8>,
    failure: Option<OutputFailure>,
}

impl BoundedOutputWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            failure: None,
        }
    }
}

impl io::Write for BoundedOutputWriter {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let Some(new_len) = self.bytes.len().checked_add(buffer.len()) else {
            self.failure = Some(OutputFailure::SizeLimit);
            return Err(io::Error::other("protocol response size limit exceeded"));
        };
        if new_len > MAX_AGENT_NDJSON_RESPONSE_BYTES {
            self.failure = Some(OutputFailure::SizeLimit);
            return Err(io::Error::other("protocol response size limit exceeded"));
        }
        if self.bytes.try_reserve(buffer.len()).is_err() {
            self.failure = Some(OutputFailure::Allocation);
            return Err(io::Error::other("protocol response allocation failed"));
        }
        self.bytes.extend_from_slice(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct BoundedDisplayWriter {
    rendered: String,
    max_bytes: usize,
}

impl BoundedDisplayWriter {
    fn new() -> Self {
        Self {
            rendered: String::new(),
            max_bytes: MAX_AGENT_NDJSON_RESPONSE_BYTES,
        }
    }

    #[cfg(test)]
    fn with_limit(max_bytes: usize) -> Self {
        Self {
            rendered: String::new(),
            max_bytes,
        }
    }
}

impl fmt::Write for BoundedDisplayWriter {
    fn write_str(&mut self, value: &str) -> fmt::Result {
        let new_len = self
            .rendered
            .len()
            .checked_add(value.len())
            .ok_or(fmt::Error)?;
        if new_len > self.max_bytes {
            return Err(fmt::Error);
        }
        self.rendered
            .try_reserve(value.len())
            .map_err(|_| fmt::Error)?;
        self.rendered.push_str(value);
        Ok(())
    }
}

/// Stable error categories for the provisional wire response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProtocolErrorCode {
    MalformedRequest,
    RequestTooLarge,
    InvalidName,
    ParseFailed,
    ResourceLimitExceeded,
    UnsupportedOperation,
    OutputLimitExceeded,
    InternalSerialization,
}

/// Request packet accepted by the provisional facade.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "payload")]
#[serde(deny_unknown_fields)]
pub enum AgentRequest {
    Bind { symbol: String, expr: String },
    Eval { expr: String },
    Simplify { expr: String },
    Diff { expr: String, var: String },
    Fork { branch_name: String },
}

/// Response packet emitted by the provisional facade.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "data")]
#[serde(deny_unknown_fields)]
pub enum AgentResponse {
    Success {
        result: String,
    },
    Error {
        code: ProtocolErrorCode,
        error: String,
    },
}

fn error_response(code: ProtocolErrorCode, error: &'static str) -> AgentResponse {
    AgentResponse::Error {
        code,
        error: error.to_owned(),
    }
}

fn encode_response(response: &AgentResponse) -> String {
    let mut writer = BoundedOutputWriter::new();
    if serde_json::to_writer(&mut writer, response).is_err() {
        return match writer.failure {
            Some(OutputFailure::SizeLimit) => OUTPUT_LIMIT_RESPONSE.to_owned(),
            Some(OutputFailure::Allocation) | None => SERIALIZATION_FAILURE_RESPONSE.to_owned(),
        };
    }
    String::from_utf8(writer.bytes).unwrap_or_else(|_| SERIALIZATION_FAILURE_RESPONSE.to_owned())
}

fn valid_symbol_name(name: &str) -> bool {
    if name.is_empty() || name.len() > MAX_AGENT_NAME_BYTES {
        return false;
    }
    matches!(
        fsym_core::parse(name),
        Ok(Expr::Sym(symbol)) if symbol.name == name
    )
}

fn node_payload_bytes(node: &Expr) -> Result<usize, ExpressionValidationError> {
    let bits_to_bytes = |bits: u64| {
        usize::try_from(bits.div_ceil(8)).map_err(|_| ExpressionValidationError::PayloadLimit)
    };
    match node {
        Expr::Sym(symbol) => Ok(symbol.name.len()),
        Expr::Integer(integer) => bits_to_bytes(integer.bits()),
        Expr::Rational(rational) => bits_to_bytes(rational.numer().bits())?
            .checked_add(bits_to_bytes(rational.denom().bits())?)
            .ok_or(ExpressionValidationError::PayloadLimit),
        Expr::Function(name, _) => Ok(name.len()),
        Expr::Const(_) | Expr::Add(_) | Expr::Mul(_) | Expr::Pow(_, _) => Ok(0),
    }
}

fn validate_expression_shape<'a>(
    expression: &'a Expr,
    workspace: Option<&'a SemanticWorkspace>,
    limits: ExpressionLimits,
) -> Result<(), ExpressionValidationError> {
    let mut stack = Vec::new();
    stack
        .try_reserve(1)
        .map_err(|_| ExpressionValidationError::StructuralLimit)?;
    stack.push((expression, 0_usize, workspace.is_some()));
    let mut nodes = 0_usize;
    let mut payload_bytes = 0_usize;

    while let Some((node, depth, substitute_symbols)) = stack.pop() {
        if substitute_symbols
            && let Expr::Sym(symbol) = node
            && let Some(binding) = workspace.and_then(|active| active.bindings.get(symbol))
        {
            stack
                .try_reserve(1)
                .map_err(|_| ExpressionValidationError::StructuralLimit)?;
            stack.push((binding, depth, false));
            continue;
        }

        nodes = nodes
            .checked_add(1)
            .ok_or(ExpressionValidationError::StructuralLimit)?;
        if nodes > limits.max_nodes || depth > limits.max_depth {
            return Err(ExpressionValidationError::StructuralLimit);
        }
        payload_bytes = payload_bytes
            .checked_add(node_payload_bytes(node)?)
            .ok_or(ExpressionValidationError::PayloadLimit)?;
        if payload_bytes > limits.max_payload_bytes {
            return Err(ExpressionValidationError::PayloadLimit);
        }

        let children: &[Expr] = match node {
            Expr::Add(children) | Expr::Mul(children) | Expr::Function(_, children) => children,
            Expr::Pow(base, exponent) => {
                let child_depth = depth
                    .checked_add(1)
                    .ok_or(ExpressionValidationError::StructuralLimit)?;
                if child_depth > limits.max_depth {
                    return Err(ExpressionValidationError::StructuralLimit);
                }
                stack
                    .try_reserve(2)
                    .map_err(|_| ExpressionValidationError::StructuralLimit)?;
                stack.push((exponent, child_depth, substitute_symbols));
                stack.push((base, child_depth, substitute_symbols));
                continue;
            }
            Expr::Sym(_) | Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => continue,
        };

        if children.len() > limits.max_fanout {
            return Err(ExpressionValidationError::StructuralLimit);
        }
        let child_depth = depth
            .checked_add(1)
            .ok_or(ExpressionValidationError::StructuralLimit)?;
        if !children.is_empty() && child_depth > limits.max_depth {
            return Err(ExpressionValidationError::StructuralLimit);
        }
        stack
            .try_reserve(children.len())
            .map_err(|_| ExpressionValidationError::StructuralLimit)?;
        for child in children.iter().rev() {
            stack.push((child, child_depth, substitute_symbols));
        }
    }
    Ok(())
}

fn parse_expression(source: &str, limits: ExpressionLimits) -> Result<Expr, AgentResponse> {
    if source.len() > MAX_AGENT_EXPRESSION_BYTES {
        return Err(error_response(
            ProtocolErrorCode::ResourceLimitExceeded,
            "expression exceeds protocol source limit",
        ));
    }
    let expression = fsym_core::parse(source).map_err(|_| {
        error_response(
            ProtocolErrorCode::ParseFailed,
            "expression could not be parsed",
        )
    })?;
    validate_expression_shape(&expression, None, limits).map_err(|_| {
        error_response(
            ProtocolErrorCode::ResourceLimitExceeded,
            "expression exceeds protocol structural limits",
        )
    })?;
    Ok(expression)
}

fn render_expression(expression: &Expr) -> Result<String, AgentResponse> {
    let mut writer = BoundedDisplayWriter::new();
    write!(&mut writer, "{expression}").map_err(|_| {
        error_response(
            ProtocolErrorCode::OutputLimitExceeded,
            "expression view exceeds protocol output limit",
        )
    })?;
    Ok(writer.rendered)
}

fn render_binding(symbol: &str, expression: &Expr) -> Result<String, AgentResponse> {
    let mut writer = BoundedDisplayWriter::new();
    write!(&mut writer, "Bound {symbol} = {expression}").map_err(|_| {
        error_response(
            ProtocolErrorCode::OutputLimitExceeded,
            "expression view exceeds protocol output limit",
        )
    })?;
    Ok(writer.rendered)
}

/// Dispatches a single NDJSON line against the active semantic workspace.
pub fn handle_agent_ndjson(line: &str, workspace: &mut SemanticWorkspace) -> String {
    if line.len() > MAX_AGENT_NDJSON_REQUEST_BYTES {
        return encode_response(&error_response(
            ProtocolErrorCode::RequestTooLarge,
            "request exceeds protocol envelope limit",
        ));
    }
    let req: AgentRequest = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(_) => {
            return encode_response(&error_response(
                ProtocolErrorCode::MalformedRequest,
                "request is not a recognized provisional envelope",
            ));
        }
    };

    let resp = match req {
        AgentRequest::Bind { symbol, expr } => {
            if !valid_symbol_name(&symbol) {
                error_response(
                    ProtocolErrorCode::InvalidName,
                    "symbol name is not a bounded parser identifier",
                )
            } else {
                match parse_expression(&expr, GENERAL_EXPRESSION_LIMITS) {
                    Ok(expression) => match render_binding(&symbol, &expression) {
                        Ok(result) => {
                            workspace.bind(Symbol::new(symbol), expression);
                            AgentResponse::Success { result }
                        }
                        Err(error) => error,
                    },
                    Err(error) => error,
                }
            }
        }
        AgentRequest::Eval { expr } | AgentRequest::Simplify { expr } => {
            match parse_expression(&expr, GENERAL_EXPRESSION_LIMITS) {
                Ok(expression) => {
                    match validate_expression_shape(
                        &expression,
                        Some(workspace),
                        EVALUATED_EXPRESSION_LIMITS,
                    ) {
                        Err(ExpressionValidationError::StructuralLimit) => error_response(
                            ProtocolErrorCode::ResourceLimitExceeded,
                            "workspace substitution exceeds protocol structural limits",
                        ),
                        Err(ExpressionValidationError::PayloadLimit) => error_response(
                            ProtocolErrorCode::OutputLimitExceeded,
                            "workspace substitution payload exceeds protocol output limit",
                        ),
                        Ok(()) => {
                            let result = workspace.eval(&expression);
                            match render_expression(&result) {
                                Ok(result) => AgentResponse::Success { result },
                                Err(error) => error,
                            }
                        }
                    }
                }
                Err(error) => error,
            }
        }
        AgentRequest::Diff { expr, var } => {
            if !valid_symbol_name(&var) {
                error_response(
                    ProtocolErrorCode::InvalidName,
                    "differentiation variable is not a bounded parser identifier",
                )
            } else {
                match parse_expression(&expr, DIFFERENTIATION_LIMITS) {
                    Ok(expression) => {
                        let result = diff(&expression, &Symbol::new(var));
                        match render_expression(&result) {
                            Ok(result) => AgentResponse::Success { result },
                            Err(error) => error,
                        }
                    }
                    Err(error) => error,
                }
            }
        }
        AgentRequest::Fork { branch_name: _ } => error_response(
            ProtocolErrorCode::UnsupportedOperation,
            "fork requires the planned versioned workspace-reference protocol",
        ),
    };

    encode_response(&resp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_response(wire: &str) -> AgentResponse {
        serde_json::from_str(wire).expect("protocol response must be valid JSON")
    }

    #[test]
    fn oversized_and_unknown_requests_fail_before_mutation() {
        let mut workspace = SemanticWorkspace::new("test");
        let oversized = " ".repeat(MAX_AGENT_NDJSON_REQUEST_BYTES + 1);
        assert!(matches!(
            decode_response(&handle_agent_ndjson(&oversized, &mut workspace)),
            AgentResponse::Error {
                code: ProtocolErrorCode::RequestTooLarge,
                ..
            }
        ));
        assert!(workspace.bindings.is_empty());

        let duplicate_field = r#"{"type":"Eval","payload":{"expr":"1","expr":"2"}}"#;
        assert!(matches!(
            decode_response(&handle_agent_ndjson(duplicate_field, &mut workspace)),
            AgentResponse::Error {
                code: ProtocolErrorCode::MalformedRequest,
                ..
            }
        ));
        assert!(workspace.bindings.is_empty());

        let unknown_field = r#"{"type":"Bind","payload":{"symbol":"x","expr":"1","extra":true}}"#;
        assert!(matches!(
            decode_response(&handle_agent_ndjson(unknown_field, &mut workspace)),
            AgentResponse::Error {
                code: ProtocolErrorCode::MalformedRequest,
                ..
            }
        ));
        assert!(workspace.bindings.is_empty());
    }

    #[test]
    fn refused_binding_does_not_publish_partial_state() {
        let mut workspace = SemanticWorkspace::new("test");
        let oversized_expression = "x".repeat(MAX_AGENT_EXPRESSION_BYTES + 1);
        let request = serde_json::to_string(&AgentRequest::Bind {
            symbol: "x".to_owned(),
            expr: oversized_expression,
        })
        .unwrap();
        assert!(matches!(
            decode_response(&handle_agent_ndjson(&request, &mut workspace)),
            AgentResponse::Error {
                code: ProtocolErrorCode::ResourceLimitExceeded,
                ..
            }
        ));
        assert!(workspace.bindings.is_empty());
    }

    #[test]
    fn parser_constants_cannot_be_published_as_unreachable_bindings() {
        for reserved in ["pi", "Pi", "E", "I", "oo", "zoo", "nan", "NaN"] {
            let mut workspace = SemanticWorkspace::new("test");
            let request = serde_json::to_string(&AgentRequest::Bind {
                symbol: reserved.to_owned(),
                expr: "1".to_owned(),
            })
            .unwrap();
            assert!(matches!(
                decode_response(&handle_agent_ndjson(&request, &mut workspace)),
                AgentResponse::Error {
                    code: ProtocolErrorCode::InvalidName,
                    ..
                }
            ));
            assert!(workspace.bindings.is_empty());
        }

        let mut workspace = SemanticWorkspace::new("test");
        let request = serde_json::to_string(&AgentRequest::Diff {
            expr: "x".to_owned(),
            var: "pi".to_owned(),
        })
        .unwrap();
        assert!(matches!(
            decode_response(&handle_agent_ndjson(&request, &mut workspace)),
            AgentResponse::Error {
                code: ProtocolErrorCode::InvalidName,
                ..
            }
        ));
    }

    #[test]
    fn discarded_fork_is_replaced_by_typed_refusal() {
        let mut workspace = SemanticWorkspace::new("test");
        let request = serde_json::to_string(&AgentRequest::Fork {
            branch_name: "child".to_owned(),
        })
        .unwrap();
        assert!(matches!(
            decode_response(&handle_agent_ndjson(&request, &mut workspace)),
            AgentResponse::Error {
                code: ProtocolErrorCode::UnsupportedOperation,
                ..
            }
        ));
    }

    #[test]
    fn expression_views_are_bounded_even_for_preloaded_workspace_state() {
        let mut workspace = SemanticWorkspace::new("test");
        workspace.bind(
            Symbol::new("x"),
            Expr::symbol("y".repeat(MAX_AGENT_NDJSON_RESPONSE_BYTES + 1)),
        );
        assert_eq!(
            validate_expression_shape(
                &Expr::symbol("x"),
                Some(&workspace),
                EVALUATED_EXPRESSION_LIMITS,
            ),
            Err(ExpressionValidationError::PayloadLimit)
        );
        let request = r#"{"type":"Eval","payload":{"expr":"x"}}"#;
        assert!(matches!(
            decode_response(&handle_agent_ndjson(request, &mut workspace)),
            AgentResponse::Error {
                code: ProtocolErrorCode::OutputLimitExceeded,
                ..
            }
        ));
    }

    #[test]
    fn response_encoder_limit_fallback_is_typed_valid_json() {
        let oversized = AgentResponse::Success {
            result: "x".repeat(MAX_AGENT_NDJSON_RESPONSE_BYTES + 1),
        };
        assert!(matches!(
            decode_response(&encode_response(&oversized)),
            AgentResponse::Error {
                code: ProtocolErrorCode::OutputLimitExceeded,
                ..
            }
        ));
    }

    #[test]
    fn structural_renderer_matches_display_and_stops_inside_large_children() {
        for (expression, expected) in [
            (fsym_core::parse("x").unwrap(), "x"),
            (
                fsym_core::parse("12345678901234567890").unwrap(),
                "12345678901234567890",
            ),
            (fsym_core::parse("1/3").unwrap(), "1/3"),
            (fsym_core::parse("pi").unwrap(), "pi"),
            (fsym_core::parse("x + 1").unwrap(), "(x + 1)"),
            (fsym_core::parse("2*x*y").unwrap(), "2*x*y"),
            (fsym_core::parse("(x + 1)^2").unwrap(), "((x + 1)**2)"),
            (fsym_core::parse("f(x, y + 1)").unwrap(), "f(x, (y + 1))"),
            (fsym_core::parse("f()").unwrap(), "f()"),
            (Expr::Add(Vec::new()), "()"),
            (Expr::Mul(Vec::new()), ""),
        ] {
            assert_eq!(render_expression(&expression).unwrap(), expected);
        }

        let expression = Expr::Add(vec![Expr::symbol("x".repeat(128))]);
        let mut writer = BoundedDisplayWriter::with_limit(8);
        assert_eq!(write!(&mut writer, "{expression}"), Err(fmt::Error));
        assert_eq!(writer.rendered, "(");
    }
}
