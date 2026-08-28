//! Non-recursive admission checks for expression rendering.

use fsym_core::{BigInt, Constant, Expr};
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrintingLimits {
    pub max_depth: usize,
    pub max_nodes: usize,
    pub max_output_bytes: usize,
    pub max_lines: usize,
    pub max_line_bytes: usize,
}

impl Default for PrintingLimits {
    fn default() -> Self {
        Self {
            max_depth: 256,
            max_nodes: 100_000,
            max_output_bytes: 1_048_576,
            max_lines: 1,
            max_line_bytes: 1_048_576,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum PrintingError {
    #[error("printer depth limit exceeded (maximum {max_depth})")]
    DepthLimitExceeded { max_depth: usize },
    #[error("printer node limit exceeded (maximum {max_nodes})")]
    NodeLimitExceeded { max_nodes: usize },
    #[error("printer output limit exceeded (maximum {max_output_bytes} bytes)")]
    OutputLimitExceeded { max_output_bytes: usize },
    #[error("printer line limit exceeded (maximum {max_lines})")]
    LineLimitExceeded { max_lines: usize },
    #[error("printer line-width limit exceeded (maximum {max_line_bytes} bytes)")]
    LineWidthLimitExceeded { max_line_bytes: usize },
    #[error("printer traversal allocation failed")]
    AllocationFailure,
    #[error("symbol or function name contains a control character")]
    InvalidNameControlCharacter,
    #[error("symbol or function name is not a supported Rust identifier")]
    InvalidRustIdentifier,
    #[error("symbol or function name is not a supported Python identifier")]
    InvalidPythonIdentifier,
    #[error("symbol or function name is not a supported C identifier")]
    InvalidCIdentifier,
    #[error("numeric value exceeds the Rust emitter's supported i64 literal lane")]
    RustNumericValueOutOfRange,
    #[error("constant {0} is not supported by the real-valued Rust emitter")]
    UnsupportedRustConstant(Constant),
    #[error("constant {0} is not supported by the real-valued C emitter")]
    UnsupportedCConstant(Constant),
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum RenderTarget {
    Latex,
    Pretty,
    Rust,
    Python,
    C,
}

fn decimal_digits_upper_bound(value: &BigInt) -> usize {
    let magnitude_digits = if value.bits() == 0 {
        1_u64
    } else {
        // 30103/100000 is a strict upper approximation of log10(2).
        value.bits().saturating_mul(30_103) / 100_000 + 1
    };
    let sign = u64::from(value.is_negative());
    usize::try_from(magnitude_digits.saturating_add(sign)).unwrap_or(usize::MAX)
}

fn latex_name_bytes(name: &str) -> usize {
    name.chars().fold(0_usize, |total, ch| {
        let bytes = match ch {
            '\\' => 12,
            '{' | '}' | '$' | '&' | '#' | '%' | '_' => 2,
            '^' | '~' => 4,
            _ => ch.len_utf8(),
        };
        total.saturating_add(bytes)
    })
}

fn rust_keyword(identifier: &str) -> bool {
    matches!(
        identifier,
        "as" | "async"
            | "await"
            | "break"
            | "const"
            | "continue"
            | "crate"
            | "dyn"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "fn"
            | "for"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "type"
            | "union"
            | "unsafe"
            | "use"
            | "where"
            | "while"
            | "abstract"
            | "become"
            | "box"
            | "do"
            | "final"
            | "macro"
            | "override"
            | "priv"
            | "typeof"
            | "unsized"
            | "virtual"
            | "yield"
            | "try"
    )
}

fn valid_rust_identifier(identifier: &str) -> bool {
    let mut chars = identifier.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
        && identifier != "_"
        && !rust_keyword(identifier)
}

fn python_keyword(identifier: &str) -> bool {
    matches!(
        identifier,
        "False"
            | "None"
            | "True"
            | "and"
            | "as"
            | "assert"
            | "async"
            | "await"
            | "break"
            | "class"
            | "continue"
            | "def"
            | "del"
            | "elif"
            | "else"
            | "except"
            | "finally"
            | "for"
            | "from"
            | "global"
            | "if"
            | "import"
            | "in"
            | "is"
            | "lambda"
            | "nonlocal"
            | "not"
            | "or"
            | "pass"
            | "raise"
            | "return"
            | "try"
            | "while"
            | "with"
            | "yield"
    )
}

fn valid_python_identifier(identifier: &str) -> bool {
    let mut chars = identifier.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
        && !python_keyword(identifier)
}

fn c_keyword(identifier: &str) -> bool {
    matches!(
        identifier,
        "auto"
            | "break"
            | "case"
            | "char"
            | "const"
            | "continue"
            | "default"
            | "do"
            | "double"
            | "else"
            | "enum"
            | "extern"
            | "float"
            | "for"
            | "goto"
            | "if"
            | "inline"
            | "int"
            | "long"
            | "register"
            | "restrict"
            | "return"
            | "short"
            | "signed"
            | "sizeof"
            | "static"
            | "struct"
            | "switch"
            | "typedef"
            | "union"
            | "unsigned"
            | "void"
            | "volatile"
            | "while"
    )
}

fn valid_c_identifier(identifier: &str) -> bool {
    let mut chars = identifier.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
        && !c_keyword(identifier)
}

fn node_output_upper_bound(expr: &Expr, target: RenderTarget) -> Result<usize, PrintingError> {
    let name_bytes = |name: &str| -> Result<usize, PrintingError> {
        if name.chars().any(char::is_control) {
            return Err(PrintingError::InvalidNameControlCharacter);
        }
        match target {
            RenderTarget::Rust if !valid_rust_identifier(name) => {
                return Err(PrintingError::InvalidRustIdentifier);
            }
            RenderTarget::Python if !valid_python_identifier(name) => {
                return Err(PrintingError::InvalidPythonIdentifier);
            }
            RenderTarget::C if !valid_c_identifier(name) => {
                return Err(PrintingError::InvalidCIdentifier);
            }
            _ => {}
        }
        Ok(match target {
            RenderTarget::Latex => latex_name_bytes(name),
            RenderTarget::Pretty | RenderTarget::Rust | RenderTarget::Python | RenderTarget::C => {
                name.len()
            }
        })
    };

    let value = match expr {
        Expr::Sym(symbol) => name_bytes(&symbol.name)?.saturating_add(13),
        Expr::Integer(value) => {
            if matches!(target, RenderTarget::Rust | RenderTarget::C) && value.to_i64().is_none() {
                return Err(PrintingError::RustNumericValueOutOfRange);
            }
            let multiplier = usize::from(matches!(target, RenderTarget::Pretty)) * 2 + 1;
            decimal_digits_upper_bound(value)
                .saturating_mul(multiplier)
                .saturating_add(16)
        }
        Expr::Rational(value) => {
            if matches!(target, RenderTarget::Rust | RenderTarget::C)
                && (value.numer().to_i64().is_none() || value.denom().to_i64().is_none())
            {
                return Err(PrintingError::RustNumericValueOutOfRange);
            }
            decimal_digits_upper_bound(value.numer())
                .saturating_add(decimal_digits_upper_bound(value.denom()))
                .saturating_mul(3)
                .saturating_add(24)
        }
        Expr::Const(constant) => {
            if matches!(target, RenderTarget::Rust)
                && matches!(constant, Constant::I | Constant::ComplexInfinity)
            {
                return Err(PrintingError::UnsupportedRustConstant(*constant));
            }
            if matches!(target, RenderTarget::C)
                && matches!(constant, Constant::I | Constant::ComplexInfinity)
            {
                return Err(PrintingError::UnsupportedCConstant(*constant));
            }
            32
        }
        Expr::Add(items) | Expr::Mul(items) => items.len().saturating_mul(3).saturating_add(16),
        Expr::Pow(..) => 48,
        Expr::Function(name, args) => name_bytes(name)?
            .saturating_add(args.len().saturating_mul(2))
            .saturating_add(32),
    };
    Ok(value)
}

pub(crate) fn validate_render(
    expr: &Expr,
    target: RenderTarget,
    limits: PrintingLimits,
) -> Result<(), PrintingError> {
    if limits.max_lines == 0 {
        return Err(PrintingError::LineLimitExceeded { max_lines: 0 });
    }
    if limits.max_nodes == 0 {
        return Err(PrintingError::NodeLimitExceeded { max_nodes: 0 });
    }

    let mut stack = Vec::new();
    stack
        .try_reserve(1)
        .map_err(|_| PrintingError::AllocationFailure)?;
    stack.push((expr, 0_usize));
    let mut scheduled = 1_usize;
    let mut output_upper_bound = 0_usize;

    while let Some((node, depth)) = stack.pop() {
        if depth > limits.max_depth {
            return Err(PrintingError::DepthLimitExceeded {
                max_depth: limits.max_depth,
            });
        }
        let node_output = node_output_upper_bound(node, target)?;
        output_upper_bound = output_upper_bound.saturating_add(node_output);
        if output_upper_bound > limits.max_output_bytes {
            return Err(PrintingError::OutputLimitExceeded {
                max_output_bytes: limits.max_output_bytes,
            });
        }
        if output_upper_bound > limits.max_line_bytes {
            return Err(PrintingError::LineWidthLimitExceeded {
                max_line_bytes: limits.max_line_bytes,
            });
        }

        let child_count = match node {
            Expr::Add(items) | Expr::Mul(items) | Expr::Function(_, items) => items.len(),
            Expr::Pow(..) => 2,
            Expr::Sym(_) | Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => 0,
        };
        if child_count == 0 {
            continue;
        }
        if depth >= limits.max_depth {
            return Err(PrintingError::DepthLimitExceeded {
                max_depth: limits.max_depth,
            });
        }
        scheduled = scheduled
            .checked_add(child_count)
            .filter(|count| *count <= limits.max_nodes)
            .ok_or(PrintingError::NodeLimitExceeded {
                max_nodes: limits.max_nodes,
            })?;
        stack
            .try_reserve(child_count)
            .map_err(|_| PrintingError::AllocationFailure)?;
        match node {
            Expr::Add(items) | Expr::Mul(items) | Expr::Function(_, items) => {
                stack.extend(items.iter().map(|child| (child, depth + 1)));
            }
            Expr::Pow(base, exponent) => {
                stack.push((base.as_ref(), depth + 1));
                stack.push((exponent.as_ref(), depth + 1));
            }
            Expr::Sym(_) | Expr::Integer(_) | Expr::Rational(_) | Expr::Const(_) => {}
        }
    }

    Ok(())
}
