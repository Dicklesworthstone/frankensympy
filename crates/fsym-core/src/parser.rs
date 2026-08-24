//! Recursive-descent expression parser producing canonical [`Expr`] ASTs.
//!
//! Grammar (standard precedence, `^` right-associative):
//!
//! ```text
//! expr  := term (('+' | '-') term)*
//! term  := unary (('*' | '/') unary)*
//! unary := ('-' | '+') unary | power
//! power := atom ('^' unary)?
//! atom  := number | constant | symbol | func '(' args ')' | '(' expr ')'
//! ```
//!
//! Division of two integer literals folds to an exact rational; decimal
//! literals parse as exact rationals scaled by powers of ten. There are no
//! floating-point leaves anywhere in the numeric tower.

use crate::{Constant, CoreError, Expr, Symbol};
use fsym_bigint::{BigInt, BigRational};
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Int(String),
    Dec(String),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
}

fn tokenize(input: &str) -> Result<Vec<Tok>, CoreError> {
    let mut toks = Vec::new();
    let mut chars = input.chars().peekable();
    while let Some(&c) = chars.peek() {
        match c {
            c if c.is_whitespace() => {
                chars.next();
            }
            '+' => {
                chars.next();
                toks.push(Tok::Plus);
            }
            '-' => {
                chars.next();
                toks.push(Tok::Minus);
            }
            '*' => {
                chars.next();
                toks.push(Tok::Star);
            }
            '/' => {
                chars.next();
                toks.push(Tok::Slash);
            }
            '^' => {
                chars.next();
                toks.push(Tok::Caret);
            }
            '(' => {
                chars.next();
                toks.push(Tok::LParen);
            }
            ')' => {
                chars.next();
                toks.push(Tok::RParen);
            }
            ',' => {
                chars.next();
                toks.push(Tok::Comma);
            }
            '0'..='9' => {
                let mut s = String::new();
                let mut dot_seen = false;
                while let Some(&c) = chars.peek() {
                    if c.is_ascii_digit() || (c == '.' && !dot_seen) {
                        if c == '.' {
                            dot_seen = true;
                        }
                        s.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                if dot_seen {
                    toks.push(Tok::Dec(s));
                } else {
                    toks.push(Tok::Int(s));
                }
            }
            c if c.is_alphabetic() || c == '_' => {
                let mut s = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_alphanumeric() || c == '_' {
                        s.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                toks.push(Tok::Ident(s));
            }
            other => {
                return Err(CoreError::ParseError(format!(
                    "unexpected character `{other}` in `{input}`"
                )));
            }
        }
    }
    Ok(toks)
}

struct Parser<'a> {
    toks: &'a [Tok],
    pos: usize,
    src: &'a str,
}

impl<'a> Parser<'a> {
    fn err(&self, why: &str) -> CoreError {
        CoreError::ParseError(format!("{} in `{}`", why, self.src))
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn bump(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expect(&mut self, tok: &Tok) -> Result<(), CoreError> {
        if self.peek() == Some(tok) {
            self.pos += 1;
            Ok(())
        } else {
            Err(self.err(&format!("expected {tok:?}")))
        }
    }

    fn expr(&mut self) -> Result<Expr, CoreError> {
        let mut left = self.term()?;
        loop {
            match self.peek() {
                Some(Tok::Plus) => {
                    self.bump();
                    let right = self.term()?;
                    left = left + right;
                }
                Some(Tok::Minus) => {
                    self.bump();
                    let right = self.term()?;
                    left = left + neg(right);
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn term(&mut self) -> Result<Expr, CoreError> {
        let mut left = self.unary()?;
        loop {
            match self.peek() {
                Some(Tok::Star) => {
                    self.bump();
                    let right = self.unary()?;
                    left = left * right;
                }
                Some(Tok::Slash) => {
                    self.bump();
                    let right = self.unary()?;
                    left = divide(left, right)?;
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn unary(&mut self) -> Result<Expr, CoreError> {
        match self.peek() {
            Some(Tok::Minus) => {
                self.bump();
                Ok(neg(self.unary()?))
            }
            Some(Tok::Plus) => {
                self.bump();
                self.unary()
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Result<Expr, CoreError> {
        let base = self.atom()?;
        if matches!(self.peek(), Some(Tok::Caret)) {
            self.bump();
            let exp = self.unary()?;
            Ok(Expr::Pow(Arc::new(base), Arc::new(exp)))
        } else {
            Ok(base)
        }
    }

    fn atom(&mut self) -> Result<Expr, CoreError> {
        match self.bump() {
            Some(Tok::Int(s)) => Ok(Expr::Integer(
                BigInt::parse_bytes(s.as_bytes(), 10).expect("digit string"),
            )),
            Some(Tok::Dec(s)) => decimal_expr(&s),
            Some(Tok::Ident(name)) => {
                if matches!(self.peek(), Some(Tok::LParen)) && constant_named(&name).is_none() {
                    self.bump();
                    let mut args = Vec::new();
                    if !matches!(self.peek(), Some(Tok::RParen)) {
                        args.push(self.expr()?);
                        while matches!(self.peek(), Some(Tok::Comma)) {
                            self.bump();
                            args.push(self.expr()?);
                        }
                    }
                    self.expect(&Tok::RParen)?;
                    Ok(Expr::Function(name, args))
                } else {
                    match constant_named(&name) {
                        Some(c) => Ok(Expr::Const(c)),
                        None => Ok(Expr::Sym(Symbol::new(name))),
                    }
                }
            }
            Some(Tok::LParen) => {
                let e = self.expr()?;
                self.expect(&Tok::RParen)?;
                Ok(e)
            }
            other => Err(self.err(&format!("unexpected token {other:?}"))),
        }
    }
}

/// Negation with constant folding so `-oo` becomes `NegativeInfinity`.
fn neg(e: Expr) -> Expr {
    match e {
        Expr::Integer(n) => Expr::Integer(-n),
        Expr::Rational(r) => Expr::Rational(-r),
        Expr::Const(Constant::Infinity) => Expr::Const(Constant::NegativeInfinity),
        other => Expr::Mul(vec![Expr::from_i64(-1), other]),
    }
}

/// Exact division: numeric literals fold to an exact integer or rational; anything else
/// becomes multiplication by the reciprocal.
fn divide(a: Expr, b: Expr) -> Result<Expr, CoreError> {
    match &b {
        Expr::Integer(n) if n.is_zero() => return Err(CoreError::DivisionByZero),
        Expr::Rational(r) if r.numer().is_zero() => return Err(CoreError::DivisionByZero),
        _ => {}
    }
    match (&a, &b) {
        (Expr::Integer(an), Expr::Integer(bn)) => {
            let r = BigRational::new(an.clone(), bn.clone());
            if r.is_integer() {
                Ok(Expr::Integer(r.to_integer()))
            } else {
                Ok(Expr::Rational(r))
            }
        }
        (Expr::Integer(an), Expr::Rational(bn)) => {
            let an_r = BigRational::from_integer(an.clone());
            let r = an_r / bn;
            if r.is_integer() {
                Ok(Expr::Integer(r.to_integer()))
            } else {
                Ok(Expr::Rational(r))
            }
        }
        (Expr::Rational(an), Expr::Integer(bn)) => {
            let bn_r = BigRational::from_integer(bn.clone());
            let r = an / bn_r;
            if r.is_integer() {
                Ok(Expr::Integer(r.to_integer()))
            } else {
                Ok(Expr::Rational(r))
            }
        }
        (Expr::Rational(an), Expr::Rational(bn)) => {
            let r = an / bn;
            if r.is_integer() {
                Ok(Expr::Integer(r.to_integer()))
            } else {
                Ok(Expr::Rational(r))
            }
        }
        _ => Ok(Expr::Mul(vec![
            a,
            Expr::Pow(Arc::new(b), Arc::new(Expr::from_i64(-1))),
        ])),
    }
}

/// Decimal literal as an exact rational: `"3.14"` → `157/50`.
fn decimal_expr(s: &str) -> Result<Expr, CoreError> {
    let (whole, frac) = match s.split_once('.') {
        Some((w, f)) => (w, f),
        None => (s, ""),
    };
    let digits = format!("{whole}{frac}");
    let numerator = BigInt::parse_bytes(digits.as_bytes(), 10)
        .ok_or_else(|| CoreError::ParseError(format!("bad number literal `{s}`")))?;
    let denominator = BigInt::from(10u32).pow(frac.len() as u32);
    Ok(Expr::Rational(BigRational::new(numerator, denominator)))
}

fn constant_named(name: &str) -> Option<Constant> {
    match name {
        "pi" | "Pi" => Some(Constant::Pi),
        "E" => Some(Constant::E),
        "I" => Some(Constant::I),
        "oo" => Some(Constant::Infinity),
        "zoo" => Some(Constant::ComplexInfinity),
        "nan" | "NaN" => Some(Constant::NaN),
        _ => None,
    }
}

/// Parse an expression string into an [`Expr`].
///
/// # Errors
/// [`CoreError::ParseError`] on malformed input; [`CoreError::DivisionByZero`]
/// for literal division by zero.
pub fn parse(input: &str) -> Result<Expr, CoreError> {
    let toks = tokenize(input)?;
    if toks.is_empty() {
        return Err(CoreError::ParseError(format!("empty input `{input}`")));
    }
    let mut p = Parser {
        toks: &toks,
        pos: 0,
        src: input,
    };
    let e = p.expr()?;
    if p.pos != toks.len() {
        return Err(p.err("unexpected trailing input"));
    }
    Ok(e)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precedence_and_associativity() {
        // Pure-integer subexpressions fold eagerly (canonical constant
        // folding via the ops impls), so numeric literals collapse.
        assert_eq!(parse("2+3*4").unwrap(), Expr::from_i64(14));
        assert_eq!(parse("(2+3)*4").unwrap(), Expr::from_i64(20));
        // Integer products fold even inside symbolic sums.
        assert_eq!(
            parse("x+3*4").unwrap(),
            Expr::Add(vec![Expr::symbol("x"), Expr::from_i64(12)])
        );
        // ^ is right-associative: 2^3^2 = 2^(3^2).
        assert_eq!(
            parse("2^3^2").unwrap(),
            Expr::Pow(
                Arc::new(Expr::from_i64(2)),
                Arc::new(Expr::Pow(
                    Arc::new(Expr::from_i64(3)),
                    Arc::new(Expr::from_i64(2))
                ))
            )
        );
    }

    #[test]
    fn test_exact_division_folds_to_rational() {
        assert_eq!(
            parse("1/2").unwrap(),
            Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)))
        );
        assert_eq!(
            parse("0.5/2").unwrap(),
            Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(4)))
        );
        // Symbolic denominators stay structural.
        assert_eq!(
            parse("x/2").unwrap(),
            Expr::Mul(vec![
                Expr::symbol("x"),
                Expr::Pow(Arc::new(Expr::from_i64(2)), Arc::new(Expr::from_i64(-1)))
            ])
        );
        assert_eq!(parse("1/0"), Err(CoreError::DivisionByZero));
        assert_eq!(parse("1/0.0"), Err(CoreError::DivisionByZero));
    }

    #[test]
    fn test_decimal_literals_are_exact() {
        assert_eq!(
            parse("0.5").unwrap(),
            Expr::Rational(BigRational::new(BigInt::from(1), BigInt::from(2)))
        );
        // 3.14 = 157/50 exactly.
        assert_eq!(
            parse("3.14").unwrap(),
            Expr::Rational(BigRational::new(BigInt::from(157), BigInt::from(50)))
        );
    }

    #[test]
    fn test_unary_minus_and_constants() {
        assert_eq!(
            parse("-x").unwrap(),
            Expr::Mul(vec![Expr::from_i64(-1), Expr::symbol("x")])
        );
        assert_eq!(
            parse("-oo").unwrap(),
            Expr::Const(Constant::NegativeInfinity)
        );
        assert_eq!(parse("pi").unwrap(), Expr::Const(Constant::Pi));
        assert_eq!(parse("E").unwrap(), Expr::Const(Constant::E));
        // Lowercase `e` stays a free symbol; only `E` is Euler's number.
        assert_eq!(parse("e").unwrap(), Expr::symbol("e"));
    }

    #[test]
    fn test_function_calls() {
        assert_eq!(
            parse("sin(pi)").unwrap(),
            Expr::Function("sin".to_string(), vec![Expr::Const(Constant::Pi)])
        );
        assert_eq!(
            parse("f()").unwrap(),
            Expr::Function("f".to_string(), vec![])
        );
        assert_eq!(
            parse("f(x, y+1)").unwrap(),
            Expr::Function(
                "f".to_string(),
                vec![Expr::symbol("x"), parse("y+1").unwrap()]
            )
        );
    }

    #[test]
    fn test_parse_errors() {
        assert!(matches!(parse(""), Err(CoreError::ParseError(_))));
        assert!(matches!(parse("2+"), Err(CoreError::ParseError(_))));
        assert!(matches!(parse("(x"), Err(CoreError::ParseError(_))));
        assert!(matches!(parse("2 3"), Err(CoreError::ParseError(_))));
        assert!(matches!(parse("x @ y"), Err(CoreError::ParseError(_))));
    }

    #[test]
    fn test_evalf_through_parser() {
        let approx = |s: &str| parse(s).unwrap().evalf().unwrap();
        assert!((approx("2+3*4") - 14.0).abs() < 1e-12);
        assert!((approx("1/2") - 0.5).abs() < 1e-12);
        assert!(approx("sin(pi)").abs() < 1e-12);
        assert!((approx("pi") - std::f64::consts::PI).abs() < 1e-12);
        assert!(approx("2^-2").abs() - 0.25 < 1e-12);
        assert!(matches!(
            parse("x+1").unwrap().evalf(),
            Err(CoreError::InvalidOperation(_))
        ));
        assert!(Expr::Integer(BigInt::zero()).evalf().unwrap() == 0.0);
    }
}
