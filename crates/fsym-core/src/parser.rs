//! Recursive-descent expression parser producing canonical [`Expr`] ASTs.
//!
//! Grammar (standard precedence, `^` and `**` right-associative):
//!
//! ```text
//! expr  := term (('+' | '-') term)*
//! term  := unary (('*' | '/') unary)*
//! unary := ('-' | '+') unary | power
//! power := atom (('^' | '**') unary)?
//! atom  := number | constant | symbol | func '(' args ')' | '(' expr ')'
//! ```
//!
//! Division of two integer literals folds to an exact rational; decimal
//! literals parse as exact rationals scaled by powers of ten. There are no
//! floating-point leaves anywhere in the numeric tower.

use crate::{BigInt, BigRational};
use crate::{Constant, CoreError, Expr, Symbol};
use std::sync::Arc;

/// Maximum source size admitted by the unsupervised convenience parser.
///
/// Callers needing a different policy must use a future supervised parser
/// lane rather than making this allocation boundary implicit.
const MAX_PARSE_INPUT_BYTES: usize = 1024 * 1024;

/// Maximum recursive syntactic nesting admitted before descending further.
const MAX_PARSE_NESTING_DEPTH: usize = 256;

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
                if matches!(chars.peek(), Some('*')) {
                    chars.next();
                    toks.push(Tok::Caret);
                } else {
                    toks.push(Tok::Star);
                }
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
    nesting_depth: usize,
    preserve_source: bool,
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

    fn nested<T>(
        &mut self,
        parse_nested: impl FnOnce(&mut Self) -> Result<T, CoreError>,
    ) -> Result<T, CoreError> {
        if self.nesting_depth >= MAX_PARSE_NESTING_DEPTH {
            return Err(self.err("maximum parser nesting depth exceeded"));
        }
        self.nesting_depth += 1;
        let result = parse_nested(self);
        self.nesting_depth -= 1;
        result
    }

    fn expr(&mut self) -> Result<Expr, CoreError> {
        let mut left = self.term()?;
        loop {
            match self.peek() {
                Some(Tok::Plus) => {
                    self.bump();
                    let right = self.term()?;
                    left = self.add(left, right);
                }
                Some(Tok::Minus) => {
                    self.bump();
                    let right = self.term()?;
                    left = self.add(left, neg(right));
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
                    left = self.multiply(left, right);
                }
                Some(Tok::Slash) => {
                    self.bump();
                    let right = self.unary()?;
                    left = if self.preserve_source {
                        divide_raw(left, right)?
                    } else {
                        divide(left, right)?
                    };
                }
                _ => break,
            }
        }
        Ok(left)
    }

    fn add(&self, left: Expr, right: Expr) -> Expr {
        if !self.preserve_source {
            return left + right;
        }
        // Flatten only the already visited left-associated spine. Every
        // operand survives in its original evaluation order, including zero,
        // repeated, and opposite terms. A long flat input does not create an
        // unbounded recursive clone/drop spine.
        if let Expr::Add(mut terms) = left {
            terms.push(right);
            Expr::Add(terms)
        } else {
            Expr::Add(vec![left, right])
        }
    }

    fn multiply(&self, left: Expr, right: Expr) -> Expr {
        if !self.preserve_source {
            return left * right;
        }
        if let Expr::Mul(mut factors) = left {
            factors.push(right);
            Expr::Mul(factors)
        } else {
            Expr::Mul(vec![left, right])
        }
    }

    fn unary(&mut self) -> Result<Expr, CoreError> {
        match self.peek() {
            Some(Tok::Minus) => {
                self.bump();
                Ok(neg(self.nested(Self::unary)?))
            }
            Some(Tok::Plus) => {
                self.bump();
                self.nested(Self::unary)
            }
            _ => self.power(),
        }
    }

    fn power(&mut self) -> Result<Expr, CoreError> {
        let base = self.atom()?;
        if matches!(self.peek(), Some(Tok::Caret)) {
            self.bump();
            let exp = self.nested(Self::unary)?;
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
                    self.nested(|parser| {
                        let mut args = Vec::new();
                        if !matches!(parser.peek(), Some(Tok::RParen)) {
                            args.push(parser.expr()?);
                            while matches!(parser.peek(), Some(Tok::Comma)) {
                                parser.bump();
                                args.push(parser.expr()?);
                            }
                        }
                        parser.expect(&Tok::RParen)?;
                        // `sqrt(a)` is notation for `a**(1/2)` (the printers
                        // render half powers that way); keep the AST uniform.
                        if name == "sqrt" && args.len() == 1 {
                            let arg = args.pop().expect("one argument");
                            return Ok(Expr::Pow(
                                Arc::new(arg),
                                Arc::new(Expr::Rational(BigRational::new(
                                    BigInt::from(1),
                                    BigInt::from(2),
                                ))),
                            ));
                        }
                        Ok(Expr::Function(name, args))
                    })
                } else {
                    match constant_named(&name) {
                        Some(c) => Ok(Expr::Const(c)),
                        None => Ok(Expr::Sym(Symbol::new(name))),
                    }
                }
            }
            Some(Tok::LParen) => self.nested(|parser| {
                let e = parser.expr()?;
                parser.expect(&Tok::RParen)?;
                Ok(e)
            }),
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

/// Preserve division's domain, including a zero literal denominator in an
/// inactive branch. Folding two finite numeric literals with a nonzero
/// denominator is exact and cannot erase a source-domain obligation.
fn divide_raw(a: Expr, b: Expr) -> Result<Expr, CoreError> {
    let numerator_is_literal = matches!(&a, Expr::Integer(_) | Expr::Rational(_));
    let denominator_is_nonzero_literal = match &b {
        Expr::Integer(n) => !n.is_zero(),
        Expr::Rational(q) => !q.numer().is_zero(),
        _ => false,
    };
    if numerator_is_literal && denominator_is_nonzero_literal {
        divide(a, b)
    } else {
        let reciprocal = Expr::Pow(Arc::new(b), Arc::new(Expr::from_i64(-1)));
        Ok(if let Expr::Mul(mut factors) = a {
            factors.push(reciprocal);
            Expr::Mul(factors)
        } else {
            Expr::Mul(vec![a, reciprocal])
        })
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
    let scale = u32::try_from(frac.len())
        .map_err(|_| CoreError::ParseError("decimal scale exceeds u32".to_string()))?;
    let denominator = BigInt::from(10u32).pow(scale);
    let r = BigRational::new(numerator, denominator);
    if r.is_integer() {
        Ok(Expr::Integer(r.to_integer()))
    } else {
        Ok(Expr::Rational(r))
    }
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
/// [`CoreError::ParseError`] on malformed, oversized, or excessively nested
/// input; [`CoreError::DivisionByZero`] for literal division by zero.
pub fn parse(input: &str) -> Result<Expr, CoreError> {
    parse_with_policy(input, false)
}

/// Parse exact arithmetic while retaining original domain obligations.
///
/// Addends and factors keep their input order and multiplicity. In particular,
/// zero products, opposite terms, and zero-denominator reciprocals remain in
/// the typed source for an independent domain checker. Powers never evaluate
/// during parsing. Decimal literals, unary numeric negation, and division of
/// two numeric literals with a nonzero denominator remain exact.
///
/// This uses the same grammar and intrinsic source/nesting bounds as [`parse`].
/// It is a construction API, not a domain, differentiation, or root proof.
pub fn parse_raw(input: &str) -> Result<Expr, CoreError> {
    parse_with_policy(input, true)
}

fn parse_with_policy(input: &str, preserve_source: bool) -> Result<Expr, CoreError> {
    if input.len() > MAX_PARSE_INPUT_BYTES {
        return Err(CoreError::ParseError(format!(
            "input exceeds {MAX_PARSE_INPUT_BYTES}-byte parser limit"
        )));
    }
    let toks = tokenize(input)?;
    if toks.is_empty() {
        return Err(CoreError::ParseError(format!("empty input `{input}`")));
    }
    let mut p = Parser {
        toks: &toks,
        pos: 0,
        src: input,
        nesting_depth: 0,
        preserve_source,
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
            // Canonical Add order (fra-add-args-canonical-order-o1i): exact
            // numbers precede symbols, matching pinned SymPy 1.14.0 args.
            Expr::Add(vec![Expr::from_i64(12), Expr::symbol("x")])
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
        assert_eq!(parse("x**2**3").unwrap(), parse("x^2^3").unwrap());
    }

    #[test]
    fn displayed_powers_round_trip_through_the_parser() {
        let expression = parse("(x + 1)^(-2)").unwrap();
        assert_eq!(parse(&expression.to_string()).unwrap(), expression);
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
        // Integer-valued decimals canonicalize to Expr::Integer
        assert_eq!(parse("2.0").unwrap(), Expr::from_i64(2));
        assert_eq!(parse("0.0").unwrap(), Expr::from_i64(0));
        assert_eq!(parse("-4.00").unwrap(), Expr::from_i64(-4));
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
    fn raw_parser_preserves_zero_products_cancelled_terms_and_operand_order() {
        let x = Expr::symbol("x");
        let y = Expr::symbol("y");
        assert_eq!(
            parse_raw("y+x+y+0").unwrap(),
            Expr::Add(vec![y.clone(), x.clone(), y.clone(), Expr::from_i64(0)])
        );
        assert_eq!(
            parse_raw("y*x*y*0").unwrap(),
            Expr::Mul(vec![y.clone(), x.clone(), y, Expr::from_i64(0)])
        );
        assert_eq!(
            parse_raw("x-x").unwrap(),
            Expr::Add(vec![x.clone(), Expr::Mul(vec![Expr::from_i64(-1), x])])
        );
        assert_eq!(parse("0*(1/x)").unwrap(), Expr::from_i64(0));
        let raw = parse_raw("0*(1/x)").unwrap();
        assert!(matches!(&raw, Expr::Mul(factors) if factors.len() == 2));
        assert!(
            crate::enclose_residual_jacobian(
                &[raw],
                &[Symbol::new("x")],
                &[crate::RealBall::from_i64(0)],
                16,
            )
            .is_err()
        );
        let raw = parse_raw("log(x)-log(x)").unwrap();
        assert!(
            crate::enclose_residual_jacobian(
                &[raw],
                &[Symbol::new("x")],
                &[crate::RealBall::from_i64(-1)],
                16,
            )
            .is_err()
        );
    }

    #[test]
    fn raw_parser_keeps_exact_literals_without_evaluating_powers_or_zero_division() {
        for source in ["1/2", "-1/2", "0.125", "sqrt(x)", "x**2**3"] {
            assert_eq!(parse_raw(source).unwrap(), parse(source).unwrap());
        }
        assert_eq!(parse("1/0"), Err(CoreError::DivisionByZero));
        assert_eq!(
            parse_raw("1/0").unwrap(),
            Expr::Mul(vec![
                Expr::from_i64(1),
                Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1))),
            ])
        );
        assert_eq!(
            parse_raw("2^1000000000").unwrap(),
            Expr::Pow(
                Arc::new(Expr::from_i64(2)),
                Arc::new(Expr::from_i64(1_000_000_000)),
            )
        );
        let flat = std::iter::repeat_n("x", 1024).collect::<Vec<_>>().join("+");
        assert!(matches!(parse_raw(&flat).unwrap(), Expr::Add(terms) if terms.len() == 1024));
        let division = std::iter::repeat_n("x", 1024).collect::<Vec<_>>().join("/");
        let divided = parse_raw(&division).unwrap();
        let Expr::Mul(factors) = divided else {
            panic!("a raw division chain must remain a flat ordered product");
        };
        assert_eq!(factors.len(), 1024);
        assert_eq!(factors[0], Expr::symbol("x"));
        for reciprocal in &factors[1..] {
            assert_eq!(
                reciprocal,
                &Expr::Pow(Arc::new(Expr::symbol("x")), Arc::new(Expr::from_i64(-1)))
            );
        }
        let undefined = parse_raw("0/(x-x)").unwrap();
        assert!(
            crate::enclose_residual_jacobian(
                &[undefined],
                &[Symbol::new("x")],
                &[crate::RealBall::from_i64(1)],
                16,
            )
            .is_err()
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
    fn parser_preflights_input_size_and_recursive_nesting() {
        let oversized = "1".repeat(MAX_PARSE_INPUT_BYTES + 1);
        assert_eq!(
            parse(&oversized),
            Err(CoreError::ParseError(format!(
                "input exceeds {MAX_PARSE_INPUT_BYTES}-byte parser limit"
            )))
        );

        let admitted = format!(
            "{}x{}",
            "(".repeat(MAX_PARSE_NESTING_DEPTH),
            ")".repeat(MAX_PARSE_NESTING_DEPTH)
        );
        assert_eq!(parse(&admitted).unwrap(), Expr::symbol("x"));

        let refused = format!(
            "{}x{}",
            "(".repeat(MAX_PARSE_NESTING_DEPTH + 1),
            ")".repeat(MAX_PARSE_NESTING_DEPTH + 1)
        );
        assert!(matches!(parse(&refused), Err(CoreError::ParseError(_))));
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
