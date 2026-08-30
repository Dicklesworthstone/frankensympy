//! # fsym-functions
//!
//! Constructors for sine, cosine, tangent, cotangent, secant, cosecant,
//! inverse trigonometric, exponential, logarithm, hyperbolic, gamma,
//! zeta, factorial, binomial, fibonacci, lucas, harmonic, catalan,
//! bernoulli, bell, subfactorial, floor, and ceiling expressions, with exact identity values.

#![forbid(unsafe_code)]

use fsym_core::{BigInt, BigRational, Constant, Expr};

/// Create a sine function expression: sin(x).
pub fn sin(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    Expr::Function("sin".to_string(), vec![arg])
}

/// Create a cosine function expression: cos(x).
pub fn cos(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(1);
    }
    Expr::Function("cos".to_string(), vec![arg])
}

/// Create a tangent function expression: tan(x).
pub fn tan(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    Expr::Function("tan".to_string(), vec![arg])
}

/// Create a cotangent function expression: cot(x).
pub fn cot(arg: Expr) -> Expr {
    Expr::Function("cot".to_string(), vec![arg])
}

/// Create a secant function expression: sec(x).
pub fn sec(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(1);
    }
    Expr::Function("sec".to_string(), vec![arg])
}

/// Create a cosecant function expression: csc(x).
pub fn csc(arg: Expr) -> Expr {
    Expr::Function("csc".to_string(), vec![arg])
}

/// Create an arcsine function expression: asin(x).
pub fn asin(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    Expr::Function("asin".to_string(), vec![arg])
}

/// Create an arccosine function expression: acos(x).
pub fn acos(arg: Expr) -> Expr {
    if arg.is_one() {
        return Expr::from_i64(0);
    }
    Expr::Function("acos".to_string(), vec![arg])
}

/// Create an arctangent function expression: atan(x).
pub fn atan(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    Expr::Function("atan".to_string(), vec![arg])
}

/// Create an arccotangent function expression: acot(x).
pub fn acot(arg: Expr) -> Expr {
    Expr::Function("acot".to_string(), vec![arg])
}

/// Create an arcsecant function expression: asec(x).
pub fn asec(arg: Expr) -> Expr {
    if arg.is_one() {
        return Expr::from_i64(0);
    }
    Expr::Function("asec".to_string(), vec![arg])
}

/// Create an arccosecant function expression: acsc(x).
pub fn acsc(arg: Expr) -> Expr {
    Expr::Function("acsc".to_string(), vec![arg])
}

/// Create a hyperbolic sine function expression: sinh(x).
pub fn sinh(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    Expr::Function("sinh".to_string(), vec![arg])
}

/// Create a hyperbolic cosine function expression: cosh(x).
pub fn cosh(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(1);
    }
    Expr::Function("cosh".to_string(), vec![arg])
}

/// Create a hyperbolic tangent function expression: tanh(x).
pub fn tanh(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    Expr::Function("tanh".to_string(), vec![arg])
}

/// Create an inverse hyperbolic sine expression: asinh(x).
pub fn asinh(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    Expr::Function("asinh".to_string(), vec![arg])
}

/// Create an inverse hyperbolic cosine expression: acosh(x).
pub fn acosh(arg: Expr) -> Expr {
    if arg.is_one() {
        return Expr::from_i64(0);
    }
    Expr::Function("acosh".to_string(), vec![arg])
}

/// Create an inverse hyperbolic tangent expression: atanh(x).
pub fn atanh(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    Expr::Function("atanh".to_string(), vec![arg])
}

/// Create an inverse hyperbolic cotangent expression: acoth(x).
pub fn acoth(arg: Expr) -> Expr {
    Expr::Function("acoth".to_string(), vec![arg])
}

/// Create an inverse hyperbolic secant expression: asech(x).
pub fn asech(arg: Expr) -> Expr {
    if arg.is_one() {
        return Expr::from_i64(0);
    }
    Expr::Function("asech".to_string(), vec![arg])
}

/// Create an inverse hyperbolic cosecant expression: acsch(x).
pub fn acsch(arg: Expr) -> Expr {
    Expr::Function("acsch".to_string(), vec![arg])
}

/// Create an unnormalized sinc expression: sinc(x) = sin(x)/x with sinc(0) = 1.
pub fn sinc(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(1);
    }
    Expr::Function("sinc".to_string(), vec![arg])
}

/// Create an error function expression: erf(x).
pub fn erf(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    Expr::Function("erf".to_string(), vec![arg])
}

/// Create a complementary error function expression: erfc(x).
pub fn erfc(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(1);
    }
    Expr::Function("erfc".to_string(), vec![arg])
}

/// Create an absolute value function expression: |x|.
pub fn abs_val(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    if let Expr::Integer(n) = &arg {
        return Expr::Integer(if n < &BigInt::from(0) {
            -n.clone()
        } else {
            n.clone()
        });
    }
    if let Expr::Rational(r) = &arg {
        return Expr::Rational(if r < &BigRational::from_integer(0.into()) {
            -r.clone()
        } else {
            r.clone()
        });
    }
    if matches!(
        arg,
        Expr::Const(Constant::Pi | Constant::E | Constant::Infinity)
    ) {
        return arg;
    }
    if arg == Expr::Const(Constant::NegativeInfinity) {
        return Expr::Const(Constant::Infinity);
    }
    Expr::Function("Abs".to_string(), vec![arg])
}

/// Create a signum function expression: sign(x).
pub fn sign(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    if let Expr::Integer(n) = &arg {
        return Expr::from_i64(if n > &BigInt::from(0) { 1 } else { -1 });
    }
    if let Expr::Rational(r) = &arg {
        return Expr::from_i64(if r > &BigRational::from_integer(0.into()) {
            1
        } else {
            -1
        });
    }
    if matches!(
        arg,
        Expr::Const(Constant::Pi | Constant::E | Constant::Infinity)
    ) {
        return Expr::from_i64(1);
    }
    if arg == Expr::Const(Constant::NegativeInfinity) {
        return Expr::from_i64(-1);
    }
    Expr::Function("sign".to_string(), vec![arg])
}

/// Create an exponential function expression: exp(x).
pub fn exp(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(1);
    }
    Expr::Function("exp".to_string(), vec![arg])
}

/// Create a natural logarithm function expression: log(x).
pub fn log(arg: Expr) -> Expr {
    if arg.is_one() {
        return Expr::from_i64(0);
    }
    if arg == Expr::Const(Constant::E) {
        return Expr::from_i64(1);
    }
    Expr::Function("log".to_string(), vec![arg])
}

/// Create a Gamma function expression: Γ(x).
pub fn gamma(arg: Expr) -> Expr {
    if arg.is_one() {
        return Expr::from_i64(1);
    }
    if arg == Expr::from_i64(2) {
        return Expr::from_i64(1);
    }
    if let Expr::Integer(n) = &arg
        && n > &BigInt::from(0)
        && n <= &BigInt::from(100)
    {
        let n_u64 = n.to_u64().unwrap_or(0);
        if n_u64 >= 1 {
            let mut acc = BigInt::from(1);
            for i in 1..n_u64 {
                acc *= BigInt::from(i);
            }
            return Expr::Integer(acc);
        }
    }
    Expr::Function("gamma".to_string(), vec![arg])
}

/// Create a factorial function expression: n!.
pub fn factorial(arg: Expr) -> Expr {
    if arg.is_zero() || arg.is_one() {
        return Expr::from_i64(1);
    }
    if let Expr::Integer(n) = &arg
        && n > &BigInt::from(0)
        && n <= &BigInt::from(100)
    {
        let n_u64 = n.to_u64().unwrap_or(0);
        let mut acc = BigInt::from(1);
        for i in 1..=n_u64 {
            acc *= BigInt::from(i);
        }
        return Expr::Integer(acc);
    }
    Expr::Function("factorial".to_string(), vec![arg])
}

fn is_exact_negative_integer(expr: &Expr) -> bool {
    match expr {
        Expr::Integer(value) => value < &BigInt::from(0),
        Expr::Rational(value) => value.is_integer() && value < &BigRational::from_integer(0.into()),
        _ => false,
    }
}

fn self_binomial_is_unconditionally_one(expr: &Expr) -> bool {
    match expr {
        Expr::Integer(value) => value >= &BigInt::from(0),
        Expr::Rational(value) => {
            !value.is_integer() || value >= &BigRational::from_integer(0.into())
        }
        Expr::Const(Constant::Pi | Constant::E | Constant::I) => true,
        _ => false,
    }
}

/// Create a binomial coefficient expression: (n choose k).
pub fn binomial(n: Expr, k: Expr) -> Expr {
    if k.is_zero() {
        return Expr::from_i64(1);
    }
    // The generalized binomial coefficient is zero for every exact negative
    // integer lower index. This must precede the equal-argument shortcut:
    // binomial(-1, -1) is zero, not one.
    if is_exact_negative_integer(&k) {
        return Expr::from_i64(0);
    }
    // x choose x is conditional for a symbolic x because x may be a negative
    // integer. Fold only inputs whose concrete value rules that pole out.
    if n == k && self_binomial_is_unconditionally_one(&n) {
        return Expr::from_i64(1);
    }
    if let (Expr::Integer(ni), Expr::Integer(ki)) = (&n, &k)
        && ki >= &BigInt::from(0)
        && ni >= &BigInt::from(0)
        && ki > ni
    {
        // For non-negative integer n < k, the binomial coefficient
        // is 0 by the standard convention, not an unevaluated
        // function. This matches the existing top-level "k is zero"
        // and "n equals k" early-exit paths.
        return Expr::from_i64(0);
    }
    if let (Expr::Integer(ni), Expr::Integer(ki)) = (&n, &k)
        && ki >= &BigInt::from(0)
        && ni >= ki
        && ni <= &BigInt::from(100)
    {
        let n_val = ni.to_u64().unwrap_or(0);
        let k_val = ki.to_u64().unwrap_or(0);
        let k_opt = k_val.min(n_val - k_val);
        let mut result = BigInt::from(1);
        for i in 0..k_opt {
            result = (result * BigInt::from(n_val - i)) / BigInt::from(i + 1);
        }
        return Expr::Integer(result);
    }
    Expr::Function("binomial".to_string(), vec![n, k])
}

/// Create a Fibonacci number expression: F_n.
pub fn fibonacci(n: Expr) -> Expr {
    if n.is_zero() {
        return Expr::from_i64(0);
    }
    if n.is_one() {
        return Expr::from_i64(1);
    }
    if let Expr::Integer(ni) = &n
        && ni > &BigInt::from(0)
        && ni <= &BigInt::from(200)
    {
        let n_val = ni.to_u64().unwrap_or(0);
        let mut a = BigInt::from(0);
        let mut b = BigInt::from(1);
        for _ in 2..=n_val {
            let c = &a + &b;
            a = b;
            b = c;
        }
        return Expr::Integer(b);
    }
    Expr::Function("fibonacci".to_string(), vec![n])
}

/// Create a Lucas number expression: L_n (L_0 = 2, L_1 = 1, L_n = L_{n-1} + L_{n-2}).
pub fn lucas(n: Expr) -> Expr {
    if n.is_zero() {
        return Expr::from_i64(2);
    }
    if n.is_one() {
        return Expr::from_i64(1);
    }
    if let Expr::Integer(ni) = &n
        && ni > &BigInt::from(0)
        && ni <= &BigInt::from(200)
    {
        let n_val = ni.to_u64().unwrap_or(0);
        let mut a = BigInt::from(2);
        let mut b = BigInt::from(1);
        for _ in 2..=n_val {
            let c = &a + &b;
            a = b;
            b = c;
        }
        return Expr::Integer(b);
    }
    Expr::Function("lucas".to_string(), vec![n])
}

/// Create a Harmonic number expression: H_n = Σ_{k=1}^n 1/k.
pub fn harmonic(n: Expr) -> Expr {
    if n.is_zero() {
        return Expr::from_i64(0);
    }
    if n.is_one() {
        return Expr::from_i64(1);
    }
    if let Expr::Integer(ni) = &n
        && ni > &BigInt::from(0)
        && ni <= &BigInt::from(100)
    {
        let n_val = ni.to_u64().unwrap_or(0);
        let mut sum = BigRational::from_integer(0.into());
        for k in 1..=n_val {
            sum += BigRational::new(1.into(), (k as i64).into());
        }
        if sum.is_integer() {
            return Expr::Integer(sum.to_integer());
        } else {
            return Expr::Rational(sum);
        }
    }
    Expr::Function("harmonic".to_string(), vec![n])
}

/// Create a Catalan number expression: C_n = 1/(n+1) * (2n choose n).
pub fn catalan(n: Expr) -> Expr {
    if n.is_zero() || n.is_one() {
        return Expr::from_i64(1);
    }
    if let Expr::Integer(ni) = &n
        && ni > &BigInt::from(0)
        && ni <= &BigInt::from(60)
    {
        let n_val = ni.to_u64().unwrap_or(0);
        let mut c = BigInt::from(1);
        for k in 0..n_val {
            c = (c * BigInt::from(2 * (2 * k + 1))) / BigInt::from(k + 2);
        }
        return Expr::Integer(c);
    }
    Expr::Function("catalan".to_string(), vec![n])
}

/// Create a Bernoulli number expression: B_n (using the standard B_1 = -1/2 convention).
pub fn bernoulli(n: Expr) -> Expr {
    if n.is_zero() {
        return Expr::from_i64(1);
    }
    if n.is_one() {
        return Expr::rational(-1, 2).expect("valid rational -1/2");
    }
    if let Expr::Integer(ni) = &n {
        if ni > &BigInt::from(1) && (ni % BigInt::from(2)) != BigInt::from(0) {
            // All odd Bernoulli numbers > 1 are zero
            return Expr::from_i64(0);
        }
        if ni > &BigInt::from(0) && ni <= &BigInt::from(60) {
            let n_val = ni.to_u64().unwrap_or(0) as usize;
            let mut a: Vec<BigRational> = (0..=n_val)
                .map(|m| BigRational::new(1.into(), ((m + 1) as i64).into()))
                .collect();
            for j in 1..=n_val {
                for m in 0..=(n_val - j) {
                    let diff = &a[m] - &a[m + 1];
                    a[m] = BigRational::from_integer(((m + 1) as i64).into()) * diff;
                }
            }
            let b_n = a[0].clone();
            return if b_n.is_integer() {
                Expr::Integer(b_n.to_integer())
            } else {
                Expr::Rational(b_n)
            };
        }
    }
    Expr::Function("bernoulli".to_string(), vec![n])
}

/// Create a Bell number expression: B_n (number of partitions of a set of n elements).
pub fn bell(n: Expr) -> Expr {
    if n.is_zero() || n.is_one() {
        return Expr::from_i64(1);
    }
    if let Expr::Integer(ni) = &n
        && ni > &BigInt::from(0)
        && ni <= &BigInt::from(60)
    {
        let n_val = ni.to_u64().unwrap_or(0) as usize;
        let mut row = vec![BigInt::from(1)];
        for _ in 1..=n_val {
            let mut next_row = Vec::with_capacity(row.len() + 1);
            next_row.push(row.last().unwrap().clone());
            for j in 0..row.len() {
                let sum = &next_row[j] + &row[j];
                next_row.push(sum);
            }
            row = next_row;
        }
        return Expr::Integer(row.first().unwrap().clone());
    }
    Expr::Function("bell".to_string(), vec![n])
}

/// Create a subfactorial expression: !n (number of derangements of n elements).
pub fn subfactorial(n: Expr) -> Expr {
    if n.is_zero() {
        return Expr::from_i64(1);
    }
    if n.is_one() {
        return Expr::from_i64(0);
    }
    if let Expr::Integer(ni) = &n
        && ni > &BigInt::from(0)
        && ni <= &BigInt::from(60)
    {
        let n_val = ni.to_u64().unwrap_or(0);
        let mut d_prev2 = BigInt::from(1); // !0 = 1
        let mut d_prev1 = BigInt::from(0); // !1 = 0
        let mut curr = BigInt::from(0);
        for i in 2..=n_val {
            curr = BigInt::from(i - 1) * (&d_prev1 + &d_prev2);
            d_prev2 = d_prev1;
            d_prev1 = curr.clone();
        }
        return Expr::Integer(curr);
    }
    Expr::Function("subfactorial".to_string(), vec![n])
}

/// Create a floor function expression: ⌊x⌋.
pub fn floor(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    if let Expr::Integer(n) = &arg {
        return Expr::Integer(n.clone());
    }
    if let Expr::Rational(r) = &arg {
        let p = r.numer();
        let q = r.denom();
        let div = p / q;
        let rem = p % q;
        let fl = if rem < BigInt::from(0) {
            div - BigInt::from(1)
        } else {
            div
        };
        return Expr::Integer(fl);
    }
    if arg == Expr::Const(Constant::Pi) {
        return Expr::from_i64(3);
    }
    if arg == Expr::Const(Constant::E) {
        return Expr::from_i64(2);
    }
    if matches!(
        arg,
        Expr::Const(Constant::Infinity | Constant::NegativeInfinity)
    ) {
        return arg;
    }
    Expr::Function("floor".to_string(), vec![arg])
}

/// Create a ceiling function expression: ⌈x⌉.
pub fn ceiling(arg: Expr) -> Expr {
    if arg.is_zero() {
        return Expr::from_i64(0);
    }
    if let Expr::Integer(n) = &arg {
        return Expr::Integer(n.clone());
    }
    if let Expr::Rational(r) = &arg {
        let p = r.numer();
        let q = r.denom();
        let div = p / q;
        let rem = p % q;
        let ceil = if rem > BigInt::from(0) {
            div + BigInt::from(1)
        } else {
            div
        };
        return Expr::Integer(ceil);
    }
    if arg == Expr::Const(Constant::Pi) {
        return Expr::from_i64(4);
    }
    if arg == Expr::Const(Constant::E) {
        return Expr::from_i64(3);
    }
    if matches!(
        arg,
        Expr::Const(Constant::Infinity | Constant::NegativeInfinity)
    ) {
        return arg;
    }
    Expr::Function("ceiling".to_string(), vec![arg])
}

/// Create a Riemann Zeta function expression: ζ(s).
pub fn zeta(arg: Expr) -> Expr {
    Expr::Function("zeta".to_string(), vec![arg])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elementary_evaluations() {
        assert_eq!(sin(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(cos(Expr::from_i64(0)), Expr::from_i64(1));
        assert_eq!(tan(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(sec(Expr::from_i64(0)), Expr::from_i64(1));
        assert_eq!(asin(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(acos(Expr::from_i64(1)), Expr::from_i64(0));
        assert_eq!(atan(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(asec(Expr::from_i64(1)), Expr::from_i64(0));
        assert_eq!(sinh(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(cosh(Expr::from_i64(0)), Expr::from_i64(1));
        assert_eq!(tanh(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(asinh(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(acosh(Expr::from_i64(1)), Expr::from_i64(0));
        assert_eq!(atanh(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(asech(Expr::from_i64(1)), Expr::from_i64(0));
        assert_eq!(sinc(Expr::from_i64(0)), Expr::from_i64(1));
        assert_eq!(erf(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(erfc(Expr::from_i64(0)), Expr::from_i64(1));
        assert_eq!(exp(Expr::from_i64(0)), Expr::from_i64(1));
        assert_eq!(log(Expr::from_i64(1)), Expr::from_i64(0));
        assert_eq!(log(Expr::Const(Constant::E)), Expr::from_i64(1));
    }

    #[test]
    fn test_abs_and_sign() {
        assert_eq!(abs_val(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(abs_val(Expr::from_i64(5)), Expr::from_i64(5));
        assert_eq!(abs_val(Expr::from_i64(-7)), Expr::from_i64(7));
        assert_eq!(
            abs_val(Expr::rational(-3, 4).unwrap()),
            Expr::rational(3, 4).unwrap()
        );

        assert_eq!(sign(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(sign(Expr::from_i64(42)), Expr::from_i64(1));
        assert_eq!(sign(Expr::from_i64(-99)), Expr::from_i64(-1));
        assert_eq!(sign(Expr::rational(5, 3).unwrap()), Expr::from_i64(1));
        assert_eq!(sign(Expr::rational(-5, 3).unwrap()), Expr::from_i64(-1));
    }

    #[test]
    fn test_combinatorial_and_special() {
        assert_eq!(factorial(Expr::from_i64(0)), Expr::from_i64(1));
        assert_eq!(factorial(Expr::from_i64(1)), Expr::from_i64(1));
        assert_eq!(factorial(Expr::from_i64(5)), Expr::from_i64(120));

        assert_eq!(gamma(Expr::from_i64(1)), Expr::from_i64(1));
        assert_eq!(gamma(Expr::from_i64(2)), Expr::from_i64(1));
        assert_eq!(gamma(Expr::from_i64(5)), Expr::from_i64(24));

        assert_eq!(
            binomial(Expr::from_i64(5), Expr::from_i64(2)),
            Expr::from_i64(10)
        );
        assert_eq!(
            binomial(Expr::from_i64(10), Expr::from_i64(0)),
            Expr::from_i64(1)
        );
        assert_eq!(
            binomial(Expr::from_i64(10), Expr::from_i64(10)),
            Expr::from_i64(1)
        );
        assert_eq!(
            binomial(Expr::from_i64(5), Expr::from_i64(-1)),
            Expr::from_i64(0)
        );
        assert_eq!(
            binomial(Expr::from_i64(-1), Expr::from_i64(-1)),
            Expr::from_i64(0)
        );

        // A public Expr can contain an integer-valued Rational directly, so
        // the negative-lower-index guard must not depend on constructor
        // canonicalization.
        let rational_negative_one = Expr::Rational(BigRational::from_integer(BigInt::from(-1)));
        assert_eq!(
            binomial(Expr::symbol("n"), rational_negative_one.clone()),
            Expr::from_i64(0)
        );
        assert_eq!(
            binomial(rational_negative_one.clone(), rational_negative_one),
            Expr::from_i64(0)
        );

        let x = Expr::symbol("x");
        assert_eq!(
            binomial(x.clone(), x.clone()),
            Expr::Function("binomial".to_string(), vec![x.clone(), x])
        );
        assert_eq!(
            binomial(
                Expr::Const(Constant::Infinity),
                Expr::Const(Constant::Infinity),
            ),
            Expr::Function(
                "binomial".to_string(),
                vec![
                    Expr::Const(Constant::Infinity),
                    Expr::Const(Constant::Infinity),
                ],
            )
        );
        let negative_half = Expr::rational(-1, 2).unwrap();
        assert_eq!(
            binomial(negative_half.clone(), negative_half),
            Expr::from_i64(1)
        );
        // k > n with both non-negative: standard convention is 0,
        // not a deferred function. n = 0 / k = 1 covers the empty-set
        // boundary; 5 / 6 covers a non-trivial k > n.
        assert_eq!(
            binomial(Expr::from_i64(0), Expr::from_i64(1)),
            Expr::from_i64(0)
        );
        assert_eq!(
            binomial(Expr::from_i64(5), Expr::from_i64(6)),
            Expr::from_i64(0)
        );
        assert_eq!(
            binomial(Expr::from_i64(10), Expr::from_i64(15)),
            Expr::from_i64(0)
        );

        assert_eq!(fibonacci(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(fibonacci(Expr::from_i64(1)), Expr::from_i64(1));
        assert_eq!(fibonacci(Expr::from_i64(2)), Expr::from_i64(1));
        assert_eq!(fibonacci(Expr::from_i64(10)), Expr::from_i64(55));

        assert_eq!(lucas(Expr::from_i64(0)), Expr::from_i64(2));
        assert_eq!(lucas(Expr::from_i64(1)), Expr::from_i64(1));
        assert_eq!(lucas(Expr::from_i64(2)), Expr::from_i64(3));
        assert_eq!(lucas(Expr::from_i64(5)), Expr::from_i64(11));

        assert_eq!(harmonic(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(harmonic(Expr::from_i64(1)), Expr::from_i64(1));
        assert_eq!(harmonic(Expr::from_i64(2)), Expr::rational(3, 2).unwrap());
        assert_eq!(harmonic(Expr::from_i64(4)), Expr::rational(25, 12).unwrap());
    }

    #[test]
    fn test_fibonacci_lucas_and_harmonic() {
        // The constructors fold specific integer inputs to the exact value
        // (F_n, L_n, H_n) and return an opaque function form for non-integer
        // inputs. Pin both the fold-to-value contract and the no-fold
        // contract for symbolic inputs so any change is loud in code review.
        // Fibonacci: F_0 = 0, F_1 = 1, F_n = F_{n-1} + F_{n-2}.
        assert_eq!(fibonacci(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(fibonacci(Expr::from_i64(1)), Expr::from_i64(1));
        assert_eq!(fibonacci(Expr::from_i64(2)), Expr::from_i64(1));
        assert_eq!(fibonacci(Expr::from_i64(3)), Expr::from_i64(2));
        assert_eq!(fibonacci(Expr::from_i64(10)), Expr::from_i64(55));
        // Lucas: L_0 = 2, L_1 = 1, L_n = L_{n-1} + L_{n-2}.
        assert_eq!(lucas(Expr::from_i64(0)), Expr::from_i64(2));
        assert_eq!(lucas(Expr::from_i64(1)), Expr::from_i64(1));
        assert_eq!(lucas(Expr::from_i64(2)), Expr::from_i64(3));
        assert_eq!(lucas(Expr::from_i64(3)), Expr::from_i64(4));
        assert_eq!(lucas(Expr::from_i64(10)), Expr::from_i64(123));
        // Harmonic: H_0 = 0, H_1 = 1, H_n = Σ_{k=1}^n 1/k.
        assert_eq!(harmonic(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(harmonic(Expr::from_i64(1)), Expr::from_i64(1));
        assert_eq!(harmonic(Expr::from_i64(2)), Expr::rational(3, 2).unwrap());
        assert_eq!(harmonic(Expr::from_i64(3)), Expr::rational(11, 6).unwrap());
        assert_eq!(harmonic(Expr::from_i64(4)), Expr::rational(25, 12).unwrap());
        // Symbolic inputs return the opaque function form (no fold).
        let x = Expr::symbol("x");
        assert_eq!(
            fibonacci(x.clone()),
            Expr::Function("fibonacci".to_string(), vec![x.clone()])
        );
        assert_eq!(
            lucas(x.clone()),
            Expr::Function("lucas".to_string(), vec![x.clone()])
        );
        assert_eq!(
            harmonic(x.clone()),
            Expr::Function("harmonic".to_string(), vec![x.clone()])
        );
    }

    #[test]
    fn test_opaque_no_fold_functions_keep_function_form() {
        // cot, csc, acot, acsc, acoth, acsch, and zeta all have no
        // identity-point fold in the current constructor. They emit
        // Function("name", [arg]) for any input — symbolic or numeric.
        // Pin that contract so any future fold addition is loud in code
        // review (a fold is a real semantic change, not a refactor).
        let x = Expr::symbol("x");
        let cases: Vec<(&str, Expr, &str)> = vec![
            ("cot", cot(x.clone()), "cot"),
            ("csc", csc(x.clone()), "csc"),
            ("acot", acot(x.clone()), "acot"),
            ("acsc", acsc(x.clone()), "acsc"),
            ("acoth", acoth(x.clone()), "acoth"),
            ("acsch", acsch(x.clone()), "acsch"),
            ("zeta", zeta(x.clone()), "zeta"),
        ];
        for (label, constructed, expected_name) in &cases {
            let inner = match constructed {
                Expr::Function(name, args) => (name.clone(), args.clone()),
                other => panic!("{label}: expected Function form, got {other:?}"),
            };
            assert_eq!(inner.0, *expected_name, "{label}: function name");
            assert_eq!(inner.1, vec![Expr::symbol("x")], "{label}: arg passthrough");
        }
        // Same constructors also emit the function form for integer / Rational
        // inputs that would be singular (cot 0, csc 0, zeta 1, ...). The
        // constructor does not refuse or refuse-fold these; it surfaces them
        // to the simplifier / kernel as opaque forms.
        for (label, value) in [
            ("cot(0)", cot(Expr::from_i64(0))),
            ("csc(0)", csc(Expr::from_i64(0))),
            ("acot(0)", acot(Expr::from_i64(0))),
            ("acsc(0)", acsc(Expr::from_i64(0))),
            ("acoth(0)", acoth(Expr::from_i64(0))),
            ("acsch(0)", acsch(Expr::from_i64(0))),
            ("zeta(1)", zeta(Expr::from_i64(1))),
        ] {
            let (_name, args) = match value {
                Expr::Function(name, args) => (name, args),
                other => panic!("{label}: expected Function form, got {other:?}"),
            };
            assert_eq!(args.len(), 1, "{label}: arity");
        }
    }

    #[test]
    fn test_catalan_and_bernoulli_and_bell_and_subfactorial() {
        assert_eq!(catalan(Expr::from_i64(1)), Expr::from_i64(1));
        assert_eq!(catalan(Expr::from_i64(2)), Expr::from_i64(2));
        assert_eq!(catalan(Expr::from_i64(3)), Expr::from_i64(5));
        assert_eq!(catalan(Expr::from_i64(4)), Expr::from_i64(14));
        assert_eq!(catalan(Expr::from_i64(5)), Expr::from_i64(42));
        let x = Expr::symbol("x");
        assert_eq!(
            catalan(x.clone()),
            Expr::Function("catalan".to_string(), vec![x.clone()])
        );

        assert_eq!(bernoulli(Expr::from_i64(0)), Expr::from_i64(1));
        assert_eq!(bernoulli(Expr::from_i64(1)), Expr::rational(-1, 2).unwrap());
        assert_eq!(bernoulli(Expr::from_i64(2)), Expr::rational(1, 6).unwrap());
        assert_eq!(bernoulli(Expr::from_i64(3)), Expr::from_i64(0));
        assert_eq!(
            bernoulli(Expr::from_i64(4)),
            Expr::rational(-1, 30).unwrap()
        );
        assert_eq!(bernoulli(Expr::from_i64(5)), Expr::from_i64(0));
        assert_eq!(bernoulli(Expr::from_i64(6)), Expr::rational(1, 42).unwrap());
        assert_eq!(bernoulli(Expr::from_i64(7)), Expr::from_i64(0));
        assert_eq!(
            bernoulli(Expr::from_i64(8)),
            Expr::rational(-1, 30).unwrap()
        );

        assert_eq!(bell(Expr::from_i64(0)), Expr::from_i64(1));
        assert_eq!(bell(Expr::from_i64(1)), Expr::from_i64(1));
        assert_eq!(bell(Expr::from_i64(2)), Expr::from_i64(2));
        assert_eq!(bell(Expr::from_i64(3)), Expr::from_i64(5));
        assert_eq!(bell(Expr::from_i64(4)), Expr::from_i64(15));
        assert_eq!(bell(Expr::from_i64(5)), Expr::from_i64(52));

        assert_eq!(subfactorial(Expr::from_i64(0)), Expr::from_i64(1));
        assert_eq!(subfactorial(Expr::from_i64(1)), Expr::from_i64(0));
        assert_eq!(subfactorial(Expr::from_i64(2)), Expr::from_i64(1));
        assert_eq!(subfactorial(Expr::from_i64(3)), Expr::from_i64(2));
        assert_eq!(subfactorial(Expr::from_i64(4)), Expr::from_i64(9));
        assert_eq!(subfactorial(Expr::from_i64(5)), Expr::from_i64(44));
        assert_eq!(subfactorial(Expr::from_i64(6)), Expr::from_i64(265));
    }

    #[test]
    fn test_floor_and_ceiling() {
        assert_eq!(floor(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(floor(Expr::from_i64(5)), Expr::from_i64(5));
        assert_eq!(floor(Expr::from_i64(-4)), Expr::from_i64(-4));
        assert_eq!(floor(Expr::rational(7, 2).unwrap()), Expr::from_i64(3));
        assert_eq!(floor(Expr::rational(-7, 2).unwrap()), Expr::from_i64(-4));
        assert_eq!(floor(Expr::rational(6, 2).unwrap()), Expr::from_i64(3));
        assert_eq!(floor(Expr::rational(-6, 2).unwrap()), Expr::from_i64(-3));
        assert_eq!(floor(Expr::Const(Constant::Pi)), Expr::from_i64(3));
        assert_eq!(floor(Expr::Const(Constant::E)), Expr::from_i64(2));
        assert_eq!(
            floor(Expr::Const(Constant::Infinity)),
            Expr::Const(Constant::Infinity)
        );
        let x = Expr::symbol("x");
        assert_eq!(
            floor(x.clone()),
            Expr::Function("floor".to_string(), vec![x.clone()])
        );

        assert_eq!(ceiling(Expr::from_i64(0)), Expr::from_i64(0));
        assert_eq!(ceiling(Expr::from_i64(5)), Expr::from_i64(5));
        assert_eq!(ceiling(Expr::from_i64(-4)), Expr::from_i64(-4));
        assert_eq!(ceiling(Expr::rational(7, 2).unwrap()), Expr::from_i64(4));
        assert_eq!(ceiling(Expr::rational(-7, 2).unwrap()), Expr::from_i64(-3));
        assert_eq!(ceiling(Expr::rational(6, 2).unwrap()), Expr::from_i64(3));
        assert_eq!(ceiling(Expr::rational(-6, 2).unwrap()), Expr::from_i64(-3));
        assert_eq!(ceiling(Expr::Const(Constant::Pi)), Expr::from_i64(4));
        assert_eq!(ceiling(Expr::Const(Constant::E)), Expr::from_i64(3));
        assert_eq!(
            ceiling(Expr::Const(Constant::Infinity)),
            Expr::Const(Constant::Infinity)
        );
        assert_eq!(
            ceiling(x.clone()),
            Expr::Function("ceiling".to_string(), vec![x])
        );

        assert_eq!(
            abs_val(Expr::Const(Constant::Pi)),
            Expr::Const(Constant::Pi)
        );
        assert_eq!(
            abs_val(Expr::Const(Constant::NegativeInfinity)),
            Expr::Const(Constant::Infinity)
        );
        assert_eq!(sign(Expr::Const(Constant::Pi)), Expr::from_i64(1));
        assert_eq!(
            sign(Expr::Const(Constant::NegativeInfinity)),
            Expr::from_i64(-1)
        );
    }
}
