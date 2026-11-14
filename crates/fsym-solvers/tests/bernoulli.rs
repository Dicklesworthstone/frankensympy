use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigRational, Expr, Symbol};
use fsym_solvers::ode::bernoulli::{
    BernoulliCertificate, BernoulliError, bernoulli, generate_bernoulli_certificate,
    metered_bernoulli, verify_bernoulli_certificate,
};

fn rational(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}
fn axes() -> (Symbol, Symbol) {
    (Symbol::new("x"), Symbol::new("C1"))
}

#[test]
fn logistic_and_cubic_nonlinearity_have_exact_transformed_families() {
    let (x, c) = axes();
    let q = [rational(0, 1), rational(1, 1)];
    let solution = bernoulli(&rational(2, 1), &q, 2, &x, &c).unwrap();
    assert_eq!(
        solution.certificate().particular,
        [rational(1, 4), rational(1, 2)]
    );
    assert_eq!(solution.p(), &rational(2, 1));
    assert_eq!(solution.q(), &q);
    assert!(solution.has_zero_solution());
    let solution = bernoulli(&rational(2, 1), &q, 3, &x, &c).unwrap();
    assert_eq!(
        solution.certificate().particular,
        [rational(1, 8), rational(1, 2)]
    );
    assert_eq!(solution.exponent(), 3);
    assert_eq!(solution.variable(), &x);
    assert_eq!(solution.constant(), &c);
}

#[test]
fn negative_exponent_excludes_the_singular_zero_branch() {
    let (x, c) = axes();
    let solution = bernoulli(
        &rational(2, 1),
        &[rational(0, 1), rational(1, 1)],
        -1,
        &x,
        &c,
    )
    .unwrap();
    assert_eq!(
        solution.certificate().particular,
        [rational(-1, 8), rational(1, 2)]
    );
    assert!(!solution.has_zero_solution());
}

#[test]
fn vanishing_linear_coefficient_uses_an_exact_polynomial_primitive() {
    let (x, c) = axes();
    let solution = bernoulli(
        &rational(0, 1),
        &[rational(2, 3), rational(-7, 2)],
        4,
        &x,
        &c,
    )
    .unwrap();
    assert_eq!(
        solution.certificate().particular,
        [rational(0, 1), rational(-2, 1), rational(21, 4)]
    );
    let zero = bernoulli(&rational(0, 1), &[rational(0, 1)], 2, &x, &c).unwrap();
    assert_eq!(zero.transformed(), &Expr::Sym(c));
}

#[test]
fn independent_differentiation_checks_lifted_output_across_signed_exponents() {
    let (x, c) = axes();
    for n in [-16, -5, -2, -1, 2, 3, 8, 16] {
        for p in [rational(-3, 2), rational(0, 1), rational(2, 5)] {
            let source = [rational(5, 7), rational(-2, 3), rational(1, 6)];
            let solution = bernoulli(&p, &source, n, &x, &c).unwrap();
            let u = solution.transformed();
            let q = Expr::Rational(source[0].clone())
                + Expr::Rational(source[1].clone()) * Expr::Sym(x.clone())
                + Expr::Rational(source[2].clone()) * Expr::Sym(x.clone()).pow(Expr::from_i64(2));
            let m = Expr::from_i64(i64::from(1 - n));
            let residual = fsym_calculus::diff(u, &x)
                + m.clone() * Expr::Rational(p.clone()) * u.clone()
                - m * q;
            let residual = fsym_simplify::expand(&residual);
            assert!(fsym_simplify::simplify(&residual).is_zero(), "n={n}, p={p}");
        }
    }
}

#[test]
fn degree_sixteen_and_zero_forcing_retain_complete_polynomial_directions() {
    let (x, c) = axes();
    let mut q = vec![rational(0, 1); 17];
    q[16] = rational(17, 3);
    let s = bernoulli(&rational(0, 1), &q, -2, &x, &c).unwrap();
    assert_eq!(s.certificate().particular.len(), 18);
    assert_eq!(s.certificate().particular[17], rational(1, 1));
    let s = bernoulli(&rational(9, 2), &[rational(0, 1)], 2, &x, &c).unwrap();
    assert_eq!(s.certificate().particular, [rational(0, 1)]);
}

#[test]
fn original_coefficients_and_every_candidate_coefficient_are_checked() {
    let (x, c) = axes();
    let p = rational(2, 1);
    let q = [rational(0, 1), rational(1, 1)];
    let certificate = generate_bernoulli_certificate(&p, &q, 2, &x, &c, &mut Unbounded).unwrap();
    for index in 0..certificate.particular.len() {
        let mut altered = certificate.clone();
        altered.particular[index] += rational(1, 1_000_000);
        assert!(matches!(
            verify_bernoulli_certificate(&p, &q, 2, &x, &c, altered, &mut Unbounded),
            Err(BernoulliError::Verification(_))
        ));
    }
    for (p2, q2, n2) in [
        (rational(3, 1), q.to_vec(), 2),
        (p.clone(), vec![rational(0, 1), rational(2, 1)], 2),
        (p.clone(), q.to_vec(), 3),
    ] {
        assert!(
            verify_bernoulli_certificate(&p2, &q2, n2, &x, &c, certificate.clone(), &mut Unbounded)
                .is_err()
        );
    }
    let mut altered = certificate;
    altered.particular.push(rational(1, 1));
    assert!(verify_bernoulli_certificate(&p, &q, 2, &x, &c, altered, &mut Unbounded).is_err());
}

#[test]
fn malformed_sources_and_certificates_refuse_before_publication() {
    let (x, c) = axes();
    for n in [i32::MIN, -17, 0, 1, 17, i32::MAX] {
        assert!(bernoulli(&rational(1, 1), &[rational(1, 1)], n, &x, &c).is_err());
    }
    for q in [
        vec![],
        vec![rational(1, 1), rational(0, 1)],
        vec![rational(1, 1); 18],
    ] {
        assert!(bernoulli(&rational(1, 1), &q, 2, &x, &c).is_err());
    }
    assert!(bernoulli(&rational(1, 1), &[rational(1, 1)], 2, &x, &x).is_err());
    assert!(
        verify_bernoulli_certificate(
            &rational(1, 1),
            &[rational(1, 1)],
            2,
            &x,
            &c,
            BernoulliCertificate { particular: vec![] },
            &mut Unbounded
        )
        .is_err()
    );
    let huge = BigRational::from(fsym_core::BigInt::from(1) << 1024u32);
    assert!(bernoulli(&huge, &[rational(1, 1)], 2, &x, &c).is_err());
}

#[derive(Default)]
struct Counts {
    checks: usize,
    stop: Option<usize>,
    charged: [u64; 3],
}
impl BudgetMeter for Counts {
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checks += 1;
        if self.stop == Some(self.checks) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        if dimension.index() < 3 {
            self.charged[dimension.index()] += amount;
        }
        Ok(())
    }
    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        for &(dimension, amount) in charges {
            self.charge(dimension, amount)?;
        }
        Ok(())
    }
}

#[test]
fn each_checkpoint_and_exact_or_one_short_budget_in_both_phases_is_effective() {
    let (x, c) = axes();
    let p = rational(2, 3);
    let q = [rational(1, 7), rational(-2, 5)];
    let mut generator = Counts::default();
    let mut verifier = Counts::default();
    metered_bernoulli(&p, &q, 3, &x, &c, &mut generator, &mut verifier).unwrap();
    for (phase, baseline) in [(true, generator), (false, verifier)] {
        for stop in 1..=baseline.checks {
            let mut stopped = Counts {
                stop: Some(stop),
                ..Counts::default()
            };
            let result = if phase {
                metered_bernoulli(&p, &q, 3, &x, &c, &mut stopped, &mut Unbounded)
            } else {
                metered_bernoulli(&p, &q, 3, &x, &c, &mut Unbounded, &mut stopped)
            };
            assert!(matches!(
                result,
                Err(BernoulliError::Meter(MeterError::Cancelled))
            ));
        }
        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            for short in [false, true] {
                let mut limits = BudgetLimits::uniform(u64::MAX, 0);
                limits.dimensions[dimension.index()] =
                    baseline.charged[dimension.index()] - u64::from(short);
                let mut budget = Budget::new(limits);
                let result = if phase {
                    metered_bernoulli(&p, &q, 3, &x, &c, &mut budget, &mut Unbounded)
                } else {
                    metered_bernoulli(&p, &q, 3, &x, &c, &mut Unbounded, &mut budget)
                };
                assert_eq!(result.is_ok(), !short);
            }
        }
    }
}
