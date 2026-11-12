//! Recursive QQ/ZZ division: whole leading coefficients, not monomial reduction.
//!
//! The public examples and domain boundary were independently checked against
//! isolated SymPy1.14.0; expected polynomials below are hand supplied. No oracle
//! is used by the implementation or by these native tests.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Symbol, SymbolIdentity, parse};
use fsym_polys::{
    MultivariatePoly as Polynomial, RecursiveCoefficientObstruction as Obstruction,
    RecursiveDivisionCertificate as Certificate, RecursiveDivisionDomain as Domain, TermOrder,
    VerifiedRecursiveDivision as Checked, bounded_polynomial_from_expr,
    bounded_recursive_division_certificate as propose, checked_polynomial_division,
    checked_recursive_division as divide, verify_recursive_division as verify,
};
use std::collections::BTreeMap;

fn ring(names: &[&str]) -> Vec<Symbol> {
    names.iter().map(|name| Symbol::new(*name)).collect()
}

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn p(expression: &str, variables: &[Symbol]) -> Polynomial {
    bounded_polynomial_from_expr(&parse(expression).unwrap(), variables, &mut Unbounded).unwrap()
}

fn solve(dividend: &Polynomial, divisor: &Polynomial, domain: Domain) -> Checked {
    divide(dividend, divisor, domain, &mut Unbounded, &mut Unbounded).unwrap()
}

fn rejected(dividend: &Polynomial, divisor: &Polynomial, domain: Domain, claim: &Certificate) {
    assert!(verify(dividend, divisor, domain, claim, &mut Unbounded).is_err());
}

#[test]
fn recursive_qq_division_stops_before_a_nonpolynomial_coefficient_quotient() {
    let variables = ring(&["x", "y"]);
    let f = p("x*y+1", &variables);
    let g = p("x*(y+1)+1", &variables);
    let result = solve(&f, &g, Domain::QQ);
    assert_eq!(result.dividend(), &f);
    assert_eq!(result.divisor(), &g);
    assert_eq!(result.domain(), Domain::QQ);
    assert!(result.quotient().is_zero());
    assert_eq!(result.remainder(), &f);
    let certificate = propose(&f, &g, Domain::QQ, &mut Unbounded).unwrap();
    let terminal = certificate.terminal.as_ref().unwrap();
    assert_eq!(terminal.quotient, p("1", &variables));
    assert_eq!(terminal.remainder, p("-1", &variables));
    assert_eq!(
        verify(&f, &g, Domain::QQ, &certificate, &mut Unbounded).unwrap(),
        result
    );

    // The old public route computed this valid distributed normal form. Its
    // quotient/remainder are not the profile's recursive division result.
    let (quotients, remainder) = checked_polynomial_division(
        &f,
        std::slice::from_ref(&g),
        TermOrder::Lex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(quotients, [p("1", &variables)]);
    assert_eq!(remainder, p("-x", &variables));
    let forged = Certificate {
        quotient: quotients[0].clone(),
        remainder,
        terminal: None,
    };
    rejected(&f, &g, Domain::QQ, &forged);
}

#[test]
fn earlier_exact_steps_survive_a_later_leading_coefficient_obstruction() {
    let variables = ring(&["x", "y"]);
    let f = p("3*x**2*y+2*x*y**2+4*y+5", &variables);
    let g = p("2*x*y+3*y**2+1", &variables);
    let result = solve(&f, &g, Domain::QQ);
    assert_eq!(result.quotient(), &p("3/2*x", &variables));
    assert_eq!(
        result.remainder(),
        &p("-5/2*x*y**2-3/2*x+4*y+5", &variables)
    );
    let certificate = propose(&f, &g, Domain::QQ, &mut Unbounded).unwrap();
    let terminal = certificate.terminal.as_ref().unwrap();
    assert_eq!(terminal.quotient, p("-5/4*y", &variables));
    assert_eq!(terminal.remainder, p("-3/2", &variables));
    let (quotient, remainder) = result.clone().into_parts();
    assert_eq!(&quotient, result.quotient());
    assert_eq!(&remainder, result.remainder());
}

#[test]
fn full_nonmonomial_coefficients_divide_in_three_variables() {
    let variables = ring(&["x", "y", "z"]);
    let g = p("(y+z)*x+y*z+1", &variables);
    let f = p("((y-z)*x+z)*((y+z)*x+y*z+1)+y+1", &variables);
    for domain in [Domain::QQ, Domain::ZZ] {
        let result = solve(&f, &g, domain);
        assert_eq!(result.quotient(), &p("(y-z)*x+z", &variables));
        assert_eq!(result.remainder(), &p("y+1", &variables));
        let certificate = propose(&f, &g, domain, &mut Unbounded).unwrap();
        assert!(certificate.terminal.is_none());
        assert_eq!(
            verify(&f, &g, domain, &certificate, &mut Unbounded).unwrap(),
            result
        );
    }
}

#[test]
fn generator_order_changes_the_primary_variable_and_the_obstruction() {
    let xy = ring(&["x", "y"]);
    let yx = ring(&["y", "x"]);
    let result = solve(&p("x*y+1", &yx), &p("x*(y+1)+1", &yx), Domain::QQ);
    assert_eq!(result.quotient(), &p("1", &yx));
    assert_eq!(result.remainder(), &p("-x", &yx));
    let certificate = propose(
        &p("x*y+1", &yx),
        &p("x*(y+1)+1", &yx),
        Domain::QQ,
        &mut Unbounded,
    )
    .unwrap();
    rejected(
        &p("x*y+1", &xy),
        &p("x*(y+1)+1", &xy),
        Domain::QQ,
        &certificate,
    );
}

#[test]
fn strict_integer_division_does_not_promote_a_fractional_coefficient_step() {
    let variables = ring(&["x", "y"]);
    let f = p("x", &variables);
    let g = p("2*x", &variables);
    let rational = solve(&f, &g, Domain::QQ);
    assert_eq!(rational.quotient(), &p("1/2", &variables));
    assert!(rational.remainder().is_zero());
    let integer = solve(&f, &g, Domain::ZZ);
    assert!(integer.quotient().is_zero());
    assert_eq!(integer.remainder(), &f);
    let certificate = propose(&f, &g, Domain::ZZ, &mut Unbounded).unwrap();
    let terminal = certificate.terminal.as_ref().unwrap();
    assert_eq!(terminal.quotient, p("1/2", &variables));
    assert!(terminal.remainder.is_zero());
    rejected(&f, &g, Domain::QQ, &certificate);
    let rational_certificate = propose(&f, &g, Domain::QQ, &mut Unbounded).unwrap();
    rejected(&f, &g, Domain::ZZ, &rational_certificate);
}

#[test]
fn strict_integer_obstruction_is_checked_after_all_prior_integral_steps() {
    let variables = ring(&["x", "y"]);
    let f = p("2*x**2+3*x+2", &variables);
    let g = p("2*x+1", &variables);
    let result = solve(&f, &g, Domain::ZZ);
    assert_eq!(result.quotient(), &p("x+1", &variables));
    assert_eq!(result.remainder(), &p("1", &variables));
    assert!(
        propose(&f, &g, Domain::ZZ, &mut Unbounded)
            .unwrap()
            .terminal
            .is_none()
    );
    let f = p("2*x**2+2*x+1", &variables);
    let result = solve(&f, &g, Domain::ZZ);
    assert_eq!(result.quotient(), &p("x", &variables));
    assert_eq!(result.remainder(), &p("x+1", &variables));
    let rational = solve(&f, &g, Domain::QQ);
    assert_eq!(rational.quotient(), &p("x+1/2", &variables));
    assert_eq!(rational.remainder(), &p("1/2", &variables));
}

#[test]
fn primary_constant_divisors_still_obey_whole_coefficient_divisibility() {
    let variables = ring(&["x", "y"]);
    for domain in [Domain::QQ, Domain::ZZ] {
        let f = p("y*x+1", &variables);
        let g = p("y", &variables);
        let result = solve(&f, &g, domain);
        assert_eq!(result.quotient(), &p("x", &variables));
        assert_eq!(result.remainder(), &p("1", &variables));
        assert!(
            propose(&f, &g, domain, &mut Unbounded)
                .unwrap()
                .terminal
                .is_some()
        );
        let f = p("x+y", &variables);
        let result = solve(&f, &g, domain);
        assert!(result.quotient().is_zero());
        assert_eq!(result.remainder(), &f);
    }
}

#[test]
fn exact_improper_univariate_zero_and_scalar_divisions_are_normalized() {
    let variables = ring(&["x", "y"]);
    for (a, b, quotient, remainder) in [
        ("x**3-1", "x-1", "x**2+x+1", "0"),
        ("x**2+1", "x+1", "x-1", "2"),
        ("x+1", "x**2+1", "0", "x+1"),
        ("0", "2*x+1", "0", "0"),
        ("3", "2", "3/2", "0"),
        ("x**2+y", "-2", "-x**2/2-y/2", "0"),
        ("(x+y)*(x-y)", "x-y", "x+y", "0"),
    ] {
        let f = p(a, &variables);
        let g = p(b, &variables);
        let result = solve(&f, &g, Domain::QQ);
        assert_eq!(result.quotient(), &p(quotient, &variables));
        assert_eq!(result.remainder(), &p(remainder, &variables));
    }
}

#[test]
fn degree256_boundary_retains_every_forced_recursive_step() {
    let variables = ring(&["x", "y"]);
    let f = p("x**256", &variables);
    let g = p("x-y", &variables);
    let expected = Polynomial::new(
        variables.clone(),
        (0..256u32).map(|k| (vec![255 - k, k], q(1, 1))).collect(),
    )
    .unwrap();
    let result = solve(&f, &g, Domain::ZZ);
    assert_eq!(result.quotient(), &expected);
    assert_eq!(result.remainder(), &p("y**256", &variables));
}

#[test]
fn eight_axis_coefficients_and_same_name_typed_symbols_remain_distinct() {
    let variables: Vec<_> = (0..8).map(|i| Symbol::new(format!("x{i}"))).collect();
    let f = p("(x7+1)*x0**2+x0*x1+x7", &variables);
    let g = p("(x7+1)*x0+1", &variables);
    let result = solve(&f, &g, Domain::QQ);
    assert_eq!(result.quotient(), &p("x0", &variables));
    assert_eq!(result.remainder(), &p("x0*x1-x0+x7", &variables));
    let a = Symbol::with_identity("t", SymbolIdentity::from_assumptions_digest([1; 32]));
    let b = Symbol::with_identity("t", SymbolIdentity::from_assumptions_digest([2; 32]));
    let variables = vec![a, b];
    let f = Polynomial::new(
        variables.clone(),
        BTreeMap::from([(vec![1, 1], q(1, 1)), (vec![0, 0], q(1, 1))]),
    )
    .unwrap();
    let g = Polynomial::new(
        variables.clone(),
        BTreeMap::from([
            (vec![1, 1], q(1, 1)),
            (vec![1, 0], q(1, 1)),
            (vec![0, 0], q(1, 1)),
        ]),
    )
    .unwrap();
    let result = solve(&f, &g, Domain::QQ);
    assert!(result.quotient().is_zero());
    assert_eq!(result.remainder(), &f);
    let mut certificate = propose(&f, &g, Domain::QQ, &mut Unbounded).unwrap();
    certificate.remainder.generators[1] = Symbol::new("t");
    rejected(&f, &g, Domain::QQ, &certificate);
}

#[test]
fn reconstruction_cannot_replace_the_forced_recursive_step_obligation() {
    let variables = ring(&["x", "y"]);
    let f = p("(y+1)*x**2+x*y+1", &variables);
    let g = p("(y+1)*x+1", &variables);
    let certificate = propose(&f, &g, Domain::QQ, &mut Unbounded).unwrap();
    assert_eq!(certificate.quotient, p("x", &variables));
    // Skipping a forced exact step still satisfies f=q*g+r, but the copied
    // terminal witness belongs to a different leading coefficient.
    let mut skipped = certificate.clone();
    skipped.quotient = p("0", &variables);
    skipped.remainder = f.clone();
    rejected(&f, &g, Domain::QQ, &skipped);
    // Extra lower steps also preserve reconstruction, but cannot cancel the
    // whole coefficient y-1 by y+1 at the first actual obstruction.
    let mut extra = certificate.clone();
    extra.quotient = p("x+1", &variables);
    extra.remainder = p("-2*x", &variables);
    rejected(&f, &g, Domain::QQ, &extra);
    let mut changed = certificate.clone();
    *changed.quotient.terms.values_mut().next().unwrap() += q(1, 1_000_000_007);
    rejected(&f, &g, Domain::QQ, &changed);
    let mut changed = certificate;
    changed.remainder.terms.pop_first();
    rejected(&f, &g, Domain::QQ, &changed);
}

#[test]
fn terminal_coefficient_witnesses_require_exact_identity_and_complete_reduction() {
    let variables = ring(&["x", "y"]);
    let f = p("x*y+1", &variables);
    let g = p("x*(y+1)+1", &variables);
    let certificate = propose(&f, &g, Domain::QQ, &mut Unbounded).unwrap();
    for terminal in [
        None,
        Some(Obstruction {
            quotient: p("1", &variables),
            remainder: p("0", &variables),
        }),
        // y=0*(y+1)+y is a correct identity but its remainder is reducible.
        Some(Obstruction {
            quotient: p("0", &variables),
            remainder: p("y", &variables),
        }),
        Some(Obstruction {
            quotient: p("2", &variables),
            remainder: p("-1", &variables),
        }),
        Some(Obstruction {
            quotient: p("x", &variables),
            remainder: p("-1", &variables),
        }),
    ] {
        let mut changed = certificate.clone();
        changed.terminal = terminal;
        rejected(&f, &g, Domain::QQ, &changed);
    }
    let f = p("x**2-1", &variables);
    let g = p("x-1", &variables);
    let mut changed = propose(&f, &g, Domain::QQ, &mut Unbounded).unwrap();
    changed.terminal = certificate.terminal;
    rejected(&f, &g, Domain::QQ, &changed);
}

#[test]
fn exact_original_coefficients_divisor_and_typed_ring_are_not_rebindable() {
    let variables = ring(&["x", "y"]);
    let f = p("(x+y)*(x-y)+1", &variables);
    let g = p("x-y", &variables);
    let certificate = propose(&f, &g, Domain::QQ, &mut Unbounded).unwrap();
    rejected(
        &p("(x+y)*(x-y)+2", &variables),
        &g,
        Domain::QQ,
        &certificate,
    );
    rejected(&f, &p("x+y", &variables), Domain::QQ, &certificate);
    rejected(&g, &f, Domain::QQ, &certificate);
    let mut renamed = certificate.clone();
    renamed.quotient.generators[1] = Symbol::new("z");
    rejected(&f, &g, Domain::QQ, &renamed);
}

#[test]
fn scalar_and_zero_shortcuts_do_not_bypass_domain_or_shape_admission() {
    let variables = ring(&["x", "y"]);
    for (f, g) in [
        (p("0", &variables), p("0", &variables)),
        (p("1", &variables), p("0", &variables)),
    ] {
        for domain in [Domain::QQ, Domain::ZZ] {
            assert!(divide(&f, &g, domain, &mut Unbounded, &mut Unbounded).is_err());
        }
    }
    for (f, g) in [
        (p("0", &variables), p("1/2", &variables)),
        (p("1/2", &variables), p("1", &variables)),
    ] {
        assert!(divide(&f, &g, Domain::ZZ, &mut Unbounded, &mut Unbounded).is_err());
        let certificate = propose(&f, &g, Domain::QQ, &mut Unbounded).unwrap();
        rejected(&f, &g, Domain::ZZ, &certificate);
    }
    let empty = Polynomial::new(vec![], BTreeMap::from([(vec![], q(1, 1))])).unwrap();
    assert!(divide(&empty, &empty, Domain::QQ, &mut Unbounded, &mut Unbounded).is_err());
    let one = p("1", &variables);
    for malformed in [
        Polynomial {
            generators: variables.clone(),
            terms: BTreeMap::from([(vec![0], q(1, 1))]),
        },
        Polynomial {
            generators: variables.clone(),
            terms: BTreeMap::from([(vec![0, 0], q(0, 1))]),
        },
        Polynomial {
            generators: vec![variables[0].clone(), variables[0].clone()],
            terms: BTreeMap::new(),
        },
        Polynomial {
            generators: vec![Symbol::new("x".repeat(65_537))],
            terms: BTreeMap::new(),
        },
        Polynomial {
            generators: (0..9).map(|i| Symbol::new(format!("v{i}"))).collect(),
            terms: BTreeMap::new(),
        },
    ] {
        assert!(divide(&malformed, &one, Domain::QQ, &mut Unbounded, &mut Unbounded).is_err());
        assert!(divide(&one, &malformed, Domain::QQ, &mut Unbounded, &mut Unbounded).is_err());
    }
}

#[test]
fn coefficient_degree_and_untrusted_output_growth_refuse_without_partial_answers() {
    let variables = ring(&["x", "y"]);
    let f = p("x**256", &variables);
    let g = p("x-y**2", &variables);
    assert!(divide(&f, &g, Domain::QQ, &mut Unbounded, &mut Unbounded).is_err());
    let one = p("1", &variables);
    let huge = Polynomial::new(
        variables.clone(),
        BTreeMap::from([(
            vec![0, 0],
            BigRational::from_integer(BigInt::one() << 4096u32),
        )]),
    )
    .unwrap();
    assert!(divide(&huge, &one, Domain::QQ, &mut Unbounded, &mut Unbounded).is_err());
    let f = p("x**2-y**2", &variables);
    let g = p("x-y", &variables);
    let certificate = propose(&f, &g, Domain::QQ, &mut Unbounded).unwrap();
    let mut changed = certificate.clone();
    changed.quotient.terms.insert(vec![257, 0], q(1, 1));
    rejected(&f, &g, Domain::QQ, &changed);
    let mut changed = certificate.clone();
    changed.remainder = huge;
    rejected(&f, &g, Domain::QQ, &changed);
    let mut changed = certificate;
    changed.quotient.generators[0] = Symbol::new("x".repeat(65_537));
    rejected(&f, &g, Domain::QQ, &changed);
}

#[derive(Default)]
struct Meter {
    steps: u64,
    memory: u64,
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        if dimension == Dimension::ComputeSteps {
            self.steps += amount;
        }
        if dimension == Dimension::MemoryBytes {
            self.memory += amount;
        }
        Ok(())
    }
    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        for &(dimension, amount) in charges {
            self.charge(dimension, amount)?;
        }
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn separate_phase_budgets_remain_cumulative_through_acceptance() {
    let variables = ring(&["x", "y"]);
    let f = p("x**3+y", &variables);
    let g = p("x+1", &variables);
    let (mut generation, mut verification) = (Meter::default(), Meter::default());
    divide(&f, &g, Domain::QQ, &mut generation, &mut verification).unwrap();
    for phase in [false, true] {
        let observed = if phase { &verification } else { &generation };
        assert!(observed.steps > 0 && observed.memory > 0);
        for cancel_at in [1, observed.checkpoints / 2, observed.checkpoints] {
            let mut interrupted = Meter {
                cancel_at: Some(cancel_at),
                ..Meter::default()
            };
            let error = if phase {
                divide(&f, &g, Domain::QQ, &mut Unbounded, &mut interrupted)
            } else {
                divide(&f, &g, Domain::QQ, &mut interrupted, &mut Unbounded)
            }
            .unwrap_err();
            assert!(error.to_string().to_lowercase().contains("cancel"));
            assert_eq!(interrupted.checkpoints, cancel_at);
        }
        for dimension in [Dimension::ComputeSteps, Dimension::MemoryBytes] {
            let consumed = if dimension == Dimension::ComputeSteps {
                observed.steps
            } else {
                observed.memory
            };
            let mut limits = BudgetLimits::uniform(u64::MAX / 4, 0);
            limits.dimensions[dimension.index()] = consumed - 1;
            let mut budget = Budget::new(limits);
            assert!(
                if phase {
                    divide(&f, &g, Domain::QQ, &mut Unbounded, &mut budget)
                } else {
                    divide(&f, &g, Domain::QQ, &mut budget, &mut Unbounded)
                }
                .is_err()
            );
        }
    }
    let mut limits = BudgetLimits::uniform(u64::MAX / 4, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = generation.steps;
    let mut budget = Budget::new(limits);
    divide(&f, &g, Domain::QQ, &mut budget, &mut Unbounded).unwrap();
    assert!(divide(&f, &g, Domain::QQ, &mut budget, &mut Unbounded).is_err());
}

#[test]
fn every_reference_checkpoint_can_cancel_without_publishing_a_receipt() {
    let variables = ring(&["x"]);
    let f = p("x", &variables);
    let g = p("2*x", &variables);
    let certificate = propose(&f, &g, Domain::ZZ, &mut Unbounded).unwrap();
    let mut observed = Meter::default();
    verify(&f, &g, Domain::ZZ, &certificate, &mut observed).unwrap();
    for cancel_at in 1..=observed.checkpoints {
        let mut meter = Meter {
            cancel_at: Some(cancel_at),
            ..Meter::default()
        };
        let error = verify(&f, &g, Domain::ZZ, &certificate, &mut meter).unwrap_err();
        assert!(error.to_string().to_lowercase().contains("cancel"));
        assert_eq!(meter.checkpoints, cancel_at);
    }
}
