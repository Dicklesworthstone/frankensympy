//! Exact pseudo-division values and adversarial acceptance boundaries.
//!
//! Expected coefficients are hand-derived, including an independent closed
//! formula for quadratic/linear operands. Production never invokes an oracle.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Symbol, SymbolIdentity, parse};
use fsym_polys::{
    MultivariatePoly as Polynomial, PseudoDivisionCertificate as Certificate,
    RecursiveDivisionDomain as Domain, VerifiedPseudoDivision as Checked,
    bounded_polynomial_from_expr, bounded_pseudo_division_certificate as propose,
    checked_pseudo_division as divide, verify_pseudo_division as verify,
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
fn integer_pseudo_division_preserves_the_coefficient_ring() {
    let variables = ring(&["x", "y"]);
    for domain in [Domain::ZZ, Domain::QQ] {
        for (a, b, quotient, remainder, exponent) in [
            ("x**2+1", "2*x-4", "2*x+4", "20", 2),
            ("x**3-1", "2*x-2", "4*x**2+4*x+4", "0", 3),
            ("x**2+1", "2*x+2", "2*x-2", "8", 2),
            ("x**2+1", "(y+1)*x+y", "(y+1)*x-y", "2*y**2+2*y+1", 2),
            (
                "(y+1)*x**2+x*y+1",
                "(y+1)*x+1",
                "(y+1)**2*x+y**2-1",
                "2*y+2",
                2,
            ),
        ] {
            let f = p(a, &variables);
            let g = p(b, &variables);
            let result = solve(&f, &g, domain);
            assert_eq!(result.dividend(), &f);
            assert_eq!(result.divisor(), &g);
            assert_eq!(result.domain(), domain);
            assert_eq!(result.scale_exponent(), exponent);
            assert_eq!(result.quotient(), &p(quotient, &variables));
            assert_eq!(result.remainder(), &p(remainder, &variables));
            let certificate = propose(&f, &g, domain, &mut Unbounded).unwrap();
            assert_eq!(
                verify(&f, &g, domain, &certificate, &mut Unbounded).unwrap(),
                result
            );
        }
    }
}

#[test]
fn skipped_degrees_and_early_exact_division_keep_the_full_original_scale() {
    let variables = ring(&["x"]);
    for (f, g, quotient, remainder, exponent) in [
        ("x**5", "2*x**4", "2*x", "0", 2),
        ("x**4+1", "3*x**2", "9*x**2", "27", 3),
        ("x**4", "3*x**2", "9*x**2", "0", 3),
        ("x**3+1", "-2*x**2", "-2*x", "4", 2),
    ] {
        let result = solve(&p(f, &variables), &p(g, &variables), Domain::ZZ);
        assert_eq!(result.quotient(), &p(quotient, &variables));
        assert_eq!(result.remainder(), &p(remainder, &variables));
        assert_eq!(result.scale_exponent(), exponent);
    }
}

#[test]
fn constants_and_primary_degree_zero_divisors_use_the_same_definition() {
    let variables = ring(&["x", "y"]);
    for (f, g, quotient, exponent) in [
        ("3", "2", "3", 1),
        ("x**2+y", "2", "4*x**2+4*y", 3),
        ("x+1", "y+1", "(y+1)*(x+1)", 2),
        ("y**2+1", "y+1", "y**2+1", 1),
    ] {
        let result = solve(&p(f, &variables), &p(g, &variables), Domain::ZZ);
        assert_eq!(result.quotient(), &p(quotient, &variables));
        assert!(result.remainder().is_zero());
        assert_eq!(result.scale_exponent(), exponent);
    }
}

#[test]
fn zero_and_lower_degree_dividends_have_zero_scaling_exponent() {
    let variables = ring(&["x", "y"]);
    for f in ["0", "x+y"] {
        let f = p(f, &variables);
        let g = p("(y+1)*x**2+y", &variables);
        let result = solve(&f, &g, Domain::ZZ);
        assert!(result.quotient().is_zero());
        assert_eq!(result.remainder(), &f);
        assert_eq!(result.scale_exponent(), 0);
    }
}

#[test]
fn rational_coefficients_require_qq_and_keep_their_exact_scale() {
    let variables = ring(&["x"]);
    let f = p("x**2/2+1/3", &variables);
    let g = p("x/2+1/3", &variables);
    let result = solve(&f, &g, Domain::QQ);
    assert_eq!(result.quotient(), &p("x/4-1/6", &variables));
    assert_eq!(result.remainder(), &p("5/36", &variables));
    assert_eq!(result.scale_exponent(), 2);
    assert!(divide(&f, &g, Domain::ZZ, &mut Unbounded, &mut Unbounded).is_err());
    let certificate = propose(&f, &g, Domain::QQ, &mut Unbounded).unwrap();
    rejected(&f, &g, Domain::ZZ, &certificate);
}

#[test]
fn the_first_declared_axis_is_primary_and_all_other_axes_remain_coefficients() {
    for (variables, quotient, remainder, exponent) in [
        (ring(&["y", "x"]), "x*y-1", "x**3+1", 2),
        (ring(&["x", "y"]), "1", "y**3-1", 1),
    ] {
        let result = solve(
            &p("y**2+x", &variables),
            &p("x*y+1", &variables),
            Domain::ZZ,
        );
        assert_eq!(result.quotient(), &p(quotient, &variables));
        assert_eq!(result.remainder(), &p(remainder, &variables));
        assert_eq!(result.scale_exponent(), exponent);
    }
    let variables: Vec<_> = (0..8).map(|i| Symbol::new(format!("x{i}"))).collect();
    let result = solve(
        &p("x0**2+x7", &variables),
        &p("x7*x0+1", &variables),
        Domain::ZZ,
    );
    assert_eq!(result.quotient(), &p("x7*x0-1", &variables));
    assert_eq!(result.remainder(), &p("x7**3+1", &variables));
    assert_eq!(result.quotient().generators(), variables);
}

#[test]
fn same_name_symbols_keep_distinct_typed_coordinates() {
    let variables = vec![
        Symbol::with_identity("t", SymbolIdentity::from_assumptions_digest([1; 32])),
        Symbol::with_identity("t", SymbolIdentity::from_assumptions_digest([2; 32])),
    ];
    let f = Polynomial::new(
        variables.clone(),
        BTreeMap::from([(vec![2, 0], q(1, 1)), (vec![0, 1], q(1, 1))]),
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
    let result = solve(&f, &g, Domain::ZZ);
    assert_eq!(
        result.quotient().terms(),
        &BTreeMap::from([
            (vec![1, 1], q(1, 1)),
            (vec![1, 0], q(1, 1)),
            (vec![0, 0], q(-1, 1)),
        ])
    );
    assert_eq!(
        result.remainder().terms(),
        &BTreeMap::from([
            (vec![0, 3], q(1, 1)),
            (vec![0, 2], q(2, 1)),
            (vec![0, 1], q(1, 1)),
            (vec![0, 0], q(1, 1)),
        ])
    );
    let mut certificate = propose(&f, &g, Domain::ZZ, &mut Unbounded).unwrap();
    certificate.quotient.generators.swap(0, 1);
    rejected(&f, &g, Domain::ZZ, &certificate);
}

#[test]
fn degree256_inputs_and_degree257_constant_scale_are_supported() {
    let variables = ring(&["x"]);
    let f = p("x**256+1", &variables);
    let result = solve(&f, &p("x**256+2", &variables), Domain::ZZ);
    assert_eq!(result.quotient(), &p("1", &variables));
    assert_eq!(result.remainder(), &p("-1", &variables));
    let result = solve(&f, &p("1", &variables), Domain::ZZ);
    assert_eq!(result.quotient(), &f);
    assert!(result.remainder().is_zero());
    assert_eq!(result.scale_exponent(), 257);
}

#[test]
fn quadratic_linear_corpus_matches_independently_derived_coefficients() {
    let variables = ring(&["x"]);
    for a in [-2i64, 1, 3] {
        for b in [-1i64, 0, 2] {
            for c in [-1i64, 0, 2] {
                for d in [-2i64, 1, 3] {
                    for e in [-1i64, 0, 2] {
                        let f = p(&format!("{a}*x**2+({b})*x+({c})"), &variables);
                        let g = p(&format!("{d}*x+({e})"), &variables);
                        let result = solve(&f, &g, Domain::ZZ);
                        let quotient = p(&format!("{}*x+({})", d * a, d * b - a * e), &variables);
                        let remainder = p(
                            &format!("{}", d * d * c - d * b * e + a * e * e),
                            &variables,
                        );
                        assert_eq!(result.quotient(), &quotient);
                        assert_eq!(result.remainder(), &remainder);
                        assert_eq!(result.scale_exponent(), 2);
                    }
                }
            }
        }
    }
}

#[test]
fn scaling_and_remainder_degree_mutants_are_rejected_independently() {
    let variables = ring(&["x"]);
    let f = p("x**2+1", &variables);
    let g = p("2*x+1", &variables);
    let certificate = propose(&f, &g, Domain::QQ, &mut Unbounded).unwrap();
    let mut high_remainder = certificate.clone();
    // This retains the exact scaled identity while skipping every division step.
    high_remainder.quotient = p("0", &variables);
    high_remainder.remainder = p("4*x**2+4", &variables);
    rejected(&f, &g, Domain::QQ, &high_remainder);
    let mut unscaled = certificate.clone();
    // This is the ordinary QQ division identity, not pseudo-division.
    unscaled.quotient = p("x/2-1/4", &variables);
    unscaled.remainder = p("5/4", &variables);
    rejected(&f, &g, Domain::QQ, &unscaled);
    let monic = p("x+1", &variables);
    let certificate = propose(&f, &monic, Domain::ZZ, &mut Unbounded).unwrap();
    for exponent in [0, 1, 3, u32::MAX] {
        let mut changed = certificate.clone();
        changed.scale_exponent = exponent;
        rejected(&f, &monic, Domain::ZZ, &changed);
    }
}

#[test]
fn coefficient_omission_perturbation_and_original_operand_rebinding_are_rejected() {
    let variables = ring(&["x", "y"]);
    let f = p("x**2+y", &variables);
    let g = p("(y+1)*x+1", &variables);
    let certificate = propose(&f, &g, Domain::QQ, &mut Unbounded).unwrap();
    let mut changed = certificate.clone();
    changed.quotient.terms.pop_first();
    rejected(&f, &g, Domain::QQ, &changed);
    let mut changed = certificate.clone();
    *changed.remainder.terms.values_mut().next().unwrap() += q(1, 1_000_000_007);
    rejected(&f, &g, Domain::QQ, &changed);
    rejected(&p("x**2+y+1", &variables), &g, Domain::QQ, &certificate);
    rejected(&f, &p("(y+1)*x+2", &variables), Domain::QQ, &certificate);
    let mut changed = certificate;
    changed.remainder.generators[1] = Symbol::new("z");
    rejected(&f, &g, Domain::QQ, &changed);
}

#[test]
fn malformed_and_zero_shortcuts_cannot_bypass_original_admission() {
    let variables = ring(&["x", "y"]);
    let zero = p("0", &variables);
    let one = p("1", &variables);
    assert!(divide(&one, &zero, Domain::ZZ, &mut Unbounded, &mut Unbounded).is_err());
    assert!(divide(&zero, &zero, Domain::QQ, &mut Unbounded, &mut Unbounded).is_err());
    assert!(
        divide(
            &zero,
            &p("1/2", &variables),
            Domain::ZZ,
            &mut Unbounded,
            &mut Unbounded,
        )
        .is_err()
    );
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
            generators: vec![],
            terms: BTreeMap::new(),
        },
        Polynomial {
            generators: (0..9).map(|i| Symbol::new(format!("x{i}"))).collect(),
            terms: BTreeMap::new(),
        },
        Polynomial {
            generators: vec![Symbol::new("x".repeat(65_537))],
            terms: BTreeMap::new(),
        },
    ] {
        assert!(divide(&malformed, &one, Domain::QQ, &mut Unbounded, &mut Unbounded).is_err());
        assert!(
            divide(
                &zero,
                &malformed,
                Domain::QQ,
                &mut Unbounded,
                &mut Unbounded
            )
            .is_err()
        );
    }
}

#[test]
fn coefficient_and_intermediate_growth_refuse_without_partial_results() {
    let variables = ring(&["x", "y"]);
    let one = p("1", &variables);
    let huge = Polynomial::new(
        variables.clone(),
        BTreeMap::from([(
            vec![0, 0],
            BigRational::from_integer(BigInt::from(1) << 4096u32),
        )]),
    )
    .unwrap();
    assert!(divide(&huge, &one, Domain::QQ, &mut Unbounded, &mut Unbounded).is_err());
    assert!(
        divide(
            &p("x**256", &variables),
            &p("y*x+1", &variables),
            Domain::ZZ,
            &mut Unbounded,
            &mut Unbounded,
        )
        .is_err()
    );
    let f = p("x", &variables);
    let mut certificate = propose(&f, &one, Domain::ZZ, &mut Unbounded).unwrap();
    certificate.quotient.terms.insert(vec![257, 0], q(1, 1));
    rejected(&f, &one, Domain::ZZ, &certificate);
    let mut certificate = propose(&f, &one, Domain::QQ, &mut Unbounded).unwrap();
    certificate.remainder = huge;
    rejected(&f, &one, Domain::QQ, &certificate);
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
fn separate_phase_budgets_are_cumulative_through_final_publication() {
    let variables = ring(&["x", "y"]);
    let f = p("x**3+y", &variables);
    let g = p("2*x+1", &variables);
    let (mut generation, mut verification) = (Meter::default(), Meter::default());
    divide(&f, &g, Domain::ZZ, &mut generation, &mut verification).unwrap();
    for phase in [false, true] {
        let observed = if phase { &verification } else { &generation };
        assert!(observed.steps > 0 && observed.memory > 0);
        for cancel_at in [1, observed.checkpoints / 2, observed.checkpoints] {
            let mut interrupted = Meter {
                cancel_at: Some(cancel_at),
                ..Meter::default()
            };
            let error = if phase {
                divide(&f, &g, Domain::ZZ, &mut Unbounded, &mut interrupted)
            } else {
                divide(&f, &g, Domain::ZZ, &mut interrupted, &mut Unbounded)
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
                    divide(&f, &g, Domain::ZZ, &mut Unbounded, &mut budget)
                } else {
                    divide(&f, &g, Domain::ZZ, &mut budget, &mut Unbounded)
                }
                .is_err()
            );
        }
    }
    let mut limits = BudgetLimits::uniform(u64::MAX / 4, 0);
    limits.dimensions[Dimension::ComputeSteps.index()] = generation.steps;
    let mut budget = Budget::new(limits);
    divide(&f, &g, Domain::ZZ, &mut budget, &mut Unbounded).unwrap();
    assert!(divide(&f, &g, Domain::ZZ, &mut budget, &mut Unbounded).is_err());
}

#[test]
fn every_verifier_checkpoint_can_cancel_without_an_accepted_receipt() {
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
