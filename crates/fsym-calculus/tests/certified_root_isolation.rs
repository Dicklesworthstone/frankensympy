use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::root_isolation::{RootIsolationOptions, propose_root_isolation};
use fsym_core::{
    BigRational, Expr, RealBall, RootBoxError, RootIsolationError, RootIsolationReceipt,
    RootIsolationStep, RootIsolationWitness, Symbol, verify_root_isolation,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn interval(lower: i64, upper: i64) -> RealBall {
    RealBall::new(q(lower + upper, 2), q(upper - lower, 2)).unwrap()
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn shifted_square(variable: &Symbol, constant: i64) -> Expr {
    Expr::Add(vec![
        power(Expr::Sym(variable.clone()), 2),
        Expr::from_i64(constant),
    ])
}

fn options() -> RootIsolationOptions {
    RootIsolationOptions {
        precision_digits: 16,
        max_leaves: 512,
    }
}

fn checked(source: &[Expr], variables: &[Symbol], domain: &[RealBall]) -> RootIsolationReceipt {
    let mut budget = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    let witness =
        propose_root_isolation(source, variables, domain, options(), &mut budget).unwrap();
    let receipt = verify_root_isolation(source, variables, domain, &witness, &mut budget).unwrap();
    assert_eq!(receipt.residuals(), source);
    assert_eq!(receipt.variables(), variables);
    assert_eq!(receipt.domain(), domain);
    assert_eq!(receipt.precision_digits(), 16);
    assert_eq!(receipt.root_count(), witness.roots.len());
    assert_eq!(receipt.roots().len(), witness.roots.len());
    assert_eq!(receipt.witness(), &witness);
    let leaves = witness
        .partition
        .iter()
        .filter(|step| !matches!(step, RootIsolationStep::Split { .. }))
        .count();
    let excluded = witness
        .partition
        .iter()
        .filter(|step| matches!(step, RootIsolationStep::Exclude))
        .count();
    assert_eq!(receipt.leaf_count(), leaves);
    assert_eq!(receipt.excluded_leaf_count(), excluded);
    assert_eq!(witness.partition.len(), 2 * leaves - 1);
    assert_eq!(receipt.is_root_free(), witness.roots.is_empty());
    for root in receipt.roots() {
        assert_eq!(root.residuals(), source);
        assert_eq!(root.variables(), variables);
        assert!(root.contraction_bound() < &q(1, 1));
        assert!(
            domain
                .iter()
                .zip(&root.witness().domain)
                .all(|(outer, inner)| outer.contains_ball(inner))
        );
    }
    receipt
}

#[test]
fn finds_both_irrational_roots_without_initial_guesses() {
    let x = Symbol::new("x");
    let result = checked(&[shifted_square(&x, -2)], &[x], &[interval(-2, 2)]);
    assert_eq!(result.root_count(), 2);
    assert!(result.excluded_leaf_count() > 0);
    let first = &result.roots()[0].root_enclosure()[0];
    let second = &result.roots()[1].root_enclosure()[0];
    assert!(first.upper() < q(0, 1));
    assert!(second.lower() > q(0, 1));
    // Independently known roots +/-sqrt(2), bounded using exact squares.
    assert!(&first.lower() * &first.lower() >= q(2, 1));
    assert!(&first.upper() * &first.upper() <= q(2, 1));
    assert!(&second.lower() * &second.lower() <= q(2, 1));
    assert!(&second.upper() * &second.upper() >= q(2, 1));
}

#[test]
fn nonlinear_two_dimensional_system_has_all_four_distinct_roots() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let px = shifted_square(&x, -1);
    let py = shifted_square(&y, -1);
    let result = checked(
        &[px.clone(), Expr::Add(vec![px, py])],
        &[x, y],
        &[interval(-2, 2), interval(-2, 2)],
    );
    assert_eq!(result.root_count(), 4);
    for a in [-1, 1] {
        for b in [-1, 1] {
            assert_eq!(
                result
                    .roots()
                    .iter()
                    .filter(|root| {
                        root.root_enclosure()[0].contains(&q(a, 1))
                            && root.root_enclosure()[1].contains(&q(b, 1))
                    })
                    .count(),
                1
            );
        }
    }
}

#[test]
fn transcendental_cover_counts_every_root_in_the_requested_interval() {
    let x = Symbol::new("x");
    let source = Expr::Function("sin".into(), vec![Expr::Sym(x.clone())]);
    let result = checked(&[source], &[x], &[interval(-4, 4)]);
    assert_eq!(result.root_count(), 3);
    let pi = RealBall::pi(32).unwrap();
    for expected in [pi.neg(), RealBall::from_i64(0), pi] {
        assert_eq!(
            result
                .roots()
                .iter()
                .filter(|root| root.root_enclosure()[0].contains_ball(&expected))
                .count(),
            1
        );
    }
}

#[test]
fn root_boxes_cross_internal_split_faces_without_duplicate_roots() {
    let x = Symbol::new("x");
    let source = Expr::Add(vec![
        power(Expr::Sym(x.clone()), 3),
        Expr::Mul(vec![Expr::from_i64(-1), Expr::Sym(x.clone())]),
    ]);
    let result = checked(&[source], &[x], &[interval(-2, 2)]);
    assert_eq!(result.root_count(), 3);
    for expected in [-1, 0, 1] {
        assert_eq!(
            result
                .roots()
                .iter()
                .filter(|root| root.root_enclosure()[0].contains(&q(expected, 1)))
                .count(),
            1
        );
    }
    let index = result
        .roots()
        .iter()
        .position(|root| root.root_enclosure()[0].contains(&q(0, 1)))
        .unwrap();
    let zero_box = &result.roots()[index].witness().domain[0];
    assert!(zero_box.lower() < q(0, 1) && zero_box.upper() > q(0, 1));
    assert!(
        result
            .witness()
            .partition
            .iter()
            .filter(|step| **step == RootIsolationStep::Root { index })
            .count()
            >= 2
    );
}

#[test]
fn a_complete_exclusion_cover_proves_the_domain_root_free() {
    let x = Symbol::new("x");
    let result = checked(&[shifted_square(&x, 1)], &[x], &[interval(-2, 2)]);
    assert!(result.is_root_free());
    assert_eq!(result.root_count(), 0);
    assert_eq!(result.leaf_count(), 1);
    assert_eq!(result.excluded_leaf_count(), 1);
}

#[test]
fn maximum_dimension_linear_system_is_certified_in_one_leaf() {
    let variables: Vec<_> = (0..16).map(|i| Symbol::new(format!("x{i}"))).collect();
    let residuals: Vec<_> = variables.iter().cloned().map(Expr::Sym).collect();
    let result = checked(&residuals, &variables, &vec![interval(-1, 1); 16]);
    assert_eq!(result.root_count(), 1);
    assert_eq!(result.leaf_count(), 1);
    assert!(
        result.roots()[0]
            .root_enclosure()
            .iter()
            .all(|ball| ball == &RealBall::from_i64(0))
    );
}

#[test]
fn original_source_interval_dependency_can_be_resolved_by_subdivision() {
    let x = Symbol::new("x");
    let sx = Expr::Sym(x.clone());
    // Held x*x on [-1,1] overestimates [-1,1]. Only subdivisions establish
    // the original denominator's positivity; no algebraic cancellation is used.
    let denominator = Expr::Add(vec![Expr::from_i64(1), Expr::Mul(vec![sx.clone(), sx])]);
    let result = checked(&[power(denominator, -1)], &[x], &[interval(-1, 1)]);
    assert!(result.is_root_free());
    assert_eq!(result.leaf_count(), 2);
    assert_eq!(result.excluded_leaf_count(), 2);
}

#[test]
fn an_excluded_first_residual_cannot_hide_an_original_pole_in_another() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let inverse = power(Expr::Add(vec![Expr::Sym(y.clone()), Expr::from_i64(1)]), -1);
    for hidden in [
        Expr::Mul(vec![Expr::from_i64(0), inverse.clone()]),
        power(inverse, 0),
    ] {
        assert!(matches!(
            propose_root_isolation(
                &[Expr::from_i64(1), hidden],
                &[x.clone(), y.clone()],
                &[interval(-1, 1), interval(-1, 1)],
                RootIsolationOptions {
                    max_leaves: 8,
                    ..options()
                },
                &mut Unbounded,
            ),
            Err(RootIsolationError::Verification(
                RootBoxError::Domain(_) | RootBoxError::Ball(_)
            ))
        ));
    }
}

#[test]
fn singular_or_outer_boundary_roots_never_publish_partial_completeness() {
    let x = Symbol::new("x");
    let sx = Expr::Sym(x.clone());
    for (source, domain) in [
        (power(sx.clone(), 2), interval(-1, 1)),
        (sx.clone(), interval(0, 1)),
        (
            Expr::Mul(vec![
                power(sx.clone(), 2),
                Expr::Add(vec![sx, Expr::from_i64(1)]),
            ]),
            interval(-2, 2),
        ),
    ] {
        assert!(matches!(
            propose_root_isolation(
                &[source],
                std::slice::from_ref(&x),
                &[domain],
                RootIsolationOptions {
                    max_leaves: 16,
                    ..options()
                },
                &mut Unbounded,
            ),
            Err(RootIsolationError::Incomplete(_))
        ));
    }
}

#[test]
fn independent_replay_rejects_missing_regions_and_wrong_source_or_root_boxes() {
    let x = Symbol::new("x");
    let source = [shifted_square(&x, -2)];
    let variables = [x.clone()];
    let domain = [interval(-2, 2)];
    let witness =
        propose_root_isolation(&source, &variables, &domain, options(), &mut Unbounded).unwrap();
    let verify = |source: &[Expr], witness: &RootIsolationWitness| {
        verify_root_isolation(source, &variables, &domain, witness, &mut Unbounded)
    };
    assert!(verify(&source, &witness).is_ok());
    let mut missing = witness.clone();
    missing.partition.pop();
    assert!(matches!(
        verify(&source, &missing),
        Err(RootIsolationError::InvalidPartition(_))
    ));
    let mut forged_exclusion = witness.clone();
    let retained = forged_exclusion
        .partition
        .iter_mut()
        .find(|step| matches!(step, RootIsolationStep::Root { .. }))
        .unwrap();
    *retained = RootIsolationStep::Exclude;
    assert!(verify(&source, &forged_exclusion).is_err());
    let mut duplicate = witness.clone();
    duplicate.roots[1] = duplicate.roots[0].clone();
    assert!(matches!(
        verify(&source, &duplicate),
        Err(RootIsolationError::OverlappingRoots { .. })
    ));
    let mut outside = witness.clone();
    outside.roots[0].center[0] = q(3, 1);
    assert!(verify(&source, &outside).is_err());
    assert!(verify(&[shifted_square(&x, 2)], &witness).is_err());
    assert!(
        verify_root_isolation(
            &source,
            &[Symbol::new("different")],
            &domain,
            &witness,
            &mut Unbounded,
        )
        .is_err()
    );
}

#[test]
fn deterministic_traversal_and_exact_inversion_reproduce_identical_witnesses() {
    let x = Symbol::new("x");
    let source = [shifted_square(&x, -2)];
    let propose = || {
        propose_root_isolation(
            &source,
            std::slice::from_ref(&x),
            &[interval(-2, 2)],
            options(),
            &mut Unbounded,
        )
        .unwrap()
    };
    assert_eq!(propose(), propose());
}

#[test]
fn invalid_controls_shapes_and_unsupported_sources_do_not_start_unbounded_search() {
    let x = Symbol::new("x");
    let source = [shifted_square(&x, -2)];
    for max_leaves in [0, 1025, usize::MAX] {
        assert!(matches!(
            propose_root_isolation(
                &source,
                std::slice::from_ref(&x),
                &[interval(-2, 2)],
                RootIsolationOptions {
                    max_leaves,
                    ..options()
                },
                &mut Unbounded,
            ),
            Err(RootIsolationError::Verification(
                RootBoxError::ResourceLimit(_)
            ))
        ));
    }
    for precision_digits in [7, 129, u32::MAX] {
        assert!(matches!(
            propose_root_isolation(
                &source,
                std::slice::from_ref(&x),
                &[interval(-2, 2)],
                RootIsolationOptions {
                    precision_digits,
                    ..options()
                },
                &mut Unbounded,
            ),
            Err(RootIsolationError::Verification(
                RootBoxError::InvalidPrecision
            ))
        ));
    }
    assert!(matches!(
        propose_root_isolation(
            &source,
            std::slice::from_ref(&x),
            &[interval(-2, 2)],
            RootIsolationOptions {
                max_leaves: 1,
                ..options()
            },
            &mut Unbounded,
        ),
        Err(RootIsolationError::Incomplete(_))
    ));
    for unsupported in [
        Expr::Function("gamma".into(), vec![Expr::Sym(x.clone())]),
        Expr::Sym(Symbol::new("unbound")),
    ] {
        assert!(matches!(
            propose_root_isolation(
                &[unsupported],
                std::slice::from_ref(&x),
                &[interval(1, 2)],
                options(),
                &mut Unbounded,
            ),
            Err(RootIsolationError::Verification(RootBoxError::Unsupported(
                _
            )))
        ));
    }
    assert!(matches!(
        propose_root_isolation(
            &source,
            std::slice::from_ref(&x),
            &[RealBall::from_i64(0)],
            options(),
            &mut Unbounded,
        ),
        Err(RootIsolationError::Verification(
            RootBoxError::DegenerateBox { .. }
        ))
    ));
    assert!(matches!(
        propose_root_isolation(
            &[source[0].clone(), source[0].clone()],
            &[x],
            &[interval(-2, 2)],
            options(),
            &mut Unbounded,
        ),
        Err(RootIsolationError::Verification(
            RootBoxError::DimensionMismatch
        ))
    ));
}

#[test]
fn generation_and_independent_verification_share_the_callers_cumulative_budget() {
    let x = Symbol::new("x");
    let source = [Expr::Sym(x.clone())];
    let variables = [x];
    let domain = [interval(-1, 1)];
    let limit = 100 * 1024 * 1024;
    let mut budget = Budget::new(BudgetLimits::uniform(limit, 0));
    let witness =
        propose_root_isolation(&source, &variables, &domain, options(), &mut budget).unwrap();
    let remaining = budget.remaining(Dimension::MemoryBytes);
    assert!(remaining < limit - 64 * 1024 * 1024);
    assert!(matches!(
        verify_root_isolation(&source, &variables, &domain, &witness, &mut budget),
        Err(RootIsolationError::Verification(RootBoxError::Meter(
            MeterError::Budget(_)
        )))
    ));
    assert_eq!(budget.remaining(Dimension::MemoryBytes), remaining);
}

#[test]
fn failed_search_keeps_its_consumption_before_the_next_call() {
    let x = Symbol::new("x");
    let source = [shifted_square(&x, -2)];
    let mut budget = Budget::new(BudgetLimits::uniform(100 * 1024 * 1024, 0));
    assert!(matches!(
        propose_root_isolation(
            &source,
            std::slice::from_ref(&x),
            &[interval(-2, 2)],
            RootIsolationOptions {
                max_leaves: 1,
                ..options()
            },
            &mut budget,
        ),
        Err(RootIsolationError::Incomplete(_))
    ));
    assert!(budget.remaining(Dimension::ComputeSteps) < 100 * 1024 * 1024);
    assert!(matches!(
        propose_root_isolation(
            &[Expr::Sym(x.clone())],
            &[x],
            &[interval(-1, 1)],
            options(),
            &mut budget,
        ),
        Err(RootIsolationError::Verification(RootBoxError::Meter(
            MeterError::Budget(_)
        )))
    ));
}

#[derive(Default)]
struct CheckpointMeter {
    checks: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for CheckpointMeter {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checks += 1;
        if self.cancel_at == Some(self.checks) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_at_entry_during_search_and_before_publication_has_no_witness() {
    let x = Symbol::new("x");
    let source = [shifted_square(&x, -2)];
    let propose = |meter: &mut CheckpointMeter| {
        propose_root_isolation(
            &source,
            std::slice::from_ref(&x),
            &[interval(-2, 2)],
            options(),
            meter,
        )
    };
    let mut completed = CheckpointMeter::default();
    assert!(propose(&mut completed).is_ok());
    for cancel_at in [1, completed.checks / 2, completed.checks] {
        let mut meter = CheckpointMeter {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert_eq!(
            propose(&mut meter),
            Err(RootIsolationError::Verification(RootBoxError::Meter(
                MeterError::Cancelled
            )))
        );
        assert_eq!(meter.checks, cancel_at);
    }
}
