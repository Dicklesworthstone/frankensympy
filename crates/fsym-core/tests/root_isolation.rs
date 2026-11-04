//! Analytic completeness checks and adversarial cover/root mutations.
//! Fixtures are exact and constructed without a candidate search algorithm.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{
    BigInt, BigRational, Expr, MAX_ROOT_ISOLATION_DEPTH, MAX_ROOT_ISOLATION_NODES,
    MAX_ROOT_ISOLATION_ROOTS, RealBall, RootBoxError, RootBoxWitness, RootIsolationError,
    RootIsolationStep, RootIsolationWitness, RootIsolationWorkspace, Symbol, SymbolIdentity,
    certify_root_isolation, verify_root_isolation,
};
use std::{error::Error, sync::Arc};

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}
fn ball(center: BigRational, radius: BigRational) -> RealBall {
    RealBall::new(center, radius).unwrap()
}
fn bounds(lo: i64, hi: i64) -> RealBall {
    ball(q(lo + hi, 2), q(hi - lo, 2))
}
fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}
fn local(center: Vec<BigRational>, radius: BigRational) -> RootBoxWitness {
    let n = center.len();
    RootBoxWitness {
        domain: center
            .iter()
            .map(|value| ball(value.clone(), radius.clone()))
            .collect(),
        center,
        preconditioner: (0..n)
            .map(|i| (0..n).map(|j| q(i64::from(i == j), 1)).collect())
            .collect(),
        precision_digits: 24,
    }
}

// Fixed-depth dyadic fixture geometry, independent of any residual evaluator.
// Known root neighborhoods determine leaf labels; every remaining region
// must subsequently pass the production checker's exclusion obligations.
fn cover(domain: &[RealBall], roots: &[RootBoxWitness], depth: usize) -> Vec<RootIsolationStep> {
    fn visit(
        cell: &[RealBall],
        roots: &[RootBoxWitness],
        depth: usize,
        axis: usize,
        output: &mut Vec<RootIsolationStep>,
    ) {
        if depth == 0 {
            let root = roots.iter().position(|root| {
                root.domain
                    .iter()
                    .zip(cell)
                    .all(|(outer, inner)| outer.contains_ball(inner))
            });
            output.push(match root {
                Some(index) => RootIsolationStep::Root { index },
                None => RootIsolationStep::Exclude,
            });
        } else {
            output.push(RootIsolationStep::Split { axis });
            let middle = cell[axis].midpoint().clone();
            let radius = cell[axis].radius() / q(2, 1);
            let mut left = cell.to_vec();
            let mut right = cell.to_vec();
            left[axis] = ball(&middle - &radius, radius.clone());
            right[axis] = ball(middle + &radius, radius);
            visit(&left, roots, depth - 1, (axis + 1) % cell.len(), output);
            visit(&right, roots, depth - 1, (axis + 1) % cell.len(), output);
        }
    }
    let mut result = Vec::new();
    visit(domain, roots, depth, 0, &mut result);
    result
}

fn two_roots() -> (Vec<Expr>, Vec<Symbol>, Vec<RealBall>, RootIsolationWitness) {
    let variables = vec![Symbol::new("x")];
    let residuals = vec![power(Expr::Sym(variables[0].clone()), 2) - Expr::from_i64(1)];
    let domain = vec![ball(q(0, 1), q(3, 2))];
    let roots: Vec<_> = [-1, 1]
        .into_iter()
        .map(|sign| {
            let mut root = local(vec![q(sign, 1)], q(1, 2));
            root.preconditioner[0][0] = q(sign, 2);
            root
        })
        .collect();
    let partition = cover(&domain, &roots, 2);
    (
        residuals,
        variables,
        domain,
        RootIsolationWitness {
            roots,
            partition,
            precision_digits: 24,
        },
    )
}

#[test]
fn complete_cover_proves_both_irrational_roots_and_binds_the_original_system() {
    let variables = [Symbol::new("x")];
    let residuals = [power(Expr::Sym(variables[0].clone()), 2) - Expr::from_i64(2)];
    let domain = [bounds(-2, 2)];
    let roots: Vec<_> = [-1, 1]
        .into_iter()
        .map(|sign| {
            let mut root = local(vec![q(sign * 11, 8)], q(1, 8));
            root.preconditioner[0][0] = q(sign * 4, 11);
            root
        })
        .collect();
    let witness = RootIsolationWitness {
        partition: cover(&domain, &roots, 4),
        roots,
        precision_digits: 24,
    };
    let accepted = certify_root_isolation(&residuals, &variables, &domain, &witness).unwrap();
    assert_eq!(accepted.root_count(), 2);
    assert!(!accepted.is_root_free());
    assert_eq!(accepted.residuals(), residuals);
    assert_eq!(accepted.variables(), variables);
    assert_eq!(accepted.domain(), domain);
    assert_eq!(accepted.precision_digits(), 24);
    assert_eq!(accepted.witness(), &witness);
    assert_eq!(accepted.leaf_count(), 16);
    assert_eq!(accepted.excluded_leaf_count(), 14);
    for (index, root) in accepted.roots().iter().enumerate() {
        assert_eq!(root.residuals(), residuals);
        assert_eq!(root.variables(), variables);
        assert_eq!(root.witness(), &witness.roots[index]);
        let interval = &root.root_enclosure()[0];
        if index == 0 {
            assert!(&interval.lower() * &interval.lower() > q(2, 1));
            assert!(&interval.upper() * &interval.upper() < q(2, 1));
        } else {
            assert!(&interval.lower() * &interval.lower() < q(2, 1));
            assert!(&interval.upper() * &interval.upper() > q(2, 1));
        }
    }
}

#[test]
fn all_four_roots_of_a_two_dimensional_system_are_counted_once() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let residuals: Vec<_> = variables
        .iter()
        .map(|variable| power(Expr::Sym(variable.clone()), 2) - Expr::from_i64(1))
        .collect();
    let domain = vec![ball(q(0, 1), q(3, 2)); 2];
    let mut roots = Vec::new();
    for x in [-1, 1] {
        for y in [-1, 1] {
            let mut root = local(vec![q(x, 1), q(y, 1)], q(1, 2));
            root.preconditioner[0][0] = q(x, 2);
            root.preconditioner[1][1] = q(y, 2);
            roots.push(root);
        }
    }
    let witness = RootIsolationWitness {
        partition: cover(&domain, &roots, 4),
        roots,
        precision_digits: 24,
    };
    let accepted = certify_root_isolation(&residuals, &variables, &domain, &witness).unwrap();
    assert_eq!(accepted.root_count(), 4);
    assert_eq!(accepted.leaf_count(), 16);
    assert_eq!(accepted.excluded_leaf_count(), 12);
    for (root, exact) in accepted
        .roots()
        .iter()
        .zip([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    {
        for (coordinate, value) in root.root_enclosure().iter().zip(exact) {
            assert!(coordinate.contains(&q(value, 1)));
        }
    }
}

#[test]
fn a_root_on_internal_split_faces_is_neither_lost_nor_duplicated() {
    let variables = [Symbol::new("x")];
    let residuals = [Expr::Sym(variables[0].clone())];
    let domain = [bounds(-1, 1)];
    let roots = vec![local(vec![q(0, 1)], q(1, 4))];
    let witness = RootIsolationWitness {
        partition: cover(&domain, &roots, 3),
        roots,
        precision_digits: 24,
    };
    let accepted = certify_root_isolation(&residuals, &variables, &domain, &witness).unwrap();
    assert_eq!(accepted.root_count(), 1);
    assert_eq!(accepted.leaf_count(), 8);
    assert_eq!(accepted.excluded_leaf_count(), 6);
    assert_eq!(
        accepted.roots()[0].root_enclosure(),
        [RealBall::from_i64(0)]
    );
    assert_eq!(
        witness
            .partition
            .iter()
            .filter(|step| matches!(step, RootIsolationStep::Root { index: 0 }))
            .count(),
        2
    );
}

#[test]
fn an_empty_root_set_requires_and_receives_a_complete_exclusion_proof() {
    let variables = [Symbol::new("x")];
    let residuals = [power(Expr::Sym(variables[0].clone()), 2) + Expr::from_i64(1)];
    let domain = [bounds(-10, 10)];
    let witness = RootIsolationWitness {
        roots: vec![],
        partition: vec![RootIsolationStep::Exclude],
        precision_digits: 24,
    };
    let accepted = certify_root_isolation(&residuals, &variables, &domain, &witness).unwrap();
    assert!(accepted.is_root_free());
    assert_eq!(accepted.root_count(), 0);
    assert_eq!(accepted.leaf_count(), 1);
    assert_eq!(accepted.excluded_leaf_count(), 1);

    let (residuals, variables, domain, _) = two_roots();
    assert!(matches!(
        certify_root_isolation(&residuals, &variables, &domain, &witness),
        Err(RootIsolationError::Incomplete(_))
    ));
}

#[test]
fn omitting_regions_or_falsifying_exclusion_never_proves_completeness() {
    let (residuals, variables, domain, original) = two_roots();
    let accepted = certify_root_isolation(&residuals, &variables, &domain, &original).unwrap();
    assert_eq!(accepted.root_count(), 2);
    let mut missing = original.clone();
    missing.partition.pop();
    assert!(matches!(
        certify_root_isolation(&residuals, &variables, &domain, &missing),
        Err(RootIsolationError::InvalidPartition(_))
    ));
    let mut trailing = original.clone();
    trailing.partition.push(RootIsolationStep::Exclude);
    assert!(matches!(
        certify_root_isolation(&residuals, &variables, &domain, &trailing),
        Err(RootIsolationError::InvalidPartition(_))
    ));
    let mut false_exclusion = original.clone();
    for step in &mut false_exclusion.partition {
        if *step == (RootIsolationStep::Root { index: 1 }) {
            *step = RootIsolationStep::Exclude;
        }
    }
    assert!(matches!(
        certify_root_isolation(&residuals, &variables, &domain, &false_exclusion),
        Err(RootIsolationError::Incomplete(_))
    ));
    false_exclusion.roots.pop();
    assert!(certify_root_isolation(&residuals, &variables, &domain, &false_exclusion).is_err());
}

#[test]
fn local_root_acceptance_does_not_authorize_a_larger_region_or_wrong_source() {
    let (residuals, variables, domain, original) = two_roots();
    let mut oversized_leaf = original.clone();
    oversized_leaf.partition = vec![RootIsolationStep::Root { index: 0 }];
    assert!(matches!(
        certify_root_isolation(&residuals, &variables, &domain, &oversized_leaf),
        Err(RootIsolationError::InvalidPartition(_))
    ));
    let mut wrong_root = original.clone();
    for step in &mut wrong_root.partition {
        if *step == (RootIsolationStep::Root { index: 1 }) {
            *step = RootIsolationStep::Root { index: 0 };
        }
    }
    assert!(matches!(
        certify_root_isolation(&residuals, &variables, &domain, &wrong_root),
        Err(RootIsolationError::InvalidPartition(_))
    ));
    let changed_source = [power(Expr::Sym(variables[0].clone()), 2) - Expr::from_i64(4)];
    assert!(certify_root_isolation(&changed_source, &variables, &domain, &original).is_err());
    let changed_variables = [Symbol::with_identity(
        "x",
        SymbolIdentity::from_assumptions_digest([7; 32]),
    )];
    assert!(certify_root_isolation(&residuals, &changed_variables, &domain, &original).is_err());
    let mut wrong_inverse = original.clone();
    wrong_inverse.roots[0].preconditioner[0][0] = q(1, 2);
    assert!(matches!(
        certify_root_isolation(&residuals, &variables, &domain, &wrong_inverse),
        Err(RootIsolationError::Verification(
            RootBoxError::NotContractive
        ))
    ));
}

#[test]
fn overlapping_duplicate_foreign_and_mismatched_root_boxes_refuse() {
    let (residuals, variables, domain, original) = two_roots();
    let mut duplicate = original.clone();
    duplicate.roots.push(duplicate.roots[0].clone());
    assert_eq!(
        certify_root_isolation(&residuals, &variables, &domain, &duplicate),
        Err(RootIsolationError::OverlappingRoots {
            first: 0,
            second: 2
        })
    );
    let mut outside = original.clone();
    outside.roots[0].domain[0] = bounds(-2, 0);
    assert!(matches!(
        certify_root_isolation(&residuals, &variables, &domain, &outside),
        Err(RootIsolationError::InvalidPartition(_))
    ));
    let mut precision = original.clone();
    precision.roots[0].precision_digits = 32;
    assert_eq!(
        certify_root_isolation(&residuals, &variables, &domain, &precision),
        Err(RootIsolationError::Verification(
            RootBoxError::InvalidPrecision
        ))
    );
    let mut index = original;
    index.partition[2] = RootIsolationStep::Root { index: usize::MAX };
    assert!(matches!(
        certify_root_isolation(&residuals, &variables, &domain, &index),
        Err(RootIsolationError::InvalidPartition(_))
    ));
}

#[test]
fn nonzero_residual_rows_do_not_hide_other_original_domain_obligations() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let x = Expr::Sym(variables[0].clone());
    let hidden = Expr::Mul(vec![Expr::from_i64(0), power(x.clone(), -1)]);
    let witness = RootIsolationWitness {
        roots: vec![],
        partition: vec![RootIsolationStep::Exclude],
        precision_digits: 24,
    };
    let domain = [bounds(-1, 1), bounds(-1, 1)];
    for invalid in [hidden, power(power(x, -1), 0)] {
        let residuals = [Expr::from_i64(1), invalid];
        assert!(matches!(
            certify_root_isolation(&residuals, &variables, &domain, &witness),
            Err(RootIsolationError::Verification(_))
        ));
    }
}

#[test]
fn outer_boundary_and_singular_roots_never_receive_a_complete_receipt() {
    let variables = [Symbol::new("x")];
    let x = Expr::Sym(variables[0].clone());
    let boundary = RootIsolationWitness {
        roots: vec![local(vec![q(1, 2)], q(1, 2))],
        partition: vec![RootIsolationStep::Root { index: 0 }],
        precision_digits: 24,
    };
    assert_eq!(
        certify_root_isolation(
            std::slice::from_ref(&x),
            &variables,
            &[bounds(0, 1)],
            &boundary
        ),
        Err(RootIsolationError::Verification(
            RootBoxError::NoInteriorInclusion { coordinate: 0 }
        ))
    );
    let singular = RootIsolationWitness {
        roots: vec![local(vec![q(0, 1)], q(1, 4))],
        partition: vec![RootIsolationStep::Root { index: 0 }],
        precision_digits: 24,
    };
    assert_eq!(
        certify_root_isolation(
            &[power(x, 2)],
            &variables,
            &[ball(q(0, 1), q(1, 4))],
            &singular
        ),
        Err(RootIsolationError::Verification(
            RootBoxError::NotContractive
        ))
    );
}

#[test]
fn partition_shape_counts_depth_dimensions_and_precision_are_bounded() {
    let variables = [Symbol::new("x")];
    let residuals = [Expr::from_i64(1)];
    let domain = [bounds(-1, 1)];
    let mut witness = RootIsolationWitness {
        roots: vec![],
        partition: vec![],
        precision_digits: 24,
    };
    assert!(certify_root_isolation(&residuals, &variables, &domain, &witness).is_err());
    witness.partition = vec![RootIsolationStep::Exclude; MAX_ROOT_ISOLATION_NODES + 1];
    assert!(matches!(
        certify_root_isolation(&residuals, &variables, &domain, &witness),
        Err(RootIsolationError::InvalidPartition(_))
    ));
    witness.partition = vec![RootIsolationStep::Split { axis: 0 }; MAX_ROOT_ISOLATION_DEPTH + 1];
    witness.partition.extend(vec![
        RootIsolationStep::Exclude;
        MAX_ROOT_ISOLATION_DEPTH + 2
    ]);
    assert!(matches!(
        certify_root_isolation(&residuals, &variables, &domain, &witness),
        Err(RootIsolationError::InvalidPartition(_))
    ));
    witness.partition = vec![RootIsolationStep::Split { axis: usize::MAX }];
    assert!(matches!(
        certify_root_isolation(&residuals, &variables, &domain, &witness),
        Err(RootIsolationError::InvalidPartition(_))
    ));
    witness.partition = vec![RootIsolationStep::Exclude];
    witness.roots = vec![local(vec![q(0, 1)], q(1, 4)); MAX_ROOT_ISOLATION_ROOTS + 1];
    assert!(matches!(
        certify_root_isolation(&residuals, &variables, &domain, &witness),
        Err(RootIsolationError::InvalidPartition(_))
    ));
    witness.roots.clear();
    for precision in [7, 129, u32::MAX] {
        witness.precision_digits = precision;
        assert_eq!(
            certify_root_isolation(&residuals, &variables, &domain, &witness),
            Err(RootIsolationError::Verification(
                RootBoxError::InvalidPrecision
            ))
        );
    }
    witness.precision_digits = 24;
    assert_eq!(
        certify_root_isolation(&residuals, &variables, &[bounds(0, 0)], &witness),
        Err(RootIsolationError::Verification(
            RootBoxError::DegenerateBox { coordinate: 0 }
        ))
    );
    assert_eq!(
        certify_root_isolation(
            &[Expr::from_i64(1), Expr::from_i64(2)],
            &variables,
            &domain,
            &witness
        ),
        Err(RootIsolationError::Verification(
            RootBoxError::DimensionMismatch
        ))
    );
}

#[test]
fn exact_sub_binary64_root_coordinates_and_empty_decisions_are_preserved() {
    let variables = [Symbol::new("x")];
    let tiny = BigRational::new(BigInt::from(1), BigInt::from(1) << 1200u32);
    let mut root = local(vec![tiny.clone()], &tiny / q(2, 1));
    let domain = root.domain.clone();
    let source = [Expr::Sym(variables[0].clone()) - Expr::Rational(tiny.clone())];
    let mut witness = RootIsolationWitness {
        roots: vec![root.clone()],
        partition: vec![RootIsolationStep::Root { index: 0 }],
        precision_digits: 24,
    };
    let accepted = certify_root_isolation(&source, &variables, &domain, &witness).unwrap();
    assert_eq!(accepted.root_count(), 1);
    assert_eq!(
        accepted.roots()[0].root_enclosure(),
        [RealBall::exact(tiny.clone())]
    );
    root.center[0] = q(0, 1);
    witness.roots[0] = root;
    assert!(certify_root_isolation(&source, &variables, &domain, &witness).is_err());
    witness.roots.clear();
    witness.partition = vec![RootIsolationStep::Exclude];
    let accepted =
        certify_root_isolation(&[Expr::Rational(tiny)], &variables, &domain, &witness).unwrap();
    assert!(accepted.is_root_free());
}

#[test]
fn maximum_dimension_root_count_uses_the_full_square_system() {
    let variables: Vec<_> = (0..16).map(|i| Symbol::new(format!("x{i}"))).collect();
    let residuals: Vec<_> = variables.iter().cloned().map(Expr::Sym).collect();
    let domain = vec![bounds(-1, 1); variables.len()];
    let witness = RootIsolationWitness {
        roots: vec![local(vec![q(0, 1); variables.len()], q(1, 1))],
        partition: vec![RootIsolationStep::Root { index: 0 }],
        precision_digits: 24,
    };
    let accepted = certify_root_isolation(&residuals, &variables, &domain, &witness).unwrap();
    assert_eq!(accepted.root_count(), 1);
    assert_eq!(
        accepted.roots()[0].root_enclosure(),
        vec![RealBall::from_i64(0); 16]
    );
}

#[test]
fn reusable_workspace_keeps_scratch_and_failed_local_work_cumulative() {
    let variables = [Symbol::new("x")];
    let residuals = [Expr::Sym(variables[0].clone())];
    let domain = [bounds(-1, 1)];
    let root = local(vec![q(0, 1)], q(1, 4));
    let mut budget = Budget::new(BudgetLimits::uniform(100_000_000, 10_000));
    {
        let mut workspace =
            RootIsolationWorkspace::new(&residuals, &variables, &domain, 24, &mut budget).unwrap();
        for _ in 0..8 {
            let values = workspace.enclose_cell(&domain).unwrap();
            assert_eq!(values.residual_values(), domain);
            assert_eq!(values.jacobian(), [vec![RealBall::from_i64(1)]]);
            workspace.verify_local(&root).unwrap();
        }
        let point = workspace.enclose_cell(&[RealBall::from_i64(0)]).unwrap();
        assert_eq!(point.residual_values(), [RealBall::from_i64(0)]);
        let mut rejected = root.clone();
        rejected.preconditioner[0][0] = q(0, 1);
        assert_eq!(
            workspace.verify_local(&rejected),
            Err(RootIsolationError::Verification(
                RootBoxError::NotContractive
            ))
        );
        workspace.verify_local(&root).unwrap();
        let (left, right) = workspace.bisect(&domain, 0).unwrap();
        assert_eq!(left[0].upper(), right[0].lower());
        assert_eq!(left[0].lower(), domain[0].lower());
        assert_eq!(right[0].upper(), domain[0].upper());
        let expanded = workspace.expand_cell(&left).unwrap();
        assert_eq!(expanded[0].lower(), q(-1, 1));
        assert_eq!(expanded[0].upper(), q(1, 2));
        assert!(workspace.enclose_cell(&[bounds(-2, 1)]).is_err());
        assert!(
            workspace
                .verify_local(&local(vec![q(2, 1)], q(1, 4)))
                .is_err()
        );
        assert_eq!(workspace.add(&q(1, 3), &q(1, 6)).unwrap(), q(1, 2));
        assert_eq!(workspace.subtract(&q(1, 3), &q(1, 6)).unwrap(), q(1, 6));
        assert_eq!(workspace.multiply(&q(2, 3), &q(3, 5)).unwrap(), q(2, 5));
        assert_eq!(workspace.divide(&q(2, 3), &q(4, 5)).unwrap(), q(5, 6));
        assert!(workspace.divide(&q(1, 1), &q(0, 1)).is_err());
        workspace.account(0, 0, 0).unwrap();
    }
    assert!(budget.remaining(Dimension::MemoryBytes) > 20_000_000);
    assert_eq!(budget.verifier_remaining(), 10_000);
}

#[test]
fn a_later_phase_cannot_refund_consumed_caller_resources() {
    let (residuals, variables, domain, witness) = two_roots();
    let mut budget = Budget::new(BudgetLimits::uniform(100_000_000, 10_000));
    verify_root_isolation(&residuals, &variables, &domain, &witness, &mut budget).unwrap();
    let consumed = budget.snapshot();
    assert!(matches!(
        verify_root_isolation(&residuals, &variables, &domain, &witness, &mut budget),
        Err(RootIsolationError::Verification(RootBoxError::Meter(
            MeterError::Budget(_)
        )))
    ));
    assert_eq!(budget.snapshot(), consumed);
    assert_eq!(budget.verifier_remaining(), 10_000);
    let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(verify_root_isolation(&residuals, &variables, &domain, &witness, &mut empty).is_err());
}

#[derive(Default)]
struct Cancel {
    visits: usize,
    stop: Option<usize>,
}
impl BudgetMeter for Cancel {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        let here = self.visits;
        self.visits += 1;
        if self.stop == Some(here) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_at_every_checkpoint_prevents_complete_receipt_publication() {
    let (residuals, variables, domain, witness) = two_roots();
    let mut count = Cancel::default();
    verify_root_isolation(&residuals, &variables, &domain, &witness, &mut count).unwrap();
    assert!(count.visits > 20);
    for stop in 0..count.visits {
        let mut cancel = Cancel {
            visits: 0,
            stop: Some(stop),
        };
        let error = verify_root_isolation(&residuals, &variables, &domain, &witness, &mut cancel)
            .unwrap_err();
        assert_eq!(
            error,
            RootIsolationError::Verification(RootBoxError::Meter(MeterError::Cancelled)),
            "safe point {stop}"
        );
        assert!(error.source().unwrap().is::<RootBoxError>());
    }
    let mut meter = Unbounded;
    assert_eq!(
        verify_root_isolation(&residuals, &variables, &domain, &witness, &mut meter)
            .unwrap()
            .root_count(),
        2
    );
}
