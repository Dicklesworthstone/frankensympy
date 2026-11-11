//! Native facade coverage for complete parameter families and their sensitivities.

use frankensympy::certification::{MAX_CERTIFICATION_NAME_BYTES, MAX_CERTIFICATION_SOURCE_BYTES};
use frankensympy::parametric::{ParametricError, ParametricProblem};
use frankensympy::{CertificationError, Expr, RealBall, Symbol};
use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_calculus::parametric_cover::ParametricRootCoverOptions;
use fsym_core::{
    BigInt, BigRational, MAX_ROOT_BOX_VARIABLES, ParametricRootBoxWitness,
    ParametricRootCoverError, ParametricRootCoverStep, RootBoxError, SymbolIdentity,
    metered_enclose_parametric_cover_sensitivity, metered_enclose_parametric_root_sensitivity,
};

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
}

fn interval(lower: i64, upper: i64) -> RealBall {
    RealBall::new(q(lower + upper, 2), q(upper - lower, 2)).unwrap()
}

fn options() -> ParametricRootCoverOptions {
    ParametricRootCoverOptions {
        precision_digits: 16,
        max_leaves: 128,
    }
}

fn affine() -> ParametricProblem {
    ParametricProblem::parse_sources(&["x-p"], &["x"], &["p"], &mut Unbounded).unwrap()
}

fn affine_witness() -> ParametricRootBoxWitness {
    ParametricRootBoxWitness {
        unknown_domain: vec![interval(-2, 2)],
        parameter_domain: vec![interval(-1, 1)],
        center: vec![q(0, 1)],
        preconditioner: vec![vec![q(1, 1)]],
        precision_digits: 16,
    }
}

#[test]
fn affine_family_certifies_the_entire_parameter_interval_and_exact_sensitivity() {
    let problem = affine();
    let proposed = affine_witness();
    let mut budget = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    let accepted = problem.verify_root_box(&proposed, &mut budget).unwrap();
    assert_eq!(accepted.residuals(), problem.residuals());
    assert_eq!(accepted.unknowns(), problem.unknowns());
    assert_eq!(accepted.parameters(), problem.parameters());
    assert_eq!(accepted.witness(), &proposed);
    assert_eq!(accepted.root_enclosure(), [interval(-1, 1)]);
    assert_eq!(accepted.contraction_bound(), &q(0, 1));
    let derivative = metered_enclose_parametric_root_sensitivity(&accepted, &mut budget).unwrap();
    assert_eq!(
        derivative.solution_jacobian(),
        [vec![RealBall::from_i64(1)]]
    );
    assert_eq!(derivative.parameter_domain(), proposed.parameter_domain);
    assert_eq!(derivative.unknown_domain(), proposed.unknown_domain);
    assert_eq!(derivative.leaf_count(), 1);
    assert!(budget.remaining(Dimension::MemoryBytes) < 512 * 1024 * 1024 - 128 * 1024 * 1024);
}

#[test]
fn coupled_system_preserves_unknown_rows_and_parameter_columns() {
    // In declared unknown order [y,x] the exact inverse is [[3,-1],[-5,2]].
    // With parameter order [q,p], dy/d(q,p)=(-1,3), dx/d(q,p)=(2,-5).
    let problem = ParametricProblem::parse_sources(
        &["x+2*y-p", "3*x+5*y-q"],
        &["y", "x"],
        &["q", "p"],
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(problem.unknowns(), [Symbol::new("y"), Symbol::new("x")]);
    assert_eq!(problem.parameters(), [Symbol::new("q"), Symbol::new("p")]);
    let proposed = ParametricRootBoxWitness {
        unknown_domain: vec![interval(-10, 10), interval(-15, 15)],
        parameter_domain: vec![interval(-1, 1); 2],
        center: vec![q(0, 1); 2],
        preconditioner: vec![vec![q(3, 1), q(-1, 1)], vec![q(-5, 1), q(2, 1)]],
        precision_digits: 16,
    };
    let accepted = problem.verify_root_box(&proposed, &mut Unbounded).unwrap();
    assert_eq!(
        accepted.root_enclosure(),
        [interval(-4, 4), interval(-7, 7)]
    );
    let derivative =
        metered_enclose_parametric_root_sensitivity(&accepted, &mut Unbounded).unwrap();
    assert_eq!(
        derivative.solution_jacobian(),
        [
            vec![RealBall::from_i64(-1), RealBall::from_i64(3)],
            vec![RealBall::from_i64(2), RealBall::from_i64(-5)],
        ]
    );
    let mut transposed = proposed;
    transposed.preconditioner = vec![vec![q(3, 1), q(-5, 1)], vec![q(-1, 1), q(2, 1)]];
    assert_eq!(
        problem
            .verify_root_box(&transposed, &mut Unbounded)
            .unwrap_err(),
        ParametricError::Certification(CertificationError::Root(RootBoxError::NotContractive))
    );
}

#[test]
fn explicit_same_name_typed_coordinates_are_distinct_and_raw_sources_remain_owned() {
    let unknown = Symbol::with_identity("z", SymbolIdentity::from_assumptions_digest([1; 32]));
    let parameter = Symbol::with_identity("z", SymbolIdentity::from_assumptions_digest([2; 32]));
    let source = Expr::Sym(unknown.clone()) - Expr::Sym(parameter.clone());
    let problem = ParametricProblem::from_exprs(
        vec![source.clone()],
        vec![unknown.clone()],
        vec![parameter.clone()],
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(problem.residuals(), [source]);
    assert_eq!(problem.unknowns(), [unknown]);
    assert_eq!(problem.parameters(), [parameter]);
    let accepted = problem
        .verify_root_box(&affine_witness(), &mut Unbounded)
        .unwrap();
    assert_eq!(accepted.root_enclosure(), [interval(-1, 1)]);
    let derivative =
        metered_enclose_parametric_root_sensitivity(&accepted, &mut Unbounded).unwrap();
    assert_eq!(
        derivative.solution_jacobian(),
        [vec![RealBall::from_i64(1)]]
    );
}

#[test]
fn adaptive_cover_certifies_a_family_that_a_coarse_box_cannot_and_checks_all_leaves() {
    let problem =
        ParametricProblem::parse_sources(&["(1+p^2)*x-1"], &["x"], &["p"], &mut Unbounded).unwrap();
    let unknown_domain = [interval(0, 2)];
    let parameter_domain = [interval(0, 2)];
    let coarse = ParametricRootBoxWitness {
        unknown_domain: unknown_domain.to_vec(),
        parameter_domain: parameter_domain.to_vec(),
        center: vec![q(1, 1)],
        preconditioner: vec![vec![q(1, 2)]],
        precision_digits: 16,
    };
    assert_eq!(
        problem
            .verify_root_box(&coarse, &mut Unbounded)
            .unwrap_err(),
        ParametricError::Certification(CertificationError::Root(RootBoxError::NotContractive))
    );
    assert!(matches!(
        problem.propose_cover(
            &unknown_domain,
            &parameter_domain,
            ParametricRootCoverOptions {
                max_leaves: 1,
                ..options()
            },
            &mut Unbounded
        ),
        Err(ParametricError::Cover(
            ParametricRootCoverError::Incomplete(_)
        ))
    ));
    let mut budget = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    let witness = problem
        .propose_cover(&unknown_domain, &parameter_domain, options(), &mut budget)
        .unwrap();
    assert!(witness.leaves.len() > 1);
    let accepted = problem
        .verify_cover(&unknown_domain, &parameter_domain, &witness, &mut budget)
        .unwrap();
    assert_eq!(accepted.witness(), &witness);
    assert!(accepted.root_enclosure()[0].contains(&q(1, 5)));
    assert!(accepted.root_enclosure()[0].contains(&q(1, 1)));
    for leaf in accepted.leaves() {
        let parameters = &leaf.witness().parameter_domain[0];
        // The exact family x=1/(1+p^2) supplies independent endpoint checks.
        for p in [parameters.lower(), parameters.upper()] {
            assert!(leaf.root_enclosure()[0].contains(&(q(1, 1) / (q(1, 1) + &p * &p))));
        }
    }
    let derivative = metered_enclose_parametric_cover_sensitivity(&accepted, &mut budget).unwrap();
    for p in [q(0, 1), q(1, 1), q(2, 1)] {
        let denominator = q(1, 1) + &p * &p;
        let exact = -q(2, 1) * p / (&denominator * &denominator);
        assert!(derivative.solution_jacobian()[0][0].contains(&exact));
    }
    assert_eq!(derivative.leaf_count(), accepted.leaf_count());

    let mut omitted = witness.clone();
    omitted.partition.pop();
    assert!(matches!(
        problem.verify_cover(&unknown_domain, &parameter_domain, &omitted, &mut Unbounded),
        Err(ParametricError::Cover(
            ParametricRootCoverError::InvalidPartition(_)
        ))
    ));
    let mut substituted = witness.clone();
    substituted.leaves[0].parameter_domain[0] = interval(10, 11);
    assert!(matches!(
        problem.verify_cover(
            &unknown_domain,
            &parameter_domain,
            &substituted,
            &mut Unbounded
        ),
        Err(ParametricError::Cover(
            ParametricRootCoverError::InvalidPartition(_)
        ))
    ));
    let mut reused = witness;
    for token in &mut reused.partition {
        if let ParametricRootCoverStep::Leaf { index } = token {
            *index = 0;
        }
    }
    assert!(matches!(
        problem.verify_cover(&unknown_domain, &parameter_domain, &reused, &mut Unbounded),
        Err(ParametricError::Cover(
            ParametricRootCoverError::InvalidPartition(_)
        ))
    ));
}

#[test]
fn quadratic_parameter_sensitivity_and_fixed_unused_axes_remain_exact() {
    let problem =
        ParametricProblem::parse_sources(&["x-p^2"], &["x"], &["unused", "p"], &mut Unbounded)
            .unwrap();
    let mut proposed = affine_witness();
    proposed.unknown_domain = vec![interval(-1, 2)];
    proposed.parameter_domain = vec![RealBall::from_i64(3), interval(-1, 1)];
    let accepted = problem.verify_root_box(&proposed, &mut Unbounded).unwrap();
    assert_eq!(accepted.root_enclosure(), [interval(0, 1)]);
    let derivative =
        metered_enclose_parametric_root_sensitivity(&accepted, &mut Unbounded).unwrap();
    assert_eq!(
        derivative.solution_jacobian(),
        [vec![RealBall::from_i64(0), interval(-2, 2)]]
    );
    let witness = problem
        .propose_cover(
            &proposed.unknown_domain,
            &proposed.parameter_domain,
            options(),
            &mut Unbounded,
        )
        .unwrap();
    let family = problem
        .verify_cover(
            &proposed.unknown_domain,
            &proposed.parameter_domain,
            &witness,
            &mut Unbounded,
        )
        .unwrap();
    assert_eq!(family.leaf_count(), 1);
    assert_eq!(family.parameter_domain(), proposed.parameter_domain);
}

#[test]
fn endpoint_nonexistence_bifurcations_and_original_hidden_holes_refuse() {
    let mut wide = affine_witness();
    wide.unknown_domain = vec![interval(-1, 1)];
    wide.parameter_domain = vec![interval(-2, 2)];
    assert_eq!(
        affine().verify_root_box(&wide, &mut Unbounded).unwrap_err(),
        ParametricError::Certification(CertificationError::Root(
            RootBoxError::NoInteriorInclusion { coordinate: 0 }
        ))
    );
    let singular =
        ParametricProblem::parse_sources(&["p*x-1"], &["x"], &["p"], &mut Unbounded).unwrap();
    let mut proposed = affine_witness();
    proposed.unknown_domain = vec![interval(-1, 5)];
    proposed.parameter_domain = vec![interval(0, 1)];
    proposed.preconditioner[0][0] = q(2, 1);
    assert_eq!(
        singular
            .verify_root_box(&proposed, &mut Unbounded)
            .unwrap_err(),
        ParametricError::Certification(CertificationError::Root(RootBoxError::NotContractive))
    );
    for source in ["x-p+0/p", "x-p+(1/p)^0", "x-p+log(p)-log(p)"] {
        let hidden =
            ParametricProblem::parse_sources(&[source], &["x"], &["p"], &mut Unbounded).unwrap();
        let error = hidden
            .verify_root_box(&affine_witness(), &mut Unbounded)
            .unwrap_err();
        assert!(matches!(
            error,
            ParametricError::Certification(CertificationError::Root(
                RootBoxError::Ball(_) | RootBoxError::Domain(_)
            ))
        ));
        assert!(
            hidden
                .propose_cover(
                    &[interval(-2, 2)],
                    &[interval(-1, 1)],
                    ParametricRootCoverOptions {
                        max_leaves: 2,
                        ..options()
                    },
                    &mut Unbounded
                )
                .is_err()
        );
    }
}

#[test]
fn group_shape_duplicates_and_text_bounds_fail_before_source_ownership() {
    for (residuals, unknowns, parameters) in [
        (vec!["x-p"], vec![], vec!["p"]),
        (vec!["x-p"], vec!["x"], vec![]),
        (vec![], vec!["x"], vec!["p"]),
        (vec!["x-p", "x+p"], vec!["x"], vec!["p"]),
    ] {
        assert_eq!(
            ParametricProblem::parse_sources(&residuals, &unknowns, &parameters, &mut Unbounded)
                .unwrap_err(),
            ParametricError::Certification(CertificationError::Root(
                RootBoxError::DimensionMismatch
            ))
        );
    }
    let axes: Vec<_> = (0..MAX_ROOT_BOX_VARIABLES)
        .map(|i| Symbol::new(format!("x{i}")))
        .collect();
    assert_eq!(
        ParametricProblem::from_exprs(
            vec![Expr::from_i64(0); axes.len()],
            axes,
            vec![Symbol::new("p")],
            &mut Unbounded
        )
        .unwrap_err(),
        ParametricError::Certification(CertificationError::Root(RootBoxError::DimensionMismatch))
    );
    for (unknowns, parameters) in [(vec!["x"], vec!["x"]), (vec!["x"], vec!["p", "p"])] {
        assert_eq!(
            ParametricProblem::parse_sources(&["x-p"], &unknowns, &parameters, &mut Unbounded)
                .unwrap_err(),
            ParametricError::Certification(CertificationError::Root(
                RootBoxError::DuplicateVariable
            ))
        );
    }
    let shared = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([3; 32]));
    assert_eq!(
        ParametricProblem::from_exprs(
            vec![Expr::Sym(shared.clone())],
            vec![shared.clone()],
            vec![shared],
            &mut Unbounded
        )
        .unwrap_err(),
        ParametricError::Certification(CertificationError::Root(RootBoxError::DuplicateVariable))
    );
    let huge_source = "x".repeat(MAX_CERTIFICATION_SOURCE_BYTES + 1);
    assert_eq!(
        ParametricProblem::parse_sources(&[&huge_source], &["x"], &["p"], &mut Unbounded)
            .unwrap_err(),
        ParametricError::Certification(CertificationError::SourceLimit("textual residual bytes"))
    );
    let huge_name = "x".repeat(MAX_CERTIFICATION_NAME_BYTES + 1);
    for name in ["", "pi", "x+p", &huge_name] {
        assert_eq!(
            ParametricProblem::parse_sources(&["x-p"], &[name], &["p"], &mut Unbounded)
                .unwrap_err(),
            ParametricError::Certification(CertificationError::InvalidVariable)
        );
    }
}

#[derive(Default)]
struct Probe {
    checkpoints: usize,
    interrupt: Option<usize>,
    stopped: bool,
    charges: [u64; DIMENSION_COUNT],
}

impl BudgetMeter for Probe {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }
    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.stopped);
        for &(dimension, amount) in charges {
            assert_ne!(amount, 0);
            self.charges[dimension.index()] += amount;
        }
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped);
        self.checkpoints += 1;
        if self.interrupt == Some(self.checkpoints) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn construction_uniform_cover_generation_and_verification_propagate_each_cancellation() {
    let problem = affine();
    let witness = affine_witness();
    let cover = problem
        .propose_cover(
            &witness.unknown_domain,
            &witness.parameter_domain,
            options(),
            &mut Unbounded,
        )
        .unwrap();
    for operation in 0..5 {
        let run = |meter: &mut Probe| -> Result<(), ParametricError> {
            match operation {
                0 => ParametricProblem::parse_sources(&["x-p"], &["x"], &["p"], meter).map(|_| ()),
                1 => ParametricProblem::from_exprs(
                    problem.residuals().to_vec(),
                    problem.unknowns().to_vec(),
                    problem.parameters().to_vec(),
                    meter,
                )
                .map(|_| ()),
                2 => problem.verify_root_box(&witness, meter).map(|_| ()),
                3 => problem
                    .propose_cover(
                        &witness.unknown_domain,
                        &witness.parameter_domain,
                        options(),
                        meter,
                    )
                    .map(|_| ()),
                _ => problem
                    .verify_cover(
                        &witness.unknown_domain,
                        &witness.parameter_domain,
                        &cover,
                        meter,
                    )
                    .map(|_| ()),
            }
        };
        let mut measured = Probe::default();
        run(&mut measured).unwrap();
        for checkpoint in 1..=measured.checkpoints {
            let mut interrupted = Probe {
                interrupt: Some(checkpoint),
                ..Probe::default()
            };
            assert_eq!(
                run(&mut interrupted).unwrap_err(),
                ParametricError::Certification(CertificationError::Meter(MeterError::Cancelled))
            );
            assert!(interrupted.stopped);
        }
    }
}

#[test]
fn failure_retry_and_protected_verification_share_cumulative_budgets() {
    let problem = affine();
    let witness = affine_witness();
    let mut measured = Probe::default();
    problem.verify_root_box(&witness, &mut measured).unwrap();
    let mut budget = Budget::new(BudgetLimits {
        dimensions: measured.charges.map(|amount| amount * 2),
        verifier_pool: 29,
    });
    problem.verify_root_box(&witness, &mut budget).unwrap();
    problem.verify_root_box(&witness, &mut budget).unwrap();
    assert_eq!(budget.remaining(Dimension::ComputeSteps), 0);
    assert_eq!(budget.remaining(Dimension::MemoryBytes), 0);
    assert!(matches!(
        problem.verify_root_box(&witness, &mut budget),
        Err(ParametricError::Certification(CertificationError::Meter(
            MeterError::Budget(_)
        )))
    ));
    assert_eq!(budget.verifier_remaining(), 29);

    let mut empty = Budget::new(BudgetLimits::uniform(0, 999));
    assert!(matches!(
        problem.propose_cover(
            &witness.unknown_domain,
            &witness.parameter_domain,
            options(),
            &mut empty
        ),
        Err(ParametricError::Certification(CertificationError::Meter(
            MeterError::Budget(_)
        )))
    ));
    assert_eq!(empty.verifier_remaining(), 999);
    let candidate = problem
        .propose_cover(
            &witness.unknown_domain,
            &witness.parameter_domain,
            options(),
            &mut Unbounded,
        )
        .unwrap();
    assert!(matches!(
        problem.verify_cover(
            &witness.unknown_domain,
            &witness.parameter_domain,
            &candidate,
            &mut empty
        ),
        Err(ParametricError::Certification(CertificationError::Meter(
            MeterError::Budget(_)
        )))
    ));
    assert_eq!(empty.verifier_remaining(), 999);

    let mut protected = Budget::new(BudgetLimits::uniform(1 << 30, 1 << 20));
    let lease = protected.verifier_lease().unwrap();
    let before = protected.verifier_remaining();
    let accepted = problem
        .verify_cover(
            &witness.unknown_domain,
            &witness.parameter_domain,
            &candidate,
            &mut protected.verifier_meter(&lease).unwrap(),
        )
        .unwrap();
    assert_eq!(accepted.root_enclosure(), [interval(-1, 1)]);
    assert!(protected.verifier_remaining() < before);
}

#[test]
fn invalid_cover_options_and_dimension_mismatches_remain_terminal() {
    let problem = affine();
    let witness = affine_witness();
    for max_leaves in [0, 1025, usize::MAX] {
        assert!(matches!(
            problem.propose_cover(
                &witness.unknown_domain,
                &witness.parameter_domain,
                ParametricRootCoverOptions {
                    max_leaves,
                    ..options()
                },
                &mut Unbounded
            ),
            Err(ParametricError::Certification(CertificationError::Root(
                RootBoxError::ResourceLimit(_)
            )))
        ));
    }
    for precision_digits in [0, 7, 129, u32::MAX] {
        assert_eq!(
            problem
                .propose_cover(
                    &witness.unknown_domain,
                    &witness.parameter_domain,
                    ParametricRootCoverOptions {
                        precision_digits,
                        ..options()
                    },
                    &mut Unbounded
                )
                .unwrap_err(),
            ParametricError::Certification(CertificationError::Root(
                RootBoxError::InvalidPrecision
            ))
        );
    }
    let mut malformed = witness.clone();
    malformed.parameter_domain.clear();
    assert_eq!(
        problem
            .verify_root_box(&malformed, &mut Unbounded)
            .unwrap_err(),
        ParametricError::Certification(CertificationError::Root(RootBoxError::DimensionMismatch))
    );
    assert_eq!(
        problem
            .propose_cover(&[], &witness.parameter_domain, options(), &mut Unbounded)
            .unwrap_err(),
        ParametricError::Certification(CertificationError::Root(RootBoxError::DimensionMismatch))
    );
}
