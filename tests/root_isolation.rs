use frankensympy::{
    BigRational, Budget, BudgetLimits, BudgetMeter, CertificationError, Dimension, MeterError,
    RealBall, ResidualProblem, RootIsolationError, RootIsolationOptions, RootIsolationStep,
    RootIsolationWitness, export_root_certificate, replay_root_certificate,
};
use fsym_budget::{BudgetError, Unbounded};
use fsym_core::RootBoxError;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}
fn interval(lower: i64, upper: i64) -> RealBall {
    RealBall::new(q(lower + upper, 2), q(upper - lower, 2)).unwrap()
}
fn problem(residuals: &[&str], variables: &[&str]) -> ResidualProblem {
    ResidualProblem::parse_sources(
        &residuals
            .iter()
            .map(|text| (*text).to_owned())
            .collect::<Vec<_>>(),
        &variables
            .iter()
            .map(|text| (*text).to_owned())
            .collect::<Vec<_>>(),
        &mut Unbounded,
    )
    .unwrap()
}
fn options() -> RootIsolationOptions {
    RootIsolationOptions {
        precision_digits: 16,
        max_leaves: 512,
    }
}

#[test]
fn automatic_isolation_finds_and_replays_both_irrational_roots() {
    let system = problem(&["x^2-2"], &["x"]);
    let domain = [interval(-2, 2)];
    let receipt = system
        .isolate_roots(&domain, options(), &mut Unbounded, &mut Unbounded)
        .unwrap();
    assert_eq!(receipt.root_count(), 2);
    assert_eq!(receipt.residuals(), system.residuals());
    assert_eq!(receipt.variables(), system.variables());
    assert_eq!(receipt.domain(), domain);
    assert!(receipt.excluded_leaf_count() > 0);
    let negative = &receipt.roots()[0].root_enclosure()[0];
    let positive = &receipt.roots()[1].root_enclosure()[0];
    assert!(negative.upper() < q(0, 1));
    assert!(positive.lower() > q(0, 1));
    assert!(&negative.lower() * &negative.lower() >= q(2, 1));
    assert!(&negative.upper() * &negative.upper() <= q(2, 1));
    assert!(&positive.lower() * &positive.lower() <= q(2, 1));
    assert!(&positive.upper() * &positive.upper() >= q(2, 1));
    let receiver = problem(&["x^2-2"], &["x"]);
    for root in receipt.roots() {
        let artifact = export_root_certificate(root, [23; 32], &mut Unbounded).unwrap();
        assert_eq!(
            replay_root_certificate(&artifact, &receiver, [23; 32], &mut Unbounded).unwrap(),
            *root,
        );
    }
}

#[test]
fn coupled_systems_retain_all_roots_and_explicit_coordinate_order() {
    let system = problem(&["x^2-1", "x^2-1+y^2-1"], &["x", "y"]);
    let receipt = system
        .isolate_roots(
            &[interval(-2, 2), interval(-2, 2)],
            options(),
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
    assert_eq!(receipt.root_count(), 4);
    for x in [-1, 1] {
        for y in [-1, 1] {
            assert_eq!(
                receipt
                    .roots()
                    .iter()
                    .filter(|root| {
                        root.root_enclosure()[0].contains(&q(x, 1))
                            && root.root_enclosure()[1].contains(&q(y, 1))
                    })
                    .count(),
                1,
            );
        }
    }
    let reversed = problem(&["x+2*y-5", "3*x-y-1"], &["y", "x"]);
    let affine = reversed
        .isolate_roots(
            &[interval(0, 4), interval(0, 2)],
            options(),
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
    assert_eq!(affine.root_count(), 1);
    assert_eq!(affine.variables(), reversed.variables());
    assert_eq!(affine.residuals(), reversed.residuals());
    assert_eq!(
        affine.roots()[0].root_enclosure(),
        &[RealBall::from_i64(2), RealBall::from_i64(1)]
    );
}

#[test]
fn root_free_receipts_require_a_complete_original_source_exclusion() {
    let domain = [interval(-2, 2)];
    let root_free = problem(&["x^2+1"], &["x"])
        .isolate_roots(&domain, options(), &mut Unbounded, &mut Unbounded)
        .unwrap();
    assert!(root_free.is_root_free());
    assert_eq!(root_free.root_count(), 0);
    assert_eq!(root_free.leaf_count(), 1);
    assert_eq!(root_free.excluded_leaf_count(), 1);
    let fake_empty = RootIsolationWitness {
        roots: Vec::new(),
        partition: vec![RootIsolationStep::Exclude],
        precision_digits: 16,
    };
    assert!(matches!(
        problem(&["x^2-2"], &["x"]).verify_isolation(&domain, &fake_empty, &mut Unbounded),
        Err(CertificationError::Isolation(
            RootIsolationError::Incomplete(_)
        )),
    ));
}

#[test]
fn independent_cover_replay_rejects_missing_regions_overlap_and_changed_sources() {
    let system = problem(&["x^2-2"], &["x"]);
    let domain = [interval(-2, 2)];
    let witness = system
        .propose_isolation(&domain, options(), &mut Unbounded)
        .unwrap();
    let mut missing = witness.clone();
    missing.partition.pop();
    assert!(matches!(
        system.verify_isolation(&domain, &missing, &mut Unbounded),
        Err(CertificationError::Isolation(
            RootIsolationError::InvalidPartition(_)
        )),
    ));
    let mut overlap = witness.clone();
    overlap.roots[1] = overlap.roots[0].clone();
    assert!(matches!(
        system.verify_isolation(&domain, &overlap, &mut Unbounded),
        Err(CertificationError::Isolation(
            RootIsolationError::OverlappingRoots { .. }
        )),
    ));
    for changed in [problem(&["x^2+2"], &["x"]), problem(&["x^2-2"], &["y"])] {
        assert!(
            changed
                .verify_isolation(&domain, &witness, &mut Unbounded)
                .is_err()
        );
    }
    assert!(
        system
            .verify_isolation(&[interval(2, 3)], &witness, &mut Unbounded)
            .is_err()
    );
}

#[test]
fn zero_products_cancelled_logs_and_zero_powers_retain_original_poles() {
    let bounded = RootIsolationOptions {
        max_leaves: 8,
        ..options()
    };
    for source in ["1+0*(1/x)", "log(x)-log(x)+1", "1+(1/x)^0"] {
        assert!(
            matches!(
                problem(&[source], &["x"]).isolate_roots(
                    &[interval(-1, 1)],
                    bounded,
                    &mut Unbounded,
                    &mut Unbounded,
                ),
                Err(CertificationError::Root(
                    RootBoxError::Domain(_) | RootBoxError::Ball(_)
                )),
            ),
            "source: {source}"
        );
    }
    assert!(matches!(
        problem(&["1", "0*(1/y)"], &["x", "y"]).isolate_roots(
            &[interval(-1, 1), interval(-1, 1)],
            bounded,
            &mut Unbounded,
            &mut Unbounded,
        ),
        Err(CertificationError::Root(
            RootBoxError::Domain(_) | RootBoxError::Ball(_)
        )),
    ));
}

#[test]
fn singular_boundary_and_exhausted_searches_never_return_partial_root_sets() {
    for (source, domain) in [
        ("x^2", interval(-1, 1)),
        ("x", interval(0, 1)),
        ("x^2*(x+1)", interval(-2, 2)),
    ] {
        assert!(matches!(
            problem(&[source], &["x"]).isolate_roots(
                &[domain],
                RootIsolationOptions {
                    max_leaves: 16,
                    ..options()
                },
                &mut Unbounded,
                &mut Unbounded,
            ),
            Err(CertificationError::Isolation(
                RootIsolationError::Incomplete(_)
            )),
        ));
    }
    let system = problem(&["x^2-2"], &["x"]);
    for max_leaves in [0, 1025, usize::MAX] {
        assert!(matches!(
            system.propose_isolation(
                &[interval(-2, 2)],
                RootIsolationOptions {
                    max_leaves,
                    ..options()
                },
                &mut Unbounded,
            ),
            Err(CertificationError::Root(RootBoxError::ResourceLimit(_))),
        ));
    }
    for precision_digits in [7, 129, u32::MAX] {
        assert!(matches!(
            system.propose_isolation(
                &[interval(-2, 2)],
                RootIsolationOptions {
                    precision_digits,
                    ..options()
                },
                &mut Unbounded,
            ),
            Err(CertificationError::Root(RootBoxError::InvalidPrecision)),
        ));
    }
    assert!(matches!(
        problem(&["x", "x+1"], &["x"]).propose_isolation(
            &[interval(-1, 1)],
            options(),
            &mut Unbounded,
        ),
        Err(CertificationError::Root(RootBoxError::DimensionMismatch)),
    ));
    assert!(matches!(
        system.propose_isolation(&[RealBall::from_i64(0)], options(), &mut Unbounded),
        Err(CertificationError::Root(RootBoxError::DegenerateBox { .. })),
    ));
}

#[test]
fn one_cumulative_ledger_keeps_proposals_out_of_protected_verifier_fuel() {
    let system = problem(&["x^2-2"], &["x"]);
    let domain = [interval(-2, 2)];
    let mut ledger = Budget::new(BudgetLimits::uniform(1 << 40, 1 << 30));
    let lease = ledger.verifier_lease().unwrap();
    let ordinary_before = ledger.remaining(Dimension::ComputeSteps);
    let protected_before = ledger.verifier_remaining();
    let witness = system
        .propose_isolation(&domain, options(), &mut ledger)
        .unwrap();
    assert!(ledger.remaining(Dimension::ComputeSteps) < ordinary_before);
    assert_eq!(ledger.verifier_remaining(), protected_before);
    let ordinary_after = ledger.remaining(Dimension::ComputeSteps);
    let accepted = system
        .verify_isolation(
            &domain,
            &witness,
            &mut ledger.verifier_meter(&lease).unwrap(),
        )
        .unwrap();
    assert_eq!(accepted.root_count(), 2);
    assert_eq!(ledger.remaining(Dimension::ComputeSteps), ordinary_after);
    assert!(ledger.verifier_remaining() < protected_before);
    let mut incomplete = witness;
    incomplete.partition.pop();
    let protected_after = ledger.verifier_remaining();
    assert!(
        system
            .verify_isolation(
                &domain,
                &incomplete,
                &mut ledger.verifier_meter(&lease).unwrap(),
            )
            .is_err()
    );
    assert!(ledger.verifier_remaining() < protected_after);
    assert_eq!(ledger.remaining(Dimension::ComputeSteps), ordinary_after);
}

#[test]
fn convenience_uses_distinct_meters_and_never_verifies_a_failed_proposal() {
    let system = problem(&["x^2-2"], &["x"]);
    let domain = [interval(-2, 2)];
    let mut proposer = Budget::new(BudgetLimits::uniform(1 << 40, 73));
    let mut verification_limits = BudgetLimits::uniform(1 << 40, 1 << 30);
    verification_limits.dimensions[Dimension::ComputeSteps.index()] = 0;
    let mut verifier = Budget::new(verification_limits);
    let lease = verifier.verifier_lease().unwrap();
    let accepted = system
        .isolate_roots(
            &domain,
            options(),
            &mut proposer,
            &mut verifier.verifier_meter(&lease).unwrap(),
        )
        .unwrap();
    assert_eq!(accepted.root_count(), 2);
    assert_eq!(proposer.verifier_remaining(), 73);
    assert_eq!(verifier.remaining(Dimension::ComputeSteps), 0);
    assert!(verifier.verifier_remaining() < 1 << 30);
    let mut untouched = Probe::default();
    let mut exhausted_limits = BudgetLimits::uniform(1 << 40, 73);
    exhausted_limits.dimensions[Dimension::ComputeSteps.index()] = 0;
    let mut exhausted = Budget::new(exhausted_limits);
    assert!(matches!(
        system.isolate_roots(&domain, options(), &mut exhausted, &mut untouched),
        Err(CertificationError::Meter(MeterError::Budget(
            BudgetError::Exhausted { .. }
        ))),
    ));
    assert_eq!(exhausted.verifier_remaining(), 73);
    assert_eq!(untouched.checkpoints, 0);
    assert!(matches!(
        system.isolate_roots(
            &domain,
            RootIsolationOptions {
                max_leaves: 1,
                ..options()
            },
            &mut Unbounded,
            &mut untouched,
        ),
        Err(CertificationError::Isolation(
            RootIsolationError::Incomplete(_)
        )),
    ));
    assert_eq!(untouched.checkpoints, 0);
}

#[derive(Default)]
struct Probe {
    checkpoints: usize,
    interrupt: Option<usize>,
    stopped: bool,
}
impl BudgetMeter for Probe {
    fn charge(&mut self, _dimension: Dimension, _amount: u64) -> Result<(), MeterError> {
        assert!(!self.stopped, "work resumed after cancellation");
        Ok(())
    }
    fn charge_batch(&mut self, _charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.stopped, "work resumed after cancellation");
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped, "work resumed after cancellation");
        self.checkpoints += 1;
        if self.interrupt == Some(self.checkpoints) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}
fn cancellation_stays_typed(
    mut operation: impl FnMut(&mut Probe) -> Result<(), CertificationError>,
) {
    let mut measured = Probe::default();
    operation(&mut measured).unwrap();
    assert!(measured.checkpoints > 1);
    for interrupt in [1, measured.checkpoints.div_ceil(2), measured.checkpoints] {
        let mut meter = Probe {
            interrupt: Some(interrupt),
            ..Probe::default()
        };
        assert_eq!(
            operation(&mut meter),
            Err(CertificationError::Meter(MeterError::Cancelled))
        );
        assert!(meter.stopped);
    }
}
#[test]
fn cancellation_in_either_phase_stays_an_operational_refusal() {
    let system = problem(&["x^2-2"], &["x"]);
    let domain = [interval(-2, 2)];
    let witness = system
        .propose_isolation(&domain, options(), &mut Unbounded)
        .unwrap();
    cancellation_stays_typed(|meter| {
        system
            .propose_isolation(&domain, options(), meter)
            .map(|_| ())
    });
    cancellation_stays_typed(|meter| {
        system
            .verify_isolation(&domain, &witness, meter)
            .map(|_| ())
    });
    cancellation_stays_typed(|meter| {
        system
            .isolate_roots(&domain, options(), &mut Unbounded, meter)
            .map(|_| ())
    });
}
