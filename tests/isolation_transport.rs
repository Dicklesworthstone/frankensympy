use frankensympy::certification::{
    CertificationError, MAX_ROOT_CERTIFICATE_BYTES, MAX_ROOT_CERTIFICATE_JSON_DEPTH,
    ResidualProblem,
};
use frankensympy::isolation_transport::{export_root_isolation, replay_root_isolation};
use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::root_isolation::RootIsolationOptions;
use fsym_core::{
    BigRational, MAX_ROOT_ISOLATION_NODES, MAX_ROOT_ISOLATION_ROOTS, RealBall, RootIsolationError,
    RootIsolationReceipt, Symbol,
};

fn problem(source: &str) -> ResidualProblem {
    ResidualProblem::parse_sources(&[source.into()], &["x".into()], &mut Unbounded).unwrap()
}

fn domain() -> Vec<RealBall> {
    vec![
        RealBall::new(
            BigRational::from_integer(0.into()),
            BigRational::from_integer(2.into()),
        )
        .unwrap(),
    ]
}

fn receipt(problem: &ResidualProblem) -> RootIsolationReceipt {
    problem
        .isolate_roots(
            &domain(),
            RootIsolationOptions {
                precision_digits: 16,
                max_leaves: 128,
            },
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap()
}

#[test]
fn complete_two_root_and_root_free_proofs_replay_in_fresh_receivers() {
    for (source, count) in [("x^2-2", 2), ("x^2+1", 0)] {
        let sender = problem(source);
        let original = receipt(&sender);
        let bytes = export_root_isolation(&original, [7; 32], &mut Unbounded).unwrap();
        let receiver = problem(source);
        let replayed =
            replay_root_isolation(&bytes, &receiver, &domain(), [7; 32], &mut Unbounded).unwrap();
        assert_eq!(replayed, original);
        assert_eq!(replayed.root_count(), count);
        assert_eq!(replayed.is_root_free(), count == 0);
        assert_eq!(
            export_root_isolation(&replayed, [7; 32], &mut Unbounded).unwrap(),
            bytes,
        );
        let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(wire.get("root_count").is_none());
        assert!(wire.get("accepted").is_none());
        assert!(wire["witness"].get("root_enclosure").is_none());
    }
}

#[test]
fn inverse_circular_sources_isolate_and_replay_against_current_core() {
    let sender = problem("asin(x)");
    let exact_zero = BigRational::from_integer(0.into());
    let domain = [RealBall::new(exact_zero.clone(), BigRational::new(1.into(), 2.into())).unwrap()];
    let accepted = sender
        .isolate_roots(
            &domain,
            RootIsolationOptions {
                precision_digits: 16,
                max_leaves: 128,
            },
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
    assert_eq!(accepted.root_count(), 1);
    assert_eq!(accepted.residuals(), sender.residuals());
    assert_eq!(accepted.domain(), domain);
    assert_eq!(
        accepted.roots()[0].witness().center.as_slice(),
        std::slice::from_ref(&exact_zero),
    );
    assert!(accepted.roots()[0].root_enclosure()[0].contains(&exact_zero));
    let bytes = export_root_isolation(&accepted, [19; 32], &mut Unbounded).unwrap();
    let receiver = problem("asin(x)");
    let replayed =
        replay_root_isolation(&bytes, &receiver, &domain, [19; 32], &mut Unbounded).unwrap();
    assert_eq!(replayed, accepted);
    assert_eq!(replayed.residuals(), receiver.residuals());
    assert_eq!(
        replay_root_isolation(&bytes, &problem("x"), &domain, [19; 32], &mut Unbounded),
        Err(CertificationError::SourceMismatch),
    );
}

#[test]
fn receiving_source_axes_context_and_complete_region_are_bound() {
    let sender = problem("x^2-2");
    let bytes = export_root_isolation(&receipt(&sender), [7; 32], &mut Unbounded).unwrap();
    assert_eq!(
        replay_root_isolation(
            &bytes,
            &problem("x^2-3"),
            &domain(),
            [7; 32],
            &mut Unbounded
        ),
        Err(CertificationError::SourceMismatch),
    );
    let changed_axes = ResidualProblem::new(
        sender.residuals().to_vec(),
        vec![Symbol::new("y")],
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(
        replay_root_isolation(&bytes, &changed_axes, &domain(), [7; 32], &mut Unbounded),
        Err(CertificationError::SourceMismatch),
    );
    let wider = [RealBall::new(
        BigRational::from_integer(0.into()),
        BigRational::from_integer(3.into()),
    )
    .unwrap()];
    assert_eq!(
        replay_root_isolation(&bytes, &sender, &wider, [7; 32], &mut Unbounded),
        Err(CertificationError::DomainMismatch),
    );
    assert!(matches!(
        replay_root_isolation(&bytes, &sender, &domain(), [8; 32], &mut Unbounded),
        Err(CertificationError::UniverseMismatch { .. }),
    ));
}

#[test]
fn canonical_false_exclusion_reaches_and_fails_the_independent_checker() {
    let empty = problem("x^2+1");
    let target = problem("x^2-2");
    let original = String::from_utf8(
        export_root_isolation(&receipt(&empty), [7; 32], &mut Unbounded).unwrap(),
    )
    .unwrap();
    let old = format!(
        "\"residuals\":{}",
        serde_json::to_string(empty.residuals()).unwrap()
    );
    let new = format!(
        "\"residuals\":{}",
        serde_json::to_string(target.residuals()).unwrap()
    );
    assert!(original.contains(&old));
    let forged = original.replacen(&old, &new, 1);
    assert!(matches!(
        replay_root_isolation(
            forged.as_bytes(),
            &target,
            &domain(),
            [7; 32],
            &mut Unbounded
        ),
        Err(CertificationError::Isolation(
            RootIsolationError::Incomplete(_)
        )),
    ));
}

#[test]
fn malformed_noncanonical_and_oversized_covers_never_receive_acceptance() {
    let problem = problem("x^2+1");
    let original = String::from_utf8(
        export_root_isolation(&receipt(&problem), [7; 32], &mut Unbounded).unwrap(),
    )
    .unwrap();
    let extra_steps = format!(
        "\"partition\":[{}]",
        vec!["\"exclude\""; MAX_ROOT_ISOLATION_NODES + 1].join(","),
    );
    let extra_roots = format!(
        "\"roots\":[{}]",
        vec!["{}"; MAX_ROOT_ISOLATION_ROOTS + 1].join(","),
    );
    for mutant in [
        format!(" {original}"),
        original.replacen(
            "\"schema_version\":1",
            "\"schema_version\":1,\"schema_version\":1",
            1,
        ),
        original.replacen(
            "\"precision_digits\":16",
            "\"accepted\":true,\"precision_digits\":16",
            1,
        ),
        original.replacen("\"partition\":[\"exclude\"]", "\"partition\":[]", 1),
        original.replacen("\"partition\":[\"exclude\"]", &extra_steps, 1),
        original.replacen("\"roots\":[]", &extra_roots, 1),
    ] {
        assert_ne!(mutant, original);
        assert!(
            replay_root_isolation(
                mutant.as_bytes(),
                &problem,
                &domain(),
                [7; 32],
                &mut Unbounded
            )
            .is_err(),
        );
    }
    assert!(matches!(
        replay_root_isolation(
            &vec![b' '; MAX_ROOT_CERTIFICATE_BYTES + 1],
            &problem,
            &domain(),
            [7; 32],
            &mut Unbounded,
        ),
        Err(CertificationError::ByteLimit { .. }),
    ));
    let depth = MAX_ROOT_CERTIFICATE_JSON_DEPTH + 1;
    let nested = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
    assert!(matches!(
        replay_root_isolation(
            nested.as_bytes(),
            &problem,
            &domain(),
            [7; 32],
            &mut Unbounded
        ),
        Err(CertificationError::DepthLimit { .. }),
    ));
}

struct Stop {
    remaining: usize,
    checks: usize,
    stopped: bool,
}

impl BudgetMeter for Stop {
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
        self.checks += 1;
        if self.remaining == 0 {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            self.remaining -= 1;
            Ok(())
        }
    }
}

fn cancellation_sweep(mut operation: impl FnMut(&mut Stop) -> Result<(), CertificationError>) {
    let mut measured = Stop {
        remaining: usize::MAX,
        checks: 0,
        stopped: false,
    };
    operation(&mut measured).unwrap();
    assert!(measured.checks > 1);
    for remaining in 0..measured.checks {
        let mut stop = Stop {
            remaining,
            checks: 0,
            stopped: false,
        };
        assert_eq!(
            operation(&mut stop),
            Err(CertificationError::Meter(MeterError::Cancelled)),
        );
        assert!(stop.stopped);
    }
}

#[test]
fn transport_cancels_at_every_safe_point_and_uses_the_receiving_protected_pool() {
    let affine = problem("x");
    let accepted = receipt(&affine);
    let bytes = export_root_isolation(&accepted, [7; 32], &mut Unbounded).unwrap();
    cancellation_sweep(|meter| export_root_isolation(&accepted, [7; 32], meter).map(|_| ()));
    cancellation_sweep(|meter| {
        replay_root_isolation(&bytes, &affine, &domain(), [7; 32], meter).map(|_| ())
    });

    let two_roots = problem("x^2-2");
    let bytes = export_root_isolation(&receipt(&two_roots), [7; 32], &mut Unbounded).unwrap();
    let mut limits = BudgetLimits::uniform(1 << 40, 100_000_000);
    limits.dimensions[Dimension::ComputeSteps.index()] = 0;
    let mut budget = Budget::new(limits);
    let lease = budget.verifier_lease().unwrap();
    let accepted = replay_root_isolation(
        &bytes,
        &two_roots,
        &domain(),
        [7; 32],
        &mut budget.verifier_meter(&lease).unwrap(),
    )
    .unwrap();
    assert_eq!(accepted.root_count(), 2);
    assert_eq!(budget.remaining(Dimension::ComputeSteps), 0);
    assert!(budget.verifier_remaining() < 100_000_000);
}
