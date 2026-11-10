use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::root_isolation::{
    MAX_ROOT_ISOLATION_ADVANCE_CELLS, RootIsolationOptions, RootIsolationSearch,
    RootIsolationSearchCheckpoint, propose_root_isolation,
};
use fsym_core::{
    BigRational, Expr, MAX_ROOT_BOX_ALLOCATION_BYTES, MAX_ROOT_BOX_STEPS,
    MAX_ROOT_ISOLATION_LEAVES, RealBall, RootBoxError, RootIsolationError, RootIsolationStep,
    Symbol, SymbolIdentity, verify_root_isolation,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn interval(a: i64, b: i64) -> RealBall {
    RealBall::new(q(a + b, 2), q(b - a, 2)).unwrap()
}

fn square(x: &Symbol, constant: i64) -> Expr {
    Expr::Add(vec![
        Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(2))),
        Expr::from_i64(constant),
    ])
}

fn options() -> RootIsolationOptions {
    RootIsolationOptions {
        precision_digits: 16,
        max_leaves: 512,
    }
}

fn budget() -> Budget {
    Budget::new(BudgetLimits::uniform(1 << 35, 100_000_000))
}

fn fixture() -> (Vec<Expr>, Vec<Symbol>, Vec<RealBall>) {
    let x = Symbol::new("x");
    (vec![square(&x, -2)], vec![x], vec![interval(-2, 2)])
}

fn completed_checkpoint() -> RootIsolationSearchCheckpoint {
    let (sources, variables, domain) = fixture();
    let mut search =
        RootIsolationSearch::new(&sources, &variables, &domain, options(), &mut Unbounded).unwrap();
    search
        .advance(MAX_ROOT_ISOLATION_ADVANCE_CELLS, &mut Unbounded)
        .unwrap();
    search.checkpoint(&mut Unbounded).unwrap()
}

fn restore(
    checkpoint: &RootIsolationSearchCheckpoint,
) -> Result<RootIsolationSearch, RootIsolationError> {
    let (sources, variables, domain) = fixture();
    RootIsolationSearch::from_checkpoint(&sources, &variables, &domain, checkpoint, &mut Unbounded)
}

#[test]
fn every_saved_prefix_resumes_to_the_identical_independently_verified_cover() {
    let (sources, variables, domain) = fixture();
    let expected =
        propose_root_isolation(&sources, &variables, &domain, options(), &mut Unbounded).unwrap();
    let mut original =
        RootIsolationSearch::new(&sources, &variables, &domain, options(), &mut Unbounded).unwrap();
    let mut saw_root_and_pending = false;
    let mut saw_exclusion = false;
    for prefix_len in 0..=expected.partition.len() {
        let before = original.progress();
        assert_eq!(before.processed_cells, prefix_len);
        let checkpoint = original.checkpoint(&mut Unbounded).unwrap();
        assert_eq!(checkpoint.counters, original.counters());
        let frozen = checkpoint.clone();
        let mut receiver = budget();
        let mut resumed = RootIsolationSearch::from_checkpoint(
            &sources,
            &variables,
            &domain,
            &checkpoint,
            &mut receiver,
        )
        .unwrap();
        assert_eq!(checkpoint, frozen);
        assert_eq!(resumed.progress(), before);
        assert!(resumed.counters().steps > checkpoint.counters.steps);
        assert!(resumed.counters().allocation_bytes > checkpoint.counters.allocation_bytes);
        assert!(receiver.remaining(Dimension::ComputeSteps) < 1 << 35);
        assert_eq!(receiver.verifier_remaining(), 100_000_000);
        saw_root_and_pending |= before.candidate_root_count > 0 && before.pending_cells > 0;
        saw_exclusion |= before.excluded_cells > 0;
        resumed
            .advance(MAX_ROOT_ISOLATION_ADVANCE_CELLS, &mut receiver)
            .unwrap();
        assert_eq!(resumed.completed_witness(), Some(&expected));
        let lease = receiver.verifier_lease().unwrap();
        let receipt = verify_root_isolation(
            &sources,
            &variables,
            &domain,
            resumed.completed_witness().unwrap(),
            &mut receiver.verifier_meter(&lease).unwrap(),
        )
        .unwrap();
        assert_eq!(receipt.root_count(), 2);
        assert_eq!(receipt.witness(), &expected);
        if !before.ready_for_verification {
            original.advance(1, &mut Unbounded).unwrap();
        }
    }
    assert!(saw_root_and_pending && saw_exclusion);
}

#[test]
fn multiaxis_pending_geometry_is_rebuilt_with_distinct_same_name_coordinates() {
    let x = Symbol::with_identity("axis", SymbolIdentity::from_assumptions_digest([1; 32]));
    let y = Symbol::with_identity("axis", SymbolIdentity::from_assumptions_digest([2; 32]));
    let px = square(&x, -1);
    let sources = [px.clone(), Expr::Add(vec![px, square(&y, -1)])];
    let variables = [x, y];
    let domain = [interval(-2, 2), interval(-2, 2)];
    let expected =
        propose_root_isolation(&sources, &variables, &domain, options(), &mut Unbounded).unwrap();
    let mut original =
        RootIsolationSearch::new(&sources, &variables, &domain, options(), &mut Unbounded).unwrap();
    original.advance(7, &mut Unbounded).unwrap();
    assert!(!original.progress().ready_for_verification);
    let checkpoint = original.checkpoint(&mut Unbounded).unwrap();
    let mut resumed = RootIsolationSearch::from_checkpoint(
        &sources,
        &variables,
        &domain,
        &checkpoint,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(resumed.variables(), variables);
    assert_eq!(resumed.progress(), original.progress());
    resumed
        .advance(MAX_ROOT_ISOLATION_ADVANCE_CELLS, &mut Unbounded)
        .unwrap();
    assert_eq!(resumed.completed_witness(), Some(&expected));
    let receipt = verify_root_isolation(
        &sources,
        &variables,
        &domain,
        resumed.completed_witness().unwrap(),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(receipt.root_count(), 4);
}

#[test]
fn independent_replay_rejects_forged_exclusions_roots_and_coverage() {
    let original = completed_checkpoint();
    let root_position = original
        .witness
        .partition
        .iter()
        .position(|step| matches!(step, RootIsolationStep::Root { .. }))
        .unwrap();
    let mut bad = original.clone();
    bad.witness.partition[root_position] = RootIsolationStep::Exclude;
    assert!(matches!(
        restore(&bad),
        Err(RootIsolationError::Incomplete(
            "checkpoint exclusion still admits a zero residual vector"
        ))
    ));

    let mut bad = original.clone();
    bad.witness.roots[0].preconditioner[0][0] = q(0, 1);
    assert!(matches!(
        restore(&bad),
        Err(RootIsolationError::Verification(
            RootBoxError::NotContractive
        ))
    ));

    let mut bad = original.clone();
    bad.witness.roots[0].center[0] += q(1, 1 << 30);
    bad.witness.roots[0].domain = vec![interval(5, 6)];
    assert!(
        restore(&bad).is_err(),
        "root boxes must remain in the requested box"
    );

    let mut bad = original.clone();
    bad.witness.partition[root_position] = RootIsolationStep::Root { index: 1 };
    assert!(
        restore(&bad).is_err(),
        "new root references must follow discovery order"
    );

    let mut bad = original.clone();
    bad.witness.partition[root_position] = RootIsolationStep::Root { index: usize::MAX };
    assert!(restore(&bad).is_err());

    let mut bad = original.clone();
    bad.witness.partition[0] = RootIsolationStep::Split { axis: usize::MAX };
    assert!(restore(&bad).is_err());

    let mut bad = original.clone();
    bad.witness.partition.push(RootIsolationStep::Exclude);
    assert!(
        restore(&bad).is_err(),
        "tokens after a completed cover are invalid"
    );

    let mut bad = original.clone();
    bad.witness.roots.push(bad.witness.roots[0].clone());
    assert!(
        restore(&bad).is_err(),
        "overlapping or unreferenced roots are invalid"
    );
    assert!(
        restore(&original)
            .unwrap()
            .progress()
            .ready_for_verification
    );
}

#[test]
fn invalid_options_and_intrinsic_history_cannot_restart_lifetime_limits() {
    let original = completed_checkpoint();
    for bad_count in [0, MAX_ROOT_ISOLATION_LEAVES + 1, usize::MAX] {
        let mut bad = original.clone();
        bad.options.max_leaves = bad_count;
        assert!(restore(&bad).is_err());
    }
    let mut bad = original.clone();
    bad.options.max_leaves = 1;
    assert!(restore(&bad).is_err());
    let mut bad = original.clone();
    bad.options.precision_digits += 1;
    assert!(restore(&bad).is_err());
    for field in 0..4 {
        let mut bad = original.clone();
        match field {
            0 => bad.counters.steps = 0,
            1 => bad.counters.allocation_bytes = 0,
            2 => bad.counters.source_bytes += 1,
            _ => bad.counters.evaluations = 0,
        }
        assert!(restore(&bad).is_err(), "understated history field {field}");
    }
    for value in [MAX_ROOT_BOX_STEPS, u64::MAX] {
        let mut bad = original.clone();
        bad.counters.steps = value;
        assert!(
            restore(&bad).is_err(),
            "receiver preparation is additional work"
        );
    }
    for value in [MAX_ROOT_BOX_ALLOCATION_BYTES, u64::MAX] {
        let mut bad = original.clone();
        bad.counters.allocation_bytes = value;
        assert!(restore(&bad).is_err());
    }
    let mut bad = original;
    bad.counters.evaluations = usize::MAX;
    assert!(restore(&bad).is_err());
}

#[test]
fn repeated_honest_restore_accumulates_validation_and_eventually_refuses_without_reset() {
    let mut checkpoint = completed_checkpoint();
    let (sources, variables, domain) = fixture();
    let mut meter = budget();
    let initial_remaining = meter.remaining(Dimension::MemoryBytes);
    let mut restored = 0;
    for _ in 0..8 {
        let before = meter.remaining(Dimension::MemoryBytes);
        match RootIsolationSearch::from_checkpoint(
            &sources,
            &variables,
            &domain,
            &checkpoint,
            &mut meter,
        ) {
            Ok(mut search) => {
                assert!(search.counters().allocation_bytes > checkpoint.counters.allocation_bytes);
                checkpoint = search.checkpoint(&mut meter).unwrap();
                restored += 1;
            }
            Err(RootIsolationError::Verification(RootBoxError::ResourceLimit(_))) => {
                assert!(meter.remaining(Dimension::MemoryBytes) < before);
                assert!(restored >= 1);
                assert!(meter.remaining(Dimension::MemoryBytes) < initial_remaining);
                return;
            }
            Err(error) => panic!("unexpected refusal: {error}"),
        }
    }
    panic!("restoring must not reset the cumulative intrinsic allocation limit");
}

#[test]
fn no_root_and_partial_pole_searches_restore_without_false_completion() {
    let x = Symbol::new("x");
    let domain = [interval(-2, 2)];
    let variables = [x.clone()];
    let sources = [square(&x, 1)];
    let mut search =
        RootIsolationSearch::new(&sources, &variables, &domain, options(), &mut Unbounded).unwrap();
    search
        .advance(MAX_ROOT_ISOLATION_ADVANCE_CELLS, &mut Unbounded)
        .unwrap();
    let saved = search.checkpoint(&mut Unbounded).unwrap();
    let restored =
        RootIsolationSearch::from_checkpoint(&sources, &variables, &domain, &saved, &mut Unbounded)
            .unwrap();
    assert_eq!(restored.progress().candidate_root_count, 0);
    assert!(restored.progress().ready_for_verification);
    assert!(
        verify_root_isolation(
            &sources,
            &variables,
            &domain,
            restored.completed_witness().unwrap(),
            &mut Unbounded
        )
        .unwrap()
        .is_root_free()
    );

    let source = Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::Pow(Arc::new(Expr::Sym(x)), Arc::new(Expr::from_i64(-1))),
    ]);
    let sources = [source];
    let mut search =
        RootIsolationSearch::new(&sources, &variables, &domain, options(), &mut Unbounded).unwrap();
    search.advance(2, &mut Unbounded).unwrap();
    let saved = search.checkpoint(&mut Unbounded).unwrap();
    let restored =
        RootIsolationSearch::from_checkpoint(&sources, &variables, &domain, &saved, &mut Unbounded)
            .unwrap();
    assert_eq!(restored.progress(), search.progress());
    assert!(!restored.progress().ready_for_verification);
    assert!(restored.completed_witness().is_none());
}

#[test]
fn source_domain_and_typed_axis_changes_are_rechecked() {
    let original = completed_checkpoint();
    let (sources, variables, domain) = fixture();
    let other = [Symbol::with_identity(
        "x",
        SymbolIdentity::from_assumptions_digest([1; 32]),
    )];
    assert!(
        RootIsolationSearch::from_checkpoint(&sources, &other, &domain, &original, &mut Unbounded)
            .is_err()
    );
    assert!(
        RootIsolationSearch::from_checkpoint(
            &[square(&variables[0], 1)],
            &variables,
            &domain,
            &original,
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        RootIsolationSearch::from_checkpoint(
            &sources,
            &variables,
            &[interval(2, 3)],
            &original,
            &mut Unbounded
        )
        .is_err()
    );
    // The native checkpoint intentionally contains no authorization or source
    // digest. The transport must bind exact raw sources/context before here;
    // this assertion establishes the independent mathematical replay itself.
}

struct CancelMeter<'a> {
    ledger: &'a mut Budget,
    count: usize,
    at: Option<usize>,
}

impl BudgetMeter for CancelMeter<'_> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.ledger.charge(dimension, amount)
    }
    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        self.ledger.charge_batch(charges)
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.count += 1;
        if self.at == Some(self.count) {
            Err(MeterError::Cancelled)
        } else {
            self.ledger.checkpoint()
        }
    }
}

#[test]
fn restore_cancellation_preserves_input_and_charges_each_receiver_attempt() {
    let original = completed_checkpoint();
    let (sources, variables, domain) = fixture();
    let mut ledger = budget();
    let mut probe = CancelMeter {
        ledger: &mut ledger,
        count: 0,
        at: None,
    };
    RootIsolationSearch::from_checkpoint(&sources, &variables, &domain, &original, &mut probe)
        .unwrap();
    let total = probe.count;
    for at in [1, 2, total / 4, total / 2, total - 1, total] {
        let mut meter = CancelMeter {
            ledger: &mut ledger,
            count: 0,
            at: Some(at),
        };
        let before = meter.ledger.remaining(Dimension::ComputeSteps);
        let saved = original.clone();
        assert!(matches!(
            RootIsolationSearch::from_checkpoint(
                &sources, &variables, &domain, &original, &mut meter
            ),
            Err(RootIsolationError::Verification(RootBoxError::Meter(
                MeterError::Cancelled
            )))
        ));
        assert_eq!(meter.count, at);
        assert_eq!(original, saved);
        assert!(meter.ledger.remaining(Dimension::ComputeSteps) <= before);
        assert_eq!(meter.ledger.verifier_remaining(), 100_000_000);
    }
    assert!(
        RootIsolationSearch::from_checkpoint(&sources, &variables, &domain, &original, &mut ledger)
            .is_ok()
    );
}

#[test]
fn exhausted_receiver_budget_stays_exhausted_on_repeated_import() {
    let original = completed_checkpoint();
    let (sources, variables, domain) = fixture();
    let mut limits = BudgetLimits::uniform(1 << 35, 100_000_000);
    limits.dimensions[Dimension::ComputeSteps.index()] = 1;
    let mut ledger = Budget::new(limits);
    for _ in 0..3 {
        assert!(matches!(
            RootIsolationSearch::from_checkpoint(
                &sources,
                &variables,
                &domain,
                &original,
                &mut ledger
            ),
            Err(RootIsolationError::Verification(RootBoxError::Meter(
                MeterError::Budget(_)
            )))
        ));
        assert_eq!(ledger.remaining(Dimension::ComputeSteps), 0);
        assert_eq!(ledger.verifier_remaining(), 100_000_000);
    }
}
