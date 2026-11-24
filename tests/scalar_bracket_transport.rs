//! Independent root-existence replay binds the full receiver statement.

use frankensympy::certification::{CertificationError, ResidualProblem};
use frankensympy::scalar_bracket::{
    MAX_SCALAR_ROOT_BRACKET_BYTES, SCALAR_ROOT_BRACKET_SCHEMA_VERSION, ScalarBracketError,
    export_scalar_root_bracket, replay_scalar_root_bracket,
};
use fsym_budget::{
    Budget, BudgetLimits, BudgetMeter, DIMENSION_COUNT, Dimension, MeterError, Unbounded,
};
use fsym_calculus::scalar_bracket::ScalarRootBracketOptions;
use fsym_core::{
    BigInt, BigRational, Expr, MAX_SCALAR_ROOT_INPUT_BITS, RealBall, RootBoxError,
    ScalarRootBracketError, ScalarRootBracketReceipt, ScalarRootBracketWitness, Symbol,
    SymbolIdentity,
};
use serde::Serialize;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn problem(source: &str) -> ResidualProblem {
    ResidualProblem::parse_sources(&[source.into()], &["x".into()], &mut Unbounded).unwrap()
}

fn options() -> ScalarRootBracketOptions {
    ScalarRootBracketOptions {
        precision_digits: 24,
        max_bisections: 64,
    }
}

fn checked(
    problem: &ResidualProblem,
    lower: &BigRational,
    upper: &BigRational,
    tolerance: &BigRational,
) -> ScalarRootBracketReceipt {
    let witness = problem
        .propose_scalar_root_bracket(lower, upper, tolerance, options(), &mut Unbounded)
        .unwrap();
    problem
        .verify_scalar_root_bracket(lower, upper, tolerance, &witness, &mut Unbounded)
        .unwrap()
}

#[derive(Serialize)]
struct Proposal<'a> {
    schema_version: u32,
    context_digest: [u8; 32],
    residuals: &'a [Expr],
    variables: &'a [Symbol],
    lower: &'a BigRational,
    upper: &'a BigRational,
    absolute_tolerance: &'a BigRational,
    witness: &'a ScalarRootBracketWitness,
}

fn proposed_bytes(
    problem: &ResidualProblem,
    lower: &BigRational,
    upper: &BigRational,
    tolerance: &BigRational,
    witness: &ScalarRootBracketWitness,
) -> Vec<u8> {
    serde_json::to_vec(&Proposal {
        schema_version: SCALAR_ROOT_BRACKET_SCHEMA_VERSION,
        context_digest: [7; 32],
        residuals: problem.residuals(),
        variables: problem.variables(),
        lower,
        upper,
        absolute_tolerance: tolerance,
        witness,
    })
    .unwrap()
}

fn replay(
    bytes: &[u8],
    problem: &ResidualProblem,
) -> Result<ScalarRootBracketReceipt, ScalarBracketError> {
    replay_scalar_root_bracket(
        bytes,
        problem,
        &q(0, 1),
        &q(1, 1),
        &q(1, 1000),
        [7; 32],
        &mut Unbounded,
    )
}

#[test]
fn odd_repeated_root_roundtrip_recomputes_an_exact_enclosure() {
    let original = problem("(x-1/3)^3");
    let accepted = checked(&original, &q(0, 1), &q(1, 1), &q(1, 1000));
    // This polynomial's zero is 1/3, independently of the interval engine.
    assert!(accepted.root_enclosure().contains(&q(1, 3)));
    assert!(accepted.root_enclosure().width() <= q(1, 1000));
    let bytes = export_scalar_root_bracket(&original, &accepted, [7; 32], &mut Unbounded).unwrap();
    let fresh = problem("(x-1/3)^3");
    let replayed = replay(&bytes, &fresh).unwrap();
    assert_eq!(replayed, accepted);
    assert_eq!(
        export_scalar_root_bracket(&fresh, &replayed, [7; 32], &mut Unbounded).unwrap(),
        bytes
    );
    let decoded: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(decoded.as_object().unwrap().len(), 8);
    for key in [
        "accepted",
        "verified",
        "unique",
        "multiplicity",
        "enclosure",
        "endpoint_values",
        "root",
    ] {
        assert!(decoded.get(key).is_none(), "{key}");
    }
}

#[test]
fn nonsmooth_and_endpoint_roots_do_not_require_a_nonzero_derivative() {
    for (source, lower, upper, exact) in [
        ("x+abs(x)/2", q(-1, 1), q(1, 1), q(0, 1)),
        ("abs(x)", q(0, 1), q(1, 1), q(0, 1)),
        ("(x-1)^4", q(0, 1), q(1, 1), q(1, 1)),
        ("(x-1/3)^2", q(1, 3), q(1, 3), q(1, 3)),
    ] {
        let original = problem(source);
        let accepted = checked(&original, &lower, &upper, &q(1, 1000));
        let bytes =
            export_scalar_root_bracket(&original, &accepted, [7; 32], &mut Unbounded).unwrap();
        let replayed = replay_scalar_root_bracket(
            &bytes,
            &problem(source),
            &lower,
            &upper,
            &q(1, 1000),
            [7; 32],
            &mut Unbounded,
        )
        .unwrap();
        assert_eq!(
            replayed.root_enclosure(),
            &RealBall::new(exact, q(0, 1)).unwrap()
        );
        assert_eq!(accepted, replayed);
    }
}

#[test]
fn an_existence_receipt_may_contain_several_roots_without_claiming_completeness() {
    let problem = problem("(x+1)*x*(x-1)");
    let witness = ScalarRootBracketWitness {
        lower: q(-2, 1),
        upper: q(2, 1),
        precision_digits: 24,
    };
    let bytes = proposed_bytes(&problem, &q(-2, 1), &q(2, 1), &q(4, 1), &witness);
    let receipt = replay_scalar_root_bracket(
        &bytes,
        &problem,
        &q(-2, 1),
        &q(2, 1),
        &q(4, 1),
        [7; 32],
        &mut Unbounded,
    )
    .unwrap();
    for root in [-1, 0, 1] {
        assert!(receipt.root_enclosure().contains(&q(root, 1)));
    }
    assert_eq!(receipt.root_enclosure().width(), q(4, 1));
}

#[test]
fn no_sign_change_and_insufficient_search_remain_distinct_failures() {
    let even_root = problem("(x-1/3)^2");
    assert!(matches!(
        even_root.propose_scalar_root_bracket(
            &q(0, 1),
            &q(1, 1),
            &q(1, 1000),
            options(),
            &mut Unbounded,
        ),
        Err(ScalarBracketError::Bracket(
            ScalarRootBracketError::NoSignChange
        ))
    ));
    assert!(matches!(
        problem("x-1/3").propose_scalar_root_bracket(
            &q(0, 1),
            &q(1, 1),
            &q(1, 1000),
            ScalarRootBracketOptions {
                max_bisections: 0,
                ..options()
            },
            &mut Unbounded,
        ),
        Err(ScalarBracketError::Bracket(
            ScalarRootBracketError::AccuracyNotReached
        ))
    ));
}

#[test]
fn wrong_residual_or_variable_count_refuses_before_generation_and_replay() {
    let original = problem("x");
    let accepted = checked(&original, &q(0, 1), &q(1, 1), &q(1, 1000));
    let bytes = export_scalar_root_bracket(&original, &accepted, [7; 32], &mut Unbounded).unwrap();
    for (sources, variables) in [
        (vec!["x".into(), "x^2".into()], vec!["x".into()]),
        (vec!["x+y".into()], vec!["x".into(), "y".into()]),
    ] {
        let target = ResidualProblem::parse_sources(&sources, &variables, &mut Unbounded).unwrap();
        assert!(matches!(
            target.propose_scalar_root_bracket(
                &q(0, 1),
                &q(1, 1),
                &q(1, 1000),
                options(),
                &mut Unbounded,
            ),
            Err(ScalarBracketError::ProblemDimension)
        ));
        assert!(matches!(
            target.verify_scalar_root_bracket(
                &q(0, 1),
                &q(1, 1),
                &q(1, 1000),
                accepted.witness(),
                &mut Unbounded,
            ),
            Err(ScalarBracketError::ProblemDimension)
        ));
        assert!(matches!(
            replay(&bytes, &target),
            Err(ScalarBracketError::ProblemDimension)
        ));
        assert!(matches!(
            export_scalar_root_bracket(&target, &accepted, [7; 32], &mut Unbounded),
            Err(ScalarBracketError::ProblemDimension)
        ));
    }
}

#[test]
fn raw_source_structure_and_same_name_typed_coordinates_are_bound() {
    let original = problem("x");
    let accepted = checked(&original, &q(0, 1), &q(1, 1), &q(1, 1000));
    let bytes = export_scalar_root_bracket(&original, &accepted, [7; 32], &mut Unbounded).unwrap();
    let symbol = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([3; 32]));
    let typed = ResidualProblem::new(
        vec![Expr::Sym(symbol.clone())],
        vec![symbol],
        &mut Unbounded,
    )
    .unwrap();
    for different in [problem("x+0"), problem("2*x"), typed] {
        assert!(matches!(
            replay(&bytes, &different),
            Err(ScalarBracketError::Certification(
                CertificationError::SourceMismatch
            ))
        ));
        assert!(matches!(
            export_scalar_root_bracket(&different, &accepted, [7; 32], &mut Unbounded),
            Err(ScalarBracketError::Certification(
                CertificationError::SourceMismatch
            ))
        ));
    }
}

#[test]
fn mismatched_context_source_and_statement_precede_untrusted_witness_decoding() {
    let original = problem("x");
    let accepted = checked(&original, &q(0, 1), &q(1, 1), &q(1, 1000));
    let bytes = export_scalar_root_bracket(&original, &accepted, [7; 32], &mut Unbounded).unwrap();
    let mut object: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    object["witness"] = serde_json::json!({"lower": "invalid rational"});
    let malformed = serde_json::to_vec(&object).unwrap();
    for (lower, upper, tolerance) in [
        (q(-1, 1), q(1, 1), q(1, 1000)),
        (q(0, 1), q(2, 1), q(1, 1000)),
        (q(0, 1), q(1, 1), q(1, 100)),
    ] {
        assert!(matches!(
            replay_scalar_root_bracket(
                &malformed,
                &original,
                &lower,
                &upper,
                &tolerance,
                [7; 32],
                &mut Unbounded,
            ),
            Err(ScalarBracketError::StatementMismatch)
        ));
    }
    assert!(matches!(
        replay_scalar_root_bracket(
            &malformed,
            &original,
            &q(0, 1),
            &q(1, 1),
            &q(1, 1000),
            [8; 32],
            &mut Unbounded,
        ),
        Err(ScalarBracketError::Certification(
            CertificationError::UniverseMismatch { .. }
        ))
    ));
    assert!(matches!(
        replay(&malformed, &problem("x+0")),
        Err(ScalarBracketError::Certification(
            CertificationError::SourceMismatch
        ))
    ));
}

#[test]
fn canonical_witnesses_still_need_containment_width_and_root_evidence() {
    let problem = problem("x-1/3");
    for (lower, upper, expected) in [
        (q(-1, 1), q(1, 1), "containment"),
        (q(0, 1), q(2, 1), "containment"),
        (q(0, 1), q(1, 1), "width"),
        (q(1, 2), q(1, 2), "sign"),
    ] {
        let witness = ScalarRootBracketWitness {
            lower,
            upper,
            precision_digits: 24,
        };
        let bytes = proposed_bytes(&problem, &q(0, 1), &q(1, 1), &q(1, 1000), &witness);
        let error = replay(&bytes, &problem).unwrap_err();
        match expected {
            "containment" => assert!(matches!(
                error,
                ScalarBracketError::Bracket(ScalarRootBracketError::InvalidBracket(_))
            )),
            "width" => assert!(matches!(
                error,
                ScalarBracketError::Bracket(ScalarRootBracketError::AccuracyNotReached)
            )),
            "sign" => assert!(matches!(
                error,
                ScalarBracketError::Bracket(ScalarRootBracketError::NoSignChange)
            )),
            _ => unreachable!(),
        }
    }
}

#[test]
fn narrow_valid_roots_cannot_hide_a_pole_in_the_original_interval() {
    // x=1/2 is a root with a regular neighborhood, but the full requested
    // interval includes a reciprocal hole at zero. The receiver must reject.
    let source = problem("(x-1/2)/x");
    let witness = ScalarRootBracketWitness {
        lower: q(1, 2),
        upper: q(1, 2),
        precision_digits: 24,
    };
    let bytes = proposed_bytes(&source, &q(0, 1), &q(1, 1), &q(1, 1000), &witness);
    assert!(matches!(
        replay(&bytes, &source),
        Err(ScalarBracketError::Bracket(
            ScalarRootBracketError::Enclosure(RootBoxError::Domain(_))
        ))
    ));
    for expression in ["0*(1/x)", "x+(1/x)^0-1", "x+log(x)-log(x)"] {
        let source = problem(expression);
        let witness = ScalarRootBracketWitness {
            lower: q(0, 1),
            upper: q(0, 1),
            precision_digits: 24,
        };
        let bytes = proposed_bytes(&source, &q(0, 1), &q(1, 1), &q(1, 1000), &witness);
        assert!(
            matches!(
                replay(&bytes, &source),
                Err(ScalarBracketError::Bracket(
                    ScalarRootBracketError::Enclosure(_)
                ))
            ),
            "{expression}"
        );
    }
}

#[test]
fn changed_endpoint_witnesses_are_recomputed_without_a_generator_retry() {
    let source = problem("(x-1/3)^3");
    // The original statement has a root; a forged point at 1/2 does not. An
    // implementation that silently reruns candidate search would accept it.
    let witness = ScalarRootBracketWitness {
        lower: q(1, 2),
        upper: q(1, 2),
        precision_digits: 24,
    };
    let bytes = proposed_bytes(&source, &q(0, 1), &q(1, 1), &q(1, 1000), &witness);
    assert!(matches!(
        replay(&bytes, &source),
        Err(ScalarBracketError::Bracket(
            ScalarRootBracketError::NoSignChange
        ))
    ));
}

#[test]
fn unknown_versions_fields_duplicate_keys_and_noncanonical_bytes_refuse() {
    let source = problem("x");
    let accepted = checked(&source, &q(0, 1), &q(1, 1), &q(1, 1000));
    let bytes = export_scalar_root_bracket(&source, &accepted, [7; 32], &mut Unbounded).unwrap();
    let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let mut future = original.clone();
    future["schema_version"] = serde_json::json!(SCALAR_ROOT_BRACKET_SCHEMA_VERSION + 1);
    assert!(matches!(
        replay(&serde_json::to_vec(&future).unwrap(), &source),
        Err(ScalarBracketError::Certification(
            CertificationError::UnsupportedVersion { .. }
        ))
    ));
    for key in ["accepted", "unique", "endpoint_values", "root_enclosure"] {
        let mut extra = original.clone();
        extra[key] = serde_json::json!(true);
        assert!(matches!(
            replay(&serde_json::to_vec(&extra).unwrap(), &source),
            Err(ScalarBracketError::Certification(
                CertificationError::Encoding(_)
            ))
        ));
        let mut extra = original.clone();
        extra["witness"][key] = serde_json::json!(true);
        assert!(matches!(
            replay(&serde_json::to_vec(&extra).unwrap(), &source),
            Err(ScalarBracketError::Certification(
                CertificationError::Encoding(_)
            ))
        ));
    }
    let duplicate = format!(
        "{{\"schema_version\":1,{}",
        std::str::from_utf8(&bytes[1..]).unwrap()
    );
    assert!(matches!(
        replay(duplicate.as_bytes(), &source),
        Err(ScalarBracketError::Certification(
            CertificationError::Encoding(_)
        ))
    ));
    let mut padded = bytes.clone();
    padded.push(b'\n');
    assert!(matches!(
        replay(&padded, &source),
        Err(ScalarBracketError::Certification(
            CertificationError::NonCanonical
        ))
    ));
    // Explicitly exchange the first fields rather than depending on whether
    // serde_json's transitive feature set preserves object insertion order.
    let fields = [
        "context_digest",
        "schema_version",
        "residuals",
        "variables",
        "lower",
        "upper",
        "absolute_tolerance",
        "witness",
    ];
    let reordered = format!(
        "{{{}}}",
        fields
            .iter()
            .map(|key| format!("{key:?}:{}", original[*key]))
            .collect::<Vec<_>>()
            .join(",")
    );
    assert_ne!(reordered.as_bytes(), bytes);
    assert!(matches!(
        replay(reordered.as_bytes(), &source),
        Err(ScalarBracketError::Certification(
            CertificationError::NonCanonical
        ))
    ));
}

#[test]
fn oversized_deep_and_invalid_witness_payloads_refuse() {
    let source = problem("x");
    assert!(matches!(
        replay(&vec![b' '; MAX_SCALAR_ROOT_BRACKET_BYTES + 1], &source),
        Err(ScalarBracketError::Certification(
            CertificationError::ByteLimit { .. }
        ))
    ));
    let deep = format!("{}0{}", "[".repeat(100), "]".repeat(100));
    assert!(matches!(
        replay(deep.as_bytes(), &source),
        Err(ScalarBracketError::Certification(
            CertificationError::DepthLimit { .. }
        ))
    ));
    for witness in [
        ScalarRootBracketWitness {
            lower: q(1, 1),
            upper: q(0, 1),
            precision_digits: 24,
        },
        ScalarRootBracketWitness {
            lower: q(0, 1),
            upper: q(0, 1),
            precision_digits: 7,
        },
        ScalarRootBracketWitness {
            lower: q(0, 1),
            upper: q(0, 1),
            precision_digits: 129,
        },
        ScalarRootBracketWitness {
            lower: q(0, 1),
            upper: BigRational::from_integer(BigInt::from(1) << MAX_SCALAR_ROOT_INPUT_BITS as u32),
            precision_digits: 24,
        },
    ] {
        let bytes = proposed_bytes(&source, &q(0, 1), &q(1, 1), &q(1, 1000), &witness);
        assert!(matches!(
            replay(&bytes, &source),
            Err(ScalarBracketError::Certification(
                CertificationError::Encoding(_)
            ))
        ));
    }
}

#[test]
fn malformed_or_noncanonical_rationals_never_reach_mathematical_verification() {
    let source = problem("x");
    let accepted = checked(&source, &q(0, 1), &q(1, 1), &q(1, 1000));
    let bytes = export_scalar_root_bracket(&source, &accepted, [7; 32], &mut Unbounded).unwrap();
    let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    for rational in [
        serde_json::json!([[1, [2]], [1, [2]]]),
        serde_json::json!([[1, [1]], [-1, [1]]]),
        serde_json::json!([[0, [1]], [1, [1]]]),
        serde_json::json!([[1, [1]], [0, []]]),
    ] {
        let mut bad = original.clone();
        bad["witness"]["lower"] = rational;
        assert!(matches!(
            replay(&serde_json::to_vec(&bad).unwrap(), &source),
            Err(ScalarBracketError::Certification(
                CertificationError::Encoding(_)
            ))
        ));
    }
}

#[test]
fn invalid_expected_intervals_and_tolerances_refuse_even_for_valid_bytes() {
    let source = problem("x");
    let accepted = checked(&source, &q(0, 1), &q(1, 1), &q(1, 1000));
    let bytes = export_scalar_root_bracket(&source, &accepted, [7; 32], &mut Unbounded).unwrap();
    for (lower, upper, tolerance) in [
        (q(1, 1), q(0, 1), q(1, 1000)),
        (q(0, 1), q(1, 1), q(0, 1)),
        (q(0, 1), q(1, 1), q(-1, 1000)),
    ] {
        assert!(matches!(
            replay_scalar_root_bracket(
                &bytes,
                &source,
                &lower,
                &upper,
                &tolerance,
                [7; 32],
                &mut Unbounded
            ),
            Err(ScalarBracketError::Bracket(
                ScalarRootBracketError::InvalidBracket(_)
                    | ScalarRootBracketError::InvalidTolerance
            ))
        ));
    }
}

#[test]
fn generation_cannot_consume_the_independent_verifiers_reserved_pool() {
    let source = problem("(x-1/3)^3");
    let mut budget = Budget::new(BudgetLimits::uniform(1 << 40, 1 << 32));
    let lease = budget.verifier_lease().unwrap();
    let protected_before = budget.verifier_remaining();
    let witness = source
        .propose_scalar_root_bracket(&q(0, 1), &q(1, 1), &q(1, 1000), options(), &mut budget)
        .unwrap();
    assert_eq!(budget.verifier_remaining(), protected_before);
    let ordinary_remaining = budget.remaining(Dimension::ComputeSteps);
    assert!(ordinary_remaining < 1 << 40);
    budget
        .try_charge(Dimension::ComputeSteps, ordinary_remaining)
        .unwrap();
    assert!(
        source
            .propose_scalar_root_bracket(&q(0, 1), &q(1, 1), &q(1, 1000), options(), &mut budget,)
            .is_err()
    );
    let receipt = source
        .verify_scalar_root_bracket(
            &q(0, 1),
            &q(1, 1),
            &q(1, 1000),
            &witness,
            &mut budget.verifier_meter(&lease).unwrap(),
        )
        .unwrap();
    assert!(receipt.root_enclosure().contains(&q(1, 3)));
    assert_eq!(budget.remaining(Dimension::ComputeSteps), 0);
    assert!(budget.verifier_remaining() < protected_before);
}

#[derive(Default)]
struct Counted {
    charges: [u64; DIMENSION_COUNT],
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for Counted {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charges[dimension.index()] += amount;
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
fn repeated_replay_exhausts_one_cumulative_ledger_without_refunds() {
    let source = problem("x");
    let accepted = checked(&source, &q(0, 1), &q(1, 1), &q(1, 1000));
    let bytes = export_scalar_root_bracket(&source, &accepted, [7; 32], &mut Unbounded).unwrap();
    let mut counter = Counted::default();
    replay_scalar_root_bracket(
        &bytes,
        &source,
        &q(0, 1),
        &q(1, 1),
        &q(1, 1000),
        [7; 32],
        &mut counter,
    )
    .unwrap();
    let mut budget = Budget::new(BudgetLimits {
        dimensions: counter.charges.map(|value| value * 2),
        verifier_pool: 193,
    });
    for _ in 0..2 {
        replay_scalar_root_bracket(
            &bytes,
            &source,
            &q(0, 1),
            &q(1, 1),
            &q(1, 1000),
            [7; 32],
            &mut budget,
        )
        .unwrap();
    }
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        assert!(counter.charges[dimension.index()] > 0);
        assert_eq!(budget.remaining(dimension), 0);
    }
    assert!(
        replay_scalar_root_bracket(
            &bytes,
            &source,
            &q(0, 1),
            &q(1, 1),
            &q(1, 1000),
            [7; 32],
            &mut budget,
        )
        .is_err()
    );
    assert_eq!(budget.verifier_remaining(), 193);
}

fn assert_cancelled(error: ScalarBracketError) {
    assert!(matches!(
        error,
        ScalarBracketError::Certification(CertificationError::Meter(MeterError::Cancelled))
            | ScalarBracketError::Bracket(ScalarRootBracketError::Enclosure(RootBoxError::Meter(
                MeterError::Cancelled
            )))
    ));
}

fn cancel_each_phase<T>(mut operation: impl FnMut(&mut Counted) -> Result<T, ScalarBracketError>) {
    let mut counter = Counted::default();
    assert!(operation(&mut counter).is_ok());
    assert!(counter.checkpoints > 3);
    for cancel_at in [1, counter.checkpoints / 2, counter.checkpoints] {
        let mut cancelled = Counted {
            cancel_at: Some(cancel_at),
            ..Counted::default()
        };
        let error = match operation(&mut cancelled) {
            Err(error) => error,
            Ok(_) => panic!("cancellation at checkpoint {cancel_at} published a result"),
        };
        assert_cancelled(error);
        assert_eq!(cancelled.checkpoints, cancel_at);
    }
}

#[test]
fn cancellation_before_during_and_at_final_publication_returns_no_receipt_or_bytes() {
    let source = problem("x");
    let accepted = checked(&source, &q(0, 1), &q(1, 1), &q(1, 1000));
    let bytes = export_scalar_root_bracket(&source, &accepted, [7; 32], &mut Unbounded).unwrap();
    cancel_each_phase(|meter| {
        source.propose_scalar_root_bracket(&q(0, 1), &q(1, 1), &q(1, 1000), options(), meter)
    });
    cancel_each_phase(|meter| {
        source.verify_scalar_root_bracket(
            &q(0, 1),
            &q(1, 1),
            &q(1, 1000),
            accepted.witness(),
            meter,
        )
    });
    cancel_each_phase(|meter| export_scalar_root_bracket(&source, &accepted, [7; 32], meter));
    cancel_each_phase(|meter| {
        replay_scalar_root_bracket(
            &bytes,
            &source,
            &q(0, 1),
            &q(1, 1),
            &q(1, 1000),
            [7; 32],
            meter,
        )
    });
}
