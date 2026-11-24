//! Wide numerical objective registration keeps exact certificate limits intact.

use frankensympy::{
    CertificationError, ObjectiveProblem, export_objective_program, replay_objective_program,
};
use fsym_budget::{
    Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded,
};
use fsym_calculus::VerifiedObjective;
use fsym_core::root_certificate::artifact::{RootCertificateSources, check_dimensions};
use fsym_core::{
    BigInt, BigRational, Expr, RealBall, RootBoxError, RootBoxWitness, Symbol, SymbolIdentity,
};
use fsym_numeric_consumer::{MinimizeOptions, sparse_minimize};

fn names(n: usize) -> Vec<String> {
    (0..n).map(|index| format!("v{index}")).collect()
}

fn symbols(n: usize) -> Vec<Symbol> {
    names(n).into_iter().map(Symbol::new).collect()
}

#[test]
fn sixty_four_axes_compile_replay_and_minimize_without_dense_registration_limits() {
    let problem = ObjectiveProblem::parse_source("(v0-3)^2", &names(64), &mut Unbounded).unwrap();
    let program = VerifiedObjective::try_compile(problem.objective(), problem.variables()).unwrap();
    let artifact = export_objective_program(&program, [9; 32], &mut Unbounded).unwrap();
    let replayed = replay_objective_program(&artifact, &problem, [9; 32], &mut Unbounded).unwrap();
    assert_eq!(problem.variables().len(), 64);
    assert_eq!(replayed.variables(), problem.variables());
    assert_eq!(replayed.objective(), problem.objective());
    for source in [&program, &replayed] {
        let result = sparse_minimize(source, &[7.0; 64], &MinimizeOptions::default()).unwrap();
        assert!((result.point[0] - 3.0).abs() <= 1e-10);
        assert!(result.point.iter().skip(1).all(|value| *value == 7.0));
        assert!(result.gradient_norm <= 1e-10);
    }
}

#[test]
fn wide_typed_same_name_axes_keep_distinct_identities_and_order() {
    let first = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    let second = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32]));
    let source = Expr::Add(vec![
        Expr::Sym(first.clone()),
        Expr::Mul(vec![Expr::from_i64(2), Expr::Sym(second.clone())]),
    ]);
    let mut variables = vec![second.clone(), first.clone()];
    variables.extend(symbols(15));
    let problem = ObjectiveProblem::new(source.clone(), variables.clone(), &mut Unbounded).unwrap();
    assert_eq!(problem.objective(), &source);
    assert_eq!(problem.variables(), variables);
    let program = VerifiedObjective::try_compile(problem.objective(), problem.variables()).unwrap();
    let mut point = vec![0.0; 17];
    point[0] = 3.0;
    point[1] = 7.0;
    let result = program.evaluate(&point).unwrap();
    assert_eq!(result.value, 13.0);
    assert_eq!(&result.gradient[..2], &[2.0, 1.0]);
    assert!(result.gradient[2..].iter().all(|value| *value == 0.0));

    let mut missing = variables;
    missing[0] = Symbol::new("x");
    let unbound = ObjectiveProblem::new(source, missing, &mut Unbounded).unwrap();
    // Source registration, as in the original narrow API, does not silently
    // rebind an undeclared identity. Executable compilation must reject it.
    assert!(VerifiedObjective::try_compile(unbound.objective(), unbound.variables()).is_err());
}

#[test]
fn wide_registration_rejects_duplicate_axes_and_invalid_parsed_names() {
    let mut duplicate = symbols(17);
    duplicate[16] = duplicate[0].clone();
    assert_eq!(
        ObjectiveProblem::new(Expr::from_i64(0), duplicate, &mut Unbounded).unwrap_err(),
        CertificationError::Root(RootBoxError::DuplicateVariable)
    );
    for name in [
        "", "1x", "x-y", "pi", "Pi", "E", "I", "oo", "zoo", "nan", "NaN",
    ] {
        let mut variables = names(17);
        variables[16] = name.into();
        assert_eq!(
            ObjectiveProblem::parse_source("v0", &variables, &mut Unbounded).unwrap_err(),
            CertificationError::InvalidVariable
        );
    }
    let mut variables = names(17);
    variables[16] = "a".repeat(4097);
    assert_eq!(
        ObjectiveProblem::parse_source("v0", &variables, &mut Unbounded).unwrap_err(),
        CertificationError::InvalidVariable
    );
}

#[test]
fn duplicate_detection_accounts_for_long_equal_names_with_distinct_identities() {
    let name = "x".repeat(4096);
    let variables: Vec<_> = (0u8..64)
        .map(|identity| {
            Symbol::with_identity(
                name.clone(),
                SymbolIdentity::from_assumptions_digest([identity; 32]),
            )
        })
        .collect();
    let source = Expr::Sym(variables[0].clone());
    assert!(ObjectiveProblem::new(source.clone(), variables.clone(), &mut Unbounded).is_ok());
    // Pairwise equality must inspect megabytes of matching name bytes before
    // reaching the distinct identities. A small compute budget cannot admit
    // that work merely because the coordinate count itself is at most 64.
    let mut limits = BudgetLimits::uniform(1_000_000_000, 29);
    limits.dimensions[Dimension::ComputeSteps.index()] = 100_000;
    let mut budget = Budget::new(limits);
    assert!(matches!(
        ObjectiveProblem::new(source, variables, &mut budget),
        Err(CertificationError::Meter(MeterError::Budget(
            BudgetError::Exhausted {
                dimension: Dimension::ComputeSteps,
                ..
            }
        )))
    ));
    assert_eq!(budget.verifier_remaining(), 29);
}

#[test]
fn wide_registered_sources_retain_held_domain_obligations() {
    for source in ["v0^2+0*(1/v0)", "log(v0)-log(v0)", "(1/v0)^0", "sqrt(v0)"] {
        let problem = ObjectiveProblem::parse_source(source, &names(17), &mut Unbounded).unwrap();
        let program =
            VerifiedObjective::try_compile(problem.objective(), problem.variables()).unwrap();
        assert!(program.evaluate(&[0.0; 17]).is_err(), "{source}");
    }
}

#[test]
fn existing_narrow_exact_sources_keep_their_larger_admission_envelope() {
    let large = Expr::Add(vec![Expr::Sym(Symbol::new("v0")); 300]);
    let deep_scalar = Expr::Integer(BigInt::from(1u64) << 5000u32);
    for source in [large, deep_scalar] {
        for n in [1, 17] {
            let problem =
                ObjectiveProblem::new(source.clone(), symbols(n), &mut Unbounded).unwrap();
            assert_eq!(problem.objective(), &source);
            assert!(
                VerifiedObjective::try_compile(problem.objective(), problem.variables()).is_err()
            );
        }
    }
}

#[test]
fn source_only_preflight_does_not_widen_exact_root_or_objective_certificates() {
    let problem = ObjectiveProblem::parse_source("v0^2", &names(17), &mut Unbounded).unwrap();
    assert_eq!(
        check_dimensions(1, 17).unwrap_err(),
        CertificationError::Root(RootBoxError::DimensionMismatch)
    );
    assert!(
        RootCertificateSources::new(
            vec![problem.objective().clone()],
            problem.variables().to_vec(),
            &mut Unbounded,
        )
        .is_err()
    );
    let domain = vec![RealBall::from_i64(0); 17];
    assert_eq!(
        problem.enclose(&domain, 24, &mut Unbounded).unwrap_err(),
        CertificationError::Root(RootBoxError::DimensionMismatch)
    );
    let witness = RootBoxWitness {
        domain,
        center: vec![BigRational::from_integer(BigInt::from(0)); 17],
        preconditioner: vec![vec![BigRational::from_integer(BigInt::from(0)); 17]; 17],
        precision_digits: 24,
    };
    assert!(problem.verify_stationary(&witness, &mut Unbounded).is_err());
}

#[test]
fn oversized_shape_payload_and_raw_sources_refuse_before_compilation() {
    for count in [0, 65] {
        assert!(ObjectiveProblem::parse_source("0", &names(count), &mut Unbounded).is_err());
        assert!(ObjectiveProblem::new(Expr::from_i64(0), symbols(count), &mut Unbounded).is_err());
    }
    assert!(ObjectiveProblem::parse_source(&"v".repeat(4097), &names(17), &mut Unbounded).is_err());
    let mut too_deep = Expr::Sym(Symbol::new("v0"));
    for _ in 0..33 {
        too_deep = Expr::Function("sin".into(), vec![too_deep]);
    }
    let too_wide = Expr::Add(vec![Expr::from_i64(0); 16_384]);
    let too_tall = Expr::Integer(BigInt::from_bytes_le(&vec![255; 16_385]));
    let long_name = Expr::Sym(Symbol::new("x".repeat(4097)));
    for source in [too_deep, too_wide, too_tall, long_name] {
        assert!(ObjectiveProblem::new(source, symbols(17), &mut Unbounded).is_err());
    }
}

#[derive(Default)]
struct Recorder {
    checkpoints: usize,
    stop: Option<usize>,
    stopped: bool,
    totals: [u64; 3],
}

impl BudgetMeter for Recorder {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.stopped, "continued after cancellation");
        if dimension.index() < self.totals.len() {
            self.totals[dimension.index()] += amount;
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
        assert!(!self.stopped, "continued after cancellation");
        self.checkpoints += 1;
        if self.stop == Some(self.checkpoints) {
            self.stopped = true;
            return Err(MeterError::Cancelled);
        }
        Ok(())
    }
}

#[test]
fn every_wide_registration_safe_point_preserves_terminal_cancellation() {
    let variables = names(64);
    let mut measured = Recorder::default();
    ObjectiveProblem::parse_source("(v0-3)^2", &variables, &mut measured).unwrap();
    for stop in 1..=measured.checkpoints {
        let mut meter = Recorder {
            stop: Some(stop),
            ..Recorder::default()
        };
        assert_eq!(
            ObjectiveProblem::parse_source("(v0-3)^2", &variables, &mut meter).unwrap_err(),
            CertificationError::Meter(MeterError::Cancelled)
        );
        assert!(meter.stopped);
        assert_eq!(meter.checkpoints, stop);
    }
}

#[test]
fn wide_registration_uses_one_cumulative_ledger_in_every_charged_dimension() {
    let variables = names(64);
    let mut measured = Recorder::default();
    ObjectiveProblem::parse_source("(v0-3)^2", &variables, &mut measured).unwrap();
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut limits = BudgetLimits::uniform(1_000_000_000, 29);
        limits.dimensions[dimension.index()] = measured.totals[dimension.index()];
        let mut exact = Budget::new(limits);
        ObjectiveProblem::parse_source("(v0-3)^2", &variables, &mut exact).unwrap();
        assert_eq!(exact.remaining(dimension), 0);
        assert_eq!(exact.verifier_remaining(), 29);
        assert!(ObjectiveProblem::parse_source("(v0-3)^2", &variables, &mut exact).is_err());
        assert_eq!(exact.remaining(dimension), 0);
        limits.dimensions[dimension.index()] -= 1;
        let mut short = Budget::new(limits);
        assert!(matches!(
            ObjectiveProblem::parse_source("(v0-3)^2", &variables, &mut short),
            Err(CertificationError::Meter(MeterError::Budget(BudgetError::Exhausted { dimension: exhausted, .. })))
                if exhausted == dimension
        ));
        assert_eq!(short.verifier_remaining(), 29);
    }
}
