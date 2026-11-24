//! Shared checkpoints preserve exact global-optimization statements and evidence.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::box_optimization::checkpoint_transport::{
    MAX_BOX_OPTIMIZATION_CHECKPOINT_BYTES, OptimizationCheckpointError,
    export_box_optimization_checkpoint, restore_box_optimization_checkpoint,
};
use fsym_calculus::box_optimization::{
    BoxOptimizationOptions, BoxOptimizationSearch, BoxOptimizationSearchStatus,
    MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS,
};
use fsym_core::root_certificate::artifact::CertificationError;
use fsym_core::{
    BigRational, BoxExtremumKind, Expr, RealBall, Symbol, SymbolIdentity, verify_box_optimization,
};
use serde_json::{Value, value::RawValue};
use std::{collections::HashMap, sync::Arc};

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn interval(lower: i64, upper: i64) -> RealBall {
    RealBall::new(q(lower + upper, 2), q(upper - lower, 2)).unwrap()
}

fn power(source: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(source), Arc::new(Expr::from_i64(exponent)))
}

#[derive(Clone)]
struct Statement {
    objective: Expr,
    variables: Vec<Symbol>,
    domain: Vec<RealBall>,
    kind: BoxExtremumKind,
    tolerance: BigRational,
    options: BoxOptimizationOptions,
    context: [u8; 32],
}

impl Statement {
    fn double_well() -> Self {
        let x = Symbol::new("x");
        Self {
            objective: power(
                Expr::Add(vec![power(Expr::Sym(x.clone()), 2), Expr::from_i64(-2)]),
                2,
            ),
            variables: vec![x],
            domain: vec![interval(-2, 2)],
            kind: BoxExtremumKind::Minimum,
            tolerance: q(1, 1000),
            options: BoxOptimizationOptions {
                precision_digits: 16,
                max_leaves: 256,
            },
            context: [3; 32],
        }
    }

    fn begin(&self) -> BoxOptimizationSearch {
        BoxOptimizationSearch::new(
            &self.objective,
            &self.variables,
            &self.domain,
            self.kind,
            &self.tolerance,
            self.options,
            &mut Unbounded,
        )
        .unwrap()
    }

    fn restore(
        &self,
        bytes: &[u8],
        meter: &mut impl BudgetMeter,
    ) -> Result<BoxOptimizationSearch, OptimizationCheckpointError> {
        restore_box_optimization_checkpoint(
            bytes,
            &self.objective,
            &self.variables,
            &self.domain,
            self.kind,
            &self.tolerance,
            self.options,
            self.context,
            meter,
        )
    }
}

fn saved(statement: &Statement) -> Vec<u8> {
    let mut search = statement.begin();
    search.advance(3, &mut Unbounded).unwrap();
    export_box_optimization_checkpoint(&mut search, statement.context, &mut Unbounded).unwrap()
}

// Preserve every other byte and field order. Structural attacks must reach the
// bounded decoder or topology checker instead of merely reordering the object.
fn replace_field(bytes: &[u8], field: &str, replacement: &str) -> Vec<u8> {
    let fields: HashMap<&str, &RawValue> = serde_json::from_slice(bytes).unwrap();
    let old = format!("\"{field}\":{}", fields[field].get());
    let new = format!("\"{field}\":{replacement}");
    let text = std::str::from_utf8(bytes).unwrap();
    assert_eq!(text.matches(&old).count(), 1);
    text.replacen(&old, &new, 1).into_bytes()
}

#[test]
fn portable_nonconvex_prefixes_finish_with_the_same_independently_checked_witness() {
    let statement = Statement::double_well();
    let mut baseline = statement.begin();
    baseline.advance(MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS, &mut Unbounded).unwrap();
    let expected = baseline.completed_witness().unwrap();
    let mut saw_unfinished = false;
    for prefix in [0, 3, MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS] {
        let mut search = statement.begin();
        if prefix != 0 {
            search.advance(prefix, &mut Unbounded).unwrap();
        }
        let progress = search.progress();
        saw_unfinished |= progress.status == BoxOptimizationSearchStatus::Searching;
        let bytes = export_box_optimization_checkpoint(
            &mut search, statement.context, &mut Unbounded,
        ).unwrap();
        let wire: Value = serde_json::from_slice(&bytes).unwrap();
        for absent in ["accepted", "verified", "value", "gap", "bounds", "priority_queue"] {
            assert!(wire.get(absent).is_none(), "{absent}");
        }
        let mut restored = statement.restore(&bytes, &mut Unbounded).unwrap();
        assert_eq!(restored.progress().leaf_count, progress.leaf_count);
        assert_eq!(restored.progress().status, progress.status);
        assert!(restored.counters().steps > search.counters().steps);
        restored.advance(MAX_BOX_OPTIMIZATION_ADVANCE_SPLITS, &mut Unbounded).unwrap();
        assert_eq!(restored.completed_witness(), Some(expected));
        let receipt = verify_box_optimization(
            &statement.objective,
            &statement.variables,
            &statement.domain,
            statement.kind,
            &statement.tolerance,
            restored.completed_witness().unwrap(),
            &mut Unbounded,
        ).unwrap();
        assert!(receipt.value().contains(&q(0, 1)));
        assert!(receipt.gap() <= &statement.tolerance);
    }
    assert!(saw_unfinished);
}

#[test]
fn maximum_checkpoints_preserve_same_name_typed_axes_and_fixed_coordinates() {
    let x = Symbol::with_identity("axis", SymbolIdentity::from_assumptions_digest([1; 32]));
    let y = Symbol::with_identity("axis", SymbolIdentity::from_assumptions_digest([2; 32]));
    let statement = Statement {
        objective: Expr::Add(vec![Expr::Sym(x.clone()), Expr::Sym(y.clone())]),
        variables: vec![x, y],
        domain: vec![interval(-2, 2), interval(1, 1)],
        kind: BoxExtremumKind::Maximum,
        ..Statement::double_well()
    };
    let bytes = saved(&statement);
    let restored = statement.restore(&bytes, &mut Unbounded).unwrap();
    assert_eq!(restored.variables(), statement.variables);
    assert_eq!(restored.domain(), statement.domain);
    assert_eq!(restored.kind(), BoxExtremumKind::Maximum);
    let receipt = verify_box_optimization(
        &statement.objective, &statement.variables, &statement.domain,
        statement.kind, &statement.tolerance,
        restored.completed_witness().unwrap(), &mut Unbounded,
    ).unwrap();
    assert_eq!(receipt.point(), &[q(2, 1), q(1, 1)]);
    assert!(receipt.value().contains(&q(3, 1)));
    assert_eq!(receipt.gap(), &q(0, 1));
}

#[test]
fn every_expected_statement_component_is_bound_before_foreign_payload_decoding() {
    let statement = Statement::double_well();
    // The invalid counters would otherwise fail decoding. Binding must refuse
    // first without constructing any arithmetic supplied by the sender.
    let bytes = replace_field(&saved(&statement), "counters", "null");
    let mut changed = statement.clone();
    changed.objective = Expr::Add(vec![changed.objective, Expr::from_i64(0)]);
    assert!(matches!(changed.restore(&bytes, &mut Unbounded),
        Err(OptimizationCheckpointError::Certification(CertificationError::SourceMismatch))));
    let mut changed = statement.clone();
    changed.variables[0] = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([9; 32]));
    assert!(matches!(changed.restore(&bytes, &mut Unbounded),
        Err(OptimizationCheckpointError::Certification(CertificationError::SourceMismatch))));
    let mut changed = statement.clone();
    changed.domain = vec![interval(-3, 3)];
    assert!(matches!(changed.restore(&bytes, &mut Unbounded),
        Err(OptimizationCheckpointError::Certification(CertificationError::DomainMismatch))));
    for change in 0..4 {
        let mut changed = statement.clone();
        match change {
            0 => changed.kind = BoxExtremumKind::Maximum,
            1 => changed.tolerance = q(1, 500),
            2 => changed.options.precision_digits = 32,
            3 => changed.options.max_leaves = 512,
            _ => unreachable!(),
        }
        assert!(matches!(changed.restore(&bytes, &mut Unbounded),
            Err(OptimizationCheckpointError::StatementMismatch)));
    }
    let mut changed = statement.clone();
    changed.context = [4; 32];
    assert!(matches!(changed.restore(&bytes, &mut Unbounded),
        Err(OptimizationCheckpointError::Certification(CertificationError::UniverseMismatch { .. }))));
}

#[test]
fn structural_corruption_understated_work_and_noncanonical_bytes_cannot_restore() {
    let statement = Statement::double_well();
    let bytes = saved(&statement);
    for (field, replacement) in [
        ("schema_version", "2"),
        ("partition", "[]"),
        ("split_order", "[]"),
        ("incumbent_point", "[\"malformed scalar\"]"),
        ("counters", "{\"steps\":0,\"allocation_bytes\":0,\"source_bytes\":0,\"evaluations\":0}"),
    ] {
        assert!(statement.restore(&replace_field(&bytes, field, replacement), &mut Unbounded).is_err(), "{field}");
    }
    let mut unknown = bytes.clone();
    unknown.pop();
    unknown.extend_from_slice(b",\"accepted\":true}");
    assert!(statement.restore(&unknown, &mut Unbounded).is_err());
    let mut noncanonical = bytes.clone();
    noncanonical.push(b' ');
    assert!(matches!(statement.restore(&noncanonical, &mut Unbounded),
        Err(OptimizationCheckpointError::Certification(CertificationError::NonCanonical))));
    assert!(statement.restore(b"{}", &mut Unbounded).is_err());
    let oversized = vec![b' '; MAX_BOX_OPTIMIZATION_CHECKPOINT_BYTES + 1];
    assert!(matches!(statement.restore(&oversized, &mut Unbounded),
        Err(OptimizationCheckpointError::Certification(CertificationError::ByteLimit { .. }))));
}

#[test]
fn checkpoint_copy_and_reconstruction_charge_one_cumulative_ledger() {
    let statement = Statement::double_well();
    let mut search = statement.begin();
    let allowance = 1 << 35;
    let mut budget = Budget::new(BudgetLimits::uniform(allowance, 1_000_000));
    let first = export_box_optimization_checkpoint(&mut search, statement.context, &mut budget).unwrap();
    let steps = search.counters().steps;
    let remaining = budget.remaining(Dimension::ComputeSteps);
    let second = export_box_optimization_checkpoint(&mut search, statement.context, &mut budget).unwrap();
    assert!(search.counters().steps > steps);
    assert!(budget.remaining(Dimension::ComputeSteps) < remaining);
    assert_ne!(first, second);
    let before_restore = budget.remaining(Dimension::ComputeSteps);
    let restored = statement.restore(&second, &mut budget).unwrap();
    assert!(restored.counters().steps > search.counters().steps);
    assert!(budget.remaining(Dimension::ComputeSteps) < before_restore);
    assert_eq!(budget.verifier_remaining(), 1_000_000);
    let mut exhausted = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(matches!(statement.restore(&second, &mut exhausted),
        Err(OptimizationCheckpointError::Certification(CertificationError::Meter(MeterError::Budget(_))))));
    assert!(matches!(export_box_optimization_checkpoint(&mut search, statement.context, &mut exhausted),
        Err(OptimizationCheckpointError::Certification(CertificationError::Meter(MeterError::Budget(_))))));
}

struct Cancelled;

impl BudgetMeter for Cancelled {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        panic!("charge after cancellation")
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        panic!("charge after cancellation")
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        Err(MeterError::Cancelled)
    }
}

#[test]
fn cancelled_transport_never_returns_candidate_state_or_partial_bytes() {
    let statement = Statement::double_well();
    let bytes = saved(&statement);
    assert!(matches!(statement.restore(&bytes, &mut Cancelled),
        Err(OptimizationCheckpointError::Certification(CertificationError::Meter(MeterError::Cancelled)))));
    let mut search = statement.begin();
    let before = search.progress();
    assert!(matches!(export_box_optimization_checkpoint(&mut search, statement.context, &mut Cancelled),
        Err(OptimizationCheckpointError::Certification(CertificationError::Meter(MeterError::Cancelled)))));
    assert_eq!(search.progress(), before);
}
