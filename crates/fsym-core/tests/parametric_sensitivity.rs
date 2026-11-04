//! Analytic implicit derivatives, nonzero-defect counterexamples and ledger tests.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{
    BigInt, BigRational, Expr, ParametricRootBoxReceipt, ParametricRootBoxWitness,
    ParametricRootCoverReceipt, ParametricRootCoverStep, ParametricRootCoverWitness, RealBall,
    RootBoxError, Symbol, SymbolIdentity, certify_parametric_root_box,
    certify_parametric_root_cover, enclose_parametric_cover_sensitivity,
    enclose_parametric_root_sensitivity, metered_enclose_parametric_cover_sensitivity,
    metered_enclose_parametric_root_sensitivity,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}
fn interval(lo: BigRational, hi: BigRational) -> RealBall {
    RealBall::new((&lo + &hi) / q(2, 1), (hi - lo) / q(2, 1)).unwrap()
}
fn bounds(lo: i64, hi: i64) -> RealBall {
    interval(q(lo, 1), q(hi, 1))
}
fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}
fn witness(x: Vec<RealBall>, p: Vec<RealBall>) -> ParametricRootBoxWitness {
    let n = x.len();
    ParametricRootBoxWitness {
        center: x.iter().map(|v| v.midpoint().clone()).collect(),
        unknown_domain: x,
        parameter_domain: p,
        preconditioner: (0..n)
            .map(|i| (0..n).map(|j| q(i64::from(i == j), 1)).collect())
            .collect(),
        precision_digits: 16,
    }
}
fn affine() -> ParametricRootBoxReceipt {
    let x = Symbol::new("x");
    let p = Symbol::new("p");
    let source = [Expr::Sym(x.clone()) - Expr::Sym(p.clone())];
    certify_parametric_root_box(
        &source,
        &[x],
        &[p],
        &witness(vec![bounds(-2, 2)], vec![bounds(-1, 1)]),
    )
    .unwrap()
}

#[test]
fn affine_family_has_exact_signed_solution_derivatives_and_source_binding() {
    let unknowns = [Symbol::new("x"), Symbol::new("y")];
    let parameters = [Symbol::new("p")];
    let x = Expr::Sym(unknowns[0].clone());
    let y = Expr::Sym(unknowns[1].clone());
    let p = Expr::Sym(parameters[0].clone());
    let source = [x + Expr::from_i64(2) * y.clone() - p.clone(), y - p];
    let mut proposal = witness(vec![bounds(-4, 4); 2], vec![bounds(-1, 1)]);
    proposal.preconditioner = vec![vec![q(1, 1), q(-2, 1)], vec![q(0, 1), q(1, 1)]];
    let family = certify_parametric_root_box(&source, &unknowns, &parameters, &proposal).unwrap();
    let result = enclose_parametric_root_sensitivity(&family).unwrap();
    assert_eq!(
        result.solution_jacobian(),
        [vec![bounds(-1, -1)], vec![bounds(1, 1)]]
    );
    assert_eq!(result.residuals(), source);
    assert_eq!(result.unknowns(), unknowns);
    assert_eq!(result.parameters(), parameters);
    assert_eq!(result.unknown_domain(), proposal.unknown_domain);
    assert_eq!(result.parameter_domain(), proposal.parameter_domain);
    assert_eq!(result.root_enclosure(), family.root_enclosure());
    assert_eq!(result.contraction_bound(), &q(0, 1));
    assert_eq!(result.precision_digits(), 16);
    assert_eq!(result.leaf_count(), 1);
}

#[test]
fn square_root_sensitivity_covers_analytic_extremes_not_just_the_midpoint() {
    let x = Symbol::new("x");
    let p = Symbol::new("p");
    let source = [power(Expr::Sym(x.clone()), 2) - Expr::Sym(p.clone())];
    let mut proposal = witness(vec![bounds(1, 2)], vec![bounds(2, 3)]);
    proposal.preconditioner[0][0] = q(1, 3);
    let family = certify_parametric_root_box(&source, &[x], &[p], &proposal).unwrap();
    let result = enclose_parametric_root_sensitivity(&family).unwrap();
    let derivative = &result.solution_jacobian()[0][0];
    assert_eq!(result.contraction_bound(), &q(1, 3));
    assert!(derivative.lower() > q(0, 1));
    // dx/dp = 1/(2 sqrt(p)); its extreme squares on [2,3] are 1/12 and 1/8.
    assert!(&derivative.lower() * &derivative.lower() <= q(1, 12));
    assert!(&derivative.upper() * &derivative.upper() >= q(1, 8));
    assert!(derivative.lower() < q(1, 3) && derivative.upper() > q(1, 3));
}

#[test]
fn noninverse_preconditioner_requires_the_full_geometric_series_remainder() {
    let x = Symbol::new("x");
    let p = Symbol::new("p");
    let source = [Expr::Rational(q(1, 2)) * Expr::Sym(x.clone()) - Expr::Sym(p.clone())];
    let proposal = witness(vec![bounds(-4, 4)], vec![interval(q(-1, 2), q(1, 2))]);
    let family = certify_parametric_root_box(&source, &[x], &[p], &proposal).unwrap();
    let result = enclose_parametric_root_sensitivity(&family).unwrap();
    assert_eq!(result.contraction_bound(), &q(1, 2));
    assert!(result.solution_jacobian()[0][0].contains(&q(2, 1)));
    // B=1 and q=1/2. Omitting /(1-q) would incorrectly cap the result at 3/2.
    assert!(result.solution_jacobian()[0][0].upper() > q(3, 2));
}

#[test]
fn absolute_row_sums_are_required_for_coupled_derivative_bounds() {
    let unknowns = [Symbol::new("x"), Symbol::new("y")];
    let p = Symbol::new("p");
    let x = Expr::Sym(unknowns[0].clone());
    let y = Expr::Sym(unknowns[1].clone());
    let parameter = Expr::Sym(p.clone());
    let source = [
        Expr::Rational(q(3, 5)) * x.clone()
            - Expr::Rational(q(2, 5)) * y.clone()
            - parameter.clone(),
        Expr::Rational(q(3, 5)) * y - Expr::Rational(q(2, 5)) * x - parameter,
    ];
    let proposal = witness(
        vec![bounds(-10, 10); 2],
        vec![interval(q(-1, 10), q(1, 10))],
    );
    let family = certify_parametric_root_box(&source, &unknowns, &[p], &proposal).unwrap();
    let result = enclose_parametric_root_sensitivity(&family).unwrap();
    assert_eq!(result.contraction_bound(), &q(4, 5));
    for row in result.solution_jacobian() {
        // Exact solution is x=y=5p. A maximum-entry norm would miss this value.
        assert!(row[0].contains(&q(5, 1)));
    }
}

#[test]
fn negative_defects_cannot_be_used_as_unsigned_contraction_bounds() {
    let x = Symbol::new("x");
    let p = Symbol::new("p");
    let source = [Expr::Rational(q(3, 2)) * Expr::Sym(x.clone()) - Expr::Sym(p.clone())];
    let proposal = witness(vec![bounds(-4, 4)], vec![bounds(-1, 1)]);
    let family = certify_parametric_root_box(&source, &[x], &[p], &proposal).unwrap();
    let result = enclose_parametric_root_sensitivity(&family).unwrap();
    assert_eq!(result.contraction_bound(), &q(1, 2));
    assert!(result.solution_jacobian()[0][0].contains(&q(2, 3)));
}

#[test]
fn original_parameter_jacobian_keeps_full_interval_variation() {
    let x = Symbol::new("x");
    let p = Symbol::new("p");
    let source = [Expr::Sym(x.clone()) - power(Expr::Sym(p.clone()), 2)];
    let family = certify_parametric_root_box(
        &source,
        &[x],
        &[p],
        &witness(vec![bounds(-1, 2)], vec![bounds(0, 1)]),
    )
    .unwrap();
    let result = enclose_parametric_root_sensitivity(&family).unwrap();
    assert_eq!(result.solution_jacobian(), [vec![bounds(0, 2)]]);
}

#[test]
fn transcendental_sensitivity_retains_a_useful_positive_sign() {
    let x = Symbol::new("x");
    let p = Symbol::new("p");
    let sine = Expr::Function("sin".into(), vec![Expr::Sym(p.clone())]);
    let source = [Expr::Sym(x.clone()) - sine];
    let family = certify_parametric_root_box(
        &source,
        &[x],
        &[p],
        &witness(vec![bounds(-2, 2)], vec![bounds(-1, 1)]),
    )
    .unwrap();
    let result = enclose_parametric_root_sensitivity(&family).unwrap();
    let derivative = &result.solution_jacobian()[0][0];
    // The current interval Taylor recurrence over-encloses cosine on [-1,1]
    // by roughly [2-cosh(1), cosh(1)]; it still proves a positive derivative.
    assert!(derivative.lower() > q(2, 5));
    assert!(derivative.lower() < q(3, 5));
    assert!(derivative.contains(&q(1, 1)));
}

#[test]
fn ordered_parameter_columns_include_unused_and_fixed_parameters() {
    let x = Symbol::new("x");
    let parameters = [Symbol::new("p"), Symbol::new("unused"), Symbol::new("r")];
    let residual = Expr::Sym(x.clone())
        - Expr::Sym(parameters[0].clone())
        - Expr::from_i64(10) * Expr::Sym(parameters[2].clone());
    let source = [residual];
    let proposal = witness(
        vec![bounds(-20, 20)],
        vec![bounds(-1, 1), bounds(0, 0), bounds(-1, 1)],
    );
    let family = certify_parametric_root_box(&source, &[x], &parameters, &proposal).unwrap();
    let result = enclose_parametric_root_sensitivity(&family).unwrap();
    assert_eq!(
        result.solution_jacobian(),
        [vec![bounds(1, 1), bounds(0, 0), bounds(10, 10)]]
    );
    assert_eq!(result.parameters(), parameters);
}

#[test]
fn typed_same_name_unknown_and_parameter_remain_distinct() {
    let x = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([11; 32]));
    let p = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([12; 32]));
    let source = [Expr::Sym(x.clone()) - Expr::from_i64(3) * Expr::Sym(p.clone())];
    let family = certify_parametric_root_box(
        &source,
        std::slice::from_ref(&x),
        std::slice::from_ref(&p),
        &witness(vec![bounds(-4, 4)], vec![bounds(-1, 1)]),
    )
    .unwrap();
    let result = enclose_parametric_root_sensitivity(&family).unwrap();
    assert_eq!(result.solution_jacobian(), [vec![bounds(3, 3)]]);
    assert_eq!(result.unknowns(), [x]);
    assert_eq!(result.parameters(), [p]);
}

#[test]
fn exact_sensitivity_can_exceed_binary64_range_without_overflow() {
    let x = Symbol::new("x");
    let p = Symbol::new("p");
    let huge = BigRational::from_integer(BigInt::from(10).pow(400));
    let tiny = q(1, 1) / &huge;
    let source = [Expr::Rational(tiny.clone()) * Expr::Sym(x.clone()) - Expr::Sym(p.clone())];
    let mut proposal = witness(vec![bounds(-2, 2)], vec![interval(-tiny.clone(), tiny)]);
    proposal.preconditioner[0][0] = huge.clone();
    let family = certify_parametric_root_box(&source, &[x], &[p], &proposal).unwrap();
    let result = enclose_parametric_root_sensitivity(&family).unwrap();
    assert_eq!(result.solution_jacobian(), [vec![RealBall::exact(huge)]]);
}

#[test]
fn maximum_unknown_dimension_preserves_every_derivative_row() {
    let unknowns: Vec<_> = (0..15).map(|i| Symbol::new(format!("x{i}"))).collect();
    let p = Symbol::new("p");
    let source: Vec<_> = unknowns
        .iter()
        .map(|x| Expr::Sym(x.clone()) - Expr::Sym(p.clone()))
        .collect();
    let family = certify_parametric_root_box(
        &source,
        &unknowns,
        &[p],
        &witness(vec![bounds(-2, 2); 15], vec![bounds(-1, 1)]),
    )
    .unwrap();
    let result = enclose_parametric_root_sensitivity(&family).unwrap();
    assert_eq!(result.solution_jacobian(), vec![vec![bounds(1, 1)]; 15]);
}

#[test]
fn maximum_parameter_dimension_preserves_every_derivative_column() {
    let x = Symbol::new("x");
    let parameters: Vec<_> = (0..15).map(|i| Symbol::new(format!("p{i}"))).collect();
    let mut residual = Expr::Sym(x.clone());
    for (j, p) in parameters.iter().enumerate() {
        residual = residual - Expr::from_i64((j + 1) as i64) * Expr::Sym(p.clone());
    }
    let family = certify_parametric_root_box(
        &[residual],
        &[x],
        &parameters,
        &witness(vec![bounds(-1, 1)], vec![bounds(0, 0); 15]),
    )
    .unwrap();
    let result = enclose_parametric_root_sensitivity(&family).unwrap();
    let expected: Vec<_> = (1..=15).map(|j| bounds(j, j)).collect();
    assert_eq!(result.solution_jacobian(), [expected]);
}

fn squared_parameter_cover() -> ParametricRootCoverReceipt {
    let x = Symbol::new("x");
    let p = Symbol::new("p");
    let source = [Expr::Sym(x.clone()) - power(Expr::Sym(p.clone()), 2)];
    let unknown_domain = [bounds(-1, 2)];
    let parameter_domain = [bounds(-1, 1)];
    let cover = ParametricRootCoverWitness {
        leaves: vec![
            witness(unknown_domain.to_vec(), vec![bounds(-1, 0)]),
            witness(unknown_domain.to_vec(), vec![bounds(0, 1)]),
        ],
        partition: vec![
            ParametricRootCoverStep::Split { axis: 0 },
            ParametricRootCoverStep::Leaf { index: 0 },
            ParametricRootCoverStep::Leaf { index: 1 },
        ],
        precision_digits: 16,
    };
    certify_parametric_root_cover(
        &source,
        &[x],
        &[p],
        &unknown_domain,
        &parameter_domain,
        &cover,
    )
    .unwrap()
}

#[test]
fn complete_cover_sensitivity_includes_both_sides_and_the_shared_face() {
    let family = squared_parameter_cover();
    let result = enclose_parametric_cover_sensitivity(&family).unwrap();
    assert_eq!(result.leaf_count(), 2);
    assert_eq!(result.solution_jacobian(), [vec![bounds(-2, 2)]]);
    assert_eq!(result.unknown_domain(), family.unknown_domain());
    assert_eq!(result.parameter_domain(), family.parameter_domain());
    assert_eq!(result.root_enclosure(), family.root_enclosure());
}

#[test]
fn a_complete_cover_shares_one_scratch_reservation_and_never_refunds_the_caller() {
    let family = squared_parameter_cover();
    let mut budget = Budget::new(BudgetLimits::uniform(100_000_000, 10_000));
    metered_enclose_parametric_cover_sensitivity(&family, &mut budget).unwrap();
    assert!(budget.remaining(Dimension::MemoryBytes) < 100_000_000 - 64 * 1024 * 1024);
    let after = budget.snapshot();
    assert!(matches!(
        metered_enclose_parametric_cover_sensitivity(&family, &mut budget),
        Err(RootBoxError::Meter(MeterError::Budget(_)))
    ));
    assert_eq!(budget.snapshot(), after);
    assert_eq!(budget.verifier_remaining(), 10_000);
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
fn every_sensitivity_checkpoint_including_publication_preserves_cancellation() {
    let family = affine();
    let mut count = Cancel::default();
    metered_enclose_parametric_root_sensitivity(&family, &mut count).unwrap();
    for stop in 0..count.visits {
        let mut cancel = Cancel {
            visits: 0,
            stop: Some(stop),
        };
        assert_eq!(
            metered_enclose_parametric_root_sensitivity(&family, &mut cancel),
            Err(RootBoxError::Meter(MeterError::Cancelled)),
            "safe point {stop}"
        );
    }
}
