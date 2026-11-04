//! Uniform-family claims checked against analytic extrema and mutations.
//! Parameter intervals are never reduced to sampled or midpoint values.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError};
use fsym_core::{
    BigInt, BigRational, Expr, MAX_ROOT_BOX_LIMBS, ParametricRootBoxWitness, RealBall,
    RootBoxError, Symbol, SymbolIdentity, certify_parametric_root_box, verify_parametric_root_box,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}
fn ball(center: BigRational, radius: BigRational) -> RealBall {
    RealBall::new(center, radius).unwrap()
}
fn bounds(lower: i64, upper: i64) -> RealBall {
    ball(q(lower + upper, 2), q(upper - lower, 2))
}
fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}
fn witness(
    unknown_domain: Vec<RealBall>,
    parameter_domain: Vec<RealBall>,
) -> ParametricRootBoxWitness {
    let n = unknown_domain.len();
    ParametricRootBoxWitness {
        center: unknown_domain
            .iter()
            .map(|value| value.midpoint().clone())
            .collect(),
        unknown_domain,
        parameter_domain,
        preconditioner: (0..n)
            .map(|i| (0..n).map(|j| q(i64::from(i == j), 1)).collect())
            .collect(),
        precision_digits: 24,
    }
}
fn affine_family() -> (
    Vec<Expr>,
    Vec<Symbol>,
    Vec<Symbol>,
    ParametricRootBoxWitness,
) {
    let unknowns = vec![Symbol::new("x"), Symbol::new("y")];
    let parameters = vec![Symbol::new("p")];
    let p = Expr::Sym(parameters[0].clone());
    let source = vec![
        Expr::Sym(unknowns[0].clone()) - p.clone(),
        Expr::Sym(unknowns[1].clone()) - Expr::from_i64(2) * p,
    ];
    let proposed = witness(vec![bounds(-2, 2), bounds(-3, 3)], vec![bounds(-1, 1)]);
    (source, unknowns, parameters, proposed)
}

#[test]
fn positive_square_root_family_is_uniformly_enclosed_for_every_parameter() {
    let unknowns = [Symbol::new("x")];
    let parameters = [Symbol::new("p")];
    let source = [power(Expr::Sym(unknowns[0].clone()), 2) - Expr::Sym(parameters[0].clone())];
    let mut proposed = witness(vec![bounds(1, 2)], vec![bounds(2, 3)]);
    proposed.preconditioner[0][0] = q(1, 3);
    let accepted = certify_parametric_root_box(&source, &unknowns, &parameters, &proposed).unwrap();
    assert_eq!(accepted.residuals(), source);
    assert_eq!(accepted.unknowns(), unknowns);
    assert_eq!(accepted.parameters(), parameters);
    assert_eq!(accepted.witness(), &proposed);
    assert_eq!(accepted.contraction_bound(), &q(1, 3));
    let enclosure = &accepted.root_enclosure()[0];
    assert_eq!(enclosure.lower(), q(5, 4));
    assert_eq!(enclosure.upper(), q(23, 12));
    // The family contains sqrt(2) AND sqrt(3). Evaluating only at p=5/2
    // would produce a lower bound whose square is greater than 2.
    assert!(&enclosure.lower() * &enclosure.lower() < q(2, 1));
    assert!(&enclosure.upper() * &enclosure.upper() > q(3, 1));
}

#[test]
fn coupled_affine_family_retains_the_entire_parameter_variation() {
    let (source, unknowns, parameters, proposed) = affine_family();
    let accepted = certify_parametric_root_box(&source, &unknowns, &parameters, &proposed).unwrap();
    assert_eq!(accepted.root_enclosure(), [bounds(-1, 1), bounds(-2, 2)]);
    assert_eq!(accepted.contraction_bound(), &q(0, 1));
    assert_eq!(accepted.unknowns(), unknowns);
    assert_eq!(accepted.parameters(), parameters);
    assert_eq!(accepted.residuals(), source);
}

#[test]
fn nonsymmetric_preconditioner_orientation_and_unknown_order_are_checked() {
    let unknowns = [Symbol::new("x"), Symbol::new("y")];
    let parameters = [Symbol::new("p")];
    let x = Expr::Sym(unknowns[0].clone());
    let y = Expr::Sym(unknowns[1].clone());
    let p = Expr::Sym(parameters[0].clone());
    let source = [x + Expr::from_i64(2) * y.clone() - p.clone(), y - p];
    let mut proposed = witness(vec![bounds(-4, 4); 2], vec![bounds(-1, 1)]);
    proposed.preconditioner = vec![vec![q(1, 1), q(-2, 1)], vec![q(0, 1), q(1, 1)]];
    let accepted = certify_parametric_root_box(&source, &unknowns, &parameters, &proposed).unwrap();
    // Exact affine family is x=-p, y=p. Interval dependency in C*F(c,P)
    // enlarges the first coordinate, so the unknown box retains that width.
    assert!(accepted.root_enclosure()[0].contains(&q(-1, 1)));
    assert!(accepted.root_enclosure()[0].contains(&q(1, 1)));
    assert_eq!(accepted.contraction_bound(), &q(0, 1));
    let mut transposed = proposed.clone();
    transposed.preconditioner = vec![vec![q(1, 1), q(0, 1)], vec![q(-2, 1), q(1, 1)]];
    assert_eq!(
        certify_parametric_root_box(&source, &unknowns, &parameters, &transposed),
        Err(RootBoxError::NotContractive)
    );
    assert_eq!(
        certify_parametric_root_box(
            &source,
            &[unknowns[1].clone(), unknowns[0].clone()],
            &parameters,
            &proposed
        ),
        Err(RootBoxError::NotContractive)
    );
    proposed.preconditioner.swap(0, 1);
    let reordered = [unknowns[1].clone(), unknowns[0].clone()];
    let accepted =
        certify_parametric_root_box(&source, &reordered, &parameters, &proposed).unwrap();
    assert_eq!(accepted.unknowns(), reordered);
}

#[test]
fn parameter_order_and_same_name_typed_identities_remain_semantic_inputs() {
    let unknowns = [Symbol::new("x")];
    let parameters = [Symbol::new("p"), Symbol::new("q")];
    let residual = Expr::Sym(unknowns[0].clone())
        - Expr::Sym(parameters[0].clone())
        - Expr::from_i64(10) * Expr::Sym(parameters[1].clone());
    let source = [residual];
    let proposed = witness(vec![bounds(0, 50)], vec![bounds(1, 2), bounds(3, 4)]);
    let accepted = certify_parametric_root_box(&source, &unknowns, &parameters, &proposed).unwrap();
    assert_eq!(accepted.root_enclosure(), [bounds(31, 42)]);
    let reordered = [parameters[1].clone(), parameters[0].clone()];
    let accepted = certify_parametric_root_box(&source, &unknowns, &reordered, &proposed).unwrap();
    assert_eq!(accepted.parameters(), reordered);
    assert_eq!(accepted.root_enclosure(), [bounds(13, 24)]);

    let x = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([1; 32]));
    let p = Symbol::with_identity("same", SymbolIdentity::from_assumptions_digest([2; 32]));
    let source = [Expr::Sym(x.clone()) - Expr::Sym(p.clone())];
    let proposed = witness(vec![bounds(-2, 2)], vec![bounds(-1, 1)]);
    let accepted = certify_parametric_root_box(
        &source,
        std::slice::from_ref(&x),
        std::slice::from_ref(&p),
        &proposed,
    )
    .unwrap();
    assert_eq!(accepted.unknowns(), std::slice::from_ref(&x));
    assert_eq!(accepted.parameters(), [p]);
    assert_eq!(accepted.root_enclosure(), [bounds(-1, 1)]);
    assert!(matches!(
        certify_parametric_root_box(&source, &[x], &[Symbol::new("same")], &proposed),
        Err(RootBoxError::Unsupported(_))
    ));
}

#[test]
fn retaining_parameter_intervals_prevents_midpoint_only_false_acceptance() {
    let unknowns = [Symbol::new("x")];
    let parameters = [Symbol::new("p")];
    let x = Expr::Sym(unknowns[0].clone());
    let p = Expr::Sym(parameters[0].clone());
    let affine = [x.clone() - p.clone()];
    let proposed = witness(vec![bounds(-1, 1)], vec![bounds(-2, 2)]);
    assert_eq!(
        certify_parametric_root_box(&affine, &unknowns, &parameters, &proposed),
        Err(RootBoxError::NoInteriorInclusion { coordinate: 0 })
    );
    let nonlinear = [x - power(p, 2)];
    let proposed = witness(vec![ball(q(0, 1), q(1, 4))], vec![bounds(-1, 1)]);
    assert_eq!(
        certify_parametric_root_box(&nonlinear, &unknowns, &parameters, &proposed),
        Err(RootBoxError::NoInteriorInclusion { coordinate: 0 })
    );
}

#[test]
fn uniform_jacobian_bounds_reject_bifurcation_and_parameter_endpoint_singularity() {
    let unknowns = [Symbol::new("x")];
    let parameters = [Symbol::new("p")];
    let x = Expr::Sym(unknowns[0].clone());
    let p = Expr::Sym(parameters[0].clone());
    let source = [power(x.clone(), 2) - p.clone()];
    let mut proposed = witness(vec![bounds(-1, 2)], vec![bounds(0, 1)]);
    proposed.center[0] = q(1, 1);
    proposed.preconditioner[0][0] = q(1, 2);
    assert_eq!(
        certify_parametric_root_box(&source, &unknowns, &parameters, &proposed),
        Err(RootBoxError::NotContractive)
    );

    let source = [p * x - Expr::from_i64(1)];
    let mut proposed = witness(vec![bounds(-1, 5)], vec![bounds(0, 1)]);
    proposed.preconditioner[0][0] = q(2, 1);
    // J_x=p includes the singular endpoint p=0. Freezing only J_x at
    // p=1/2 would incorrectly make the defect zero and image [0,4] valid.
    assert_eq!(
        certify_parametric_root_box(&source, &unknowns, &parameters, &proposed),
        Err(RootBoxError::NotContractive)
    );
}

#[test]
fn original_held_parameter_holes_are_checked_over_the_full_product_box() {
    let unknowns = [Symbol::new("x")];
    let parameters = [Symbol::new("p")];
    let x = Expr::Sym(unknowns[0].clone());
    let inverse = power(Expr::Sym(parameters[0].clone()), -1);
    let proposed = witness(vec![bounds(-2, 2)], vec![bounds(-1, 1)]);
    for hidden in [
        Expr::Mul(vec![Expr::from_i64(0), inverse.clone()]),
        power(inverse, 0),
    ] {
        let source = [Expr::Add(vec![x.clone(), hidden])];
        assert!(matches!(
            certify_parametric_root_box(&source, &unknowns, &parameters, &proposed),
            Err(RootBoxError::Ball(_)) | Err(RootBoxError::Domain(_))
        ));
    }
}

#[test]
fn parameter_point_boxes_are_allowed_but_boundary_unknown_roots_are_not() {
    let unknowns = [Symbol::new("x")];
    let parameters = [Symbol::new("p")];
    let source = [Expr::Sym(unknowns[0].clone()) - Expr::Sym(parameters[0].clone())];
    let proposed = witness(vec![bounds(1, 3)], vec![bounds(2, 2)]);
    let accepted = certify_parametric_root_box(&source, &unknowns, &parameters, &proposed).unwrap();
    assert_eq!(accepted.root_enclosure(), [RealBall::from_i64(2)]);
    assert_eq!(accepted.contraction_bound(), &q(0, 1));
    let boundary = witness(vec![bounds(1, 2)], vec![bounds(2, 2)]);
    assert_eq!(
        certify_parametric_root_box(&source, &unknowns, &parameters, &boundary),
        Err(RootBoxError::NoInteriorInclusion { coordinate: 0 })
    );
    let fixed = witness(vec![bounds(2, 2)], vec![bounds(2, 2)]);
    assert_eq!(
        certify_parametric_root_box(&source, &unknowns, &parameters, &fixed),
        Err(RootBoxError::DegenerateBox { coordinate: 0 })
    );
}

#[test]
fn certified_transcendental_parameter_dependence_is_not_sampled_away() {
    let unknowns = [Symbol::new("x")];
    let parameters = [Symbol::new("p")];
    let residual = Expr::Sym(unknowns[0].clone())
        - Expr::Function("sin".into(), vec![Expr::Sym(parameters[0].clone())]);
    let source = [residual];
    let proposed = witness(vec![bounds(-2, 2)], vec![bounds(-1, 1)]);
    let accepted = certify_parametric_root_box(&source, &unknowns, &parameters, &proposed).unwrap();
    let interval = &accepted.root_enclosure()[0];
    // sin(1) > 1 - 1/6 > 4/5, by its alternating-series lower bound.
    assert!(interval.contains(&q(-4, 5)));
    assert!(interval.contains(&q(4, 5)));
    // The interval extension may conservatively exceed the exact [-1,1]
    // sine range; it still must retain a strict uniform root neighborhood.
    assert!(interval.lower() > q(-2, 1));
    assert!(interval.upper() < q(2, 1));
    assert_eq!(accepted.contraction_bound(), &q(0, 1));
}

#[test]
fn a_family_below_binary64_range_retains_exact_endpoint_roots() {
    let unknowns = [Symbol::new("x")];
    let parameters = [Symbol::new("p")];
    let tiny = BigRational::new(BigInt::from(1), BigInt::from(10).pow(400));
    let source = [Expr::Sym(unknowns[0].clone()) - Expr::Sym(parameters[0].clone())];
    let proposed = witness(
        vec![ball(&tiny * q(3, 2), &tiny * q(3, 2))],
        vec![ball(&tiny * q(3, 2), &tiny / q(2, 1))],
    );
    let accepted = certify_parametric_root_box(&source, &unknowns, &parameters, &proposed).unwrap();
    assert_eq!(accepted.root_enclosure()[0].lower(), tiny);
    assert_eq!(accepted.root_enclosure()[0].upper(), &tiny * q(2, 1));
    assert_eq!(accepted.root_enclosure(), proposed.parameter_domain);
}

#[test]
fn unknown_and_parameter_groups_must_be_nonempty_disjoint_and_shape_correct() {
    let unknowns = [Symbol::new("x")];
    let parameters = [Symbol::new("p")];
    let source = [Expr::Sym(unknowns[0].clone()) - Expr::Sym(parameters[0].clone())];
    let original = witness(vec![bounds(-2, 2)], vec![bounds(-1, 1)]);
    for (x, p) in [(&[][..], &parameters[..]), (&unknowns[..], &[][..])] {
        assert_eq!(
            certify_parametric_root_box(&source, x, p, &original),
            Err(RootBoxError::DimensionMismatch)
        );
    }
    assert_eq!(
        certify_parametric_root_box(&source, &unknowns, &unknowns, &original),
        Err(RootBoxError::DuplicateVariable)
    );
    assert_eq!(
        certify_parametric_root_box(&[], &unknowns, &parameters, &original),
        Err(RootBoxError::DimensionMismatch)
    );
    let mut cases = Vec::new();
    let mut bad = original.clone();
    bad.unknown_domain.clear();
    cases.push(bad);
    let mut bad = original.clone();
    bad.parameter_domain.clear();
    cases.push(bad);
    let mut bad = original.clone();
    bad.center.clear();
    cases.push(bad);
    let mut bad = original.clone();
    bad.preconditioner.clear();
    cases.push(bad);
    let mut bad = original.clone();
    bad.preconditioner[0].clear();
    cases.push(bad);
    for bad in cases {
        assert_eq!(
            certify_parametric_root_box(&source, &unknowns, &parameters, &bad),
            Err(RootBoxError::DimensionMismatch)
        );
    }
    let mut outside = original.clone();
    outside.center[0] = q(3, 1);
    assert_eq!(
        certify_parametric_root_box(&source, &unknowns, &parameters, &outside),
        Err(RootBoxError::CenterOutsideBox { coordinate: 0 })
    );
    for precision in [0, 7, 129, u32::MAX] {
        let mut bad = original.clone();
        bad.precision_digits = precision;
        assert_eq!(
            certify_parametric_root_box(&source, &unknowns, &parameters, &bad),
            Err(RootBoxError::InvalidPrecision)
        );
    }
}

#[test]
fn dimensions_names_scalars_and_original_source_are_bounded_before_publication() {
    let unknowns = [Symbol::new("x")];
    let parameters = [Symbol::new("p")];
    let source = [Expr::Sym(unknowns[0].clone()) - Expr::Sym(parameters[0].clone())];
    let proposed = witness(vec![bounds(-2, 2)], vec![bounds(-1, 1)]);
    let many: Vec<_> = (0..16).map(|i| Symbol::new(format!("p{i}"))).collect();
    let huge_domain = witness(vec![bounds(-2, 2)], vec![bounds(-1, 1); 16]);
    assert_eq!(
        certify_parametric_root_box(&source, &unknowns, &many, &huge_domain),
        Err(RootBoxError::DimensionMismatch)
    );
    assert!(matches!(
        certify_parametric_root_box(
            &source,
            &[Symbol::new("x".repeat(4097))],
            &parameters,
            &proposed
        ),
        Err(RootBoxError::ResourceLimit(_))
    ));
    assert!(matches!(
        certify_parametric_root_box(
            &source,
            &unknowns,
            &[Symbol::new("p".repeat(4097))],
            &proposed
        ),
        Err(RootBoxError::ResourceLimit(_))
    ));
    let huge = BigRational::from_integer(BigInt::from(1) << (64 * MAX_ROOT_BOX_LIMBS) as u32);
    let mut bad = proposed.clone();
    bad.parameter_domain[0] = RealBall::exact(huge.clone());
    assert!(matches!(
        certify_parametric_root_box(&source, &unknowns, &parameters, &bad),
        Err(RootBoxError::ResourceLimit(_))
    ));
    let mut bad = proposed.clone();
    bad.center[0] = huge.clone();
    assert!(matches!(
        certify_parametric_root_box(&source, &unknowns, &parameters, &bad),
        Err(RootBoxError::ResourceLimit(_))
    ));
    let mut bad = proposed.clone();
    bad.preconditioner[0][0] = huge.clone();
    assert!(matches!(
        certify_parametric_root_box(&source, &unknowns, &parameters, &bad),
        Err(RootBoxError::ResourceLimit(_))
    ));
    let hidden = Expr::Mul(vec![Expr::from_i64(0), Expr::Rational(huge)]);
    assert!(matches!(
        certify_parametric_root_box(
            &[Expr::Add(vec![source[0].clone(), hidden])],
            &unknowns,
            &parameters,
            &proposed
        ),
        Err(RootBoxError::ResourceLimit(_))
    ));
}

#[test]
fn maximum_total_dimension_keeps_parameter_columns_out_of_the_unknown_map() {
    let unknowns: Vec<_> = (0..15).map(|i| Symbol::new(format!("x{i}"))).collect();
    let parameters = [Symbol::new("p")];
    let source: Vec<_> = unknowns
        .iter()
        .map(|x| Expr::Sym(x.clone()) - Expr::Sym(parameters[0].clone()))
        .collect();
    let proposed = witness(vec![bounds(-2, 2); unknowns.len()], vec![bounds(-1, 1)]);
    let accepted = certify_parametric_root_box(&source, &unknowns, &parameters, &proposed).unwrap();
    assert_eq!(
        accepted.root_enclosure(),
        vec![bounds(-1, 1); unknowns.len()]
    );
    assert_eq!(accepted.contraction_bound(), &q(0, 1));
}

#[test]
fn actual_caller_accounting_survives_success_and_mathematical_refusal() {
    let (source, unknowns, parameters, proposed) = affine_family();
    let mut budget = Budget::new(BudgetLimits::uniform(100_000_000, 10_000));
    verify_parametric_root_box(&source, &unknowns, &parameters, &proposed, &mut budget).unwrap();
    let consumed = budget.snapshot();
    assert!(matches!(
        verify_parametric_root_box(&source, &unknowns, &parameters, &proposed, &mut budget),
        Err(RootBoxError::Meter(MeterError::Budget(_)))
    ));
    assert_eq!(budget.snapshot(), consumed);
    assert_eq!(budget.verifier_remaining(), 10_000);

    let mut budget = Budget::new(BudgetLimits::uniform(100_000_000, 10_000));
    let mut bad = proposed.clone();
    bad.parameter_domain[0] = bounds(-10, 10);
    assert!(matches!(
        verify_parametric_root_box(&source, &unknowns, &parameters, &bad, &mut budget),
        Err(RootBoxError::NoInteriorInclusion { .. })
    ));
    let refused_consumption = budget.snapshot();
    assert!(budget.remaining(Dimension::MemoryBytes) < 64 * 1024 * 1024);
    assert!(matches!(
        verify_parametric_root_box(&source, &unknowns, &parameters, &proposed, &mut budget),
        Err(RootBoxError::Meter(MeterError::Budget(_)))
    ));
    assert_eq!(budget.snapshot(), refused_consumption);
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
fn cancellation_at_every_checkpoint_prevents_uniform_receipt_publication() {
    let (source, unknowns, parameters, proposed) = affine_family();
    let mut count = Cancel::default();
    verify_parametric_root_box(&source, &unknowns, &parameters, &proposed, &mut count).unwrap();
    for stop in 0..count.visits {
        let mut cancel = Cancel {
            visits: 0,
            stop: Some(stop),
        };
        assert_eq!(
            verify_parametric_root_box(&source, &unknowns, &parameters, &proposed, &mut cancel),
            Err(RootBoxError::Meter(MeterError::Cancelled)),
            "safe point {stop}"
        );
    }
    let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(matches!(
        verify_parametric_root_box(&source, &unknowns, &parameters, &proposed, &mut empty),
        Err(RootBoxError::Meter(MeterError::Budget(_)))
    ));
}
