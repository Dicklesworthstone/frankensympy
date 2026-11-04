//! Analytic families, complete closed parameter coverage and hostile replay.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{
    BigInt, BigRational, Expr, MAX_PARAMETRIC_ROOT_COVER_DEPTH, MAX_PARAMETRIC_ROOT_COVER_LEAVES,
    MAX_PARAMETRIC_ROOT_COVER_NODES, MAX_ROOT_BOX_LIMBS, MAX_ROOT_BOX_STEPS,
    ParametricRootBoxWitness, ParametricRootCoverError, ParametricRootCoverReceipt,
    ParametricRootCoverStep, ParametricRootCoverWitness, ParametricRootCoverWorkspace, RealBall,
    RootBoxError, Symbol, SymbolIdentity, certify_parametric_root_box,
    certify_parametric_root_cover, verify_parametric_root_cover,
};
use std::{cell::Cell, error::Error, sync::Arc};

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
fn local(unknown_domain: &[RealBall], parameter_domain: Vec<RealBall>) -> ParametricRootBoxWitness {
    let n = unknown_domain.len();
    ParametricRootBoxWitness {
        unknown_domain: unknown_domain.to_vec(),
        parameter_domain,
        center: unknown_domain
            .iter()
            .map(|value| value.midpoint().clone())
            .collect(),
        preconditioner: (0..n)
            .map(|i| (0..n).map(|j| q(i64::from(i == j), 1)).collect())
            .collect(),
        precision_digits: 24,
    }
}

#[derive(Clone)]
struct Fixture {
    residuals: Vec<Expr>,
    unknowns: Vec<Symbol>,
    parameters: Vec<Symbol>,
    unknown_domain: Vec<RealBall>,
    parameter_domain: Vec<RealBall>,
    witness: ParametricRootCoverWitness,
}

impl Fixture {
    fn verify(&self) -> Result<ParametricRootCoverReceipt, ParametricRootCoverError> {
        certify_parametric_root_cover(
            &self.residuals,
            &self.unknowns,
            &self.parameters,
            &self.unknown_domain,
            &self.parameter_domain,
            &self.witness,
        )
    }
    fn metered(
        &self,
        meter: &mut impl BudgetMeter,
    ) -> Result<ParametricRootCoverReceipt, ParametricRootCoverError> {
        verify_parametric_root_cover(
            &self.residuals,
            &self.unknowns,
            &self.parameters,
            &self.unknown_domain,
            &self.parameter_domain,
            &self.witness,
            meter,
        )
    }
    fn one_leaf(&mut self) {
        self.witness.leaves = vec![local(&self.unknown_domain, self.parameter_domain.clone())];
        self.witness.partition = vec![ParametricRootCoverStep::Leaf { index: 0 }];
    }
}

fn linear() -> Fixture {
    let unknowns = vec![Symbol::new("x")];
    let parameters = vec![Symbol::new("p")];
    let residuals = vec![Expr::Sym(unknowns[0].clone()) - Expr::Sym(parameters[0].clone())];
    let unknown_domain = vec![bounds(-2, 2)];
    let leaves = vec![
        local(&unknown_domain, vec![bounds(-1, 0)]),
        local(&unknown_domain, vec![bounds(0, 1)]),
    ];
    Fixture {
        residuals,
        unknowns,
        parameters,
        unknown_domain,
        parameter_domain: vec![bounds(-1, 1)],
        witness: ParametricRootCoverWitness {
            leaves,
            partition: vec![
                ParametricRootCoverStep::Split { axis: 0 },
                ParametricRootCoverStep::Leaf { index: 0 },
                ParametricRootCoverStep::Leaf { index: 1 },
            ],
            precision_digits: 24,
        },
    }
}

fn balanced(partition: &mut Vec<ParametricRootCoverStep>, start: usize, count: usize) {
    if count == 1 {
        partition.push(ParametricRootCoverStep::Leaf { index: start });
    } else {
        partition.push(ParametricRootCoverStep::Split { axis: 0 });
        balanced(partition, start, count / 2);
        balanced(partition, start + count / 2, count / 2);
    }
}

#[test]
fn subdivision_certifies_a_family_the_single_uniform_box_cannot_accept() {
    let mut fixture = linear();
    let p = Expr::Sym(fixture.parameters[0].clone());
    fixture.residuals = vec![
        (Expr::from_i64(1) + power(p, 2)) * Expr::Sym(fixture.unknowns[0].clone())
            - Expr::from_i64(1),
    ];
    fixture.unknown_domain = vec![bounds(0, 2)];
    fixture.parameter_domain = vec![bounds(0, 2)];
    let mut coarse = local(&fixture.unknown_domain, fixture.parameter_domain.clone());
    coarse.preconditioner[0][0] = q(1, 2);
    assert_eq!(
        certify_parametric_root_box(
            &fixture.residuals,
            &fixture.unknowns,
            &fixture.parameters,
            &coarse
        ),
        Err(RootBoxError::NotContractive)
    );
    fixture.witness.leaves = (0..8)
        .map(|i| {
            let mut proposed = local(
                &fixture.unknown_domain,
                vec![ball(q(2 * i + 1, 8), q(1, 8))],
            );
            proposed.preconditioner[0][0] = q(64, 64 + (2 * i + 1).pow(2));
            proposed
        })
        .collect();
    fixture.witness.partition.clear();
    balanced(&mut fixture.witness.partition, 0, 8);
    let accepted = fixture.verify().unwrap();
    assert_eq!(accepted.leaf_count(), 8);
    assert_eq!(accepted.residuals(), fixture.residuals);
    assert_eq!(accepted.unknowns(), fixture.unknowns);
    assert_eq!(accepted.parameters(), fixture.parameters);
    assert_eq!(accepted.unknown_domain(), fixture.unknown_domain);
    assert_eq!(accepted.parameter_domain(), fixture.parameter_domain);
    assert_eq!(accepted.precision_digits(), 24);
    assert_eq!(accepted.witness(), &fixture.witness);
    for (i, leaf) in accepted.leaves().iter().enumerate() {
        let i = i as i64;
        let denominator = 64 + (2 * i + 1).pow(2);
        // Direct interval algebra for c=1, X=[0,2], P_i=[i/4,(i+1)/4]:
        // q_i=(4i+3)/denominator and K_i=[(58-8i)/denominator,(68+8i)/denominator].
        assert_eq!(leaf.contraction_bound(), &q(4 * i + 3, denominator));
        assert_eq!(leaf.root_enclosure()[0].lower(), q(58 - 8 * i, denominator));
        assert_eq!(leaf.root_enclosure()[0].upper(), q(68 + 8 * i, denominator));
        // The analytic family x(p)=1/(1+p²) is decreasing on [0,2].
        assert!(leaf.root_enclosure()[0].contains(&q(16, 16 + i * i)));
        assert!(leaf.root_enclosure()[0].contains(&q(16, 16 + (i + 1).pow(2))));
    }
    assert_eq!(accepted.root_enclosure()[0].lower(), q(2, 289));
    assert_eq!(accepted.root_enclosure()[0].upper(), q(68, 65));
    assert!(accepted.root_enclosure()[0].contains(&q(1, 5)));
    assert!(accepted.root_enclosure()[0].contains(&q(1, 1)));
}

#[test]
fn coupled_two_parameter_cover_preserves_all_faces_and_the_exact_family_hull() {
    let unknowns = vec![Symbol::new("x"), Symbol::new("y")];
    let parameters = vec![Symbol::new("p"), Symbol::new("q")];
    let p = Expr::Sym(parameters[0].clone());
    let q_expr = Expr::Sym(parameters[1].clone());
    let residuals = vec![
        Expr::Sym(unknowns[0].clone()) - p.clone() - q_expr.clone(),
        Expr::Sym(unknowns[1].clone()) - p + q_expr,
    ];
    let unknown_domain = vec![bounds(-3, 3); 2];
    let mut leaves = Vec::new();
    for p in [bounds(-1, 0), bounds(0, 1)] {
        for q in [bounds(-1, 0), bounds(0, 1)] {
            leaves.push(local(&unknown_domain, vec![p.clone(), q]));
        }
    }
    let fixture = Fixture {
        residuals,
        unknowns,
        parameters,
        unknown_domain,
        parameter_domain: vec![bounds(-1, 1); 2],
        witness: ParametricRootCoverWitness {
            leaves,
            partition: vec![
                ParametricRootCoverStep::Split { axis: 0 },
                ParametricRootCoverStep::Split { axis: 1 },
                ParametricRootCoverStep::Leaf { index: 0 },
                ParametricRootCoverStep::Leaf { index: 1 },
                ParametricRootCoverStep::Split { axis: 1 },
                ParametricRootCoverStep::Leaf { index: 2 },
                ParametricRootCoverStep::Leaf { index: 3 },
            ],
            precision_digits: 24,
        },
    };
    let accepted = fixture.verify().unwrap();
    assert_eq!(accepted.root_enclosure(), [bounds(-2, 2), bounds(-2, 2)]);
    for leaf in accepted.leaves() {
        assert_eq!(leaf.contraction_bound(), &q(0, 1));
        // Both parameter coordinates are zero at the common four-way face.
        // All four proofs agree there on the same unknown root (0,0).
        assert!(
            leaf.root_enclosure()
                .iter()
                .all(|value| value.contains(&q(0, 1)))
        );
    }
}

#[test]
fn point_parameters_are_allowed_but_splitting_a_point_is_not_coverage() {
    let mut fixture = linear();
    fixture.parameters.push(Symbol::new("fixed"));
    fixture.parameter_domain.push(RealBall::from_i64(2));
    for leaf in &mut fixture.witness.leaves {
        leaf.parameter_domain.push(RealBall::from_i64(2));
    }
    let accepted = fixture.verify().unwrap();
    assert_eq!(accepted.root_enclosure(), [bounds(-1, 1)]);
    fixture.witness.partition[0] = ParametricRootCoverStep::Split { axis: 1 };
    assert!(matches!(
        fixture.verify(),
        Err(ParametricRootCoverError::InvalidPartition(_))
    ));

    let mut fixture = linear();
    fixture.parameter_domain = vec![RealBall::from_i64(1)];
    fixture.one_leaf();
    assert_eq!(
        fixture.verify().unwrap().root_enclosure(),
        [RealBall::from_i64(1)]
    );
    fixture
        .witness
        .partition
        .insert(0, ParametricRootCoverStep::Split { axis: 0 });
    assert!(matches!(
        fixture.verify(),
        Err(ParametricRootCoverError::InvalidPartition(_))
    ));
}

#[test]
fn every_leaf_reference_and_both_children_are_required_exactly_once() {
    let original = linear();
    let mut cases = Vec::new();
    let mut bad = original.clone();
    bad.witness.partition.pop();
    cases.push(("missing right child", bad));
    let mut bad = original.clone();
    bad.witness
        .partition
        .push(ParametricRootCoverStep::Leaf { index: 0 });
    cases.push(("trailing token", bad));
    let mut bad = original.clone();
    bad.witness.partition[2] = ParametricRootCoverStep::Leaf { index: 0 };
    cases.push(("duplicate leaf reference", bad));
    let mut bad = original.clone();
    bad.witness.partition[2] = ParametricRootCoverStep::Leaf { index: 2 };
    cases.push(("out-of-range reference", bad));
    let mut bad = original.clone();
    bad.witness.leaves.push(bad.witness.leaves[0].clone());
    cases.push(("unreferenced proposal", bad));
    let mut bad = original.clone();
    bad.witness.partition.clear();
    cases.push(("empty partition", bad));
    let mut bad = original.clone();
    bad.witness.leaves.clear();
    cases.push(("empty leaf list", bad));
    for (name, bad) in cases {
        assert!(
            matches!(
                bad.verify(),
                Err(ParametricRootCoverError::InvalidPartition(_))
            ),
            "{name}"
        );
    }
    let mut reordered = original;
    reordered.witness.leaves.swap(0, 1);
    reordered.witness.partition[1] = ParametricRootCoverStep::Leaf { index: 1 };
    reordered.witness.partition[2] = ParametricRootCoverStep::Leaf { index: 0 };
    let accepted = reordered.verify().unwrap();
    // Receipt indices retain witness indexing even if replay order differs.
    assert_eq!(accepted.leaves()[0].witness(), &reordered.witness.leaves[0]);
    assert_eq!(accepted.leaves()[1].witness(), &reordered.witness.leaves[1]);
}

#[test]
fn sampled_shifted_or_broadened_parameter_boxes_cannot_replace_exact_cells() {
    let original = linear();
    for replacement in [
        RealBall::exact(q(-1, 2)),
        bounds(-2, -1),
        bounds(-1, 1),
        bounds(0, 1),
    ] {
        let mut bad = original.clone();
        bad.witness.leaves[0].parameter_domain[0] = replacement;
        assert!(matches!(
            bad.verify(),
            Err(ParametricRootCoverError::InvalidPartition(_))
        ));
    }
    let mut bad = original;
    bad.parameter_domain[0] = bounds(-2, 2);
    assert!(matches!(
        bad.verify(),
        Err(ParametricRootCoverError::InvalidPartition(_))
    ));
}

#[test]
fn a_valid_smaller_unknown_box_cannot_hide_other_roots_in_the_declared_box() {
    let mut fixture = linear();
    fixture.residuals = vec![power(Expr::Sym(fixture.unknowns[0].clone()), 2) - Expr::from_i64(1)];
    let mut smaller = local(&[ball(q(1, 1), q(1, 2))], fixture.parameter_domain.clone());
    smaller.preconditioner[0][0] = q(1, 2);
    certify_parametric_root_box(
        &fixture.residuals,
        &fixture.unknowns,
        &fixture.parameters,
        &smaller,
    )
    .unwrap();
    fixture.witness.leaves = vec![smaller];
    fixture.witness.partition = vec![ParametricRootCoverStep::Leaf { index: 0 }];
    // x=+1 is locally unique in [1/2,3/2], but the requested [-2,2]
    // contains both +1 and -1 for every parameter value.
    assert!(matches!(
        fixture.verify(),
        Err(ParametricRootCoverError::InvalidPartition(_))
    ));
}

#[test]
fn every_leaf_is_recomputed_with_original_source_precision_and_typed_roles() {
    let original = linear();
    let mut bad = original.clone();
    bad.witness.leaves[1].preconditioner[0][0] = q(0, 1);
    assert_eq!(
        bad.verify(),
        Err(ParametricRootCoverError::Verification(
            RootBoxError::NotContractive
        ))
    );
    let mut bad = original.clone();
    bad.witness.leaves[1].precision_digits += 1;
    assert_eq!(
        bad.verify(),
        Err(ParametricRootCoverError::Verification(
            RootBoxError::InvalidPrecision
        ))
    );
    let mut bad = original.clone();
    bad.residuals[0] = bad.residuals[0].clone() + Expr::from_i64(5);
    assert!(matches!(
        bad.verify(),
        Err(ParametricRootCoverError::Verification(
            RootBoxError::NoInteriorInclusion { .. }
        ))
    ));
    let mut bad = original.clone();
    bad.parameters[0] =
        Symbol::with_identity("p", SymbolIdentity::from_assumptions_digest([9; 32]));
    assert!(matches!(
        bad.verify(),
        Err(ParametricRootCoverError::Verification(
            RootBoxError::Unsupported(_)
        ))
    ));
    let mut bad = original;
    std::mem::swap(&mut bad.unknowns[0], &mut bad.parameters[0]);
    assert_eq!(
        bad.verify(),
        Err(ParametricRootCoverError::Verification(
            RootBoxError::NotContractive
        ))
    );
}

#[test]
fn closed_split_faces_cannot_omit_a_singular_parameter_or_boundary_root() {
    let mut fixture = linear();
    fixture.unknown_domain = vec![bounds(-1, 5)];
    fixture.parameter_domain = vec![bounds(0, 1)];
    fixture.residuals = vec![
        Expr::Sym(fixture.parameters[0].clone()) * Expr::Sym(fixture.unknowns[0].clone())
            - Expr::from_i64(1),
    ];
    fixture.witness.leaves = vec![
        local(&fixture.unknown_domain, vec![ball(q(1, 4), q(1, 4))]),
        local(&fixture.unknown_domain, vec![ball(q(3, 4), q(1, 4))]),
    ];
    fixture.witness.leaves[0].preconditioner[0][0] = q(4, 1);
    fixture.witness.leaves[1].preconditioner[0][0] = q(4, 3);
    assert_eq!(
        fixture.verify(),
        Err(ParametricRootCoverError::Verification(
            RootBoxError::NotContractive
        ))
    );

    let mut fixture = linear();
    fixture.unknown_domain = vec![bounds(-1, 1)];
    for leaf in &mut fixture.witness.leaves {
        leaf.unknown_domain = fixture.unknown_domain.clone();
    }
    assert!(matches!(
        fixture.verify(),
        Err(ParametricRootCoverError::Verification(
            RootBoxError::NoInteriorInclusion { .. }
        ))
    ));
}

#[test]
fn original_held_parameter_poles_survive_all_residual_and_seam_checks() {
    let mut fixture = linear();
    fixture.unknowns.push(Symbol::new("y"));
    fixture.unknown_domain.push(bounds(-2, 2));
    let pole = power(Expr::Sym(fixture.parameters[0].clone()), -1);
    for hidden in [
        Expr::Mul(vec![Expr::from_i64(0), pole.clone()]),
        power(pole, 0),
    ] {
        let mut bad = fixture.clone();
        bad.residuals = vec![
            Expr::from_i64(1),
            Expr::Add(vec![Expr::Sym(bad.unknowns[1].clone()), hidden]),
        ];
        bad.witness.leaves = vec![
            local(&bad.unknown_domain, vec![bounds(-1, 0)]),
            local(&bad.unknown_domain, vec![bounds(0, 1)]),
        ];
        // A nonzero first residual may not hide the second row's original
        // domain hole; both closed leaves retain the pole at p=0.
        assert!(matches!(
            bad.verify(),
            Err(ParametricRootCoverError::Verification(
                RootBoxError::Domain(_) | RootBoxError::Ball(_)
            ))
        ));
    }
}

#[test]
fn transcendental_family_bounds_retain_full_parameter_intervals() {
    let mut fixture = linear();
    fixture.residuals = vec![
        Expr::Sym(fixture.unknowns[0].clone())
            - Expr::Function("sin".into(), vec![Expr::Sym(fixture.parameters[0].clone())]),
    ];
    let accepted = fixture.verify().unwrap();
    // sin(1)>1-1/6>4/5 from an independent alternating-series bound.
    assert!(accepted.root_enclosure()[0].contains(&q(-4, 5)));
    assert!(accepted.root_enclosure()[0].contains(&q(4, 5)));
    assert!(accepted.root_enclosure()[0].lower() > q(-2, 1));
    assert!(accepted.root_enclosure()[0].upper() < q(2, 1));
}

#[test]
fn exact_sub_binary64_parameter_cells_preserve_the_whole_root_family() {
    let tiny = BigRational::new(BigInt::from(1), BigInt::from(10).pow(400));
    let mut fixture = linear();
    fixture.unknown_domain = vec![ball(&tiny * q(3, 2), &tiny * q(3, 2))];
    fixture.parameter_domain = vec![ball(&tiny * q(3, 2), &tiny / q(2, 1))];
    fixture.witness.leaves = vec![
        local(
            &fixture.unknown_domain,
            vec![ball(&tiny * q(5, 4), &tiny / q(4, 1))],
        ),
        local(
            &fixture.unknown_domain,
            vec![ball(&tiny * q(7, 4), &tiny / q(4, 1))],
        ),
    ];
    let accepted = fixture.verify().unwrap();
    assert_eq!(accepted.root_enclosure(), fixture.parameter_domain);
    assert_eq!(accepted.root_enclosure()[0].lower(), tiny);
    assert_eq!(accepted.root_enclosure()[0].upper(), &tiny * q(2, 1));
}

#[test]
fn structural_and_scalar_bounds_precede_untrusted_payload_copies() {
    let original = linear();
    let mut bad = original.clone();
    bad.witness.partition =
        vec![ParametricRootCoverStep::Split { axis: 0 }; MAX_PARAMETRIC_ROOT_COVER_NODES + 1];
    assert!(matches!(
        bad.verify(),
        Err(ParametricRootCoverError::InvalidPartition(_))
    ));
    let mut bad = original.clone();
    bad.witness.leaves = vec![bad.witness.leaves[0].clone(); MAX_PARAMETRIC_ROOT_COVER_LEAVES + 1];
    assert!(matches!(
        bad.verify(),
        Err(ParametricRootCoverError::InvalidPartition(_))
    ));
    let mut bad = original.clone();
    bad.witness.partition =
        vec![ParametricRootCoverStep::Split { axis: 0 }; MAX_PARAMETRIC_ROOT_COVER_DEPTH + 1];
    assert!(matches!(
        bad.verify(),
        Err(ParametricRootCoverError::InvalidPartition(_))
    ));
    let mut bad = original.clone();
    bad.witness.partition[0] = ParametricRootCoverStep::Split { axis: 1 };
    assert!(matches!(
        bad.verify(),
        Err(ParametricRootCoverError::InvalidPartition(_))
    ));
    let mut bad = original.clone();
    bad.parameters = (0..16).map(|i| Symbol::new(format!("p{i}"))).collect();
    bad.parameter_domain = vec![bounds(-1, 1); 16];
    assert_eq!(
        bad.verify(),
        Err(ParametricRootCoverError::Verification(
            RootBoxError::DimensionMismatch
        ))
    );
    let mut bad = original.clone();
    bad.parameters[0] = bad.unknowns[0].clone();
    assert_eq!(
        bad.verify(),
        Err(ParametricRootCoverError::Verification(
            RootBoxError::DuplicateVariable
        ))
    );
    let mut bad = original.clone();
    bad.parameters[0] = Symbol::new("p".repeat(4097));
    assert!(matches!(
        bad.verify(),
        Err(ParametricRootCoverError::Verification(
            RootBoxError::ResourceLimit(_)
        ))
    ));
    let huge = BigRational::from_integer(BigInt::from(1) << (64 * MAX_ROOT_BOX_LIMBS) as u32);
    let mut bad = original.clone();
    bad.witness.leaves[0].preconditioner[0][0] = huge.clone();
    assert!(matches!(
        bad.verify(),
        Err(ParametricRootCoverError::Verification(
            RootBoxError::ResourceLimit(_)
        ))
    ));
    let mut bad = original;
    bad.residuals[0] = Expr::Add(vec![
        bad.residuals[0].clone(),
        Expr::Mul(vec![Expr::from_i64(0), Expr::Rational(huge)]),
    ]);
    assert!(matches!(
        bad.verify(),
        Err(ParametricRootCoverError::Verification(
            RootBoxError::ResourceLimit(_)
        ))
    ));
}

struct Tracking<'a> {
    totals: &'a Cell<[u64; 3]>,
    scratch_reservations: &'a Cell<usize>,
}

impl BudgetMeter for Tracking<'_> {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        let index = match dimension {
            Dimension::ComputeSteps => 0,
            Dimension::MemoryBytes => 1,
            Dimension::AllocationCount => 2,
            _ => return Ok(()),
        };
        let mut totals = self.totals.get();
        totals[index] += amount;
        self.totals.set(totals);
        if dimension == Dimension::MemoryBytes && amount == 64 * 1024 * 1024 {
            self.scratch_reservations
                .set(self.scratch_reservations.get() + 1);
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
        Ok(())
    }
}

#[test]
fn workspace_reuses_one_scratch_and_one_source_preflight_through_failed_trials() {
    let fixture = linear();
    let totals = Cell::new([0; 3]);
    let scratch = Cell::new(0);
    let mut meter = Tracking {
        totals: &totals,
        scratch_reservations: &scratch,
    };
    let mut workspace = ParametricRootCoverWorkspace::new(
        &fixture.residuals,
        &fixture.unknowns,
        &fixture.parameters,
        &fixture.unknown_domain,
        &fixture.parameter_domain,
        24,
        &mut meter,
    )
    .unwrap();
    let initial = totals.get();
    workspace.verify_local(&fixture.witness.leaves[0]).unwrap();
    let first = totals.get();
    workspace.verify_local(&fixture.witness.leaves[0]).unwrap();
    let second = totals.get();
    for i in 0..3 {
        assert_eq!(first[i] - initial[i], second[i] - first[i]);
    }
    let mut bad = fixture.witness.leaves[0].clone();
    bad.preconditioner[0][0] = q(0, 1);
    assert_eq!(
        workspace.verify_local(&bad),
        Err(ParametricRootCoverError::Verification(
            RootBoxError::NotContractive
        ))
    );
    assert!(totals.get()[0] > second[0]);
    assert!(totals.get()[1] > second[1]);
    workspace.verify_local(&fixture.witness.leaves[0]).unwrap();
    assert_eq!(scratch.get(), 1);
    let enclosure = workspace
        .enclose_cell(&[RealBall::from_i64(0)], &[RealBall::from_i64(0)])
        .unwrap();
    assert_eq!(enclosure.residual_values(), [RealBall::from_i64(0)]);
    assert_eq!(
        enclosure.jacobian(),
        [vec![RealBall::from_i64(1), RealBall::from_i64(-1)]]
    );
}

#[test]
fn workspace_arithmetic_and_intrinsic_limits_remain_cumulative_without_a_caller_budget() {
    let fixture = linear();
    let mut meter = Unbounded;
    let mut workspace = ParametricRootCoverWorkspace::new(
        &fixture.residuals,
        &fixture.unknowns,
        &fixture.parameters,
        &fixture.unknown_domain,
        &fixture.parameter_domain,
        24,
        &mut meter,
    )
    .unwrap();
    assert_eq!(workspace.add(&q(1, 3), &q(1, 6)).unwrap(), q(1, 2));
    assert_eq!(workspace.subtract(&q(1, 3), &q(1, 6)).unwrap(), q(1, 6));
    assert_eq!(workspace.multiply(&q(2, 3), &q(3, 4)).unwrap(), q(1, 2));
    assert_eq!(workspace.divide(&q(2, 3), &q(4, 3)).unwrap(), q(1, 2));
    assert!(workspace.compare(&q(1, 3), &q(1, 2)).unwrap().is_lt());
    assert_eq!(
        workspace.divide(&q(1, 1), &q(0, 1)),
        Err(ParametricRootCoverError::Verification(
            RootBoxError::SingularPreconditioner
        ))
    );
    for _ in 0..4 {
        workspace.account(MAX_ROOT_BOX_STEPS / 5, 0, 0).unwrap();
    }
    assert!(matches!(
        workspace.account(MAX_ROOT_BOX_STEPS / 5, 0, 0),
        Err(ParametricRootCoverError::Verification(
            RootBoxError::ResourceLimit(_)
        ))
    ));
}

#[test]
fn actual_shared_budget_cannot_be_reset_after_success_or_late_leaf_refusal() {
    let fixture = linear();
    let mut budget = Budget::new(BudgetLimits::uniform(100_000_000, 10_000));
    fixture.metered(&mut budget).unwrap();
    let consumed = budget.snapshot();
    assert!(matches!(
        fixture.metered(&mut budget),
        Err(ParametricRootCoverError::Verification(RootBoxError::Meter(
            MeterError::Budget(_)
        )))
    ));
    assert_eq!(budget.snapshot(), consumed);
    assert_eq!(budget.verifier_remaining(), 10_000);

    let mut bad = fixture.clone();
    bad.witness.leaves[1].preconditioner[0][0] = q(0, 1);
    let mut budget = Budget::new(BudgetLimits::uniform(100_000_000, 10_000));
    assert_eq!(
        bad.metered(&mut budget),
        Err(ParametricRootCoverError::Verification(
            RootBoxError::NotContractive
        ))
    );
    assert!(budget.remaining(Dimension::MemoryBytes) < 64 * 1024 * 1024);
    let consumed = budget.snapshot();
    assert!(matches!(
        fixture.metered(&mut budget),
        Err(ParametricRootCoverError::Verification(RootBoxError::Meter(
            MeterError::Budget(_)
        )))
    ));
    assert_eq!(budget.snapshot(), consumed);
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
fn every_checkpoint_including_final_publication_cancels_without_a_partial_cover() {
    let fixture = linear();
    let mut count = Cancel::default();
    fixture.metered(&mut count).unwrap();
    for stop in 0..count.visits {
        let mut meter = Cancel {
            visits: 0,
            stop: Some(stop),
        };
        let error = fixture.metered(&mut meter).unwrap_err();
        assert_eq!(
            error,
            ParametricRootCoverError::Verification(RootBoxError::Meter(MeterError::Cancelled)),
            "safe point {stop}"
        );
        assert!(error.source().unwrap().is::<RootBoxError>());
        assert!(matches!(
            error
                .source()
                .unwrap()
                .source()
                .unwrap()
                .downcast_ref::<MeterError>(),
            Some(MeterError::Cancelled)
        ));
    }
    let mut empty = Budget::new(BudgetLimits::uniform(0, 0));
    assert!(matches!(
        fixture.metered(&mut empty),
        Err(ParametricRootCoverError::Verification(RootBoxError::Meter(
            MeterError::Budget(_)
        )))
    ));
}
