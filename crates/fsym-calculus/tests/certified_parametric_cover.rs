use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::parametric_cover::{ParametricRootCoverOptions, propose_parametric_root_cover};
use fsym_core::{
    BigRational, Expr, ParametricRootBoxWitness, ParametricRootCoverError,
    ParametricRootCoverReceipt, ParametricRootCoverStep, ParametricRootCoverWitness, RealBall,
    RootBoxError, Symbol, certify_parametric_root_box, verify_parametric_root_cover,
};
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn bounds(lower: BigRational, upper: BigRational) -> RealBall {
    RealBall::new((&lower + &upper) / q(2, 1), (upper - lower) / q(2, 1)).unwrap()
}

fn interval(lower: i64, upper: i64) -> RealBall {
    bounds(q(lower, 1), q(upper, 1))
}

fn power(base: Expr, exponent: i64) -> Expr {
    Expr::Pow(Arc::new(base), Arc::new(Expr::from_i64(exponent)))
}

fn regular_source(x: &Symbol, p: &Symbol) -> Expr {
    (Expr::from_i64(1) + power(Expr::Sym(p.clone()), 2)) * Expr::Sym(x.clone()) - Expr::from_i64(1)
}

fn options() -> ParametricRootCoverOptions {
    ParametricRootCoverOptions {
        precision_digits: 16,
        max_leaves: 512,
    }
}

fn checked(
    source: &[Expr],
    unknowns: &[Symbol],
    parameters: &[Symbol],
    unknown_domain: &[RealBall],
    parameter_domain: &[RealBall],
) -> ParametricRootCoverReceipt {
    let mut budget = Budget::new(BudgetLimits::uniform(512 * 1024 * 1024, 0));
    let witness = propose_parametric_root_cover(
        source,
        unknowns,
        parameters,
        unknown_domain,
        parameter_domain,
        options(),
        &mut budget,
    )
    .unwrap();
    let receipt = verify_parametric_root_cover(
        source,
        unknowns,
        parameters,
        unknown_domain,
        parameter_domain,
        &witness,
        &mut budget,
    )
    .unwrap();
    assert_eq!(receipt.residuals(), source);
    assert_eq!(receipt.unknowns(), unknowns);
    assert_eq!(receipt.parameters(), parameters);
    assert_eq!(receipt.unknown_domain(), unknown_domain);
    assert_eq!(receipt.parameter_domain(), parameter_domain);
    assert_eq!(receipt.precision_digits(), 16);
    assert_eq!(receipt.leaf_count(), witness.leaves.len());
    assert_eq!(receipt.leaves().len(), witness.leaves.len());
    assert_eq!(receipt.witness(), &witness);
    assert_eq!(witness.partition.len(), 2 * receipt.leaf_count() - 1);
    for leaf in receipt.leaves() {
        assert_eq!(leaf.residuals(), source);
        assert_eq!(leaf.unknowns(), unknowns);
        assert_eq!(leaf.parameters(), parameters);
        assert_eq!(leaf.witness().unknown_domain, unknown_domain);
        assert_eq!(leaf.witness().precision_digits, 16);
        assert!(leaf.contraction_bound() < &q(1, 1));
        for (outer, inner) in parameter_domain
            .iter()
            .zip(&leaf.witness().parameter_domain)
        {
            assert!(outer.contains_ball(inner));
        }
        for (hull, image) in receipt.root_enclosure().iter().zip(leaf.root_enclosure()) {
            assert!(hull.contains_ball(image));
        }
    }
    for (outer, hull) in unknown_domain.iter().zip(receipt.root_enclosure()) {
        assert!(outer.lower() < hull.lower());
        assert!(hull.upper() < outer.upper());
    }
    receipt
}

#[test]
fn subdivision_certifies_a_regular_family_that_one_uniform_proposal_cannot() {
    let unknowns = [Symbol::new("x")];
    let parameters = [Symbol::new("p")];
    let source = [regular_source(&unknowns[0], &parameters[0])];
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
        certify_parametric_root_box(&source, &unknowns, &parameters, &coarse),
        Err(RootBoxError::NotContractive)
    );
    let receipt = checked(
        &source,
        &unknowns,
        &parameters,
        &unknown_domain,
        &parameter_domain,
    );
    assert!(receipt.leaf_count() > 1);
    assert!(receipt.root_enclosure()[0].contains(&q(1, 5)));
    assert!(receipt.root_enclosure()[0].contains(&q(1, 1)));
    for leaf in receipt.leaves() {
        let domain = &leaf.witness().parameter_domain[0];
        // Analytic extreme roots are 1/(1+p^2), independently of search.
        for p in [domain.lower(), domain.upper()] {
            let root = q(1, 1) / (q(1, 1) + &p * &p);
            assert!(leaf.root_enclosure()[0].contains(&root));
        }
    }
}

#[test]
fn nonlinear_unknown_family_can_remain_one_complete_leaf() {
    let x = Symbol::new("x");
    let p = Symbol::new("p");
    let source = [power(Expr::Sym(x.clone()), 2) - Expr::Sym(p.clone())];
    let receipt = checked(&source, &[x], &[p], &[interval(1, 2)], &[interval(2, 3)]);
    assert_eq!(receipt.leaf_count(), 1);
    let image = &receipt.root_enclosure()[0];
    assert!(&image.lower() * &image.lower() <= q(2, 1));
    assert!(&image.upper() * &image.upper() >= q(3, 1));
}

#[test]
fn transcendental_parameter_family_covers_the_whole_interval_without_sampling_claims() {
    let x = Symbol::new("x");
    let p = Symbol::new("p");
    let coefficient = Expr::Function("exp".into(), vec![Expr::Sym(p.clone())]);
    let source = [coefficient * Expr::Sym(x.clone()) - Expr::from_i64(1)];
    let receipt = checked(&source, &[x], &[p], &[interval(0, 2)], &[interval(0, 1)]);
    assert!(receipt.leaf_count() > 1);
    for leaf in receipt.leaves() {
        let cell = &leaf.witness().parameter_domain[0];
        // Analytic roots are exp(-p). These independent finer enclosures
        // must fit the certified family images at both extrema.
        for p in [cell.lower(), cell.upper()] {
            let root = RealBall::exact(-p).exp(32).unwrap();
            assert!(leaf.root_enclosure()[0].contains_ball(&root));
        }
    }
}

#[test]
fn coupled_unknown_inverse_uses_only_the_correct_jacobian_columns() {
    let x = Symbol::new("x");
    let y = Symbol::new("y");
    let p = Symbol::new("p");
    let r = Symbol::new("r");
    let source = [
        Expr::Sym(x.clone()) + Expr::from_i64(2) * Expr::Sym(y.clone()) - Expr::Sym(p.clone()),
        Expr::Sym(y.clone()) - Expr::Sym(r.clone()),
    ];
    let receipt = checked(
        &source,
        &[x, y],
        &[p, r],
        &[interval(-4, 4), interval(-2, 2)],
        &[interval(-1, 1), interval(-1, 1)],
    );
    assert_eq!(receipt.leaf_count(), 1);
    assert_eq!(receipt.root_enclosure(), [interval(-3, 3), interval(-1, 1)]);
    assert_eq!(
        receipt.leaves()[0].witness().preconditioner,
        [vec![q(1, 1), q(-2, 1)], vec![q(0, 1), q(1, 1)]]
    );
}

#[test]
fn multidimensional_parameter_cover_skips_points_and_breaks_width_ties_by_axis() {
    let x = Symbol::new("x");
    let fixed = Symbol::new("fixed");
    let p = Symbol::new("p");
    let r = Symbol::new("r");
    let residual =
        (Expr::from_i64(1) + power(Expr::Sym(p.clone()), 2) + power(Expr::Sym(r.clone()), 2))
            * Expr::Sym(x.clone())
            - Expr::from_i64(1);
    let source = [residual];
    let receipt = checked(
        &source,
        &[x],
        &[fixed, p, r],
        &[interval(0, 2)],
        &[interval(7, 7), interval(0, 1), interval(0, 1)],
    );
    assert_eq!(
        receipt.witness().partition[0],
        ParametricRootCoverStep::Split { axis: 1 }
    );
    assert!(
        receipt
            .witness()
            .partition
            .contains(&ParametricRootCoverStep::Split { axis: 2 })
    );
    for leaf in receipt.leaves() {
        let cell = &leaf.witness().parameter_domain;
        assert_eq!(cell[0], RealBall::from_i64(7));
        for p in [cell[1].lower(), cell[1].upper()] {
            for r in [cell[2].lower(), cell[2].upper()] {
                let root = q(1, 1) / (q(1, 1) + &p * &p + &r * &r);
                assert!(leaf.root_enclosure()[0].contains(&root));
            }
        }
    }
}

#[test]
fn all_point_parameters_are_checked_once_without_degenerate_subdivision() {
    let x = Symbol::new("x");
    let p = Symbol::new("p");
    let source = [Expr::Sym(x.clone()) - Expr::Sym(p.clone())];
    let receipt = checked(&source, &[x], &[p], &[interval(0, 2)], &[interval(1, 1)]);
    assert_eq!(receipt.leaf_count(), 1);
    assert_eq!(receipt.root_enclosure(), [RealBall::from_i64(1)]);
}

#[test]
fn parameter_subdivision_can_resolve_original_source_domain_overestimation() {
    let x = Symbol::new("x");
    let p = Symbol::new("p");
    let sp = Expr::Sym(p.clone());
    // Held p*p on [-1,1] encloses [-1,1], making 1+p*p appear to
    // contain zero. Each half proves the denominator positive, but its
    // reciprocal image [1/2,2] still touches X's boundary. Quarter cells
    // give [1/2,8/9] and [4/5,8/7], which lie strictly inside X.
    let denominator = Expr::Add(vec![Expr::from_i64(1), Expr::Mul(vec![sp.clone(), sp])]);
    let source = [Expr::Sym(x.clone()) - power(denominator, -1)];
    let receipt = checked(&source, &[x], &[p], &[interval(0, 2)], &[interval(-1, 1)]);
    assert_eq!(receipt.leaf_count(), 4);
    assert_eq!(receipt.root_enclosure(), [bounds(q(1, 2), q(8, 7))]);
}

#[test]
fn maximum_total_dimension_keeps_all_unknowns_and_the_parameter_binding() {
    let unknowns: Vec<_> = (0..15).map(|i| Symbol::new(format!("x{i}"))).collect();
    let parameters = [Symbol::new("p")];
    let source: Vec<_> = unknowns
        .iter()
        .map(|x| Expr::Sym(x.clone()) - Expr::Sym(parameters[0].clone()))
        .collect();
    let receipt = checked(
        &source,
        &unknowns,
        &parameters,
        &vec![interval(-2, 2); 15],
        &[interval(-1, 1)],
    );
    assert_eq!(receipt.leaf_count(), 1);
    assert_eq!(receipt.root_enclosure(), vec![interval(-1, 1); 15]);
}

#[test]
fn unresolved_outer_parameter_endpoints_do_not_publish_successful_partial_leaves() {
    let x = Symbol::new("x");
    let p = Symbol::new("p");
    let source = [Expr::Sym(x.clone()) - power(Expr::Sym(p.clone()), 2)];
    assert!(matches!(
        propose_parametric_root_cover(
            &source,
            &[x],
            &[p],
            &[bounds(q(-1, 4), q(1, 4))],
            &[interval(0, 1)],
            ParametricRootCoverOptions {
                max_leaves: 16,
                ..options()
            },
            &mut Unbounded,
        ),
        Err(ParametricRootCoverError::Incomplete(_))
    ));
}

#[test]
fn singular_and_unknown_boundary_roots_remain_unproved_after_parameter_refinement() {
    let x = Symbol::new("x");
    let p = Symbol::new("p");
    for (source, unknown_domain, parameter_domain) in [
        (
            power(Expr::Sym(x.clone()), 2) - Expr::Sym(p.clone()),
            interval(-1, 1),
            interval(0, 1),
        ),
        (
            Expr::Sym(p.clone()) * Expr::Sym(x.clone()) - Expr::from_i64(1),
            interval(0, 2),
            interval(0, 1),
        ),
        (
            Expr::Sym(x.clone()) - Expr::Sym(p.clone()),
            interval(-1, 1),
            interval(-1, 1),
        ),
    ] {
        assert!(matches!(
            propose_parametric_root_cover(
                &[source],
                std::slice::from_ref(&x),
                std::slice::from_ref(&p),
                &[unknown_domain],
                &[parameter_domain],
                ParametricRootCoverOptions {
                    max_leaves: 8,
                    ..options()
                },
                &mut Unbounded,
            ),
            Err(ParametricRootCoverError::Incomplete(_))
        ));
    }
}

#[test]
fn an_unresolved_point_parameter_returns_no_cover_instead_of_splitting_a_zero_width() {
    let x = Symbol::new("x");
    let p = Symbol::new("p");
    let source = [power(Expr::Sym(x.clone()), 2) - Expr::Sym(p.clone())];
    assert_eq!(
        propose_parametric_root_cover(
            &source,
            &[x],
            &[p],
            &[interval(-1, 1)],
            &[interval(0, 0)],
            options(),
            &mut Unbounded,
        ),
        Err(ParametricRootCoverError::Incomplete(
            "unresolved parameter cell has no varying coordinate"
        ))
    );
}

#[test]
fn held_parameter_poles_are_never_erased_by_zero_products_or_zero_powers() {
    let x = Symbol::new("x");
    let p = Symbol::new("p");
    let inverse = power(Expr::Sym(p.clone()) - Expr::from_i64(1), -1);
    for hidden in [
        Expr::Mul(vec![Expr::from_i64(0), inverse.clone()]),
        Expr::Add(vec![power(inverse, 0), Expr::from_i64(-1)]),
    ] {
        let source = [Expr::Add(vec![Expr::Sym(x.clone()), hidden])];
        assert!(matches!(
            propose_parametric_root_cover(
                &source,
                std::slice::from_ref(&x),
                std::slice::from_ref(&p),
                &[interval(-1, 1)],
                &[interval(0, 1)],
                ParametricRootCoverOptions {
                    max_leaves: 8,
                    ..options()
                },
                &mut Unbounded,
            ),
            Err(ParametricRootCoverError::Verification(
                RootBoxError::Domain(_) | RootBoxError::Ball(_)
            ))
        ));
    }
}

#[test]
fn independent_verification_rejects_gaps_wrong_leaf_domains_and_wrong_sources() {
    let unknowns = [Symbol::new("x")];
    let parameters = [Symbol::new("p")];
    let source = [regular_source(&unknowns[0], &parameters[0])];
    let unknown_domain = [interval(0, 2)];
    let parameter_domain = [interval(0, 2)];
    let proposed = propose_parametric_root_cover(
        &source,
        &unknowns,
        &parameters,
        &unknown_domain,
        &parameter_domain,
        options(),
        &mut Unbounded,
    )
    .unwrap();
    let verify = |source: &[Expr], witness: &ParametricRootCoverWitness| {
        verify_parametric_root_cover(
            source,
            &unknowns,
            &parameters,
            &unknown_domain,
            &parameter_domain,
            witness,
            &mut Unbounded,
        )
    };
    assert!(verify(&source, &proposed).is_ok());
    let mut missing = proposed.clone();
    missing.partition.pop();
    assert!(matches!(
        verify(&source, &missing),
        Err(ParametricRootCoverError::InvalidPartition(_))
    ));
    let mut sampled = proposed.clone();
    sampled.leaves[0].parameter_domain[0] =
        RealBall::exact(sampled.leaves[0].parameter_domain[0].midpoint().clone());
    assert!(verify(&source, &sampled).is_err());
    let mut changed_unknown = proposed.clone();
    changed_unknown.leaves[0].unknown_domain[0] = interval(-1, 2);
    assert!(verify(&source, &changed_unknown).is_err());
    let changed_source = [source[0].clone() + Expr::from_i64(100)];
    assert!(verify(&changed_source, &proposed).is_err());
}

#[test]
fn deterministic_preorder_and_exact_inverse_proposals_reproduce_the_same_cover() {
    let unknowns = [Symbol::new("x")];
    let parameters = [Symbol::new("p")];
    let source = [regular_source(&unknowns[0], &parameters[0])];
    let propose = || {
        propose_parametric_root_cover(
            &source,
            &unknowns,
            &parameters,
            &[interval(0, 2)],
            &[interval(0, 2)],
            options(),
            &mut Unbounded,
        )
        .unwrap()
    };
    assert_eq!(propose(), propose());
}

#[test]
fn controls_and_unsupported_sources_refuse_before_unbounded_subdivision() {
    let unknowns = [Symbol::new("x")];
    let parameters = [Symbol::new("p")];
    let source = [regular_source(&unknowns[0], &parameters[0])];
    for max_leaves in [0, 1025, usize::MAX] {
        assert!(matches!(
            propose_parametric_root_cover(
                &source,
                &unknowns,
                &parameters,
                &[interval(0, 2)],
                &[interval(0, 2)],
                ParametricRootCoverOptions {
                    max_leaves,
                    ..options()
                },
                &mut Unbounded,
            ),
            Err(ParametricRootCoverError::Verification(
                RootBoxError::ResourceLimit(_)
            ))
        ));
    }
    for precision_digits in [7, 129, u32::MAX] {
        assert!(matches!(
            propose_parametric_root_cover(
                &source,
                &unknowns,
                &parameters,
                &[interval(0, 2)],
                &[interval(0, 2)],
                ParametricRootCoverOptions {
                    precision_digits,
                    ..options()
                },
                &mut Unbounded,
            ),
            Err(ParametricRootCoverError::Verification(
                RootBoxError::InvalidPrecision
            ))
        ));
    }
    for unsupported in [
        Expr::Function("gamma".into(), vec![Expr::Sym(unknowns[0].clone())]),
        Expr::Sym(Symbol::new("unbound")),
    ] {
        assert!(matches!(
            propose_parametric_root_cover(
                &[unsupported],
                &unknowns,
                &parameters,
                &[interval(1, 2)],
                &[interval(0, 2)],
                options(),
                &mut Unbounded,
            ),
            Err(ParametricRootCoverError::Verification(
                RootBoxError::Unsupported(_)
            ))
        ));
    }
}

#[test]
fn generation_and_verification_consume_one_ledger_without_per_leaf_scratch_resets() {
    let unknowns = [Symbol::new("x")];
    let parameters = [Symbol::new("p")];
    let source = [regular_source(&unknowns[0], &parameters[0])];
    let unknown_domain = [interval(0, 2)];
    let parameter_domain = [interval(0, 2)];
    let limit = 100 * 1024 * 1024;
    let mut budget = Budget::new(BudgetLimits::uniform(limit, 0));
    let proposed = propose_parametric_root_cover(
        &source,
        &unknowns,
        &parameters,
        &unknown_domain,
        &parameter_domain,
        options(),
        &mut budget,
    )
    .unwrap();
    assert!(proposed.leaves.len() > 1);
    let remaining = budget.remaining(Dimension::MemoryBytes);
    assert!(remaining < limit - 64 * 1024 * 1024);
    assert!(matches!(
        verify_parametric_root_cover(
            &source,
            &unknowns,
            &parameters,
            &unknown_domain,
            &parameter_domain,
            &proposed,
            &mut budget,
        ),
        Err(ParametricRootCoverError::Verification(RootBoxError::Meter(
            MeterError::Budget(_)
        )))
    ));
    assert_eq!(budget.remaining(Dimension::MemoryBytes), remaining);
}

#[test]
fn unsuccessful_search_retains_work_and_cannot_reset_the_callers_budget() {
    let unknowns = [Symbol::new("x")];
    let parameters = [Symbol::new("p")];
    let source = [regular_source(&unknowns[0], &parameters[0])];
    let mut budget = Budget::new(BudgetLimits::uniform(100 * 1024 * 1024, 0));
    assert!(matches!(
        propose_parametric_root_cover(
            &source,
            &unknowns,
            &parameters,
            &[interval(0, 2)],
            &[interval(0, 2)],
            ParametricRootCoverOptions {
                max_leaves: 1,
                ..options()
            },
            &mut budget,
        ),
        Err(ParametricRootCoverError::Incomplete(_))
    ));
    assert!(budget.remaining(Dimension::ComputeSteps) < 100 * 1024 * 1024);
    assert!(matches!(
        propose_parametric_root_cover(
            &source,
            &unknowns,
            &parameters,
            &[interval(0, 2)],
            &[interval(0, 2)],
            options(),
            &mut budget,
        ),
        Err(ParametricRootCoverError::Verification(RootBoxError::Meter(
            MeterError::Budget(_)
        )))
    ));
}

#[derive(Default)]
struct CheckpointMeter {
    checks: usize,
    cancel_at: Option<usize>,
}

impl BudgetMeter for CheckpointMeter {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checks += 1;
        if self.cancel_at == Some(self.checks) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_at_entry_during_refinement_and_before_publication_returns_no_witness() {
    let unknowns = [Symbol::new("x")];
    let parameters = [Symbol::new("p")];
    let source = [regular_source(&unknowns[0], &parameters[0])];
    let propose = |meter: &mut CheckpointMeter| {
        propose_parametric_root_cover(
            &source,
            &unknowns,
            &parameters,
            &[interval(0, 2)],
            &[interval(0, 2)],
            options(),
            meter,
        )
    };
    let mut completed = CheckpointMeter::default();
    assert!(propose(&mut completed).is_ok());
    for cancel_at in [1, completed.checks / 2, completed.checks] {
        let mut meter = CheckpointMeter {
            cancel_at: Some(cancel_at),
            ..Default::default()
        };
        assert_eq!(
            propose(&mut meter),
            Err(ParametricRootCoverError::Verification(RootBoxError::Meter(
                MeterError::Cancelled
            )))
        );
        assert_eq!(meter.checks, cancel_at);
    }
}
