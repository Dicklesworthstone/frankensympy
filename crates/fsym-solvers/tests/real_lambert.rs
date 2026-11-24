use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigRational, Expr, RealLambertError, Symbol, SymbolIdentity};
use fsym_solvers::lambert::{metered_solve_real_lambert, propose_real_lambert, solve_real_lambert};
use std::{cell::Cell, rc::Rc, sync::Arc};

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}
fn exp(e: Expr) -> Expr {
    Expr::Function("exp".into(), vec![e])
}
fn add(values: Vec<Expr>) -> Expr {
    Expr::Add(values)
}
fn mul(values: Vec<Expr>) -> Expr {
    Expr::Mul(values)
}
fn power(e: Expr, n: i64) -> Expr {
    Expr::Pow(Arc::new(e), Arc::new(Expr::from_i64(n)))
}
fn branch_count(roots: &[Expr], branch: i64) -> usize {
    fn contains(e: &Expr, branch: i64) -> bool {
        match e {
            Expr::Function(name, args) if name == "LambertW" => {
                matches!(args.as_slice(),[_] if branch==0)
                    || matches!(args.as_slice(),[_,Expr::Integer(k)] if k.to_i64()==Some(branch))
            }
            Expr::Add(v) | Expr::Mul(v) | Expr::Function(_, v) => {
                v.iter().any(|child| contains(child, branch))
            }
            Expr::Pow(a, b) => contains(a, branch) || contains(b, branch),
            _ => false,
        }
    }
    roots.iter().filter(|root| contains(root, branch)).count()
}

#[test]
fn common_mixed_transcendental_equations_have_complete_real_solutions() {
    let x = Symbol::new("lambert_x");
    let sx = Expr::Sym(x.clone());
    let exponential = exp(sx.clone());
    let cases = [
        (
            add(vec![
                mul(vec![sx.clone(), exponential.clone()]),
                Expr::from_i64(-1),
            ]),
            1,
        ),
        (
            add(vec![
                mul(vec![sx.clone(), exponential.clone()]),
                Expr::rational(1, 4).unwrap(),
            ]),
            2,
        ),
        (
            add(vec![
                exponential.clone(),
                mul(vec![Expr::from_i64(-3), sx.clone()]),
            ]),
            2,
        ),
        (add(vec![sx.clone(), exponential.clone()]), 1),
        (
            add(vec![
                mul(vec![sx.clone(), exponential.clone()]),
                Expr::from_i64(1),
            ]),
            0,
        ),
        (
            add(vec![sx.clone(), mul(vec![Expr::from_i64(-1), exponential])]),
            0,
        ),
    ];
    for (source, count) in cases {
        let result = solve_real_lambert(&source, &x).unwrap().unwrap();
        assert_eq!(result.expression(), &source);
        assert_eq!(result.variable(), &x);
        assert_eq!(result.roots().len(), count);
        if count == 2 {
            assert_eq!(branch_count(result.roots(), -1), 1);
            assert_eq!(branch_count(result.roots(), 0), 1);
        }
    }
}

#[test]
fn translated_rescaled_equations_expose_exact_normal_form_coefficients() {
    let x = Symbol::new("translated_x");
    let sx = Expr::Sym(x.clone());
    let source = add(vec![
        mul(vec![
            add(vec![
                mul(vec![Expr::from_i64(2), sx.clone()]),
                Expr::from_i64(3),
            ]),
            exp(add(vec![
                mul(vec![Expr::from_i64(3), sx]),
                Expr::from_i64(-2),
            ])),
        ]),
        Expr::from_i64(-5),
    ]);
    let candidate = propose_real_lambert(&source, &x, &mut Unbounded)
        .unwrap()
        .unwrap();
    assert_eq!(candidate.linear_slope, q(0, 1));
    assert_eq!(candidate.linear_offset, q(-5, 1));
    assert_eq!(candidate.exponential_slope, q(2, 1));
    assert_eq!(candidate.exponential_offset, q(3, 1));
    assert_eq!(candidate.argument_slope, q(3, 1));
    assert_eq!(candidate.argument_offset, q(-2, 1));
    let accepted = solve_real_lambert(&source, &x).unwrap().unwrap();
    assert_eq!(accepted.normal_form(), &candidate);
    assert_eq!(accepted.roots().len(), 1);
}

#[test]
fn tangencies_and_vanishing_rhs_return_one_exact_finite_root() {
    let x = Symbol::new("tangent_x");
    let sx = Expr::Sym(x.clone());
    let cases = [
        (
            add(vec![
                mul(vec![
                    sx.clone(),
                    exp(add(vec![sx.clone(), Expr::from_i64(1)])),
                ]),
                Expr::from_i64(1),
            ]),
            -1,
        ),
        (
            add(vec![
                sx.clone(),
                Expr::from_i64(1),
                mul(vec![Expr::from_i64(-1), exp(sx.clone())]),
            ]),
            0,
        ),
        (
            mul(vec![
                add(vec![sx.clone(), Expr::from_i64(2)]),
                exp(sx.clone()),
            ]),
            -2,
        ),
        (
            add(vec![
                mul(vec![Expr::from_i64(0), exp(sx.clone())]),
                sx.clone(),
                Expr::from_i64(2),
            ]),
            -2,
        ),
        (
            add(vec![
                mul(vec![sx.clone(), exp(add(vec![sx, Expr::from_i64(-1)]))]),
                Expr::from_i64(-1),
            ]),
            1,
        ),
    ];
    for (source, root) in cases {
        let result = solve_real_lambert(&source, &x).unwrap().unwrap();
        assert_eq!(result.roots(), &[Expr::from_i64(root)]);
    }
}

#[test]
fn exact_sparse_expansion_and_affine_argument_normalization_compose() {
    let x = Symbol::new("polynomial_form_x");
    let sx = Expr::Sym(x.clone());
    // Two differently spelled affine arguments agree exactly. The quadratic
    // source terms cancel only in the proposed formal coefficient algebra.
    let e1 = exp(add(vec![sx.clone(), sx.clone(), Expr::from_i64(1)]));
    let e2 = exp(add(vec![
        mul(vec![Expr::from_i64(2), sx.clone()]),
        Expr::from_i64(1),
    ]));
    let source = add(vec![
        mul(vec![sx.clone(), e1]),
        mul(vec![Expr::from_i64(2), e2]),
        power(sx.clone(), 2),
        mul(vec![Expr::from_i64(-1), power(sx, 2)]),
        Expr::from_i64(-3),
    ]);
    let result = solve_real_lambert(&source, &x).unwrap().unwrap();
    assert_eq!(result.normal_form().exponential_slope, q(1, 1));
    assert_eq!(result.normal_form().exponential_offset, q(2, 1));
    assert_eq!(result.normal_form().argument_slope, q(2, 1));
    assert_eq!(result.roots().len(), 1);
}

#[test]
fn reciprocal_rational_constants_are_exact_and_original_dynamic_poles_refuse() {
    let x = Symbol::new("rational_constant_x");
    let sx = Expr::Sym(x.clone());
    let base = mul(vec![sx.clone(), exp(sx.clone())]);
    let source = add(vec![base.clone(), power(Expr::from_i64(4), -1)]);
    assert_eq!(
        solve_real_lambert(&source, &x)
            .unwrap()
            .unwrap()
            .roots()
            .len(),
        2
    );
    let pole = power(sx, -1);
    for hidden in [mul(vec![Expr::from_i64(0), pole.clone()]), power(pole, 0)] {
        assert!(
            solve_real_lambert(&add(vec![base.clone(), hidden]), &x)
                .unwrap()
                .is_none()
        );
    }
    let undefined = add(vec![
        base,
        mul(vec![Expr::from_i64(0), power(Expr::from_i64(0), -1)]),
    ]);
    assert!(matches!(
        solve_real_lambert(&undefined, &x),
        Err(RealLambertError::InvalidSource(_))
    ));
}

#[test]
fn nonmatching_equations_remain_available_to_existing_solver_lanes() {
    let x = Symbol::new("nonmatch_x");
    let sx = Expr::Sym(x.clone());
    let sources = [
        add(vec![sx.clone(), Expr::from_i64(-1)]),
        add(vec![exp(sx.clone()), Expr::from_i64(-2)]),
        add(vec![power(sx.clone(), 2), exp(sx.clone())]),
        add(vec![
            sx.clone(),
            mul(vec![sx.clone(), exp(sx.clone())]),
            Expr::from_i64(1),
        ]),
        add(vec![
            sx.clone(),
            exp(sx.clone()),
            exp(mul(vec![Expr::from_i64(2), sx.clone()])),
        ]),
        add(vec![sx.clone(), exp(exp(sx.clone()))]),
        add(vec![sx.clone(), exp(Expr::from_i64(1))]),
        add(vec![
            sx,
            Expr::Function("sin".into(), vec![Expr::from_i64(0)]),
        ]),
    ];
    for source in sources {
        assert!(solve_real_lambert(&source, &x).unwrap().is_none());
    }
}

#[test]
fn typed_coordinate_identity_survives_generator_and_verifier() {
    let plain = Symbol::new("same_name");
    let typed = Symbol::with_identity(
        "same_name",
        SymbolIdentity::from_assumptions_digest([3; 32]),
    );
    let expression = add(vec![
        mul(vec![
            Expr::Sym(typed.clone()),
            exp(Expr::Sym(typed.clone())),
        ]),
        Expr::from_i64(-1),
    ]);
    assert!(solve_real_lambert(&expression, &plain).unwrap().is_none());
    let accepted = solve_real_lambert(&expression, &typed).unwrap().unwrap();
    assert_eq!(accepted.variable(), &typed);
}

#[test]
fn generator_and_verifier_failures_keep_their_separate_cumulative_ledgers() {
    let x = Symbol::new("budget_lambert_x");
    let source = add(vec![
        mul(vec![Expr::Sym(x.clone()), exp(Expr::Sym(x.clone()))]),
        Expr::rational(1, 4).unwrap(),
    ]);
    let allowance = 512 * 1024 * 1024;
    let mut generation = Budget::new(BudgetLimits::uniform(allowance, 0));
    let mut verification = Budget::new(BudgetLimits::uniform(allowance, 0));
    let accepted = metered_solve_real_lambert(&source, &x, &mut generation, &mut verification)
        .unwrap()
        .unwrap();
    assert_eq!(accepted.roots().len(), 2);
    for meter in [&generation, &verification] {
        assert!(meter.remaining(Dimension::ComputeSteps) < allowance);
        assert!(meter.remaining(Dimension::MemoryBytes) < allowance);
    }
    let mut generation = Budget::new(BudgetLimits::uniform(1, 0));
    let before = verification.snapshot();
    assert!(metered_solve_real_lambert(&source, &x, &mut generation, &mut verification).is_err());
    assert_eq!(verification.snapshot(), before);
    let mut generation = Budget::new(BudgetLimits::uniform(allowance, 0));
    let mut verification = Budget::new(BudgetLimits::uniform(1, 0));
    assert!(metered_solve_real_lambert(&source, &x, &mut generation, &mut verification).is_err());
    assert!(generation.remaining(Dimension::ComputeSteps) < allowance);
}

struct SharedCancellation {
    flag: Rc<Cell<bool>>,
    trigger: bool,
}
impl BudgetMeter for SharedCancellation {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        if self.trigger {
            self.flag.set(true);
            Ok(())
        } else if self.flag.get() {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn cancellation_during_verification_cannot_publish_after_generator_cancelled() {
    let x = Symbol::new("publication_x");
    let source = add(vec![
        mul(vec![Expr::Sym(x.clone()), exp(Expr::Sym(x.clone()))]),
        Expr::from_i64(-1),
    ]);
    let flag = Rc::new(Cell::new(false));
    let mut generation = SharedCancellation {
        flag: flag.clone(),
        trigger: false,
    };
    let mut verification = SharedCancellation {
        flag,
        trigger: true,
    };
    assert!(matches!(
        metered_solve_real_lambert(&source, &x, &mut generation, &mut verification),
        Err(RealLambertError::Meter(MeterError::Cancelled))
    ));
}
