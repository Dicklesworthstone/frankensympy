//! Independent fundamental matrices, IVPs, mutations and resource boundaries.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_solvers::ode::system::{
    LinearOdeSystemCertificate, LinearOdeSystemError, RationalMatrix2, linear_ode_fundamental,
    linear_ode_system, metered_linear_ode_system, verify_linear_ode_fundamental,
};
use std::collections::HashMap;
use std::sync::Arc;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}
fn matrix(values: [[i64; 2]; 2]) -> RationalMatrix2 {
    values.map(|row| row.map(|value| q(value, 1)))
}
fn e(source: &str) -> Expr {
    parse(source).unwrap()
}
fn at(expression: &Expr, time: &Symbol, value: BigRational) -> Expr {
    fsym_simplify::simplify(&expression.subs(&HashMap::from([(
        time.clone(),
        fsym_core::elementary::rational_expr(value),
    )])))
}
fn rotation() -> LinearOdeSystemCertificate {
    LinearOdeSystemCertificate {
        fundamental: [[e("cos(t)"), e("sin(t)")], [e("-sin(t)"), e("cos(t)")]],
    }
}

#[test]
fn independent_rotation_and_jordan_matrices_are_accepted() {
    let t = Symbol::new("t");
    let source = matrix([[0, 1], [-1, 0]]);
    let checked =
        verify_linear_ode_fundamental(&source, &t, &q(0, 1), rotation(), &mut Unbounded).unwrap();
    assert_eq!(checked.coefficients(), &source);
    assert_eq!(checked.initial_point(), &q(0, 1));
    assert_eq!(checked.time(), &t);
    let jordan = LinearOdeSystemCertificate {
        fundamental: [[e("exp(t)"), e("t*exp(t)")], [e("0"), e("exp(t)")]],
    };
    verify_linear_ode_fundamental(
        &matrix([[1, 1], [0, 1]]),
        &t,
        &q(0, 1),
        jordan,
        &mut Unbounded,
    )
    .unwrap();
    let nilpotent = LinearOdeSystemCertificate {
        fundamental: [[e("1+t"), e("t")], [e("-t"), e("1-t")]],
    };
    verify_linear_ode_fundamental(
        &matrix([[1, 1], [-1, -1]]),
        &t,
        &q(0, 1),
        nilpotent,
        &mut Unbounded,
    )
    .unwrap();
}

#[test]
fn all_discriminant_regimes_and_nondiagonal_rational_matrices_have_complete_families() {
    let t = Symbol::new("t");
    let c1 = Expr::Sym(Symbol::new("C1"));
    let c2 = Expr::Sym(Symbol::new("C2"));
    let mut matrices = vec![
        matrix([[0, 0], [0, 0]]),
        matrix([[0, 1], [-1, 0]]),
        matrix([[0, 1], [2, 0]]),
        matrix([[1, 1], [0, 1]]),
        matrix([[1, 1], [-1, -1]]),
        matrix([[2, 0], [0, -3]]),
        matrix([[2, 3], [-4, 1]]),
    ];
    matrices.push([[q(1, 2), q(2, 3)], [q(-3, 5), q(-7, 4)]]);
    for source in matrices {
        let result = linear_ode_system(&source, &t, &q(0, 1), [&c1, &c2])
            .unwrap_or_else(|error| panic!("{source:?}: {error}"));
        for (row, expected) in [&c1, &c2].into_iter().enumerate() {
            assert_eq!(at(&result.solutions()[row], &t, q(0, 1)), *expected);
        }
        assert_eq!(result.fundamental().coefficients(), &source);
    }
}

#[test]
fn exact_initial_values_at_zero_and_nonzero_times_are_retained() {
    let t = Symbol::new("t");
    let source = matrix([[0, 1], [-1, 0]]);
    let initial = [Expr::from_i64(1), Expr::from_i64(2)];
    let result = linear_ode_system(&source, &t, &q(0, 1), [&initial[0], &initial[1]]).unwrap();
    assert_eq!(
        fsym_simplify::simplify(&result.solutions()[0]),
        fsym_simplify::simplify(&e("cos(t)+2*sin(t)"))
    );
    assert_eq!(
        fsym_simplify::simplify(&result.solutions()[1]),
        fsym_simplify::simplify(&e("2*cos(t)-sin(t)"))
    );
    for point in [q(2, 1), q(-3, 2), q(1, 3)] {
        let result = linear_ode_system(&source, &t, &point, [&initial[0], &initial[1]]).unwrap();
        assert_eq!(result.fundamental().initial_point(), &point);
        assert_eq!(result.initial_values(), &initial);
        for (row, expected) in initial.iter().enumerate() {
            assert_eq!(at(&result.solutions()[row], &t, point.clone()), *expected);
        }
    }
}

#[test]
fn initial_normalization_rejects_dependent_or_rescaled_solution_columns() {
    let t = Symbol::new("t");
    let source = matrix([[0, 1], [-1, 0]]);
    let good = rotation();
    let mut swapped = good.clone();
    for row in &mut swapped.fundamental {
        row.swap(0, 1);
    }
    let mut dependent = good.clone();
    for row in &mut dependent.fundamental {
        row[1] = row[0].clone();
    }
    let mut scaled = good.clone();
    for row in &mut scaled.fundamental {
        row[1] = Expr::from_i64(2) * row[1].clone();
    }
    for candidate in [swapped, dependent, scaled] {
        assert!(matches!(
            verify_linear_ode_fundamental(&source, &t, &q(0, 1), candidate, &mut Unbounded),
            Err(LinearOdeSystemError::Verification(
                "fundamental initial identity"
            ))
        ));
    }
}

#[test]
fn coefficient_time_point_and_tiny_candidate_mutations_cannot_rebind_the_claim() {
    let t = Symbol::new("t");
    let source = matrix([[0, 1], [-1, 0]]);
    let mut changed = rotation();
    changed.fundamental[0][0] = Expr::Add(vec![
        changed.fundamental[0][0].clone(),
        Expr::Rational(BigRational::new(1.into(), BigInt::from(10).pow(80))),
    ]);
    assert!(verify_linear_ode_fundamental(&source, &t, &q(0, 1), changed, &mut Unbounded).is_err());
    assert!(
        verify_linear_ode_fundamental(
            &matrix([[0, 1], [-2, 0]]),
            &t,
            &q(0, 1),
            rotation(),
            &mut Unbounded
        )
        .is_err()
    );
    assert!(
        verify_linear_ode_fundamental(&source, &t, &q(1, 1), rotation(), &mut Unbounded).is_err()
    );
    let rebound = Symbol::with_identity("t", SymbolIdentity::from_assumptions_digest([7; 32]));
    assert!(
        verify_linear_ode_fundamental(&source, &rebound, &q(0, 1), rotation(), &mut Unbounded)
            .is_err()
    );
}

#[test]
fn original_undefined_children_and_opaque_functions_refuse_before_cancellation() {
    let t = Symbol::new("t");
    for invalid in [
        Expr::Pow(Arc::new(Expr::from_i64(0)), Arc::new(Expr::from_i64(-1))),
        Expr::Function("opaque".into(), vec![Expr::Sym(t.clone())]),
        Expr::Sym(Symbol::new("foreign")),
    ] {
        let mut candidate = rotation();
        candidate.fundamental[0][0] = Expr::Add(vec![
            candidate.fundamental[0][0].clone(),
            Expr::Mul(vec![Expr::from_i64(0), invalid]),
        ]);
        assert!(
            verify_linear_ode_fundamental(
                &matrix([[0, 1], [-1, 0]]),
                &t,
                &q(0, 1),
                candidate,
                &mut Unbounded
            )
            .is_err()
        );
    }
}

#[test]
fn typed_time_and_constants_are_preserved_without_ambient_assumption_capture() {
    let t = Symbol::with_identity("C1", SymbolIdentity::from_assumptions_digest([9; 32]));
    let c1 = Expr::Sym(Symbol::new("C2"));
    let c2 = Expr::Sym(Symbol::new("C3"));
    let result = fsym_core::assume::with_facts(
        HashMap::from([
            (
                "C1".into(),
                fsym_core::assume::Facts::from_names(&["integer"]),
            ),
            (
                "C2".into(),
                fsym_core::assume::Facts::from_names(&["negative"]),
            ),
        ]),
        || linear_ode_system(&matrix([[0, 1], [-1, 0]]), &t, &q(0, 1), [&c1, &c2]),
    )
    .unwrap();
    assert!(
        result
            .solutions()
            .iter()
            .all(|value| value.free_symbols().contains(&t))
    );
    assert_eq!(result.initial_values(), &[c1.clone(), c2]);
    assert!(linear_ode_system(&matrix([[0, 1], [-1, 0]]), &t, &q(0, 1), [&c1, &c1]).is_err());
    assert!(
        linear_ode_system(
            &matrix([[0, 1], [-1, 0]]),
            &t,
            &q(0, 1),
            [&Expr::Sym(t.clone()), &Expr::from_i64(1)]
        )
        .is_err()
    );
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    cancel_at: Option<usize>,
    charges: [u64; 3],
}
impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        let index = match dimension {
            Dimension::ComputeSteps => Some(0),
            Dimension::MemoryBytes => Some(1),
            Dimension::AllocationCount => Some(2),
            _ => None,
        };
        if let Some(index) = index {
            self.charges[index] += amount;
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
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn separate_cumulative_budgets_and_final_cancellation_are_terminal() {
    let t = Symbol::new("t");
    let source = matrix([[0, 1], [-1, 0]]);
    let values = [Expr::from_i64(1), Expr::from_i64(2)];
    let (mut g, mut v) = (Meter::default(), Meter::default());
    metered_linear_ode_system(
        &source,
        &t,
        &q(0, 1),
        [&values[0], &values[1]],
        &mut g,
        &mut v,
    )
    .unwrap();
    for (generator, measured) in [(true, &g), (false, &v)] {
        assert!(measured.charges.iter().all(|&value| value > 0));
        for stop in [
            1,
            measured.checkpoints / 2,
            measured.checkpoints - 1,
            measured.checkpoints,
        ] {
            let mut meter = Meter {
                cancel_at: Some(stop),
                ..Default::default()
            };
            let result = if generator {
                metered_linear_ode_system(
                    &source,
                    &t,
                    &q(0, 1),
                    [&values[0], &values[1]],
                    &mut meter,
                    &mut Unbounded,
                )
            } else {
                metered_linear_ode_system(
                    &source,
                    &t,
                    &q(0, 1),
                    [&values[0], &values[1]],
                    &mut Unbounded,
                    &mut meter,
                )
            };
            assert!(matches!(
                result,
                Err(LinearOdeSystemError::Meter(MeterError::Cancelled))
            ));
        }
        for (index, dimension) in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ]
        .into_iter()
        .enumerate()
        {
            let mut limits = BudgetLimits::uniform(u64::MAX, 0);
            limits.dimensions[dimension.index()] = measured.charges[index] - 1;
            let mut meter = Budget::new(limits);
            let result = if generator {
                metered_linear_ode_system(
                    &source,
                    &t,
                    &q(0, 1),
                    [&values[0], &values[1]],
                    &mut meter,
                    &mut Unbounded,
                )
            } else {
                metered_linear_ode_system(
                    &source,
                    &t,
                    &q(0, 1),
                    [&values[0], &values[1]],
                    &mut Unbounded,
                    &mut meter,
                )
            };
            assert!(matches!(
                result,
                Err(LinearOdeSystemError::Meter(MeterError::Budget(_)))
            ));
        }
    }
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    let mut unused = Meter::default();
    assert!(
        metered_linear_ode_system(
            &source,
            &t,
            &q(0, 1),
            [&values[0], &values[1]],
            &mut tiny,
            &mut unused
        )
        .is_err()
    );
    assert_eq!(unused.charges, [0; 3]);
    assert_eq!(unused.checkpoints, 0);
}

#[test]
fn coefficient_and_candidate_bounds_apply_even_with_unbounded_outer_meters() {
    let t = Symbol::new("t");
    let mut source = matrix([[0, 1], [-1, 0]]);
    source[0][0] = BigRational::from_integer(BigInt::from(1) << 1024u32);
    assert!(matches!(
        linear_ode_fundamental(&source, &t, &q(0, 1)),
        Err(LinearOdeSystemError::ResourceLimit(_))
    ));
    let mut candidate = rotation();
    candidate.fundamental[0][0] = Expr::Pow(
        Arc::new(e("1+t")),
        Arc::new(Expr::Integer(BigInt::from(1) << 2048u32)),
    );
    assert!(
        verify_linear_ode_fundamental(
            &matrix([[0, 1], [-1, 0]]),
            &t,
            &q(0, 1),
            candidate,
            &mut Unbounded
        )
        .is_err()
    );
}
