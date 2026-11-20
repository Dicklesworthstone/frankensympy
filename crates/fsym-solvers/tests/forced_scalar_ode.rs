//! Scalar forcing must be checked against the original equation and domains.

use fsym_budget::{
    Budget, BudgetError, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded,
};
use fsym_calculus::diff;
use fsym_core::{BigInt, Constant, Expr, Symbol, SymbolIdentity, parse};
use fsym_simplify::{try_expand, try_simplify};
use fsym_solvers::ode::forcing::{metered_second_order, second_order};
use fsym_solvers::ode::system::general::GeneralOdeSystemError;
use fsym_solvers::{
    dsolve_const_coeff_second_order_nonhomogeneous,
    verify_const_coeff_second_order_nonhomogeneous_solution,
};
use std::collections::HashMap;
use std::sync::Arc;

fn e(source: &str) -> Expr {
    parse(source).unwrap()
}

fn equal(actual: &Expr, expected: &Expr) {
    let residual = Expr::Add(vec![
        actual.clone(),
        Expr::Mul(vec![Expr::from_i64(-1), expected.clone()]),
    ]);
    let residual = try_simplify(&try_expand(&residual).unwrap()).unwrap();
    assert!(residual.is_zero(), "{actual} != {expected}: {residual}");
}

fn check_initial_values(solution: &Expr, x: &Symbol, c1: &Symbol, c2: &Symbol) {
    let at_zero = HashMap::from([(x.clone(), Expr::from_i64(0))]);
    equal(&solution.subs(&at_zero), &Expr::Sym(c1.clone()));
    equal(&diff(solution, x).subs(&at_zero), &Expr::Sym(c2.clone()));
}

#[test]
fn polynomial_forcing_reaches_the_existing_scalar_api() {
    let (x, c1, c2) = (Symbol::new("x"), Symbol::new("C1"), Symbol::new("C2"));
    for (source, expected) in [
        ("x", "C1+C2*x+x**3/6"),
        ("x**2+2*x+3", "C1+C2*x+x**4/12+x**3/3+3*x**2/2"),
        ("(x+1)*(x-1)", "C1+C2*x+x**4/12-x**2/2"),
    ] {
        let source = e(source);
        let result =
            dsolve_const_coeff_second_order_nonhomogeneous(1, 0, 0, &source, &x, &c1, &c2).unwrap();
        equal(&result, &e(expected));
        check_initial_values(&result, &x, &c1, &c2);
        assert!(verify_const_coeff_second_order_nonhomogeneous_solution(
            &result, 1, 0, 0, &source, &x,
        ));
    }
}

#[test]
fn resonant_exponential_polynomial_retains_both_constants() {
    let (x, c1, c2) = (Symbol::new("x"), Symbol::new("C1"), Symbol::new("C2"));
    let source = e("x*exp(x)");
    let result = second_order([1, -2, 1], &source, &x, [&c1, &c2]).unwrap();
    equal(&result, &e("exp(x)*(C1+(C2-C1)*x+x**3/6)"));
    check_initial_values(&result, &x, &c1, &c2);
    assert!(verify_const_coeff_second_order_nonhomogeneous_solution(
        &result, 1, -2, 1, &source, &x,
    ));
}

#[test]
fn resonant_oscillatory_polynomial_has_the_exact_particular() {
    let (x, c1, c2) = (Symbol::new("x"), Symbol::new("C1"), Symbol::new("C2"));
    let source = e("x*cos(x)");
    let result = second_order([1, 0, 1], &source, &x, [&c1, &c2]).unwrap();
    equal(
        &result,
        &e("C1*cos(x)+C2*sin(x)+(x**2*sin(x)+x*cos(x)-sin(x))/4"),
    );
    check_initial_values(&result, &x, &c1, &c2);
}

#[test]
fn combined_modes_and_negative_frequencies_bind_the_original_source() {
    let (x, c1, c2) = (Symbol::new("x"), Symbol::new("C1"), Symbol::new("C2"));
    let mut refused = Vec::new();
    for (operator, source) in [
        ([1, 0, -1], "(x+1)*(x-1)*exp(x)"),
        ([2, 1, -1], "x*exp(x)+sin(2*x)"),
        ([1, 0, 1], "x*sin(-x)"),
        ([1, 0, 1], "x*cos(-x)"),
        ([-2, 2, 0], "x**2"),
    ] {
        let source = e(source);
        let result = match second_order(operator, &source, &x, [&c1, &c2]) {
            Ok(result) => result,
            Err(error) => {
                refused.push(format!("operator {operator:?}, source {source}: {error}"));
                continue;
            }
        };
        assert!(verify_const_coeff_second_order_nonhomogeneous_solution(
            &result,
            operator[0],
            operator[1],
            operator[2],
            &source,
            &x,
        ));
        check_initial_values(&result, &x, &c1, &c2);
        assert!(!verify_const_coeff_second_order_nonhomogeneous_solution(
            &result,
            operator[0],
            operator[1],
            operator[2],
            &Expr::Add(vec![source, Expr::from_i64(1)]),
            &x,
        ));
    }
    assert!(
        refused.is_empty(),
        "refused mixed cases:\n{}",
        refused.join("\n")
    );
}

#[test]
fn resonant_damped_oscillatory_forcing_retains_the_actual_budget_refusal() {
    let (x, c1, c2) = (Symbol::new("x"), Symbol::new("C1"), Symbol::new("C2"));
    // This original input fits the grammar and six-state realization, but
    // the current independently checked system engine exhausts its cumulative
    // compute allowance. Keep the difficult case visible; do not reset its
    // accounting, increase the cap or return a partial solution.
    assert!(matches!(
        second_order([1, -4, 13], &e("x*exp(2*x)*sin(3*x)"), &x, [&c1, &c2]),
        Err(GeneralOdeSystemError::Meter(MeterError::Budget(
            BudgetError::Exhausted {
                dimension: Dimension::ComputeSteps,
                ..
            }
        ))),
    ));
}

#[test]
fn scalar_verifier_accepts_independent_witnesses_and_rejects_mutations() {
    let x = Symbol::new("x");
    // These witnesses are supplied independently of both source decomposition
    // and the system/forcing generators.
    for (operator, forcing, witness) in [
        ([1, 0, 0], "x", "x**3/6"),
        ([1, -2, 1], "x*exp(x)", "x**3*exp(x)/6"),
        ([1, 0, 1], "x*cos(x)", "(x**2*sin(x)+x*cos(x))/4"),
        ([1, 0, 1], "x*sin(-x)", "(x**2*cos(x)-x*sin(x))/4"),
        ([1, 0, 1], "x*cos(-x)", "(x**2*sin(x)+x*cos(x))/4"),
    ] {
        let source = e(forcing);
        let witness = e(witness);
        assert!(verify_const_coeff_second_order_nonhomogeneous_solution(
            &witness,
            operator[0],
            operator[1],
            operator[2],
            &source,
            &x,
        ));
        let mutated = Expr::Mul(vec![Expr::from_i64(2), witness]);
        assert!(!verify_const_coeff_second_order_nonhomogeneous_solution(
            &mutated,
            operator[0],
            operator[1],
            operator[2],
            &source,
            &x,
        ));
        assert!(!verify_const_coeff_second_order_nonhomogeneous_solution(
            &e("0"),
            operator[0],
            operator[1],
            operator[2],
            &source,
            &x,
        ));
    }
}

#[test]
fn original_undefined_and_foreign_sources_cannot_disappear() {
    let (x, c1, c2) = (Symbol::new("x"), Symbol::new("C1"), Symbol::new("C2"));
    let rebound = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([37; 32]));
    for bad in [
        Expr::Const(Constant::NaN),
        Expr::Const(Constant::Infinity),
        Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(-1))),
        Expr::Function("opaque".into(), vec![Expr::Sym(x.clone())]),
        Expr::Sym(rebound),
    ] {
        for source in [
            bad.clone(),
            Expr::Mul(vec![Expr::from_i64(0), bad.clone()]),
            Expr::Add(vec![
                bad.clone(),
                Expr::Mul(vec![Expr::from_i64(-1), bad.clone()]),
            ]),
        ] {
            assert!(second_order([1, 0, 1], &source, &x, [&c1, &c2]).is_err());
        }
    }
    assert!(second_order([0, 1, 1], &e("x"), &x, [&c1, &c2]).is_err());
    assert!(second_order([1, 0, 1], &e("x"), &x, [&c1, &c1]).is_err());
    assert!(second_order([1, 0, 1], &e("x+C1"), &x, [&c1, &c2]).is_err());
}

#[test]
fn unsupported_and_oversized_forcing_remain_distinct_failures() {
    let (x, c1, c2) = (Symbol::new("x"), Symbol::new("C1"), Symbol::new("C2"));
    for source in [
        "x**17",
        "x**3*sin(x)",
        "x**6*exp(x)",
        "sin(x)*cos(x)",
        "exp(x**2)",
        "exp(x+1)",
        "sqrt(x)",
    ] {
        assert!(
            second_order([1, 0, 1], &e(source), &x, [&c1, &c2]).is_err(),
            "{source}"
        );
    }
    let oversized = Expr::Integer(BigInt::one() << 1024u32);
    assert!(matches!(
        second_order([1, 0, 1], &oversized, &x, [&c1, &c2]),
        Err(GeneralOdeSystemError::ResourceLimit(_)),
    ));
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    cancel_at: Option<usize>,
    stopped: bool,
    totals: [u64; 3],
}

impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        self.charge_batch(&[(dimension, amount)])
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        assert!(!self.stopped, "charged after cancellation");
        for &(dimension, amount) in charges {
            let index = match dimension {
                Dimension::ComputeSteps => Some(0),
                Dimension::MemoryBytes => Some(1),
                Dimension::AllocationCount => Some(2),
                _ => None,
            };
            if let Some(index) = index {
                self.totals[index] += amount;
            }
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped, "checked after cancellation");
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn separate_phase_budgets_and_cancellation_reach_final_source_verification() {
    let (x, c1, c2) = (Symbol::new("x"), Symbol::new("C1"), Symbol::new("C2"));
    let source = e("x");
    let (mut generation, mut verification) = (Meter::default(), Meter::default());
    metered_second_order(
        [1, 0, 0],
        &source,
        &x,
        [&c1, &c2],
        &mut generation,
        &mut verification,
    )
    .unwrap();
    for (generating, measured) in [(true, generation), (false, verification)] {
        for stop in [1, measured.checkpoints / 2, measured.checkpoints] {
            let mut meter = Meter {
                cancel_at: Some(stop),
                ..Meter::default()
            };
            let result = if generating {
                metered_second_order(
                    [1, 0, 0],
                    &source,
                    &x,
                    [&c1, &c2],
                    &mut meter,
                    &mut Unbounded,
                )
            } else {
                metered_second_order(
                    [1, 0, 0],
                    &source,
                    &x,
                    [&c1, &c2],
                    &mut Unbounded,
                    &mut meter,
                )
            };
            assert!(matches!(
                result,
                Err(GeneralOdeSystemError::Meter(MeterError::Cancelled))
            ));
            assert_eq!(meter.checkpoints, stop);
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
            limits.dimensions[dimension.index()] = measured.totals[index] - 1;
            let mut meter = Budget::new(limits);
            let result = if generating {
                metered_second_order(
                    [1, 0, 0],
                    &source,
                    &x,
                    [&c1, &c2],
                    &mut meter,
                    &mut Unbounded,
                )
            } else {
                metered_second_order(
                    [1, 0, 0],
                    &source,
                    &x,
                    [&c1, &c2],
                    &mut Unbounded,
                    &mut meter,
                )
            };
            assert!(matches!(
                result,
                Err(GeneralOdeSystemError::Meter(MeterError::Budget(_)))
            ));
        }
    }
}
