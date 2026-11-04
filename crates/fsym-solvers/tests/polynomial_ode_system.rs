//! Independent particular identities, resonant systems, IVPs and mutations.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_solvers::ode::system::polynomial::{
    PolynomialOdeParticularCertificate, PolynomialVector2, metered_polynomial_ode_system,
    polynomial_ode_system, verify_polynomial_ode_particular,
};
use fsym_solvers::ode::system::{LinearOdeSystemError, RationalMatrix2};
use std::collections::HashMap;

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}
fn matrix(entries: [[i64; 2]; 2]) -> RationalMatrix2 {
    entries.map(|row| row.map(|value| q(value, 1)))
}
fn e(source: &str) -> Expr {
    parse(source).unwrap()
}
fn simplified(expression: &Expr) -> Expr {
    fsym_simplify::try_simplify(&fsym_simplify::try_expand(expression).unwrap()).unwrap()
}
fn poly(coefficients: &[BigRational], time: &Symbol) -> Expr {
    coefficients
        .iter()
        .rev()
        .fold(Expr::from_i64(0), |value, coefficient| {
            value * Expr::Sym(time.clone()) + rational_expr(coefficient.clone())
        })
}
fn initial(expression: &Expr, time: &Symbol, point: &BigRational) -> Expr {
    simplified(&expression.subs(&HashMap::from([(
        time.clone(),
        rational_expr(point.clone()),
    )])))
}
fn assert_residuals(
    result: &[Expr; 2],
    coefficients: &RationalMatrix2,
    forcing: &PolynomialVector2,
    time: &Symbol,
) {
    for (row, value) in result.iter().enumerate() {
        let rhs = rational_expr(coefficients[row][0].clone()) * result[0].clone()
            + rational_expr(coefficients[row][1].clone()) * result[1].clone()
            + poly(&forcing[row], time);
        let derivative = fsym_calculus::diff(value, time);
        let raw_residual = derivative.clone() - rhs.clone();
        let residual = simplified(&raw_residual);
        assert!(
            residual.is_zero(),
            "matrix {coefficients:?}, row {row}: residual {residual}; solution {value}; other {}; derivative {derivative}; rhs {rhs}; raw residual {raw_residual}",
            result[1 - row]
        );
    }
}

#[test]
fn hand_supplied_polynomial_particular_is_checked_without_generation() {
    let coefficients = matrix([[0, 1], [-1, 0]]);
    let forcing = [vec![q(3, 1)], vec![q(3, 1), q(0, 1), q(1, 1)]];
    let proposed = [vec![q(1, 1), q(0, 1), q(1, 1)], vec![q(-3, 1), q(2, 1)]];
    let accepted = verify_polynomial_ode_particular(
        &coefficients,
        &forcing,
        PolynomialOdeParticularCertificate {
            particular: proposed.clone(),
        },
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(accepted.coefficients(), &coefficients);
    assert_eq!(accepted.forcing(), &forcing);
    assert_eq!(accepted.particular(), &proposed);
}

#[test]
fn independent_coefficient_source_and_matrix_mutations_refuse() {
    let coefficients = matrix([[0, 1], [-1, 0]]);
    let forcing = [vec![q(3, 1)], vec![q(3, 1), q(0, 1), q(1, 1)]];
    let proposed = [vec![q(1, 1), q(0, 1), q(1, 1)], vec![q(-3, 1), q(2, 1)]];
    for row in 0..2 {
        for degree in 0..proposed[row].len() {
            let mut changed = proposed.clone();
            changed[row][degree] += q(1, 1_000_000_007);
            assert!(matches!(
                verify_polynomial_ode_particular(
                    &coefficients,
                    &forcing,
                    PolynomialOdeParticularCertificate {
                        particular: changed
                    },
                    &mut Unbounded,
                ),
                Err(LinearOdeSystemError::Verification(
                    "polynomial particular coefficient identity"
                ))
            ));
        }
    }
    let mut changed_forcing = forcing.clone();
    changed_forcing[0][0] += q(1, 1);
    for (source, force) in [
        (coefficients.clone(), changed_forcing),
        (matrix([[0, 2], [-1, 0]]), forcing),
    ] {
        assert!(
            verify_polynomial_ode_particular(
                &source,
                &force,
                PolynomialOdeParticularCertificate {
                    particular: proposed.clone()
                },
                &mut Unbounded,
            )
            .is_err()
        );
    }
}

#[test]
fn explicit_forced_rotation_ivp_has_the_hand_derived_unique_solution() {
    let t = Symbol::new("t");
    let coefficients = matrix([[0, 1], [-1, 0]]);
    let forcing = [vec![q(3, 1)], vec![q(3, 1), q(0, 1), q(1, 1)]];
    let values = [Expr::from_i64(1), Expr::from_i64(2)];
    let result = polynomial_ode_system(
        &coefficients,
        &t,
        &q(0, 1),
        &forcing,
        [&values[0], &values[1]],
    )
    .unwrap();
    assert_residuals(result.solutions(), &coefficients, &forcing, &t);
    assert!(simplified(&(result.solutions()[0].clone() - e("t^2+1+5*sin(t)"))).is_zero());
    assert!(simplified(&(result.solutions()[1].clone() - e("2*t-3+5*cos(t)"))).is_zero());
    assert_eq!(result.homogeneous().initial_values(), &values);
}

#[test]
fn every_matrix_regime_has_complete_forced_families_and_shifted_ivps() {
    let t = Symbol::new("t");
    let constants = [Expr::Sym(Symbol::new("C1")), Expr::Sym(Symbol::new("C2"))];
    let forcing = [vec![q(2, 3), q(-3, 2), q(5, 7)], vec![q(-4, 5), q(7, 11)]];
    let mut sources = vec![
        matrix([[0, 0], [0, 0]]),
        matrix([[0, 1], [0, 0]]),
        matrix([[1, 1], [-1, -1]]),
        matrix([[1, 1], [0, 0]]),
        matrix([[2, 0], [0, -3]]),
        matrix([[1, 1], [0, 1]]),
        matrix([[0, 1], [-1, 0]]),
        matrix([[0, 1], [2, 0]]),
    ];
    sources.push([[q(1, 2), q(2, 3)], [q(-3, 5), q(-7, 4)]]);
    for source in sources {
        for point in [q(0, 1), q(-2, 3)] {
            let result = polynomial_ode_system(
                &source,
                &t,
                &point,
                &forcing,
                [&constants[0], &constants[1]],
            )
            .unwrap_or_else(|error| panic!("{source:?}, t0={point}: {error}"));
            assert_residuals(result.solutions(), &source, &forcing, &t);
            for (row, constant) in constants.iter().enumerate() {
                assert_eq!(initial(&result.solutions()[row], &t, &point), *constant);
            }
            let first = Expr::Rational(q(2, 5));
            let second = Expr::Rational(q(-3, 7));
            let fixed =
                polynomial_ode_system(&source, &t, &point, &forcing, [&first, &second]).unwrap();
            assert_residuals(fixed.solutions(), &source, &forcing, &t);
            assert_eq!(initial(&fixed.solutions()[0], &t, &point), first);
            assert_eq!(initial(&fixed.solutions()[1], &t, &point), second);
        }
    }
}

#[test]
fn nilpotent_resonance_retains_the_degree_eighteen_particular() {
    let t = Symbol::new("t");
    let mut highest = vec![q(0, 1); 17];
    highest[16] = q(1, 1);
    let forcing = [vec![q(0, 1)], highest];
    let c1 = Expr::Sym(Symbol::new("C1"));
    let c2 = Expr::Sym(Symbol::new("C2"));
    let result = polynomial_ode_system(
        &matrix([[0, 1], [0, 0]]),
        &t,
        &q(0, 1),
        &forcing,
        [&c1, &c2],
    )
    .unwrap();
    assert_eq!(result.particular().particular()[0].len(), 19);
    assert_eq!(result.particular().particular()[0][18], q(1, 306));
    assert_eq!(result.particular().particular()[1][17], q(1, 17));
    assert!(simplified(&(result.solutions()[0].clone() - e("C1+C2*t+t^18/306"))).is_zero());
    assert!(simplified(&(result.solutions()[1].clone() - e("C2+t^17/17"))).is_zero());
    assert_residuals(result.solutions(), &matrix([[0, 1], [0, 0]]), &forcing, &t);
}

#[test]
fn partial_data_and_typed_time_do_not_capture_constants_or_ambient_facts() {
    let t = Symbol::with_identity("C1", SymbolIdentity::from_assumptions_digest([56; 32]));
    let fixed = Expr::Rational(q(3, 5));
    let free = Expr::Sym(Symbol::new("C2"));
    let forcing = [vec![q(1, 1), q(2, 1)], vec![q(-1, 1)]];
    let point = q(2, 3);
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
        || {
            polynomial_ode_system(
                &matrix([[0, 1], [-1, 0]]),
                &t,
                &point,
                &forcing,
                [&fixed, &free],
            )
        },
    )
    .unwrap();
    assert_eq!(initial(&result.solutions()[0], &t, &point), fixed);
    assert_eq!(initial(&result.solutions()[1], &t, &point), free);
    for value in result.solutions() {
        assert!(value.free_symbols().contains(&t));
        assert!(!value.free_symbols().contains(&Symbol::new("C1")));
    }
}

#[test]
fn source_and_candidate_bounds_hold_with_unbounded_outer_meters() {
    let t = Symbol::new("t");
    let source = matrix([[0, 0], [0, 0]]);
    let initial = Expr::from_i64(0);
    for bad in [
        [vec![], vec![q(0, 1)]],
        [vec![q(1, 1); 18], vec![q(0, 1)]],
        [vec![q(1, 1), q(0, 1)], vec![q(0, 1)]],
        [
            vec![BigRational::from_integer(BigInt::from(1) << 1024u32)],
            vec![q(0, 1)],
        ],
    ] {
        assert!(polynomial_ode_system(&source, &t, &q(0, 1), &bad, [&initial, &initial]).is_err());
    }
    for bad in [
        vec![],
        vec![q(1, 1); 20],
        vec![BigRational::from_integer(BigInt::from(1) << 16_384u32)],
    ] {
        assert!(
            verify_polynomial_ode_particular(
                &source,
                &[vec![q(0, 1)], vec![q(0, 1)]],
                PolynomialOdeParticularCertificate {
                    particular: [bad, vec![q(0, 1)]]
                },
                &mut Unbounded,
            )
            .is_err()
        );
    }
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    cancel_at: Option<usize>,
    charges: [u64; 3],
}
impl BudgetMeter for Meter {
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.cancel_at == Some(self.checkpoints) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        if let Some(index) = match dimension {
            Dimension::ComputeSteps => Some(0),
            Dimension::MemoryBytes => Some(1),
            Dimension::AllocationCount => Some(2),
            _ => None,
        } {
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
}

#[test]
fn combined_phase_budgets_and_final_cancellation_are_terminal() {
    let t = Symbol::new("t");
    let source = matrix([[1, 1], [0, 0]]);
    let force = [vec![q(1, 1), q(2, 3)], vec![q(-1, 1), q(0, 1), q(4, 5)]];
    let first = Expr::from_i64(1);
    let second = Expr::Sym(Symbol::new("C2"));
    let mut generation = Meter::default();
    let mut verification = Meter::default();
    metered_polynomial_ode_system(
        &source,
        &t,
        &q(1, 3),
        &force,
        [&first, &second],
        &mut generation,
        &mut verification,
    )
    .unwrap();
    for (phase, total) in [
        (true, generation.checkpoints),
        (false, verification.checkpoints),
    ] {
        for stop in [1, total / 2, total - 1, total] {
            let mut stopped = Meter {
                cancel_at: Some(stop),
                ..Meter::default()
            };
            let error = if phase {
                metered_polynomial_ode_system(
                    &source,
                    &t,
                    &q(1, 3),
                    &force,
                    [&first, &second],
                    &mut stopped,
                    &mut Unbounded,
                )
            } else {
                metered_polynomial_ode_system(
                    &source,
                    &t,
                    &q(1, 3),
                    &force,
                    [&first, &second],
                    &mut Unbounded,
                    &mut stopped,
                )
            };
            assert!(matches!(
                error,
                Err(LinearOdeSystemError::Meter(MeterError::Cancelled))
            ));
        }
    }
    for (phase, totals) in [(true, generation.charges), (false, verification.charges)] {
        for (index, dimension) in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ]
        .into_iter()
        .enumerate()
        {
            let mut limits = BudgetLimits::uniform(u64::MAX, 0);
            limits.dimensions[dimension.index()] = totals[index] - 1;
            let mut stopped = Budget::new(limits);
            let error = if phase {
                metered_polynomial_ode_system(
                    &source,
                    &t,
                    &q(1, 3),
                    &force,
                    [&first, &second],
                    &mut stopped,
                    &mut Unbounded,
                )
            } else {
                metered_polynomial_ode_system(
                    &source,
                    &t,
                    &q(1, 3),
                    &force,
                    [&first, &second],
                    &mut Unbounded,
                    &mut stopped,
                )
            };
            assert!(matches!(
                error,
                Err(LinearOdeSystemError::Meter(MeterError::Budget(_)))
            ));
        }
    }
}
