//! Independent rational block identities, general systems, forcing and limits.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::rational_laplace::{LaplaceBlock, RationalLaplaceCertificate};
use fsym_core::elementary::rational_expr;
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_solvers::ode::system::general::{
    GeneralOdeSystemCertificate, GeneralOdeSystemError, RationalMatrix, general_ode_system,
    metered_general_ode_system, verify_general_ode_system,
};
use std::collections::HashMap;

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}
fn e(source: &str) -> Expr {
    parse(source).unwrap()
}
fn matrix(rows: &[&[i64]]) -> RationalMatrix {
    rows.iter()
        .map(|row| row.iter().map(|&value| q(value, 1)).collect())
        .collect()
}
fn zeros(n: usize) -> Vec<Vec<BigRational>> {
    vec![vec![q(0, 1)]; n]
}
fn zero() -> RationalLaplaceCertificate {
    RationalLaplaceCertificate {
        blocks: Vec::new(),
    }
}
fn polynomial(coefficients: &[BigRational]) -> RationalLaplaceCertificate {
    RationalLaplaceCertificate {
        blocks: vec![LaplaceBlock::Exponential {
            rate: q(0, 1),
            coefficients: coefficients.to_vec(),
        }],
    }
}
fn exponential(rate: i64) -> RationalLaplaceCertificate {
    RationalLaplaceCertificate {
        blocks: vec![LaplaceBlock::Exponential {
            rate: q(rate, 1),
            coefficients: vec![q(1, 1)],
        }],
    }
}
fn quadratic(d: i64, cosine: i64, sine: i64) -> RationalLaplaceCertificate {
    RationalLaplaceCertificate {
        blocks: vec![LaplaceBlock::Quadratic {
            rate: q(0, 1),
            squared_frequency: q(d, 1),
            cosine: vec![q(cosine, 1)],
            sine: vec![q(sine, 1)],
        }],
    }
}
fn simplified(expression: &Expr) -> Expr {
    fsym_simplify::try_simplify(&fsym_simplify::try_expand(expression).unwrap()).unwrap()
}
fn at(expression: &Expr, time: &Symbol, point: &BigRational) -> Expr {
    simplified(&expression.subs(&HashMap::from([(
        time.clone(),
        rational_expr(point.clone()),
    )])))
}
fn assert_equal(actual: &Expr, expected: &Expr) {
    let residual = simplified(&(actual.clone() - expected.clone()));
    assert!(
        residual.is_zero(),
        "{actual} != {expected}; residual {residual}"
    );
}
fn polynomial_expression(values: &[BigRational], time: &Symbol) -> Expr {
    values
        .iter()
        .rev()
        .fold(Expr::from_i64(0), |value, coefficient| {
            value * Expr::Sym(time.clone()) + rational_expr(coefficient.clone())
        })
}
fn assert_equations(
    solutions: &[Expr],
    coefficients: &[Vec<BigRational>],
    forcing: &[Vec<BigRational>],
    time: &Symbol,
) {
    for (row, solution) in solutions.iter().enumerate() {
        let mut rhs = polynomial_expression(&forcing[row], time);
        for (coefficient, value) in coefficients[row].iter().zip(solutions) {
            rhs = rhs + rational_expr(coefficient.clone()) * value.clone();
        }
        assert_equal(&fsym_calculus::diff(solution, time), &rhs);
    }
}
fn chain(n: usize) -> RationalMatrix {
    let mut result = vec![vec![q(0, 1); n]; n];
    for (row, entries) in result.iter_mut().enumerate().take(n - 1) {
        entries[row + 1] = q(1, 1);
    }
    result
}
fn chain_certificate() -> GeneralOdeSystemCertificate {
    let one = polynomial(&[q(1, 1)]);
    let local = polynomial(&[q(0, 1), q(1, 1)]);
    let half_square = polynomial(&[q(0, 1), q(0, 1), q(1, 2)]);
    GeneralOdeSystemCertificate {
        fundamental: vec![
            vec![one.clone(), local.clone(), half_square],
            vec![zero(), one.clone(), local],
            vec![zero(), zero(), one],
        ],
        particular: vec![zero(), zero(), zero()],
    }
}

#[test]
fn independent_three_component_fundamental_needs_no_generator() {
    let t = Symbol::new("t");
    let coefficients = chain(3);
    let initial = vec![e("C1"), e("C2"), e("C3")];
    let receipt = verify_general_ode_system(
        &coefficients,
        &t,
        &q(2, 1),
        &zeros(3),
        &initial,
        &chain_certificate(),
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(receipt.coefficients(), coefficients);
    assert_eq!(receipt.initial_point(), &q(2, 1));
    assert_eq!(receipt.initial_values(), initial);
    assert_eq!(receipt.time(), &t);
    assert_equal(&receipt.solutions()[0], &e("C1+C2*(t-2)+C3*(t-2)**2/2"));
    for (actual, expected) in receipt.solutions().iter().zip(&initial) {
        assert_eq!(at(actual, &t, &q(2, 1)), *expected);
    }
    assert_equations(receipt.solutions(), &coefficients, &zeros(3), &t);
}

#[test]
fn normalized_sine_identity_is_checked_for_both_frequency_signs() {
    let t = Symbol::new("t");
    for d in [2, -2] {
        let coefficients = matrix(&[&[0, 1, 0], &[-d, 0, 0], &[0, 0, 3]]);
        let candidate = GeneralOdeSystemCertificate {
            fundamental: vec![
                vec![quadratic(d, 1, 0), quadratic(d, 0, 1), zero()],
                vec![quadratic(d, 0, -d), quadratic(d, 1, 0), zero()],
                vec![zero(), zero(), exponential(3)],
            ],
            particular: vec![zero(), zero(), zero()],
        };
        let initial = [e("2"), e("3"), e("5")];
        let receipt = verify_general_ode_system(
            &coefficients,
            &t,
            &q(0, 1),
            &zeros(3),
            &initial,
            &candidate,
            &mut Unbounded,
        )
        .unwrap();
        assert_equations(receipt.solutions(), &coefficients, &zeros(3), &t);
        for (actual, expected) in receipt.solutions().iter().zip(&initial) {
            assert_eq!(at(actual, &t, &q(0, 1)), *expected);
        }
        let mut changed = candidate.clone();
        let LaplaceBlock::Quadratic { sine, .. } =
            &mut changed.fundamental[1][0].blocks[0]
        else {
            panic!("quadratic fixture")
        };
        sine[0] += q(1, 7);
        assert!(matches!(
            verify_general_ode_system(
                &coefficients,
                &t,
                &q(0, 1),
                &zeros(3),
                &initial,
                &changed,
                &mut Unbounded,
            ),
            Err(GeneralOdeSystemError::Verification("fundamental differential identity"))
        ));
    }
}

#[test]
fn generated_dense_similarity_preserves_coordinate_order() {
    // A = S diag([[0,1],[-1,0]], 2) S^-1, with
    // S = [[1,1,1],[1,2,1],[1,1,2]] and initial S*[1,0,1].
    let t = Symbol::new("t");
    let coefficients = matrix(&[&[-6, 2, 3], &[-9, 3, 4], &[-8, 2, 5]]);
    let initial = [e("2"), e("2"), e("3")];
    let receipt = general_ode_system(&coefficients, &t, &q(0, 1), &zeros(3), &initial).unwrap();
    let expected = [
        e("cos(t)-sin(t)+exp(2*t)"),
        e("cos(t)-2*sin(t)+exp(2*t)"),
        e("cos(t)-sin(t)+2*exp(2*t)"),
    ];
    for (actual, expected) in receipt.solutions().iter().zip(&expected) {
        assert_equal(actual, expected);
    }
    assert_equations(receipt.solutions(), &coefficients, &zeros(3), &t);
}

#[test]
fn repeated_eigenvalue_jordan_chain_generates_every_independent_constant() {
    let t = Symbol::new("t");
    let coefficients = matrix(&[
        &[2, 1, 0, 0],
        &[0, 2, 1, 0],
        &[0, 0, 2, 1],
        &[0, 0, 0, 2],
    ]);
    let initial = [e("C1"), e("C2"), e("C3"), e("C4")];
    let receipt = general_ode_system(&coefficients, &t, &q(0, 1), &zeros(4), &initial).unwrap();
    assert_equal(
        &receipt.solutions()[0],
        &e("exp(2*t)*(C1+C2*t+C3*t**2/2+C4*t**3/6)"),
    );
    assert_equations(receipt.solutions(), &coefficients, &zeros(4), &t);
    for (actual, expected) in receipt.solutions().iter().zip(&initial) {
        assert_eq!(at(actual, &t, &q(0, 1)), *expected);
    }
}

#[test]
fn original_forcing_and_nonzero_initial_time_are_checked_independently() {
    let t = Symbol::new("t");
    let coefficients = chain(3);
    let forcing = vec![
        vec![q(0, 1)],
        vec![q(0, 1)],
        vec![q(0, 1), q(0, 1), q(1, 1)],
    ];
    let initial = [e("1"), e("2"), e("3")];
    // Exact zero-initial particular in tau=t-2; the forcing is original t^2.
    let mut candidate = chain_certificate();
    candidate.particular = vec![
        polynomial(&[
            q(0, 1),
            q(0, 1),
            q(0, 1),
            q(2, 3),
            q(1, 6),
            q(1, 60),
        ]),
        polynomial(&[q(0, 1), q(0, 1), q(2, 1), q(2, 3), q(1, 12)]),
        polynomial(&[q(0, 1), q(4, 1), q(2, 1), q(1, 3)]),
    ];
    let checked = verify_general_ode_system(
        &coefficients,
        &t,
        &q(2, 1),
        &forcing,
        &initial,
        &candidate,
        &mut Unbounded,
    )
    .unwrap();
    let generated = general_ode_system(&coefficients, &t, &q(2, 1), &forcing, &initial).unwrap();
    for (actual, expected) in generated.solutions().iter().zip(checked.solutions()) {
        assert_equal(actual, expected);
    }
    assert_equations(generated.solutions(), &coefficients, &forcing, &t);
    for value in generated.particular() {
        assert_eq!(at(value, &t, &q(2, 1)), e("0"));
    }
    assert!(matches!(
        verify_general_ode_system(
            &coefficients,
            &t,
            &q(0, 1),
            &forcing,
            &initial,
            &candidate,
            &mut Unbounded,
        ),
        Err(GeneralOdeSystemError::Verification("particular differential identity"))
    ));
}

#[test]
fn eight_component_degree_sixteen_forcing_reaches_degree_twenty_four() {
    let t = Symbol::new("t");
    let coefficients = chain(8);
    let mut forcing = zeros(8);
    forcing[7] = vec![q(0, 1); 17];
    forcing[7][16] = q(1, 1);
    let initial = vec![e("0"); 8];
    let receipt = general_ode_system(&coefficients, &t, &q(0, 1), &forcing, &initial).unwrap();
    let mut denominator = BigInt::from(1);
    for (row, solution) in receipt.solutions().iter().enumerate().rev() {
        denominator *= BigInt::from(24 - row);
        let power = Expr::Sym(t.clone()).pow(Expr::from_i64((24 - row) as i64));
        let expected = power * rational_expr(BigRational::new(1.into(), denominator.clone()));
        assert_equal(solution, &expected);
        assert_eq!(at(solution, &t, &q(0, 1)), e("0"));
    }
    assert_equations(receipt.solutions(), &coefficients, &forcing, &t);
}

#[test]
fn singleton_and_typed_time_do_not_capture_internal_or_initial_names() {
    let t = Symbol::with_identity(
        "_fsym_ode_transform_0",
        SymbolIdentity::from_assumptions_digest([7; 32]),
    );
    let parameter = Symbol::new("_fsym_ode_local_0");
    let initial = [Expr::Sym(parameter.clone())];
    let receipt = general_ode_system(
        &[vec![q(0, 1)]],
        &t,
        &q(3, 2),
        &[vec![q(1, 1)]],
        &initial,
    )
    .unwrap();
    assert_equal(
        &receipt.solutions()[0],
        &(Expr::Sym(parameter.clone()) + Expr::Sym(t.clone()) - rational_expr(q(3, 2))),
    );
    let symbols = receipt.solutions()[0].free_symbols();
    assert!(symbols.contains(&t));
    assert!(symbols.contains(&parameter));
    assert_eq!(symbols.len(), 2);
    assert_eq!(at(&receipt.solutions()[0], &t, &q(3, 2)), initial[0]);
}

#[test]
fn parameter_slots_are_distinct_and_cannot_rebind_the_time_coordinate() {
    let t = Symbol::new("t");
    let coefficients = chain(3);
    for initial in [
        vec![e("C1"), e("C1"), e("C3")],
        vec![e("C1"), Expr::Sym(t.clone()), e("C3")],
        vec![e("C1+C2"), e("0"), e("0")],
    ] {
        assert!(matches!(
            general_ode_system(&coefficients, &t, &q(0, 1), &zeros(3), &initial),
            Err(GeneralOdeSystemError::InvalidSource(_))
        ));
    }
    let same_name = Symbol::with_identity("t", SymbolIdentity::from_assumptions_digest([8; 32]));
    let initial = [Expr::Sym(same_name.clone()), e("2"), e("3")];
    let receipt = general_ode_system(&coefficients, &t, &q(0, 1), &zeros(3), &initial).unwrap();
    assert_eq!(at(&receipt.solutions()[0], &t, &q(0, 1)), Expr::Sym(same_name));
}

#[test]
fn irreducible_cubic_spectrum_is_an_explicit_unsupported_family() {
    let coefficients = matrix(&[&[0, 1, 0], &[0, 0, 1], &[2, 0, 0]]);
    let error = general_ode_system(
        &coefficients,
        &Symbol::new("t"),
        &q(0, 1),
        &zeros(3),
        &[e("1"), e("0"), e("0")],
    )
    .unwrap_err();
    assert!(error.is_unsupported(), "{error}");
}

#[test]
fn omitted_modes_changed_matrix_and_particular_constants_are_rejected() {
    let t = Symbol::new("t");
    let coefficients = chain(3);
    let initial = [e("0"), e("0"), e("0")];
    let candidate = chain_certificate();
    let mut changed = candidate.clone();
    changed.fundamental[0][2] = zero();
    assert!(matches!(
        verify_general_ode_system(
            &coefficients,
            &t,
            &q(0, 1),
            &zeros(3),
            &initial,
            &changed,
            &mut Unbounded,
        ),
        Err(GeneralOdeSystemError::Verification("fundamental differential identity"))
    ));
    let mut changed_matrix = coefficients.clone();
    changed_matrix[1][2] += q(1, 1_000_000_007);
    assert!(
        verify_general_ode_system(
            &changed_matrix,
            &t,
            &q(0, 1),
            &zeros(3),
            &initial,
            &candidate,
            &mut Unbounded,
        )
        .is_err()
    );
    let mut changed = candidate.clone();
    changed.particular[0] = polynomial(&[q(1, 1)]);
    assert!(matches!(
        verify_general_ode_system(
            &coefficients,
            &t,
            &q(0, 1),
            &zeros(3),
            &initial,
            &changed,
            &mut Unbounded,
        ),
        Err(GeneralOdeSystemError::Verification("particular initial vector"))
    ));
    let mut changed = candidate.clone();
    changed.fundamental[0][0] = polynomial(&[q(2, 1)]);
    assert!(matches!(
        verify_general_ode_system(
            &coefficients,
            &t,
            &q(0, 1),
            &zeros(3),
            &initial,
            &changed,
            &mut Unbounded,
        ),
        Err(GeneralOdeSystemError::Verification("fundamental initial matrix"))
    ));
}

#[test]
fn degenerate_frequencies_noncanonical_blocks_and_oversized_certificates_refuse() {
    let t = Symbol::new("t");
    let coefficients = chain(3);
    let initial = [e("0"), e("0"), e("0")];
    let mut candidates = Vec::new();
    let mut changed = chain_certificate();
    changed.fundamental[0][0] = quadratic(0, 1, 0);
    candidates.push(changed);
    let mut changed = chain_certificate();
    changed.fundamental[0][0] = polynomial(&[q(1, 1), q(0, 1)]);
    candidates.push(changed);
    let mut changed = chain_certificate();
    changed.fundamental[0][0] = polynomial(&vec![q(1, 1); 33]);
    candidates.push(changed);
    let mut changed = chain_certificate();
    changed.fundamental.pop();
    candidates.push(changed);
    for candidate in candidates {
        assert!(
            verify_general_ode_system(
                &coefficients,
                &t,
                &q(0, 1),
                &zeros(3),
                &initial,
                &candidate,
                &mut Unbounded,
            )
            .is_err()
        );
    }
}

#[test]
fn all_input_bounds_apply_before_transform_search() {
    let t = Symbol::new("t");
    for n in [0, 9] {
        let coefficients = vec![vec![q(0, 1); n]; n];
        assert!(matches!(
            general_ode_system(&coefficients, &t, &q(0, 1), &zeros(n), &vec![e("0"); n]),
            Err(GeneralOdeSystemError::InvalidSource(_))
        ));
    }
    let mut coefficients = chain(3);
    coefficients[0][0] = BigRational::from_integer(BigInt::from(1) << 1024usize);
    assert!(matches!(
        general_ode_system(&coefficients, &t, &q(0, 1), &zeros(3), &[e("0"), e("0"), e("0")]),
        Err(GeneralOdeSystemError::ResourceLimit(_))
    ));
    let mut forcing = zeros(3);
    forcing[1] = vec![q(1, 1); 18];
    assert!(matches!(
        general_ode_system(&chain(3), &t, &q(0, 1), &forcing, &[e("0"), e("0"), e("0")]),
        Err(GeneralOdeSystemError::InvalidSource(_))
    ));
}

#[derive(Default)]
struct Meter {
    checkpoints: usize,
    cancel_at: Option<usize>,
    stopped: bool,
    charges: [u64; 3],
}
impl BudgetMeter for Meter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.stopped, "charged after cancellation");
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
fn cumulative_ledgers_and_late_cancellation_never_publish_partial_families() {
    let t = Symbol::new("t");
    let coefficients = chain(3);
    let forcing = vec![vec![q(0, 1)], vec![q(0, 1)], vec![q(1, 1)]];
    let initial = [e("1"), e("2"), e("3")];
    let (mut generator, mut verifier) = (Meter::default(), Meter::default());
    metered_general_ode_system(
        &coefficients,
        &t,
        &q(0, 1),
        &forcing,
        &initial,
        &mut generator,
        &mut verifier,
    )
    .unwrap();
    for (in_generation, measured) in [(true, &generator), (false, &verifier)] {
        assert!(measured.charges.iter().all(|&value| value > 0));
        for stop in [
            1,
            measured.checkpoints / 2,
            measured.checkpoints - 1,
            measured.checkpoints,
        ] {
            let mut meter = Meter {
                cancel_at: Some(stop),
                ..Meter::default()
            };
            let result = if in_generation {
                metered_general_ode_system(
                    &coefficients,
                    &t,
                    &q(0, 1),
                    &forcing,
                    &initial,
                    &mut meter,
                    &mut Unbounded,
                )
            } else {
                metered_general_ode_system(
                    &coefficients,
                    &t,
                    &q(0, 1),
                    &forcing,
                    &initial,
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
            limits.dimensions[dimension.index()] = measured.charges[index] - 1;
            let mut meter = Budget::new(limits);
            let result = if in_generation {
                metered_general_ode_system(
                    &coefficients,
                    &t,
                    &q(0, 1),
                    &forcing,
                    &initial,
                    &mut meter,
                    &mut Unbounded,
                )
            } else {
                metered_general_ode_system(
                    &coefficients,
                    &t,
                    &q(0, 1),
                    &forcing,
                    &initial,
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
    let mut tiny = Budget::new(BudgetLimits::uniform(1, 0));
    let mut untouched = Meter::default();
    assert!(
        metered_general_ode_system(
            &coefficients,
            &t,
            &q(0, 1),
            &forcing,
            &initial,
            &mut tiny,
            &mut untouched,
        )
        .is_err()
    );
    assert_eq!(untouched.checkpoints, 0);
    assert_eq!(untouched.charges, [0; 3]);
}
