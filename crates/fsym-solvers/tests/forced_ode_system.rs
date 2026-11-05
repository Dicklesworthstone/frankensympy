//! Original-source checks for exponential/trigonometric coupled systems.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_calculus::rational_laplace::{LaplaceBlock, RationalLaplaceCertificate};
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity, parse};
use fsym_solvers::ode::system::general::forcing::{
    ElementaryForcing, ForcedOdeSystemCertificate, ForcingBasis, ForcingRealization,
    forced_ode_system, metered_forced_ode_system, verify_forced_ode_system,
};
use fsym_solvers::ode::system::general::{GeneralOdeSystemCertificate, GeneralOdeSystemError};
use std::collections::HashMap;

fn q(value: i64) -> BigRational {
    BigRational::from_integer(value.into())
}
fn e(source: &str) -> Expr {
    parse(source).unwrap()
}
fn zero() -> RationalLaplaceCertificate {
    RationalLaplaceCertificate { blocks: Vec::new() }
}
fn exponential(rate: i64, coefficients: &[i64]) -> RationalLaplaceCertificate {
    RationalLaplaceCertificate {
        blocks: vec![LaplaceBlock::Exponential {
            rate: q(rate),
            coefficients: coefficients.iter().map(|&value| q(value)).collect(),
        }],
    }
}
fn mode(
    power: usize,
    rate: i64,
    frequency: i64,
    cosine: &[i64],
    sine: &[i64],
) -> ElementaryForcing {
    ElementaryForcing {
        power,
        rate: q(rate),
        frequency: q(frequency),
        cosine: cosine.iter().map(|&value| q(value)).collect(),
        sine: sine.iter().map(|&value| q(value)).collect(),
    }
}
fn simplified(value: &Expr) -> Expr {
    fsym_simplify::try_simplify(&fsym_simplify::try_expand(value).unwrap()).unwrap()
}
fn equal(actual: &Expr, expected: &Expr) {
    let residual = simplified(&(actual.clone() - expected.clone()));
    assert!(
        residual.is_zero(),
        "{actual} != {expected}; residual {residual}"
    );
}
fn at(value: &Expr, time: &Symbol, point: i64) -> Expr {
    simplified(&value.subs(&HashMap::from([(time.clone(), Expr::from_i64(point))])))
}

// This certificate is written directly from elementary functions. It never
// runs the realization or resolvent generators.
fn independent_exponential_certificate() -> ForcedOdeSystemCertificate {
    let one = exponential(0, &[1]);
    let mut increment = exponential(1, &[1]);
    increment.blocks.push(LaplaceBlock::Exponential {
        rate: q(0),
        coefficients: vec![q(-1)],
    });
    ForcedOdeSystemCertificate {
        realization: ForcingRealization {
            basis: vec![ForcingBasis {
                rate: q(1),
                frequency: q(0),
                power: 0,
                sine: false,
            }],
            coefficients: vec![vec![q(0), q(1)], vec![q(0), q(1)]],
        },
        baseline: GeneralOdeSystemCertificate {
            fundamental: vec![vec![one.clone()]],
            particular: vec![zero()],
        },
        driven: GeneralOdeSystemCertificate {
            fundamental: vec![vec![one, increment], vec![zero(), exponential(1, &[1])]],
            particular: vec![zero(), zero()],
        },
    }
}

#[test]
fn independent_certificate_binds_the_original_forcing_and_nonzero_initial_time() {
    let t = Symbol::new("t");
    let candidate = independent_exponential_certificate();
    let result = verify_forced_ode_system(
        &[vec![q(0)]],
        &t,
        &q(2),
        &[vec![q(0)]],
        &[mode(0, 1, 0, &[1], &[0])],
        &[e("C1")],
        &candidate,
        &mut Unbounded,
    )
    .unwrap();
    equal(&result.solutions()[0], &e("C1+exp(t)-exp(2)"));
    equal(
        &fsym_calculus::diff(&result.solutions()[0], &t),
        &e("exp(t)"),
    );
    assert_eq!(at(&result.solutions()[0], &t, 2), e("C1"));
    assert_eq!(result.initial_point(), &q(2));
    assert_eq!(result.time(), &t);
    assert_eq!(result.coefficients(), &[vec![q(0)]]);
    assert_eq!(result.polynomial_forcing(), &[vec![q(0)]]);
    assert_eq!(result.elementary_forcing(), &[mode(0, 1, 0, &[1], &[0])]);
    assert_eq!(result.initial_values(), &[e("C1")]);
    assert_eq!(result.fundamental(), &[vec![e("1")]]);
}

#[test]
fn resonant_exponential_forcing_preserves_the_complete_initial_family() {
    let t = Symbol::new("t");
    let result = forced_ode_system(
        &[vec![q(2)]],
        &t,
        &q(0),
        &[vec![q(0)]],
        &[mode(0, 2, 0, &[1], &[0])],
        &[e("C1")],
    )
    .unwrap();
    equal(&result.solutions()[0], &e("(C1+t)*exp(2*t)"));
    equal(
        &fsym_calculus::diff(&result.solutions()[0], &t),
        &(e("2") * result.solutions()[0].clone() + e("exp(2*t)")),
    );
    assert_eq!(at(&result.solutions()[0], &t, 0), e("C1"));
}

#[test]
fn coupled_oscillator_handles_resonant_sine_forcing() {
    let t = Symbol::new("t");
    let matrix = vec![vec![q(0), q(1)], vec![q(-1), q(0)]];
    let result = forced_ode_system(
        &matrix,
        &t,
        &q(0),
        &[vec![q(0)], vec![q(0)]],
        &[mode(0, 0, 1, &[0, 0], &[0, 1])],
        &[e("0"), e("0")],
    )
    .unwrap();
    equal(&result.solutions()[0], &e("(sin(t)-t*cos(t))/2"));
    equal(&result.solutions()[1], &e("t*sin(t)/2"));
    equal(
        &fsym_calculus::diff(&result.solutions()[0], &t),
        &result.solutions()[1],
    );
    equal(
        &fsym_calculus::diff(&result.solutions()[1], &t),
        &(e("-1") * result.solutions()[0].clone() + e("sin(t)")),
    );
}

#[test]
fn polynomial_times_exponential_forcing_uses_a_closed_derivative_chain() {
    let t = Symbol::new("t");
    let result = forced_ode_system(
        &[vec![q(0)]],
        &t,
        &q(0),
        &[vec![q(0)]],
        &[mode(2, 1, 0, &[1], &[0])],
        &[e("0")],
    )
    .unwrap();
    equal(&result.solutions()[0], &e("(t**2-2*t+2)*exp(t)-2"));
    equal(
        &fsym_calculus::diff(&result.solutions()[0], &t),
        &e("t**2*exp(t)"),
    );
    assert_eq!(result.certificate().realization.basis.len(), 3);
}

#[test]
fn mixed_polynomial_exponential_and_cosine_loads_keep_absolute_time() {
    let t = Symbol::new("t");
    let matrix = vec![vec![q(0); 2]; 2];
    let result = forced_ode_system(
        &matrix,
        &t,
        &q(2),
        &[vec![q(0), q(1)], vec![q(1)]],
        &[
            mode(0, 1, 0, &[1, 0], &[0, 0]),
            mode(0, 0, 2, &[0, 2], &[0, 0]),
        ],
        &[e("1"), e("3")],
    )
    .unwrap();
    equal(&result.solutions()[0], &e("1+(t**2-4)/2+exp(t)-exp(2)"));
    equal(&result.solutions()[1], &e("3+t-2+sin(2*t)-sin(4)"));
    equal(
        &fsym_calculus::diff(&result.solutions()[0], &t),
        &e("t+exp(t)"),
    );
    equal(
        &fsym_calculus::diff(&result.solutions()[1], &t),
        &e("1+2*cos(2*t)"),
    );
    assert_eq!(at(&result.solutions()[0], &t, 2), e("1"));
    assert_eq!(at(&result.solutions()[1], &t, 2), e("3"));
}

#[test]
fn duplicate_modes_accumulate_instead_of_overwriting_the_source() {
    let t = Symbol::new("t");
    let result = forced_ode_system(
        &[vec![q(0)]],
        &t,
        &q(0),
        &[vec![q(0)]],
        &[mode(0, 1, 0, &[2], &[0]), mode(0, 1, 0, &[-1], &[0])],
        &[e("0")],
    )
    .unwrap();
    equal(&result.solutions()[0], &e("exp(t)-1"));
    assert_eq!(result.certificate().realization.basis.len(), 1);
}

#[test]
fn source_and_coordinated_realization_mutations_are_rejected() {
    let t = Symbol::new("t");
    let source = [mode(0, 1, 0, &[1], &[0])];
    let candidate = independent_exponential_certificate();
    let verify = |value: &ForcedOdeSystemCertificate, modes: &[ElementaryForcing]| {
        verify_forced_ode_system(
            &[vec![q(0)]],
            &t,
            &q(0),
            &[vec![q(0)]],
            modes,
            &[e("0")],
            value,
            &mut Unbounded,
        )
    };
    let mut mutations = Vec::new();
    let mut changed = candidate.clone();
    changed.realization.coefficients[0][1] = q(2);
    mutations.push(changed);
    let mut changed = candidate.clone();
    changed.realization.coefficients[1][0] = q(1);
    mutations.push(changed);
    let mut changed = candidate.clone();
    changed.driven.fundamental[0][1] = zero();
    mutations.push(changed);
    let mut changed = candidate.clone();
    changed.baseline.fundamental[0][0] = zero();
    mutations.push(changed);
    // The altered augmented system and its exponential agree with one
    // another, but they no longer realize the original exp(t) source.
    let mut changed = candidate.clone();
    changed.realization.coefficients[0][1] = q(2);
    changed.realization.coefficients[1][1] = q(2);
    changed.driven.fundamental[0][1] = RationalLaplaceCertificate {
        blocks: vec![
            LaplaceBlock::Exponential {
                rate: q(2),
                coefficients: vec![q(1)],
            },
            LaplaceBlock::Exponential {
                rate: q(0),
                coefficients: vec![q(-1)],
            },
        ],
    };
    changed.driven.fundamental[1][1] = exponential(2, &[1]);
    mutations.push(changed);
    for changed in mutations {
        assert!(verify(&changed, &source).is_err());
    }
    assert!(verify(&candidate, &[mode(0, 2, 0, &[1], &[0])]).is_err());
    assert!(verify(&candidate, &[mode(1, 1, 0, &[1], &[0])]).is_err());
    let mut tiny = source[0].clone();
    tiny.cosine[0] += BigRational::new(1.into(), BigInt::from(1) << 200u32);
    assert!(verify(&candidate, &[tiny]).is_err());
}

#[test]
fn original_modes_and_augmented_size_are_admitted_before_search() {
    let t = Symbol::new("t");
    for invalid in [
        mode(17, 1, 0, &[1], &[0]),
        mode(0, 1, -1, &[0], &[0]),
        mode(0, 1, 0, &[1], &[1]),
        mode(0, 1, 0, &[1, 2], &[0]),
    ] {
        assert!(
            forced_ode_system(
                &[vec![q(0)]],
                &t,
                &q(0),
                &[vec![q(0)]],
                &[invalid],
                &[e("0")],
            )
            .is_err()
        );
    }
    let mut oversized = mode(0, 0, 0, &[0], &[0]);
    oversized.rate = BigRational::from_integer(BigInt::from(1) << 1024u32);
    assert!(
        forced_ode_system(
            &[vec![q(0)]],
            &t,
            &q(0),
            &[vec![q(0)]],
            &[oversized],
            &[e("0")],
        )
        .is_err()
    );
    let matrix = vec![vec![q(0); 8]; 8];
    let error = forced_ode_system(
        &matrix,
        &t,
        &q(0),
        &vec![vec![q(0)]; 8],
        &[mode(0, 1, 0, &[1; 8], &[0; 8])],
        &vec![e("0"); 8],
    )
    .unwrap_err();
    assert!(error.is_unsupported(), "{error}");
}

#[test]
fn malformed_realization_is_rejected_without_panics() {
    let t = Symbol::new("t");
    for mutation in 0..5 {
        let mut candidate = independent_exponential_certificate();
        match mutation {
            0 => {
                let _ = candidate.realization.coefficients.pop();
            }
            1 => candidate.realization.coefficients[0].push(q(0)),
            2 => candidate.realization.basis[0].sine = true,
            3 => candidate
                .realization
                .basis
                .push(candidate.realization.basis[0].clone()),
            _ => candidate.realization.basis[0].power = usize::MAX,
        }
        assert!(
            verify_forced_ode_system(
                &[vec![q(0)]],
                &t,
                &q(0),
                &[vec![q(0)]],
                &[mode(0, 1, 0, &[1], &[0])],
                &[e("0")],
                &candidate,
                &mut Unbounded,
            )
            .is_err()
        );
    }
}

#[test]
fn oversized_basis_scalars_are_rejected_before_any_rational_ordering() {
    let t = Symbol::new("t");
    let mut candidate = independent_exponential_certificate();
    let height = BigInt::from(1) << 2048u32;
    candidate.realization.basis = [1i64, 2]
        .into_iter()
        .map(|offset| ForcingBasis {
            rate: BigRational::new(1.into(), &height + BigInt::from(offset)),
            frequency: q(0),
            power: 0,
            sine: false,
        })
        .collect();
    candidate.realization.coefficients = vec![vec![q(0); 3]; 3];
    let error = verify_forced_ode_system(
        &[vec![q(0)]],
        &t,
        &q(0),
        &[vec![q(0)]],
        &[mode(0, 1, 0, &[1], &[0])],
        &[e("0")],
        &candidate,
        &mut Unbounded,
    )
    .unwrap_err();
    assert!(matches!(error, GeneralOdeSystemError::ResourceLimit(_)));
}

#[test]
fn typed_time_and_same_print_constant_keep_distinct_ownership() {
    let time = Symbol::with_identity("t", SymbolIdentity::from_assumptions_digest([23; 32]));
    let constant = Symbol::new("t");
    let result = forced_ode_system(
        &[vec![q(0)]],
        &time,
        &q(0),
        &[vec![q(0)]],
        &[mode(0, 1, 0, &[1], &[0])],
        &[Expr::Sym(constant.clone())],
    )
    .unwrap();
    let symbols = result.solutions()[0].free_symbols();
    assert!(symbols.contains(&time));
    assert!(symbols.contains(&constant));
    assert_eq!(symbols.len(), 2);
    assert_eq!(at(&result.solutions()[0], &time, 0), Expr::Sym(constant));
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
fn cancellation_and_cumulative_phase_budgets_cover_composition() {
    let matrix = [vec![q(0)]];
    let polynomial = [vec![q(0)]];
    let modes = [mode(0, 1, 0, &[1], &[0])];
    let initial = [e("0")];
    let t = Symbol::new("t");
    let (mut generator, mut verifier) = (Meter::default(), Meter::default());
    metered_forced_ode_system(
        &matrix,
        &t,
        &q(2),
        &polynomial,
        &modes,
        &initial,
        &mut generator,
        &mut verifier,
    )
    .unwrap();
    for (generating, measured) in [(true, &generator), (false, &verifier)] {
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
            let result = if generating {
                metered_forced_ode_system(
                    &matrix,
                    &t,
                    &q(2),
                    &polynomial,
                    &modes,
                    &initial,
                    &mut meter,
                    &mut Unbounded,
                )
            } else {
                metered_forced_ode_system(
                    &matrix,
                    &t,
                    &q(2),
                    &polynomial,
                    &modes,
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
            let result = if generating {
                metered_forced_ode_system(
                    &matrix,
                    &t,
                    &q(2),
                    &polynomial,
                    &modes,
                    &initial,
                    &mut meter,
                    &mut Unbounded,
                )
            } else {
                metered_forced_ode_system(
                    &matrix,
                    &t,
                    &q(2),
                    &polynomial,
                    &modes,
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
}
