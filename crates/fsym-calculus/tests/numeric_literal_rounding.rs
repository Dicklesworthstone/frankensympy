//! Exact rounding cells and boundary cases for approximate numeric literals.

use fsym_calculus::VerifiedResidualSystem;
use fsym_calculus::compile::{CompileError, CompiledExpr, CompiledOp, EvalError};
use fsym_calculus::sparse_hessian::VerifiedObjective;
use fsym_calculus::sparse_jacobian::compiled_expression_matches;
use fsym_core::{BigInt, BigRational, Expr, Symbol};
use std::collections::HashMap;
use std::sync::Arc;

fn power_of_two(exponent: u32) -> BigInt {
    BigInt::one() << exponent
}

fn rational(numerator: BigInt, denominator: BigInt) -> Expr {
    Expr::Rational(BigRational::new(numerator, denominator))
}

fn compiled_literal_bits(source: &Expr) -> u64 {
    let variables = HashMap::new();
    let compiled = CompiledExpr::try_compile(source, &variables).unwrap();
    assert!(compiled_expression_matches(source, &variables, &compiled).unwrap());
    let [CompiledOp::LoadConst(value)] = compiled.ops() else {
        panic!("a literal must compile to exactly one constant load");
    };
    assert_eq!(
        compiled.try_eval_finite(&[]).unwrap().to_bits(),
        value.to_bits()
    );
    value.to_bits()
}

#[test]
fn coprime_ratios_with_overflowing_components_compile_to_finite_values() {
    for exponent in [1024, 2000, 4095] {
        let denominator = power_of_two(exponent);
        let numerator = &denominator + 1i64;
        // The components are coprime and each exceeds binary64's finite range.
        assert_eq!(numerator.gcd(&denominator), BigInt::one());
        for (numerator, expected) in [
            (numerator.clone(), 1.0f64.to_bits()),
            (-numerator, (-1.0f64).to_bits()),
        ] {
            assert_eq!(
                compiled_literal_bits(&rational(numerator, denominator.clone())),
                expected
            );
        }
    }
}

#[test]
fn exact_quotients_are_rounded_once_and_old_component_rounding_is_rejected() {
    let base = power_of_two(53);
    let denominator = &base + 1i64;
    // The first exact quotient is just above the predecessor of 1. The
    // second is just below the midpoint between 1 and its successor.
    // Rounding the denominator to 2^53 first gets both answers wrong.
    for (numerator, expected, wrong) in [
        (base.clone(), 0x3fef_ffff_ffff_ffff, 1.0f64),
        (
            &base + 2i64,
            1.0f64.to_bits(),
            f64::from_bits(0x3ff0_0000_0000_0001),
        ),
    ] {
        let source = rational(numerator.clone(), denominator.clone());
        assert_eq!(compiled_literal_bits(&source), expected);
        let mutant = CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(wrong)]).unwrap();
        assert!(!compiled_expression_matches(&source, &HashMap::new(), &mutant).unwrap());
        assert_eq!(
            compiled_literal_bits(&rational(-numerator, denominator.clone())),
            expected | (1u64 << 63)
        );
    }
}

#[test]
fn normal_midpoints_use_ties_to_even() {
    let denominator = power_of_two(53);
    for (offset, expected) in [
        (1i64, 0x3ff0_0000_0000_0000),
        (3, 0x3ff0_0000_0000_0002),
        (5, 0x3ff0_0000_0000_0002),
        (7, 0x3ff0_0000_0000_0004),
    ] {
        let numerator = &denominator + offset;
        for (numerator, sign) in [(numerator.clone(), 0), (-numerator, 1u64 << 63)] {
            assert_eq!(
                compiled_literal_bits(&rational(numerator, denominator.clone())),
                expected | sign
            );
        }
    }
}

#[test]
fn subnormal_rounding_and_underflow_keep_canonical_bytecode_zero() {
    for (numerator, exponent, expected) in [
        (1i64, 1074, 1u64),
        (1, 1075, 0),
        (3, 1075, 2),
        (5, 1075, 2),
        (7, 1075, 4),
        (1, 2000, 0),
    ] {
        for sign in [1i64, -1] {
            let expected = if expected == 0 || sign > 0 {
                expected
            } else {
                expected | (1u64 << 63)
            };
            assert_eq!(
                compiled_literal_bits(&rational(
                    BigInt::from(numerator * sign),
                    power_of_two(exponent),
                )),
                expected
            );
        }
    }
    let transition_midpoint = rational(power_of_two(53) - 1i64, power_of_two(1075));
    assert_eq!(
        compiled_literal_bits(&transition_midpoint),
        0x0010_0000_0000_0000
    );
    // The fix does not relax the external bytecode boundary.
    assert_eq!(
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(-0.0)]),
        Err(EvalError::NonCanonicalConstant)
    );
}

#[test]
fn integers_round_directly_and_true_literal_overflow_still_refuses() {
    let exact_boundary = power_of_two(53);
    for (offset, expected) in [(1i64, 0x4340_0000_0000_0000), (3, 0x4340_0000_0000_0002)] {
        assert_eq!(
            compiled_literal_bits(&Expr::Integer(&exact_boundary + offset)),
            expected
        );
        assert_eq!(
            compiled_literal_bits(&Expr::Integer(-(&exact_boundary + offset))),
            expected | (1u64 << 63)
        );
    }
    let maximum_finite = (power_of_two(53) - 1i64) << 971u32;
    let overflow_midpoint = (power_of_two(54) - 1i64) << 970u32;
    for numerator in [maximum_finite, &overflow_midpoint - 1i64] {
        assert_eq!(
            compiled_literal_bits(&Expr::Integer(numerator)),
            f64::MAX.to_bits()
        );
    }
    // This coprime quotient lies less than one unit below the overflow
    // midpoint, so it still rounds to the maximum finite binary64 value.
    assert_eq!(
        compiled_literal_bits(&rational(&overflow_midpoint * 3i64 - 1i64, 3i64.into())),
        f64::MAX.to_bits()
    );
    for source in [
        Expr::Integer(overflow_midpoint.clone()),
        Expr::Integer(-overflow_midpoint),
        rational(power_of_two(2050) + 1i64, power_of_two(1024)),
        rational(-power_of_two(2050) - 1i64, power_of_two(1024)),
    ] {
        assert!(matches!(
            CompiledExpr::try_compile(&source, &HashMap::new()),
            Err(CompileError::NumericConversion(_))
        ));
    }
}

// Decode IEEE-754 bits by their exact dyadic definition. This reference does
// not call either the rational conversion routine or the numeric compiler.
fn exact_positive_float(bits: u64) -> BigRational {
    let fraction = bits & ((1u64 << 52) - 1);
    let biased_exponent = (bits >> 52) as i32;
    let (significand, exponent) = if biased_exponent == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1u64 << 52), biased_exponent - 1023 - 52)
    };
    if exponent >= 0 {
        BigRational::from_integer(BigInt::from(significand) << exponent as u32)
    } else {
        BigRational::new(BigInt::from(significand), power_of_two((-exponent) as u32))
    }
}

#[test]
fn compiled_ratios_stay_in_their_independent_exact_rounding_cells() {
    let base = power_of_two(53);
    let two = BigRational::from_integer(2i64.into());
    for scale in [-900i32, -52, 0, 52, 900] {
        for numerator_offset in -5i64..=5 {
            for denominator_offset in -4i64..=4 {
                let mut numerator = &base + numerator_offset;
                let mut denominator = &base + denominator_offset;
                if scale >= 0 {
                    numerator = numerator << scale as u32;
                } else {
                    denominator = denominator << (-scale) as u32;
                }
                let exact = BigRational::new(numerator, denominator);
                let bits = compiled_literal_bits(&Expr::Rational(exact.clone()));
                let lower_midpoint =
                    (exact_positive_float(bits - 1) + exact_positive_float(bits)) / &two;
                let upper_midpoint =
                    (exact_positive_float(bits) + exact_positive_float(bits + 1)) / &two;
                assert!(exact >= lower_midpoint && exact <= upper_midpoint);
                if exact == lower_midpoint || exact == upper_midpoint {
                    assert_eq!(bits & 1, 0, "a midpoint must choose the even significand");
                }
            }
        }
    }
}

#[test]
fn huge_finite_coefficients_reach_verified_residuals_and_objectives() {
    let x = Symbol::new("x");
    let coefficient = rational(power_of_two(2000) + 1i64, power_of_two(2000));
    let source = Expr::Mul(vec![
        coefficient,
        Expr::Pow(Arc::new(Expr::Sym(x.clone())), Arc::new(Expr::from_i64(2))),
    ]);
    let residual = VerifiedResidualSystem::try_compile(
        std::slice::from_ref(&source),
        std::slice::from_ref(&x),
    )
    .unwrap();
    let mut values = [f64::NAN];
    let mut jacobian = [f64::NAN];
    residual
        .compiled()
        .try_eval_system(&[2.0], &mut values, &mut jacobian)
        .unwrap();
    assert_eq!(values, [4.0]);
    assert_eq!(jacobian, [4.0]);
    // Import reconstructs and independently rechecks the source-bound program.
    let restored =
        VerifiedResidualSystem::from_json_bytes(&residual.to_json_bytes().unwrap()).unwrap();
    assert_eq!(
        restored.compiled().compiled_residuals[0]
            .try_eval_finite(&[2.0])
            .unwrap(),
        4.0
    );

    let objective = VerifiedObjective::try_compile(&source, &[x]).unwrap();
    let evaluated = objective.evaluate(&[2.0]).unwrap();
    assert_eq!(evaluated.value, 4.0);
    assert_eq!(evaluated.gradient, [4.0]);
    assert_eq!(evaluated.hessian, [2.0]);
    assert_eq!(
        objective.hessian_vector_product(&[2.0], &[3.0]).unwrap(),
        [6.0]
    );
}

#[test]
fn exponent_components_keep_the_existing_exact_integer_admission_bound() {
    let variables = HashMap::from([(Symbol::new("x"), 0)]);
    let in_range = Expr::Pow(
        Arc::new(Expr::symbol("x")),
        Arc::new(Expr::Integer(power_of_two(53) - 1i64)),
    );
    let compiled = CompiledExpr::try_compile(&in_range, &variables).unwrap();
    assert!(compiled_expression_matches(&in_range, &variables, &compiled).unwrap());
    assert_eq!(compiled.try_eval_finite(&[1.0]).unwrap(), 1.0);
    for exponent in [
        Expr::Integer(power_of_two(53)),
        rational(power_of_two(53) + 1i64, power_of_two(53) - 1i64),
    ] {
        let source = Expr::Pow(Arc::new(Expr::symbol("x")), Arc::new(exponent));
        assert!(matches!(
            CompiledExpr::try_compile(&source, &variables),
            Err(CompileError::UnsupportedExponent(_))
        ));
    }
}
