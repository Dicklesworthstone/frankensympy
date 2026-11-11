//! Independent hand formulas, binding mutants and fail-closed second-order use.

use fsym_calculus::compile::{CompiledExpr, CompiledOp};
use fsym_calculus::sparse_hessian::{
    MAX_OBJECTIVE_PAYLOAD_BITS, MAX_OBJECTIVE_SOURCE_DEPTH, MAX_OBJECTIVE_SOURCE_NODES,
    MAX_OBJECTIVE_VARIABLES, ObjectiveCandidate, VerifiedObjective,
};
use fsym_calculus::sparse_jacobian::SparseJacobian;
use fsym_core::{BigInt, BigRational, Expr, Symbol, SymbolIdentity, parse};
use std::sync::Arc;

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-11 * expected.abs().max(1.0),
        "{actual} != {expected}"
    );
}

fn dense(program: &VerifiedObjective, entries: &[f64]) -> Vec<Vec<f64>> {
    let n = program.variables().len();
    let mut rows = vec![vec![0.0; n]; n];
    for (&(row, col), &entry) in program.hessian_pattern().iter().zip(entries) {
        rows[row][col] = entry;
    }
    rows
}

fn drop_entry(jacobian: &mut SparseJacobian, index: usize) {
    jacobian.entries.remove(index);
    jacobian.pattern.nonzeros.remove(index);
    jacobian.pattern.nnz -= 1;
}

#[test]
fn rosenbrock_value_gradient_hessian_and_product_match_hand_formulas() {
    let source = parse("(1-x)^2+100*(y-x^2)^2").unwrap();
    let program =
        VerifiedObjective::try_compile(&source, &[Symbol::new("x"), Symbol::new("y")]).unwrap();
    let result = program.evaluate(&[-1.2, 1.0]).unwrap();
    close(result.value, 24.2);
    close(result.gradient[0], -215.6);
    close(result.gradient[1], -88.0);
    for (row, expected) in dense(&program, &result.hessian)
        .iter()
        .zip([[1330.0, 480.0], [480.0, 200.0]])
    {
        for (&actual, expected) in row.iter().zip(expected) {
            close(actual, expected);
        }
    }
    let product = program
        .hessian_vector_product(&[-1.2, 1.0], &[2.0, -3.0])
        .unwrap();
    close(product[0], 1220.0);
    close(product[1], 360.0);
    assert_eq!(program.receipt().gradient_proofs_verified, 2);
    assert_eq!(program.receipt().hessian_proofs_verified, 4);
    assert_eq!(program.receipt().zero_entries_checked, 0);
}

#[test]
fn transcendental_chain_rules_match_independent_closed_form() {
    let source = parse("exp(x*y)+sin(x)+log(y)").unwrap();
    let program =
        VerifiedObjective::try_compile(&source, &[Symbol::new("x"), Symbol::new("y")]).unwrap();
    let result = program.evaluate(&[0.5, 2.0]).unwrap();
    let e = std::f64::consts::E;
    close(result.value, e + 0.5f64.sin() + 2.0f64.ln());
    close(result.gradient[0], 2.0 * e + 0.5f64.cos());
    close(result.gradient[1], 0.5 * e + 0.5);
    for (row, expected) in dense(&program, &result.hessian).iter().zip([
        [4.0 * e - 0.5f64.sin(), 2.0 * e],
        [2.0 * e, 0.25 * e - 0.25],
    ]) {
        for (&actual, expected) in row.iter().zip(expected) {
            close(actual, expected);
        }
    }
}

#[test]
fn dynamic_powers_retain_positive_domain_through_both_derivative_stages() {
    let source = parse("x^x").unwrap();
    let program = VerifiedObjective::try_compile(&source, &[Symbol::new("x")]).unwrap();
    let result = program.evaluate(&[2.0]).unwrap();
    let log_two = 2.0f64.ln();
    close(result.value, 4.0);
    close(result.gradient[0], 4.0 * (log_two + 1.0));
    close(result.hessian[0], 4.0 * ((log_two + 1.0).powi(2) + 0.5));
    for value in [-2.0, 0.0] {
        assert!(program.evaluate(&[value]).is_err());
        assert!(program.hessian_vector_product(&[value], &[0.0]).is_err());
    }
}

#[test]
fn both_mixed_orders_and_all_omitted_zeros_are_independently_checked() {
    let source = parse("x*y+z^3").unwrap();
    let variables = [Symbol::new("x"), Symbol::new("y"), Symbol::new("z")];
    let program = VerifiedObjective::try_compile(&source, &variables).unwrap();
    assert_eq!(program.hessian_pattern(), [(0, 1), (1, 0), (2, 2)]);
    let result = program.evaluate(&[2.0, 3.0, 4.0]).unwrap();
    assert_eq!(result.value, 70.0);
    assert_eq!(result.gradient, [3.0, 2.0, 48.0]);
    assert_eq!(result.hessian, [1.0, 1.0, 24.0]);
    assert_eq!(program.receipt().zero_entries_checked, 6);
    assert_eq!(
        program
            .hessian_vector_product(&[2.0, 3.0, 4.0], &[7.0, 11.0, 13.0])
            .unwrap(),
        [11.0, 7.0, 312.0]
    );
    for omitted in [0, 1, 2] {
        let mut candidate = program.candidate().clone();
        drop_entry(&mut candidate.hessian, omitted);
        assert!(VerifiedObjective::try_from_candidate(&source, &variables, candidate).is_err());
    }
    let mut candidate = program.candidate().clone();
    drop_entry(&mut candidate.gradient, 0);
    assert!(VerifiedObjective::try_from_candidate(&source, &variables, candidate).is_err());
}

#[test]
fn constants_linear_sources_and_empty_axes_preserve_dimensions() {
    let constant = VerifiedObjective::try_compile(&Expr::from_i64(7), &[]).unwrap();
    let result = constant.evaluate(&[]).unwrap();
    assert_eq!(result.value, 7.0);
    assert!(result.gradient.is_empty() && result.hessian.is_empty());
    assert!(
        constant
            .hessian_vector_product(&[], &[])
            .unwrap()
            .is_empty()
    );
    assert!(constant.evaluate(&[1.0]).is_err());
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let linear = VerifiedObjective::try_compile(&parse("2*x-y+7").unwrap(), &variables).unwrap();
    let result = linear.evaluate(&[3.0, 4.0]).unwrap();
    assert_eq!(result.value, 9.0);
    assert_eq!(result.gradient, [2.0, -1.0]);
    assert!(result.hessian.is_empty());
    assert_eq!(
        linear
            .hessian_vector_product(&[3.0, 4.0], &[9.0, 8.0])
            .unwrap(),
        [0.0, 0.0]
    );
    assert!(linear.evaluate(&[f64::NAN, 0.0]).is_err());
}

#[test]
fn equal_printed_symbol_names_keep_distinct_typed_coordinates() {
    let plain = Symbol::new("x");
    let typed = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([3; 32]));
    let source = Expr::Add(vec![
        Expr::pow(Expr::Sym(plain.clone()), Expr::from_i64(2)),
        Expr::Mul(vec![
            Expr::from_i64(3),
            Expr::pow(Expr::Sym(typed.clone()), Expr::from_i64(2)),
        ]),
    ]);
    let variables = [plain.clone(), typed.clone()];
    let program = VerifiedObjective::try_compile(&source, &variables).unwrap();
    assert_eq!(program.variables(), &variables);
    assert_eq!(program.hessian_pattern(), [(0, 0), (1, 1)]);
    let result = program.evaluate(&[2.0, 3.0]).unwrap();
    assert_eq!(result.value, 31.0);
    assert_eq!(result.gradient, [4.0, 18.0]);
    assert_eq!(result.hessian, [2.0, 6.0]);
    assert!(VerifiedObjective::try_compile(&source, &[plain.clone(), plain]).is_err());
    assert!(VerifiedObjective::try_compile(&source, &[typed]).is_err());
}

#[test]
fn sixty_four_variable_product_uses_the_sparse_ordered_pattern() {
    let variables: Vec<_> = (0..MAX_OBJECTIVE_VARIABLES)
        .map(|index| Symbol::new(format!("x{index}")))
        .collect();
    let source = Expr::Add(vec![
        Expr::Mul(vec![
            Expr::Sym(variables[0].clone()),
            Expr::Sym(variables[63].clone()),
        ]),
        Expr::pow(Expr::Sym(variables[32].clone()), Expr::from_i64(2)),
    ]);
    let program = VerifiedObjective::try_compile(&source, &variables).unwrap();
    assert_eq!(program.hessian_pattern(), [(0, 63), (32, 32), (63, 0)]);
    let direction: Vec<_> = (0..MAX_OBJECTIVE_VARIABLES)
        .map(|index| index as f64 + 1.0)
        .collect();
    let product = program
        .hessian_vector_product(&vec![2.0; MAX_OBJECTIVE_VARIABLES], &direction)
        .unwrap();
    assert_eq!(product[0], 64.0);
    assert_eq!(product[32], 66.0);
    assert_eq!(product[63], 1.0);
    for (index, value) in product.iter().enumerate() {
        if ![0, 32, 63].contains(&index) {
            assert_eq!(*value, 0.0);
        }
    }
}

#[test]
fn original_domain_survives_zero_products_and_zero_hessian_directions() {
    let x = Symbol::new("x");
    let source = Expr::Add(vec![
        Expr::Sym(x.clone()),
        Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::Function("log".into(), vec![Expr::Sym(x.clone())]),
        ]),
    ]);
    let program = VerifiedObjective::try_compile(&source, std::slice::from_ref(&x)).unwrap();
    assert!(program.hessian_pattern().is_empty());
    assert_eq!(program.evaluate(&[2.0]).unwrap().gradient, [1.0]);
    for point in [-1.0, 0.0] {
        assert!(program.evaluate(&[point]).is_err());
        assert!(program.hessian_vector_product(&[point], &[0.0]).is_err());
    }
}

#[test]
fn first_and_second_derivative_singularities_refuse_even_zero_directions() {
    let x = Symbol::new("x");
    for numerator in [1, 3] {
        let source = Expr::pow(
            Expr::Sym(x.clone()),
            Expr::Rational(BigRational::new(BigInt::from(numerator), BigInt::from(2))),
        );
        let program = VerifiedObjective::try_compile(&source, std::slice::from_ref(&x)).unwrap();
        assert!(program.evaluate(&[0.0]).is_err());
        assert!(program.hessian_vector_product(&[0.0], &[0.0]).is_err());
        assert!(program.evaluate(&[4.0]).is_ok());
    }
}

#[test]
fn failures_are_atomic_and_the_artifact_remains_reusable() {
    let program =
        VerifiedObjective::try_compile(&parse("exp(x)+x^2").unwrap(), &[Symbol::new("x")]).unwrap();
    let mut gradient = [17.0];
    let mut hessian = [19.0];
    assert!(
        program
            .try_eval_into(&[1000.0], &mut gradient, &mut hessian)
            .is_err()
    );
    assert_eq!(gradient, [17.0]);
    assert_eq!(hessian, [19.0]);
    let mut output = [23.0];
    assert!(
        program
            .try_hessian_vector_product_into(&[0.0], &[f64::MAX], &mut output)
            .is_err()
    );
    assert_eq!(output, [23.0]);
    assert!(
        program
            .try_eval_into(&[], &mut gradient, &mut hessian)
            .is_err()
    );
    assert_eq!(gradient, [17.0]);
    assert_eq!(hessian, [19.0]);
    assert_eq!(
        program
            .try_eval_into(&[0.0], &mut gradient, &mut hessian)
            .unwrap(),
        1.0
    );
    assert_eq!(gradient, [1.0]);
    assert_eq!(hessian, [3.0]);
}

#[test]
fn untrusted_compiled_source_and_derivative_instructions_are_rejected() {
    let source = parse("x^2+y^2").unwrap();
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let original = ObjectiveCandidate::try_build(&source, &variables).unwrap();
    let mut wrong = original.clone();
    wrong.compiled_objective =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(2.0)]).unwrap();
    assert!(VerifiedObjective::try_from_candidate(&source, &variables, wrong).is_err());
    let mut wrong = original.clone();
    // Sample agreement at x=1 cannot stand in for compiled provenance.
    wrong.gradient.entries[0].compiled_deriv =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(2.0)]).unwrap();
    assert!(VerifiedObjective::try_from_candidate(&source, &variables, wrong).is_err());
    let mut wrong = original;
    wrong.hessian.entries[0].compiled_deriv =
        CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(0.0)]).unwrap();
    assert!(VerifiedObjective::try_from_candidate(&source, &variables, wrong).is_err());
}

#[test]
fn certificate_source_variable_order_and_second_stage_binding_are_exact() {
    let source = parse("x^2+y^2").unwrap();
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let original = ObjectiveCandidate::try_build(&source, &variables).unwrap();
    assert!(
        VerifiedObjective::try_from_candidate(
            &parse("x^2+y^2+1").unwrap(),
            &variables,
            original.clone()
        )
        .is_err()
    );
    assert!(
        VerifiedObjective::try_from_candidate(
            &source,
            &[variables[1].clone(), variables[0].clone()],
            original.clone()
        )
        .is_err()
    );
    let mut wrong = original.clone();
    wrong.hessian.residuals[0] = Expr::from_i64(0);
    assert!(VerifiedObjective::try_from_candidate(&source, &variables, wrong).is_err());
    let mut wrong = original;
    wrong.hessian.entries[0].symbolic_deriv = Expr::from_i64(0);
    assert!(VerifiedObjective::try_from_candidate(&source, &variables, wrong).is_err());
}

#[test]
fn malformed_coordinates_and_derivation_envelopes_cannot_bypass_checks() {
    let source = parse("x*y").unwrap();
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let original = ObjectiveCandidate::try_build(&source, &variables).unwrap();
    let mut wrong = original.clone();
    wrong.hessian.entries[0].col = 1000;
    assert!(VerifiedObjective::try_from_candidate(&source, &variables, wrong).is_err());
    let mut wrong = original.clone();
    wrong.hessian.pattern.nonzeros.reverse();
    assert!(VerifiedObjective::try_from_candidate(&source, &variables, wrong).is_err());
    let mut wrong = original.clone();
    wrong.hessian.entries[0].derivation.steps.clear();
    assert!(VerifiedObjective::try_from_candidate(&source, &variables, wrong).is_err());
    let mut wrong = original;
    wrong.gradient.entries[0].derivation.root = fsym_proof_kernel::StepId(9);
    assert!(VerifiedObjective::try_from_candidate(&source, &variables, wrong).is_err());
}

#[test]
fn shape_depth_payload_and_untrusted_proof_limits_precede_recursive_work() {
    let variables: Vec<_> = (0..=MAX_OBJECTIVE_VARIABLES)
        .map(|index| Symbol::new(format!("x{index}")))
        .collect();
    assert!(VerifiedObjective::try_compile(&Expr::from_i64(0), &variables).is_err());
    let x = Symbol::new("x");
    let wide = Expr::Add(vec![Expr::Sym(x.clone()); MAX_OBJECTIVE_SOURCE_NODES]);
    assert!(VerifiedObjective::try_compile(&wide, std::slice::from_ref(&x)).is_err());
    let mut deep = Expr::Sym(x.clone());
    for _ in 0..=MAX_OBJECTIVE_SOURCE_DEPTH {
        deep = Expr::Function("sin".into(), vec![deep]);
    }
    assert!(VerifiedObjective::try_compile(&deep, std::slice::from_ref(&x)).is_err());
    let enormous = Expr::Integer(BigInt::from(1) << (MAX_OBJECTIVE_PAYLOAD_BITS as u32 + 1));
    assert!(VerifiedObjective::try_compile(&enormous, &[]).is_err());
    let source = parse("x^2").unwrap();
    let mut candidate = ObjectiveCandidate::try_build(&source, std::slice::from_ref(&x)).unwrap();
    let mut proof_expression = Expr::Sym(x.clone());
    for _ in 0..40 {
        proof_expression = Expr::Pow(Arc::new(proof_expression), Arc::new(Expr::from_i64(2)));
    }
    candidate.hessian.entries[0].derivation.steps[0].claim =
        fsym_proof_kernel::Claim::equality(proof_expression, Expr::from_i64(2));
    assert!(VerifiedObjective::try_from_candidate(&source, &[x], candidate).is_err());
}

#[test]
fn unsupported_original_nodes_cannot_disappear_through_differentiation() {
    let x = Symbol::new("x");
    let source = Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::Function("opaque".into(), vec![Expr::Sym(x.clone())]),
    ]);
    assert!(VerifiedObjective::try_compile(&source, &[x]).is_err());
}

fn assert_float_bits(actual: &[f64], expected: &[f64]) {
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
    );
}

#[test]
fn shared_objective_and_products_match_independent_scalar_execution_bit_for_bit() {
    let source = parse("exp(x*y)+sin(x*y)+(x*y)^3").unwrap();
    let variables = [Symbol::new("x"), Symbol::new("y"), Symbol::new("unused")];
    let program = VerifiedObjective::try_compile(&source, &variables).unwrap();
    assert!(program.shared_subexpression_count() > 0);
    let original_candidate = program.candidate().clone();
    let original_receipt = program.receipt().clone();
    let restored =
        VerifiedObjective::try_from_candidate(&source, &variables, original_candidate.clone())
            .unwrap();
    assert_eq!(
        restored.shared_subexpression_count(),
        program.shared_subexpression_count(),
    );
    let direction = [2.0, -3.0, 5.0];
    for point in [
        [0.0, -0.0, 4.0],
        [-0.0, 0.0, 1.0],
        [0.5, 2.0, -3.0],
        [-0.75, 1.3, 0.0],
    ] {
        // The reference interpreter executes each original program without
        // any shared memo. This checks operation order, typed axis placement,
        // sparse zeros and floating-point bits, not just approximate equality.
        let value = original_candidate
            .compiled_objective
            .try_eval_finite(&point)
            .unwrap();
        let mut gradient = vec![0.0; variables.len()];
        for entry in &original_candidate.gradient.entries {
            gradient[entry.col] = entry.compiled_deriv.try_eval_finite(&point).unwrap();
        }
        let hessian: Vec<_> = original_candidate
            .hessian
            .entries
            .iter()
            .map(|entry| entry.compiled_deriv.try_eval_finite(&point).unwrap())
            .collect();
        let mut product = vec![0.0; variables.len()];
        for (&(row, col), &derivative) in program.hessian_pattern().iter().zip(&hessian) {
            product[row] += derivative * direction[col];
        }
        for checked in [&program, &restored] {
            let actual = checked.evaluate(&point).unwrap();
            assert_eq!(actual.value.to_bits(), value.to_bits());
            assert_float_bits(&actual.gradient, &gradient);
            assert_float_bits(&actual.hessian, &hessian);
            assert_float_bits(
                &checked.hessian_vector_product(&point, &direction).unwrap(),
                &product,
            );
            let mut into_gradient = vec![901.0; variables.len()];
            let mut into_hessian = vec![902.0; hessian.len()];
            assert_eq!(
                checked
                    .try_eval_into(&point, &mut into_gradient, &mut into_hessian)
                    .unwrap()
                    .to_bits(),
                value.to_bits(),
            );
            assert_float_bits(&into_gradient, &gradient);
            assert_float_bits(&into_hessian, &hessian);
            let mut into_product = vec![903.0; variables.len()];
            checked
                .try_hessian_vector_product_into(&point, &direction, &mut into_product)
                .unwrap();
            assert_float_bits(&into_product, &product);
        }
    }
    assert_eq!(program.candidate(), &original_candidate);
    assert_eq!(program.receipt(), &original_receipt);
}

#[test]
fn shared_objective_memos_do_not_cross_points_failures_or_imports() {
    let source = parse("exp(x*y)").unwrap();
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let program = VerifiedObjective::try_compile(&source, &variables).unwrap();
    assert!(program.shared_subexpression_count() > 0);
    let first = program.evaluate(&[0.5, 2.0]).unwrap();
    let second = program.evaluate(&[0.5, -2.0]).unwrap();
    close(first.value, std::f64::consts::E);
    close(second.value, (-1.0f64).exp());
    close(second.gradient[0], -2.0 * (-1.0f64).exp());
    assert_ne!(first.value.to_bits(), second.value.to_bits());
    let mut gradient = [901.0; 2];
    let mut hessian = [902.0; 4];
    assert!(
        program
            .try_eval_into(&[1000.0, 1000.0], &mut gradient, &mut hessian)
            .is_err()
    );
    assert_eq!(gradient, [901.0; 2]);
    assert_eq!(hessian, [902.0; 4]);
    let again = program.evaluate(&[0.5, 2.0]).unwrap();
    assert_eq!(again.value.to_bits(), first.value.to_bits());
    assert_float_bits(&again.gradient, &first.gradient);
    assert_float_bits(&again.hessian, &first.hessian);

    // Even when the source and several derivatives share exp(x*y), replacing
    // one derivative program cannot reuse a trusted source result to escape
    // the independent source/proof admission boundary.
    let mut forged = program.candidate().clone();
    forged.hessian.entries[0].compiled_deriv = forged.compiled_objective.clone();
    assert!(VerifiedObjective::try_from_candidate(&source, &variables, forged).is_err());
}

#[test]
fn memoized_values_do_not_hide_source_or_second_derivative_singularities() {
    for (text, invalid) in [("log(x)", -1.0), ("x^(3/2)", 0.0)] {
        let expression = parse(text).unwrap();
        // Preserve the repeated source regions so they actually participate
        // in shared evaluation rather than relying on a canonicalized sum.
        let source = Expr::Add(vec![expression.clone(), expression]);
        let program = VerifiedObjective::try_compile(&source, &[Symbol::new("x")]).unwrap();
        assert!(program.shared_subexpression_count() > 0);
        let valid = program.evaluate(&[4.0]).unwrap();
        if text == "x^(3/2)" {
            let candidate = program.candidate();
            assert!(
                candidate
                    .compiled_objective
                    .try_eval_finite(&[invalid])
                    .is_ok()
            );
            assert!(
                candidate
                    .gradient
                    .entries
                    .iter()
                    .all(|entry| { entry.compiled_deriv.try_eval_finite(&[invalid]).is_ok() })
            );
        }
        let mut gradient = [901.0];
        let mut hessian = [902.0];
        assert!(
            program
                .try_eval_into(&[invalid], &mut gradient, &mut hessian)
                .is_err()
        );
        assert_eq!(gradient, [901.0]);
        assert_eq!(hessian, [902.0]);
        let mut product = [903.0];
        assert!(
            program
                .try_hessian_vector_product_into(&[invalid], &[0.0], &mut product)
                .is_err()
        );
        assert_eq!(product, [903.0]);
        let again = program.evaluate(&[4.0]).unwrap();
        assert_eq!(again.value.to_bits(), valid.value.to_bits());
        assert_float_bits(&again.gradient, &valid.gradient);
        assert_float_bits(&again.hessian, &valid.hessian);
    }
}
