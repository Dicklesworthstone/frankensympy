//! Sparse dependency admission and independent omission/coloring adversaries.

use fsym_calculus::compile::{
    CompileError, CompiledExpr, CompiledOp, EvalError, MAX_COMPILE_DEPTH,
};
use fsym_calculus::sparse_jacobian::{
    JacobianVerificationError, MAX_SPARSE_AXES, MAX_SPARSE_SOURCE_WORK, SparseJacobian,
    SparsityPattern, plan_sparse_sources, verify_sparse_jacobian_certificate,
};
use fsym_core::{Expr, Symbol, SymbolIdentity};
use fsym_proof_kernel::StepId;
use std::collections::HashMap;
use std::sync::Arc;

fn variables(count: usize) -> Vec<Symbol> {
    (0..count)
        .map(|index| Symbol::new(format!("x{index}")))
        .collect()
}

fn empty_candidate(residuals: Vec<Expr>, vars: Vec<Symbol>) -> SparseJacobian {
    SparseJacobian {
        pattern: SparsityPattern {
            num_rows: residuals.len(),
            num_cols: vars.len(),
            nonzeros: Vec::new(),
            nnz: 0,
        },
        column_colors: vec![0; vars.len()],
        num_colors: usize::from(!vars.is_empty()),
        residuals,
        vars,
        entries: Vec::new(),
    }
}

#[test]
fn planning_visits_power_exponents_held_factors_and_every_function_argument() {
    let vars = variables(3);
    let x = Expr::Sym(vars[0].clone());
    let y = Expr::Sym(vars[1].clone());
    let z = Expr::Sym(vars[2].clone());
    let sources = [
        Expr::Pow(Arc::new(x.clone()), Arc::new(y.clone())),
        Expr::Mul(vec![
            Expr::from_i64(0),
            Expr::Function("log".into(), vec![z.clone()]),
        ]),
        Expr::Function("opaque".into(), vec![z, Expr::Add(vec![y, x])]),
    ];
    let plan = plan_sparse_sources(&sources, &vars).unwrap();
    assert_eq!(plan.row_dependencies, [vec![0, 1], vec![2], vec![0, 1, 2]]);
    assert_eq!(plan.source_nodes, 12);
    assert_eq!(plan.differentiation_units, 25);
}

#[test]
fn dependency_binding_distinguishes_symbols_that_print_identically() {
    let plain = Symbol::new("x");
    let typed = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([9; 32]));
    let vars = [plain.clone(), typed.clone(), Symbol::new("unused")];
    let source = Expr::Sym(typed.clone());
    let plan = plan_sparse_sources(std::slice::from_ref(&source), &vars).unwrap();
    assert_eq!(plan.row_dependencies, [vec![1]]);
    let good = SparseJacobian::try_build(&[source], &vars).unwrap();
    assert_eq!(good.pattern.nonzeros, [(0, 1)]);
    verify_sparse_jacobian_certificate(&good, &[]).unwrap();

    let omitted = empty_candidate(good.residuals.clone(), good.vars.clone());
    assert!(matches!(
        verify_sparse_jacobian_certificate(&omitted, &[]),
        Err(JacobianVerificationError::OmittedNonzero { row: 0, col: 1, .. })
    ));
    let mut wrong_axis = good.clone();
    wrong_axis.vars.swap(0, 1);
    assert!(matches!(
        verify_sparse_jacobian_certificate(&wrong_axis, &[]),
        Err(JacobianVerificationError::ProofFailed { .. })
    ));
    assert!(matches!(
        plan_sparse_sources(&[], &[typed.clone(), typed]),
        Err(CompileError::DuplicateVariable(_))
    ));
}

#[test]
fn absence_of_a_declared_axis_never_maps_a_foreign_symbol_by_its_name() {
    let plain = Symbol::new("x");
    let foreign = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([3; 32]));
    let source = Expr::Sym(foreign.clone());
    let plan =
        plan_sparse_sources(std::slice::from_ref(&source), std::slice::from_ref(&plain)).unwrap();
    assert_eq!(plan.row_dependencies, [Vec::<usize>::new()]);
    assert_eq!(plan.differentiation_units, 1);
    assert!(matches!(
        CompiledExpr::try_compile(&source, &HashMap::from([(plain, 0)])),
        Err(CompileError::UnmappedVariable(symbol)) if symbol == foreign
    ));
}

#[test]
fn thousands_of_rows_use_local_work_and_single_color_in_a_million_cell_shape() {
    let vars = variables(256);
    let sources: Vec<_> = (0..4096)
        .map(|row| {
            Expr::Add(vec![
                Expr::Sym(vars[row % vars.len()].clone()),
                Expr::from_i64(-((row % 13) as i64)),
            ])
        })
        .collect();
    let plan = plan_sparse_sources(&sources, &vars).unwrap();
    assert_eq!(plan.source_nodes, 12_288);
    assert_eq!(plan.differentiation_units, 12_288);
    assert_eq!(
        plan.row_dependencies.iter().map(Vec::len).sum::<usize>(),
        4096
    );
    let system = SparseJacobian::try_build(&sources, &vars).unwrap();
    assert_eq!(system.entries.len(), 4096);
    assert_eq!(system.num_colors, 1);
    assert!(system.column_colors.iter().all(|&color| color == 0));
    let receipt = verify_sparse_jacobian_certificate(&system, &[]).unwrap();
    assert_eq!(receipt.proofs_verified, 4096);
    assert_eq!(receipt.nnz, 4096);
    assert_eq!(receipt.density, 1.0 / 256.0);
    let mut values = vec![0.0; 4096];
    system
        .try_eval_sparse(&vec![0.0; 256], &mut values)
        .unwrap();
    assert!(values.iter().all(|&value| value == 1.0));
}

#[test]
fn unused_native_axes_do_not_expand_the_zero_coverage_certificate() {
    let vars = variables(MAX_SPARSE_AXES);
    let sources = [
        Expr::Sym(vars[0].clone()),
        Expr::Sym(vars[MAX_SPARSE_AXES - 1].clone()),
    ];
    let plan = plan_sparse_sources(&sources, &vars).unwrap();
    assert_eq!(plan.source_nodes, 2);
    assert_eq!(plan.differentiation_units, 2);
    assert_eq!(plan.row_dependencies, [vec![0], vec![MAX_SPARSE_AXES - 1]]);
    let system = SparseJacobian::try_build(&sources, &vars).unwrap();
    assert_eq!(system.pattern.nonzeros, [(0, 0), (1, MAX_SPARSE_AXES - 1)]);
    assert_eq!(system.entries.len(), 2);
    assert_eq!(system.num_colors, 1);
    assert_eq!(
        verify_sparse_jacobian_certificate(&system, &[])
            .unwrap()
            .proofs_verified,
        2
    );
}

#[test]
fn incidence_coloring_matches_ascending_greedy_conflicts_and_rejects_collisions() {
    let vars = variables(7);
    let rows = [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3], [0, 5]];
    let sources: Vec<_> = rows
        .iter()
        .map(|columns| {
            Expr::Add(
                columns
                    .iter()
                    .map(|&column| Expr::Sym(vars[column].clone()))
                    .collect(),
            )
        })
        .collect();
    let first = SparseJacobian::try_build(&sources, &vars).unwrap();
    let second = SparseJacobian::try_build(&sources, &vars).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.column_colors, [0, 1, 2, 0, 1, 2, 0]);
    assert_eq!(first.num_colors, 3);
    verify_sparse_jacobian_certificate(&first, &[]).unwrap();
    let mut collision = first;
    collision.column_colors[1] = 0;
    assert!(matches!(
        verify_sparse_jacobian_certificate(&collision, &[]),
        Err(JacobianVerificationError::ColoringCollision { row: 0, .. })
    ));
}

#[test]
fn deleted_present_coordinates_are_recomputed_even_without_diagnostic_points() {
    let vars = variables(3);
    let sources = [
        Expr::Pow(
            Arc::new(Expr::Sym(vars[0].clone())),
            Arc::new(Expr::Sym(vars[1].clone())),
        ),
        Expr::Function("sin".into(), vec![Expr::Sym(vars[2].clone())]),
    ];
    let original = SparseJacobian::try_build(&sources, &vars).unwrap();
    for index in 0..original.entries.len() {
        let mut forged = original.clone();
        let (row, col) = forged.pattern.nonzeros.remove(index);
        forged.entries.remove(index);
        forged.pattern.nnz -= 1;
        assert!(matches!(
            verify_sparse_jacobian_certificate(&forged, &[]),
            Err(JacobianVerificationError::OmittedNonzero { row: found_row, col: found_col, .. })
                if found_row == row && found_col == col
        ));
    }
}

#[test]
fn true_zero_omissions_retain_original_held_source_domain_obligations() {
    let x = Symbol::new("x");
    let raw = Expr::Mul(vec![
        Expr::from_i64(0),
        Expr::Function("log".into(), vec![Expr::Sym(x.clone())]),
    ]);
    let candidate = empty_candidate(vec![raw.clone()], vec![x.clone(), Symbol::new("unused")]);
    let receipt = verify_sparse_jacobian_certificate(&candidate, &[]).unwrap();
    assert_eq!(receipt.nnz, 0);
    assert_eq!(candidate.residuals.as_slice(), std::slice::from_ref(&raw));
    let original = CompiledExpr::try_compile(&raw, &HashMap::from([(x, 0)])).unwrap();
    assert_eq!(original.try_eval_finite(&[1.0]), Ok(0.0));
    assert_eq!(
        original.try_eval_finite(&[0.0]),
        Err(EvalError::NonFiniteValue)
    );
    assert_eq!(
        original.try_eval_finite(&[-1.0]),
        Err(EvalError::NonFiniteValue)
    );
}

#[test]
fn every_retained_entry_still_requires_its_proof_and_exact_compiled_provenance() {
    let vars = variables(3);
    let sources: Vec<_> = vars
        .iter()
        .map(|var| {
            Expr::Pow(
                Arc::new(Expr::Sym(var.clone())),
                Arc::new(Expr::from_i64(2)),
            )
        })
        .collect();
    let original = SparseJacobian::try_build(&sources, &vars).unwrap();
    for index in 0..original.entries.len() {
        let mut bad_program = original.clone();
        bad_program.entries[index].compiled_deriv =
            CompiledExpr::try_from_ops(vec![CompiledOp::LoadConst(42.0)]).unwrap();
        assert!(matches!(
            verify_sparse_jacobian_certificate(&bad_program, &[]),
            Err(JacobianVerificationError::CompiledProvenanceMismatch { .. })
        ));
        let mut bad_proof = original.clone();
        bad_proof.entries[index].derivation.root = StepId(999);
        assert!(matches!(
            verify_sparse_jacobian_certificate(&bad_proof, &[]),
            Err(JacobianVerificationError::ProofFailed { .. })
        ));
    }
}

#[test]
fn serialized_source_and_typed_coordinate_mutations_do_not_reuse_zero_coverage() {
    let vars = variables(2);
    let original = SparseJacobian::try_build(&[Expr::Sym(vars[0].clone())], &vars).unwrap();
    let bytes = serde_json::to_vec(&original).unwrap();
    let restored: SparseJacobian = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(restored, original);
    verify_sparse_jacobian_certificate(&restored, &[]).unwrap();
    let mut source_changed = restored.clone();
    source_changed.residuals[0] =
        Expr::Add(vec![Expr::Sym(vars[0].clone()), Expr::Sym(vars[1].clone())]);
    assert!(verify_sparse_jacobian_certificate(&source_changed, &[]).is_err());

    // An independently supplied omission claim cannot hide a new raw child.
    let omitted = empty_candidate(source_changed.residuals, source_changed.vars);
    assert!(matches!(
        verify_sparse_jacobian_certificate(&omitted, &[]),
        Err(JacobianVerificationError::OmittedNonzero { .. })
    ));
}

#[test]
fn aggregate_dependency_work_and_raw_product_growth_refuse_before_proof_expansion() {
    let vars = variables(100);
    let row = Expr::Add(vars.iter().cloned().map(Expr::Sym).collect());
    let sources = vec![row; 105];
    assert!(matches!(
        plan_sparse_sources(&sources, &vars),
        Err(CompileError::OpLimitExceeded {
            max: MAX_SPARSE_SOURCE_WORK,
            ..
        })
    ));
    assert!(matches!(
        verify_sparse_jacobian_certificate(&empty_candidate(sources, vars), &[]),
        Err(JacobianVerificationError::Compile(
            CompileError::OpLimitExceeded {
                max: MAX_SPARSE_SOURCE_WORK,
                ..
            }
        ))
    ));

    let x = Symbol::new("x");
    let explosive = Expr::Mul(vec![Expr::Sym(x.clone()); 96]);
    assert!(matches!(
        SparseJacobian::try_build(std::slice::from_ref(&explosive), std::slice::from_ref(&x)),
        Err(CompileError::OpLimitExceeded { .. })
    ));
    assert!(matches!(
        verify_sparse_jacobian_certificate(&empty_candidate(vec![explosive], vec![x]), &[]),
        Err(JacobianVerificationError::Compile(
            CompileError::OpLimitExceeded { .. }
        ))
    ));
}

#[test]
fn axis_and_depth_caps_precede_untrusted_sparse_work() {
    let vars = variables(MAX_SPARSE_AXES + 1);
    assert!(matches!(
        plan_sparse_sources(&[], &vars),
        Err(CompileError::OpLimitExceeded {
            max: MAX_SPARSE_AXES,
            ..
        })
    ));
    assert!(matches!(
        verify_sparse_jacobian_certificate(&empty_candidate(Vec::new(), vars), &[]),
        Err(JacobianVerificationError::Compile(
            CompileError::OpLimitExceeded {
                max: MAX_SPARSE_AXES,
                ..
            }
        ))
    ));
    let x = Symbol::new("x");
    let mut deep = Expr::Sym(x.clone());
    for _ in 0..=MAX_COMPILE_DEPTH {
        deep = Expr::Function("sin".into(), vec![deep]);
    }
    assert!(matches!(
        SparseJacobian::try_build(std::slice::from_ref(&deep), std::slice::from_ref(&x)),
        Err(CompileError::DepthLimitExceeded { .. })
    ));
    assert!(matches!(
        verify_sparse_jacobian_certificate(&empty_candidate(vec![deep], vec![x]), &[]),
        Err(JacobianVerificationError::Compile(
            CompileError::DepthLimitExceeded { .. }
        ))
    ));
}

#[test]
fn shared_derivative_forests_are_bounded_before_any_proof_replay() {
    let x = Symbol::new("x");
    let sources = vec![Expr::Sym(x.clone()); 133];
    let mut candidate = SparseJacobian::try_build(&sources, &[x]).unwrap();
    // A cheap untrusted clone can repeat an Arc-backed large derivative while
    // retaining tiny stale proofs. Aggregate admission must stop the complete
    // forest before replay starts, not scan every derivative independently.
    let expanded = Expr::Pow(
        Arc::new(Expr::Add(vec![Expr::from_i64(1); 7900])),
        Arc::new(Expr::from_i64(1)),
    );
    for entry in &mut candidate.entries {
        entry.symbolic_deriv = expanded.clone();
    }
    assert!(matches!(
        verify_sparse_jacobian_certificate(&candidate, &[]),
        Err(JacobianVerificationError::Compile(
            CompileError::OpLimitExceeded {
                max: MAX_SPARSE_SOURCE_WORK,
                ..
            }
        ))
    ));
}
