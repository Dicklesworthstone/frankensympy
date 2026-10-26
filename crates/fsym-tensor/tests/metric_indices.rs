use fsym_core::{BigInt, BigRational, Expr, Symbol};
use fsym_tensor::{
    IndexVariance, MetricTensor, TensorError, TensorExpr, TensorIndex, verify_metric_index_change,
};

fn ints(values: &[i64]) -> Vec<Expr> {
    values.iter().copied().map(Expr::from_i64).collect()
}

fn metric() -> MetricTensor {
    MetricTensor::new(
        "g",
        2,
        ints(&[2, 1, 1, 2]),
        vec![
            Expr::Rational(BigRational::new(2.into(), 3.into())),
            Expr::Rational(BigRational::new((-1).into(), 3.into())),
            Expr::Rational(BigRational::new((-1).into(), 3.into())),
            Expr::Rational(BigRational::new(2.into(), 3.into())),
        ],
    )
    .unwrap()
}

fn tensor(values: &[i64], rank: usize) -> TensorExpr {
    TensorExpr::with_components(
        "T",
        2,
        (0..rank)
            .map(|i| TensorIndex::upper(format!("a{i}")))
            .collect(),
        ints(values),
    )
    .unwrap()
}

#[test]
fn rank_two_axes_and_double_lowering_match_exact_components() {
    let g = metric();
    let t = tensor(&[1, 2, 3, 4], 2);
    let first = g.lower_index(&t, 0, "A").unwrap();
    let second = g.lower_index(&t, 1, "B").unwrap();
    assert_eq!(first.components, Some(ints(&[5, 8, 7, 10])));
    assert_eq!(second.components, Some(ints(&[4, 5, 10, 11])));
    assert_eq!(first.indices[0].variance, IndexVariance::Lower);
    assert_eq!(first.indices[1], t.indices[1]);
    let both = g.lower_index(&first, 1, "C").unwrap();
    assert_eq!(both.components, Some(ints(&[18, 21, 24, 27])));
    let other_order = g.lower_index(&second, 0, "C").unwrap();
    assert_eq!(both, other_order);
    let raised_once = g.raise_index(&both, 0, "D").unwrap();
    assert_eq!(raised_once.components, second.components);
    assert_eq!(g.raise_index(&raised_once, 1, "T").unwrap(), t);
    assert_eq!(t.components, Some(ints(&[1, 2, 3, 4])));
}

#[test]
fn rank_three_transforms_middle_and_terminal_slots() {
    let g = metric();
    let t = tensor(&[1, 2, 3, 4, 5, 6, 7, 8], 3);
    for (axis, expected) in [
        (0, [7, 10, 13, 16, 11, 14, 17, 20]),
        (1, [5, 8, 7, 10, 17, 20, 19, 22]),
        (2, [4, 5, 10, 11, 16, 17, 22, 23]),
    ] {
        let changed = g.lower_index(&t, axis, "U").unwrap();
        assert_eq!(changed.components, Some(ints(&expected)));
        assert_eq!(g.raise_index(&changed, axis, "T").unwrap(), t);
    }
}

#[test]
fn independent_checker_rejects_every_tiny_component_mutation() {
    let g = metric();
    let t = tensor(&[1, 2, 3, 4], 2);
    // The reference checker accepts a manually specified result, independent
    // of the contraction generator.
    let mut candidate = tensor(&[5, 8, 7, 10], 2);
    candidate.indices[0].variance = IndexVariance::Lower;
    verify_metric_index_change(&g, &t, 0, &candidate).unwrap();
    let tiny = BigRational::new(1.into(), BigInt::from(10).pow(80));
    for index in 0..4 {
        let mut forged = candidate.clone();
        let Expr::Integer(value) = &candidate.components.as_ref().unwrap()[index] else {
            panic!()
        };
        forged.components.as_mut().unwrap()[index] =
            Expr::Rational(BigRational::from_integer(value.clone()) + &tiny);
        assert!(verify_metric_index_change(&g, &t, 0, &forged).is_err());
    }
    let mut wrong_symbol = candidate.clone();
    wrong_symbol.indices[1].symbol = Symbol::new("impostor");
    assert!(verify_metric_index_change(&g, &t, 0, &wrong_symbol).is_err());
    let mut wrong_slot = candidate.clone();
    wrong_slot.indices[0].variance = IndexVariance::Upper;
    wrong_slot.indices[1].variance = IndexVariance::Lower;
    assert!(verify_metric_index_change(&g, &t, 0, &wrong_slot).is_err());
    assert!(verify_metric_index_change(&g, &t, 1, &candidate).is_err());
}

#[test]
fn indefinite_metrics_and_metric_contraction_are_exact() {
    let g = MetricTensor::new("null", 2, ints(&[0, 1, 1, 0]), ints(&[0, 1, 1, 0])).unwrap();
    let t = tensor(&[1, 2, 3, 4], 2);
    let lower = g.lower_index(&t, 1, "L").unwrap();
    assert_eq!(lower.components, Some(ints(&[2, 1, 4, 3])));
    let contracted = lower
        .self_contract(&Symbol::new("a0"), &Symbol::new("a1"), "scalar")
        .unwrap();
    assert_eq!(contracted.components, Some(ints(&[5])));
    let eta = MetricTensor::minkowski_4d("eta");
    let v = TensorExpr::with_components("v", 4, vec![TensorIndex::upper("a")], ints(&[2, 3, 4, 5]))
        .unwrap();
    assert_eq!(
        eta.lower_vector(&v).unwrap().components,
        Some(ints(&[-2, 3, 4, 5]))
    );
    assert_eq!(eta.norm_squared(&v).unwrap(), Expr::from_i64(46));
}

#[test]
fn rational_constructor_rejects_bad_inverse_asymmetry_and_singularity() {
    for (matrix, inverse) in [
        ([2, 1, 1, 2], [1, 0, 0, 1]),
        ([1, 1, 0, 1], [1, -1, 0, 1]),
        ([1, 1, 1, 1], [1, 0, 0, 1]),
    ] {
        assert!(matches!(
            MetricTensor::new("bad", 2, ints(&matrix), ints(&inverse)),
            Err(TensorError::InvalidMetric(_))
        ));
    }
}

#[test]
fn public_storage_mutations_cannot_bypass_metric_or_shape_checks() {
    let t = tensor(&[1, 2], 1);
    let mut g = metric();
    g.inverse[0] = Expr::from_i64(1);
    assert!(g.lower_vector(&t).is_err());
    let mut abstract_tensor = TensorExpr::new("A", vec![TensorIndex::upper("a")]);
    abstract_tensor.dimension = 2;
    assert!(g.lower_vector(&abstract_tensor).is_err());
    g.matrix.pop();
    assert!(g.lower_index(&t, 0, "U").is_err());
    let g = metric();
    let mut truncated = t.clone();
    truncated.components.as_mut().unwrap().pop();
    assert!(g.lower_vector(&truncated).is_err());
    assert!(g.inner_product(&t, &truncated).is_err());
    assert!(g.inner_product(&t, &tensor(&[1, 2, 3, 4], 2)).is_err());
    let lower = g.lower_index(&t, 0, "L").unwrap();
    assert!(g.inner_product(&t, &lower).is_err());
}

#[test]
fn unsupported_original_entries_are_rejected_even_under_zero_coefficients() {
    let g = MetricTensor::euclidean("e", 2).unwrap();
    let mut t = tensor(&[0, 0], 1);
    t.components.as_mut().unwrap()[1] = Expr::Pow(
        std::sync::Arc::new(Expr::from_i64(0)),
        std::sync::Arc::new(Expr::from_i64(-1)),
    );
    assert_eq!(
        g.lower_index(&t, 0, "L"),
        Err(TensorError::UnsupportedMetricDomain)
    );
    t.components.as_mut().unwrap()[1] = Expr::Sym(Symbol::new("x"));
    assert_eq!(
        g.lower_index(&t, 0, "L"),
        Err(TensorError::UnsupportedMetricDomain)
    );
    let abstract_tensor = TensorExpr::new("A", vec![TensorIndex::upper("i")]);
    let mut abstract_tensor = abstract_tensor;
    abstract_tensor.dimension = 2;
    assert_eq!(
        g.lower_index(&abstract_tensor, 0, "L"),
        Err(TensorError::ConcreteComponentsRequired)
    );
}

#[test]
fn empty_dimensions_wrong_variances_and_resource_limits_are_explicit() {
    let g = MetricTensor::euclidean("e", 0).unwrap();
    let t = TensorExpr::with_components("T", 0, vec![TensorIndex::upper("a")], vec![]).unwrap();
    assert_eq!(g.lower_index(&t, 0, "L").unwrap().components, Some(vec![]));
    let g = metric();
    let mut t = tensor(&[1, 2], 1);
    assert!(g.lower_index(&t, 1, "L").is_err());
    assert!(g.raise_index(&t, 0, "U").is_err());
    t.components.as_mut().unwrap()[0] = Expr::Integer(BigInt::from(1) << 4096u32);
    assert!(matches!(
        g.lower_index(&t, 0, "L"),
        Err(TensorError::MetricResourceLimit(_))
    ));
    assert!(matches!(
        MetricTensor::euclidean("large", 65),
        Err(TensorError::MetricResourceLimit(_))
    ));
    let g = MetricTensor::euclidean("one", 1).unwrap();
    let t = TensorExpr::with_components(
        "T",
        1,
        (0..33)
            .map(|i| TensorIndex::upper(format!("a{i}")))
            .collect(),
        ints(&[1]),
    )
    .unwrap();
    assert!(matches!(
        g.lower_index(&t, 0, "L"),
        Err(TensorError::MetricResourceLimit(_))
    ));
}
