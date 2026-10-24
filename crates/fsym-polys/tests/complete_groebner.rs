use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{BigRational, Symbol};
use fsym_polys::groebner::{
    CompleteGroebnerBasisCertificate, groebner_basis_with_complete_certificate,
    verify_complete_groebner_certificate,
};
use fsym_polys::multivariate::{MultivariatePoly, TermOrder};
use std::collections::BTreeMap;

fn poly(g: &[Symbol], terms: &[(Vec<u32>, i64, i64)]) -> MultivariatePoly {
    MultivariatePoly::new(
        g.to_vec(),
        terms
            .iter()
            .map(|(p, n, d)| (p.clone(), BigRational::new((*n).into(), (*d).into())))
            .collect(),
    )
    .unwrap()
}
fn system() -> (Vec<Symbol>, Vec<MultivariatePoly>) {
    let g = vec![Symbol::new("x"), Symbol::new("y")];
    let input = vec![
        poly(&g, &[(vec![1, 0], 3, 2), (vec![0, 1], -1, 1)]),
        poly(&g, &[(vec![0, 2], 1, 2), (vec![0, 0], -2, 1)]),
    ];
    (g, input)
}
#[test]
fn rational_nonmonic_inputs_are_independently_certified_in_all_orders() {
    let (g, input) = system();
    let expected = [
        poly(&g, &[(vec![1, 0], 1, 1), (vec![0, 1], -2, 3)]),
        poly(&g, &[(vec![0, 2], 1, 1), (vec![0, 0], -4, 1)]),
    ];
    for order in [TermOrder::Lex, TermOrder::DegLex, TermOrder::DegRevLex] {
        let cert = groebner_basis_with_complete_certificate(&input, order).unwrap();
        let decoded: CompleteGroebnerBasisCertificate =
            serde_json::from_slice(&serde_json::to_vec(&cert).unwrap()).unwrap();
        assert_eq!(cert, decoded);
        let verified =
            verify_complete_groebner_certificate(&input, &decoded, &mut Unbounded).unwrap();
        assert_eq!(verified.order(), order);
        assert_eq!(verified.input_basis(), input);
        assert_eq!(verified.basis().len(), 2);
        for polynomial in &expected {
            assert!(verified.basis().contains(polynomial));
        }
        assert_eq!(verified.receipt().s_pairs_verified(), 1);
    }
}
#[test]
fn verified_artifact_keeps_exact_binding_after_untrusted_data_is_edited() {
    let (g, mut input) = system();
    let original = input.clone();
    let mut cert = groebner_basis_with_complete_certificate(&input, TermOrder::Lex).unwrap();
    let original_cert = cert.clone();
    let verified = verify_complete_groebner_certificate(&input, &cert, &mut Unbounded).unwrap();
    let basis = verified.basis().to_vec();
    cert.candidate.basis[0] = MultivariatePoly::one(g.clone());
    input[0] = MultivariatePoly::one(g);
    assert_eq!(verified.input_basis(), original);
    assert_eq!(verified.basis(), basis);
    assert!(verify_complete_groebner_certificate(&input, &original_cert, &mut Unbounded).is_err());
    assert!(verify_complete_groebner_certificate(&original, &cert, &mut Unbounded).is_err());
}
#[test]
fn verified_lex_basis_proves_the_complete_leading_block_elimination_ideal() {
    let g = vec![Symbol::new("y"), Symbol::new("x")];
    let input = vec![
        poly(&g, &[(vec![1, 0], 1, 1), (vec![0, 1], -1, 1)]),
        poly(&g, &[(vec![2, 0], 1, 1), (vec![0, 0], -1, 1)]),
    ];
    let cert = groebner_basis_with_complete_certificate(&input, TermOrder::Lex).unwrap();
    let verified = verify_complete_groebner_certificate(&input, &cert, &mut Unbounded).unwrap();
    assert_eq!(
        verified.elimination_basis(&g[..1]).unwrap(),
        vec![poly(&g, &[(vec![0, 2], 1, 1), (vec![0, 0], -1, 1)])]
    );
    assert_eq!(verified.elimination_basis(&[]).unwrap(), verified.basis());
    assert!(verified.elimination_basis(&g).unwrap().is_empty());
    assert!(verified.elimination_basis(&g[1..]).is_err());
    let other = groebner_basis_with_complete_certificate(&input, TermOrder::DegLex).unwrap();
    assert!(
        verify_complete_groebner_certificate(&input, &other, &mut Unbounded)
            .unwrap()
            .elimination_basis(&g[..1])
            .is_err()
    );
}
#[test]
fn zero_and_unit_ideals_keep_exact_complete_certificate_semantics() {
    let g = vec![Symbol::new("x")];
    for input in [vec![], vec![MultivariatePoly::zero(g.clone())]] {
        let cert = groebner_basis_with_complete_certificate(&input, TermOrder::Lex).unwrap();
        let verified = verify_complete_groebner_certificate(&input, &cert, &mut Unbounded).unwrap();
        assert!(verified.basis().is_empty());
        assert_eq!(verified.receipt().s_pairs_verified(), 0);
    }
    let input = vec![poly(&g, &[(vec![0], 2, 1)])];
    let cert = groebner_basis_with_complete_certificate(&input, TermOrder::Lex).unwrap();
    let verified = verify_complete_groebner_certificate(&input, &cert, &mut Unbounded).unwrap();
    assert_eq!(verified.basis(), &[MultivariatePoly::one(g.clone())]);
    assert_eq!(verified.elimination_basis(&g).unwrap(), verified.basis());
}
#[test]
fn malformed_ring_matrices_and_s_pairs_are_refused_without_panics() {
    let (_, input) = system();
    let cert = groebner_basis_with_complete_certificate(&input, TermOrder::Lex).unwrap();
    let mut bad = cert.clone();
    bad.candidate.input_ideal_witnesses[0][0].generators[0] = Symbol::new("foreign");
    assert!(verify_complete_groebner_certificate(&input, &bad, &mut Unbounded).is_err());
    let mut bad = cert.clone();
    bad.input_basis_witnesses[0].pop();
    assert!(verify_complete_groebner_certificate(&input, &bad, &mut Unbounded).is_err());
    let mut bad = cert.clone();
    bad.s_pairs[0].left = usize::MAX;
    assert!(verify_complete_groebner_certificate(&input, &bad, &mut Unbounded).is_err());
    let mut bad = cert;
    bad.candidate.basis[0].terms = BTreeMap::from([(vec![1], BigRational::from_integer(1.into()))]);
    assert!(verify_complete_groebner_certificate(&input, &bad, &mut Unbounded).is_err());
}
#[derive(Default)]
struct Checkpoints {
    seen: usize,
    cancel_at: Option<usize>,
}
impl BudgetMeter for Checkpoints {
    fn charge(&mut self, _: Dimension, _: u64) -> Result<(), MeterError> {
        Ok(())
    }
    fn charge_batch(&mut self, _: &[(Dimension, u64)]) -> Result<(), MeterError> {
        Ok(())
    }
    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.seen += 1;
        if self.cancel_at == Some(self.seen) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}
#[test]
fn resource_refusal_and_final_checkpoint_cancellation_do_not_publish_a_basis() {
    let (_, input) = system();
    let cert = groebner_basis_with_complete_certificate(&input, TermOrder::Lex).unwrap();
    let mut count = Checkpoints::default();
    verify_complete_groebner_certificate(&input, &cert, &mut count).unwrap();
    let mut cancel = Checkpoints {
        seen: 0,
        cancel_at: Some(count.seen),
    };
    assert!(verify_complete_groebner_certificate(&input, &cert, &mut cancel).is_err());
    for dimension in [
        Dimension::ComputeSteps,
        Dimension::MemoryBytes,
        Dimension::AllocationCount,
    ] {
        let mut limits = BudgetLimits::uniform(1_000_000, 0);
        limits.dimensions[dimension.index()] = 0;
        assert!(
            verify_complete_groebner_certificate(&input, &cert, &mut Budget::new(limits)).is_err()
        );
    }
}

#[test]
fn finite_budgets_accept_empty_zero_and_zero_variable_unit_ideals() {
    for input in [
        vec![],
        vec![MultivariatePoly::zero(vec![Symbol::new("x")])],
        vec![poly(&[], &[(vec![], 2, 1)])],
    ] {
        let cert = groebner_basis_with_complete_certificate(&input, TermOrder::Lex).unwrap();
        let mut budget = Budget::new(BudgetLimits::uniform(1_000_000, 0));
        let verified = verify_complete_groebner_certificate(&input, &cert, &mut budget).unwrap();
        assert_eq!(verified.input_basis(), input);
        assert_eq!(verified.receipt().s_pairs_verified(), 0);
    }
}
