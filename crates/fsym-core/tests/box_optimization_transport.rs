//! Bounded candidate decoding never grants a global optimization receipt.

use fsym_budget::Unbounded;
use fsym_core::{
    BigRational, BoxExtremumKind, BoxOptimizationError, BoxOptimizationWitness, BoxPartitionStep,
    Expr, MAX_BOX_OPTIMIZATION_NODES, MAX_ROOT_BOX_LIMBS, MAX_ROOT_BOX_VARIABLES, RealBall, Symbol,
    verify_box_optimization,
};
use serde::de::{DeserializeSeed, IntoDeserializer, SeqAccess, Visitor, value::Error};
use serde::{Deserialize, Deserializer};
use serde_json::{Value, json};

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
}

fn candidate() -> BoxOptimizationWitness {
    BoxOptimizationWitness {
        partition: vec![
            BoxPartitionStep::Split { axis: 0 },
            BoxPartitionStep::Leaf,
            BoxPartitionStep::Leaf,
        ],
        point: vec![q(-1, 1)],
        precision_digits: 24,
    }
}

fn check(witness: &BoxOptimizationWitness) -> Result<(), BoxOptimizationError> {
    let x = Symbol::new("x");
    verify_box_optimization(
        &Expr::Sym(x.clone()),
        &[x],
        &[RealBall::new(q(0, 1), q(1, 1)).unwrap()],
        BoxExtremumKind::Minimum,
        &q(1, 100),
        witness,
        &mut Unbounded,
    )
    .map(|_| ())
}

#[test]
fn exact_round_trip_keeps_only_a_candidate_until_independent_verification() {
    let witness = candidate();
    let encoded = serde_json::to_string(&witness).unwrap();
    assert!(encoded.starts_with("{\"partition\":[{\"split\":{\"axis\":0}},\"leaf\",\"leaf\"]"));
    let decoded: BoxOptimizationWitness = serde_json::from_str(&encoded).unwrap();
    assert_eq!(decoded, witness);
    assert_eq!(serde_json::to_string(&decoded).unwrap(), encoded);
    assert_eq!(check(&decoded), Ok(()));

    let mut missing_child = decoded;
    missing_child.partition.pop();
    let bytes = serde_json::to_vec(&missing_child).unwrap();
    let decoded: BoxOptimizationWitness = serde_json::from_slice(&bytes).unwrap();
    assert!(matches!(
        check(&decoded),
        Err(BoxOptimizationError::InvalidPartition(_))
    ));

    let mut poor_point = candidate();
    poor_point.point[0] = q(1, 1);
    let bytes = serde_json::to_vec(&poor_point).unwrap();
    let decoded: BoxOptimizationWitness = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        check(&decoded),
        Err(BoxOptimizationError::AccuracyNotReached)
    );
}

#[test]
fn extrema_kind_and_partition_forms_are_explicit_and_closed() {
    for (kind, expected) in [
        (BoxExtremumKind::Minimum, "\"minimum\""),
        (BoxExtremumKind::Maximum, "\"maximum\""),
    ] {
        assert_eq!(serde_json::to_string(&kind).unwrap(), expected);
        assert_eq!(
            serde_json::from_str::<BoxExtremumKind>(expected).unwrap(),
            kind
        );
    }
    for bad in ["\"Minimum\"", "\"min\"", "\"stationary\"", "1", "null"] {
        assert!(serde_json::from_str::<BoxExtremumKind>(bad).is_err());
    }
    for bad in [
        "{\"split\":{\"axis\":0,\"accepted\":true}}",
        "{\"split\":{\"axis\":0,\"axis\":1}}",
        "{\"split\":{\"axis\":-1}}",
        "{\"split\":{}}",
        "\"discard\"",
        "{\"leaf\":{\"bound\":0}}",
    ] {
        assert!(
            serde_json::from_str::<BoxPartitionStep>(bad).is_err(),
            "{bad}"
        );
    }
}

#[test]
fn witness_rejects_unknown_duplicate_fields_and_invalid_precision() {
    let encoded = serde_json::to_string(&candidate()).unwrap();
    for suffix in [",\"accepted\":true}", ",\"precision_digits\":24}"] {
        let altered = format!("{}{suffix}", &encoded[..encoded.len() - 1]);
        assert!(serde_json::from_str::<BoxOptimizationWitness>(&altered).is_err());
    }
    for digits in [0, 7, 129, u32::MAX] {
        let mut wire = serde_json::to_value(candidate()).unwrap();
        wire["precision_digits"] = json!(digits);
        assert!(serde_json::from_value::<BoxOptimizationWitness>(wire).is_err());
    }
    for digits in [8, 128] {
        let mut wire = serde_json::to_value(candidate()).unwrap();
        wire["precision_digits"] = json!(digits);
        assert!(serde_json::from_value::<BoxOptimizationWitness>(wire).is_ok());
    }
}

#[test]
fn bounded_sequences_reject_empty_or_excess_nodes_and_coordinates() {
    for (field, replacement) in [
        ("partition", json!([])),
        ("point", json!([])),
        (
            "partition",
            json!(vec!["leaf"; MAX_BOX_OPTIMIZATION_NODES + 1]),
        ),
        ("point", json!(vec![q(0, 1); MAX_ROOT_BOX_VARIABLES + 1])),
    ] {
        let mut wire = serde_json::to_value(candidate()).unwrap();
        wire[field] = replacement;
        assert!(serde_json::from_value::<BoxOptimizationWitness>(wire).is_err());
    }
    let mut wire = serde_json::to_value(candidate()).unwrap();
    wire["partition"] = json!(vec!["leaf"; MAX_BOX_OPTIMIZATION_NODES]);
    wire["point"] = json!(vec![q(0, 1); MAX_ROOT_BOX_VARIABLES]);
    // Maximum bounded shape is a candidate, not a valid coverage proof.
    assert!(serde_json::from_value::<BoxOptimizationWitness>(wire).is_ok());
}

#[test]
fn bounded_rationals_reject_zero_denominators_noncanonical_forms_and_limb_excess() {
    let too_many_limbs = vec![1u32; MAX_ROOT_BOX_LIMBS as usize * 2 + 1];
    let malformed = [
        json!([[1, [1]], [0, []]]),
        json!([[1, [1]], [-1, [2]]]),
        json!([[1, [2]], [1, [4]]]),
        json!([[0, [1]], [1, [1]]]),
        json!([[1, []], [1, [1]]]),
        json!([[1, [1, 0]], [1, [1]]]),
        json!([[2, [1]], [1, [1]]]),
        json!([[1, too_many_limbs], [1, [1]]]),
    ];
    for number in malformed {
        let mut wire = serde_json::to_value(candidate()).unwrap();
        wire["point"] = Value::Array(vec![number]);
        assert!(serde_json::from_value::<BoxOptimizationWitness>(wire).is_err());
    }
}

#[test]
fn fixed_coordinates_and_full_dimension_survive_transport_without_new_domain_rules() {
    let vars: Vec<_> = (0..MAX_ROOT_BOX_VARIABLES)
        .map(|i| Symbol::new(format!("x{i}")))
        .collect();
    let witness = BoxOptimizationWitness {
        partition: vec![BoxPartitionStep::Leaf],
        point: vec![q(2, 1); vars.len()],
        precision_digits: 24,
    };
    let bytes = serde_json::to_vec(&witness).unwrap();
    let decoded: BoxOptimizationWitness = serde_json::from_slice(&bytes).unwrap();
    let receipt = verify_box_optimization(
        &Expr::from_i64(7),
        &vars,
        &vec![RealBall::exact(q(2, 1)); vars.len()],
        BoxExtremumKind::Maximum,
        &q(1, 100),
        &decoded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(receipt.value(), &RealBall::from_i64(7));
    assert_eq!(receipt.gap(), &q(0, 1));
}

struct OversizedSequence;

impl<'de> IntoDeserializer<'de, Error> for OversizedSequence {
    type Deserializer = Self;

    fn into_deserializer(self) -> Self {
        self
    }
}

impl<'de> Deserializer<'de> for OversizedSequence {
    type Error = Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Error> {
        visitor.visit_seq(self)
    }

    serde::forward_to_deserialize_any! {
        bool i8 i16 i32 i64 u8 u16 u32 u64 f32 f64 char str string bytes
        byte_buf option unit unit_struct newtype_struct seq tuple tuple_struct
        map struct enum identifier ignored_any
    }
}

impl<'de> SeqAccess<'de> for OversizedSequence {
    type Error = Error;

    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        _: T,
    ) -> Result<Option<T::Value>, Error> {
        panic!("oversized sequence must be rejected before reading or allocating an element")
    }

    fn size_hint(&self) -> Option<usize> {
        Some(usize::MAX)
    }
}

#[test]
fn hostile_sequence_size_hints_refuse_before_decoding_an_element() {
    for key in ["partition", "point"] {
        let input = serde::de::value::MapDeserializer::new([(key, OversizedSequence)].into_iter());
        assert!(BoxOptimizationWitness::deserialize(input).is_err());
    }
}
