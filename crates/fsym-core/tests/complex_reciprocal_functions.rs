//! Complex reciprocals retain domains, complete rectangles and caller budgets.
//!
//! Frozen 40-digit values below were separated by exact rational Taylor
//! bounds (120 terms for exp, sine and cosine), followed by rational complex
//! division. Both reference endpoints select the same half-even decimal bin.
//! No binary64 value or production elementary kernel supplies the oracle.

use fsym_budget::{Budget, BudgetLimits, BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{
    BallError, BigInt, BigRational, ComplexBall, ComplexDecimalError, Constant, Expr,
    MeteredComplexDecimalError, RealBall,
};
use std::sync::Arc;

const NAMES: [&str; 6] = ["cot", "sec", "csc", "coth", "sech", "csch"];

const DECIMALS: [(&str, i64, i64, &str, &str); 18] = [
    (
        "cot",
        1,
        1,
        "2.176215618544026813651342436052380735208e-1",
        "-8.680141428959249486358492089162738882734e-1",
    ),
    (
        "sec",
        1,
        1,
        "4.983370305551867852138058917721695344329e-1",
        "5.910838417210450480503916929743350715037e-1",
    ),
    (
        "csc",
        1,
        1,
        "6.215180171704284212349078058559201481675e-1",
        "-3.039310016284264503344856045097032734887e-1",
    ),
    (
        "coth",
        1,
        1,
        "8.680141428959249486358492089162738882734e-1",
        "-2.176215618544026813651342436052380735208e-1",
    ),
    (
        "sech",
        1,
        1,
        "4.983370305551867852138058917721695344329e-1",
        "-5.910838417210450480503916929743350715037e-1",
    ),
    (
        "csch",
        1,
        1,
        "3.039310016284264503344856045097032734887e-1",
        "-6.215180171704284212349078058559201481675e-1",
    ),
    (
        "cot",
        1,
        0,
        "6.420926159343307030064199865942656202303e-1",
        "0",
    ),
    (
        "sec",
        1,
        0,
        "1.850815717680925617911753241398650193470e0",
        "0",
    ),
    (
        "csc",
        1,
        0,
        "1.188395105778121216261599452374551003528e0",
        "0",
    ),
    (
        "coth",
        1,
        0,
        "1.313035285499331303636161246930847832912e0",
        "0",
    ),
    (
        "sech",
        1,
        0,
        "6.480542736638853995749773532261503231085e-1",
        "0",
    ),
    (
        "csch",
        1,
        0,
        "8.509181282393215451338427632871752841817e-1",
        "0",
    ),
    (
        "cot",
        0,
        1,
        "0",
        "-1.313035285499331303636161246930847832912e0",
    ),
    (
        "sec",
        0,
        1,
        "6.480542736638853995749773532261503231085e-1",
        "0",
    ),
    (
        "csc",
        0,
        1,
        "0",
        "-8.509181282393215451338427632871752841817e-1",
    ),
    (
        "coth",
        0,
        1,
        "0",
        "-6.420926159343307030064199865942656202303e-1",
    ),
    (
        "sech",
        0,
        1,
        "1.850815717680925617911753241398650193470e0",
        "0",
    ),
    (
        "csch",
        0,
        1,
        "0",
        "-1.188395105778121216261599452374551003528e0",
    ),
];

fn q(numerator: i64, denominator: i64) -> BigRational {
    BigRational::new(numerator.into(), denominator.into())
}

fn interval(lower: BigRational, upper: BigRational) -> RealBall {
    let two = q(2, 1);
    RealBall::new((&lower + &upper) / &two, (upper - lower) / two).unwrap()
}

fn function(name: &str, argument: Expr) -> Expr {
    Expr::Function(name.into(), vec![argument])
}

fn complex(real: i64, imaginary: i64) -> Expr {
    Expr::Add(vec![
        Expr::from_i64(real),
        Expr::Mul(vec![Expr::from_i64(imaginary), Expr::Const(Constant::I)]),
    ])
}

fn kernel(name: &str, source: &ComplexBall, digits: u32) -> Result<ComplexBall, BallError> {
    match name {
        "cot" => source.cot(digits),
        "sec" => source.sec(digits),
        "csc" => source.csc(digits),
        "coth" => source.coth(digits),
        "sech" => source.sech(digits),
        "csch" => source.csch(digits),
        _ => unreachable!("test kernel roster"),
    }
}

fn decimal_rational(value: &str) -> BigRational {
    if value == "0" {
        return q(0, 1);
    }
    let (coefficient, exponent) = value.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    let fractional = coefficient.split_once('.').unwrap().1.len() as i32;
    let integer: BigInt = coefficient.replace('.', "").parse().unwrap();
    let scale = exponent - fractional;
    let power = BigInt::from(10).pow(scale.unsigned_abs());
    if scale >= 0 {
        BigRational::from_integer(integer * power)
    } else {
        BigRational::new(integer, power)
    }
}

fn reference_component(value: &str) -> RealBall {
    let midpoint = decimal_rational(value);
    if value == "0" {
        return RealBall::exact(midpoint);
    }
    // All nonzero fixtures have decimal exponents -1 or 0. This encloses
    // their entire 40-digit rounding bins with room to spare.
    RealBall::new(
        midpoint,
        BigRational::new(1.into(), BigInt::from(10).pow(38)),
    )
    .unwrap()
}

fn assert_pair(source: &Expr, real: &str, imaginary: &str) {
    let expected = (real.to_string(), imaginary.to_string());
    assert_eq!(
        source.evalf_complex_decimal(40).unwrap(),
        expected,
        "{source}"
    );
    assert_eq!(
        source
            .evalf_complex_decimal_metered(40, &mut Unbounded)
            .unwrap(),
        expected,
        "{source}"
    );
}

#[test]
fn all_six_functions_round_independent_complex_and_axis_references() {
    for (name, x, y, real, imaginary) in DECIMALS {
        assert_pair(&function(name, complex(x, y)), real, imaginary);
    }
}

#[test]
fn complete_rectangles_retain_both_input_coordinates() {
    let inputs = [
        // Contains i and 1+i: variation in the real input must survive.
        ComplexBall::new(
            interval(q(0, 1), q(1, 1)),
            interval(q(99, 100), q(101, 100)),
        ),
        // Contains 1 and 1+i: variation in the imaginary input must survive.
        ComplexBall::new(
            interval(q(99, 100), q(101, 100)),
            interval(q(0, 1), q(1, 1)),
        ),
    ];
    for (index, input) in inputs.iter().enumerate() {
        for name in NAMES {
            let image = kernel(name, input, 20).unwrap();
            for (reference_name, x, y, real, imaginary) in DECIMALS {
                if reference_name != name || (index == 0 && y == 0) || (index == 1 && x == 0) {
                    continue;
                }
                assert!(
                    image.real().contains_ball(&reference_component(real)),
                    "{name} lost real image of ({x},{y}) for input {index}"
                );
                assert!(
                    image.imag().contains_ball(&reference_component(imaginary)),
                    "{name} lost imaginary image of ({x},{y}) for input {index}"
                );
            }
        }
    }
}

#[test]
fn axis_identities_have_exact_zero_components_without_numerical_chopping() {
    let log_two = function("log", Expr::from_i64(2));
    let imaginary_log_two = Expr::Mul(vec![Expr::Const(Constant::I), log_two.clone()]);
    // exp(log(2))=2 gives sinh(log(2))=3/4 and cosh(log(2))=5/4.
    for (name, imaginary_argument, numerator, denominator, imaginary_result) in [
        ("cot", true, -5, 3, true),
        ("sec", true, 4, 5, false),
        ("csc", true, -4, 3, true),
        ("coth", false, 5, 3, false),
        ("sech", false, 4, 5, false),
        ("csch", false, 4, 3, false),
    ] {
        let argument = if imaginary_argument {
            imaginary_log_two.clone()
        } else {
            log_two.clone()
        };
        let expected = Expr::Rational(q(numerator, denominator))
            .evalf_complex_decimal(40)
            .unwrap()
            .0;
        let source = function(name, argument);
        if imaginary_result {
            assert_pair(&source, "0", &expected);
        } else {
            assert_pair(&source, &expected, "0");
        }
    }
    for name in ["sec", "sech"] {
        assert_pair(
            &function(name, Expr::from_i64(0)),
            &format!("1.{}e0", "0".repeat(39)),
            "0",
        );
    }
}

#[test]
fn real_axis_quotients_retain_high_precision_admission() {
    for name in ["sec", "sech"] {
        let image = kernel(name, &ComplexBall::from_i64(1), 700).unwrap();
        let (_, _, _, reference, _) = DECIMALS
            .iter()
            .find(|entry| entry.0 == name && entry.1 == 1 && entry.2 == 0)
            .unwrap();
        assert!(reference_component(reference).contains_ball(image.real()));
        assert_eq!(image.imag(), &RealBall::from_i64(0));
        assert!(image.real().width() < BigRational::new(1.into(), BigInt::from(10).pow(650)));
    }
}

#[test]
fn complete_denominator_intervals_refuse_poles_away_from_the_midpoint() {
    for name in NAMES {
        let input = match name {
            "cot" | "csc" => ComplexBall::from_real(interval(q(-1, 4), q(3, 4))),
            "sec" => ComplexBall::from_real(interval(q(1, 1), q(2, 1))),
            "sech" => ComplexBall::new(RealBall::from_i64(0), interval(q(1, 1), q(2, 1))),
            _ => ComplexBall::new(RealBall::from_i64(0), interval(q(3, 1), q(4, 1))),
        };
        assert!(
            matches!(kernel(name, &input, 25), Err(BallError::DivisionByZero(_))),
            "{name}"
        );
    }
    for name in ["cot", "csc", "coth", "csch"] {
        let source = function(name, Expr::from_i64(0));
        assert!(matches!(
            source.evalf_complex_decimal(30),
            Err(ComplexDecimalError::Enclosure(BallError::DivisionByZero(_)))
        ));
        assert!(matches!(
            source.evalf_complex_decimal_metered(30, &mut Unbounded),
            Err(MeteredComplexDecimalError::Ball(BallError::DivisionByZero(
                _
            )))
        ));
    }
}

#[test]
fn original_invalid_children_and_invalid_precisions_cannot_be_hidden() {
    let invalid = function("log", Expr::from_i64(0));
    for name in NAMES {
        for argument in [
            Expr::Mul(vec![Expr::from_i64(0), invalid.clone()]),
            Expr::Pow(Arc::new(invalid.clone()), Arc::new(Expr::from_i64(0))),
            Expr::symbol("unbound"),
            Expr::Const(Constant::NaN),
        ] {
            let source = function(name, argument);
            assert!(source.evalf_complex_decimal(30).is_err());
            assert!(
                source
                    .evalf_complex_decimal_metered(30, &mut Unbounded)
                    .is_err()
            );
        }
        for arguments in [vec![], vec![Expr::from_i64(1), Expr::from_i64(2)]] {
            let source = Expr::Function(name.into(), arguments);
            assert!(matches!(
                source.evalf_complex_decimal(30),
                Err(ComplexDecimalError::UnsupportedSource(_))
            ));
            assert!(matches!(
                source.evalf_complex_decimal_metered(30, &mut Unbounded),
                Err(MeteredComplexDecimalError::Unsupported(_))
            ));
        }
        for digits in [
            0,
            ComplexBall::MAX_ELEMENTARY_PRECISION_DIGITS + 1,
            u32::MAX,
        ] {
            assert!(matches!(
                kernel(name, &ComplexBall::i(), digits),
                Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
            ));
        }
    }
}

struct ObservedMeter {
    budget: Budget,
    checkpoints: usize,
    cancel_at: Option<usize>,
}

impl ObservedMeter {
    fn new(cancel_at: Option<usize>) -> Self {
        Self {
            budget: Budget::new(BudgetLimits::uniform(1 << 45, 0)),
            checkpoints: 0,
            cancel_at,
        }
    }
}

impl BudgetMeter for ObservedMeter {
    fn charge(&mut self, dimension: Dimension, amount: u64) -> Result<(), MeterError> {
        BudgetMeter::charge(&mut self.budget, dimension, amount)
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        BudgetMeter::charge_batch(&mut self.budget, charges)
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        self.checkpoints += 1;
        if self.cancel_at.is_some_and(|stop| self.checkpoints >= stop) {
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn caller_cancellation_is_terminal_at_admission_work_and_pair_publication() {
    for name in NAMES {
        let source = function(name, complex(1, 1));
        let mut baseline = ObservedMeter::new(None);
        source
            .evalf_complex_decimal_metered(30, &mut baseline)
            .unwrap();
        for stop in [1, baseline.checkpoints / 2, baseline.checkpoints] {
            let mut meter = ObservedMeter::new(Some(stop));
            assert_eq!(
                source.evalf_complex_decimal_metered(30, &mut meter),
                Err(MeteredComplexDecimalError::Meter(MeterError::Cancelled)),
                "{name}, checkpoint {stop}"
            );
        }
    }
}

#[test]
fn caller_resource_exhaustion_and_reuse_keep_one_cumulative_ledger() {
    for name in NAMES {
        let source = function(name, complex(1, 1));
        for dimension in [
            Dimension::ComputeSteps,
            Dimension::MemoryBytes,
            Dimension::AllocationCount,
        ] {
            let mut limits = BudgetLimits::uniform(1 << 45, 0);
            limits.dimensions[dimension.index()] = 0;
            let mut budget = Budget::new(limits);
            assert!(matches!(
                source.evalf_complex_decimal_metered(30, &mut budget),
                Err(MeteredComplexDecimalError::Meter(MeterError::Budget(_)))
            ));
        }
        let mut budget = Budget::new(BudgetLimits::uniform(1 << 45, 0));
        let first = source
            .evalf_complex_decimal_metered(30, &mut budget)
            .unwrap();
        let remaining = budget.remaining(Dimension::ComputeSteps);
        assert_eq!(
            source
                .evalf_complex_decimal_metered(30, &mut budget)
                .unwrap(),
            first
        );
        assert!(budget.remaining(Dimension::ComputeSteps) < remaining);
        let mut small = Budget::new(BudgetLimits::uniform(64, 0));
        let before = small.remaining(Dimension::ComputeSteps);
        assert!(
            source
                .evalf_complex_decimal_metered(30, &mut small)
                .is_err()
        );
        let after = small.remaining(Dimension::ComputeSteps);
        assert!(after < before);
        let _ = source.evalf_complex_decimal_metered(30, &mut small);
        assert!(small.remaining(Dimension::ComputeSteps) <= after);
    }
}
