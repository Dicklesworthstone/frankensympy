//! Certified real ball arithmetic (WS11).
//!
//! A real ball $\mathcal{B}(m, r) = [m - r, m + r]$ represents a certified enclosure
//! of a real number with rational midpoint $m \in \mathbb{Q}$ and non-negative rational radius $r \in \mathbb{Q}_{\ge 0}$.

#![forbid(unsafe_code)]

use crate::{BigInt, BigRational};
use num_traits::{One, Signed, Zero};
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BallError {
    #[error("Division by ball containing zero: {0}")]
    DivisionByZero(String),
    #[error("Negative radius is invalid: {0}")]
    NegativeRadius(String),
    #[error("Indeterminate integer power: {0}")]
    IndeterminatePower(String),
    #[error("Square root is not real over the complete ball: {0}")]
    NonRealSquareRoot(String),
    #[error(
        "{operation} requires an estimated {required_bits} numeric work bits, exceeding the supported limit of {limit_bits}"
    )]
    ResourceLimitExceeded {
        operation: &'static str,
        required_bits: u64,
        limit_bits: u64,
    },
    #[error(
        "{0} lies outside the declared certified-ball envelope (argument magnitude, \
         result representability, or requested precision bound exceeded)"
    )]
    ArgumentOutsideDeclaredEnvelope(String),
    #[error("Canonical RealBall encoding failed: {0}")]
    CanonicalEncoding(String),
}

/// Maximum estimated numeric width admitted by a single power or square-root operation.
///
/// This matches 16,384 64-bit charging limbs. It is a fixed fail-closed guard for this
/// convenience API, not a substitute for the metered evaluator required by WS03/WS11.
pub const MAX_REAL_BALL_WORK_BITS: u64 = 16_384 * 64;

/// Certified real ball $\mathcal{B}(m, r) = [m - r, m + r]$.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct RealBall {
    midpoint: BigRational,
    radius: BigRational,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RealBallWire {
    midpoint: BigRational,
    radius: BigRational,
}

impl<'de> Deserialize<'de> for RealBall {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RealBallWire::deserialize(deserializer)?;
        Self::new(wire.midpoint, wire.radius).map_err(serde::de::Error::custom)
    }
}

impl RealBall {
    /// Constructs a certified ball with given midpoint and non-negative radius.
    pub fn new(midpoint: BigRational, radius: BigRational) -> Result<Self, BallError> {
        if radius < BigRational::zero() {
            return Err(BallError::NegativeRadius(radius.to_string()));
        }
        Ok(Self { midpoint, radius })
    }

    /// Construct an exact point ball with zero radius: $\mathcal{B}(q, 0) = [q, q]$.
    pub fn exact(q: BigRational) -> Self {
        Self {
            midpoint: q,
            radius: BigRational::zero(),
        }
    }

    /// Construct an exact integer ball: $\mathcal{B}(n, 0)$.
    pub fn from_i64(n: i64) -> Self {
        Self::exact(BigRational::from_integer(BigInt::from(n)))
    }

    /// Lower bound $m - r$.
    pub fn lower(&self) -> BigRational {
        &self.midpoint - &self.radius
    }

    /// Upper bound $m + r$.
    pub fn upper(&self) -> BigRational {
        &self.midpoint + &self.radius
    }

    /// Midpoint $m$.
    pub fn midpoint(&self) -> &BigRational {
        &self.midpoint
    }

    /// Radius $r$.
    pub fn radius(&self) -> &BigRational {
        &self.radius
    }

    /// Width / diameter $2r$.
    pub fn width(&self) -> BigRational {
        self.diameter()
    }

    /// Checks if this ball is completely disjoint from another ball.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        self.upper() < other.lower() || other.upper() < self.lower()
    }

    /// Diameter $2r$.
    pub fn diameter(&self) -> BigRational {
        &self.radius * BigRational::from_integer(BigInt::from(2))
    }

    /// Check if point $x \in \mathcal{B}(m, r)$.
    pub fn contains(&self, x: &BigRational) -> bool {
        x >= &self.lower() && x <= &self.upper()
    }

    /// Check if another ball is entirely contained within this ball: $B_2 \subseteq B_1$.
    pub fn contains_ball(&self, other: &Self) -> bool {
        other.lower() >= self.lower() && other.upper() <= self.upper()
    }

    /// Check if 0 is contained within the ball ($0 \in [m - r, m + r]$).
    pub fn contains_zero(&self) -> bool {
        let magnitude = self.midpoint.abs();
        magnitude <= self.radius
    }

    /// Strictly positive: lower bound > 0.
    pub fn is_positive(&self) -> bool {
        self.lower() > BigRational::zero()
    }

    /// Strictly negative: upper bound < 0.
    pub fn is_negative(&self) -> bool {
        self.upper() < BigRational::zero()
    }

    /// Certified ball addition: $[m_1 - r_1, m_1 + r_1] + [m_2 - r_2, m_2 + r_2] = [m_1 + m_2, r_1 + r_2]$.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            midpoint: &self.midpoint + &other.midpoint,
            radius: &self.radius + &other.radius,
        }
    }

    /// Certified ball subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            midpoint: &self.midpoint - &other.midpoint,
            radius: &self.radius + &other.radius,
        }
    }

    /// Certified ball negation.
    pub fn neg(&self) -> Self {
        Self {
            midpoint: -&self.midpoint,
            radius: self.radius.clone(),
        }
    }

    /// Certified ball multiplication:
    /// $\mathcal{B}(m_1, r_1) \cdot \mathcal{B}(m_2, r_2) = \mathcal{B}(m_1 m_2, |m_1| r_2 + |m_2| r_1 + r_1 r_2)$.
    pub fn mul(&self, other: &Self) -> Self {
        let new_midpoint = &self.midpoint * &other.midpoint;
        let abs_m1 = self.midpoint.abs();
        let abs_m2 = other.midpoint.abs();

        let new_radius =
            (&abs_m1 * &other.radius) + (&abs_m2 * &self.radius) + (&self.radius * &other.radius);

        Self {
            midpoint: new_midpoint,
            radius: new_radius,
        }
    }

    /// Certified ball inversion: $1 / \mathcal{B}(m, r)$ when $0 \notin \mathcal{B}(m, r)$.
    pub fn inv(&self) -> Result<Self, BallError> {
        if self.contains_zero() {
            return Err(BallError::DivisionByZero(self.to_string()));
        }
        let low = self.lower();
        let high = self.upper();

        let inv_low = BigRational::one() / &high;
        let inv_high = BigRational::one() / &low;

        let mid = (&inv_low + &inv_high) / BigRational::from_integer(BigInt::from(2));
        let rad = (&inv_high - &inv_low) / BigRational::from_integer(BigInt::from(2));

        Ok(Self {
            midpoint: mid,
            radius: rad.abs(),
        })
    }

    /// Certified ball division: $\mathcal{B}_1 / \mathcal{B}_2$.
    pub fn div(&self, other: &Self) -> Result<Self, BallError> {
        let inv_other = other.inv()?;
        Ok(self.mul(&inv_other))
    }

    /// Intersect two balls: returns tightest enclosing ball of the intersection if non-empty.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let low = self.lower().max(other.lower());
        let high = self.upper().min(other.upper());

        if low > high {
            None
        } else {
            let mid = (&low + &high) / BigRational::from_integer(BigInt::from(2));
            let rad = (&high - &low) / BigRational::from_integer(BigInt::from(2));
            Some(Self {
                midpoint: mid,
                radius: rad,
            })
        }
    }

    /// Certified absolute value $|\mathcal{B}|$.
    pub fn abs(&self) -> Self {
        let low = self.lower();
        let high = self.upper();
        if low >= BigRational::zero() {
            self.clone()
        } else if high <= BigRational::zero() {
            self.neg()
        } else {
            let max_val = (-low).max(high);
            let two = BigRational::from_integer(BigInt::from(2));
            let half = &max_val / &two;
            Self {
                midpoint: half.clone(),
                radius: half,
            }
        }
    }

    /// Certified integer power $\mathcal{B}^k$.
    pub fn pow(&self, exp: i32) -> Result<Self, BallError> {
        if exp == 0 {
            if self.contains_zero() {
                return Err(BallError::IndeterminatePower(
                    "zero exponent requires a base ball that excludes zero".to_string(),
                ));
            }
            return Ok(Self::from_i64(1));
        }
        if exp == 1 {
            return Ok(self.clone());
        }

        let magnitude = exp.unsigned_abs();
        if exp < 0 {
            ensure_endpoint_construction_fits(self, "RealBall negative integer power")?;
            return self.inv()?.pow_nonnegative(magnitude);
        }
        self.pow_nonnegative(magnitude)
    }

    fn pow_nonnegative(&self, exp: u32) -> Result<Self, BallError> {
        debug_assert!(exp > 0);
        let k = usize::try_from(exp).map_err(|_| BallError::ResourceLimitExceeded {
            operation: "RealBall integer power",
            required_bits: u64::MAX,
            limit_bits: MAX_REAL_BALL_WORK_BITS,
        })?;
        ensure_endpoint_construction_fits(self, "RealBall integer power")?;
        let low = self.lower();
        let high = self.upper();
        checked_work_bits(
            "RealBall integer power",
            [
                rational_power_bit_bound(&low, exp),
                rational_power_bit_bound(&high, exp),
            ],
        )?;
        let two = BigRational::from_integer(BigInt::from(2));

        if k % 2 == 1 || low >= BigRational::zero() {
            let low_k = rational_pow(&low, k);
            let high_k = rational_pow(&high, k);
            let mid = (&low_k + &high_k) / &two;
            let rad = (&high_k - &low_k) / &two;
            Ok(Self {
                midpoint: mid,
                radius: rad,
            })
        } else if high <= BigRational::zero() {
            let low_k = rational_pow(&high, k);
            let high_k = rational_pow(&low, k);
            let mid = (&low_k + &high_k) / &two;
            let rad = (&high_k - &low_k) / &two;
            Ok(Self {
                midpoint: mid,
                radius: rad,
            })
        } else {
            let max_val = (-&low).max(high);
            let high_k = rational_pow(&max_val, k);
            let half = &high_k / &two;
            Ok(Self {
                midpoint: half.clone(),
                radius: half,
            })
        }
    }

    /// Certified square root $\sqrt{\mathcal{B}}$ with specified precision bits.
    pub fn sqrt(&self, precision_bits: u32) -> Result<Self, BallError> {
        ensure_endpoint_construction_fits(self, "RealBall square root")?;
        let low = self.lower();
        let high = self.upper();
        if low < BigRational::zero() {
            return Err(BallError::NonRealSquareRoot(format!(
                "cannot compute an unconditional real square root of ball {self}"
            )));
        }
        if high.is_zero() {
            return Ok(Self::from_i64(0));
        }

        let l = rational_sqrt_bound(&low, precision_bits, false)?;
        let u = rational_sqrt_bound(&high, precision_bits, true)?;
        checked_work_bits(
            "RealBall square-root enclosure",
            [l.height().max_bits(), u.height().max_bits()],
        )?;
        let two = BigRational::from_integer(BigInt::from(2));
        let mid = (&l + &u) / &two;
        let rad = (&u - &l) / &two;
        Ok(Self {
            midpoint: mid,
            radius: rad,
        })
    }

    /// Computes the bounded canonical BLAKE3 content digest of this certified ball.
    pub fn digest(&self) -> Result<[u8; 32], BallError> {
        let midpoint = crate::canonical::canonical_rational_bytes(&self.midpoint)
            .map_err(|error| BallError::CanonicalEncoding(error.to_string()))?;
        let radius = crate::canonical::canonical_rational_bytes(&self.radius)
            .map_err(|error| BallError::CanonicalEncoding(error.to_string()))?;
        let midpoint_len = u64::try_from(midpoint.len()).map_err(|_| {
            BallError::CanonicalEncoding("midpoint length exceeds u64 framing".to_string())
        })?;
        let radius_len = u64::try_from(radius.len()).map_err(|_| {
            BallError::CanonicalEncoding("radius length exceeds u64 framing".to_string())
        })?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"fsym.real_ball.v2\0");
        hasher.update(&midpoint_len.to_le_bytes());
        hasher.update(&midpoint);
        hasher.update(&radius_len.to_le_bytes());
        hasher.update(&radius);
        Ok(*hasher.finalize().as_bytes())
    }

    /// Declared envelope for the certified transcendental subset.
    pub const MAX_TRANSCENDENTAL_PRECISION_DIGITS: u32 = 2_000;
    /// sin/cos range reduction refuses argument magnitudes beyond 10^1000:
    /// the integer k = round(x/pi) and the matching pi enclosure stay inside
    /// the declared work-bit budget at the maximum supported precision.
    pub const MAX_REDUCTION_DECIMAL_MAGNITUDE: u32 = 1_000;
    /// exp refuses arguments beyond 10^4: the enclosure midpoint of e^x
    /// would need more than ~4350 decimal digits and is not representable
    /// inside the declared envelope.
    pub const MAX_EXP_DECIMAL_MAGNITUDE: u32 = 4;

    fn decimal_magnitude_bound(&self) -> u32 {
        let upper_abs = if self.lower().abs() > self.upper().abs() {
            self.lower().abs()
        } else {
            self.upper().abs()
        };
        // Magnitude of the VALUE, not of the rational's exact height: use
        // the integer part's bit width so high-precision (large-denominator)
        // balls of modest size are not misread as huge magnitudes.
        if upper_abs < BigRational::one() {
            return 0;
        }
        let int_part = upper_abs.numer() / upper_abs.denom();
        ((int_part.bits().saturating_mul(30_103) / 100_000) as u32).saturating_add(1)
    }

    fn precision_target(precision_digits: u32) -> Result<BigRational, BallError> {
        if precision_digits == 0 || precision_digits > Self::MAX_TRANSCENDENTAL_PRECISION_DIGITS {
            return Err(BallError::ArgumentOutsideDeclaredEnvelope(format!(
                "requested precision {precision_digits} digits (supported: 1..={})",
                Self::MAX_TRANSCENDENTAL_PRECISION_DIGITS
            )));
        }
        Ok(ten_pow_rational(precision_digits + 2))
    }

    /// Certified enclosure of pi via the Machin identity
    /// pi = 16*atan(1/5) - 4*atan(1/239) with exact rational partial sums
    /// and Leibniz tail bounds. The radius is <= 10^-(precision_digits + 1)
    /// by construction.
    pub fn pi(precision_digits: u32) -> Result<Self, BallError> {
        Self::precision_target(precision_digits)?;
        // Each tail is <= 10^-(digits+3); scaling by 16 and 4 keeps the
        // combined radius <= 20 * 10^-(digits+3) <= 10^-(digits+1) for
        // digits >= 1.
        let digits = precision_digits + 2;
        let atan5 = arctan_inverse_enclosure(5, digits)?;
        let atan239 = arctan_inverse_enclosure(239, digits)?;
        let sixteen = BigRational::from_integer(BigInt::from(16));
        let four = BigRational::from_integer(BigInt::from(4));
        let mid = atan5.midpoint() * &sixteen - atan239.midpoint() * &four;
        let rad = atan5.radius() * &sixteen + atan239.radius() * &four;
        let ball = Self::new(mid, rad)?;
        checked_work_bits(
            "RealBall pi enclosure",
            [
                ball.midpoint.height().max_bits(),
                ball.radius.height().max_bits(),
            ],
        )?;
        Ok(ball)
    }

    /// Certified enclosure of sin(x) over the declared envelope: exact
    /// range reduction by k*pi, Taylor polynomial of degree 2m+1 evaluated
    /// with exact rational interval arithmetic, and the Lagrange tail bound
    /// |R| <= u^(2m+2)/(2m+2)!.
    pub fn sin(&self, precision_digits: u32) -> Result<Self, BallError> {
        let target = Self::precision_target(precision_digits)?;
        let magnitude = self.decimal_magnitude_bound();
        if magnitude > Self::MAX_REDUCTION_DECIMAL_MAGNITUDE {
            return Err(BallError::ArgumentOutsideDeclaredEnvelope(format!(
                "sin argument magnitude 10^{magnitude} exceeds the declared \
                 range-reduction bound of 10^{}",
                Self::MAX_REDUCTION_DECIMAL_MAGNITUDE
            )));
        }
        if self.upper().is_zero() && self.lower().is_zero() {
            return Ok(Self::from_i64(0));
        }
        let (reduced, parity) = self.reduce_mod_pi(precision_digits)?;
        // Degree schedule: u <= 2 after reduction, so degree 2m+1 with
        // u^(2m+2)/(2m+2)! <= target needs m ~ digits; double until met.
        let u = upper_abs_rational(&reduced);
        let mut m = (precision_digits + 8) as usize;
        let tail = loop {
            let tail = u
                .pow((2 * m + 2) as i32)
                .map_err(|_| transcendental_arithmetic_overflow())?
                / BigRational::from_integer(factorial(2 * m + 2));
            if tail <= target {
                break tail;
            }
            m *= 2;
            if m > (4 * Self::MAX_TRANSCENDENTAL_PRECISION_DIGITS as usize).pow(2) {
                return Err(BallError::ResourceLimitExceeded {
                    operation: "RealBall sin enclosure",
                    required_bits: u64::MAX,
                    limit_bits: MAX_REAL_BALL_WORK_BITS,
                });
            }
        };
        let series = sin_series_interval(&reduced, m)?;
        let ball = Self::new(series.midpoint().clone(), series.radius() + tail)?;
        checked_work_bits(
            "RealBall sin enclosure",
            [
                ball.midpoint.height().max_bits(),
                ball.radius.height().max_bits(),
            ],
        )?;
        if parity == 1 {
            Ok(ball.neg())
        } else {
            Ok(ball)
        }
    }

    /// Certified enclosure of cos(x): same reduction and tail discipline as
    /// sin, with the even Taylor polynomial of degree 2m and Lagrange bound
    /// |R| <= u^(2m+1)/(2m+1)!.
    pub fn cos(&self, precision_digits: u32) -> Result<Self, BallError> {
        let target = Self::precision_target(precision_digits)?;
        let magnitude = self.decimal_magnitude_bound();
        if magnitude > Self::MAX_REDUCTION_DECIMAL_MAGNITUDE {
            return Err(BallError::ArgumentOutsideDeclaredEnvelope(format!(
                "cos argument magnitude 10^{magnitude} exceeds the declared \
                 range-reduction bound of 10^{}",
                Self::MAX_REDUCTION_DECIMAL_MAGNITUDE
            )));
        }
        let (reduced, parity) = self.reduce_mod_pi(precision_digits)?;
        let u = upper_abs_rational(&reduced);
        let mut m = (precision_digits + 8) as usize;
        let tail = loop {
            let tail = u
                .pow((2 * m + 1) as i32)
                .map_err(|_| transcendental_arithmetic_overflow())?
                / BigRational::from_integer(factorial(2 * m + 1));
            if tail <= target {
                break tail;
            }
            m *= 2;
            if m > (4 * Self::MAX_TRANSCENDENTAL_PRECISION_DIGITS as usize).pow(2) {
                return Err(BallError::ResourceLimitExceeded {
                    operation: "RealBall cos enclosure",
                    required_bits: u64::MAX,
                    limit_bits: MAX_REAL_BALL_WORK_BITS,
                });
            }
        };
        let series = cos_series_interval(&reduced, m)?;
        let ball = Self::new(series.midpoint().clone(), series.radius() + tail)?;
        checked_work_bits(
            "RealBall cos enclosure",
            [
                ball.midpoint.height().max_bits(),
                ball.radius.height().max_bits(),
            ],
        )?;
        if parity == 1 {
            Ok(ball.neg())
        } else {
            Ok(ball)
        }
    }

    /// Certified enclosure of exp(x): the argument is halved until the
    /// remainder has magnitude <= 1/2 (tail <= 2*u^(n+1)/(n+1)! since
    /// e^(1/2) < 2), evaluated exactly, then squared back with interval
    /// arithmetic. Arguments beyond 10^MAX_EXP_DECIMAL_MAGNITUDE refuse:
    /// the enclosure midpoint itself would not be representable.
    pub fn exp(&self, precision_digits: u32) -> Result<Self, BallError> {
        Self::precision_target(precision_digits)?;
        let magnitude = self.decimal_magnitude_bound();
        if magnitude > Self::MAX_EXP_DECIMAL_MAGNITUDE {
            return Err(BallError::ArgumentOutsideDeclaredEnvelope(format!(
                "exp argument magnitude 10^{magnitude} exceeds the declared \
                 representable-enclosure bound of 10^{} (the enclosure of \
                 e^x itself would need more than 4350 decimal digits)",
                Self::MAX_EXP_DECIMAL_MAGNITUDE
            )));
        }
        if self.upper().is_zero() && self.lower().is_zero() {
            return Ok(Self::from_i64(1));
        }
        let u = upper_abs_rational(self);
        let two = BigRational::from_integer(BigInt::from(2));
        let threshold = BigRational::new(BigInt::one(), BigInt::from(2));
        let mut s: u32 = 0;
        let mut t = u.clone();
        while t > threshold {
            t /= &two;
            s += 1;
        }
        // Extra guard digits: each squaring roughly quadruples the absolute
        // radius relative to the midpoint scale.
        let series_digits = precision_digits + 3 + 2 * s;
        let two_pow_s = two
            .pow(s as i32)
            .map_err(|_| transcendental_arithmetic_overflow())?;
        let scaled = self.div(&Self::new(two_pow_s, BigRational::zero())?)?;
        let mut value = exp_series_interval(&scaled, series_digits)?;
        for _ in 0..s {
            value = value.mul(&value);
        }
        if value.radius() > &ten_pow_rational(precision_digits) {
            return Err(BallError::ResourceLimitExceeded {
                operation: "RealBall exp enclosure",
                required_bits: u64::MAX,
                limit_bits: MAX_REAL_BALL_WORK_BITS,
            });
        }
        checked_work_bits(
            "RealBall exp enclosure",
            [
                value.midpoint.height().max_bits(),
                value.radius.height().max_bits(),
            ],
        )?;
        Ok(value)
    }

    /// Exact range reduction: computes k = round(x/pi) on midpoints, then
    /// the certified interval x - k*pi with pi enclosed to enough digits
    /// that |k| * radius(pi) stays below the precision floor. Returns the
    /// reduced ball and the parity of k (sin(x) and cos(x) pick up the sign
    /// (-1)^k).
    fn reduce_mod_pi(&self, precision_digits: u32) -> Result<(Self, u32), BallError> {
        let extra_digits = self.decimal_magnitude_bound();
        let pi = Self::pi(precision_digits + extra_digits + 4)?;
        let xm = self.midpoint();
        let pm = pi.midpoint();
        let num = xm.numer() * pm.denom();
        let den = xm.denom() * pm.numer();
        let k = bigdiv_floor(&(&num * 2 + &den), &(&den * 2));
        let pi_lo = pi.lower();
        let pi_hi = pi.upper();
        let kp_lo = scale_rational_by_bigint(&pi_lo, &k);
        let kp_hi = scale_rational_by_bigint(&pi_hi, &k);
        let (kp_lo, kp_hi) = if k.is_negative() {
            (kp_hi, kp_lo)
        } else {
            (kp_lo, kp_hi)
        };
        let lo = self.lower() - &kp_hi;
        let hi = self.upper() - &kp_lo;
        let two = BigRational::from_integer(BigInt::from(2));
        let mid = (&lo + &hi) / &two;
        let rad = (&hi - &lo) / &two;
        let reduced = Self::new(mid, rad)?;
        let mut parity_rem = &k % 2;
        if parity_rem.is_negative() {
            parity_rem += 2;
        }
        let parity = if parity_rem.is_zero() { 0u32 } else { 1u32 };
        Ok((reduced, parity))
    }
}

fn checked_work_bits(
    operation: &'static str,
    terms: impl IntoIterator<Item = u64>,
) -> Result<u64, BallError> {
    let required_bits = terms
        .into_iter()
        .try_fold(0u64, u64::checked_add)
        .unwrap_or(u64::MAX);
    if required_bits > MAX_REAL_BALL_WORK_BITS {
        return Err(BallError::ResourceLimitExceeded {
            operation,
            required_bits,
            limit_bits: MAX_REAL_BALL_WORK_BITS,
        });
    }
    Ok(required_bits)
}

fn integer_power_bit_bound(value: &BigInt, exp: u32) -> u64 {
    let bits = value.bits();
    if bits <= 1 {
        bits
    } else {
        bits.saturating_mul(u64::from(exp))
    }
}

fn rational_power_bit_bound(value: &BigRational, exp: u32) -> u64 {
    integer_power_bit_bound(value.numer(), exp).max(integer_power_bit_bound(value.denom(), exp))
}

fn ensure_endpoint_construction_fits(
    ball: &RealBall,
    operation: &'static str,
) -> Result<(), BallError> {
    checked_work_bits(
        operation,
        [
            ball.midpoint.height().max_bits(),
            ball.radius.height().max_bits(),
        ],
    )
    .map(|_| ())
}

fn rational_sqrt_bound(
    value: &BigRational,
    precision_bits: u32,
    is_upper: bool,
) -> Result<BigRational, BallError> {
    if value.is_zero() {
        return Ok(BigRational::zero());
    }

    let shift = u64::from(precision_bits);
    let doubled_shift = shift.saturating_mul(2);
    checked_work_bits(
        "RealBall square root",
        [value.numer().bits(), value.denom().bits(), doubled_shift],
    )?;
    checked_work_bits(
        "RealBall square-root denominator",
        [value.denom().bits(), shift],
    )?;

    let doubled_shift = precision_bits
        .checked_mul(2)
        .ok_or(BallError::ResourceLimitExceeded {
            operation: "RealBall square root",
            required_bits: u64::MAX,
            limit_bits: MAX_REAL_BALL_WORK_BITS,
        })?;
    let p = value.numer();
    let q = value.denom();
    let p_scaled = p * (BigInt::one() << doubled_shift);
    let num_prod = &p_scaled * q;
    let s_floor = num_prod.sqrt().ok_or_else(|| {
        BallError::NonRealSquareRoot(
            "internal square-root radicand unexpectedly became negative".to_string(),
        )
    })?;
    let s = if is_upper && &s_floor * &s_floor != num_prod {
        &s_floor + BigInt::one()
    } else {
        s_floor
    };
    let denom_scaled = q * (BigInt::one() << precision_bits);
    Ok(BigRational::new(s, denom_scaled))
}

fn rational_pow(r: &BigRational, mut exp: usize) -> BigRational {
    if exp == 0 {
        return BigRational::one();
    }
    let mut base = r.clone();
    let mut acc = BigRational::one();
    while exp > 0 {
        if exp % 2 == 1 {
            acc *= &base;
        }
        exp /= 2;
        if exp > 0 {
            base = &base * &base;
        }
    }
    acc
}

impl fmt::Display for RealBall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{} ± {}]", self.midpoint, self.radius)
    }
}

/// Canonical typed refusal for checked-rational overflow inside the
/// transcendental enclosure helpers.
fn transcendental_arithmetic_overflow() -> BallError {
    BallError::ResourceLimitExceeded {
        operation: "RealBall transcendental enclosure",
        required_bits: u64::MAX,
        limit_bits: MAX_REAL_BALL_WORK_BITS,
    }
}

/// 10^-n as an exact rational (n = 0 gives 1).
fn ten_pow_rational(n: u32) -> BigRational {
    let ten = BigInt::from(10);
    let denom = ten.pow(n);
    BigRational::new(BigInt::one(), denom)
}

/// n! for modest n (series degree schedules).
fn factorial(n: usize) -> BigInt {
    let mut acc = BigInt::one();
    for k in 2..=n {
        acc *= BigInt::from(k as u64);
    }
    acc
}

/// Floor division for signed BigInts (no num-integer dependency).
fn bigdiv_floor(a: &BigInt, b: &BigInt) -> BigInt {
    let q = a / b;
    let r = a % b;
    if !r.is_zero() && r.is_negative() != b.is_negative() {
        q - 1
    } else {
        q
    }
}

/// |lower| if it exceeds |upper|, else |upper|: an upper bound of |x| over
/// the whole ball.
fn upper_abs_rational(ball: &RealBall) -> BigRational {
    let lo = ball.lower().abs();
    let hi = ball.upper().abs();
    if lo > hi { lo } else { hi }
}

fn scale_rational_by_bigint(q: &BigRational, k: &BigInt) -> BigRational {
    BigRational::new(q.numer() * k, q.denom().clone())
}

/// Certified enclosure of atan(1/t) for integer t >= 2: exact alternating
/// rational partial sums; the Leibniz tail is bounded by the first omitted
/// term, which is <= 10^-(digits+2) by the stop condition.
fn arctan_inverse_enclosure(t: u64, digits: u32) -> Result<RealBall, BallError> {
    let target = ten_pow_rational(digits + 2);
    let t_big = BigInt::from(t);
    let t_sq = &t_big * &t_big;
    let mut t_pow = t_big.clone(); // t^(2j+1)
    let mut sum = BigRational::zero();
    let mut j: u64 = 0;
    loop {
        let denom = BigInt::from(2 * j + 1) * &t_pow;
        let term = BigRational::new(BigInt::one(), denom.clone());
        if term <= target {
            break;
        }
        if j.is_multiple_of(2) {
            sum += term;
        } else {
            sum -= term;
        }
        t_pow *= &t_sq;
        j += 1;
        if j > 100_000_000 {
            return Err(BallError::ResourceLimitExceeded {
                operation: "RealBall pi enclosure",
                required_bits: u64::MAX,
                limit_bits: MAX_REAL_BALL_WORK_BITS,
            });
        }
    }
    let tail = BigRational::new(BigInt::one(), BigInt::from(2 * j + 1) * &t_pow);
    let ball = RealBall::new(sum, tail)?;
    checked_work_bits(
        "RealBall pi arctan series",
        [
            ball.midpoint.height().max_bits(),
            ball.radius.height().max_bits(),
        ],
    )?;
    Ok(ball)
}

/// Exact interval evaluation of the odd Taylor polynomial of degree 2m+1
/// for sin, with all arithmetic on certified balls (no rounding anywhere:
/// the coefficients are exact rationals).
fn sin_series_interval(r: &RealBall, m: usize) -> Result<RealBall, BallError> {
    let mut acc = RealBall::from_i64(0);
    let mut even_pow = RealBall::from_i64(1); // r^(2j)
    let r2 = r.mul(r);
    let mut fact = BigInt::one(); // (2j+1)!
    for j in 0..=m {
        if j > 0 {
            even_pow = even_pow.mul(&r2);
            fact *= BigInt::from(2 * j as u64) * BigInt::from((2 * j + 1) as u64);
        }
        let odd_pow = even_pow.mul(r);
        let coeff = BigRational::new(BigInt::one(), fact.clone());
        let term = RealBall::new(odd_pow.midpoint() * &coeff, odd_pow.radius() * &coeff)?;
        acc = if j % 2 == 0 {
            acc.add(&term)
        } else {
            acc.sub(&term)
        };
    }
    Ok(acc)
}

/// Exact interval evaluation of the even Taylor polynomial of degree 2m
/// for cos.
fn cos_series_interval(r: &RealBall, m: usize) -> Result<RealBall, BallError> {
    let mut acc = RealBall::from_i64(1);
    let mut even_pow = RealBall::from_i64(1); // r^(2j)
    let r2 = r.mul(r);
    let mut fact = BigInt::one(); // (2j)!
    for j in 1..=m {
        even_pow = even_pow.mul(&r2);
        fact *= BigInt::from(2 * j as u64 - 1) * BigInt::from(2 * j as u64);
        let coeff = BigRational::new(BigInt::one(), fact.clone());
        let term = RealBall::new(even_pow.midpoint() * &coeff, even_pow.radius() * &coeff)?;
        acc = if j % 2 == 1 {
            acc.sub(&term)
        } else {
            acc.add(&term)
        };
    }
    Ok(acc)
}

/// Exact interval evaluation of exp's Taylor polynomial to the smallest
/// degree whose Lagrange tail (factor e^u <= 2 for u <= 1/2) is within the
/// target radius; returns the enclosure already carrying the tail bound.
fn exp_series_interval(x: &RealBall, digits: u32) -> Result<RealBall, BallError> {
    let target = ten_pow_rational(digits + 2);
    let u = upper_abs_rational(x);
    let two = BigRational::from_integer(BigInt::from(2));
    let mut n = (digits + 8) as usize;
    let tail = loop {
        let tail = &two
            * u.pow(n as i32)
                .map_err(|_| transcendental_arithmetic_overflow())?
            / BigRational::from_integer(factorial(n + 1));
        if tail <= target {
            break tail;
        }
        n *= 2;
        if n > 4_000_000 {
            return Err(BallError::ResourceLimitExceeded {
                operation: "RealBall exp enclosure",
                required_bits: u64::MAX,
                limit_bits: MAX_REAL_BALL_WORK_BITS,
            });
        }
    };
    let mut acc = RealBall::from_i64(0);
    let mut pow = RealBall::from_i64(1); // x^j
    let mut fact = BigInt::one(); // j!
    for j in 0..=n {
        if j > 0 {
            pow = pow.mul(x);
            fact *= BigInt::from(j as u64);
        }
        let coeff = BigRational::new(BigInt::one(), fact.clone());
        let term = RealBall::new(pow.midpoint() * &coeff, pow.radius() * &coeff)?;
        acc = acc.add(&term);
    }
    RealBall::new(acc.midpoint().clone(), acc.radius() + tail)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(i: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(i))
    }

    #[test]
    fn new_rejects_negative_radius() {
        let err = RealBall::new(q(0), BigRational::from_integer(BigInt::from(-1))).unwrap_err();
        assert!(matches!(err, BallError::NegativeRadius(_)));
        // Zero radius is allowed: an exact point ball.
        assert!(RealBall::new(q(0), q(0)).is_ok());
    }

    #[test]
    fn exact_and_from_i64_construct_point_balls() {
        let b = RealBall::exact(BigRational::new(1.into(), 3.into()));
        assert_eq!(b.lower(), BigRational::new(1.into(), 3.into()));
        assert_eq!(b.upper(), BigRational::new(1.into(), 3.into()));
        assert!(b.radius().is_zero());

        let b = RealBall::from_i64(5);
        assert_eq!(b.lower(), q(5));
        assert_eq!(b.upper(), q(5));
    }

    #[test]
    fn contains_and_contains_ball_and_contains_zero() {
        let ball = RealBall::new(q(0), q(2)).unwrap();
        assert!(ball.contains(&q(0)));
        assert!(ball.contains(&q(2)));
        assert!(ball.contains(&BigRational::from_integer(BigInt::from(-2))));
        assert!(!ball.contains(&q(3)));
        assert!(!ball.contains(&BigRational::from_integer(BigInt::from(-3))));
        assert!(ball.contains_zero());

        // contains_ball is inclusive on the boundaries.
        let inner = RealBall::new(q(0), q(1)).unwrap();
        let boundary = RealBall::new(q(2), q(0)).unwrap();
        let outside = RealBall::new(q(5), q(0)).unwrap();
        assert!(ball.contains_ball(&inner));
        assert!(ball.contains_ball(&boundary));
        assert!(!ball.contains_ball(&outside));
    }

    #[test]
    fn is_positive_and_is_negative_use_strict_inequalities() {
        // The boundary point zero is neither strictly positive nor
        // strictly negative.
        let zero = RealBall::exact(q(0));
        assert!(!zero.is_positive());
        assert!(!zero.is_negative());

        // A ball that includes zero but is mostly positive is not
        // strictly positive.
        let cross = RealBall::new(q(0), q(1)).unwrap();
        assert!(!cross.is_positive());
        assert!(!cross.is_negative());

        // A ball entirely above zero.
        let pos = RealBall::new(q(2), q(1)).unwrap();
        assert!(pos.is_positive());
        assert!(!pos.is_negative());
    }

    #[test]
    fn add_sub_neg_preserve_radius_rules() {
        let a = RealBall::new(q(1), q(2)).unwrap();
        let b = RealBall::new(q(3), q(4)).unwrap();
        let sum = a.add(&b);
        assert_eq!(sum.midpoint(), &q(4));
        // Radii ADD under ball addition, not the triangle inequality.
        assert_eq!(sum.radius(), &q(6));

        let diff = a.sub(&b);
        assert_eq!(
            diff.midpoint(),
            &BigRational::from_integer(BigInt::from(-2))
        );
        assert_eq!(diff.radius(), &q(6));

        let neg = a.neg();
        assert_eq!(neg.midpoint(), &BigRational::from_integer(BigInt::from(-1)));
        assert_eq!(neg.radius(), &q(2));
    }

    #[test]
    fn mul_uses_abs_midpoint_radius_formula() {
        // (m1, r1) * (m2, r2) has midpoint m1*m2 and radius
        // |m1|*r2 + |m2|*r1 + r1*r2.
        let a = RealBall::new(q(2), q(1)).unwrap();
        let b = RealBall::new(q(3), q(1)).unwrap();
        let prod = a.mul(&b);
        assert_eq!(prod.midpoint(), &q(6));
        // 2*1 + 3*1 + 1*1 = 6
        assert_eq!(prod.radius(), &q(6));

        // Negative midpoint: (m1=-2, r1=1)*(m2=3, r2=1) has midpoint
        // -6 and radius |-2|*1 + 3*1 + 1*1 = 6.
        let neg = RealBall::new(BigRational::from_integer(BigInt::from(-2)), q(1)).unwrap();
        let prod = neg.mul(&b);
        assert_eq!(
            prod.midpoint(),
            &BigRational::from_integer(BigInt::from(-6))
        );
        assert_eq!(prod.radius(), &q(6));
    }

    #[test]
    fn inv_rejects_balls_containing_zero() {
        let crossing = RealBall::new(q(0), q(1)).unwrap();
        assert!(matches!(crossing.inv(), Err(BallError::DivisionByZero(_))));

        let zero_point = RealBall::exact(q(0));
        assert!(matches!(
            zero_point.inv(),
            Err(BallError::DivisionByZero(_))
        ));
    }

    #[test]
    fn inv_is_consistent_for_positive_balls() {
        // 1 / [3, 5] should land in [1/5, 1/3] with midpoint 4/15.
        let ball = RealBall::new(q(3), q(2)).unwrap();
        let inv = ball.inv().unwrap();
        // Lower = 1/5, upper = 1/3.
        assert!(inv.lower() <= BigRational::new(1.into(), 5.into()));
        assert!(inv.upper() >= BigRational::new(1.into(), 3.into()));
    }

    #[test]
    fn intersect_returns_none_for_disjoint() {
        let a = RealBall::new(q(0), q(1)).unwrap();
        let b = RealBall::new(q(5), q(1)).unwrap();
        assert!(a.intersect(&b).is_none());
    }

    #[test]
    fn intersect_returns_tightest_common_enclosure() {
        let a = RealBall::new(q(0), q(3)).unwrap();
        // Ball a = [-3, 3]
        let b = RealBall::new(q(2), q(3)).unwrap();
        // Ball b = [-1, 5]
        let intersection = a.intersect(&b).unwrap();
        // max(-3, -1) = -1, min(3, 5) = 3
        assert_eq!(
            intersection.lower(),
            BigRational::from_integer(BigInt::from(-1))
        );
        assert_eq!(intersection.upper(), q(3));
    }

    #[test]
    fn digest_is_deterministic_and_distinct() {
        let b1 = RealBall::new(q(1), q(2)).unwrap();
        let b2 = RealBall::new(q(1), q(2)).unwrap();
        let b3 = RealBall::new(q(1), q(3)).unwrap();
        assert_eq!(b1.digest().unwrap(), b2.digest().unwrap());
        assert_ne!(b1.digest().unwrap(), b3.digest().unwrap());
        assert_ne!(b1.digest().unwrap(), [0u8; 32]);
    }

    #[test]
    fn digest_refuses_oversized_canonical_numeric_payload() {
        let oversized_bits = u32::try_from(crate::canonical::MAX_SERIALIZED_BYTES)
            .unwrap()
            .checked_mul(8)
            .unwrap();
        let oversized = RealBall::exact(BigRational::from_integer(BigInt::one() << oversized_bits));
        assert!(matches!(
            oversized.digest(),
            Err(BallError::CanonicalEncoding(_))
        ));
    }

    #[test]
    fn test_real_ball_abs_pow_sqrt() {
        // 1. Abs of [-2, 3] is [0, 3]
        let b = RealBall::new(
            BigRational::new(1.into(), 2.into()),
            BigRational::new(5.into(), 2.into()),
        )
        .unwrap();
        assert_eq!(b.lower(), q(-2));
        assert_eq!(b.upper(), q(3));
        let b_abs = b.abs();
        assert_eq!(b_abs.lower(), q(0));
        assert_eq!(b_abs.upper(), q(3));

        // 2. Square of [-2, 3] is [0, 9]
        let b_sq = b.pow(2).unwrap();
        assert_eq!(b_sq.lower(), q(0));
        assert_eq!(b_sq.upper(), q(9));

        // 3. Cube of [-2, 3] is [-8, 27]
        let b_cube = b.pow(3).unwrap();
        assert_eq!(b_cube.lower(), q(-8));
        assert_eq!(b_cube.upper(), q(27));

        // 4. Sqrt of [4, 9] is [2, 3]
        let b_pos = RealBall::new(
            BigRational::new(13.into(), 2.into()),
            BigRational::new(5.into(), 2.into()),
        )
        .unwrap();
        assert_eq!(b_pos.lower(), q(4));
        assert_eq!(b_pos.upper(), q(9));
        let b_sqrt = b_pos.sqrt(16).unwrap();
        assert!(b_sqrt.lower() <= q(2));
        assert!(b_sqrt.upper() >= q(3));
        assert!(b_sqrt.contains(&q(2)));
        assert!(b_sqrt.contains(&q(3)));
    }

    #[test]
    fn power_handles_extreme_signed_exponents_without_recursion() {
        assert_eq!(
            RealBall::from_i64(1).pow(i32::MIN).unwrap(),
            RealBall::from_i64(1)
        );
        assert_eq!(
            RealBall::from_i64(-1).pow(i32::MIN).unwrap(),
            RealBall::from_i64(1)
        );
        assert!(matches!(
            RealBall::from_i64(2).pow(i32::MIN),
            Err(BallError::ResourceLimitExceeded {
                operation: "RealBall integer power",
                ..
            })
        ));
    }

    #[test]
    fn zero_exponent_requires_ball_to_exclude_zero() {
        let crossing = RealBall::new(q(0), q(1)).unwrap();
        assert!(matches!(
            crossing.pow(0),
            Err(BallError::IndeterminatePower(_))
        ));
        assert_eq!(RealBall::from_i64(2).pow(0).unwrap(), RealBall::from_i64(1));
    }

    #[test]
    fn square_root_requires_complete_nonnegative_domain() {
        let crossing = RealBall::new(q(0), q(1)).unwrap();
        assert!(matches!(
            crossing.sqrt(8),
            Err(BallError::NonRealSquareRoot(_))
        ));
        assert!(matches!(
            RealBall::from_i64(-1).sqrt(8),
            Err(BallError::NonRealSquareRoot(_))
        ));
    }

    #[test]
    fn square_root_refuses_extreme_precision_before_shifting() {
        assert!(matches!(
            RealBall::from_i64(2).sqrt(u32::MAX),
            Err(BallError::ResourceLimitExceeded {
                operation: "RealBall square root",
                ..
            })
        ));
    }

    /// Exact rational of a plain decimal string (oracle-derived frozen
    /// reference; the true transcendental value is within 10^-(frac_len+2)
    /// of it, far inside the requested-precision radius).
    fn decimal_rational(s: &str) -> BigRational {
        let (int_part, frac_part) = match s.split_once('.') {
            Some((a, b)) => (a.to_string(), b.to_string()),
            None => (s.to_string(), String::new()),
        };
        let negative = int_part.starts_with('-');
        let digits = format!("{int_part}{frac_part}")
            .trim_start_matches('-')
            .to_string();
        let numer = BigInt::parse_bytes(digits.as_bytes(), 10).expect("decimal digits");
        let denom = BigInt::from(10).pow(frac_part.len() as u32);
        let signed = if negative { -numer } else { numer };
        BigRational::new(signed, denom)
    }

    fn contains_reference(ball: &RealBall, reference: &BigRational) -> bool {
        ball.lower() <= *reference && *reference <= ball.upper()
    }

    #[test]
    fn transcendentals_enclose_oracle_references_at_requested_precision() {
        type OracleCase = (
            Box<dyn Fn(&RealBall, u32) -> Result<RealBall, BallError>>,
            &'static str,
            &'static str,
        );
        let cases: Vec<OracleCase> = vec![
            (
                Box::new(|x: &RealBall, d: u32| x.sin(d)),
                "1",
                "0.841470984807896506652502321630299",
            ),
            (
                Box::new(|x: &RealBall, d: u32| x.cos(d)),
                "1",
                "0.540302305868139717400936607442976604",
            ),
            (
                Box::new(|x: &RealBall, d: u32| x.exp(d)),
                "1",
                "2.7182818284590452353602874713526625",
            ),
            (
                Box::new(|x: &RealBall, d: u32| x.sin(d)),
                "2.5",
                "0.598472144103956494051854702186162272",
            ),
            (
                Box::new(|x: &RealBall, d: u32| x.cos(d)),
                "7.75",
                "0.103794357219252971027694067713823037",
            ),
            (
                Box::new(|x: &RealBall, d: u32| x.sin(d)),
                "-3",
                "-0.14112000805986722210074480280811028",
            ),
            (
                Box::new(|x: &RealBall, d: u32| x.exp(d)),
                "-10",
                "0.0000453999297624848515355915155605506102",
            ),
            (
                Box::new(|x: &RealBall, d: u32| x.exp(d)),
                "3",
                "20.0855369231876677409285296545817179",
            ),
        ];
        for (op, arg, reference) in &cases {
            let x = RealBall::exact(decimal_rational(arg));
            let ball = op(&x, 30).expect("enclosure succeeds");
            // The recorded reference is a 34-digit truncation of the true
            // value: tolerate its own <= 10^-33 rounding error when checking
            // containment of the enclosure.
            let r = decimal_rational(reference);
            let tol = ten_pow_rational(33);
            let ref_lo = r.clone() - &tol;
            let ref_hi = &r + &tol;
            assert!(
                ball.lower() <= ref_hi && ref_lo <= ball.upper(),
                "enclosure {ball} must overlap oracle reference {r} (+/-10^-33) for {arg}"
            );
            assert!(
                ball.radius() <= &ten_pow_rational(28),
                "radius must respect the 30-digit request for {arg}"
            );
        }
    }

    #[test]
    fn large_arguments_reduce_and_stay_certified() {
        let huge = RealBall::exact(BigRational::from_integer(BigInt::from(10).pow(30)));
        let sin_ball = huge.sin(24).expect("sin(10^30)");
        let sin_ref = decimal_rational("-0.0901169019121380580303864289529873303");
        assert!(contains_reference(&sin_ball, &sin_ref));
        let cos_ball = huge.cos(24).expect("cos(10^30)");
        let cos_ref = decimal_rational("-0.995931194405395702394248587997048641");
        assert!(contains_reference(&cos_ball, &cos_ref));
        assert!(sin_ball.radius() <= &ten_pow_rational(22));
        assert!(cos_ball.radius() <= &ten_pow_rational(22));
    }

    #[test]
    fn pi_ball_is_sound_across_precision_requests() {
        for digits in [1u32, 8, 30] {
            let pi = RealBall::pi(digits).expect("pi enclosure");
            let lower_ref = decimal_rational("3.14159265358979323846264338327950288");
            let upper_ref = decimal_rational("3.14159265358979323846264338327950289");
            assert!(pi.lower() <= lower_ref, "pi ball {pi} too high");
            assert!(pi.upper() >= upper_ref, "pi ball {pi} too low");
            assert!(pi.radius() <= &ten_pow_rational(digits));
        }
    }

    #[test]
    fn refinement_monotonically_narrows_the_enclosure() {
        let x = RealBall::exact(decimal_rational("2.5"));
        let coarse = x.sin(12).expect("coarse");
        let fine = x.sin(20).expect("fine");
        // Both enclose the same true value, so they must overlap; the
        // finer request must have strictly smaller radius.
        assert!(
            fine.intersect(&coarse).is_some(),
            "fine and coarse enclosures of the same value must overlap"
        );
        assert!(fine.radius() < coarse.radius());
    }

    #[test]
    fn deterministic_precision_schedule_is_reproducible() {
        let x = RealBall::exact(decimal_rational("7.75"));
        let a = x.cos(20).expect("cos A");
        let b = x.cos(20).expect("cos B");
        assert_eq!(a, b, "same input and precision must give the same ball");
    }

    #[test]
    fn envelope_and_precision_refusals_are_typed() {
        let x = RealBall::from_i64(1);
        assert!(matches!(
            x.sin(0),
            Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
        ));
        assert!(matches!(
            x.exp(RealBall::MAX_TRANSCENDENTAL_PRECISION_DIGITS + 1),
            Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
        ));
        let exp_huge = RealBall::exact(BigRational::from_integer(
            BigInt::from(10).pow(RealBall::MAX_EXP_DECIMAL_MAGNITUDE + 1),
        ));
        assert!(matches!(
            exp_huge.exp(10),
            Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
        ));
        let sin_huge = RealBall::exact(BigRational::from_integer(
            BigInt::from(10).pow(RealBall::MAX_REDUCTION_DECIMAL_MAGNITUDE + 1),
        ));
        assert!(matches!(
            sin_huge.sin(10),
            Err(BallError::ArgumentOutsideDeclaredEnvelope(_))
        ));
    }
}
