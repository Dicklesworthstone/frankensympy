//! Independent second-order interval automatic differentiation.
//!
//! The value, gradient and Hessian are recomputed from the original typed
//! expression. No symbolic differentiator, compiled derivative, numerical
//! sample or caller-supplied derivative is used. Every admitted elementary
//! rule is C2 on a neighborhood of the complete real box; in particular,
//! logarithms, square roots and noninteger powers require strictly positive
//! arguments. Every original child is visited before algebraic shortcuts.

use super::*;

/// Complete-box bounds for one real objective and its first two derivatives.
/// This is an enclosure, not a stationarity or convexity assertion. There is
/// no unchecked constructor or deserialization path for this receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectiveEnclosure {
    objective: Expr,
    variables: Vec<Symbol>,
    domain: Vec<RealBall>,
    precision_digits: u32,
    value: RealBall,
    gradient: Vec<RealBall>,
    hessian: Vec<Vec<RealBall>>,
}

impl ObjectiveEnclosure {
    pub fn objective(&self) -> &Expr {
        &self.objective
    }
    pub fn variables(&self) -> &[Symbol] {
        &self.variables
    }
    pub fn domain(&self) -> &[RealBall] {
        &self.domain
    }
    pub fn precision_digits(&self) -> u32 {
        self.precision_digits
    }
    pub fn value(&self) -> &RealBall {
        &self.value
    }
    pub fn gradient(&self) -> &[RealBall] {
        &self.gradient
    }
    pub fn hessian(&self) -> &[Vec<RealBall>] {
        &self.hessian
    }
}

/// Independently enclose one objective on a 1..=16 dimensional real box.
/// Point coordinates are allowed. The existing root-box precision, expression
/// height/depth, arithmetic and cumulative allocation limits apply.
pub fn enclose_objective(
    objective: &Expr,
    variables: &[Symbol],
    domain: &[RealBall],
    precision_digits: u32,
) -> Result<ObjectiveEnclosure, RootBoxError> {
    metered_enclose_objective(
        objective,
        variables,
        domain,
        precision_digits,
        &mut Unbounded,
    )
}

/// Use one caller-owned cumulative ledger, including all original-source
/// domain checks and both derivative orders. Any refusal or cancellation
/// returns no partial receipt and retains charges already consumed.
pub fn metered_enclose_objective(
    objective: &Expr,
    variables: &[Symbol],
    domain: &[RealBall],
    precision_digits: u32,
    meter: &mut impl BudgetMeter,
) -> Result<ObjectiveEnclosure, RootBoxError> {
    let mut work = Work::new(meter, precision_digits)?;
    validate_inputs(
        std::slice::from_ref(objective),
        variables,
        domain,
        &mut work,
    )?;
    let result = evaluate_objective(objective, variables, domain, &mut work)?;
    work.charge(1, work.source_bytes + binding_bytes(variables, domain), 4)?;
    let receipt = ObjectiveEnclosure {
        objective: objective.clone(),
        variables: variables.to_vec(),
        domain: domain.to_vec(),
        precision_digits,
        value: result.value,
        gradient: result.gradient,
        hessian: result.hessian,
    };
    work.step()?;
    Ok(receipt)
}

/// Private derivative data, usable by sibling source-bound checkers only.
pub(super) struct ObjectiveJet {
    pub(super) value: RealBall,
    pub(super) gradient: Vec<RealBall>,
    pub(super) hessian: Vec<Vec<RealBall>>,
}

impl ObjectiveJet {
    fn constant(
        value: RealBall,
        dimension: usize,
        work: &mut Work<'_, impl BudgetMeter>,
    ) -> Result<Self, RootBoxError> {
        let entries = 1 + dimension + dimension * dimension;
        work.charge(1, (entries * 128) as u64, (dimension + 2) as u64)?;
        let mut gradient = Vec::new();
        gradient
            .try_reserve_exact(dimension)
            .map_err(|_| RootBoxError::ResourceLimit("objective gradient allocation"))?;
        gradient.resize_with(dimension, || RealBall::from_i64(0));
        let mut hessian = Vec::new();
        hessian
            .try_reserve_exact(dimension)
            .map_err(|_| RootBoxError::ResourceLimit("objective Hessian allocation"))?;
        for _ in 0..dimension {
            let mut row = Vec::new();
            row.try_reserve_exact(dimension)
                .map_err(|_| RootBoxError::ResourceLimit("objective Hessian row allocation"))?;
            row.resize_with(dimension, || RealBall::from_i64(0));
            hessian.push(row);
        }
        Ok(Self {
            value,
            gradient,
            hessian,
        })
    }

    fn checked(self) -> Result<Self, RootBoxError> {
        check_ball(&self.value)?;
        for entry in self.gradient.iter().chain(self.hessian.iter().flatten()) {
            check_ball(entry)?;
        }
        Ok(self)
    }
}

fn add_jets(
    a: ObjectiveJet,
    b: ObjectiveJet,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<ObjectiveJet, RootBoxError> {
    let n = a.gradient.len();
    let value = work.add(&a.value, &b.value)?;
    let mut result = ObjectiveJet::constant(value, n, work)?;
    for i in 0..n {
        result.gradient[i] = work.add(&a.gradient[i], &b.gradient[i])?;
        for j in 0..n {
            result.hessian[i][j] = work.add(&a.hessian[i][j], &b.hessian[i][j])?;
        }
    }
    result.checked()
}

fn multiply_jets(
    a: ObjectiveJet,
    b: ObjectiveJet,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<ObjectiveJet, RootBoxError> {
    let n = a.gradient.len();
    let value = work.mul(&a.value, &b.value)?;
    let mut result = ObjectiveJet::constant(value, n, work)?;
    for i in 0..n {
        let left = work.mul(&a.gradient[i], &b.value)?;
        let right = work.mul(&a.value, &b.gradient[i])?;
        result.gradient[i] = work.add(&left, &right)?;
        for j in 0..n {
            let left = work.mul(&a.hessian[i][j], &b.value)?;
            let right = work.mul(&a.value, &b.hessian[i][j])?;
            let first = work.mul(&a.gradient[i], &b.gradient[j])?;
            let second = work.mul(&a.gradient[j], &b.gradient[i])?;
            let sum = work.add(&left, &right)?;
            let cross = work.add(&first, &second)?;
            result.hessian[i][j] = work.add(&sum, &cross)?;
        }
    }
    result.checked()
}

/// For h=f(g), h_ij = f'(g) g_ij + f''(g) g_i g_j.
fn chain_jet(
    argument: ObjectiveJet,
    value: RealBall,
    slope: RealBall,
    curvature: RealBall,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<ObjectiveJet, RootBoxError> {
    check_ball(&value)?;
    check_ball(&slope)?;
    check_ball(&curvature)?;
    let n = argument.gradient.len();
    let mut result = ObjectiveJet::constant(value, n, work)?;
    for i in 0..n {
        result.gradient[i] = work.mul(&slope, &argument.gradient[i])?;
        for j in 0..n {
            let first = work.mul(&slope, &argument.hessian[i][j])?;
            let product = work.mul(&argument.gradient[i], &argument.gradient[j])?;
            let second = work.mul(&curvature, &product)?;
            result.hessian[i][j] = work.add(&first, &second)?;
        }
    }
    result.checked()
}

fn logarithm(
    argument: ObjectiveJet,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<ObjectiveJet, RootBoxError> {
    if !argument.value.is_positive() {
        return Err(RootBoxError::Domain(
            "objective logarithm argument is not strictly positive",
        ));
    }
    work.elementary(&argument.value)?;
    let value = checked(argument.value.ln(work.precision)?)?;
    let slope = work.inv(&argument.value)?;
    let curvature = work.power(&slope, 2)?.neg();
    chain_jet(argument, value, slope, curvature, work)
}

fn exponential(
    argument: ObjectiveJet,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<ObjectiveJet, RootBoxError> {
    let value = work.exp(&argument.value)?;
    work.arithmetic(&[&value], &[])?;
    chain_jet(argument, value.clone(), value.clone(), value, work)
}

/// The caller validates source, variables and the box once before invoking
/// this evaluator. Repeated evaluations share this same work counter and meter.
pub(super) fn evaluate_objective(
    expression: &Expr,
    variables: &[Symbol],
    domain: &[RealBall],
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<ObjectiveJet, RootBoxError> {
    evaluate_inner(expression, variables, domain, 0, work)
}

fn evaluate_inner(
    expression: &Expr,
    variables: &[Symbol],
    domain: &[RealBall],
    depth: usize,
    work: &mut Work<'_, impl BudgetMeter>,
) -> Result<ObjectiveJet, RootBoxError> {
    work.step()?;
    // Three passes permit a stationary-point checker to inspect the original
    // box, a proposed center and the accepted root enclosure with one ledger.
    if depth > MAX_ROOT_BOX_DEPTH || work.nodes >= MAX_ROOT_BOX_NODES * 3 {
        return Err(RootBoxError::ResourceLimit(
            "objective expression depth or node count",
        ));
    }
    work.charge(1, 512, 4)?;
    work.nodes += 1;
    let n = variables.len();
    let digits = work.precision;
    let result = match expression {
        Expr::Integer(integer) => {
            work.charge(1, 128 + integer.limb_count() * 16, 2)?;
            ObjectiveJet::constant(
                RealBall::exact(BigRational::from_integer(integer.clone())),
                n,
                work,
            )?
        }
        Expr::Rational(rational) => {
            work.arithmetic(&[], &[rational])?;
            ObjectiveJet::constant(RealBall::exact(rational.clone()), n, work)?
        }
        Expr::Const(Constant::Pi) => {
            work.elementary(&RealBall::from_i64(1))?;
            ObjectiveJet::constant(RealBall::pi(digits)?, n, work)?
        }
        Expr::Const(Constant::E) => {
            let value = work.exp(&RealBall::from_i64(1))?;
            ObjectiveJet::constant(value, n, work)?
        }
        Expr::Const(_) => return Err(RootBoxError::Domain("non-real or non-finite constant")),
        Expr::Sym(symbol) => {
            let index = variables
                .iter()
                .position(|variable| variable == symbol)
                .ok_or(RootBoxError::Unsupported("unbound symbol identity"))?;
            work.arithmetic(&[&domain[index]], &[])?;
            let mut jet = ObjectiveJet::constant(domain[index].clone(), n, work)?;
            jet.gradient[index] = RealBall::from_i64(1);
            jet
        }
        Expr::Add(children) | Expr::Mul(children) => {
            let is_sum = matches!(expression, Expr::Add(_));
            let mut jet = ObjectiveJet::constant(RealBall::from_i64(i64::from(!is_sum)), n, work)?;
            for child in children {
                // In particular, a leading zero never hides a pole or a
                // nonsmooth branch in a later original child.
                let next = evaluate_inner(child, variables, domain, depth + 1, work)?;
                jet = if is_sum {
                    add_jets(jet, next, work)?
                } else {
                    multiply_jets(jet, next, work)?
                };
            }
            jet
        }
        Expr::Pow(base, exponent) => {
            let base_jet = evaluate_inner(base, variables, domain, depth + 1, work)?;
            let exponent_jet = evaluate_inner(exponent, variables, domain, depth + 1, work)?;
            let integer = match exponent.as_ref() {
                Expr::Integer(value) => Some(value),
                Expr::Rational(value) if value.denom().is_one() => Some(value.numer()),
                _ => None,
            };
            if let Some(integer) = integer {
                let power = integer
                    .to_i32()
                    .filter(|power| (-64..=64).contains(power))
                    .ok_or(RootBoxError::ResourceLimit("objective integer exponent"))?;
                let value = work.power(&base_jet.value, power)?;
                let slope = match power {
                    0 => RealBall::from_i64(0),
                    1 => RealBall::from_i64(1),
                    _ => {
                        let inner = work.power(&base_jet.value, power - 1)?;
                        work.mul(&inner, &RealBall::from_i64(i64::from(power)))?
                    }
                };
                let curvature = match power {
                    0 | 1 => RealBall::from_i64(0),
                    2 => RealBall::from_i64(2),
                    _ => {
                        let inner = work.power(&base_jet.value, power - 2)?;
                        work.mul(&inner, &RealBall::from_i64(i64::from(power * (power - 1))))?
                    }
                };
                chain_jet(base_jet, value, slope, curvature, work)?
            } else {
                // x^y = exp(y log(x)) only on the strictly positive real
                // branch. Both y derivatives and both mixed terms survive.
                if !base_jet.value.is_positive() {
                    return Err(RootBoxError::Domain(
                        "objective noninteger power base is not strictly positive",
                    ));
                }
                let log_base = logarithm(base_jet, work)?;
                let product = multiply_jets(exponent_jet, log_base, work)?;
                exponential(product, work)?
            }
        }
        Expr::Function(name, arguments) => {
            if name == "Piecewise" {
                let selected = work.select_stable_piecewise(arguments, variables, domain, depth)?;
                return evaluate_inner(selected, variables, domain, depth + 2, work);
            }
            let [argument] = arguments.as_slice() else {
                return Err(RootBoxError::Unsupported("non-unary objective function"));
            };
            let argument = evaluate_inner(argument, variables, domain, depth + 1, work)?;
            work.elementary(&argument.value)?;
            if matches!(name.as_str(), "log" | "ln") {
                return logarithm(argument, work);
            }
            if name == "exp" {
                return exponential(argument, work);
            }
            let (value, slope, curvature) = match name.as_str() {
                "sin" => {
                    let value = checked(argument.value.sin(digits)?)?;
                    let slope = checked(argument.value.cos(digits)?)?;
                    work.arithmetic(&[&value], &[])?;
                    (value.clone(), slope, value.neg())
                }
                "cos" => {
                    let value = checked(argument.value.cos(digits)?)?;
                    let slope = checked(argument.value.sin(digits)?)?.neg();
                    work.arithmetic(&[&value], &[])?;
                    (value.clone(), slope, value.neg())
                }
                "sqrt" => {
                    if !argument.value.is_positive() {
                        return Err(RootBoxError::Domain(
                            "objective square root is not C2 on this box",
                        ));
                    }
                    let value = checked(
                        argument
                            .value
                            .sqrt(digits.saturating_mul(4).saturating_add(64))?,
                    )?;
                    let twice = work.mul(&value, &RealBall::from_i64(2))?;
                    let slope = work.inv(&twice)?;
                    let cube = work.power(&value, 3)?;
                    let four_cubes = work.mul(&cube, &RealBall::from_i64(4))?;
                    let curvature = work.inv(&four_cubes)?.neg();
                    (value, slope, curvature)
                }
                "atan" => {
                    let square = work.power(&argument.value, 2)?;
                    let denominator = work.add(&RealBall::from_i64(1), &square)?;
                    let slope = work.inv(&denominator)?;
                    let inverse_square = work.power(&slope, 2)?;
                    let numerator = work.mul(&RealBall::from_i64(-2), &argument.value)?;
                    let curvature = work.mul(&numerator, &inverse_square)?;
                    (argument.value.atan(digits)?, slope, curvature)
                }
                "tan" => {
                    let cosine = checked(argument.value.cos(digits)?)?;
                    let inverse = work.inv(&cosine)?;
                    let sine = checked(argument.value.sin(digits)?)?;
                    let value = work.mul(&sine, &inverse)?;
                    let slope = work.power(&inverse, 2)?;
                    let product = work.mul(&value, &slope)?;
                    let curvature = work.mul(&RealBall::from_i64(2), &product)?;
                    (value, slope, curvature)
                }
                "sinh" | "cosh" | "tanh" => {
                    let plus = work.exp(&argument.value)?;
                    let minus = work.exp(&argument.value.neg())?;
                    let difference = work.sub(&plus, &minus)?;
                    let sum = work.add(&plus, &minus)?;
                    let sinh = work.div(&difference, &RealBall::from_i64(2))?;
                    let cosh = work.div(&sum, &RealBall::from_i64(2))?;
                    match name.as_str() {
                        "sinh" => {
                            work.arithmetic(&[&sinh], &[])?;
                            (sinh.clone(), cosh, sinh)
                        }
                        "cosh" => {
                            work.arithmetic(&[&cosh], &[])?;
                            (cosh.clone(), sinh, cosh)
                        }
                        _ => {
                            let value = work.div(&sinh, &cosh)?;
                            let inverse = work.inv(&cosh)?;
                            let slope = work.power(&inverse, 2)?;
                            let product = work.mul(&value, &slope)?;
                            let curvature = work.mul(&RealBall::from_i64(-2), &product)?;
                            (value, slope, curvature)
                        }
                    }
                }
                _ => {
                    return Err(RootBoxError::Unsupported(
                        "function has no C2 interval rule",
                    ));
                }
            };
            chain_jet(argument, value, slope, curvature, work)?
        }
    };
    result.checked()
}
