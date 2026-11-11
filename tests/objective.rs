//! Exact expectations for original-source second-order objective requests.

use frankensympy::{CertificationError, ObjectiveProblem};
use fsym_budget::{BudgetMeter, Dimension, MeterError, Unbounded};
use fsym_core::{
    BigInt, BigRational, Expr, RealBall, RootBoxError, RootBoxWitness, StationaryPointKind, Symbol,
    SymbolIdentity,
};

fn q(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

fn problem(source: &str, variables: &[&str]) -> ObjectiveProblem {
    ObjectiveProblem::parse_source(
        source,
        &variables
            .iter()
            .map(|name| (*name).into())
            .collect::<Vec<_>>(),
        &mut Unbounded,
    )
    .unwrap()
}

fn witness(preconditioner: Vec<Vec<BigRational>>) -> RootBoxWitness {
    let n = preconditioner.len();
    RootBoxWitness {
        domain: vec![RealBall::new(q(0, 1), q(1, 1)).unwrap(); n],
        center: vec![q(0, 1); n],
        preconditioner,
        precision_digits: 24,
    }
}

#[test]
fn independent_exact_gradient_and_hessian_follow_declared_axis_order() {
    let problem = problem("x^2 + 3*x*y + 2*y^2", &["y", "x"]);
    let enclosed = problem
        .enclose(
            &[RealBall::from_i64(3), RealBall::from_i64(2)],
            24,
            &mut Unbounded,
        )
        .unwrap();
    assert_eq!(enclosed.value(), &RealBall::from_i64(40));
    assert_eq!(
        enclosed.gradient(),
        &[RealBall::from_i64(18), RealBall::from_i64(13)]
    );
    assert_eq!(
        enclosed.hessian(),
        &[
            vec![RealBall::from_i64(4), RealBall::from_i64(3)],
            vec![RealBall::from_i64(3), RealBall::from_i64(2)],
        ]
    );
    assert_eq!(enclosed.objective(), problem.objective());
    assert_eq!(enclosed.variables(), problem.variables());
}

#[test]
fn minimum_maximum_saddle_and_unclassified_are_distinct_local_claims() {
    for (source, inverse, kind) in [
        (
            "x^2+y^2",
            vec![vec![q(1, 2), q(0, 1)], vec![q(0, 1), q(1, 2)]],
            StationaryPointKind::StrictMinimum,
        ),
        (
            "-x^2-y^2",
            vec![vec![q(-1, 2), q(0, 1)], vec![q(0, 1), q(-1, 2)]],
            StationaryPointKind::StrictMaximum,
        ),
        (
            "x^2-y^2",
            vec![vec![q(1, 2), q(0, 1)], vec![q(0, 1), q(-1, 2)]],
            StationaryPointKind::Saddle,
        ),
        (
            "x*y",
            vec![vec![q(0, 1), q(1, 1)], vec![q(1, 1), q(0, 1)]],
            StationaryPointKind::Unclassified,
        ),
    ] {
        let problem = problem(source, &["x", "y"]);
        let proposed = witness(inverse);
        let receipt = problem
            .verify_stationary(&proposed, &mut Unbounded)
            .unwrap();
        assert_eq!(receipt.kind(), kind, "{source}");
        assert_eq!(receipt.contraction_bound(), &q(0, 1));
        assert_eq!(
            receipt.stationary_enclosure(),
            &[RealBall::from_i64(0), RealBall::from_i64(0)]
        );
        assert_eq!(receipt.witness(), &proposed);
    }
}

#[test]
fn typed_same_name_axes_never_alias_or_silently_rebind() {
    let x = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([1; 32]));
    let other = Symbol::with_identity("x", SymbolIdentity::from_assumptions_digest([2; 32]));
    let source = Expr::Add(vec![
        Expr::Sym(x.clone()),
        Expr::Mul(vec![Expr::from_i64(2), Expr::Sym(other.clone())]),
    ]);
    let problem = ObjectiveProblem::new(source, vec![other.clone(), x], &mut Unbounded).unwrap();
    let enclosed = problem
        .enclose(
            &[RealBall::from_i64(3), RealBall::from_i64(7)],
            24,
            &mut Unbounded,
        )
        .unwrap();
    assert_eq!(enclosed.value(), &RealBall::from_i64(13));
    assert_eq!(
        enclosed.gradient(),
        &[RealBall::from_i64(2), RealBall::from_i64(1)]
    );
    let unbound =
        ObjectiveProblem::new(Expr::Sym(other), vec![Symbol::new("x")], &mut Unbounded).unwrap();
    assert!(
        unbound
            .enclose(&[RealBall::from_i64(1)], 24, &mut Unbounded)
            .is_err()
    );
}

#[test]
fn source_domains_and_singular_stationary_points_cannot_be_erased() {
    for source in ["x^2+0*(1/x)", "log(x)-log(x)", "(1/x)^0", "sqrt(x)"] {
        let problem = problem(source, &["x"]);
        assert!(
            problem
                .enclose(&[RealBall::from_i64(0)], 24, &mut Unbounded)
                .is_err(),
            "{source}"
        );
        assert!(
            problem
                .verify_stationary(&witness(vec![vec![q(1, 2)]]), &mut Unbounded)
                .is_err(),
            "{source}"
        );
    }
    assert!(
        problem("x^4", &["x"])
            .verify_stationary(&witness(vec![vec![q(1, 1)]]), &mut Unbounded)
            .is_err()
    );
    assert_eq!(
        problem("x^2", &["x"])
            .verify_stationary(&witness(vec![vec![q(0, 1)]]), &mut Unbounded)
            .unwrap_err(),
        CertificationError::Root(RootBoxError::NotContractive)
    );
}

#[test]
fn registration_and_witness_shape_limits_refuse_before_math_acceptance() {
    for variables in [vec![], vec!["x".into(), "x".into()], vec!["pi".into()]] {
        assert!(ObjectiveProblem::parse_source("x", &variables, &mut Unbounded).is_err());
    }
    assert!(
        ObjectiveProblem::parse_source(&"x".repeat(4097), &["x".into()], &mut Unbounded,).is_err()
    );
    let problem = problem("x^2", &["x"]);
    let mut invalid = witness(vec![vec![q(1, 2), q(0, 1)]]);
    assert!(problem.verify_stationary(&invalid, &mut Unbounded).is_err());
    invalid = witness(vec![vec![q(1, 2)]]);
    invalid.precision_digits = 129;
    assert!(problem.verify_stationary(&invalid, &mut Unbounded).is_err());
    invalid.precision_digits = 24;
    invalid.center[0] = q(2, 1);
    assert!(problem.verify_stationary(&invalid, &mut Unbounded).is_err());
}

#[derive(Default)]
struct CancelAt {
    calls: usize,
    at: usize,
    stopped: bool,
    charged: u64,
}

impl BudgetMeter for CancelAt {
    fn charge(&mut self, _: Dimension, amount: u64) -> Result<(), MeterError> {
        assert!(!self.stopped);
        self.charged += amount;
        Ok(())
    }

    fn charge_batch(&mut self, charges: &[(Dimension, u64)]) -> Result<(), MeterError> {
        for &(dimension, amount) in charges {
            self.charge(dimension, amount)?;
        }
        Ok(())
    }

    fn checkpoint(&mut self) -> Result<(), MeterError> {
        assert!(!self.stopped);
        self.calls += 1;
        if self.calls == self.at {
            self.stopped = true;
            Err(MeterError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[test]
fn every_objective_and_stationary_safe_point_propagates_cancellation() {
    let problem = problem("x^2", &["x"]);
    let witness = witness(vec![vec![q(1, 2)]]);
    for stationary in [false, true] {
        let run = |meter: &mut CancelAt| {
            if stationary {
                problem.verify_stationary(&witness, meter).map(|_| ())
            } else {
                problem.enclose(&witness.domain, 24, meter).map(|_| ())
            }
        };
        let mut measured = CancelAt::default();
        run(&mut measured).unwrap();
        assert!(measured.calls > 2);
        for at in 1..=measured.calls {
            let mut cancelled = CancelAt {
                at,
                ..CancelAt::default()
            };
            assert_eq!(
                run(&mut cancelled),
                Err(CertificationError::Meter(MeterError::Cancelled))
            );
            assert!(cancelled.stopped);
            assert!(cancelled.charged <= measured.charged);
        }
    }
}
