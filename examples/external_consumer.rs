//! External small consumer: construct → differentiate → verify → export →
//! replay one supported claim with explicit budgets, using only the public
//! `frankensympy` library surface.

#![forbid(unsafe_code)]

use frankensympy::{ClaimStatus, Session, SessionBudgets};

fn main() {
    let mut session = Session::new(SessionBudgets::default());

    // Construct: bind a workspace term.
    session
        .construct("f", "sin(x) + x^2")
        .expect("constructs the binding");

    // Differentiate.
    let derivative = session
        .differentiate("sin(x) + x^2", "x")
        .expect("differentiates");
    assert!(derivative.contains("cos"), "derivative must mention cos(x)");

    // Verify: candidate first, accepted only after independent verification.
    let candidate = session.construct_claim("1 + 1").expect("claim");
    assert_eq!(candidate.status, ClaimStatus::Candidate);
    let accepted = session.verify_claim(&candidate).expect("verifies");
    assert_eq!(accepted.status, ClaimStatus::Accepted);

    // Export: canonical paginated JSON of the accepted claim.
    let export = session.export_claim(&accepted, 0, 16).expect("exports");
    assert_eq!(export["total_pages"], 1);

    // Replay: independent re-verification of the exported derivation.
    let digest = session
        .replay_claim(&export, accepted.context_digest)
        .expect("replays");
    assert_eq!(digest, accepted.derivation_digest.expect("accepted"));

    println!("external consumer OK: construct/differentiate/verify/export/replay");
}
