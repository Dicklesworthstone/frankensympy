//! Fundamental proof inference rules and proof terms for FrankenSymPy (WS06).
//!
//! Layer: L2 (claims and proof kernel).

#![forbid(unsafe_code)]

use crate::claim::Claim;
use fsym_assumptions::{Domain, Predicate};
use fsym_core::{Expr, Symbol};
use serde::{Deserialize, Serialize};

/// Reference handle to an established step in a proof derivation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StepId(pub u32);

/// Core deductive inference rules accepted by the proof kernel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofRule {
    /// $\vdash e = e$
    Reflexivity(Expr),
    /// $a = b \implies b = a$
    Symmetry(StepId),
    /// $a = b \land b = c \implies a = c$
    Transitivity(StepId, StepId),
    /// $a_i = b_i \implies \sum a_i = \sum b_i$
    CongruenceAdd(Vec<StepId>),
    /// $a_i = b_i \implies \prod a_i = \prod b_i$
    CongruenceMul(Vec<StepId>),
    /// $a = c \land b = d \implies a^b = c^d$
    CongruencePow { base: StepId, exp: StepId },
    /// $a_i = b_i \implies f(a_1..a_n) = f(b_1..b_n)$
    CongruenceFunction { name: String, args: Vec<StepId> },
    /// Context predicate query: $\Gamma \vdash P(e)$
    ContextPredicate { expr: Expr, predicate: Predicate },
    /// Context domain query: $\Gamma \vdash e \in \mathcal{D}$
    ContextDomain { expr: Expr, domain: Domain },
    /// Capture-safe term substitution: $a = b \implies T[x \mapsto a] = T[x \mapsto b]$
    Substitution {
        template: Expr,
        var: Symbol,
        step: StepId,
    },
    /// Verified elementary arithmetic / reduction steps:
    /// e.g., $x + 0 \to x$, $x \times 1 \to x$, $x \times 0 \to 0$, $x - x \to 0$, $c_1 + c_2 \to c_3$
    DefinitionalReduction {
        lhs: Expr,
        rhs: Expr,
        rule_name: String,
    },
    /// Reference to an independently verified certificate lemma from a certified family crate.
    CertificateLemma {
        family: String,
        claim: Claim,
        receipt_digest: [u8; 32],
    },
}
