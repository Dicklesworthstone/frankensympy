//! # fsym-logic
//!
//! Boolean algebra, truth tables, normal forms (CNF, DNF), and SAT solving.

#![forbid(unsafe_code)]

use fsym_core::Symbol;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum LogicError {
    #[error("Variable not found in truth assignment: {0}")]
    UnassignedVariable(String),
    #[error("Truth table exceeds supported variable count: {0} > 20")]
    TableTooLarge(usize),
}

/// Propositional logic formula.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BoolExpr {
    /// Boolean constant: True or False.
    Const(bool),
    /// Boolean variable symbol.
    Var(Symbol),
    /// Logical NOT (¬A).
    Not(Box<BoolExpr>),
    /// Logical AND (A ∧ B ∧ ...).
    And(Vec<BoolExpr>),
    /// Logical OR (A ∨ B ∨ ...).
    Or(Vec<BoolExpr>),
    /// Logical Implication (A → B).
    Implies(Box<BoolExpr>, Box<BoolExpr>),
    /// Logical Equivalence (A ↔ B).
    Equivalent(Box<BoolExpr>, Box<BoolExpr>),
}

impl BoolExpr {
    pub fn var(name: impl Into<String>) -> Self {
        BoolExpr::Var(Symbol::new(name))
    }

    pub fn and(self, other: BoolExpr) -> Self {
        match (self, other) {
            (BoolExpr::Const(false), _) | (_, BoolExpr::Const(false)) => BoolExpr::Const(false),
            (BoolExpr::Const(true), b) | (b, BoolExpr::Const(true)) => b,
            (BoolExpr::And(mut a), BoolExpr::And(b)) => {
                a.extend(b);
                BoolExpr::And(a)
            }
            (BoolExpr::And(mut a), b) | (b, BoolExpr::And(mut a)) => {
                a.push(b);
                BoolExpr::And(a)
            }
            (a, b) => BoolExpr::And(vec![a, b]),
        }
    }

    pub fn or(self, other: BoolExpr) -> Self {
        match (self, other) {
            (BoolExpr::Const(true), _) | (_, BoolExpr::Const(true)) => BoolExpr::Const(true),
            (BoolExpr::Const(false), b) | (b, BoolExpr::Const(false)) => b,
            (BoolExpr::Or(mut a), BoolExpr::Or(b)) => {
                a.extend(b);
                BoolExpr::Or(a)
            }
            (BoolExpr::Or(mut a), b) | (b, BoolExpr::Or(mut a)) => {
                a.push(b);
                BoolExpr::Or(a)
            }
            (a, b) => BoolExpr::Or(vec![a, b]),
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn not(self) -> Self {
        BoolExpr::Not(Box::new(self))
    }

    pub fn implies(self, other: BoolExpr) -> Self {
        BoolExpr::Implies(Box::new(self), Box::new(other))
    }

    /// `self` as the CONSEQUENT: `other` implies `self`.
    pub fn implied_by(self, other: BoolExpr) -> Self {
        other.implies(self)
    }

    pub fn equiv(self, other: BoolExpr) -> Self {
        BoolExpr::Equivalent(Box::new(self), Box::new(other))
    }

    /// Evaluate expression under variable assignment.
    pub fn evaluate(&self, env: &HashMap<Symbol, bool>) -> Result<bool, LogicError> {
        match self {
            BoolExpr::Const(b) => Ok(*b),
            BoolExpr::Var(s) => env
                .get(s)
                .copied()
                .ok_or_else(|| LogicError::UnassignedVariable(s.name.clone())),
            BoolExpr::Not(inner) => Ok(!inner.evaluate(env)?),
            BoolExpr::And(terms) => {
                for t in terms {
                    if !t.evaluate(env)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            BoolExpr::Or(terms) => {
                for t in terms {
                    if t.evaluate(env)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            BoolExpr::Implies(a, b) => Ok(!a.evaluate(env)? || b.evaluate(env)?),
            BoolExpr::Equivalent(a, b) => Ok(a.evaluate(env)? == b.evaluate(env)?),
        }
    }

    /// Collect all boolean variables in the expression.
    pub fn variables(&self) -> Vec<Symbol> {
        let mut vars = Vec::new();
        self.collect_vars(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_vars(&self, acc: &mut Vec<Symbol>) {
        match self {
            BoolExpr::Const(_) => {}
            BoolExpr::Var(s) => acc.push(s.clone()),
            BoolExpr::Not(inner) => inner.collect_vars(acc),
            BoolExpr::And(terms) | BoolExpr::Or(terms) => {
                for t in terms {
                    t.collect_vars(acc);
                }
            }
            BoolExpr::Implies(a, b) | BoolExpr::Equivalent(a, b) => {
                a.collect_vars(acc);
                b.collect_vars(acc);
            }
        }
    }

    /// Negation normal form: `Implies`/`Equivalent` expanded, `Not` pushed onto literals.
    pub fn nnf(&self) -> BoolExpr {
        fn go(e: &BoolExpr, negated: bool) -> BoolExpr {
            match e {
                BoolExpr::Const(b) => BoolExpr::Const(*b != negated),
                BoolExpr::Var(s) => {
                    if negated {
                        BoolExpr::Not(Box::new(BoolExpr::Var(s.clone())))
                    } else {
                        BoolExpr::Var(s.clone())
                    }
                }
                BoolExpr::Not(inner) => go(inner, !negated),
                BoolExpr::And(terms) => {
                    let parts: Vec<BoolExpr> = terms.iter().map(|t| go(t, negated)).collect();
                    if negated {
                        BoolExpr::Or(parts)
                    } else {
                        BoolExpr::And(parts)
                    }
                }
                BoolExpr::Or(terms) => {
                    let parts: Vec<BoolExpr> = terms.iter().map(|t| go(t, negated)).collect();
                    if negated {
                        BoolExpr::And(parts)
                    } else {
                        BoolExpr::Or(parts)
                    }
                }
                BoolExpr::Implies(a, b) => {
                    // a -> b  ==  ~a | b;   ~(a -> b)  ==  a & ~b
                    let (flip_a, flip_b) = if negated {
                        (false, true)
                    } else {
                        (true, false)
                    };
                    let parts = vec![go(a, flip_a), go(b, flip_b)];
                    if negated {
                        BoolExpr::And(parts)
                    } else {
                        BoolExpr::Or(parts)
                    }
                }
                BoolExpr::Equivalent(a, b) => {
                    // a <-> b  ==  (a & b) | (~a & ~b)
                    // ~(a <-> b)  ==  (a & ~b) | (~a & b)
                    let (a1, b1, a2, b2) = if negated {
                        (false, true, true, false)
                    } else {
                        (false, false, true, true)
                    };
                    BoolExpr::Or(vec![
                        BoolExpr::And(vec![go(a, a1), go(b, b1)]),
                        BoolExpr::And(vec![go(a, a2), go(b, b2)]),
                    ])
                }
            }
        }
        go(self, false)
    }

    /// Exhaustive truth table over the sorted variable set.
    ///
    /// Fails with [`LogicError::TableTooLarge`] beyond 20 variables.
    pub fn truth_table(&self) -> Result<TruthTable, LogicError> {
        TruthTable::of(self)
    }

    /// Canonical CNF: conjunction of maxterms, one per falsifying assignment.
    /// A tautology yields the empty conjunction; a contradiction yields a
    /// single empty clause.
    pub fn to_cnf(&self) -> Result<Cnf, LogicError> {
        let tt = self.truth_table()?;
        let falsifying: Vec<&TruthRow> = tt.rows.iter().filter(|r| !r.value).collect();
        // A contradiction collapses to a single empty clause rather than
        // one maxterm per assignment.
        if !tt.rows.is_empty() && falsifying.len() == tt.rows.len() {
            return Ok(vec![Vec::new()]);
        }
        Ok(falsifying.iter().map(|r| maxterm(r, &tt.vars)).collect())
    }

    /// Canonical DNF: disjunction of minterms, one per satisfying assignment.
    /// A contradiction yields the empty disjunction.
    pub fn to_dnf(&self) -> Result<Cnf, LogicError> {
        let tt = self.truth_table()?;
        Ok(tt
            .rows
            .iter()
            .filter(|r| r.value)
            .map(|r| minterm(r, &tt.vars))
            .collect())
    }
}

impl std::ops::Not for BoolExpr {
    type Output = BoolExpr;

    fn not(self) -> BoolExpr {
        match self {
            BoolExpr::Const(b) => BoolExpr::Const(!b),
            BoolExpr::Not(inner) => *inner,
            other => BoolExpr::Not(Box::new(other)),
        }
    }
}

impl fmt::Display for BoolExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BoolExpr::Const(b) => write!(f, "{}", if *b { "True" } else { "False" }),
            BoolExpr::Var(s) => write!(f, "{}", s),
            BoolExpr::Not(inner) => write!(f, "~{}", inner),
            BoolExpr::And(terms) => {
                let s = terms
                    .iter()
                    .map(|t| format!("{}", t))
                    .collect::<Vec<_>>()
                    .join(" & ");
                write!(f, "({})", s)
            }
            BoolExpr::Or(terms) => {
                let s = terms
                    .iter()
                    .map(|t| format!("{}", t))
                    .collect::<Vec<_>>()
                    .join(" | ");
                write!(f, "({})", s)
            }
            BoolExpr::Implies(a, b) => write!(f, "({} >> {})", a, b),
            BoolExpr::Equivalent(a, b) => write!(f, "({} <=> {})", a, b),
        }
    }
}

/// A signed boolean literal: a variable or its negation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Literal {
    Pos(Symbol),
    Neg(Symbol),
}

impl Literal {
    /// The underlying variable of the literal.
    pub fn variable(&self) -> &Symbol {
        match self {
            Literal::Pos(s) | Literal::Neg(s) => s,
        }
    }

    /// Whether the literal is unnegated.
    pub fn is_positive(&self) -> bool {
        matches!(self, Literal::Pos(_))
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Pos(s) => write!(f, "{}", s),
            Literal::Neg(s) => write!(f, "~{}", s),
        }
    }
}

/// Clause set shape shared by both normal forms: conjunction (CNF) or
/// disjunction (DNF) of literal groups.
pub type Cnf = Vec<Vec<Literal>>;

fn maxterm(row: &TruthRow, vars: &[Symbol]) -> Vec<Literal> {
    row.assignment
        .iter()
        .zip(vars)
        .map(|(&v, s)| {
            if v {
                Literal::Neg(s.clone())
            } else {
                Literal::Pos(s.clone())
            }
        })
        .collect()
}

fn minterm(row: &TruthRow, vars: &[Symbol]) -> Vec<Literal> {
    row.assignment
        .iter()
        .zip(vars)
        .map(|(&v, s)| {
            if v {
                Literal::Pos(s.clone())
            } else {
                Literal::Neg(s.clone())
            }
        })
        .collect()
}

/// Exhaustive truth table over an expression's sorted variable set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TruthTable {
    pub vars: Vec<Symbol>,
    pub rows: Vec<TruthRow>,
}

/// One assignment row of a [`TruthTable`]: per-variable values aligned with
/// `TruthTable::vars`, plus the expression's truth value under that assignment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TruthRow {
    pub assignment: Vec<bool>,
    pub value: bool,
}

impl TruthTable {
    /// Build the table for `expr`, rejecting more than 20 variables.
    pub fn of(expr: &BoolExpr) -> Result<Self, LogicError> {
        const MAX_VARS: usize = 20;
        let vars = expr.variables();
        if vars.len() > MAX_VARS {
            return Err(LogicError::TableTooLarge(vars.len()));
        }
        let mut rows = Vec::with_capacity(1 << vars.len());
        for mask in 0u32..(1u32 << vars.len()) {
            let assignment: Vec<bool> = (0..vars.len()).map(|i| (mask >> i) & 1 == 1).collect();
            let env: HashMap<Symbol, bool> = vars
                .iter()
                .cloned()
                .zip(assignment.iter().copied())
                .collect();
            let value = expr.evaluate(&env)?;
            rows.push(TruthRow { assignment, value });
        }
        Ok(TruthTable { vars, rows })
    }

    /// Whether every assignment satisfies the expression.
    pub fn is_tautology(&self) -> bool {
        self.rows.iter().all(|r| r.value)
    }

    /// Whether no assignment satisfies the expression.
    pub fn is_contradiction(&self) -> bool {
        self.rows.iter().all(|r| !r.value)
    }
}

/// Decide satisfiability with DPLL (unit propagation + pure-literal elimination).
///
/// Returns one satisfying assignment when one exists.
pub fn dpll_satisfiable(expr: &BoolExpr) -> Option<HashMap<Symbol, bool>> {
    let mut model = dpll(&structural_cnf(&expr.nnf()), HashMap::new())?;
    for var in expr.variables() {
        model.entry(var).or_insert(false);
    }
    Some(model)
}

/// Convert an NNF formula to CNF by distribution.
fn structural_cnf(e: &BoolExpr) -> Cnf {
    match e {
        BoolExpr::Const(true) => Cnf::new(),
        BoolExpr::Const(false) => vec![Vec::new()],
        BoolExpr::Var(s) => vec![vec![Literal::Pos(s.clone())]],
        BoolExpr::Not(inner) => match inner.as_ref() {
            BoolExpr::Var(s) => vec![vec![Literal::Neg(s.clone())]],
            // Post-NNF, Not wraps literals only.
            other => structural_cnf(other),
        },
        BoolExpr::And(terms) => terms.iter().flat_map(structural_cnf).collect(),
        BoolExpr::Or(terms) => terms
            .iter()
            .map(structural_cnf)
            .reduce(|acc, next| {
                let mut out = Vec::with_capacity(acc.len() * next.len());
                for a in &acc {
                    for c in &next {
                        let mut merged = a.clone();
                        merged.extend(c.iter().cloned());
                        out.push(merged);
                    }
                }
                out
            })
            .unwrap_or_else(|| vec![Vec::new()]),
        // NNF removes these connectives before conversion.
        BoolExpr::Implies(..) | BoolExpr::Equivalent(..) => {
            unreachable!("nnf() eliminates Implies/Equivalent")
        }
    }
}

/// Clause value under a partial model: `Some(true)` satisfied, `Some(false)`
/// falsified (all literals assigned false), `None` undetermined.
fn clause_value(clause: &[Literal], model: &HashMap<Symbol, bool>) -> Option<bool> {
    // Self-contained literal evaluation: `Some(true)` once any literal
    // holds, `Some(false)` when every literal is assigned and false,
    // `None` while the clause is still undetermined.
    let mut all_assigned_false = true;
    for lit in clause {
        match model.get(lit.variable()) {
            Some(&v) if v == lit.is_positive() => return Some(true),
            Some(_) => {}
            None => all_assigned_false = false,
        }
    }
    if all_assigned_false {
        Some(false)
    } else {
        None
    }
}

fn unit_literal(clause: &[Literal], model: &HashMap<Symbol, bool>) -> Option<Literal> {
    if clause_value(clause, model) == Some(true) {
        return None;
    }
    let mut unassigned: Option<&Literal> = None;
    for lit in clause {
        if !model.contains_key(lit.variable()) {
            if unassigned.is_some() {
                return None;
            }
            unassigned = Some(lit);
        }
    }
    unassigned.cloned()
}

fn dpll(clauses: &[Vec<Literal>], model: HashMap<Symbol, bool>) -> Option<HashMap<Symbol, bool>> {
    if clauses
        .iter()
        .any(|c| clause_value(c, &model) == Some(false))
    {
        return None;
    }
    if clauses
        .iter()
        .all(|c| clause_value(c, &model) == Some(true))
    {
        return Some(model);
    }

    // Unit propagation.
    if let Some(lit) = clauses.iter().find_map(|c| unit_literal(c, &model)) {
        let mut m = model;
        m.insert(lit.variable().clone(), lit.is_positive());
        return dpll(clauses, m);
    }

    // Pure-literal elimination on unassigned variables in unresolved clauses.
    let mut seen = HashMap::<&Symbol, (bool, bool)>::new(); // (pos_seen, neg_seen)
    for clause in clauses {
        if clause_value(clause, &model) == Some(true) {
            continue;
        }
        for lit in clause {
            if !model.contains_key(lit.variable()) {
                let entry = seen.entry(lit.variable()).or_insert((false, false));
                if lit.is_positive() {
                    entry.0 = true;
                } else {
                    entry.1 = true;
                }
            }
        }
    }
    for (&var, &(pos_seen, neg_seen)) in seen.iter() {
        if pos_seen != neg_seen {
            let mut m = model;
            m.insert(var.clone(), pos_seen);
            return dpll(clauses, m);
        }
    }

    // Branch on the first unassigned variable.
    let var = clauses
        .iter()
        .flat_map(|c| c.iter())
        .map(|lit| lit.variable())
        .find(|s| !model.contains_key(*s))?
        .clone();
    for branch_value in [true, false] {
        let mut m = model.clone();
        m.insert(var.clone(), branch_value);
        if let Some(solution) = dpll(clauses, m) {
            return Some(solution);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logic_eval() {
        let p = Symbol::new("p");
        let q = Symbol::new("q");
        let expr = BoolExpr::var("p").and(!BoolExpr::var("q"));
        let mut env = HashMap::new();
        env.insert(p.clone(), true);
        env.insert(q.clone(), false);
        assert_eq!(expr.evaluate(&env), Ok(true));
    }

    fn env_of(vars: &[Symbol], assignment: &[bool]) -> HashMap<Symbol, bool> {
        vars.iter()
            .cloned()
            .zip(assignment.iter().copied())
            .collect()
    }

    /// Rebuild a BoolExpr from a clause set for equivalence checks.
    fn clauses_to_expr(clauses: &Cnf) -> BoolExpr {
        let disjunctions: Vec<BoolExpr> = clauses
            .iter()
            .map(|clause| {
                let lits: Vec<BoolExpr> = clause
                    .iter()
                    .map(|lit| match lit {
                        Literal::Pos(s) => BoolExpr::Var(s.clone()),
                        Literal::Neg(s) => !BoolExpr::Var(s.clone()),
                    })
                    .collect();
                lits.into_iter()
                    .reduce(|a, b| a.or(b))
                    .unwrap_or(BoolExpr::Const(false))
            })
            .collect();
        disjunctions
            .into_iter()
            .reduce(|a, b| a.and(b))
            .unwrap_or(BoolExpr::Const(true))
    }

    /// Mirror of [`clauses_to_expr`] for DNF shape: OR over groups,
    /// each group an AND of literals; empty DNF is false.
    fn groups_to_expr(groups: &Cnf) -> BoolExpr {
        let conjunctions: Vec<BoolExpr> = groups
            .iter()
            .map(|group| {
                let lits: Vec<BoolExpr> = group
                    .iter()
                    .map(|lit| match lit {
                        Literal::Pos(s) => BoolExpr::Var(s.clone()),
                        Literal::Neg(s) => !BoolExpr::Var(s.clone()),
                    })
                    .collect();
                lits.into_iter()
                    .reduce(|a, b| a.and(b))
                    .unwrap_or(BoolExpr::Const(true))
            })
            .collect();
        conjunctions
            .into_iter()
            .reduce(|a, b| a.or(b))
            .unwrap_or(BoolExpr::Const(false))
    }

    #[test]
    fn test_truth_table_rows() {
        // p & ~q: exactly one satisfying assignment.
        let expr = BoolExpr::var("p").and(!BoolExpr::var("q"));
        let tt = expr.truth_table().unwrap();
        assert_eq!(tt.vars.len(), 2);
        assert_eq!(tt.rows.len(), 4);
        assert_eq!(tt.rows.iter().filter(|r| r.value).count(), 1);
    }
    #[test]
    fn test_truth_table_size_limit() {
        let expr: BoolExpr = (0..21)
            .map(|i| BoolExpr::var(format!("v{i}")))
            .reduce(|a, b| a.and(b))
            .unwrap();
        assert!(matches!(
            expr.truth_table(),
            Err(LogicError::TableTooLarge(21))
        ));
    }

    #[test]
    fn test_tautology_and_contradiction() {
        let p = BoolExpr::var("p");
        let taut = p.clone().or(!p.clone());
        assert!(taut.truth_table().unwrap().is_tautology());
        assert_eq!(taut.to_cnf().unwrap(), Vec::<Vec<Literal>>::new());
        assert!(dpll_satisfiable(&taut).is_some());

        let contra = p.clone().and(!p);
        assert!(contra.truth_table().unwrap().is_contradiction());
        assert_eq!(contra.to_cnf().unwrap(), vec![Vec::<Literal>::new()]);
        assert!(dpll_satisfiable(&contra).is_none());
    }

    #[test]
    fn test_cnf_dnf_equivalence_with_original() {
        // Mixed connectives: ((p -> q) <=> (~r | p)) & ~(p -> r)
        let expr = BoolExpr::Equivalent(
            Box::new(BoolExpr::Implies(
                Box::new(BoolExpr::var("p")),
                Box::new(BoolExpr::var("q")),
            )),
            Box::new((!BoolExpr::var("r")).or(BoolExpr::var("p"))),
        )
        .and(!BoolExpr::Implies(
            Box::new(BoolExpr::var("p")),
            Box::new(BoolExpr::var("r")),
        ));
        let vars = expr.variables();
        let cnf_expr = clauses_to_expr(&expr.to_cnf().unwrap());
        let dnf_expr = groups_to_expr(&expr.to_dnf().unwrap());
        for mask in 0u32..(1u32 << vars.len()) {
            let assignment: Vec<bool> = (0..vars.len()).map(|i| (mask >> i) & 1 == 1).collect();
            let env = env_of(&vars, &assignment);
            let expected = expr.evaluate(&env);
            assert_eq!(cnf_expr.evaluate(&env), expected);
            assert_eq!(dnf_expr.evaluate(&env), expected);
        }
    }

    #[test]
    fn test_dpll_matches_bruteforce() {
        // (p -> q) & p & ~q is unsatisfiable.
        let unsat = BoolExpr::Implies(Box::new(BoolExpr::var("p")), Box::new(BoolExpr::var("q")))
            .and(BoolExpr::var("p"))
            .and(!BoolExpr::var("q"));
        assert!(dpll_satisfiable(&unsat).is_none());

        // (p | q) & (~p | r) & (q | r) is satisfiable.
        let sat = BoolExpr::var("p")
            .or(BoolExpr::var("q"))
            .and((!BoolExpr::var("p")).or(BoolExpr::var("r")))
            .and(BoolExpr::var("q").or(BoolExpr::var("r")));
        let model = dpll_satisfiable(&sat).expect("expected a model");
        assert_eq!(sat.evaluate(&model), Ok(true));
    }
}
