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
    #[error("Boolean logic resource limit exceeded for {resource}: {actual} > {limit}")]
    SolverLimitExceeded {
        resource: &'static str,
        actual: usize,
        limit: usize,
    },
    #[error("SAT solver invariant violation: {0}")]
    SolverInvariantViolation(String),
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

    /// Logical XOR (A ⊕ B).
    pub fn xor(self, other: BoolExpr) -> Self {
        self.equiv(other).not()
    }

    /// Logical NAND (¬(A ∧ B)).
    pub fn nand(self, other: BoolExpr) -> Self {
        self.and(other).not()
    }

    /// Logical NOR (¬(A ∨ B)).
    pub fn nor(self, other: BoolExpr) -> Self {
        self.or(other).not()
    }

    /// Logical XNOR (¬(A ⊕ B) ≡ A ↔ B).
    pub fn xnor(self, other: BoolExpr) -> Self {
        self.equiv(other)
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
    /// Fails with [`LogicError::TableTooLarge`] beyond 20 variables and with
    /// [`LogicError::SolverLimitExceeded`] when the formula shape or the
    /// worst-case evaluation work exceeds the native safety limits.
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
    /// Build the table for `expr`, rejecting structurally unsafe or excessive work.
    pub fn of(expr: &BoolExpr) -> Result<Self, LogicError> {
        let formula_nodes = validate_formula_shape(expr)?;
        let vars = expr.variables();
        if vars.len() > MAX_TRUTH_TABLE_VARIABLES {
            return Err(LogicError::TableTooLarge(vars.len()));
        }
        let row_count = 1usize << vars.len();
        let evaluation_work =
            row_count
                .checked_mul(formula_nodes)
                .ok_or(LogicError::SolverLimitExceeded {
                    resource: "truth-table evaluation work",
                    actual: usize::MAX,
                    limit: MAX_TRUTH_TABLE_EVALUATION_WORK,
                })?;
        if evaluation_work > MAX_TRUTH_TABLE_EVALUATION_WORK {
            return Err(LogicError::SolverLimitExceeded {
                resource: "truth-table evaluation work",
                actual: evaluation_work,
                limit: MAX_TRUTH_TABLE_EVALUATION_WORK,
            });
        }

        let mut rows = Vec::with_capacity(row_count);
        for mask in 0..row_count {
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

const MAX_FORMULA_DEPTH: usize = 256;
const MAX_FORMULA_NODES: usize = 65_536;
const MAX_TRUTH_TABLE_VARIABLES: usize = 20;
// Worst-case source-formula node visits across all assignments.
const MAX_TRUTH_TABLE_EVALUATION_WORK: usize = 33_554_432;
const MAX_DPLL_VARIABLES: usize = 256;
const MAX_DPLL_CLAUSES: usize = 65_536;
const MAX_DPLL_CLAUSE_LITERALS: usize = 65_536;
const MAX_DPLL_SEARCH_WORK: usize = 16_777_216;

fn validate_formula_shape(expr: &BoolExpr) -> Result<usize, LogicError> {
    let mut visited = 0usize;
    let mut stack = vec![(expr, 0usize)];

    while let Some((current, depth)) = stack.pop() {
        if depth > MAX_FORMULA_DEPTH {
            return Err(LogicError::SolverLimitExceeded {
                resource: "formula depth",
                actual: depth,
                limit: MAX_FORMULA_DEPTH,
            });
        }
        visited = visited
            .checked_add(1)
            .ok_or(LogicError::SolverLimitExceeded {
                resource: "formula nodes",
                actual: usize::MAX,
                limit: MAX_FORMULA_NODES,
            })?;
        if visited > MAX_FORMULA_NODES {
            return Err(LogicError::SolverLimitExceeded {
                resource: "formula nodes",
                actual: visited,
                limit: MAX_FORMULA_NODES,
            });
        }

        let child_count = match current {
            BoolExpr::Const(_) | BoolExpr::Var(_) => 0,
            BoolExpr::Not(_) => 1,
            BoolExpr::And(terms) | BoolExpr::Or(terms) => terms.len(),
            BoolExpr::Implies(_, _) | BoolExpr::Equivalent(_, _) => 2,
        };
        let admitted_nodes = visited
            .checked_add(stack.len())
            .and_then(|count| count.checked_add(child_count))
            .ok_or(LogicError::SolverLimitExceeded {
                resource: "formula nodes",
                actual: usize::MAX,
                limit: MAX_FORMULA_NODES,
            })?;
        if admitted_nodes > MAX_FORMULA_NODES {
            return Err(LogicError::SolverLimitExceeded {
                resource: "formula nodes",
                actual: admitted_nodes,
                limit: MAX_FORMULA_NODES,
            });
        }

        if child_count == 0 {
            continue;
        }
        let child_depth = depth
            .checked_add(1)
            .ok_or(LogicError::SolverLimitExceeded {
                resource: "formula depth",
                actual: usize::MAX,
                limit: MAX_FORMULA_DEPTH,
            })?;
        match current {
            BoolExpr::Not(inner) => stack.push((inner, child_depth)),
            BoolExpr::And(terms) | BoolExpr::Or(terms) => {
                stack.extend(terms.iter().map(|term| (term, child_depth)));
            }
            BoolExpr::Implies(left, right) | BoolExpr::Equivalent(left, right) => {
                stack.push((left, child_depth));
                stack.push((right, child_depth));
            }
            BoolExpr::Const(_) | BoolExpr::Var(_) => {}
        }
    }

    Ok(visited)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct SatVar(usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SatLiteral {
    variable: SatVar,
    positive: bool,
}

impl SatLiteral {
    fn positive(variable: SatVar) -> Self {
        Self {
            variable,
            positive: true,
        }
    }

    fn negative(variable: SatVar) -> Self {
        Self {
            variable,
            positive: false,
        }
    }
}

struct TseitinEncoder<'a> {
    clauses: Vec<Vec<SatLiteral>>,
    original_variables: HashMap<Symbol, SatVar>,
    memo: HashMap<&'a BoolExpr, SatVar>,
    variable_count: usize,
}

impl<'a> TseitinEncoder<'a> {
    fn new() -> Self {
        Self {
            clauses: Vec::new(),
            original_variables: HashMap::new(),
            memo: HashMap::new(),
            variable_count: 0,
        }
    }

    fn new_variable(&mut self) -> Result<SatVar, LogicError> {
        let actual = self
            .variable_count
            .checked_add(1)
            .ok_or(LogicError::SolverLimitExceeded {
                resource: "variables",
                actual: usize::MAX,
                limit: MAX_DPLL_VARIABLES,
            })?;
        if actual > MAX_DPLL_VARIABLES {
            return Err(LogicError::SolverLimitExceeded {
                resource: "variables",
                actual,
                limit: MAX_DPLL_VARIABLES,
            });
        }
        let variable = SatVar(self.variable_count);
        self.variable_count = actual;
        Ok(variable)
    }

    fn push_clause(&mut self, clause: Vec<SatLiteral>) -> Result<(), LogicError> {
        if clause.len() > MAX_DPLL_CLAUSE_LITERALS {
            return Err(LogicError::SolverLimitExceeded {
                resource: "literals per clause",
                actual: clause.len(),
                limit: MAX_DPLL_CLAUSE_LITERALS,
            });
        }
        let actual = self
            .clauses
            .len()
            .checked_add(1)
            .ok_or(LogicError::SolverLimitExceeded {
                resource: "clauses",
                actual: usize::MAX,
                limit: MAX_DPLL_CLAUSES,
            })?;
        if actual > MAX_DPLL_CLAUSES {
            return Err(LogicError::SolverLimitExceeded {
                resource: "clauses",
                actual,
                limit: MAX_DPLL_CLAUSES,
            });
        }
        self.clauses.push(clause);
        Ok(())
    }

    fn encode(&mut self, expr: &'a BoolExpr, depth: usize) -> Result<SatVar, LogicError> {
        if depth > MAX_FORMULA_DEPTH {
            return Err(LogicError::SolverLimitExceeded {
                resource: "formula depth",
                actual: depth,
                limit: MAX_FORMULA_DEPTH,
            });
        }
        if let Some(&variable) = self.memo.get(expr) {
            return Ok(variable);
        }

        let variable = match expr {
            BoolExpr::Var(symbol) => {
                if let Some(&variable) = self.original_variables.get(symbol) {
                    variable
                } else {
                    let variable = self.new_variable()?;
                    self.original_variables.insert(symbol.clone(), variable);
                    variable
                }
            }
            BoolExpr::Const(value) => {
                let variable = self.new_variable()?;
                self.push_clause(vec![if *value {
                    SatLiteral::positive(variable)
                } else {
                    SatLiteral::negative(variable)
                }])?;
                variable
            }
            BoolExpr::Not(inner) => {
                let inner = self.encode(inner, depth + 1)?;
                let variable = self.new_variable()?;
                self.push_clause(vec![
                    SatLiteral::negative(variable),
                    SatLiteral::negative(inner),
                ])?;
                self.push_clause(vec![
                    SatLiteral::positive(variable),
                    SatLiteral::positive(inner),
                ])?;
                variable
            }
            BoolExpr::And(terms) | BoolExpr::Or(terms) => {
                if terms.len() > MAX_DPLL_CLAUSE_LITERALS {
                    return Err(LogicError::SolverLimitExceeded {
                        resource: "operator fanout",
                        actual: terms.len(),
                        limit: MAX_DPLL_CLAUSE_LITERALS,
                    });
                }
                let mut children = Vec::with_capacity(terms.len());
                for term in terms {
                    children.push(self.encode(term, depth + 1)?);
                }
                let variable = self.new_variable()?;
                match expr {
                    BoolExpr::And(_) => {
                        for &child in &children {
                            self.push_clause(vec![
                                SatLiteral::negative(variable),
                                SatLiteral::positive(child),
                            ])?;
                        }
                        let mut clause = Vec::with_capacity(children.len() + 1);
                        clause.push(SatLiteral::positive(variable));
                        clause.extend(children.into_iter().map(SatLiteral::negative));
                        self.push_clause(clause)?;
                    }
                    BoolExpr::Or(_) => {
                        for &child in &children {
                            self.push_clause(vec![
                                SatLiteral::positive(variable),
                                SatLiteral::negative(child),
                            ])?;
                        }
                        let mut clause = Vec::with_capacity(children.len() + 1);
                        clause.push(SatLiteral::negative(variable));
                        clause.extend(children.into_iter().map(SatLiteral::positive));
                        self.push_clause(clause)?;
                    }
                    _ => {
                        return Err(LogicError::SolverInvariantViolation(
                            "n-ary encoder reached a non-n-ary expression".to_string(),
                        ));
                    }
                }
                variable
            }
            BoolExpr::Implies(antecedent, consequent) => {
                let antecedent = self.encode(antecedent, depth + 1)?;
                let consequent = self.encode(consequent, depth + 1)?;
                let variable = self.new_variable()?;
                self.push_clause(vec![
                    SatLiteral::positive(variable),
                    SatLiteral::positive(antecedent),
                ])?;
                self.push_clause(vec![
                    SatLiteral::positive(variable),
                    SatLiteral::negative(consequent),
                ])?;
                self.push_clause(vec![
                    SatLiteral::negative(variable),
                    SatLiteral::negative(antecedent),
                    SatLiteral::positive(consequent),
                ])?;
                variable
            }
            BoolExpr::Equivalent(left, right) => {
                let left = self.encode(left, depth + 1)?;
                let right = self.encode(right, depth + 1)?;
                let variable = self.new_variable()?;
                self.push_clause(vec![
                    SatLiteral::negative(variable),
                    SatLiteral::negative(left),
                    SatLiteral::positive(right),
                ])?;
                self.push_clause(vec![
                    SatLiteral::negative(variable),
                    SatLiteral::positive(left),
                    SatLiteral::negative(right),
                ])?;
                self.push_clause(vec![
                    SatLiteral::positive(variable),
                    SatLiteral::positive(left),
                    SatLiteral::positive(right),
                ])?;
                self.push_clause(vec![
                    SatLiteral::positive(variable),
                    SatLiteral::negative(left),
                    SatLiteral::negative(right),
                ])?;
                variable
            }
        };
        self.memo.insert(expr, variable);
        Ok(variable)
    }
}

/// Decide satisfiability with a bounded Tseitin encoding and DPLL.
///
/// Returns one satisfying assignment when one exists, `Ok(None)` only for an
/// established UNSAT result, and a typed error when the solver's structural
/// or search-work limits refuse the input.
pub fn dpll_satisfiable(expr: &BoolExpr) -> Result<Option<HashMap<Symbol, bool>>, LogicError> {
    dpll_with_root_value(expr, true)
}

/// Search for a model in which the source formula has `required_value`.
///
/// Constraining the encoder's existing root avoids materializing a transformed
/// copy of the source formula and keeps SAT admission independent of wrapper
/// syntax introduced by the caller.
fn dpll_with_root_value(
    expr: &BoolExpr,
    required_value: bool,
) -> Result<Option<HashMap<Symbol, bool>>, LogicError> {
    validate_formula_shape(expr)?;
    let mut encoder = TseitinEncoder::new();
    let root = encoder.encode(expr, 0)?;
    encoder.push_clause(vec![if required_value {
        SatLiteral::positive(root)
    } else {
        SatLiteral::negative(root)
    }])?;
    let mut search_budget = SearchBudget::new(MAX_DPLL_SEARCH_WORK);
    let Some(model) = dpll(
        &encoder.clauses,
        vec![None; encoder.variable_count],
        &mut search_budget,
    )?
    else {
        return Ok(None);
    };

    let mut public_model = HashMap::with_capacity(encoder.original_variables.len());
    for (symbol, variable) in encoder.original_variables {
        public_model.insert(symbol, model_value(&model, variable)?.unwrap_or(false));
    }
    if expr.evaluate(&public_model)? != required_value {
        return Err(LogicError::SolverInvariantViolation(
            "Tseitin/DPLL model does not satisfy the required source-formula value".to_string(),
        ));
    }
    Ok(Some(public_model))
}

/// Returns `true` if the formula is satisfiable, `false` if unsatisfiable.
pub fn is_satisfiable(expr: &BoolExpr) -> Result<bool, LogicError> {
    Ok(dpll_satisfiable(expr)?.is_some())
}

/// Returns `true` if the formula is valid (a tautology), `false` otherwise.
pub fn is_valid(expr: &BoolExpr) -> Result<bool, LogicError> {
    Ok(dpll_with_root_value(expr, false)?.is_none())
}

/// Returns `true` if the formula is a contradiction (unsatisfiable), `false` otherwise.
pub fn is_contradiction_sat(expr: &BoolExpr) -> Result<bool, LogicError> {
    Ok(dpll_satisfiable(expr)?.is_none())
}

fn is_negation_of(a: &BoolExpr, b: &BoolExpr) -> bool {
    match (a, b) {
        (BoolExpr::Not(inner), other) | (other, BoolExpr::Not(inner)) => inner.as_ref() == other,
        _ => false,
    }
}

/// Algebraic simplification of propositional logic formulas.
pub fn simplify_logic(expr: &BoolExpr) -> BoolExpr {
    match expr {
        BoolExpr::Const(b) => BoolExpr::Const(*b),
        BoolExpr::Var(s) => BoolExpr::Var(s.clone()),
        BoolExpr::Not(inner) => {
            let simplified = simplify_logic(inner);
            match simplified {
                BoolExpr::Const(b) => BoolExpr::Const(!b),
                BoolExpr::Not(sub) => *sub,
                other => BoolExpr::Not(Box::new(other)),
            }
        }
        BoolExpr::And(terms) => {
            let mut flat = Vec::new();
            for term in terms {
                let s = simplify_logic(term);
                match s {
                    BoolExpr::And(sub) => flat.extend(sub),
                    other => flat.push(other),
                }
            }
            if flat.iter().any(|t| matches!(t, BoolExpr::Const(false))) {
                return BoolExpr::Const(false);
            }
            flat.retain(|t| !matches!(t, BoolExpr::Const(true)));
            if flat.is_empty() {
                return BoolExpr::Const(true);
            }
            let mut deduped = Vec::new();
            for item in flat {
                if !deduped.contains(&item) {
                    deduped.push(item);
                }
            }
            for i in 0..deduped.len() {
                for j in (i + 1)..deduped.len() {
                    if is_negation_of(&deduped[i], &deduped[j]) {
                        return BoolExpr::Const(false);
                    }
                }
            }
            if deduped.len() == 1 {
                deduped.pop().unwrap()
            } else {
                BoolExpr::And(deduped)
            }
        }
        BoolExpr::Or(terms) => {
            let mut flat = Vec::new();
            for term in terms {
                let s = simplify_logic(term);
                match s {
                    BoolExpr::Or(sub) => flat.extend(sub),
                    other => flat.push(other),
                }
            }
            if flat.iter().any(|t| matches!(t, BoolExpr::Const(true))) {
                return BoolExpr::Const(true);
            }
            flat.retain(|t| !matches!(t, BoolExpr::Const(false)));
            if flat.is_empty() {
                return BoolExpr::Const(false);
            }
            let mut deduped = Vec::new();
            for item in flat {
                if !deduped.contains(&item) {
                    deduped.push(item);
                }
            }
            for i in 0..deduped.len() {
                for j in (i + 1)..deduped.len() {
                    if is_negation_of(&deduped[i], &deduped[j]) {
                        return BoolExpr::Const(true);
                    }
                }
            }
            if deduped.len() == 1 {
                deduped.pop().unwrap()
            } else {
                BoolExpr::Or(deduped)
            }
        }
        BoolExpr::Implies(a, b) => {
            let sa = simplify_logic(a);
            let sb = simplify_logic(b);
            simplify_logic(&BoolExpr::Or(vec![BoolExpr::Not(Box::new(sa)), sb]))
        }
        BoolExpr::Equivalent(a, b) => {
            let sa = simplify_logic(a);
            let sb = simplify_logic(b);
            if sa == sb {
                return BoolExpr::Const(true);
            }
            if is_negation_of(&sa, &sb) {
                return BoolExpr::Const(false);
            }
            match (&sa, &sb) {
                (BoolExpr::Const(true), other) | (other, BoolExpr::Const(true)) => other.clone(),
                (BoolExpr::Const(false), other) | (other, BoolExpr::Const(false)) => {
                    simplify_logic(&BoolExpr::Not(Box::new(other.clone())))
                }
                _ => BoolExpr::Equivalent(Box::new(sa), Box::new(sb)),
            }
        }
    }
}

struct SearchBudget {
    used: usize,
    limit: usize,
}

impl SearchBudget {
    fn new(limit: usize) -> Self {
        Self { used: 0, limit }
    }

    fn charge(&mut self, amount: usize) -> Result<(), LogicError> {
        let actual = self
            .used
            .checked_add(amount)
            .ok_or(LogicError::SolverLimitExceeded {
                resource: "search work units",
                actual: usize::MAX,
                limit: self.limit,
            })?;
        if actual > self.limit {
            return Err(LogicError::SolverLimitExceeded {
                resource: "search work units",
                actual,
                limit: self.limit,
            });
        }
        self.used = actual;
        Ok(())
    }
}

fn model_value(model: &[Option<bool>], variable: SatVar) -> Result<Option<bool>, LogicError> {
    model.get(variable.0).copied().ok_or_else(|| {
        LogicError::SolverInvariantViolation(format!(
            "SAT variable {} is outside model length {}",
            variable.0,
            model.len()
        ))
    })
}

fn assign_model(
    model: &mut [Option<bool>],
    variable: SatVar,
    value: bool,
) -> Result<(), LogicError> {
    let model_len = model.len();
    let slot = model.get_mut(variable.0).ok_or_else(|| {
        LogicError::SolverInvariantViolation(format!(
            "SAT variable {} is outside model length {model_len}",
            variable.0
        ))
    })?;
    *slot = Some(value);
    Ok(())
}

/// Clause value under a partial model: `Some(true)` satisfied, `Some(false)`
/// falsified (all literals assigned false), `None` undetermined.
fn clause_value(
    clause: &[SatLiteral],
    model: &[Option<bool>],
    budget: &mut SearchBudget,
) -> Result<Option<bool>, LogicError> {
    budget.charge(clause.len().max(1))?;
    let mut all_assigned_false = true;
    for literal in clause {
        match model_value(model, literal.variable)? {
            Some(value) if value == literal.positive => return Ok(Some(true)),
            Some(_) => {}
            None => all_assigned_false = false,
        }
    }
    Ok(all_assigned_false.then_some(false))
}

fn unit_literal(
    clause: &[SatLiteral],
    model: &[Option<bool>],
    budget: &mut SearchBudget,
) -> Result<Option<SatLiteral>, LogicError> {
    if clause_value(clause, model, budget)? == Some(true) {
        return Ok(None);
    }
    budget.charge(clause.len().max(1))?;
    let mut unassigned = None;
    for &literal in clause {
        if model_value(model, literal.variable)?.is_none() {
            if unassigned.is_some() {
                return Ok(None);
            }
            unassigned = Some(literal);
        }
    }
    Ok(unassigned)
}

fn dpll(
    clauses: &[Vec<SatLiteral>],
    model: Vec<Option<bool>>,
    budget: &mut SearchBudget,
) -> Result<Option<Vec<Option<bool>>>, LogicError> {
    budget.charge(1)?;
    for clause in clauses {
        if clause_value(clause, &model, budget)? == Some(false) {
            return Ok(None);
        }
    }

    let mut all_satisfied = true;
    for clause in clauses {
        if clause_value(clause, &model, budget)? != Some(true) {
            all_satisfied = false;
            break;
        }
    }
    if all_satisfied {
        return Ok(Some(model));
    }

    for clause in clauses {
        if let Some(literal) = unit_literal(clause, &model, budget)? {
            let mut next = model;
            assign_model(&mut next, literal.variable, literal.positive)?;
            return dpll(clauses, next, budget);
        }
    }

    budget.charge(model.len().max(1))?;
    let mut seen = vec![(false, false); model.len()];
    for clause in clauses {
        if clause_value(clause, &model, budget)? == Some(true) {
            continue;
        }
        budget.charge(clause.len().max(1))?;
        for literal in clause {
            if model_value(&model, literal.variable)?.is_none() {
                let seen_len = seen.len();
                let entry = seen.get_mut(literal.variable.0).ok_or_else(|| {
                    LogicError::SolverInvariantViolation(format!(
                        "SAT variable {} is outside polarity table length {seen_len}",
                        literal.variable.0
                    ))
                })?;
                if literal.positive {
                    entry.0 = true;
                } else {
                    entry.1 = true;
                }
            }
        }
    }
    if let Some((variable, &(positive, _))) = seen
        .iter()
        .enumerate()
        .find(|(_, (positive, negative))| positive != negative)
    {
        let mut next = model;
        assign_model(&mut next, SatVar(variable), positive)?;
        return dpll(clauses, next, budget);
    }

    let Some(variable) = model.iter().position(Option::is_none) else {
        return Err(LogicError::SolverInvariantViolation(
            "undetermined clause remained after every SAT variable was assigned".to_string(),
        ));
    };
    for value in [true, false] {
        budget.charge(model.len().max(1))?;
        let mut next = model.clone();
        assign_model(&mut next, SatVar(variable), value)?;
        if let Some(solution) = dpll(clauses, next, budget)? {
            return Ok(Some(solution));
        }
    }
    Ok(None)
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
    fn truth_table_preflights_formula_depth_and_evaluation_work() {
        let mut too_deep = BoolExpr::Const(true);
        for _ in 0..=MAX_FORMULA_DEPTH {
            too_deep = BoolExpr::Not(Box::new(too_deep));
        }
        assert!(matches!(
            too_deep.truth_table(),
            Err(LogicError::SolverLimitExceeded {
                resource: "formula depth",
                actual,
                limit: MAX_FORMULA_DEPTH,
            }) if actual > MAX_FORMULA_DEPTH
        ));

        let mut terms: Vec<BoolExpr> = (0..MAX_TRUTH_TABLE_VARIABLES)
            .map(|index| BoolExpr::var(format!("v{index}")))
            .collect();
        terms.extend(vec![BoolExpr::Const(true); 13]);
        let excessive_work = BoolExpr::And(terms);
        assert!(matches!(
            excessive_work.truth_table(),
            Err(LogicError::SolverLimitExceeded {
                resource: "truth-table evaluation work",
                actual,
                limit: MAX_TRUTH_TABLE_EVALUATION_WORK,
            }) if actual > MAX_TRUTH_TABLE_EVALUATION_WORK
        ));
    }

    #[test]
    fn test_tautology_and_contradiction() {
        let p = BoolExpr::var("p");
        let taut = p.clone().or(!p.clone());
        assert!(taut.truth_table().unwrap().is_tautology());
        assert_eq!(taut.to_cnf().unwrap(), Vec::<Vec<Literal>>::new());
        assert!(dpll_satisfiable(&taut).unwrap().is_some());

        let contra = p.clone().and(!p);
        assert!(contra.truth_table().unwrap().is_contradiction());
        assert_eq!(contra.to_cnf().unwrap(), vec![Vec::<Literal>::new()]);
        assert!(dpll_satisfiable(&contra).unwrap().is_none());
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
        assert!(dpll_satisfiable(&unsat).unwrap().is_none());

        // (p | q) & (~p | r) & (q | r) is satisfiable.
        let sat = BoolExpr::var("p")
            .or(BoolExpr::var("q"))
            .and((!BoolExpr::var("p")).or(BoolExpr::var("r")))
            .and(BoolExpr::var("q").or(BoolExpr::var("r")));
        let model = dpll_satisfiable(&sat)
            .expect("solver should admit the formula")
            .expect("expected a model");
        assert_eq!(sat.evaluate(&model), Ok(true));
    }

    #[test]
    fn tseitin_encoding_avoids_distributive_cnf_explosion() {
        // Raw CNF distribution requires 2^18 clauses. The Tseitin encoding is
        // linear in the source tree and retains a source-variable model.
        let expr = (0..18)
            .map(|index| BoolExpr::var(format!("p{index}")).and(BoolExpr::var(format!("q{index}"))))
            .reduce(BoolExpr::or)
            .unwrap();
        let model = dpll_satisfiable(&expr)
            .expect("bounded encoding should admit the formula")
            .expect("formula is satisfiable");
        assert_eq!(expr.evaluate(&model), Ok(true));
    }

    #[test]
    fn tseitin_dpll_matches_exhaustive_semantics_for_all_connectives() {
        let p = BoolExpr::var("p");
        let q = BoolExpr::var("q");
        let atoms = vec![
            BoolExpr::Const(false),
            BoolExpr::Const(true),
            p.clone(),
            q.clone(),
            !p,
            !q,
            BoolExpr::And(Vec::new()),
            BoolExpr::Or(Vec::new()),
        ];
        let mut formulas = atoms.clone();
        for left in &atoms {
            for right in &atoms {
                formulas.push(left.clone().and(right.clone()));
                formulas.push(left.clone().or(right.clone()));
                formulas.push(left.clone().implies(right.clone()));
                formulas.push(left.clone().equiv(right.clone()));
            }
        }

        for formula in formulas {
            let expected_sat = !formula.truth_table().unwrap().is_contradiction();
            let model = dpll_satisfiable(&formula).expect("small formula must be admitted");
            assert_eq!(model.is_some(), expected_sat, "formula: {formula}");
            if let Some(model) = model {
                assert_eq!(formula.evaluate(&model), Ok(true), "formula: {formula}");
            }
        }
    }

    #[test]
    fn dpll_resource_refusal_is_not_reported_as_unsat() {
        let expr = BoolExpr::And(
            (0..=MAX_DPLL_VARIABLES)
                .map(|index| BoolExpr::var(format!("v{index}")))
                .collect(),
        );
        assert!(matches!(
            dpll_satisfiable(&expr),
            Err(LogicError::SolverLimitExceeded {
                resource: "variables",
                ..
            })
        ));
    }

    #[test]
    fn dpll_preflights_formula_depth_and_node_count() {
        let mut too_deep = BoolExpr::Const(true);
        for _ in 0..=MAX_FORMULA_DEPTH {
            too_deep = BoolExpr::Not(Box::new(too_deep));
        }
        assert!(matches!(
            dpll_satisfiable(&too_deep),
            Err(LogicError::SolverLimitExceeded {
                resource: "formula depth",
                actual,
                limit: MAX_FORMULA_DEPTH,
            }) if actual > MAX_FORMULA_DEPTH
        ));

        let too_wide = BoolExpr::And(vec![BoolExpr::var("p"); MAX_FORMULA_NODES]);
        assert!(matches!(
            dpll_satisfiable(&too_wide),
            Err(LogicError::SolverLimitExceeded {
                resource: "formula nodes",
                actual,
                limit: MAX_FORMULA_NODES,
            }) if actual > MAX_FORMULA_NODES
        ));
    }

    #[test]
    fn validity_uses_the_existing_tseitin_root_and_validates_the_source() {
        let at_variable_limit = BoolExpr::And(
            (0..(MAX_DPLL_VARIABLES - 1))
                .map(|index| BoolExpr::var(format!("v{index}")))
                .collect(),
        );
        assert!(dpll_satisfiable(&at_variable_limit).unwrap().is_some());
        assert_eq!(is_valid(&at_variable_limit), Ok(false));

        let mut too_deep = BoolExpr::Const(true);
        for _ in 0..=MAX_FORMULA_DEPTH {
            too_deep = BoolExpr::Not(Box::new(too_deep));
        }
        assert!(matches!(
            is_valid(&too_deep),
            Err(LogicError::SolverLimitExceeded {
                resource: "formula depth",
                actual,
                limit: MAX_FORMULA_DEPTH,
            }) if actual > MAX_FORMULA_DEPTH
        ));
    }

    #[test]
    fn dpll_search_budget_refuses_instead_of_reporting_unsat() {
        let variable = SatVar(0);
        let clauses = vec![vec![SatLiteral::positive(variable)]];
        let mut budget = SearchBudget::new(1);
        assert_eq!(
            dpll(&clauses, vec![None], &mut budget),
            Err(LogicError::SolverLimitExceeded {
                resource: "search work units",
                actual: 2,
                limit: 1,
            })
        );
    }

    #[test]
    fn malformed_sat_variable_is_a_typed_invariant_error() {
        let malformed = vec![SatLiteral::positive(SatVar(1))];
        let mut budget = SearchBudget::new(16);
        assert!(matches!(
            clause_value(&malformed, &[None], &mut budget),
            Err(LogicError::SolverInvariantViolation(message))
                if message.contains("outside model length")
        ));
    }

    #[test]
    fn test_xor_nand_nor_xnor() {
        fn node_count(expr: &BoolExpr) -> usize {
            match expr {
                BoolExpr::Const(_) | BoolExpr::Var(_) => 1,
                BoolExpr::Not(inner) => 1 + node_count(inner),
                BoolExpr::And(terms) | BoolExpr::Or(terms) => {
                    1 + terms.iter().map(node_count).sum::<usize>()
                }
                BoolExpr::Implies(left, right) | BoolExpr::Equivalent(left, right) => {
                    1 + node_count(left) + node_count(right)
                }
            }
        }

        let p = BoolExpr::var("p");
        let q = BoolExpr::var("q");
        let env_tt = HashMap::from([(Symbol::new("p"), true), (Symbol::new("q"), true)]);
        let env_tf = HashMap::from([(Symbol::new("p"), true), (Symbol::new("q"), false)]);
        let env_ft = HashMap::from([(Symbol::new("p"), false), (Symbol::new("q"), true)]);
        let env_ff = HashMap::from([(Symbol::new("p"), false), (Symbol::new("q"), false)]);

        let xor = p.clone().xor(q.clone());
        assert!(!xor.evaluate(&env_tt).unwrap());
        assert!(xor.evaluate(&env_tf).unwrap());
        assert!(xor.evaluate(&env_ft).unwrap());
        assert!(!xor.evaluate(&env_ff).unwrap());

        let nand = p.clone().nand(q.clone());
        assert!(!nand.evaluate(&env_tt).unwrap());
        assert!(nand.evaluate(&env_tf).unwrap());
        assert!(nand.evaluate(&env_ft).unwrap());
        assert!(nand.evaluate(&env_ff).unwrap());

        let nor = p.clone().nor(q.clone());
        assert!(!nor.evaluate(&env_tt).unwrap());
        assert!(!nor.evaluate(&env_tf).unwrap());
        assert!(!nor.evaluate(&env_ft).unwrap());
        assert!(nor.evaluate(&env_ff).unwrap());

        let xnor = p.xnor(q);
        assert!(xnor.evaluate(&env_tt).unwrap());
        assert!(!xnor.evaluate(&env_tf).unwrap());
        assert!(!xnor.evaluate(&env_ft).unwrap());
        assert!(xnor.evaluate(&env_ff).unwrap());

        let mut xor_chain = BoolExpr::var("x0");
        for index in 1..=12 {
            xor_chain = xor_chain.xor(BoolExpr::var(format!("x{index}")));
        }
        assert_eq!(
            node_count(&xor_chain),
            37,
            "each XOR must retain one copy of each operand"
        );
    }

    #[test]
    fn test_simplify_logic_algebraic_identities() {
        let x = BoolExpr::var("x");
        let y = BoolExpr::var("y");

        // Double negation
        assert_eq!(simplify_logic(&!(!x.clone())), x);

        // Constants
        assert_eq!(
            simplify_logic(&BoolExpr::Const(true)),
            BoolExpr::Const(true)
        );
        assert_eq!(
            simplify_logic(&!BoolExpr::Const(true)),
            BoolExpr::Const(false)
        );

        // And identities
        assert_eq!(simplify_logic(&x.clone().and(BoolExpr::Const(true))), x);
        assert_eq!(
            simplify_logic(&x.clone().and(BoolExpr::Const(false))),
            BoolExpr::Const(false)
        );
        assert_eq!(simplify_logic(&x.clone().and(x.clone())), x);
        assert_eq!(
            simplify_logic(&x.clone().and(!x.clone())),
            BoolExpr::Const(false)
        );

        // Or identities
        assert_eq!(simplify_logic(&x.clone().or(BoolExpr::Const(false))), x);
        assert_eq!(
            simplify_logic(&x.clone().or(BoolExpr::Const(true))),
            BoolExpr::Const(true)
        );
        assert_eq!(simplify_logic(&x.clone().or(x.clone())), x);
        assert_eq!(
            simplify_logic(&x.clone().or(!x.clone())),
            BoolExpr::Const(true)
        );

        // Implication
        assert_eq!(
            simplify_logic(&x.clone().implies(x.clone())),
            BoolExpr::Const(true)
        );
        assert_eq!(
            simplify_logic(&x.clone().implies(y.clone())),
            (!x.clone()).or(y.clone())
        );

        // Equivalence
        assert_eq!(
            simplify_logic(&x.clone().equiv(x.clone())),
            BoolExpr::Const(true)
        );
        assert_eq!(
            simplify_logic(&x.clone().equiv(!x.clone())),
            BoolExpr::Const(false)
        );
    }

    #[test]
    fn test_is_satisfiable_is_valid_and_is_contradiction() {
        let x = BoolExpr::var("x");
        let y = BoolExpr::var("y");

        // Tautology: x | ~x
        let tautology = x.clone().or(!x.clone());
        assert_eq!(is_satisfiable(&tautology), Ok(true));
        assert_eq!(is_valid(&tautology), Ok(true));
        assert_eq!(is_contradiction_sat(&tautology), Ok(false));

        // Contradiction: x & ~x
        let contradiction = x.clone().and(!x.clone());
        assert_eq!(is_satisfiable(&contradiction), Ok(false));
        assert_eq!(is_valid(&contradiction), Ok(false));
        assert_eq!(is_contradiction_sat(&contradiction), Ok(true));

        // Contingency: x & y
        let contingency = x.and(y);
        assert_eq!(is_satisfiable(&contingency), Ok(true));
        assert_eq!(is_valid(&contingency), Ok(false));
        assert_eq!(is_contradiction_sat(&contingency), Ok(false));
    }
}
