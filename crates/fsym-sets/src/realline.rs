//! Exact real-line normal form for unions/intersections/differences of
//! intervals and finite point sets.
//!
//! A set built only from intervals and finite sets whose endpoints are
//! comparable extended-real constants is normalized to a sorted list of
//! disjoint pieces (degenerate closed pieces are isolated points). Endpoint
//! order is decided exactly for rationals and infinities and by certified
//! ball enclosures for other real constants; any undecided comparison makes
//! the normalization refuse (`None`) so callers keep the unevaluated form.

use std::cmp::Ordering;

use fsym_core::{Constant, Expr};

use crate::SymSet;

const ORDER_DIGITS: [u32; 2] = [30, 120];

fn is_neg_inf(e: &Expr) -> bool {
    matches!(e, Expr::Const(Constant::NegativeInfinity))
}

fn is_pos_inf(e: &Expr) -> bool {
    matches!(e, Expr::Const(Constant::Infinity))
}

fn is_symbol_free_real_candidate(e: &Expr) -> bool {
    match e {
        Expr::Sym(_) => false,
        Expr::Const(c) => matches!(c, Constant::Pi | Constant::E),
        Expr::Integer(_) | Expr::Rational(_) => true,
        Expr::Add(xs) | Expr::Mul(xs) | Expr::Function(_, xs) => {
            xs.iter().all(is_symbol_free_real_candidate)
        }
        Expr::Pow(b, x) => is_symbol_free_real_candidate(b) && is_symbol_free_real_candidate(x),
    }
}

/// Certified order of two extended-real constants; `None` when undecided.
pub fn real_order(a: &Expr, b: &Expr) -> Option<Ordering> {
    if a == b {
        return Some(Ordering::Equal);
    }
    match (is_neg_inf(a), is_pos_inf(a), is_neg_inf(b), is_pos_inf(b)) {
        (true, _, _, _) => return Some(Ordering::Less),
        (_, true, _, _) => return Some(Ordering::Greater),
        (_, _, true, _) => return Some(Ordering::Greater),
        (_, _, _, true) => return Some(Ordering::Less),
        _ => {}
    }
    if let (Some(x), Some(y)) = (rational(a), rational(b)) {
        return Some(x.cmp(&y));
    }
    if !is_symbol_free_real_candidate(a) || !is_symbol_free_real_candidate(b) {
        return None;
    }
    for digits in ORDER_DIGITS {
        let (Ok(ba), Ok(bb)) = (a.evalf_ball(digits), b.evalf_ball(digits)) else {
            return None;
        };
        if ba.upper() < bb.lower() {
            return Some(Ordering::Less);
        }
        if ba.lower() > bb.upper() {
            return Some(Ordering::Greater);
        }
    }
    None
}

fn rational(e: &Expr) -> Option<fsym_core::BigRational> {
    match e {
        Expr::Integer(n) => Some(fsym_core::BigRational::from_integer(n.clone())),
        Expr::Rational(r) => Some(r.clone()),
        _ => None,
    }
}

/// One connected component: `lo..hi` with openness flags. A point is the
/// closed degenerate piece `[p, p]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    pub lo: Expr,
    pub hi: Expr,
    pub lo_open: bool,
    pub hi_open: bool,
}

impl Piece {
    fn is_point(&self) -> bool {
        self.lo == self.hi && !self.lo_open && !self.hi_open
    }
}

/// Lower-bound order: smaller value first; at equal value a closed bound
/// starts earlier than an open one.
fn cmp_lo(a: &Piece, b: &Piece) -> Option<Ordering> {
    Some(match real_order(&a.lo, &b.lo)? {
        Ordering::Equal => a.lo_open.cmp(&b.lo_open),
        o => o,
    })
}

fn checked_sort(pieces: &mut [Piece]) -> Option<()> {
    // Insertion sort so every comparison is checked (no panics on None).
    for i in 1..pieces.len() {
        let mut j = i;
        while j > 0 {
            if cmp_lo(&pieces[j - 1], &pieces[j])? == Ordering::Greater {
                pieces.swap(j - 1, j);
                j -= 1;
            } else {
                break;
            }
        }
    }
    Some(())
}

/// Sort and merge overlapping/touching pieces.
pub fn normalize(mut pieces: Vec<Piece>) -> Option<Vec<Piece>> {
    checked_sort(&mut pieces)?;
    let mut out: Vec<Piece> = Vec::new();
    for p in pieces {
        if let Some(last) = out.last_mut() {
            let o = real_order(&last.hi, &p.lo)?;
            let touches =
                o == Ordering::Greater || (o == Ordering::Equal && !(last.hi_open && p.lo_open));
            if touches {
                match real_order(&p.hi, &last.hi)? {
                    Ordering::Greater => {
                        last.hi = p.hi;
                        last.hi_open = p.hi_open;
                    }
                    Ordering::Equal => last.hi_open = last.hi_open && p.hi_open,
                    Ordering::Less => {}
                }
                continue;
            }
        }
        out.push(p);
    }
    Some(out)
}

fn intersect_piece(a: &Piece, b: &Piece) -> Option<Option<Piece>> {
    let (lo, lo_open) = match real_order(&a.lo, &b.lo)? {
        Ordering::Greater => (a.lo.clone(), a.lo_open),
        Ordering::Less => (b.lo.clone(), b.lo_open),
        Ordering::Equal => (a.lo.clone(), a.lo_open || b.lo_open),
    };
    let (hi, hi_open) = match real_order(&a.hi, &b.hi)? {
        Ordering::Less => (a.hi.clone(), a.hi_open),
        Ordering::Greater => (b.hi.clone(), b.hi_open),
        Ordering::Equal => (a.hi.clone(), a.hi_open || b.hi_open),
    };
    Some(match real_order(&lo, &hi)? {
        Ordering::Less => Some(Piece {
            lo,
            hi,
            lo_open,
            hi_open,
        }),
        Ordering::Equal if !lo_open && !hi_open && !is_neg_inf(&lo) && !is_pos_inf(&lo) => {
            Some(Piece {
                lo,
                hi,
                lo_open,
                hi_open,
            })
        }
        _ => None,
    })
}

pub fn intersect(a: &[Piece], b: &[Piece]) -> Option<Vec<Piece>> {
    let mut out = Vec::new();
    for p in a {
        for q in b {
            if let Some(r) = intersect_piece(p, q)? {
                out.push(r);
            }
        }
    }
    normalize(out)
}

/// Complement within the extended real line `(-oo, oo)`.
pub fn complement(a: &[Piece]) -> Option<Vec<Piece>> {
    let mut out = Vec::new();
    let mut lo = Expr::Const(Constant::NegativeInfinity);
    let mut lo_open = true;
    for p in a {
        let gap = Piece {
            lo: lo.clone(),
            hi: p.lo.clone(),
            lo_open,
            hi_open: !p.lo_open,
        };
        if real_order(&gap.lo, &gap.hi)? == Ordering::Less
            || (gap.lo == gap.hi && !gap.lo_open && !gap.hi_open)
        {
            out.push(gap);
        }
        lo = p.hi.clone();
        lo_open = !p.hi_open;
    }
    let tail = Piece {
        lo,
        hi: Expr::Const(Constant::Infinity),
        lo_open,
        hi_open: true,
    };
    if real_order(&tail.lo, &tail.hi)? == Ordering::Less {
        out.push(tail);
    }
    Some(out)
}

/// Pieces of a real-line set, or `None` when the set is not decidably a
/// finite union of intervals with comparable endpoints.
pub fn pieces_of(set: &SymSet) -> Option<Vec<Piece>> {
    match set {
        SymSet::EmptySet => Some(Vec::new()),
        SymSet::Interval {
            start,
            end,
            left_open,
            right_open,
        } => {
            let lo_open = *left_open || is_neg_inf(start);
            let hi_open = *right_open || is_pos_inf(end);
            let p = Piece {
                lo: start.clone(),
                hi: end.clone(),
                lo_open,
                hi_open,
            };
            match real_order(start, end)? {
                Ordering::Less => Some(vec![p]),
                Ordering::Equal if p.is_point() => Some(vec![p]),
                _ => Some(Vec::new()),
            }
        }
        SymSet::FiniteSet(elems) => {
            let mut out = Vec::new();
            for e in elems {
                if is_neg_inf(e) || is_pos_inf(e) {
                    return None;
                }
                // Every point must be a certified real constant.
                real_order(e, &Expr::Integer(0.into()))?;
                out.push(Piece {
                    lo: e.clone(),
                    hi: e.clone(),
                    lo_open: false,
                    hi_open: false,
                });
            }
            normalize(out)
        }
        SymSet::Union(parts) => {
            let mut all = Vec::new();
            for p in parts {
                all.extend(pieces_of(p)?);
            }
            normalize(all)
        }
        SymSet::Intersection(parts) => {
            let mut iter = parts.iter();
            let mut acc = pieces_of(iter.next()?)?;
            for p in iter {
                acc = intersect(&acc, &pieces_of(p)?)?;
            }
            Some(acc)
        }
        SymSet::UniversalSet | SymSet::Complement(_) => None,
    }
}

/// Rebuild a `SymSet` from normalized pieces: isolated points are gathered
/// into one finite set, intervals stay in increasing order.
pub fn to_set(pieces: Vec<Piece>) -> SymSet {
    let mut points = Vec::new();
    let mut parts = Vec::new();
    for p in pieces {
        if p.is_point() {
            points.push(p.lo);
        } else {
            parts.push(SymSet::Interval {
                start: p.lo,
                end: p.hi,
                left_open: p.lo_open,
                right_open: p.hi_open,
            });
        }
    }
    if !points.is_empty() {
        parts.insert(0, SymSet::finite(points));
    }
    match parts.len() {
        0 => SymSet::EmptySet,
        1 => parts.pop().expect("one part"),
        _ => SymSet::Union(parts),
    }
}

/// Normalize a real-line set when every component is decidable.
pub fn simplify(set: &SymSet) -> Option<SymSet> {
    pieces_of(set).map(to_set)
}

/// `a \ b` for real-line sets.
pub fn difference(a: &SymSet, b: &SymSet) -> Option<SymSet> {
    let pa = pieces_of(a)?;
    let pb = pieces_of(b)?;
    Some(to_set(intersect(&pa, &complement(&pb)?)?))
}

/// Certified membership of a real constant in a real-line set.
pub fn contains(set: &SymSet, elem: &Expr) -> Option<bool> {
    let pieces = pieces_of(set)?;
    real_order(elem, &Expr::Integer(0.into()))?;
    if is_neg_inf(elem) || is_pos_inf(elem) {
        return Some(false);
    }
    for p in &pieces {
        let lo = real_order(elem, &p.lo)?;
        let hi = real_order(elem, &p.hi)?;
        let above = lo == Ordering::Greater || (lo == Ordering::Equal && !p.lo_open);
        let below = hi == Ordering::Less || (hi == Ordering::Equal && !p.hi_open);
        if above && below {
            return Some(true);
        }
    }
    Some(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(n: i64) -> Expr {
        Expr::Integer(n.into())
    }

    fn iv(a: i64, b: i64, lo: bool, ro: bool) -> SymSet {
        SymSet::Interval {
            start: int(a),
            end: int(b),
            left_open: lo,
            right_open: ro,
        }
    }

    #[test]
    fn union_merges_touching_intervals() {
        let u = SymSet::Union(vec![iv(0, 1, false, false), iv(1, 2, false, false)]);
        assert_eq!(simplify(&u), Some(iv(0, 2, false, false)));
        let open_touch = SymSet::Union(vec![iv(0, 1, false, true), iv(1, 2, true, false)]);
        assert_eq!(simplify(&open_touch), Some(open_touch.clone()));
    }

    #[test]
    fn point_fills_open_endpoint() {
        let u = SymSet::Union(vec![iv(0, 1, true, true), SymSet::finite([int(1)])]);
        assert_eq!(simplify(&u), Some(iv(0, 1, true, false)));
    }

    #[test]
    fn difference_punches_holes() {
        let d = difference(&iv(0, 2, false, false), &SymSet::finite([int(1)])).unwrap();
        assert_eq!(
            d,
            SymSet::Union(vec![iv(0, 1, false, true), iv(1, 2, true, false)])
        );
        let d = difference(&iv(0, 2, false, false), &iv(1, 3, false, false)).unwrap();
        assert_eq!(d, iv(0, 1, false, true));
    }

    #[test]
    fn intersection_of_disjoint_is_empty() {
        let i = SymSet::Intersection(vec![iv(0, 1, false, false), iv(2, 3, false, false)]);
        assert_eq!(simplify(&i), Some(SymSet::EmptySet));
    }

    #[test]
    fn irrational_endpoints_are_ordered_by_certified_balls() {
        let sqrt2 = Expr::Pow(
            std::sync::Arc::new(int(2)),
            std::sync::Arc::new(Expr::Rational(fsym_core::BigRational::new(
                1.into(),
                2.into(),
            ))),
        );
        let pi = Expr::Const(Constant::Pi);
        assert_eq!(real_order(&sqrt2, &pi), Some(Ordering::Less));
        let x = Expr::symbol("x");
        assert_eq!(real_order(&x, &pi), None);
    }

    #[test]
    fn symbolic_points_refuse_normalization() {
        let x = Expr::symbol("x");
        assert_eq!(
            simplify(&SymSet::Union(vec![
                iv(0, 1, false, false),
                SymSet::finite([x])
            ])),
            None
        );
    }
}
