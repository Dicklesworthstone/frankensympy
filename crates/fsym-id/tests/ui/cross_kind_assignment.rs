use fsym_id::{ContextId, TermId};

fn main() {
    // Distinct newtype kinds never unify: assignment across kinds is a
    // type error even though both wrap a u64 payload.
    let term: TermId = ContextId::new(7).unwrap();
    println!("{term}");
}
