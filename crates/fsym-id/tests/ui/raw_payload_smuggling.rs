use fsym_id::{ReceiptId, TermId};

fn main() {
    // Raw payloads are not portable across kinds either, even numerically
    // equal ones: there is no From<u64> blanket conversion to smuggle a
    // term payload into a receipt identity.
    let receipt: ReceiptId = ReceiptId::from(TermId::new(3).unwrap().raw());
    println!("{receipt}");
}
