use fsym_core::{BigInt, BigRational, Expr};

fn parse_decimal(s: &str) -> BigRational {
    let neg = s.starts_with('-');
    let body = s.trim_start_matches('-');
    let (int_part, frac_part) = match body.split_once('.') {
        Some((a, b)) => (a, b),
        None => (body, ""),
    };
    let numer =
        BigInt::parse_bytes(format!("{int_part}{frac_part}").as_bytes(), 10).expect("digits");
    let denom = BigInt::from(10).pow(frac_part.len() as u32);
    let r = BigRational::new(numer, denom);
    if neg { -r } else { r }
}

fn main() {
    let expr = Expr::Function("sin".into(), vec![Expr::from_i64(4)]);
    let p50 = parse_decimal("-0.75680249530792825137263909451182909413591288733647");
    for digits in [20u32, 30, 40, 50] {
        match expr.evalf_ball(digits) {
            Ok(ball) => {
                let mid = ball.midpoint().clone();
                let radius = ball.radius().clone();
                let distance = if mid >= p50 { &mid - &p50 } else { &p50 - &mid };
                let ratio_ok = distance <= radius;
                let outside_by = distance - &radius;
                println!(
                    "digits={digits} dist_le_rad={ratio_ok} outside_by>0 means ball excludes truth; outside_by_num_bits={} radius_den_bits={} ball_contains_p50={}",
                    outside_by.numer().bits(),
                    radius.denom().bits(),
                    ball.contains(&p50),
                );
            }
            Err(e) => println!("digits={digits} REFUSED {e}"),
        }
    }
    let deep = expr.evalf_ball(200).expect("deep");
    let deep_distance = if deep.midpoint() >= &p50 {
        deep.midpoint() - &p50
    } else {
        &p50 - deep.midpoint()
    };
    println!(
        "deep(200) contains p50: {} (radius_den_bits={})",
        deep_distance <= *deep.radius(),
        deep.radius().denom().bits()
    );
}
