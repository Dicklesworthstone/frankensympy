//! Recursive coefficient-ring division is distinct from sparse normal forms.

use fsym_budget::Unbounded;
use fsym_core::Symbol;
use fsym_polys::{
    PrimeFieldMultivariatePolynomial as Polynomial, PrimeFieldMultivariateTerms as Terms,
    PrimeFieldRecursiveDivisionCertificate, TermOrder,
    bounded_prime_field_recursive_division_certificate as candidate,
    checked_prime_field_multivariate_division as sparse_divide,
    checked_prime_field_recursive_division as divide,
    verify_prime_field_recursive_division as verify,
};

fn terms(values: &[(&[u32], u64)]) -> Terms {
    values
        .iter()
        .map(|(monomial, value)| (monomial.to_vec(), *value))
        .collect()
}

fn polynomial(generators: &[Symbol], prime: u64, values: &[(&[u32], u64)]) -> Polynomial {
    Polynomial::new(generators, prime, &terms(values)).unwrap()
}

#[test]
fn constant_fields_use_sparse_division_and_recursive_entry_points_refuse() {
    let divisor = polynomial(&[], 5, &[(&[], 3)]);
    for dividend in [polynomial(&[], 5, &[]), polynomial(&[], 5, &[(&[], 2)])] {
        assert!(candidate(&dividend, &divisor, &mut Unbounded).is_err());
        assert!(divide(&dividend, &divisor, &mut Unbounded, &mut Unbounded).is_err());
        let untrusted = PrimeFieldRecursiveDivisionCertificate {
            quotient: Terms::new(),
            remainder: dividend.terms().clone(),
            terminal: None,
        };
        assert!(verify(&dividend, &divisor, &untrusted, &mut Unbounded).is_err());
        let result = sparse_divide(
            &dividend,
            std::slice::from_ref(&divisor),
            TermOrder::Lex,
            &mut Unbounded,
            &mut Unbounded,
        )
        .unwrap();
        let expected = if dividend.is_zero() {
            Terms::new()
        } else {
            terms(&[(&[], 4)])
        };
        assert_eq!(result.quotients()[0].terms(), &expected);
        assert!(result.quotients()[0].generators().is_empty());
        assert!(result.remainder().is_zero());
    }
}

#[test]
fn public_division_stops_at_the_first_coefficient_obstruction_over_gf7() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let f = polynomial(
        &variables,
        7,
        &[(&[2, 1], 3), (&[1, 2], 2), (&[0, 1], 4), (&[0, 0], 5)],
    );
    let g = polynomial(&variables, 7, &[(&[1, 1], 2), (&[0, 2], 3), (&[0, 0], 1)]);
    let accepted = divide(&f, &g, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(accepted.dividend(), &f);
    assert_eq!(accepted.divisor(), &g);
    assert_eq!(accepted.quotient().terms(), &terms(&[(&[1, 0], 5)]));
    assert_eq!(
        accepted.remainder().terms(),
        &terms(&[(&[1, 2], 1), (&[1, 0], 2), (&[0, 1], 4), (&[0, 0], 5),])
    );
    // Recursive division asks whether y^2+2 is divisible by 2*y and
    // stops. Sparse reduction can still remove the x*y^2 monomial.
    let sparse = sparse_divide(
        &f,
        std::slice::from_ref(&g),
        TermOrder::Lex,
        &mut Unbounded,
        &mut Unbounded,
    )
    .unwrap();
    assert_eq!(
        sparse.quotients()[0].terms(),
        &terms(&[(&[1, 0], 5), (&[0, 1], 4)])
    );
    assert_eq!(
        sparse.remainder().terms(),
        &terms(&[(&[1, 0], 2), (&[0, 3], 2), (&[0, 0], 5),])
    );
    let certificate = candidate(&f, &g, &mut Unbounded).unwrap();
    let terminal = certificate.terminal.as_ref().unwrap();
    assert_eq!(terminal.quotients, [terms(&[(&[0, 1], 4)])]);
    assert_eq!(terminal.remainder, terms(&[(&[0, 0], 2)]));
    assert_eq!(
        verify(&f, &g, &certificate, &mut Unbounded).unwrap(),
        accepted
    );
}

#[test]
fn complete_nonmonomial_coefficient_products_cancel_in_three_variables() {
    let variables = [Symbol::new("x"), Symbol::new("y"), Symbol::new("z")];
    // f = ((y-z)*x+z) * ((y+z)*x+y*z+1) + y+1 over F_5.
    let f = polynomial(
        &variables,
        5,
        &[
            (&[2, 2, 0], 1),
            (&[2, 0, 2], 4),
            (&[1, 2, 1], 1),
            (&[1, 1, 2], 4),
            (&[1, 1, 0], 1),
            (&[1, 0, 1], 4),
            (&[1, 1, 1], 1),
            (&[1, 0, 2], 1),
            (&[0, 1, 2], 1),
            (&[0, 0, 1], 1),
            (&[0, 1, 0], 1),
            (&[0, 0, 0], 1),
        ],
    );
    let g = polynomial(
        &variables,
        5,
        &[
            (&[1, 1, 0], 1),
            (&[1, 0, 1], 1),
            (&[0, 1, 1], 1),
            (&[0, 0, 0], 1),
        ],
    );
    let accepted = divide(&f, &g, &mut Unbounded, &mut Unbounded).unwrap();
    assert_eq!(
        accepted.quotient().terms(),
        &terms(&[(&[1, 1, 0], 1), (&[1, 0, 1], 4), (&[0, 0, 1], 1),])
    );
    assert_eq!(
        accepted.remainder().terms(),
        &terms(&[(&[0, 1, 0], 1), (&[0, 0, 0], 1),])
    );
    let certificate = candidate(&f, &g, &mut Unbounded).unwrap();
    assert!(certificate.terminal.is_none());
    let parts = accepted.clone().into_parts();
    assert_eq!(&parts.0, accepted.quotient());
    assert_eq!(&parts.1, accepted.remainder());
}

#[test]
fn a_leading_coefficient_obstruction_prevents_even_the_first_main_step() {
    let variables = [Symbol::new("x"), Symbol::new("y"), Symbol::new("z")];
    let f = polynomial(&variables, 5, &[(&[1, 1, 0], 1), (&[0, 0, 1], 1)]);
    let g = polynomial(
        &variables,
        5,
        &[(&[1, 1, 0], 1), (&[1, 0, 1], 1), (&[0, 0, 0], 1)],
    );
    let accepted = divide(&f, &g, &mut Unbounded, &mut Unbounded).unwrap();
    assert!(accepted.quotient().is_zero());
    assert_eq!(accepted.remainder(), &f);
    let certificate = candidate(&f, &g, &mut Unbounded).unwrap();
    let terminal = certificate.terminal.unwrap();
    assert_eq!(terminal.quotients, [terms(&[(&[0, 0, 0], 1)])]);
    assert_eq!(terminal.remainder, terms(&[(&[0, 0, 1], 4)]));
}

#[test]
fn univariate_division_agrees_including_the_largest_supported_prime() {
    let variables = [Symbol::new("x")];
    for prime in [2, 3, 5, 7, 65521] {
        let f = polynomial(&variables, prime, &[(&[3], 1), (&[0], prime - 1)]);
        let g = polynomial(&variables, prime, &[(&[1], 1), (&[0], prime - 1)]);
        let accepted = divide(&f, &g, &mut Unbounded, &mut Unbounded).unwrap();
        assert_eq!(
            accepted.quotient().terms(),
            &terms(&[(&[2], 1), (&[1], 1), (&[0], 1)])
        );
        assert!(accepted.remainder().is_zero());
        assert!(
            candidate(&f, &g, &mut Unbounded)
                .unwrap()
                .terminal
                .is_none()
        );
    }
}

#[test]
fn degree64_boundary_preserves_all_forced_recursive_steps() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let f = polynomial(&variables, 7, &[(&[64, 0], 1)]);
    let g = polynomial(&variables, 7, &[(&[1, 0], 1), (&[0, 1], 6)]);
    let accepted = divide(&f, &g, &mut Unbounded, &mut Unbounded).unwrap();
    let expected: Terms = (0..64).map(|k| (vec![63 - k, k], 1)).collect();
    assert_eq!(accepted.quotient().terms(), &expected);
    assert_eq!(accepted.remainder().terms(), &terms(&[(&[0, 64], 1)]));
}

#[test]
fn excessive_recursive_output_growth_is_refused_without_a_partial_result() {
    let variables = [Symbol::new("x"), Symbol::new("y")];
    let f = polynomial(&variables, 7, &[(&[64, 0], 1)]);
    let g = polynomial(&variables, 7, &[(&[1, 0], 1), (&[0, 2], 6)]);
    // The exact remainder would be y^128, beyond the accepted degree64
    // output bound. Temporary growth must not turn it into a partial answer.
    assert!(divide(&f, &g, &mut Unbounded, &mut Unbounded).is_err());
}
