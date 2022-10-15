use egg::Runner;
use he_vectorizer::circ::optimizer::Optimizer;

fn test_equiv(s1: &str, s2: &str) -> bool {
    let optimizer = Optimizer::new(16);
    let expr1 = s1.parse().unwrap();
    let expr2 = s2.parse().unwrap();
    let runner =
        Runner::default()
        // .with_explanations_enabled()
        .with_expr(&expr1)
        .run(optimizer.rules());

    runner.egraph.equivs(&expr1, &expr2).len() > 0
}

// #[ignore] ensures that these long-running equality saturation tests
// don't run when calling `cargo test`.

#[test]
#[ignore]
fn test_mul_to_add() {
    assert!(test_equiv("(* x 2)", "(+ x x)"));
}

#[test]
#[ignore]
fn test_add_to_mul() {
    assert!(test_equiv("(+ (+ x x) x)", "(* x 3)"));
}

#[test]
#[ignore]
fn test_factor() {
    assert!(test_equiv("(+ (* x x) (* 2 x))", "(* x (+ x 2))"));
}

#[test]
#[ignore]
fn test_constant_fold() {
    assert!(test_equiv("(+ x (* 2 3))", "(+ x 6)"));
}
