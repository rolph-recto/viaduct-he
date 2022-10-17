use egg::{Runner, RecExpr, LpExtractor};
use good_lp::ResolutionError;
use he_vectorizer::circ::optimizer::{ Optimizer, lp_extractor::OpSizeFunction, HEOptCircuit, HELatencyModel };

fn run_equiv(s1: &str, s2: &str) -> (bool, String) {
    let optimizer = Optimizer::new(16);
    let expr1 = s1.parse().unwrap();
    let expr2 = s2.parse().unwrap();
    let mut runner =
        Runner::default()
        .with_explanations_enabled()
        .with_expr(&expr1)
        .run(optimizer.rules());

    let equiv = runner.egraph.equivs(&expr1, &expr2).len() > 0;
    if equiv {
        (true, runner.explain_equivalence(&expr1, &expr2).get_flat_string())

    } else {
        (false, String::from(""))
    }
}

fn run_extractor(s: &str) -> RecExpr<HEOptCircuit> {
    let optimizer = Optimizer::new(16);
    let expr = s.parse().unwrap();
    let runner =
        Runner::default()
        // .with_explanations_enabled()
        .with_expr(&expr)
        .run(optimizer.rules());
    let root = *runner.roots.first().unwrap();

    // let mut extractor = HEExtractor::new(&runner.egraph, root);
    // extractor.solve().unwrap()

    let mut extractor =
        LpExtractor::new(
            &runner.egraph, 
            OpSizeFunction { latency: HELatencyModel::default() }
        );
    extractor.solve(root)
}

// #[ignore] ensures that these long-running equality saturation tests
// don't run when calling `cargo test`.

#[test]
#[ignore]
fn test_mul_to_add() {
    assert!(run_equiv("(* x 2)", "(+ x x)").0);
}

#[test]
#[ignore]
fn test_add_to_mul() {
    assert!(run_equiv("(+ (+ x x) x)", "(* x 3)").0);
}

#[test]
#[ignore]
fn test_factor() {
    assert!(run_equiv("(+ (* x x) (* 2 x))", "(* x (+ x 2))").0);
}

#[test]
#[ignore]
fn test_constant_fold() {
    assert!(run_equiv("(+ x (* 2 3))", "(+ x 6)").0);
}

#[test]
#[ignore]
fn test_neg_equiv() {
    assert!(!run_equiv("(+ (* x x) (* 2 x))", "(+ (+ x x) x)").0);
}

#[test]
#[ignore]
fn test_neg_equiv2() {
    let (equiv, explain) = run_equiv("(+ (* x x) (* 2 x))", "(* x (+ 3 x))");
    println!("{}", explain);
    assert!(!equiv);
}

#[test]
#[ignore]
fn test_extract() {
    let res = run_extractor("(+ (* x x) (* 2 x))");
    println!("{}", res);
}
