use he_vectorizer::lang::{
    parser::ExprParser,
    *,
    normalizer::ExprNormalizer,
    typechecker::TypeChecker,
};
use im::vector;
use interval::{Interval, ops::Range};

#[test]
fn test_parse_positive() {
    let parser = ExprParser::new();
    assert!(parser.parse("42").is_ok());
    assert!(parser.parse("(42)").is_ok());
    assert!(parser.parse("42 + 56").is_ok());
    assert!(parser.parse("42 * 56").is_ok());
    assert!(parser.parse("for x : (0,16) { 42 }").is_ok());
    assert!(parser.parse("for x: (0,16) { for y: (0, 16) { img[x][y] + 2  }}").is_ok());
}

#[test]
fn test_parse_negative() {
    let parser = ExprParser::new();
    assert!(parser.parse("for x : (0,16) in { 42").is_err());
}

#[test]
fn test_typechecker_positive() {
    let parser = ExprParser::new();
    let typechecker = TypeChecker::new();
    let store: im::HashMap<ArrayName, Shape> =
        im::HashMap::from(vec![
            (String::from("img"), vector![Interval::new(0,16), Interval::new(0,16)])
        ]);

    let expr1 = parser.parse("42").unwrap();
    let expr2 = parser.parse("42 + 56").unwrap();
    let expr3 = parser.parse("for x: (0,16) { img[x] }").unwrap();

    assert!(typechecker.run(&expr1, &store).is_ok());
    assert!(typechecker.run(&expr2, &store).is_ok());
    assert!(typechecker.run(&expr3, &store).is_ok());
}

#[test]
fn test_typechecker_negative() {
    let parser = ExprParser::new();
    let typechecker = TypeChecker::new();
    let store: im::HashMap<ArrayName, Shape> =
        im::HashMap::from(vec![
            (String::from("img"), vector![Interval::new(0,16), Interval::new(0,16)])
        ]);

    let expr1 = parser.parse("sum(42)").unwrap();
    let expr2 = parser.parse("for x: (0,16) { for y: (0, 16) { for z: (0, 16) { img[x][y][z] }}}").unwrap();

    assert!(typechecker.run(&expr1, &store).is_err());
    assert!(typechecker.run(&expr2, &store).is_err());
}


#[test]
fn imgblur() {
    let parser = ExprParser::new();
    let store: im::HashMap<ArrayName, Shape> =
        im::HashMap::from(vec![
            (String::from("img"), vector![Interval::new(0,16), Interval::new(0,16)])
        ]);

    let expr: SourceExpr =
        parser.parse(
            "for x: (0, 16) {
                for y: (0, 16) {
                    img[x-1][y-1] + img[x+1][y+1]
                }
            }"
        ).unwrap();

    let norm_expr = ExprNormalizer::new().run(&expr, &store);
    println!("{}", norm_expr)
}

fn matmatmul() {
    let parser = ExprParser::new();
    let store: ArrayEnvironment =
        im::HashMap::from(vec![
            (String::from("A"), vector![Interval::new(0,4), Interval::new(0,4)]),
            (String::from("B"), vector![Interval::new(0,4), Interval::new(0,4)]),
        ]);

    let expr: SourceExpr =
        parser.parse(
            "for x: (0, 4) {
                for y: (0, 4) {
                    img[x-1][y-1] + img[x+1][y+1]
                }
            }"
        ).unwrap();

    let norm_expr = ExprNormalizer::new().run(&expr, &store);
    println!("{}", norm_expr)
}