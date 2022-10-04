use std::collections::HashMap;
use he_vectorizer::lang::{
    parser::ExprParser,
    *,
    ExprOperator::*,
    normalizer::ExprNormalizer,
    IndexExpr::*,
    SourceExpr::*
};
use im::vector;
use interval::{Interval, ops::Range};

#[test]
fn test_parse_positive() {
    let parser = ExprParser::new();
    assert!(parser.parse("42").is_ok());
    assert!(parser.parse("-42").is_ok());
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
fn imgblur() {
    let parser = ExprParser::new();
    let store: HashMap<ArrayName, Vec<Extent>> =
        HashMap::from([
            (String::from("img"), vec![Interval::new(0,16), Interval::new(0,16)])
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