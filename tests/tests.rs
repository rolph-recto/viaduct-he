use std::collections::HashMap;
use he_vectorizer::lang::{
    *,
    ExprOperator::*,
    normalizer::ExprNormalizer,
    IndexExpr::*,
    SourceExpr::*
};
use interval::{Interval, ops::Range};

#[test]
fn imgblur() {
    let store: HashMap<ArrayName, Vec<Extent>> =
        HashMap::from([
            (String::from("img"), vec![Interval::new(0,16), Interval::new(0,16)])
        ]);

    let expr =
        ForNode(String::from("x"), Interval::new(0, 16),
        Box::new(
            ForNode(String::from("y"), Interval::new(0, 16),
            Box::new(
                OpNode(OpAdd,
                    Box::new(
                        IndexingNode(
                            String::from("img"),
                            vec![
                                IndexOp(OpAdd, Box::new(IndexVar(String::from("x"))), Box::new(IndexLiteral(1))),
                                IndexOp(OpAdd, Box::new(IndexVar(String::from("y"))), Box::new(IndexLiteral(1))),
                            ],
                        )
                    ),
                    Box::new(
                        IndexingNode(
                            String::from("img"),
                            vec![
                                IndexOp(OpSub, Box::new(IndexVar(String::from("x"))), Box::new(IndexLiteral(1))),
                                IndexOp(OpSub, Box::new(IndexVar(String::from("y"))), Box::new(IndexLiteral(1))),
                            ],
                        )
                    )
                )
            )
        )
    ));

    let norm_expr = ExprNormalizer::new().run(&expr, &store);
    println!("{}", norm_expr)
}