use std::collections::HashMap;

use interval::Interval;

pub(crate) mod normalizer;

type Extent = Interval<isize>;
type ExtentStore = HashMap<String, Vec<Extent>>;

type IndexName = String;
type ArrayName = String;

type ExprId = usize;

#[derive(Copy,Clone,Debug)]
pub(crate) enum ExprOperator { OP_ADD, OP_SUB, OP_MUL }

#[derive(Clone,Debug)]
pub(crate) enum IndexExpr {
    IndexVar(IndexName),

    IndexLiteral(isize),

    IndexOp(ExprOperator, Box<IndexExpr>, Box<IndexExpr>)
}

impl IndexExpr {
    fn get_single_var(&self) -> Option<IndexName> {
        let vars = self.get_vars();
        if vars.len() == 1 {
            vars.into_iter().last()

        } else {
            None
        }
    }

    fn get_vars(&self) -> im::HashSet<IndexName> {
        match self {
            IndexExpr::IndexVar(var) => im::HashSet::unit(var.clone()),

            IndexExpr::IndexLiteral(_) => im::HashSet::new(),

            IndexExpr::IndexOp(_, expr1, expr2) => {
                expr1.get_vars().union(expr2.get_vars())
            }
        }
    }
}

#[derive(Clone,Debug)]
pub(crate) enum SourceExpr {
    ForNode { index: String, extent: Extent, body: Box<SourceExpr> },

    ReduceNode { op: ExprOperator, body: Box<SourceExpr> },

    OpNode { op: ExprOperator, expr1: Box<SourceExpr>, expr2: Box<SourceExpr> },

    IndexingNode { arr: ArrayName, index_list: Vec<IndexExpr> }
}

#[derive(Clone,Debug)]
pub(crate) enum NormalizedExpr {
    ReduceNode { op: ExprOperator, body: Box<NormalizedExpr> },

    OpNode { op: ExprOperator, expr1: Box<NormalizedExpr>, expr2: Box<NormalizedExpr> },

    TransformNode { id: ExprId, arr: ArrayName, transform: ArrayTransform }
}

type PadSize = (usize, usize);

#[derive(Clone,Debug)]
pub(crate) struct ArrayTransform {
    fill_sizes: Vec<usize>,
    transpose: Vec<usize>,
    pad_sizes: Vec<PadSize>,
    extent_list: Vec<Extent>
}