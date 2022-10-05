use core::fmt::Display;
use std::collections::HashMap;

use interval::Interval;
use lalrpop_util::lalrpop_mod;

lalrpop_mod!(pub parser);

pub mod normalizer;
pub mod typechecker;

pub type Extent = Interval<i64>;
pub type Shape = im::Vector<Extent>;

pub type IndexName = String;
pub type ArrayName = String;

pub type ArrayEnvironment = im::HashMap<ArrayName, Shape>;
pub type IndexEnvironment = im::HashMap<IndexName, Extent>;

pub type ExprId = usize;

#[derive(Copy,Clone,Debug)]
pub enum ExprOperator { OpAdd, OpSub, OpMul }

#[derive(Clone,Debug)]
pub enum IndexExpr {
    IndexVar(IndexName),
    IndexLiteral(i64),
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

impl Display for IndexExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexExpr::IndexVar(var) => {
                write!(f, "{}", var)
            },

            IndexExpr::IndexLiteral(val) => {
                write!(f, "{}", val)
            },

            IndexExpr::IndexOp(op, expr1, expr2) => {
                let op_str =
                    match op {
                        ExprOperator::OpAdd => "+",
                        ExprOperator::OpSub => "-",
                        ExprOperator::OpMul => "*"
                    };

                write!(f, "({} {} {})", expr1, op_str, expr2)
            }
        }
    }
}

#[derive(Clone,Debug)]
pub enum SourceExpr {
    ForNode(IndexName, Extent, Box<SourceExpr>),
    ReduceNode(ExprOperator, Box<SourceExpr>),
    OpNode(ExprOperator, Box<SourceExpr>, Box<SourceExpr>),
    IndexingNode(ArrayName, im::Vector<IndexExpr>),
    LiteralNode(i64)
}

impl Display for SourceExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use SourceExpr::*;
        match self {
            ForNode(index, extent, body) => {
                write!(f, "for {} : {} in {}", index, extent, body)
            },

            ReduceNode(op, body) => {
                let reduce_op_str = 
                    match op {
                        ExprOperator::OpAdd => "sum",
                        ExprOperator::OpSub => "sum_sub",
                        ExprOperator::OpMul => "product"
                    };

                write!(f, "{}({})", reduce_op_str, body)
            },

            OpNode(op, expr1, expr2) => {
                let op_str =
                    match op {
                        ExprOperator::OpAdd => "+",
                        ExprOperator::OpSub => "-",
                        ExprOperator::OpMul => "*"
                    };

                write!(f, "({} {} {})", expr1, op_str, expr2)
            },

            IndexingNode(arr, index_list) => {
                write!(f, "{}{:?}", arr, index_list)
            },

            LiteralNode(val) => {
                write!(f, "{}", val)
            },
        }
    }
}

#[derive(Clone,Debug)]
pub enum NormalizedExpr {
    ReduceNode(ExprOperator, Box<NormalizedExpr>),
    OpNode(ExprOperator, Box<NormalizedExpr>, Box<NormalizedExpr>),
    TransformNode(ExprId, ArrayName, ArrayTransform),
    LiteralNode(i64)
}

type PadSize = (u64, u64);

#[derive(Clone,Debug)]
pub struct ArrayTransform {
    fill_sizes: Vec<usize>,
    transpose: Vec<usize>,
    pad_sizes: Vec<PadSize>,
    extent_list: Vec<Extent>
}

impl Display for NormalizedExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NormalizedExpr::ReduceNode(op, body) => {
                let reduce_op_str = 
                    match op {
                        ExprOperator::OpAdd => "sum",
                        ExprOperator::OpSub => "sum_sub",
                        ExprOperator::OpMul => "product"
                    };

                write!(f, "{}({})", reduce_op_str, body)
            },

            NormalizedExpr::OpNode(op, expr1, expr2) => {
                let op_str = 
                    match op {
                        ExprOperator::OpAdd => "+",
                        ExprOperator::OpSub => "-",
                        ExprOperator::OpMul => "*"
                    };

                write!(f, "({} {} {})", expr1, op_str, expr2)
            },

            NormalizedExpr::TransformNode(_, arr, transform) => {
                write!(
                    f,
                    "transpose(fill(pad({}, {:?}), {:?}), {:?})",
                    arr,
                    transform.pad_sizes,
                    transform.fill_sizes,
                    transform.transpose
                )
            },

            NormalizedExpr::LiteralNode(val) => {
                write!(f, "{}", val)
            }
        }
    }
}