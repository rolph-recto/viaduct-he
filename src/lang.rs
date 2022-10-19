use core::fmt::Display;
use std::collections::HashMap;

use interval::Interval;
use lalrpop_util::lalrpop_mod;

lalrpop_mod!(pub parser);

pub mod source;
pub mod normalized;
pub mod typechecker;

pub type Extent = Interval<i64>;
pub type Shape = im::Vector<Extent>;

pub type IndexName = String;
pub type ArrayName = String;

pub type ArrayEnvironment = HashMap<ArrayName, Shape>;
pub type IndexEnvironment = HashMap<IndexName, Extent>;

pub type ExprId = usize;

#[derive(Copy,Clone,Debug)]
pub enum ExprOperator { OpAdd, OpSub, OpMul }

impl Display for ExprOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExprOperator::OpAdd => write!(f, "+"),
            ExprOperator::OpSub => write!(f, "-"),
            ExprOperator::OpMul => write!(f, "*"),
        }
    }
}
