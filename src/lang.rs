use core::fmt::Display;
use std::collections::HashMap;

use interval::Interval;
use lalrpop_util::lalrpop_mod;

lalrpop_mod!(pub parser);

pub mod extent_analysis;
pub mod source;
pub mod index_elim;
pub mod index_free;
pub mod typechecker;

pub use self::source::*;
pub use self::index_free::*;

pub type Extent = Interval<i64>;
pub type Shape = im::Vector<Extent>;

pub type IndexName = String;
pub type ArrayName = String;

pub type ArrayEnvironment = HashMap<ArrayName, Shape>;
pub type IndexEnvironment = HashMap<IndexName, Extent>;

pub type ExprId = usize;

#[derive(Copy,Clone,Debug)]
pub enum Operator { Add, Sub, Mul }

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Add => write!(f, "+"),
            Operator::Sub => write!(f, "-"),
            Operator::Mul => write!(f, "*"),
        }
    }
}
