use std::{collections::HashMap, fmt::Display};
use super::Operator;

pub type HEObjectName = String;

#[derive(Clone,Debug)]
pub enum ClientTransform {
    InputArray(HEObjectName),

    // reorder dimensions
    Transpose(Box<ClientTransform>, im::Vector<usize>),

    // add list of dimensions to the vector, intially filled with 0
    Expand(Box<ClientTransform>, im::Vector<usize>),

    // extend existing dimensions
    Pad(Box<ClientTransform>, im::Vector<(usize, usize)>),
}

impl ClientTransform {
    pub fn as_python_str(&self) -> String {
        match self {
            ClientTransform::InputArray(arr) => arr.clone(),

            ClientTransform::Transpose(expr, dims) =>
                format!("transpose({},{:?})", expr.as_python_str(), dims),

            ClientTransform::Expand(expr, num_dims) => 
                format!("expand({},{:?})", expr.as_python_str(), num_dims),

            ClientTransform::Pad(expr, pad_list) =>
                format!("pad({},{:?})", expr.as_python_str(), pad_list),
        }
    }
}

impl Display for ClientTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_python_str())
    }
}

pub type HEClientStore = HashMap<HEObjectName, ClientTransform>;

#[derive(Clone,Debug)]
pub enum IndexFreeExpr {
    // reduction
    Reduce(usize, Operator, Box<IndexFreeExpr>),

    // element-wise operation
    Op(Operator, Box<IndexFreeExpr>, Box<IndexFreeExpr>),

    // array received from the client
    InputArray(HEObjectName),

    // integer literal; must be treated as "shapeless" since literals can
    // denote arrays of *any* dimension
    Literal(isize),

    // TRANSFORMATIONS

    // fill the following dimensions of an array by rotating it
    Fill(Box<IndexFreeExpr>, usize),

    // offset array by a given amount in each dimension
    Offset(Box<IndexFreeExpr>, im::Vector<isize>),

    // zero out a dimension
    Zero(Box<IndexFreeExpr>, usize),
}

impl Display for IndexFreeExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexFreeExpr::Reduce(dim, op, body) => {
                write!(f, "reduce({}, {}, {})", dim, op, body)
            },

            IndexFreeExpr::Op(op,expr1, expr2) => {
                write!(f, "({} {} {})", expr1, op, expr2)
            },

            IndexFreeExpr::InputArray(arr) => {
                write!(f, "{}", arr)
            },

            IndexFreeExpr::Literal(lit) => {
                write!(f, "{}", lit)
            },

            IndexFreeExpr::Fill(expr, dim) => {
                write!(f, "fill({}, {})", expr, dim)
            },

            IndexFreeExpr::Offset(expr, dim_offsets) => {
                write!(f, "offset({}, {:?})", expr, dim_offsets)
            },
            
            IndexFreeExpr::Zero(expr, dim) => {
                write!(f, "zero({}, {})", expr, dim)
            },
        }
    }
}

pub struct IndexFreeProgram {
    pub client_store: HEClientStore,
    pub expr: IndexFreeExpr,
}