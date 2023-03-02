use core::fmt::Display;
use std::collections::{HashMap, HashSet};
use std::ops::Index;

use gcollections::ops::Bounded;
use interval::Interval;
use lalrpop_util::lalrpop_mod;

lalrpop_mod!(pub parser);

pub mod elaborated;
pub mod extent_analysis;
pub mod index_elim;
pub mod source;
pub mod typechecker;

pub static OUTPUT_EXPR_NAME: &'static str = "__root__";

// name of a dimension in a schedule
pub type DimName = String;
pub type Extent = usize;
pub type Shape = im::Vector<Extent>;

// index variable in source
pub type IndexVar = String;

// identifier for an array (either input or let-bound)
pub type ArrayName = String;

// identifier for indexing site
pub type IndexingId = String;
pub type VectorId = usize;

pub type ArrayEnvironment = HashMap<ArrayName, Shape>;
pub type IndexEnvironment = HashMap<IndexVar, Extent>;

#[derive(Copy, Clone, Debug)]
pub enum ArrayType {
    Ciphertext,
    Plaintext,
}

impl ArrayType {
    pub fn join(&self, other: &ArrayType) -> ArrayType {
        match self {
            ArrayType::Ciphertext => ArrayType::Ciphertext,
            ArrayType::Plaintext => *other,
        }
    }
}

impl Display for ArrayType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArrayType::Ciphertext => write!(f, "client"),
            ArrayType::Plaintext => write!(f, "server"),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum Operator {
    Add,
    Sub,
    Mul,
}

impl Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Add => write!(f, "+"),
            Operator::Sub => write!(f, "-"),
            Operator::Mul => write!(f, "*"),
        }
    }
}

pub type DimIndex = usize;

// an dimension of an abstract (read: not materialized) array
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DimContent {
    // a dimension where array elements change along one specific dimension
    // of the array being indexed
    FilledDim {
        dim: DimIndex,
        extent: usize,
        stride: usize,
    },

    // a dimension where array elements do not change
    EmptyDim {
        extent: usize,
    },
}

impl DimContent {
    pub fn extent(&self) -> usize {
        match self {
            DimContent::FilledDim {
                dim: _,
                extent,
                stride: _,
            } => *extent,
            DimContent::EmptyDim { extent } => *extent,
        }
    }
}

impl Display for DimContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DimContent::FilledDim {
                dim,
                extent,
                stride,
            } => {
                write!(f, "{{{}:{}::{}}}", dim, extent, stride)
            }

            DimContent::EmptyDim { extent } => {
                write!(f, "{{{}}}", extent)
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OffsetMap<T> {
    map: Vec<T>,
}

pub type BaseOffsetMap = OffsetMap<isize>;

impl<T: Clone + Default + Display + Eq> OffsetMap<T> {
    pub fn new(num_dims: usize) -> Self {
        let map = vec![T::default(); num_dims];
        OffsetMap { map }
    }

    pub fn set(&mut self, dim: DimIndex, offset: T) {
        self.map[dim] = offset
    }

    pub fn get(&self, dim: usize) -> &T {
        &self.map[dim]
    }

    pub fn num_dims(&self) -> usize {
        self.map.len()
    }

    pub fn map<F: FnMut(&T) -> S, S: Default + Display + Clone + Eq>(&self, f: F) -> OffsetMap<S> {
        OffsetMap {
            map: self.map.iter().map(f).collect(),
        }
    }
}

impl<T: Clone + Display> Display for OffsetMap<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.map
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

impl<T: Clone + Display> Index<usize> for OffsetMap<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.map.get(index).unwrap()
    }
}

#[derive(Clone, Debug)]
pub struct ArrayTransform {
    pub array: ArrayName,
    pub offset_map: OffsetMap<isize>,
    pub dims: im::Vector<DimContent>,
}

impl Display for ArrayTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}[{}]<{}>",
            self.array,
            self.offset_map,
            self.dims
                .iter()
                .map(|dim| dim.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

impl ArrayTransform {
    /// return the "identity" transform for a shape
    pub fn from_shape(name: ArrayName, shape: &Shape) -> ArrayTransform {
        ArrayTransform {
            array: name,
            offset_map: OffsetMap::new(shape.len()),
            dims: shape
                .iter()
                .enumerate()
                .map(|(i, dim)| DimContent::FilledDim {
                    dim: i,
                    extent: dim.upper() as usize,
                    stride: 1,
                })
                .collect(),
        }
    }

    /// convert a transform into a shape
    pub fn as_shape(&self) -> Shape {
        self.dims
            .iter()
            .map(|dim| match dim {
                DimContent::FilledDim {
                    dim: _,
                    extent,
                    stride: _,
                }
                | DimContent::EmptyDim { extent } => *extent,
            })
            .collect()
    }
}

pub struct OffsetEnvironment {
    index_map: HashMap<DimName, usize>,
    function_values: HashMap<String, isize>,
}

impl OffsetEnvironment {
    pub fn new(index_map: HashMap<DimName, usize>) -> Self {
        OffsetEnvironment {
            index_map,
            function_values: HashMap::new(),
        }
    }

    pub fn set_function_value(&mut self, func: String, value: isize) {
        self.function_values.insert(func, value);
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum OffsetExpr {
    Add(Box<OffsetExpr>, Box<OffsetExpr>),
    Mul(Box<OffsetExpr>, Box<OffsetExpr>),
    Literal(isize),
    Var(DimName),
    FunctionVar(String, im::Vector<DimName>),
}

impl OffsetExpr {
    pub fn eval(&self, store: &OffsetEnvironment) -> isize {
        match self {
            OffsetExpr::Add(expr1, expr2) => {
                let val1 = expr1.eval(store);
                let val2 = expr2.eval(store);
                val1 + val2
            }

            OffsetExpr::Mul(expr1, expr2) => {
                let val1 = expr1.eval(store);
                let val2 = expr2.eval(store);
                val1 * val2
            }

            OffsetExpr::Literal(lit) => *lit,

            OffsetExpr::Var(var) => store.index_map[var] as isize,

            OffsetExpr::FunctionVar(func, _) => store.function_values[func],
        }
    }

    pub fn const_value(&self) -> Option<isize> {
        match self {
            OffsetExpr::Add(expr1, expr2) => {
                let const1 = expr1.const_value()?;
                let const2 = expr2.const_value()?;
                Some(const1 + const2)
            }

            OffsetExpr::Mul(expr1, expr2) => {
                let const1 = expr1.const_value()?;
                let const2 = expr2.const_value()?;
                Some(const1 + const2)
            }

            OffsetExpr::Literal(lit) => Some(*lit),

            OffsetExpr::Var(_) => None,

            OffsetExpr::FunctionVar(_, _) => None,
        }
    }

    pub fn function_vars(&self) -> HashSet<String> {
        match self {
            OffsetExpr::Add(expr1, expr2) | OffsetExpr::Mul(expr1, expr2) => {
                let mut vars1 = expr1.function_vars();
                let vars2 = expr2.function_vars();
                vars1.extend(vars2);
                vars1
            }

            OffsetExpr::Literal(_) | OffsetExpr::Var(_) => HashSet::new(),

            OffsetExpr::FunctionVar(fvar, _) => HashSet::from([fvar.clone()]),
        }
    }
}

impl Display for OffsetExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OffsetExpr::Add(expr1, expr2) => {
                write!(f, "({} + {})", expr1, expr2)
            }

            OffsetExpr::Mul(expr1, expr2) => {
                write!(f, "({} * {})", expr1, expr2)
            }

            OffsetExpr::Literal(lit) => {
                write!(f, "{}", lit)
            }

            OffsetExpr::Var(var) => {
                write!(f, "{}", var)
            }

            OffsetExpr::FunctionVar(func, vars) => {
                write!(f, "{}{:?}", func, vars)
            }
        }
    }
}

impl Default for OffsetExpr {
    fn default() -> Self {
        OffsetExpr::Literal(0)
    }
}
