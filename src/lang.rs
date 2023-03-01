use core::fmt::Display;
use std::collections::HashMap;
use std::ops::Index;

use gcollections::ops::Bounded;
use interval::Interval;
use lalrpop_util::lalrpop_mod;

lalrpop_mod!(pub parser);

pub mod extent_analysis;
pub mod source;
pub mod elaborated;
pub mod index_elim;
pub mod index_free;
pub mod typechecker;

pub use self::source::*;
pub use self::index_free::*;

pub static OUTPUT_EXPR_NAME: &'static str = "__root__";

pub type DimSize = usize;
pub type Extent = usize;
pub type Shape = im::Vector<Extent>;

// index variable in source
pub type IndexVar = String;

// identifier for an array (either input or let-bound)
pub type ArrayName = String;

// identifier for indexing site
pub type IndexingId = String;

pub type ArrayEnvironment = HashMap<ArrayName, Shape>;
pub type IndexEnvironment = HashMap<IndexVar, Extent>;

#[derive(Copy, Clone, Debug)]
pub enum InputType { Client, Server }

impl Display for InputType  {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputType::Client => write!(f, "client"),
            InputType::Server => write!(f, "server"),
        }
    }
}

#[derive(Copy, Clone, Debug)]
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

pub type DimIndex = usize;

// an dimension of an abstract (read: not materialized) array
#[derive(Clone,Debug,PartialEq,Eq,Hash)]
pub enum DimContent {
    // a dimension where array elements change along one specific dimension
    // of the array being indexed
    FilledDim { dim: DimIndex, extent: usize, stride: usize },

    // a dimension where array elements do not change
    EmptyDim { extent: usize }
}

impl DimContent {
    pub fn extent(&self) -> usize {
        match self {
            DimContent::FilledDim { dim: _, extent, stride: _ } => *extent,
            DimContent::EmptyDim { extent } => *extent
        }
    }
}

impl Display for DimContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DimContent::FilledDim { dim, extent, stride } => {
                write!(f, "{{{}:{}::{}}}", dim, extent, stride)
            },

            DimContent::EmptyDim { extent } => {
                write!(f, "{{{}}}", extent)
            },
        }
    }
}

#[derive(Clone,Debug,PartialEq,Eq,Hash)]
pub struct OffsetMap<T> { map: Vec<T> }

pub type BaseOffsetMap = OffsetMap<isize>;

impl<T: Clone+Default+Display+Eq> OffsetMap<T> {
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

    pub fn map<F: FnMut(&T) -> S, S: Default+Display+Clone+Eq>(&self, f: F) -> OffsetMap<S> {
        OffsetMap { map: self.map.iter().map(f).collect() }
    }
}

impl<T: Clone+Display> Display for OffsetMap<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}",
            self.map.iter()
                .map(|x| x.to_string())
                .collect::<Vec<String>>()
                .join(", ")
        )
    }
}

impl<T: Clone+Display> Index<usize> for OffsetMap<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.map.get(index).unwrap()
    }
}

#[derive(Clone,Debug)]
pub struct ArrayTransform {
    pub array: ArrayName,
    pub offset_map: OffsetMap<isize>,
    pub dims: im::Vector<DimContent>,
}

impl Display for ArrayTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}[{}]<{}>",
            self.array,
            self.offset_map,
            self.dims.iter()
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
            dims:
                shape.iter().enumerate().map(|(i, dim)| {
                    DimContent::FilledDim { dim: i, extent: dim.upper() as usize, stride: 1 }
                }).collect()
        }
    }

    /// convert a transform into a shape
    pub fn as_shape(&self) -> Shape {
        self.dims.iter().map(|dim| {
            match dim {
                DimContent::FilledDim { dim: _, extent, stride: _ } |
                DimContent::EmptyDim { extent } => {
                    *extent
                }
            }
        }).collect()
    }
}