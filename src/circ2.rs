use itertools::{MultiProduct, Itertools};
use std::{collections::{HashSet, HashMap}, fmt::Display, ops::Range};

use crate::{
    lang::{Operator, BaseArrayTransform, DimSize},
    scheduling::{OffsetExpr, ScheduleDim, ExprSchedule, DimName}
};

use self::materializer::VectorInfo;

pub mod materializer;
pub mod cost;

type VarName = String;

#[derive(Copy,Clone,Debug,PartialEq,Eq)]
pub enum VectorType { Ciphertext, Plaintext }

/// parameterized circuit expr that represents an *array* of circuit exprs
/// these exprs are parameterized by exploded dim coordinate variables
#[derive(Clone,Debug)]
pub enum ParamCircuitExpr {
    CiphertextVar(VarName),
    PlaintextVar(VarName),
    Literal(isize),
    Op(Operator, Box<ParamCircuitExpr>, Box<ParamCircuitExpr>),
    Rotate(Box<OffsetExpr>, Box<ParamCircuitExpr>),
    ReduceVectors(HashSet<(DimName,DimSize)>, Operator, Box<ParamCircuitExpr>),
}

impl Display for ParamCircuitExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamCircuitExpr::CiphertextVar(name) => {
                write!(f, "CT({})", name)
            },

            ParamCircuitExpr::PlaintextVar(name) => {
                write!(f, "PT({})", name)
            },

            ParamCircuitExpr::Literal(lit) => {
                write!(f, "{}", lit)
            },

            ParamCircuitExpr::Op(op, expr1, expr2) => {
                write!(f, "({} {} {})", expr1, op, expr2)
            },

            ParamCircuitExpr::Rotate(offset, expr) => {
                write!(f, "rot({}, {})", offset, expr)
            },

            ParamCircuitExpr::ReduceVectors(indices, op, expr) => {
                write!(f, "reduce({:?}, {}, {})", indices, op, expr)
            },
        }
    }
}

type IndexCoord = im::Vector<usize>;

pub struct IndexCoordinateSystem(im::Vector<(DimName,usize)>);

impl IndexCoordinateSystem {
    pub fn new<'a, A: Iterator<Item=&'a ScheduleDim>>(dims: A) -> Self {
        IndexCoordinateSystem(
            dims.map(|dim| {
                (dim.name.clone(), dim.extent)
            }).collect()
        )
    }

    pub fn coord_iter(&self) -> impl Iterator<Item=im::Vector<usize>> + Clone {
        self.0.iter()
        .map(|(_, extent)| (0..*extent))
        .multi_cartesian_product()
        .into_iter()
        .map(|coord| im::Vector::from(coord))
    }

    // iterate through coordinates while fixing an index to be a subset of its range
    pub fn coord_iter_subset(&self, dim: &DimName, range: Range<usize>) -> impl Iterator<Item=im::Vector<usize>> + Clone {
        self.0.iter()
        .map(|(idim, extent)| {
            if idim == dim {
                range.clone()

            } else {
                0..*extent
            }
        }).multi_cartesian_product()
        .into_iter()
        .map(|coord| im::Vector::from(coord))
    }

    pub fn index_vars(&self) -> Vec<String> {
        self.0.iter()
        .map(|(var, _)| var.clone())
        .collect()
    }

    pub fn in_range(&self, coord: IndexCoord) -> bool {
        if self.0.len() == coord.len() {
            self.0.iter()
            .zip(coord.iter())
            .map(|((_, extent), point)| *point <= *extent)
            .all(|x| x)
            
        } else {
            false
        }
    }

    pub fn is_empty(&self) -> bool {
        self.0.len() == 0 ||
        self.0.iter().all(|(_, extent)| *extent == 0)
    }

    /// count how many items this coordinate system represents
    pub fn multiplicity(&self) -> usize {
        self.0.iter().fold(1, |acc, (_, extent)| acc * (*extent))
    }
}

// map from index variable coordinates to values
pub struct IndexCoordinateMap<T: Default> {
    coord_system: IndexCoordinateSystem,
    coord_map: HashMap<IndexCoord, T>
}

impl<T: Default> IndexCoordinateMap<T> {
    pub fn new<'a, A: Iterator<Item = &'a ScheduleDim>>(dims: A) -> Self {
        let coord_system = IndexCoordinateSystem::new(dims);
        let mut coord_map: HashMap<IndexCoord, T> = HashMap::new();
        for coord in coord_system.coord_iter() {
            let im_coord = im::Vector::from(coord);
            coord_map.insert(im_coord, T::default());
        }
        
        IndexCoordinateMap { coord_system, coord_map }
    }

    pub fn coord_iter(&self) -> impl Iterator<Item=IndexCoord> + Clone {
        self.coord_system.coord_iter()
    }

    pub fn coord_iter_subset(&self, dim: &DimName, range: Range<usize>) -> impl Iterator<Item=IndexCoord> + Clone {
        self.coord_system.coord_iter_subset(dim, range)
    }

    pub fn value_iter(&self) -> impl Iterator<Item=(IndexCoord,&T)> + Clone {
        self.coord_iter().map(|coord| {
            let value = &self.coord_map[&coord];
            (coord, value)
        })
    }

    pub fn index_vars(&self) -> Vec<String> {
        self.coord_system.index_vars()
    }

    pub fn set(&mut self, coord: IndexCoord, value: T) {
        assert!(self.coord_map.contains_key(&coord));
        self.coord_map.insert(coord, value);
    }

    pub fn get(&self, coord: &IndexCoord) -> &T {
        &self.coord_map[coord]
    }

    pub fn multiplicity(&self) -> usize {
        self.coord_system.multiplicity()
    }

    pub fn is_empty(&self) -> bool {
        self.coord_system.is_empty()
    }
}

#[derive(Clone,Debug)]
pub enum CiphertextObject {
    Null,
    Vector(VectorInfo),
}

impl Default for CiphertextObject {
    fn default() -> Self { CiphertextObject::Null }
}

impl Display for CiphertextObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CiphertextObject::Null => write!(f, "null"),

            CiphertextObject::Vector(v) => write!(f, "{}", v)
        }
    }
}

#[derive(Clone,Debug)]
pub enum PlaintextObject { Null }

impl Default for PlaintextObject {
    fn default() -> Self { PlaintextObject::Null }
}

pub enum CircuitVarValue<T: Default> {
    CoordMap(IndexCoordinateMap<T>),
    Object(T)
}

type CiphertextVarValue = CircuitVarValue<CiphertextObject>;
type PlaintextVarValue = CircuitVarValue<PlaintextObject>;
type OffsetVarValue = CircuitVarValue<usize>;

pub struct CircuitRegistry {
    ct_var_values: HashMap<VarName, CiphertextVarValue>,
    ct_var_id: usize,

    pt_var_values: HashMap<VarName, PlaintextVarValue>,
    pt_var_id: usize
}

impl CircuitRegistry {
    fn new() -> Self {
        CircuitRegistry {
            ct_var_values: HashMap::new(),
            ct_var_id: 1,
            pt_var_values: HashMap::new(),
            pt_var_id: 1
        }
    }

    pub fn fresh_ciphertext_var(&mut self) -> VarName {
        let id = self.ct_var_id;
        self.ct_var_id += 1;
        format!("ct{}", id)
    }

    pub fn fresh_plaintext_var(&mut self) -> VarName {
        let id = self.pt_var_id;
        self.pt_var_id += 1;
        format!("pt{}", id)
    }

    pub fn set_ct_var_value(&mut self, ct_var: VarName, value: CiphertextVarValue) {
        self.ct_var_values.insert(ct_var, value);
    }

    pub fn set_pt_var_value(&mut self, pt_var: VarName, value: PlaintextVarValue) {
        self.pt_var_values.insert(pt_var, value);
    }

    pub fn get_ct_var_value(&mut self, ct_var: VarName) -> &CiphertextVarValue {
        self.ct_var_values.get(&ct_var).unwrap()
    }

    pub fn get_pt_var_value(&mut self, pt_var: VarName) -> &PlaintextVarValue {
        self.pt_var_values.get(&pt_var).unwrap()
    }

    // TODO implement
    pub fn get_ciphertext_objects(&self) -> Vec<CiphertextObject> {
        vec![]
    }

    // TODO implement
    pub fn get_plaintext_objects(&self) -> Vec<PlaintextObject> {
        vec![]
    }
}

/// parameterized circuit packaged with information about input ciphertexts/plaintexts used
/// the schedule defining exploded dims
pub struct ParamCircuitProgram {
    pub schedule: ExprSchedule,
    pub expr: ParamCircuitExpr,
    pub registry: CircuitRegistry,
}

impl Display for ParamCircuitProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.schedule, self.expr)
    }
}


