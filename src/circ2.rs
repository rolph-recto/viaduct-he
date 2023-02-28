use egg::{RecExpr, Symbol};
use itertools::Itertools;
use std::{collections::{HashSet, HashMap}, fmt::Display, ops::Range};

use crate::{
    circ2::{vector_info::VectorInfo},
    lang::{Operator, DimSize, Extent, ArrayName},
    scheduling::{OffsetExpr, ScheduleDim, DimName}, circ::optimizer::HEOptCircuit
};

pub mod cost;
pub mod materializer;
pub mod vector_info;

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
    ReduceVectors(DimName, DimSize, Operator, Box<ParamCircuitExpr>),
}

impl ParamCircuitExpr {
    fn to_opt_circuit_recur(&self, opt_expr: &mut RecExpr<HEOptCircuit>) -> egg::Id {
        match self {
            ParamCircuitExpr::CiphertextVar(var) => {
                opt_expr.add(HEOptCircuit::CiphertextRef(Symbol::from(var)))
            },

            ParamCircuitExpr::PlaintextVar(var) => {
                opt_expr.add(HEOptCircuit::CiphertextRef(Symbol::from(var)))
            },

            ParamCircuitExpr::Literal(lit) => {
                opt_expr.add(HEOptCircuit::Num(*lit))
            },

            ParamCircuitExpr::Op(op, expr1, expr2) => {
                let id1 = expr1.to_opt_circuit_recur(opt_expr);
                let id2 = expr2.to_opt_circuit_recur(opt_expr);
                match op {
                    Operator::Add => opt_expr.add(HEOptCircuit::Add([id1, id2])),

                    Operator::Sub => opt_expr.add(HEOptCircuit::Sub([id1, id2])),

                    Operator::Mul => opt_expr.add(HEOptCircuit::Mul([id1, id2])),
                }
            },

            ParamCircuitExpr::Rotate(steps, body) => {
                let steps_id = steps.to_opt_circuit_recur(opt_expr);
                let body_id = body.to_opt_circuit_recur(opt_expr);
                opt_expr.add(HEOptCircuit::Rot([steps_id, body_id]))
            },

            ParamCircuitExpr::ReduceVectors(_, _, _, _) => todo!(),
        }
    }

    pub fn to_opt_circuit(&self) -> RecExpr<HEOptCircuit> {
        let mut opt_expr: RecExpr<HEOptCircuit> = RecExpr::default();
        self.to_opt_circuit_recur(&mut opt_expr);
        opt_expr
    }
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

            ParamCircuitExpr::ReduceVectors(index, extent, op, expr) => {
                write!(f, "reduce({}, {}, {})", index, op, expr)
            },
        }
    }
}

impl ParamCircuitExpr {
    // get the ciphertext vars in the expression
    pub fn ciphertext_vars(&self) -> HashSet<VarName> {
        match self {
            ParamCircuitExpr::CiphertextVar(name) => {
                let mut res = HashSet::new();
                res.insert(name.clone());
                res
            },

            ParamCircuitExpr::PlaintextVar(_) => HashSet::new(),
            
            ParamCircuitExpr::Literal(_) => HashSet::new(),

            ParamCircuitExpr::Op(_, expr1, expr2) => {
                let mut res = HashSet::new();
                res.extend(expr1.ciphertext_vars());
                res.extend(expr2.ciphertext_vars());
                res
            },

            ParamCircuitExpr::Rotate(_, body) => {
                body.ciphertext_vars()
            },

            ParamCircuitExpr::ReduceVectors(_, _, _, body) => {
                body.ciphertext_vars()
            }
        }
    }

    // get the ciphertext vars in the expression
    pub fn plaintext_vars(&self) -> HashSet<VarName> {
        match self {
            ParamCircuitExpr::PlaintextVar(name) => {
                let mut res = HashSet::new();
                res.insert(name.clone());
                res
            },

            ParamCircuitExpr::CiphertextVar(_) => HashSet::new(),
            
            ParamCircuitExpr::Literal(_) => HashSet::new(),

            ParamCircuitExpr::Op(_, expr1, expr2) => {
                let mut res = HashSet::new();
                res.extend(expr1.plaintext_vars());
                res.extend(expr2.plaintext_vars());
                res
            },

            ParamCircuitExpr::Rotate(_, body) => {
                body.plaintext_vars()
            },

            ParamCircuitExpr::ReduceVectors(_, _, _, body) => {
                body.plaintext_vars()
            }
        }
    }

    pub fn offset_fvars(&self) -> HashSet<String> {
        match self {
            ParamCircuitExpr::CiphertextVar(_) |
            ParamCircuitExpr::PlaintextVar(_) |
            ParamCircuitExpr::Literal(_) =>
                HashSet::new(),

            ParamCircuitExpr::Op(_, expr1, expr2) => {
                let mut fvars1 = expr1.offset_fvars();
                let fvars2 = expr1.offset_fvars();
                fvars1.extend(fvars2);
                fvars1
            },

            ParamCircuitExpr::Rotate(offset, body) => {
                let mut offset_fvars = offset.function_vars();
                let body_fvar = body.offset_fvars();
                offset_fvars.extend(body_fvar);
                offset_fvars
            },

            ParamCircuitExpr::ReduceVectors(_, _, _, body) =>
                body.offset_fvars()
        }
    }
}

type IndexCoord = im::Vector<usize>;

#[derive(Debug,Clone)]
pub struct IndexCoordinateSystem(im::Vector<(DimName,usize)>);

impl IndexCoordinateSystem {
    pub fn new<'a, A: Iterator<Item=&'a ScheduleDim>>(dims: A) -> Self {
        IndexCoordinateSystem(
            dims.map(|dim| {
                (dim.name.clone(), dim.extent)
            }).collect()
        )
    }

    pub fn from_coord_system(coord_system: &IndexCoordinateSystem) -> IndexCoordinateSystem {
        IndexCoordinateSystem(coord_system.0.clone())
    }

    pub fn coord_iter(&self) -> impl Iterator<Item=im::Vector<usize>> + Clone {
        self.0.iter()
        .map(|(_, extent)| (0..*extent))
        .multi_cartesian_product()
        .into_iter()
        .map(|coord| im::Vector::from(coord))
    }

    pub fn index_map_iter(&self) -> impl Iterator<Item=HashMap<DimName,usize>> + Clone {
        let index_vars = self.index_vars();
        self.coord_iter().map(move |coord| {
            index_vars.clone().into_iter().zip(coord).collect()
        })
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
#[derive(Debug)]
pub struct IndexCoordinateMap<T> {
    coord_system: IndexCoordinateSystem,
    coord_map: HashMap<IndexCoord, T>
}

impl<T> IndexCoordinateMap<T> {
    pub fn new<'a, A: Iterator<Item = &'a ScheduleDim>>(dims: A) -> Self {
        let coord_system = IndexCoordinateSystem::new(dims);
        let coord_map: HashMap<IndexCoord, T> = HashMap::new();
        IndexCoordinateMap { coord_system, coord_map }
    }

    pub fn from_coord_system(coord_system: IndexCoordinateSystem) -> Self {
        IndexCoordinateMap { coord_system, coord_map: HashMap::new() }
    }
    
    pub fn coord_iter(&self) -> impl Iterator<Item=IndexCoord> + Clone {
        self.coord_system.coord_iter()
    }

    pub fn index_map_iter(&self) -> impl Iterator<Item=HashMap<DimName,usize>> + Clone {
        self.coord_system.index_map_iter()
    }

    pub fn coord_as_index_map(&self, coord: IndexCoord) -> HashMap<DimName, usize> {
        self.coord_system.0.iter()
        .map(|(dim_name, _)| dim_name.clone())
        .zip(coord)
        .collect()
    }

    pub fn index_map_as_coord(&self, index_map: HashMap<DimName,usize>) -> IndexCoord {
        self.coord_system.0.iter()
        .map(|(dim_name, _)| index_map[dim_name])
        .collect()
    }

    pub fn coord_iter_subset(&self, dim: &DimName, range: Range<usize>) -> impl Iterator<Item=IndexCoord> + Clone {
        self.coord_system.coord_iter_subset(dim, range)
    }

    pub fn value_iter(&self) -> impl Iterator<Item=(IndexCoord,Option<&T>)> + Clone {
        self.coord_iter().map(|coord| {
            let value = self.coord_map.get(&coord);
            (coord, value)
        })
    }

    pub fn map<F,U>(&self, f: F) -> IndexCoordinateMap<U>
        where F: Fn(&IndexCoord, &T) -> U
    {
        let mut coord_map = IndexCoordinateMap::from_coord_system(self.coord_system.clone());
        for (coord, value_opt) in self.value_iter() {
            if let Some(value) = value_opt {
                let res = f(&coord, value);
                coord_map.set(coord, res);
            }
        }

        coord_map
    }

    pub fn index_vars(&self) -> Vec<String> {
        self.coord_system.index_vars()
    }

    pub fn set(&mut self, coord: IndexCoord, value: T) {
        self.coord_map.insert(coord, value);
    }

    pub fn get(&self, coord: &IndexCoord) -> Option<&T> {
        self.coord_map.get(coord)
    }

    pub fn multiplicity(&self) -> usize {
        self.coord_system.multiplicity()
    }

    pub fn is_empty(&self) -> bool {
        self.coord_system.is_empty()
    }
}

impl<T: Display> Display for IndexCoordinateMap<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.value_iter().try_for_each(|(coord, value_opt)| {
            if let Some(value) = value_opt {
                write!(f, "{:?} => {}\n", coord, value)

            } else {
                write!(f, "{:?} => null\n", coord)
            }
        })
    }
}

#[derive(Clone,Debug,PartialEq,Eq)]
pub enum CiphertextObject {
    InputVector(VectorInfo),
    VectorRef(ArrayName, IndexCoord)
}

impl Display for CiphertextObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CiphertextObject::InputVector(v) =>
                write!(f, "{}", v),

            CiphertextObject::VectorRef(array, coords) => {
                let mut coord_str = String::new();
                for coord in coords {
                    coord_str.push_str(&coord.to_string());
                }
                write!(f, "{}{}", array, coord_str)
            }
        }
    }
}

pub type MaskVector = im::Vector<(usize, usize, usize)>;

#[derive(Clone,Debug,PartialEq,Eq)]
pub enum PlaintextObject {
    // plaintext filled with a constant value
    Const(isize),

    // plaintext of 1s and 0s used to mask 
    // the mask is a vector of (dim_size, lower, upper)
    // where [lower, upper) defines the interval filled with 1s;
    // values outside of this interval is 0, so when multiplied with a vector
    // elements outside of the interval will be zeroed out
    Mask(MaskVector),
}

impl Default for PlaintextObject {
    fn default() -> Self { PlaintextObject::Const(1) }
}

impl Display for PlaintextObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlaintextObject::Const(lit) => write!(f, "{}", lit),

            PlaintextObject::Mask(mask) => write!(f, "{:?}", mask)
        }
    }
}

/// objects that are referenced in a param circuit
#[derive(Debug)]
pub enum CircuitValue<T> {
    CoordMap(IndexCoordinateMap<T>),
    Single(T)
}

impl<T> CircuitValue<T> {
    pub fn map<U,F>(&self, f: F) -> CircuitValue<U> where F: Fn(&IndexCoord, &T)->U {
        match self {
            CircuitValue::CoordMap(coord_map) => 
                CircuitValue::CoordMap(coord_map.map(f)),

            CircuitValue::Single(obj) =>
                CircuitValue::Single(f(&im::Vector::new(), obj))
        }
    }
}

impl<T: Display> Display for CircuitValue<T> where T: Display {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitValue::CoordMap(map) => write!(f, "{}", map),
            CircuitValue::Single(obj) => write!(f, "{}", obj),
        }
    }
}

type CiphertextVarValue = CircuitValue<CiphertextObject>;
type PlaintextVarValue = CircuitValue<PlaintextObject>;
type OffsetFunctionVarValue = CircuitValue<isize>;

/// data structure that maintains values for variables in parameterized circuits
#[derive(Debug)]
pub struct CircuitRegistry {
    pub ct_var_values: HashMap<VarName, CiphertextVarValue>,
    ct_var_id: usize,

    pub pt_var_values: HashMap<VarName, PlaintextVarValue>,
    pt_var_id: usize,

    pub offset_fvar_values: HashMap<DimName, OffsetFunctionVarValue>,
    offset_fvar_id: usize,
}

impl CircuitRegistry {
    fn new() -> Self {
        CircuitRegistry {
            ct_var_values: HashMap::new(),
            ct_var_id: 1,
            pt_var_values: HashMap::new(),
            pt_var_id: 1,
            offset_fvar_values: HashMap::new(),
            offset_fvar_id: 1,
        }
    }

    pub fn fresh_ct_var(&mut self) -> VarName {
        let id = self.ct_var_id;
        self.ct_var_id += 1;
        format!("ct{}", id)
    }

    pub fn fresh_pt_var(&mut self) -> VarName {
        let id = self.pt_var_id;
        self.pt_var_id += 1;
        format!("pt{}", id)
    }

    pub fn fresh_offset_fvar(&mut self) -> VarName {
        let id = self.offset_fvar_id;
        self.offset_fvar_id += 1;
        format!("offset{}", id)
    }

    pub fn set_ct_var_value(&mut self, ct_var: VarName, value: CiphertextVarValue) {
        self.ct_var_values.insert(ct_var, value);
    }

    pub fn set_pt_var_value(&mut self, pt_var: VarName, value: PlaintextVarValue) {
        self.pt_var_values.insert(pt_var, value);
    }

    pub fn set_offset_var_value(&mut self, offset_var: DimName, value: OffsetFunctionVarValue) {
        self.offset_fvar_values.insert(offset_var, value);
    }

    pub fn get_ct_var_value(&self, ct_var: &VarName) -> &CiphertextVarValue {
        self.ct_var_values.get(ct_var).unwrap()
    }

    pub fn get_pt_var_value(&self, pt_var: &VarName) -> &PlaintextVarValue {
        self.pt_var_values.get(pt_var).unwrap()
    }

    pub fn get_offset_fvar_value(&self, offset_fvar: &DimName) -> &OffsetFunctionVarValue {
        self.offset_fvar_values.get(offset_fvar).unwrap()
    }

    // TODO implement
    pub fn get_ct_objects(&self) -> Vec<CiphertextObject> {
        todo!()
    }

    // TODO implement
    pub fn get_pt_objects(&self) -> Vec<PlaintextObject> {
        todo!()
    }
}

impl Display for CircuitRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.ct_var_values.iter().try_for_each(|(ct_var, val)| {
            write!(f, "{} => \n{}\n", ct_var, val)
        })?;

        self.pt_var_values.iter().try_for_each(|(pt_var, val)| {
            write!(f, "{} => \n{}\n", pt_var, val)
        })?;

        self.offset_fvar_values.iter().try_for_each(|(offset_var, val)| {
            write!(f, "{} => \n{}\n", offset_var, val)
        })?;

        Ok(())
    }
}

/// parameterized circuit packaged with information about input ciphertexts/plaintexts used
#[derive(Debug)]
pub struct ParamCircuitProgram {
    pub registry: CircuitRegistry,
    pub circuit_list: Vec<(String, Vec<(DimName,Extent)>, ParamCircuitExpr)>
}

impl Display for ParamCircuitProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.registry)?;

        self.circuit_list.iter().try_for_each(|(name, dims, circuit)| {
            let mut dims_str = String::new();
            for dim in dims.iter() {
                dims_str.push_str(&format!("[{}: {}]", dim.0, dim.1))
            }
            write!(f, "let {}{} = {}\n", name, dims_str, circuit)
        })
    }
}
