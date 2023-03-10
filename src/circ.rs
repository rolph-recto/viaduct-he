use egg::{RecExpr, Symbol};
use itertools::Itertools;
use std::{
    collections::{HashMap, HashSet, LinkedList},
    fmt::Display,
    ops::Range,
};

use crate::{
    circ::{optimizer::{cost::HECostContext, HEOptCircuit}, vector_info::VectorInfo},
    lang::*,
    scheduling::ScheduleDim,
};

pub mod cost;
pub mod materializer;
pub mod optimizer;
pub mod partial_eval;
pub mod vector_deriver;
pub mod vector_info;

pub type CircuitId = usize;
pub type VarName = String;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VectorType {
    Ciphertext,
    Plaintext,
}

pub trait CanCreateObjectVar<T: CircuitObject> {
    fn obj_var(var: VarName) -> Self;
}

/// parameterized circuit expr that represents an *array* of circuit exprs
/// these exprs are parameterized by exploded dim coordinate variables
///
/// Note that this is not a recursive data structure; instead it uses
/// circuit ids to allow sharing of subcircuits---i.e. circuits can be DAGs,
/// not just trees; this eases equality saturation and lowering
#[derive(Clone, Debug)]
pub enum ParamCircuitExpr {
    CiphertextVar(VarName),
    PlaintextVar(VarName),
    Literal(isize),
    Op(Operator, CircuitId, CircuitId),
    Rotate(OffsetExpr, CircuitId),
    ReduceDim(DimName, Extent, Operator, CircuitId),
}

impl ParamCircuitExpr {
    pub fn circuit_refs(&self) -> Vec<CircuitId> {
        match self {
            ParamCircuitExpr::CiphertextVar(_)
            | ParamCircuitExpr::PlaintextVar(_)
            | ParamCircuitExpr::Literal(_) => vec![],

            ParamCircuitExpr::Op(_, id1, id2) => vec![*id1, *id2],

            ParamCircuitExpr::Rotate(_, id) => vec![*id],

            ParamCircuitExpr::ReduceDim(_, _, _, id) => vec![*id],
        }
    }
}

impl Display for ParamCircuitExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamCircuitExpr::CiphertextVar(name) => {
                write!(f, "CT({})", name)
            }

            ParamCircuitExpr::PlaintextVar(name) => {
                write!(f, "PT({})", name)
            }

            ParamCircuitExpr::Literal(lit) => {
                write!(f, "{}", lit)
            }

            ParamCircuitExpr::Op(op, expr1, expr2) => {
                write!(f, "({} {} {})", expr1, op, expr2)
            }

            ParamCircuitExpr::Rotate(offset, expr) => {
                write!(f, "rot({}, {})", offset, expr)
            }

            ParamCircuitExpr::ReduceDim(index, _, op, expr) => {
                write!(f, "reduce({}, {}, {})", index, op, expr)
            }
        }
    }
}

impl CanCreateObjectVar<CiphertextObject> for ParamCircuitExpr {
    fn obj_var(var: VarName) -> Self {
        ParamCircuitExpr::CiphertextVar(var)
    }
}

impl CanCreateObjectVar<PlaintextObject> for ParamCircuitExpr {
    fn obj_var(var: VarName) -> Self {
        ParamCircuitExpr::PlaintextVar(var)
    }
}

pub type IndexCoord = im::Vector<usize>;

#[derive(Debug, Clone)]
pub struct IndexCoordinateSystem(Vec<(DimName, usize)>);

impl IndexCoordinateSystem {
    pub fn new<'a, A: Iterator<Item = &'a ScheduleDim>>(dims: A) -> Self {
        IndexCoordinateSystem(dims.map(|dim| (dim.name.clone(), dim.extent)).collect())
    }

    pub fn from_dim_list(dims: Vec<(DimName, usize)>) -> IndexCoordinateSystem {
        IndexCoordinateSystem(dims)
    }

    pub fn from_coord_system(coord_system: &IndexCoordinateSystem) -> IndexCoordinateSystem {
        IndexCoordinateSystem(coord_system.0.clone())
    }

    pub fn coord_iter(&self) -> impl Iterator<Item = im::Vector<usize>> + Clone {
        self.0
            .iter()
            .map(|(_, extent)| (0..*extent))
            .multi_cartesian_product()
            .into_iter()
            .map(|coord| im::Vector::from(coord))
    }

    pub fn index_map_iter(&self) -> impl Iterator<Item = HashMap<DimName, usize>> + Clone {
        let index_vars = self.index_vars();
        self.coord_iter()
            .map(move |coord| index_vars.clone().into_iter().zip(coord).collect())
    }

    // iterate through coordinates while fixing an index to be a subset of its range
    pub fn coord_iter_subset(
        &self,
        dim: &DimName,
        range: Range<usize>,
    ) -> impl Iterator<Item = im::Vector<usize>> + Clone {
        self.0
            .iter()
            .map(|(idim, extent)| {
                if idim == dim {
                    range.clone()
                } else {
                    0..*extent
                }
            })
            .multi_cartesian_product()
            .into_iter()
            .map(|coord| im::Vector::from(coord))
    }

    pub fn index_vars(&self) -> Vec<String> {
        self.0.iter().map(|(var, _)| var.clone()).collect()
    }
    
    pub fn extents(&self) -> Vec<Extent> {
        self.0.iter().map(|(_, extent)| *extent).collect()
    }

    pub fn in_range(&self, coord: IndexCoord) -> bool {
        if self.0.len() == coord.len() {
            self.0
                .iter()
                .zip(coord.iter())
                .map(|((_, extent), point)| *point <= *extent)
                .all(|x| x)
        } else {
            false
        }
    }

    pub fn is_empty(&self) -> bool {
        self.0.len() == 0 || self.0.iter().all(|(_, extent)| *extent == 0)
    }

    /// count how many items this coordinate system represents
    pub fn multiplicity(&self) -> usize {
        self.0.iter().fold(1, |acc, (_, extent)| acc * (*extent))
    }
}

// map from index variable coordinates to values
#[derive(Clone, Debug)]
pub struct IndexCoordinateMap<T: Clone> {
    coord_system: IndexCoordinateSystem,
    coord_map: HashMap<IndexCoord, T>,
}

impl<T: Clone> IndexCoordinateMap<T> {
    pub fn new<'a, A: Iterator<Item = &'a ScheduleDim>>(dims: A) -> Self {
        let coord_system = IndexCoordinateSystem::new(dims);
        let coord_map: HashMap<IndexCoord, T> = HashMap::new();
        IndexCoordinateMap {
            coord_system,
            coord_map,
        }
    }

    pub fn from_coord_system(coord_system: IndexCoordinateSystem) -> Self {
        IndexCoordinateMap {
            coord_system,
            coord_map: HashMap::new(),
        }
    }

    pub fn coord_iter(&self) -> impl Iterator<Item = IndexCoord> + Clone {
        self.coord_system.coord_iter()
    }

    pub fn index_map_iter(&self) -> impl Iterator<Item = HashMap<DimName, usize>> + Clone {
        self.coord_system.index_map_iter()
    }

    pub fn coord_as_index_map(&self, coord: IndexCoord) -> HashMap<DimName, usize> {
        self.coord_system
            .0
            .iter()
            .map(|(dim_name, _)| dim_name.clone())
            .zip(coord)
            .collect()
    }

    pub fn index_map_as_coord(&self, index_map: HashMap<DimName, usize>) -> IndexCoord {
        self.coord_system
            .0
            .iter()
            .map(|(dim_name, _)| index_map[dim_name])
            .collect()
    }

    pub fn coord_iter_subset(
        &self,
        dim: &DimName,
        range: Range<usize>,
    ) -> impl Iterator<Item = IndexCoord> + Clone {
        self.coord_system.coord_iter_subset(dim, range)
    }

    pub fn value_iter(&self) -> impl Iterator<Item = (IndexCoord, Option<&T>)> + Clone {
        self.coord_iter().map(|coord| {
            let value = self.coord_map.get(&coord);
            (coord, value)
        })
    }

    pub fn map<F, U>(&self, f: F) -> IndexCoordinateMap<U>
    where
        U: Clone, F: Fn(&IndexCoord, &T) -> U,
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

    pub fn extents(&self) -> Vec<Extent> {
        self.coord_system.extents()
    }

    pub fn index_vars_and_extents(&self) -> Vec<(String, Extent)> {
        self.coord_system.0.clone()
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

impl<T: Clone+Display> Display for IndexCoordinateMap<T> {
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

pub trait CircuitObject: Clone {
    fn input_vector(vector: VectorInfo) -> Self;
    fn expr_vector(array: String, coord: IndexCoord) -> Self;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CiphertextObject {
    InputVector(VectorInfo),
    ExprVector(ArrayName, IndexCoord),
}

impl CircuitObject for CiphertextObject {
    fn input_vector(vector: VectorInfo) -> Self {
        CiphertextObject::InputVector(vector)
    }

    fn expr_vector(array: String, coord: IndexCoord) -> Self {
        CiphertextObject::ExprVector(array, coord)
    }
}

impl Display for CiphertextObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CiphertextObject::InputVector(v) => write!(f, "{}", v),

            CiphertextObject::ExprVector(array, coords) => {
                let coord_str = coords
                    .iter()
                    .map(|coord| format!("[{}]", coord))
                    .collect::<Vec<String>>()
                    .join("");

                write!(f, "{}{}", array, coord_str)
            }
        }
    }
}

pub type MaskVector = im::Vector<(usize, usize, usize)>;

pub enum PlaintextExpr {
    Op(Operator, Box<PlaintextExpr>, Box<PlaintextExpr>),
    Reduce(DimName, Operator, Box<PlaintextExpr>, Box<PlaintextExpr>),
    Object(PlaintextObject),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlaintextObject {
    InputVector(VectorInfo),
    ExprVector(ArrayName, IndexCoord),

    // plaintext filled with a constant value
    Const(isize),

    // plaintext of 1s and 0s used to mask
    // the mask is a vector of (dim_size, lower, upper)
    // where [lower, upper) defines the interval filled with 1s;
    // values outside of this interval is 0, so when multiplied with a vector
    // elements outside of the interval will be zeroed out
    Mask(MaskVector),
}

impl CircuitObject for PlaintextObject {
    fn input_vector(vector: VectorInfo) -> Self {
        PlaintextObject::InputVector(vector)
    }

    fn expr_vector(array: String, coord: IndexCoord) -> Self {
        PlaintextObject::ExprVector(array, coord)
    }
}

impl Default for PlaintextObject {
    fn default() -> Self {
        PlaintextObject::Const(1)
    }
}

impl Display for PlaintextObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlaintextObject::InputVector(vector) => write!(f, "{}", vector),

            PlaintextObject::ExprVector(vector, coords) => {
                let coords_str = coords
                    .iter()
                    .map(|coord| format!("[{}]", coord))
                    .collect::<Vec<String>>()
                    .join("");

                write!(f, "{}{}", vector, coords_str)
            }

            PlaintextObject::Const(lit) => write!(f, "{}", lit),

            PlaintextObject::Mask(mask) => write!(f, "{:?}", mask),
        }
    }
}

/// objects that are referenced in a param circuit
#[derive(Clone, Debug)]
pub enum CircuitValue<T: Clone> {
    CoordMap(IndexCoordinateMap<T>),
    Single(T),
}

impl<T: Clone> CircuitValue<T> {
    pub fn map<U, F>(&self, f: F) -> CircuitValue<U>
    where
        U: Clone, F: Fn(&IndexCoord, &T) -> U,
    {
        match self {
            CircuitValue::CoordMap(coord_map) => CircuitValue::CoordMap(coord_map.map(f)),

            CircuitValue::Single(obj) => CircuitValue::Single(f(&im::Vector::new(), obj)),
        }
    }
}

impl<T: Clone+Display> Display for CircuitValue<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitValue::CoordMap(map) => write!(f, "{}", map),
            CircuitValue::Single(obj) => write!(f, "{}", obj),
        }
    }
}

pub trait CanRegisterObject<'a, T: CircuitObject> {
    fn fresh_obj_var(&mut self) -> VarName;
    fn set_obj_var_value(&mut self, var: VarName, val: CircuitValue<T>);
    fn get_var_value(&'a self, var: &String) -> &'a CircuitValue<T>;
}

/// data structure that maintains values for variables in parameterized circuits
#[derive(Debug)]
pub struct CircuitObjectRegistry {
    pub circuit_map: HashMap<CircuitId, ParamCircuitExpr>,
    pub cur_circuit_id: CircuitId,

    pub ct_var_values: HashMap<VarName, CircuitValue<CiphertextObject>>,
    pub ct_var_id: usize,

    pub pt_var_values: HashMap<VarName, CircuitValue<PlaintextObject>>,
    pub pt_var_id: usize,

    pub offset_fvar_values: HashMap<DimName, CircuitValue<isize>>,
    pub offset_fvar_id: usize,
}

impl CircuitObjectRegistry {
    pub fn new() -> Self {
        CircuitObjectRegistry {
            circuit_map: HashMap::new(),
            cur_circuit_id: 1,
            ct_var_values: HashMap::new(),
            ct_var_id: 1,
            pt_var_values: HashMap::new(),
            pt_var_id: 1,
            offset_fvar_values: HashMap::new(),
            offset_fvar_id: 1,
        }
    }

    pub fn register_circuit(&mut self, circuit: ParamCircuitExpr) -> CircuitId {
        let id = self.cur_circuit_id;
        self.cur_circuit_id += 1;
        self.circuit_map.insert(id, circuit);
        id
    }

    pub fn get_circuit(&self, id: CircuitId) -> &ParamCircuitExpr {
        self.circuit_map.get(&id).unwrap()
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

    pub fn set_ct_var_value(&mut self, ct_var: VarName, value: CircuitValue<CiphertextObject>) {
        self.ct_var_values.insert(ct_var, value);
    }

    pub fn set_pt_var_value(&mut self, pt_var: VarName, value: CircuitValue<PlaintextObject>) {
        self.pt_var_values.insert(pt_var, value);
    }

    pub fn set_offset_var_value(&mut self, offset_var: DimName, value: CircuitValue<isize>) {
        self.offset_fvar_values.insert(offset_var, value);
    }

    pub fn get_ct_var_value(&self, ct_var: &VarName) -> &CircuitValue<CiphertextObject> {
        self.ct_var_values.get(ct_var).unwrap()
    }

    pub fn get_pt_var_value(&self, pt_var: &VarName) -> &CircuitValue<PlaintextObject> {
        self.pt_var_values.get(pt_var).unwrap()
    }

    pub fn get_offset_fvar_value(&self, offset_fvar: &DimName) -> &CircuitValue<isize> {
        self.offset_fvar_values.get(offset_fvar).unwrap()
    }

    pub fn get_ciphertext_input_vectors(&self, root_vars: Option<&HashSet<VarName>>) -> HashSet<VectorInfo> {
        let processed_ct_vars: HashSet<&VarName> =
            if let Some(roots) = root_vars {
                roots.iter().collect()

            } else {
                self.ct_var_values.keys().collect()
            };
        
        let mut set = HashSet::new();
        for ct_var in processed_ct_vars {
            match self.get_ct_var_value(ct_var) {
                CircuitValue::CoordMap(coord_map) => {
                    for (_, obj) in coord_map.value_iter() {
                        if let Some(CiphertextObject::InputVector(vector)) = obj {
                            set.insert(vector.clone());
                        }
                    }
                }

                CircuitValue::Single(obj) => {
                    if let CiphertextObject::InputVector(vector) = obj {
                        set.insert(vector.clone());
                    }
                }
            }
        }

        set
    }

    pub fn get_plaintext_input_vectors(&self, root_vars: Option<&HashSet<VarName>>) -> HashSet<VectorInfo> {
        let processed_pt_vars: HashSet<&VarName> =
            if let Some(roots) = root_vars {
                roots.iter().collect()

            } else {
                self.pt_var_values.keys().collect()
            };
        
        let mut set = HashSet::new();
        for pt_var in processed_pt_vars {
            match self.get_pt_var_value(pt_var) {
                CircuitValue::CoordMap(coord_map) => {
                    for (_, obj) in coord_map.value_iter() {
                        if let Some(PlaintextObject::InputVector(vector)) = obj {
                            set.insert(vector.clone());
                        }
                    }
                }

                CircuitValue::Single(obj) => {
                    if let PlaintextObject::InputVector(vector) = obj {
                        set.insert(vector.clone());
                    }
                }
            }
        }

        set
    }

    pub fn get_plaintext_expr_vectors(&self, root_vars: Option<&HashSet<VarName>>) -> HashSet<ArrayName> {
        let processed_pt_vars: HashSet<&VarName> =
            if let Some(roots) = root_vars {
                roots.iter().collect()

            } else {
                self.pt_var_values.keys().collect()
            };
        
        let mut set = HashSet::new();
        for pt_var in processed_pt_vars {
            match self.get_pt_var_value(pt_var) {
                CircuitValue::CoordMap(coord_map) => {
                    for (_, obj) in coord_map.value_iter() {
                        if let Some(PlaintextObject::ExprVector(vector, _)) = obj {
                            set.insert(vector.clone());
                        }
                    }
                }

                CircuitValue::Single(obj) => {
                    if let PlaintextObject::ExprVector(vector, _) = obj {
                        set.insert(vector.clone());
                    }
                }
            }
        }

        set
    }

    pub fn get_masks(&self, root_vars: Option<&HashSet<VarName>>) -> HashSet<MaskVector> {
        let processed_pt_vars: HashSet<&VarName> =
            if let Some(roots) = root_vars {
                roots.iter().collect()

            } else {
                self.pt_var_values.keys().collect()
            };
        
        let mut set = HashSet::new();
        for pt_var in processed_pt_vars {
            match self.get_pt_var_value(pt_var) {
                CircuitValue::CoordMap(coord_map) => {
                    for (_, obj) in coord_map.value_iter() {
                        if let Some(PlaintextObject::Mask(vector)) = obj {
                            set.insert(vector.clone());
                        }
                    }
                }

                CircuitValue::Single(obj) => {
                    if let PlaintextObject::Mask(vector) = obj {
                        set.insert(vector.clone());
                    }
                }
            }
        }

        set
    }

    pub fn get_constants(
        &self,
        root_vars: Option<&HashSet<VarName>>,
        root_circuits: Option<&HashSet<CircuitId>>,
    ) -> HashSet<isize> {
        let processed_pt_vars: HashSet<&VarName> =
            if let Some(roots) = root_vars {
                roots.iter().collect()

            } else {
                self.pt_var_values.keys().collect()
            };
        
        let mut set = HashSet::new();
        for pt_var in processed_pt_vars {
            match self.get_pt_var_value(pt_var) {
                CircuitValue::CoordMap(coord_map) => {
                    for (_, obj) in coord_map.value_iter() {
                        if let Some(PlaintextObject::Const(constval)) = obj {
                            set.insert(*constval);
                        }
                    }
                }

                CircuitValue::Single(obj) => {
                    if let PlaintextObject::Const(constval) = obj {
                        set.insert(*constval);
                    }
                }
            }
        }

        let relevant_circuits: Vec<&ParamCircuitExpr> =
            if let Some(ids) = root_circuits {
                ids.iter()
                .map(|id| self.get_circuit(*id))
                .collect()

            } else {
                self.circuit_map.iter()
                .map(|(_, circuit)| circuit)
                .collect()
            };

        // must look for literals in circuits as well
        for circuit in relevant_circuits {
            if let ParamCircuitExpr::Literal(lit) = circuit {
                set.insert(*lit);
            }
        }

        set
    }

    // get a list of expressions reachable from a circuit root
    // this starts from children and proceeds to the circuit root
    pub fn expr_list(&self, id: CircuitId) -> Vec<CircuitId> {
        let mut expr_ids: Vec<CircuitId> = vec![id];
        let mut worklist: LinkedList<CircuitId> = LinkedList::from([id]);

        while !worklist.is_empty() {
            let id = worklist.pop_front().unwrap();
            let circuit = self.circuit_map.get(&id).unwrap();
            for child_id in circuit.circuit_refs() {
                if !expr_ids.contains(&child_id) {
                    expr_ids.push(child_id);
                    worklist.push_back(child_id);
                }
            }
        }

        expr_ids.reverse();
        expr_ids
    }

    pub fn circuit_ciphertext_vars(&self, id: CircuitId) -> HashSet<VarName> {
        let mut ct_vars: HashSet<VarName> = HashSet::new();
        for child_id in self.expr_list(id) {
            if let ParamCircuitExpr::CiphertextVar(var) = self.get_circuit(child_id) {
                ct_vars.insert(var.clone());
            }
        }

        ct_vars
    }

    pub fn circuit_plaintext_vars(&self, id: CircuitId) -> HashSet<VarName> {
        let mut pt_vars: HashSet<VarName> = HashSet::new();
        for child_id in self.expr_list(id) {
            if let ParamCircuitExpr::PlaintextVar(var) = self.get_circuit(child_id) {
                pt_vars.insert(var.clone());
            }
        }

        pt_vars
    }

    pub fn circuit_offset_fvars(&self, id: CircuitId) -> HashSet<String> {
        let mut offset_fvars: HashSet<VarName> = HashSet::new();
        for child_id in self.expr_list(id) {
            if let ParamCircuitExpr::Rotate(offset, _) = self.get_circuit(child_id) {
                offset_fvars.extend(offset.function_vars());
            }
        }

        offset_fvars
    }

    /// remove circuits not reachable from a set of roots
    pub fn collect_garbage(&mut self, roots: HashSet<CircuitId>) {
        let reachable: HashSet<CircuitId> =
            roots.into_iter()
            .flat_map(|root| {
                HashSet::<CircuitId>::from_iter(
                    self.expr_list(root).into_iter()
                )
            })
            .collect();

        let unreachable: HashSet<CircuitId> =
            self.circuit_map.keys()
            .filter_map(|id|
                if !reachable.contains(id) {
                    Some(*id)
                } else {
                    None
                }
            )
            .collect();

        for id in unreachable {
            self.circuit_map.remove(&id);
        }
    }
}

impl<'a> CanRegisterObject<'a, CiphertextObject> for CircuitObjectRegistry {
    fn fresh_obj_var(&mut self) -> VarName {
        self.fresh_ct_var()
    }

    fn set_obj_var_value(&mut self, var: VarName, val: CircuitValue<CiphertextObject>) {
        self.set_ct_var_value(var, val)
    }

    fn get_var_value(&'a self, var: &VarName) -> &'a CircuitValue<CiphertextObject> {
        self.get_ct_var_value(var)
    }
}

impl<'a> CanRegisterObject<'a, PlaintextObject> for CircuitObjectRegistry {
    fn fresh_obj_var(&mut self) -> VarName {
        self.fresh_pt_var()
    }

    fn set_obj_var_value(&mut self, var: VarName, val: CircuitValue<PlaintextObject>) {
        self.set_pt_var_value(var, val)
    }

    fn get_var_value(&'a self, var: &VarName) -> &'a CircuitValue<PlaintextObject> {
        self.get_pt_var_value(var)
    }
}

impl Display for CircuitObjectRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.ct_var_values
            .iter()
            .try_for_each(|(ct_var, val)| write!(f, "{} => \n{}\n", ct_var, val))?;

        self.pt_var_values
            .iter()
            .try_for_each(|(pt_var, val)| write!(f, "{} => \n{}\n", pt_var, val))?;

        self.offset_fvar_values
            .iter()
            .try_for_each(|(offset_var, val)| write!(f, "{} => \n{}\n", offset_var, val))?;

        Ok(())
    }
}

/// parameterized circuit packaged with information about input ciphertexts/plaintexts used
#[derive(Debug)]
pub struct ParamCircuitProgram {
    pub registry: CircuitObjectRegistry,

    // the expressions that will be executed natively by the server
    pub native_expr_list: Vec<(String, Vec<(DimName, Extent)>, CircuitId)>,

    // the expressions that will be executed in HE
    pub circuit_expr_list: Vec<(String, Vec<(DimName, Extent)>, CircuitId)>,
}

impl ParamCircuitProgram {
    pub fn output_circuit(&self) -> (Vec<(DimName, Extent)>, CircuitId) {
        let (_, dims, id) =
            self.circuit_expr_list.iter()
            .find(|(name, _, _)| name == OUTPUT_EXPR_NAME)
            .unwrap();

        (dims.clone(), *id)
    }

    pub fn to_opt_circuit(&self) -> (Vec<RecExpr<HEOptCircuit>>, HECostContext) {
        // only run optimization *before* partial evaluation
        assert!(self.native_expr_list.len() == 0);
        let mut cost_ctx =
            HECostContext {
                ct_multiplicity_map: HashMap::new(),
                pt_multiplicity_map: HashMap::new(),
                dim_extent_map: HashMap::new(),
            };

        let mut rec_exprs: Vec<RecExpr<HEOptCircuit>> = Vec::new();
        for (_, _, expr_id) in self.circuit_expr_list.iter() {
            let mut circuit_ids: Vec<CircuitId> = Vec::new();
            let mut eclasses: Vec<egg::Id> = Vec::new();
            let mut rec_expr: RecExpr<HEOptCircuit> = RecExpr::default();
            for id in self.registry.expr_list(*expr_id) {
                circuit_ids.push(id);

                match self.registry.get_circuit(id) {
                    ParamCircuitExpr::CiphertextVar(var) => {
                        let eclass =
                            rec_expr.add(HEOptCircuit::CiphertextVar(Symbol::from(var)));

                        let mult =
                            match self.registry.get_ct_var_value(var) {
                                CircuitValue::CoordMap(coord_map) => coord_map.multiplicity(),
                                CircuitValue::Single(_) => 1
                            };

                        cost_ctx.ct_multiplicity_map.insert(var.clone(), mult);
                        eclasses.push(eclass);
                    },

                    ParamCircuitExpr::PlaintextVar(var) => {
                        let eclass =
                            rec_expr.add(HEOptCircuit::PlaintextVar(Symbol::from(var)));

                        let mult =
                            match self.registry.get_pt_var_value(var) {
                                CircuitValue::CoordMap(coord_map) => coord_map.multiplicity(),
                                CircuitValue::Single(_) => 1
                            };

                        cost_ctx.pt_multiplicity_map.insert(var.clone(), mult);
                        eclasses.push(eclass);
                    },

                    ParamCircuitExpr::Literal(lit) => {
                        let eclass =
                            rec_expr.add(HEOptCircuit::Literal(*lit));

                        eclasses.push(eclass);
                    },

                    ParamCircuitExpr::Op(op, id1, id2) => {
                        let index1 =
                            circuit_ids.iter()
                            .position(|cid| cid == id1).unwrap();

                        let index2 =
                            circuit_ids.iter()
                            .position(|cid| cid == id2).unwrap();

                        let eclass1 = eclasses[index1];
                        let eclass2 = eclasses[index2];
                        let opt_expr = 
                            match op {
                                Operator::Add =>
                                    HEOptCircuit::Add([eclass1, eclass2]),

                                Operator::Sub =>
                                    HEOptCircuit::Sub([eclass1, eclass2]),

                                Operator::Mul =>
                                    HEOptCircuit::Mul([eclass1, eclass2])
                            };

                        let eclass = rec_expr.add(opt_expr);
                        eclasses.push(eclass);
                    },

                    ParamCircuitExpr::Rotate(offset, body_id) => {
                        let offset_eclass =
                            self.offset_to_opt_circuit(offset, &mut rec_expr);

                        let body_index =
                            circuit_ids.iter()
                            .position(|cid| cid == body_id).unwrap();

                        let body_eclass = eclasses[body_index];

                        let eclass =
                            rec_expr.add(HEOptCircuit::Rot([offset_eclass, body_eclass]));

                        eclasses.push(eclass);
                    },
                    
                    ParamCircuitExpr::ReduceDim(dim, extent, op, body_id) => {
                        let dim_eclass =
                            rec_expr.add(HEOptCircuit::IndexVar(Symbol::from(dim)));

                        let body_index =
                            circuit_ids.iter()
                            .position(|cid| cid == body_id).unwrap();

                        let body_eclass = eclasses[body_index];

                        let opt_expr =
                            match op {
                                Operator::Add =>
                                    HEOptCircuit::SumVectors([dim_eclass, body_eclass]),

                                Operator::Mul =>
                                    HEOptCircuit::ProductVectors([dim_eclass, body_eclass]),

                                Operator::Sub =>
                                    panic!("reducing with subtraction not supported yet")
                            };
                        
                        cost_ctx.dim_extent_map.insert(dim.clone(), *extent);
                        let eclass = rec_expr.add(opt_expr);
                        eclasses.push(eclass);
                    },
                }
            }

            rec_exprs.push(rec_expr);
        }

        (rec_exprs, cost_ctx)
    }

    fn offset_to_opt_circuit(&self, offset: &OffsetExpr, rec_expr: &mut RecExpr<HEOptCircuit>) -> egg::Id {
        match offset {
            OffsetExpr::Add(expr1, expr2) => {
                let id1 = self.offset_to_opt_circuit(expr1, rec_expr);
                let id2 = self.offset_to_opt_circuit(expr2, rec_expr); 
                let id = rec_expr.add(HEOptCircuit::Add([id1,id2]));
                id
            },
            
            OffsetExpr::Mul(expr1, expr2) => {
                let id1 = self.offset_to_opt_circuit(expr1, rec_expr);
                let id2 = self.offset_to_opt_circuit(expr2, rec_expr); 
                let id = rec_expr.add(HEOptCircuit::Mul([id1,id2]));
                id
            },

            OffsetExpr::Literal(lit) => {
                let id = rec_expr.add(HEOptCircuit::Literal(*lit));
                id
            },

            OffsetExpr::Var(var) => {
                let id = rec_expr.add(HEOptCircuit::IndexVar(Symbol::from(var)));
                id
            },

            OffsetExpr::FunctionVar(fv, indices) => {
                let index_ids =
                    indices.iter().map(|index|
                        rec_expr.add(HEOptCircuit::IndexVar(Symbol::from(index)))
                    ).collect();

                let id =
                    rec_expr.add(
                        HEOptCircuit::FunctionVar(Symbol::from(fv), index_ids)
                    );

                id
            },
        }
    }

    fn offset_from_opt_circuit_recur(
        &self,
        id: egg::Id,
        rec_expr: &RecExpr<HEOptCircuit>,
        new_registry: &mut CircuitObjectRegistry,
    ) -> OffsetExpr {
        match &rec_expr[id] {
            HEOptCircuit::Literal(lit) => {
                OffsetExpr::Literal(*lit)
            },

            HEOptCircuit::Add([id1, id2]) => {
                let expr1 = 
                    self.offset_from_opt_circuit_recur(id, rec_expr, new_registry);

                let expr2 =
                    self.offset_from_opt_circuit_recur(id, rec_expr, new_registry);

                OffsetExpr::Add(Box::new(expr1), Box::new(expr2))
            },

            HEOptCircuit::Sub(_) => todo!(),

            HEOptCircuit::Mul([id1, id2]) => {
                let expr1 =
                    self.offset_from_opt_circuit_recur(id, rec_expr, new_registry);

                let expr2 =
                    self.offset_from_opt_circuit_recur(id, rec_expr, new_registry);

                OffsetExpr::Mul(Box::new(expr1), Box::new(expr2))
            },

            HEOptCircuit::IndexVar(var) => {
                OffsetExpr::Var(var.to_string())
            },

            HEOptCircuit::FunctionVar(fv, var_ids) => {
                let index_vars: im::Vector<DimName> =
                    var_ids.iter().map(|id| {
                        match rec_expr[*id] {
                            HEOptCircuit::IndexVar(var) => var.to_string(),
                            _ => unreachable!()
                        }
                    }).collect();

                let old_var = fv.to_string();
                let new_var = new_registry.fresh_offset_fvar();
                let circval = self.registry.get_offset_fvar_value(&old_var);
                new_registry.set_offset_var_value(new_var.clone(), circval.clone());

                OffsetExpr::FunctionVar(new_var, index_vars)
            },

            HEOptCircuit::Rot(_) |
            HEOptCircuit::SumVectors(_) |
            HEOptCircuit::ProductVectors(_) |
            HEOptCircuit::CiphertextVar(_) |
            HEOptCircuit::PlaintextVar(_) =>
                unreachable!()
        }
    }

    fn from_opt_circuit_recur(
        &self,
        id: egg::Id,
        rec_expr: &RecExpr<HEOptCircuit>,
        extent_map: &HashMap<DimName, Extent>,
        id_map: &mut HashMap<egg::Id, CircuitId>,
        new_registry: &mut CircuitObjectRegistry,
    ) -> CircuitId {
        if let Some(circ_id) = id_map.get(&id) {
            return *circ_id
        }

        match &rec_expr[id] {
            HEOptCircuit::Literal(lit) => {
                let circ_id = new_registry.register_circuit(ParamCircuitExpr::Literal(*lit));
                id_map.insert(id, circ_id);
                circ_id
            },

            HEOptCircuit::Add([id1, id2]) => {
                let circ_id1 =
                    self.from_opt_circuit_recur(
                         *id1,
                         rec_expr,
                         extent_map,
                         id_map,
                         new_registry
                    );

                let circ_id2 =
                    self.from_opt_circuit_recur(
                         *id2,
                         rec_expr,
                         extent_map,
                         id_map,
                         new_registry
                    );

                let circ_id =
                    new_registry.register_circuit(
                        ParamCircuitExpr::Op(Operator::Add, circ_id1, circ_id2)
                    );

                id_map.insert(id, circ_id);
                circ_id
            },

            HEOptCircuit::Sub([id1, id2]) => {
                let circ_id1 =
                    self.from_opt_circuit_recur(
                         *id1,
                         rec_expr,
                         extent_map,
                         id_map,
                         new_registry
                    );

                let circ_id2 =
                    self.from_opt_circuit_recur(
                         *id2,
                         rec_expr,
                         extent_map,
                         id_map,
                         new_registry
                    );

                let circ_id =
                    new_registry.register_circuit(
                        ParamCircuitExpr::Op(Operator::Sub, circ_id1, circ_id2)
                    );

                id_map.insert(id, circ_id);
                circ_id

            },

            HEOptCircuit::Mul([id1, id2]) => {
                let circ_id1 =
                    self.from_opt_circuit_recur(
                         *id1,
                         rec_expr,
                         extent_map,
                         id_map,
                         new_registry
                    );

                let circ_id2 =
                    self.from_opt_circuit_recur(
                         *id2,
                         rec_expr,
                         extent_map,
                         id_map,
                         new_registry
                    );

                let circ_id =
                    new_registry.register_circuit(
                        ParamCircuitExpr::Op(Operator::Mul, circ_id1, circ_id2)
                    );

                id_map.insert(id, circ_id);
                circ_id
            },

            HEOptCircuit::Rot([offset_id, body_id]) => {
                let offset =
                    self.offset_from_opt_circuit_recur(*offset_id, &rec_expr, new_registry);

                let body =
                    self.from_opt_circuit_recur(
                        *body_id,
                        rec_expr, 
                        extent_map,
                        id_map,
                        new_registry
                    );

                let circuit = ParamCircuitExpr::Rotate(offset, body);
                let circ_id = new_registry.register_circuit(circuit);
                id_map.insert(id, circ_id);
                circ_id
            },

            HEOptCircuit::SumVectors([var_id, body_id]) => {
                let dim_var =
                    match &rec_expr[*var_id] {
                        HEOptCircuit::IndexVar(var) => var.to_string(),
                        _ => unreachable!(),
                    };

                let body =
                    self.from_opt_circuit_recur(
                        *body_id,
                        rec_expr,
                        extent_map,
                        id_map,
                        new_registry
                    );

                let extent = extent_map[&dim_var];
                let circuit = 
                    ParamCircuitExpr::ReduceDim(
                        dim_var,
                        extent,
                        Operator::Add,
                        body
                    );

                let circ_id = new_registry.register_circuit(circuit);
                id_map.insert(id, circ_id);
                circ_id
            },

            HEOptCircuit::ProductVectors([var_id, body_id]) => {
                let dim_var =
                    match &rec_expr[*var_id] {
                        HEOptCircuit::IndexVar(var) => var.to_string(),
                        _ => unreachable!(),
                    };

                let body =
                    self.from_opt_circuit_recur(
                        *body_id,
                        rec_expr,
                        extent_map,
                        id_map,
                        new_registry
                    );

                let extent = extent_map[&dim_var];
                let circuit = 
                    ParamCircuitExpr::ReduceDim(
                        dim_var,
                        extent,
                        Operator::Mul,
                        body
                    );

                let circ_id = new_registry.register_circuit(circuit);
                id_map.insert(id, circ_id);
                circ_id

            },
            
            HEOptCircuit::CiphertextVar(var) => {
                let ct_var = var.to_string();
                let new_var = new_registry.fresh_ct_var();
                let circval = self.registry.get_ct_var_value(&ct_var);
                new_registry.set_ct_var_value(new_var.clone(), circval.clone());

                let circ_id =
                    new_registry.register_circuit(
                        ParamCircuitExpr::CiphertextVar(new_var)
                    );
                id_map.insert(id, circ_id);
                circ_id
            },

            HEOptCircuit::PlaintextVar(var) => {
                let pt_var = var.to_string();
                let new_var = new_registry.fresh_pt_var();
                let circval = self.registry.get_pt_var_value(&pt_var);
                new_registry.set_pt_var_value(new_var.clone(), circval.clone());

                let circ_id =
                    new_registry.register_circuit(
                        ParamCircuitExpr::PlaintextVar(new_var)
                    );
                id_map.insert(id, circ_id);
                circ_id
            },

            HEOptCircuit::IndexVar(_) | HEOptCircuit::FunctionVar(_, _) => 
                unreachable!()
        }
    }

    pub fn from_opt_circuit(&self, rec_exprs: Vec<RecExpr<HEOptCircuit>>) -> Self {
        assert!(self.circuit_expr_list.len() == rec_exprs.len());

        // build extent map
        let mut extent_map: HashMap<DimName, Extent> = HashMap::new();
        self.registry.circuit_map.iter().for_each(|(_, circ)| {
            if let ParamCircuitExpr::ReduceDim(dim, extent, _, _) = circ {
                if let Some(old_extent) = extent_map.insert(dim.clone(), *extent) {
                    if *extent != old_extent {
                        panic!("multiple extents for dim var {}", dim)
                    }
                }
            }
        });

        let mut new_registry = CircuitObjectRegistry::new();
        let mut new_circuit_expr_list: Vec<(ArrayName, Vec<(DimName, Extent)>, CircuitId)> = Vec::new();
        for (i, rec_expr) in rec_exprs.into_iter().enumerate() {
            let mut id_map: HashMap<egg::Id, CircuitId> = HashMap::new();
            let root = rec_expr.as_ref().len() - 1;
            let expr_circ_id =
                self.from_opt_circuit_recur(
                    egg::Id::from(root),
                    &rec_expr,
                    &extent_map,
                    &mut id_map,
                    &mut new_registry
                );

            let (array, dims, _) =
                self.circuit_expr_list.get(i).unwrap();

            new_circuit_expr_list.push(
                (array.clone(), dims.clone(), expr_circ_id)
            );
        }

        ParamCircuitProgram {
            registry: new_registry,
            native_expr_list: Vec::new(),
            circuit_expr_list: new_circuit_expr_list
        }
    }

    fn display_circuit(&self, id: CircuitId) -> String {
        match self.registry.get_circuit(id) {
            ParamCircuitExpr::CiphertextVar(var) => {
                var.clone()
            },

            ParamCircuitExpr::PlaintextVar(var) => {
                var.clone()
            },

            ParamCircuitExpr::Literal(lit) => {
                lit.to_string()
            },

            ParamCircuitExpr::Op(op, id1, id2) => {
                let str1 = self.display_circuit(*id1);
                let str2 = self.display_circuit(*id2);
                format!("({} {} {})", str1, op, str2)
            },
            
            ParamCircuitExpr::Rotate(offset, body_id) => {
                let body_str = self.display_circuit(*body_id);
                format!("rot({}, {})", offset, body_str)
            },

            ParamCircuitExpr::ReduceDim(dim, extent, op, body_id) => {
                let body_str = self.display_circuit(*body_id);
                format!("reduce_dim({}:{}, {}, {})", dim, extent, op, body_str)
            }
        }
    }
}

impl Display for ParamCircuitProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.registry)?;

        self.circuit_expr_list.iter().try_for_each(|(name, dims, circuit)| {
            let mut dims_str = String::new();
            for dim in dims.iter() {
                dims_str.push_str(&format!("[{}: {}]", dim.0, dim.1))
            }
            write!(f, "let {}{} = {}\n", name, dims_str, self.display_circuit(*circuit))
        })
    }
}