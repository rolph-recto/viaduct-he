use egg::{RecExpr, Symbol};
use itertools::Itertools;
use std::{
    collections::{HashMap, HashSet, LinkedList},
    fmt::Display,
    ops::Range,
};

use crate::{
    circ::{optimizer::HEOptCircuit, vector_info::VectorInfo},
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

    fn to_opt_circuit_recur(&self, opt_expr: &mut RecExpr<HEOptCircuit>) -> egg::Id {
        todo!()
        /*
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
        */
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

impl Display for ParamCircuitProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.registry)?;

        self.circuit_expr_list.iter().try_for_each(|(name, dims, circuit)| {
            let mut dims_str = String::new();
            for dim in dims.iter() {
                dims_str.push_str(&format!("[{}: {}]", dim.0, dim.1))
            }
            write!(f, "let {}{} = {}\n", name, dims_str, circuit)
        })
    }
}
