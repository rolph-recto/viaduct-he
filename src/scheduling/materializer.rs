use std::{collections::{HashMap, HashSet}, fmt::Display, ops::Range};
use itertools::{Itertools, MultiProduct};

use crate::{
    lang::{
        Operator, BaseArrayTransform,
        index_elim2::{TransformedProgram, TransformedExpr}, ExprRefId
    },
    scheduling::{ArraySchedule, ExprSchedule, OffsetExpr, DimName, Schedule, ScheduleDim}
};

use super::ParamArrayTransform;

type VarName = String;

// parameterized circuit expr that represents an *array* of circuit exprs
#[derive(Clone,Debug)]
pub enum ParamCircuitExpr {
    CiphertextVar(VarName),
    PlaintextVar(VarName),
    Literal(isize),
    Op(Operator, Box<ParamCircuitExpr>, Box<ParamCircuitExpr>),
    Rotate(Box<OffsetExpr>, Box<ParamCircuitExpr>),
    Reduce(HashSet<DimName>, Operator, Box<ParamCircuitExpr>),
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

            ParamCircuitExpr::Reduce(indices, op, expr) => {
                write!(f, "reduce({:?}, {}, {})", indices, op, expr)
            },
        }
    }
}

pub struct ParamCircuitProgram {
    schedule: ExprSchedule,
    expr: ParamCircuitExpr,
    registry: VectorRegistry,
}

impl Display for ParamCircuitProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}", self.schedule, self.expr)
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

    pub fn coord_iter(&self) -> MultiProduct<Range<usize>> {
        self.0.iter()
        .map(|(_, extent)| (0..*extent))
        .multi_cartesian_product()
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

    pub fn coord_iter(&self) -> MultiProduct<Range<usize>> {
        self.coord_system.coord_iter()
    }

    pub fn set(&mut self, coord: IndexCoord, value: T) {
        assert!(self.coord_map.contains_key(&coord));
        self.coord_map.insert(coord, value);
    }

    pub fn get(&self, coord: IndexCoord) -> &T {
        &self.coord_map[&coord]
    }
}

pub enum CiphertextObject {
    Null,
    ArrayVector(BaseArrayTransform)
}

impl Default for CiphertextObject {
    fn default() -> Self { CiphertextObject::Null }
}

pub enum PlaintextObject { Null }

impl Default for PlaintextObject {
    fn default() -> Self { PlaintextObject::Null }
}

pub struct VectorRegistry {
    ct_var_values: HashMap<VarName, IndexCoordinateMap<CiphertextObject>>,
    ct_var_id: usize,

    pt_var_values: HashMap<VarName, IndexCoordinateMap<PlaintextObject>>,
    pt_var_id: usize
}

impl VectorRegistry {
    fn new() -> Self {
        VectorRegistry {
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

    pub fn set_ciphertext_coord_map(&mut self, ct_var: VarName, coord_map: IndexCoordinateMap<CiphertextObject>) {
        self.ct_var_values.insert(ct_var, coord_map);
    }

    pub fn set_plaintext_coord_map(&mut self, pt_var: VarName, coord_map: IndexCoordinateMap<PlaintextObject>) {
        self.pt_var_values.insert(pt_var, coord_map);
    }

    pub fn get_ciphertext_coord_map(&mut self, ct_var: VarName) -> &IndexCoordinateMap<CiphertextObject> {
        self.ct_var_values.get(&ct_var).unwrap()
    }

    pub fn get_plaintext_coord_map(&mut self, pt_var: VarName) -> &IndexCoordinateMap<PlaintextObject> {
        self.pt_var_values.get(&pt_var).unwrap()
    }
}

pub trait ArrayMaterializer {
    fn can_materialize(&self, param_transform: &ParamArrayTransform) -> bool;
    fn materialize(&self, param_transform: &ParamArrayTransform, registry: &mut VectorRegistry) -> ParamCircuitExpr;
}

pub struct Materializer {
    array_materializers: Vec<Box<dyn ArrayMaterializer>>,
    registry: VectorRegistry,
}

impl Materializer {
    pub fn new(array_materializers: Vec<Box<dyn ArrayMaterializer>>) -> Self {
        Materializer { array_materializers, registry: VectorRegistry::new() }
    }

    pub fn materialize(
        mut self,
        program: &TransformedProgram,
        schedule: &Schedule
    ) -> Result<ParamCircuitProgram, String> {
        let (schedule, expr) = self.materialize_expr(&program.inputs, &program.expr, schedule)?;
        Ok(
            ParamCircuitProgram {
                schedule, expr, registry: self.registry
            }
        )
    }

    // TODO: refactor logic of computing output schedules into a separate struct
    // since this share the same logic as Schedule::compute_output_schedule
    fn materialize_expr(
        &mut self,
        inputs: &HashMap<ExprRefId, BaseArrayTransform>,
        expr: &TransformedExpr,
        schedule: &Schedule
    ) -> Result<(ExprSchedule, ParamCircuitExpr), String> {
        match expr {
            TransformedExpr::Literal(lit) => {
                Ok((ExprSchedule::Any, ParamCircuitExpr::Literal(*lit)))
            },

            TransformedExpr::Op(op, expr1, expr2) => {
                let (sched1, mat1) = self.materialize_expr(inputs, expr1, schedule)?;
                let (sched2, mat2) = self.materialize_expr(inputs, expr2, schedule)?;

                let expr = 
                    ParamCircuitExpr::Op(op.clone(), Box::new(mat1), Box::new(mat2));

                let schedule = 
                    match (sched1, sched2) {
                        (ExprSchedule::Any, ExprSchedule::Any) =>
                            ExprSchedule::Any,

                        (ExprSchedule::Any, ExprSchedule::Specific(sched2)) => 
                            ExprSchedule::Specific(sched2),

                        (ExprSchedule::Specific(sched1), ExprSchedule::Any) =>
                            ExprSchedule::Specific(sched1),

                        (ExprSchedule::Specific(sched1), ExprSchedule::Specific(sched2)) => {
                            println!("sched1: {}; sched2: {}", sched1, sched2);
                            assert!(sched1 == sched2);
                            ExprSchedule::Specific(sched1)
                        }
                    };

                Ok((schedule, expr))
            },

            // TODO support reduction in vectorized dims
            TransformedExpr::ReduceNode(reduced_index, op, body) => {
                let (body_sched, mat_body) =
                    self.materialize_expr(inputs, body, schedule)?;

                match body_sched {
                    ExprSchedule::Any => Err("Cannot reduce a literal expression".to_string()),

                    ExprSchedule::Specific(body_sched_spec) => {
                        let mut new_exploded_dims: im::Vector<ScheduleDim> = im::Vector::new();
                        let mut reduced_index_vars: HashSet<DimName> = HashSet::new();
                        for mut dim in body_sched_spec.exploded_dims {
                            if dim.index == *reduced_index { // dim is reduced, remove it
                                reduced_index_vars.insert(dim.name);

                            } else if dim.index > *reduced_index { // decrease dim index
                                dim.index -= 1;
                                new_exploded_dims.push_back(dim);

                            } else {
                                new_exploded_dims.push_back(dim);
                            }
                        }

                        let schedule = 
                            ExprSchedule::Specific(
                                ArraySchedule {
                                    exploded_dims: new_exploded_dims,
                                    vectorized_dims: body_sched_spec.vectorized_dims,
                                }
                            );

                        let expr = 
                            ParamCircuitExpr::Reduce(
                                reduced_index_vars,
                                op.clone(),
                                Box::new(mat_body)
                            );

                        Ok((schedule, expr))
                    }
                }
            },

            TransformedExpr::ExprRef(ref_id) => {
                let transform = &inputs[ref_id];
                let transform_schedule = &schedule.schedule_map[ref_id];
                let param_transform = transform_schedule.apply_schedule(transform);

                for amat in self.array_materializers.iter() {
                    if amat.can_materialize(&param_transform) {
                        let expr = amat.materialize(&param_transform, &mut self.registry);
                        let schedule = ExprSchedule::Specific(transform_schedule.clone());
                        return Ok((schedule, expr))
                    }
                }

                Err(format!("No array materializer can process {}", param_transform))
            },
        }
    }
}

// array materializer that doesn't attempt to derive vectors
pub struct DummyArrayMaterializer {}

impl ArrayMaterializer for DummyArrayMaterializer {
    // the dummy materializer can materialize any transform
    fn can_materialize(&self, _param_transform: &ParamArrayTransform) -> bool {
        true
    }

    fn materialize(&self, param_transform: &ParamArrayTransform, registry: &mut VectorRegistry) -> ParamCircuitExpr {
        let ct_var = registry.fresh_ciphertext_var();
        let coord_map: IndexCoordinateMap<CiphertextObject> =
            IndexCoordinateMap::new(param_transform.exploded_dims.iter());
        registry.set_ciphertext_coord_map(ct_var.clone(), coord_map);
        
        ParamCircuitExpr::CiphertextVar(ct_var)
    }
}

#[cfg(test)]
mod tests{
    use crate::lang::{parser::ProgramParser, index_elim2::IndexElimination2, source::SourceProgram};
    use super::*;

    // generate an initial schedule for a program
    fn test_materializer(src: &str) {
        let parser = ProgramParser::new();
        let program: SourceProgram = parser.parse(src).unwrap();

        let mut index_elim = IndexElimination2::new();
        let res = index_elim.run(&program);
        
        assert!(res.is_ok());

        let program = res.unwrap();
        let init_schedule = Schedule::gen_initial_schedule(&program);

        let materializer =
            Materializer::new(vec![Box::new(DummyArrayMaterializer {})]);

        let res_mat = materializer.materialize(&program, &init_schedule);
        assert!(res_mat.is_ok());

        let param_circ = res_mat.unwrap();
        println!("{}", param_circ.schedule);
        println!("{}", param_circ.expr);
    }

    #[test]
    fn test_imgblur() {
        test_materializer(
        "input img: [(0,16),(0,16)]
            for x: (0, 16) {
                for y: (0, 16) {
                    img[x-1][y-1] + img[x+1][y+1]
                }
            }"
        );
    }

    #[test]
    fn test_imgblur2() {
        test_materializer(
        "input img: [(0,16),(0,16)]
            let res = 
                for x: (0, 16) {
                    for y: (0, 16) {
                        img[x-1][y-1] + img[x+1][y+1]
                    }
                }
            in
            for x: (0, 16) {
                for y: (0, 16) {
                    res[x-2][y-2] + res[x+2][y+2]
                }
            }
            "
        );
    }

    #[test]
    fn test_convolve() {
        test_materializer(
        "input img: [(0,16),(0,16)]
            let conv1 = 
                for x: (0, 15) {
                    for y: (0, 15) {
                        img[x][y] + img[x+1][y+1]
                    }
                }
            in
            for x: (0, 14) {
                for y: (0, 14) {
                    conv1[x][y] + conv1[x+1][y+1]
                }
            }
            "
        );
    }

    #[test]
    fn test_matmatmul() {
        test_materializer(
            "input A: [(0,4),(0,4)]
            input B: [(0,4),(0,4)]
            for i: (0,4) {
                for j: (0,4) {
                    sum(for k: (0,4) { A[i][k] * B[k][j] })
                }
            }"
        );
    }

    #[test]
    fn test_matmatmul2() {
        test_materializer(
            "input A1: [(0,4),(0,4)]
            input A2: [(0,4),(0,4)]
            input B: [(0,4),(0,4)]
            let res =
                for i: (0,4) {
                    for j: (0,4) {
                        sum(for k: (0,4) { A1[i][k] * B[k][j] })
                    }
                }
            in
            for i: (0,4) {
                for j: (0,4) {
                    sum(for k: (0,4) { A2[i][k] * res[k][j] })
                }
            }
            "
        );
    }

    #[test]
    fn test_dotprod_pointless() {
        test_materializer(
        "
            input A: [(0,3)]
            input B: [(0,3)]
            sum(A * B)
            "
        );
    }

    #[test]
    fn test_matvecmul() {
        test_materializer(
        "
            input M: [(0,1),(0,1)]
            input v: [(0,1)]
            for i: (0,1) {
                sum(M[i] * v)
            }
            "
        );
    }
}