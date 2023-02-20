use std::collections::{HashMap, HashSet};
use bimap::BiHashMap;

use crate::{
    circ2::{
        IndexCoordinateMap, CiphertextObject, PlaintextObject, ParamCircuitExpr,
        CircuitRegistry, ParamCircuitProgram, IndexCoord, CircuitVarValue,
        vector_info::VectorInfo,
    },
    lang::{
        ArrayTransform, Shape, DimContent, Operator, 
        index_elim2::{TransformedProgram, TransformedExpr},
    },
    scheduling::{
        ArraySchedule, ExprScheduleType, DimName, OffsetExpr, Schedule,
        ClientPreprocessing, VectorScheduleDim
    },
    util
};

pub trait ArrayMaterializer {
    fn can_materialize(
        &self,
        array_shape: &Shape,
        schedule: &ArraySchedule,
        transform: &ArrayTransform,
    ) -> bool;

    fn materialize(
        &mut self,
        array_shape: &Shape,
        schedule: &ArraySchedule,
        transform: &ArrayTransform,
        registry: &mut CircuitRegistry
    ) -> ParamCircuitExpr;
}

/// materializes a schedule for an index-free program.
pub struct Materializer {
    array_materializers: Vec<Box<dyn ArrayMaterializer>>,
    registry: CircuitRegistry,
}

impl Materializer {
    pub fn new(array_materializers: Vec<Box<dyn ArrayMaterializer>>) -> Self {
        Materializer { array_materializers, registry: CircuitRegistry::new() }
    }

    /// packages the materialized expr with the vector registry
    pub fn materialize(
        mut self,
        program: &TransformedProgram,
        schedule: &Schedule
    ) -> Result<ParamCircuitProgram, String> {
        let (schedule, expr) = self.materialize_expr(&program, &program.expr, schedule)?;
        Ok(
            ParamCircuitProgram {
                schedule, expr, registry: self.registry
            }
        )
    }

    fn materialize_expr(
        &mut self,
        program: &TransformedProgram,
        expr: &TransformedExpr,
        schedule: &Schedule
    ) -> Result<(ExprScheduleType, ParamCircuitExpr), String> {
        match expr {
            TransformedExpr::Literal(lit) => {
                let sched_lit = Schedule::schedule_literal()?;
                Ok((sched_lit, ParamCircuitExpr::Literal(*lit)))
            },

            TransformedExpr::Op(op, expr1, expr2) => {
                let (sched1, mat1) = self.materialize_expr(program, expr1, schedule)?;
                let (sched2, mat2) = self.materialize_expr(program, expr2, schedule)?;
                let schedule = Schedule::schedule_op(&sched1, &sched2)?;
                let expr = ParamCircuitExpr::Op(op.clone(), Box::new(mat1), Box::new(mat2));
                Ok((schedule, expr))
            },

            // TODO support client transforms
            TransformedExpr::ReduceNode(reduced_index, op, body) => {
                let (body_sched, mat_body) =
                    self.materialize_expr(program, body, schedule)?;

                let schedule = Schedule::schedule_reduce(*reduced_index, &body_sched)?;

                if let ExprScheduleType::Specific(body_sched_spec) = body_sched {
                    let mut reduced_index_vars: HashSet<(DimName,usize)> = HashSet::new();
                    for dim in body_sched_spec.exploded_dims.iter() {
                        if dim.index == *reduced_index { // dim is reduced, remove it
                            reduced_index_vars.insert((dim.name.clone(), dim.extent));
                        }
                    }

                    let mut reduction_list: Vec<usize> = Vec::new();
                    let mut block_size: usize = body_sched_spec.vector_size();

                    for dim in body_sched_spec.vectorized_dims.into_iter() {
                        block_size /= dim.extent();

                        if let VectorScheduleDim::Filled(sched_dim) = dim {
                            // if extent is 1, there's nothing to reduce!
                            if sched_dim.index == *reduced_index && sched_dim.extent > 1 {
                                reduction_list.extend(
                                    util::gen_pow2_list(sched_dim.extent >> 1)
                                    .iter().map(|x| x * block_size)
                                );
                            }
                        }
                    }

                    let expr_vec = 
                        reduced_index_vars.into_iter().fold(
                            mat_body,
                            |acc, (var, extent)| {
                                ParamCircuitExpr::ReduceVectors(
                                    var,
                                    extent,
                                    op.clone(),
                                    Box::new(acc)
                                )
                            }
                        );

                    let expr = 
                        reduction_list.into_iter().fold(
                            expr_vec,
                            |acc, n| {
                                ParamCircuitExpr::Op(
                                    Operator::Add,
                                    Box::new(acc.clone()),
                                    Box::new(
                                        ParamCircuitExpr::Rotate(
                                            Box::new(OffsetExpr::Literal(-(n as isize))),
                                            Box::new(acc)
                                        )
                                    )
                                )
                            }
                        );

                    Ok((schedule, expr))

                } else {
                    unreachable!()
                }
            },

            // this is assumed to be a transformation of an input array
            TransformedExpr::ExprRef(ref_id) => {
                let schedule = &schedule.schedule_map[ref_id];
                let transform = &program.inputs[ref_id];
                let array_shape = &program.array_shapes[&transform.array];

                for amat in self.array_materializers.iter_mut() {
                    if amat.can_materialize(array_shape, schedule, transform) {
                        let expr = amat.materialize(array_shape, schedule, transform, &mut self.registry);
                        let schedule = ExprScheduleType::Specific(schedule.to_expr_schedule());
                        return Ok((schedule, expr))
                    }
                }

                Err(format!("No array materializer can process expr ref {}", ref_id))
            },
        }
    }
}

// array materializer that doesn't attempt to derive vectors
pub struct DummyArrayMaterializer {}

impl ArrayMaterializer for DummyArrayMaterializer {
    // the dummy materializer can only materialize arrays w/o client preprocessing
    fn can_materialize(
        &self,
        _array_shape: &Shape,
        schedule: &ArraySchedule,
        _transform: &ArrayTransform,
    ) -> bool {
        schedule.preprocessing.is_none()
    }

    fn materialize(
        &mut self,
        array_shape: &Shape,
        schedule: &ArraySchedule,
        transform: &ArrayTransform,
        registry: &mut CircuitRegistry
    ) -> ParamCircuitExpr {
        let ct_var = registry.fresh_ct_var();
        let mut coord_map: IndexCoordinateMap<CiphertextObject> =
            IndexCoordinateMap::new(schedule.exploded_dims.iter());

        let index_vars = coord_map.index_vars();

        // register vectors
        for coord in coord_map.coord_iter() {
            let index_map: HashMap<DimName, usize> =
                index_vars.clone().into_iter().zip(coord.clone()).collect();

            let vector =
                VectorInfo::get_vector_at_coord(array_shape, &index_map, schedule, transform, None);

            coord_map.set(coord, CiphertextObject::Vector(vector));
        }
        
        registry.set_ct_var_value(ct_var.clone(), CircuitVarValue::CoordMap(coord_map));
        ParamCircuitExpr::CiphertextVar(ct_var)
    }
}

// array materialize that will derive vectors through rotation and masking
type VectorId = usize;

pub struct VectorDeriver {
    cur_vector_id: VectorId,
    vector_map: BiHashMap<VectorId, VectorInfo>,
    parent_map: HashMap<VectorId, VectorId>,
}

impl VectorDeriver {
    pub fn new() -> Self {
        VectorDeriver {
            cur_vector_id: 1,
            vector_map: BiHashMap::new(),
            parent_map: HashMap::new(),
        }
    }

    pub fn register_vector(&mut self, vector: VectorInfo) -> VectorId {
        if let Some(id) = self.vector_map.get_by_right(&vector) {
            *id

        } else {
            let id = self.cur_vector_id;
            self.cur_vector_id += 1;
            self.vector_map.insert(id, vector);
            id
        }
    }

    pub fn find_immediate_parent(&self, id: VectorId) -> VectorId {
        let vector = self.vector_map.get_by_left(&id).unwrap();
        for (id2, vector2) in self.vector_map.iter() {
            if id != *id2 {
                if vector2.derive(vector).is_some() {
                    return *id2
                }
            }
        }

        id
    }

    // find immediate parent for each vector
    pub fn compute_immediate_parents(&mut self) {
        for (vector_id, _) in self.vector_map.iter() {
            let parent_id = self.find_immediate_parent(*vector_id);
            self.parent_map.insert(*vector_id, parent_id);
        }
    }

    pub fn find_transitive_parent(&self, id: VectorId) -> VectorId {
        let mut cur_id = id;
        let mut parent_id = self.parent_map[&cur_id];

        while parent_id != cur_id {
            cur_id = parent_id;
            parent_id = self.parent_map[&cur_id];
        }

        parent_id
    }

    pub fn get_vector(&self, id: VectorId) -> &VectorInfo {
        self.vector_map.get_by_left(&id).unwrap()
    }

    pub fn register_and_derive_vectors(
        &mut self,
        array_shape: &Shape,
        schedule: &ArraySchedule,
        transform: &ArrayTransform,
        preprocessing: Option<ClientPreprocessing>,
        coords: impl Iterator<Item=IndexCoord> + Clone,
        obj_map: &mut IndexCoordinateMap<CiphertextObject>,
        mask_map: &mut IndexCoordinateMap<PlaintextObject>,
        step_map: &mut IndexCoordinateMap<isize>,
    ) {
        let mut vector_id_map: HashMap<IndexCoord, VectorId> = HashMap::new();
        let index_vars = obj_map.index_vars();

        for coord in coords.clone() {
            let index_map: HashMap<DimName, usize> =
                index_vars.clone().into_iter().zip(coord.clone()).collect();

            let vector =
                VectorInfo::get_vector_at_coord(array_shape, &index_map, schedule, transform, preprocessing);

            let vector_id = self.register_vector(vector);
            vector_id_map.insert(coord, vector_id);
        }

        self.compute_immediate_parents();

        // find transitive parents
        for coord in coords {
            let vector_id = *vector_id_map.get(&coord).unwrap();
            let parent_id = self.find_transitive_parent(vector_id);

            if vector_id != parent_id { // the vector is derived from some parent 
                let vector = self.get_vector(vector_id);
                let parent = self.get_vector(parent_id);
                let (steps, mask) = parent.derive(vector).unwrap();

                step_map.set(coord.clone(), steps);
                mask_map.set(coord.clone(), mask);
                obj_map.set(coord, CiphertextObject::Vector(parent.clone()));

            } else { // the vector is not derived
                let vector = self.get_vector(vector_id);
                step_map.set(coord.clone(), 0);
                obj_map.set(coord, CiphertextObject::Vector(vector.clone()));
            }
        }
    }

    // assume that the rotation steps have a linear relationship to the index vars,
    // then probe certain coordinates to compute an offset expr
    // this can compute linear offsets for a *subset* of defined coords;
    // hence this function takes in extra arguments
    // valid_coords and processed_index_vars
    pub fn compute_linear_offset(
        &self,
        step_map: &IndexCoordinateMap<isize>,
        valid_coords: impl Iterator<Item=IndexCoord> + Clone,
        processed_index_vars: Vec<DimName>,
    ) -> Option<OffsetExpr> {
        let index_vars = step_map.index_vars();

        // probe at (0,...,0) to get the base offset
        let base_coord: im::Vector<usize> = im::Vector::from(vec![0; index_vars.len()]);
        let base_offset: isize = *step_map.get(&base_coord);

        // probe at (0,..,1,..,0) to get the coefficient for the ith index var
        // only do this for processed_index_vars, not *all* index vars
        let mut coefficients: Vec<isize> = Vec::new();
        for i in 0..processed_index_vars.len() {
            let mut index_coord = base_coord.clone();
            index_coord[i] = 1;

            let step_offset = *step_map.get(&index_coord);
            coefficients.push(step_offset - base_offset);
        }

        // build offset expr from base offset and coefficients
        let offset_expr =
            coefficients.iter()
            .zip(index_vars.clone())
            .fold(OffsetExpr::Literal(base_offset), |acc, (coeff, index_var)| {
                if *coeff != 0 {
                    OffsetExpr::Add(
                        Box::new(acc),
                        Box::new(
                            OffsetExpr::Mul(
                                Box::new(OffsetExpr::Literal(*coeff)),
                                Box::new(OffsetExpr::Var(index_var.clone()))
                            )
                        )
                    )
                } else {
                    acc
                }
            });

        // validate computed offset expr
        for coord in valid_coords {
            let value = *step_map.get(&coord);
            let index_map: HashMap<DimName, usize> =
                index_vars.clone().into_iter().zip(coord.clone()).collect();

            let predicted_value = offset_expr.eval(&index_map);
            if value != predicted_value {
                return None
            }
        }

        // this expression is correct for all valid_coords; return it
        Some(offset_expr)
    }

    // default method for generating circuit expression for an array materializer
    pub fn gen_circuit_expr(
        &mut self,
        array_shape: &Shape,
        schedule: &ArraySchedule,
        transform: &ArrayTransform,
        registry: &mut CircuitRegistry,
        preprocessing: Option<ClientPreprocessing>,
    ) -> ParamCircuitExpr {
        let mut obj_map: IndexCoordinateMap<CiphertextObject> =
            IndexCoordinateMap::new(schedule.exploded_dims.iter());
        let mut mask_map: IndexCoordinateMap<PlaintextObject> =
            IndexCoordinateMap::new(schedule.exploded_dims.iter());
        let mut step_map: IndexCoordinateMap<isize> =
            IndexCoordinateMap::new(schedule.exploded_dims.iter());
        let index_vars = obj_map.index_vars();
        let coords = obj_map.coord_iter();

        self.register_and_derive_vectors(
            array_shape,
            schedule,
            transform,
            preprocessing,
            coords.clone(),
            &mut obj_map,
            &mut mask_map,
            &mut step_map);

        let ct_var = registry.fresh_ct_var();
        let pt_var = registry.fresh_pt_var();

        if !obj_map.is_empty() { // there is an array of vectors
            let mask_is_nonconst =
                mask_map.value_iter().any(|(_, mask)| {
                    match mask {
                        PlaintextObject::Const(_) => false,
                        PlaintextObject::Mask(_) => true,
                    }
                });

            registry.set_ct_var_value(ct_var.clone(), CircuitVarValue::CoordMap(obj_map));
            registry.set_pt_var_value(pt_var.clone(), CircuitVarValue::CoordMap(mask_map));

            let masked_expr =
                if mask_is_nonconst {
                    ParamCircuitExpr::Op(
                        Operator::Mul,
                        Box::new(ParamCircuitExpr::CiphertextVar(ct_var)),
                        Box::new(ParamCircuitExpr::PlaintextVar(pt_var))
                    )

                } else {
                    ParamCircuitExpr::CiphertextVar(ct_var)
                };

            // attempt to compute offset expr
            let offset_expr_opt =
                self.compute_linear_offset(&step_map, coords, index_vars);

            if let Some(linear_offset_expr) = offset_expr_opt {
                if let Some(0) = linear_offset_expr.const_value() {
                    masked_expr

                } else {
                    ParamCircuitExpr::Rotate(
                        Box::new(linear_offset_expr),
                        Box::new(masked_expr)
                    )
                }

            } else { // introduce new offset variable, since we can't create an offset expr
                let offset_var = registry.fresh_offset_var();
                registry.set_offset_var_value(offset_var.clone(), CircuitVarValue::CoordMap(step_map));

                ParamCircuitExpr::Rotate(
                    Box::new(OffsetExpr::Var(offset_var)),
                    Box::new(masked_expr)
                )
            }

        } else { // there is only a single vector
            let index_map: HashMap<DimName, usize> = HashMap::new();
            let vector =
                VectorInfo::get_vector_at_coord(array_shape, &index_map, schedule, transform, preprocessing);

            registry.set_ct_var_value(
                ct_var.clone(),
                CircuitVarValue::Object(CiphertextObject::Vector(vector))
            );
            ParamCircuitExpr::CiphertextVar(ct_var)
        }
    }
}

pub struct DefaultArrayMaterializer {
    deriver: VectorDeriver,
}

impl DefaultArrayMaterializer {
    pub fn new() -> Self {
        DefaultArrayMaterializer {
            deriver: VectorDeriver::new(),
        }
    }
}

impl ArrayMaterializer for DefaultArrayMaterializer {
    /// the default materializer can only apply when there is no client preprocessing
    fn can_materialize(
        &self,
        _array_shape: &Shape,
        schedule: &ArraySchedule,
        _transform: &ArrayTransform,
    ) -> bool {
        schedule.preprocessing.is_none()
    }

    fn materialize(
        &mut self,
        shape: &Shape,
        schedule: &ArraySchedule,
        transform: &ArrayTransform,
        registry: &mut CircuitRegistry
    ) -> ParamCircuitExpr {
        self.deriver.gen_circuit_expr(shape, schedule, transform, registry, None)
    }
}

pub struct DiagonalArrayMaterializer { deriver: VectorDeriver }

impl DiagonalArrayMaterializer {
    pub fn new() -> Self {
        DiagonalArrayMaterializer { deriver: VectorDeriver::new() }
    }
}

impl ArrayMaterializer for DiagonalArrayMaterializer {
    fn can_materialize(
        &self,
        _array_shape: &Shape,
        schedule: &ArraySchedule,
        _base: &ArrayTransform,
    ) -> bool {
        if let Some(ClientPreprocessing::Permute(dim_i, dim_j)) = schedule.preprocessing {
            // dim i must be exploded and dim j must be the outermost vectorized dim
            let i_exploded = 
                schedule.exploded_dims.iter().any(|edim| edim.index == dim_i);

            let j_outermost_vectorized =
                schedule.vectorized_dims.len() > 0 &&
                schedule.vectorized_dims.head().unwrap().index == dim_j;

            // dim i and j must have both have the same tiling that corresponds
            // to the permutation transform
            // TODO: for now, assume i and j are NOT tiled
            let tiling_i = schedule.get_tiling(dim_i);
            let tiling_j = schedule.get_tiling(dim_j);
            tiling_i == tiling_j && tiling_i.len() == 1 && i_exploded && j_outermost_vectorized

        } else {
            false
        }
    }
    
    fn materialize(
        &mut self,
        shape: &Shape,
        schedule: &ArraySchedule,
        transform: &ArrayTransform,
        registry: &mut CircuitRegistry
    ) -> ParamCircuitExpr {
        if let Some(ClientPreprocessing::Permute(dim_i, dim_j)) = schedule.preprocessing {
            match (&transform.dims[dim_i], &transform.dims[dim_j]) {
                // if dim i is empty, then the permutation is a no-op
                // materialize the schedule normally
                (DimContent::EmptyDim { extent: _ }, _) => {
                    self.deriver.gen_circuit_expr(
                        shape,
                        schedule,
                        transform,
                        registry, 
                        None
                    )
                },

                // if dim j is a filled dim, then the permutation must actually
                // be done by the client; record this fact and then materialize
                // the schedule normally
                (DimContent::FilledDim { dim: idim_i, extent: _, stride: _ },
                DimContent::FilledDim { dim: idim_j, extent: _, stride: _ }) => {
                    self.deriver.gen_circuit_expr(
                        shape,
                        schedule,
                        transform,
                        registry,
                        Some(ClientPreprocessing::Permute(*idim_i, *idim_j))
                    )
                },

                // if dim j is an empty dim, then we can apply the "diagonal"
                // trick from Halevi and Schoup for matrix-vector multiplication
                // to do this, follow these steps:
                // 1. switch innermost tiles of dim i and dim j
                //    (assuming all tiles of i is exploded and only innermost tile of j is vectorized)
                // 2. derive vectors assuming j = 0
                // 3. to fill in the rest of the vectors along dim j by rotating
                //    the vectors at dim j = 0
                (DimContent::FilledDim { dim: dim_i_dim, extent: extent_i, stride: stride_i },
                DimContent::EmptyDim { extent: extent_j }) => {
                    let mut obj_map: IndexCoordinateMap<CiphertextObject> =
                        IndexCoordinateMap::new(schedule.exploded_dims.iter());
                    let mut mask_map: IndexCoordinateMap<PlaintextObject> =
                        IndexCoordinateMap::new(schedule.exploded_dims.iter());
                    let mut step_map: IndexCoordinateMap<isize> =
                        IndexCoordinateMap::new(schedule.exploded_dims.iter());
                    let mut new_schedule = schedule.clone();

                    // switch innermost tiles of i and j in the schedule
                    let inner_i_dim = 
                        new_schedule.exploded_dims.iter_mut()
                        .find(|dim| dim.index == dim_i && dim.stride == 1)
                        .unwrap();
                    let inner_i_dim_name = inner_i_dim.name.clone();
                    let inner_i_dim_extent = inner_i_dim.extent;
                    inner_i_dim.index = dim_j;

                    let inner_j_dim = new_schedule.vectorized_dims.get_mut(0).unwrap();
                    inner_j_dim.index = dim_i;

                    let zero_inner_j_coords =
                        obj_map.coord_iter_subset(&inner_i_dim_name, 0..1);

                    self.deriver.register_and_derive_vectors(
                        shape,
                        &new_schedule,
                        transform,
                        None,
                        zero_inner_j_coords.clone(),
                        &mut obj_map,
                        &mut mask_map,
                        &mut step_map);

                    let mut processed_index_vars = obj_map.index_vars();

                    // remember, inner i and inner j are swapped,
                    // so inner j now has the name of inner i!
                    let inner_j_name_index =
                        processed_index_vars.iter()
                        .position(|name| *name == inner_i_dim_name)
                        .unwrap();

                    processed_index_vars.remove(inner_j_name_index);

                    let ct_var = registry.fresh_ct_var();

                    // given expr e is at coord where inner_j=0,
                    // expr rot(inner_j, e) is at coord where inner_j != 0
                    let rest_inner_j_coords =
                        obj_map.coord_iter_subset(&inner_i_dim_name, 1..inner_i_dim_extent);

                    for coord in rest_inner_j_coords {
                        let mut ref_coord = coord.clone();
                        ref_coord[inner_j_name_index] = 0;

                        let ref_obj = obj_map.get(&ref_coord).clone();
                        let ref_step = *step_map.get(&ref_coord);
                        let inner_j_value = coord[inner_j_name_index];

                        obj_map.set(coord.clone(), ref_obj);
                        step_map.set(coord, ref_step + (inner_j_value as isize));
                    }

                    // attempt to compute offset expr
                    let offset_expr_opt =
                        if processed_index_vars.len() > 0 {
                            self.deriver.compute_linear_offset(
                                &step_map,
                                zero_inner_j_coords,
                                processed_index_vars
                            )

                        } else {
                            let zero_j_coord = im::vector![0];
                            let step = *step_map.get(&zero_j_coord);
                            Some(OffsetExpr::Literal(step))
                        };

                    registry.set_ct_var_value(ct_var.clone(), CircuitVarValue::CoordMap(obj_map));

                    if let Some(offset_expr) = offset_expr_opt {
                        let new_offset_expr =
                            OffsetExpr::Add(
                                Box::new(offset_expr),
                                Box::new(OffsetExpr::Var(inner_i_dim_name.clone()))
                            );

                        ParamCircuitExpr::Rotate(
                            Box::new(new_offset_expr),
                            Box::new(ParamCircuitExpr::CiphertextVar(ct_var))
                        )

                    } else {
                        let offset_var = registry.fresh_offset_var();
                        registry.set_offset_var_value(offset_var.clone(), CircuitVarValue::CoordMap(step_map));

                        ParamCircuitExpr::Rotate(
                            Box::new(OffsetExpr::Var(offset_var)),
                            Box::new(ParamCircuitExpr::CiphertextVar(ct_var))
                        )
                    }
                },
            }

        } else {
            unreachable!()
        }
    }
}

#[cfg(test)]
mod tests {
    use interval::{Interval, ops::Range};

    use crate::{lang::{parser::ProgramParser, index_elim2::IndexElimination2, source::SourceProgram, BaseOffsetMap, index_elim::Transform}, scheduling::ScheduleDim};
    use super::*;

    fn test_materializer(program: TransformedProgram, schedule: Schedule) -> ParamCircuitProgram {
        assert!(schedule.is_schedule_valid(&program.expr));

        let materializer =
            Materializer::new(vec![
                Box::new(DefaultArrayMaterializer::new())
            ]);

        let res_mat = materializer.materialize(&program, &schedule);
        assert!(res_mat.is_ok());

        let param_circ = res_mat.unwrap();
        println!("{}", param_circ.schedule);
        println!("{}", param_circ.expr);

        param_circ
    }

    // generate an initial schedule for a program
    fn test_materializer_from_src(src: &str) {
        let parser = ProgramParser::new();
        let program: SourceProgram = parser.parse(src).unwrap();

        let mut index_elim = IndexElimination2::new();
        let res = index_elim.run(&program);
        
        assert!(res.is_ok());

        let program = res.unwrap();
        let init_schedule = Schedule::gen_initial_schedule(&program);
        test_materializer(program, init_schedule);
    }

    fn test_array_materializer(
        mut amat: Box<dyn ArrayMaterializer>,
        shape: Shape,
        schedule: ArraySchedule,
        transform: ArrayTransform, 
    ) -> (CircuitRegistry, ParamCircuitExpr) {
        let mut registry = CircuitRegistry::new();
        let circ = amat.materialize(&shape, &schedule, &transform, &mut registry);

        println!("{}", circ);
        for ct_var in circ.ciphertext_vars() {
            println!("{} =>\n{}", ct_var, registry.get_ct_var_value(&ct_var));
        }
        for pt_var in circ.plaintext_vars() {
            println!("{} =>\n{}", pt_var, registry.get_pt_var_value(&pt_var));
        }

        (registry, circ)
    }

    #[test]
    fn test_imgblur() {
        test_materializer_from_src(
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
        test_materializer_from_src(
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
        test_materializer_from_src(
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
        test_materializer_from_src(
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
        test_materializer_from_src(
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
        test_materializer_from_src(
        "
            input A: [(0,3)]
            input B: [(0,3)]
            sum(A * B)
            "
        );
    }

    #[test]
    fn test_matvecmul() {
        test_materializer_from_src(
        "
            input M: [(0,2),(0,2)]
            input v: [(0,2)]
            for i: (0,2) {
                sum(M[i] * v)
            }
            "
        );
    }

    #[test]
    fn test_materialize_img_array() {
        let shape: Shape = im::vector![Interval::new(0, 16), Interval::new(0, 16)];

        let base =
            ArrayTransform {
                array: String::from("img"),
                offset_map: BaseOffsetMap::new(2),
                dims: im::vector![
                    DimContent::FilledDim { dim: 0, extent: 3, stride: 1 },
                    DimContent::FilledDim { dim: 1, extent: 3, stride: 1 },
                    DimContent::FilledDim { dim: 0, extent: 16, stride: 1 },
                    DimContent::FilledDim { dim: 1, extent: 16, stride: 1 },
                ]
            };

        let schedule = 
            ArraySchedule {
                preprocessing: None,
                exploded_dims: im::vector![
                    ScheduleDim { index: 0, stride: 1, extent: 3, name: String::from("i"), pad_left: 0, pad_right: 0 },
                    ScheduleDim { index: 1, stride: 1, extent: 3, name: String::from("j"), pad_left: 0, pad_right: 0 }
                ],
                vectorized_dims: im::vector![
                    ScheduleDim { index: 2, stride: 1, extent: 16, name: String::from("x"), pad_left: 0, pad_right: 0 },
                    ScheduleDim { index: 3, stride: 1, extent: 16, name: String::from("y"), pad_left: 0, pad_right: 0 }
                ]
            };

        let (mut registry, circ) =
            test_array_materializer(
                Box::new(DefaultArrayMaterializer::new()),
                shape, 
                schedule, 
                base, 
            );

        let ct_var = circ.ciphertext_vars().iter().next().unwrap().clone();
        if let CircuitVarValue::CoordMap(coord_map) = registry.get_ct_var_value(&ct_var) {
            // ct_var should be mapped to the same vector at all coords
            assert!(coord_map.multiplicity() == 9);
            let values: Vec<&CiphertextObject> =
                coord_map.value_iter()
                .map(|(_, value)| value)
                .collect();

            let first = *values.first().unwrap();
            assert!(
                values.iter().all(|x| **x == *first)
            )

        } else {
            assert!(false)
        }
    }

    #[test]
    fn test_materialize_diagonal() {
        let shape: Shape = im::vector![Interval::new(0, 4)];

        let transform =
            ArrayTransform {
                array: String::from("v"),
                offset_map: BaseOffsetMap::new(2),
                dims: im::vector![
                    DimContent::FilledDim { dim: 0, extent: 4, stride: 1 },
                    DimContent::EmptyDim { extent: 4 },
                ]
            };

        let schedule = 
            ArraySchedule {
                preprocessing: Some(ClientPreprocessing::Permute(0, 1)),
                exploded_dims: im::vector![
                    ScheduleDim { index: 0, stride: 1, extent: 4, name: String::from("x"), pad_left: 0, pad_right: 0 },
                ],
                vectorized_dims: im::vector![
                    ScheduleDim { index: 1, stride: 1, extent: 4, name: String::from("y"), pad_left: 0, pad_right: 0 },
                ]
            };

        test_array_materializer(
            Box::new(DiagonalArrayMaterializer::new()),
            shape, 
            schedule, 
            transform, 
        );
    }

    #[test]
    fn test_materialize_diagonal2() {
        let shape: Shape = im::vector![Interval::new(0, 4), Interval::new(0, 4)];

        let transform =
            ArrayTransform {
                array: String::from("img"),
                offset_map: BaseOffsetMap::new(2),
                dims: im::vector![
                    DimContent::FilledDim { dim: 0, extent: 4, stride: 1 },
                    DimContent::FilledDim { dim: 1, extent: 4, stride: 1 },
                ]
            };

        let schedule = 
            ArraySchedule {
                preprocessing: Some(ClientPreprocessing::Permute(0, 1)),
                exploded_dims: im::vector![
                    ScheduleDim { index: 0, stride: 1, extent: 4, name: String::from("x"), pad_left: 0, pad_right: 0 },
                ],
                vectorized_dims: im::vector![
                    ScheduleDim { index: 1, stride: 1, extent: 4, name: String::from("y"), pad_left: 0, pad_right: 0 },
                ]
            };

        test_array_materializer(
            Box::new(DiagonalArrayMaterializer::new()),
            shape, 
            schedule, 
            transform, 
        );
    }

    #[test]
    fn test_vectorized_reduce() {
        let program =
            TransformedProgram {
                expr:
                    TransformedExpr::ReduceNode(
                        1,
                        Operator::Add,
                        Box::new(TransformedExpr::ExprRef(1))
                    ),

                inputs: HashMap::from([
                    (1,
                        ArrayTransform {
                            array: String::from("a"),
                            offset_map: BaseOffsetMap::new(2),
                            dims: im::vector![
                                DimContent::FilledDim { dim: 0, extent: 4, stride: 1 },
                                DimContent::FilledDim { dim: 1, extent: 4, stride: 1 },
                            ]
                        }
                    )
                ]),

                array_shapes: HashMap::from([
                    (String::from("a"), im::vector![Interval::new(0, 4), Interval::new(0, 4)])
                ])
            };

        let schedule =
            Schedule {
                schedule_map: im::HashMap::from(vec![
                    (1, ArraySchedule {
                        preprocessing: None,
                        exploded_dims: im::vector![],
                        vectorized_dims: im::vector![
                            ScheduleDim { index: 0, stride: 1, extent: 4, name: String::from("i"), pad_left: 0, pad_right: 0 },
                            ScheduleDim { index: 1, stride: 1, extent: 4, name: String::from("j"), pad_left: 0, pad_right: 0 },
                        ],
                    })
                ])
            };

        test_materializer(program, schedule);
    }
}