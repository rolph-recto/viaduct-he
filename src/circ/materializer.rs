use std::{collections::{HashMap, HashSet}};

use crate::{
    circ::{vector_deriver::VectorDeriver, vector_info::VectorInfo, *},
    lang::index_elim::{InlinedExpr, InlinedProgram},
    scheduling::*,
    util,
};

pub trait InputArrayMaterializer<'a> {
    fn create(&self) -> Box<dyn InputArrayMaterializer + 'a>;

    fn can_materialize(
        &self,
        array_type: ArrayType,
        array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
    ) -> bool;

    fn materialize(
        &mut self,
        array_type: ArrayType,
        array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
        registry: &mut CircuitObjectRegistry,
    ) -> CircuitId;
}

pub trait MaterializerFactory {
    fn create<'a>(&self) -> Materializer<'a>;
}

pub struct DefaultMaterializerFactory;

impl MaterializerFactory for DefaultMaterializerFactory {
    fn create<'a>(&self) -> Materializer<'a> {
        let amats: Vec<Box<dyn InputArrayMaterializer + 'a>> =
            vec![
                Box::new(DefaultArrayMaterializer::new())
            ];

        Materializer::new(amats)
    }
}

/// materializes a schedule for an index-free program.
pub struct Materializer<'a> {
    array_materializers: Vec<Box<dyn InputArrayMaterializer<'a> + 'a>>,
    registry: CircuitObjectRegistry,
    expr_circuit_map: HashMap<ArrayName, CircuitId>,
    expr_schedule_map: HashMap<ArrayName, ExprScheduleType>,
    expr_array_type_map: HashMap<ArrayName, ArrayType>,
}

impl<'a> Materializer<'a> {
    pub fn new(array_materializers: Vec<Box<dyn InputArrayMaterializer<'a> + 'a>>) -> Self {
        Materializer {
            array_materializers,
            registry: CircuitObjectRegistry::new(),
            expr_circuit_map: HashMap::new(),
            expr_schedule_map: HashMap::new(),
            expr_array_type_map: HashMap::new(),
        }
    }

    /// packages the materialized expr with the vector registry
    pub fn run(
        mut self,
        program: &InlinedProgram,
        schedule: &Schedule
    ) -> Result<ParamCircuitProgram, String> {
        let mut circuit_list: Vec<(String, Vec<(DimName, Extent)>, CircuitId)> = vec![];

        // need to clone expr_map here because the iteration through it is mutating
        let expr_list: Vec<(ArrayName, InlinedExpr)> = 
            program
            .expr_map
            .iter()
            .map(|(array, expr)| (array.clone(), expr.clone()))
            .collect();

        expr_list
            .into_iter()
            .try_for_each(|(array, expr)| -> Result<(), String> {
                let (expr_schedule, array_type, circuit_id) =
                    self.materialize_expr(program, &expr, schedule)?;

                let dims = match &expr_schedule {
                    ExprScheduleType::Any => vec![],

                    ExprScheduleType::Specific(spec_sched) => spec_sched
                        .exploded_dims
                        .iter()
                        .map(|dim| (dim.name.clone(), dim.extent))
                        .collect(),
                };

                self.expr_circuit_map.insert(array.clone(), circuit_id);
                self.expr_schedule_map.insert(array.clone(), expr_schedule);
                self.expr_array_type_map.insert(array.clone(), array_type);
                circuit_list.push((array.clone(), dims, circuit_id));
                Ok(())
            })?;

        Ok(ParamCircuitProgram {
            registry: self.registry,
            native_expr_list: vec![],
            circuit_expr_list: circuit_list,
        })
    }

    fn materialize_expr_indexing_site<'b, T: CircuitObject>(
        &mut self,
        indexing_id: &IndexingId,
        array_type: ArrayType,
        schedule: &IndexingSiteSchedule,
        ref_expr_sched: &ExprSchedule,
        expr_circ_val: CircuitValue<VectorInfo>,
        transform_circ_val: CircuitValue<VectorInfo>,
    ) -> Result<(ExprScheduleType, ArrayType, CircuitId), String>
    where
        CircuitObjectRegistry: CanRegisterObject<'b, T>,
        ParamCircuitExpr: CanCreateObjectVar<T>,
    {
        let derivation_opt =
            VectorDeriver::derive_from_source::<T>(&expr_circ_val, &transform_circ_val);

        match derivation_opt {
            Some((obj_val, step_val, mask_val)) => {
                let circuit_id = VectorDeriver::gen_circuit_expr(
                    obj_val,
                    step_val,
                    mask_val,
                    &mut self.registry,
                );

                let expr_schedule = schedule.to_expr_schedule(ref_expr_sched.shape.clone());

                Ok((
                    ExprScheduleType::Specific(expr_schedule),
                    array_type,
                    circuit_id,
                ))
            }

            None => Err(format!("cannot derive transform at {}", indexing_id)),
        }
    }

    fn materialize_expr(
        &mut self,
        program: &InlinedProgram,
        expr: &InlinedExpr,
        schedule: &Schedule,
    ) -> Result<(ExprScheduleType, ArrayType, CircuitId), String> {
        match expr {
            InlinedExpr::Literal(lit) => {
                let sched_lit = Schedule::schedule_literal()?;
                let circuit_id = self
                    .registry
                    .register_circuit(ParamCircuitExpr::Literal(*lit));
                Ok((sched_lit, ArrayType::Plaintext, circuit_id))
            }

            InlinedExpr::Op(op, expr1, expr2) => {
                let (sched1, type1, id1) =
                    self.materialize_expr(program, expr1, schedule)?;

                let (sched2, type2, id2) =
                    self.materialize_expr(program, expr2, schedule)?;

                let schedule = Schedule::schedule_op(&sched1, &sched2)?;

                let expr = ParamCircuitExpr::Op(op.clone(), id1, id2);
                let id = self.registry.register_circuit(expr);

                Ok((schedule, type1.join(&type2), id))
            }

            // TODO support client transforms
            InlinedExpr::ReduceNode(reduced_index, op, body) => {
                let (body_sched, body_type, mat_body) =
                    self.materialize_expr(program, body, schedule)?;

                let schedule = Schedule::schedule_reduce(*reduced_index, &body_sched)?;

                if let ExprScheduleType::Specific(body_sched_spec) = body_sched {
                    let mut reduced_index_vars: HashSet<(DimName, usize)> = HashSet::new();
                    for dim in body_sched_spec.exploded_dims.iter() {
                        if dim.index == *reduced_index {
                            // dim is reduced, remove it
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
                                        .iter()
                                        .map(|x| x * block_size),
                                );
                            }
                        }
                    }

                    let expr_vec =
                        reduced_index_vars
                            .into_iter()
                            .fold(mat_body, |acc, (var, extent)| {
                                let reduce_id = self.registry.register_circuit(
                                    ParamCircuitExpr::ReduceDim(var, extent, op.clone(), acc),
                                );

                                reduce_id
                            });

                    let expr = reduction_list.into_iter().fold(expr_vec, |acc, n| {
                        let rot_id = self.registry.register_circuit(ParamCircuitExpr::Rotate(
                            OffsetExpr::Literal(-(n as isize)),
                            acc,
                        ));

                        let op_id = self.registry.register_circuit(ParamCircuitExpr::Op(
                            Operator::Add,
                            acc,
                            rot_id,
                        ));

                        op_id
                    });

                    Ok((schedule, body_type, expr))
                } else {
                    unreachable!()
                }
            }

            InlinedExpr::ExprRef(indexing_id, transform) => {
                let schedule = &schedule.schedule_map[indexing_id];
                if program.is_expr(&transform.array) {
                    // indexing an expression
                    let ref_schedule_type = self
                        .expr_schedule_map
                        .get(&transform.array)
                        .unwrap()
                        .clone();
                    let array_type = *self.expr_array_type_map.get(&transform.array).unwrap();
                    match ref_schedule_type {
                        ExprScheduleType::Any => {
                            panic!("no support for top-level literal exprs yet")
                        }

                        ExprScheduleType::Specific(ref_expr_sched) => {
                            // TODO: refactor this so we don't inline derivation logic here
                            let coord_system =
                                IndexCoordinateSystem::new(schedule.exploded_dims.iter());
                            let expr_circ_val = ref_expr_sched.materialize(&transform.array);
                            let transform_circ_val = VectorInfo::get_input_vector_value(
                                coord_system,
                                &ref_expr_sched.shape,
                                schedule,
                                transform,
                                schedule.preprocessing,
                            );

                            match array_type {
                                ArrayType::Ciphertext => self
                                    .materialize_expr_indexing_site::<CiphertextObject>(
                                        indexing_id,
                                        array_type,
                                        schedule,
                                        &ref_expr_sched,
                                        expr_circ_val,
                                        transform_circ_val,
                                    ),

                                ArrayType::Plaintext => self
                                    .materialize_expr_indexing_site::<PlaintextObject>(
                                        indexing_id,
                                        array_type,
                                        schedule,
                                        &ref_expr_sched,
                                        expr_circ_val,
                                        transform_circ_val,
                                    ),
                            }
                        }
                    }
                } else {
                    // indexing an input array
                    let (array_shape, array_type) =
                        &program.input_map[&transform.array];

                    for amat in self.array_materializers.iter_mut() {
                        if amat.can_materialize(*array_type, array_shape, schedule, transform) {
                            let expr_id = amat.materialize(
                                *array_type,
                                array_shape,
                                schedule,
                                transform,
                                &mut self.registry,
                            );

                            let shape = transform.as_shape();
                            let expr_schedule =
                                ExprScheduleType::Specific(schedule.to_expr_schedule(shape));

                            return Ok((expr_schedule, *array_type, expr_id));
                        }
                    }

                    Err(format!(
                        "No array materializer can process expr ref {}",
                        indexing_id
                    ))
                }
            }
        }
    }
}

// array materializer that doesn't attempt to derive vectors
pub struct DummyArrayMaterializer;

impl<'a> InputArrayMaterializer<'a> for DummyArrayMaterializer {
    fn create(&self) -> Box<dyn InputArrayMaterializer + 'a> {
        Box::new(Self {})
    }

    // the dummy materializer can only materialize arrays w/o client preprocessing
    fn can_materialize(
        &self,
        _array_type: ArrayType,
        _array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        _transform: &ArrayTransform,
    ) -> bool {
        schedule.preprocessing.is_none()
    }

    fn materialize(
        &mut self,
        _array_type: ArrayType,
        shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
        registry: &mut CircuitObjectRegistry,
    ) -> CircuitId {
        let ct_var = registry.fresh_ct_var();

        // register vectors
        let circuit_val = VectorInfo::get_input_vector_value(
            IndexCoordinateSystem::new(schedule.exploded_dims.iter()),
            shape,
            schedule,
            transform,
            schedule.preprocessing,
        )
        .map(|_, vector| CiphertextObject::InputVector(vector.clone()));

        registry.set_ct_var_value(ct_var.clone(), circuit_val);
        registry.register_circuit(ParamCircuitExpr::CiphertextVar(ct_var))
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

impl<'a> InputArrayMaterializer<'a> for DefaultArrayMaterializer {
    fn create(&self) -> Box<dyn InputArrayMaterializer + 'a> {
        Box::new(DefaultArrayMaterializer::new())
    }

    /// the default materializer can only apply when there is no client preprocessing
    fn can_materialize(
        &self,
        _array_type: ArrayType,
        _array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        _transform: &ArrayTransform,
    ) -> bool {
        schedule.preprocessing.is_none()
    }

    fn materialize(
        &mut self,
        array_type: ArrayType,
        shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
        registry: &mut CircuitObjectRegistry,
    ) -> CircuitId {
        match array_type {
            ArrayType::Ciphertext => self
                .deriver
                .derive_vectors_and_gen_circuit_expr::<CiphertextObject>(
                    shape, schedule, transform, None, registry,
                ),

            ArrayType::Plaintext => self
                .deriver
                .derive_vectors_and_gen_circuit_expr::<PlaintextObject>(
                    shape, schedule, transform, None, registry,
                ),
        }
    }
}

pub struct DiagonalArrayMaterializer {
    deriver: VectorDeriver,
}

impl DiagonalArrayMaterializer {
    pub fn new() -> Self {
        DiagonalArrayMaterializer {
            deriver: VectorDeriver::new(),
        }
    }

    // if dim j is an empty dim, then we can apply the "diagonal"
    // trick from Halevi and Schoup for matrix-vector multiplication
    // to do this, follow these steps:
    // 1. switch innermost tiles of dim i and dim j
    //    (assuming all tiles of i is exploded and only innermost tile of j is vectorized)
    // 2. derive vectors assuming j = 0
    // 3. to fill in the rest of the vectors along dim j by rotating
    //    the vectors at dim j = 0
    fn diagonal_materialize<'a, T: CircuitObject + Clone>(
        &mut self,
        dim_i: usize,
        dim_j: usize,
        shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
        registry: &mut CircuitObjectRegistry,
    ) -> CircuitId
    where
        CircuitObjectRegistry: CanRegisterObject<'a, T>,
        ParamCircuitExpr: CanCreateObjectVar<T>,
    {
        let mut obj_map: IndexCoordinateMap<T> =
            IndexCoordinateMap::new(schedule.exploded_dims.iter());
        let mut mask_map: IndexCoordinateMap<PlaintextObject> =
            IndexCoordinateMap::new(schedule.exploded_dims.iter());
        let mut step_map: IndexCoordinateMap<isize> =
            IndexCoordinateMap::new(schedule.exploded_dims.iter());
        let mut new_schedule = schedule.clone();

        // switch innermost tiles of i and j in the schedule
        let inner_i_dim = new_schedule
            .exploded_dims
            .iter_mut()
            .find(|dim| dim.index == dim_i && dim.stride == 1)
            .unwrap();
        let inner_i_dim_name = inner_i_dim.name.clone();
        let inner_i_dim_extent = inner_i_dim.extent;
        inner_i_dim.index = dim_j;

        let inner_j_dim = new_schedule.vectorized_dims.get_mut(0).unwrap();
        inner_j_dim.index = dim_i;

        let zero_inner_j_coords = obj_map.coord_iter_subset(&inner_i_dim_name, 0..1);

        self.deriver.register_and_derive_vectors::<T>(
            shape,
            &new_schedule,
            transform,
            None,
            zero_inner_j_coords.clone(),
            &mut obj_map,
            &mut mask_map,
            &mut step_map,
        );

        let mut processed_index_vars = obj_map.index_vars();

        // remember, inner i and inner j are swapped,
        // so inner j now has the name of inner i!
        let inner_j_name_index = processed_index_vars
            .iter()
            .position(|name| *name == inner_i_dim_name)
            .unwrap();

        processed_index_vars.remove(inner_j_name_index);

        let obj_var = registry.fresh_obj_var();

        // given expr e is at coord where inner_j=0,
        // expr rot(inner_j, e) is at coord where inner_j != 0
        let rest_inner_j_coords =
            obj_map.coord_iter_subset(&inner_i_dim_name, 1..inner_i_dim_extent);

        for coord in rest_inner_j_coords {
            let mut ref_coord = coord.clone();
            ref_coord[inner_j_name_index] = 0;

            let ref_obj: T = obj_map.get(&ref_coord).unwrap().clone();
            let ref_step = *step_map.get(&ref_coord).unwrap();
            let inner_j_value = coord[inner_j_name_index];

            obj_map.set(coord.clone(), ref_obj);
            step_map.set(coord, ref_step + (inner_j_value as isize));
        }

        // attempt to compute offset expr
        let offset_expr_opt = if processed_index_vars.len() > 0 {
            VectorDeriver::compute_linear_offset(
                &step_map,
                Box::new(zero_inner_j_coords),
                processed_index_vars,
            )
        } else {
            let zero_j_coord = im::vector![0];
            let step = *step_map.get(&zero_j_coord).unwrap();
            Some(OffsetExpr::Literal(step))
        };

        registry.set_obj_var_value(obj_var.clone(), CircuitValue::CoordMap(obj_map));

        let obj_var_id = registry.register_circuit(ParamCircuitExpr::obj_var(obj_var));

        let output_expr = if let Some(offset_expr) = offset_expr_opt {
            let new_offset_expr = OffsetExpr::Add(
                Box::new(offset_expr),
                Box::new(OffsetExpr::Var(inner_i_dim_name.clone())),
            );

            ParamCircuitExpr::Rotate(new_offset_expr, obj_var_id)
        } else {
            let offset_var = registry.fresh_offset_fvar();
            registry.set_offset_var_value(offset_var.clone(), CircuitValue::CoordMap(step_map));

            ParamCircuitExpr::Rotate(OffsetExpr::Var(offset_var), obj_var_id)
        };

        registry.register_circuit(output_expr)
    }
}

impl<'a> InputArrayMaterializer<'a> for DiagonalArrayMaterializer {
    fn create(&self) -> Box<dyn InputArrayMaterializer + 'a> {
        Box::new(DiagonalArrayMaterializer::new())
    }

    fn can_materialize(
        &self,
        _array_type: ArrayType,
        _array_shape: &Shape,
        schedule: &IndexingSiteSchedule,
        _base: &ArrayTransform,
    ) -> bool {
        if let Some(ArrayPreprocessing::Permute(dim_i, dim_j)) = schedule.preprocessing {
            // dim i must be exploded and dim j must be the outermost vectorized dim
            let i_exploded = schedule
                .exploded_dims
                .iter()
                .any(|edim| edim.index == dim_i);

            let j_outermost_vectorized = schedule.vectorized_dims.len() > 0
                && schedule.vectorized_dims.head().unwrap().index == dim_j;

            // dim i and j must have both have the same tiling that corresponds
            // to the permutation transform
            // TODO: for now, assume i and j are NOT tiled
            let tiling_i = schedule.get_dim_tiling(dim_i);
            let tiling_j = schedule.get_dim_tiling(dim_j);

            // dim i and j cannot have any padding
            let no_padding = schedule.vectorized_dims.iter().all(|dim| {
                (dim.index == dim_i && dim.pad_left == 0 && dim.pad_right == 0)
                    || (dim.index == dim_j && dim.pad_left == 0 && dim.pad_right == 0)
                    || (dim.index != dim_i || dim.index != dim_j)
            });

            // TODO: for now, assume i and j are NOT tiled
            tiling_i == tiling_j
                && tiling_i.len() == 1
                && i_exploded
                && j_outermost_vectorized
                && no_padding
        } else {
            false
        }
    }

    fn materialize(
        &mut self,
        array_type: ArrayType,
        shape: &Shape,
        schedule: &IndexingSiteSchedule,
        transform: &ArrayTransform,
        registry: &mut CircuitObjectRegistry,
    ) -> CircuitId {
        if let Some(ArrayPreprocessing::Permute(dim_i, dim_j)) = schedule.preprocessing {
            match (&transform.dims[dim_i], &transform.dims[dim_j]) {
                // if dim i is empty, then the permutation is a no-op
                // materialize the schedule normally
                (DimContent::EmptyDim { extent: _ }, _) => match array_type {
                    ArrayType::Ciphertext => self
                        .deriver
                        .derive_vectors_and_gen_circuit_expr::<CiphertextObject>(
                            shape, schedule, transform, None, registry,
                        ),

                    ArrayType::Plaintext => self
                        .deriver
                        .derive_vectors_and_gen_circuit_expr::<PlaintextObject>(
                            shape, schedule, transform, None, registry,
                        ),
                },

                // if dim j is a filled dim, then the permutation must actually
                // be done by the client; record this fact and then materialize
                // the schedule normally
                (
                    DimContent::FilledDim {
                        dim: idim_i,
                        extent: _,
                        stride: _,
                    },
                    DimContent::FilledDim {
                        dim: idim_j,
                        extent: _,
                        stride: _,
                    },
                ) => match array_type {
                    ArrayType::Ciphertext => self
                        .deriver
                        .derive_vectors_and_gen_circuit_expr::<CiphertextObject>(
                            shape,
                            schedule,
                            transform,
                            Some(ArrayPreprocessing::Permute(*idim_i, *idim_j)),
                            registry,
                        ),

                    ArrayType::Plaintext => self
                        .deriver
                        .derive_vectors_and_gen_circuit_expr::<PlaintextObject>(
                            shape,
                            schedule,
                            transform,
                            Some(ArrayPreprocessing::Permute(*idim_i, *idim_j)),
                            registry,
                        ),
                },

                (
                    DimContent::FilledDim {
                        dim: _,
                        extent: _,
                        stride: _,
                    },
                    DimContent::EmptyDim { extent: _ },
                ) => match array_type {
                    ArrayType::Ciphertext => self.diagonal_materialize::<CiphertextObject>(
                        dim_i, dim_j, shape, schedule, transform, registry,
                    ),

                    ArrayType::Plaintext => self.diagonal_materialize::<PlaintextObject>(
                        dim_i, dim_j, shape, schedule, transform, registry,
                    ),
                },
            }
        } else {
            unreachable!()
        }
    }
}

#[cfg(test)]
mod tests {
    use indexmap::IndexMap;

    use super::*;
    use crate::{
        lang::{
            elaborated::Elaborator, index_elim::IndexElimination, parser::ProgramParser,
            source::SourceProgram, ArrayType, BaseOffsetMap, OUTPUT_EXPR_NAME,
        },
        scheduling::ScheduleDim,
    };

    fn test_materializer(program: InlinedProgram, schedule: Schedule) -> ParamCircuitProgram {
        assert!(schedule.is_schedule_valid(&program));

        let amats: Vec<Box<dyn InputArrayMaterializer>> = vec![Box::new(DefaultArrayMaterializer::new())];
        let materializer = Materializer::new(amats);

        let res_mat = materializer.run(&program, &schedule);
        assert!(res_mat.is_ok());

        let param_circ = res_mat.unwrap();
        println!("{}", param_circ);

        param_circ
    }

    // generate an initial schedule for a program
    fn test_materializer_from_src(src: &str) {
        let parser = ProgramParser::new();
        let program: SourceProgram = parser.parse(src).unwrap();

        let elaborated = Elaborator::new().run(program);
        let inline_set = elaborated.default_inline_set();
        let array_group_map = elaborated.default_array_group_map();

        let res =   
            IndexElimination::new()
            .run(&inline_set, &array_group_map, &elaborated);

        assert!(res.is_ok());

        let tprogram = res.unwrap();
        let init_schedule = Schedule::gen_initial_schedule(&tprogram);
        test_materializer(tprogram, init_schedule);
    }

    fn test_array_materializer(
        mut amat: Box<dyn InputArrayMaterializer>,
        shape: Shape,
        schedule: IndexingSiteSchedule,
        transform: ArrayTransform,
    ) -> (CircuitObjectRegistry, CircuitId) {
        let mut registry = CircuitObjectRegistry::new();
        let circ = amat.materialize(
            ArrayType::Ciphertext,
            &shape,
            &schedule,
            &transform,
            &mut registry,
        );

        println!("{}", circ);
        for ct_var in registry.circuit_ciphertext_vars(circ) {
            println!("{} =>\n{}", ct_var, registry.get_ct_var_value(&ct_var));
        }
        for pt_var in registry.circuit_plaintext_vars(circ) {
            println!("{} =>\n{}", pt_var, registry.get_pt_var_value(&pt_var));
        }

        (registry, circ)
    }

    #[test]
    fn test_imgblur() {
        test_materializer_from_src(
            "input img: [16,16] from client
            for x: 16 {
                for y: 16 {
                    img[x-1][y-1] + img[x+1][y+1]
                }
            }",
        );
    }

    #[test]
    fn test_imgblur2() {
        test_materializer_from_src(
            "input img: [16,16] from client
            let res = 
                for x: 16 {
                    for y: 16 {
                        img[x-1][y-1] + img[x+1][y+1]
                    }
                }
            in
            for x: 16 {
                for y: 16 {
                    res[x-2][y-2] + res[x+2][y+2]
                }
            }
            ",
        );
    }

    #[test]
    fn test_convolve() {
        test_materializer_from_src(
            "input img: [16,16] from client
            let conv1 = 
                for x: 15 {
                    for y: 15 {
                        img[x][y] + img[x+1][y+1]
                    }
                }
            in
            for x: 14 {
                for y: 14 {
                    conv1[x][y] + conv1[x+1][y+1]
                }
            }
            ",
        );
    }

    #[test]
    fn test_matmatmul() {
        test_materializer_from_src(
            "input A: [4,4] from client
            input B: [4,4] from client
            for i: 4 {
                for j: 4 {
                    sum(for k: 4 { A[i][k] * B[k][j] })
                }
            }",
        );
    }

    #[test]
    fn test_matmatmul2() {
        test_materializer_from_src(
            "input A1: [4,4] from client
            input A2: [4,4] from client
            input B: [4,4] from client
            let res =
                for i: 4 {
                    for j: 4 {
                        sum(for k: 4 { A1[i][k] * B[k][j] })
                    }
                }
            in
            for i: 4 {
                for j: 4 {
                    sum(for k: 4 { A2[i][k] * res[k][j] })
                }
            }
            ",
        );
    }

    #[test]
    fn test_dotprod_pointless() {
        test_materializer_from_src(
            "
            input A: [3] from client
            input B: [3] from client
            sum(A * B)
            ",
        );
    }

    #[test]
    fn test_matvecmul() {
        test_materializer_from_src(
            "
            input M: [2,2] from client
            input v: [2] from client
            for i: 2 {
                sum(M[i] * v)
            }
            ",
        );
    }

    // convolution with masking for out-of-bounds accessesyy
    #[test]
    fn test_materialize_img_array() {
        let shape: Shape = im::vector![16, 16];

        let base = ArrayTransform {
            array: String::from("img"),
            offset_map: BaseOffsetMap::new(2),
            dims: im::vector![
                DimContent::FilledDim {
                    dim: 0,
                    extent: 3,
                    stride: 1
                },
                DimContent::FilledDim {
                    dim: 1,
                    extent: 3,
                    stride: 1
                },
                DimContent::FilledDim {
                    dim: 0,
                    extent: 16,
                    stride: 1
                },
                DimContent::FilledDim {
                    dim: 1,
                    extent: 16,
                    stride: 1
                },
            ],
        };

        let schedule = IndexingSiteSchedule {
            preprocessing: None,
            exploded_dims: im::vector![
                ScheduleDim {
                    index: 0,
                    stride: 1,
                    extent: 3,
                    name: String::from("i"),
                    pad_left: 0,
                    pad_right: 0
                },
                ScheduleDim {
                    index: 1,
                    stride: 1,
                    extent: 3,
                    name: String::from("j"),
                    pad_left: 0,
                    pad_right: 0
                }
            ],
            vectorized_dims: im::vector![
                ScheduleDim {
                    index: 2,
                    stride: 1,
                    extent: 16,
                    name: String::from("x"),
                    pad_left: 0,
                    pad_right: 0
                },
                ScheduleDim {
                    index: 3,
                    stride: 1,
                    extent: 16,
                    name: String::from("y"),
                    pad_left: 0,
                    pad_right: 0
                }
            ],
        };

        let (registry, circ) = test_array_materializer(
            Box::new(DefaultArrayMaterializer::new()),
            shape,
            schedule,
            base,
        );

        let ct_var = registry
            .circuit_ciphertext_vars(circ)
            .iter()
            .next()
            .unwrap()
            .clone();

        if let CircuitValue::CoordMap(coord_map) = registry.get_ct_var_value(&ct_var) {
            // ct_var should be mapped to the same vector at all coords
            assert!(coord_map.multiplicity() == 9);
            let values: Vec<&CiphertextObject> = coord_map
                .value_iter()
                .map(|(_, value)| value.unwrap())
                .collect();

            let first = *values.first().unwrap();
            assert!(values.iter().all(|x| **x == *first))
        } else {
            assert!(false)
        }
    }

    // convolution with padding for out-of-bounds accesses
    #[test]
    fn test_materialize_img_array_padding() {
        let shape: Shape = im::vector![16, 16];

        let base = ArrayTransform {
            array: String::from("img"),
            offset_map: BaseOffsetMap::new(2),
            dims: im::vector![
                DimContent::FilledDim {
                    dim: 0,
                    extent: 3,
                    stride: 1
                },
                DimContent::FilledDim {
                    dim: 1,
                    extent: 3,
                    stride: 1
                },
                DimContent::FilledDim {
                    dim: 0,
                    extent: 16,
                    stride: 1
                },
                DimContent::FilledDim {
                    dim: 1,
                    extent: 16,
                    stride: 1
                },
            ],
        };

        let schedule = IndexingSiteSchedule {
            preprocessing: None,
            exploded_dims: im::vector![
                ScheduleDim {
                    index: 0,
                    stride: 1,
                    extent: 3,
                    name: String::from("i"),
                    pad_left: 0,
                    pad_right: 0
                },
                ScheduleDim {
                    index: 1,
                    stride: 1,
                    extent: 3,
                    name: String::from("j"),
                    pad_left: 0,
                    pad_right: 0
                }
            ],
            vectorized_dims: im::vector![
                ScheduleDim {
                    index: 2,
                    stride: 1,
                    extent: 16,
                    name: String::from("x"),
                    pad_left: 3,
                    pad_right: 3
                },
                ScheduleDim {
                    index: 3,
                    stride: 1,
                    extent: 16,
                    name: String::from("y"),
                    pad_left: 3,
                    pad_right: 3
                }
            ],
        };

        let (registry, circ) = test_array_materializer(
            Box::new(DefaultArrayMaterializer::new()),
            shape,
            schedule,
            base,
        );

        let ct_var = registry
            .circuit_ciphertext_vars(circ)
            .iter()
            .next()
            .unwrap()
            .clone();

        if let CircuitValue::CoordMap(coord_map) = registry.get_ct_var_value(&ct_var) {
            // ct_var should be mapped to the same vector at all coords
            assert!(coord_map.multiplicity() == 9);
            let values: Vec<&CiphertextObject> = coord_map
                .value_iter()
                .map(|(_, value)| value.unwrap())
                .collect();

            let first = *values.first().unwrap();
            assert!(values.iter().all(|x| **x == *first))
        } else {
            assert!(false)
        }
    }

    #[test]
    fn test_materialize_diagonal() {
        let shape: Shape = im::vector![4];

        let transform = ArrayTransform {
            array: String::from("v"),
            offset_map: BaseOffsetMap::new(2),
            dims: im::vector![
                DimContent::FilledDim {
                    dim: 0,
                    extent: 4,
                    stride: 1
                },
                DimContent::EmptyDim { extent: 4 },
            ],
        };

        let schedule = IndexingSiteSchedule {
            preprocessing: Some(ArrayPreprocessing::Permute(0, 1)),
            exploded_dims: im::vector![ScheduleDim {
                index: 0,
                stride: 1,
                extent: 4,
                name: String::from("x"),
                pad_left: 0,
                pad_right: 0
            },],
            vectorized_dims: im::vector![ScheduleDim {
                index: 1,
                stride: 1,
                extent: 4,
                name: String::from("y"),
                pad_left: 0,
                pad_right: 0
            },],
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
        let shape: Shape = im::vector![4, 4,];

        let transform = ArrayTransform {
            array: String::from("img"),
            offset_map: BaseOffsetMap::new(2),
            dims: im::vector![
                DimContent::FilledDim {
                    dim: 0,
                    extent: 4,
                    stride: 1
                },
                DimContent::FilledDim {
                    dim: 1,
                    extent: 4,
                    stride: 1
                },
            ],
        };

        let schedule = IndexingSiteSchedule {
            preprocessing: Some(ArrayPreprocessing::Permute(0, 1)),
            exploded_dims: im::vector![ScheduleDim {
                index: 0,
                stride: 1,
                extent: 4,
                name: String::from("x"),
                pad_left: 0,
                pad_right: 0
            },],
            vectorized_dims: im::vector![ScheduleDim {
                index: 1,
                stride: 1,
                extent: 4,
                name: String::from("y"),
                pad_left: 0,
                pad_right: 0
            },],
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
        let program = InlinedProgram {
            expr_map: IndexMap::from([(
                String::from(OUTPUT_EXPR_NAME),
                InlinedExpr::ReduceNode(
                    1,
                    Operator::Add,
                    Box::new(InlinedExpr::ExprRef(
                        String::from("a1"),
                        ArrayTransform {
                            array: String::from("a"),
                            offset_map: BaseOffsetMap::new(2),
                            dims: im::vector![
                                DimContent::FilledDim {
                                    dim: 0,
                                    extent: 4,
                                    stride: 1
                                },
                                DimContent::FilledDim {
                                    dim: 1,
                                    extent: 4,
                                    stride: 1
                                },
                            ],
                        },
                    )),
                ),
            )]),

            input_map: IndexMap::from([(
                String::from("a"),
                (im::vector![4, 4], ArrayType::Ciphertext),
            )]),
        };

        let schedule = Schedule {
            schedule_map: im::HashMap::from(vec![(
                String::from("a1"),
                IndexingSiteSchedule {
                    preprocessing: None,
                    exploded_dims: im::vector![],
                    vectorized_dims: im::vector![
                        ScheduleDim {
                            index: 0,
                            stride: 1,
                            extent: 4,
                            name: String::from("i"),
                            pad_left: 0,
                            pad_right: 0
                        },
                        ScheduleDim {
                            index: 1,
                            stride: 1,
                            extent: 4,
                            name: String::from("j"),
                            pad_left: 0,
                            pad_right: 0
                        },
                    ],
                },
            )]),
        };

        test_materializer(program, schedule);
    }

    #[test]
    fn test_read() {
        let program = InlinedProgram {
            expr_map: IndexMap::from([
                (
                    String::from("res"),
                    InlinedExpr::Op(
                        Operator::Add,
                        Box::new(InlinedExpr::ExprRef(
                            String::from("a_1"),
                            ArrayTransform {
                                array: String::from("a"),
                                offset_map: BaseOffsetMap::new(2),
                                dims: im::vector![
                                    DimContent::FilledDim {
                                        dim: 0,
                                        extent: 4,
                                        stride: 1
                                    },
                                    DimContent::FilledDim {
                                        dim: 1,
                                        extent: 4,
                                        stride: 1
                                    },
                                ],
                            },
                        )),
                        Box::new(InlinedExpr::Literal(3)),
                    ),
                ),
                (
                    String::from(OUTPUT_EXPR_NAME),
                    InlinedExpr::Op(
                        Operator::Add,
                        Box::new(InlinedExpr::ExprRef(
                            String::from("res_1"),
                            ArrayTransform {
                                array: String::from("res"),
                                offset_map: BaseOffsetMap::new(2),
                                dims: im::vector![
                                    DimContent::FilledDim {
                                        dim: 0,
                                        extent: 4,
                                        stride: 1
                                    },
                                    DimContent::FilledDim {
                                        dim: 1,
                                        extent: 4,
                                        stride: 1
                                    },
                                ],
                            },
                        )),
                        Box::new(InlinedExpr::Literal(2)),
                    ),
                ),
            ]),

            input_map: IndexMap::from([(
                String::from("a"),
                (im::vector![4, 4], ArrayType::Ciphertext),
            )]),
        };

        let schedule = Schedule {
            schedule_map: im::HashMap::from(vec![
                (
                    String::from("a_1"),
                    IndexingSiteSchedule {
                        preprocessing: None,
                        exploded_dims: im::vector![],
                        vectorized_dims: im::vector![
                            ScheduleDim {
                                index: 0,
                                stride: 1,
                                extent: 4,
                                name: String::from("i"),
                                pad_left: 0,
                                pad_right: 0
                            },
                            ScheduleDim {
                                index: 1,
                                stride: 1,
                                extent: 4,
                                name: String::from("j"),
                                pad_left: 0,
                                pad_right: 0
                            },
                        ],
                    },
                ),
                (
                    String::from("res_1"),
                    IndexingSiteSchedule {
                        preprocessing: None,
                        exploded_dims: im::vector![],
                        vectorized_dims: im::vector![
                            ScheduleDim {
                                index: 0,
                                stride: 1,
                                extent: 4,
                                name: String::from("i"),
                                pad_left: 0,
                                pad_right: 0
                            },
                            ScheduleDim {
                                index: 1,
                                stride: 1,
                                extent: 4,
                                name: String::from("j"),
                                pad_left: 0,
                                pad_right: 0
                            },
                        ],
                    },
                ),
            ]),
        };

        test_materializer(program, schedule);
    }
}
