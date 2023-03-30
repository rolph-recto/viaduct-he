use std::{
    collections::{HashMap, HashSet},
    hash::Hash, ops::Index, rc::Rc, cell::RefCell
};

use disjoint_sets::UnionFindNode;
use log::{info, debug};

use crate::{
    circ::{ *,
        array_materializer::*,
        vector_deriver::VectorDeriver,
        vector_info::VectorInfo
    },
    lang::index_elim::{InlinedExpr, InlinedProgram},
    scheduling::*,
    util::{self, NameGenerator},
};

pub trait MaterializerFactory {
    fn create<'a>(&self) -> Materializer<'a>;
}

pub struct DefaultMaterializerFactory;

impl MaterializerFactory for DefaultMaterializerFactory {
    fn create<'a>(&self) -> Materializer<'a> {
        let amats: Vec<Box<dyn InputArrayMaterializer + 'a>> =
            vec![
                Box::new(RollArrayMaterializer::new()),
                Box::new(DefaultArrayMaterializer::new()),
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
    name_generator: NameGenerator,
}

impl<'a> Materializer<'a> {
    pub fn new(array_materializers: Vec<Box<dyn InputArrayMaterializer<'a> + 'a>>) -> Self {
        Materializer {
            array_materializers,
            registry: CircuitObjectRegistry::new(),
            expr_circuit_map: HashMap::new(),
            expr_schedule_map: HashMap::new(),
            expr_array_type_map: HashMap::new(),
            name_generator: NameGenerator::new(),
        }
    }

    /// packages the materialized expr with the vector registry
    pub fn run(
        mut self,
        program: &InlinedProgram,
        schedule: &Schedule
    ) -> Result<ParamCircuitProgram, String> {
        let mut circuit_list: Vec<CircuitDecl> = vec![];

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
                self.register_input_indexing_sites(program, &expr, schedule)?;

                let mut expr_preludes: Vec<CircuitDecl> = Vec::new();
                let (expr_schedule, array_type, circuit_id) =
                    self.materialize_expr(program, &expr, schedule, &mut expr_preludes)?;

                let dims = match &expr_schedule {
                    ExprScheduleType::Any => vec![],

                    ExprScheduleType::Specific(spec_sched) => spec_sched
                        .exploded_dims
                        .iter()
                        .map(|dim| (dim.name.clone(), dim.extent))
                        .collect(),
                };

                // add preludes before actual circuit
                for (array, dims, circuit_id) in expr_preludes {
                    circuit_list.push((array, dims, circuit_id));
                }

                // coalesce cipher/plaintext vars that map to equal values
                // this allows for value numbering optimizations
                self.coalesce_vars(circuit_id);

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

    fn coalesce_vars(&mut self, expr_id: CircuitId) {
        let mut varctx_map: HashMap<im::Vector<DimName>, Vec<CircuitId>> = HashMap::new();
        self.get_vars_with_context(expr_id, &mut im::Vector::new(), &mut varctx_map);

        let var_classes = self.get_var_equiv_classes(varctx_map);

        let mut removed_ct_vars: Vec<VarName> = Vec::new();
        let mut removed_pt_vars: Vec<VarName> = Vec::new();
        self.repoint_vars(var_classes, &mut removed_ct_vars, &mut removed_pt_vars);

        for ct_var in removed_ct_vars {
            self.registry.ct_var_values.remove(&ct_var);
        }

        for pt_var in removed_pt_vars {
            self.registry.pt_var_values.remove(&pt_var);
        }
    }

    fn get_vars_with_context(
        &self,
        expr_id: CircuitId,
        reduced_dims: &mut im::Vector<DimName>,
        varctx_map: &mut HashMap<im::Vector<DimName>, Vec<CircuitId>>
    ) {
        match self.registry.get_circuit(expr_id) {
            ParamCircuitExpr::ReduceDim(dim, _, _, body) => {
                reduced_dims.push_back(dim.clone());
                self.get_vars_with_context(
                    *body,
                    reduced_dims,
                    varctx_map
                )
            },

            ParamCircuitExpr::Op(_, expr1, expr2) => {
                self.get_vars_with_context(*expr1, reduced_dims, varctx_map);
                self.get_vars_with_context(*expr2, reduced_dims, varctx_map);
            },

            ParamCircuitExpr::Literal(_) => {},

            ParamCircuitExpr::Rotate(_, body) => {
                self.get_vars_with_context(*body, reduced_dims, varctx_map)
            },

            ParamCircuitExpr::CiphertextVar(_) |
            ParamCircuitExpr::PlaintextVar(_) => {
                if let Some(vars) = varctx_map.get_mut(&reduced_dims) {
                    vars.push(expr_id);

                } else {
                    varctx_map.insert(reduced_dims.clone(), vec![expr_id]);
                }
            },
        }
    }

    fn get_var_equiv_classes(
        &self,
        varctx_map: HashMap<im::Vector<DimName>, Vec<CircuitId>>,
    ) -> HashMap<CircuitId,CircuitId> {
        let mut uf_nodes: HashMap<CircuitId, Rc<RefCell<UnionFindNode<CircuitId>>>> =
            HashMap::new();

        for (_, var_ids) in varctx_map {
            for id in var_ids.iter() {
                uf_nodes.insert(*id, Rc::new(RefCell::new(UnionFindNode::new(*id))));
            }

            for (i, id1) in var_ids.iter().enumerate() {
                for j in (i+1)..var_ids.len() {
                    let id2 = var_ids[j];
                    if *id1 != id2 {
                        match (self.registry.get_circuit(*id1), self.registry.get_circuit(id2)) {
                            (ParamCircuitExpr::CiphertextVar(var1),
                            ParamCircuitExpr::CiphertextVar(var2)) => {
                                let val1 = self.registry.get_ct_var_value(var1);
                                let val2 = self.registry.get_ct_var_value(var2);
                                if val1 == val2 {
                                    let mut node1 = uf_nodes.get(id1).unwrap().borrow_mut();
                                    let mut node2 = uf_nodes.get(&id2).unwrap().borrow_mut();
                                    node1.union(&mut node2);
                                }
                            }

                            (ParamCircuitExpr::PlaintextVar(var1),
                            ParamCircuitExpr::PlaintextVar(var2)) => {
                                let val1 = self.registry.get_pt_var_value(var1);
                                let val2 = self.registry.get_pt_var_value(var2);
                                if val1 == val2 {
                                    let mut node1 = uf_nodes.get(id1).unwrap().borrow_mut();
                                    let mut node2 = uf_nodes.get(&id2).unwrap().borrow_mut();
                                    node1.union(&mut node2);
                                }
                            }
    
                            _ => {}
                        }
                    }
                }
            }
        }

        uf_nodes.into_iter().map(|(id, node)| {
            (id, node.borrow().clone_data())
        }).collect()
    }

    fn repoint_vars(
        &mut self,
        var_classes: HashMap<CircuitId, CircuitId>,
        removed_ct_vars: &mut Vec<VarName>,
        removed_pt_vars: &mut Vec<VarName>,
    ) {
        for (id, parent) in var_classes {
            if id != parent {
                let circ = self.registry.circuit_map.remove(&id).unwrap();
                let parent_circ = self.registry.circuit_map.get(&parent).unwrap();
                self.registry.circuit_map.insert(id, parent_circ.clone());

                match circ {
                    ParamCircuitExpr::CiphertextVar(var) => {
                        removed_ct_vars.push(var);
                    },

                    ParamCircuitExpr::PlaintextVar(var) => {
                        removed_pt_vars.push(var);
                    }

                    _ => unreachable!()
                }
            }
        }
    }

    fn register_input_indexing_sites(
        &mut self,
        program: &InlinedProgram,
        expr: &InlinedExpr,
        schedule: &Schedule,
    ) -> Result<(), String> {
        match expr {
            InlinedExpr::ReduceNode(_, _, body) => {
                self.register_input_indexing_sites(program, body, schedule)
            },

            InlinedExpr::Op(_, expr1, expr2) => {
                self.register_input_indexing_sites(program, expr1, schedule)?;
                self.register_input_indexing_sites(program, expr2, schedule)
            },

            InlinedExpr::Literal(_) => Ok(()),

            InlinedExpr::ExprRef(indexing_id, transform) => {
                if !program.is_expr(&transform.array) {
                    let schedule =
                        &schedule.schedule_map[indexing_id];

                    let (array_shape, array_type) =
                        &program.input_map[&transform.array];

                    let mut processed = false;
                    for amat in self.array_materializers.iter_mut() {
                        if amat.can_materialize(*array_type, array_shape, schedule, transform) {
                            amat.register(
                                *array_type,
                                array_shape,
                                schedule,
                                transform,
                            );

                            processed = true;
                            break
                        }
                    }

                    if !processed {
                        Err(format!("No array materializer can process expr ref {}", indexing_id))

                    } else {
                        Ok(())
                    }

                } else {
                    Ok(())
                }
            }
        }
    }

    // clean and fill reduced dimensions so that they can be used again
    fn clean_and_fill<T: CircuitObject+Eq+Hash>(
        &mut self,
        objects: HashSet<T>,
        old_val: CircuitValue<T>,
        ref_expr_sched: &ExprSchedule,
    ) -> (CircuitValue<T>, Option<CircuitDecl>) {
        let (dims_to_fill, mask_vector) = ref_expr_sched.dims_to_fill();

        if dims_to_fill.len() == 0 {
            (old_val, None)

        } else {
            let dims = vec![(String::from("i"), objects.len())];
            let coord_system = IndexCoordinateSystem::from_dim_list(dims.clone());

            let obj_var = T::get_fresh_variable(&mut self.registry);
            let mut obj_map: IndexCoordinateMap<T> =
                IndexCoordinateMap::from_coord_system(coord_system.clone());
            let mut obj_index: HashMap<T, im::Vector<usize>> = HashMap::new();

            let mask_var_name = self.registry.fresh_pt_var();
            let mask_var = ParamCircuitExpr::PlaintextVar(mask_var_name.clone());
            let mut mask_map: IndexCoordinateMap<PlaintextObject> =
                IndexCoordinateMap::from_coord_system(coord_system.clone());

            for (coord, obj) in obj_map.coord_iter().zip(objects.into_iter()) {
                obj_index.insert(obj.clone(), coord.clone());
                obj_map.set(coord.clone(), obj);
                mask_map.set(coord, PlaintextObject::Mask(mask_vector.clone()));
            }

            T::set_var_value(
                &mut self.registry,
                obj_var.clone(),
                CircuitValue::CoordMap(obj_map)
            );

            self.registry.set_pt_var_value(mask_var_name, CircuitValue::CoordMap(mask_map));

            // multiply objects with masks
            let obj_id = self.registry.register_circuit(obj_var);
            let mask_id = self.registry.register_circuit(mask_var);
            let mul_id =
                self.registry.register_circuit(ParamCircuitExpr::Op(Operator::Mul, obj_id, mask_id));

            // fill reduced dimensions!
            let reduction_list  = util::get_reduction_list(dims_to_fill);

            let expr_id = 
                reduction_list.into_iter()
                .fold(mul_id, |acc, n| {
                    let rot_id =
                        self.registry.register_circuit(
                            ParamCircuitExpr::Rotate(
                                OffsetExpr::Literal(n as isize),
                                acc
                            )
                        );

                    let op_id =
                        self.registry.register_circuit(
                            ParamCircuitExpr::Op(Operator::Add, acc, rot_id)
                        );

                    op_id
                });

            // create new prelude circuit for the clean and fill
            let new_circ_name = self.name_generator.get_fresh_name("__circ");
            let new_obj_val: CircuitValue<T> = match old_val {
                CircuitValue::CoordMap(coord_map) => {
                    let mut new_coord_map: IndexCoordinateMap<T> =
                        IndexCoordinateMap::from_coord_system(coord_map.coord_system.clone());

                    for (coord, obj_opt) in coord_map.object_iter() {
                        if let Some(obj) = obj_opt {
                            new_coord_map.set(
                                coord,
                                T::expr_vector(new_circ_name.clone(), obj_index[obj].clone())
                            )
                        }
                    }

                    CircuitValue::CoordMap(new_coord_map)
                },

                CircuitValue::Single(obj) => {
                    CircuitValue::Single(
                        T::expr_vector(new_circ_name.clone(), obj_index[&obj].clone())
                    )
                }
            };

            (new_obj_val, Some((new_circ_name, dims, expr_id)))
        }
    }

    fn materialize_expr_indexing_site<'b, T: CircuitObject+Eq+Hash>(
        &mut self,
        indexing_id: &IndexingId,
        array_type: ArrayType,
        _transform: &ArrayTransform,
        schedule: &IndexingSiteSchedule,
        ref_expr_sched: &ExprSchedule,
        expr_circ_val: CircuitValue<VectorInfo>,
        transform_circ_val: CircuitValue<VectorInfo>,
    ) -> Result<(ExprScheduleType, ArrayType, CircuitId, Option<CircuitDecl>), String>
    where
        CircuitObjectRegistry: CanRegisterObject<'b, T>,
        ParamCircuitExpr: CanCreateObjectVar<T>,
    {
        let derivation_opt =
            VectorDeriver::derive_from_source::<T>(&expr_circ_val, &transform_circ_val);

        match derivation_opt {
            Some((obj_val, step_val, mask_val)) => {
                let objects: HashSet<T> = match &obj_val {
                    CircuitValue::CoordMap(coord_map) => {
                        coord_map.object_iter()
                        .filter_map(|(_, obj_opt)| {
                            obj_opt.map(|obj| obj.clone())
                        }).collect()
                    },

                    CircuitValue::Single(obj) => HashSet::from([obj.clone()])
                };

                let (new_obj_val, prelude_circ_opt) =
                    self.clean_and_fill(objects, obj_val, ref_expr_sched);

                let circuit_id = VectorDeriver::gen_circuit_expr(
                    new_obj_val,
                    step_val,
                    mask_val,
                    &mut self.registry,
                );

                let expr_schedule = schedule.to_expr_schedule(ref_expr_sched.shape.clone());

                Ok((
                    ExprScheduleType::Specific(expr_schedule),
                    array_type,
                    circuit_id,
                    prelude_circ_opt,
                ))
            }

            None => {
                Err(format!("{}: cannot derive {} from {}", indexing_id, expr_circ_val, transform_circ_val))
            }
        }
    }

    fn materialize_expr(
        &mut self,
        program: &InlinedProgram,
        expr: &InlinedExpr,
        schedule: &Schedule,
        preludes: &mut Vec<CircuitDecl>,
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
                    self.materialize_expr(program, expr1, schedule, preludes)?;

                let (sched2, type2, id2) =
                    self.materialize_expr(program, expr2, schedule, preludes)?;

                let schedule = Schedule::schedule_op(&sched1, &sched2)?;

                let expr = ParamCircuitExpr::Op(op.clone(), id1, id2);
                let id = self.registry.register_circuit(expr);

                Ok((schedule, type1.join(&type2), id))
            }

            // TODO support client transforms
            InlinedExpr::ReduceNode(reduced_index, op, body) => {
                let (body_sched, body_type, mat_body) =
                    self.materialize_expr(program, body, schedule, preludes)?;

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
                        block_size /= dim.size();

                        if let VectorScheduleDim::Filled(sched_dim) = dim {
                            // if extent is 1, there's nothing to reduce!
                            if sched_dim.index == *reduced_index && sched_dim.extent > 1 {
                                reduction_list.extend(
                                    util::descending_pow2_list(sched_dim.extent >> 1)
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
                                    ParamCircuitExpr::ReduceDim(var, extent, *op, acc),
                                );

                                reduce_id
                            });

                    let expr =
                        reduction_list.into_iter()
                        .fold(expr_vec, |acc, n| {
                            let rot_id =
                                self.registry.register_circuit(
                                    ParamCircuitExpr::Rotate(
                                        OffsetExpr::Literal(-(n as isize)),
                                        acc,
                                    )
                                );

                            let op_id =
                                self.registry.register_circuit(
                                    ParamCircuitExpr::Op(*op, acc, rot_id)
                                );

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
                            
                            // vectors of the expr array being indexed
                            let expr_circ_val = ref_expr_sched.materialize(&transform.array);

                            // vectors of the indexing site
                            let transform_circ_val = VectorInfo::get_input_vector_value(
                                coord_system,
                                &ref_expr_sched.shape,
                                schedule,
                                transform,
                            );

                            let (sched_type, array_type, circ_id, prelude_circ_opt) = match array_type {
                                ArrayType::Ciphertext => self
                                    .materialize_expr_indexing_site::<CiphertextObject>(
                                        indexing_id,
                                        array_type,
                                        transform,
                                        schedule,
                                        &ref_expr_sched,
                                        expr_circ_val,
                                        transform_circ_val,
                                    ),

                                ArrayType::Plaintext => self
                                    .materialize_expr_indexing_site::<PlaintextObject>(
                                        indexing_id,
                                        array_type,
                                        transform,
                                        schedule,
                                        &ref_expr_sched,
                                        expr_circ_val,
                                        transform_circ_val,
                                    ),
                            }?;

                            if let Some(prelude) = prelude_circ_opt {
                                preludes.push(prelude);
                            }

                            Ok((sched_type, array_type, circ_id))
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

#[cfg(test)]
mod tests {
    use indexmap::IndexMap;

    use super::*;
    use crate::{
        lang::{
            elaborated::Elaborator, index_elim::IndexElimination, parser::ProgramParser,
            source::SourceProgram, ArrayType, BaseOffsetMap, OUTPUT_EXPR_NAME,
        },
        scheduling::ScheduleDim, circ::array_materializer::*
    };

    fn test_materializer(program: InlinedProgram, schedule: Schedule) -> ParamCircuitProgram {
        assert!(schedule.is_schedule_valid(&program).is_ok());

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
        let inline_set = elaborated.all_inlined_set();
        let array_group_map = elaborated.array_group_from_inline_set(&inline_set);

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

        amat.register(
            ArrayType::Ciphertext,
            &shape,
            &schedule,
            &transform,
        );

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
                .object_iter()
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
                .object_iter()
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
            preprocessing: Some(ArrayPreprocessing::Roll(0, 1)),
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
            Box::new(RollArrayMaterializer::new()),
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
            preprocessing: Some(ArrayPreprocessing::Roll(0, 1)),
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
            Box::new(RollArrayMaterializer::new()),
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
