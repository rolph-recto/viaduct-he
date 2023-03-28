use std::{collections::{HashMap, HashSet}, fmt::Display};

use log::info;

use crate::{
    circ::{CircuitId, ParamCircuitExpr},
    lang::{
        ArrayType, ArrayName, DimName, OffsetExpr, Operator,
        index_elim::{InlinedExpr, InlinedProgram},
    },
    scheduling::{ExprScheduleType, Schedule, VectorScheduleDim},
    util
};

use super::{CircuitObjectRegistry, materializer::{InputArrayMaterializer, DefaultArrayMaterializer}, cost::CostFeatures, vector_deriver::VectorDeriver, IndexCoordinateSystem, vector_info::VectorInfo, CiphertextObject, PlaintextObject};

pub struct PseudoCircuitProgram {
    pub registry: CircuitObjectRegistry,
    pub expr_list: Vec<(usize, CircuitId)>,
    pub ref_cost: CostFeatures,
}

impl Display for PseudoCircuitProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.expr_list.iter().try_for_each(|(m, expr)| {
            write!(f, "multiplicity: {}\n{}", m, self.registry.display_circuit(*expr))
        })
    }
}

/// a fake materializer used to quickly estimate circuit costs during scheduling
pub struct PseudoMaterializer<'a> {
    array_materializers: Vec<Box<dyn InputArrayMaterializer<'a> + 'a>>,
    registry: CircuitObjectRegistry,
    expr_circuit_map: HashMap<ArrayName, CircuitId>,
    expr_schedule_map: HashMap<ArrayName, ExprScheduleType>,
    expr_array_type_map: HashMap<ArrayName, ArrayType>,
}

impl<'a> PseudoMaterializer<'a> {
    pub fn new(array_materializers: Vec<Box<dyn InputArrayMaterializer<'a> + 'a>>) -> Self {
        Self {
            array_materializers,
            registry: CircuitObjectRegistry::new(),
            expr_circuit_map: HashMap::new(),
            expr_schedule_map: HashMap::new(),
            expr_array_type_map: HashMap::new(),
        }
    }

    pub fn run(
        mut self,
        program: &InlinedProgram,
        schedule: &Schedule
    ) -> Result<PseudoCircuitProgram, String> {
        let mut circuit_list: Vec<(usize, CircuitId)> = vec![];

        // need to clone expr_map here because the iteration through it is mutating
        let expr_list: Vec<(ArrayName, InlinedExpr)> = 
            program
            .expr_map
            .iter()
            .map(|(array, expr)| (array.clone(), expr.clone()))
            .collect();

        let mut ref_cost = CostFeatures::default();
        expr_list
            .into_iter()
            .try_for_each(|(array, expr)| -> Result<(), String> {
                let (expr_schedule, array_type, circuit_id) =
                    self.materialize_expr(program, &expr, schedule, &mut ref_cost)?;

                let multiplicity: usize =
                    match &expr_schedule {
                        ExprScheduleType::Any => 1,

                        ExprScheduleType::Specific(spec_sched) =>
                            spec_sched
                            .exploded_dims
                            .iter()
                            .fold(1, |acc, dim| acc * dim.extent)
                    };

                self.expr_circuit_map.insert(array.clone(), circuit_id);
                self.expr_schedule_map.insert(array.clone(), expr_schedule);
                self.expr_array_type_map.insert(array.clone(), array_type);
                circuit_list.push((multiplicity, circuit_id));
                Ok(())
            })?;

        Ok(PseudoCircuitProgram {
            registry: self.registry,
            expr_list: circuit_list,
            ref_cost,
        })
    }

    fn materialize_expr(
        &mut self,
        program: &InlinedProgram,
        expr: &InlinedExpr,
        schedule: &Schedule,
        ref_cost: &mut CostFeatures,
    ) -> Result<(ExprScheduleType, ArrayType, CircuitId), String> {
        match expr {
            InlinedExpr::Literal(lit) => {
                let sched_lit = Schedule::schedule_literal()?;
                let circuit_id =
                    self.registry.register_circuit(ParamCircuitExpr::Literal(*lit));
                Ok((sched_lit, ArrayType::Plaintext, circuit_id))
            }

            InlinedExpr::Op(op, expr1, expr2) => {
                let (sched1, type1, id1) =
                    self.materialize_expr(program, expr1, schedule, ref_cost)?;

                let (sched2, type2, id2) =
                    self.materialize_expr(program, expr2, schedule, ref_cost)?;

                let schedule = Schedule::schedule_op(&sched1, &sched2)?;

                let expr = ParamCircuitExpr::Op(op.clone(), id1, id2);
                let id = self.registry.register_circuit(expr);

                Ok((schedule, type1.join(&type2), id))
            }

            InlinedExpr::ReduceNode(reduced_index, op, body) => {
                let (body_sched, body_type, mat_body) =
                    self.materialize_expr(program, body, schedule, ref_cost)?;

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
            },

            InlinedExpr::ExprRef(indexing_id, transform) => {
                let schedule = &schedule.schedule_map[indexing_id];

                // TODO have a better estimate for expr indexing site
                if program.is_expr(&transform.array) {
                    let array_type = *self.expr_array_type_map.get(&transform.array).unwrap();
                    let ref_expr_sched =
                        match self.expr_schedule_map.get(&transform.array).unwrap() {
                            ExprScheduleType::Any => panic!("array literals not supported"),
                            ExprScheduleType::Specific(sched) => sched.clone(),
                        };

                    let var_expr =
                        match array_type {
                            ArrayType::Ciphertext =>
                                ParamCircuitExpr::CiphertextVar(String::from("var")),

                            ArrayType::Plaintext =>
                                ParamCircuitExpr::PlaintextVar(String::from("var")),
                        };

                    let var_id = self.registry.register_circuit(var_expr);

                    let expr_schedule = 
                        schedule.to_expr_schedule(transform.as_shape());
                            
                    // vectors of the expr array being indexed
                    let expr_circ_val = ref_expr_sched.materialize(&transform.array);

                    let coord_system =
                        IndexCoordinateSystem::new(schedule.exploded_dims.iter());

                    // vectors of the indexing site
                    let transform_circ_val = VectorInfo::get_input_vector_value(
                        coord_system,
                        &ref_expr_sched.shape,
                        schedule,
                        transform,
                        schedule.preprocessing,
                    );

                    let derive_opt =
                        match array_type {
                            ArrayType::Ciphertext => {
                                VectorDeriver::probe_from_source::<CiphertextObject>(
                                    &expr_circ_val,
                                    &transform_circ_val,
                                    &ref_expr_sched,
                                    array_type
                                )
                            },

                            ArrayType::Plaintext => {
                                VectorDeriver::probe_from_source::<PlaintextObject>(
                                    &expr_circ_val,
                                    &transform_circ_val,
                                    &ref_expr_sched,
                                    array_type
                                )

                            }
                        };

                    match derive_opt {
                        Some(est_cost) => {
                            *ref_cost = *ref_cost + est_cost;

                            Ok((ExprScheduleType::Specific(expr_schedule), array_type, var_id))
                        },

                        None => {
                            Err(format!("cannot derive estimate for {}", indexing_id))
                        }
                    }

                } else {
                    // indexing an input array
                    let (array_shape, array_type) =
                        &program.input_map[&transform.array];

                    for amat in self.array_materializers.iter_mut() {
                        if amat.can_materialize(*array_type, array_shape, schedule, transform) {
                            let cost = amat.estimate_cost(
                                *array_type,
                                array_shape,
                                schedule,
                                transform,
                            );

                            let shape = transform.as_shape();
                            let expr_schedule =
                                ExprScheduleType::Specific(schedule.to_expr_schedule(shape));

                            let expr_id = 
                                self.registry.register_circuit(match array_type {
                                    ArrayType::Ciphertext => ParamCircuitExpr::CiphertextVar(String::new()),
                                    ArrayType::Plaintext => ParamCircuitExpr::PlaintextVar(String::new()),
                                });

                            info!("input indexing site {} cost: {:?}", indexing_id, cost);
                            *ref_cost = *ref_cost + cost;
                            info!("ref_cost: {:?}", ref_cost);
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

pub trait PseudoMaterializerFactory {
    fn create<'a>(&self) -> PseudoMaterializer<'a>;
}

pub struct DefaultPseudoMaterializerFactory;

impl PseudoMaterializerFactory for DefaultPseudoMaterializerFactory {
    fn create<'a>(&self) -> PseudoMaterializer<'a> {
        let amats: Vec<Box<dyn InputArrayMaterializer + 'a>> =
            vec![
                Box::new(DefaultArrayMaterializer::new())
            ];

        PseudoMaterializer::new(amats)
    }
}