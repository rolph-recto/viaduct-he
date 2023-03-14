use std::collections::{HashMap, HashSet};

use crate::{
    circ::{CircuitId, ParamCircuitExpr},
    lang::{
        ArrayType, ArrayName, DimName, OffsetExpr, Operator,
        index_elim::{InlinedExpr, InlinedProgram},
    },
    scheduling::{ExprScheduleType, Schedule, VectorScheduleDim},
    util
};

use super::CircuitObjectRegistry;

pub struct PseudoCircuitProgram {
    pub registry: CircuitObjectRegistry,
    pub expr_list: Vec<(usize, CircuitId)>
}

/// a fake materializer used to quickly estimate circuit costs during scheduling
pub struct PseudoMaterializer {
    registry: CircuitObjectRegistry,
    expr_circuit_map: HashMap<ArrayName, CircuitId>,
    expr_schedule_map: HashMap<ArrayName, ExprScheduleType>,
    expr_array_type_map: HashMap<ArrayName, ArrayType>,
}

impl PseudoMaterializer {
    pub fn new() -> Self {
        Self {
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

        expr_list
            .into_iter()
            .try_for_each(|(array, expr)| -> Result<(), String> {
                let (expr_schedule, array_type, circuit_id) =
                    self.materialize_expr(program, &expr, schedule)?;

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
        })
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
                let circuit_id =
                    self.registry.register_circuit(ParamCircuitExpr::Literal(*lit));
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
            },

            InlinedExpr::ExprRef(indexing_id, transform) => {
                let schedule = &schedule.schedule_map[indexing_id];
                let array_type: ArrayType =
                    if program.is_expr(&transform.array) {
                        *self.expr_array_type_map.get(&transform.array).unwrap()

                    } else {
                        program.input_map[&transform.array].1
                    };

                let var_expr =
                    match array_type {
                        ArrayType::Ciphertext =>
                            ParamCircuitExpr::CiphertextVar(String::from("var")),

                        ArrayType::Plaintext =>
                            ParamCircuitExpr::PlaintextVar(String::from("var")),
                    };

                let var_id = self.registry.register_circuit(var_expr);
                let mask_id =
                    self.registry.register_circuit(
                        ParamCircuitExpr::PlaintextVar(String::from("mask")),
                    );

                let op_id = 
                    self.registry.register_circuit(
                        ParamCircuitExpr::Op(Operator::Mul, var_id, mask_id)
                    );

                let rot_id = 
                    self.registry.register_circuit(
                        ParamCircuitExpr::Rotate(
                            OffsetExpr::Literal(0),
                            op_id
                        )
                    );

                let expr_schedule = 
                    schedule.to_expr_schedule(transform.as_shape());

                Ok((ExprScheduleType::Specific(expr_schedule), array_type, rot_id))
            }
        }
    }
}