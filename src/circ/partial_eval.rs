use super::ParamCircuitExpr;
use crate::{circ::*, util::NameGenerator};

#[derive(Clone, Debug)]
pub enum PartialCircuit {
    Ciphertext(ParamCircuitExpr),
    Plaintext(ParamCircuitExpr)
}

/// partially evaluate plaintexts to remove them from the HE circuit
pub struct PlaintextHoisting { name_generator: NameGenerator }

impl PlaintextHoisting {
    pub fn new() -> Self {
        Self { name_generator: NameGenerator::new() }
    }

    fn build_ref_obj(
        array: ArrayName,
        dims: Vec<(DimName, Extent)>
    ) -> CircuitValue<PlaintextObject> {
        if dims.len() > 0 {
            let mut coord_map: IndexCoordinateMap<PlaintextObject> =
                IndexCoordinateMap::from_coord_system(IndexCoordinateSystem::from_dim_list(dims));

            for coord in coord_map.coord_iter() {
                coord_map.set(
                    coord.clone(),
                    PlaintextObject::ExprVector(array.clone(), coord)
                );
            }

            CircuitValue::CoordMap(coord_map)

        } else {
            CircuitValue::Single(PlaintextObject::ExprVector(array, im::Vector::new()))
        }
    }

    fn partial_eval_op(
        &mut self,
        op: Operator,
        circ1: PartialCircuit,
        new_id1: CircuitId,
        circ2: PartialCircuit,
        new_id2: CircuitId,
        dims: &Vec<(DimName,Extent)>,
        new_registry: &mut CircuitObjectRegistry,
        native_expr_list: &mut Vec<(ArrayName, Vec<(DimName, Extent)>, CircuitId)>,
    ) -> (PartialCircuit, CircuitId) {
        match (circ1, circ2) {
            (PartialCircuit::Ciphertext(_), PartialCircuit::Ciphertext(_)) => {
                let new_circuit =
                    ParamCircuitExpr::Op(op, new_id1, new_id2);

                let new_id =
                    new_registry.register_circuit(new_circuit.clone());

                (PartialCircuit::Ciphertext(new_circuit), new_id)
            },

            (PartialCircuit::Ciphertext(_), PartialCircuit::Plaintext(p2)) => {
                match p2 {
                    ParamCircuitExpr::CiphertextVar(_) =>
                        unreachable!(),

                    // don't create a new circuit
                    ParamCircuitExpr::PlaintextVar(_) |
                    ParamCircuitExpr::Literal(_) => {
                        let new_circuit =
                            ParamCircuitExpr::Op(op, new_id1, new_id2);

                        let new_id =
                            new_registry.register_circuit(new_circuit.clone());

                        (PartialCircuit::Ciphertext(new_circuit), new_id)
                    },

                    // create a new native circuit
                    ParamCircuitExpr::Op(_, _, _) |
                    ParamCircuitExpr::Rotate(_, _) |
                    ParamCircuitExpr::ReduceDim(_, _, _, _) => {
                        let new_array =
                            self.name_generator.get_fresh_name("__partial");

                        let new_var = new_registry.fresh_pt_var();

                        let new_circval: CircuitValue<PlaintextObject> =
                            Self::build_ref_obj(new_array.clone(), dims.clone());

                        new_registry.set_pt_var_value(new_var.clone(), new_circval);

                        native_expr_list.push(
                            (new_array, dims.clone(), new_id2)
                        );

                        let pt_var_id =
                            new_registry.register_circuit(
                                ParamCircuitExpr::PlaintextVar(new_var)
                            );

                        let new_circuit =
                            ParamCircuitExpr::Op(op, new_id1, pt_var_id);

                        let new_id =
                            new_registry.register_circuit(new_circuit.clone());

                        (PartialCircuit::Ciphertext(new_circuit), new_id)
                    }
                }
            },

            (PartialCircuit::Plaintext(p1), PartialCircuit::Ciphertext(_)) => {
                match p1 {
                    ParamCircuitExpr::CiphertextVar(_) =>
                        unreachable!(),

                    // don't create a new circuit
                    ParamCircuitExpr::PlaintextVar(_) |
                    ParamCircuitExpr::Literal(_) => {
                        let new_circuit =
                            ParamCircuitExpr::Op(op, new_id1, new_id2);

                        let new_id =
                            new_registry.register_circuit(new_circuit.clone());

                        (PartialCircuit::Ciphertext(new_circuit), new_id)
                    },

                    // create a new native circuit
                    ParamCircuitExpr::Op(_, _, _) |
                    ParamCircuitExpr::Rotate(_, _) |
                    ParamCircuitExpr::ReduceDim(_, _, _, _) => {
                        let new_array =
                            self.name_generator.get_fresh_name("__partial__");

                        let new_var = new_registry.fresh_pt_var();

                        let new_circval: CircuitValue<PlaintextObject> =
                            Self::build_ref_obj(new_array.clone(), dims.clone());

                        new_registry.set_pt_var_value(new_var.clone(), new_circval);

                        native_expr_list.push(
                            (new_array, dims.clone(), new_id1)
                        );

                        let pt_var_id =
                            new_registry.register_circuit(
                                ParamCircuitExpr::PlaintextVar(new_var)
                            );

                        let new_circuit =
                            ParamCircuitExpr::Op(op, pt_var_id, new_id2);

                        let new_id =
                            new_registry.register_circuit(new_circuit.clone());

                        (PartialCircuit::Ciphertext(new_circuit), new_id)
                    }
                }
            },

            (PartialCircuit::Plaintext(p1), PartialCircuit::Plaintext(p2)) => {
                let new_circuit =
                    if let (ParamCircuitExpr::Literal(lit1), ParamCircuitExpr::Literal(lit2)) = (p1, p2) {
                        match op {
                            Operator::Add =>
                                ParamCircuitExpr::Literal(lit1 + lit2),

                            Operator::Sub => 
                                ParamCircuitExpr::Literal(lit1 - lit2),

                            Operator::Mul =>
                                ParamCircuitExpr::Literal(lit1 * lit2),
                        }

                    } else {
                        ParamCircuitExpr::Op(op, new_id1, new_id2)
                    };

                let new_id =
                    new_registry.register_circuit(new_circuit.clone());

                (PartialCircuit::Plaintext(new_circuit), new_id)
            },
        }
    }

    // this copies the partially evaluated circuit to a new registry
    fn partial_eval(
        &mut self,
        circuit_id: CircuitId,
        dims: &Vec<(DimName,Extent)>,
        old_registry: &CircuitObjectRegistry,
        new_registry: &mut CircuitObjectRegistry,
        native_expr_list: &mut Vec<(ArrayName, Vec<(DimName, Extent)>, CircuitId)>,
        cache: &mut HashMap<CircuitId, (PartialCircuit, CircuitId)>,
    ) -> (PartialCircuit, CircuitId) {
        if let Some((partial_circ, new_id)) = cache.get(&circuit_id) {
            return (partial_circ.clone(), *new_id);
        }

        let circuit = old_registry.get_circuit(circuit_id);
        match circuit {
            ParamCircuitExpr::CiphertextVar(var) => {
                let new_var = new_registry.fresh_ct_var();
                let circval = old_registry.get_ct_var_value(var);
                new_registry.set_ct_var_value(new_var.clone(), circval.clone());

                let new_circuit = ParamCircuitExpr::CiphertextVar(new_var);
                let new_id = new_registry.register_circuit(new_circuit.clone());
                let partial_circ = PartialCircuit::Ciphertext(new_circuit);
                cache.insert(circuit_id, (partial_circ.clone(), new_id));
                (partial_circ, new_id)
            },

            ParamCircuitExpr::PlaintextVar(var) => {
                let new_var = new_registry.fresh_pt_var();
                let circval = old_registry.get_pt_var_value(var);
                new_registry.set_pt_var_value(new_var.clone(), circval.clone());

                let new_circuit = ParamCircuitExpr::PlaintextVar(new_var);
                let new_id = new_registry.register_circuit(new_circuit.clone());
                let partial_circ = PartialCircuit::Plaintext(new_circuit);
                cache.insert(circuit_id, (partial_circ.clone(), new_id));
                (partial_circ, new_id)
            },

            ParamCircuitExpr::Literal(lit) => {
                let new_id = new_registry.register_circuit(circuit.clone());
                let partial_circ = PartialCircuit::Plaintext(circuit.clone());
                cache.insert(circuit_id, (partial_circ.clone(), new_id));
                (partial_circ, new_id)
            },

            ParamCircuitExpr::Op(op, expr1, expr2) => {
                let (circ1, new_id1) =
                    self.partial_eval(
                        *expr1, 
                        dims,
                        old_registry,
                        new_registry, 
                        native_expr_list, 
                        cache
                    );

                let (circ2, new_id2) =
                    self.partial_eval(
                        *expr2, 
                        dims,
                        old_registry, new_registry,
                        native_expr_list,
                        cache
                    );

                let (partial_circ, new_id) =
                    self.partial_eval_op(
                        *op, circ1, new_id1, circ2, new_id2,
                        dims,
                        new_registry,
                        native_expr_list,
                    );

                cache.insert(circuit_id, (partial_circ.clone(), new_id));
                (partial_circ, new_id)
            },

            ParamCircuitExpr::Rotate(steps, body)  => {
                let (body_circ, new_body_id) =
                    self.partial_eval(
                        *body,
                        dims,
                         old_registry,
                         new_registry,
                         native_expr_list,
                         cache,
                    );

                let new_circuit =
                    ParamCircuitExpr::Rotate(steps.clone(), new_body_id);

                // TODO registry old fvars in new registry?
                for fvar in steps.function_vars() {
                    let fvar_circval =
                        old_registry.get_offset_fvar_value(&fvar);

                    new_registry.offset_fvar_values.insert(
                        fvar, fvar_circval.clone()
                    );
                }

                let new_id = new_registry.register_circuit(new_circuit.clone());
                let partial_circ =
                    match body_circ {
                        PartialCircuit::Ciphertext(_) => {
                            PartialCircuit::Ciphertext(new_circuit)
                        },

                        PartialCircuit::Plaintext(_) => {
                            PartialCircuit::Plaintext(new_circuit)
                        },
                    };
                cache.insert(circuit_id, (partial_circ.clone(), new_id));
                (partial_circ, new_id)
            },

            ParamCircuitExpr::ReduceDim(dim, extent, op, body) => {
                let (body_circ, new_body_id) =
                    self.partial_eval(
                        *body,
                        dims,
                         old_registry,
                         new_registry,
                         native_expr_list,
                         cache,
                    );

                let new_circuit =
                    ParamCircuitExpr::ReduceDim(dim.clone(), *extent, *op, new_body_id);
                let new_id = new_registry.register_circuit(new_circuit.clone());
                let partial_circ =
                    match body_circ {
                        PartialCircuit::Ciphertext(_) => {
                            PartialCircuit::Ciphertext(new_circuit)
                        },

                        PartialCircuit::Plaintext(_) => {
                            PartialCircuit::Plaintext(new_circuit)
                        },
                    };
                cache.insert(circuit_id, (partial_circ.clone(), new_id));
                (partial_circ, new_id)
            },
        }
    }

    pub fn run(&mut self, program: ParamCircuitProgram) -> ParamCircuitProgram {
        // assume circuit has not been partially evaluated yet
        assert!(program.native_expr_list.len() == 0);

        let mut native_expr_list: Vec<(ArrayName, Vec<(DimName, Extent)>, CircuitId)> = Vec::new();
        let mut new_circuit_expr_list: Vec<(ArrayName, Vec<(DimName, Extent)>, CircuitId)> = Vec::new();
        let mut new_registry = CircuitObjectRegistry::new();

        for (array, dims, circuit_id) in program.circuit_expr_list {
            let (partial_circ, new_id) =
                self.partial_eval(
                    circuit_id,
                    &dims,
                    &program.registry,
                    &mut new_registry,
                    &mut native_expr_list,
                    &mut HashMap::new(),
                );

            match partial_circ {
                PartialCircuit::Ciphertext(_) => {
                    new_circuit_expr_list.push((array, dims, new_id));
                },

                PartialCircuit::Plaintext(_) => {
                    native_expr_list.push((array, dims, new_id));
                },
            }
        }

        ParamCircuitProgram {
            registry: new_registry,
            native_expr_list,
            circuit_expr_list: new_circuit_expr_list,
        }
    }
}
