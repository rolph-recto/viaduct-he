use std::collections::{HashMap, HashSet};

use crate::{circ::*, lang::*, program::*, util::NameGenerator};

/// pass that lowers ParamCircuitProgram into HEProgram
pub struct CircuitLowering {
    name_generator: NameGenerator,
    cur_instr_id: InstructionId,
    circuit_instr_map: HashMap<CircuitId, InstructionId>,
}

impl CircuitLowering {
    pub fn new() -> Self {
        Self {
            name_generator: NameGenerator::new(),
            cur_instr_id: 0,
            circuit_instr_map: HashMap::new(),
        }
    }

    fn fresh_instr_id(&mut self) -> InstructionId {
        let id = self.cur_instr_id;
        self.cur_instr_id += 1;
        id
    }

    fn resolve_ciphertext_object(obj: &CiphertextObject, input: &HEProgramContext) -> HERef {
        match obj {
            CiphertextObject::InputVector(vector) => {
                let vector_var = input.vector_map.get(vector).unwrap().clone();
                HERef::Ciphertext(vector_var, vec![])
            }

            CiphertextObject::ExprVector(array, coords) => {
                let coord_index = coords.iter().map(|coord| HEIndex::Literal(*coord as isize));
                HERef::Ciphertext(array.clone(), Vec::from_iter(coord_index))
            }
        }
    }

    // TODO finish this
    fn compute_coord_relationship(
        index_vars: Vec<DimName>,
        coord_val_map: HashMap<IndexCoord, IndexCoord>,
    ) -> Option<Vec<HEIndex>> {
        None
    }

    fn inline_ciphertext_object(
        circval: &CircuitValue<CiphertextObject>,
        input: &HEProgramContext,
    ) -> Option<HERef> {
        match circval {
            CircuitValue::CoordMap(coord_map) => {
                let mut vector_var_set: HashSet<String> = HashSet::new();
                let mut coord_val_map: HashMap<IndexCoord, IndexCoord> = HashMap::new();

                for (coord, obj_opt) in coord_map.value_iter() {
                    let obj = obj_opt.unwrap();
                    match obj {
                        CiphertextObject::InputVector(vector) => {
                            vector_var_set.insert(input.vector_map.get(vector).unwrap().clone());
                        }

                        CiphertextObject::ExprVector(ref_vector, ref_coord) => {
                            vector_var_set.insert(ref_vector.clone());
                            coord_val_map.insert(coord, ref_coord.clone());
                        }
                    }
                }

                if vector_var_set.len() == 1 {
                    let vector_var = vector_var_set.into_iter().next().unwrap();
                    if coord_val_map.len() == 0 {
                        Some(HERef::Ciphertext(vector_var, vec![]))
                    } else {
                        let index_opt =
                            Self::compute_coord_relationship(coord_map.index_vars(), coord_val_map);

                        if let Some(index) = index_opt {
                            Some(HERef::Ciphertext(vector_var, index))
                        } else {
                            None
                        }
                    }
                } else {
                    None
                }
            }

            CircuitValue::Single(obj) => Some(Self::resolve_ciphertext_object(obj, input)),
        }
    }

    fn inline_plaintext_object(
        circval: &CircuitValue<PlaintextObject>,
        input: &HEProgramContext,
    ) -> Option<HERef> {
        match circval {
            CircuitValue::CoordMap(coord_map) => {
                let mut vector_var_set: HashSet<String> = HashSet::new();
                let mut coord_val_map: HashMap<IndexCoord, IndexCoord> = HashMap::new();

                for (coord, obj_opt) in coord_map.value_iter() {
                    let obj = obj_opt.unwrap();
                    match obj {
                        PlaintextObject::InputVector(vector) => {
                            vector_var_set.insert(input.vector_map.get(vector).unwrap().clone());
                        }

                        PlaintextObject::ExprVector(ref_vector, ref_coord) => {
                            vector_var_set.insert(ref_vector.clone());
                            coord_val_map.insert(coord, ref_coord.clone());
                        }

                        PlaintextObject::Mask(mask) => {
                            vector_var_set.insert(input.mask_map.get(mask).unwrap().clone());
                        }

                        PlaintextObject::Const(constval) => {
                            vector_var_set.insert(input.const_map.get(constval).unwrap().clone());
                        }
                    }
                }

                if vector_var_set.len() == 1 {
                    let vector_var = vector_var_set.into_iter().next().unwrap();
                    if coord_val_map.len() == 0 {
                        Some(HERef::Plaintext(vector_var, vec![]))
                    } else {
                        let index_opt =
                            Self::compute_coord_relationship(coord_map.index_vars(), coord_val_map);

                        if let Some(index) = index_opt {
                            Some(HERef::Plaintext(vector_var, index))
                        } else {
                            None
                        }
                    }
                } else {
                    None
                }
            }

            CircuitValue::Single(obj) => Some(Self::resolve_plaintext_object(obj, input)),
        }
    }

    fn resolve_plaintext_object(obj: &PlaintextObject, input: &HEProgramContext) -> HERef {
        match obj {
            PlaintextObject::InputVector(vector) => {
                let vector_var = input.vector_map.get(vector).unwrap().clone();
                HERef::Plaintext(vector_var, vec![])
            }

            PlaintextObject::ExprVector(array, coords) => {
                let coord_index = coords.iter().map(|coord| HEIndex::Literal(*coord as isize));
                HERef::Plaintext(array.clone(), Vec::from_iter(coord_index))
            }

            PlaintextObject::Const(val) => {
                let const_var = input.const_map.get(val).unwrap().clone();
                HERef::Plaintext(const_var, vec![])
            }

            PlaintextObject::Mask(mask) => {
                let mask_var = input.mask_map.get(mask).unwrap().clone();
                HERef::Plaintext(mask_var, vec![])
            }
        }
    }

    fn resolve_offset(offset: &isize, _input: &HEProgramContext) -> HEOperand {
        HEOperand::Literal(*offset)
    }

    fn gen_program_context(&mut self, registry: &CircuitObjectRegistry) -> HEProgramContext {
        let mut vector_map = HashMap::new();
        let mut mask_map = HashMap::new();
        let mut const_map = HashMap::new();

        for vector in registry.get_vectors() {
            let vector_name = self.name_generator.get_fresh_name("vector");
            vector_map.insert(vector, vector_name);
        }

        for mask in registry.get_masks() {
            let mask_name = self.name_generator.get_fresh_name("mask");
            mask_map.insert(mask, mask_name);
        }

        for constval in registry.get_constants() {
            const_map.insert(constval, format!("const_{}", constval));
        }

        HEProgramContext {
            vector_map,
            mask_map,
            const_map,
        }
    }

    fn process_circuit_val<T>(
        value: &CircuitValue<T>,
        var: String,
        input: &HEProgramContext,
        f: fn(&T, &HEProgramContext) -> HEOperand,
        statements: &mut Vec<HEStatement>,
    ) {
        match value {
            CircuitValue::CoordMap(coord_map) => {
                for (coords, obj) in coord_map.value_iter() {
                    let operand = f(obj.unwrap(), &input);
                    let coord_index = Vec::from_iter(
                        coords.iter().map(|coord| HEIndex::Literal(*coord as isize)),
                    );

                    statements.push(HEStatement::SetVar(var.clone(), coord_index, operand));
                }
            }

            CircuitValue::Single(obj) => {
                let operand = f(obj, &input);
                statements.push(HEStatement::SetVar(var, vec![], operand));
            }
        }
    }

    pub fn lower(&mut self, program: ParamCircuitProgram) -> HEProgram {
        let mut statements: Vec<HEStatement> = Vec::new();
        let context = self.gen_program_context(&program.registry);

        let mut ct_inline_map: HashMap<VarName, HERef> = HashMap::new();
        let mut pt_inline_map: HashMap<VarName, HERef> = HashMap::new();

        // process statements
        for (array, dims, circuit_id) in program.expr_list {
            // preamble: allocate arrays referenced in the circuit expr
            for ct_var in program.registry.circuit_ciphertext_vars(circuit_id) {
                let circval = program.registry.get_ct_var_value(&ct_var);

                if let Some(operand) = Self::inline_ciphertext_object(circval, &context) {
                    ct_inline_map.insert(ct_var, operand);
                } else {
                    CircuitLowering::process_circuit_val(
                        program.registry.get_ct_var_value(&ct_var),
                        ct_var,
                        &context,
                        |obj, input| {
                            HEOperand::Ref(CircuitLowering::resolve_ciphertext_object(obj, input))
                        },
                        &mut statements,
                    );
                }
            }

            for pt_var in program.registry.circuit_plaintext_vars(circuit_id) {
                let circval = program.registry.get_pt_var_value(&pt_var);

                if let Some(operand) = Self::inline_plaintext_object(circval, &context) {
                    pt_inline_map.insert(pt_var, operand);
                } else {
                    CircuitLowering::process_circuit_val(
                        program.registry.get_pt_var_value(&pt_var),
                        pt_var,
                        &context,
                        |obj, input| {
                            HEOperand::Ref(CircuitLowering::resolve_plaintext_object(obj, input))
                        },
                        &mut statements,
                    );
                }
            }

            for offset_fvar in program.registry.circuit_offset_fvars(circuit_id) {
                CircuitLowering::process_circuit_val(
                    program.registry.get_offset_fvar_value(&offset_fvar),
                    offset_fvar,
                    &context,
                    CircuitLowering::resolve_offset,
                    &mut statements,
                );
            }

            let dim_vars: Vec<String> = dims.iter().map(|(var, _)| var.clone()).collect();

            let dim_extents = dims.iter().map(|(_, extent)| *extent).collect();

            // generate statements for array expr
            // first, declare array
            statements.push(HEStatement::DeclareVar(array.clone(), dim_extents));

            // generate statements in the array expr's body
            let mut body_statements: Vec<HEStatement> = Vec::new();
            let array_id = self.gen_expr_instrs_recur(
                circuit_id,
                &program.registry,
                &context,
                &dims,
                &ct_inline_map,
                &pt_inline_map,
                &mut HashMap::new(),
                &mut body_statements,
            );

            // set the array's value to the body statement's computed value
            body_statements.push(HEStatement::SetVar(
                array,
                dim_vars.into_iter().map(|var| HEIndex::Var(var)).collect(),
                HEOperand::Ref(HERef::Instruction(array_id)),
            ));

            let mut dims_reversed = dims.clone();
            dims_reversed.reverse();

            // wrap the body in a nest of for loops
            let array_statement = dims_reversed
                .into_iter()
                .fold(body_statements, |acc, (dim, extent)| {
                    vec![HEStatement::ForNode(dim, extent, acc)]
                })
                .pop()
                .unwrap();

            statements.push(array_statement);
        }

        HEProgram {
            context,
            statements,
        }
    }

    pub fn gen_expr_instrs_recur(
        &mut self,
        expr_id: CircuitId,
        registry: &CircuitObjectRegistry,
        context: &HEProgramContext,
        indices: &Vec<(String, usize)>,
        ct_inline_map: &HashMap<VarName, HERef>,
        pt_inline_map: &HashMap<VarName, HERef>,
        ref_map: &mut HashMap<usize, HERef>,
        stmts: &mut Vec<HEStatement>,
    ) -> InstructionId {
        if let Some(instr_id) = self.circuit_instr_map.get(&expr_id) {
            *instr_id
        } else {
            match registry.get_circuit(expr_id) {
                ParamCircuitExpr::CiphertextVar(var) => {
                    let instr_id = self.fresh_instr_id();
                    if let Some(var_ref) = ct_inline_map.get(var) {
                        ref_map.insert(instr_id, var_ref.clone());
                    } else {
                        let index_vars = indices
                            .iter()
                            .map(|(var, _)| HEIndex::Var(var.clone()))
                            .collect();
                        let var_ref = HERef::Ciphertext(var.clone(), index_vars);
                        ref_map.insert(instr_id, var_ref);
                    }

                    self.circuit_instr_map.insert(expr_id, instr_id);
                    instr_id
                }

                ParamCircuitExpr::PlaintextVar(var) => {
                    let instr_id = self.fresh_instr_id();
                    if let Some(var_ref) = pt_inline_map.get(var) {
                        ref_map.insert(instr_id, var_ref.clone());
                    } else {
                        let index_vars = indices
                            .iter()
                            .map(|(var, _)| HEIndex::Var(var.clone()))
                            .collect();
                        let var_ref = HERef::Plaintext(var.clone(), index_vars);
                        ref_map.insert(instr_id, var_ref);
                    }

                    instr_id
                }

                ParamCircuitExpr::Literal(val) => {
                    let lit_ref = context.const_map.get(val).unwrap();
                    let instr_id = self.fresh_instr_id();
                    let lit_ref = HERef::Plaintext(lit_ref.clone(), vec![]);
                    ref_map.insert(instr_id, lit_ref);
                    self.circuit_instr_map.insert(expr_id, instr_id);
                    instr_id
                }

                ParamCircuitExpr::Op(op, expr1, expr2) => {
                    let id1 = self.gen_expr_instrs_recur(
                        *expr1,
                        registry,
                        context,
                        indices,
                        ct_inline_map,
                        pt_inline_map,
                        ref_map,
                        stmts,
                    );

                    let id2 = self.gen_expr_instrs_recur(
                        *expr2,
                        registry,
                        context,
                        indices,
                        ct_inline_map,
                        pt_inline_map,
                        ref_map,
                        stmts,
                    );

                    let ref1 = ref_map.get(&id1).unwrap().clone();
                    let ref2 = ref_map.get(&id2).unwrap().clone();

                    let id = self.fresh_instr_id();
                    let instr = match op {
                        Operator::Add => HEInstruction::Add(id, ref1, ref2),
                        Operator::Sub => HEInstruction::Sub(id, ref1, ref2),
                        Operator::Mul => HEInstruction::Mul(id, ref1, ref2),
                    };

                    stmts.push(HEStatement::Instruction(instr));
                    ref_map.insert(id, HERef::Instruction(id));
                    id
                }

                ParamCircuitExpr::Rotate(steps, body) => {
                    let body_id = self.gen_expr_instrs_recur(
                        *body,
                        registry,
                        context,
                        indices,
                        ct_inline_map,
                        pt_inline_map,
                        ref_map,
                        stmts,
                    );

                    let id = self.fresh_instr_id();

                    let body_operand = ref_map.get(&body_id).unwrap().clone();
                    stmts.push(HEStatement::Instruction(HEInstruction::Rot(
                        id,
                        steps.clone(),
                        body_operand,
                    )));
                    ref_map.insert(id, HERef::Instruction(id));

                    id
                }

                ParamCircuitExpr::ReduceVectors(dim, extent, op, body) => {
                    let mut body_indices = indices.clone();
                    body_indices.push((dim.clone(), *extent));

                    let mut body_stmts: Vec<HEStatement> = Vec::new();

                    let body_id = self.gen_expr_instrs_recur(
                        *body,
                        registry,
                        context,
                        &body_indices,
                        ct_inline_map,
                        pt_inline_map,
                        ref_map,
                        &mut body_stmts,
                    );

                    let body_operand = ref_map.get(&body_id).unwrap().clone();

                    let reduce_var = self.name_generator.get_fresh_name("reduce");

                    let reduce_var_ref = HERef::Ciphertext(reduce_var.clone(), vec![]);

                    let reduce_id = self.fresh_instr_id();

                    let reduce_stmt = match op {
                        Operator::Add => {
                            HEInstruction::Add(reduce_id, reduce_var_ref.clone(), body_operand)
                        }

                        Operator::Sub => {
                            HEInstruction::Sub(reduce_id, reduce_var_ref.clone(), body_operand)
                        }

                        Operator::Mul => {
                            HEInstruction::Mul(reduce_id, reduce_var_ref.clone(), body_operand)
                        }
                    };

                    body_stmts.push(HEStatement::Instruction(reduce_stmt));
                    body_stmts.push(HEStatement::SetVar(
                        reduce_var.clone(),
                        vec![],
                        HEOperand::Ref(HERef::Instruction(reduce_id)),
                    ));

                    stmts.extend([
                        HEStatement::DeclareVar(reduce_var.clone(), vec![]),
                        HEStatement::ForNode(dim.clone(), *extent, body_stmts),
                    ]);

                    let id = self.fresh_instr_id();
                    ref_map.insert(id, reduce_var_ref);

                    id
                }
            }
        }
    }
}
