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
                let vector_var = input.ct_vector_map.get(vector).unwrap().clone();
                HERef::Array(vector_var, vec![])
            }

            CiphertextObject::ExprVector(array, coords) => {
                let coord_index = coords.iter().map(|coord| HEIndex::Literal(*coord as isize));
                HERef::Array(array.clone(), Vec::from_iter(coord_index))
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
                            vector_var_set.insert(input.ct_vector_map.get(vector).unwrap().clone());
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
                        Some(HERef::Array(vector_var, vec![]))
                    } else {
                        let index_opt =
                            Self::compute_coord_relationship(coord_map.index_vars(), coord_val_map);

                        if let Some(index) = index_opt {
                            Some(HERef::Array(vector_var, index))
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
                            vector_var_set.insert(input.ct_vector_map.get(vector).unwrap().clone());
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
                        Some(HERef::Array(vector_var, vec![]))
                    } else {
                        let index_opt =
                            Self::compute_coord_relationship(coord_map.index_vars(), coord_val_map);

                        if let Some(index) = index_opt {
                            Some(HERef::Array(vector_var, index))
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
                let vector_var = input.pt_vector_map.get(vector).unwrap().clone();
                HERef::Array(vector_var, vec![])
            }

            PlaintextObject::ExprVector(array, coords) => {
                let coord_index = coords.iter().map(|coord| HEIndex::Literal(*coord as isize));
                HERef::Array(array.clone(), Vec::from_iter(coord_index))
            }

            PlaintextObject::Const(val) => {
                let const_var = input.const_map.get(val).unwrap().clone();
                HERef::Array(const_var, vec![])
            }

            PlaintextObject::Mask(mask) => {
                let mask_var = input.mask_map.get(mask).unwrap().clone();
                HERef::Array(mask_var, vec![])
            }
        }
    }

    fn resolve_offset(offset: &isize, _input: &HEProgramContext) -> HEOperand {
        HEOperand::Literal(*offset)
    }

    fn gen_program_context(&mut self, registry: &CircuitObjectRegistry) -> HEProgramContext {
        let mut ct_vector_map = HashMap::new();
        let mut pt_vector_map = HashMap::new();
        let mut mask_map = HashMap::new();
        let mut const_map = HashMap::new();

        for vector in registry.get_ciphertext_vectors() {
            let vector_name = self.name_generator.get_fresh_name("ct");
            ct_vector_map.insert(vector, vector_name);
        }

        for vector in registry.get_plaintext_vectors() {
            let vector_name = self.name_generator.get_fresh_name("pt");
            pt_vector_map.insert(vector, vector_name);
        }

        for mask in registry.get_masks() {
            let mask_name = self.name_generator.get_fresh_name("mask");
            mask_map.insert(mask, mask_name);
        }

        for constval in registry.get_constants() {
            const_map.insert(constval, format!("const_{}", constval));
        }

        HEProgramContext {
            ct_vector_map,
            pt_vector_map,
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

    pub fn run(&mut self, program: ParamCircuitProgram) -> HEProgram {
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

            let dim_vars: Vec<DimName> =
                dims.iter().map(|(var, _)| var.clone()).collect();

            let dim_extents: Vec<Extent> =
                dims.iter().map(|(_, extent)| *extent).collect();

            // generate statements for array expr
            // first, declare array
            statements.push(HEStatement::DeclareVar(array.clone(), HEType::Ciphertext, dim_extents));

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
        ref_map: &mut HashMap<usize, (HERef, HEType)>,
        stmts: &mut Vec<HEStatement>,
    ) -> InstructionId {
        if let Some(instr_id) = self.circuit_instr_map.get(&expr_id) {
            *instr_id
        } else {
            match registry.get_circuit(expr_id) {
                ParamCircuitExpr::CiphertextVar(var) => {
                    let instr_id = self.fresh_instr_id();
                    if let Some(var_ref) = ct_inline_map.get(var) {
                        ref_map.insert(instr_id, (var_ref.clone(), HEType::Ciphertext));
                    } else {
                        let index_vars = indices
                            .iter()
                            .map(|(var, _)| HEIndex::Var(var.clone()))
                            .collect();
                        let var_ref = HERef::Array(var.clone(), index_vars);
                        ref_map.insert(instr_id, (var_ref, HEType::Ciphertext));
                    }

                    self.circuit_instr_map.insert(expr_id, instr_id);
                    instr_id
                }

                ParamCircuitExpr::PlaintextVar(var) => {
                    let instr_id = self.fresh_instr_id();
                    if let Some(var_ref) = pt_inline_map.get(var) {
                        ref_map.insert(instr_id, (var_ref.clone(), HEType::Plaintext));
                    } else {
                        let index_vars = indices
                            .iter()
                            .map(|(var, _)| HEIndex::Var(var.clone()))
                            .collect();
                        let var_ref = HERef::Array(var.clone(), index_vars);
                        ref_map.insert(instr_id, (var_ref, HEType::Plaintext));
                    }

                    instr_id
                }

                ParamCircuitExpr::Literal(val) => {
                    let lit_ref = context.const_map.get(val).unwrap();
                    let instr_id = self.fresh_instr_id();
                    let lit_ref = HERef::Array(lit_ref.clone(), vec![]);
                    ref_map.insert(instr_id, (lit_ref, HEType::Plaintext));
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

                    let (ref1, type1) = ref_map.get(&id1).unwrap().clone();
                    let (ref2, type2) = ref_map.get(&id2).unwrap().clone();

                    // make sure that operand 1 is always ciphertext
                    let (optype, ref1_final, ref2_final) =
                        match (type1, type2) {
                            (HEType::Ciphertext, HEType::Ciphertext) =>
                                (HEInstructionType::CipherCipher, ref1, ref2),

                            (HEType::Plaintext, HEType::Ciphertext) =>
                                (HEInstructionType::CipherPlain, ref2, ref1),

                            (HEType::Ciphertext, HEType::Plaintext) =>
                                (HEInstructionType::CipherPlain, ref1, ref2),

                            (HEType::Native, _) | (_, HEType::Native) |
                            (HEType::Plaintext, HEType::Plaintext) =>
                                unreachable!()
                        };

                    let id = self.fresh_instr_id();
                    let instr = match op {
                        Operator::Add => HEInstruction::Add(optype, id, ref1_final, ref2_final),
                        Operator::Sub => HEInstruction::Sub(optype, id, ref1_final, ref2_final),
                        Operator::Mul => HEInstruction::Mul(optype, id, ref1_final, ref2_final),
                    };

                    stmts.push(HEStatement::Instruction(instr));
                    ref_map.insert(id, (HERef::Instruction(id), HEType::Ciphertext));
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

                    let (body_operand, body_type) = ref_map.get(&body_id).unwrap().clone();

                    // plaintext rotations should have been partially evaluated
                    assert!(body_type == HEType::Ciphertext);

                    stmts.push(HEStatement::Instruction(HEInstruction::Rot(
                        HEInstructionType::CipherCipher,
                        id,
                        steps.clone(),
                        body_operand,
                    )));
                    ref_map.insert(id, (HERef::Instruction(id), HEType::Ciphertext));

                    id
                }

                ParamCircuitExpr::ReduceDim(dim, extent, op, body) => {
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

                    let (body_operand, body_type) = ref_map.get(&body_id).unwrap().clone();

                    // the body must be ciphertext, or else
                    // it would've been partially evaluated
                    assert!(body_type == HEType::Ciphertext);

                    let reduce_var = self.name_generator.get_fresh_name("reduce");

                    let reduce_var_ref = HERef::Array(reduce_var.clone(), vec![]);

                    let reduce_id = self.fresh_instr_id();

                    let reduce_type = HEInstructionType::CipherCipher;

                    let reduce_stmt = match op {
                        Operator::Add => {
                            HEInstruction::Add(reduce_type, reduce_id, reduce_var_ref.clone(), body_operand)
                        }

                        Operator::Sub => {
                            HEInstruction::Sub(reduce_type, reduce_id, reduce_var_ref.clone(), body_operand)
                        }

                        Operator::Mul => {
                            HEInstruction::Mul(reduce_type, reduce_id, reduce_var_ref.clone(), body_operand)
                        }
                    };

                    body_stmts.push(HEStatement::Instruction(reduce_stmt));
                    body_stmts.push(HEStatement::SetVar(
                        reduce_var.clone(),
                        vec![],
                        HEOperand::Ref(HERef::Instruction(reduce_id)),
                    ));

                    stmts.extend([
                        HEStatement::DeclareVar(reduce_var.clone(), HEType::Ciphertext, vec![]),
                        HEStatement::ForNode(dim.clone(), *extent, body_stmts),
                    ]);

                    let id = self.fresh_instr_id();
                    ref_map.insert(id, (reduce_var_ref, HEType::Ciphertext));

                    id
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        circ::{
            CiphertextObject, CircuitObjectRegistry, CircuitValue, IndexCoordinateMap,
            IndexCoordinateSystem, ParamCircuitExpr, ParamCircuitProgram,
        },
        lang::{BaseOffsetMap, Operator},
        program::lowering::CircuitLowering,
    };

    use super::*;

    fn test_lowering(program: ParamCircuitProgram) {
        let mut lowering = CircuitLowering::new();
        let lowered_program = lowering.run(program);
        println!("{}", lowered_program);
    }

    #[test]
    fn test_reduce() {
        let mut coord_map =
            IndexCoordinateMap::from_coord_system(IndexCoordinateSystem::from_dim_list(vec![
                (String::from("i"), 2),
                (String::from("j"), 2),
            ]));

        let vector = VectorInfo {
            array: String::from("arr"),
            preprocessing: None,
            offset_map: BaseOffsetMap::new(2),
            dims: im::Vector::new(),
        };

        let ct_obj = CiphertextObject::InputVector(vector);

        coord_map.set(im::vector![0, 0], ct_obj.clone());
        coord_map.set(im::vector![0, 1], ct_obj.clone());
        coord_map.set(im::vector![1, 0], ct_obj.clone());
        coord_map.set(im::vector![1, 1], ct_obj.clone());

        let mut registry = CircuitObjectRegistry::new();

        let lit_2 = registry.register_circuit(ParamCircuitExpr::Literal(2));
        let ct = registry.register_circuit(ParamCircuitExpr::CiphertextVar(String::from("ct")));
        let reduce_vec = registry.register_circuit(ParamCircuitExpr::ReduceDim(
            String::from("j"),
            2,
            Operator::Add,
            ct,
        ));

        let circuit =
            registry.register_circuit(ParamCircuitExpr::Op(Operator::Add, reduce_vec, lit_2));

        registry
            .ct_var_values
            .insert(String::from("ct"), CircuitValue::CoordMap(coord_map));

        let circuit_program = ParamCircuitProgram {
            registry,
            expr_list: vec![(String::from("out"), vec![(String::from("i"), 2)], circuit)],
        };

        test_lowering(circuit_program);
    }
}