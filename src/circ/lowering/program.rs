use itertools::Itertools;
/// program.rs
/// instruction representation of HE programs

use pretty::RcDoc;
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display};

use crate::circ2::{*, vector_info::VectorInfo};
use crate::lang::{Operator, Extent, HEObjectName};
use crate::scheduling::{OffsetExpr, DimName};
use crate::util::NameGenerator;

pub type NodeId = usize;

#[derive(Clone, Debug)]
pub enum HEIndex {
    Var(String),
    Literal(isize),
}

impl fmt::Display for HEIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HEIndex::Var(var) =>
                write!(f, "{}", var),

            HEIndex::Literal(val) =>
                write!(f, "{}", val),
        }
    }
}

#[derive(Clone, Debug)]
pub enum HERef {
    // reference to a previous instruction's output
    Node(NodeId),

    // index to a ciphertext array (variable if no indices)
    Ciphertext(HEObjectName, Vec<HEIndex>),

    // index to a plaintext array (variable if no indices)
    Plaintext(HEObjectName, Vec<HEIndex>),
}

#[derive(Clone, Debug)]
pub enum HEOperand {
    Ref(HERef),
    Literal(isize),
}

impl fmt::Display for HEOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HEOperand::Ref(HERef::Node(i)) => {
                write!(f, "i{}", i)
            },

            HEOperand::Ref(HERef::Ciphertext(sym, index_vars)) => {
                let index_str =
                    index_vars.iter()
                    .map(|var| format!("[{}]", var))
                    .collect::<Vec<String>>()
                    .join("");
                write!(f, "{}{}", sym, index_str)
            },

            HEOperand::Ref(HERef::Plaintext(sym, index_vars)) => {
                let index_str =
                    index_vars.iter()
                    .map(|var| format!("[{}]", var))
                    .collect::<Vec<String>>()
                    .join("");
                write!(f, "{}{}", sym, index_str)
            },

            HEOperand::Literal(n) => {
                write!(f, "{}", n)
            },
        }
    }
}

#[derive(Clone, Debug)]
pub enum HEInstruction {
    Add(usize, HEOperand, HEOperand),
    Sub(usize, HEOperand, HEOperand),
    Mul(usize, HEOperand, HEOperand),
    Rot(usize, OffsetExpr, HEOperand),
}

impl fmt::Display for HEInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HEInstruction::Add(id, op1, op2) =>
                write!(f, "i{} = {} + {}", id, op1, op2),

            HEInstruction::Sub(id, op1, op2) =>
                write!(f, "i{} = {} - {}", id, op1, op2),

            HEInstruction::Mul(id, op1, op2) =>
                write!(f, "i{} = {} * {}", id, op1, op2),

            HEInstruction::Rot(id, op1, op2) =>
                write!(f, "i{} = rot {} {}", id, op1, op2),
        }
    }
}

#[derive(Clone, Debug)]
pub enum HEStatement {
    ForNode(String, usize, Vec<HEStatement>),
    DeclareVar(String, Vec<Extent>),
    SetVar(String, Vec<HEIndex>, HEOperand),
    Instruction(HEInstruction),
}

impl HEStatement {
    pub fn to_doc(&self) -> RcDoc<()> {
        match self {
            HEStatement::ForNode(dim, extent, body) => {
                let body_doc =
                    RcDoc::intersperse(
                        body.iter().map(|stmt| stmt.to_doc()),
                        RcDoc::hardline()
                    );

                RcDoc::text(format!("for {}: {} {{", dim, extent))
                .append(
                    RcDoc::hardline()
                    .append(body_doc)
                    .nest(4)
                )
                .append(RcDoc::hardline())
                .append(RcDoc::text("}"))
            },

            HEStatement::DeclareVar(var, extents) => {
                let extent_str = 
                    extents.iter()
                    .map(|i| format!("[{}]", i))
                    .collect::<Vec<String>>()
                    .join("");

                RcDoc::text(format!("var {}{}", var, extent_str))
            },

            HEStatement::SetVar(var, indices, val) => {
                let index_str = 
                    indices.iter()
                    .map(|i| format!("[{}]", i))
                    .collect::<Vec<String>>()
                    .join("");

                RcDoc::text(format!("{}{} = {}", var, index_str, val))
            },

            HEStatement::Instruction(instr) =>
                RcDoc::text(instr.to_string())
        }
    }
}

impl Display for HEStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_doc().render_fmt(80, f)
    }
}

pub struct HEProgramContext {
    vector_map: HashMap<VectorInfo, String>,
    mask_map: HashMap<MaskVector, String>,
    const_map: HashMap<isize, String>,
}

pub struct HEProgram {
    pub context: HEProgramContext,
    pub statements: Vec<HEStatement>,
}

impl HEProgram {
    pub fn to_doc(&self) -> RcDoc<()> {
        RcDoc::intersperse(
            self.context.vector_map.iter().map(|(vector, name)| {
                RcDoc::text(format!("{} = vector({})", name, vector))
            })
            ,RcDoc::hardline()
        )
        .append(
            RcDoc::intersperse(
                self.context.mask_map.iter().map(|(mask, name)| {
                    RcDoc::text(format!("{} = mask({:?})", name, mask))
                })
                ,RcDoc::hardline()
            )
        )
        .append(
            RcDoc::intersperse(
                self.context.const_map.iter().map(|(constval, name)| {
                    RcDoc::text(format!("{} = const({})", name, constval))
                })
                ,RcDoc::hardline()
            )
        )
        .append(RcDoc::hardline())
        .append(
            RcDoc::intersperse(
                self.statements.iter().map(|s| s.to_doc()),
                RcDoc::hardline()
            )
        )
    }
}

impl Display for HEProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_doc().render_fmt(80, f)
    }
}

pub struct CircuitLowering {
    name_generator: NameGenerator,
    cur_instr_id: NodeId,
}

impl CircuitLowering {
    pub fn new() -> Self {
        Self {
            name_generator: NameGenerator::new(),
            cur_instr_id: 0,
        }
    }

    fn fresh_instr_id(&mut self) -> NodeId {
        let id = self.cur_instr_id;
        self.cur_instr_id += 1;
        id
    }

    fn resolve_ciphertext_object(obj: &CiphertextObject, input: &HEProgramContext) -> HEOperand {
        match obj {
            CiphertextObject::InputVector(vector) => {
                let vector_var = input.vector_map.get(vector).unwrap().clone();
                HEOperand::Ref(HERef::Ciphertext(vector_var, vec![]))
            },

            CiphertextObject::VectorRef(array, coords) => {
                let coord_index =
                    coords.iter().map(|coord| HEIndex::Literal(*coord as isize));
                HEOperand::Ref(HERef::Ciphertext(array.clone(), Vec::from_iter(coord_index)))
            }
        }
    }

    // TODO finish this
    fn compute_coord_relationship(
        index_vars: Vec<DimName>,
        coord_val_map: HashMap<IndexCoord, IndexCoord>
    ) -> Option<Vec<HEIndex>> {
        None
    }

    fn inline_ciphertext_object(
        circval: &CircuitValue<CiphertextObject>,
        input: &HEProgramContext
    ) -> Option<HEOperand> {
        match circval {
            CircuitValue::CoordMap(coord_map) => {
                let mut vector_var_set: HashSet<String> = HashSet::new();
                let mut coord_val_map: HashMap<IndexCoord, IndexCoord> = HashMap::new();

                for (coord, obj_opt) in coord_map.value_iter() {
                    let obj = obj_opt.unwrap();
                    match obj {
                        CiphertextObject::InputVector(vector) => {
                            vector_var_set.insert(
                                input.vector_map.get(vector).unwrap().clone()
                            );
                        },

                        CiphertextObject::VectorRef(ref_vector, ref_coord) => {
                            vector_var_set.insert(ref_vector.clone());
                            coord_val_map.insert(coord, ref_coord.clone());
                        }
                    }
                }

                if vector_var_set.len() == 1 {
                    let vector_var = vector_var_set.into_iter().next().unwrap();
                    if coord_val_map.len() == 0 {
                        Some(HEOperand::Ref(HERef::Ciphertext(vector_var, vec![])))

                    } else {
                        let index_opt =
                            Self::compute_coord_relationship(
                                coord_map.index_vars(),
                                coord_val_map,
                            );

                        if let Some(index) = index_opt {
                            Some(HEOperand::Ref(HERef::Ciphertext(vector_var, index)))

                        } else {
                            None
                        }
                    }

                } else {
                    None
                }
            },

            CircuitValue::Single(obj) => {
                Some(Self::resolve_ciphertext_object(obj, input))
            }
        }
    }

    fn inline_plaintext_object(
        circval: &CircuitValue<PlaintextObject>,
        input: &HEProgramContext
    ) -> Option<HEOperand> {
        match circval {
            CircuitValue::CoordMap(coord_map) => {
                let mut vector_var_set: HashSet<String> = HashSet::new();

                for (_, obj_opt) in coord_map.value_iter() {
                    let obj = obj_opt.unwrap();
                    match obj {
                        PlaintextObject::Mask(mask) => {
                            vector_var_set.insert(
                                input.mask_map.get(mask).unwrap().clone()
                            );
                        },

                        PlaintextObject::Const(constval) => {
                            vector_var_set.insert(
                                input.const_map.get(constval).unwrap().clone()
                            );
                        }
                    }
                }

                if vector_var_set.len() == 1 {
                    let vector_var = vector_var_set.into_iter().next().unwrap();
                    Some(HEOperand::Ref(HERef::Plaintext(vector_var, vec![])))

                } else {
                    None
                }
            },

            CircuitValue::Single(obj) => {
                Some(Self::resolve_plaintext_object(obj, input))
            }
        }
    }

    fn resolve_plaintext_object(obj: &PlaintextObject, input: &HEProgramContext) -> HEOperand {
        match obj {
            PlaintextObject::Const(val) => {
                let const_var = input.const_map.get(val).unwrap().clone();
                HEOperand::Ref(HERef::Plaintext(const_var, vec![]))
            },

            PlaintextObject::Mask(mask) => {
                let mask_var = input.mask_map.get(mask).unwrap().clone();
                HEOperand::Ref(HERef::Plaintext(mask_var, vec![]))
            }
        }
    }

    fn resolve_offset(offset: &isize, _input: &HEProgramContext) -> HEOperand {
        HEOperand::Literal(*offset)
    }

    fn gen_program_context(&mut self, registry: &CircuitRegistry) -> HEProgramContext {
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

        HEProgramContext { vector_map, mask_map, const_map, }
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
                    let coord_index =
                        Vec::from_iter(
                            coords.iter()
                            .map(|coord| HEIndex::Literal(*coord as isize))
                        );

                    statements.push(
                        HEStatement::SetVar(var.clone(), coord_index, operand)
                    );
                }
            },

            CircuitValue::Single(obj) => {
                let operand = f(obj, &input);
                statements.push(
                    HEStatement::SetVar(var, vec![], operand)
                );
            }
        }
    }

    pub fn lower(&mut self, program: ParamCircuitProgram) -> HEProgram {
        let mut statements: Vec<HEStatement> = Vec::new();
        let context = self.gen_program_context(&program.registry);

        let mut ct_inline_map: HashMap<VarName, HEOperand> = HashMap::new();
        let mut pt_inline_map: HashMap<VarName, HEOperand> = HashMap::new();

        // process statements
        for (array, dims, circuit) in program.circuit_list {
            // preamble: allocate arrays referenced in the circuit expr
            for ct_var in circuit.ciphertext_vars() {
                let circval = program.registry.get_ct_var_value(&ct_var);

                if let Some(operand) = Self::inline_ciphertext_object(circval, &context) {
                    ct_inline_map.insert(ct_var, operand);

                } else {
                    CircuitLowering::process_circuit_val(
                        program.registry.get_ct_var_value(&ct_var),
                        ct_var,
                        &context,
                        CircuitLowering::resolve_ciphertext_object,
                        &mut statements,
                    );
                }
            }

            for pt_var in circuit.plaintext_vars() {
                let circval = program.registry.get_pt_var_value(&pt_var);

                if let Some(operand) = Self::inline_plaintext_object(circval, &context) {
                    pt_inline_map.insert(pt_var, operand);

                } else {
                    CircuitLowering::process_circuit_val(
                        program.registry.get_pt_var_value(&pt_var),
                        pt_var,
                        &context,
                        CircuitLowering::resolve_plaintext_object,
                        &mut statements,
                    );
                }
            }

            for offset_fvar in circuit.offset_fvars() {
                CircuitLowering::process_circuit_val(
                    program.registry.get_offset_fvar_value(&offset_fvar),
                    offset_fvar,
                    &context,
                    CircuitLowering::resolve_offset,
                    &mut statements,
                );
            }

            let dim_vars: Vec<String> = 
                dims.iter().map(|(var, _)| var.clone()).collect();

            let dim_extents = 
                dims.iter().map(|(_, extent)| *extent).collect();

            // generate statements for array expr
            // first, declare array
            statements.push(
                HEStatement::DeclareVar(array.clone(), dim_extents)
            );

            // generate statements in the array expr's body
            let mut body_statements: Vec<HEStatement> = Vec::new();
            let array_id =
                self.gen_expr_instrs_recur(
                    circuit,
                    &dims, 
                    &ct_inline_map,
                    &pt_inline_map,
                    &mut HashMap::new(), 
                    &mut body_statements,
                );

            // set the array's value to the body statement's computed value
            body_statements.push(
                HEStatement::SetVar(
                    array,
                    dim_vars.into_iter().map(|var| HEIndex::Var(var)).collect(),
                    HEOperand::Ref(HERef::Node(array_id))
                )
            );

            let mut dims_reversed = dims.clone();
            dims_reversed.reverse();

            // wrap the body in a nest of for loops
            let array_statement = 
                dims_reversed.into_iter()
                .fold(body_statements, |acc, (dim, extent)| {
                    vec![HEStatement::ForNode(dim, extent, acc)]
                }).pop().unwrap();

            statements.push(array_statement);
        }

        HEProgram { context, statements }
    }

    pub fn gen_expr_instrs_recur(
        &mut self,
        expr: ParamCircuitExpr,
        indices: &Vec<(String, usize)>,
        ct_inline_map: &HashMap<VarName, HEOperand>,
        pt_inline_map: &HashMap<VarName, HEOperand>,
        operand_map: &mut HashMap<usize, HEOperand>,
        stmts: &mut Vec<HEStatement>
    ) -> NodeId {
        match expr {
            ParamCircuitExpr::CiphertextVar(var) => {
                let id = self.fresh_instr_id();
                if let Some(operand) = ct_inline_map.get(&var) {
                    operand_map.insert(id, operand.clone());

                }  else {
                    let index_vars =
                        indices.iter()
                        .map(|(var, _)| HEIndex::Var(var.clone()))
                        .collect();
                    let operand = 
                        HEOperand::Ref(HERef::Ciphertext(var, index_vars));
                    operand_map.insert(id, operand);
                }

                id
            },

            ParamCircuitExpr::PlaintextVar(var) => {
                let id = self.fresh_instr_id();
                if let Some(operand) = pt_inline_map.get(&var) {
                    operand_map.insert(id, operand.clone());

                } else {
                    let index_vars =
                        indices.iter()
                        .map(|(var, _)| HEIndex::Var(var.clone()))
                        .collect();
                    let operand =
                        HEOperand::Ref(HERef::Plaintext(var, index_vars));
                    operand_map.insert(id, operand);
                }

                id
            },

            ParamCircuitExpr::Literal(val) => {
                let id = self.fresh_instr_id();
                let operand = HEOperand::Literal(val);
                operand_map.insert(id, operand);
                id
            },

            ParamCircuitExpr::Op(op, expr1, expr2) => {
                let id1 =
                    self.gen_expr_instrs_recur(
                        *expr1,
                        indices, 
                        ct_inline_map,
                        pt_inline_map,
                        operand_map,
                        stmts
                    );
                    
                let id2 =
                    self.gen_expr_instrs_recur(
                        *expr2,
                        indices, 
                        ct_inline_map,
                        pt_inline_map,
                        operand_map,
                        stmts
                    );

                let operand1 = operand_map.get(&id1).unwrap().clone();
                let operand2 = operand_map.get(&id2).unwrap().clone();

                let id = self.fresh_instr_id();
                let instr =
                    match op {
                        Operator::Add => HEInstruction::Add(id, operand1, operand2),
                        Operator::Sub => HEInstruction::Sub(id, operand1, operand2),
                        Operator::Mul => HEInstruction::Mul(id, operand1, operand2),
                    };

                stmts.push(HEStatement::Instruction(instr));
                operand_map.insert(id, HEOperand::Ref(HERef::Node(id)));
                id
            },

            ParamCircuitExpr::Rotate(steps, body) => {
                let body_id =
                    self.gen_expr_instrs_recur(
                        *body,
                        indices,
                        ct_inline_map,
                        pt_inline_map,
                        operand_map,
                        stmts
                    );

                let id = self.fresh_instr_id();

                let body_operand = operand_map.get(&body_id).unwrap().clone();
                stmts.push(
                    HEStatement::Instruction(HEInstruction::Rot(id, *steps, body_operand))
                );
                operand_map.insert(id, HEOperand::Ref(HERef::Node(id)));

                id
            },

            ParamCircuitExpr::ReduceVectors(dim, extent, op, body) => {
                let mut body_indices = indices.clone();
                body_indices.push((dim.clone(), extent));

                let mut body_stmts: Vec<HEStatement> = Vec::new();

                let body_id =
                    self.gen_expr_instrs_recur(
                        *body,
                        &body_indices,
                        ct_inline_map,
                        pt_inline_map,
                        operand_map,
                        &mut body_stmts,
                    );

                let body_operand = operand_map.get(&body_id).unwrap().clone();

                let reduce_var = self.name_generator.get_fresh_name("reduce");

                let reduce_var_ref = 
                    HEOperand::Ref(HERef::Ciphertext(reduce_var.clone(), vec![]));

                let reduce_id = self.fresh_instr_id();

                let reduce_stmt = 
                    match op {
                        Operator::Add =>
                            HEInstruction::Add(reduce_id, reduce_var_ref.clone(), body_operand),

                        Operator::Sub =>
                            HEInstruction::Sub(reduce_id, reduce_var_ref.clone(), body_operand),

                        Operator::Mul => 
                            HEInstruction::Mul(reduce_id, reduce_var_ref.clone(), body_operand),
                    };

                body_stmts.push(HEStatement::Instruction(reduce_stmt));
                body_stmts.push(
                    HEStatement::SetVar(
                        reduce_var.clone(),
                        vec![],
                        HEOperand::Ref(HERef::Node(reduce_id)),
                    )
                );

                stmts.extend([
                    HEStatement::DeclareVar(reduce_var.clone(), vec![]),
                    HEStatement::ForNode(dim, extent, body_stmts),
                ]);
                
                let id = self.fresh_instr_id();
                operand_map.insert(id, reduce_var_ref);

                id
            },
        }
    }
}

/*
pub struct HEProgram {
    pub(crate) instrs: Vec<HEInstruction>,
}

impl HEProgram {
    /// calculate the multiplicative depth of the program.
    pub fn get_muldepth(&self) -> usize {
        let mut max_depth: usize = 0;
        let mut depth_list: Vec<usize> = vec![];

        let get_opdepth = |dlist: &Vec<usize>, op: &HEOperand| -> usize {
            match op {
                HEOperand::Ref(HERef::Node(r)) => dlist[*r],
                HEOperand::Ref(HERef::Ciphertext(_, _)) => 0,
                HEOperand::Ref(HERef::Plaintext(_, _)) => 0,
                HEOperand::Literal(_) => 0,
            }
        };

        for instr in self.instrs.iter() {
            let dlist: &Vec<usize> = &depth_list;
            let depth: usize =
                match instr {
                    HEInstruction::Add { id: _, op1, op2 } => {
                        let op1_depth = get_opdepth(dlist, op1);
                        let op2_depth: usize = get_opdepth(dlist, op2);
                        max(op1_depth, op2_depth)
                    },
                    HEInstruction::Sub { id: _, op1, op2 } => {
                        let op1_depth = get_opdepth(dlist, op1);
                        let op2_depth: usize = get_opdepth(dlist, op2);
                        max(op1_depth, op2_depth)
                    },
                    HEInstruction::Mul { id: _, op1, op2 } => {
                        let op1_depth = get_opdepth(dlist, op1);
                        let op2_depth: usize = get_opdepth(dlist, op2);

                        match (op1, op2) {
                            (HEOperand::Literal(_), _) | (_, HEOperand::Literal(_)) =>
                                max(op1_depth, op2_depth),

                            _ => max(op1_depth, op2_depth) + 1
                        }
                    },
                    HEInstruction::Rot { id: _, op1, op2 } => {
                        let op1_depth = get_opdepth(dlist, op1);
                        let op2_depth: usize = get_opdepth(dlist, op2);
                        max(op1_depth, op2_depth)
                    },
                };

            if depth > max_depth {
                max_depth = depth;
            }
            depth_list.push(depth);
        }
        max_depth
    }

    /// calculate the latency of a program
    pub fn get_latency(&self, model: &HELatencyModel) -> f64 {
        let mut latency: f64 = 0.0;
        for instr in self.instrs.iter() {
            match instr {
                HEInstruction::Add { id: _, op1, op2 } => {
                    match (op1, op2) {
                        (HEOperand::Literal(_), _) | (_, HEOperand::Literal(_)) =>  {
                            latency += model.add_plain;
                        },

                        _ => {
                            latency += model.add
                        }
                    }
                },

                HEInstruction::Sub { id: _, op1, op2 } => {
                    match (op1, op2) {
                        (HEOperand::Literal(_), _) | (_, HEOperand::Literal(_)) =>  {
                            latency += model.add_plain;
                        },

                        _ => {
                            latency += model.add;
                        }
                    }
                },

                HEInstruction::Mul { id: _, op1, op2 } => {
                    match (op1, op2) {
                        (HEOperand::Literal(_), _) | (_, HEOperand::Literal(_)) =>  {
                            latency += model.mul_plain;
                        },

                        _ => {
                            latency += model.mul;
                        }
                    }
                },
                
                HEInstruction::Rot { id: _, op1: _, op2: _ } => {
                    latency += model.rot;
                }
            }
        }
        latency
    }

    /// get the symbols used in this program.
    pub fn get_ciphertext_symbols(&self) -> HashSet<String> {
        let mut symset = HashSet::new();

        for instr in self.instrs.iter() {
            for op in instr.get_operands() {
                if let HEOperand::Ref(HERef::Ciphertext(sym, index)) = op{
                    symset.insert(sym.to_string());
                }
            }
        }

        symset
    }

    /// get the plaintext symbols used in this program.
    pub fn get_plaintext_symbols(&self) -> HashSet<String> {
        let mut symset = HashSet::new();

        for instr in self.instrs.iter() {
            for op in instr.get_operands() {
                if let HEOperand::Ref(HERef::Plaintext(sym, index)) = op {
                    symset.insert(sym.to_string());
                }
            }
        }

        symset
    }

    /// generate a random sym store for this program
    /*
    pub(crate) fn gen_sym_store(&self, vec_size: usize, range: RangeInclusive<isize>) -> HESymStore {
        let symbols = self.get_ciphertext_symbols();
        let mut sym_store: HESymStore = HashMap::new();
        let mut rng = rand::thread_rng();

        for symbol in symbols {
            let new_vec: Vec<isize> = (0..vec_size)
                .into_iter()
                .map(|_| rng.gen_range(range.clone()))
                .collect();

            sym_store.insert(symbol.to_string(), HEValue::HEVector(new_vec));
        }

        sym_store
    }
    */

    /// compute the required vectors at every program point
    /// this is used in the lowering pass for computing when relinearizations
    /// and in-place operations can be used
    pub(crate) fn analyze_use(&self) -> Vec<HashSet<usize>> {
        let mut uses: Vec<HashSet<usize>> = Vec::new();
        uses.push(HashSet::new());

        for instr in self.instrs.iter().rev() {
            let mut new_use: HashSet<usize> = uses.last().unwrap().clone();
            match instr {
                HEInstruction::Add { id: _, op1, op2 } |
                HEInstruction::Sub { id: _, op1, op2 } |
                HEInstruction::Mul { id: _, op1, op2 } |
                HEInstruction::Rot { id: _, op1, op2 } => {
                    if let HEOperand::Ref(HERef::Node(nr)) = op1 {
                        new_use.insert(*nr);
                    }

                    if let HEOperand::Ref(HERef::Node(nr)) = op2 {
                        new_use.insert(*nr);
                    }
                }
            }

            uses.push(new_use);
        }

        uses.reverse();
        uses
    }
}

impl From<&RecExpr<HEOptCircuit>> for HEProgram {
    fn from(expr: &RecExpr<HEOptCircuit>) -> Self {
        let mut node_map: HashMap<Id, HEOperand> = HashMap::new();
        let mut program: HEProgram = HEProgram { instrs: Vec::new() };
        let mut cur_instr: NodeId = 0;

        let mut op_processor =
            |nmap: &mut HashMap<Id, HEOperand>, id: Id,
            ctor: fn(usize, HEOperand, HEOperand) -> HEInstruction,
            id_op1: &Id, id_op2: &Id|
        {
            let op1 = &nmap[id_op1];
            let op2 = &nmap[id_op2];

            program.instrs.push(ctor(cur_instr, op1.clone(), op2.clone()));
            nmap.insert(id, HEOperand::Ref(HERef::Node(cur_instr)));

            cur_instr += 1;
        };

        for (i, node) in expr.as_ref().iter().enumerate() {
            let id = Id::from(i);
            match node {
                HEOptCircuit::Num(n) => {
                    node_map.insert(id, HEOperand::Literal(*n));
                }

                HEOptCircuit::CiphertextRef(sym) => {
                    node_map.insert(id, HEOperand::Ref(HERef::Ciphertext(sym.to_string(), vec![])));
                }

                HEOptCircuit::PlaintextRef(sym) => {
                    node_map.insert(id, HEOperand::Ref(HERef::Plaintext(sym.to_string(), vec![])));
                }

                HEOptCircuit::Add([id1, id2]) => {
                    op_processor(
                        &mut node_map, id, 
                        |index, op1, op2| {
                            HEInstruction::Add { id: index, op1, op2 }
                        },
                        id1, id2);
                }
                
                HEOptCircuit::Sub([id1, id2]) => {
                    op_processor(
                        &mut node_map, id, 
                        |index, op1, op2| {
                            HEInstruction::Sub { id: index, op1, op2 }
                        },
                        id1, id2);
                }

                HEOptCircuit::Mul([id1, id2]) => {
                    op_processor(
                        &mut node_map, id, 
                        |index, op1, op2| HEInstruction::Mul { id: index, op1, op2 }, 
                        id1, id2);
                }

                HEOptCircuit::Rot([id1, id2]) => {
                    op_processor(
                        &mut node_map, id, 
                        |index, op1, op2| HEInstruction::Rot { id: index, op1, op2 }, 
                        id1, id2);
                }
            }
        }

        program
    }
}

impl fmt::Display for HEProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.instrs
            .iter()
            .enumerate()
            .try_for_each(|(i, instr)| write!(f, "let i{} = {}\n", i + 1, instr))
    }
}

#[derive(PartialEq, Eq, Clone)]
pub(crate) enum HEValue {
    HEScalar(isize),
    HEVector(Vec<isize>),
}

impl fmt::Display for HEValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HEValue::HEScalar(s) => write!(f, "{}", s),
            HEValue::HEVector(v) => write!(f, "{:?}", v),
        }
    }
}

type HESymStore = HashMap<String, HEValue>;
type HERefStore = HashMap<NodeId, HEValue>;

/// interpret HEPrograms against a given store
struct HEProgramInterpreter {
    sym_store: HESymStore,
    ref_store: HERefStore,
    vec_size: usize,
}

impl HEProgramInterpreter {
    fn new(sym_store: HESymStore, ref_store: HERefStore, vec_size: usize) -> Self {
        HEProgramInterpreter { sym_store, ref_store, vec_size }
    }

    fn interp_operand(&self, op: &HEOperand) -> HEValue {
        match op {
            HEOperand::Ref(HERef::Node(ref_i)) => self.ref_store[ref_i].clone(),

            HEOperand::Ref(HERef::Ciphertext(sym)) => self.sym_store[sym.as_str()].clone(),

            HEOperand::Ref(HERef::Plaintext(sym)) => self.sym_store[sym.as_str()].clone(),

            HEOperand::ConstNum(n) => HEValue::HEScalar(*n),
        }
    }

    fn interp_instr(&self, instr: &HEInstruction) -> HEValue {
        let exec_binop = |op1: &HEOperand, op2: &HEOperand, f: fn(&isize, &isize) -> isize| -> HEValue {
            let val1 = self.interp_operand(op1);
            let val2 = self.interp_operand(op2);
            match (val1, val2) {
                (HEValue::HEScalar(s1), HEValue::HEScalar(s2)) => HEValue::HEScalar(f(&s1, &s2)),

                (HEValue::HEScalar(s1), HEValue::HEVector(v2)) => {
                    let new_vec = v2.iter().map(|x| f(x, &s1)).collect();
                    HEValue::HEVector(new_vec)
                }

                (HEValue::HEVector(v1), HEValue::HEScalar(s2)) => {
                    let new_vec = v1.iter().map(|x| f(x, &s2)).collect();
                    HEValue::HEVector(new_vec)
                }

                (HEValue::HEVector(v1), HEValue::HEVector(v2)) => {
                    let new_vec = v1.iter().zip(v2).map(|(x1, x2)| f(x1, &x2)).collect();
                    HEValue::HEVector(new_vec)
                }
            }
        };

        match instr {
            HEInstruction::Add { id: _, op1, op2 }=>
                exec_binop(op1, op2, |x1, x2| x1 + x2),

            HEInstruction::Sub { id: _, op1, op2 }=>
                exec_binop(op1, op2, |x1, x2| x1 + x2),

            HEInstruction::Mul { id: _, op1, op2 }=>
                exec_binop(op1, op2, |x1, x2| x1 * x2),

            HEInstruction::Rot { id: _, op1, op2 }=> {
                let val1 = self.interp_operand(op1);
                let val2 = self.interp_operand(op2);
                match (val1, val2) {
                    (HEValue::HEVector(v1), HEValue::HEScalar(s2)) => {
                        let rot_val = s2 % (self.vec_size as isize);
                        let mut new_vec: Vec<isize> = v1;
                        if rot_val < 0 {
                            new_vec.rotate_left((-rot_val) as usize)
                        } else {
                            new_vec.rotate_right(rot_val as usize)
                        }

                        HEValue::HEVector(new_vec)
                    }

                    _ => panic!("Rotate must have vector has 1st operand and scalar as 2nd operand"),
                }
            }
        }
    }

    pub fn interp_program(&self, program: &HEProgram) -> Option<HEValue> {
        let mut ref_store: HERefStore = HashMap::new();

        let mut last_instr = None;
        for (i, instr) in program.instrs.iter().enumerate() {
            let val = self.interp_instr(instr);
            ref_store.insert(i, val);
            last_instr = Some(i);
        }

        last_instr.and_then(|i| ref_store.remove(&i))
    }
}
*/

#[cfg(test)]
mod tests {
    use crate::{
        lang::BaseOffsetMap,
        circ2::{IndexCoordinateMap, IndexCoordinateSystem}
    };

    use super::*;

    fn test_lowering(program: ParamCircuitProgram) {
        let mut lowering = CircuitLowering::new();
        let lowered_program = lowering.lower(program);
        println!("{}", lowered_program);
    }

    #[test]
    fn test_reduce() {
        let circuit =
            ParamCircuitExpr::Op(
                Operator::Add,
                Box::new(ParamCircuitExpr::ReduceVectors(
                    String::from("j"), 2, Operator::Add,
                    Box::new(ParamCircuitExpr::CiphertextVar(String::from("ct"))),
                )),
                Box::new(ParamCircuitExpr::Literal(2))
            );

        let mut coord_map =
            IndexCoordinateMap::from_coord_system(
                IndexCoordinateSystem::from_dim_list(
                    vec![(String::from("i"), 2), (String::from("j"), 2)]
                )
            );

        let vector =
            VectorInfo {
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

        let mut registry = CircuitRegistry::new();
        registry.ct_var_values.insert(
            String::from("ct"),
            CircuitValue::CoordMap(coord_map)
        );

        let circuit_program =
            ParamCircuitProgram {
                registry,
                circuit_list: vec![
                    (String::from("out"),
                    vec![(String::from("i"), 2)],
                    circuit)
                ]
            };

        test_lowering(circuit_program);
    }
}