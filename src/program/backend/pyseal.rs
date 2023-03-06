/// lowered_program.rs
/// lowered program for generating output programs from templates

use std::collections::{HashMap, HashSet};

use crate::{program::*, lang::Operator};

enum SEALOpType {
    // HE operations
    Add,
    AddInplace,
    AddPlain,
    AddPlainInplace,
    AddNative,
    AddNativeInplace,

    Sub,
    SubInplace,
    SubPlain,
    SubPlainInplace,
    SubNative,
    SubNativeInplace,

    Mul,
    MulInplace,
    MulPlain,
    MulPlainInplace,
    MulNative,
    MulNativeInplace,
    
    Rot,
    RotInplace,
    RotNative,
    RotNativeInplace,

    // low-level HE operations
    Encode,
    RelinearizeInplace,

    Assign,

    // HE datatypes
    DeclareNativeArray,
    DeclareCiphertextArray,

    DeclareMask,
    DeclareConst,
    ServerGetCiphertextInputVector,
    ServerDeclarePlaintextInputVector,
    ServerRecv,
}

impl SEALOpType {
    pub fn to_doc(&self) -> RcDoc<()> {
        match self {
            SEALOpType::Add => RcDoc::text("seal.add"),
            SEALOpType::AddInplace => RcDoc::text("seal.add_inplace"),
            SEALOpType::AddPlain => RcDoc::text("seal.add_plain"),
            SEALOpType::AddPlainInplace => RcDoc::text("seal.add_plain_inplace"),
            SEALOpType::AddNative => RcDoc::text("seal.add_native"),
            SEALOpType::AddNativeInplace => RcDoc::text("seal.add_native_inplace"),

            SEALOpType::Sub => RcDoc::text("seal.sub"),
            SEALOpType::SubInplace => RcDoc::text("seal.sub_inplace"),
            SEALOpType::SubPlain => RcDoc::text("seal.sub_plain"),
            SEALOpType::SubPlainInplace => RcDoc::text("seal.sub_plain_inplace"),
            SEALOpType::SubNative => RcDoc::text("seal.sub_native"),
            SEALOpType::SubNativeInplace => RcDoc::text("seal.sub_native_inplace"),

            SEALOpType::Mul => RcDoc::text("seal.mul"),
            SEALOpType::MulInplace => RcDoc::text("seal.mul_inplace"),
            SEALOpType::MulPlain => RcDoc::text("seal.mul_plain"),
            SEALOpType::MulPlainInplace => RcDoc::text("seal.mul_plain_inplace"),
            SEALOpType::MulNative => RcDoc::text("seal.mul_native"),
            SEALOpType::MulNativeInplace => RcDoc::text("seal.mul_native_inplace"),

            SEALOpType::Rot => RcDoc::text("seal.rot"),
            SEALOpType::RotInplace => RcDoc::text("seal.rot_inplace"),
            SEALOpType::RotNative => RcDoc::text("seal.rot_native"),
            SEALOpType::RotNativeInplace => RcDoc::text("seal.rot_native_inplace"),

            SEALOpType::Encode => RcDoc::text("seal.encode"),
            SEALOpType::RelinearizeInplace => RcDoc::text("seal.relinearize_inplace"),

            SEALOpType::Assign => RcDoc::text("seal.set"),

            SEALOpType::DeclareNativeArray => RcDoc::text("NativeArray"),
            SEALOpType::DeclareCiphertextArray => RcDoc::text("CiphertextArray"),

            SEALOpType::DeclareMask => RcDoc::text("Mask"),
            SEALOpType::DeclareConst => RcDoc::text("Const"),
            SEALOpType::ServerGetCiphertextInputVector => RcDoc::text("seal.get_ct_input"),
            SEALOpType::ServerDeclarePlaintextInputVector => RcDoc::text("PlaintextInput"),
            SEALOpType::ServerRecv => RcDoc::text("seal.server_recv"),
        }
    }
}

enum SEALInstruction {
    Op(SEALOpType, String, Vec<String>),
    OpInplace(SEALOpType, Vec<String>),
}

impl SEALInstruction {
    pub fn to_doc(&self) -> RcDoc<()> {
        match self {
            SEALInstruction::Op(optype, id, ops) => {
                let ops_str = ops.join(", ");
                RcDoc::text(format!("{} = ", id))
                .append(optype.to_doc())
                .append(RcDoc::text(format!("({})", ops_str)))
            },

            SEALInstruction::OpInplace(optype, ops) => {
                let ops_str = ops.join(", ");
                optype.to_doc()
                .append(RcDoc::text(format!("({})", ops_str)))
            },
        }
    }
}

enum SEALStatement {
    Instruction(SEALInstruction),
    ForNode(String, usize, Vec<SEALStatement>),
}

impl SEALStatement {
    pub fn to_doc(&self) -> RcDoc<()> {
        match self {
            SEALStatement::Instruction(instr) => {
                instr.to_doc()
            },

            SEALStatement::ForNode(dim, extent, body) => {
                let body_doc =
                    RcDoc::intersperse(
                        body.iter().map(|stmt| stmt.to_doc()),
                        RcDoc::hardline()
                    );

                RcDoc::text(format!("for {} in range({}):", dim, extent))
                    .append(RcDoc::hardline().append(body_doc).nest(4))
                    .append(RcDoc::hardline())
            }
        }
    }
}

struct SEALProgram {
    client_code: Vec<SEALStatement>,
    server_code: Vec<SEALStatement>,
}

impl SEALProgram {
    pub fn to_doc(&self) -> RcDoc<()> {
        let client_doc =
            if self.client_code.len() > 0 {
                RcDoc::intersperse(
                    self.client_code.iter().map(|stmt| {
                        stmt.to_doc()
                    }),
                    RcDoc::hardline()
                ) 

            } else {
                RcDoc::text("pass")
            };

        let server_doc =
            if self.server_code.len() > 0 {
                RcDoc::intersperse(
                    self.server_code.iter().map(|stmt| {
                        stmt.to_doc()
                    }),
                    RcDoc::hardline()
                )

            } else {
                RcDoc::text("pass")
            };

        RcDoc::text("def client(seal):")
        .append(
            RcDoc::hardline().append(client_doc).nest(4)
        )
        .append(RcDoc::hardline())
        .append(RcDoc::hardline())
        .append(RcDoc::text("def server(seal):"))
        .append(
            RcDoc::hardline().append(server_doc).nest(4)
        )
        .append(RcDoc::hardline())
    }
}

impl Display for SEALProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_doc().render_fmt(80, f)
    }
}

// backwards analysis to determine if an instruction is live
// at a particular program point
struct UseAnalysis(HashMap<InstructionId, HashSet<InstructionId>>);

impl UseAnalysis {
    fn get_atomic_statements(stmt: &HEStatement) -> Vec<&HEStatement> {
        match stmt {
            HEStatement::ForNode(_, _, body) => {
                body.iter().rev().flat_map(|bstmt| {
                    Self::get_atomic_statements(bstmt)
                })
                .collect()
            },

            HEStatement::DeclareVar(_, _, _) |
            HEStatement::AssignVar(_, _, _) |
            HEStatement::Encode(_, _) |
            HEStatement::Instruction(_) => {
                vec![stmt]
            },
        }
    }

    pub fn can_reuse(&self, id: InstructionId, he_ref: &HERef) -> bool {
        match he_ref {
            HERef::Instruction(ref_id) => {
                let use_map = self.0.get(&id).unwrap();
                !use_map.contains(ref_id)
            },

            HERef::Array(_, _) => false,
        }
    }

    fn _analyze(mut self, program: &HEProgram) -> Self {
        let atomic_stmts: Vec<&HEStatement> =
            program.statements.iter().rev().flat_map(|stmt| {
                Self::get_atomic_statements(stmt)
            })
            .collect();

        let mut out_set: HashSet<InstructionId> = HashSet::new();
        for stmt in atomic_stmts {
            match stmt {
                HEStatement::DeclareVar(_, _, _) | HEStatement::Encode(_, _) => {},

                HEStatement::ForNode(_, _, _) =>
                    unreachable!(),

                HEStatement::AssignVar(_, _, op) => {
                    if let HEOperand::Ref(HERef::Instruction(instr)) = op {
                        out_set.insert(*instr);
                    }
                },

                HEStatement::Instruction(instr) => {
                    self.0.insert(instr.get_id(), out_set.clone());
                    out_set.extend(instr.get_instr_refs());
                },
            }
        }

        self
    }

    pub fn analyze(program: &HEProgram) -> Self {
        Self(HashMap::new())._analyze(program)
    }
}

struct SEALBackend {
    use_analysis: UseAnalysis,
    program: HEProgram,
}

impl SEALBackend {
    fn new(program: HEProgram) -> Self {
        let use_analysis = UseAnalysis::analyze(&program);
        Self { use_analysis, program }
    }

    fn array_ref<T: Display>(array: ArrayName, indices: Vec<T>) -> String {
        let index_str =
            indices.into_iter()
            .map(|i| format!("[{}]", i.to_string()))
            .collect::<Vec<String>>()
            .join("");

        format!("{}{}", array, index_str)
    }

    fn vec<T: Display>(vec: Vec<T>) -> String {
        format!(
            "[{}]",
            vec.into_iter()
            .map(|x| x.to_string())
            .collect::<Vec<String>>()
            .join(", ")
        )
    }

    fn instruction(id: InstructionId) -> String {
        format!("instr{}", id)
    }

    fn he_ref(heref: HERef) -> String {
        match heref {
            HERef::Instruction(i) =>
                Self::instruction(i),

            HERef::Array(array, indices) =>
                format!("{}.get({})", array, Self::vec(indices)),
        }
    }

    fn operand(op: HEOperand) -> String {
        match op {
            HEOperand::Ref(heref) => Self::he_ref(heref),

            HEOperand::Literal(lit) => lit.to_string(),
        }
    }

    pub fn get_binop(
        op: Operator,
        optype: HEInstructionType,
        inplace: bool,
        id: InstructionId,
        ref1: HERef,
        ref2: HERef,
    ) -> (SEALInstruction, String) {
        if inplace {
            let seal_optype = 
                match (op, optype) {
                    (Operator::Add, HEInstructionType::Native) =>
                        SEALOpType::AddNativeInplace,

                    (Operator::Add, HEInstructionType::CipherPlain) =>
                        SEALOpType::AddPlainInplace,

                    (Operator::Add, HEInstructionType::CipherCipher) =>
                        SEALOpType::AddInplace,

                    (Operator::Sub, HEInstructionType::Native) =>
                        SEALOpType::SubNativeInplace,

                    (Operator::Sub, HEInstructionType::CipherPlain) =>
                        SEALOpType::SubPlainInplace,

                    (Operator::Sub, HEInstructionType::CipherCipher) =>
                        SEALOpType::SubInplace,

                    (Operator::Mul, HEInstructionType::Native) =>
                        SEALOpType::MulNativeInplace,

                    (Operator::Mul, HEInstructionType::CipherPlain) =>
                        SEALOpType::MulPlainInplace,

                    (Operator::Mul, HEInstructionType::CipherCipher) =>
                        SEALOpType::MulInplace,
                };

            let ref1_str = Self::he_ref(ref1);
            let instr = 
                SEALInstruction::OpInplace(
                    seal_optype,
                    vec![ref1_str.clone(), Self::he_ref(ref2)],
                );

            (instr, ref1_str)

        } else {
            let seal_optype = 
                match (op, optype) {
                    (Operator::Add, HEInstructionType::Native) =>
                        SEALOpType::AddNative,

                    (Operator::Add, HEInstructionType::CipherPlain) =>
                        SEALOpType::AddPlain,

                    (Operator::Add, HEInstructionType::CipherCipher) =>
                        SEALOpType::Add,

                    (Operator::Sub, HEInstructionType::Native) =>
                        SEALOpType::SubNative,

                    (Operator::Sub, HEInstructionType::CipherPlain) =>
                        SEALOpType::SubPlain,

                    (Operator::Sub, HEInstructionType::CipherCipher) =>
                        SEALOpType::Sub,

                    (Operator::Mul, HEInstructionType::Native) =>
                        SEALOpType::MulNative,

                    (Operator::Mul, HEInstructionType::CipherPlain) =>
                        SEALOpType::MulPlain,

                    (Operator::Mul, HEInstructionType::CipherCipher) =>
                        SEALOpType::Mul,
                };

            let id_str = Self::instruction(id);
            let instr = 
                SEALInstruction::Op(
                    seal_optype,
                    id_str.clone(),
                    vec![Self::he_ref(ref1), Self::he_ref(ref2)]
                );
            
            (instr, id_str)
        }
    }

    fn offset(offset: OffsetExpr) -> String {
        match offset {
            OffsetExpr::Add(op1, op2) => {
                let str1 = Self::offset(*op1);
                let str2 = Self::offset(*op2);
                format!("({} + {})", str1, str2)
            }

            OffsetExpr::Mul(op1, op2) => {
                let str1 = Self::offset(*op1);
                let str2 = Self::offset(*op2);
                format!("({} * {})", str1, str2)
            }

            OffsetExpr::Literal(lit) =>
                lit.to_string(),

            OffsetExpr::Var(var) =>
                var.to_string(),

            OffsetExpr::FunctionVar(fvar, indices) =>
                Self::array_ref(fvar.clone(), indices.into_iter().collect())
        }
    }

    fn lower_recur(&self, stmt: HEStatement, seal_stmts: &mut Vec<SEALStatement>) {
        match stmt {
            HEStatement::ForNode(dim, extent, body) => {
                let mut body_stmts = Vec::new();
                for body_stmt in body {
                    self.lower_recur(body_stmt, &mut body_stmts);
                }

                seal_stmts.push(
                    SEALStatement::ForNode(dim, extent, body_stmts)
                );
            },

            HEStatement::DeclareVar(array, optype, extent) => {
                let extent_str =
                    extent.into_iter()
                    .map(|e| e.to_string()).
                    collect::<Vec<String>>()
                    .join(", ");

                let instr_type =
                    match optype {
                        HEType::Native => SEALOpType::DeclareNativeArray,
                        HEType::Ciphertext => SEALOpType::DeclareCiphertextArray,
                        HEType::Plaintext => unreachable!()
                    };

                seal_stmts.push(
                    SEALStatement::Instruction(
                        SEALInstruction::Op(
                            instr_type,
                            array,
                            vec![format!("[{}]", extent_str)]
                        )
                    )
                );
            },

            HEStatement::AssignVar(array, indices, op) => {
                seal_stmts.push(
                    SEALStatement::Instruction(
                        SEALInstruction::OpInplace(
                            SEALOpType::Assign,
                            vec![array, Self::vec(indices), Self::operand(op)]
                        )
                    )
                )
            },

            HEStatement::Encode(array, indices) => {
                seal_stmts.push(
                    SEALStatement::Instruction(
                        SEALInstruction::OpInplace(
                            SEALOpType::Encode,
                            vec![array, Self::vec(indices)]
                        )
                    )
                );
            },

            HEStatement::Instruction(instr) => {
                match instr {
                    HEInstruction::Add(optype, id, ref1, ref2) => {
                        let use1 = self.use_analysis.can_reuse(id, &ref1);
                        let use2 = self.use_analysis.can_reuse(id, &ref2);
                        let (instr, _) =
                            match (use1, use2) {
                                (false, false) => {
                                    Self::get_binop(
                                        Operator::Add, optype, false, id, ref1, ref2
                                    )
                                },

                                (false, true) => {
                                    Self::get_binop(
                                        Operator::Add, optype, true, id, ref2, ref1
                                    )
                                },

                                (true, false) | (true, true) => {
                                    Self::get_binop(
                                        Operator::Add, optype, true, id, ref1, ref2
                                    )
                                },
                            };

                        seal_stmts.push(SEALStatement::Instruction(instr));
                    },
 
                    HEInstruction::Sub(optype, id, ref1, ref2) => {
                        let use1 = self.use_analysis.can_reuse(id, &ref1);
                        let use2 = self.use_analysis.can_reuse(id, &ref2);
                        let (instr, _) =
                            match (use1, use2) {
                                (false, false) | (false, true) => {
                                    Self::get_binop(
                                        Operator::Sub, optype, false, id, ref1, ref2
                                    )
                                },

                                (true, false) | (true, true) => {
                                    Self::get_binop(
                                        Operator::Sub, optype, true, id, ref1, ref2
                                    )
                                },
                            };

                        seal_stmts.push(SEALStatement::Instruction(instr));
                    },
                    
                    HEInstruction::Mul(optype, id, ref1, ref2) => {
                        let use1 = self.use_analysis.can_reuse(id, &ref1);
                        let use2 = self.use_analysis.can_reuse(id, &ref2);
                        let (instr, res) =
                            match (use1, use2) {
                                (false, false) => {
                                    Self::get_binop(
                                        Operator::Mul, optype, false, id, ref1, ref2
                                    )
                                },

                                (false, true) => {
                                    Self::get_binop(
                                        Operator::Mul, optype, true, id, ref2, ref1
                                    )
                                },

                                (true, false) | (true, true) => {
                                    Self::get_binop(
                                        Operator::Mul, optype, true, id, ref1, ref2
                                    )
                                },
                            };

                        seal_stmts.push(SEALStatement::Instruction(instr));

                        // insert relinearization statement for cipher-cipher multiplies
                        if let HEInstructionType::CipherCipher = optype {
                            seal_stmts.push(
                                SEALStatement::Instruction(
                                    SEALInstruction::OpInplace(
                                        SEALOpType::RelinearizeInplace,
                                        vec![res]
                                    )
                                )
                            );
                        }
                    },
                    
                    HEInstruction::Rot(optype, id, offset, he_ref) => {
                        let ref_use = self.use_analysis.can_reuse(id, &he_ref);
                        let instr = 
                            match (optype, ref_use) {
                                (HEInstructionType::Native, true) => {
                                    SEALInstruction::Op(
                                        SEALOpType::RotNative,
                                        Self::instruction(id),
                                        vec![Self::offset(offset), Self::he_ref(he_ref)]
                                    )
                                },

                                (HEInstructionType::Native, false) => {
                                    SEALInstruction::OpInplace(
                                        SEALOpType::RotNativeInplace,
                                        vec![Self::offset(offset), Self::he_ref(he_ref)]
                                    )
                                },

                                (HEInstructionType::CipherCipher, true) => {
                                    SEALInstruction::Op(
                                        SEALOpType::Rot,
                                        Self::instruction(id),
                                        vec![Self::offset(offset), Self::he_ref(he_ref)]
                                    )
                                },
                                
                                (HEInstructionType::CipherCipher, false) => {
                                    SEALInstruction::OpInplace(
                                        SEALOpType::RotInplace,
                                        vec![Self::offset(offset), Self::he_ref(he_ref)]
                                    )
                                },

                                (HEInstructionType::CipherPlain, _) =>
                                    unreachable!()
                            };

                        seal_stmts.push(SEALStatement::Instruction(instr));
                    }
                }
            }
        }
    }

    fn lower_context(
        &self,
        context: HEProgramContext,
        client_code: &mut Vec<SEALStatement>,
        server_code: &mut Vec<SEALStatement>
    ) {
        // TODO finish for client code
        for (vector, name) in context.ct_vector_map {
            server_code.extend([
                SEALStatement::Instruction(
                    SEALInstruction::OpInplace(
                        SEALOpType::ServerRecv,
                        vec![format!("\"{}\"", name)]
                    )
                ),
                SEALStatement::Instruction(
                    SEALInstruction::Op(
                        SEALOpType::ServerGetCiphertextInputVector,
                        name.clone(),
                        vec![format!("\"{}\"", name)]
                    )
                )
            ])
        }

        for (constval, name) in context.const_map {
            server_code.push(
                SEALStatement::Instruction(
                    SEALInstruction::Op(
                        SEALOpType::DeclareConst,
                        name,
                        vec![constval.to_string()]
                    )
                )
            )
        }

        for (mask, name) in context.mask_map {
            let mask_str =
                format!(
                    "[{}]",
                    mask.into_iter().map(|(extent, lo, hi)| {
                        format!("({}, {}, {})", extent, lo, hi)
                    })
                    .collect::<Vec<String>>()
                    .join(", ")
                );

            server_code.push(
                SEALStatement::Instruction(
                    SEALInstruction::Op(
                        SEALOpType::DeclareMask,
                        name,
                        vec![mask_str]
                    )
                )
            )
        }
    } 

    fn lower(mut self) -> SEALProgram {
        let mut client_code: Vec<SEALStatement> = Vec::new();
        let mut server_code: Vec<SEALStatement> = Vec::new();
        let mut program = HEProgram::default();
        std::mem::swap(&mut program, &mut self.program);

        self.lower_context(
            program.context, 
            &mut client_code,
            &mut server_code
        );
        for stmt in program.statements {
            self.lower_recur(stmt, &mut server_code);
        }

        SEALProgram { client_code, server_code }
    }
}

#[cfg(test)]
mod tests {
    use crate::{circ::{*, partial_eval::HEPartialEvaluator}, program::{*, lowering::CircuitLowering}, lang::*};
    use super::*;

    fn test_lowering(program: HEProgram) {
        let seal_program = SEALBackend::new(program).lower();
        println!("{}", seal_program);
    }

    #[test]
    fn test_partial_eval() {
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
        let add_2 = registry.register_circuit(ParamCircuitExpr::Op(Operator::Add, lit_2, lit_2));
        let ct = registry.register_circuit(ParamCircuitExpr::CiphertextVar(String::from("ct")));
        let reduce_vec = registry.register_circuit(ParamCircuitExpr::ReduceDim(
            String::from("j"),
            2,
            Operator::Add,
            ct,
        ));

        let circuit =
            registry.register_circuit(ParamCircuitExpr::Op(Operator::Add, reduce_vec, add_2));

        registry
            .ct_var_values
            .insert(String::from("ct"), CircuitValue::CoordMap(coord_map));

        let circuit_program = ParamCircuitProgram {
            registry,
            native_expr_list: vec![],
            circuit_expr_list: vec![(String::from("out"), vec![(String::from("i"), 2)], circuit)],
        };

        let circuit_program2 = HEPartialEvaluator::new().run(circuit_program);

        let mut lowering = CircuitLowering::new();
        let he_program = lowering.run(circuit_program2);

        test_lowering(he_program);
    }
}