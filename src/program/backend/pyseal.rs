/// pyseal.rs;
/// PySEAL backend

use std::{collections::{HashMap, HashSet}, fs::rename};

use handlebars::{Handlebars, RenderError};
use log::info;
use serde::Serialize;

use crate::{
    program::*,
    lang::Operator,
    circ::vector_info::VectorDimContent
};

use super::HEBackend;

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
    DeclarePlaintextArray,

    DeclareMask,
    DeclareConst,
    BuildVector,

    ServerInput,
    ServerRecv,
    ServerSend,

    ClientInput,
    ClientOutput,
    ClientSend,
    ClientRecv,

    StartServerExec,
    EndServerExec,
}

impl SEALOpType {
    pub fn to_doc(&self) -> RcDoc<()> {
        match self {
            SEALOpType::Add => RcDoc::text("wrapper.add"),
            SEALOpType::AddInplace => RcDoc::text("wrapper.add_inplace"),
            SEALOpType::AddPlain => RcDoc::text("wrapper.add_plain"),
            SEALOpType::AddPlainInplace => RcDoc::text("wrapper.add_plain_inplace"),
            SEALOpType::AddNative => RcDoc::text("wrapper.add_native"),
            SEALOpType::AddNativeInplace => RcDoc::text("wrapper.add_native_inplace"),

            SEALOpType::Sub => RcDoc::text("wrapper.subtract"),
            SEALOpType::SubInplace => RcDoc::text("wrapper.subtract_inplace"),
            SEALOpType::SubPlain => RcDoc::text("wrapper.subtract_plain"),
            SEALOpType::SubPlainInplace => RcDoc::text("wrapper.subtract_plain_inplace"),
            SEALOpType::SubNative => RcDoc::text("wrapper.subtract_native"),
            SEALOpType::SubNativeInplace => RcDoc::text("wrapper.subtract_native_inplace"),

            SEALOpType::Mul => RcDoc::text("wrapper.multiply"),
            SEALOpType::MulInplace => RcDoc::text("wrapper.multiply_inplace"),
            SEALOpType::MulPlain => RcDoc::text("wrapper.multiply_plain"),
            SEALOpType::MulPlainInplace => RcDoc::text("wrapper.multiply_plain_inplace"),
            SEALOpType::MulNative => RcDoc::text("wrapper.multiply_native"),
            SEALOpType::MulNativeInplace => RcDoc::text("wrapper.mul_native_inplace"),

            SEALOpType::Rot => RcDoc::text("wrapper.rotate_rows"),
            SEALOpType::RotInplace => RcDoc::text("wrapper.rotate_rows_inplace"),
            SEALOpType::RotNative => RcDoc::text("wrapper.rotate_rows_native"),
            SEALOpType::RotNativeInplace => RcDoc::text("wrapper.rotate_rows_native_inplace"),

            SEALOpType::Encode => RcDoc::text("wrapper.encode"),
            SEALOpType::RelinearizeInplace => RcDoc::text("wrapper.relinearize_inplace"),

            SEALOpType::Assign => RcDoc::text("wrapper.set"),

            SEALOpType::DeclareNativeArray => RcDoc::text("wrapper.native_array"),
            SEALOpType::DeclareCiphertextArray => RcDoc::text("wrapper.ciphertext_array"),
            SEALOpType::DeclarePlaintextArray => RcDoc::text("wrapper.plaintext_array"),
            SEALOpType::DeclareMask => RcDoc::text("wrapper.mask"),
            SEALOpType::DeclareConst => RcDoc::text("wrapper.const"),

            SEALOpType::BuildVector => RcDoc::text("wrapper.build_vector"),

            SEALOpType::ServerInput => RcDoc::text("wrapper.server_input"),
            SEALOpType::ServerRecv => RcDoc::text("wrapper.server_recv"),
            SEALOpType::ServerSend => RcDoc::text("wrapper.server_send"),

            SEALOpType::ClientInput => RcDoc::text("wrapper.client_input"),
            SEALOpType::ClientOutput => RcDoc::text("wrapper.client_output"),
            SEALOpType::ClientSend => RcDoc::text("wrapper.client_send"),
            SEALOpType::ClientRecv => RcDoc::text("wrapper.client_recv"),

            SEALOpType::StartServerExec => RcDoc::text("wrapper.start_server_exec"),
            SEALOpType::EndServerExec => RcDoc::text("wrapper.end_server_exec"),
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
    client_pre_code: Vec<SEALStatement>,
    client_post_code: Vec<SEALStatement>,
    server_code: Vec<SEALStatement>,
}

impl SEALProgram {
    pub fn to_doc(&self) -> RcDoc<()> {
        let client_pre_doc =
            if self.client_pre_code.len() > 0 {
                RcDoc::intersperse(
                    self.client_pre_code.iter().map(|stmt| {
                        stmt.to_doc()
                    }),
                    RcDoc::hardline()
                ) 

            } else {
                RcDoc::text("pass")
            };

        let client_post_doc =
            if self.client_post_code.len() > 0 {
                RcDoc::intersperse(
                    self.client_post_code.iter().map(|stmt| {
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

        RcDoc::text("def client_pre(wrapper):")
        .append(
            RcDoc::hardline().append(client_pre_doc).nest(4)
        )
        .append(RcDoc::hardline())
        .append(RcDoc::hardline())
        .append(RcDoc::text("def client_post(wrapper):"))
        .append(
            RcDoc::hardline().append(client_post_doc).nest(4)
        )
        .append(RcDoc::hardline())
        .append(RcDoc::hardline())
        .append(RcDoc::text("def server(wrapper):"))
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

    pub fn can_reuse(
        &self,
        id: InstructionId,
        he_ref: &HERef
    ) -> Option<InstructionId> {
        match he_ref {
            HERef::Instruction(ref_id) => {
                let use_map = self.0.get(&id).unwrap();

                if !use_map.contains(ref_id) {
                    Some(*ref_id)

                } else {
                    None
                }
            },

            HERef::Array(_, _) => None,
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

struct SEALLowering {
    use_analysis: UseAnalysis,
    program: HEProgram,
    enable_inplace: bool
}

impl SEALLowering {
    fn new(program: HEProgram, enable_inplace: bool) -> Self {
        let use_analysis = UseAnalysis::analyze(&program);
        Self { use_analysis, program, enable_inplace }
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

            HERef::Array(array, indices) => {
                let vec_str = 
                    if indices.len() > 0 {
                        Self::vec(indices)

                    } else {
                        String::from("")
                    };

                format!("{}.get({})", array, vec_str)
            }
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

    fn can_reuse(&self, id: InstructionId, r: &HERef) -> Option<InstructionId> {
        if self.enable_inplace {
            self.use_analysis.can_reuse(id, r)

        } else {
            None
        }
    }

    fn resolve_ref(&self, rename_map: &HashMap<InstructionId, InstructionId>, r: HERef) -> HERef {
        match r {
            HERef::Instruction(id) => {
                let mut cur = id;

                while let Some(rid) = rename_map.get(&cur) {
                    if *rid != cur {
                        cur = *rid;

                    } else {
                        break;
                    }
                }
                
                HERef::Instruction(cur)
            },

            HERef::Array(_, _) => r
        }
    }

    fn lower_recur(
        &self,
        stmt: HEStatement,
        seal_stmts: &mut Vec<SEALStatement>,
        rename_map: &mut HashMap<InstructionId, InstructionId>,
    ) {
        match stmt {
            HEStatement::ForNode(dim, extent, body) => {
                let mut body_stmts = Vec::new();
                for body_stmt in body {
                    self.lower_recur(body_stmt, &mut body_stmts, rename_map);
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
                        HEType::Plaintext => SEALOpType::DeclarePlaintextArray,
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
                let nop =
                    match op {
                        HEOperand::Ref(r) => {
                            let nref = self.resolve_ref(rename_map, r);
                            HEOperand::Ref(nref)
                        },

                        HEOperand::Literal(_) => op
                    };

                seal_stmts.push(
                    SEALStatement::Instruction(
                        SEALInstruction::OpInplace(
                            SEALOpType::Assign,
                            vec![array, Self::vec(indices), Self::operand(nop)]
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
                        let use1 = self.can_reuse(id, &ref1);
                        let use2 = self.can_reuse(id, &ref2);
                        let nref1 = self.resolve_ref(rename_map, ref1);
                        let nref2 = self.resolve_ref(rename_map, ref2);
                        let (instr, _) =
                            match (use1, use2) {
                                (None, None) => {
                                    Self::get_binop(
                                        Operator::Add, optype, false, id, nref1, nref2
                                    )
                                },

                                (None, Some(rid2)) => {
                                    rename_map.insert(id, rid2);
                                    Self::get_binop(
                                        Operator::Add, optype, true, id, nref2, nref1
                                    )
                                },

                                (Some(rid1), None) | (Some(rid1), Some(_)) => {
                                    rename_map.insert(id, rid1);
                                    Self::get_binop(
                                        Operator::Add, optype, true, id, nref1, nref2
                                    )
                                },
                            };

                        seal_stmts.push(SEALStatement::Instruction(instr));
                    },
 
                    HEInstruction::Sub(optype, id, ref1, ref2) => {
                        let use1 = self.can_reuse(id, &ref1);
                        let use2 = self.can_reuse(id, &ref2);
                        let nref1 = self.resolve_ref(rename_map, ref1);
                        let nref2 = self.resolve_ref(rename_map, ref2);
                        let (instr, _) =
                            match (use1, use2) {
                                (None, None) | (None, Some(_)) => {
                                    Self::get_binop(
                                        Operator::Sub, optype, false, id, nref1, nref2
                                    )
                                },

                                (Some(rid1), None) | (Some(rid1), Some(_)) => {
                                    rename_map.insert(id, rid1);
                                    Self::get_binop(
                                        Operator::Sub, optype, true, id, nref1, nref2
                                    )
                                },
                            };

                        seal_stmts.push(SEALStatement::Instruction(instr));
                    },
                    
                    HEInstruction::Mul(optype, id, ref1, ref2) => {
                        let use1 = self.can_reuse(id, &ref1);
                        let use2 = self.can_reuse(id, &ref2);
                        let nref1 = self.resolve_ref(rename_map, ref1);
                        let nref2 = self.resolve_ref(rename_map, ref2);
                        let (instr, res) =
                            match (use1, use2) {
                                (None, None) => {
                                    Self::get_binop(
                                        Operator::Mul, optype, false, id, nref1, nref2
                                    )
                                },

                                (None, Some(rid2)) => {
                                    rename_map.insert(id, rid2);
                                    Self::get_binop(
                                        Operator::Mul, optype, true, id, nref2, nref1
                                    )
                                },

                                (Some(rid1), None) | (Some(rid1), Some(_)) => {
                                    rename_map.insert(id, rid1);
                                    Self::get_binop(
                                        Operator::Mul, optype, true, id, nref1, nref2
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
                    
                    HEInstruction::Rot(optype, id, offset, body_ref) => {
                        let ref_use = self.can_reuse(id, &body_ref);
                        let body_nref = self.resolve_ref(rename_map, body_ref);
                        let instr = 
                            match (optype, ref_use) {
                                (HEInstructionType::Native, None) => {
                                    SEALInstruction::Op(
                                        SEALOpType::RotNative,
                                        Self::instruction(id),
                                        vec![Self::offset(offset), Self::he_ref(body_nref)]
                                    )
                                },

                                (HEInstructionType::Native, Some(rid2)) => {
                                    rename_map.insert(id, rid2);
                                    SEALInstruction::OpInplace(
                                        SEALOpType::RotNativeInplace,
                                        vec![Self::offset(offset), Self::he_ref(body_nref)]
                                    )
                                },

                                (HEInstructionType::CipherCipher, None) => {
                                    SEALInstruction::Op(
                                        SEALOpType::Rot,
                                        Self::instruction(id),
                                        vec![Self::offset(offset), Self::he_ref(body_nref)]
                                    )
                                },
                                
                                (HEInstructionType::CipherCipher, Some(rid2)) => {
                                    rename_map.insert(id, rid2);
                                    SEALInstruction::OpInplace(
                                        SEALOpType::RotInplace,
                                        vec![Self::offset(offset), Self::he_ref(body_nref)]
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

    fn vector_dim_content(dim: &VectorDimContent) -> String {
        match dim {
            VectorDimContent::FilledDim { dim, extent, stride, oob_left, oob_right, pad_left, pad_right } =>
                format!(
                    "FilledDim({}, {}, {}, {}, {}, {}, {})",
                    dim, extent, stride, oob_left, oob_right, pad_left, pad_right
                ),

            VectorDimContent::EmptyDim { extent, pad_left, pad_right, oob_right } =>
                format!(
                    "EmptyDim({}, {}, {}, {})",
                    extent, pad_left, pad_right, oob_right
                ),

            VectorDimContent::ReducedDim { extent: _, pad_left: _, pad_right: _ } => 
                unreachable!("input vector has reduced dim")
        }
    }

    /// return argument list for vector info
    /// args should be [array, preprocessing, offset, dims]
    fn vector_info(vector: &VectorInfo) -> Vec<String> {
        let mut args = Vec::new();
        args.push(format!("\"{}\"", vector.array));

        let preprocess_str = 
            if let Some(preprocess) = vector.preprocessing {
                preprocess.to_string()

            } else {
                String::from("None")
            };

        args.push(preprocess_str);
        args.push(Self::vec(vector.offset_map.map.clone()));

        let dims_vec: Vec<String> = 
            vector.dims.iter()
            .map(|dim| Self::vector_dim_content(dim))
            .collect();

        args.push(Self::vec(dims_vec));
        args
    }

    fn lower_context_client(
        &self,
        context: &HEProgramContext,
        client_code: &mut Vec<SEALStatement>
    ) {
        let ct_inputs: HashSet<String> =
            context.ct_vector_map.iter().map(|(vec, _)| {
                vec.array.clone()
            }).collect();

        // get ct inputs
        for input in ct_inputs {
            client_code.push(
                SEALStatement::Instruction(
                    SEALInstruction::OpInplace(
                        SEALOpType::ClientInput,
                        vec![format!("\"{}\"", input)]
                    )
                )
            );
        }

        // recv ct vectors from client 
        for (vector, name) in context.ct_vector_map.iter() {
            client_code.extend([
                SEALStatement::Instruction(
                    SEALInstruction::Op(
                        SEALOpType::BuildVector,
                        name.clone(),
                        Self::vector_info(vector)
                    )
                ),
                SEALStatement::Instruction(
                    SEALInstruction::OpInplace(
                        SEALOpType::ClientSend,
                        vec![format!("\"{}\"", name), name.clone()],
                    )
                )
            ])
        }
    }

    fn lower_context_server(
        &self,
        context: &HEProgramContext,
        server_code: &mut Vec<SEALStatement>
    ) {
        let pt_inputs: HashSet<String> =
            context.pt_vector_map.iter().map(|(vec, _)| {
                vec.array.clone()
            }).collect();

        // get pt inputs
        for input in pt_inputs {
            server_code.push(
                SEALStatement::Instruction(
                    SEALInstruction::OpInplace(
                        SEALOpType::ServerInput,
                        vec![format!("\"{}\"", input)]
                    )
                )
            );
        }

        // build pt vectors
        for (vector, name) in context.pt_vector_map.iter() {
            server_code.push(
                SEALStatement::Instruction(
                    SEALInstruction::Op(
                        SEALOpType::BuildVector,
                        name.clone(),
                        Self::vector_info(vector)
                    )
                )
            )
        }

        // recv ct vectors from client 
        for (_, name) in context.ct_vector_map.iter() {
            server_code.push(
                SEALStatement::Instruction(
                    SEALInstruction::Op(
                        SEALOpType::ServerRecv,
                        name.clone(),
                        vec![format!("\"{}\"", name)]
                    )
                )
            );
        }

        for (constval, name) in context.const_map.iter() {
            server_code.push(
                SEALStatement::Instruction(
                    SEALInstruction::Op(
                        SEALOpType::DeclareConst,
                        name.clone(),
                        vec![constval.to_string()]
                    )
                )
            )
        }

        for (mask, name) in context.mask_map.iter() {
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
                        name.clone(),
                        vec![mask_str]
                    )
                )
            )
        }
    } 

    fn lower(mut self) -> SEALProgram {
        let mut client_pre_code: Vec<SEALStatement> = Vec::new();
        let mut server_code: Vec<SEALStatement> = Vec::new();
        let mut program = HEProgram::default();
        std::mem::swap(&mut program, &mut self.program);

        self.lower_context_client(&program.context, &mut client_pre_code);
        self.lower_context_server(&program.context, &mut server_code);

        server_code.push(
            SEALStatement::Instruction(
                SEALInstruction::OpInplace(SEALOpType::StartServerExec, vec![])
            )
        );

        let mut rename_map = HashMap::new();
        for stmt in program.statements {
            self.lower_recur(stmt, &mut server_code, &mut rename_map);
        }

        server_code.push(
            SEALStatement::Instruction(
                SEALInstruction::OpInplace(SEALOpType::EndServerExec, vec![])
            )
        );

        server_code.push(
            SEALStatement::Instruction(
                SEALInstruction::OpInplace(
                    SEALOpType::ServerSend,
                    vec![format!("\"{}\"", program.output), program.output.clone()]
                )
            )
        );

        let client_post_code = Vec::from([
            SEALStatement::Instruction(
                SEALInstruction::Op(
                    SEALOpType::ClientRecv,
                    program.output.clone(),
                    vec![format!("\"{}\"", program.output.clone())]
                )
            ),
            SEALStatement::Instruction(
                SEALInstruction::OpInplace(
                    SEALOpType::ClientOutput,
                    vec![program.output]
                )
            ),
        ]);

        SEALProgram { client_pre_code, client_post_code, server_code }
    }
}

#[derive(Serialize)]
struct SEALHandlebarsData {
    program: String,
    size: usize,
}

pub struct SEALBackend {
    template_file_opt: Option<String>,
    enable_inplace: bool,
    size: usize,
}

impl SEALBackend {
    pub fn new(template_file_opt: Option<String>, enable_inplace: bool, size: usize) -> Self {
        Self { template_file_opt, enable_inplace, size }
    }

    fn codegen_template(
        &mut self,
        program: SEALProgram,
    ) -> Result<String, RenderError> {
        let template_file = self.template_file_opt.as_ref().unwrap();
        let template_str =
            std::fs::read_to_string(template_file)
            .expect(&format!("Could not read file {}", &template_file));

        let mut handlebars = Handlebars::new();
        handlebars.register_template_string("template", template_str)?;

        let seal_program =
            SEALHandlebarsData { program: program.to_string(), size: self.size };

        handlebars.render("template", &seal_program)
    }
}

impl<'a> HEBackend<'a> for SEALBackend {
    fn name(&self) -> &str { "pyseal" }
        
    fn compile(
        &mut self,
        program: HEProgram,
        mut writer: Box<dyn std::io::Write + 'a>,
    ) -> std::io::Result<()> {
        let backend = SEALLowering::new(program, self.enable_inplace);
        let program = backend.lower();

        if let Some(_) = &self.template_file_opt {
            match self.codegen_template(program) {
                Ok(prog_str) => {
                    write!(writer, "{}", prog_str)?;
                    Ok(())
                },

                Err(err) => {
                    info!("{}", err);
                    Err(std::io::Error::new(std::io::ErrorKind::Other, err.to_string()))
                }
            }

        } else {
            write!(writer, "{}", &program.to_string())?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{circ::{*, plaintext_hoisting::*}, program::{*, lowering::*}, lang::*};
    use super::*;

    fn test_lowering(program: HEProgram) {
        let seal_program = SEALLowering::new(program, false).lower();
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

        let circuit_program2 = PlaintextHoisting::new().run(circuit_program);

        let mut lowering = CircuitLowering::new();
        let he_program = lowering.run(circuit_program2);

        test_lowering(he_program);
    }

    #[test]
    fn test_partial_eval2() {
        let mut registry = CircuitObjectRegistry::new();

        let vector = VectorInfo {
            array: String::from("arr"),
            preprocessing: None,
            offset_map: BaseOffsetMap::new(2),
            dims: im::Vector::new(),
        };

        let ct_obj = CiphertextObject::InputVector(vector);

        let mut coord_map =
            IndexCoordinateMap::from_coord_system(IndexCoordinateSystem::from_dim_list(vec![
                (String::from("i"), 2),
                (String::from("j"), 2),
            ]));

        coord_map.set(im::vector![0, 0], ct_obj.clone());
        coord_map.set(im::vector![0, 1], ct_obj.clone());
        coord_map.set(im::vector![1, 0], ct_obj.clone());
        coord_map.set(im::vector![1, 1], ct_obj.clone());

        registry
            .ct_var_values
            .insert(String::from("ct"), CircuitValue::CoordMap(coord_map));

        let vector2 = VectorInfo {
            array: String::from("parr"),
            preprocessing: None,
            offset_map: BaseOffsetMap::new(2),
            dims: im::Vector::new(),
        };

        let pt_obj = PlaintextObject::InputVector(vector2);

        let mut coord_map2 =
            IndexCoordinateMap::from_coord_system(IndexCoordinateSystem::from_dim_list(vec![
                (String::from("i"), 2),
                (String::from("j"), 2),
            ]));

        coord_map2.set(im::vector![0, 0], pt_obj.clone());
        coord_map2.set(im::vector![0, 1], pt_obj.clone());
        coord_map2.set(im::vector![1, 0], pt_obj.clone());
        coord_map2.set(im::vector![1, 1], pt_obj.clone());

        registry
            .pt_var_values
            .insert(String::from("pt"), CircuitValue::CoordMap(coord_map2));

        let pt = registry.register_circuit(ParamCircuitExpr::PlaintextVar(String::from("pt")));
        let add_pt = registry.register_circuit(ParamCircuitExpr::Op(Operator::Add, pt, pt));
        let ct = registry.register_circuit(ParamCircuitExpr::CiphertextVar(String::from("ct")));
        let reduce_vec = registry.register_circuit(ParamCircuitExpr::ReduceDim(
            String::from("j"),
            2,
            Operator::Add,
            ct,
        ));

        let circuit =
            registry.register_circuit(ParamCircuitExpr::Op(Operator::Add, reduce_vec, add_pt));

        let circuit_program = ParamCircuitProgram {
            registry,
            native_expr_list: vec![],
            circuit_expr_list: vec![(String::from("out"), vec![(String::from("i"), 2)], circuit)],
        };

        let circuit_program2 = PlaintextHoisting::new().run(circuit_program);

        let mut lowering = CircuitLowering::new();
        let he_program = lowering.run(circuit_program2);

        test_lowering(he_program);
    }
}