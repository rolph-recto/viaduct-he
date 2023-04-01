use indexmap::IndexMap;
/// program.rs
/// instruction representation of HE programs

use pretty::RcDoc;
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Display};

use crate::{
    circ::{vector_info::VectorInfo, MaskVector},
    lang::{ArrayName, Extent, OffsetExpr},
};

pub type InstructionId = usize;

pub mod lowering;
pub mod backend;

/// data types for arrays in an HE program.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum HEType {
    // machine-native data and operation
    Native,

    // native data encoded into a plaintext
    Plaintext,

    // ciphertext encrypted by the client
    Ciphertext, 
}

impl Display for HEType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HEType::Native => write!(f, "N"),
            HEType::Plaintext => write!(f, "P"),
            HEType::Ciphertext => write!(f, "C"),
        }
    }
}

/// instructions types in an HE program.
#[derive(Copy, Clone, Debug)]
pub enum HEInstructionType {
    // native (cleartext) operations
    Native,

    // ciphertext-plaintext operations
    CipherPlain,

    // ciphertext-ciphertext operations
    CipherCipher,
}

impl Display for HEInstructionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HEInstructionType::Native => write!(f, "N"),
            HEInstructionType::CipherPlain => write!(f, "CP"),
            HEInstructionType::CipherCipher => write!(f, "CC"),
        }
    }
}

#[derive(Clone, Debug)]
pub enum HEIndex {
    Var(String),
    Literal(isize),
}

impl Display for HEIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HEIndex::Var(var) => write!(f, "{}", var),

            HEIndex::Literal(val) => write!(f, "{}", val),
        }
    }
}

#[derive(Clone, Debug)]
pub enum HERef {
    // reference to a previous instruction's output
    Instruction(InstructionId),

    // index to an array (variable if no indices)
    Array(ArrayName, Vec<HEIndex>),
}

impl Display for HERef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HERef::Instruction(id) => write!(f, "instr{}", id),

            HERef::Array(vector, indices) => {
                let index_str = indices
                    .iter()
                    .map(|i| format!("[{}]", i))
                    .collect::<Vec<String>>()
                    .join("");

                write!(f, "{}{}", vector, index_str)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum HEOperand {
    Ref(HERef),
    Literal(isize),
}

impl fmt::Display for HEOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HEOperand::Ref(r) => {
                write!(f, "{}", r)
            }

            HEOperand::Literal(n) => {
                write!(f, "{}", n)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum HEInstruction {
    Add(HEInstructionType, usize, HERef, HERef),
    Sub(HEInstructionType, usize, HERef, HERef),
    Mul(HEInstructionType, usize, HERef, HERef),
    Rot(HEInstructionType, usize, OffsetExpr, HERef),
}

impl HEInstruction {
    pub fn get_id(&self) -> InstructionId {
        match self {
            HEInstruction::Add(_, id, _, _) |
            HEInstruction::Sub(_, id, _, _) |
            HEInstruction::Mul(_, id, _, _) |
            HEInstruction::Rot(_, id, _, _) =>
                *id
        }
    }

    pub fn get_refs(&self) -> Vec<HERef> {
        match self {
            HEInstruction::Add(_, _, ref1, ref2) |
            HEInstruction::Sub(_, _, ref1, ref2) |
            HEInstruction::Mul(_, _, ref1, ref2) =>
                vec![ref1.clone(), ref2.clone()],

            HEInstruction::Rot(_, _, _, ref1) => 
                vec![ref1.clone()]
        }
    }

    /// get the other instuctions this one reads
    pub fn get_instr_refs(&self) -> Vec<InstructionId> {
        self.get_refs().into_iter().filter_map(|r| {
            match r {
                HERef::Instruction(id) => Some(id),
                HERef::Array(_, _) => None,
            }
        }).collect()
    }
}

impl fmt::Display for HEInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HEInstruction::Add(optype, id, op1, op2) =>
                write!(f, "{:<2}: instr{} = {} + {}", optype, id, op1, op2),

            HEInstruction::Sub(optype, id, op1, op2) =>
                write!(f, "{:<2}: instr{} = {} - {}", optype, id, op1, op2),

            HEInstruction::Mul(optype, id, op1, op2) =>
                write!(f, "{:<2}: instr{} = {} * {}", optype, id, op1, op2),

            HEInstruction::Rot(optype, id, op1, op2) =>
                write!(f, "{:<2}: instr{} = rot {} {}", optype, id, op1, op2),
        }
    }
}

#[derive(Clone, Debug)]
pub enum HEStatement {
    // for loop
    ForNode(String, usize, Vec<HEStatement>),

    // declare a new array
    DeclareVar(String, HEType, Vec<Extent>, isize),

    // assign to a variable
    AssignVar(String, Vec<HEIndex>, HEOperand),

    // encode native array into plaintext
    Encode(String, Vec<HEIndex>),

    // HE or native array instruction
    Instruction(HEInstruction),
}

impl HEStatement {
    pub fn to_doc(&self) -> RcDoc<()> {
        match self {
            HEStatement::ForNode(dim, extent, body) => {
                let body_doc =
                    RcDoc::intersperse(body.iter().map(|stmt| stmt.to_doc()), RcDoc::hardline());

                RcDoc::text(format!("for {}: {} {{", dim, extent))
                    .append(RcDoc::hardline().append(body_doc).nest(4))
                    .append(RcDoc::hardline())
                    .append(RcDoc::text("}"))
            },

            HEStatement::DeclareVar(var, ty, extents, default) => {
                let extent_str = extents
                    .iter()
                    .map(|i| format!("[{}]", i))
                    .collect::<Vec<String>>()
                    .join("");

                RcDoc::text(format!("var {}: {}{} = {}", var, ty, extent_str, default))
            },

            HEStatement::AssignVar(var, indices, val) => {
                let index_str = indices
                    .iter()
                    .map(|i| format!("[{}]", i))
                    .collect::<Vec<String>>()
                    .join("");

                RcDoc::text(format!("{}{} = {}", var, index_str, val))
            },

            HEStatement::Encode(var, indices) => {
                let index_str = indices
                    .iter()
                    .map(|i| format!("[{}]", i))
                    .collect::<Vec<String>>()
                    .join("");

                RcDoc::text(format!("encode({}{})", var, index_str))
            },

            HEStatement::Instruction(instr) => RcDoc::text(instr.to_string()),
        }
    }
}

impl Display for HEStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_doc().render_fmt(80, f)
    }
}

#[derive(Default)]
pub struct HEProgramContext {
    ct_vector_map: IndexMap<VectorInfo, ArrayName>,
    pt_vector_map: IndexMap<VectorInfo, ArrayName>,
    mask_map: IndexMap<MaskVector, ArrayName>,
    const_map: IndexMap<isize, ArrayName>,
}

#[derive(Default)]
pub struct HEProgram {
    pub context: HEProgramContext,
    pub statements: Vec<HEStatement>,
    pub output: ArrayName,
}

impl HEProgram {
    pub fn to_doc(&self) -> RcDoc<()> {
        let ct_vector_doc =
            if self.context.ct_vector_map.len() > 0 {
                RcDoc::intersperse(
                    self.context
                        .ct_vector_map
                        .iter()
                        .map(|(vector, name)| {
                            RcDoc::text(format!("val {}: C = vector({})", name, vector))
                        }),
                    RcDoc::hardline(),
                )
            } else {
                RcDoc::nil()
            };
        
        let pt_vector_doc =
            if self.context.pt_vector_map.len() > 0 {
                RcDoc::hardline()
                .append(RcDoc::intersperse(
                    self.context
                        .pt_vector_map
                        .iter()
                        .map(|(mask, name)| {
                            RcDoc::text(format!("val {}: N = vector({})", name, mask))
                        }),
                    RcDoc::hardline(),
                ))
            } else {
                RcDoc::nil()
            };
        
        let mask_doc =
            if self.context.mask_map.len() > 0 {
                RcDoc::hardline()
                .append(RcDoc::intersperse(
                    self.context
                        .mask_map
                        .iter()
                        .map(|(mask, name)| {
                            RcDoc::text(format!("val {}: N = mask({:?})", name, mask))
                        }),
                    RcDoc::hardline(),
                ))
            } else {
                RcDoc::nil()
            };

        let const_doc =
            if self.context.const_map.len() > 0 {
                RcDoc::hardline()
                .append(RcDoc::intersperse(
                    self.context
                        .const_map
                        .iter()
                        .map(|(constval, name)| {
                            RcDoc::text(format!("val {}: N = const({})", name, constval))
                        }),
                    RcDoc::hardline(),
                ))
            } else {
                RcDoc::nil()
            };

        let stmts_doc =
            RcDoc::hardline()
            .append(RcDoc::intersperse(
                self.statements.iter().map(|s| s.to_doc()),
                RcDoc::hardline(),
            ));

        ct_vector_doc
        .append(pt_vector_doc)
        .append(mask_doc)
        .append(const_doc)
        .append(stmts_doc)
    }
}

impl Display for HEProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_doc().render_fmt(80, f)
    }
}
