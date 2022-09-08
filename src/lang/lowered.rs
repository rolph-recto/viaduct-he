/// lowered.rs
/// lowered program for generating output programs from templates

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::lang::instr::*;
 
type HELoweredOperand = String;

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "op")]
pub(crate) enum HELoweredInstr {
    Add { id: HELoweredOperand, op1: HELoweredOperand, op2: HELoweredOperand },
    AddInplace { op1: HELoweredOperand, op2: HELoweredOperand },
    AddPlain { id: HELoweredOperand, op1: HELoweredOperand, op2: HELoweredOperand },
    AddPlainInplace { op1: HELoweredOperand, op2: HELoweredOperand },
    Mul { id: HELoweredOperand, op1: HELoweredOperand, op2: HELoweredOperand },
    MulInplace { op1: HELoweredOperand, op2: HELoweredOperand },
    MulPlain { id: HELoweredOperand, op1: HELoweredOperand, op2: HELoweredOperand },
    MulPlainInplace { op1: HELoweredOperand, op2: HELoweredOperand },
    Rot { id: HELoweredOperand, op1: HELoweredOperand, op2: HELoweredOperand },
    RelinearizeInplace { op1: HELoweredOperand },
}

#[derive(Serialize)]
pub(crate) struct HELoweredProgram {
    vec_size: usize,
    symbols: HashSet<String>,
    instrs: Vec<HELoweredInstr>,
}

pub(crate) fn resolve_inplace_indirection(inplace_map: &HashMap<NodeId,NodeId>, id: NodeId) -> NodeId {
    let mut cur_id = id;
    while inplace_map.contains_key(&cur_id) {
        cur_id = inplace_map[&cur_id];
    }
    cur_id
}

pub(crate) fn lower_operand(inplace_map: &HashMap<NodeId,NodeId>, op: &HEOperand) -> String {
    match op {
        HEOperand::Ref(HERef::NodeRef(nr)) => {
            let resolved_nr = resolve_inplace_indirection(inplace_map, *nr);
            format!("i{}", resolved_nr)
        },
        HEOperand::Ref(HERef::ConstSym(sym)) => sym.clone(),
        HEOperand::ConstNum(n) => n.to_string(),
    }
}

pub(crate) fn lower_program(prog: &HEProgram, vec_size: usize) -> HELoweredProgram {
    let uses = prog.analyze_use();
    let mut inplace_map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut instrs: Vec<HELoweredInstr> = Vec::new();
    for instr in prog.instrs.iter() {
        match instr {
            HEInstr::Add { id, op1, op2 } => {
                let lid = format!("i{}", id);
                let lop1 = lower_operand(&inplace_map, op1);
                let lop2 = lower_operand(&inplace_map, op2);
                match (op1, op2) {
                    (HEOperand::Ref(r1), HEOperand::Ref(r2)) => {
                        match (r1, r2) {
                            (HERef::NodeRef(nr1), HERef::NodeRef(_)) if !uses[id+1].contains(nr1) => {
                                instrs.push(
                                    HELoweredInstr::AddInplace { op1: lop1, op2: lop2 }
                                );
                                inplace_map.insert(*id, *nr1);
                            },

                            (HERef::NodeRef(_), HERef::NodeRef(nr2)) if !uses[id+1].contains(nr2) => {
                                instrs.push(
                                    HELoweredInstr::AddInplace { op1: lop2, op2: lop1 }
                                );
                                inplace_map.insert(*id, *nr2);
                            },

                            _ => {
                                instrs.push(
                                    HELoweredInstr::Add { id: lid, op1: lop1, op2: lop2 }
                                )
                            }
                        }
                    },

                    (HEOperand::Ref(r1), HEOperand::ConstNum(_)) => {
                        match r1 {
                            HERef::NodeRef(nr1) if !uses[id+1].contains(nr1) => {
                                instrs.push(
                                    HELoweredInstr::AddPlainInplace { op1: lop1, op2: lop2 }
                                );
                                inplace_map.insert(*id, *nr1);
                            },

                            _ => {
                                instrs.push(
                                    HELoweredInstr::AddPlain { id: lid, op1: lop1, op2: lop2 }
                                )
                            }
                        }
                    },

                    (HEOperand::ConstNum(_), HEOperand::Ref(r2)) => {
                        match r2 {
                            HERef::NodeRef(nr2) if !uses[id+1].contains(nr2) => {
                                instrs.push(
                                    HELoweredInstr::AddPlainInplace { op1: lop2, op2: lop1 }
                                );
                                inplace_map.insert(*id, *nr2);
                            },

                            _ => {
                                instrs.push(
                                    HELoweredInstr::AddPlain { id: lid, op1: lop2, op2: lop1 }
                                )
                            }
                        }
                    },

                    (HEOperand::ConstNum(_), HEOperand::ConstNum(_)) => {
                        panic!("attempting to add two plaintexts---this should be constant folded")
                    }
                }
            },

            HEInstr::Mul { id, op1, op2 } => {
                let lid = format!("i{}", id);
                let lop1 = lower_operand(&inplace_map, op1);
                let lop2 = lower_operand(&inplace_map, op2);
                match (op1, op2) {
                    (HEOperand::Ref(_), HEOperand::Ref(_)) => {
                        instrs.push(
                            HELoweredInstr::Mul { id: lid.clone(), op1: lop1, op2: lop2 }
                        );
                    },

                    (HEOperand::Ref(_), HEOperand::ConstNum(_)) => {
                        instrs.push(
                            HELoweredInstr::MulPlain { id: lid.clone(), op1: lop1, op2: lop2 }
                        )
                    },

                    (HEOperand::ConstNum(_), HEOperand::Ref(_)) => {
                        instrs.push(
                            HELoweredInstr::MulPlain { id: lid.clone(), op1: lop2, op2: lop1 }
                        )
                    },

                    (HEOperand::ConstNum(_), HEOperand::ConstNum(_)) => {
                        panic!("attempting to multiply two plaintexts---this should be constant folded")
                    }
                }

                // relinearize at every multiplication, except for outputs;
                // outputs will not be used in the future,
                // so there's no need to minimize noise for them
                if uses[id+1].contains(id) {
                    instrs.push(
                        HELoweredInstr::RelinearizeInplace { op1: lid }
                    )
                }
            },
            
            HEInstr::Rot { id: index, op1, op2 } => {
                let lid = format!("i{}", index);
                let lop1 = lower_operand(&inplace_map, op1);
                let lop2 = lower_operand(&inplace_map, op2);
                match (op1, op2) {
                    (HEOperand::Ref(_), HEOperand::ConstNum(_)) => {
                        instrs.push(
                            HELoweredInstr::Rot { id: lid, op1: lop1, op2: lop2 }
                        )
                    }

                    _ => panic!("must rotate ciphertext with constant value")
                }
            }
        }
    }
    HELoweredProgram { vec_size, symbols: prog.get_symbols(), instrs }
}