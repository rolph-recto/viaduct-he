/// lowered.rs
/// lowered program for generating output programs from templates

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::ir::instr::*;
 
type HELoweredNodeId = String;
type HELoweredOperand = String;

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "op")]
pub enum HELoweredInstr {
    Add { id: HELoweredNodeId, op1: HELoweredOperand, op2: HELoweredOperand },
    AddInplace { op1: HELoweredOperand, op2: HELoweredOperand },
    AddPlain { id: HELoweredOperand, op1: HELoweredOperand, op2: HELoweredOperand },
    AddPlainInplace { op1: HELoweredOperand, op2: HELoweredOperand },
    Mul { id: HELoweredNodeId, op1: HELoweredOperand, op2: HELoweredOperand },
    MulInplace { op1: HELoweredOperand, op2: HELoweredOperand },
    MulPlain { id: HELoweredOperand, op1: HELoweredOperand, op2: HELoweredOperand },
    MulPlainInplace { op1: HELoweredOperand, op2: HELoweredOperand },
    Rot { id: HELoweredNodeId, op1: HELoweredOperand, op2: HELoweredOperand },
    RotInplace { op1: HELoweredOperand, op2: HELoweredOperand },
    RelinearizeInplace { op1: HELoweredNodeId },
}

#[derive(Serialize)]
pub struct HELoweredProgram {
    vec_size: usize,
    symbols: HashSet<String>,
    constants: Vec<(i32, String)>,
    instrs: Vec<HELoweredInstr>,
    output: HELoweredNodeId
}

pub(crate) fn resolve_inplace_indirection(inplace_map: &HashMap<NodeId,NodeId>, id: NodeId) -> NodeId {
    let mut cur_id = id;
    while inplace_map.contains_key(&cur_id) {
        cur_id = inplace_map[&cur_id];
    }
    cur_id
}

pub(crate) fn lower_operand(inplace_map: &HashMap<NodeId,NodeId>, const_map: &mut HashMap<i32, String>, op: &HEOperand) -> String {
    match op {
        HEOperand::Ref(HERef::NodeRef(nr)) => {
            let resolved_nr = resolve_inplace_indirection(inplace_map, *nr);
            format!("i{}", resolved_nr)
        },
        HEOperand::Ref(HERef::ConstSym(sym)) => sym.clone(),
        HEOperand::ConstNum(n) => {
            if const_map.contains_key(n) {
                const_map[n].clone()

            } else {
                if *n >= 0 {
                    const_map.insert(*n, format!("const_{}", n));

                } else {
                    const_map.insert(*n, format!("const_neg{}", -n));
                }
                const_map[n].clone()
            }
        },
    }
}

pub fn lower_program(prog: &HEProgram, vec_size: usize) -> HELoweredProgram {
    let uses = prog.analyze_use();
    let mut inplace_map: HashMap<NodeId, NodeId> = HashMap::new();
    let mut const_map: HashMap<i32, String> = HashMap::new();
    let mut instrs: Vec<HELoweredInstr> = Vec::new();

    for instr in prog.instrs.iter() {
        match instr {
            HEInstr::Add { id, op1, op2 } => {
                let lid = format!("i{}", id);
                let lop1 = lower_operand(&inplace_map, &mut const_map, op1);
                let lop2 = lower_operand(&inplace_map, &mut const_map, op2);
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
                let lop1 = lower_operand(&inplace_map, &mut const_map, op1);
                let lop2 = lower_operand(&inplace_map, &mut const_map, op2);
                let mut relin_id = lid.clone();
                match (op1, op2) {
                    (HEOperand::Ref(r1), HEOperand::Ref(r2)) => {
                        match (r1, r2) {
                            (HERef::NodeRef(nr1), HERef::NodeRef(_)) if !uses[id+1].contains(nr1) => {
                                instrs.push(
                                    HELoweredInstr::MulInplace { op1: lop1.clone(), op2: lop2 }
                                );
                                inplace_map.insert(*id, *nr1);
                                relin_id = lop1;
                            },

                            (HERef::NodeRef(_), HERef::NodeRef(nr2)) if !uses[id+1].contains(nr2) => {
                                instrs.push(
                                    HELoweredInstr::MulInplace { op1: lop2.clone(), op2: lop1 }
                                );
                                inplace_map.insert(*id, *nr2);
                                relin_id = lop2;
                            },

                            _ => {
                                instrs.push(
                                    HELoweredInstr::Mul { id: lid, op1: lop1, op2: lop2 }
                                )
                            }
                        }

                        // relinearize at every ciphertext-ciphertext multiplication,
                        // except for outputs, since these will not be used in the future
                        // so there's no need to minimize noise for them
                        // TODO: follow EVA and only relinearize if there will be
                        // a future *multiplication* that will use the result
                        if uses[id+1].contains(id) {
                            instrs.push(
                                HELoweredInstr::RelinearizeInplace { op1: relin_id }
                            )
                        }
                    },

                    (HEOperand::Ref(r1), HEOperand::ConstNum(_)) => {
                        match r1 {
                            HERef::NodeRef(nr1) if !uses[id+1].contains(nr1) => {
                                instrs.push(
                                    HELoweredInstr::MulPlainInplace { op1: lop1.clone(), op2: lop2 }
                                );
                                inplace_map.insert(*id, *nr1);
                                relin_id = lop1
                            },

                            _ => {
                                instrs.push(
                                    HELoweredInstr::MulPlain { id: lid.clone(), op1: lop1, op2: lop2 }
                                )
                            }
                        }
                    },

                    (HEOperand::ConstNum(_), HEOperand::Ref(r2)) => {
                        match r2 {
                            HERef::NodeRef(nr2) if !uses[id+1].contains(nr2) => {
                                instrs.push(
                                    HELoweredInstr::MulPlainInplace { op1: lop2.clone(), op2: lop1 }
                                );
                                inplace_map.insert(*id, *nr2);
                                relin_id = lop2
                            },

                            _ => {
                                instrs.push(
                                    HELoweredInstr::MulPlain { id: lid.clone(), op1: lop2, op2: lop1 }
                                )
                            }
                        }
                    },

                    (HEOperand::ConstNum(_), HEOperand::ConstNum(_)) => {
                        panic!("attempting to multiply two plaintexts---this should be constant folded")
                    }
                }
            },
            
            HEInstr::Rot { id, op1, op2 } => {
                let lid = format!("i{}", id);
                let lop1 = lower_operand(&inplace_map, &mut const_map, op1);
                match (op1, op2) {
                    (HEOperand::Ref(r1), HEOperand::ConstNum(cn2)) => {
                        let lop2 = cn2.to_string();
                        match r1 {
                            HERef::NodeRef(nr1) if !uses[id+1].contains(nr1) => {
                                instrs.push(
                                    HELoweredInstr::RotInplace { op1: lop1, op2: lop2 }
                                );
                                inplace_map.insert(*id, *nr1);
                            },

                            _ => {
                                instrs.push(
                                    HELoweredInstr::Rot { id: lid, op1: lop1, op2: lop2 }
                                );
                            }
                        }
                    }

                    _ => panic!("must rotate ciphertext with constant value")
                }
            }
        }
    }

    let constants: Vec<(i32, String)> = const_map.into_iter().collect();
    let symbols: HashSet<String> = prog.get_symbols();
    let output: HELoweredNodeId = match &instrs.last().unwrap() {
        HELoweredInstr::Add { id, op1, op2 } => id.clone(),
        HELoweredInstr::AddInplace { op1, op2 } => op1.clone(),
        HELoweredInstr::AddPlain { id, op1, op2 } => id.clone(),
        HELoweredInstr::AddPlainInplace { op1, op2 } => op1.clone(),
        HELoweredInstr::Mul { id, op1, op2 } => id.clone(),
        HELoweredInstr::MulInplace { op1, op2 } => op1.clone(),
        HELoweredInstr::MulPlain { id, op1, op2 } => id.clone(),
        HELoweredInstr::MulPlainInplace { op1, op2 } => op1.clone(),
        HELoweredInstr::Rot { id, op1, op2 } => id.clone(),
        HELoweredInstr::RotInplace { op1, op2 } => op1.clone(),
        HELoweredInstr::RelinearizeInplace { op1 } => op1.clone(),
    };
    HELoweredProgram { vec_size, constants, symbols, instrs, output }
}