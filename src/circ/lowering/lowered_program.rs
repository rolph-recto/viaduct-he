/// lowered_program.rs
/// lowered program for generating output programs from templates

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::{circ::{*, lowering::program::*}, lang::HEClientStore};
 
type HELoweredNodeId = String;
type HELoweredOperand = String;

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "op")]
pub enum HELoweredInstr {
    Add { id: HELoweredNodeId, op1: HELoweredOperand, op2: HELoweredOperand },
    AddInplace { op1: HELoweredOperand, op2: HELoweredOperand },
    AddPlain { id: HELoweredOperand, op1: HELoweredOperand, op2: HELoweredOperand },
    AddPlainInplace { op1: HELoweredOperand, op2: HELoweredOperand },
    Sub { id: HELoweredNodeId, op1: HELoweredOperand, op2: HELoweredOperand },
    SubInplace { op1: HELoweredOperand, op2: HELoweredOperand },
    SubPlain { id: HELoweredOperand, op1: HELoweredOperand, op2: HELoweredOperand },
    SubPlainInplace { op1: HELoweredOperand, op2: HELoweredOperand },
    Negate { id: HELoweredNodeId, op1: HELoweredOperand },
    NegateInplace { op1: HELoweredOperand },
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
    literals: Vec<(isize, String)>,
    constants: Vec<(Vec<isize>, String)>,
    instrs: Vec<HELoweredInstr>,
    output: HELoweredNodeId,
    client_inputs: Vec<(String, im::Vector<usize>)>,
    server_inputs: Vec<(String, im::Vector<usize>)>,
    ciphertexts: HashSet<String>,
    client_preprocess: Vec<(String, String)>,
}

impl HELoweredProgram {
    fn resolve_inplace_indirection(inplace_map: &HashMap<NodeId,NodeId>, id: NodeId) -> NodeId {
        let mut cur_id = id;
        while inplace_map.contains_key(&cur_id) {
            cur_id = inplace_map[&cur_id];
        }
        cur_id
    }

    fn lower_operand(inplace_map: &HashMap<NodeId,NodeId>, const_map: &mut HashMap<isize, String>, op: &HEOperand) -> String {
        match op {
            HEOperand::Ref(HERef::Node(nr)) => {
                let resolved_nr = Self::resolve_inplace_indirection(inplace_map, *nr);
                format!("i{}", resolved_nr)
            },

            HEOperand::Ref(HERef::Ciphertext(sym)) |
            HEOperand::Ref(HERef::Plaintext(sym)) => {
                sym.clone()
            },

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

    pub fn lower_program(
        vec_size: usize,
        noinplace: bool,
        prog: &HEProgram,
        store: &HECircuitStore,
        client_store: HEClientStore,
        server_inputs: HashMap<HEObjectName,Dimensions>,
        client_inputs: HashMap<HEObjectName,Dimensions>,
    ) -> HELoweredProgram {
        let uses = prog.analyze_use();
        let mut inplace_map: HashMap<NodeId, NodeId> = HashMap::new();
        let mut const_map: HashMap<isize, String> = HashMap::new();
        let mut instrs: Vec<HELoweredInstr> = Vec::new();

        for instr in prog.instrs.iter() {
            match instr {
                HEInstruction::Add { id, op1, op2 } => {
                    let lid = format!("i{}", id);
                    let lop1 = Self::lower_operand(&inplace_map, &mut const_map, op1);
                    let lop2 = Self::lower_operand(&inplace_map, &mut const_map, op2);
                    match (op1, op2) {
                        (HEOperand::Ref(r1), HEOperand::Ref(r2)) => {
                            match (r1, r2) {
                                (HERef::Node(nr1), HERef::Node(_))
                                if !uses[id+1].contains(nr1) && !noinplace => {
                                    instrs.push(
                                        HELoweredInstr::AddInplace { op1: lop1, op2: lop2 }
                                    );
                                    inplace_map.insert(*id, *nr1);
                                },

                                (HERef::Node(_), HERef::Node(nr2))
                                if !uses[id+1].contains(nr2) && !noinplace => {
                                    instrs.push(
                                        HELoweredInstr::AddInplace { op1: lop2, op2: lop1 }
                                    );
                                    inplace_map.insert(*id, *nr2);
                                },

                                (HERef::Plaintext(_), HERef::Plaintext(_)) => {
                                    panic!("attempting to add two plaintexts: {:?} and {:?}", r1, r2)
                                },

                                (_, HERef::Plaintext(_)) => {
                                    instrs.push(
                                        HELoweredInstr::AddPlain { id: lid, op1: lop1, op2: lop2 }
                                    )
                                },

                                (HERef::Plaintext(_), _) => {
                                    instrs.push(
                                        HELoweredInstr::AddPlain { id: lid, op1: lop2, op2: lop1 }
                                    )
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
                                HERef::Node(nr1)
                                if !uses[id+1].contains(nr1) && !noinplace => {
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
                                HERef::Node(nr2)
                                if !uses[id+1].contains(nr2) && !noinplace => {
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
                            panic!("attempting to add two constants---this should be constant folded")
                        }
                    }
                },

                HEInstruction::Sub { id, op1, op2 } => {
                    let lid = format!("i{}", id);
                    let lop1 = Self::lower_operand(&inplace_map, &mut const_map, op1);
                    let lop2 = Self::lower_operand(&inplace_map, &mut const_map, op2);
                    match (op1, op2) {
                        (HEOperand::Ref(r1), HEOperand::Ref(r2)) => {
                            match (r1, r2) {
                                (HERef::Node(nr1), HERef::Node(_))
                                if !uses[id+1].contains(nr1) && !noinplace => {
                                    instrs.push(
                                        HELoweredInstr::SubInplace { op1: lop1, op2: lop2 }
                                    );
                                    inplace_map.insert(*id, *nr1);
                                },

                                (HERef::Plaintext(_), HERef::Plaintext(_)) => {
                                    panic!("attempting to subtract two plaintexts: {:?} and {:?}", r1, r2)
                                },

                                (_, HERef::Plaintext(_)) => {
                                    instrs.push(
                                        HELoweredInstr::SubPlain { id: lid, op1: lop1, op2: lop2 }
                                    )
                                },

                                (HERef::Plaintext(_), _) => {
                                    instrs.push(
                                        HELoweredInstr::Negate { id: lid.clone(), op1: lop2 }
                                    );
                                    instrs.push(
                                        HELoweredInstr::AddPlainInplace { op1: lid, op2: lop1 }
                                    )
                                },

                                _ => {
                                    instrs.push(
                                        HELoweredInstr::Sub { id: lid, op1: lop1, op2: lop2 }
                                    )
                                }
                            }
                        },

                        (HEOperand::Ref(r1), HEOperand::ConstNum(_)) => {
                            match r1 {
                                HERef::Node(nr1)
                                if !uses[id+1].contains(nr1) && !noinplace => {
                                    instrs.push(
                                        HELoweredInstr::SubPlainInplace { op1: lop1, op2: lop2 }
                                    );
                                    inplace_map.insert(*id, *nr1);
                                },

                                _ => {
                                    instrs.push(
                                        HELoweredInstr::SubPlain { id: lid, op1: lop1, op2: lop2 }
                                    )
                                }
                            }
                        },

                        (HEOperand::ConstNum(_), HEOperand::Ref(r2)) => {
                            match r2 {
                                HERef::Node(nr2)
                                if !uses[id+1].contains(nr2) && !noinplace => {
                                    instrs.push(
                                        HELoweredInstr::NegateInplace { op1: lop2.clone() }
                                    );
                                    instrs.push(
                                        HELoweredInstr::AddPlainInplace { op1: lop2, op2: lop1 }
                                    );
                                    inplace_map.insert(*id, *nr2);
                                },

                                _ => {
                                    instrs.push(
                                        HELoweredInstr::Negate { id: lid.clone(), op1: lop2.clone() }
                                    );
                                    instrs.push(
                                        HELoweredInstr::AddPlainInplace { op1: lid, op2: lop1 }
                                    )
                                }
                            }
                        },

                        (HEOperand::ConstNum(_), HEOperand::ConstNum(_)) => {
                            panic!("attempting to subtract two constants---this should be constant folded")
                        }
                    }
                },

                HEInstruction::Mul { id, op1, op2 } => {
                    let lid = format!("i{}", id);
                    let lop1 = Self::lower_operand(&inplace_map, &mut const_map, op1);
                    let lop2 = Self::lower_operand(&inplace_map, &mut const_map, op2);
                    let mut relin_id = lid.clone();
                    match (op1, op2) {
                        (HEOperand::Ref(r1), HEOperand::Ref(r2)) => {
                            let mut cipher_cipher_op = false;
                            match (r1, r2) {
                                (HERef::Node(nr1), HERef::Node(_))
                                if !uses[id+1].contains(nr1) && !noinplace => {
                                    instrs.push(
                                        HELoweredInstr::MulInplace { op1: lop1.clone(), op2: lop2 }
                                    );
                                    inplace_map.insert(*id, *nr1);
                                    relin_id = lop1;
                                    cipher_cipher_op = true;
                                },

                                (HERef::Node(_), HERef::Node(nr2))
                                if !uses[id+1].contains(nr2) && !noinplace => {
                                    instrs.push(
                                        HELoweredInstr::MulInplace { op1: lop2.clone(), op2: lop1 }
                                    );
                                    inplace_map.insert(*id, *nr2);
                                    relin_id = lop2;
                                    cipher_cipher_op = true;
                                },

                                (HERef::Plaintext(_), HERef::Plaintext(_)) => {
                                    panic!("attempting to add two plaintexts: {:?} and {:?}", r1, r2)
                                },

                                (_, HERef::Plaintext(_)) => {
                                    instrs.push(
                                        HELoweredInstr::MulPlain { id: lid, op1: lop1, op2: lop2 }
                                    )
                                },

                                (HERef::Plaintext(_), _) => {
                                    instrs.push(
                                        HELoweredInstr::MulPlain { id: lid, op1: lop2, op2: lop1 }
                                    )
                                },

                                _ => {
                                    instrs.push(
                                        HELoweredInstr::Mul { id: lid, op1: lop1, op2: lop2 }
                                    );
                                    cipher_cipher_op = true;
                                }
                            }

                            // relinearize at every ciphertext-ciphertext multiplication,
                            // except for outputs, since these will not be used in the future
                            // so there's no need to minimize noise for them
                            // TODO: follow EVA and only relinearize if there will be
                            // a future *multiplication* that will use the result
                            if cipher_cipher_op && uses[id+1].contains(id) {
                                instrs.push(
                                    HELoweredInstr::RelinearizeInplace { op1: relin_id }
                                )
                            }
                        },

                        (HEOperand::Ref(r1), HEOperand::ConstNum(_)) => {
                            match r1 {
                                HERef::Node(nr1)
                                if !uses[id+1].contains(nr1) && !noinplace => {
                                    instrs.push(
                                        HELoweredInstr::MulPlainInplace { op1: lop1.clone(), op2: lop2 }
                                    );
                                    inplace_map.insert(*id, *nr1);
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
                                HERef::Node(nr2)
                                if !uses[id+1].contains(nr2) && !noinplace => {
                                    instrs.push(
                                        HELoweredInstr::MulPlainInplace { op1: lop2.clone(), op2: lop1 }
                                    );
                                    inplace_map.insert(*id, *nr2);
                                },

                                _ => {
                                    instrs.push(
                                        HELoweredInstr::MulPlain { id: lid.clone(), op1: lop2, op2: lop1 }
                                    )
                                }
                            }
                        },

                        (HEOperand::ConstNum(_), HEOperand::ConstNum(_)) => {
                            panic!("attempting to multiply two constants---this should be constant folded")
                        }
                    }
                },
                
                HEInstruction::Rot { id, op1, op2 } => {
                    let lid = format!("i{}", id);
                    let lop1 = Self::lower_operand(&inplace_map, &mut const_map, op1);
                    match (op1, op2) {
                        (HEOperand::Ref(r1), HEOperand::ConstNum(cn2)) => {
                            let lop2 = cn2.to_string();
                            match r1 {
                                HERef::Node(nr1)
                                if !uses[id+1].contains(nr1) && !noinplace => {
                                    instrs.push(
                                        HELoweredInstr::RotInplace { op1: lop1, op2: lop2 }
                                    );
                                    inplace_map.insert(*id, *nr1);
                                },

                                HERef::Plaintext(_) =>  {
                                    panic!("attempting to rotate plaintext {:?}", r1)
                                }

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

        let literals: Vec<(isize, String)> = const_map.into_iter().collect();
        let ciphertexts: HashSet<String> = prog.get_ciphertext_symbols();

        let constants: Vec<(Vec<isize>, String)> =
            prog.get_plaintext_symbols().into_iter().map(|sym|
                match store.plaintexts.get(&sym) {
                    Some(Plaintext { dimensions: _, value }) => {
                        (Vec::from_iter(value.clone().into_iter()), sym)
                    },
                    _ => {
                        panic!("symbol {} does not map into a plaintext object", &sym)
                    },
                }
            ).collect();

        let output: HELoweredNodeId = match &instrs.last().unwrap() {
            HELoweredInstr::Add { id, op1, op2 } => id.clone(),
            HELoweredInstr::AddInplace { op1, op2 } => op1.clone(),
            HELoweredInstr::AddPlain { id, op1, op2 } => id.clone(),
            HELoweredInstr::AddPlainInplace { op1, op2 } => op1.clone(),
            HELoweredInstr::Sub { id, op1, op2 } => id.clone(),
            HELoweredInstr::SubInplace { op1, op2 } => op1.clone(),
            HELoweredInstr::SubPlain { id, op1, op2 } => id.clone(),
            HELoweredInstr::SubPlainInplace { op1, op2 } => op1.clone(),
            HELoweredInstr::Negate { id, op1 } => id.clone(),
            HELoweredInstr::NegateInplace { op1 } => op1.clone(),
            HELoweredInstr::Mul { id, op1, op2 } => id.clone(),
            HELoweredInstr::MulInplace { op1, op2 } => op1.clone(),
            HELoweredInstr::MulPlain { id, op1, op2 } => id.clone(),
            HELoweredInstr::MulPlainInplace { op1, op2 } => op1.clone(),
            HELoweredInstr::Rot { id, op1, op2 } => id.clone(),
            HELoweredInstr::RotInplace { op1, op2 } => op1.clone(),
            HELoweredInstr::RelinearizeInplace { op1 } => op1.clone(),
        };
        HELoweredProgram {
            vec_size, literals, constants, instrs, output,
            server_inputs:
                server_inputs.into_iter()
                .map(|(k,v)| (k, v.as_vec().clone()))
                .collect(),
            client_inputs:
                client_inputs.into_iter()
                .map(|(k,v)| (k, v.as_vec().clone()))
                .collect(),
            ciphertexts,
            client_preprocess:
                client_store.into_iter()
                .map(|(k,v)| (k, v.as_python_str()))
                .collect(),
        }
    }
}