/// lowered_program.rs
/// lowered program for generating output programs from templates

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::{circ::{*, lowering::program::*}, lang::HEClientStore};

/*
 
type SEALNodeId = String;
type SEALOperand = String;

#[derive(Clone, Deserialize, Serialize, Debug)]
#[serde(tag = "op")]
pub enum SEALInstr {
    Add { id: SEALNodeId, op1: SEALOperand, op2: SEALOperand },
    AddInplace { op1: SEALOperand, op2: SEALOperand },
    AddPlain { id: SEALOperand, op1: SEALOperand, op2: SEALOperand },
    AddPlainInplace { op1: SEALOperand, op2: SEALOperand },
    Sub { id: SEALNodeId, op1: SEALOperand, op2: SEALOperand },
    SubInplace { op1: SEALOperand, op2: SEALOperand },
    SubPlain { id: SEALOperand, op1: SEALOperand, op2: SEALOperand },
    SubPlainInplace { op1: SEALOperand, op2: SEALOperand },
    Negate { id: SEALNodeId, op1: SEALOperand },
    NegateInplace { op1: SEALOperand },
    Mul { id: SEALNodeId, op1: SEALOperand, op2: SEALOperand },
    MulInplace { op1: SEALOperand, op2: SEALOperand },
    MulPlain { id: SEALOperand, op1: SEALOperand, op2: SEALOperand },
    MulPlainInplace { op1: SEALOperand, op2: SEALOperand },
    Rot { id: SEALNodeId, op1: SEALOperand, op2: SEALOperand },
    RotInplace { op1: SEALOperand, op2: SEALOperand },
    RelinearizeInplace { op1: SEALNodeId },
}

impl SEALInstr {
    pub fn is_inplace(&self) -> bool {
        match self {
            SEALInstr::Add { id: _, op1: _, op2: _} => false,
            SEALInstr::AddInplace { op1: _, op2: _} => true,
            SEALInstr::AddPlain { id: _, op1: _, op2: _ } => false,
            SEALInstr::AddPlainInplace { op1: _, op2: _ } => true,
            SEALInstr::Sub { id: _, op1: _, op2: _} => false,
            SEALInstr::SubInplace { op1: _, op2: _} => true,
            SEALInstr::SubPlain { id: _, op1: _, op2: _ } => false,
            SEALInstr::SubPlainInplace { op1: _, op2: _ } => true,
            SEALInstr::Negate { id: _, op1: _ } => false,
            SEALInstr::NegateInplace { op1: _ } => true,
            SEALInstr::Mul { id: _, op1: _, op2: _ } => false,
            SEALInstr::MulInplace { op1: _, op2: _ } => true,
            SEALInstr::MulPlain { id: _, op1: _, op2: _ } => false,
            SEALInstr::MulPlainInplace { op1: _, op2: _ } => true,
            SEALInstr::Rot { id: _, op1: _, op2: _ } => false,
            SEALInstr::RotInplace { op1: _, op2: _ } => true,
            SEALInstr::RelinearizeInplace { op1: _ } => true
        }
    }

    pub fn is_binary(&self) -> bool {
        match self {
            SEALInstr::Add { id: _, op1: _, op2: _} => true,
            SEALInstr::AddInplace { op1: _, op2: _} => true,
            SEALInstr::AddPlain { id: _, op1: _, op2: _ } => true,
            SEALInstr::AddPlainInplace { op1: _, op2: _ } => true,
            SEALInstr::Sub { id: _, op1: _, op2: _} => true,
            SEALInstr::SubInplace { op1: _, op2: _} => true,
            SEALInstr::SubPlain { id: _, op1: _, op2: _ } => true,
            SEALInstr::SubPlainInplace { op1: _, op2: _ } => true,
            SEALInstr::Negate { id: _, op1: _ } => false,
            SEALInstr::NegateInplace { op1: _ } => false,
            SEALInstr::Mul { id: _, op1: _, op2: _ } => true,
            SEALInstr::MulInplace { op1: _, op2: _ } => true,
            SEALInstr::MulPlain { id: _, op1: _, op2: _ } => true,
            SEALInstr::MulPlainInplace { op1: _, op2: _ } => true,
            SEALInstr::Rot { id: _, op1: _, op2: _ } => true,
            SEALInstr::RotInplace { op1: _, op2: _ } => true,
            SEALInstr::RelinearizeInplace { op1: _ } => false
        }
    }

    pub fn name(&self) -> String {
        match self {
            SEALInstr::Add { id: _, op1: _, op2: _} => "add",
            SEALInstr::AddInplace { op1: _, op2: _} => "add_inplace",
            SEALInstr::AddPlain { id: _, op1: _, op2: _ } => "add_plain",
            SEALInstr::AddPlainInplace { op1: _, op2: _ } => "add_plain_inplace",
            SEALInstr::Sub { id: _, op1: _, op2: _} => "sub",
            SEALInstr::SubInplace { op1: _, op2: _} => "sub_inplace",
            SEALInstr::SubPlain { id: _, op1: _, op2: _ } => "sub_plain",
            SEALInstr::SubPlainInplace { op1: _, op2: _ } => "sub_plain_inplace",
            SEALInstr::Negate { id: _, op1: _ } => "negate",
            SEALInstr::NegateInplace { op1: _ } => "negate_inplace",
            SEALInstr::Mul { id: _, op1: _, op2: _ } => "multiply",
            SEALInstr::MulInplace { op1: _, op2: _ } => "multiply_inplace",
            SEALInstr::MulPlain { id: _, op1: _, op2: _ } => "multiply_plain",
            SEALInstr::MulPlainInplace { op1: _, op2: _ } => "multiply_plain_inplace",
            SEALInstr::Rot { id: _, op1: _, op2: _ } => "rotate_rows",
            SEALInstr::RotInplace { op1: _, op2: _ } => "rotate_rows_inplace",
            SEALInstr::RelinearizeInplace { op1: _ } => "relinearize_inplace"
        }.to_string()
    }

    pub fn operands(&self) -> Vec<SEALOperand> {
        match self {
            SEALInstr::Add { id: _, op1, op2 } |
            SEALInstr::AddInplace { op1, op2 } |
            SEALInstr::AddPlain { id: _, op1, op2 } |
            SEALInstr::AddPlainInplace { op1, op2 } |
            SEALInstr::Sub { id: _, op1, op2 } |
            SEALInstr::SubInplace { op1, op2 } |
            SEALInstr::SubPlain { id: _, op1, op2 } |
            SEALInstr::SubPlainInplace { op1, op2 } |
            SEALInstr::Mul { id: _, op1, op2 } |
            SEALInstr::MulInplace { op1, op2 } |
            SEALInstr::MulPlain { id: _, op1, op2 } |
            SEALInstr::MulPlainInplace { op1, op2 } |
            SEALInstr::Rot { id: _, op1, op2 } |
            SEALInstr::RotInplace { op1, op2 } =>
                vec![op1.to_string(),op2.to_string()],

            SEALInstr::Negate { id: _, op1 } |
            SEALInstr::NegateInplace { op1 } |
            SEALInstr::RelinearizeInplace { op1 } =>
                vec![op1.to_string()]
        }
    }

    fn id(&self) -> String {
        match self {
            SEALInstr::Add { id, op1: _, op2: _ } |
            SEALInstr::AddPlain { id, op1: _, op2: _ } |
            SEALInstr::Sub { id, op1: _, op2: _ } |
            SEALInstr::SubPlain { id, op1: _, op2: _ } |
            SEALInstr::Negate { id, op1: _ } |
            SEALInstr::Mul { id, op1: _, op2: _ } |
            SEALInstr::MulPlain { id, op1: _, op2: _ } |
            SEALInstr::Rot { id, op1: _, op2: _ } =>
                id.to_string(),

            _ => "".to_string(),
        }
    }
}

impl Display for SEALInstr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let operands_str = self.operands().join(", ");
        if self.is_inplace() {
            write!(f, "{}({})", self.name(), operands_str)

        } else {
            write!(f, "{} = {}({})", self.id(), self.name(), operands_str)
        }
    }
}

#[derive(Serialize)]
pub struct HELoweredProgram {
    pub vec_size: usize,
    pub literals: Vec<(isize, String)>,
    pub constants: Vec<(Vec<isize>, String)>,
    pub instrs: Vec<SEALInstr>,
    pub output: SEALNodeId,
    pub client_inputs: Vec<(String, im::Vector<usize>)>,
    pub server_inputs: Vec<(String, im::Vector<usize>)>,
    pub ciphertexts: HashSet<String>,
    pub client_preprocess: Vec<(String, String)>,
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

            HEOperand::Literal(n) => {
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
        let mut instrs: Vec<SEALInstr> = Vec::new();

        for instr in prog.instrs.iter() {
            match instr {
                HEInstruction::Add(id, op1, op2)=> {
                    let lid = format!("i{}", id);
                    let lop1 = Self::lower_operand(&inplace_map, &mut const_map, op1);
                    let lop2 = Self::lower_operand(&inplace_map, &mut const_map, op2);
                    match (op1, op2) {
                        (HEOperand::Ref(r1), HEOperand::Ref(r2)) => {
                            match (r1, r2) {
                                (HERef::Node(nr1), HERef::Node(_))
                                if !uses[id+1].contains(nr1) && !noinplace => {
                                    instrs.push(
                                        SEALInstr::AddInplace { op1: lop1, op2: lop2 }
                                    );
                                    inplace_map.insert(*id, *nr1);
                                },

                                (HERef::Node(_), HERef::Node(nr2))
                                if !uses[id+1].contains(nr2) && !noinplace => {
                                    instrs.push(
                                        SEALInstr::AddInplace { op1: lop2, op2: lop1 }
                                    );
                                    inplace_map.insert(*id, *nr2);
                                },

                                (HERef::Plaintext(_), HERef::Plaintext(_)) => {
                                    panic!("attempting to add two plaintexts: {:?} and {:?}", r1, r2)
                                },

                                (_, HERef::Plaintext(_)) => {
                                    instrs.push(
                                        SEALInstr::AddPlain { id: lid, op1: lop1, op2: lop2 }
                                    )
                                },

                                (HERef::Plaintext(_), _) => {
                                    instrs.push(
                                        SEALInstr::AddPlain { id: lid, op1: lop2, op2: lop1 }
                                    )
                                },

                                _ => {
                                    instrs.push(
                                        SEALInstr::Add { id: lid, op1: lop1, op2: lop2 }
                                    )
                                }
                            }
                        },

                        (HEOperand::Ref(r1), HEOperand::Literal(_)) => {
                            match r1 {
                                HERef::Node(nr1)
                                if !uses[id+1].contains(nr1) && !noinplace => {
                                    instrs.push(
                                        SEALInstr::AddPlainInplace { op1: lop1, op2: lop2 }
                                    );
                                    inplace_map.insert(*id, *nr1);
                                },

                                _ => {
                                    instrs.push(
                                        SEALInstr::AddPlain { id: lid, op1: lop1, op2: lop2 }
                                    )
                                }
                            }
                        },

                        (HEOperand::Literal(_), HEOperand::Ref(r2)) => {
                            match r2 {
                                HERef::Node(nr2)
                                if !uses[id+1].contains(nr2) && !noinplace => {
                                    instrs.push(
                                        SEALInstr::AddPlainInplace { op1: lop2, op2: lop1 }
                                    );
                                    inplace_map.insert(*id, *nr2);
                                },

                                _ => {
                                    instrs.push(
                                        SEALInstr::AddPlain { id: lid, op1: lop2, op2: lop1 }
                                    )
                                }
                            }
                        },

                        (HEOperand::Literal(_), HEOperand::Literal(_)) => {
                            panic!("attempting to add two constants---this should be constant folded")
                        }
                    }
                },

                HEInstruction::Sub(id, op1, op2) => {
                    let lid = format!("i{}", id);
                    let lop1 = Self::lower_operand(&inplace_map, &mut const_map, op1);
                    let lop2 = Self::lower_operand(&inplace_map, &mut const_map, op2);
                    match (op1, op2) {
                        (HEOperand::Ref(r1), HEOperand::Ref(r2)) => {
                            match (r1, r2) {
                                (HERef::Node(nr1), HERef::Node(_))
                                if !uses[id+1].contains(nr1) && !noinplace => {
                                    instrs.push(
                                        SEALInstr::SubInplace { op1: lop1, op2: lop2 }
                                    );
                                    inplace_map.insert(*id, *nr1);
                                },

                                (HERef::Plaintext(_), HERef::Plaintext(_)) => {
                                    panic!("attempting to subtract two plaintexts: {:?} and {:?}", r1, r2)
                                },

                                (_, HERef::Plaintext(_)) => {
                                    instrs.push(
                                        SEALInstr::SubPlain { id: lid, op1: lop1, op2: lop2 }
                                    )
                                },

                                (HERef::Plaintext(_), _) => {
                                    instrs.push(
                                        SEALInstr::Negate { id: lid.clone(), op1: lop2 }
                                    );
                                    instrs.push(
                                        SEALInstr::AddPlainInplace { op1: lid, op2: lop1 }
                                    )
                                },

                                _ => {
                                    instrs.push(
                                        SEALInstr::Sub { id: lid, op1: lop1, op2: lop2 }
                                    )
                                }
                            }
                        },

                        (HEOperand::Ref(r1), HEOperand::Literal(_)) => {
                            match r1 {
                                HERef::Node(nr1)
                                if !uses[id+1].contains(nr1) && !noinplace => {
                                    instrs.push(
                                        SEALInstr::SubPlainInplace { op1: lop1, op2: lop2 }
                                    );
                                    inplace_map.insert(*id, *nr1);
                                },

                                _ => {
                                    instrs.push(
                                        SEALInstr::SubPlain { id: lid, op1: lop1, op2: lop2 }
                                    )
                                }
                            }
                        },

                        (HEOperand::Literal(_), HEOperand::Ref(r2)) => {
                            match r2 {
                                HERef::Node(nr2)
                                if !uses[id+1].contains(nr2) && !noinplace => {
                                    instrs.push(
                                        SEALInstr::NegateInplace { op1: lop2.clone() }
                                    );
                                    instrs.push(
                                        SEALInstr::AddPlainInplace { op1: lop2, op2: lop1 }
                                    );
                                    inplace_map.insert(*id, *nr2);
                                },

                                _ => {
                                    instrs.push(
                                        SEALInstr::Negate { id: lid.clone(), op1: lop2.clone() }
                                    );
                                    instrs.push(
                                        SEALInstr::AddPlainInplace { op1: lid, op2: lop1 }
                                    )
                                }
                            }
                        },

                        (HEOperand::Literal(_), HEOperand::Literal(_)) => {
                            panic!("attempting to subtract two constants---this should be constant folded")
                        }
                    }
                },

                HEInstruction::Mul(id, op1, op2) => {
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
                                        SEALInstr::MulInplace { op1: lop1.clone(), op2: lop2 }
                                    );
                                    inplace_map.insert(*id, *nr1);
                                    relin_id = lop1;
                                    cipher_cipher_op = true;
                                },

                                (HERef::Node(_), HERef::Node(nr2))
                                if !uses[id+1].contains(nr2) && !noinplace => {
                                    instrs.push(
                                        SEALInstr::MulInplace { op1: lop2.clone(), op2: lop1 }
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
                                        SEALInstr::MulPlain { id: lid, op1: lop1, op2: lop2 }
                                    )
                                },

                                (HERef::Plaintext(_), _) => {
                                    instrs.push(
                                        SEALInstr::MulPlain { id: lid, op1: lop2, op2: lop1 }
                                    )
                                },

                                _ => {
                                    instrs.push(
                                        SEALInstr::Mul { id: lid, op1: lop1, op2: lop2 }
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
                                    SEALInstr::RelinearizeInplace { op1: relin_id }
                                )
                            }
                        },

                        (HEOperand::Ref(r1), HEOperand::Literal(_)) => {
                            match r1 {
                                HERef::Node(nr1)
                                if !uses[id+1].contains(nr1) && !noinplace => {
                                    instrs.push(
                                        SEALInstr::MulPlainInplace { op1: lop1.clone(), op2: lop2 }
                                    );
                                    inplace_map.insert(*id, *nr1);
                                },

                                _ => {
                                    instrs.push(
                                        SEALInstr::MulPlain { id: lid.clone(), op1: lop1, op2: lop2 }
                                    )
                                }
                            }
                        },

                        (HEOperand::Literal(_), HEOperand::Ref(r2)) => {
                            match r2 {
                                HERef::Node(nr2)
                                if !uses[id+1].contains(nr2) && !noinplace => {
                                    instrs.push(
                                        SEALInstr::MulPlainInplace { op1: lop2.clone(), op2: lop1 }
                                    );
                                    inplace_map.insert(*id, *nr2);
                                },

                                _ => {
                                    instrs.push(
                                        SEALInstr::MulPlain { id: lid.clone(), op1: lop2, op2: lop1 }
                                    )
                                }
                            }
                        },

                        (HEOperand::Literal(_), HEOperand::Literal(_)) => {
                            panic!("attempting to multiply two constants---this should be constant folded")
                        }
                    }
                },
                
                HEInstruction::Rot(id, op1, op2) => {
                    let lid = format!("i{}", id);
                    let lop1 = Self::lower_operand(&inplace_map, &mut const_map, op1);
                    match (op1, op2) {
                        (HEOperand::Ref(r1), HEOperand::Literal(cn2)) => {
                            let lop2 = cn2.to_string();
                            match r1 {
                                HERef::Node(nr1)
                                if !uses[id+1].contains(nr1) && !noinplace => {
                                    instrs.push(
                                        SEALInstr::RotInplace { op1: lop1, op2: lop2 }
                                    );
                                    inplace_map.insert(*id, *nr1);
                                },

                                HERef::Plaintext(_) =>  {
                                    panic!("attempting to rotate plaintext {:?}", r1)
                                }

                                _ => {
                                    instrs.push(
                                        SEALInstr::Rot { id: lid, op1: lop1, op2: lop2 }
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

        let output: SEALNodeId = match &instrs.last().unwrap() {
            SEALInstr::Add { id, op1, op2 } => id.clone(),
            SEALInstr::AddInplace { op1, op2 } => op1.clone(),
            SEALInstr::AddPlain { id, op1, op2 } => id.clone(),
            SEALInstr::AddPlainInplace { op1, op2 } => op1.clone(),
            SEALInstr::Sub { id, op1, op2 } => id.clone(),
            SEALInstr::SubInplace { op1, op2 } => op1.clone(),
            SEALInstr::SubPlain { id, op1, op2 } => id.clone(),
            SEALInstr::SubPlainInplace { op1, op2 } => op1.clone(),
            SEALInstr::Negate { id, op1 } => id.clone(),
            SEALInstr::NegateInplace { op1 } => op1.clone(),
            SEALInstr::Mul { id, op1, op2 } => id.clone(),
            SEALInstr::MulInplace { op1, op2 } => op1.clone(),
            SEALInstr::MulPlain { id, op1, op2 } => id.clone(),
            SEALInstr::MulPlainInplace { op1, op2 } => op1.clone(),
            SEALInstr::Rot { id, op1, op2 } => id.clone(),
            SEALInstr::RotInplace { op1, op2 } => op1.clone(),
            SEALInstr::RelinearizeInplace { op1 } => op1.clone(),
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
*/