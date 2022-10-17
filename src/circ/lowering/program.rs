/// program.rs
/// instruction representation of HE programs

use egg::*;
use rand::Rng;
use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::RangeInclusive;

use crate::circ::optimizer::HELatencyModel;
use crate::circ::{*, optimizer, optimizer::HEOptCircuit};

pub(crate) type NodeId = usize;

#[derive(Clone, Debug)]
pub(crate) enum HERef {
    Node(NodeId),
    Ciphertext(HEObjectName),
    Plaintext(HEObjectName),
}

#[derive(Clone, Debug)]
pub(crate) enum HEOperand {
    Ref(HERef),
    ConstNum(isize),
}

impl fmt::Display for HEOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HEOperand::Ref(HERef::Node(i)) => {
                write!(f, "i{}", i)
            },
            HEOperand::Ref(HERef::Ciphertext(sym)) => {
                write!(f, "{}", sym)
            },
            HEOperand::Ref(HERef::Plaintext(sym)) => {
                write!(f, "{}", sym)
            },
            HEOperand::ConstNum(n) => {
                write!(f, "{}", n)
            },
        }
    }
}

#[derive(Clone)]
pub(crate) enum HEInstruction {
    Add { id: NodeId, op1: HEOperand, op2: HEOperand },
    Sub { id: NodeId, op1: HEOperand, op2: HEOperand },
    Mul{ id: NodeId, op1: HEOperand, op2: HEOperand },
    Rot { id: NodeId, op1: HEOperand, op2: HEOperand} ,
}

impl HEInstruction {
    fn get_operands(&self) -> [&HEOperand;2] {
        match self {
            HEInstruction::Add { id: _, op1, op2 } |
            HEInstruction::Sub { id: _, op1, op2 } |
            HEInstruction::Mul { id: _, op1, op2 } |
            HEInstruction::Rot { id: _, op1, op2 } =>
                [op1, op2],
        }
    }
}

impl fmt::Display for HEInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HEInstruction::Add { id: _, op1, op2 } =>
                write!(f, "{} + {}", op1, op2),

            HEInstruction::Sub { id: _, op1, op2 } =>
                write!(f, "{} - {}", op1, op2),

            HEInstruction::Mul { id: _, op1, op2 } =>
                write!(f, "{} * {}", op1, op2),

            HEInstruction::Rot { id: _, op1, op2 } =>
                write!(f, "rot {} {}", op1, op2),
        }
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
                HEOperand::Ref(HERef::Ciphertext(_)) => 0,
                HEOperand::Ref(HERef::Plaintext(_)) => 0,
                HEOperand::ConstNum(_) => 0,
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
                            (HEOperand::ConstNum(_), _) | (_, HEOperand::ConstNum(_)) =>
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
                        (HEOperand::ConstNum(_), _) | (_, HEOperand::ConstNum(_)) =>  {
                            latency += model.add_plain;
                        },

                        _ => {
                            latency += model.add
                        }
                    }
                },

                HEInstruction::Sub { id: _, op1, op2 } => {
                    match (op1, op2) {
                        (HEOperand::ConstNum(_), _) | (_, HEOperand::ConstNum(_)) =>  {
                            latency += model.add_plain;
                        },

                        _ => {
                            latency += model.add;
                        }
                    }
                },

                HEInstruction::Mul { id: _, op1, op2 } => {
                    match (op1, op2) {
                        (HEOperand::ConstNum(_), _) | (_, HEOperand::ConstNum(_)) =>  {
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
                if let HEOperand::Ref(HERef::Ciphertext(sym)) = op{
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
                if let HEOperand::Ref(HERef::Plaintext(sym)) = op {
                    symset.insert(sym.to_string());
                }
            }
        }

        symset
    }

    /// generate a random sym store for this program
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
                    node_map.insert(id, HEOperand::ConstNum(*n));
                }

                HEOptCircuit::CiphertextRef(sym) => {
                    node_map.insert(id, HEOperand::Ref(HERef::Ciphertext(sym.to_string())));
                }

                HEOptCircuit::PlaintextRef(sym) => {
                    node_map.insert(id, HEOperand::Ref(HERef::Plaintext(sym.to_string())));
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