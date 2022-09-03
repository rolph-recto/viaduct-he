use egg::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::max;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::RangeInclusive;

define_language! {
    /// The language used by egg e-graph engine.
    pub(crate) enum HE {
        Num(i32),
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        "rot" = Rot([Id; 2]),
        Symbol(Symbol),
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub(crate) enum HERef {
    NodeRef(usize),
    ConstSym(String)
}

#[derive(Clone, Deserialize, Serialize)]
pub(crate) enum HEOperand {
    Ref(HERef),
    ConstNum(i32),
}

impl fmt::Display for HEOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}",
            match self {
                HEOperand::Ref(HERef::NodeRef(i)) => format!("i{}", i),
                HEOperand::Ref(HERef::ConstSym(sym)) => sym.to_string(),
                HEOperand::ConstNum(n) => n.to_string(),
            }
        )
    }
}

#[derive(Clone)]
pub(crate) enum HEInstr {
    Add { index: usize, op1: HEOperand, op2: HEOperand },
    Mul{ index: usize, op1: HEOperand, op2: HEOperand },
    Rot { index: usize, op1: HEOperand, op2: HEOperand} ,
}

impl HEInstr {
    fn get_operands(&self) -> [&HEOperand; 2] {
        match self {
            HEInstr::Add { index: _, op1, op2 }|
            HEInstr::Mul { index: _, op1, op2 } |
            HEInstr::Rot { index: _, op1, op2 } =>
                [op1, op2],
        }
    }
}

impl fmt::Display for HEInstr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HEInstr::Add { index, op1, op2 } =>
                write!(f, "{} + {}", op1, op2),

            HEInstr::Mul { index, op1, op2 } =>
                write!(f, "{} * {}", op1, op2),

            HEInstr::Rot { index, op1, op2 } =>
                write!(f, "rot {} {}", op1, op2),
        }
    }
}

pub(crate) struct HEProgram {
    instrs: Vec<HEInstr>,
}

#[derive(PartialEq, Eq, Clone)]
pub(crate) enum HEValue {
    HEScalar(i32),
    HEVector(Vec<i32>),
}

impl fmt::Display for HEValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HEValue::HEScalar(s) => write!(f, "{}", s),
            HEValue::HEVector(v) => write!(f, "{:?}", v),
        }
    }
}

pub(crate) type HESymStore = HashMap<String, HEValue>;
pub(crate) type HERefStore = HashMap<usize, HEValue>;

impl HEProgram {
    /// calculate the multiplicative depth of the program.
    pub(crate) fn get_muldepth(&self) -> usize {
        let mut max_depth: usize = 0;
        let mut depth_list: Vec<usize> = vec![];
        let mut i: usize = 0;

        let get_opdepth = |dlist: &Vec<usize>, op: &HEOperand| -> usize {
            match op {
                HEOperand::Ref(HERef::NodeRef(r)) => dlist[*r],
                HEOperand::Ref(HERef::ConstSym(_)) => 0,
                HEOperand::ConstNum(_) => 0,
            }
        };

        for instr in self.instrs.iter() {
            let dlist: &Vec<usize> = &depth_list;
            let depth: usize =
                match instr {
                    HEInstr::Add { index, op1, op2 } => {
                        let op1_depth = get_opdepth(dlist, op1);
                        let op2_depth: usize = get_opdepth(dlist, op2);
                        max(op1_depth, op2_depth)
                    },
                    HEInstr::Mul { index, op1, op2 } => {
                        let op1_depth = get_opdepth(dlist, op1);
                        let op2_depth: usize = get_opdepth(dlist, op2);

                        match (op1, op2) {
                            (HEOperand::ConstNum(_), _) | (_, HEOperand::ConstNum(_)) =>
                                max(op1_depth, op2_depth),

                            _ => max(op1_depth, op2_depth) + 1
                        }
                    },
                    HEInstr::Rot { index, op1, op2 } => {
                        let op1_depth = get_opdepth(dlist, op1);
                        let op2_depth: usize = get_opdepth(dlist, op2);
                        max(op1_depth, op2_depth)
                    },
                };

            if depth > max_depth {
                max_depth = depth;
            }
            depth_list.push(depth);
            i += 1;
        }
        max_depth
    }

    /// calculate the latency of a program
    pub fn get_latency(&self) -> f64 {
        let mut latency = 0.0;
        for instr in self.instrs.iter() {
            match instr {
                HEInstr::Add { index, op1, op2 } => {
                    match (op1, op2) {
                        (HEOperand::ConstNum(_), _) | (_, HEOperand::ConstNum(_)) =>  {
                            latency += crate::optimizer::ADD_PLAIN_LATENCY
                        },

                        _ => {
                            latency += crate::optimizer::ADD_LATENCY
                        }
                    }
                },

                HEInstr::Mul { index, op1, op2 } => {
                    match (op1, op2) {
                        (HEOperand::ConstNum(_), _) | (_, HEOperand::ConstNum(_)) =>  {
                            latency += crate::optimizer::MUL_PLAIN_LATENCY
                        },

                        _ => {
                            latency += crate::optimizer::MUL_LATENCY
                        }
                    }
                },
                
                HEInstr::Rot { index, op1, op2 } => {
                    latency += crate::optimizer::ROT_LATENCY
                }
            }
        }
        latency
    }

    /// get the symbols used in this program.
    fn get_symbols(&self) -> HashSet<String> {
        let mut symset = HashSet::new();

        for instr in self.instrs.iter() {
            for op in instr.get_operands() {
                if let HEOperand::Ref(HERef::ConstSym(sym)) = op {
                    symset.insert(sym.to_string());
                }
            }
        }

        symset
    }

    /// generate a random sym store for this program
    pub(crate) fn gen_sym_store(&self, vec_size: usize, range: RangeInclusive<i32>) -> HESymStore {
        let symbols = self.get_symbols();
        let mut sym_store: HESymStore = HashMap::new();
        let mut rng = rand::thread_rng();

        for symbol in symbols {
            let new_vec: Vec<i32> = (0..vec_size)
                .into_iter()
                .map(|_| rng.gen_range(range.clone()))
                .collect();

            sym_store.insert(symbol.to_string(), HEValue::HEVector(new_vec));
        }

        sym_store
    }
}

impl fmt::Display for HEProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.instrs
            .iter()
            .enumerate()
            .map(|(i, instr)| write!(f, "let i{} = {}\n", i + 1, instr))
            .collect()
    }
}

pub(crate) fn gen_program(expr: &RecExpr<HE>) -> HEProgram {
    let mut node_map: HashMap<Id, HEOperand> = HashMap::new();
    let mut program: HEProgram = HEProgram { instrs: Vec::new() };
    let mut cur_instr: usize = 0;

    let mut op_processor =
        |nmap: &mut HashMap<Id, HEOperand>, id: Id,
        ctor: fn(usize, HEOperand, HEOperand) -> HEInstr,
        id_op1: &Id, id_op2: &Id|
    {
        let op1 = &nmap[id_op1];
        let op2 = &nmap[id_op2];

        program.instrs.push(ctor(cur_instr, op1.clone(), op2.clone()));
        nmap.insert(id, HEOperand::Ref(HERef::NodeRef(cur_instr)));

        cur_instr += 1;
    };

    for (i, node) in expr.as_ref().iter().enumerate() {
        let id = Id::from(i);
        match node {
            HE::Num(n) => {
                node_map.insert(id, HEOperand::ConstNum(*n));
            }

            HE::Symbol(sym) => {
                node_map.insert(id, HEOperand::Ref(HERef::ConstSym(sym.to_string())));
            }

            HE::Add([id1, id2]) => {
                op_processor(
                    &mut node_map, id, 
                    |index, op1, op2| {
                        HEInstr::Add { index, op1, op2 }
                    },
                    id1, id2);
            }

            HE::Mul([id1, id2]) => {
                op_processor(
                    &mut node_map, id, 
                    |index, op1, op2| HEInstr::Mul { index, op1, op2 }, 
                    id1, id2);
            }

            HE::Rot([id1, id2]) => {
                op_processor(
                    &mut node_map, id, 
                    |index, op1, op2| HEInstr::Rot { index, op1, op2 }, 
                    id1, id2);
            }
        }
    }

    program
}

pub(crate) fn interp_operand(sym_store: &HESymStore, ref_store: &HERefStore, op: &HEOperand) -> HEValue {
    match op {
        HEOperand::Ref(HERef::NodeRef(ref_i)) => ref_store[ref_i].clone(),

        HEOperand::Ref(HERef::ConstSym(sym)) => sym_store[sym.as_str()].clone(),

        HEOperand::ConstNum(n) => HEValue::HEScalar(*n),
    }
}

pub(crate) fn interp_instr(
    sym_store: &HESymStore,
    ref_store: &HERefStore,
    instr: &HEInstr,
    vec_size: usize,
) -> HEValue {
    let exec_binop = |op1: &HEOperand, op2: &HEOperand, f: fn(&i32, &i32) -> i32| -> HEValue {
        let val1 = interp_operand(sym_store, ref_store, op1);
        let val2 = interp_operand(sym_store, ref_store, op2);
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
        HEInstr::Add { index, op1, op2 }=>
            exec_binop(op1, op2, |x1, x2| x1 + x2),

        HEInstr::Mul { index, op1, op2 }=>
            exec_binop(op1, op2, |x1, x2| x1 * x2),

        HEInstr::Rot { index, op1, op2 }=> {
            let val1 = interp_operand(sym_store, ref_store, op1);
            let val2 = interp_operand(sym_store, ref_store, op2);
            match (val1, val2) {
                (HEValue::HEVector(v1), HEValue::HEScalar(s2)) => {
                    let rot_val = s2 % (vec_size as i32);
                    let mut new_vec: Vec<i32> = v1.clone();
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

pub(crate) fn interp_program(sym_store: &HESymStore, program: &HEProgram, vec_size: usize) -> Option<HEValue> {
    let mut ref_store: HERefStore = HashMap::new();

    let mut last_instr = None;
    for (i, instr) in program.instrs.iter().enumerate() {
        let val = interp_instr(&sym_store, &ref_store, instr, vec_size);
        ref_store.insert(i, val);
        last_instr = Some(i);
    }

    last_instr.and_then(|i| ref_store.remove(&i))
}

type HELoweredOperand = String;

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "op")]
pub(crate) enum HELoweredInstr {
    Add { index: HELoweredOperand, op1: HELoweredOperand, op2: HELoweredOperand },
    AddInplace { op1: HELoweredOperand, op2: HELoweredOperand },
    AddPlain { index: HELoweredOperand, op1: HELoweredOperand, op2: HELoweredOperand },
    AddPlainInplace { op1: HELoweredOperand, op2: HELoweredOperand },
    Mul { index: HELoweredOperand, op1: HELoweredOperand, op2: HELoweredOperand },
    MulPlain { index: HELoweredOperand, op1: HELoweredOperand, op2: HELoweredOperand },
    MulPlainInplace { op1: HELoweredOperand, op2: HELoweredOperand },
    Rot { index: HELoweredOperand, op1: HELoweredOperand, op2: HELoweredOperand },
    RelinearizeInplace { op1: HELoweredOperand },
}

#[derive(Serialize)]
pub(crate) struct HELoweredProgram {
    instrs: Vec<HELoweredInstr>,
}

pub(crate) fn lower_operand(op: &HEOperand) -> String {
    match op {
        HEOperand::Ref(HERef::NodeRef(nr)) => format!("i{}", nr),
        HEOperand::Ref(HERef::ConstSym(sym)) => sym.clone(),
        HEOperand::ConstNum(n) => n.to_string(),
    }
}

/// compute the required vectors at every program point
/// this is used in the lowering pass for computing when relinearizations
/// and in-place operations can be used
pub(crate) fn analyze_use(prog: &HEProgram) -> Vec<HashSet<usize>> {
    let mut uses: Vec<HashSet<usize>> = Vec::new();
    uses.push(HashSet::new());

    for instr in prog.instrs.iter().rev() {
        let mut new_use: HashSet<usize> = uses.last().unwrap().clone();
        match instr {
            HEInstr::Add { index: _, op1, op2 } |
            HEInstr::Mul { index: _, op1, op2 } |
            HEInstr::Rot { index: _, op1, op2 } => {
                match op1 {
                    HEOperand::Ref(HERef::NodeRef(nr)) => {
                        new_use.insert(*nr);
                    }
                    _ => ()
                };

                match op2 {
                    HEOperand::Ref(HERef::NodeRef(nr)) => {
                        new_use.insert(*nr);
                    }
                    _ => ()
                };
            }
        }

        uses.push(new_use);
    }

    uses.reverse();
    uses
}

pub(crate) fn lower_program(prog: &HEProgram) -> HELoweredProgram {
    let uses = analyze_use(prog);
    let mut inplace_map: HashMap<usize, usize> = HashMap::new();
    let mut instrs: Vec<HELoweredInstr> = Vec::new();
    for instr in prog.instrs.iter() {
        match instr {
            HEInstr::Add { index, op1, op2 } => {
                let lindex = format!("i{}", index);
                let lop1 = lower_operand(op1);
                let lop2 = lower_operand(op2);
                match (op1, op2) {
                    (HEOperand::Ref(_), HEOperand::Ref(_)) => {
                        instrs.push(
                            HELoweredInstr::Add { index: lindex, op1: lop1, op2: lop2 }
                        )
                    },

                    (HEOperand::Ref(r1), HEOperand::ConstNum(_)) => {
                        match r1 {
                            HERef::NodeRef(nr1) if !uses[index+1].contains(nr1) => {
                                instrs.push(
                                    HELoweredInstr::AddPlainInplace { op1: lop1, op2: lop2 }
                                );
                                inplace_map.insert(*index, *nr1);
                            },
                            _ => {
                                instrs.push(
                                    HELoweredInstr::AddPlain { index: lindex, op1: lop1, op2: lop2 }
                                )
                            }
                        }
                    },

                    (HEOperand::ConstNum(_), HEOperand::Ref(_)) => {
                        instrs.push(
                            HELoweredInstr::AddPlain { index: lindex, op1: lop2, op2: lop1 }
                        )
                    },

                    (HEOperand::ConstNum(_), HEOperand::ConstNum(_)) => {
                        panic!("attempting to add two plaintexts---this should be constant folded")
                    }
                }
            },

            HEInstr::Mul { index, op1, op2 } => {
                let lindex = format!("i{}", index);
                let lop1 = lower_operand(op1);
                let lop2 = lower_operand(op2);
                match (op1, op2) {
                    (HEOperand::Ref(_), HEOperand::Ref(_)) => {
                        instrs.push(
                            HELoweredInstr::Mul { index: lindex.clone(), op1: lop1, op2: lop2 }
                        );
                    },

                    (HEOperand::Ref(_), HEOperand::ConstNum(_)) => {
                        instrs.push(
                            HELoweredInstr::MulPlain { index: lindex.clone(), op1: lop1, op2: lop2 }
                        )
                    },

                    (HEOperand::ConstNum(_), HEOperand::Ref(_)) => {
                        instrs.push(
                            HELoweredInstr::MulPlain { index: lindex.clone(), op1: lop2, op2: lop1 }
                        )
                    },

                    (HEOperand::ConstNum(_), HEOperand::ConstNum(_)) => {
                        panic!("attempting to multiply two plaintexts---this should be constant folded")
                    }
                }

                // relinearize at every multiplication, except for outputs;
                // outputs will not be used in the future,
                // so there's no need to minimize noise for them
                if uses[index+1].contains(index) {
                    instrs.push(
                        HELoweredInstr::RelinearizeInplace { op1: lindex }
                    )
                }
            },
            
            HEInstr::Rot { index, op1, op2 } => {
                let lindex = format!("i{}", index);
                let lop1 = lower_operand(op1);
                let lop2 = lower_operand(op2);
                match (op1, op2) {
                    (HEOperand::Ref(_), HEOperand::ConstNum(_)) => {
                        instrs.push(
                            HELoweredInstr::Rot { index: lindex, op1: lop1, op2: lop2 }
                        )
                    }

                    _ => panic!("must rotate ciphertext with constant value")
                }
            }
        }
    }
    HELoweredProgram { instrs }
}