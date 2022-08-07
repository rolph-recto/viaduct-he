use egg::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
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
pub(crate) enum HEOperand {
    NodeRef(usize),
    ConstSym(String),
    ConstNum(i32),
}

impl fmt::Display for HEOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                HEOperand::NodeRef(i) => format!("i{}", i),
                HEOperand::ConstSym(sym) => sym.to_string(),
                HEOperand::ConstNum(n) => n.to_string(),
            }
        )
    }
}

#[derive(Clone, Serialize)]
#[serde(tag = "op")]
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

#[derive(Serialize)]
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
    /// get the symbols used in this program.
    fn get_symbols(&self) -> HashSet<String> {
        let mut symset = HashSet::new();

        for instr in self.instrs.iter() {
            for op in instr.get_operands() {
                if let HEOperand::ConstSym(sym) = op {
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
        nmap.insert(id, HEOperand::NodeRef(cur_instr));

        cur_instr += 1;
    };

    for (i, node) in expr.as_ref().iter().enumerate() {
        let id = Id::from(i);
        match node {
            HE::Num(n) => {
                node_map.insert(id, HEOperand::ConstNum(*n));
            }

            HE::Symbol(sym) => {
                node_map.insert(id, HEOperand::ConstSym(sym.to_string()));
            }

            HE::Add([id1, id2]) => {
                op_processor(
                    &mut node_map, id, 
                    |index, op1, op2| HEInstr::Add { index, op1, op2 },
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
        HEOperand::NodeRef(ref_i) => ref_store[ref_i].clone(),

        HEOperand::ConstSym(sym) => sym_store[sym.as_str()].clone(),

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