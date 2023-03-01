/// program.rs
/// instruction representation of HE programs

use pretty::RcDoc;
use std::collections::HashMap;
use std::fmt::{self, Display};

use crate::circ::{MaskVector, vector_info::VectorInfo};
use crate::lang::{Extent, ArrayName, OffsetExpr};

pub type InstructionId = usize;

pub mod lowering;

#[derive(Clone, Debug)]
pub enum HEIndex {
    Var(String),
    Literal(isize),
}

impl Display for HEIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HEIndex::Var(var) =>
                write!(f, "{}", var),

            HEIndex::Literal(val) =>
                write!(f, "{}", val),
        }
    }
}

#[derive(Clone, Debug)]
pub enum HERef {
    // reference to a previous instruction's output
    Instruction(InstructionId),

    // index to a ciphertext array (variable if no indices)
    Ciphertext(ArrayName, Vec<HEIndex>),

    // index to a plaintext array (variable if no indices)
    Plaintext(ArrayName, Vec<HEIndex>)
}

impl Display for HERef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HERef::Instruction(id) =>
                write!(f, "i{}", id),

            HERef::Ciphertext(vector, indices) => {
                let index_str =
                    indices.iter()
                    .map(|i| format!("[{}]", i))
                    .collect::<Vec<String>>()
                    .join("");

                write!(f, "{}{}", vector, index_str)
            },

            HERef::Plaintext(vector, indices) => {
                let index_str =
                    indices.iter()
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
            },

            HEOperand::Literal(n) => {
                write!(f, "{}", n)
            },
        }
    }
}

#[derive(Clone, Debug)]
pub enum HEInstruction {
    Add(usize, HERef, HERef),
    Sub(usize, HERef, HERef),
    Mul(usize, HERef, HERef),
    Rot(usize, OffsetExpr, HERef),
}

impl fmt::Display for HEInstruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HEInstruction::Add(id, op1, op2) =>
                write!(f, "i{} = {} + {}", id, op1, op2),

            HEInstruction::Sub(id, op1, op2) =>
                write!(f, "i{} = {} - {}", id, op1, op2),

            HEInstruction::Mul(id, op1, op2) =>
                write!(f, "i{} = {} * {}", id, op1, op2),

            HEInstruction::Rot(id, op1, op2) =>
                write!(f, "i{} = rot {} {}", id, op1, op2),
        }
    }
}

#[derive(Clone, Debug)]
pub enum HEStatement {
    ForNode(String, usize, Vec<HEStatement>),
    DeclareVar(String, Vec<Extent>),
    SetVar(String, Vec<HEIndex>, HEOperand),
    Instruction(HEInstruction),
}

impl HEStatement {
    pub fn to_doc(&self) -> RcDoc<()> {
        match self {
            HEStatement::ForNode(dim, extent, body) => {
                let body_doc =
                    RcDoc::intersperse(
                        body.iter().map(|stmt| stmt.to_doc()),
                        RcDoc::hardline()
                    );

                RcDoc::text(format!("for {}: {} {{", dim, extent))
                .append(
                    RcDoc::hardline()
                    .append(body_doc)
                    .nest(4)
                )
                .append(RcDoc::hardline())
                .append(RcDoc::text("}"))
            },

            HEStatement::DeclareVar(var, extents) => {
                let extent_str = 
                    extents.iter()
                    .map(|i| format!("[{}]", i))
                    .collect::<Vec<String>>()
                    .join("");

                RcDoc::text(format!("var {}{}", var, extent_str))
            },

            HEStatement::SetVar(var, indices, val) => {
                let index_str = 
                    indices.iter()
                    .map(|i| format!("[{}]", i))
                    .collect::<Vec<String>>()
                    .join("");

                RcDoc::text(format!("{}{} = {}", var, index_str, val))
            },

            HEStatement::Instruction(instr) =>
                RcDoc::text(instr.to_string())
        }
    }
}

impl Display for HEStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_doc().render_fmt(80, f)
    }
}

pub struct HEProgramContext {
    vector_map: HashMap<VectorInfo, String>,
    mask_map: HashMap<MaskVector, String>,
    const_map: HashMap<isize, String>,
}

pub struct HEProgram {
    pub context: HEProgramContext,
    pub statements: Vec<HEStatement>,
}

impl HEProgram {
    pub fn to_doc(&self) -> RcDoc<()> {
        RcDoc::intersperse(
            self.context.vector_map.iter().map(|(vector, name)| {
                RcDoc::text(format!("{} = vector({})", name, vector))
            })
            ,RcDoc::hardline()
        )
        .append(RcDoc::hardline())
        .append(
            RcDoc::intersperse(
                self.context.mask_map.iter().map(|(mask, name)| {
                    RcDoc::text(format!("{} = mask({:?})", name, mask))
                })
                ,RcDoc::hardline()
            )
        )
        .append(RcDoc::hardline())
        .append(
            RcDoc::intersperse(
                self.context.const_map.iter().map(|(constval, name)| {
                    RcDoc::text(format!("{} = const({})", name, constval))
                })
                ,RcDoc::hardline()
            )
        )
        .append(RcDoc::hardline())
        .append(
            RcDoc::intersperse(
                self.statements.iter().map(|s| s.to_doc()),
                RcDoc::hardline()
            )
        )
    }
}

impl Display for HEProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_doc().render_fmt(80, f)
    }
}


/*
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
                HEOperand::Ref(HERef::Ciphertext(_, _)) => 0,
                HEOperand::Ref(HERef::Plaintext(_, _)) => 0,
                HEOperand::Literal(_) => 0,
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
                            (HEOperand::Literal(_), _) | (_, HEOperand::Literal(_)) =>
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
                        (HEOperand::Literal(_), _) | (_, HEOperand::Literal(_)) =>  {
                            latency += model.add_plain;
                        },

                        _ => {
                            latency += model.add
                        }
                    }
                },

                HEInstruction::Sub { id: _, op1, op2 } => {
                    match (op1, op2) {
                        (HEOperand::Literal(_), _) | (_, HEOperand::Literal(_)) =>  {
                            latency += model.add_plain;
                        },

                        _ => {
                            latency += model.add;
                        }
                    }
                },

                HEInstruction::Mul { id: _, op1, op2 } => {
                    match (op1, op2) {
                        (HEOperand::Literal(_), _) | (_, HEOperand::Literal(_)) =>  {
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
                if let HEOperand::Ref(HERef::Ciphertext(sym, index)) = op{
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
                if let HEOperand::Ref(HERef::Plaintext(sym, index)) = op {
                    symset.insert(sym.to_string());
                }
            }
        }

        symset
    }

    /// generate a random sym store for this program
    /*
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
    */

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
                    node_map.insert(id, HEOperand::Literal(*n));
                }

                HEOptCircuit::CiphertextRef(sym) => {
                    node_map.insert(id, HEOperand::Ref(HERef::Ciphertext(sym.to_string(), vec![])));
                }

                HEOptCircuit::PlaintextRef(sym) => {
                    node_map.insert(id, HEOperand::Ref(HERef::Plaintext(sym.to_string(), vec![])));
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
*/

#[cfg(test)]
mod tests {
    use crate::{
        lang::{BaseOffsetMap, Operator},
        circ::{
            IndexCoordinateMap,
            IndexCoordinateSystem,
            ParamCircuitProgram,
            CiphertextObject,
            CircuitObjectRegistry,
            ParamCircuitExpr,
            CircuitValue
        },
        program::lowering::CircuitLowering,
    };

    use super::*;

    fn test_lowering(program: ParamCircuitProgram) {
        let mut lowering = CircuitLowering::new();
        let lowered_program = lowering.lower(program);
        println!("{}", lowered_program);
    }

    #[test]
    fn test_reduce() {
        let mut coord_map =
            IndexCoordinateMap::from_coord_system(
                IndexCoordinateSystem::from_dim_list(
                    vec![(String::from("i"), 2), (String::from("j"), 2)]
                )
            );

        let vector =
            VectorInfo {
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
        let ct = registry.register_circuit(ParamCircuitExpr::CiphertextVar(String::from("ct")));
        let reduce_vec =
            registry.register_circuit(
                ParamCircuitExpr::ReduceVectors(
                    String::from("j"), 2, Operator::Add,
                    ct
                )
            );

        let circuit =
            registry.register_circuit(
                ParamCircuitExpr::Op(Operator::Add, reduce_vec, lit_2)
            );

        registry.ct_var_values.insert(
            String::from("ct"),
            CircuitValue::CoordMap(coord_map)
        );

        let circuit_program =
            ParamCircuitProgram {
                registry,
                expr_list: vec![
                    (String::from("out"),
                    vec![(String::from("i"), 2)],
                    circuit)
                ]
            };

        test_lowering(circuit_program);
    }
}