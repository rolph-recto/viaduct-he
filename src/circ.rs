use std::{collections::HashMap, fmt::Display, ops::Index};

use egg::{RecExpr, Symbol, Id};

use self::optimizer::HEOptCircuit;

pub mod circ_gen;
pub mod optimizer;
pub mod lowering;
pub mod partial_eval;

pub type Dimension = usize;
pub type HEObjectName = String;

#[derive(Clone,Debug)]
pub struct Shape(im::Vector<usize>);

impl Shape {
    pub fn num_dims(&self) -> usize {
        self.0.len()
    }

    pub fn size(&self) -> usize {
        self.0.iter().fold(1, |acc, x| acc*x)
    }

    pub fn block_size(&self, dim: usize) -> usize {
        let mut block_size: usize = 1;
        for i in (dim+1)..self.0.len() {
            block_size *= self.0[i];
        }
        block_size
    }

    pub fn as_vec(&self) -> &im::Vector<usize> {
        &self.0
    }

    pub fn wrap_offset(&self, offset: isize) -> usize {
        if offset < 0 {
            ((self.size() as isize) - offset) as usize

        } else {
            offset as usize
        }
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl From<im::Vector<usize>> for Shape {
    fn from(vec: im::Vector<usize>) -> Self {
        Shape(vec)
    }
}

#[derive(Clone,Debug)]
pub enum HECircuit {
    CiphertextRef(HEObjectName),
    PlaintextRef(HEObjectName),
    Literal(isize),
    Add(Box<HECircuit>, Box<HECircuit>),
    Sub(Box<HECircuit>, Box<HECircuit>),
    Mul(Box<HECircuit>, Box<HECircuit>),
    Rotate(Box<HECircuit>, usize),
}

impl HECircuit {
    fn to_optimizer_circuit_recur(&self, expr: &mut RecExpr<HEOptCircuit>) -> Id {
        match self {
            HECircuit::CiphertextRef(name) => 
                expr.add(HEOptCircuit::CiphertextRef(Symbol::from(name))),

            HECircuit::PlaintextRef(name) => 
                expr.add(HEOptCircuit::PlaintextRef(Symbol::from(name))),

            HECircuit::Literal(lit) => 
                expr.add(HEOptCircuit::Num(*lit)),

            HECircuit::Add(op1, op2) => {
                let id1 = op1.to_optimizer_circuit_recur(expr);
                let id2 = op2.to_optimizer_circuit_recur(expr);
                expr.add(HEOptCircuit::Add([id1,id2]))
            },

            HECircuit::Sub(op1, op2) => {
                let id1 = op1.to_optimizer_circuit_recur(expr);
                let id2 = op2.to_optimizer_circuit_recur(expr);
                // expr.add(HEOptCircuit::Sub([id1,id2]))
                todo!()
            },

            HECircuit::Mul(op1, op2) => {
                let id1 = op1.to_optimizer_circuit_recur(expr);
                let id2 = op2.to_optimizer_circuit_recur(expr);
                expr.add(HEOptCircuit::Mul([id1,id2]))
            },

            HECircuit::Rotate(op1, op2) => {
                let id1 = op1.to_optimizer_circuit_recur(expr);
                let id2 = expr.add(HEOptCircuit::Num(*op2 as isize));
                expr.add(HEOptCircuit::Rot([id1,id2]))
            },
        }
    }

    pub fn to_opt_circuit(&self) -> RecExpr<HEOptCircuit> {
        let mut expr = RecExpr::default();
        self.to_optimizer_circuit_recur(&mut expr);
        expr
    }
}

impl Display for HECircuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HECircuit::CiphertextRef(name) =>
                write!(f, "{}", name),

            HECircuit::PlaintextRef(name) =>
                write!(f, "{}", name),

            HECircuit::Literal(lit) =>
                write!(f, "{}", lit),

            HECircuit::Add(op1, op2) =>
                write!(f, "({} + {})", op1, op2),

            HECircuit::Sub(op1, op2) =>
                write!(f, "({} - {})", op1, op2),

            HECircuit::Mul(op1, op2) =>
                write!(f, "({} * {})", op1, op2),

            HECircuit::Rotate(op, amt) =>
                write!(f, "rot({},{})", op, amt)
        }
    }
}

#[derive(Clone,Debug)]
pub struct Ciphertext { pub shape: Shape }

#[derive(Clone,Debug)]
pub struct Plaintext { pub shape: Shape, pub value: im::Vector<isize> }

pub struct HECircuitStore {
    pub ciphertexts: HashMap<HEObjectName, Ciphertext>,
    pub plaintexts: HashMap<HEObjectName, Plaintext>,
}

impl HECircuitStore {
    pub fn new(inputs: &HashMap<HEObjectName,Ciphertext>) -> Self {
        HECircuitStore { ciphertexts: inputs.clone(), plaintexts: HashMap::new() }
    }
}

impl Default for HECircuitStore {
    fn default() -> Self {
        HECircuitStore { ciphertexts: HashMap::new(), plaintexts: HashMap::new() }
    }
}