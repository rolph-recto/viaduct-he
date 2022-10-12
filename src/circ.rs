use std::{collections::HashMap, fmt::Display};

pub mod circ_gen;
pub mod optimizer;
pub mod lowering;
pub mod partial_eval;

pub type Dimension = usize;
pub type Shape = im::Vector<usize>;
pub type HEObjectName = String;

#[derive(Clone,Debug)]
pub enum HECircuit {
    CiphertextRef(HEObjectName),
    PlaintextRef(HEObjectName),
    Literal(isize),
    Add(Box<HECircuit>, Box<HECircuit>),
    Sub(Box<HECircuit>, Box<HECircuit>),
    Mul(Box<HECircuit>, Box<HECircuit>),
    Rotate(Box<HECircuit>, isize),
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