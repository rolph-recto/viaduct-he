use std::collections::HashMap;

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

#[derive(Clone,Debug)]
pub struct Ciphertext { shape: Shape }

#[derive(Clone,Debug)]
pub struct Plaintext { shape: Shape, value: im::Vector<isize> }

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