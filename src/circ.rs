pub mod circ_gen;
pub mod optimizer;
pub mod lowering;

pub type Dimension = usize;
pub type Shape = im::Vector<usize>;
pub type ObjectName = String;

#[derive(Clone,Debug)]
pub enum HECircuit {
    CiphertextRef(ObjectName),
    PlaintextRef(ObjectName),
    Add(Box<HECircuit>, Box<HECircuit>),
    Sub(Box<HECircuit>, Box<HECircuit>),
    Mul(Box<HECircuit>, Box<HECircuit>),
    Rotate(Box<HECircuit>, isize),
}

#[derive(Clone,Debug)]
pub enum HEObject {
    Ciphertext(Shape),
    Plaintext(Shape, im::Vector<isize>),
}

impl HEObject {
    fn shape(&self) -> &Shape {
        match self {
            HEObject::Ciphertext(shape) => shape,
            HEObject::Plaintext(shape, _) => shape,
        }
    }
}