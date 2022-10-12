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