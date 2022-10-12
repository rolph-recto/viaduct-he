use std::collections::HashMap;
use crate::{
    circ::*,
    util::NameGenerator,
};

/// partially evaluate operations on plaintexts.
struct HECircuitPartialEvaluator {
    object_map: HashMap<HEObjectName, HEObject>,
    name_generator: NameGenerator
}

enum PEValue {
    Vector(Shape, im::Vector<isize>),
    Scalar(isize),
}

enum PEResult {
    Static(PEValue),
    Dynamic(HECircuit),
}

/// TODO finish this to prevent operations on plaintexts 
impl HECircuitPartialEvaluator {
    fn new() -> Self {
        HECircuitPartialEvaluator {
            object_map: HashMap::new(),
            name_generator: NameGenerator::new(),
        }
    }

    fn partial_eval(&mut self, circ: &HECircuit) -> PEResult {
        match circ {
            HECircuit::CiphertextRef(_) => {
                PEResult::Dynamic(circ.clone())
            },

            HECircuit::PlaintextRef(name) => {
                let object = &self.object_map[name];
                match object {
                    HEObject::Ciphertext(_) => {
                        panic!("plaintext ref {} points to a ciphertext object", name)
                    },

                    HEObject::Plaintext(shape, values) => {
                        PEResult::Static(
                            PEValue::Vector(shape.clone(), values.clone())
                        )
                    }
                }
            },

            HECircuit::Literal(lit) => {
                PEResult::Static(PEValue::Scalar(*lit))
            },

            HECircuit::Add(_, _) => todo!(),

            HECircuit::Sub(_, _) => todo!(),
            HECircuit::Mul(_, _) => todo!(),
            HECircuit::Rotate(_, _) => todo!(),
        }
    }
}
