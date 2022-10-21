use std::collections::HashMap;
use crate::{
    circ::*,
    util::NameGenerator, lang::Shape,
};

/// partially evaluate operations on plaintexts.
struct HECircuitPartialEvaluator {
    name_generator: NameGenerator,
    store: HECircuitStore,
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
            name_generator: NameGenerator::default(),
            store: HECircuitStore::default(),
        }
    }

    fn partial_eval(&mut self, circ: &HECircuit) -> PEResult {
        match circ {
            HECircuit::CiphertextRef(_) => {
                PEResult::Dynamic(circ.clone())
            },

            HECircuit::PlaintextRef(name) => {
                todo!()
                /*
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
                */
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
