use super::ParamCircuitExpr;
use crate::circ::*;

/// partially evaluate plaintexts to remove them from the HE circuit
pub struct HEPartialEvaluator {}

impl HEPartialEvaluator {
    pub fn new() -> Self {
        Self {}
    }

    pub fn run(&self, program: ParamCircuitProgram) -> ParamCircuitProgram {
        
        todo!()
    }

    fn eval(
        &self,
        circuit_id: CircuitId,
        registry: &mut CircuitObjectRegistry
    ) -> (Option<ParamCircuitExpr>, Option<ParamCircuitExpr>) {
        let circuit = registry.get_circuit(circuit_id);
        match circuit {
            ParamCircuitExpr::CiphertextVar(_) => {
                (Some(circuit.clone()), None)
            },

            ParamCircuitExpr::PlaintextVar(var) => {
                (None, Some(circuit.clone()))
            },

            ParamCircuitExpr::Literal(lit) => {
                (None, Some(circuit.clone()))
            },

            ParamCircuitExpr::Op(op, expr1, expr2) => {
                // let (circ1, pe1) = self.eval(*expr1, registry);
                // let (circ1, pe2) = self.eval(*expr2, registry);
                todo!()
            },

            ParamCircuitExpr::Rotate(_, _) => todo!(),
            ParamCircuitExpr::ReduceDim(_, _, _, _) => todo!(),
        }
    }
}
