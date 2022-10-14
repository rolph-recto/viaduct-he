use crate::circ::optimizer::*;

pub(crate) struct OpSizeFunction;

impl LpCostFunction<HEOptCircuit, HEData> for OpSizeFunction {
    fn node_cost(&mut self, egraph: &HEGraph, id: Id, enode: &HEOptCircuit) -> f64 {
        match enode {
            HEOptCircuit::Num(_) => 0.1,

            HEOptCircuit::Add([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                6.0 * (muldepth as f64)
            },

            HEOptCircuit::Mul([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                15.0 * (muldepth as f64)
            },

            HEOptCircuit::Rot([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                0.1 * (muldepth as f64)
            },

            HEOptCircuit::CiphertextRef(_) => 0.1,

            HEOptCircuit::PlaintextRef(_) => 0.1,
        }
    }
}