use crate::circ::optimizer::*;

pub(crate) struct OpSizeFunction;

impl LpCostFunction<HEOptimizerCircuit, HEData> for OpSizeFunction {
    fn node_cost(&mut self, egraph: &HEGraph, id: Id, enode: &HEOptimizerCircuit) -> f64 {
        match enode {
            HEOptimizerCircuit::Num(_) => 0.1,

            HEOptimizerCircuit::Add([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                6.0 * (muldepth as f64)
            },

            HEOptimizerCircuit::Mul([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                15.0 * (muldepth as f64)
            },

            HEOptimizerCircuit::Rot([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                0.1 * (muldepth as f64)
            },

            HEOptimizerCircuit::Symbol(_) => 0.1,
        }
    }
}