use crate::{ir::expr::HE, optimizer::*};

pub(crate) struct OpSizeFunction;

impl LpCostFunction<HE, HEData> for OpSizeFunction {
    fn node_cost(&mut self, egraph: &HEGraph, id: Id, enode: &HE) -> f64 {
        match enode {
            HE::Num(_) => 0.1,

            HE::Add([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                6.0 * (muldepth as f64)
            },

            HE::Mul([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                15.0 * (muldepth as f64)
            },

            HE::Rot([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                0.1 * (muldepth as f64)
            },

            HE::Symbol(_) => 0.1,
        }
    }
}