use crate::ir::{expr::HEExpr, optimizer::*};

pub(crate) struct OpSizeFunction;

impl LpCostFunction<HEExpr, HEData> for OpSizeFunction {
    fn node_cost(&mut self, egraph: &HEGraph, id: Id, enode: &HEExpr) -> f64 {
        match enode {
            HEExpr::Num(_) => 0.1,

            HEExpr::Add([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                6.0 * (muldepth as f64)
            },

            HEExpr::Mul([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                15.0 * (muldepth as f64)
            },

            HEExpr::Rot([id1, id2]) => {
                let d1 = egraph[*id1].data.muldepth;
                let d2 = egraph[*id2].data.muldepth;
                let muldepth = max(d1, d2) + 1;
                0.1 * (muldepth as f64)
            },

            HEExpr::Symbol(_) => 0.1,
        }
    }
}