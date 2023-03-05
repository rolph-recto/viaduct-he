use crate::circ::optimizer::*;
use egg::*;
use std::{cmp::Ordering, collections::HashMap};

#[derive(Debug, Clone)]
pub(crate) struct HECost {
    pub muldepth: usize,
    pub latency: f64,
}

impl HECost {
    fn cost(&self) -> f64 {
        (self.muldepth + 1) as f64 * self.latency
    }
}

impl PartialEq for HECost {
    fn eq(&self, other: &Self) -> bool {
        self.cost().eq(&other.cost())
    }
}

impl PartialOrd for HECost {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.cost().partial_cmp(&other.cost())
    }
}

/*
pub(crate) struct HECostFunction<'a> {
    pub egraph: &'a HEGraph,
}

impl<'a> CostFunction<HEOptCircuit> for HECostFunction<'a> {
    type Cost = HECost;

    fn cost<C>(&mut self, enode: &HEOptCircuit, mut costs: C) -> Self::Cost
    where
        C: FnMut(Id) -> Self::Cost,
    {
        let child_muldepth = enode
            .children()
            .iter()
            .fold(0, |acc, child| max(acc, costs(*child).muldepth));

        let child_latency: f64 = enode
            .children()
            .iter()
            .map(|child| costs(*child).latency)
            .sum();

        let is_plainop = enode
            .children()
            .iter()
            .any(|child| self.egraph[*child].data.constval.is_some());

        let mut muldepth = child_muldepth;
        let latency = child_latency
            + match enode {
                HEOptCircuit::Literal(_) => self.latency.num,

                HEOptCircuit::Add(_) => {
                    if is_plainop {
                        self.latency.add_plain
                    } else {
                        self.latency.add
                    }
                }

                HEOptCircuit::Sub(_) => {
                    if is_plainop {
                        self.latency.sub_plain
                    } else {
                        self.latency.sub
                    }
                }

                HEOptCircuit::Mul(_) => {
                    if is_plainop {
                        self.latency.mul_plain
                    } else {
                        muldepth += 1;
                        self.latency.mul
                    }
                }

                HEOptCircuit::Rot(_) => self.latency.rot,

                HEOptCircuit::CiphertextVar(_) => self.latency.sym,

                HEOptCircuit::PlaintextVar(_) => self.latency.sym,

                HEOptCircuit::SumVectors(_) |
                HEOptCircuit::ProductVectors(_) |
                HEOptCircuit::IndexVar(_) |
                HEOptCircuit::FunctionVar(_, _) => 0.0
            };

        HECost { muldepth, latency }
    }
}
*/