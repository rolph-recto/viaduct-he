use std::{collections::HashMap, fmt::Display};

use crate::lang::DimName;

use super::*;

pub struct HELatencyModel {
    pub add_cipher: usize,
    pub add_cipherplain: usize,
    pub add_native: usize,
    pub sub_cipher: usize,
    pub sub_cipherplain: usize,
    pub sub_native: usize,
    pub mul_cipher: usize,
    pub mul_cipherplain: usize,
    pub mul_native: usize,
    pub rot_cipher: usize,
    pub rot_native: usize,
}

impl Default for HELatencyModel {
    fn default() -> Self {
        Self {
            add_cipher: 20,
            add_cipherplain: 5,
            add_native: 1,
            sub_cipher: 20,
            sub_cipherplain: 5,
            sub_native: 1,
            mul_cipher: 30,
            mul_cipherplain: 5,
            mul_native: 1,
            rot_cipher: 40,
            rot_native: 1,
        }
    }
}

#[derive(Clone, Default, Debug, Eq)]
pub(crate) struct HECost {
    pub muldepth: usize,
    pub latency: usize,
    pub multiplicity: Option<usize>,
}

impl HECost {
    fn cost(&self) -> usize {
        (self.muldepth + 1) * self.latency * self.multiplicity.unwrap_or(1)
    }
}

impl Display for HECost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,
            "multiplicity: {:?} muldepth: {:<2} latency: {}",
            self.multiplicity, self.muldepth, self.latency)
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

impl Ord for HECost {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cost().cmp(&other.cost())
    }
}

#[derive(Clone, Default)]
pub struct HEOptimizerContext {
    pub ct_multiplicity_map: HashMap<VarName, usize>,
    pub uniform_ct_set: HashSet<VarName>,
    pub pt_multiplicity_map: HashMap<VarName, usize>,
    pub uniform_pt_set: HashSet<VarName>,
    pub dim_extent_map: HashMap<DimName, usize>,
}

pub(crate) struct HECostFunction<'a> {
    pub latency: HELatencyModel,
    pub egraph: &'a EGraph<HEOptCircuit, HEAnalysis>,
}

impl<'a> CostFunction<HEOptCircuit> for HECostFunction<'a> {
    type Cost = HECost;

    fn cost<C>(&mut self, enode: &HEOptCircuit, mut costs: C) -> Self::Cost
    where C: FnMut(Id) -> Self::Cost
    {
        let (muldepth, latency, multiplicity) =
        match enode {
            HEOptCircuit::Literal(_) => {
                (0, 0, None)
            },

            HEOptCircuit::Add([id1, id2]) |
            HEOptCircuit::Sub([id1, id2]) |
            HEOptCircuit::Mul([id1, id2]) => {
                let type1 = &self.egraph[*id1].data.node_type;
                let type2 = &self.egraph[*id2].data.node_type;
                let cost1 = costs(*id1);
                let cost2 = costs(*id2);

                let child_latency = cost1.latency + cost2.latency;
                let child_muldepth = max(cost1.muldepth, cost1.muldepth);

                let mult1_opt = costs(*id1).multiplicity;
                let mult2_opt = costs(*id2).multiplicity;

                let mult =
                    mult1_opt.and_then(|mult1|
                    mult2_opt.and_then(|mult2| {
                        assert!(mult1 == mult2);
                        Some(mult1)
                    }));

                let node_latency =
                    match (type1, type2) {
                        (HEOptNodeType::Cipher, HEOptNodeType::Cipher) =>
                            match enode {
                                HEOptCircuit::Add(_) => self.latency.add_cipher,
                                HEOptCircuit::Sub(_) => self.latency.sub_cipher,
                                HEOptCircuit::Mul(_) => self.latency.mul_cipher,
                                _ => unreachable!()
                            }

                        (HEOptNodeType::Cipher, HEOptNodeType::Plain) |
                        (HEOptNodeType::Plain, HEOptNodeType::Cipher) => 
                            match enode {
                                HEOptCircuit::Add(_) => self.latency.add_cipherplain,
                                HEOptCircuit::Sub(_) => self.latency.sub_cipherplain,
                                HEOptCircuit::Mul(_) => self.latency.mul_cipherplain,
                                _ => unreachable!()
                            }

                        (HEOptNodeType::Plain, HEOptNodeType::Plain) =>
                            match enode {
                                HEOptCircuit::Add(_) => self.latency.add_native,
                                HEOptCircuit::Sub(_) => self.latency.sub_native,
                                HEOptCircuit::Mul(_) => self.latency.mul_native,
                                _ => unreachable!()
                            }
                    };

                let muldepth =
                    match enode {
                        HEOptCircuit::Add(_) => child_muldepth,
                        HEOptCircuit::Sub(_) => child_muldepth,
                        HEOptCircuit::Mul(_) => {
                            match (type1, type2) {
                                (HEOptNodeType::Cipher, HEOptNodeType::Cipher) =>
                                    child_muldepth + 1,

                                (HEOptNodeType::Cipher, HEOptNodeType::Plain) |
                                (HEOptNodeType::Plain, HEOptNodeType::Cipher) |
                                (HEOptNodeType::Plain, HEOptNodeType::Plain) =>
                                    child_muldepth,
                            }
                        },

                        _ => unreachable!()
                    };

                (muldepth, child_latency + node_latency, mult)
            },

            HEOptCircuit::Rot([_, body_id]) => {
                let body_cost = costs(*body_id);
                let body_type = &self.egraph[*body_id].data.node_type;
                let node_latency = 
                    match body_type {
                        HEOptNodeType::Cipher => self.latency.rot_cipher,
                        HEOptNodeType::Plain => self.latency.rot_native,
                    };

                (body_cost.muldepth, body_cost.latency + node_latency, body_cost.multiplicity)
            },

            HEOptCircuit::CiphertextVar(var) => {
                let mult = self.egraph.analysis.context.ct_multiplicity_map.get(var.as_str()).unwrap();
                (0, 0, Some(*mult))
            },
            
            HEOptCircuit::PlaintextVar(var) => {
                let mult = self.egraph.analysis.context.pt_multiplicity_map.get(var.as_str()).unwrap();
                (0, 0, Some(*mult))
            },

            HEOptCircuit::SumVectors([ind_id, body_id]) |
            HEOptCircuit::ProductVectors([ind_id, body_id]) => {
                let body_cost = costs(*body_id);
                let body_data = &self.egraph[*ind_id].data;

                let ind_data = 
                    self.egraph[*ind_id].data.index_vars
                    .iter().next().unwrap();

                let extent =
                    *self.egraph.analysis.context.dim_extent_map.get(ind_data).unwrap();

                let multiplicity =
                    body_cost.multiplicity.map(|body_mult| body_mult / extent);

                let node_latency =
                    match enode {
                        HEOptCircuit::SumVectors(_) => {
                            match body_data.node_type {
                                HEOptNodeType::Cipher => {
                                    self.latency.add_cipher * extent
                                },

                                HEOptNodeType::Plain => {
                                    self.latency.add_native * extent
                                }
                            }
                        },

                        HEOptCircuit::ProductVectors(_) => {
                            match body_data.node_type {
                                HEOptNodeType::Cipher => {
                                    self.latency.mul_cipher * extent
                                },

                                HEOptNodeType::Plain => {
                                    self.latency.mul_native * extent
                                }
                            }
                        }
                        _ => unreachable!()
                    };

                let muldepth =
                    match enode {
                        HEOptCircuit::SumVectors(_) => body_cost.muldepth,
                        HEOptCircuit::ProductVectors(_) => {
                            match body_data.node_type {
                                HEOptNodeType::Cipher => body_cost.muldepth + extent - 1,
                                HEOptNodeType::Plain => body_cost.muldepth,
                            }
                        },
                        _ => unreachable!(),
                    };

                (muldepth, body_cost.latency + node_latency, multiplicity)
            },
            
            HEOptCircuit::IndexVar(_) | HEOptCircuit::FunctionVar(_, _) => {
                (0, 0, Some(0))
            },
        };

        HECost { muldepth, latency, multiplicity }
    }
}

pub struct HELpCostFunction {
    pub latency: HELatencyModel,
}

impl LpCostFunction<HEOptCircuit, HEAnalysis> for HELpCostFunction {
    fn node_cost(&mut self, egraph: &HEGraph, eclass: Id, enode: &HEOptCircuit) -> f64 {
        let latency = match enode {
            HEOptCircuit::Literal(_) =>
                1.0,

            HEOptCircuit::Add([id1, id2]) => {
                let type1 = &egraph[*id1].data.node_type;
                let type2 = &egraph[*id2].data.node_type;
                match (type1, type2) {
                    (HEOptNodeType::Cipher, HEOptNodeType::Cipher) => {
                        self.latency.add_cipher as f64
                    },

                    (HEOptNodeType::Cipher, HEOptNodeType::Plain) |
                    (HEOptNodeType::Plain, HEOptNodeType::Cipher) => {
                        self.latency.add_cipherplain as f64
                    },

                    (HEOptNodeType::Plain, HEOptNodeType::Plain) => {
                        self.latency.add_native as f64
                    }
                } 
            }

            HEOptCircuit::Sub([id1, id2]) => {
                let type1 = &egraph[*id1].data.node_type;
                let type2 = &egraph[*id2].data.node_type;
                match (type1, type2) {
                    (HEOptNodeType::Cipher, HEOptNodeType::Cipher) => {
                        self.latency.sub_cipher as f64
                    },

                    (HEOptNodeType::Cipher, HEOptNodeType::Plain) |
                    (HEOptNodeType::Plain, HEOptNodeType::Cipher) => {
                        self.latency.sub_cipherplain as f64
                    },

                    (HEOptNodeType::Plain, HEOptNodeType::Plain) => {
                        self.latency.sub_native as f64
                    },
                } 
            }

            HEOptCircuit::Mul([id1, id2]) => {
                let type1 = &egraph[*id1].data.node_type;
                let type2 = &egraph[*id2].data.node_type;
                match (type1, type2) {
                    (HEOptNodeType::Cipher, HEOptNodeType::Cipher) => {
                        self.latency.mul_cipher as f64
                    },

                    (HEOptNodeType::Cipher, HEOptNodeType::Plain) |
                    (HEOptNodeType::Plain, HEOptNodeType::Cipher) => {
                        self.latency.mul_cipherplain as f64
                    },

                    (HEOptNodeType::Plain, HEOptNodeType::Plain) => {
                        self.latency.mul_native as f64
                    },
                } 
            },

            HEOptCircuit::Rot([_, body_id]) => {
                let body_type = &egraph[*body_id].data.node_type;
                match body_type {
                    HEOptNodeType::Cipher => self.latency.rot_cipher as f64,
                    HEOptNodeType::Plain => self.latency.rot_native as f64,
                }
            },

            HEOptCircuit::CiphertextVar(_) | HEOptCircuit::PlaintextVar(_) => {
                1.0
            },

            HEOptCircuit::SumVectors(_) | HEOptCircuit::ProductVectors(_) => {
                1.0
            },

            HEOptCircuit::IndexVar(_) | HEOptCircuit::FunctionVar(_, _) => 1.0
        };

        let muldepth = egraph[eclass].data.muldepth;
        let multiplicity =
            egraph[eclass].data.multiplicity.unwrap_or(1) as f64;

        ((muldepth + 1) as f64) * latency * multiplicity
    }
}