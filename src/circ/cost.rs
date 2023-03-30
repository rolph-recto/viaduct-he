use std::{
    cmp::{max, Ordering},
    collections::HashMap, ops::Add,
};

use crate::{
    circ::{ParamCircuitExpr, ParamCircuitProgram, VectorType},
    lang::Operator,
};

use super::{CircuitId, CircuitObjectRegistry};
use crate::circ::pseudomaterializer::PseudoCircuitProgram;

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CostFeatures {
    pub input_ciphertexts: usize,
    pub input_plaintexts: usize,
    pub output_ciphertexts: usize,
    pub ct_rotations: usize,
    pub pt_rotations: usize,
    pub ct_ct_add: usize,
    pub ct_pt_add: usize,
    pub pt_pt_add: usize,
    pub ct_ct_mul: usize,
    pub ct_pt_mul: usize,
    pub pt_pt_mul: usize,
    pub ct_ct_sub: usize,
    pub ct_pt_sub: usize,
    pub pt_pt_sub: usize,
    pub ct_ct_muldepth: usize,
    pub ct_pt_muldepth: usize,
}

impl CostFeatures {
    // return a default weighting on features
    // basically just turns off all plaintext weights
    pub fn default_weights() -> Self {
        CostFeatures {
            input_ciphertexts: 1,
            input_plaintexts: 0,
            output_ciphertexts: 1,
            ct_rotations: 1,
            pt_rotations: 1,
            ct_ct_add: 1,
            ct_pt_add: 1,
            pt_pt_add: 0,
            ct_ct_mul: 1,
            ct_pt_mul: 1,
            pt_pt_mul: 0,
            ct_ct_sub: 1,
            ct_pt_sub: 1,
            pt_pt_sub: 0,
            ct_ct_muldepth: 1,
            ct_pt_muldepth: 1
        }
    }

    pub fn weighted_cost(&self, weights: &CostFeatures) -> usize {
        let weighted = (*self).weight(*weights);
        weighted.input_ciphertexts
            + weighted.input_plaintexts
            + weighted.output_ciphertexts
            + weighted.pt_rotations
            + weighted.ct_ct_add
            + weighted.ct_pt_add
            + weighted.pt_pt_add
            + weighted.ct_ct_mul
            + weighted.ct_pt_mul
            + weighted.pt_pt_mul
            + weighted.ct_ct_sub
            + weighted.ct_pt_sub
            + weighted.pt_pt_sub
            + weighted.ct_ct_muldepth
            + weighted.ct_pt_muldepth
    }

    /// because this operation is meant for combining features of subexprs,
    /// everything is pointwise *except* for muldepth, of which max is taken
    pub fn combine(&self, other: &Self) -> Self {
        CostFeatures {
            input_ciphertexts: self.input_ciphertexts + other.input_ciphertexts,
            input_plaintexts: self.input_plaintexts + other.input_plaintexts,
            output_ciphertexts: self.output_ciphertexts + other.output_ciphertexts,
            ct_rotations: self.ct_rotations + other.ct_rotations,
            pt_rotations: self.pt_rotations + other.pt_rotations,
            ct_ct_add: self.ct_ct_add + other.ct_ct_add,
            ct_pt_add: self.ct_pt_add + other.ct_pt_add,
            pt_pt_add: self.pt_pt_add + other.pt_pt_add,
            ct_ct_mul: self.ct_ct_mul + other.ct_ct_mul,
            ct_pt_mul: self.ct_pt_mul + other.ct_pt_mul,
            pt_pt_mul: self.pt_pt_mul + other.pt_pt_mul,
            ct_ct_sub: self.ct_ct_sub + other.ct_ct_sub,
            ct_pt_sub: self.ct_pt_sub + other.ct_pt_sub,
            pt_pt_sub: self.pt_pt_sub + other.pt_pt_sub,
            ct_ct_muldepth: max(self.ct_ct_muldepth, other.ct_ct_muldepth),
            ct_pt_muldepth: max(self.ct_pt_muldepth, other.ct_pt_muldepth),
        }
    }

    pub fn weight(self, other: Self) -> Self {
        CostFeatures {
            input_ciphertexts: self.input_ciphertexts * other.input_ciphertexts,
            input_plaintexts: self.input_plaintexts * other.input_plaintexts,
            output_ciphertexts: self.output_ciphertexts * other.output_ciphertexts,
            ct_rotations: self.ct_rotations * other.ct_rotations,
            pt_rotations: self.pt_rotations * other.pt_rotations,
            ct_ct_add: self.ct_ct_add * other.ct_ct_add,
            ct_pt_add: self.ct_pt_add * other.ct_pt_add,
            pt_pt_add: self.pt_pt_add * other.pt_pt_add,
            ct_ct_mul: self.ct_ct_mul * other.ct_ct_mul,
            ct_pt_mul: self.ct_pt_mul * other.ct_pt_mul,
            pt_pt_mul: self.pt_pt_mul * other.pt_pt_mul,
            ct_ct_sub: self.ct_ct_sub * other.ct_ct_sub,
            ct_pt_sub: self.ct_pt_sub * other.ct_pt_sub,
            pt_pt_sub: self.pt_pt_sub * other.pt_pt_sub,
            ct_ct_muldepth: self.ct_ct_muldepth * other.ct_ct_muldepth,
            ct_pt_muldepth: self.ct_pt_muldepth * other.ct_pt_muldepth,
        }
    }

    pub fn dominates(&self, other: &Self) -> bool {
        let pairs = vec![
            (self.input_ciphertexts, other.input_ciphertexts),
            (self.input_plaintexts, other.input_plaintexts),
            (self.output_ciphertexts, other.output_ciphertexts),
            (self.ct_rotations, other.ct_rotations),
            (self.pt_rotations, other.pt_rotations),
            (self.ct_ct_add, other.ct_ct_add),
            (self.ct_pt_add, other.ct_pt_add),
            (self.pt_pt_add, other.pt_pt_add),
            (self.ct_ct_sub, other.ct_ct_sub),
            (self.ct_pt_sub, other.ct_pt_sub),
            (self.pt_pt_sub, other.pt_pt_sub),
            (self.ct_ct_mul, other.ct_ct_mul),
            (self.ct_pt_mul, other.ct_pt_mul),
            (self.pt_pt_mul, other.pt_pt_mul),
            (self.ct_ct_muldepth, other.ct_ct_muldepth),
            (self.ct_pt_muldepth, other.ct_pt_muldepth)
        ];

        pairs.iter().all(|(s, o)| s <= o)
    }
}

impl Add for CostFeatures {
    type Output = CostFeatures;

    fn add(self, rhs: Self) -> Self::Output {
        self.combine(&rhs)
    }
}

/// estimate cost of an param circuit program
#[derive(Default)]
pub struct CostEstimator;

impl CostEstimator {
    pub fn estimate_cost(&self, program: &ParamCircuitProgram) -> CostFeatures {
        let mut cost: CostFeatures =
            program.native_expr_list.iter()
            .chain(program.circuit_expr_list.iter())
            .fold(CostFeatures::default(), |acc, (_, dims, id)| {
                let multiplicity =
                    dims.iter().fold(1, |acc, (_, extent)| acc * extent);

                let mut cost = CostFeatures::default();
                self.estimate_cost_expr(
                    *id,
                    multiplicity,
                    &program.registry, 
                    &mut HashMap::new(),
                    &mut cost
                );

                acc.combine(&cost)
            });

        let (output_dims, _) = program.output_circuit();
        let output_multiplicity =
            output_dims.iter()
            .fold(1, |acc, (_, extent)| acc * extent);

        cost.input_ciphertexts += program.registry.get_ciphertext_input_vectors(None).len();
        cost.input_plaintexts += program.registry.get_plaintext_input_vectors(None).len();
        cost.input_plaintexts += program.registry.get_constants(None, None).len();
        cost.input_plaintexts += program.registry.get_masks(None).len();
        cost.output_ciphertexts += output_multiplicity;

        cost
    }

    pub fn estimate_pseudo_cost(&self, program: &PseudoCircuitProgram) -> CostFeatures {
        let expr_cost: CostFeatures =
            program.expr_list.iter()
            .fold(CostFeatures::default(), |acc, (multiplicity, id)| {
                let mut cost = CostFeatures::default();
                self.estimate_cost_expr(
                    *id,
                    *multiplicity,
                    &program.registry, 
                    &mut HashMap::new(),
                    &mut cost
                );

                acc + cost
            });

        expr_cost + program.ref_cost

        /*
        let (output_multiplicity, _) = program.expr_list.last().unwrap();
        // cost.input_ciphertexts += program.registry.get_ciphertext_input_vectors(None).len();
        // cost.input_plaintexts += program.registry.get_plaintext_input_vectors(None).len();

        cost.input_plaintexts += program.registry.get_constants(None, None).len();

        // assume the circuit only uses 1 mask globally
        cost.input_plaintexts += 1;

        cost.output_ciphertexts += output_multiplicity;

        cost
        */
    }

    // this must visit subcircuits exactly once
    fn estimate_cost_expr(
        &self,
        id: CircuitId,
        multiplicity: usize,
        registry: &CircuitObjectRegistry,
        type_map: &mut HashMap<CircuitId, VectorType>,
        cost: &mut CostFeatures,
    ) -> VectorType {
        if type_map.contains_key(&id) {
            return type_map[&id]
        }

        match registry.get_circuit(id) {
            ParamCircuitExpr::CiphertextVar(_) => {
                VectorType::Ciphertext
            },

            ParamCircuitExpr::PlaintextVar(_) |
            ParamCircuitExpr::Literal(_) => {
                VectorType::Plaintext
            },

            ParamCircuitExpr::Op(op, id1, id2) => {
                let type1 =
                    self.estimate_cost_expr(*id1, multiplicity, registry, type_map, cost);

                let type2 =
                    self.estimate_cost_expr(*id2, multiplicity, registry, type_map, cost);

                let node_type =
                    match (op, type1, type2) {
                        (Operator::Add, VectorType::Ciphertext, VectorType::Ciphertext) => {
                            cost.ct_ct_add += multiplicity;
                            VectorType::Ciphertext
                        },

                        (Operator::Add, VectorType::Ciphertext, VectorType::Plaintext) |
                        (Operator::Add, VectorType::Plaintext, VectorType::Ciphertext) => {
                            cost.ct_pt_add += multiplicity;
                            VectorType::Ciphertext
                        },

                        (Operator::Add, VectorType::Plaintext, VectorType::Plaintext) => {
                            cost.pt_pt_add += multiplicity;
                            VectorType::Plaintext
                        },

                        (Operator::Sub, VectorType::Ciphertext, VectorType::Ciphertext) => {
                            cost.ct_ct_sub += multiplicity;
                            VectorType::Ciphertext
                        },

                        (Operator::Sub, VectorType::Ciphertext, VectorType::Plaintext) |
                        (Operator::Sub, VectorType::Plaintext, VectorType::Ciphertext) => {
                            cost.ct_pt_sub += multiplicity;
                            VectorType::Ciphertext
                        },

                        (Operator::Sub, VectorType::Plaintext, VectorType::Plaintext) => {
                            cost.pt_pt_sub += multiplicity;
                            VectorType::Plaintext
                        },

                        (Operator::Mul, VectorType::Ciphertext, VectorType::Ciphertext) => {
                            cost.ct_ct_mul += multiplicity;
                            VectorType::Ciphertext
                        },

                        (Operator::Mul, VectorType::Ciphertext, VectorType::Plaintext) |
                        (Operator::Mul, VectorType::Plaintext, VectorType::Ciphertext) => {
                            cost.ct_pt_mul += multiplicity;
                            VectorType::Ciphertext
                        },

                        (Operator::Mul, VectorType::Plaintext, VectorType::Plaintext) => {
                            cost.pt_pt_mul += multiplicity;
                            VectorType::Plaintext
                        },
                    };

                type_map.insert(id, node_type);
                node_type
            },

            ParamCircuitExpr::Rotate(_, body_id) => {
                let body_type =
                    self.estimate_cost_expr(*body_id, multiplicity, registry, type_map, cost);

                match body_type {
                    VectorType::Ciphertext => {
                        cost.ct_rotations += multiplicity;
                    },
                    VectorType::Plaintext => {
                        cost.pt_rotations += multiplicity;
                    }
                }

                type_map.insert(id, body_type);
                body_type
            },

            ParamCircuitExpr::ReduceDim(_, extent, op, body_id) => {
                let body_type =
                    self.estimate_cost_expr(*body_id, extent * multiplicity, registry, type_map, cost);

                match (op, body_type) {
                    (Operator::Add, VectorType::Ciphertext) => {
                        cost.ct_ct_add += extent - 1;
                    },

                    (Operator::Add, VectorType::Plaintext) => {
                        cost.pt_pt_add += extent - 1;
                    },

                    (Operator::Sub, VectorType::Ciphertext) => {
                        cost.pt_pt_add += extent - 1;
                    },
                    
                    (Operator::Sub, VectorType::Plaintext) => {
                        cost.pt_pt_sub += extent - 1;
                    },

                    (Operator::Mul, VectorType::Ciphertext) => {
                        cost.ct_ct_mul += extent - 1;
                        cost.ct_ct_muldepth += extent - 1;
                    },

                    (Operator::Mul, VectorType::Plaintext) => {
                        cost.pt_pt_mul += extent - 1;
                    },
                }

                type_map.insert(id, body_type);
                body_type
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circ::{array_materializer::*, materializer::{Materializer}},
        lang::{
            elaborated::Elaborator, index_elim::IndexElimination, parser::ProgramParser,
            source::SourceProgram,
        },
        scheduling::Schedule,
    };

    // generate an initial schedule for a program
    fn test_cost_estimator(src: &str) {
        let parser = ProgramParser::new();
        let program: SourceProgram = parser.parse(src).unwrap();

        let elaborated = Elaborator::new().run(program);
        let inline_set = elaborated.all_inlined_set();
        let array_group_map = elaborated.array_group_from_inline_set(&inline_set);

        let res =
            IndexElimination::new()
            .run(&inline_set, &array_group_map, &elaborated);

        assert!(res.is_ok());

        let inlined = res.unwrap();
        let init_schedule = Schedule::gen_initial_schedule(&inlined);

        let materializer = Materializer::new(vec![Box::new(DummyArrayMaterializer {})]);

        let res_mat =
            materializer.run(&inlined, &init_schedule);

        assert!(res_mat.is_ok());

        let param_circ = res_mat.unwrap();
        let cost = CostEstimator::default().estimate_cost(&param_circ);
        println!("{:?}", cost);
    }

    #[test]
    fn test_imgblur() {
        test_cost_estimator(
            "input img: [16,16] from client
            for x: 16 {
                for y: 16 {
                    img[x-1][y-1] + img[x+1][y+1]
                }
            }",
        );
    }
}
