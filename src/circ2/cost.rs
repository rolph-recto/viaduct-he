use std::{ops::{Add, Mul}, cmp::max};

use crate::{circ2::{
    ParamCircuitProgram, ParamCircuitExpr, IndexCoordinateSystem, VectorType
}, lang::Operator, scheduling::ExprScheduleType};

#[derive(Copy,Clone,Debug,PartialEq,Eq)]
pub struct CostFeatures {
    input_ciphertexts: usize,
    input_plaintexts: usize,
    output_ciphertexts: usize,
    rotations: usize,
    ct_ct_add: usize,
    pt_ct_add: usize,
    pt_pt_add: usize,
    ct_ct_mult: usize,
    pt_ct_mult: usize,
    pt_pt_mult: usize,
    ct_ct_sub: usize,
    pt_ct_sub: usize,
    pt_pt_sub: usize,
    ct_ct_muldepth: usize,
    pt_ct_muldepth: usize,
}

impl CostFeatures {
    fn new() -> Self {
        CostFeatures {
            input_ciphertexts: 0,
            input_plaintexts: 0,
            output_ciphertexts: 0,
            rotations: 0,
            ct_ct_add: 0,
            pt_ct_add: 0,
            pt_pt_add: 0,
            ct_ct_mult: 0,
            pt_ct_mult: 0,
            pt_pt_mult: 0,
            ct_ct_sub: 0,
            pt_ct_sub: 0,
            pt_pt_sub: 0,
            ct_ct_muldepth: 0,
            pt_ct_muldepth: 0,
        }
    }

    fn weighted_cost(&self, weights: &CostFeatures) -> usize {
        let weighted = (*self).weight(*weights);
weighted.input_ciphertexts
        + weighted.input_plaintexts
        + weighted.output_ciphertexts
        + weighted.rotations
        + weighted.ct_ct_add
        + weighted.pt_ct_add
        + weighted.pt_pt_add
        + weighted.ct_ct_mult
        + weighted.pt_ct_mult
        + weighted.pt_pt_mult
        + weighted.ct_ct_sub
        + weighted.pt_ct_sub
        + weighted.pt_pt_sub
        + weighted.ct_ct_muldepth
        + weighted.pt_ct_muldepth
    }

    /// because this operation is meant for combining features of subexprs,
    /// everything is pointwise *except* for muldepth, of which max is taken
    pub fn combine(self, other: Self) -> Self {
        CostFeatures {
            input_ciphertexts: self.input_ciphertexts + other.input_ciphertexts,
            input_plaintexts: self.input_plaintexts + other.input_plaintexts,
            output_ciphertexts: self.output_ciphertexts + other.output_ciphertexts,
            rotations: self.rotations + other.rotations,
            ct_ct_add: self.ct_ct_add + other.ct_ct_add,
            pt_ct_add: self.pt_ct_add + other.pt_ct_add,
            pt_pt_add: self.pt_pt_add + other.pt_pt_add,
            ct_ct_mult: self.ct_ct_mult + other.ct_ct_mult,
            pt_ct_mult: self.pt_ct_mult + other.pt_ct_mult,
            pt_pt_mult: self.pt_pt_mult + other.pt_pt_mult,
            ct_ct_sub: self.ct_ct_sub + other.ct_ct_sub,
            pt_ct_sub: self.pt_ct_sub + other.pt_ct_sub,
            pt_pt_sub: self.pt_pt_sub + other.pt_pt_sub,
            ct_ct_muldepth: max(self.ct_ct_muldepth, other.ct_ct_muldepth),
            pt_ct_muldepth: max(self.pt_ct_muldepth, other.pt_ct_muldepth),
        }
    }

    pub fn weight(self, other: Self) -> Self {
        CostFeatures {
            input_ciphertexts: self.input_ciphertexts * other.input_ciphertexts,
            input_plaintexts: self.input_plaintexts * other.input_plaintexts,
            output_ciphertexts: self.output_ciphertexts * other.output_ciphertexts,
            rotations: self.rotations * other.rotations,
            ct_ct_add: self.ct_ct_add * other.ct_ct_add,
            pt_ct_add: self.pt_ct_add * other.pt_ct_add,
            pt_pt_add: self.pt_pt_add * other.pt_pt_add,
            ct_ct_mult: self.ct_ct_mult * other.ct_ct_mult,
            pt_ct_mult: self.pt_ct_mult * other.pt_ct_mult,
            pt_pt_mult: self.pt_pt_mult * other.pt_pt_mult,
            ct_ct_sub: self.ct_ct_sub * other.ct_ct_sub,
            pt_ct_sub: self.pt_ct_sub * other.pt_ct_sub,
            pt_pt_sub: self.pt_pt_sub * other.pt_pt_sub,
            ct_ct_muldepth: self.ct_ct_muldepth * other.ct_ct_muldepth,
            pt_ct_muldepth: self.pt_ct_muldepth * other.pt_ct_muldepth,
        }
    }
}

impl Default for CostFeatures {
    fn default() -> Self {
        Self {
            input_ciphertexts: 0,
            input_plaintexts: 0,
            output_ciphertexts: 0,
            rotations: 0,
            ct_ct_add: 0,
            pt_ct_add: 0,
            pt_pt_add: 0,
            ct_ct_mult: 0,
            pt_ct_mult: 0,
            pt_pt_mult: 0,
            ct_ct_sub: 0,
            pt_ct_sub: 0,
            pt_pt_sub: 0,
            ct_ct_muldepth: 0,
            pt_ct_muldepth: 0,
        }
    }
}

/// estimate cost of an param circuit program
pub struct CostEstimator {}

impl CostEstimator {
    pub fn new() -> Self { CostEstimator {} }

    // TODO finish
    pub fn estimate_cost(&self, program: &ParamCircuitProgram) -> Result<CostFeatures,String> {
        Ok(CostFeatures::default())
        /*
        if let ExprScheduleType::Specific(schedule) = &program.schedule {
            let coord_system = IndexCoordinateSystem::new(schedule.exploded_dims.iter());
            let (vec_type, mult, mut cost) = self.estimate_cost_expr(&program.expr, &coord_system);

            if let VectorType::Ciphertext = vec_type {
                cost.input_ciphertexts = program.registry.get_ct_objects().len();
                cost.input_plaintexts = program.registry.get_pt_objects().len();
                cost.output_ciphertexts = mult;
                Ok(cost)

            } else {
                Err("Cannot estimate cost of plaintext computation".to_string())
            }

        } else {
            Err("Cannot estimate cost of literal expression".to_string())
        }
        */
    }

    fn estimate_cost_expr(&self, expr: &ParamCircuitExpr, coord_system: &IndexCoordinateSystem) -> (VectorType, usize, CostFeatures) {
        match expr {
            ParamCircuitExpr::CiphertextVar(_) => {
                (VectorType::Ciphertext, coord_system.multiplicity(), CostFeatures::new())
            },

            ParamCircuitExpr::PlaintextVar(_) => {
                (VectorType::Plaintext, coord_system.multiplicity(), CostFeatures::new())
            },

            ParamCircuitExpr::Literal(_) => {
                (VectorType::Plaintext, coord_system.multiplicity(), CostFeatures::new())
            },

            ParamCircuitExpr::Op(op, expr1, expr2) => {
                let (type1, mult1, cost1) = self.estimate_cost_expr(expr1, coord_system);
                let (type2, mult2, cost2) = self.estimate_cost_expr(expr2, coord_system);
                assert!(mult1 == mult2);

                let mut out_cost = CostFeatures::new();
                let out_type = 
                    match (op, type1, type2) {
                        (Operator::Add, VectorType::Ciphertext, VectorType::Ciphertext) => {
                            out_cost.ct_ct_add = mult1;
                            VectorType::Ciphertext
                        },

                        (Operator::Add, VectorType::Ciphertext, VectorType::Plaintext) |
                        (Operator::Add, VectorType::Plaintext, VectorType::Ciphertext) => {
                            out_cost.pt_ct_add = mult1;
                            VectorType::Ciphertext
                        },

                        (Operator::Add, VectorType::Plaintext, VectorType::Plaintext) => {
                            out_cost.pt_pt_add = mult1;
                            VectorType::Plaintext
                        },

                        (Operator::Sub, VectorType::Ciphertext, VectorType::Ciphertext) => {
                            out_cost.ct_ct_sub = mult1;
                            VectorType::Ciphertext
                        },

                        (Operator::Sub, VectorType::Ciphertext, VectorType::Plaintext) |
                        (Operator::Sub, VectorType::Plaintext, VectorType::Ciphertext) => {
                            out_cost.pt_ct_sub = mult1;
                            VectorType::Ciphertext
                        },

                        (Operator::Sub, VectorType::Plaintext, VectorType::Plaintext) => {
                            out_cost.pt_pt_sub = mult1;
                            VectorType::Plaintext
                        },

                        (Operator::Mul, VectorType::Ciphertext, VectorType::Ciphertext) => {
                            out_cost.ct_ct_sub = mult1;
                            out_cost.ct_ct_muldepth = max(cost1.ct_ct_muldepth, cost2.ct_ct_muldepth) + 1;
                            VectorType::Ciphertext
                        },

                        (Operator::Mul, VectorType::Ciphertext, VectorType::Plaintext) |
                        (Operator::Mul, VectorType::Plaintext, VectorType::Ciphertext) => {
                            out_cost.pt_ct_sub = mult1;
                            out_cost.pt_ct_muldepth = max(cost1.pt_ct_muldepth, cost2.pt_ct_muldepth) + 1;
                            VectorType::Ciphertext
                        },

                        (Operator::Mul, VectorType::Plaintext, VectorType::Plaintext) => {
                            out_cost.pt_pt_mult = mult1;
                            VectorType::Plaintext
                        }
                    };

                (out_type, mult1, cost1.combine(cost2).combine(out_cost))
            },

            ParamCircuitExpr::Rotate(_, body) => {
                let (body_type, body_mult, body_cost) = self.estimate_cost_expr(body, coord_system);
                let mut out_cost = CostFeatures::new();
                out_cost.rotations = body_mult;
                (body_type, body_mult, body_cost.combine(out_cost))
            },

            ParamCircuitExpr::ReduceVectors(_, extent, op, body) => {
                let (body_type, body_mult, body_cost) = self.estimate_cost_expr(body, coord_system);
                let out_mult = body_mult / extent;
                let mut out_cost = CostFeatures::new();

                match (op, body_type) {
                    (Operator::Add, VectorType::Ciphertext) => {
                        out_cost.ct_ct_add = out_mult * (extent - 1);
                    },

                    (Operator::Add, VectorType::Plaintext) => {
                        out_cost.pt_pt_add = out_mult * (extent - 1);
                    },

                    (Operator::Sub, VectorType::Ciphertext) => {
                        out_cost.ct_ct_sub  = out_mult * (extent - 1);
                    },

                    (Operator::Sub, VectorType::Plaintext) => {
                        out_cost.pt_pt_add = out_mult * (extent - 1);
                    },

                    (Operator::Mul, VectorType::Ciphertext) => {
                        out_cost.ct_ct_mult = out_mult * (extent - 1);
                        
                        // compilation of reduction will try to create a
                        // balanced tree of mults, so there is only log-factor
                        // addition to muldepth
                        out_cost.ct_ct_muldepth = ((extent - 1) as f64).log2() as usize;
                    },

                    (Operator::Mul, VectorType::Plaintext) => {
                        out_cost.pt_pt_mult = out_mult * (extent - 1);
                    }
                }

                (body_type, out_mult, body_cost.combine(out_cost))
            },
        }
    }
}

#[cfg(test)]
mod tests{
    use crate::{lang::{parser::ProgramParser, index_elim::IndexElimination, source::SourceProgram, elaborated::Elaborator}, circ2::materializer::{Materializer, DummyArrayMaterializer}, scheduling::Schedule};
    use super::*;

    // generate an initial schedule for a program
    fn test_cost_estimator(src: &str) {
        let parser = ProgramParser::new();
        let program: SourceProgram = parser.parse(src).unwrap();

        let elaborated = Elaborator::new().run(program);
        let inline_set = elaborated.get_default_inline_set();
        let array_group_map = elaborated.get_default_array_group_map();

        let res =
            IndexElimination::new()
            .run(&inline_set, &array_group_map, elaborated);
        
        assert!(res.is_ok());

        let tprogram = res.unwrap();
        let init_schedule = Schedule::gen_initial_schedule(&tprogram);

        let materializer =
            Materializer::new(
                vec![Box::new(DummyArrayMaterializer {})],
                tprogram,
            );

        let res_mat = materializer.materialize(&init_schedule);
        assert!(res_mat.is_ok());

        let param_circ = res_mat.unwrap();
        let res_cost = CostEstimator::new().estimate_cost(&param_circ);
        assert!(res_cost.is_ok());
        println!("{:?}", res_cost.unwrap());
    }

    #[test]
    fn test_imgblur() {
        test_cost_estimator(
        "input img: [16,16]
            for x: 16 {
                for y: 16 {
                    img[x-1][y-1] + img[x+1][y+1]
                }
            }"
        );
    }
}