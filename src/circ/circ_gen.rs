use std::collections::HashMap;

use crate::{
    circ::{Plaintext, Dimensions, HECircuit, HECircuitStore},
    lang::{IndexFreeExpr, HEObjectName, Operator, IndexFreeProgram},
    util::NameGenerator,
};

pub struct HECircuitGenerator {
    name_generator: NameGenerator,
    store: HECircuitStore,
}

impl HECircuitGenerator {
    pub fn new() -> Self {
        HECircuitGenerator {
            name_generator: NameGenerator::default(),
            store: HECircuitStore::new(),
        }
    }

    pub fn gen_circuit(&mut self, program: &IndexFreeProgram) -> Result<(HECircuit, HashMap<HEObjectName,Plaintext>), String> {
        for (name, ciphertext) in program.ciphertexts.iter() {
            self.store.ciphertexts.insert(name.clone(), ciphertext.clone());
        }
        let (circuit, _) = self._gen_circuit(&program.expr)?;
        Ok((circuit, self.store.plaintexts.clone()))
    }

    fn _gen_circuit(&mut self, expr: &IndexFreeExpr) -> Result<(HECircuit, Option<Dimensions>), String> {
        match expr {
            // TODO optimize this
            IndexFreeExpr::Reduce(dim, op, body) => {
                let (circ, shape_opt) = self._gen_circuit(body)?;
                let shape =
                    shape_opt.ok_or(String::from("Cannot reduce dimensionless array"))?;
            
                let mut cur =
                    if let Operator::Sub = op {
                        HECircuit::Sub(
                            Box::new(HECircuit::Literal(0)),
                            Box::new(circ.clone())
                        )

                    } else {
                        circ.clone()
                    };

                let block_size: usize = shape.block_size(*dim);

                for i in 1..shape[*dim] {
                    let rot_circ = 
                        HECircuit::Rotate(
                            Box::new(circ.clone()),
                            i*block_size
                        );

                    cur = match op {
                        Operator::Add => 
                            HECircuit::Add(Box::new(cur), Box::new(rot_circ)),

                        Operator::Mul =>
                            HECircuit::Mul(Box::new(cur), Box::new(rot_circ)),

                        Operator::Sub =>
                            HECircuit::Sub(Box::new(cur), Box::new(rot_circ)),
                    }
                }

                Ok((cur, Some(shape)))
            },

            IndexFreeExpr::Op(op, expr1, expr2) => {
                let (circ1, shape1_opt) = self._gen_circuit(expr1)?;
                let (circ2, shape2_opt) = self._gen_circuit(expr2)?;
                let out_circ =
                    match op {
                        Operator::Add => {
                            HECircuit::Add(Box::new(circ1), Box::new(circ2))
                        },

                        Operator::Mul => {
                            HECircuit::Mul(Box::new(circ1), Box::new(circ2))
                        },

                        Operator::Sub => {
                            HECircuit::Sub(Box::new(circ1), Box::new(circ2))
                        },
                    };
                let out_shape = 
                    match (shape1_opt, shape2_opt) {
                        (None, None) => None,
                        (None, Some(shape2)) => Some(shape2),
                        (Some(shape1), None) => Some(shape1),
                        (Some(shape1), Some(_)) => Some(shape1),
                    };
                Ok((out_circ, out_shape))
            }

            IndexFreeExpr::InputArray(arr) => {
                let object =
                    self.store.ciphertexts.get(arr)
                    .ok_or(format!("input array {} not found", arr))?;
                Ok((HECircuit::CiphertextRef(arr.clone()), Some(object.dimensions.clone())))
            },

            IndexFreeExpr::Literal(lit) => {
                Ok((HECircuit::Literal(*lit), None))
            },

            IndexFreeExpr::Offset(expr, amounts) => {
                let (circ, shape_opt) = self._gen_circuit(expr)?;
                let shape =
                    shape_opt.ok_or(String::from("Cannot apply offset transform to dimensionless array"))?;

                let mut total_offset = 0;
                let mut factor = 1;
                for (&dim, &offset) in shape.as_vec().iter().zip(amounts).rev() {
                    total_offset += offset * factor;
                    factor *= dim as isize;
                }

                Ok((HECircuit::Rotate(Box::new(circ), shape.wrap_offset(total_offset)), Some(shape)))
            },

            IndexFreeExpr::Fill(expr, dim) => {
                let (circ, shape_opt) = self._gen_circuit(expr)?;
                let shape =
                    shape_opt.ok_or(String::from("Cannot apply fill transform to dimensionless array"))?;
                if *dim >= shape.num_dims() {
                    Err(format!("Dimension {} is out of bounds for fill operation", dim))

                } else {
                    let block_size = shape.block_size(*dim);

                    let mut res_circ = circ;
                    for i in 1..shape[*dim] {
                        res_circ =
                            HECircuit::Add(
                                Box::new(res_circ.clone()),
                                Box::new(
                                    HECircuit::Rotate(
                                        Box::new(res_circ),
                                        shape.wrap_offset(-((i * block_size) as isize))
                                    )
                                )
                            );
                    }

                    Ok((res_circ, Some(shape)))
                }
            },

            IndexFreeExpr::Zero(expr, zero_dim) => {
                let (circ, shape_opt) = self._gen_circuit(expr)?;
                let shape =
                    shape_opt.ok_or(String::from("Cannot apply zero transform to dimensionless array"))?;

                let zero_region: im::Vector<(usize,usize)> =
                    shape.0.iter().enumerate().map(|(i, &dim_size)| {
                        if i == *zero_dim {
                            (1, dim_size-1)
                        } else {
                            (0, dim_size-1)
                        }
                    }).collect();

                let iter_domain = Self::get_iteration_domain(&shape.as_vec());
                let mut mask: Vec<isize> = Vec::new();
                for point in iter_domain.iter() {
                    let mut is_zero = true;
                    for (i, (lb, ub)) in point.iter().zip(zero_region.iter()) {
                        if !(lb <= i && i <= ub) {
                            is_zero = false;
                        }
                    }
                    mask.push(if is_zero { 0 } else { 1 })
                }

                let mask_name =
                    self.register_plaintext(
                        "zero_mask",
                        &shape, 
                        im::Vector::from(mask)
                    );

                let new_circ =
                    HECircuit::Mul(
                        Box::new(circ),
                        Box::new(HECircuit::PlaintextRef(mask_name))
                    );

                Ok((new_circ, Some(shape)))
            }
        }
    }

    fn get_iteration_domain_recur(
        dim: usize,
        head: im::Vector<usize>,
        rest: im::Vector<usize>
    ) -> im::Vector<im::Vector<usize>> {
        if rest.is_empty() {
            (0..dim)
                .map(|i| head.clone() + im::Vector::unit(i))
                .collect()

        } else {
            let (head_list, tail) = rest.split_at(1);
            let next = *head_list.head().unwrap();
            (0..dim).flat_map(|i|
                Self::get_iteration_domain_recur(
                    next,
                    head.clone() + im::Vector::unit(i),
                    tail.clone()
                )
            ).collect()
        }
    }

    pub fn get_iteration_domain(dims: &im::Vector<usize>) -> im::Vector<im::Vector<usize>> {
        if dims.is_empty() {
            im::Vector::new()

        } else {
            let (head_list, tail) = dims.clone().split_at(1);
            let head = *head_list.head().unwrap();
            Self::get_iteration_domain_recur(head, im::Vector::new(), tail)
        }
    }

    fn register_plaintext(&mut self, name: &str, shape: &Dimensions, value: im::Vector<isize>) -> String {
        let fresh_name = self.name_generator.get_fresh_name(name);
        self.store.plaintexts.insert(
            fresh_name.clone(),
            Plaintext { dimensions: shape.clone(), value }
        );
        fresh_name
    }
}

#[cfg(test)]
mod tests {
    use crate::lang::HEObjectName;

    use super::{ *, IndexFreeExpr::* };

    #[test]
    fn test_mask_iteration_domain() {
        let shape: im::Vector<usize> = im::vector![4, 2, 3];
        let iteration_dom = HECircuitGenerator::get_iteration_domain(&shape);
        assert_eq!(iteration_dom.len(), 24);
        assert_eq!(iteration_dom[0], im::vector![0,0,0]);
        assert_eq!(iteration_dom.last().unwrap().clone(), im::vector![3,1,2]);
    }

    fn test_circ_gen(ciphertexts: HashMap<HEObjectName, Ciphertext>, expr: IndexFreeExpr) {
        let program =
            IndexFreeProgram {
                client_store: HashMap::new(),
                ciphertexts,
                expr
            };

        let mut circ_gen = HECircuitGenerator::new();
        let res = circ_gen.gen_circuit(&program);
        assert!(res.is_ok());
        println!("{}", res.unwrap().0)
    }

    #[test]
    fn test_literal() {
        test_circ_gen(
            HashMap::from([
                (
                    String::from("img"),
                    Ciphertext { dimensions: Dimensions::from(im::vector![10, 10]) }
                )
            ]),
            Op(
                Operator::Add,
                Box::new(
                    Offset(
                        Box::new(InputArray("img".to_owned())),
                        im::vector![2,2]
                    )
                ),
                Box::new(Literal(2)),
            )
        );
    }

    #[test]
    fn test_blur() {
        test_circ_gen(
            HashMap::from([
                (
                    String::from("img"),
                    Ciphertext { dimensions: Dimensions::from(im::vector![10, 10]) }
                )
            ]),
            Op(
                Operator::Add,
                Box::new(
                    Op(
                        Operator::Add,
                        Box::new(
                            Offset(
                                Box::new(InputArray("img".to_owned())),
                                im::vector![2,2]
                            )
                        ),
                        Box::new(
                            Offset(
                                Box::new(InputArray("img".to_owned())),
                                im::vector![-2,-2]
                            )
                        ),
                    )
                ),
                Box::new(
                    Op(
                        Operator::Add,
                        Box::new(
                            Offset(
                                Box::new(InputArray("img".to_owned())),
                                im::vector![1,2]
                            )
                        ),
                        Box::new(
                            Offset(
                                Box::new(InputArray("img".to_owned())),
                                im::vector![-1,-1]
                            )
                        ),
                    )
                )
            )
        );
    }

    // this replicates Figure 1 in Dathathri, PLDI 2019 (CHET)
    #[test]
    fn test_matmul() {
        let filled_a =
            Fill(Box::new(InputArray("A".to_owned())), 2);

        let filled_b = 
            Fill(Box::new(InputArray("B".to_owned())), 0);

        let mul_ab =
            Op(
                Operator::Mul,
                Box::new(filled_a),
                Box::new(filled_b),
            );
        
        let add_ab = 
            Reduce(
                1,
                Operator::Add,
                Box::new(mul_ab)
            );

        let zero_expr = 
            Zero(Box::new(add_ab), 1);

        test_circ_gen(
            HashMap::from([
                (
                    String::from("A"),
                    Ciphertext { dimensions: Dimensions::from(im::vector![2, 2, 2]) }
                ),
                (
                    String::from("B"),
                    Ciphertext { dimensions: Dimensions::from(im::vector![2, 2, 2]) }
                ),
            ]),
            zero_expr
        );
    }
}