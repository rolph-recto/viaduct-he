use std::collections::HashMap;
use crate::util::NameGenerator;
use super::*;

#[derive(Clone,Debug)]
pub enum IndexFreeExprOperator {
    OpAdd, OpMul, OpSub
}

#[derive(Clone,Debug)]
pub enum IndexFreeExpr {
    // reduction
    ReduceNode(IndexFreeExprOperator, usize, Box<IndexFreeExpr>),

    // element-wise operation
    OpNode(IndexFreeExprOperator, Box<IndexFreeExpr>, Box<IndexFreeExpr>),

    // array received from the client
    InputArray(HEObjectName),

    // TRANSFORMATIONS

    // fill the following dimensions of an array by rotating it
    Fill(Box<IndexFreeExpr>, Dimension),

    // offset array by a given amount in each dimension
    Offset(Box<IndexFreeExpr>, im::Vector<isize>),

    // zero out specific ranges in an array
    Zero(Box<IndexFreeExpr>, im::Vector<(usize, usize)>),
}

pub struct HECircuitGenerator {
    name_generator: NameGenerator,
    store: HECircuitStore,
}

impl HECircuitGenerator {
    pub fn new(inputs: &HashMap<HEObjectName,Ciphertext>) -> Self {
        HECircuitGenerator {
            name_generator: NameGenerator::default(),
            store: HECircuitStore::new(inputs),
        }
    }

    pub fn gen_circuit(&mut self, expr: &IndexFreeExpr) -> Result<(HECircuit, HashMap<HEObjectName,Plaintext>), String> {
        let (circuit, _) = self._gen_circuit(expr)?;
        Ok((circuit, self.store.plaintexts.clone()))
    }

    fn _gen_circuit(&mut self, expr: &IndexFreeExpr) -> Result<(HECircuit, Shape), String> {
        match expr {
            // TODO optimize this
            IndexFreeExpr::ReduceNode(op, dim, body) => {
                let (circ, shape) = self._gen_circuit(body)?;

                let mut cur =
                    if let IndexFreeExprOperator::OpSub = op {
                        HECircuit::Sub(
                            Box::new(HECircuit::Literal(0)),
                            Box::new(circ.clone())
                        )

                    } else {
                        circ.clone()
                    };

                let mut block_size: usize = 1;
                for i in (*dim+1)..shape.len() {
                    block_size *= shape[i];
                }

                for i in 1..shape[*dim] {
                    let rot_circ = 
                        HECircuit::Rotate(
                            Box::new(circ.clone()),
                            -((i*block_size) as isize)
                        );

                    cur = match op {
                        IndexFreeExprOperator::OpAdd => 
                            HECircuit::Add(Box::new(cur), Box::new(rot_circ)),

                        IndexFreeExprOperator::OpMul =>
                            HECircuit::Mul(Box::new(cur), Box::new(rot_circ)),

                        IndexFreeExprOperator::OpSub =>
                            HECircuit::Sub(Box::new(cur), Box::new(rot_circ)),
                    }
                }

                Ok((cur, shape))
            },

            IndexFreeExpr::OpNode(op, expr1, expr2) => {
                let (circ1, shape1) = self._gen_circuit(expr1)?;
                let (circ2, _) = self._gen_circuit(expr2)?;
                let out_circ =
                    match op {
                        IndexFreeExprOperator::OpAdd => {
                            HECircuit::Add(Box::new(circ1), Box::new(circ2))
                        },

                        IndexFreeExprOperator::OpMul => {
                            HECircuit::Mul(Box::new(circ1), Box::new(circ2))
                        },

                        IndexFreeExprOperator::OpSub => {
                            HECircuit::Sub(Box::new(circ1), Box::new(circ2))
                        },
                    };
                Ok((out_circ, shape1))
            }

            IndexFreeExpr::InputArray(arr) => {
                let object =
                    self.store.ciphertexts.get(arr)
                    .ok_or(format!("input array {} not found", arr))?;
                Ok((HECircuit::CiphertextRef(arr.clone()), object.shape.clone()))
            },

            IndexFreeExpr::Offset(expr, amounts) => {
                let (circ, shape) = self._gen_circuit(expr)?;

                let mut total_offset = 0;
                let mut factor = 1;
                for (&dim, &offset) in shape.iter().zip(amounts).rev() {
                    total_offset += offset * factor;
                    factor *= dim as isize;
                }

                Ok((HECircuit::Rotate(Box::new(circ), total_offset), shape))
            },

            IndexFreeExpr::Fill(expr, dim) => {
                let (circ, shape) = self._gen_circuit(expr)?;
                if *dim >= shape.len() {
                    Err(format!("Dimension {} is out of bounds for fill operation", dim))

                } else {
                    let mut block_size = 1;
                    for i in dim+1..shape.len() {
                        block_size *= shape[i]
                    }

                    let mut res_circ = circ;
                    for i in 1..shape[*dim] {
                        res_circ =
                            HECircuit::Add(
                                Box::new(res_circ.clone()),
                                Box::new(
                                    HECircuit::Rotate(
                                        Box::new(res_circ),
                                        (i * block_size) as isize
                                    )
                                )
                            );
                    }

                    Ok((res_circ, shape))
                }
            },

            IndexFreeExpr::Zero(expr, zero_region) => {
                let (circ, shape) = self._gen_circuit(expr)?;

                let iter_domain = Self::get_iteration_domain(&shape);
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

                Ok((new_circ, shape))
            }
        }
    }

    fn get_iteration_domain_recur(
        dim: usize,
        head: im::Vector<usize>,
        rest: Shape
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

    pub fn get_iteration_domain(dims: &Shape) -> im::Vector<im::Vector<usize>> {
        if dims.is_empty() {
            im::Vector::new()

        } else {
            let (head_list, tail) = dims.clone().split_at(1);
            let head = *head_list.head().unwrap();
            Self::get_iteration_domain_recur(head, im::Vector::new(), tail)
        }
    }

    fn register_plaintext(&mut self, name: &str, shape: &Shape, value: im::Vector<isize>) -> String {
        let fresh_name = self.name_generator.get_fresh_name(name);
        self.store.plaintexts.insert(
            fresh_name.clone(),
            Plaintext { shape: shape.clone(), value }
        );
        fresh_name
    }
}