use std::collections::HashMap;
use crate::util::NameGenerator;
use super::*;

#[derive(Clone,Debug)]
pub enum IndexFreeExprOperator {
    OpAdd, OpMul, OpSub
}

#[derive(Clone,Debug)]
pub enum IndexFreeExpr {
    ReduceNode(IndexFreeExprOperator, usize, Box<IndexFreeExpr>),
    OpNode(IndexFreeExprOperator, Box<IndexFreeExpr>, Box<IndexFreeExpr>),
    ArrayNode(TransformedArray),
}

#[derive(Clone,Debug)]
pub enum TransformedArray {
    // array received from the client
    InputArray(HEObjectName),

    // fill the following dimensions of an array by rotating it
    Fill(Box<TransformedArray>, Dimension),

    // offset array by a given amount in each dimension
    Offset(Box<TransformedArray>, im::Vector<isize>),

    // zero out specific ranges in an array
    Zero(Box<TransformedArray>, im::Vector<(usize, usize)>),
}

pub struct HECircuitGenerator {
    name_generator: NameGenerator,
    object_map: HashMap<HEObjectName, HEObject>
}

impl HECircuitGenerator {
    pub fn new() -> Self {
        HECircuitGenerator {
            name_generator: NameGenerator::new(),
            object_map: HashMap::new(),
        }
    }

    fn gen_circuit_expr(&mut self, expr: &IndexFreeExpr) -> Result<(HECircuit, Shape), String> {
        match expr {
            // TODO optimize this
            IndexFreeExpr::ReduceNode(op, dim, body) => {
                let (circ, shape) = self.gen_circuit_expr(body)?;

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
                for i in (*dim)..shape.len() {
                    block_size *= shape[i];
                }

                for i in 0..shape[*dim] {
                    let rot_circ = 
                        HECircuit::Rotate(
                            Box::new(circ.clone()),
                            -(((i+1)*block_size) as isize));

                    match op {
                        IndexFreeExprOperator::OpAdd => {
                            cur = HECircuit::Add(Box::new(cur), Box::new(rot_circ));
                        }
                        IndexFreeExprOperator::OpMul => {
                            cur = HECircuit::Mul(Box::new(cur), Box::new(rot_circ));
                        }
                        IndexFreeExprOperator::OpSub => {
                            cur = HECircuit::Sub(Box::new(cur), Box::new(rot_circ));
                        }
                    }
                }

                Ok((cur, shape))
            },

            IndexFreeExpr::OpNode(op, expr1, expr2) => {
                let (circ1, shape1) = self.gen_circuit_expr(expr1)?;
                let (circ2, _) = self.gen_circuit_expr(expr2)?;
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

            IndexFreeExpr::ArrayNode(arr) => {
                self.gen_circuit_array(arr)
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
        self.object_map.insert(
            fresh_name.clone(),
            HEObject::Plaintext(shape.clone(), value)
        );
        fresh_name
    }

    fn gen_circuit_array(&mut self, array: &TransformedArray) -> Result<(HECircuit, Shape), String> {
        match array {
            TransformedArray::InputArray(arr) => {
                let object =
                    self.object_map.get(arr)
                    .ok_or(format!("input array {} not found", arr))?;
                Ok((HECircuit::CiphertextRef(arr.clone()), object.shape().clone()))
            },

            TransformedArray::Offset(arr, amounts) => {
                let (circ, shape) = self.gen_circuit_array(arr)?;

                let mut total_offset = 0;
                let mut factor = 1;
                for (&dim, &offset) in shape.iter().zip(amounts).rev() {
                    total_offset += offset * factor;
                    factor *= dim as isize;
                }

                Ok((HECircuit::Rotate(Box::new(circ), total_offset), shape))
            },

            TransformedArray::Fill(arr, dim) => {
                let (circ, shape) = self.gen_circuit_array(arr)?;
                if *dim >= shape.len() - 1 {
                    Err(format!("Dimension {} is out of bounds for fill operation", dim))

                } else {
                    let mut block_size = 0;
                    for i in dim+1..shape.len() {
                        block_size += shape[i]
                    }

                    let mut res_circ = circ;
                    for i in 0..shape[*dim] {
                        res_circ =
                            HECircuit::Add(
                                Box::new(res_circ.clone()),
                                Box::new(
                                    HECircuit::Rotate(
                                        Box::new(res_circ),
                                        ((i+1) * block_size) as isize
                                    )
                                )
                            );
                    }

                    Ok((res_circ, shape))
                }
            },

            TransformedArray::Zero(arr, zero_region) => {
                let (circ, shape) = self.gen_circuit_array(arr)?;

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
}