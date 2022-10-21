use std::{collections::HashMap};

use crate::util::NameGenerator;
use super::*;

#[derive(Clone,Debug)]
pub enum IndexFreeExprOperator {
    OpAdd, OpMul, OpSub
}

#[derive(Clone,Debug)]
pub enum ClientTransform {
    InputArray(HEObjectName),

    // reorder dimensions
    Transpose(Box<ClientTransform>, im::Vector<usize>),

    // add dimensions to the vector, intially filled with 0
    Expand(Box<ClientTransform>, usize),

    // extend existing dimensions
    Pad(Box<ClientTransform>, im::Vector<(usize, usize)>),
}

impl ClientTransform {
    pub fn as_python_str(&self) -> String {
        match self {
            ClientTransform::InputArray(arr) => arr.clone(),

            ClientTransform::Transpose(expr, dims) =>
                format!("transpose({},{:?})", expr.as_python_str(), dims),

            ClientTransform::Expand(expr, num_dims) => 
                format!("expand({},{})", expr.as_python_str(), num_dims),

            ClientTransform::Pad(expr, pad_list) =>
                format!("pad({},{:?})", expr.as_python_str(), pad_list),
        }
    }
}

pub type HEClientStore = HashMap<HEObjectName, ClientTransform>;

#[derive(Clone,Debug)]
pub enum IndexFreeExpr {
    // reduction
    ReduceNode(usize, IndexFreeExprOperator, Box<IndexFreeExpr>),

    // element-wise operation
    OpNode(IndexFreeExprOperator, Box<IndexFreeExpr>, Box<IndexFreeExpr>),

    // array received from the client
    InputArray(HEObjectName),

    // integer literal; must be treated as "shapeless" since literals can
    // denote arrays of *any* dimension
    Literal(isize),

    // TRANSFORMATIONS

    // fill the following dimensions of an array by rotating it
    Fill(Box<IndexFreeExpr>, Dimension),

    // offset array by a given amount in each dimension
    Offset(Box<IndexFreeExpr>, im::Vector<isize>),

    // zero out a dimension
    Zero(Box<IndexFreeExpr>, usize),
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

    fn _gen_circuit(&mut self, expr: &IndexFreeExpr) -> Result<(HECircuit, Option<Shape>), String> {
        match expr {
            // TODO optimize this
            IndexFreeExpr::ReduceNode(dim, op, body) => {
                let (circ, shape_opt) = self._gen_circuit(body)?;
                let shape =
                    shape_opt.ok_or(String::from("Cannot reduce dimensionless array"))?;
            
                let mut cur =
                    if let IndexFreeExprOperator::OpSub = op {
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
                        IndexFreeExprOperator::OpAdd => 
                            HECircuit::Add(Box::new(cur), Box::new(rot_circ)),

                        IndexFreeExprOperator::OpMul =>
                            HECircuit::Mul(Box::new(cur), Box::new(rot_circ)),

                        IndexFreeExprOperator::OpSub =>
                            HECircuit::Sub(Box::new(cur), Box::new(rot_circ)),
                    }
                }

                Ok((cur, Some(shape)))
            },

            IndexFreeExpr::OpNode(op, expr1, expr2) => {
                let (circ1, shape1_opt) = self._gen_circuit(expr1)?;
                let (circ2, shape2_opt) = self._gen_circuit(expr2)?;
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
                Ok((HECircuit::CiphertextRef(arr.clone()), Some(object.shape.clone())))
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

    fn register_plaintext(&mut self, name: &str, shape: &Shape, value: im::Vector<isize>) -> String {
        let fresh_name = self.name_generator.get_fresh_name(name);
        self.store.plaintexts.insert(
            fresh_name.clone(),
            Plaintext { shape: shape.clone(), value }
        );
        fresh_name
    }
}

#[cfg(test)]
mod tests {
    use super::{ *, IndexFreeExpr::* };

    #[test]
    fn test_mask_iteration_domain() {
        let shape: im::Vector<usize> = im::vector![4, 2, 3];
        let iteration_dom = HECircuitGenerator::get_iteration_domain(&shape);
        assert_eq!(iteration_dom.len(), 24);
        assert_eq!(iteration_dom[0], im::vector![0,0,0]);
        assert_eq!(iteration_dom.last().unwrap().clone(), im::vector![3,1,2]);
    }

    #[test]
    fn test_literal() {
        let mut inputs: HashMap<HEObjectName, Ciphertext> = HashMap::new();
        inputs.insert("img".to_owned(), Ciphertext { shape: Shape::from(im::vector![10, 10]) });

        let mut circ_gen = HECircuitGenerator::new(&inputs);

        let expr =
            OpNode(
                IndexFreeExprOperator::OpAdd,
                Box::new(
                    Offset(
                        Box::new(InputArray("img".to_owned())),
                        im::vector![2,2]
                    )
                ),
                Box::new(Literal(2)),
            );

        let res = circ_gen.gen_circuit(&expr);
        assert!(res.is_ok());
        println!("{}", res.unwrap().0)
    }

    #[test]
    fn test_blur() {
        let mut inputs: HashMap<HEObjectName, Ciphertext> = HashMap::new();
        inputs.insert("img".to_owned(), Ciphertext { shape: Shape::from(im::vector![10, 10]) });

        let mut circ_gen = HECircuitGenerator::new(&inputs);

        let expr = 
            OpNode(
                IndexFreeExprOperator::OpAdd,
                Box::new(
                    OpNode(
                        IndexFreeExprOperator::OpAdd,
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
                    OpNode(
                        IndexFreeExprOperator::OpAdd,
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
            );

        let res = circ_gen.gen_circuit(&expr);
        assert!(res.is_ok());
        println!("{}", res.unwrap().0)
    }

    // this replicates Figure 1 in Dathathri, PLDI 2019 (CHET)
    #[test]
    fn test_matmul() {
        let mut inputs: HashMap<HEObjectName, Ciphertext> = HashMap::new();
        inputs.insert("A".to_owned(), Ciphertext { shape: Shape::from(im::vector![2, 2, 2]) });
        inputs.insert("B".to_owned(), Ciphertext { shape: Shape::from(im::vector![2, 2, 2]) });

        let mut circ_gen = HECircuitGenerator::new(&inputs);

        let filled_A =
            Fill(Box::new(InputArray("A".to_owned())), 2);

        let filled_B = 
            Fill(Box::new(InputArray("B".to_owned())), 0);

        let mul_AB =
            OpNode(
                IndexFreeExprOperator::OpMul,
                Box::new(filled_A),
                Box::new(filled_B),
            );
        
        let add_AB = 
            ReduceNode(
                1,
                IndexFreeExprOperator::OpAdd,
                Box::new(mul_AB)
            );

        let zero_expr = 
            Zero(Box::new(add_AB), 1);

        // let expr = 
        //     Fill(Box::new(zero_expr), 1);

        let res = circ_gen.gen_circuit(&zero_expr);
        assert!(res.is_ok());
        let (circ, plaintexts) = res.unwrap();
        println!("{}", &circ);

        for (name, obj) in plaintexts.iter() {
            println!("{}: {:?}", name, obj);
        }
    }
}