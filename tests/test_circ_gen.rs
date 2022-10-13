use std::collections::HashMap;

use he_vectorizer::circ::{
    *,
    circ_gen::{
        HECircuitGenerator,
        IndexFreeExpr::*,
        IndexFreeExprOperator
    }
};

#[test]
fn test_mask_iteration_domain() {
    let shape: im::Vector<usize> = im::vector![4, 2, 3];
    let iteration_dom = HECircuitGenerator::get_iteration_domain(&shape);
    assert_eq!(iteration_dom.len(), 24);
    assert_eq!(iteration_dom[0], im::vector![0,0,0]);
    assert_eq!(iteration_dom.last().unwrap().clone(), im::vector![3,1,2]);
}

#[test]
fn test_blur() {
    let mut inputs: HashMap<HEObjectName, Ciphertext> = HashMap::new();
    inputs.insert("img".to_owned(), Ciphertext { shape: im::vector![10, 10] });

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
    inputs.insert("A".to_owned(), Ciphertext { shape: im::vector![2, 2, 2] });
    inputs.insert("B".to_owned(), Ciphertext { shape: im::vector![2, 2, 2] });

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
            IndexFreeExprOperator::OpAdd,
            1,
            Box::new(mul_AB)
        );

    let zero_expr = 
        Zero(
            Box::new(add_AB),
            im::vector![(0,1),(1,1),(0,1)]
        );

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