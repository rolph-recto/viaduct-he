use he_vectorizer::circ::circ_gen::HECircuitGenerator;

#[test]
fn test_mask_iteration_domain() {
    let shape: im::Vector<usize> = im::vector![4, 2, 3];
    let iteration_dom = HECircuitGenerator::get_iteration_domain(&shape);
    assert_eq!(iteration_dom.len(), 24);
    assert_eq!(iteration_dom[0], im::vector![0,0,0]);
    assert_eq!(iteration_dom.last().unwrap().clone(), im::vector![3,1,2]);
}