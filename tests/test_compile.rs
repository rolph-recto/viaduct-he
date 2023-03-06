use he_vectorizer::{lang::{parser::ProgramParser, elaborated::Elaborator, index_elim::IndexElimination}, scheduling::Schedule, circ::{materializer::{DefaultArrayMaterializer, InputArrayMaterializer, Materializer}, partial_eval::HEPartialEvaluator}, program::{lowering::CircuitLowering, backend::{pyseal::SEALBackend, HEBackend}}};

fn test_compile(src: &str) {
    let source = ProgramParser::new().parse(&src).unwrap();
    let elaborated = Elaborator::new().run(source);

    let inline_set = elaborated.get_default_inline_set();
    let array_group_map = elaborated.get_default_array_group_map();

    let res_index_elim = IndexElimination::new().run(&inline_set, &array_group_map, elaborated);

    let transformed = res_index_elim.unwrap();
    let init_schedule = Schedule::gen_initial_schedule(&transformed);

    let array_materializers: Vec<Box<dyn InputArrayMaterializer>> = 
        vec![Box::new(DefaultArrayMaterializer::new())];
    let materializer =
        Materializer::new(array_materializers, transformed);

    let res_materialize = materializer.run(&init_schedule);
    let circuit = res_materialize.unwrap();

    // TODO add optimizer

    let circuit_pe = HEPartialEvaluator::new().run(circuit);
    let program = CircuitLowering::new().run(circuit_pe);

    let seal_backend = SEALBackend::new(None);
    let mut code_str: String = String::new();
    seal_backend.compile(program, &mut code_str).unwrap();

    println!("{}", code_str);
}

#[test]
fn test_imgblur() {
    let src =
        "
        input img: [4,4] from client
        for x: 4 {
            for y: 4 {
                img[x-1][y-1] + img[x+1][y+1]
            }
        }
        ";

    test_compile(src);
}

#[test]
fn test_matmul() {
    let src =
        "
        input A: [2,2] from client
        input B: [2,2] from server
        for i: 2 {
            for j: 2 {
                sum(for k: 2 { A[i][k] * B[k][j] })
            }
        }
        ";

    test_compile(src);
}

#[test]
fn test_matmul2() {
    let src =
        "
        input A1: [2,2] from client
        input A2: [2,2] from client
        input B: [2,2] from client
        let res =
            for i: 2 {
                for j: 2 {
                    sum(for k: 2 { A1[i][k] * B[k][j] })
                }
            }
        in
        for i: 2 {
            for j: 2 {
                sum(for k: 2 { A2[i][k] * res[k][j] })
            }
        }
        ";

    test_compile(src);
}

#[test]
fn test_dotprod() {
    let src =
        "
        input C: [4] from client
        input P: [4] from server
        sum(for i: 4 { C[i] * P[i] })
        ";

    test_compile(src);
}
