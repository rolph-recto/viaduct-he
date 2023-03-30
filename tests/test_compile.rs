use he_vectorizer::{
    lang::{
        parser::ProgramParser,
        elaborated::Elaborator,
        index_elim::IndexElimination
    },
    scheduling::Schedule,
    circ::{
        array_materializer::{DefaultArrayMaterializer, InputArrayMaterializer},
        materializer::Materializer,
        plaintext_hoisting::PlaintextHoisting
    },
    program::{
        lowering::CircuitLowering,
        backend::{pyseal::SEALBackend, HEBackend}
    }
};

fn test_compile(src: &str) {
    let source = ProgramParser::new().parse(&src).unwrap();
    let elaborated = Elaborator::new().run(source);
    println!("elaborated program:\n{}", elaborated);

    let inline_set = elaborated.all_inlined_set();
    let array_group_map = elaborated.array_group_from_inline_set(&inline_set);

    let res_index_elim =
        IndexElimination::new()
        .run(&inline_set, &array_group_map, &elaborated);

    let transformed = res_index_elim.unwrap();
    let init_schedule = Schedule::gen_initial_schedule(&transformed);
    println!("transformed program:\n{}", transformed);

    let array_materializers: Vec<Box<dyn InputArrayMaterializer>> = 
        vec![Box::new(DefaultArrayMaterializer::new())];
    let materializer = Materializer::new(array_materializers);

    let res_materialize =
        materializer.run(&transformed, &init_schedule);
    let circuit = res_materialize.unwrap();
    println!("circuit:\n{}", circuit);

    // TODO add optimizer

    let pe_circuit = PlaintextHoisting::new().run(circuit);
    println!("partially evaluated circuit:\n{}", pe_circuit);

    let program = CircuitLowering::new().run(pe_circuit);
    println!("program:\n{}", program);

    let mut seal_backend =
        SEALBackend::new(None, true, 1024);

    // throw output away in anonymous buffer
    let writer: Box<dyn std::io::Write> = Box::new(Vec::new());
    seal_backend.compile(program, writer).unwrap();
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
