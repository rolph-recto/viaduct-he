use std::collections::HashMap;

use he_vectorizer::{circ::{circ_gen::HECircuitGenerator, optimizer::{HELatencyModel, ExtractorType, Optimizer}, lowering::{program::HEProgram, lowered_program::HELoweredProgram}}, lang::{index_elim::IndexElimination, parser::ProgramParser}};

fn test_compile(src: &str, size: usize, opt_duration: usize, extractor: ExtractorType, noinplace: bool) {
    let parser = ProgramParser::new();
    let src_program = parser.parse(src).unwrap();

    let index_elim = IndexElimination::new();
    let indfree_program = index_elim.run(&src_program).unwrap();

    println!("index-free representation:");
    println!("{}", &indfree_program.expr);

    let circ_gen = HECircuitGenerator::new();
    let (circ, store) = circ_gen.gen_circuit(&indfree_program).unwrap();

    println!("circuit:");
    println!("{}", circ);

    let latency_model = HELatencyModel::default();

    // parse the expression, the type annotation tells it which Language to use
    // let init_circ: RecExpr<HEOptCircuit> = input_str.parse().unwrap();

    let init_circ = circ.to_opt_circuit();
    let init_prog = HEProgram::from(&init_circ);

    println!("Initial HE program (muldepth {}, latency {}ms):",
        init_prog.get_muldepth(),
        init_prog.get_latency(&latency_model)
    );

    let opt_circ =
        if opt_duration > 0 {
            let optimizer = Optimizer::new(size);
            optimizer.optimize(&init_circ, size, opt_duration, extractor)

        } else {
            init_circ
        };

    let opt_prog = HEProgram::from(&opt_circ);

    if opt_duration > 0 {
        println!("Optimized HE program (muldepth {}, latency {}ms):",
            opt_prog.get_muldepth(),
            opt_prog.get_latency(&latency_model)
        );
    }

    let lowered_prog: HELoweredProgram =
        HELoweredProgram::lower_program(
            size,
            noinplace,
            &opt_prog, 
            &store,
            indfree_program.client_store,
            HashMap::new(),
            HashMap::new()
        );
}

#[test]
fn test_imgblur() {
    let src =
        "
        input img: [(0,16),(0,16)]
        for x: (0, 16) {
            for y: (0, 16) {
                img[x-1][y-1] + img[x+1][y+1]
            }
        }
        ";

    test_compile(src, 4096, 0, ExtractorType::Greedy, true);
}

#[test]
fn test_matmul() {
    let src =
        "
        input A: [(0,4),(0,4)]
        input B: [(0,4),(0,4)]
        let x = A + B
        for i: (0,4) {
            for j: (0,4) {
                sum(for k: (0,4) { A[i][k] * B[k][j] })
            }
        }
        ";

    test_compile(src, 4096, 0, ExtractorType::Greedy, true);
}

#[test]
fn test_matmul2() {
    let src =
        "
        input A1: [(0,4),(0,4)]
        input A2: [(0,4),(0,4)]
        input B: [(0,4),(0,4)]
        let res =
            for i: (0,4) {
                for j: (0,4) {
                    sum(for k: (0,4) { A1[i][k] * B[k][j] })
                }
            }
        for i: (0,4) {
            for j: (0,4) {
                sum(for k: (0,4) { A2[i][k] * res[k][j] })
            }
        }
        ";

    test_compile(src, 4096, 0, ExtractorType::Greedy, true);
}