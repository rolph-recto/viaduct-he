use he_vectorizer::circ::{*, circ_gen::HECircuitGenerator};
use he_vectorizer::lang::{
    parser::ProgramParser,
    normalized::Normalizer,
    source::SourceProgram,
    typechecker::TypeChecker,
};


#[test]
fn imgblur() {
    let parser = ProgramParser::new();
    let program: SourceProgram =
        parser.parse("
            input img: [(0,16),(0,16)]
            for x: (0, 16) {
                for y: (0, 16) {
                    img[x-1][y-1] + img[x+1][y+1]
                }
            }
        ").unwrap();

    let res = Normalizer::new().run(&program);
    assert!(res.is_ok());
    println!("{}", res.unwrap().expr)
}

#[test]
fn matmatmul() {
    let parser = ProgramParser::new();
    let prog: SourceProgram =
        parser.parse("
        input A: [(0,4),(0,4)]
        input B: [(0,4),(0,4)]
        let x = A + B
        for i: (0,4) {
            for j: (0,4) {
                sum(for k: (0,4) { A[i][k] * B[k][j] })
            }
        }
        ").unwrap();

    let res = Normalizer::new().run(&prog);
    assert!(res.is_ok());
    println!("{}", res.unwrap().expr);
}